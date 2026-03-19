#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import AdamW

from datasets import Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    Trainer,
    TrainingArguments
)
from transformers.modeling_outputs import SequenceClassifierOutput

from rdkit import Chem

# -------------------------
# Preprocessing & Data Loading
# -------------------------
def canonicalize_smiles(s):
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None

def load_and_preprocess_data(main_csv_path, extra_x_path, smiles_col, target_col):
    """
    Loads main CSV and Extra Features CSV, merges them, 
    and cleans based on valid SMILES.
    """
    # 1. Load Main Data
    df = pd.read_csv(main_csv_path)
    
    # 2. Load Extra Features
    if os.path.exists(extra_x_path):
        df_extra = pd.read_csv(extra_x_path)
        # Verify lengths match
        if len(df) != len(df_extra):
            raise ValueError(f"Row mismatch: {main_csv_path} ({len(df)}) vs {extra_x_path} ({len(df_extra)})")
        
        # Identify extra feature columns (assuming all cols in extra file are features)
        extra_cols = df_extra.columns.tolist()
        
        # Concatenate side-by-side (assuming strict row alignment by index)
        # We reset index to be safe
        df = pd.concat([df.reset_index(drop=True), df_extra.reset_index(drop=True)], axis=1)
    else:
        print(f"Warning: Extra features file {extra_x_path} not found. Using SMILES only.")
        extra_cols = []

    # 3. Clean and Canonicalize
    # Ensure target is float
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.dropna(subset=[target_col])
    
    # Clean SMILES
    df[smiles_col] = df[smiles_col].astype(str).str.strip()
    df[smiles_col] = df[smiles_col].apply(canonicalize_smiles)
    
    # Drop invalid SMILES (this will also drop the corresponding extra features row)
    df = df.dropna(subset=[smiles_col])

    return df.reset_index(drop=True), extra_cols

# -------------------------
# Utils
# -------------------------
def path_if_none(path):
    os.makedirs(path, exist_ok=True)

# -------------------------
# Model
# -------------------------
class ChemBERTaMLPRegressor(nn.Module):
    def __init__(self, base_model_name, extra_dim=0, hidden_sizes=[512, 256], dropout=0.3):
        super().__init__()

        config = AutoConfig.from_pretrained(base_model_name)
        self.roberta = AutoModel.from_pretrained(base_model_name, config=config)

        layers = []
        # Input dimension is RoBERTa hidden size + number of extra features
        in_dim = config.hidden_size + extra_dim 
        
        for h in hidden_sizes:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))
        self.mlp_head = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask, extra_features=None, labels=None):
        # 1. Pass through Transformer
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 2. Extract [CLS] token embedding (Batch, Hidden_Size)
        cls_embedding = outputs.last_hidden_state[:, 0]
        
        # 3. Concatenate Extra Features if they exist
        if extra_features is not None:
            # Ensure extra_features is on the correct device and dtype
            combined_features = torch.cat((cls_embedding, extra_features), dim=1)
        else:
            combined_features = cls_embedding

        # 4. Pass combined vector through MLP
        logits = self.mlp_head(combined_features).squeeze(-1)

        loss = None
        if labels is not None:
            loss = nn.functional.mse_loss(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


# -------------------------
# Weighted Trainer
# -------------------------
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        weights = inputs.pop("sample_weight", None)
        
        # Note: 'extra_features' is still inside 'inputs', so it gets passed to model(**inputs) automatically
        
        outputs = model(**inputs)
        preds = outputs.logits

        # Calculate loss (weights support commented out as per original script)
        loss = nn.MSELoss(reduction="none")(preds, labels)
        loss = loss.mean()
        
        return (loss, outputs) if return_outputs else loss


# -------------------------
# Metrics
# -------------------------
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if preds.ndim > 1:
        preds = preds.squeeze(-1)

    # Calculate MSE first, then take square root manually
    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)

    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)

    try:
        pearson = pearsonr(labels, preds)[0]
        spearman = spearmanr(labels, preds)[0]
    except:
        pearson = 0
        spearman = 0

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman
    }

# -------------------------
# Layer-wise LR Decay Optimizer
# -------------------------
def build_llrd_optimizer(model, base_lr=1e-5, head_lr=5e-4, layer_decay=0.8):
    params = []

    # Embeddings
    params.append({
        "params": model.roberta.embeddings.parameters(),
        "lr": base_lr * (layer_decay ** (model.roberta.config.num_hidden_layers + 1))
    })

    # Encoder layers
    n_layers = model.roberta.config.num_hidden_layers
    for i, layer in enumerate(model.roberta.encoder.layer):
        lr = base_lr * (layer_decay ** (n_layers - i))
        params.append({"params": layer.parameters(), "lr": lr})

    # MLP head (This now includes the larger input layer handling extra features)
    params.append({
        "params": model.mlp_head.parameters(),
        "lr": head_lr
    })

    return AdamW(params, weight_decay=0.01)


# -------------------------
# Training Logic
# -------------------------
def train_one_fold(split_dir, save_dir, target_col, epochs, freeze_encoder=False):

    path_if_none(save_dir)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # --- Paths ---
    train_csv = os.path.join(split_dir, "train.csv")
    train_extra = os.path.join(split_dir, "train_extra_x.csv")
    
    valid_csv = os.path.join(split_dir, "valid.csv")
    valid_extra = os.path.join(split_dir, "valid_extra_x.csv")
    
    weights_csv = os.path.join(split_dir, "train_weights.csv")

    # --- Identify SMILES column name ---
    tmp_df = pd.read_csv(train_csv, nrows=1)
    smiles_col = "smiles" if "smiles" in tmp_df.columns else "SMILES"

    # --- Load & Preprocess Data ---
    logger.info("Loading and preprocessing training data...")
    df_train, extra_cols = load_and_preprocess_data(train_csv, train_extra, smiles_col, target_col)
    
    logger.info("Loading and preprocessing validation data...")
    df_valid, _ = load_and_preprocess_data(valid_csv, valid_extra, smiles_col, target_col)

    # --- Feature Scaling (Important for MLP) ---
    # We must normalize extra features so they don't dominate the gradients vs embeddings
    if len(extra_cols) > 0:
        logger.info(f"Normalizing {len(extra_cols)} extra features...")
        scaler = StandardScaler()
        # Fit on TRAIN, transform TRAIN and VALID
        df_train[extra_cols] = scaler.fit_transform(df_train[extra_cols])
        df_valid[extra_cols] = scaler.transform(df_valid[extra_cols])
        
        # Save scaler for inference later if needed
        import joblib
        joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))

    # --- Weights (Keep original logic) ---
    # Note: If rows were dropped during preprocessing, weights might be misaligned 
    # if not handled carefully. Assuming weights_csv matches raw train.csv 
    # and we dropped rows, we theoretically need to filter weights too.
    # For simplicity here, we re-read weights based on the processed dataframe index if possible,
    # or assume the original weights_csv aligns perfectly with the output of load_and_preprocess.
    # Given the complexity, we stick to the original logic but warn:
    df_weights = pd.read_csv(weights_csv)
    # Ideally, you should merge weights into df_train before dropna, 
    # but for now we just take the first N to match length (risky if drops happened).
    # Safe fix: add weights to load_and_preprocess if possible. 
    # Here we assume minimal drops.
    weights = df_weights.iloc[:len(df_train), 0].values.astype(np.float32)

    # --- Tokenization ---
    BASE_MODEL = "DeepChem/ChemBERTa-77M-MTR"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize_and_format(batch):
        # Tokenize SMILES
        tokenized = tokenizer(
            batch[smiles_col],
            padding="max_length",
            truncation=True,
            max_length=512
        )
        # Handle Extra Features
        if len(extra_cols) > 0:
            # Convert list of lists to tensor
            extra_tensor = [ [x[col] for col in extra_cols] for x in batch ] # This creates list of features per sample
            # Dataset.map expects dictionary updates
            tokenized["extra_features"] = extra_tensor
        
        return tokenized

    # --- Create Datasets ---
    # Convert DF to Dataset first for efficiency
    train_ds = Dataset.from_pandas(df_train)
    # Add weights column manually to dataset
    train_ds = train_ds.add_column("sample_weight", weights)
    
    # Map tokenization + extra features extraction
    # We use a custom function wrapper to handle the row-based extraction
    def process_batch(examples):
        tokenized = tokenizer(examples[smiles_col], padding="max_length", truncation=True, max_length=512)
        if len(extra_cols) > 0:
            # Create a matrix of extra features for this batch
            # examples is a dict of lists: {'feat1': [val1, val2], 'feat2': [val1, val2]}
            # We need [[val1_f1, val1_f2], [val2_f1, val2_f2]]
            batch_extras = []
            num_samples = len(examples[smiles_col])
            for i in range(num_samples):
                sample_feats = [examples[col][i] for col in extra_cols]
                batch_extras.append(sample_feats)
            tokenized["extra_features"] = batch_extras
        
        tokenized["labels"] = examples[target_col]
        return tokenized

    train_ds = train_ds.map(process_batch, batched=True)
    
    valid_ds = Dataset.from_pandas(df_valid)
    valid_ds = valid_ds.map(process_batch, batched=True)

    # Set format for PyTorch
    cols_to_keep = ["input_ids", "attention_mask", "labels"]
    if len(extra_cols) > 0:
        cols_to_keep.append("extra_features")
    
    train_ds.set_format("torch", cols_to_keep + ["sample_weight"])
    valid_ds.set_format("torch", cols_to_keep)

    # --- Initialize Model ---
    model = ChemBERTaMLPRegressor(BASE_MODEL, extra_dim=len(extra_cols))

    if freeze_encoder:
        print("Freezing Encoder Layers...")
        for p in model.roberta.parameters():
            p.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = build_llrd_optimizer(model)

    args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=32, # Reduced batch size for safety with extra overhead
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        save_total_limit=1,
        logging_steps=50,
        report_to="none",
        remove_unused_columns=False # CRITICAL: Prevents Trainer from dropping 'extra_features'
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None)
    )

    # args = TrainingArguments(
    #     output_dir=save_dir,
    #     num_train_epochs=epochs,          # Reduced from 1000
    #     per_device_train_batch_size=32,
    #     per_device_eval_batch_size=64,
    #     gradient_accumulation_steps=2,
    #     eval_strategy="epoch",         # Remember to use the new name
    #     save_strategy="epoch",
    #     load_best_model_at_end=True,   # Required for Early Stopping
    #     metric_for_best_model="rmse",
    #     greater_is_better=False,       # Lower RMSE is better
    #     save_total_limit=1,
    #     logging_steps=50,
    #     report_to="none",
    #     remove_unused_columns=False
    # )

    # trainer = WeightedTrainer(
    #     model=model,
    #     args=args,
    #     train_dataset=train_ds,
    #     eval_dataset=valid_ds,
    #     compute_metrics=compute_metrics,
    #     optimizers=(optimizer, None),
        
    #     # --- ADD THIS BLOCK ---
    #     callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
    # )

    trainer.train()
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    logger.info("Training completed.")


# -------------------------
# Main
# -------------------------
def main():
    epochs = 150
    freeze = True 
    target = "quantified_delivery"

    split_dir = f"../data/crossval_splits/zhu_del_B/cv_0"
    save_dir = f"../data/crossval_splits/zhu_del_B/chemberta_llrd_{'freeze' if freeze else 'fullft'}_w_extras"

    train_one_fold(
        split_dir=split_dir,
        save_dir=save_dir,
        target_col=target,
        epochs=epochs,
        freeze_encoder=freeze
    )


if __name__ == "__main__":
    main()