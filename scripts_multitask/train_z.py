#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import AdamW

from datasets import Dataset
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

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
# Utils & Preprocessing
# -------------------------
def path_if_none(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def canonicalize_smiles(s):
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None

def load_and_preprocess_data(main_csv_path, extra_x_path, weights_path, smiles_col, target_col):
    """
    Loads main CSV, Extra Features, and Weights.
    Merges them and cleans based on valid SMILES.
    """
    # 1. Load Main Data
    if not os.path.exists(main_csv_path):
        raise FileNotFoundError(f"Main data not found: {main_csv_path}")
    df = pd.read_csv(main_csv_path)
    
    # 2. Load Extra Features
    if os.path.exists(extra_x_path):
        df_extra = pd.read_csv(extra_x_path)
        if len(df) != len(df_extra):
            raise ValueError(f"Row mismatch: {main_csv_path} ({len(df)}) vs {extra_x_path} ({len(df_extra)})")
        extra_cols = df_extra.columns.tolist()
        # Concatenate side-by-side
        df = pd.concat([df.reset_index(drop=True), df_extra.reset_index(drop=True)], axis=1)
    else:
        print(f"Warning: Extra features file {extra_x_path} not found. Using SMILES only.")
        extra_cols = []

    # 3. Load Weights
    if os.path.exists(weights_path):
        df_weights = pd.read_csv(weights_path)
        if len(df) != len(df_weights):
             raise ValueError(f"Row mismatch: {main_csv_path} ({len(df)}) vs {weights_path} ({len(df_weights)})")
        # Rename weight column to generic 'weight' for internal tracking
        weight_col_name = df_weights.columns[0]
        df['sample_weight'] = df_weights[weight_col_name].values
    else:
        print(f"Warning: Weights file {weights_path} not found. Using uniform weights.")
        df['sample_weight'] = 1.0

    # 4. Clean and Canonicalize
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.dropna(subset=[target_col])
    
    df[smiles_col] = df[smiles_col].astype(str).str.strip()
    df[smiles_col] = df[smiles_col].apply(canonicalize_smiles)
    
    # Drop invalid SMILES
    df = df.dropna(subset=[smiles_col])

    return df.reset_index(drop=True), extra_cols

# -------------------------
# Custom Model Architecture (Script 1)
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
        
        # 'extra_features' remains in inputs and is passed to model(**inputs)
        outputs = model(**inputs)
        preds = outputs.logits

        # Calculate weighted loss manually
        loss_fct = nn.MSELoss(reduction="none")
        loss = loss_fct(preds, labels)

        if weights is not None:
            weights = weights.to(loss.device)
            loss = (loss * weights).mean()
        else:
            loss = loss.mean()
        
        return (loss, outputs) if return_outputs else loss

# -------------------------
# Metrics (Simplified for Model Selection)
# -------------------------
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if preds.ndim > 1:
        preds = preds.squeeze(-1)

    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)

    return {"rmse": rmse}

# -------------------------
# Optimizer (LLRD)
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

    # MLP head
    params.append({
        "params": model.mlp_head.parameters(),
        "lr": head_lr
    })

    return AdamW(params, weight_decay=0.01)

# -------------------------
# Training Logic (Per Fold)
# -------------------------
def train_fold(split_dir, save_dir, cv_fold, target_col, epochs, freeze_encoder=True):
    path_if_none(save_dir)
    
    # Logging Setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info(f"Fold {cv_fold}: Loading data...")

    # --- Paths ---
    train_csv = os.path.join(split_dir, "train.csv")
    valid_csv = os.path.join(split_dir, "valid.csv")
    train_extra = os.path.join(split_dir, "train_extra_x.csv")
    valid_extra = os.path.join(split_dir, "valid_extra_x.csv")
    train_weights = os.path.join(split_dir, "train_weights.csv")
    
    # Need a dummy weights path for valid just to satisfy function arg, 
    # but we won't use weights in valid usually.
    valid_weights = os.path.join(split_dir, "valid_weights.csv") 

    # --- Identify SMILES column ---
    # Peek at header
    tmp_df = pd.read_csv(train_csv, nrows=1)
    smiles_col = "smiles" if "smiles" in tmp_df.columns else "SMILES"

    # --- Load Data ---
    df_train, extra_cols = load_and_preprocess_data(train_csv, train_extra, train_weights, smiles_col, target_col)
    
    # For validation, we might not have a weights file, so we pass a dummy path.
    # The loader will assign uniform weights (1.0) if missing.
    df_valid, _ = load_and_preprocess_data(valid_csv, valid_extra, valid_weights, smiles_col, target_col)

    # --- Feature Scaling ---
    if len(extra_cols) > 0:
        logger.info(f"Normalizing {len(extra_cols)} extra features...")
        scaler = StandardScaler()
        df_train[extra_cols] = scaler.fit_transform(df_train[extra_cols])
        df_valid[extra_cols] = scaler.transform(df_valid[extra_cols])
        
        # Save Scaler (Naming convention from Script 2)
        with open(os.path.join(save_dir, "extra_features_scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

    # --- Tokenization ---
    BASE_MODEL = "DeepChem/ChemBERTa-77M-MTR"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def process_batch(examples):
        tokenized = tokenizer(
            examples[smiles_col], 
            padding="max_length", 
            truncation=True, 
            max_length=512
        )
        
        if len(extra_cols) > 0:
            # Create matrix of extra features for this batch
            batch_extras = []
            num_samples = len(examples[smiles_col])
            for i in range(num_samples):
                sample_feats = [examples[col][i] for col in extra_cols]
                batch_extras.append(sample_feats)
            tokenized["extra_features"] = batch_extras
        
        tokenized["labels"] = examples[target_col]
        # Weights are already in dataset as 'sample_weight' from dataframe
        return tokenized

    # Convert to Dataset
    train_ds = Dataset.from_pandas(df_train)
    valid_ds = Dataset.from_pandas(df_valid)

    train_ds = train_ds.map(process_batch, batched=True)
    valid_ds = valid_ds.map(process_batch, batched=True)

    # Set Torch Format
    cols_to_keep = ["input_ids", "attention_mask", "labels"]
    if len(extra_cols) > 0:
        cols_to_keep.append("extra_features")
    
    # Ensure sample_weight is kept for training
    train_ds.set_format("torch", cols_to_keep + ["sample_weight"])
    valid_ds.set_format("torch", cols_to_keep) # No weights needed for eval usually

    # --- Initialize Model ---
    model = ChemBERTaMLPRegressor(BASE_MODEL, extra_dim=len(extra_cols))

    if freeze_encoder:
        logger.info("Freezing Encoder Layers (Training MLP Head Only)...")
        for p in model.roberta.parameters():
            p.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- Optimizer & Training Args ---
    optimizer = build_llrd_optimizer(model)

    args = TrainingArguments(
        output_dir=os.path.join(save_dir, "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,          # Base LR (overridden by LLRD optimizer)
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        save_total_limit=1,
        logging_steps=50,
        report_to="none",
        remove_unused_columns=False  # CRITICAL: Keeps 'extra_features'
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None)
    )

    logger.info(f"Starting training for {epochs} epochs...")
    trainer.train()
    
    # Save Final Model
    final_model_path = os.path.join(save_dir, "fine_tuned_chemberta")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    logger.info(f"Fold {cv_fold} completed. Model saved to {final_model_path}")

# -------------------------
# Main (CLI Logic from Script 2)
# -------------------------
def main(argv):
    if len(argv) < 2:
        print("Usage: python train_script.py {split_name} [--epochs N] [--cv N]")
        sys.exit(1)

    split_folder = argv[1]
    
    # Determine Target
    if 'del' in split_folder.lower():
        target_col = 'quantified_delivery'
        print("Training for Delivery (quantified_delivery)")
    elif 'tox' in split_folder.lower():
        target_col = 'quantified_toxicity'
        print("Training for Toxicity (quantified_toxicity)")
    else:
        print(f"Error: Could not infer mode (del/tox) from '{split_folder}'.")
        sys.exit(1)

    # Defaults
    epochs = 100
    cv_num = 5
    freeze = True 

    # Parse Arguments
    for i, arg in enumerate(argv):
        if arg.replace('–', '-') == '--epochs':
            epochs = int(argv[i+1])
            print('Epochs set to:', str(epochs))
        if arg.replace('–', '-') == '--cv':
            cv_num = int(argv[i+1])
            print('CV folds set to:', str(cv_num))

    # Cross-Validation Loop
    for cv in range(cv_num):
        split_dir = f'../data/crossval_splits/{split_folder}/cv_{cv}'
        save_dir = f'{split_dir}/model_{cv}'
        
        print(f"\n=== Processing CV Split {cv} ===")
        print(f"Input: {split_dir}")
        print(f"Output: {save_dir}")
        
        train_fold(
            split_dir=split_dir,
            save_dir=save_dir,
            cv_fold=cv,
            target_col=target_col,
            epochs=epochs,
            freeze_encoder=freeze
        )

if __name__ == "__main__":
    main(sys.argv)