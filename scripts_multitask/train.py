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
    TrainingArguments,
    EarlyStoppingCallback
)
from transformers.modeling_outputs import SequenceClassifierOutput

# -------------------------
# Configuration Constants
# -------------------------
BASE_MODEL = "DeepChem/ChemBERTa-77M-MTR"
MAX_LENGTH = 256  # Reduced from 512 for speed
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_EVAL = 64
DEFAULT_EPOCHS = 80
EARLY_STOPPING_PATIENCE = 20

# -------------------------
# Custom Model Architecture
# -------------------------
class ChemBERTaMLPRegressor(nn.Module):
    def __init__(self, base_model_name, extra_dim=0, hidden_sizes=[512, 256], dropout=0.3):
        super().__init__()
        
        config = AutoConfig.from_pretrained(base_model_name)
        self.roberta = AutoModel.from_pretrained(base_model_name, config=config)

        # Input dimension = RoBERTa hidden size + extra features count
        in_dim = config.hidden_size + extra_dim 
        layers = []
        
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))
        self.mlp_head = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask, extra_features=None, labels=None, **kwargs):
        # 1. Transformer Forward Pass
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        
        # 2. Extract [CLS] token (first token of last hidden state)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # 3. Concatenate Extra Features
        if extra_features is not None:
            combined_features = torch.cat((cls_embedding, extra_features), dim=1)
        else:
            combined_features = cls_embedding

        # 4. MLP Prediction
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
        
        # Forward pass
        outputs = model(**inputs)
        preds = outputs.logits

        # Weighted MSE Loss
        loss_fct = nn.MSELoss(reduction="none")
        loss = loss_fct(preds, labels)

        if weights is not None:
            weights = weights.to(loss.device)
            loss = (loss * weights).mean()
        else:
            loss = loss.mean()
        
        return (loss, outputs) if return_outputs else loss

# -------------------------
# Utils & Preprocessing
# -------------------------
def load_and_preprocess_data(main_csv, extra_csv, weights_csv, target_col):
    """
    Efficient data loader. Assumes SMILES are already canonical/valid.
    """
    # 1. Load Main Data
    if not os.path.exists(main_csv):
        raise FileNotFoundError(f"Missing: {main_csv}")
    df = pd.read_csv(main_csv)
    
    # Detect SMILES column (case-insensitive)
    cols = df.columns.tolist()
    smiles_col = next((c for c in cols if c.lower() == 'smiles'), None)
    if not smiles_col:
        raise ValueError("No 'smiles' column found in CSV.")

    # 2. Load Extra Features (Horizontal Merge)
    if os.path.exists(extra_csv):
        df_extra = pd.read_csv(extra_csv)
        if len(df) != len(df_extra):
            raise ValueError(f"Row mismatch: {len(df)} vs {len(df_extra)}")
        df = pd.concat([df.reset_index(drop=True), df_extra.reset_index(drop=True)], axis=1)
        extra_cols = df_extra.columns.tolist()
    else:
        extra_cols = []

    # 3. Load Weights
    if os.path.exists(weights_csv):
        df_weights = pd.read_csv(weights_csv)
        weight_col = df_weights.columns[0]
        df['sample_weight'] = df_weights[weight_col].values
    else:
        df['sample_weight'] = 1.0

    # 4. Basic Cleaning (No RDKit)
    df = df.dropna(subset=[target_col, smiles_col])
    df[smiles_col] = df[smiles_col].astype(str).str.strip()
    
    return df.reset_index(drop=True), extra_cols, smiles_col

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if preds.ndim > 1: preds = preds.squeeze()
    mse = mean_squared_error(labels, preds)
    return {"rmse": np.sqrt(mse)}

def build_llrd_optimizer(model, base_lr=1e-5, head_lr=5e-4, layer_decay=0.9):
    """Layer-wise Learning Rate Decay Optimizer"""
    params = []
    
    # 1. Embeddings
    params.append({
        "params": model.roberta.embeddings.parameters(),
        "lr": base_lr * (layer_decay ** (model.roberta.config.num_hidden_layers + 1))
    })

    # 2. Encoder Layers
    n_layers = model.roberta.config.num_hidden_layers
    for i, layer in enumerate(model.roberta.encoder.layer):
        # Higher layers (closer to output) get higher LR
        lr = base_lr * (layer_decay ** (n_layers - 1 - i))
        params.append({"params": layer.parameters(), "lr": lr})

    # 3. MLP Head
    params.append({"params": model.mlp_head.parameters(), "lr": head_lr})

    return AdamW(params, weight_decay=0.01)

# -------------------------
# Training Logic
# -------------------------
def train_fold(split_dir, save_dir, cv_fold, target_col, epochs, freeze_encoder=True):
    os.makedirs(save_dir, exist_ok=True)
    
    # Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(f"Fold {cv_fold}: Loading data from {split_dir}")

    # File Paths
    files = {
        "train": os.path.join(split_dir, "train.csv"),
        "valid": os.path.join(split_dir, "valid.csv"),
        "train_x": os.path.join(split_dir, "train_extra_x.csv"),
        "valid_x": os.path.join(split_dir, "valid_extra_x.csv"),
        "train_w": os.path.join(split_dir, "train_weights.csv"),
        "valid_w": os.path.join(split_dir, "valid_weights.csv")
    }

    # Load Data
    df_train, extra_cols, smiles_col = load_and_preprocess_data(files["train"], files["train_x"], files["train_w"], target_col)
    df_valid, _, _ = load_and_preprocess_data(files["valid"], files["valid_x"], files["valid_w"], target_col)

    # Feature Scaling
    if extra_cols:
        scaler = StandardScaler()
        df_train[extra_cols] = scaler.fit_transform(df_train[extra_cols])
        df_valid[extra_cols] = scaler.transform(df_valid[extra_cols])
        with open(os.path.join(save_dir, "extra_features_scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize_function(examples):
        # 1. Tokenize SMILES
        tokenized = tokenizer(
            examples[smiles_col],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )
        
        # 2. Add Extra Features
        if extra_cols:
            batch_extras = []
            for i in range(len(examples[smiles_col])):
                row_feats = [examples[col][i] for col in extra_cols]
                batch_extras.append(row_feats)
            tokenized["extra_features"] = batch_extras
            
        tokenized["labels"] = examples[target_col]
        return tokenized

    # Create Datasets
    train_ds = Dataset.from_pandas(df_train).map(tokenize_function, batched=True)
    valid_ds = Dataset.from_pandas(df_valid).map(tokenize_function, batched=True)

    # Set Format
    keep_cols = ["input_ids", "attention_mask", "labels", "sample_weight"]
    if extra_cols: keep_cols.append("extra_features")
    
    train_ds.set_format("torch", columns=keep_cols)
    # Validation doesn't need weights for metrics
    valid_ds.set_format("torch", columns=[c for c in keep_cols if c != "sample_weight"])

    # Model Setup
    model = ChemBERTaMLPRegressor(BASE_MODEL, extra_dim=len(extra_cols))
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    if freeze_encoder:
        logger.info("Freezing Encoder Layers")
        for param in model.roberta.parameters():
            param.requires_grad = False

    # Trainer Config
    args = TrainingArguments(
        output_dir=os.path.join(save_dir, "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE_TRAIN,
        per_device_eval_batch_size=BATCH_SIZE_EVAL,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,  # Base LR (overridden by optimizer)
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        save_total_limit=1,
        logging_steps=50,
        report_to="none",
        remove_unused_columns=False
    )

    optimizer = build_llrd_optimizer(model)

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
    )

    logger.info("Starting Training...")
    trainer.train()

    # Save
    final_path = os.path.join(save_dir, "fine_tuned_chemberta")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Model saved to {final_path}")

# -------------------------
# CLI Entry Point
# -------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python train_script.py {split_folder} [--epochs N] [--cv N]")
        sys.exit(1)

    split_folder = sys.argv[1]
    
    # Auto-detect target
    if 'del' in split_folder.lower():
        target = 'quantified_delivery'
        print("Mode: Delivery")
    elif 'tox' in split_folder.lower():
        target = 'quantified_toxicity'
        print("Mode: Toxicity")
    else:
        print("Error: Directory must contain 'del' or 'tox'.")
        sys.exit(1)

    # Parse Args
    epochs = DEFAULT_EPOCHS
    cv_folds = 5
    
    args = sys.argv[2:]
    for i, arg in enumerate(args):
        if arg in ['--epochs', '-e']:
            epochs = int(args[i+1])
        if arg in ['--cv', '-c']:
            cv_folds = int(args[i+1])

    print(f"Config: Epochs={epochs}, CV={cv_folds}, TokenLen={MAX_LENGTH}, Patience={EARLY_STOPPING_PATIENCE}")

    for cv in range(cv_folds):
        split_dir = f'../data/crossval_splits/{split_folder}/cv_{cv}'
        save_dir = f'{split_dir}/model_{cv}'
        
        print(f"\n=== CV Split {cv} ===")
        train_fold(split_dir, save_dir, cv, target, epochs)

if __name__ == "__main__":
    main()