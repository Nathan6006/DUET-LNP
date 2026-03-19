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
# Preprocessing
# -------------------------
def canonicalize_smiles(s):
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def preprocess_csv(csv_path, smiles_col, target_col):
    df = pd.read_csv(csv_path)
    df = df[[smiles_col, target_col]].dropna()

    df[smiles_col] = df[smiles_col].astype(str).str.strip()
    df[target_col] = df[target_col].astype(np.float32)

    df[smiles_col] = df[smiles_col].apply(canonicalize_smiles)
    df = df.dropna(subset=[smiles_col])

    return df.reset_index(drop=True)
# -------------------------
# Utils
# -------------------------
def path_if_none(path):
    os.makedirs(path, exist_ok=True)


# -------------------------
# Model
# -------------------------
class ChemBERTaMLPRegressor(nn.Module):
    def __init__(self, base_model_name, hidden_sizes=[512, 32], dropout=0.3):
        super().__init__()

        config = AutoConfig.from_pretrained(base_model_name)
        self.roberta = AutoModel.from_pretrained(base_model_name, config=config)

        layers = []
        in_dim = config.hidden_size
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

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = outputs.last_hidden_state[:, 0]
        logits = self.mlp_head(cls).squeeze(-1)

        loss = None
        if labels is not None:
            loss = nn.functional.mse_loss(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


# -------------------------
# Weighted Trainer
# -------------------------
class WeightedTrainer(Trainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,   #这一行
    ):
        labels = inputs.pop("labels")
        weights = inputs.pop("sample_weight", None)

        outputs = model(**inputs)
        preds = outputs.logits

        loss = nn.MSELoss(reduction="none")(preds, labels)
        # if weights is not None:
        #     loss = (loss * weights).mean()
        # else:
        #     loss = loss.mean()
        
        loss = loss.mean()
        return (loss, outputs) if return_outputs else loss


# -------------------------
# Metrics
# -------------------------
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if preds.ndim > 1:
        preds = preds.squeeze(-1)

    rmse = mean_squared_error(labels, preds, squared=False)
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)

    pearson = pearsonr(labels, preds)[0]
    spearman = spearmanr(labels, preds)[0]

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

    # Embeddings (smallest LR)
    params.append({
        "params": model.roberta.embeddings.parameters(),
        "lr": base_lr * (layer_decay ** (model.roberta.config.num_hidden_layers + 1))
    })

    # Encoder layers (bottom → top)
    n_layers = model.roberta.config.num_hidden_layers
    for i, layer in enumerate(model.roberta.encoder.layer):
        lr = base_lr * (layer_decay ** (n_layers - i))
        params.append({"params": layer.parameters(), "lr": lr})

    # MLP head (largest LR)
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

    train_csv = os.path.join(split_dir, "train.csv")
    valid_csv = os.path.join(split_dir, "valid.csv")
    weights_csv = os.path.join(split_dir, "train_weights.csv")

    # 自动识别 smiles 列名
    tmp_df = pd.read_csv(train_csv, nrows=1)
    smiles_col = "smiles" if "smiles" in tmp_df.columns else "SMILES"

    # ✅ 新增：使用 preprocess_csv
    df_train = preprocess_csv(train_csv, smiles_col, target_col)
    df_valid = preprocess_csv(valid_csv, smiles_col, target_col)

    # 权重保持原样（不动）
    df_weights = pd.read_csv(weights_csv)
    weights = df_weights.iloc[:, 0].values.astype(np.float32)

    BASE_MODEL = "DeepChem/ChemBERTa-77M-MTR"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(batch):
        return tokenizer(
            batch[smiles_col],
            padding="max_length",
            truncation=True,
            max_length=512
        )

    train_ds = Dataset.from_pandas(df_train)
    train_ds = train_ds.add_column("sample_weight", weights)
    train_ds = train_ds.map(tokenize, batched=True)
    train_ds = train_ds.map(lambda x: {"labels": x[target_col]}, batched=True)

    valid_ds = Dataset.from_pandas(df_valid)
    valid_ds = valid_ds.map(tokenize, batched=True)
    valid_ds = valid_ds.map(lambda x: {"labels": x[target_col]}, batched=True)

    train_ds.set_format("torch", ["input_ids", "attention_mask", "labels", "sample_weight"])
    valid_ds.set_format("torch", ["input_ids", "attention_mask", "labels"])

    model = ChemBERTaMLPRegressor(BASE_MODEL)

    if freeze_encoder:
        for p in model.roberta.parameters():
            p.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = build_llrd_optimizer(model)

    args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1024,
        per_device_eval_batch_size=1024,
        gradient_accumulation_steps=2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        save_total_limit=1,
        logging_steps=50,
        report_to="none"
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None)
    )

    trainer.train()
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    logger.info("Training completed.")



# -------------------------
# Main
# -------------------------
def main():
    epochs = 2000
    freeze = True 
    target = "quantified_delivery"

    split_dir = f"./Files/zhu_del_B/cv_4"
    save_dir = f"./chemberta_llrd_{'freeze_4' if freeze else 'fullft'}"

    train_one_fold(
        split_dir=split_dir,
        save_dir=save_dir,
        target_col=target,
        epochs=epochs,
        freeze_encoder=freeze
    )


if __name__ == "__main__":
    main()
