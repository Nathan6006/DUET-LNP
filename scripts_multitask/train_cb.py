import numpy as np 
import os
import pandas as pd  
import sys
import logging
import torch
import xgboost as xgb
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    AutoConfig
)
from datasets import Dataset


def path_if_none(path):
    if not os.path.exists(path):
        os.makedirs(path)

# --- Custom Classes ---

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        weights = inputs.pop("sample_weight", None)
        
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if logits.shape[-1] == 1:
            logits = logits.squeeze()
            
        loss_fct = torch.nn.MSELoss(reduction='none')
        loss = loss_fct(logits, labels)
        
        if weights is not None:
            weights = weights.to(loss.device)
            loss = (loss * weights).mean()
        else:
            loss = loss.mean()
            
        return (loss, outputs) if return_outputs else loss

class LogCoshObjective:
    def __init__(self):
        pass

    def __call__(self, y_true, y_pred, sample_weight=None):
        x = y_pred - y_true
        grad = np.tanh(x)
        hess = 1.0 - (grad ** 2)
        hess = np.maximum(hess, 1e-6)
        
        if sample_weight is not None:
            grad = grad * sample_weight
            hess = hess * sample_weight
            
        return grad, hess

# --- Core Logic ---

def get_embeddings(model, tokenizer, smiles_list, device, max_length=128):
    model.to("cpu")
    model.eval()
    
    embeddings = []
    batch_size = 32
    
    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i:i+batch_size]
        
        inputs = tokenizer(
            batch_smiles, 
            padding=True, 
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        ).to("cpu") 
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            cls_embeddings = outputs.hidden_states[-1][:, 0, :].numpy()
            embeddings.append(cls_embeddings)
            
    model.to(device)
    return np.vstack(embeddings)

def train_hybrid(split_dir, save_dir, cv_fold, target_columns, epochs=3, freeze_until=3):
    path_if_none(save_dir)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # 1. Load Data
    try:
        df_train = pd.read_csv(os.path.join(split_dir, "train.csv"))
        df_valid = pd.read_csv(os.path.join(split_dir, "valid.csv"))
        df_train_extra = pd.read_csv(os.path.join(split_dir, "train_extra_x.csv"))
        df_valid_extra = pd.read_csv(os.path.join(split_dir, "valid_extra_x.csv"))
        df_train_weights = pd.read_csv(os.path.join(split_dir, "train_weights.csv"))
        
        smiles_col = 'smiles' if 'smiles' in df_train.columns else 'SMILES'
        target_col = target_columns[0]
        weight_col = df_train_weights.columns[0] 
        train_weights = df_train_weights[weight_col].values.astype(np.float32)
        
    except FileNotFoundError as e:
        logger.error(f"Missing file in {split_dir}: {e}")
        return

    # 2. Scale Extra Features
    if not df_train_extra.empty:
        scaler = StandardScaler()
        X_extra_train = df_train_extra.select_dtypes(include=[np.number]).values
        X_extra_valid = df_valid_extra.select_dtypes(include=[np.number]).values
        scaler.fit(X_extra_train)
        
        with open(os.path.join(save_dir, "extra_features_scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        
        X_extra_train = scaler.transform(X_extra_train)
        X_extra_valid = scaler.transform(X_extra_valid)
    else:
        X_extra_train = np.array([])
        X_extra_valid = np.array([])

    # 3. Fine-Tune ChemBERTa (With Frozen Layers & Dropout)
    logger.info(f"Fold {cv_fold}: Fine-tuning ChemBERTa on {target_col}...")
    
    BASE_MODEL = "DeepChem/ChemBERTa-77M-MTR"
    MAX_LEN = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # CONFIG CHANGE: Increase Dropout
    config = AutoConfig.from_pretrained(BASE_MODEL)
    config.hidden_dropout_prob = 0.2        
    config.attention_probs_dropout_prob = 0.2
    config.num_labels = 1
    
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, config=config)

    total_layers = len(model.roberta.encoder.layer)

    # FREEZE LAYERS
    for name, param in model.roberta.encoder.layer.named_parameters():
        layer_num = int(name.split('.')[0])
        if layer_num < freeze_until: 
            param.requires_grad = False
    print(f"Froze {freeze_until} out of {total_layers} layers")
            
    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False
        
    model.to(device)

    def tokenize_function(examples):
        return tokenizer(examples[smiles_col], padding="max_length", truncation=True, max_length=MAX_LEN)

    train_ds = Dataset.from_pandas(df_train)
    train_ds = train_ds.add_column("sample_weight", train_weights)
    train_ds = train_ds.map(tokenize_function, batched=True)
    train_ds = train_ds.map(lambda x: {'labels': x[target_col]}, batched=True)
    
    valid_ds = Dataset.from_pandas(df_valid)
    valid_ds = valid_ds.map(tokenize_function, batched=True)
    valid_ds = valid_ds.map(lambda x: {'labels': x[target_col]}, batched=True)
    
    train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'sample_weight'])
    valid_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    training_args = TrainingArguments(
        output_dir=os.path.join(save_dir, "chemberta_checkpoints"),
        
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,  
        
        learning_rate=2e-5,           
        lr_scheduler_type="cosine",     
        warmup_ratio=0.1,               
        weight_decay=0.05,              
        
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=epochs,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,             # Saves disk space on your Mac
        
        logging_dir=os.path.join(save_dir, 'logs'),
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
    )

    trainer.train()
    
    ft_model_path = os.path.join(save_dir, "fine_tuned_chemberta")
    trainer.save_model(ft_model_path)
    tokenizer.save_pretrained(ft_model_path)

    logger.info(f"Fold {cv_fold}: Extracting Embeddings")
    train_emb = get_embeddings(model, tokenizer, df_train[smiles_col].tolist(), device)
    valid_emb = get_embeddings(model, tokenizer, df_valid[smiles_col].tolist(), device)


    logger.info(f"Fold {cv_fold}: Training XGBoost")
    
    if X_extra_train.size > 0:
        X_train_hybrid = np.hstack([train_emb, X_extra_train])
        X_valid_hybrid = np.hstack([valid_emb, X_extra_valid])
    else:
        X_train_hybrid = train_emb
        X_valid_hybrid = valid_emb

    y_train = df_train[target_col].values
    y_valid = df_valid[target_col].values

    xgb_model = xgb.XGBRegressor(
        #objective=LogCoshObjective(), 
        n_estimators=1000,           
        learning_rate=0.03,          
        max_depth=6,               
        min_child_weight=5,          
        subsample=0.8,               
        colsample_bytree=0.8,
        gamma=1,        
        n_jobs=-1,
        early_stopping_rounds=50,
        tree_method="hist"
    )

    xgb_model.fit(
        X_train_hybrid, 
        y_train, 
        sample_weight=train_weights,
        eval_set=[(X_train_hybrid, y_train), (X_valid_hybrid, y_valid)],
        verbose=False
    )

    # Evaluate
    preds = xgb_model.predict(X_valid_hybrid)
    mse = mean_squared_error(y_valid, preds)
    mae = mean_absolute_error(y_valid, preds)
    logger.info(f"Fold {cv_fold} Final Validation MSE: {mse:.4f}, MAE: {mae:.4f}")

    xgb_save_path = os.path.join(save_dir, "hybrid_xgb.json")
    xgb_model.save_model(xgb_save_path)


def main(argv):
    if len(argv) < 2:
        print("Usage: python train_cb.py {split_name} [--epochs N] [--cv N]")
        sys.exit(1)

    split_folder = argv[1]
    
    if 'del' in split_folder.lower():
        target_cols = ['quantified_delivery'] 
        print("Training for Delivery (delivery)")
    elif 'tox' in split_folder.lower():
        target_cols = ['quantified_toxicity'] 
        print("Training for Toxicity (viability)")
    else:
        print(f"Error: Could not infer mode (del/tox) from '{split_folder}'.")
        sys.exit(1)

    epochs = 3 
    cv_num = 5
    
    for i, arg in enumerate(argv):
        if arg.replace('–', '-') == '--epochs':
            epochs = int(argv[i+1])
            print('this many epochs: ', str(epochs))
        if arg.replace('–', '-') == '--cv':
            cv_num = int(argv[i+1])
            print('this many folds: ', str(cv_num))

    for cv in range(cv_num):
        split_dir = f'../data/crossval_splits/{split_folder}/cv_{cv}'
        save_dir = f'{split_dir}/model_{cv}'
        
        train_hybrid(
            split_dir=split_dir, 
            save_dir=save_dir, 
            cv_fold=cv, 
            target_columns=target_cols,
            epochs=epochs
        )

if __name__ == '__main__':
    main(sys.argv)