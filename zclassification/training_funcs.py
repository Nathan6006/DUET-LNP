import numpy as np 
import os
import pandas as pd  
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, log_loss 
import pickle
from lightning import pytorch as pl 
import torch 
from chemprop import data, models, nn, featurizers 
from lightning.pytorch.loggers import CSVLogger 
from lightning.pytorch.callbacks import ModelCheckpoint 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from helpers import smiles_to_fingerprint, path_if_none, change_column_order, load_datapoints_basic, load_datapoints_tox_only

"""
training functions
"""

def dataset_to_numpy(datapoints, smiles_column="smiles"):
    X, y = [], []
    
    for i, dp in enumerate(datapoints):
        # Safety check for missing extra features
        extra_features = dp.get("x_d", [])
        if extra_features is None: 
            extra_features = []
            
        fp = smiles_to_fingerprint(dp[smiles_column])
        
        # Ensure dimensionality matches
        feats = np.concatenate([fp, np.array(extra_features)])
        X.append(feats)
        
        # specific handling for target existence
        target = dp.get("y", [None])
        y.append(target)

    return np.array(X), np.array(y)

def train_basic(split_dir="../data", 
             save_dir="../data",
             cv_fold=0,               
             smiles_column="smiles", 
             target_columns=["class_0", "class_1", "class_2", "class_3"]):
    
    path_if_none(save_dir)

    train_datapoints = load_datapoints_basic(
        os.path.join(split_dir, "train.csv"),
        os.path.join(split_dir, "train_extra_x.csv"),
        smiles_column=smiles_column,
        target_columns=target_columns
    )
    val_datapoints = load_datapoints_basic(
        os.path.join(split_dir, "valid.csv"),
        os.path.join(split_dir, "valid_extra_x.csv"),
        smiles_column=smiles_column,
        target_columns=target_columns
    )

    # Fit StandardScaler on training extra features and save it
    train_x_d = [dp["x_d"] for dp in train_datapoints if dp["x_d"] is not None]
    
    if len(train_x_d) > 0:
        scaler = StandardScaler()
        scaler.fit(train_x_d)
        
        scaler_path = os.path.join(save_dir, "extra_features_scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
            
        # Apply scaling to train data in memory
        for dp in train_datapoints:
            if dp["x_d"] is not None and len(dp["x_d"]) > 0:
                dp["x_d"] = scaler.transform([dp["x_d"]])[0]
        
        # Apply scaling to val data in memory
        for dp in val_datapoints:
            if dp["x_d"] is not None and len(dp["x_d"]) > 0:
                dp["x_d"] = scaler.transform([dp["x_d"]])[0]
    else:
        print("Warning: No extra features found to scale.")

    # Load weights
    train_weights = pd.read_csv(os.path.join(split_dir, "train_weights.csv"))["Sample_weight"].values

    # Convert to numpy
    X_train, y_train_hot = dataset_to_numpy(train_datapoints, smiles_column)
    X_val, y_val_hot = dataset_to_numpy(val_datapoints, smiles_column)
    y_train = np.argmax(y_train_hot, axis=1)
    y_val   = np.argmax(y_val_hot, axis=1)

    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=2,
        reg_alpha=1,
        gamma=0,
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
        early_stopping_rounds=50,  # Stop if valid loss doesn't improve for 50 rounds
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=len(target_columns)

    )

    model.fit(
        X_train, y_train,
        sample_weight=train_weights,
        eval_set=[(X_train, y_train), (X_val, y_val)], 
        verbose=False
    )
    print(f"Best iteration: {model.best_iteration}")


    y_pred_class = model.predict(X_val)
    
    # Get probabilities (for Log Loss)
    y_pred_proba = model.predict_proba(X_val)

    # 1. Accuracy: Simple percentage of correct answers
    acc = accuracy_score(y_val, y_pred_class)
    
    # 2. F1 Score (Weighted): Balances Precision and Recall
    # We MUST specify average='weighted' (or 'macro') for multiclass data
    f1 = f1_score(y_val, y_pred_class, average='weighted')
    
    # 3. Log Loss: Measures confidence (penalizes confident wrong answers)
    ll = log_loss(y_val, y_pred_proba)

    print(f"Fold {cv_fold} Validation Metrics:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Log Loss: {ll:.4f}")

    model_path = os.path.join(save_dir, "basic_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {model_path}")



def train_cm(
    split_dir='../data',
    smiles_column='smiles',
    target_columns=["quantified_toxicity"],   # only toxicity
    epochs=50,
    save_dir='../data'
):
    train_datapoints = load_datapoints_tox_only(
        os.path.join(split_dir, 'train.csv'),
        os.path.join(split_dir, 'train_extra_x.csv')
    )
    val_datapoints = load_datapoints_tox_only(
        os.path.join(split_dir, 'valid.csv'),
        os.path.join(split_dir, 'valid_extra_x.csv')
    )

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    agg = nn.MeanAggregation()

    # Load pretrained Chemeleon checkpoint
    chemeleon_mp = torch.load("chemeleon_mp.pt", weights_only=False)
    mp = nn.BondMessagePassing(**chemeleon_mp['hyper_parameters'])
    mp.load_state_dict(chemeleon_mp['state_dict'])

    train_weights = pd.read_csv(os.path.join(split_dir, "train_weights.csv"))["Sample_weight"].values
    val_weights   = pd.read_csv(os.path.join(split_dir, "valid_weights.csv"))["Sample_weight"].values
    train_dset = data.MoleculeDataset(train_datapoints, featurizer, sample_weights=train_weights)
    val_dset   = data.MoleculeDataset(val_datapoints, featurizer, sample_weights=val_weights)

    # Normalize extra features
    extra_features_scaler = train_dset.normalize_inputs("X_d")
    val_dset.normalize_inputs("X_d", extra_features_scaler)

    os.makedirs(save_dir, exist_ok=True)
    scaler_path = os.path.join(save_dir, "extra_features_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(extra_features_scaler, f)

    train_loader = data.build_dataloader(
        train_dset, shuffle=False, num_workers=6,
        persistent_workers=True, prefetch_factor=2
    )
    val_loader = data.build_dataloader(
        val_dset, shuffle=False, num_workers=6,
        persistent_workers=True, prefetch_factor=2
    )

    ffn_input_dim = mp.output_dim + train_dset[0].x_d.shape[0]

    # single-task regression head
    ffn = nn.RegressionFFN(
        n_tasks=1,   # only one target
        input_dim=ffn_input_dim,
        dropout=0.1,
        hidden_dim=600,
        n_layers=3,
        activation="RELU",
        criterion=nn.metrics.MAE()
    )

    metric_list = [nn.metrics.RMSE(), nn.metrics.MAE()]

    chemprop_model = models.MPNN(
        mp, agg, ffn,
        batch_norm=False,
        metrics=metric_list
    )

    checkpointing = ModelCheckpoint(
        dirpath=save_dir,
        filename="best",
        monitor="val/rmse",
        mode="min",
        save_top_k=1,
        save_last=False
    )
    logger = CSVLogger(save_dir=save_dir, name="chemprop_runs")

    pl.seed_everything(42)
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpointing],
        enable_checkpointing=True,
        max_epochs=epochs,
        num_sanity_val_steps=0,
        accelerator="auto",
        devices=1
    )

    # Train and save fine-tuned model
    trainer.fit(chemprop_model, train_loader, val_loader)

    torch.save(
        chemprop_model.state_dict(),
        os.path.join(save_dir, "model.pt")
    )
