import numpy as np 
import os
import pandas as pd  
from sklearn.metrics import mean_squared_error 
import matplotlib.pyplot as plt 
import scipy.stats
import pickle
import sys
import random
from lightning import pytorch as pl 
import torch 
from pathlib import Path
from chemprop import data, models, nn, featurizers 
from lightning.pytorch.loggers import CSVLogger 
from lightning.pytorch.callbacks import ModelCheckpoint 
from helpers import path_if_none, change_column_order, load_datapoints_rf, load_datapoints_tox_only
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
import xgboost as xgb
from helpers import smiles_to_fingerprint

"""
training funcitons
"""

def dataset_to_numpy(datapoints, smiles_column="smiles"):
    X, y = [], []
    valid_indices = [] # Track valid indices to keep X and y aligned
    
    for i, dp in enumerate(datapoints):
        # Safety check for missing extra features
        extra_features = dp.get("x_d", [])
        if extra_features is None: 
            extra_features = []
            
        fp = smiles_to_fingerprint(dp[smiles_column], use_counts=True)
        
        # Ensure dimensionality matches
        feats = np.concatenate([fp, np.array(extra_features)])
        X.append(feats)
        
        # specific handling for target existence
        target = dp.get("y", [None])[0]
        y.append(target)

    return np.array(X), np.array(y)

def train_basic(split_dir="../data", 
             save_dir="../data",      # Reverted to save_dir to match your script call
             cv_fold=0,               
             smiles_column="smiles", 
             target_columns=["quantified_toxicity"],
             model_type="xg",         # New argument: "rf" or "xg"
             max_features="sqrt"):
    
    path_if_none(save_dir)

    train_datapoints = load_datapoints_rf(
        os.path.join(split_dir, "train.csv"),
        os.path.join(split_dir, "train_extra_x.csv"),
        smiles_column=smiles_column,
        target_columns=target_columns
    )
    val_datapoints = load_datapoints_rf(
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
    X_train, y_train = dataset_to_numpy(train_datapoints, smiles_column)
    X_val, y_val     = dataset_to_numpy(val_datapoints, smiles_column)

    print(f"Training {model_type.upper()} for fold {cv_fold}...")

    if model_type == 'xg':
        model = xgb.XGBRegressor(
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
            early_stopping_rounds=50  # Stop if valid loss doesn't improve for 50 rounds
        )

        model.fit(
            X_train, y_train,
            sample_weight=train_weights,
            eval_set=[(X_train, y_train), (X_val, y_val)], 
            verbose=False
        )
        print(f"Best iteration: {model.best_iteration}")

    elif model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=15,
            max_features=max_features,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train, sample_weight=train_weights)

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'rf' or 'xg'.")


    # Evaluate
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"Fold {cv_fold} Validation MSE: {mse:.4f}, MAE: {mae:.4f}")

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


def main(argv):
    split_folder = argv[1]
    epochs = 50
    cv_num = 5
    basic = True
    
    # Regression target
    target_cols = ["quantified_toxicity"]

    for i, arg in enumerate(argv):
        if arg.replace('–', '-') == '--epochs':
            epochs = int(argv[i+1])
            print('this many epochs: ', str(epochs))
        if arg.replace('–', '-') == '--cv':
            cv_num = int(argv[i+1])
            print('this many folds: ', str(cv_num))
        if arg.replace('–', '-') == '--basic':
            print("using basic model")
            basic = True
            
    if basic:
        for cv in range(cv_num):
            split_dir = '../data/crossval_splits/' + split_folder + '/cv_' + str(cv)
            save_dir = split_dir + '/model_' + str(cv)
            # Pass regression target to training function
            train_basic(
                split_dir=split_dir, 
                save_dir=save_dir, 
                cv_fold=cv_num, 
                target_columns=target_cols
            )
    else:      
        for cv in range(cv_num):
            split_dir = '../data/crossval_splits/' + split_folder + '/cv_' + str(cv)
            save_dir = split_dir + '/model_' + str(cv)
            # Assuming train_cm is also updated or agnostic to target columns in your implementation
            # If train_cm needs explicit targets, add target_columns=target_cols here as well
            train_cm(split_dir=split_dir, epochs=epochs, save_dir=save_dir)
        
    


if __name__ == '__main__':
    main(sys.argv)