import numpy as np 
import os
import pandas as pd  
from sklearn.metrics import mean_squared_error 
import pickle
import sys
import random
from lightning import pytorch as pl 
import torch 
from chemprop import data, models, nn, featurizers 
from lightning.pytorch.loggers import CSVLogger 
from lightning.pytorch.callbacks import ModelCheckpoint 
from helpers import path_if_none, change_column_order, load_datapoints_rf, load_datapoints_tox_only
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from embeddings import morgan_fingerprint, rdkit_descriptors, dataset_to_numpy

"""
training funcitons
"""

class PossMSEObjective:
    def __init__(self, d=0.1):
        self.d = d

    def __call__(self, y_true, y_pred, sample_weight=None):
        d = self.d
        k = 2.0 + d
        
        # Calculate residuals
        diff = y_pred - y_true 
        abs_diff = np.abs(diff)
        
        # Gradient
        grad = k * (abs_diff**(k - 1)) * np.sign(diff)
        
        # Hessian
        hess = k * (k - 1) * (abs_diff**(k - 2))
        hess = np.maximum(hess, 1e-6) # Stability fix
        
        # Apply Sample Weights manually
        if sample_weight is not None:
            grad = grad * sample_weight
            hess = hess * sample_weight
            
        return grad, hess



def train_basic(split_dir="../data", 
             save_dir="../data",      # Reverted to save_dir to match your script call
             cv_fold=0,               
             smiles_column="smiles", 
             target_columns=["quantified_toxicity"]):     
    
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

    print(f"Fold {cv_fold}...")

    model = xgb.XGBRegressor(
        #objective=PossMSEObjective(d=0.1), # custom loss function
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
        early_stopping_rounds=50 
    )

    model.fit(
        X_train, y_train,
        sample_weight=train_weights,
        eval_set=[(X_train, y_train), (X_val, y_val)], 
        verbose=False
    )
    print(f"Best iteration: {model.best_iteration}")

    # Evaluate
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"Fold {cv_fold} Validation MSE: {mse:.4f}, MAE: {mae:.4f}")

    model_path = os.path.join(save_dir, "basic_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

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
                cv_fold=cv, 
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