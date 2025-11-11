import numpy as np 
import os
import pandas as pd  
from sklearn.metrics import mean_squared_error 
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
import matplotlib.pyplot as plt 
import pickle
import sys
import random
from lightning import pytorch as pl 
import torch 
from pathlib import Path
from chemprop import data, models, nn, featurizers 
from lightning.pytorch.loggers import CSVLogger 
from lightning.pytorch.callbacks import ModelCheckpoint 
from sklearn.model_selection import train_test_split
from helpers import path_if_none, change_column_order, load_datapoints
from typing import List
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import List
from sklearn.preprocessing import StandardScaler
import joblib


def train_basic(split_dir ='../data', smiles_column='smiles', target_columns = ["quantified_delivery", "quantified_toxicity"], epochs=50, save_dir='../data'):

    train_datapoints = load_datapoints(split_dir+'/train.csv', split_dir+'/train_extra_x.csv')
    val_datapoints   = load_datapoints(split_dir+'/valid.csv', split_dir+'/valid_extra_x.csv')
    #test_datapoints  = load_datapoints(split_dir+'/test.csv', split_dir+'/test_extra_x.csv')
    
    #import chemeleon 
    agg = nn.MeanAggregation()

    train_dset = data.MoleculeDataset(train_datapoints)
    val_dset = data.MoleculeDataset(val_datapoints)

    # scale targets and extra features

    extra_features_scaler = train_dset.normalize_inputs("X_d")
    val_dset.normalize_inputs("X_d", extra_features_scaler)
    #test_dset.normalize_inputs("X_d", extra_features_scaler)

    train_loader = data.build_dataloader(train_dset, num_workers=6, persistent_workers=True, prefetch_factor=2)
    val_loader = data.build_dataloader(val_dset, shuffle=False, num_workers=6, persistent_workers=True, prefetch_factor=2)
    #test_loader = data.build_dataloader(test_dset, shuffle=False, num_workers=4, persistent_workers=True)
    
    #define the model

    mp = nn.BondMessagePassing(depth=4)
    ffn_input_dim = train_dset[0].x_d.shape[0] + mp.output_dim
    ffn = nn.RegressionFFN(
        n_tasks = len(target_columns), 
        input_dim=ffn_input_dim,
        dropout=0.1,
        hidden_dim=600,
        n_layers=3,
        activation='RELU' # only these options RELU, LEAKYRELU, PRELU, TANH, ELU"
    )

    X_d_transform = nn.ScaleTransform.from_standard_scaler(extra_features_scaler)

    #sigmoid for activation
    #cross entropy loss 
    #train until loss near 0
    metric_list = [nn.metrics.RMSE(), nn.metrics.MAE()]
    chemprop_model = models.MPNN(
        mp, agg, ffn,
        X_d_transform=X_d_transform,
        batch_norm=False,
        metrics=metric_list
    )

    #train 
    checkpointing = ModelCheckpoint(
        dirpath=save_dir,                # folder where checkpoints go
        filename="best", # naming convention
        monitor="val/rmse",                   # metric to monitor
        mode="min",                           # "min" if lower is better (RMSE/MAE), "max" if higher is better (accuracy/AUROC)
        save_last=True,                       # also save the very last checkpoint
        save_top_k=1                          # keep only the best model (set >1 if you want multiple)
    )

    logger = CSVLogger(save_dir=save_dir, name="chemprop_runs")
    pl.seed_everything(42)
    trainer = pl.Trainer(
        logger=logger, 
        enable_checkpointing=True,
        callbacks=[checkpointing], 
        max_epochs=epochs, 
        num_sanity_val_steps=0,
        accelerator="auto",
        devices=1
    )
    trainer.fit(chemprop_model, train_loader, val_loader)

def train_cm(split_dir='../data', smiles_column='smiles', target_columns=["quantified_delivery", "quantified_toxicity"], epochs=50, save_dir='../data'):
    

    train_datapoints = load_datapoints(
        os.path.join(split_dir, 'train.csv'),
        os.path.join(split_dir, 'train_extra_x.csv')
    )
    val_datapoints = load_datapoints(
        os.path.join(split_dir, 'valid.csv'),
        os.path.join(split_dir, 'valid_extra_x.csv')
    )

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    agg = nn.MeanAggregation()


    # Now load checkpoint
    chemeleon_mp = torch.load("chemeleon_mp.pt", weights_only=False)
    mp = nn.BondMessagePassing(**chemeleon_mp['hyper_parameters'])
    mp.load_state_dict(chemeleon_mp['state_dict'])

    train_dset = data.MoleculeDataset(train_datapoints, featurizer)
    val_dset = data.MoleculeDataset(val_datapoints, featurizer)

    # Normalize extra features
    extra_features_scaler = train_dset.normalize_inputs("X_d")
    val_dset.normalize_inputs("X_d", extra_features_scaler)

    os.makedirs(save_dir, exist_ok=True)
    scaler_path = os.path.join(save_dir, "extra_features_scaler.pkl")

    with open(scaler_path, "wb") as f:
        pickle.dump(extra_features_scaler, f)

    train_loader = data.build_dataloader(train_dset, shuffle=False, num_workers=6, persistent_workers=True, prefetch_factor=2)
    val_loader = data.build_dataloader(val_dset, shuffle=False, num_workers=6, persistent_workers=True, prefetch_factor=2)

    ffn_input_dim = mp.output_dim + train_dset[0].x_d.shape[0]

    ffn = nn.RegressionFFN(
        n_tasks=len(target_columns),
        input_dim=ffn_input_dim,
        dropout=0.1,
        hidden_dim=600,
        n_layers=3,
        activation="RELU"
    )

    # X_d_transform = nn.ScaleTransform.from_standard_scaler(extra_features_scaler)
    metric_list = [nn.metrics.RMSE(), nn.metrics.MAE()]

    chemprop_model = models.MPNN(
        mp, agg, ffn,
        #X_d_transform=X_d_transform,
        batch_norm=False,
        metrics=metric_list
    )

    os.makedirs(save_dir, exist_ok=True)
    checkpointing = ModelCheckpoint(
        dirpath=save_dir,
        filename="best",
        monitor="val/rmse",
        mode="min",
        save_last=True,
        save_top_k=1
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

    #train and save fine-tuned model
    trainer.fit(chemprop_model, train_loader, val_loader)

    torch.save(
        chemprop_model.state_dict(),
        os.path.join(save_dir, "chemprop_chemelon_finetuned.pt")
    )

def main(argv):
    split_folder = argv[1]
    epochs = 50
    cv_num = 2
    for i, arg in enumerate(argv):
        if arg.replace('–', '-') == '--epochs':
            epochs = int(argv[i+1])
            print('this many epochs: ',str(epochs))
        if arg.replace('–', '-') == '--cv':
            cv_num = int(argv[i+1])
            print('this many folds: ',str(cv_num))
        # if arg.replace('–', '-') == '--cm':
        #     for cv in range(cv_num):
        #         print("using cm")
        #         split_dir = '../data/crossval_splits/'+split_folder+'/cv_'+str(cv)
        #         save_dir = split_dir+'/model_'+str(cv)
        #         train_cm(split_dir=split_dir,epochs=epochs, save_dir=save_dir)
        #     return

    for cv in range(cv_num):
        split_dir = '../data/crossval_splits/'+split_folder+'/cv_'+str(cv)
        save_dir = split_dir+'/model_'+str(cv)
        train_cm(split_dir=split_dir,epochs=epochs, save_dir=save_dir)



    
if __name__ == '__main__':
    main(sys.argv)