import numpy as np 
import os
import pandas as pd  
from sklearn.metrics import mean_squared_error 
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
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
from sklearn.model_selection import train_test_split
from helpers import path_if_none, change_column_order, load_datapoints
from typing import List
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import List
from sklearn.preprocessing import StandardScaler
import joblib

def train_cm(
    split_dir='../data',
    smiles_column='smiles',
    target_columns=["quantified_toxicity"],   # only toxicity
    epochs=50,
    save_dir='../data'
):
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

    # Load pretrained Chemeleon checkpoint
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
        activation="RELU"
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

    # Train and save fine-tuned model
    trainer.fit(chemprop_model, train_loader, val_loader)

    torch.save(
        chemprop_model.state_dict(),
        os.path.join(save_dir, "chemprop_chemelon_finetuned.pt")
    )

def make_pred_vs_actual_tvt(
    split_folder,
    model_folder,
    ensemble_size=5,
    standardize_predictions=False,
    tvt='test'
):
    # Makes predictions on each split (train/valid/test) in a CV system
    # Now restricted to single target: quantified_toxicity

    for cv in range(ensemble_size):
        print(cv)
        model = f'../data/crossval_splits/{model_folder}/cv_{cv}'
        results_dir = f'../results/crossval_splits/{split_folder}/{tvt}/cv_{cv}'
        data_dir = model
        if tvt == 'test':
            data_dir = f'../data/crossval_splits/{split_folder}/test'

        df_test = pd.read_csv(f'{data_dir}/{tvt}.csv')
        metadata = pd.read_csv(f'{data_dir}/{tvt}_metadata.csv')
        output = pd.concat([metadata, df_test], axis=1)

        preds_dir = f'../data/crossval_splits/{split_folder}/preds/{tvt}'
        path_if_none(results_dir)
        path_if_none(preds_dir)

        try:
            output = pd.read_csv(f'{results_dir}/predicted_vs_actual.csv')
            print("pred vs actual already exist")
        except:
            try:
                current_predictions = pd.read_csv(f'{preds_dir}/cv_{cv}_preds.csv')
                print("already have preds.csv")
            except:
                print("running predict")
                checkpoint_path = f'{model}/model_{cv}/best.ckpt'
                loaded_model = models.MPNN.load_from_checkpoint(checkpoint_path)

                test_data = load_datapoints(
                    os.path.join(data_dir, f'{tvt}.csv'),
                    os.path.join(data_dir, f'{tvt}_extra_x.csv')
                )
                featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
                test_dataset = data.MoleculeDataset(test_data, featurizer=featurizer)

                # normalize extra features
                scaler_path = os.path.join(f"{model}/model_{cv}/extra_features_scaler.pkl")
                with open(scaler_path, "rb") as f:
                    extra_features_scaler = pickle.load(f)
                test_dataset.normalize_inputs("X_d", extra_features_scaler)
                test_loader = data.build_dataloader(test_dataset, shuffle=False)

                # run preds
                with torch.inference_mode():
                    trainer = pl.Trainer(logger=None, enable_progress_bar=False, accelerator="auto", devices=1)
                    preds = trainer.predict(loaded_model, test_loader)
                preds = np.concatenate(preds, axis=0)

                # single target only
                current_predictions = pd.DataFrame(preds, columns=["quantified_toxicity"])
                current_predictions["smiles"] = df_test["smiles"].values
                print("about to make preds.csv")
                current_predictions.to_csv(f'{preds_dir}/cv_{cv}_preds.csv', index=False)

            current_predictions.drop(columns=['smiles'], inplace=True)
            for col in current_predictions.columns:
                if standardize_predictions:
                    print("standardizing predictions")
                    preds_to_standardize = current_predictions[col]
                    std = np.std(preds_to_standardize)
                    mean = np.mean(preds_to_standardize)
                    current_predictions[col] = [(val - mean) / std for val in current_predictions[col]]
                current_predictions.rename(columns={col: f'cv_{cv}_pred_{col}'}, inplace=True)

            output = pd.concat([output, current_predictions], axis=1)

            pred_split_variables = ['Experiment_ID', 'Library_ID', 'Delivery_target', 'Route_of_administration']
            output['Prediction_split_name'] = output.apply(
                lambda row: '_'.join(str(row[v]) for v in pred_split_variables),
                axis=1
            )

            # move new prediction columns to the front
            new_cols = current_predictions.columns.tolist()
            new_cols = new_cols + ["quantified_toxicity", "smiles"]
            path = f'{results_dir}/predicted_vs_actual.csv'
            change_column_order(path, output, first_cols=new_cols)

    # Ultra held-out case
    if '_with_uho' in split_folder and tvt == 'test':
        print("uho")
        uho_dir = f'../data/crossval_splits/{split_folder}/ultra_held_out'
        results_dir = f'{uho_dir}/preds'
        path_if_none(results_dir)

        output = pd.read_csv(f'{uho_dir}/test.csv')
        metadata = pd.read_csv(f'{uho_dir}/test_metadata.csv')
        output = pd.concat([metadata, output], axis=1)

        for cv in range(ensemble_size):
            model_dir = f'../data/crossval_splits/{model_folder}/cv_{cv}'
            try:
                current_predictions = pd.read_csv(f'{results_dir}/preds_cv_{cv}.csv')
            except:
                checkpoint_path = f'{model_dir}/model_{cv}/best.ckpt'
                mpnn = models.MPNN.load_from_checkpoint(checkpoint_path)
                df_test = pd.read_csv(f'{data_dir}/test.csv')
                extra_df = pd.read_csv(f'{data_dir}/test_extra_x.csv')
                extra_features = extra_df.to_numpy(dtype=float)

                test_data = [
                    data.MoleculeDatapoint.from_smi(smi, x_d=xf)
                    for smi, xf in zip(df_test['smiles'], extra_features)
                ]
                featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
                test_dataset = data.MoleculeDataset(test_data, featurizer=featurizer)
                test_loader = data.build_dataloader(test_dataset, shuffle=False)

                with torch.inference_mode():
                    trainer = pl.Trainer(logger=None, enable_progress_bar=False, accelerator="auto", devices=1)
                    preds = trainer.predict(mpnn, test_loader)

                preds = np.concatenate(preds, axis=0)
                current_predictions = pd.DataFrame(preds, columns=["quantified_toxicity"])
                current_predictions["smiles"] = df_test["smiles"].values
                current_predictions.to_csv(f'{results_dir}/preds_cv_{cv}.csv', index=False)

            current_predictions.drop(columns=['smiles'], inplace=True)
            for col in current_predictions.columns:
                if standardize_predictions:
                    preds_to_standardize = current_predictions[col]
                    std = np.std(preds_to_standardize)
                    mean = np.mean(preds_to_standardize)
                    current_predictions[col] = [(val - mean) / std for val in current_predictions[col]]
                current_predictions.rename(columns={col: f'cv_{cv}_pred_{col}'}, inplace=True)

            output = pd.concat([output, current_predictions], axis=1)

        # collect toxicity-only preds
        pred_toxicity_cols = [col for col in output.columns if 'pred_quantified_toxicity' in col]
        output['Avg_pred_quantified_toxicity'] = output[pred_toxicity_cols].mean(axis=1)

        ultra_dir = f'../results/crossval_splits/{split_folder}/ultra_held_out'
        path_if_none(ultra_dir)
        path = f'{ultra_dir}/predicted_vs_actual.csv'

        first_col = ["quantified_toxicity"]
        for cv in range(ensemble_size):
            first_col.append(f"cv_{cv}_pred_quantified_toxicity")
        first_col += ["Avg_pred_quantified_toxicity", "smiles"]

        change_column_order(path, output, first_cols=first_col)

def analyze_predictions_cv_tvt(
    split_name,
    pred_split_variables=['Experiment_ID','Library_ID','Delivery_target','Route_of_administration'],
    path_to_preds='../results/crossval_splits/',
    ensemble_number=5,
    min_values_for_analysis=10,
    tvt='test'
):
    all_ns = {}
    all_pearson = {}
    all_r2 = {}
    all_pearson_p_val = {}
    all_kendall = {}
    all_spearman = {}
    all_rmse = {}
    all_unique = []
    
    preds_vs_actual = []
    for i in range(ensemble_number):
        df = pd.read_csv(f"{path_to_preds}{split_name}/{tvt}/cv_{i}/predicted_vs_actual.csv")
        preds_vs_actual.append(df)
        unique = set(df['Prediction_split_name'].tolist())
        all_unique.extend(unique)

    unique_pred_split_names = set(all_unique)

    for un in unique_pred_split_names:
        all_ns[un] = []
        all_pearson[un] = []
        all_r2[un] = []
        all_pearson_p_val[un] = []
        all_kendall[un] = []
        all_spearman[un] = []
        all_rmse[un] = []

    # single target only
    target_columns = [('quantified_toxicity', 'toxicity')]
    
    for i in range(ensemble_number):
        crossval_results_path = f"{path_to_preds}{split_name}/{tvt}"
        path_if_none(crossval_results_path)

        fold_results = []
        fold_df = preds_vs_actual[i]
        fold_unique = fold_df['Prediction_split_name'].unique()

        for pred_split_name in fold_unique:
            path_if_none(f"{path_to_preds}{split_name}/{tvt}/cv_{i}/results/{pred_split_name}")
            data_subset = preds_vs_actual[i][
                preds_vs_actual[i]['Prediction_split_name'] == pred_split_name
            ].reset_index(drop=True)

            value_names = set(list(data_subset.Value_name))
            if len(value_names) > 1:
                raise Exception(
                    f'Multiple types of measurement in split {pred_split_name}: {value_names}'
                )

            for actual_col, label in target_columns:
                if actual_col not in data_subset.columns:
                    print("target missing", actual_col)
                    continue

                actual = data_subset[actual_col]
                pred_col = f'cv_{i}_pred_{actual_col}'
                if pred_col not in data_subset.columns:
                    print("predictions missing", pred_col)
                    continue

                pred = data_subset[pred_col]
                analyzed_data = pd.DataFrame({
                    'smiles': data_subset.smiles, 'actual': actual, 'predicted': pred
                })

                mask = ~(actual.isna() | pred.isna() | np.isinf(actual) | np.isinf(pred))
                actual = actual[mask]
                pred = pred[mask]

                n_vals = len(actual)
                if n_vals < 2:
                    print(f"not enough data:{actual}, {label}")
                    pearson_r = rsquared = pearson_p = spearman_r = kendall_r = rmse = np.nan
                    analyzed_path = f"{path_to_preds}{split_name}/{tvt}/cv_{i}/results/{pred_split_name}/{label}"
                    path_if_none(analyzed_path)
                    analyzed_data.to_csv(analyzed_path+'/pred_vs_actual_data.csv', index=False)
                else:
                    pearson_r, pearson_p = scipy.stats.pearsonr(actual, pred)
                    rsquared = pearson_r * pearson_r
                    spearman_r, _ = scipy.stats.spearmanr(actual, pred)
                    kendall_r, _ = scipy.stats.kendalltau(actual, pred)
                    rmse = np.sqrt(mean_squared_error(actual, pred))
                    analyzed_path = f"{path_to_preds}{split_name}/{tvt}/cv_{i}/results/{pred_split_name}/{label}"
                    path_if_none(analyzed_path)
                    plt.figure()
                    plt.scatter(pred, actual, color='black')
                    plt.plot(np.unique(pred), np.poly1d(np.polyfit(pred, actual, 1))(np.unique(pred)))
                    plt.xlabel(f'Predicted {label}')
                    plt.ylabel(f'Quantified {label} ({value_names})')
                    plt.savefig(analyzed_path + '/pred_vs_actual.png')
                    plt.close()
                    analyzed_data.to_csv(analyzed_path+'/pred_vs_actual_data.csv', index=False)

                if n_vals < min_values_for_analysis:
                    print(f"Warning: only {n_vals} samples for {pred_split_name}, {label}")

                fold_results.append({
                    'fold': i,
                    'split_name': pred_split_name,
                    'target': label,
                    'pearson': pearson_r,
                    'r^2': rsquared,
                    'pearson_p_val': pearson_p,
                    'spearman': spearman_r,
                    'kendall': kendall_r,
                    'rmse': rmse,
                    'n_vals': n_vals,
                    'note': "insufficient_data" if n_vals < min_values_for_analysis else ""
                })

        fold_df = pd.DataFrame(fold_results)
        fold_df.to_csv(f"{crossval_results_path}/cv_{i}metrics.csv", index=False)

    # Ultra-held-out analysis
    try:
        preds_vs_actual = pd.read_csv(f"{path_to_preds}{split_name}/ultra_held_out/predicted_vs_actual.csv")
        preds_vs_actual['Prediction_split_name'] = preds_vs_actual[pred_split_variables].astype(str).agg('_'.join, axis=1)
        unique_pred_split_names = preds_vs_actual['Prediction_split_name'].unique()

        # only toxicity
        target_cols = ['Avg_pred_quantified_toxicity']
        actual_cols = ['quantified_toxicity']

        metrics_rows = []
        for pred_split_name in unique_pred_split_names:
            data_subset = preds_vs_actual[preds_vs_actual['Prediction_split_name'] == pred_split_name].reset_index(drop=True)
            actual = data_subset['quantified_toxicity']
            pred = data_subset['Avg_pred_quantified_toxicity']

            if actual.isna().all() or pred.isna().all():
                continue

            mask = ~(actual.isna() | pred.isna() | np.isinf(actual) | np.isinf(pred))
            actual = actual[mask]
            pred = pred[mask]

            if len(actual) < 2:
                pearson_r = pearson_p = spearman_r = kendall_r = rmse = np.nan
                n_vals = len(pred)
            else:
                pearson_r, pearson_p = scipy.stats.pearsonr(actual, pred)
                spearman_r, _ = scipy.stats.spearmanr(actual, pred)
                kendall_r, _ = scipy.stats.kendalltau(actual, pred)
                rmse = np.sqrt(mean_squared_error(actual, pred))
                analyzed_path = f"{path_to_preds}{split_name}/ultra_held_out/individual_dataset_results/{pred_split_name}"
                path_if_none(analyzed_path)
                plt.figure()
                plt.scatter(pred, actual, color='black')
                plt.plot(np.unique(pred), np.poly1d(np.polyfit(pred, actual, 1))(np.unique(pred)))
                plt.xlabel("Predicted toxicity")
                plt.ylabel("Experimental toxicity")
                plt.savefig(f"{analyzed_path}/pred_vs_actual_toxicity.png")
                plt.close()

            pd.DataFrame({
                'smiles': data_subset['smiles'],
                'actual': actual,
                'predicted': pred
            }).to_csv(f"{analyzed_path}/pred_vs_actual_toxicity_data.csv", index=False)

            metrics_rows.append({
                'dataset_ID': pred_split_name,
                'target': 'toxicity',
                'n': len(pred),
                'pearson': pearson_r,
                'pearson_p_val': pearson_p,
                'kendall': kendall_r,
                'spearman': spearman_r,
                'rmse': rmse,
                'note': "insufficient_data" if len(actual) < min_values_for_analysis else ""
            })

        uho_results_path = f"{path_to_preds}{split_name}/ultra_held_out"
        path_if_none(uho_results_path)
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(f"{uho_results_path}/ultra_held_out_metrics.csv", index=False)

    except Exception as e:
        print(f"Ultra-held-out analysis failed: {e}")

def main(argv):
    split_folder = argv[1]
    # epochs = 50
    # cv_num = 2
    # for i, arg in enumerate(argv):
    #     if arg.replace('–', '-') == '--epochs':
    #         epochs = int(argv[i+1])
    #         print('this many epochs: ',str(epochs))
    #     if arg.replace('–', '-') == '--cv':
    #         cv_num = int(argv[i+1])
    #         print('this many folds: ',str(cv_num))
    #     # if arg.replace('–', '-') == '--cm':
    #     #     for cv in range(cv_num):
    #     #         print("using cm")
    #     #         split_dir = '../data/crossval_splits/'+split_folder+'/cv_'+str(cv)
    #     #         save_dir = split_dir+'/model_'+str(cv)
    #     #         train_cm(split_dir=split_dir,epochs=epochs, save_dir=save_dir)
    #     #     return

    # for cv in range(cv_num):
    #     split_dir = '../data/crossval_splits/'+split_folder+'/cv_'+str(cv)
    #     save_dir = split_dir+'/model_'+str(cv)
    #     train_cm(split_dir=split_dir,epochs=epochs, save_dir=save_dir)
    
    
    test_dir = split_folder
    model_dir = test_dir
    cv = 2
    s = False
    to_eval = ["test", "train", "valid"]
    for i, arg in enumerate(argv):
        if arg.replace('–', '-') == '--cv':
            cv = int(argv[i+1])
            print('this many folds: ',str(cv))
        if arg.replace('–', '-') == '--diff_model':
            model_dir = argv[i+1]
        if arg.replace('–', '-') == '--standardize':
            s = True 
            print('standardize')
    for tvt in to_eval:
        print("make pva")
        make_pred_vs_actual_tvt(test_dir, model_dir, ensemble_size = cv, standardize_predictions= s, tvt=tvt)
        print("analyze preds")
        analyze_predictions_cv_tvt(test_dir, ensemble_number= cv, tvt=tvt)
        print("done with:", tvt)


    
if __name__ == '__main__':
    main(sys.argv)




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
