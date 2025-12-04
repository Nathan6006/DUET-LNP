import numpy as np 
import os
import pandas as pd  
from sklearn.metrics import mean_squared_error 
from rdkit import Chem 
from rdkit.Chem import Descriptors 
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
from helpers import path_if_none, change_column_order, load_datapoints_tox_only, load_datapoints_rf
from helpers import smiles_to_fingerprint

def make_preds(
    model_dir="../data",
    data_dir="../data",
    tvt="test",
    cv=5,
    df_test=None, # Changed default from pd.DataFrame type to None
    preds_dir="../data",
    rf=False
):
    print(f"Running predict for CV {cv}...")

    # Load data ONCE to ensure alignment for both paths
    target_datapoints = load_datapoints_rf(
        os.path.join(data_dir, f"{tvt}.csv"),
        os.path.join(data_dir, f"{tvt}_extra_x.csv"),
        target_columns=["quantified_toxicity"]
    )
    
    # Extract SMILES directly from the loaded datapoints to ensure row alignment
    aligned_smiles = [dp["smiles"] for dp in target_datapoints]

    if rf:
        # --- Random Forest / XGBoost path ---
        model_path = os.path.join(model_dir, f"model_{cv}", "basic_model.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Check train_rf save paths.")

        with open(model_path, "rb") as f:
            rf_model = pickle.load(f)

        # Load and apply scaler
        scaler_path = os.path.join(model_dir, f"model_{cv}", "extra_features_scaler.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        else:
            scaler = None
            print("Warning: Scaler not found, proceeding without scaling extra features.")

        # Build feature matrix
        X = []
        for dp in target_datapoints:
            fp = smiles_to_fingerprint(dp["smiles"])
            
            # Ensure dp["x_d"] is array-like
            x_d = np.array(dp["x_d"]) if dp["x_d"] is not None else np.array([])
            
            # Apply scaling if scaler exists and x_d is not empty
            if scaler is not None and len(x_d) > 0:
                x_d = scaler.transform([x_d])[0]
                
            feats = np.concatenate([fp, x_d])
            X.append(feats)
        X = np.array(X)

        preds = rf_model.predict(X)

    else:
        # --- Neural net path (Chemprop) ---
        test_data = load_datapoints_tox_only(
            os.path.join(data_dir, f"{tvt}.csv"),
            os.path.join(data_dir, f"{tvt}_extra_x.csv")
        )
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        test_dataset = data.MoleculeDataset(test_data, featurizer=featurizer)

        scaler_path = os.path.join(model_dir, f"model_{cv}", "extra_features_scaler.pkl")
        with open(scaler_path, "rb") as f:
            extra_features_scaler = pickle.load(f)
        test_dataset.normalize_inputs("X_d", extra_features_scaler)

        checkpoint_path = os.path.join(model_dir, f"model_{cv}", "best.ckpt")
        loaded_model = models.MPNN.load_from_checkpoint(checkpoint_path)

        test_loader = data.build_dataloader(test_dataset, shuffle=False)

        with torch.inference_mode():
            trainer = pl.Trainer(
                logger=None,
                enable_progress_bar=False,
                accelerator="auto",
                devices=1
            )
            preds = trainer.predict(loaded_model, test_loader)
        preds = np.concatenate(preds, axis=0)

    # Flatten predictions if necessary
    if preds.ndim > 1 and preds.shape[1] == 1:
        print("flattening preds")
        preds = preds.flatten()

    # Create DataFrame
    current_predictions = pd.DataFrame({
        "smiles": aligned_smiles, # Use the aligned smiles
        f"cv_{cv}_pred_quantified_toxicity": preds
    })

    #print(f"Saving preds to {preds_dir}")
    os.makedirs(preds_dir, exist_ok=True)
    out_path = os.path.join(preds_dir, f"cv_{cv}_preds.csv")
    current_predictions.to_csv(out_path, index=False)

    return current_predictions

def make_pred_vs_actual_tvt(
    split_folder,
    model_folder,
    ensemble_size=5,
    standardize_predictions=False,
    tvt='test',
    rf=False
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
                current_predictions = make_preds(model_dir=model, data_dir=data_dir, tvt=tvt, cv=cv, df_test=df_test, preds_dir=preds_dir, rf=rf)
            
            current_predictions.drop(columns=['smiles'], inplace=True)

            # for col in current_predictions.columns:
            #     if standardize_predictions:
            #         print("standardizing predictions")
            #         preds_to_standardize = current_predictions[col]
            #         std = np.std(preds_to_standardize)
            #         mean = np.mean(preds_to_standardize)
            #         current_predictions[col] = [(val - mean) / std for val in current_predictions[col]]
            #     current_predictions.rename(columns={col: f'cv_{cv}_pred_{col}'}, inplace=True)

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
    dataset_metrics_accumulator = {un: [] for un in unique_pred_split_names}
    
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

                    min_val = min(pred.min(), actual.min()) - 0.01 
                    max_val = max(pred.max(), actual.max()) + 0.01
                    plt.xlim(min_val, max_val)
                    plt.ylim(min_val, max_val)
                    plt.gca().set_aspect('equal', adjustable='box')  
                    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

                    plt.savefig(analyzed_path + '/pred_vs_actual.png')
                    plt.close()
                    analyzed_data.to_csv(analyzed_path+'/pred_vs_actual_data.csv', index=False)

                if n_vals < min_values_for_analysis:
                    print(f"Warning: only {n_vals} samples for {pred_split_name}, {label}")

                metrics_row = {
                    'fold': i,
                    'pearson': pearson_r,
                    'r^2': rsquared,
                    'pearson_p_val': pearson_p,
                    'spearman': spearman_r,
                    'kendall': kendall_r,
                    'rmse': rmse,
                    'n_vals': n_vals,
                    'note': "insufficient_data" if n_vals < min_values_for_analysis else ""
                }
                # Append the metrics to the list corresponding to the current dataset name
                dataset_metrics_accumulator[pred_split_name].append(metrics_row)

    crossval_results_path = f"{path_to_preds}{split_name}/{tvt}"
    path_if_none(f"{crossval_results_path}/metrics/")
    
    for dataset_name, metrics_list in dataset_metrics_accumulator.items():
        if metrics_list:
            df = pd.DataFrame(metrics_list)
            # Save the file as {dataset_name}_metrics.csv
            df.to_csv(f"{crossval_results_path}/metrics/{dataset_name}_metrics.csv", index=False)

    #create pooled metrics file
    path_if_none(f"{path_to_preds}{split_name}/{tvt}/pooled")
    rows = []

    for i in range(ensemble_number):
        df = pd.read_csv(f"{path_to_preds}{split_name}/{tvt}/cv_{i}/predicted_vs_actual.csv")
        pred_col = f'cv_{i}_pred_quantified_toxicity'
        if pred_col not in df.columns:
            continue

        actual = df['quantified_toxicity']
        pred   = df[pred_col]

        # mask invalids
        mask = ~(actual.isna() | pred.isna())
        actual = actual[mask]
        pred   = pred[mask]

        if len(actual) >= 2:
            pearson_r, pearson_p = scipy.stats.pearsonr(actual, pred)
            spearman_r, _ = scipy.stats.spearmanr(actual, pred)
            kendall_r, _ = scipy.stats.kendalltau(actual, pred)
            rmse = np.sqrt(mean_squared_error(actual, pred))

            plt.figure()
            plt.scatter(pred, actual, color='black')
            plt.plot(np.unique(pred), np.poly1d(np.polyfit(pred, actual, 1))(np.unique(pred)))
            plt.xlabel(f'Predicted toxicity (cv_{i})')
            plt.ylabel('Quantified toxicity')
            plt.title(f'Fold {i} pooled predictions vs actual')
            min_pooled = min(pred.min(), actual.min()) - 0.01
            max_pooled = max(pred.max(), actual.max()) + 0.01
            plt.xlim(min_pooled, max_pooled)
            plt.ylim(min_pooled, max_pooled)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.plot([min_pooled, max_pooled], [min_pooled, max_pooled], 'r--')

            plt.savefig(f"{path_to_preds}{split_name}/{tvt}/pooled/cv_{i}_pooled_pred_vs_actual.png")
            plt.close()

        else:
            pearson_r = pearson_p = spearman_r = kendall_r = rmse = np.nan

        rows.append({
            'fold': i,
            'pearson': pearson_r,
            'pearson_p_val': pearson_p,
            'spearman': spearman_r,
            'kendall': kendall_r,
            'rmse': rmse,
            'n_vals': len(actual)
        })

    # save metrics
    pooled_metrics_df = pd.DataFrame(rows)
    pooled_metrics_df.to_csv(f"{path_to_preds}{split_name}/{tvt}/pooled/pooled_metrics.csv", index=False)
    
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
