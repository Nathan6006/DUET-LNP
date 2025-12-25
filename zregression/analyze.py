import sys
import numpy as np 
import os
import pandas as pd  
import matplotlib.pyplot as plt 
import scipy.stats
import pickle
import warnings
# Added mean_absolute_error to imports
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from lightning import pytorch as pl 
import torch 
from chemprop import data, models, featurizers 
from helpers import path_if_none, change_column_order, load_datapoints_rf, load_datapoints_tox_only
from helpers import smiles_to_fingerprint

# ------------------------------------------------------------------------------
# 1. MAKE PREDICTIONS (Continuous Regression Values)
# ------------------------------------------------------------------------------
def make_preds(
    model_dir="../data",
    data_dir="../data",
    tvt="test",
    cv=5,
    df_test=None, 
    preds_dir="../data",
    rf=True,
    target_columns=["quantified_toxicity"]
):
    print(f"Running predict for CV {cv}...")

    target_datapoints = load_datapoints_rf(
        os.path.join(data_dir, f"{tvt}.csv"),
        os.path.join(data_dir, f"{tvt}_extra_x.csv"),
        target_columns=target_columns
    )
    
    aligned_smiles = [dp["smiles"] for dp in target_datapoints]

    model_path = os.path.join(model_dir, f"model_{cv}", "basic_model.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    with open(model_path, "rb") as f:
        rf_model = pickle.load(f)

    scaler_path = os.path.join(model_dir, f"model_{cv}", "extra_features_scaler.pkl")
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    else:
        print("Warning: Scaler not found, proceeding without scaling extra features.")

    X = []
    for dp in target_datapoints:
        fp = smiles_to_fingerprint(dp["smiles"])
        x_d = np.array(dp["x_d"]) if dp["x_d"] is not None else np.array([])
        
        if scaler is not None and len(x_d) > 0:
            x_d = scaler.transform([x_d])[0]
            
        feats = np.concatenate([fp, x_d])
        X.append(feats)
    X = np.array(X)

    preds = rf_model.predict(X)

    if preds.ndim > 1 and preds.shape[1] == 1:
        preds = preds.flatten()

    current_predictions = pd.DataFrame({
        "smiles": aligned_smiles,
        f"cv_{cv}_pred_quantified_toxicity": preds
    })

    os.makedirs(preds_dir, exist_ok=True)
    out_path = os.path.join(preds_dir, f"cv_{cv}_preds.csv")
    current_predictions.to_csv(out_path, index=False)

    return current_predictions


# ------------------------------------------------------------------------------
# 2. MERGE PREDICTIONS WITH ACTUALS
# ------------------------------------------------------------------------------
def make_pred_vs_actual_tvt(
    split_folder,
    model_folder,
    ensemble_size=5,
    standardize_predictions=False,
    tvt='test',
    rf=False,
    target_columns=["quantified_toxicity"]
):
    target_col_name = target_columns[0]

    for cv in range(ensemble_size):
        print(f"Processing CV {cv}")
        model_path_base = f'../data/crossval_splits/{model_folder}/cv_{cv}'
        results_dir = f'../results/crossval_splits/{split_folder}/{tvt}/cv_{cv}'
        
        data_dir = model_path_base
        if tvt == 'test':
            data_dir = f'../data/crossval_splits/{split_folder}/test'

        df_test = pd.read_csv(f'{data_dir}/{tvt}.csv')
        metadata = pd.read_csv(f'{data_dir}/{tvt}_metadata.csv')
        output = pd.concat([metadata, df_test], axis=1)

        preds_dir = f'../data/crossval_splits/{split_folder}/preds/{tvt}'
        path_if_none(results_dir)
        path_if_none(preds_dir)

        try:
            existing = pd.read_csv(f'{results_dir}/predicted_vs_actual.csv')
            print("pred vs actual already exists, skipping creation")
        except:
            pass

        try:
            current_predictions = pd.read_csv(f'{preds_dir}/cv_{cv}_preds.csv')
            print("already have preds.csv")
        except:
            current_predictions = make_preds(
                model_dir=model_path_base,
                data_dir=data_dir,
                tvt=tvt,
                cv=cv,
                df_test=df_test,
                preds_dir=preds_dir,
                rf=rf,
                target_columns=target_columns
            )
        
        if 'smiles' in current_predictions.columns:
            current_predictions.drop(columns=['smiles'], inplace=True)

        pred_col_name = f'cv_{cv}_pred_{target_col_name}'
        if standardize_predictions and pred_col_name in current_predictions.columns:
            print("standardizing predictions")
            vals = current_predictions[pred_col_name]
            current_predictions[pred_col_name] = (vals - vals.mean()) / vals.std()

        output = pd.concat([output, current_predictions], axis=1)

        pred_split_variables = ['Experiment_ID']
        if all(v in output.columns for v in pred_split_variables):
            output['Prediction_split_name'] = output.apply(
                lambda row: '_'.join(str(row[v]) for v in pred_split_variables),
                axis=1
            )
        else:
            output['Prediction_split_name'] = "All_Data"

        new_cols = [pred_col_name, target_col_name, "smiles"]
        path = f'{results_dir}/predicted_vs_actual.csv'
        change_column_order(path, output, first_cols=new_cols)


# ------------------------------------------------------------------------------
# 3. ANALYZE PREDICTIONS (MSE, MAE, R^2, Pearson)
# ------------------------------------------------------------------------------
def analyze_predictions_cv_tvt(
    split_name,
    pred_split_variables=['Experiment_ID'],
    path_to_preds='../results/crossval_splits/',
    ensemble_number=5,
    min_values_for_analysis=10,
    tvt='test',
    target_columns=["quantified_toxicity"]
):
    target_col = target_columns[0]
    
    # --- 1. Identify all unique datasets ---
    all_unique = []
    for i in range(ensemble_number):
        try:
            df = pd.read_csv(f"{path_to_preds}{split_name}/{tvt}/cv_{i}/predicted_vs_actual.csv")
            if 'Prediction_split_name' in df.columns:
                unique = set(df['Prediction_split_name'].tolist())
                all_unique.extend(unique)
        except FileNotFoundError:
            continue

    unique_pred_split_names = set(all_unique)
    dataset_metrics_accumulator = {un: [] for un in unique_pred_split_names}

    # --- 2. Process Granular Subsets (Per Fold Analysis) ---
    for i in range(ensemble_number):
        fold_dir = f"{path_to_preds}{split_name}/{tvt}/cv_{i}"
        try:
            fold_df = pd.read_csv(f"{fold_dir}/predicted_vs_actual.csv")
        except FileNotFoundError:
            continue

        if 'Prediction_split_name' not in fold_df.columns:
            continue

        for pred_split_name in fold_df['Prediction_split_name'].unique():
            results_path = f"{fold_dir}/results/{pred_split_name}"
            path_if_none(results_path)
            
            data_subset = fold_df[fold_df['Prediction_split_name'] == pred_split_name].reset_index(drop=True)

            pred_col = f'cv_{i}_pred_{target_col}'
            
            if target_col not in data_subset.columns or pred_col not in data_subset.columns:
                continue

            actual = data_subset[target_col]
            pred = data_subset[pred_col]

            mask = ~(actual.isna() | pred.isna() | np.isinf(actual) | np.isinf(pred))
            actual = actual[mask]
            pred = pred[mask]

            n_vals = len(actual)

            # --- Calculate Metrics (Subset Level) ---
            if n_vals < 2:
                pearson_r = pearson_p = spearman_r = kendall_r = mse = mae = r2 = np.nan
            else:
                pearson_r, pearson_p = scipy.stats.pearsonr(actual, pred)
                spearman_r, _ = scipy.stats.spearmanr(actual, pred)
                kendall_r, _ = scipy.stats.kendalltau(actual, pred)
                
                # UPDATED METRICS HERE
                mse = mean_squared_error(actual, pred)
                mae = mean_absolute_error(actual, pred)
                r2 = r2_score(actual, pred)

                # --- Generate Scatter Plot ---
                plt.figure(figsize=(6, 6))
                plt.scatter(pred, actual, color='black', alpha=0.6)
                
                try:
                    m, b = np.polyfit(pred, actual, 1)
                    plt.plot(pred, m*pred + b, color='blue', alpha=0.5, label='Fit')
                except:
                    pass

                min_val = min(pred.min(), actual.min()) - 0.1
                max_val = max(pred.max(), actual.max()) + 0.1
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
                
                plt.xlim(min_val, max_val)
                plt.ylim(min_val, max_val)
                plt.xlabel(f'Predicted {target_col}')
                plt.ylabel(f'Actual {target_col}')
                plt.title(f'{pred_split_name} (n={n_vals})')
                plt.legend()
                plt.savefig(f"{results_path}/pred_vs_actual.png")
                plt.close()

                pd.DataFrame({
                    'smiles': data_subset.loc[mask, 'smiles'],
                    'actual': actual,
                    'predicted': pred
                }).to_csv(f"{results_path}/pred_vs_actual_data.csv", index=False)

            metrics_row = {
                'fold': i,
                'pearson': pearson_r,
                'r2': r2,
                'spearman': spearman_r,
                'mse': mse,  # Changed from rmse
                'mae': mae,  # Added
                'n_vals': n_vals,
                'note': "insufficient_data" if n_vals < min_values_for_analysis else ""
            }
            dataset_metrics_accumulator[pred_split_name].append(metrics_row)

    # --- 3. Save Per-Fold Metrics ---
    metrics_out_path = f"{path_to_preds}{split_name}/{tvt}/metrics"
    path_if_none(metrics_out_path)

    for dataset_name, metrics_list in dataset_metrics_accumulator.items():
        if metrics_list:
            df = pd.DataFrame(metrics_list)
            if len(df) > 0:
                 # FIX: Cast to object before adding 'Mean' string
                 mean_row = df.select_dtypes(include=[np.number]).mean()
                 mean_row = mean_row.astype(object)
                 mean_row['fold'] = 'Mean'
                 
                 mean_df = pd.DataFrame([mean_row])
                 df = pd.concat([df, mean_df], ignore_index=True)
            df.to_csv(f"{metrics_out_path}/{dataset_name}_metrics.csv", index=False)

    # --- 4. POOLED METRICS (Aggregation Across Folds) ---
    path_if_none(f"{path_to_preds}{split_name}/{tvt}/pooled")
    rows = []

    for i in range(ensemble_number):
        fold_file = f"{path_to_preds}{split_name}/{tvt}/cv_{i}/predicted_vs_actual.csv"
        if not os.path.exists(fold_file):
            continue
            
        df = pd.read_csv(fold_file)
        pred_col = f'cv_{i}_pred_{target_col}'
        if pred_col not in df.columns or target_col not in df.columns:
            continue

        actual = df[target_col]
        pred   = df[pred_col]

        mask = ~(actual.isna() | pred.isna() | np.isinf(actual) | np.isinf(pred))
        actual = actual[mask]
        pred   = pred[mask]

        if len(actual) >= 2:
            pearson_r, pearson_p = scipy.stats.pearsonr(actual, pred)
            spearman_r, _ = scipy.stats.spearmanr(actual, pred)
            kendall_r, _ = scipy.stats.kendalltau(actual, pred)
            mse = mean_squared_error(actual, pred)
            mae = mean_absolute_error(actual, pred)
            r2 = r2_score(actual, pred)

            plt.figure(figsize=(6,6))
            plt.scatter(pred, actual, color='black', alpha=0.5)
            try:
                plt.plot(np.unique(pred), np.poly1d(np.polyfit(pred, actual, 1))(np.unique(pred)), 'b-')
            except: pass
            
            plt.xlabel(f'Predicted (cv_{i})')
            plt.ylabel('Quantified')
            plt.title(f'Fold {i} Pooled (All Datasets)')
            
            min_pooled = min(pred.min(), actual.min()) - 0.1
            max_pooled = max(pred.max(), actual.max()) + 0.1
            plt.xlim(min_pooled, max_pooled)
            plt.ylim(min_pooled, max_pooled)
            plt.plot([min_pooled, max_pooled], [min_pooled, max_pooled], 'r--')

            plt.savefig(f"{path_to_preds}{split_name}/{tvt}/pooled/cv_{i}_pooled_pred_vs_actual.png")
            plt.close()
        else:
            pearson_r = pearson_p = spearman_r = kendall_r = mse = mae = r2 = np.nan

        rows.append({
            'fold': i,
            'pearson': pearson_r,
            'r2': r2,
            'pearson_p_val': pearson_p,
            'spearman': spearman_r,
            'kendall': kendall_r,
            'mse': mse, # Changed from rmse
            'mae': mae, # Added
            'n_vals': len(actual)
        })

    pooled_metrics_df = pd.DataFrame(rows)
    pooled_metrics_df.to_csv(f"{path_to_preds}{split_name}/{tvt}/pooled/pooled_metrics.csv", index=False)
    
    # --- 5. ULTRA HELD OUT ANALYSIS (Optional) ---
    try:
        uho_path = f"{path_to_preds}{split_name}/ultra_held_out"
        uho_file = f"{uho_path}/predicted_vs_actual.csv"
        
        if os.path.exists(uho_file):
            print("Analyzing Ultra Held Out...")
            preds_vs_actual = pd.read_csv(uho_file)
            
            if all(v in preds_vs_actual.columns for v in pred_split_variables):
                preds_vs_actual['Prediction_split_name'] = preds_vs_actual[pred_split_variables].astype(str).agg('_'.join, axis=1)
            else:
                preds_vs_actual['Prediction_split_name'] = "UHO_Data"
                
            unique_pred_split_names = preds_vs_actual['Prediction_split_name'].unique()

            pred_cols = [c for c in preds_vs_actual.columns if f'pred_{target_col}' in c and 'Avg' not in c]
            if pred_cols:
                preds_vs_actual['Avg_pred'] = preds_vs_actual[pred_cols].mean(axis=1)
                
            actual_col = target_col
            pred_col_uho = 'Avg_pred'

            metrics_rows = []
            for pred_split_name in unique_pred_split_names:
                data_subset = preds_vs_actual[preds_vs_actual['Prediction_split_name'] == pred_split_name].reset_index(drop=True)
                
                if actual_col not in data_subset.columns or pred_col_uho not in data_subset.columns:
                    continue

                actual = data_subset[actual_col]
                pred = data_subset[pred_col_uho]

                mask = ~(actual.isna() | pred.isna() | np.isinf(actual) | np.isinf(pred))
                actual = actual[mask]
                pred = pred[mask]

                if len(actual) < 2:
                    pearson_r = pearson_p = spearman_r = kendall_r = mse = mae = r2 = np.nan
                else:
                    pearson_r, pearson_p = scipy.stats.pearsonr(actual, pred)
                    spearman_r, _ = scipy.stats.spearmanr(actual, pred)
                    kendall_r, _ = scipy.stats.kendalltau(actual, pred)
                    
                    # UPDATED METRICS HERE
                    mse = mean_squared_error(actual, pred)
                    mae = mean_absolute_error(actual, pred)
                    r2 = r2_score(actual, pred)
                    
                    analyzed_path = f"{uho_path}/individual_dataset_results/{pred_split_name}"
                    path_if_none(analyzed_path)
                    
                    plt.figure(figsize=(6,6))
                    plt.scatter(pred, actual, color='black', alpha=0.6)
                    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--')
                    plt.xlabel("Predicted (Avg)")
                    plt.ylabel("Actual")
                    plt.savefig(f"{analyzed_path}/pred_vs_actual.png")
                    plt.close()

                    pd.DataFrame({
                        'smiles': data_subset.loc[mask, 'smiles'],
                        'actual': actual,
                        'predicted': pred
                    }).to_csv(f"{analyzed_path}/pred_vs_actual_data.csv", index=False)

            metrics_rows.append({
                'dataset_ID': pred_split_name,
                'n': len(actual),
                'pearson': round(pearson_r, 6),
                'r2': round(r2, 6),
                'pearson_p_val': round(pearson_p, 6),
                'kendall': round(kendall_r, 6),
                'spearman': round(spearman_r, 6),
                'mse': round(mse, 6),  
                'mae': round(mae, 6),  
                'note': "insufficient_data" if len(actual) < min_values_for_analysis else ""
            })

            uho_metrics_df = pd.DataFrame(metrics_rows)
            uho_metrics_df.to_csv(f"{uho_path}/ultra_held_out_metrics.csv", index=False)
            
    except Exception as e:
        print(f"Ultra-held-out analysis failed or skipped: {e}")



def main(argv):
    cv_num = 5        
    test_dir = argv[1]
    model_dir = test_dir
    to_eval = ["test", "train", "valid"]
    
    for i, arg in enumerate(argv):
        if arg.replace('–', '-') == '--diff_model':
            model_dir = argv[i+1]

    for tvt in to_eval:
        print(f"Making predictions vs actual for: {tvt}")
        make_pred_vs_actual_tvt(
            test_dir, 
            model_dir, 
            ensemble_size=cv_num, 
            tvt=tvt
        )
        
        print(f"Analyzing predictions for: {tvt}")
        analyze_predictions_cv_tvt(
            test_dir, 
            ensemble_number=cv_num, 
            tvt=tvt
        )
        print("Done with:", tvt)

if __name__ == '__main__':
    main(sys.argv)