import sys
import numpy as np 
import os
import pandas as pd  
import matplotlib.pyplot as plt 
import scipy.stats
import pickle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from helpers import path_if_none, change_column_order, load_datapoints_rf, load_datapoints_tox_only
from embeddings import morgan_fingerprint
from train import PossMSEObjective
from embeddings import dataset_to_numpy

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
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Apply scaling in-memory (same pattern as training)
    for dp in target_datapoints:
        if dp.get("x_d") is not None and len(dp["x_d"]) > 0:
            dp["x_d"] = scaler.transform([dp["x_d"]])[0]

    # Convert to numpy
    X, y = dataset_to_numpy(
        target_datapoints,
        smiles_column="smiles"
    )

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


def calculate_metrics(actual, pred):

    # 1. Clean Data
    mask = ~(actual.isna() | pred.isna() | np.isinf(actual) | np.isinf(pred))
    actual_clean = actual[mask]
    pred_clean = pred[mask]
    n_vals = len(actual_clean)

    # 2. Check for Insufficient Data
    if n_vals < 2:
        return {
            'pearson': np.nan,
            'pearson_p_val': np.nan,
            'spearman': np.nan,
            'kendall': np.nan,
            'mse': np.nan,
            'mae': np.nan,
            'r2': np.nan,
            'n_vals': n_vals
        }

    # 3. Calculate Metrics
    pearson_r, pearson_p = scipy.stats.pearsonr(actual_clean, pred_clean)
    spearman_r, _ = scipy.stats.spearmanr(actual_clean, pred_clean)
    kendall_r, _ = scipy.stats.kendalltau(actual_clean, pred_clean)
    mse = mean_squared_error(actual_clean, pred_clean)
    mae = mean_absolute_error(actual_clean, pred_clean)
    r2 = r2_score(actual_clean, pred_clean)

    return {
        'pearson': round(pearson_r, 6),
        'r2': round(r2,6),
        'pearson_p_val': round(pearson_p,6),
        'spearman': round(spearman_r, 6),
        'kendall': round(kendall_r, 6),
        'mse': round(mse, 6),
        'mae': round(mae, 6),
        'n_vals': n_vals
    }

def analyze_predictions_cv_tvt(
    split_name,
    pred_split_variables=['Experiment_ID'],
    path_to_preds='../results/crossval_splits/',
    ensemble_number=5,
    min_values_for_analysis=10,
    tvt='test',
    target_columns=["quantified_toxicity"],
    class_bins = [0.0, 0.7, 0.8, 1.1]
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
    
    # Setup for Class Analysis
    bin_edges = class_bins
    bin_labels = [f"{bin_edges[j]}_{bin_edges[j+1]}" for j in range(len(bin_edges)-1)]
    class_metrics_accumulator = {label: [] for label in bin_labels}
    pooled_rows = []

    # --- 2. Main Loop (Per Fold) ---
    for i in range(ensemble_number):
        fold_dir = f"{path_to_preds}{split_name}/{tvt}/cv_{i}"
        fold_file = f"{fold_dir}/predicted_vs_actual.csv"
        
        if not os.path.exists(fold_file):
            continue
            
        fold_df = pd.read_csv(fold_file)
        pred_col = f'cv_{i}_pred_{target_col}'
        
        if pred_col not in fold_df.columns or target_col not in fold_df.columns:
            continue

        # ==========================================
        # PART A: Granular Subsets (Per Dataset)
        # ==========================================
        if 'Prediction_split_name' in fold_df.columns:
            for pred_split_name in fold_df['Prediction_split_name'].unique():
                results_path = f"{fold_dir}/results/{pred_split_name}"
                path_if_none(results_path)
                
                data_subset = fold_df[fold_df['Prediction_split_name'] == pred_split_name].reset_index(drop=True)
                
                # Get Metrics using Helper
                subset_metrics = calculate_metrics(data_subset[target_col], data_subset[pred_col])
                
                # -- Plotting (Specific to Dataset) --
                if subset_metrics['n_vals'] >= 2:
                    actual = data_subset[target_col] # Raw for plotting (cleaning handled inside helper for metrics, but we need it for plots)
                    pred = data_subset[pred_col]
                    mask = ~(actual.isna() | pred.isna() | np.isinf(actual) | np.isinf(pred))
                    actual_clean = actual[mask]
                    pred_clean = pred[mask]

                    plt.figure(figsize=(6, 6))
                    plt.scatter(pred_clean, actual_clean, color='black', alpha=0.6)
                    try:
                        m, b = np.polyfit(pred_clean, actual_clean, 1)
                        plt.plot(pred_clean, m*pred_clean + b, color='blue', alpha=0.5, label='Fit')
                    except: pass

                    min_val, max_val = 0.2, 1.05
                    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
                    plt.xlim(min_val, max_val)
                    plt.ylim(min_val, max_val)
                    plt.xlabel(f'Predicted {target_col}')
                    plt.ylabel(f'Actual {target_col}')
                    plt.title(f'{pred_split_name} (n={subset_metrics["n_vals"]})')
                    plt.legend()
                    plt.savefig(f"{results_path}/pred_vs_actual.png")
                    plt.close()

                    pd.DataFrame({
                        'smiles': data_subset.loc[mask, 'smiles'],
                        'actual': actual_clean,
                        'predicted': pred_clean
                    }).to_csv(f"{results_path}/pred_vs_actual_data.csv", index=False)

                # Store Dataset Metrics
                row = {'fold': i, 'note': "insufficient_data" if subset_metrics['n_vals'] < min_values_for_analysis else ""}
                row.update(subset_metrics)
                dataset_metrics_accumulator[pred_split_name].append(row)

        # ==========================================
        # PART B: Pooled Analysis (All Data in Fold)
        # ==========================================
        actual_all = fold_df[target_col]
        pred_all = fold_df[pred_col]
        
        # Get Metrics using Helper
        pooled_metrics = calculate_metrics(actual_all, pred_all)
        
        # -- Plotting (Specific to Pooled) --
        if pooled_metrics['n_vals'] >= 2:
            # Re-mask for plotting locally
            mask = ~(actual_all.isna() | pred_all.isna() | np.isinf(actual_all) | np.isinf(pred_all))
            a_clean = actual_all[mask]
            p_clean = pred_all[mask]

            plt.figure(figsize=(6,6))
            plt.scatter(p_clean, a_clean, color='black', alpha=0.5)
            try:
                plt.plot(np.unique(p_clean), np.poly1d(np.polyfit(p_clean, a_clean, 1))(np.unique(p_clean)), 'b-')
            except: pass
            
            min_pooled, max_pooled = 0.2, 1.05
            plt.xlim(min_pooled, max_pooled)
            plt.ylim(min_pooled, max_pooled)
            plt.plot([min_pooled, max_pooled], [min_pooled, max_pooled], 'r--')
            plt.xlabel(f'Predicted (cv_{i})')
            plt.ylabel('Quantified')
            plt.title(f'Fold {i} Pooled (All Datasets)')
            
            path_if_none(f"{path_to_preds}{split_name}/{tvt}/pooled")
            plt.savefig(f"{path_to_preds}{split_name}/{tvt}/pooled/cv_{i}_pooled_pred_vs_actual.png")
            plt.close()

        # Store Pooled Metrics
        p_row = {'fold': i}
        p_row.update(pooled_metrics)
        pooled_rows.append(p_row)

        # ==========================================
        # PART C: Class/Bin Analysis
        # ==========================================
        for j, label in enumerate(bin_labels):
            low = bin_edges[j]
            high = bin_edges[j+1]
            
            # Create mask for the specific bin based on ACTUAL values
            if j == len(bin_labels) - 1:
                bin_mask = (actual_all >= low) & (actual_all <= high)
            else:
                bin_mask = (actual_all >= low) & (actual_all < high)
            
            # Slice data
            bin_actual = actual_all[bin_mask]
            bin_pred = pred_all[bin_mask]
            
            # Get Metrics using Helper
            bin_metrics = calculate_metrics(bin_actual, bin_pred)
            
            # Store Class Metrics
            c_row = {'fold': i}
            c_row.update(bin_metrics)
            class_metrics_accumulator[label].append(c_row)

    # --- 3. Save Outputs ---

    # A. Save Per-Dataset Metrics
    metrics_out_path = f"{path_to_preds}{split_name}/{tvt}/dataset"
    path_if_none(metrics_out_path)

    for dataset_name, metrics_list in dataset_metrics_accumulator.items():
        if metrics_list:
            df = pd.DataFrame(metrics_list)
            if len(df) > 0:
                 mean_row = df.select_dtypes(include=[np.number]).mean()
                 mean_row = mean_row.astype(object)
                 mean_row['fold'] = 'Mean'
                 df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
            df.to_csv(f"{metrics_out_path}/{dataset_name}_metrics.csv", index=False)

    # B. Save Pooled Metrics
    if pooled_rows:
        path_if_none(f"{path_to_preds}{split_name}/{tvt}/pooled")
        pooled_metrics_df = pd.DataFrame(pooled_rows)
        pooled_metrics_df.to_csv(f"{path_to_preds}{split_name}/{tvt}/pooled/pooled_metrics.csv", index=False)

    # C. Save Class Metrics
    classes_out_path = f"{path_to_preds}{split_name}/{tvt}/classes"
    path_if_none(classes_out_path)

    for label, rows in class_metrics_accumulator.items():
        if rows:
            class_df = pd.DataFrame(rows)
            if len(class_df) > 0:
                 mean_row = class_df.select_dtypes(include=[np.number]).mean()
                 mean_row = mean_row.round(6)
                 mean_row = mean_row.astype(object)
                 mean_row['fold'] = 'Mean'
                 class_df = pd.concat([class_df, pd.DataFrame([mean_row])], ignore_index=True)
            class_df.drop(columns=["r2", "kendall"], inplace=True)
            class_df.to_csv(f"{classes_out_path}/metrics_{label}.csv", index=False) 


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