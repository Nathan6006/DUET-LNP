import sys
import numpy as np 
import os
import pandas as pd  
import matplotlib.pyplot as plt 
import scipy.stats
import pickle
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from helpers import path_if_none, change_column_order, load_datapoints_rf
from embeddings import dataset_to_numpy

# ------------------------------------------------------------------------------
# 0. OBJECTIVES (REQUIRED FOR UNPICKLING)
# ------------------------------------------------------------------------------
class PossMSEObjective:
    def __init__(self, d=0.1, scale = 20):
        self.d = d
        self.scale = scale

    def __call__(self, y_true, y_pred, sample_weight=None):
        d = self.d
        k = 2.0 + d
        scale = self.scale
        diff = scale * (y_pred - y_true)
        abs_diff = np.abs(diff)
        grad = k * (abs_diff**(k - 1)) * np.sign(diff)
        hess = k * (k - 1) * (abs_diff**(k - 2))
        hess = np.maximum(hess, 1e-6)
        if sample_weight is not None:
            grad = grad * sample_weight
            hess = hess * sample_weight
        return grad, hess

# ------------------------------------------------------------------------------
# 1. MAKE PREDICTIONS (Hierarchical: Classify -> Regress)
# ------------------------------------------------------------------------------
def make_preds(
    model_dir="../data",
    data_dir="../data",
    tvt="test",
    cv=5,
    df_test=None, 
    preds_dir="../data",
    rf=True, # Argument kept for compatibility, but logic is specific to XGBoost here
    target_columns=["quantified_toxicity"]
):
    print(f"Running Hierarchical Prediction for CV {cv}...")

    # Load datapoints
    target_datapoints = load_datapoints_rf(
        os.path.join(data_dir, f"{tvt}.csv"),
        os.path.join(data_dir, f"{tvt}_extra_x.csv"),
        target_columns=target_columns
    )
    
    aligned_smiles = [dp["smiles"] for dp in target_datapoints]

    # --- Load Scaler ---
    scaler_path = os.path.join(model_dir, f"model_{cv}", "extra_features_scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # --- Apply Scaling ---
    for dp in target_datapoints:
        if dp.get("x_d") is not None and len(dp["x_d"]) > 0:
            dp["x_d"] = scaler.transform([dp["x_d"]])[0]

    # --- Convert to Numpy ---
    X, _ = dataset_to_numpy(target_datapoints, smiles_column="smiles")

    # --- Load Hierarchical Models ---
    base_path = os.path.join(model_dir, f"model_{cv}")
    
    with open(os.path.join(base_path, "model_classifier.pkl"), "rb") as f:
        clf = pickle.load(f)
    with open(os.path.join(base_path, "model_reg_low.pkl"), "rb") as f:
        reg_low = pickle.load(f)
    with open(os.path.join(base_path, "model_reg_high.pkl"), "rb") as f:
        reg_high = pickle.load(f)

    # --- Step 1: Classification ---
    # Predict whether sample is Low (0) or High (1)
    preds_class = clf.predict(X)
    
    # --- Step 2: Routing ---
    preds_final = np.zeros(len(X))
    
    # Indices
    idx_low = np.where(preds_class == 0)[0]
    idx_high = np.where(preds_class == 1)[0]
    
    # Predict Low
    if len(idx_low) > 0:
        preds_final[idx_low] = reg_low.predict(X[idx_low])
        
    # Predict High
    if len(idx_high) > 0:
        preds_final[idx_high] = reg_high.predict(X[idx_high])

    # --- Save Results ---
    target_name = target_columns[0]
    
    # We save both the continuous prediction AND the classification decision
    current_predictions = pd.DataFrame({
        "smiles": aligned_smiles,
        f"cv_{cv}_pred_{target_name}": preds_final,
        f"cv_{cv}_pred_class": preds_class  # Helpful for debugging/analysis
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

        # Check if preds exist
        pred_file = f'{preds_dir}/cv_{cv}_preds.csv'
        if os.path.exists(pred_file):
            print("already have preds.csv")
            current_predictions = pd.read_csv(pred_file)
        else:
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
        
        # Clean up duplicates if merging multiple times
        if 'smiles' in current_predictions.columns:
            current_predictions_drop = current_predictions.drop(columns=['smiles'])
        else:
            current_predictions_drop = current_predictions

        # Merge
        # Note: We rely on row order alignment here (standard for this pipeline)
        output = pd.concat([output.reset_index(drop=True), current_predictions_drop.reset_index(drop=True)], axis=1)

        pred_col_name = f'cv_{cv}_pred_{target_col_name}'
        
        if pred_col_name not in output.columns:
            print(f"Warning: Prediction column {pred_col_name} missing.")
            continue

        if standardize_predictions:
            vals = output[pred_col_name]
            output[pred_col_name] = (vals - vals.mean()) / vals.std()

        # Handle splitting variable for analysis
        pred_split_variables = ['Experiment_ID']
        if all(v in output.columns for v in pred_split_variables):
            output['Prediction_split_name'] = output.apply(
                lambda row: '_'.join(str(row[v]) for v in pred_split_variables),
                axis=1
            )
        else:
            output['Prediction_split_name'] = "All_Data"

        # Organize columns
        cols_to_front = [pred_col_name, target_col_name]
        # Check if we have the class prediction column
        class_col_name = f'cv_{cv}_pred_class'
        if class_col_name in output.columns:
            cols_to_front.append(class_col_name)
        
        cols_to_front.append("smiles")
        
        path = f'{results_dir}/predicted_vs_actual.csv'
        change_column_order(path, output, first_cols=cols_to_front)


# ------------------------------------------------------------------------------
# 3. METRICS & ANALYSIS
# ------------------------------------------------------------------------------
def calculate_metrics(actual, pred):
    mask = ~(actual.isna() | pred.isna() | np.isinf(actual) | np.isinf(pred))
    actual_clean = actual[mask]
    pred_clean = pred[mask]
    n_vals = len(actual_clean)

    if n_vals < 2:
        return {
            'pearson': np.nan, 'pearson_p_val': np.nan, 'spearman': np.nan,
            'kendall': np.nan, 'mse': np.nan, 'mae': np.nan, 'r2': np.nan, 'n_vals': n_vals
        }

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
    
    # 1. Gather all unique dataset names across all folds
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
    
    bin_edges = class_bins
    bin_labels = [f"{bin_edges[j]}_{bin_edges[j+1]}" for j in range(len(bin_edges)-1)]
    class_metrics_accumulator = {label: [] for label in bin_labels}
    pooled_rows = []

    # 2. Iterate Folds
    for i in range(ensemble_number):
        fold_dir = f"{path_to_preds}{split_name}/{tvt}/cv_{i}"
        fold_file = f"{fold_dir}/predicted_vs_actual.csv"
        
        if not os.path.exists(fold_file):
            continue
            
        fold_df = pd.read_csv(fold_file)
        pred_col = f'cv_{i}_pred_{target_col}'
        
        if pred_col not in fold_df.columns:
            continue

        #  - Placeholder for the concept of plotting
        
        # --- A. Per Dataset Analysis ---
        if 'Prediction_split_name' in fold_df.columns:
            for pred_split_name in fold_df['Prediction_split_name'].unique():
                results_path = f"{fold_dir}/results/{pred_split_name}"
                path_if_none(results_path)
                
                data_subset = fold_df[fold_df['Prediction_split_name'] == pred_split_name].reset_index(drop=True)
                subset_metrics = calculate_metrics(data_subset[target_col], data_subset[pred_col])
                
                # Plot
                if subset_metrics['n_vals'] >= 2:
                    actual = data_subset[target_col]
                    pred = data_subset[pred_col]
                    mask = ~(actual.isna() | pred.isna() | np.isinf(actual) | np.isinf(pred))
                    
                    plt.figure(figsize=(6, 6))
                    plt.scatter(pred[mask], actual[mask], color='black', alpha=0.6)
                    
                    # Add reference line
                    mn, mx = min(actual.min(), pred.min()), max(actual.max(), pred.max())
                    plt.plot([mn, mx], [mn, mx], 'r--')
                    
                    plt.xlabel(f'Predicted {target_col}')
                    plt.ylabel(f'Actual {target_col}')
                    plt.title(f'{pred_split_name} (n={subset_metrics["n_vals"]})')
                    plt.savefig(f"{results_path}/pred_vs_actual.png")
                    plt.close()

                row = {'fold': i, 'note': "insufficient_data" if subset_metrics['n_vals'] < min_values_for_analysis else ""}
                row.update(subset_metrics)
                dataset_metrics_accumulator[pred_split_name].append(row)

        # --- B. Pooled Analysis ---
        pooled_metrics = calculate_metrics(fold_df[target_col], fold_df[pred_col])
        p_row = {'fold': i}
        p_row.update(pooled_metrics)
        pooled_rows.append(p_row)
        
        # Plot Pooled
        if pooled_metrics['n_vals'] >= 2:
            plt.figure(figsize=(6,6))
            plt.scatter(fold_df[pred_col], fold_df[target_col], color='black', alpha=0.5)
            mn, mx = min(fold_df[target_col].min(), fold_df[pred_col].min()), max(fold_df[target_col].max(), fold_df[pred_col].max())
            plt.plot([mn, mx], [mn, mx], 'r--')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Fold {i} Pooled')
            path_if_none(f"{path_to_preds}{split_name}/{tvt}/pooled")
            plt.savefig(f"{path_to_preds}{split_name}/{tvt}/pooled/cv_{i}_pooled.png")
            plt.close()

        # --- C. Binned Analysis ---
        actual_all = fold_df[target_col]
        pred_all = fold_df[pred_col]
        
        for j, label in enumerate(bin_labels):
            low = bin_edges[j]
            high = bin_edges[j+1]
            if j == len(bin_labels) - 1:
                bin_mask = (actual_all >= low) & (actual_all <= high)
            else:
                bin_mask = (actual_all >= low) & (actual_all < high)
            
            bin_metrics = calculate_metrics(actual_all[bin_mask], pred_all[bin_mask])
            c_row = {'fold': i}
            c_row.update(bin_metrics)
            class_metrics_accumulator[label].append(c_row)

    # 3. Save Final Aggregates
    metrics_out_path = f"{path_to_preds}{split_name}/{tvt}/dataset"
    path_if_none(metrics_out_path)
    for dataset_name, metrics_list in dataset_metrics_accumulator.items():
        if metrics_list:
            df = pd.DataFrame(metrics_list)
            # Add mean
            if len(df) > 0:
                 mean_row = df.select_dtypes(include=[np.number]).mean()
                 mean_row = mean_row.astype(object)
                 mean_row['fold'] = 'Mean'
                 df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
            df.to_csv(f"{metrics_out_path}/{dataset_name}_metrics.csv", index=False)

    if pooled_rows:
        path_if_none(f"{path_to_preds}{split_name}/{tvt}/pooled")
        pd.DataFrame(pooled_rows).to_csv(f"{path_to_preds}{split_name}/{tvt}/pooled/pooled_metrics.csv", index=False)
        
    classes_out_path = f"{path_to_preds}{split_name}/{tvt}/classes"
    path_if_none(classes_out_path)
    for label, rows in class_metrics_accumulator.items():
        if rows:
            class_df = pd.DataFrame(rows)
            # Add mean
            if len(class_df) > 0:
                 mean_row = class_df.select_dtypes(include=[np.number]).mean()
                 mean_row = mean_row.astype(object)
                 mean_row['fold'] = 'Mean'
                 class_df = pd.concat([class_df, pd.DataFrame([mean_row])], ignore_index=True)
            class_df.to_csv(f"{classes_out_path}/metrics_{label}.csv", index=False)


def main(argv):
    test_dir = argv[1]
    mode_arg = test_dir.split("_")[1]    

    if mode_arg == 'del':
        target_cols = ['quantified_delivery']
        print("Testing for Delivery (quantified_delivery)")
        analysis_bins = [-10, -1, 1, 10] 
    elif mode_arg == 'tox':
        target_cols = ['quantified_toxicity']
        print("Testing for Toxicity (quantified_toxicity)")
        analysis_bins = [0.0, 0.7, 0.8, 1.1]
    else:
        print(f"Error: Unknown mode '{mode_arg}'. Use 'del' or 'tox'.")
        sys.exit(1)

    cv_num = 5        
    model_dir = test_dir
    to_eval = ["test", "train", "valid"]
    
    for i, arg in enumerate(argv):
        if arg.replace('–', '-') == '--diff_model':
            model_dir = argv[i+1]
        if arg.replace('–', '-') == '--cv':
            cv_num = int(argv[i+1])

    # Ensure model folder is set correctly for parent function calls
    # If the user passed a direct model path, we use it, otherwise we use the test_dir name
    
    for tvt in to_eval:
        print(f"--- Processing {tvt} ---")
        make_pred_vs_actual_tvt(
            test_dir, 
            model_dir, 
            ensemble_size=cv_num, 
            tvt=tvt,
            target_columns=target_cols
        )
        
        analyze_predictions_cv_tvt(
            test_dir, 
            ensemble_number=cv_num, 
            tvt=tvt,
            target_columns=target_cols,
            class_bins=analysis_bins
        )

if __name__ == '__main__':
    main(sys.argv)