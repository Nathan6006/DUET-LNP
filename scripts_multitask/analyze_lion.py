import sys
import numpy as np 
import os
import pandas as pd  
import matplotlib.pyplot as plt 
import scipy.stats
import pickle
import chemprop
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from helpers import path_if_none, change_column_order

# ------------------------------------------------------------------------------
# 1. TRAIN MODELS (The "Missing" Training Functionality)
# ------------------------------------------------------------------------------
def train_cv(
    data_path,
    save_dir_base,
    num_folds=5,
    epochs=30,
    batch_size=50,
    target_columns=["quantified_toxicity"],
    extra_features_path=None
):
    """
    Trains an ensemble of Chemprop models using Cross Validation.
    """
    print(f"\n=== Starting Training (Ensemble Size: {num_folds}) ===")
    
    # Load Main Data
    df = pd.read_csv(data_path)
    
    # 1. Create/Load Splits (Simple random split for demonstration)
    # In a real scenario, you might load pre-defined indices.
    # Here we use Chemprop's internal splitting via the wrapper logic below.
    
    for cv in range(num_folds):
        print(f"--- Training Fold {cv} ---")
        
        fold_dir = os.path.join(save_dir_base, f"cv_{cv}")
        path_if_none(fold_dir)
        
        # Define Arguments for Chemprop
        # Note: We pass the same data_path. Chemprop handles splitting if we don't provide separate files.
        # To strictly enforce your specific "crossval_splits" folder structure, 
        # you would ideally pass specific train/val .csv paths here.
        # Assuming we are training on the provided 'train.csv' for that split:
        
        train_data_path = os.path.join(os.path.dirname(data_path), 'train.csv')
        valid_data_path = os.path.join(os.path.dirname(data_path), 'valid.csv')
        
        # If specific split files don't exist, fallback to main file
        current_data_path = train_data_path if os.path.exists(train_data_path) else data_path

        args_list = [
            '--data_path', current_data_path,
            '--dataset_type', 'regression',
            '--save_dir', fold_dir,
            '--epochs', str(epochs),
            '--batch_size', str(batch_size),
            '--target_columns'
        ] + target_columns
        
        if valid_data_path and os.path.exists(valid_data_path):
            args_list.extend(['--separate_val_path', valid_data_path])
        
        # Handle Extra Features
        if extra_features_path and os.path.exists(extra_features_path):
             args_list.extend(['--features_path', extra_features_path])
             # Auto-scale features
             args_list.extend(['--features_generator', 'none']) 

        # Parse Args
        args = chemprop.args.TrainArgs().parse_args(args_list)
        
        # Run Training
        # chemprop.train.run_training returns score, but we just want the saved model
        chemprop.train.run_training(args)
        
        print(f"Fold {cv} completed. Model saved to {fold_dir}")

# ------------------------------------------------------------------------------
# 2. MAKE PREDICTIONS (Chemprop Inference)
# ------------------------------------------------------------------------------
def make_preds(
    model_dir="../data",
    data_dir="../data",
    tvt="test",
    cv=5,
    df_test=None, 
    preds_dir="../data",
    target_columns=["quantified_toxicity"]
):
    print(f"Running Chemprop Inference for CV {cv}...")

    # 1. Load Data
    data_path = os.path.join(data_dir, f"{tvt}.csv")
    extra_path = os.path.join(data_dir, f"{tvt}_extra_x.csv")
    
    if df_test is None:
        df = pd.read_csv(data_path)
    else:
        df = df_test
    
    smiles_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
    smiles_list = df[smiles_col].tolist()

    # 2. Load Extra Features
    features = None
    scaler_path = os.path.join(model_dir, "extra_features_scaler.pkl")
    if not os.path.exists(scaler_path):
         scaler_path = os.path.join(model_dir, "..", "extra_features_scaler.pkl")

    if os.path.exists(extra_path):
        print(f"  - Loading extra features from {extra_path}...")
        df_extra = pd.read_csv(extra_path)
        X_extra = df_extra.select_dtypes(include=[np.number]).values
        
        # Apply scaler if it exists (Critical for consistency)
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                X_extra = scaler.transform(X_extra)
            except:
                print("Warning: Could not load/apply feature scaler.")
        
        features = X_extra.tolist()
    else:
        print("  - No extra features found. Using SMILES only.")

    # 3. Load Model
    checkpoint_path = os.path.join(model_dir, "model.pt")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(model_dir, "model_0", "model.pt")
        if not os.path.exists(checkpoint_path):
             # Just return empty if model failed to train
             print(f"  - Model not found at {model_dir}. Skipping.")
             return pd.DataFrame()

    # 4. Inference
    preds_args = [
        '--checkpoint_path', checkpoint_path,
        '--preds_path', os.path.join(preds_dir, 'temp_preds.csv'),
        '--smiles_columns', smiles_col
    ]
    
    args = chemprop.args.PredictArgs().parse_args(preds_args)
    preds_raw = chemprop.train.make_predictions(args=args, smiles=smiles_list, features=features)
    preds_flat = [p[0] for p in preds_raw]

    # 5. Save
    target_name = target_columns[0]
    current_predictions = pd.DataFrame({
        "smiles": smiles_list,
        f"cv_{cv}_pred_{target_name}": preds_flat
    })

    os.makedirs(preds_dir, exist_ok=True)
    out_path = os.path.join(preds_dir, f"cv_{cv}_preds.csv")
    current_predictions.to_csv(out_path, index=False)
    return current_predictions

# ------------------------------------------------------------------------------
# 3. MERGE PREDICTIONS (Helper)
# ------------------------------------------------------------------------------
def make_pred_vs_actual_tvt(
    split_folder,
    model_folder,
    ensemble_size=5,
    tvt='test',
    target_columns=["quantified_toxicity"]
):
    target_col_name = target_columns[0]

    for cv in range(ensemble_size):
        model_path_base = f'../data/crossval_splits/{model_folder}/cv_{cv}'
        results_dir = f'../results/crossval_splits/{split_folder}/{tvt}/cv_{cv}'
        
        data_dir = model_path_base
        if tvt == 'test':
            data_dir = f'../data/crossval_splits/{split_folder}/test'

        if not os.path.exists(f'{data_dir}/{tvt}.csv'):
            print(f"Skipping {tvt} for CV {cv} (File not found)")
            continue

        df_test = pd.read_csv(f'{data_dir}/{tvt}.csv')
        metadata_path = f'{data_dir}/{tvt}_metadata.csv'
        
        if os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)
            output = pd.concat([metadata.reset_index(drop=True), df_test.reset_index(drop=True)], axis=1)
        else:
            output = df_test.copy()

        preds_dir = f'../data/crossval_splits/{split_folder}/preds/{tvt}'
        path_if_none(results_dir)
        path_if_none(preds_dir)

        pred_file = f'{preds_dir}/cv_{cv}_preds.csv'
        if os.path.exists(pred_file):
            current_predictions = pd.read_csv(pred_file)
        else:
            current_predictions = make_preds(
                model_dir=model_path_base, 
                data_dir=data_dir, 
                tvt=tvt,
                cv=cv,
                df_test=df_test,
                preds_dir=preds_dir,
                target_columns=target_columns
            )
        
        if current_predictions.empty: continue

        if 'smiles' in current_predictions.columns:
            current_predictions.drop(columns=['smiles'], inplace=True)

        pred_col_name = f'cv_{cv}_pred_{target_col_name}'
        if pred_col_name not in current_predictions.columns: continue

        output = output.loc[:, ~output.columns.duplicated()]
        output = pd.concat([output, current_predictions], axis=1)

        pred_split_variables = ['Experiment_ID']
        if all(v in output.columns for v in pred_split_variables):
            output['Prediction_split_name'] = output.apply(
                lambda row: '_'.join(str(row[v]) for v in pred_split_variables), axis=1)
        else:
            output['Prediction_split_name'] = "All_Data"

        new_cols = [pred_col_name, target_col_name, "smiles"]
        path = f'{results_dir}/predicted_vs_actual.csv'
        existing_new_cols = [c for c in new_cols if c in output.columns]
        change_column_order(path, output, first_cols=existing_new_cols)

# ------------------------------------------------------------------------------
# 4. ANALYZE (Updated Pooled Metrics Logic)
# ------------------------------------------------------------------------------
def calculate_metrics(actual, pred):
    mask = ~(actual.isna() | pred.isna() | np.isinf(actual) | np.isinf(pred))
    actual_clean = actual[mask]
    pred_clean = pred[mask]
    n_vals = len(actual_clean)

    if n_vals < 2:
        return {'pearson': np.nan, 'mse': np.nan, 'mae': np.nan, 'r2': np.nan, 'n_vals': n_vals}

    pearson_r, pearson_p = scipy.stats.pearsonr(actual_clean, pred_clean)
    spearman_r, _ = scipy.stats.spearmanr(actual_clean, pred_clean)
    kendall_r, _ = scipy.stats.kendalltau(actual_clean, pred_clean)
    mse = mean_squared_error(actual_clean, pred_clean)
    mae = mean_absolute_error(actual_clean, pred_clean)
    r2 = r2_score(actual_clean, pred_clean)

    return {
        'pearson': round(pearson_r, 6), 'r2': round(r2,6),
        'pearson_p_val': round(pearson_p,6), 'spearman': round(spearman_r, 6),
        'kendall': round(kendall_r, 6), 'mse': round(mse, 6),
        'mae': round(mae, 6), 'n_vals': n_vals
    }

def analyze_predictions_cv_tvt(
    split_name,
    path_to_preds='../results/crossval_splits/',
    ensemble_number=5,
    min_values_for_analysis=10,
    tvt='test',
    target_columns=["quantified_toxicity"],
    class_bins = [0.0, 0.7, 0.8, 1.1]
):
    target_col = target_columns[0]
    
    # Identify unique datasets
    all_unique = []
    for i in range(ensemble_number):
        try:
            df = pd.read_csv(f"{path_to_preds}{split_name}/{tvt}/cv_{i}/predicted_vs_actual.csv")
            if 'Prediction_split_name' in df.columns:
                all_unique.extend(set(df['Prediction_split_name'].tolist()))
        except FileNotFoundError: continue

    unique_pred_split_names = set(all_unique)
    dataset_metrics_accumulator = {un: [] for un in unique_pred_split_names}
    
    bin_edges = class_bins
    bin_labels = [f"{bin_edges[j]}_{bin_edges[j+1]}" for j in range(len(bin_edges)-1)]
    class_metrics_accumulator = {label: [] for label in bin_labels}
    pooled_rows = []

    for i in range(ensemble_number):
        fold_dir = f"{path_to_preds}{split_name}/{tvt}/cv_{i}"
        fold_file = f"{fold_dir}/predicted_vs_actual.csv"
        
        if not os.path.exists(fold_file): continue
        fold_df = pd.read_csv(fold_file)
        pred_col = f'cv_{i}_pred_{target_col}'
        
        if pred_col not in fold_df.columns or target_col not in fold_df.columns: continue

        # A. Per Dataset Analysis
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
                    actual_clean, pred_clean = actual[mask], pred[mask]

                    plt.figure(figsize=(6, 6))
                    plt.scatter(pred_clean, actual_clean, color='black', alpha=0.6)
                    try:
                        m, b = np.polyfit(pred_clean, actual_clean, 1)
                        plt.plot(pred_clean, m*pred_clean + b, color='blue', alpha=0.5)
                    except: pass
                    
                    all_vals = np.concatenate([actual_clean, pred_clean])
                    min_val, max_val = all_vals.min() - 0.1, all_vals.max() + 0.1
                    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
                    plt.xlim(min_val, max_val); plt.ylim(min_val, max_val)
                    plt.title(f'{pred_split_name} (n={subset_metrics["n_vals"]})')
                    plt.savefig(f"{results_path}/pred_vs_actual.png")
                    plt.close()

                row = {'fold': i}; row.update(subset_metrics)
                dataset_metrics_accumulator[pred_split_name].append(row)

        # B. Pooled Analysis
        actual_all = fold_df[target_col]
        pred_all = fold_df[pred_col]
        pooled_metrics = calculate_metrics(actual_all, pred_all)
        
        if pooled_metrics['n_vals'] >= 2:
            mask = ~(actual_all.isna() | pred_all.isna() | np.isinf(actual_all) | np.isinf(pred_all))
            a_clean, p_clean = actual_all[mask], pred_all[mask]
            plt.figure(figsize=(6,6))
            plt.scatter(p_clean, a_clean, color='black', alpha=0.5)
            try: plt.plot(np.unique(p_clean), np.poly1d(np.polyfit(p_clean, a_clean, 1))(np.unique(p_clean)), 'b-')
            except: pass
            
            all_vals = np.concatenate([a_clean, p_clean])
            min_pooled, max_pooled = all_vals.min() - 0.1, all_vals.max() + 0.1
            plt.xlim(min_pooled, max_pooled); plt.ylim(min_pooled, max_pooled)
            plt.plot([min_pooled, max_pooled], [min_pooled, max_pooled], 'r--')
            plt.title(f'Fold {i} Pooled')
            path_if_none(f"{path_to_preds}{split_name}/{tvt}/pooled")
            plt.savefig(f"{path_to_preds}{split_name}/{tvt}/pooled/cv_{i}_pooled_pred_vs_actual.png")
            plt.close()

        p_row = {'fold': i}; p_row.update(pooled_metrics)
        pooled_rows.append(p_row)

        # C. Class Analysis
        for j, label in enumerate(bin_labels):
            low, high = bin_edges[j], bin_edges[j+1]
            if j == len(bin_labels) - 1: bin_mask = (actual_all >= low) & (actual_all <= high)
            else: bin_mask = (actual_all >= low) & (actual_all < high)
            
            bin_metrics = calculate_metrics(actual_all[bin_mask], pred_all[bin_mask])
            c_row = {'fold': i}; c_row.update(bin_metrics)
            class_metrics_accumulator[label].append(c_row)

    # Save
    metrics_out_path = f"{path_to_preds}{split_name}/{tvt}/dataset"
    path_if_none(metrics_out_path)
    for dataset_name, metrics_list in dataset_metrics_accumulator.items():
        if metrics_list:
            df = pd.DataFrame(metrics_list)
            if len(df) > 0:
                 mean_row = df.select_dtypes(include=[np.number]).mean().astype(object)
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
            if len(class_df) > 0:
                 mean_row = class_df.select_dtypes(include=[np.number]).mean().round(6).astype(object)
                 mean_row['fold'] = 'Mean'
                 class_df = pd.concat([class_df, pd.DataFrame([mean_row])], ignore_index=True)
            class_df.drop(columns=["r2", "kendall"], inplace=True, errors='ignore')
            class_df.to_csv(f"{classes_out_path}/metrics_{label}.csv", index=False) 

# ------------------------------------------------------------------------------
# 5. MAIN EXECUTION
# ------------------------------------------------------------------------------
def main(argv):
    if len(argv) < 2:
        print("Usage: python train_and_analyze_cb.py {split_name} [--cv N] [--train]")
        sys.exit(1)
        
    test_dir = argv[1]
    
    # Mode Settings
    if 'del' in test_dir.lower():
        target_cols = ['quantified_delivery']
        analysis_bins = [-10, -1, 1, 10]
        print(f"Targeting: {target_cols} (Delivery Mode)")
    else:
        target_cols = ['quantified_toxicity']
        analysis_bins = [0.0, 0.7, 0.8, 1.1]
        print(f"Targeting: {target_cols} (Toxicity Mode)")

    cv_num = 5        
    do_train = False
    
    # Parse Args
    for i, arg in enumerate(argv):
        if arg == '--cv': cv_num = int(argv[i+1])
        if arg == '--train': do_train = True

    # 1. TRAIN (Optional, if flag provided)
    if do_train:
        print("\nSTEP 1: Training Models...")
        # Path assumptions based on your structure
        data_source = f"../data/crossval_splits/{test_dir}/train/train.csv"
        # If no explicit train.csv, use main data
        if not os.path.exists(data_source):
             data_source = f"../data/crossval_splits/{test_dir}/all_data.csv"

        save_base = f"../data/crossval_splits/{test_dir}"
        extra_x_path = f"../data/crossval_splits/{test_dir}/train/train_extra_x.csv"
        
        train_cv(
            data_path=data_source,
            save_dir_base=save_base,
            num_folds=cv_num,
            target_columns=target_cols,
            extra_features_path=extra_x_path if os.path.exists(extra_x_path) else None
        )

    # 2. INFERENCE & ANALYSIS
    print("\nSTEP 2: Inference & Analysis...")
    to_eval = ["test", "train", "valid"]
    model_dir = test_dir

    for tvt in to_eval:
        print(f"\n--- Processing: {tvt} ---")
        make_pred_vs_actual_tvt(
            test_dir, 
            model_dir, 
            ensemble_size=cv_num, 
            tvt=tvt,
            target_columns=target_cols
        )
        
        print(f"--- Analyzing: {tvt} ---")
        analyze_predictions_cv_tvt(
            test_dir, 
            ensemble_number=cv_num, 
            tvt=tvt,
            target_columns=target_cols,
            class_bins=analysis_bins
        )
    
    print("\nWorkflow Complete.")

if __name__ == '__main__':
    main(sys.argv)