import sys
import numpy as np 
import os
import pandas as pd  
import matplotlib.pyplot as plt 
import scipy.stats
import pickle
import torch
from transformers import AutoTokenizer
from safetensors.torch import load_file

# 1. IMPORT THE CUSTOM MODEL FROM YOUR TRAINING SCRIPT
# Note: Change 'train_script' to the actual filename of your training script (without the .py)
from train import ChemBERTaMLPRegressor 

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from helpers import path_if_none, change_column_order

# ------------------------------------------------------------------------------
# 1. INFERENCE ENGINE
# ------------------------------------------------------------------------------
def run_inference(model, tokenizer, smiles_list, extra_features=None, max_length=256, batch_size=32):
    """
    Runs end-to-end inference using the ChemBERTaMLPRegressor.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    all_preds = []
    
    # Ensure extra_features is a tensor if provided
    if extra_features is not None:
        extra_features = torch.tensor(extra_features, dtype=torch.float32)
    
    total_samples = len(smiles_list)
    
    for i in range(0, total_samples, batch_size):
        batch_smiles = smiles_list[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_smiles, 
            padding=True, 
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        ).to(device)
        
        # Prepare Batch Extra Features
        batch_extra = None
        if extra_features is not None:
            batch_extra = extra_features[i:i+batch_size].to(device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'], 
                attention_mask=inputs['attention_mask'], 
                extra_features=batch_extra
            )
            # Logits are the predictions directly
            preds = outputs.logits.cpu().numpy()
            all_preds.extend(preds)
            
    return np.array(all_preds)

# ------------------------------------------------------------------------------
# 2. MAKE PREDICTIONS (End-to-End Pipeline)
# ------------------------------------------------------------------------------
def make_preds(
    model_dir="../data",
    data_dir="../data",
    tvt="test",
    cv=5,
    df_test=None, 
    preds_dir="../data",
    rf=False, # Deprecated/Ignored
    target_columns=["quantified_toxicity"]
):
    print(f"\n--- Running Inference for CV {cv} | Split: {tvt} ---")

    # 1. Define Paths
    data_path = os.path.join(data_dir, f"{tvt}.csv")
    extra_path = os.path.join(data_dir, f"{tvt}_extra_x.csv")
    scaler_path = os.path.join(model_dir, f"cv_{cv}", f"model_{cv}", "extra_features_scaler.pkl")

    # DEBUG: Print exact paths so you can see where it's looking
    print(f"  [DEBUG] Data path:   {data_path}")
    print(f"  [DEBUG] Extra X path:{extra_path}")
    print(f"  [DEBUG] Scaler path: {scaler_path}")

    # 2. Load Base Data
    if df_test is None:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Missing base data at: {data_path}")
        df = pd.read_csv(data_path)
    else:
        df = df_test
        
    smiles_col = next((c for c in df.columns if c.lower() == 'smiles'), None)
    if not smiles_col:
        raise ValueError("SMILES column not found in input data")
    
    df = df.dropna(subset=[smiles_col])
    smiles_list = df[smiles_col].astype(str).tolist()

    # 3. Load and Process Extra Features strictly
    # HARD CRASH if files are missing instead of silently ignoring them
    if not os.path.exists(extra_path):
        raise FileNotFoundError(f"STOPPING: Could not find extra features file here: \n{extra_path}")
        
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"STOPPING: Could not find scaler file here: \n{scaler_path}")

    df_extra = pd.read_csv(extra_path)
    if len(df_extra) != len(df):
        if len(df) < len(df_extra):
             df_extra = df_extra.iloc[df.index]
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    X_extra_raw = df_extra.select_dtypes(include=[np.number]).values
    X_extra = scaler.transform(X_extra_raw)
    extra_dim = X_extra.shape[1]
    
    print(f"  - Successfully loaded extra features (Dimension: {extra_dim})")


    # 3. Load Fine-Tuned Model
    ft_model_path = os.path.join(model_dir, f"cv_{cv}", f"model_{cv}", "fine_tuned_chemberta")
    
    if not os.path.exists(ft_model_path):
        raise FileNotFoundError(f"Fine-tuned model not found at {ft_model_path}")

    print("  - Loading Model Weights...")
    tokenizer = AutoTokenizer.from_pretrained(ft_model_path)
    
    # Instantiate the architecture, then load weights.
    base_model_name = "DeepChem/ChemBERTa-77M-MTR"
    model = ChemBERTaMLPRegressor(base_model_name, extra_dim=extra_dim)
    
# Load state dict appropriately based on format
    weights_path_bin = os.path.join(ft_model_path, "pytorch_model.bin")
    weights_path_safe = os.path.join(ft_model_path, "model.safetensors")
    
    try:
        if os.path.exists(weights_path_safe):
            # Use safetensors loader
            state_dict = load_file(weights_path_safe, device="cpu")
            model.load_state_dict(state_dict)
        elif os.path.exists(weights_path_bin):
            # Use standard PyTorch loader
            state_dict = torch.load(weights_path_bin, map_location="cpu", weights_only=False)
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"No valid weights found in {ft_model_path}. Looked for .bin and .safetensors")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        raise e

    # 4. Predict
    print("  - Running Forward Pass...")
    preds = run_inference(model, tokenizer, smiles_list, extra_features=X_extra)

    # 5. Save Results
    target_name = target_columns[0]
    current_predictions = pd.DataFrame({
        "smiles": smiles_list,
        f"cv_{cv}_pred_{target_name}": preds
    })

    os.makedirs(preds_dir, exist_ok=True)
    out_path = os.path.join(preds_dir, f"cv_{cv}_preds.csv")
    current_predictions.to_csv(out_path, index=False)

    return current_predictions


# ------------------------------------------------------------------------------
# 3. MERGE PREDICTIONS WITH ACTUALS 
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
        model_path_base = f'../data/crossval_splits/{model_folder}' 
        results_dir = f'../results/crossval_splits/{split_folder}/{tvt}/cv_{cv}'
        
        data_dir = f'../data/crossval_splits/{split_folder}/cv_{cv}'
        if tvt == 'test':
            data_dir = f'../data/crossval_splits/{split_folder}/test'

        df_path = f'{data_dir}/{tvt}.csv'
        if not os.path.exists(df_path):
            print(f"Skipping {tvt} for CV {cv} - file not found.")
            continue
            
        df_test = pd.read_csv(df_path)
        metadata_path = f'{data_dir}/{tvt}_metadata.csv'
        
        if os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)
            output = pd.concat([metadata, df_test], axis=1)
        else:
            output = df_test.copy()

        preds_dir = f'../data/crossval_splits/{split_folder}/preds/{tvt}'
        path_if_none(results_dir)
        path_if_none(preds_dir)

        pred_file = f'{preds_dir}/cv_{cv}_preds.csv'
        if os.path.exists(pred_file):
            print("  - Predictions already exist.")
            current_predictions = pd.read_csv(pred_file)
        else:
            current_predictions = make_preds(
                model_dir=f'../data/crossval_splits/{model_folder}', 
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
        
        if pred_col_name not in current_predictions.columns:
            print(f"Warning: Expected prediction column {pred_col_name} not found.")
            continue

        if standardize_predictions:
            vals = current_predictions[pred_col_name]
            current_predictions[pred_col_name] = (vals - vals.mean()) / vals.std()

        output = output.loc[:, ~output.columns.duplicated()]
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
        
        existing_new_cols = [c for c in new_cols if c in output.columns]
        change_column_order(path, output, first_cols=existing_new_cols)

def calculate_metrics(actual, pred):
    mask = ~(actual.isna() | pred.isna() | np.isinf(actual) | np.isinf(pred))
    actual_clean = actual[mask]
    pred_clean = pred[mask]
    n_vals = len(actual_clean)

    if n_vals < 2:
        return {
            'pearson': np.nan, 'pearson_p_val': np.nan,
            'spearman': np.nan, 'kendall': np.nan,
            'mse': np.nan, 'mae': np.nan, 'r2': np.nan,
            'n_vals': n_vals
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

    for i in range(ensemble_number):
        fold_dir = f"{path_to_preds}{split_name}/{tvt}/cv_{i}"
        fold_file = f"{fold_dir}/predicted_vs_actual.csv"
        
        if not os.path.exists(fold_file):
            continue
            
        fold_df = pd.read_csv(fold_file)
        pred_col = f'cv_{i}_pred_{target_col}'
        
        if pred_col not in fold_df.columns or target_col not in fold_df.columns:
            continue

        if 'Prediction_split_name' in fold_df.columns:
            for pred_split_name in fold_df['Prediction_split_name'].unique():
                results_path = f"{fold_dir}/results/{pred_split_name}"
                path_if_none(results_path)
                
                data_subset = fold_df[fold_df['Prediction_split_name'] == pred_split_name].reset_index(drop=True)
                subset_metrics = calculate_metrics(data_subset[target_col], data_subset[pred_col])
                
                if subset_metrics['n_vals'] >= 2:
                    actual = data_subset[target_col]
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

                    all_vals = np.concatenate([actual_clean, pred_clean])
                    min_val, max_val = all_vals.min() - 0.1, all_vals.max() + 0.1
                    
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

                row = {'fold': i, 'note': "insufficient_data" if subset_metrics['n_vals'] < min_values_for_analysis else ""}
                row.update(subset_metrics)
                dataset_metrics_accumulator[pred_split_name].append(row)

        actual_all = fold_df[target_col]
        pred_all = fold_df[pred_col]
        pooled_metrics = calculate_metrics(actual_all, pred_all)
        
        if pooled_metrics['n_vals'] >= 2:
            mask = ~(actual_all.isna() | pred_all.isna() | np.isinf(actual_all) | np.isinf(pred_all))
            a_clean = actual_all[mask]
            p_clean = pred_all[mask]

            plt.figure(figsize=(6,6))
            plt.scatter(p_clean, a_clean, color='black', alpha=0.5)
            try:
                plt.plot(np.unique(p_clean), np.poly1d(np.polyfit(p_clean, a_clean, 1))(np.unique(p_clean)), 'b-')
            except: pass
            
            all_vals = np.concatenate([a_clean, p_clean])
            min_pooled, max_pooled = all_vals.min() - 0.1, all_vals.max() + 0.1

            plt.xlim(min_pooled, max_pooled)
            plt.ylim(min_pooled, max_pooled)
            plt.plot([min_pooled, max_pooled], [min_pooled, max_pooled], 'r--')
            plt.xlabel(f'Predicted (cv_{i})')
            plt.ylabel(f'Actual {target_col}')
            plt.title(f'Fold {i} Pooled (All Datasets)')
            
            path_if_none(f"{path_to_preds}{split_name}/{tvt}/pooled")
            plt.savefig(f"{path_to_preds}{split_name}/{tvt}/pooled/cv_{i}_pooled_pred_vs_actual.png")
            plt.close()

        p_row = {'fold': i}
        p_row.update(pooled_metrics)
        pooled_rows.append(p_row)

        for j, label in enumerate(bin_labels):
            low = bin_edges[j]
            high = bin_edges[j+1]
            if j == len(bin_labels) - 1:
                bin_mask = (actual_all >= low) & (actual_all <= high)
            else:
                bin_mask = (actual_all >= low) & (actual_all < high)
            
            bin_actual = actual_all[bin_mask]
            bin_pred = pred_all[bin_mask]
            bin_metrics = calculate_metrics(bin_actual, bin_pred)
            
            c_row = {'fold': i}
            c_row.update(bin_metrics)
            class_metrics_accumulator[label].append(c_row)

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

    if pooled_rows:
        path_if_none(f"{path_to_preds}{split_name}/{tvt}/pooled")
        pooled_metrics_df = pd.DataFrame(pooled_rows)
        pooled_metrics_df.to_csv(f"{path_to_preds}{split_name}/{tvt}/pooled/pooled_metrics.csv", index=False)

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
            class_df.drop(columns=["r2", "kendall"], inplace=True, errors='ignore')
            class_df.to_csv(f"{classes_out_path}/metrics_{label}.csv", index=False) 


def main(argv):
    if len(argv) < 2:
        print("Usage: python test_cb.py {split_name} [--cv N] [--diff_model PATH]")
        sys.exit(1)
        
    test_dir = argv[1]
    
    if 'del' in test_dir.lower():
        target_cols = ['quantified_delivery']
        print("Testing for Delivery (quantified_delivery)")
        analysis_bins = [-10, -1, 1, 10] 
    elif 'tox' in test_dir.lower():
        target_cols = ['quantified_toxicity']
        print("Testing for Toxicity (quantified_toxicity)")
        analysis_bins = [0.0, 0.7, 0.8, 1.1]
    else:
        print("Warning: Could not infer 'del' or 'tox' from name. Defaulting to 'quantified_toxicity'.")
        target_cols = ['quantified_toxicity']
        analysis_bins = [0.0, 0.7, 0.8, 1.1]

    cv_num = 5        
    model_dir = test_dir
    to_eval = ["test", "train", "valid"]
    
    for i, arg in enumerate(argv):
        if arg.replace('–', '-') == '--diff_model':
            model_dir = argv[i+1]
        if arg.replace('–', '-') == '--cv':
            cv_num = int(argv[i+1])

    for tvt in to_eval:
        print(f"\n=== Processing: {tvt} ===")
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
        print(f"Done with {tvt}.")

if __name__ == '__main__':
    main(sys.argv)