import numpy as np 
import os
import pandas as pd  
import matplotlib.pyplot as plt 
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score, 
    cohen_kappa_score, 
    log_loss, 
    f1_score, 
    precision_score,
    recall_score,
    confusion_matrix, 
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
from helpers import path_if_none
import pickle
from helpers import path_if_none, change_column_order, load_datapoints_basic
from helpers import smiles_to_fingerprint

# ------------------------------------------------------------------------------
# 1. MAKE PREDICTIONS (Probabilities + Hard Labels)
# ------------------------------------------------------------------------------
def make_preds_basic(
    model_dir="../data",
    data_dir="../data",
    tvt="test",
    cv=5,
    df_test=None, 
    preds_dir="../data",
    target_columns=["class_0", "class_1", "class_2", "class_3"] 
):
    print(f"Running predict for CV {cv}...")

    # Load data
    target_datapoints = load_datapoints_basic(
        os.path.join(data_dir, f"{tvt}.csv"),
        os.path.join(data_dir, f"{tvt}_extra_x.csv"),
        target_columns=target_columns
    )
    
    aligned_smiles = [dp["smiles"] for dp in target_datapoints]

    # --- Load Model ---
    model_path = os.path.join(model_dir, f"model_{cv}", "basic_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    with open(model_path, "rb") as f:
        rf_model = pickle.load(f)

    # --- Load Scaler (if exists) ---
    scaler_path = os.path.join(model_dir, f"model_{cv}", "extra_features_scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    else:
        scaler = None
        print("Warning: Scaler not found, proceeding without scaling extra features.")

    # --- Build Feature Matrix ---
    X = []
    for dp in target_datapoints:
        fp = smiles_to_fingerprint(dp["smiles"])
        x_d = np.array(dp["x_d"]) if dp["x_d"] is not None else np.array([])
        
        if scaler is not None and len(x_d) > 0:
            x_d = scaler.transform([x_d])[0]
            
        feats = np.concatenate([fp, x_d])
        X.append(feats)
    X = np.array(X)

    # --- Generate Classification Predictions ---
    # 1. Get Probabilities
    probs = rf_model.predict_proba(X)
    
    # Normalize probabilities to ensure they sum to exactly 1.0
    row_sums = probs.sum(axis=1, keepdims=True)
    probs = probs / row_sums
    
    # 2. Get Hard Class Labels
    hard_preds = np.argmax(probs, axis=1)

    # # get probs for classes 1 and 2
    # p12 = probs[:, [1, 2]]
    # best_class = np.argmax(p12, axis=1)      # 0 or 1 → maps to class 1 or 2
    # best_prob = np.max(p12, axis=1)

    # mask = best_prob >= 0.65
    # hard_preds[mask] = best_class[mask] + 1


    # 3. Create DataFrame with ALL info dynamically
    pred_data = {
        "smiles": aligned_smiles,
        f"cv_{cv}_pred_class": hard_preds
    }
    
    # Dynamic column generation based on number of target columns
    for idx in range(len(target_columns)):
        pred_data[f"cv_{cv}_prob_{idx}"] = probs[:, idx]

    current_predictions = pd.DataFrame(pred_data)

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
    target_columns=["class_0", "class_1", "class_2", "class_3"] 
):
    class_cols = target_columns

    for cv in range(ensemble_size):
        print(f"Processing CV {cv}")
        model_path_base = f'../data/crossval_splits/{model_folder}/cv_{cv}'
        results_dir = f'../results/crossval_splits/{split_folder}/{tvt}/cv_{cv}'
        
        data_dir = model_path_base
        if tvt == 'test':
            data_dir = f'../data/crossval_splits/{split_folder}/test'

        # Load Input Data (Ground Truth)
        df_test = pd.read_csv(f'{data_dir}/{tvt}.csv')
        metadata = pd.read_csv(f'{data_dir}/{tvt}_metadata.csv')
        output = pd.concat([metadata, df_test], axis=1)

        # Convert One-Hot Ground Truth to Single Label
        if all(col in output.columns for col in class_cols):
            output['target_class'] = np.argmax(output[class_cols].values, axis=1)
        else:
            raise KeyError(f"Input CSV missing one of these columns: {class_cols}")

        preds_dir = f'../data/crossval_splits/{split_folder}/preds/{tvt}'
        path_if_none(results_dir)
        path_if_none(preds_dir)

        try:
            existing = pd.read_csv(f'{results_dir}/predicted_vs_actual.csv')
            print("pred vs actual already exists, skipping creation")
            continue 
        except:
            pass 

        try:
            current_predictions = pd.read_csv(f'{preds_dir}/cv_{cv}_preds.csv')
            print("already have preds.csv")
        except:
            current_predictions = make_preds_basic(
                model_dir=model_path_base, 
                data_dir=data_dir, 
                tvt=tvt, 
                cv=cv, 
                df_test=df_test, 
                preds_dir=preds_dir,
                target_columns=class_cols
            )
        
        if 'smiles' in current_predictions.columns:
            current_predictions.drop(columns=['smiles'], inplace=True)

        output = pd.concat([output, current_predictions], axis=1)

        pred_split_variables = ['Experiment_ID']
        if all(v in output.columns for v in pred_split_variables):
            output['Prediction_split_name'] = output.apply(
                lambda row: '_'.join(str(row[v]) for v in pred_split_variables), axis=1
            )
        else:
            output['Prediction_split_name'] = "All_Data"

        pred_cols = [c for c in current_predictions.columns if "pred" in c or "prob" in c]
        new_cols = pred_cols + ["target_class", "smiles"] 
        
        path = f'{results_dir}/predicted_vs_actual.csv'
        change_column_order(path, output, first_cols=new_cols)



def analyze_predictions_cv_tvt(
    split_name,
    pred_split_variables=['Experiment_ID'],
    path_to_preds='../results/crossval_splits/',
    ensemble_number=5,
    min_values_for_analysis=10,
    tvt='test',
    target_columns=["class_0", "class_1", "class_2", "class_3"]
):
    target_col = 'target_class'
    
    # Define dynamic variables based on target_columns
    n_classes = len(target_columns)
    class_labels = list(range(n_classes))
    display_labels = [str(c) for c in class_labels]

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

    # --- 2. Process Granular Subsets (e.g., per Experiment) ---
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

            pred_label_col = f'cv_{i}_pred_class'
            prob_cols = [f'cv_{i}_prob_{k}' for k in range(n_classes)]
            
            # Create target_class if missing
            if target_col not in data_subset.columns:
                if all(c in data_subset.columns for c in target_columns):
                    data_subset[target_col] = np.argmax(data_subset[target_columns].values, axis=1)
                else:
                    continue

            actual = data_subset[target_col]
            pred_label = data_subset[pred_label_col]
            
            if all(c in data_subset.columns for c in prob_cols):
                pred_probs = data_subset[prob_cols].values
            else:
                pred_probs = None

            mask = ~(actual.isna() | pred_label.isna())
            actual = actual[mask].astype(int)
            pred_label = pred_label[mask].astype(int)
            if pred_probs is not None:
                pred_probs = pred_probs[mask]

            n_vals = len(actual)
            n_classes_present = len(np.unique(actual))

            # --- Calculate Metrics (Subset Level) ---
            # Initialize vars as nan
            roc_auc_macro = roc_auc_weighted = pr_auc_macro = pr_auc_weighted = np.nan
            
            if n_vals < 2:
                acc = kappa = ll = macro_f1 = weighted_f1 = macro_prec = macro_rec = np.nan
            else:
                acc = accuracy_score(actual, pred_label)
                
                if n_classes_present > 1 or len(np.unique(pred_label)) > 1:
                    kappa = cohen_kappa_score(actual, pred_label)
                else:
                    kappa = 1.0 if np.array_equal(actual, pred_label) else 0.0

                macro_f1 = f1_score(actual, pred_label, average='macro', labels=class_labels, zero_division=0)
                weighted_f1 = f1_score(actual, pred_label, average='weighted', labels=class_labels, zero_division=0)
                macro_prec = precision_score(actual, pred_label, average='macro', zero_division=0, labels=class_labels)
                macro_rec = recall_score(actual, pred_label, average='macro', zero_division=0, labels=class_labels)
                
                if pred_probs is not None:
                    try:
                        # Normalize probs
                        row_sums = pred_probs.sum(axis=1, keepdims=True)
                        row_sums[row_sums == 0] = 1 
                        pred_probs_norm = pred_probs / row_sums
                        
                        ll = log_loss(actual, pred_probs_norm, labels=class_labels)
                            
                    except ValueError:
                        ll = np.nan
                else:
                    ll = np.nan

                # Plot Matrix
                cm = confusion_matrix(actual, pred_label, labels=class_labels)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
                
                fig, ax = plt.subplots(figsize=(6, 6))
                disp.plot(cmap='Blues', values_format='d', colorbar=False, ax=ax)
                ax.set_title(f'{pred_split_name} (Fold {i})\nAcc: {acc:.2f} | F1: {macro_f1:.2f}')
                fig.tight_layout()
                fig.savefig(f'{results_path}/confusion_matrix.png')
                plt.close(fig) 

            metrics_row = {
                'fold': i,
                'accuracy': acc,
                'kappa': kappa,
                'log_loss': ll,
                'macro_f1': macro_f1,
                'weighted_f1': weighted_f1,
                'macro_precision': macro_prec, 
                'macro_recall': macro_rec,
                # 'roc_auc_macro': roc_auc_macro,
                # 'roc_auc_weighted': roc_auc_weighted,
                # 'pr_auc_macro': pr_auc_macro,
                # 'pr_auc_weighted': pr_auc_weighted,
                'n_vals': n_vals,
                'note': "insufficient_data" if n_vals < min_values_for_analysis else ""
            }
            dataset_metrics_accumulator[pred_split_name].append(metrics_row)

    # Save Subset Metrics
    metrics_out_dir = f"{path_to_preds}{split_name}/{tvt}/metrics/"
    path_if_none(metrics_out_dir)
    for name, metrics in dataset_metrics_accumulator.items():
        if metrics:
            pd.DataFrame(metrics).to_csv(f"{metrics_out_dir}{name}_metrics.csv", index=False)


    # Pooled Analysis 
    print("Running Pooled Analysis (Per Fold)...")
    pooled_dir = f"{path_to_preds}{split_name}/{tvt}/pooled"
    path_if_none(pooled_dir)
    
    pooled_fold_metrics = [] 
    
    for i in range(ensemble_number):
        try:
            df = pd.read_csv(f"{path_to_preds}{split_name}/{tvt}/cv_{i}/predicted_vs_actual.csv")
            
            p_col = f'cv_{i}_pred_class'
            prob_cols = [f'cv_{i}_prob_{k}' for k in range(n_classes)]
            
            if target_col not in df.columns:
                 if all(c in df.columns for c in target_columns):
                    df[target_col] = np.argmax(df[target_columns].values, axis=1)

            if target_col in df.columns and p_col in df.columns:
                cols_to_use = [target_col, p_col]
                has_probs = all(c in df.columns for c in prob_cols)
                if has_probs:
                    cols_to_use += prob_cols
                
                sub = df[cols_to_use].dropna()
                if len(sub) == 0:
                    continue

                actual = sub[target_col].values.astype(int)
                pred = sub[p_col].values.astype(int)
                probs = sub[prob_cols].values if has_probs else None

                # Metrics
                acc = accuracy_score(actual, pred)
                
                if len(np.unique(actual)) > 1 or len(np.unique(pred)) > 1:
                    kappa = cohen_kappa_score(actual, pred)
                else:
                    kappa = 1.0 if np.array_equal(actual, pred) else 0.0

                f1 = f1_score(actual, pred, average='macro', labels=class_labels, zero_division=0)
                w_f1 = f1_score(actual, pred, average='weighted', labels=class_labels, zero_division=0)
                prec = precision_score(actual, pred, average='macro', zero_division=0, labels=class_labels)
                rec = recall_score(actual, pred, average='macro', zero_division=0, labels=class_labels)
                
                ll = np.nan
                roc_auc_macro = roc_auc_weighted = pr_auc_macro = pr_auc_weighted = np.nan

                if probs is not None:
                    try:
                        row_sums = probs.sum(axis=1, keepdims=True)
                        row_sums[row_sums == 0] = 1
                        probs_norm = probs / row_sums
                        ll = log_loss(actual, probs_norm, labels=class_labels)
                        
                        # --- For Pooled, we suppress warnings just in case, but calculate if possible ---
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                            warnings.filterwarnings("ignore", category=UserWarning)

                            if len(np.unique(actual)) > 1:
                                roc_auc_macro = roc_auc_score(
                                    actual, probs_norm, multi_class='ovr', average='macro', labels=class_labels
                                )
                                roc_auc_weighted = roc_auc_score(
                                    actual, probs_norm, multi_class='ovr', average='weighted', labels=class_labels
                                )
                                
                                y_true_bin = label_binarize(actual, classes=class_labels)
                                if y_true_bin.shape[1] == probs_norm.shape[1]:
                                    pr_auc_macro = average_precision_score(y_true_bin, probs_norm, average='macro')
                                    pr_auc_weighted = average_precision_score(y_true_bin, probs_norm, average='weighted')
                                
                    except ValueError:
                        ll = np.nan

                # Plot Pooled Matrix
                cm = confusion_matrix(actual, pred, labels=class_labels)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
                
                fig, ax = plt.subplots(figsize=(8, 8))
                disp.plot(cmap='Blues', values_format='d', ax=ax)
                ax.set_title(f'Pooled Confusion Matrix (Fold {i})\nAcc: {acc:.3f} | Macro F1: {f1:.3f}')
                fig.savefig(f"{pooled_dir}/fold_{i}_pooled_confusion_matrix.png")
                plt.close(fig) 

                pooled_fold_metrics.append({
                    'fold': i,
                    'accuracy': acc,
                    'kappa': kappa,
                    'macro_f1': f1,
                    'weighted_f1': w_f1,
                    'macro_precision': prec,
                    'macro_recall': rec,
                    'roc_auc_macro': roc_auc_macro,
                    'roc_auc_weighted': roc_auc_weighted,
                    'pr_auc_macro': pr_auc_macro,
                    'pr_auc_weighted': pr_auc_weighted,
                    'log_loss': ll,
                    'total_samples': len(actual)
                })

        except FileNotFoundError:
            continue

    if pooled_fold_metrics:
        pd.DataFrame(pooled_fold_metrics).to_csv(f"{pooled_dir}/pooled_metrics_per_fold.csv", index=False)
        
    print("Pooled Analysis Complete.")
