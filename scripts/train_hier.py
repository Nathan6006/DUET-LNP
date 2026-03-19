import numpy as np 
import os
import pandas as pd  
from sklearn.metrics import mean_squared_error 
import pickle
import sys
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
from helpers import path_if_none, load_datapoints_rf
from embeddings import dataset_to_numpy

# --- Keep your custom objectives ---
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

# --- New Training Function ---
def train_hierarchical(split_dir="../data", 
                       save_dir="../data",      
                       cv_fold=0,               
                       smiles_column="smiles", 
                       target_columns=["quantified_toxicity"],
                       threshold=0.8):     
    
    path_if_none(save_dir)

    # 1. Load Data
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

    # 2. Scale Extra Features
    train_x_d = [dp["x_d"] for dp in train_datapoints if dp["x_d"] is not None]
    if len(train_x_d) > 0:
        scaler = StandardScaler()
        scaler.fit(train_x_d)
        
        # Save Scaler
        scaler_path = os.path.join(save_dir, "extra_features_scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
            
        # Apply scaling
        for dp in train_datapoints:
            if dp["x_d"] is not None and len(dp["x_d"]) > 0:
                dp["x_d"] = scaler.transform([dp["x_d"]])[0]
        for dp in val_datapoints:
            if dp["x_d"] is not None and len(dp["x_d"]) > 0:
                dp["x_d"] = scaler.transform([dp["x_d"]])[0]

    # 3. Load Weights and Convert to Numpy
    train_weights = pd.read_csv(os.path.join(split_dir, "train_weights.csv"))["Sample_weight"].values
    
    X_train, y_train = dataset_to_numpy(train_datapoints, smiles_column)
    X_val, y_val     = dataset_to_numpy(val_datapoints, smiles_column)

    print(f"Fold {cv_fold} | Hierarchical Model (Threshold: {threshold})")

    # ---------------------------------------------------------
    # STEP A: Train Binary Classifier (Above vs Below Threshold)
    # ---------------------------------------------------------
    print("  -> Training Binary Classifier...")
    
    # Create binary targets (1 if > threshold, else 0)
    y_train_bin = (y_train > threshold).astype(int)
    y_val_bin   = (y_val > threshold).astype(int)

    # Calculate scale_pos_weight for imbalance
    num_pos = np.sum(y_train_bin)
    num_neg = len(y_train_bin) - num_pos
    scale_weight = num_neg / num_pos if num_pos > 0 else 1.0

    clf = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.7,
        scale_pos_weight=scale_weight, # Handle imbalance automatically
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
        early_stopping_rounds=30,
        eval_metric="logloss"
    )

    clf.fit(
        X_train, y_train_bin,
        sample_weight=train_weights,
        eval_set=[(X_val, y_val_bin)],
        verbose=False
    )
    
    # Evaluate Classifier
    y_pred_bin = clf.predict(X_val)
    acc = accuracy_score(y_val_bin, y_pred_bin)
    f1 = f1_score(y_val_bin, y_pred_bin)
    print(f"     Classifier Acc: {acc:.4f}, F1: {f1:.4f}")

    # ---------------------------------------------------------
    # STEP B: Split Data for Regressors
    # ---------------------------------------------------------
    mask_low_train = y_train <= threshold
    mask_high_train = y_train > threshold
    
    mask_low_val = y_val <= threshold
    mask_high_val = y_val > threshold

    # Low Dataset
    X_train_low, y_train_low = X_train[mask_low_train], y_train[mask_low_train]
    w_train_low = train_weights[mask_low_train]
    X_val_low, y_val_low = X_val[mask_low_val], y_val[mask_low_val]

    # High Dataset
    X_train_high, y_train_high = X_train[mask_high_train], y_train[mask_high_train]
    w_train_high = train_weights[mask_high_train]
    X_val_high, y_val_high = X_val[mask_high_val], y_val[mask_high_val]

    # ---------------------------------------------------------
    # STEP C: Train Low Regressor ( <= Threshold )
    # ---------------------------------------------------------
    print(f"  -> Training Low Regressor (n={len(y_train_low)})...")
    
    reg_low = xgb.XGBRegressor(
        objective=PossMSEObjective(), 
        n_estimators=500, learning_rate=0.03, max_depth=6,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
        reg_lambda=2, reg_alpha=1, gamma=0, n_jobs=-1, random_state=42,
        tree_method="hist", early_stopping_rounds=50
    )

    if len(X_train_low) > 0:
        eval_set_low = [(X_train_low, y_train_low)]
        if len(X_val_low) > 0: eval_set_low.append((X_val_low, y_val_low))
        
        reg_low.fit(
            X_train_low, y_train_low,
            sample_weight=w_train_low,
            eval_set=eval_set_low,
            verbose=False
        )
    else:
        print("     Warning: No training data for Low Regressor.")

    # ---------------------------------------------------------
    # STEP D: Train High Regressor ( > Threshold )
    # ---------------------------------------------------------
    print(f"  -> Training High Regressor (n={len(y_train_high)})...")

    reg_high = xgb.XGBRegressor(
        objective=PossMSEObjective(),
        n_estimators=500, learning_rate=0.03, max_depth=6,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
        reg_lambda=2, reg_alpha=1, gamma=0, n_jobs=-1, random_state=42,
        tree_method="hist", early_stopping_rounds=50
    )

    if len(X_train_high) > 0:
        eval_set_high = [(X_train_high, y_train_high)]
        if len(X_val_high) > 0: eval_set_high.append((X_val_high, y_val_high))

        reg_high.fit(
            X_train_high, y_train_high,
            sample_weight=w_train_high,
            eval_set=eval_set_high,
            verbose=False
        )
    else:
        print("     Warning: No training data for High Regressor.")

    # ---------------------------------------------------------
    # STEP E: Hierarchical Prediction Logic
    # ---------------------------------------------------------
    # 1. Predict Probs/Class
    val_class_preds = clf.predict(X_val) # 0 or 1
    
    # 2. Initialize final array
    y_pred_final = np.zeros_like(y_val)
    
    # 3. Route predictions
    # Indices where classifier predicts Low (0)
    idx_low = np.where(val_class_preds == 0)[0]
    # Indices where classifier predicts High (1)
    idx_high = np.where(val_class_preds == 1)[0]
    
    if len(idx_low) > 0:
        y_pred_final[idx_low] = reg_low.predict(X_val[idx_low])
        
    if len(idx_high) > 0:
        y_pred_final[idx_high] = reg_high.predict(X_val[idx_high])

    # 4. Evaluate Overall Performance
    mse = mean_squared_error(y_val, y_pred_final)
    mae = mean_absolute_error(y_val, y_pred_final)
    print(f"Fold {cv_fold} Final Combined MSE: {mse:.4f}, MAE: {mae:.4f}")

    # ---------------------------------------------------------
    # STEP F: Save Models
    # ---------------------------------------------------------
    with open(os.path.join(save_dir, "model_classifier.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(save_dir, "model_reg_low.pkl"), "wb") as f:
        pickle.dump(reg_low, f)
    with open(os.path.join(save_dir, "model_reg_high.pkl"), "wb") as f:
        pickle.dump(reg_high, f)

def main(argv):

    split_folder = argv[1]
    mode_arg = split_folder.split("_")[1]
    
    if mode_arg == 'del':
        target_cols = ['quantified_delivery']
        print("Training for Delivery (quantified_delivery)")
    elif mode_arg == 'tox':
        target_cols = ['quantified_toxicity']
        print("Training for Toxicity (quantified_toxicity)")
    else:
        print(f"Error: Unknown mode '{mode_arg}'. Use 'del' or 'tox'.")
        sys.exit(1)

    epochs = 50
    cv_num = 5
    threshold = 0.8 # Default threshold
    
    for i, arg in enumerate(argv):
        if arg.replace('–', '-') == '--epochs':
            epochs = int(argv[i+1])
        if arg.replace('–', '-') == '--cv':
            cv_num = int(argv[i+1])
        if arg.replace('–', '-') == '--threshold':
            threshold = float(argv[i+1])
            print(f"Using Threshold: {threshold}")

    for cv in range(cv_num):
        split_dir = '../data/crossval_splits/' + split_folder + '/cv_' + str(cv)
        save_dir = split_dir + '/model_' + str(cv)
        
        train_hierarchical(
            split_dir=split_dir, 
            save_dir=save_dir, 
            cv_fold=cv, 
            target_columns=target_cols,
            threshold=threshold
        )

if __name__ == '__main__':
    main(sys.argv)