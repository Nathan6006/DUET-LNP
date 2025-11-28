import numpy as np 
import os
import pandas as pd  
from sklearn.metrics import mean_squared_error 
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
from helpers import path_if_none, change_column_order, load_datapoints_rf, load_datapoints_tox_only
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs
import xgboost as xgb

#sys.argv = ['tox_testing.py', 'xg_1.1', '--basic', '--cv', '5']


def smiles_to_fingerprint(smiles, radius=2, n_bits=2048, use_counts=False):
    """
    Convert a SMILES string into a Morgan fingerprint.
    
    Args:
        use_counts (bool): If True, returns count vector (ECFP-Counts). 
                           If False, returns bit vector (ECFP/Morgan).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)

    # Correct Import usage for modern RDKit
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    if use_counts:
        # Returns counts of substructures (SparseIntVect)
        fp = gen.GetCountFingerprint(mol)
    else:
        # Returns 0/1 bits (ExplicitBitVect)
        # Note: Modern generators use GetFingerprint for the default bit vector, 
        # NOT GetFingerprintAsBitVect (which is for legacy AllChem generators)
        fp = gen.GetFingerprint(mol)

    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def dataset_to_numpy(datapoints, smiles_column="smiles"):
    X, y = [], []
    valid_indices = [] # Track valid indices to keep X and y aligned
    
    for i, dp in enumerate(datapoints):
        # Safety check for missing extra features
        extra_features = dp.get("x_d", [])
        if extra_features is None: 
            extra_features = []
            
        fp = smiles_to_fingerprint(dp[smiles_column])
        
        # Ensure dimensionality matches
        feats = np.concatenate([fp, np.array(extra_features)])
        X.append(feats)
        
        # specific handling for target existence
        target = dp.get("y", [None])[0]
        y.append(target)

    return np.array(X), np.array(y)

def train_rf(split_dir="../data", 
             save_dir="../data",      # Reverted to save_dir to match your script call
             cv_fold=0,               
             smiles_column="smiles", 
             target_columns=["quantified_toxicity"],
             model_type="rf",         # New argument: "rf" or "xg"
             n_estimators=200, 
             max_depth=10, 
             min_samples_leaf=15, 
             max_features="sqrt"):
    
    # Ensure the specific fold/model directory exists before trying to save files into it.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created directory: {save_dir}")

    print(f"Loading data from {split_dir}...")
    # Load raw datapoints
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

    # --- Feature Scaling ---
    # Fit StandardScaler on training extra features and save it
    print("Fitting and saving scaler...")
    train_x_d = [dp["x_d"] for dp in train_datapoints if dp["x_d"] is not None]
    
    if len(train_x_d) > 0:
        scaler = StandardScaler()
        scaler.fit(train_x_d)
        
        scaler_path = os.path.join(save_dir, "extra_features_scaler.pkl")
        # This will now work because save_dir was created above
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
            
        # Apply scaling to train data in memory
        for dp in train_datapoints:
            if dp["x_d"] is not None and len(dp["x_d"]) > 0:
                dp["x_d"] = scaler.transform([dp["x_d"]])[0]
        
        # Apply scaling to val data in memory
        for dp in val_datapoints:
            if dp["x_d"] is not None and len(dp["x_d"]) > 0:
                dp["x_d"] = scaler.transform([dp["x_d"]])[0]
    else:
        print("Warning: No extra features found to scale.")

    # Load weights
    train_weights = pd.read_csv(os.path.join(split_dir, "train_weights.csv"))["Sample_weight"].values

    # Convert to numpy
    X_train, y_train = dataset_to_numpy(train_datapoints, smiles_column)
    X_val, y_val     = dataset_to_numpy(val_datapoints, smiles_column)

    print(f"Training {model_type.upper()} for fold {cv_fold}...")
    
    if model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == "xg":
        # XGBoost Regressor
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=42,
            tree_method="hist"  # Often faster for larger datasets
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'rf' or 'xg'.")

    # Both RF and XGBoost support sample_weight in fit()
    model.fit(X_train, y_train, sample_weight=train_weights)

    # Evaluate
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"Fold {cv_fold} Validation MSE: {mse:.4f}, MAE: {mae:.4f}")

    model_path = os.path.join(save_dir, "basic_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {model_path}")

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

def train_cm(
    split_dir='../data',
    smiles_column='smiles',
    target_columns=["quantified_toxicity"],   # only toxicity
    epochs=50,
    save_dir='../data'
):
    train_datapoints = load_datapoints_tox_only(
        os.path.join(split_dir, 'train.csv'),
        os.path.join(split_dir, 'train_extra_x.csv')
    )
    val_datapoints = load_datapoints_tox_only(
        os.path.join(split_dir, 'valid.csv'),
        os.path.join(split_dir, 'valid_extra_x.csv')
    )

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    agg = nn.MeanAggregation()

    # Load pretrained Chemeleon checkpoint
    chemeleon_mp = torch.load("chemeleon_mp.pt", weights_only=False)
    mp = nn.BondMessagePassing(**chemeleon_mp['hyper_parameters'])
    mp.load_state_dict(chemeleon_mp['state_dict'])

    train_weights = pd.read_csv(os.path.join(split_dir, "train_weights.csv"))["Sample_weight"].values
    val_weights   = pd.read_csv(os.path.join(split_dir, "valid_weights.csv"))["Sample_weight"].values
    train_dset = data.MoleculeDataset(train_datapoints, featurizer, sample_weights=train_weights)
    val_dset   = data.MoleculeDataset(val_datapoints, featurizer, sample_weights=val_weights)

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
        activation="RELU",
        criterion=nn.metrics.MAE()
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
        save_top_k=1,
        save_last=False
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
        os.path.join(save_dir, "model.pt")
    )

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
        path_if_none(f"{crossval_results_path}/metrics/")
        fold_df.to_csv(f"{crossval_results_path}/metrics/cv_{i}metrics.csv", index=False)

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

            # --- NEW: per-fold pooled plot ---
            plt.figure()
            plt.scatter(pred, actual, color='black')
            plt.plot(np.unique(pred), np.poly1d(np.polyfit(pred, actual, 1))(np.unique(pred)))
            plt.xlabel(f'Predicted toxicity (cv_{i})')
            plt.ylabel('Quantified toxicity')
            plt.title(f'Fold {i} pooled predictions vs actual')
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
    epochs = 50
    cv_num = 2
    basic=False
    for i, arg in enumerate(argv):
        if arg.replace('–', '-') == '--epochs':
            epochs = int(argv[i+1])
            print('this many epochs: ',str(epochs))
        if arg.replace('–', '-') == '--cv':
            cv_num = int(argv[i+1])
            print('this many folds: ',str(cv_num))
        if arg.replace('–', '-') == '--basic':
            print("using basic model")
            basic = True
    if basic:
        for cv in range(cv_num):
            split_dir = '../data/crossval_splits/'+split_folder+'/cv_'+str(cv)
            save_dir = split_dir+'/model_'+str(cv)
            train_rf(split_dir=split_dir, save_dir=save_dir, model_type="xg")
    else:      
        for cv in range(cv_num):
            split_dir = '../data/crossval_splits/'+split_folder+'/cv_'+str(cv)
            save_dir = split_dir+'/model_'+str(cv)
            train_cm(split_dir=split_dir,epochs=epochs, save_dir=save_dir)
        
    
    test_dir = split_folder
    model_dir = test_dir
    s = False
    to_eval = ["test", "train", "valid"]
    for i, arg in enumerate(argv):
        if arg.replace('–', '-') == '--diff_model':
            model_dir = argv[i+1]
        if arg.replace('–', '-') == '--standardize':
            s = True 
            print('standardize')
    for tvt in to_eval:
        print("make pva")
        make_pred_vs_actual_tvt(test_dir, model_dir, ensemble_size = cv_num, standardize_predictions= s, tvt=tvt, rf=basic)
        print("analyze preds")
        print(cv)
        analyze_predictions_cv_tvt(test_dir, ensemble_number= cv_num, tvt=tvt)
        print("done with:", tvt)


    
if __name__ == '__main__':
    main(sys.argv)




def train_basic(split_dir ='../data', smiles_column='smiles', target_columns = ["quantified_delivery", "quantified_toxicity"], epochs=50, save_dir='../data'):

    train_datapoints = load_datapoints_tox_only(split_dir+'/train.csv', split_dir+'/train_extra_x.csv')
    val_datapoints   = load_datapoints_tox_only(split_dir+'/valid.csv', split_dir+'/valid_extra_x.csv')
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

    #maybe try one sigmoidl layer for activation in the future 
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
