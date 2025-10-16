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



# conda create -n lnp_ml python=3.11 
# conda activate lnp_ml 
# python -m pip install chemprop==2.2.1
# if doesn't install dependencies run pip install chemprop==2.2.1 --force-reinstall --no-cache-dir

# general helper functions 
def path_if_none(newpath):
    if not os.path.exists(newpath):
        os.makedirs(newpath)

def load_datapoints(smiles_csv, extra_csv, smiles_column='smiles', target_columns = ["quantified_delivery", "quantified_toxicity"]):
    df_smi = pd.read_csv(smiles_csv)
    df_extra = pd.read_csv(extra_csv)

    smis = df_smi[smiles_column].values
    ys = df_smi[target_columns].values
    extra_features = df_extra.to_numpy(dtype=float)

    datapoints = [
        data.MoleculeDatapoint.from_smi(smi, y, x_d=xf)
        for smi, y, xf in zip(smis, ys, extra_features)
    ]
    return datapoints

def change_column_order(path, all_df, first_cols = ['smiles','quantified_delivery','unnormalized_delivery','quantified_toxicity','unnormalized_toxicity']):
    other_cols = [col for col in all_df.columns if col not in first_cols]
    all_df = all_df[first_cols + other_cols]
    all_df.to_csv(path, index=False)

# these functions called in main 

def load_datapoints(smiles_csv, extra_csv, smiles_column='smiles', target_columns=["quantified_delivery", "quantified_toxicity"]):
    df_smi = pd.read_csv(smiles_csv)
    df_extra = pd.read_csv(extra_csv)

    smis = df_smi[smiles_column].values
    ys = df_smi[target_columns].values
    extra_features = df_extra.to_numpy(dtype=float)

    return [
        data.MoleculeDatapoint.from_smi(smi, y, x_d=xf)
        for smi, y, xf in zip(smis, ys, extra_features)
    ]

def train(split_dir ='../data', smiles_column='smiles', target_columns = ["quantified_delivery", "quantified_toxicity"], epochs=50, save_dir='../data'):

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

def train_cm(split_dir='../data', smiles_column='smiles', target_columns=["quantified_delivery", "quantified_toxicity"], epochs=50, save_dir='../data'):
    

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


    # Now load checkpoint
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

    train_loader = data.build_dataloader(train_dset, shuffle=False, num_workers=6, persistent_workers=True, prefetch_factor=2)
    val_loader = data.build_dataloader(val_dset, shuffle=False, num_workers=6, persistent_workers=True, prefetch_factor=2)

    ffn_input_dim = mp.output_dim + train_dset[0].x_d.shape[0]

    ffn = nn.RegressionFFN(
        n_tasks=len(target_columns),
        input_dim=ffn_input_dim,
        dropout=0.1,
        hidden_dim=600,
        n_layers=3,
        activation="RELU"
    )

    # X_d_transform = nn.ScaleTransform.from_standard_scaler(extra_features_scaler)
    metric_list = [nn.metrics.RMSE(), nn.metrics.MAE()]

    chemprop_model = models.MPNN(
        mp, agg, ffn,
        #X_d_transform=X_d_transform,
        batch_norm=False,
        metrics=metric_list
    )

    os.makedirs(save_dir, exist_ok=True)
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

    #train and save fine-tuned model
    trainer.fit(chemprop_model, train_loader, val_loader)

    torch.save(
        chemprop_model.state_dict(),
        os.path.join(save_dir, "chemprop_chemelon_finetuned.pt")
    )

def make_pred_vs_actual_tvt(split_folder, ensemble_size = 5, standardize_predictions = False, tvt = 'test'):
    # Makes predictions on each test set in a cross-validation-split system
    # Not used for screening a new library, used for predicting on the test set of the existing dataset
    # tvt is train valid or test- determines which files to run analysis on

    for cv in range(ensemble_size):
        print(cv)
        model = '../data/crossval_splits/'+split_folder+'/cv_'+str(cv)
        results_dir = '../results/crossval_splits/'+split_folder+'/'+tvt+'/cv_'+str(cv)
        data_dir = model
        if tvt == 'test':
            data_dir = '../data/crossval_splits/'+split_folder+'/test'
        df_test = pd.read_csv(data_dir+'/'+tvt+'.csv')
        metadata = pd.read_csv(data_dir+'/'+tvt+'_metadata.csv')
        output = pd.concat([metadata, df_test], axis = 1)
        preds_dir = '../data/crossval_splits/'+split_folder+'/preds'+'/'+tvt
        path_if_none(results_dir)
        path_if_none(preds_dir)

        try:  
            output = pd.read_csv(results_dir+'/predicted_vs_actual.csv')
            print("pred vs actual already exist")
        except:
            try:
                current_predictions = pd.read_csv(preds_dir+'/cv_'+str(cv)+'_preds.csv')
                print("already have preds.csv")
            except:
                print("running predict")
                checkpoint_path = model + '/model_'+str(cv)+'/best.ckpt'
                mpnn = models.MPNN.load_from_checkpoint(checkpoint_path)

                test_data = load_datapoints(
                    os.path.join(data_dir, f'{tvt}.csv'),
                    os.path.join(data_dir, f'{tvt}_extra_x.csv')
                )
                featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
                test_dataset = data.MoleculeDataset(test_data, featurizer=featurizer)
                
                #normalize extra features
                scaler_path = os.path.join(f"{model}/model_{cv}/extra_features_scaler.pkl")
                with open(scaler_path, "rb") as f:
                    extra_features_scaler = pickle.load(f)
                test_dataset.normalize_inputs("X_d", extra_features_scaler)
                test_loader = data.build_dataloader(test_dataset, shuffle=False)

                #run preds
                with torch.inference_mode():
                    trainer = pl.Trainer(logger=None, enable_progress_bar=False, accelerator="auto", devices=1)
                    preds = trainer.predict(mpnn, test_loader)
                preds = np.concatenate(preds, axis=0)
        
                # preds = torch.cat([torch.tensor(p) if not isinstance(p, torch.Tensor) else p for p in preds]).cpu().numpy()
                
                current_predictions = pd.DataFrame(preds, columns=["quantified_delivery", "quantified_toxicity"])
                current_predictions["smiles"] = df_test["smiles"].values
                print("about to make preds.csv")
                current_predictions.to_csv(preds_dir+'/cv_'+str(cv)+'_preds.csv', index=False)

            
            current_predictions.drop(columns = ['smiles'], inplace = True)
            for col in current_predictions.columns:
                if standardize_predictions:
                    print("standardizing predictions")
                    preds_to_standardize = current_predictions[col]
                    std = np.std(preds_to_standardize)
                    mean = np.mean(preds_to_standardize)
                    current_predictions[col] = [(val-mean)/std for val in current_predictions[col]]
                current_predictions.rename(columns = {col:('cv_'+str(cv)+'_pred_'+col[:])}, inplace = True)
            output = pd.concat([output, current_predictions], axis = 1)

            pred_split_variables = ['Experiment_ID', 'Library_ID', 'Delivery_target', 'Route_of_administration']

            output['Prediction_split_name'] = output.apply(
                lambda row: '_'.join(str(row[v]) for v in pred_split_variables),
                axis=1
            ) #add prediction_split_name column

            # move new prediction columns to the front
            new_cols = current_predictions.columns.tolist()
            new_cols = new_cols + ["quantified_delivery", "quantified_toxicity", "smiles"]
            path = results_dir+'/predicted_vs_actual.csv'
            change_column_order(path, output, first_cols= new_cols)
    
    if '_with_uho' in split_folder and tvt == 'test':
        print("uho")
        uho_dir = '../data/crossval_splits/'+split_folder+'/ultra_held_out'
        results_dir = uho_dir+'/preds'
        path_if_none(results_dir)
        output = pd.read_csv(uho_dir+'/test.csv')
        metadata = pd.read_csv(uho_dir+'/test_metadata.csv')
        output = pd.concat([metadata, output], axis = 1)

        for cv in range(ensemble_size):
            model_dir = '../data/crossval_splits/'+split_folder+'/cv_'+str(cv)
            try:
                current_predictions = pd.read_csv(results_dir+'/preds_cv_'+str(cv)+'.csv')
            except:
                checkpoint_path = model_dir + '/model_'+str(cv)+'/best.ckpt'
                mpnn = models.MPNN.load_from_checkpoint(checkpoint_path)
                df_test = pd.read_csv(data_dir+'/test.csv')
                extra_df = pd.read_csv(data_dir+'/test_extra_x.csv') 
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
                current_predictions = pd.DataFrame(preds, columns=["quantified_delivery", "quantified_toxicity"])
                current_predictions["smiles"] = df_test["smiles"].values
                current_predictions.to_csv(results_dir+ f'/preds_cv_{cv}.csv', index=False)
            
            current_predictions.drop(columns = ['smiles'], inplace = True)
            for col in current_predictions.columns:
                if standardize_predictions:
                    preds_to_standardize = current_predictions[col]
                    std = np.std(preds_to_standardize)
                    mean = np.mean(preds_to_standardize)
                    current_predictions[col] = [(val-mean)/std for val in current_predictions[col]]
                current_predictions.rename(columns = {col:('cv_'+str(cv)+'_pred_'+col)}, inplace = True)
            output = pd.concat([output, current_predictions], axis = 1)

        pred_delivery_cols = [col for col in output.columns if 'pred_quantified_delivery' in col]
        pred_toxicity_cols = [col for col in output.columns if 'pred_quantified_toxicity' in col]

        output['Avg_pred_quantified_delivery'] = output[pred_delivery_cols].mean(axis = 1)
        output['Avg_pred_quantified_toxicity'] = output[pred_toxicity_cols].mean(axis = 1)

        ultra_dir = '../results/crossval_splits/'+split_folder+'/ultra_held_out'
        path_if_none(ultra_dir)
        path = ultra_dir+'/predicted_vs_actual.csv'
        first_col = ["quantified_delivery", "quantified_toxicity"]

        for cv in range(ensemble_size):
            first_col.append(f"cv_{cv}_pred_quantified_delivery")
            first_col.append(f"cv_{cv}_pred_quantified_toxicity")

        first_col += ["Avg_pred_quantified_delivery", "Avg_pred_quantified_toxicity", "smiles"]

        change_column_order(path, output, first_cols=first_col)

def analyze_predictions_cv_tvt(split_name, pred_split_variables = ['Experiment_ID','Library_ID','Delivery_target','Route_of_administration'], path_to_preds = '../results/crossval_splits/', ensemble_number = 5, min_values_for_analysis = 10, tvt = 'test'):
    all_ns = {}
    all_pearson = {}
    all_pearson_p_val = {}
    all_kendall = {}
    all_spearman = {}
    all_rmse = {}
    all_unique = []
    
    preds_vs_actual = []
    for i in range(ensemble_number):
        df = pd.read_csv(path_to_preds + split_name + '/' + tvt + '/cv_'+str(i)+'/predicted_vs_actual.csv')
        preds_vs_actual.append(df)
        unique = set(df['Prediction_split_name'].tolist())
        all_unique.extend(unique) #get list of all unique split names 

    unique_pred_split_names = set(all_unique)

    for un in unique_pred_split_names:
        all_ns[un] = []
        all_pearson[un] = []
        all_pearson_p_val[un] = []
        all_kendall[un] = []
        all_spearman[un] = []
        all_rmse[un] = []

    target_columns = [('quantified_delivery', 'delivery'), ('quantified_toxicity', 'toxicity')]
    
    for i in range(ensemble_number): #create crossval_performance and results folder 
        crossval_results_path = f"{path_to_preds}{split_name}/{tvt}"
        path_if_none(crossval_results_path)

        fold_results = []  # stores rows for this fold

        fold_df = preds_vs_actual[i]
        fold_unique = fold_df['Prediction_split_name'].unique()

        for pred_split_name in fold_unique: #unique_pred_split_names:
            path_if_none(path_to_preds+split_name+'/'+tvt+'/cv_'+str(i)+'/results/'+pred_split_name)
            data_subset = preds_vs_actual[i][
                preds_vs_actual[i]['Prediction_split_name'] == pred_split_name
            ].reset_index(drop=True)

            value_names = set(list(data_subset.Value_name))

            if len(value_names) > 1:
                raise Exception(
                    f'Multiple types of measurement in the same prediction split: split {pred_split_name} has value names {value_names}. Try adding more pred split variables.'
                )

            for actual_col, label in target_columns:
                if actual_col not in data_subset.columns:
                    print("target missing", actual_col)
                    continue  # skip if target missing in this split

                actual = data_subset[actual_col]
                pred_col = f'cv_{i}_pred_{actual_col}'
                if pred_col not in data_subset.columns:
                    print("predictions missing", pred_col)
                    continue  # skip if predictions missing

                pred = data_subset[pred_col]

                analyzed_data = pd.DataFrame({
                    'smiles': data_subset.smiles, 'actual': actual, 'predicted': pred
                })
                
                mask = ~(actual.isna() | pred.isna() | np.isinf(actual) | np.isinf(pred))
                actual = actual[mask]
                pred = pred[mask]

                n_vals = len(actual)
                if n_vals < 2: #if not enough actual data then cannot run metrics
                    print(f"not enough data:{actual}, {label}")
                    pearson_r = pearson_p = spearman_r = kendall_r = rmse = np.nan
                    analyzed_path = f"{path_to_preds}{split_name}/{tvt}/cv_{i}/results/{pred_split_name}/{label}"
                    path_if_none(analyzed_path)

                    analyzed_data.to_csv(analyzed_path+'/pred_vs_actual_data.csv', index = False)

                else: 
                    pearson_r, pearson_p = scipy.stats.pearsonr(actual, pred)
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
                    
                    analyzed_data.to_csv(analyzed_path+'/pred_vs_actual_data.csv', index = False)
                
                if n_vals < min_values_for_analysis:
                    print(f"Warning: only {n_vals} samples for {pred_split_name}, {label} (below threshold {min_values_for_analysis})")

                fold_results.append({
                    'fold': i,
                    'split_name': pred_split_name,
                    'target': label,
                    'pearson': pearson_r,
                    'pearson_p_val': pearson_p,
                    'spearman': spearman_r,
                    'kendall': kendall_r,
                    'rmse': rmse,
                    'n_vals': n_vals,
                    'note': "insufficient_data" if n_vals < min_values_for_analysis else ""
                })

        fold_df = pd.DataFrame(fold_results)
        fold_df.to_csv(f"{crossval_results_path}/cv_{i}metrics.csv", index=False)


    # Now analyze uho
    try:
        preds_vs_actual = pd.read_csv(path_to_preds + split_name + '/ultra_held_out/predicted_vs_actual.csv')

        preds_vs_actual['Prediction_split_name'] = preds_vs_actual[pred_split_variables].astype(str).agg('_'.join, axis=1)

        unique_pred_split_names = preds_vs_actual['Prediction_split_name'].unique()

        target_cols = [col for col in preds_vs_actual.columns if col.startswith('Avg_pred_')]
        actual_cols = [col.replace('Avg_pred_', '') for col in target_cols]

        metrics_rows = []

        for pred_split_name in unique_pred_split_names:
            data_subset = preds_vs_actual[preds_vs_actual['Prediction_split_name'] == pred_split_name].reset_index(drop=True)

            # loop throughtarget columns (delivery, toxicity, etc)
            for target_col, actual_col in zip(target_cols, actual_cols):
                actual = data_subset[actual_col]
                pred = data_subset[target_col]

                if actual.isna().all() or pred.isna().all(): #skip when all are NaN
                    continue

                mask = ~(actual.isna() | pred.isna() | np.isinf(actual) | np.isinf(pred))
                actual = actual[mask]
                pred = pred[mask]
                if len(actual) < 2: #if not enough actual data then cannot run metrics
                    pearson_r = pearson_p = spearman_r = kendall_r = rmse = np.nan
                    n_vals = len(pred)
                else: 
                    pearson = scipy.stats.pearsonr(actual, pred)
                    spearman, _ = scipy.stats.spearmanr(actual, pred)
                    kendall, _ = scipy.stats.kendalltau(actual, pred)
                    rmse = np.sqrt(mean_squared_error(actual, pred))

                    # Save pred vs actual plot
                    analyzed_path = f"{path_to_preds}{split_name}/ultra_held_out/individual_dataset_results/{pred_split_name}"
                    path_if_none(analyzed_path)
                    plt.figure()
                    plt.scatter(pred, actual, color='black')
                    plt.plot(np.unique(pred), np.poly1d(np.polyfit(pred, actual, 1))(np.unique(pred)))
                    plt.xlabel(f"Predicted {actual_col}")
                    plt.ylabel(f"Experimental {actual_col}")
                    plt.savefig(f"{analyzed_path}/pred_vs_actual_{actual_col}.png")
                    plt.close()

                # Save pred vs actual data
                pd.DataFrame({
                    'smiles': data_subset['smiles'],
                    'actual': actual,
                    'predicted': pred
                }).to_csv(f"{analyzed_path}/pred_vs_actual_{actual_col}_data.csv", index=False)

                # Append to metrics table
                metrics_rows.append({
                    'dataset_ID': pred_split_name,
                    'target': actual_col,
                    'n': len(pred),
                    'pearson': pearson[0],
                    'pearson_p_val': pearson[1],
                    'kendall': kendall,
                    'spearman': spearman,
                    'rmse': rmse,
                    'note': "insufficient_data" if n_vals < min_values_for_analysis else ""


                })

        # Save metrics table in combined format
        uho_results_path = f"{path_to_preds}{split_name}/ultra_held_out"
        path_if_none(uho_results_path)
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(f"{uho_results_path}/ultra_held_out_metrics.csv", index=False)

    except Exception as e:
        print(f"Ultra-held-out analysis failed: {e}")

def merge_datasets(experiment_list, path_to_folders = '../data/data_files_to_merge', write_path = '../data'): 
    # Each folder contains the following files: 
    # main_data.csv: a csv file with columns: 'smiles', which should contain the SMILES of the ionizable lipid, the activity measurements for that measurement
    # If the same ionizable lipid is measured multiple times (i.e. for different properties, or transfection in vitro and in vivo) make separate rows, one for each measurement
    # formulations.csv: a csv file with columns:
        # Cationic_Lipid_Mol_Ratio
        # Phospholipid_Mol_Ratio
        # Cholesterol_Mol_Ratio
        # PEG_Lipid_mol_ratio
        # Cationic_Lipid_to_mRNA_weight_ratio
        # Helper_lipid_ID
        # If the dataset contains only 1 formulation in it: still provide the formulations data thing but with only one row; the model will copy it
        # Otherwise match the row to the data in formulations.csv
    # individual_metadata.csv: metadata that contains as many rows as main_data, each row is certain metadata for each lipid
        # For example, could contain the identity (SMILES) of the amine to be used in training/test splits, or contain a dosage if the dataset includes varying dosage
        # Either includes a column called "Sample_weight" with weight for each sample (each ROW, that is; weight for a kind of experiment will be determined separately)
            # alternatively, default sample weight of 1
    # experiment_metadata.csv: contains metadata about particular dataset. This includes:
        # Experiment_ID: each experiment will be given a unique ID.
        # There will be two ROWS and any number of columns

    # Based on these files, Merge_datasets will merge all the datasets into one dataset. In particular, it will output 2 files:
        # all_merged.csv: each row  will contain all the data for a measurement (SMILES, info on dose/formulation/etc, metadata, sample weights, activity value)
        # col_type.csv: two columns, column name and type. Four types: Y_val, X_val, X_val_cat (categorical X value), Metadata, Sample_weight

    # Some metadata columns that should be held consistent, in terms of names:
        # Purity ("Pure" or "Crude")
        # ng_dose (for the dose, duh)
        # Sample_weight
        # Amine_SMILES
        # Tail_SMILES
        # Library_ID
        # Experimenter_ID
        # Experiment_ID
        # Cargo (siRNA, DNA, mRNA, RNP are probably the relevant 4 options)
        # Model_type (either the cell type or the name of the animal (probably "mouse"))
    
    all_df = pd.DataFrame({})
    col_type = {'Column_name':[],'Type':[]}
    experiment_df = pd.read_csv(path_to_folders + '/experiment_metadata.csv')
    if experiment_list == None:
        print("370")
        experiment_list = list(experiment_df.Experiment_ID)
    y_val_cols = []
    helper_mol_weights = pd.read_csv(path_to_folders + '/Component_molecular_weights.csv')

    for folder in experiment_list:
        print("folder", folder)
        contin = False
        try:
            main_temp = pd.read_csv(path_to_folders + '/' + folder + '/main_data.csv')
            contin = True
        except:
            pass
        if contin:
            y_val_cols = y_val_cols + list(main_temp.columns)
            for col in main_temp.columns:
                if 'Unnamed' in col:
                    print('\n\n\nTHERE IS A BS UNNAMED COLUMN IN FOLDER: ',folder,'\n\n')
            data_n = len(main_temp)
            formulation_temp = pd.read_csv(path_to_folders + '/' + folder + '/formulations.csv')

            try:
                individual_temp = pd.read_csv(path_to_folders + '/' + folder + '/individual_metadata.csv')
            except:
                individual_temp = pd.DataFrame({})
            if len(formulation_temp) == 1:
                formulation_temp = pd.concat([formulation_temp]*data_n,ignore_index = True)
            elif len(formulation_temp) != data_n:
                print(len(formulation_temp))
                to_raise = 'For experiment ID: ',folder,': Length of formulation file (', str(len(formulation_temp))#, ') doesn\'t match length of main datafile (',str(data_n),')'
                raise ValueError(to_raise)

            # Change formulations from mass to molar ratio
            form_cols = formulation_temp.columns
            mass_ratio_variables = ['Cationic_Lipid_Mass_Ratio','Phospholipid_Mass_Ratio','Cholesterol_Mass_Ratio','PEG_Lipid_Mass_Ratio']
            molar_ratio_variables = ['Cationic_Lipid_Mol_Ratio','Phospholipid_Mol_Ratio','Cholesterol_Mol_Ratio','PEG_Lipid_Mol_Ratio']
            mass_count = 0
            molar_count = 0
            for col in form_cols:
                if col in mass_ratio_variables:
                    mass_count += 1
                elif col in molar_ratio_variables:
                    molar_count += 1
            if mass_count>0 and molar_count>0:
                raise ValueError('For experiment ID: ',folder,': Formulation information includes both mass and molar ratios.')
            elif mass_count<4 and molar_count<4:
                raise ValueError('For experiment ID: ',folder,': Formulation not completely specified, mass count: ',mass_count,', molar count: ',molar_count)
            elif mass_count == 4:
                cat_lip_mol_fracs = []
                phos_mol_fracs = []
                chol_mol_fracs = []
                peg_lip_mol_fracs = []
                # Change mass ratios to weight ratios
                for i in range(len(formulation_temp)):
                    phos_id = formulation_temp['Helper_lipid_ID'][i]
                    ion_lipid_mol = Chem.MolFromSmiles(main_temp['smiles'][i])
                    ion_lipid_mol_weight = Chem.Descriptors.MolWt(ion_lipid_mol)
                    phospholipid_mol_weight = helper_mol_weights[phos_id][0]
                    cholesterol_mol_weight = helper_mol_weights['Cholesterol']
                    PEG_lipid_mol_weight = helper_mol_weights['C14-PEG2000']
                    ion_lipid_moles = formulation_temp['Cationic_Lipid_Mass_Ratio'][i]/ion_lipid_mol_weight
                    phospholipid_moles = formulation_temp['Phospholipid_Mass_Ratio'][i]/phospholipid_mol_weight
                    cholesterol_moles = formulation_temp['Cholesterol_Mass_Ratio'][i]/cholesterol_mol_weight
                    PEG_lipid_moles = formulation_temp['PEG_Lipid_Mass_Ratio'][i]/PEG_lipid_mol_weight
                    mol_sum = ion_lipid_moles+phospholipid_moles+cholesterol_moles+PEG_lipid_moles
                    cat_lip_mol_fracs.append(float(ion_lipid_moles/mol_sum*100))
                    phos_mol_fracs.append(float(phospholipid_moles/mol_sum*100))
                    chol_mol_fracs.append(float(cholesterol_moles/mol_sum*100))
                    peg_lip_mol_fracs.append(float(PEG_lipid_moles/mol_sum*100))
                formulation_temp['Cationic_Lipid_Mol_Ratio'] = cat_lip_mol_fracs
                formulation_temp['Phospholipid_Mol_Ratio'] = phos_mol_fracs
                formulation_temp['Cholesterol_Mol_Ratio'] = chol_mol_fracs
                formulation_temp['PEG_Lipid_Mol_Ratio'] = peg_lip_mol_fracs

        
            if len(individual_temp) != data_n:
                print(len(individual_temp))
                raise ValueError('For experiment ID: ',folder,': Length of individual metadata file  (',len(individual_temp), ') doesn\'t match length of main datafile (',data_n,')')
            experiment_temp = experiment_df[experiment_df.Experiment_ID == folder]
            experiment_temp = pd.concat([experiment_temp]*data_n, ignore_index = True).reset_index(drop = True)
            to_drop = []
            for col in experiment_temp.columns:
                if col in individual_temp.columns:
                    print('Column ',col,' in experiment ID ',folder,'is being provided for each individual lipid.')
                    to_drop.append(col)
            experiment_temp = experiment_temp.drop(columns = to_drop)
            folder_df = pd.concat([main_temp, formulation_temp, individual_temp], axis = 1).reset_index(drop = True)
            folder_df = pd.concat([folder_df, experiment_temp], axis = 1)
            # print(folder_df.columns)
            if 'Sample_weight' not in folder_df.columns:
                # print(folder)
                # folder_df['Sample_weight'] = [float(folder_df.Experiment_weight[i])/list(folder_df.smiles).count(smile) for i,smile in enumerate(folder_df.smiles)]
                folder_df['Sample_weight'] = [float(folder_df.Experiment_weight[i]) for i,smile in enumerate(folder_df.smiles)]
            all_df = pd.concat([all_df,folder_df], ignore_index = True)
    # Make changes:
    all_df = all_df.replace('im','intramuscular')
    all_df = all_df.replace('iv','intravenous')
    all_df = all_df.replace('a549','lung_epithelium')
    all_df = all_df.replace('bdmc','macrophage')
    all_df = all_df.replace('bmdm','dendritic_cell')
    all_df = all_df.replace('hela','generic_cell')
    all_df = all_df.replace('hek','generic_cell')
    all_df = all_df.replace('igrov1','generic_cell')
    all_df = all_df.replace({'Model_type':'muscle'},'Mouse')


    # Make the column type dict
    extra_x_variables = ['Cationic_Lipid_Mol_Ratio','Phospholipid_Mol_Ratio','Cholesterol_Mol_Ratio','PEG_Lipid_Mol_Ratio','Cationic_Lipid_to_mRNA_weight_ratio']
    # ADD HELPER LIPID ID
    # extra_x_categorical = ['Delivery_target','Helper_lipid_ID','Route_of_administration','Batch_or_individual_or_barcoded','screen_id']
    extra_x_categorical = ['Delivery_target','Helper_lipid_ID','Route_of_administration','Batch_or_individual_or_barcoded','Cargo_type','Model_type']

    # other_x_vals = ['Target_organ']
    # form_variables.append('Helper_lipid_ID')

    for x_cat in extra_x_categorical:
        dummies = pd.get_dummies(all_df[x_cat], prefix = x_cat)
        all_df = pd.concat([all_df, dummies], axis = 1)
        extra_x_variables = extra_x_variables + list(dummies.columns)

    for column in all_df.columns:
        col_type['Column_name'].append(column)
        if column in y_val_cols:
            col_type['Type'].append('Y_val')
        elif column in extra_x_variables:
            col_type['Type'].append('X_val')
        elif column in extra_x_categorical:
            col_type['Type'].append('Metadata')
        elif column == 'Sample_weight':
            col_type['Type'].append('Sample_weight')
        else:
            col_type['Type'].append('Metadata')

    col_type_df = pd.DataFrame(col_type)

    norm_split_names, norm_del, norm_tox = generate_normalized_data_minmax(all_df)
    all_df['split_name_for_normalization'] = norm_split_names
    all_df.rename(columns = {'quantified_delivery':'unnormalized_delivery'}, inplace = True)
    all_df['quantified_delivery'] = norm_del
    all_df.rename(columns = {'quantified_toxicity':'unnormalized_toxicity'}, inplace = True)
    all_df['quantified_toxicity'] = norm_tox
    
    all_df = all_df.replace({True: 1.0, False: 0.0})
    path = write_path + '/all_data.csv'
    print("creating all_data")
    change_column_order(path, all_df)
    col_type_df.to_csv(write_path + '/col_type.csv', index = False)


def cv_split(split_spec_fname, path_to_folders='../data',
                       is_morgan=False, cv_fold=2, ultra_held_out_fraction=-1.0,
                       min_unique_vals=2.0, test_is_valid=False,
                       train_frac=0.8, valid_frac=0.1, test_frac=0.1,
                       random_state=42):
    """
    Splits the dataset according to the specifications in split_spec_fname.
    Uses sklearn to create a single fixed test set and splits the rest into train/valid.
    Supports ultra held-out sets and maintains folder structure.

    Parameters:
        split_spec_fname: CSV specifying split/train rules
        path_to_folders: folder containing all_data.csv, crossval_split_specs, etc.
        is_morgan: whether to include Morgan fingerprints
        cv_fold: number of CV folds (1–5)
        ultra_held_out_fraction: fraction to hold out from all CV splits
        min_unique_vals: minimum unique values for splitting
        test_is_valid: if True, validation = test fold (used for in-silico screening)
        train_frac, valid_frac, test_frac: fractions for train/valid/test (must sum to 1)
        random_state: random seed for reproducibility
    """

    all_df = pd.read_csv(os.path.join(path_to_folders, 'all_data.csv'))
    split_df = pd.read_csv(os.path.join(path_to_folders, 'crossval_split_specs', split_spec_fname))
    col_types = pd.read_csv(os.path.join(path_to_folders, 'col_type.csv'))

    split_path = os.path.join(path_to_folders, 'crossval_splits', split_spec_fname[:-4])
    if ultra_held_out_fraction > 0:
        split_path += '_with_uho'
    if is_morgan:
        split_path += '_morgan'
    if test_is_valid:
        split_path += '_for_iss'

    if ultra_held_out_fraction > 0:
        path_if_none(os.path.join(split_path, 'ultra_held_out'))
    
    for i in range(cv_fold):
        path_if_none(os.path.join(split_path, f'cv_{i}'))

    # --- Collect permanent train, ultra-held-out, and CV data ---
    perma_train = pd.DataFrame({})
    ultra_held_out = pd.DataFrame({})
    cv_data = pd.DataFrame({})

    for _, row in split_df.iterrows():
        dtypes = row['Data_types_for_component'].split(',')
        vals = row['Values'].split(',')
        df_to_concat = all_df.copy()

        # Filter rows according to split spec
        for i, dtype in enumerate(dtypes):
            df_to_concat = df_to_concat[df_to_concat[dtype.strip()] == vals[i].strip()].reset_index(drop=True)

        values_to_split = df_to_concat[row['Data_type_for_split']]
        unique_values_to_split = list(set(values_to_split))

        if row['Train_or_split'].lower() == 'train':
            perma_train = pd.concat([perma_train, df_to_concat])
        elif row['Train_or_split'].lower() == 'split':
            cv_split_values, ultra_held_out_values = split_for_cv(unique_values_to_split, cv_fold, ultra_held_out_fraction)
            ultra_held_out = pd.concat([ultra_held_out, df_to_concat[df_to_concat[row['Data_type_for_split']].isin(ultra_held_out_values)]])
            # Merge all CV split data (we'll split it by fraction later)
            cv_data = pd.concat([cv_data, df_to_concat[df_to_concat[row['Data_type_for_split']].isin(sum(cv_split_values, []))]])

    # --- Save ultra held-out set ---
    if ultra_held_out_fraction >= 0 and not ultra_held_out.empty:
        y, x, w, m = split_df_by_col_type(ultra_held_out, col_types)
        yxwm_to_csvs(y, x, w, m, split_path + '/ultra_held_out', 'test')

    # --- Sanity check on fractions ---
    if abs(train_frac + valid_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train_frac + valid_frac + test_frac must sum to 1.0")

    # --- Step 1: Split once into train+valid and fixed test ---
    train_valid_df, test_df = train_test_split(
        cv_data, test_size=test_frac, random_state=random_state, shuffle=True
    )
    y, x, w, m = split_df_by_col_type(test_df, col_types)
    path_if_none(split_path + '/test')
    yxwm_to_csvs(y, x, w, m, split_path + '/test', 'test')
    
    # --- Step 2: Split train_valid into train and valid ---
    valid_size = valid_frac / (train_frac + valid_frac)
    train_df, valid_df = train_test_split(
        train_valid_df, test_size=valid_size, random_state=random_state, shuffle=True
    )

    # --- Step 3: Add permanent training data ---
    if not perma_train.empty:
        train_df = pd.concat([train_df, perma_train]).drop_duplicates().reset_index(drop=True)

    # --- Save results into each cv folder (test stays fixed) ---
    for i in range(cv_fold):
        for df, split_type in zip([valid_df, train_df], ['valid', 'train']):
            y, x, w, m = split_df_by_col_type(df, col_types)
            yxwm_to_csvs(y, x, w, m, split_path + '/cv_' +str(i), split_type)


def specified_cv_split(split_spec_fname, path_to_folders='../data',
                       is_morgan=False, cv_fold=2, ultra_held_out_fraction=-1.0,
                       min_unique_vals=2.0, test_is_valid=False):
    """
    Splits the dataset according to the specifications in split_spec_fname.
    Supports CV folds from 1 to 5.
    
    Parameters:
        split_spec_fname: CSV specifying split/train rules
        path_to_folders: folder containing all_data.csv, crossval_split_specs, etc.
        is_morgan: whether to include Morgan fingerprints
        cv_fold: number of CV folds (1–5)
        ultra_held_out_fraction: fraction to hold out from all CV splits
        min_unique_vals: minimum unique values for splitting
        test_is_valid: if True, validation = test fold (used for in-silico screening)
    """
    
    all_df = pd.read_csv(path_to_folders + '/all_data.csv')
    split_df = pd.read_csv(path_to_folders + '/crossval_split_specs/' + split_spec_fname)
    
    split_path = path_to_folders + '/crossval_splits/' + split_spec_fname[:-4]
    if ultra_held_out_fraction >= 0:
        split_path += '_with_uho'
    if is_morgan:
        split_path += '_morgan'
    if test_is_valid:
        split_path += '_for_iss'
    
    if ultra_held_out_fraction >= 0:
        path_if_none(split_path + '/ultra_held_out')
    for i in range(cv_fold):
        path_if_none(split_path + '/cv_' + str(i))
    
    perma_train = pd.DataFrame({})
    ultra_held_out = pd.DataFrame({})
    cv_splits = [pd.DataFrame({}) for _ in range(cv_fold)]
    
    for _, row in split_df.iterrows():
        dtypes = row['Data_types_for_component'].split(',')
        vals = row['Values'].split(',')
        df_to_concat = all_df.copy()
        
        #filter rows according to split spec
        for i, dtype in enumerate(dtypes):
            df_to_concat = df_to_concat[df_to_concat[dtype.strip()] == vals[i].strip()].reset_index(drop=True)
        
        values_to_split = df_to_concat[row['Data_type_for_split']]
        unique_values_to_split = list(set(values_to_split))
        
        if row['Train_or_split'].lower() == 'train':
            perma_train = pd.concat([perma_train, df_to_concat])
        elif row['Train_or_split'].lower() == 'split':
            cv_split_values, ultra_held_out_values = split_for_cv(unique_values_to_split, cv_fold, ultra_held_out_fraction)
            to_concat = df_to_concat[df_to_concat[row['Data_type_for_split']].isin(ultra_held_out_values)]
            ultra_held_out = pd.concat([ultra_held_out, to_concat])
            for i, val in enumerate(cv_split_values):
                cv_splits[i] = pd.concat([cv_splits[i], df_to_concat[df_to_concat[row['Data_type_for_split']].isin(val)]])
    
    col_types = pd.read_csv(path_to_folders + '/col_type.csv')
    
    if ultra_held_out_fraction >= 0 and not ultra_held_out.empty:
        y, x, w, m = split_df_by_col_type(ultra_held_out, col_types)
        yxwm_to_csvs(y, x, w, m, split_path + '/ultra_held_out', 'test')
    
    for i in range(cv_fold):
        test_df = cv_splits[i]
        
        if test_is_valid or cv_fold == 1:
            valid_df = test_df
            train_inds = [k for k in range(cv_fold) if k != i]
        else:
            valid_df = cv_splits[(i + 1) % cv_fold] if cv_fold > 1 else test_df
            if cv_fold > 2:
                # remove both test and validation folds from training
                train_inds = [k for k in range(cv_fold) if k != i and k != ((i + 1) % cv_fold)]
            else:
                # for cv_fold=2 remove test fold
                train_inds = [k for k in range(cv_fold) if k != i]
        
        train_df = pd.concat([perma_train] + [cv_splits[k] for k in train_inds]) if train_inds else perma_train
        
        for df, split_type in zip([test_df, valid_df, train_df], ['test', 'valid', 'train']):
            y, x, w, m = split_df_by_col_type(df, col_types)
            yxwm_to_csvs(y, x, w, m, split_path + '/cv_' + str(i), split_type)


# these functions called in main 

# called in merge_datasets
def generate_normalized_data(all_df, split_variables = ['Experiment_ID','Library_ID','Delivery_target','Model_type','Route_of_administration']):
    split_names = []
    norm_dict_del = {}
    norm_dict_tox = {}
    for index, row in all_df.iterrows():
        split_name = ''
        for vbl in split_variables:
            split_name = split_name + str(row[vbl])+'_'
        split_names.append(split_name[:-1])
    unique_split_names = set(split_names)
    for split_name in unique_split_names:
        data_subset = all_df[[spl==split_name for spl in split_names]]
        norm_dict_del[split_name] = (np.mean(data_subset['quantified_delivery']), np.std(data_subset['quantified_delivery']))
        norm_dict_tox[split_name] = (np.mean(data_subset['quantified_toxicity']), np.std(data_subset['quantified_toxicity']))
    norm_delivery = []
    norm_toxicity = []
    for i, row in all_df.iterrows():
        deli = row['quantified_delivery']
        split_del = split_names[i]
        std_del = norm_dict_del[split_del][1]
        mn_del = norm_dict_del[split_del][0]
        if pd.isna(deli):
            norm_delivery.append(np.nan)
        else:
            norm_delivery.append((float(deli)-mn_del)/std_del)

        tox = row['quantified_toxicity']
        split_tox = split_names[i]
        std_tox = norm_dict_tox[split_tox][1]
        mn_tox = norm_dict_tox[split_tox][0]
        if pd.isna(tox):
            norm_toxicity.append(np.nan)
        else:
            norm_toxicity.append((float(tox)-mn_tox)/std_tox)

    return split_names, norm_delivery, norm_toxicity

def generate_normalized_data_minmax(all_df, split_variables=['Experiment_ID','Library_ID','Delivery_target','Model_type','Route_of_administration']):
    split_names = []
    norm_dict_del = {}
    norm_dict_tox = {}

    # split names
    for index, row in all_df.iterrows():
        split_name = '_'.join([str(row[vbl]) for vbl in split_variables])
        split_names.append(split_name)

    unique_split_names = set(split_names)

    for split_name in unique_split_names:
        data_subset = all_df[[spl == split_name for spl in split_names]]
        norm_dict_del[split_name] = (data_subset['quantified_delivery'].min(), data_subset['quantified_delivery'].max())
        norm_dict_tox[split_name] = (data_subset['quantified_toxicity'].min(), data_subset['quantified_toxicity'].max())

    norm_delivery = []
    norm_toxicity = []

    for i, row in all_df.iterrows():
        deli = row['quantified_delivery']
        min_del, max_del = norm_dict_del[split_names[i]]
        if pd.isna(deli) or min_del == max_del:
            norm_delivery.append(np.nan)  # avoid division by zero
        else:
            norm_delivery.append((float(deli) - min_del) / (max_del - min_del))

        tox = row['quantified_toxicity']
        min_tox, max_tox = norm_dict_tox[split_names[i]]
        if pd.isna(tox) or min_tox == max_tox:
            norm_toxicity.append(np.nan)
        else:
            norm_toxicity.append((float(tox) - min_tox) / (max_tox - min_tox))

    return split_names, norm_delivery, norm_toxicity

# these functions only used in specified_cv_split
def split_df_by_col_type(df,col_types):
    # Splits into 4 dataframes: y_vals, x_vals, sample_weights, metadata
    y_vals_cols = col_types.Column_name[col_types.Type == 'Y_val']
    x_vals_cols = col_types.Column_name[col_types.Type == 'X_val']
    xvals_df = df[x_vals_cols]
    weight_cols = col_types.Column_name[col_types.Type == 'Sample_weight']
    metadata_cols = col_types.Column_name[col_types.Type.isin(['Metadata','X_val_categorical'])]
    return df[y_vals_cols], xvals_df ,df[weight_cols] ,df[metadata_cols]

def yxwm_to_csvs(y, x, w, m, path,settype):
    # y is y values,  x is x values, w is weights, m is metadata
    # set_type is either train, valid, or test
    y.to_csv(path+'/'+settype+'.csv', index = False)
    x.to_csv(path + '/' + settype + '_extra_x.csv', index = False)
    w.to_csv(path + '/' + settype + '_weights.csv', index = False)
    m.to_csv(path + '/' + settype + '_metadata.csv', index = False)

def split_for_cv(vals,cv_fold, held_out_fraction):
    # randomly splits vals into cv_fold groups, plus held_out_fraction of vals are completely held out. So for example split_for_cv(vals,5,0.1) will hold out 10% of data and randomly put 18% into each of 5 folds
    random.seed(42)
    random.shuffle(vals)
    held_out_vals = vals[:int(held_out_fraction*len(vals))]
    cv_vals = vals[int(held_out_fraction*len(vals)):]
    return [cv_vals[i::cv_fold] for i in range(cv_fold)], held_out_vals

# these functions only used in specified_cv_split

def main(argv):
    task_type = argv[1]

    if task_type == 'split':
        split = argv[2]
        ultra_held_out = float(argv[3])
        is_morgan = False
        in_silico_screen = False
        cv_num = 2
        if len(argv)>4:
            for i, arg in enumerate(argv):
                if arg.replace('–', '-') == '--cv':
                    cv_num = int(argv[i+1])
                    print('this many folds: ',str(cv_num))
                if arg.replace('–', '-') == '--morgan':
                    is_morgan = True
                if arg.replace('–', '-') == '--in_silico':
                    in_silico_screen = True
        cv_split(split ,cv_fold=cv_num, ultra_held_out_fraction = ultra_held_out, is_morgan = is_morgan, test_is_valid = in_silico_screen)

    elif task_type == 'train':
        split_folder = argv[2]
        epochs = 50
        cv_num = 2
        for i, arg in enumerate(argv):
            if arg.replace('–', '-') == '--epochs':
                epochs = int(argv[i+1])
                print('this many epochs: ',str(epochs))
            if arg.replace('–', '-') == '--cv':
                cv_num = int(argv[i+1])
                print('this many folds: ',str(cv_num))
            if arg.replace('–', '-') == '--cm':
                for cv in range(cv_num):
                    print("using cm")
                    split_dir = '../data/crossval_splits/'+split_folder+'/cv_'+str(cv)
                    save_dir = split_dir+'/model_'+str(cv)
                    train_cm(split_dir=split_dir,epochs=epochs, save_dir=save_dir)
                return

        for cv in range(cv_num):
            split_dir = '../data/crossval_splits/'+split_folder+'/cv_'+str(cv)
            save_dir = split_dir+'/model_'+str(cv)
            train(split_dir=split_dir,epochs=epochs, save_dir=save_dir)

        return
    elif task_type == 'analyze':
        split = argv[2]
        cv = 2
        s = False
        to_eval = ["test", "train", "valid"]
        for i, arg in enumerate(argv):
            if arg.replace('–', '-') == '--cv':
                cv = int(argv[i+1])
                print('this many folds: ',str(cv))
            if arg.replace('–', '-') == '--standardize':
                s = True 
                print('standardize')
        for tvt in to_eval:
            print("make pva")
            make_pred_vs_actual_tvt(split, ensemble_size = cv, standardize_predictions= s, tvt=tvt)
            print("analyze preds")
            analyze_predictions_cv_tvt(split, ensemble_number= cv, tvt=tvt)
            print("done with:", tvt)

    elif task_type == 'predict':
        cv_num = 5
        split_model_folder = '../data/crossval_splits/'+argv[2]
        screen_name = argv[3]
        # READ THE METADATA FILE TO A DF, THEN TAG ON THE PREDICTIONS TO GENERATE A COMPLETE PREDICTIONS FILE
        all_df = pd.read_csv('../data/libraries/'+screen_name+'/'+screen_name+'_metadata.csv')
        for cv in range(cv_num):
            # results_dir = '../results/crossval_splits/'+split_model_folder+'cv_'+str(cv)
            arguments = [
                '--test_path','../data/libraries/'+screen_name+'/'+screen_name+'.csv',
                '--features_path','../data/libraries/'+screen_name+'/'+screen_name+'_extra_x.csv',
                '--checkpoint_dir', split_model_folder+'/cv_'+str(cv),
                '--preds_path','../results/screen_results/'+argv[2]+'_preds'+'/'+screen_name+'/cv_'+str(cv)+'_preds.csv'
            ]
            print(cv)
            if 'morgan' in split_model_folder:
                    arguments = arguments + ['--features_generator','morgan_count']
            print("about to make pred")
            args = chemprop.args.PredictArgs().parse_args(arguments)
            preds = chemprop.train.make_predictions(args=args)
            new_df = pd.read_csv('../results/screen_results/'+argv[2]+'_preds'+'/'+screen_name+'/cv_'+str(cv)+'_preds.csv')
            all_df['smiles'] = new_df.smiles
            all_df['cv_'+str(cv)+'_pred_delivery'] = new_df.quantified_delivery
            all_df['cv_'+str(cv)+'_pred_toxicity'] = new_df.quantified_toxicity	
        all_df['avg_pred_delivery'] = all_df[['cv_'+str(cv)+'_pred_delivery' for cv in range(cv_num)]].mean(axis=1)
        all_df['avg_pred_toxicity'] = all_df[['cv_'+str(cv)+'_pred_toxicity' for cv in range(cv_num)]].mean(axis=1)
        path = '../results/screen_results/'+argv[2]+'_preds'+'/'+screen_name+'/pred_file.csv'
        first_cols = [
            "cv_0_pred_delivery", "cv_0_pred_toxicity", "cv_1_pred_delivery", 
            "cv_1_pred_toxicity", "cv_2_pred_delivery", "cv_2_pred_toxicity", "cv_3_pred_delivery",
            "cv_3_pred_toxicity", "cv_4_pred_delivery", "cv_4_pred_toxicity", "avg_pred_delivery", "avg_pred_toxicity", "smiles"]
        change_column_order(path, all_df, first_cols = first_cols)
    
    elif task_type == 'hyperparam_optimize':
        split_folder = argv[2]
        data_dir = '../data/crossval_splits/'+split_folder+'/cv_'+str(cv)
        arguments = [
            '--data_path',data_dir+'/train.csv',
            '--features_path', data_dir+'/train_extra_x.csv',
            '--separate_val_path', data_dir+'/valid.csv',
            '--separate_val_features_path', data_dir+'/valid_extra_x.csv',
            '--separate_test_path',data_dir+'/test.csv',
            '--separate_test_features_path',data_dir+'/test_extra_x.csv',
            '--dataset_type', 'regression',
            '--num_iters', '5',
            '--config_save_path','../results/'+split_folder+'/hyp_cv_0.json',
            '--epochs', '5'
        ]
        args = chemprop.args.HyperoptArgs().parse_args(arguments)
        chemprop.hyperparameter_optimization.hyperopt(args)
    
    elif task_type == 'merge':
        print("merge")
        merge_datasets(None)


if __name__ == '__main__':
    main(sys.argv)


#og functions before adding ablity to analyze train val

def make_pred_vs_actual(split_folder, ensemble_size = 5, standardize_predictions = False):
    # Makes predictions on each test set in a cross-validation-split system
    # Not used for screening a new library, used for predicting on the test set of the existing dataset
    # set = 0 means doing on test, 1 is train, 2 is valid set
    
    for cv in range(ensemble_size):
        print(cv)
        model = '../data/crossval_splits/'+split_folder+'/cv_'+str(cv)
        results_dir = '../results/crossval_splits/'+split_folder
        data_dir = '../data/crossval_splits/'+split_folder+'/test'
        output = pd.read_csv(data_dir+'/test.csv')
        metadata = pd.read_csv(data_dir+'/test_metadata.csv')
        output = pd.concat([metadata, output], axis = 1)
        results_dir = results_dir
        path_if_none(results_dir)

        try: 
            output = pd.read_csv(results_dir+'/predicted_vs_actual.csv')
            print("pred vs actual already exist")
        except:
            try:
                current_predictions = pd.read_csv(data_dir+'/preds.csv')
                print("already have preds.csv")
            except:
                print("running predict")
                # Load model checkpoint for this fold
                checkpoint_path = model + '/model_'+str(cv)+'/best.ckpt'
                mpnn = models.MPNN.load_from_checkpoint(checkpoint_path)

                # Load test data
                df_test = pd.read_csv(data_dir+'/test.csv')
                extra_df = pd.read_csv(data_dir+'/test_extra_x.csv')  
                extra_features = extra_df.to_numpy(dtype=float)

                test_data = [
                    data.MoleculeDatapoint.from_smi(smi, x_d=xf)
                    for smi, xf in zip(df_test['smiles'], extra_features)
]
                featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
                test_dataset = data.MoleculeDataset(test_data, featurizer=featurizer)
                test_loader = data.build_dataloader(test_dataset, shuffle=False)

                # Run predictions
                with torch.inference_mode():
                    trainer = pl.Trainer(logger=None, enable_progress_bar=False, accelerator="auto", devices=1)
                    preds = trainer.predict(mpnn, test_loader)
                preds = torch.cat([torch.tensor(p) if not isinstance(p, torch.Tensor) else p for p in preds]).cpu().numpy()
                
                current_predictions = pd.DataFrame(preds, columns=["quantified_delivery", "quantified_toxicity"])
                current_predictions["smiles"] = df_test["smiles"].values
                print("about to make preds.csv")
                current_predictions.to_csv(data_dir+'/preds.csv', index=False)

            
            current_predictions.drop(columns = ['smiles'], inplace = True)
            for col in current_predictions.columns:
                if standardize_predictions:
                    preds_to_standardize = current_predictions[col]
                    std = np.std(preds_to_standardize)
                    mean = np.mean(preds_to_standardize)
                    current_predictions[col] = [(val-mean)/std for val in current_predictions[col]]
                current_predictions.rename(columns = {col:('cv_'+str(cv)+'_pred_'+col[:])}, inplace = True)
            output = pd.concat([output, current_predictions], axis = 1)

            pred_split_variables = ['Experiment_ID', 'Library_ID', 'Delivery_target', 'Route_of_administration']

            output['Prediction_split_name'] = output.apply(
                lambda row: '_'.join(str(row[v]) for v in pred_split_variables),
                axis=1
            ) #add prediction_split_name column

            # move new prediction columns to the front
            new_cols = current_predictions.columns.tolist()
            new_cols = new_cols + ["quantified_delivery", "quantified_toxicity", "smiles"]
            path = results_dir+'/predicted_vs_actual.csv'
            change_column_order(path, output, first_cols= new_cols)
    
    if '_with_uho' in split_folder:
        print("uho")
        uho_dir = '../data/crossval_splits/'+split_folder+'/ultra_held_out'
        results_dir = uho_dir+'/preds'
        path_if_none(results_dir)
        output = pd.read_csv(uho_dir+'/test.csv')
        metadata = pd.read_csv(uho_dir+'/test_metadata.csv')
        output = pd.concat([metadata, output], axis = 1)

        for cv in range(ensemble_size):
            model_dir = '../data/crossval_splits/'+split_folder+'/cv_'+str(cv)
            try:
                current_predictions = pd.read_csv(results_dir+'/preds_cv_'+str(cv)+'.csv')
            except:
                checkpoint_path = model_dir + '/model_'+str(cv)+'/best.ckpt'
                mpnn = models.MPNN.load_from_checkpoint(checkpoint_path)
                df_test = pd.read_csv(data_dir+'/test.csv')
                extra_df = pd.read_csv(data_dir+'/test_extra_x.csv') 
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
                current_predictions = pd.DataFrame(preds, columns=["quantified_delivery", "quantified_toxicity"])
                current_predictions["smiles"] = df_test["smiles"].values
                current_predictions.to_csv(results_dir+ f'/preds_cv_{cv}.csv', index=False)
            
            current_predictions.drop(columns = ['smiles'], inplace = True)
            for col in current_predictions.columns:
                if standardize_predictions:
                    preds_to_standardize = current_predictions[col]
                    std = np.std(preds_to_standardize)
                    mean = np.mean(preds_to_standardize)
                    current_predictions[col] = [(val-mean)/std for val in current_predictions[col]]
                current_predictions.rename(columns = {col:('cv_'+str(cv)+'_pred_'+col)}, inplace = True)
            output = pd.concat([output, current_predictions], axis = 1)

        pred_delivery_cols = [col for col in output.columns if 'pred_quantified_delivery' in col]
        pred_toxicity_cols = [col for col in output.columns if 'pred_quantified_toxicity' in col]

        output['Avg_pred_quantified_delivery'] = output[pred_delivery_cols].mean(axis = 1)
        output['Avg_pred_quantified_toxicity'] = output[pred_toxicity_cols].mean(axis = 1)

        ultra_dir = '../results/crossval_splits/'+split_folder+'/ultra_held_out'
        path_if_none(ultra_dir)
        path = ultra_dir+'/predicted_vs_actual.csv'
        first_col = ["quantified_delivery", "quantified_toxicity"]

        for cv in range(ensemble_size):
            first_col.append(f"cv_{cv}_pred_quantified_delivery")
            first_col.append(f"cv_{cv}_pred_quantified_toxicity")

        first_col += ["Avg_pred_quantified_delivery", "Avg_pred_quantified_toxicity", "smiles"]

        change_column_order(path, output, first_cols=first_col)

def analyze_predictions_cv(split_name, pred_split_variables = ['Experiment_ID','Library_ID','Delivery_target','Route_of_administration'], path_to_preds = '../results/crossval_splits/', ensemble_number = 5, min_values_for_analysis = 10):
    all_ns = {}
    all_pearson = {}
    all_pearson_p_val = {}
    all_kendall = {}
    all_spearman = {}
    all_rmse = {}
    all_unique = []
    
    preds_vs_actual = []
    for i in range(ensemble_number):
        df = pd.read_csv(path_to_preds + split_name + '/cv_' + str(i) + '/predicted_vs_actual.csv')
        preds_vs_actual.append(df)
        unique = set(df['Prediction_split_name'].tolist())
        all_unique.extend(unique) #get list of all unique split names 

    unique_pred_split_names = set(all_unique)

    for un in unique_pred_split_names:
        all_ns[un] = []
        all_pearson[un] = []
        all_pearson_p_val[un] = []
        all_kendall[un] = []
        all_spearman[un] = []
        all_rmse[un] = []

    target_columns = [('quantified_delivery', 'delivery'), ('quantified_toxicity', 'toxicity')]

    for i in range(ensemble_number): #create crossval_performance and results folder 
        crossval_results_path = f"{path_to_preds}{split_name}/crossval_performance"
        path_if_none(crossval_results_path)

        fold_results = []  # stores rows for this fold

        fold_df = preds_vs_actual[i]
        fold_unique = fold_df['Prediction_split_name'].unique()

        for pred_split_name in fold_unique: #unique_pred_split_names:
            path_if_none(path_to_preds+split_name+'/cv_'+str(i)+'/results/'+pred_split_name)
            data_subset = preds_vs_actual[i][
                preds_vs_actual[i]['Prediction_split_name'] == pred_split_name
            ].reset_index(drop=True)

            value_names = set(list(data_subset.Value_name))

            if len(value_names) > 1:
                raise Exception(
                    f'Multiple types of measurement in the same prediction split: split {pred_split_name} has value names {value_names}. Try adding more pred split variables.'
                )

            for actual_col, label in target_columns:
                if actual_col not in data_subset.columns:
                    print("target missing", actual_col)
                    continue  # skip if target missing in this split

                actual = data_subset[actual_col]
                pred_col = f'cv_{i}_pred_{actual_col}'
                if pred_col not in data_subset.columns:
                    print("predictions missing", pred_col)
                    continue  # skip if predictions missing

                pred = data_subset[pred_col]

                analyzed_data = pd.DataFrame({
                    'smiles': data_subset.smiles, 'actual': actual, 'predicted': pred
                })
                
                mask = ~(actual.isna() | pred.isna() | np.isinf(actual) | np.isinf(pred))
                actual = actual[mask]
                pred = pred[mask]

                n_vals = len(actual)
                if n_vals < 2: #if not enough actual data then cannot run metrics
                    pearson_r = pearson_p = spearman_r = kendall_r = rmse = np.nan
                    analyzed_path = f"{path_to_preds}{split_name}/cv_{i}/results/{pred_split_name}/{label}"
                    path_if_none(analyzed_path)

                    analyzed_data.to_csv(analyzed_path+'/pred_vs_actual_data.csv', index = False)

                else: 
                    pearson_r, pearson_p = scipy.stats.pearsonr(actual, pred)
                    spearman_r, _ = scipy.stats.spearmanr(actual, pred)
                    kendall_r, _ = scipy.stats.kendalltau(actual, pred)
                    rmse = np.sqrt(mean_squared_error(actual, pred))
                    analyzed_path = f"{path_to_preds}{split_name}/cv_{i}/results/{pred_split_name}/{label}"
                    path_if_none(analyzed_path)
                    plt.figure()
                    plt.scatter(pred, actual, color='black')
                    plt.plot(np.unique(pred), np.poly1d(np.polyfit(pred, actual, 1))(np.unique(pred)))
                    plt.xlabel(f'Predicted {label}')
                    plt.ylabel(f'Quantified {label} ({value_names})')
                    plt.savefig(analyzed_path + '/pred_vs_actual.png')
                    plt.close()
                    
                    analyzed_data.to_csv(analyzed_path+'/pred_vs_actual_data.csv', index = False)
                
                if n_vals < min_values_for_analysis:
                    print(f"⚠️ Warning: only {n_vals} samples for {pred_split_name} (below threshold {min_values_for_analysis})")

                fold_results.append({
                    'fold': i,
                    'split_name': pred_split_name,
                    'target': label,
                    'pearson': pearson_r,
                    'pearson_p_val': pearson_p,
                    'spearman': spearman_r,
                    'kendall': kendall_r,
                    'rmse': rmse,
                    'n_vals': n_vals,
                    'note': "insufficient_data" if n_vals < min_values_for_analysis else ""
                })

        fold_df = pd.DataFrame(fold_results)
        fold_df.to_csv(f"{crossval_results_path}/fold_{i}_metrics.csv", index=False)


    # Now analyze uho
    try:
        preds_vs_actual = pd.read_csv(path_to_preds + split_name + '/ultra_held_out/predicted_vs_actual.csv')

        preds_vs_actual['Prediction_split_name'] = preds_vs_actual[pred_split_variables].astype(str).agg('_'.join, axis=1)

        unique_pred_split_names = preds_vs_actual['Prediction_split_name'].unique()

        target_cols = [col for col in preds_vs_actual.columns if col.startswith('Avg_pred_')]
        actual_cols = [col.replace('Avg_pred_', '') for col in target_cols]

        metrics_rows = []

        for pred_split_name in unique_pred_split_names:
            data_subset = preds_vs_actual[preds_vs_actual['Prediction_split_name'] == pred_split_name].reset_index(drop=True)

            # loop throughtarget columns (delivery, toxicity, etc)
            for target_col, actual_col in zip(target_cols, actual_cols):
                actual = data_subset[actual_col]
                pred = data_subset[target_col]

                if actual.isna().all() or pred.isna().all(): #skip when all are NaN
                    continue

                mask = ~(actual.isna() | pred.isna() | np.isinf(actual) | np.isinf(pred))
                actual = actual[mask]
                pred = pred[mask]
                if len(actual) < 2: #if not enough actual data then cannot run metrics
                    pearson_r = pearson_p = spearman_r = kendall_r = rmse = np.nan
                    n_vals = len(pred)
                else: 
                    pearson = scipy.stats.pearsonr(actual, pred)
                    spearman, _ = scipy.stats.spearmanr(actual, pred)
                    kendall, _ = scipy.stats.kendalltau(actual, pred)
                    rmse = np.sqrt(mean_squared_error(actual, pred))

                    # Save pred vs actual plot
                    analyzed_path = f"{path_to_preds}{split_name}/ultra_held_out/individual_dataset_results/{pred_split_name}"
                    path_if_none(analyzed_path)
                    plt.figure()
                    plt.scatter(pred, actual, color='black')
                    plt.plot(np.unique(pred), np.poly1d(np.polyfit(pred, actual, 1))(np.unique(pred)))
                    plt.xlabel(f"Predicted {actual_col}")
                    plt.ylabel(f"Experimental {actual_col}")
                    plt.savefig(f"{analyzed_path}/pred_vs_actual_{actual_col}.png")
                    plt.close()

                # Save pred vs actual data
                pd.DataFrame({
                    'smiles': data_subset['smiles'],
                    'actual': actual,
                    'predicted': pred
                }).to_csv(f"{analyzed_path}/pred_vs_actual_{actual_col}_data.csv", index=False)

                # Append to metrics table
                metrics_rows.append({
                    'dataset_ID': pred_split_name,
                    'target': actual_col,
                    'n': len(pred),
                    'pearson': pearson[0],
                    'pearson_p_val': pearson[1],
                    'kendall': kendall,
                    'spearman': spearman,
                    'rmse': rmse,
                    'note': "insufficient_data" if n_vals < min_values_for_analysis else ""


                })

        # Save metrics table in combined format
        uho_results_path = f"{path_to_preds}{split_name}/ultra_held_out"
        path_if_none(uho_results_path)
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(f"{uho_results_path}/ultra_held_out_metrics.csv", index=False)

    except Exception as e:
        print(f"Ultra-held-out analysis failed: {e}")
