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
from helpers import path_if_none, change_column_order, load_datapoints
import joblib


def make_pred_vs_actual_tvt(split_folder, model_folder, ensemble_size = 5, standardize_predictions = False, tvt = 'test'):
    # Makes predictions on each test set in a cross-validation-split system
    # Not used for screening a new library, used for predicting on the test set of the existing dataset
    # tvt is train valid or test- determines which files to run analysis on
    #split folder has the datapoints, model folder has the model

    for cv in range(ensemble_size):
        print(cv)
        model = '../data/crossval_splits/'+model_folder+'/cv_'+str(cv)
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
                loaded_model = models.MPNN.load_from_checkpoint(checkpoint_path)
                
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
                    preds = trainer.predict(loaded_model, test_loader)
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
            model_dir = '../data/crossval_splits/'+model_folder+'/cv_'+str(cv)
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
    all_r2 = {}
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
        all_r2[un] = []
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
                    pearson_r = rsquared = pearson_p = spearman_r = kendall_r = rmse = np.nan
                    analyzed_path = f"{path_to_preds}{split_name}/{tvt}/cv_{i}/results/{pred_split_name}/{label}"
                    path_if_none(analyzed_path)

                    analyzed_data.to_csv(analyzed_path+'/pred_vs_actual_data.csv', index = False)

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
                    
                    analyzed_data.to_csv(analyzed_path+'/pred_vs_actual_data.csv', index = False)
                
                if n_vals < min_values_for_analysis:
                    print(f"Warning: only {n_vals} samples for {pred_split_name}, {label} (below threshold {min_values_for_analysis})")

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


def main(argv):
    test_dir = argv[1]
    model_dir = test_dir
    cv = 2
    s = False
    to_eval = ["test", "train", "valid"]
    for i, arg in enumerate(argv):
        if arg.replace('–', '-') == '--cv':
            cv = int(argv[i+1])
            print('this many folds: ',str(cv))
        if arg.replace('–', '-') == '--diff_model':
            model_dir = argv[i+1]
        if arg.replace('–', '-') == '--standardize':
            s = True 
            print('standardize')
    for tvt in to_eval:
        print("make pva")
        make_pred_vs_actual_tvt(test_dir, model_dir, ensemble_size = cv, standardize_predictions= s, tvt=tvt)
        print("analyze preds")
        analyze_predictions_cv_tvt(test_dir, ensemble_number= cv, tvt=tvt)
        print("done with:", tvt)



if __name__ == '__main__':
    main(sys.argv)