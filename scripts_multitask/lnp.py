import numpy as np 
import os
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_curve, roc_auc_score, r2_score, mean_absolute_error

from rdkit import Chem
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
import scipy.stats
import json
import sys
import random
import chemprop

# ==========================================
# GLOBAL TARGET CONFIGURATION
# ==========================================
TARGET = 'toxicity'  # Options: 'delivery', 'toxicity', etc.
TARGET_COL = f'quantified_{TARGET}'
# ==========================================


def split_df_by_col_type(df,col_types):
    # Splits into 4 dataframes: y_vals, x_vals, sample_weights, metadata
    y_vals_cols = col_types.Column_name[col_types.Type == 'Y_val']
    x_vals_cols = col_types.Column_name[col_types.Type == 'X_val']
    # print(x_vals_cols)
    xvals_df = df[x_vals_cols]
    # print('SUCCESSFUL!!!')
    weight_cols = col_types.Column_name[col_types.Type == 'Sample_weight']
    metadata_cols = col_types.Column_name[col_types.Type.isin(['Metadata','X_val_categorical'])]
    return df[y_vals_cols],xvals_df,df[weight_cols],df[metadata_cols]


def split_for_cv(vals,cv_fold, held_out_fraction):
    # randomly splits vals into cv_fold groups, plus held_out_fraction of vals are completely held out. So for example split_for_cv(vals,5,0.1) will hold out 10% of data and randomly put 18% into each of 5 folds
    random.shuffle(vals)
    held_out_vals = vals[:int(held_out_fraction*len(vals))]
    cv_vals = vals[int(held_out_fraction*len(vals)):]
    return [cv_vals[i::cv_fold] for i in range(cv_fold)],held_out_vals


def specified_cv_split(split_spec_fname, path_to_folders = '../data', is_morgan = False, cv_fold = 5, ultra_held_out_fraction = -1.0, min_unique_vals = 2.0, test_is_valid = False):
    # Splits the dataset according to the specifications in split_spec_fname
    # cv_fold: self-explanatory
    # ultra_held_out_fraction: if you want to hold a dataset out from even the cross-validation datasets this is the way to do it
    # test_is_valid: if true, then does the split where the test set is just the validation set, so that maximum data can be reserved for training set (this is for doing in siico screening)
    all_df = pd.read_csv(path_to_folders + '/all_data.csv')
    split_df = pd.read_csv(path_to_folders+'/crossval_split_specs/'+split_spec_fname)
    split_path = path_to_folders + '/crossval_splits/' + split_spec_fname[:-4]
    if ultra_held_out_fraction>-0.5:
        split_path = split_path + '_with_ultra_held_out'
    if is_morgan:
        split_path = split_path + '_morgan'
    if test_is_valid:
        split_path = split_path + '_for_in_silico_screen'
    if ultra_held_out_fraction>-0.5:
        path_if_none(split_path + '/ultra_held_out')
    for i in range(cv_fold):
        path_if_none(split_path+'/cv_'+str(i))

    perma_train = pd.DataFrame({})
    ultra_held_out = pd.DataFrame({})
    cv_splits = [pd.DataFrame({}) for _ in range(cv_fold)]

    for index, row in split_df.iterrows():
        dtypes = row['Data_types_for_component'].split(',')
        vals = row['Values'].split(',')
        df_to_concat = all_df
        for i, dtype in enumerate(dtypes):
            df_to_concat = df_to_concat[df_to_concat[dtype.strip()]==vals[i].strip()].reset_index(drop = True)
        values_to_split = df_to_concat[row['Data_type_for_split']]
        unique_values_to_split = list(set(values_to_split))
        # print(row)
        if row['Train_or_split'].lower() == 'train' or len(unique_values_to_split)<min_unique_vals*cv_fold:
            perma_train = pd.concat([perma_train, df_to_concat])
        elif row['Train_or_split'].lower() == 'split':
            cv_split_values, ultra_held_out_values = split_for_cv(unique_values_to_split, cv_fold, ultra_held_out_fraction)
            to_concat = df_to_concat[df_to_concat[row['Data_type_for_split']].isin(ultra_held_out_values)]
            # print('Type: ',type(to_concat))
            # print('Ultra held out type: ',type(ultra_held_out))
            ultra_held_out = pd.concat([ultra_held_out, to_concat])
            for i, val in enumerate(cv_split_values):
                cv_splits[i] = pd.concat([cv_splits[i], df_to_concat[df_to_concat[row['Data_type_for_split']].isin(val)]])

    col_types = pd.read_csv(path_to_folders + '/col_type.csv')

    # Now move the dfs to datafiles
    if ultra_held_out_fraction>-0.5:
        y,x,w,m = split_df_by_col_type(ultra_held_out,col_types)
        yxwm_to_csvs(y,x,w,m,split_path+'/ultra_held_out','test')

    for i in range(cv_fold):
        test_df = cv_splits[i]
        train_inds = list(range(cv_fold))
        train_inds.remove(i)
        if test_is_valid:
            valid_df = cv_splits[i]
        else:
            valid_df = cv_splits[(i+1)%cv_fold]
            train_inds.remove((i+1)%cv_fold)
        train_df = pd.concat([perma_train]+[cv_splits[k] for k in train_inds])

        y,x,w,m = split_df_by_col_type(test_df,col_types)
        yxwm_to_csvs(y,x,w,m,split_path+'/cv_'+str(i),'test')
        y,x,w,m = split_df_by_col_type(valid_df,col_types)
        yxwm_to_csvs(y,x,w,m,split_path+'/cv_'+str(i),'valid')
        y,x,w,m = split_df_by_col_type(train_df,col_types)
        yxwm_to_csvs(y,x,w,m,split_path+'/cv_'+str(i),'train')

def yxwm_to_csvs(y, x, w, m, path,settype):
    # y is y values
    # x is x values
    # w is weights
    # m is metadata
    # set_type is either train, valid, or test
    y.to_csv(path+'/'+settype+'.csv', index = False)
    x.to_csv(path + '/' + settype + '_extra_x.csv', index = False)
    w.to_csv(path + '/' + settype + '_weights.csv', index = False)
    m.to_csv(path + '/' + settype + '_metadata.csv', index = False)

def path_if_none(newpath):
    if not os.path.exists(newpath):
        os.makedirs(newpath)


def generate_normalized_data(all_df, split_variables = ['Experiment_ID','Library_ID','Delivery_target','Model_type','Route_of_administration']):
    split_names = []
    norm_dict = {}
    for index, row in all_df.iterrows():
        split_name = ''
        for vbl in split_variables:
            # print(row[vbl])
            # print(vbl)
            split_name = split_name + str(row[vbl])+'_'
        split_names.append(split_name[:-1])
    unique_split_names = set(split_names)
    for split_name in unique_split_names:
        data_subset = all_df[[spl==split_name for spl in split_names]]
        norm_dict[split_name] = (np.mean(data_subset[TARGET_COL]), np.std(data_subset[TARGET_COL]))
    norm_vals = []
    for i, row in all_df.iterrows():
        val = row[TARGET_COL]
        split = split_names[i]
        stdev = norm_dict[split][1]
        mean = norm_dict[split][0]
        norm_vals.append((float(val)-mean)/stdev)
    return split_names, norm_vals


def make_pred_vs_actual(split_folder, ensemble_size = 5, predictions_done = [], path_to_new_test = '',standardize_predictions = True):
    # Makes predictions on each test set in a cross-validation-split system
    # Not used for screening a new library, used for predicting on the test set of the existing dataset
    for cv in range(ensemble_size):
        data_dir = '../data/crossval_splits/'+split_folder+'/cv_'+str(cv)
        results_dir = '../results/crossval_splits/'+split_folder+'/cv_'+str(cv)
        path_if_none(results_dir)
        


        output = pd.read_csv(data_dir+'/test.csv')
        metadata = pd.read_csv(data_dir+'/test_metadata.csv')
        output = pd.concat([metadata, output], axis = 1)
        try:
            output = pd.read_csv(results_dir+'/predicted_vs_actual.csv')
        except:
            try:
                current_predictions = pd.read_csv(data_dir+'/preds.csv')
            except:
                arguments = [
                    '--test_path',data_dir+'/test.csv',
                    '--features_path',data_dir+'/test_extra_x.csv',
                    '--checkpoint_dir', data_dir,
                    '--preds_path',data_dir+'/preds.csv'
                ]
                if 'morgan' in split_folder:
                    arguments = arguments + ['--features_generator','morgan_count']
                args = chemprop.args.PredictArgs().parse_args(arguments)
                preds = chemprop.train.make_predictions(args=args)  
            # os.rename(path_to_folders+'/trained_model',path_to_folders + '/trained_model_'+str(i))
                current_predictions = pd.read_csv(data_dir+'/preds.csv')
            
            current_predictions.drop(columns = ['smiles'], inplace = True)
            for col in current_predictions.columns:
                if standardize_predictions:
                    preds_to_standardize = current_predictions[col]
                    std = np.std(preds_to_standardize)
                    mean = np.mean(preds_to_standardize)
                    current_predictions[col] = [(val-mean)/std for val in current_predictions[col]]
                current_predictions.rename(columns = {col:(f'cv_{cv}_pred_{col}')}, inplace = True)
            output = pd.concat([output, current_predictions], axis = 1)
            output.to_csv(results_dir+'/predicted_vs_actual.csv', index = False)
    if '_with_ultra_held_out' in split_folder:
        results_dir = '../results/crossval_splits/'+split_folder+'/ultra_held_out'
        uho_dir = '../data/crossval_splits/'+split_folder+'/ultra_held_out'
        output = pd.read_csv(uho_dir+'/test.csv')
        metadata = pd.read_csv(uho_dir+'/test_metadata.csv')
        output = pd.concat([metadata, output], axis = 1)
        for cv in range(ensemble_size):
            model_dir = '../data/crossval_splits/'+split_folder+'/cv_'+str(cv)
            try:
                current_predictions = pd.read_csv(results_dir+'/preds_cv_'+str(cv)+'.csv')
            except:
                arguments = [
                    '--test_path',uho_dir+'/test.csv',
                    '--features_path',uho_dir+'/test_extra_x.csv',
                    '--checkpoint_dir', model_dir,
                    '--preds_path',results_dir+'/preds_cv_'+str(cv)+'.csv'
                ]
                if 'morgan' in split_folder:
                    arguments = arguments + ['--features_generator','morgan_count']
                args = chemprop.args.PredictArgs().parse_args(arguments)
                preds = chemprop.train.make_predictions(args=args)
                current_predictions = pd.read_csv(results_dir+'/preds_cv_'+str(cv)+'.csv')
            current_predictions.drop(columns = ['smiles'], inplace = True)
            for col in current_predictions.columns:
                if standardize_predictions:
                    preds_to_standardize = current_predictions[col]
                    std = np.std(preds_to_standardize)
                    mean = np.mean(preds_to_standardize)
                    current_predictions[col] = [(val-mean)/std for val in current_predictions[col]]
                current_predictions.rename(columns = {col:(f'cv_{cv}_pred_{col}')}, inplace = True)
            output = pd.concat([output, current_predictions], axis = 1)
        pred_cols = [col for col in output.columns if '_pred_' in col]
        output[f'Avg_pred_{TARGET_COL}'] = output[pred_cols].mean(axis = 1)
        output.to_csv(results_dir+'/predicted_vs_actual.csv',index = False)


def analyze_predictions_cv(split_name, pred_split_variables=['Experiment_ID'], path_to_preds='../results/crossval_splits/', ensemble_number=5, min_values_for_analysis=10):
    """
    Analyzes CV predictions by:
    1. Calculating pooled metrics across the entire fold.
    2. Calculating metrics for specific metadata-based subsets.
    3. Generating pred-vs-actual plots for both pooled (per fold AND global) and subset data.
    """
    summary_table = pd.DataFrame({})
    all_ns = {}
    all_pearson = {}
    all_pearson_p_val = {}
    all_kendall = {}
    all_spearman = {}
    all_rmse = {}
    all_unique = []
    
    # List to store the row-by-row data for the requested pooled_metrics.csv
    pooled_rows = []

    # --- NEW: Lists to accumulate data across ALL folds for the global plot ---
    global_actual = []
    global_preds = []
    # ------------------------------------------------------------------------

    # Identify all unique split names across all folds to initialize dictionaries
    for i in range(ensemble_number):
        fold_csv = os.path.join(path_to_preds, split_name, f'cv_{i}', 'predicted_vs_actual.csv')
        if os.path.exists(fold_csv):
            temp_df = pd.read_csv(fold_csv)
            pred_split_names = []
            for index, row in temp_df.iterrows():
                name_parts = [str(row[vbl]) for vbl in pred_split_variables]
                pred_split_names.append("_".join(name_parts))
            all_unique += list(set(pred_split_names))
            
    unique_pred_split_names = set(all_unique)
    for un in unique_pred_split_names:
        all_ns[un] = []
        all_pearson[un] = []
        all_pearson_p_val[un] = []
        all_kendall[un] = []
        all_spearman[un] = []
        all_rmse[un] = []

    # Process each fold
    for i in range(ensemble_number):
        fold_dir = os.path.join(path_to_preds, split_name, f'cv_{i}')
        csv_path = os.path.join(fold_dir, 'predicted_vs_actual.csv')
        
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found. Skipping fold {i}.")
            continue
            
        preds_vs_actual = pd.read_csv(csv_path)
        
        # --- POOLED ANALYSIS FOR THE FOLD ---
        # This analyzes all samples in the fold together
        actual_all = preds_vs_actual[TARGET_COL]
        pred_col_name = f'cv_{i}_pred_{TARGET_COL}'
        pred_all = preds_vs_actual[pred_col_name]

        # --- NEW: Accumulate data for Global Plot ---
        global_actual.extend(actual_all.tolist())
        global_preds.extend(pred_all.tolist())
        # --------------------------------------------
        
        # Calculate pooled statistics (requested format)
        p_corr, p_pval = scipy.stats.pearsonr(actual_all, pred_all)
        s_corr, _ = scipy.stats.spearmanr(actual_all, pred_all)
        k_corr, _ = scipy.stats.kendalltau(actual_all, pred_all)
        mse_val = mean_squared_error(actual_all, pred_all)
        rmse_val = np.sqrt(mse_val)
        mae_val = mean_absolute_error(actual_all, pred_all)
        r2_val = r2_score(actual_all, pred_all)
        
        pooled_rows.append({
            'fold': i,
            'pearson': round(p_corr, 6),
            'r2': round(r2_val, 6),
            'pearson_p_val': round(p_pval, 6),
            'spearman': round(s_corr, 6),
            'kendall': round(k_corr, 6),
            'mse': round(mse_val, 6),
            'mae': round(mae_val, 6),
            'n_vals': len(actual_all)
        })
        
        # Save Pooled Plot for the fold
        pooled_plot_dir = os.path.join(fold_dir, 'results', 'pooled')
        path_if_none(pooled_plot_dir) # Assuming path_if_none is defined elsewhere in your utils
        plt.figure(figsize=(6,6))
        plt.scatter(pred_all, actual_all, color='black', alpha=0.4, s=10)
        
        # Add trendline if variance exists
        if len(np.unique(pred_all)) > 1:
            plt.plot(np.unique(pred_all), np.poly1d(np.polyfit(pred_all, actual_all, 1))(np.unique(pred_all)), color='red', lw=2)
            
        plt.xlabel(f'Predicted {TARGET.capitalize()} (Pooled)')
        plt.ylabel(f'Experimental {TARGET.capitalize()} (Pooled)')
        plt.title(f'Fold {i} Pooled: R2={r2_val:.3f}')
        plt.tight_layout()
        plt.savefig(os.path.join(pooled_plot_dir, 'pooled_pred_vs_actual.png'), dpi=300)
        plt.close()
        # ------------------------------------

        # --- SUBSET ANALYSIS (Metadata groups) ---
        pred_split_names = []
        for index, row in preds_vs_actual.iterrows():
            name_parts = [str(row[vbl]) for vbl in pred_split_variables]
            pred_split_names.append("_".join(name_parts))
        
        preds_vs_actual['Prediction_split_name'] = pred_split_names
        
        for pred_split_name in unique_pred_split_names:
            data_subset = preds_vs_actual[preds_vs_actual['Prediction_split_name'] == pred_split_name].reset_index(drop=True)
            
            if data_subset.empty:
                for d in [all_rmse, all_pearson, all_pearson_p_val, all_kendall, all_spearman]:
                    d[pred_split_name].append(float('nan'))
                all_ns[pred_split_name].append(0)
                continue

            value_name = data_subset['Value_name'].iloc[0] if 'Value_name' in data_subset.columns else "Value"
            analyzed_path = os.path.join(fold_dir, 'results', pred_split_name, pred_col_name)
            path_if_none(analyzed_path)
            
            actual = data_subset[TARGET_COL]
            pred = data_subset[pred_col_name]
            
            if len(actual) >= min_values_for_analysis:
                pearson = scipy.stats.pearsonr(actual, pred)
                spearman, _ = scipy.stats.spearmanr(actual, pred)
                kendall, _ = scipy.stats.kendalltau(actual, pred)
                rmse = np.sqrt(mean_squared_error(actual, pred))
                
                all_rmse[pred_split_name].append(rmse)
                all_pearson[pred_split_name].append(pearson[0])
                all_pearson_p_val[pred_split_name].append(pearson[1])
                all_kendall[pred_split_name].append(kendall)
                all_spearman[pred_split_name].append(spearman)
                
                # Plot individual subset
                plt.figure()
                plt.scatter(pred, actual, color='black')
                if len(np.unique(pred)) > 1:
                    plt.plot(np.unique(pred), np.poly1d(np.polyfit(pred, actual, 1))(np.unique(pred)))
                plt.xlabel(f'Predicted {value_name}')
                plt.ylabel(f'Experimental {value_name}')
                plt.title(f"{pred_split_name} (Fold {i})")
                plt.savefig(os.path.join(analyzed_path, 'pred_vs_actual.png'))
                plt.close()
            else:
                for d in [all_rmse, all_pearson, all_pearson_p_val, all_kendall, all_spearman]:
                    d[pred_split_name].append(float('nan'))

            all_ns[pred_split_name].append(len(pred))
            data_subset.to_csv(os.path.join(analyzed_path, 'pred_vs_actual_data.csv'), index=False)

    # --- FINAL EXPORT ---
    crossval_results_path = os.path.join(path_to_preds, split_name, 'crossval_performance')
    path_if_none(crossval_results_path)

    # Save the specific pooled_metrics.csv file
    pd.DataFrame(pooled_rows).to_csv(os.path.join(crossval_results_path, 'pooled_metrics.csv'), index=False)

    # --- NEW: Generate Global Pooled Plot (All folds combined) ---
    if len(global_actual) > 0:
        global_r2 = r2_score(global_actual, global_preds)
        global_pearson, _ = scipy.stats.pearsonr(global_actual, global_preds)
        
        plt.figure(figsize=(7,7))
        # Using alpha to visualize density if many points overlap
        plt.scatter(global_preds, global_actual, color='blue', alpha=0.3, s=15, label='Data Points')
        
        # Add regression line
        if len(np.unique(global_preds)) > 1:
            z = np.polyfit(global_preds, global_actual, 1)
            p = np.poly1d(z)
            plt.plot(np.unique(global_preds), p(np.unique(global_preds)), "r--", lw=2, label='Fit')

        plt.xlabel(f'Predicted {TARGET.capitalize()} (Global CV)')
        plt.ylabel(f'Experimental {TARGET.capitalize()} (Global CV)')
        plt.title(f'Global Pooled Performance\nPear={global_pearson:.3f}, R2={global_r2:.3f}, N={len(global_actual)}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(crossval_results_path, 'global_pooled_pred_vs_actual.png'), dpi=300)
        plt.close()
        
        # Save the global data for external analysis
        pd.DataFrame({
            'global_actual': global_actual,
            'global_preds': global_preds
        }).to_csv(os.path.join(crossval_results_path, 'global_pooled_data.csv'), index=False)
    # -------------------------------------------------------------

    # Export traditional metric matrices
    pd.DataFrame.from_dict(all_ns).to_csv(os.path.join(crossval_results_path, 'n_vals.csv'))
    pd.DataFrame.from_dict(all_pearson).to_csv(os.path.join(crossval_results_path, 'pearson.csv'))
    pd.DataFrame.from_dict(all_pearson_p_val).to_csv(os.path.join(crossval_results_path, 'pearson_p_val.csv'))
    pd.DataFrame.from_dict(all_kendall).to_csv(os.path.join(crossval_results_path, 'kendall.csv'))
    pd.DataFrame.from_dict(all_spearman).to_csv(os.path.join(crossval_results_path, 'spearman.csv'))
    pd.DataFrame.from_dict(all_rmse).to_csv(os.path.join(crossval_results_path, 'rmse.csv'))

    # --- ULTRA HELD OUT SECTION ---
    try:
        uho_path = os.path.join(path_to_preds, split_name, 'ultra_held_out')
        uho_csv = os.path.join(uho_path, 'predicted_vs_actual.csv')
        if os.path.exists(uho_csv):
           pass 
           # Add UHO logic here if needed
    except Exception as e:
        print(f"Ultra-held-out analysis failed: {e}")
def main(argv):
    # args = sys.argv[1:]
    task_type = argv[1]
    if task_type == 'train':
        split_folder = argv[2]
        epochs = 50
        cv_num = 5
        for i,arg in enumerate(argv):
            if arg.replace('–', '-') == '--epochs':
                epochs = argv[i+1]
                # print('this many epochs: ',str(epochs))
        # exit()
        for cv in range(cv_num):
            split_dir = '../data/crossval_splits/'+split_folder+'/cv_'+str(cv)
            arguments = [
                '--epochs',str(epochs),
                '--save_dir',split_dir,
                '--seed','42',
                '--dataset_type','regression',
                '--data_path',split_dir+'/train.csv',
                '--features_path', split_dir+'/train_extra_x.csv',
                '--separate_val_path', split_dir+'/valid.csv',
                '--separate_val_features_path', split_dir+'/valid_extra_x.csv',
                '--separate_test_path',split_dir+'/test.csv',
                '--separate_test_features_path',split_dir+'/test_extra_x.csv',
                '--data_weights_path',split_dir+'/train_weights.csv',
                '--config_path','../data/args_files/optimized_configs.json',
                '--loss_function','mse','--metric','rmse'
            ]
            if 'morgan' in split_folder:
                arguments += ['--features_generator','morgan_count']
            args = chemprop.args.TrainArgs().parse_args(arguments)
            mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
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
            if 'morgan' in split_model_folder:
                    arguments = arguments + ['--features_generator','morgan_count']
            args = chemprop.args.PredictArgs().parse_args(arguments)
            preds = chemprop.train.make_predictions(args=args)
            new_df = pd.read_csv('../results/screen_results/'+argv[2]+'_preds'+'/'+screen_name+'/cv_'+str(cv)+'_preds.csv')
            all_df['smiles'] = new_df.smiles
            all_df[f'cv_{cv}_pred_{TARGET}'] = new_df[TARGET_COL] 
        all_df[f'avg_pred_{TARGET}'] = all_df[[f'cv_{cv}_pred_{TARGET}' for cv in range(cv_num)]].mean(axis=1)
        all_df.to_csv('../results/screen_results/'+argv[2]+'_preds'+'/'+screen_name+'/pred_file.csv', index = False)
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
            '--config_save_path','..results/'+split_folder+'/hyp_cv_0.json',
            '--epochs', '5'
        ]
        args = chemprop.args.HyperoptArgs().parse_args(arguments)
        chemprop.hyperparameter_optimization.hyperopt(args)
    elif task_type == 'analyze':
        # output.to_csv(path_to_folders+'/cv_'+str(i)+'/Predicted_vs_actual.csv', index = False)
        split = argv[2]
        make_pred_vs_actual(split, predictions_done = [], ensemble_size = 5)
        analyze_predictions_cv(split)

    elif task_type == 'split':
        split = argv[2]
        ultra_held_out = float(argv[3])
        is_morgan = False
        in_silico_screen = False
        if len(argv)>4:
            if argv[4]=='morgan':
                is_morgan = True
                if len(argv)>5 and argv[5]=='in_silico_screen_split':
                    in_silico_screen = True
            elif argv[4]=='in_silico_screen_split':
                in_silico_screen = True
        specified_cv_split(split,ultra_held_out_fraction = ultra_held_out, is_morgan = is_morgan, test_is_valid = in_silico_screen)


if __name__ == '__main__':
    main(sys.argv)