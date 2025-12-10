import os
import pandas as pd  
import sys
import random
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold
from helpers import path_if_none, perform_scaffold_split

def cv_split(split_spec_fname, path_to_folders='../data',
                       is_morgan=False, cv_fold=2, ultra_held_out_fraction=-1.0,
                       min_unique_vals=2.0, test_is_valid=False,
                       train_frac=0.65, valid_frac=.175, test_frac=0.175,
                       random_state=42, 
                       split_type='scaffold',  # 'random', 'stratified', 'scaffold'
                       smiles_col='smiles'):
    
    all_df = pd.read_csv(os.path.join(path_to_folders, 'all_data.csv'))
    split_df = pd.read_csv(os.path.join(path_to_folders, 'crossval_split_specs', split_spec_fname))
    col_types = pd.read_csv(os.path.join(path_to_folders, 'col_type.csv'))

    split_path = os.path.join(path_to_folders, 'crossval_splits', split_spec_fname[:-4])
    
    # Append split type to folder name so we don't overwrite standard random splits
    if split_type != 'random':
        split_path += f'_{split_type}'
        
    if ultra_held_out_fraction > 0:
        split_path += '_with_uho'
    if is_morgan:
        split_path += '_morgan'
    if test_is_valid:
        split_path += '_for_iss'

    # Create Directories
    path_if_none(split_path)
    if ultra_held_out_fraction > 0:
        path_if_none(os.path.join(split_path, 'ultra_held_out'))
    for i in range(cv_fold):
        path_if_none(os.path.join(split_path, f'cv_{i}'))

    perma_train = pd.DataFrame({})
    ultra_held_out = pd.DataFrame({})
    cv_data = pd.DataFrame({})

    for _, row in split_df.iterrows():
        dtypes = row['Data_types_for_component'].split(',')
        vals = row['Values'].split(',')
        df_to_concat = all_df.copy()

        # Filter down to specific subset based on specs
        for i, dtype in enumerate(dtypes):
            df_to_concat = df_to_concat[df_to_concat[dtype.strip()] == vals[i].strip()].reset_index(drop=True)

        values_to_split = df_to_concat[row['Data_type_for_split']]
        unique_values_to_split = list(set(values_to_split))

        if row['Train_or_split'].lower() == 'train':
            # This data is ALWAYS in training (never validation or test)
            perma_train = pd.concat([perma_train, df_to_concat])
            
        elif row['Train_or_split'].lower() == 'split':
            # This data is eligible for Splitting (Test/Train/CV)
            
            # NOTE: We keep the original logic here for "Ultra Held Out"
            # The "split_for_cv" helper is used purely to peel off the Ultra Held Out set
            # The remaining data is passed to our new logic as "cv_data"
            
            cv_split_values, ultra_held_out_values = split_for_cv(unique_values_to_split, cv_fold, ultra_held_out_fraction)
            
            # Add to Ultra Held Out
            ultra_held_out = pd.concat([ultra_held_out, df_to_concat[df_to_concat[row['Data_type_for_split']].isin(ultra_held_out_values)]])
            
            # Add to CV pool (Test + Train + Valid)
            # We combine the lists from split_for_cv because we will re-split them ourselves using KFold/Stratified/Scaffold
            cv_data = pd.concat([cv_data, df_to_concat[df_to_concat[row['Data_type_for_split']].isin(sum(cv_split_values, []))]])

    # Save Ultra Held Out
    if ultra_held_out_fraction >= 0 and not ultra_held_out.empty:
        y, x, w, m = split_df_by_col_type(ultra_held_out, col_types)
        yxwm_to_csvs(y, x, w, m, split_path + '/ultra_held_out', 'test')

    if abs(train_frac + valid_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train_frac + valid_frac + test_frac must sum to 1.0")
    
    # --- 2. Apply New Splitting Logic on `cv_data` ---
    
    cv_data = cv_data.reset_index(drop=True)
    
    if split_type == 'scaffold':
        print(f"Performing Scaffold Split using {smiles_col}...")
        train_valid_df, test_df, kf_iterator = perform_scaffold_split(
            cv_data, smiles_col, n_splits=cv_fold, test_frac=test_frac, random_state=random_state
        )
    
    else:
        # Determine Stratification Column if needed
        stratify_col = None
        if split_type == 'stratified':
            print("Performing Stratified Split...")
            y_col_name = col_types[col_types['Type'] == 'Y_val']['Column_name'].values[0]
            stratify_col = cv_data[y_col_name]
        
        # 2a. Split Fixed Test Set
        train_valid_df, test_df = train_test_split(
            cv_data, 
            test_size=test_frac, 
            random_state=random_state, 
            shuffle=True,
            stratify=stratify_col
        )
        
        # 2b. Prepare K-Fold Iterator
        train_valid_df = train_valid_df.reset_index(drop=True)
        
        if split_type == 'stratified':
            # Re-fetch stratify column for the subset
            y_col_name = col_types[col_types['Type'] == 'Y_val']['Column_name'].values[0]
            stratify_col_subset = train_valid_df[y_col_name]
            
            kf = StratifiedKFold(n_splits=cv_fold, shuffle=True, random_state=random_state)
            kf_iterator = kf.split(train_valid_df, stratify_col_subset)
        else:
            kf = KFold(n_splits=cv_fold, shuffle=True, random_state=random_state)
            kf_iterator = kf.split(train_valid_df)

    # Save Fixed Test Set
    y, x, w, m = split_df_by_col_type(test_df, col_types)
    path_if_none(split_path + '/test')
    yxwm_to_csvs(y, x, w, m, split_path + '/test', 'test')
    
    # --- 3. Iterate Folds and Save ---
    
    for i, (train_index, valid_index) in enumerate(kf_iterator):
        
        fold_train = train_valid_df.iloc[train_index]
        fold_valid = train_valid_df.iloc[valid_index]
        
        # Merge perma_train (data forced to be in train via split_spec) into the training fold
        if not perma_train.empty:
            fold_train = pd.concat([fold_train, perma_train]).drop_duplicates().reset_index(drop=True)
        
        # Save Valid
        y, x, w, m = split_df_by_col_type(fold_valid, col_types)
        yxwm_to_csvs(y, x, w, m, split_path + '/cv_' + str(i), 'valid')

        # Save Train
        y, x, w, m = split_df_by_col_type(fold_train, col_types)
        yxwm_to_csvs(y, x, w, m, split_path + '/cv_' + str(i), 'train')

def cv_split_old(split_spec_fname, path_to_folders='../data',
                       is_morgan=False, cv_fold=2, ultra_held_out_fraction=-1.0,
                       min_unique_vals=2.0, test_is_valid=False,
                       train_frac=0.65, valid_frac=.175, test_frac=0.175,
                       random_state=42):
    """
    Splits the dataset according to the specifications in split_spec_fname.
    Uses sklearn to create a single fixed test set and splits the rest into K-fold train/valid.
    
    UPDATED: Now correctly rotates the validation fold using KFold logic.
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

    perma_train = pd.DataFrame({})
    ultra_held_out = pd.DataFrame({})
    cv_data = pd.DataFrame({})

    # Process Split Rules
    for _, row in split_df.iterrows():
        dtypes = row['Data_types_for_component'].split(',')
        vals = row['Values'].split(',')
        df_to_concat = all_df.copy()

        for i, dtype in enumerate(dtypes):
            df_to_concat = df_to_concat[df_to_concat[dtype.strip()] == vals[i].strip()].reset_index(drop=True)

        values_to_split = df_to_concat[row['Data_type_for_split']]
        unique_values_to_split = list(set(values_to_split))

        if row['Train_or_split'].lower() == 'train':
            perma_train = pd.concat([perma_train, df_to_concat])
        elif row['Train_or_split'].lower() == 'split':
            # This helper effectively just shuffles unique values here
            cv_split_values, ultra_held_out_values = split_for_cv(unique_values_to_split, cv_fold, ultra_held_out_fraction)
            
            ultra_held_out = pd.concat([ultra_held_out, df_to_concat[df_to_concat[row['Data_type_for_split']].isin(ultra_held_out_values)]])
            
            # We collect ALL potential CV data here. The actual K-Fold happens later.
            cv_data = pd.concat([cv_data, df_to_concat[df_to_concat[row['Data_type_for_split']].isin(sum(cv_split_values, []))]])

    # Save Ultra Held Out
    if ultra_held_out_fraction >= 0 and not ultra_held_out.empty:
        y, x, w, m = split_df_by_col_type(ultra_held_out, col_types)
        yxwm_to_csvs(y, x, w, m, split_path + '/ultra_held_out', 'test')

    if abs(train_frac + valid_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train_frac + valid_frac + test_frac must sum to 1.0")

    # 1. Create the fixed Test Set
    # We remove the test set first so it doesn't leak into CV
    train_valid_df, test_df = train_test_split(
        cv_data, test_size=test_frac, random_state=random_state, shuffle=True
    )
    
    # Save Test Set
    y, x, w, m = split_df_by_col_type(test_df, col_types)
    path_if_none(split_path + '/test')
    yxwm_to_csvs(y, x, w, m, split_path + '/test', 'test')
    
    # 2. Perform Real K-Fold Cross Validation on the remaining data
    # Reset index is crucial so KFold indices align with the dataframe
    train_valid_df = train_valid_df.reset_index(drop=True)
    
    kf = KFold(n_splits=cv_fold, shuffle=True, random_state=random_state)
    
    # Iterate through the folds
    for i, (train_index, valid_index) in enumerate(kf.split(train_valid_df)):
        
        fold_train = train_valid_df.iloc[train_index]
        fold_valid = train_valid_df.iloc[valid_index]
        
        # Add the 'perma_train' data (data hardcoded to always be in train) to the training fold
        if not perma_train.empty:
            fold_train = pd.concat([fold_train, perma_train]).drop_duplicates().reset_index(drop=True)
        
        # Save Valid
        y, x, w, m = split_df_by_col_type(fold_valid, col_types)
        yxwm_to_csvs(y, x, w, m, split_path + '/cv_' + str(i), 'valid')

        # Save Train
        y, x, w, m = split_df_by_col_type(fold_train, col_types)
        yxwm_to_csvs(y, x, w, m, split_path + '/cv_' + str(i), 'train')

# Helper functions for cv split

def split_df_by_col_type(df, col_types):
    y_vals_cols = col_types.Column_name[col_types.Type == 'Y_val']
    x_vals_cols = col_types.Column_name[col_types.Type == 'X_val']
    xvals_df = df[x_vals_cols]
    weight_cols = col_types.Column_name[col_types.Type == 'Sample_weight']
    metadata_cols = col_types.Column_name[col_types.Type.isin(['Metadata','X_val_categorical'])]
    return df[y_vals_cols], xvals_df, df[weight_cols], df[metadata_cols]

def yxwm_to_csvs(y, x, w, m, path, settype):
    y.to_csv(os.path.join(path, settype+'.csv'), index=False)
    x.to_csv(os.path.join(path, settype + '_extra_x.csv'), index=False)
    w.to_csv(os.path.join(path, settype + '_weights.csv'), index=False)
    m.to_csv(os.path.join(path, settype + '_metadata.csv'), index=False)

def split_for_cv(vals, cv_fold, held_out_fraction):
    random.seed(42)
    random.shuffle(vals)
    held_out_len = int(held_out_fraction * len(vals)) if held_out_fraction > 0 else 0
    held_out_vals = vals[:held_out_len]
    cv_vals = vals[held_out_len:]
    return [cv_vals[i::cv_fold] for i in range(cv_fold)], held_out_vals
def main(argv):
    split = argv[1]
    ultra_held_out = float(argv[2])
    is_morgan = False
    in_silico_screen = False
    cv_num = 2
    if len(argv)>3:
        for i, arg in enumerate(argv):
            if arg.replace('–', '-') == '--cv':
                cv_num = int(argv[i+1])
                print('this many folds: ',str(cv_num))
            if arg.replace('–', '-') == '--morgan':
                is_morgan = True
            if arg.replace('–', '-') == '--in_silico':
                in_silico_screen = True
    cv_split(split, cv_fold=cv_num, ultra_held_out_fraction = ultra_held_out, is_morgan = is_morgan, test_is_valid = in_silico_screen)

    
if __name__ == '__main__':
    main(sys.argv)