import os
import pandas as pd  
import sys
import random
from sklearn.model_selection import train_test_split, StratifiedKFold
from helpers import path_if_none, perform_scaffold_split

"""
script that splits data from all_data.csv to spilt files 
"""

def cv_split_stratified(split_spec_fname, path_to_folders='../data',
                       cv_fold=5, ultra_held_out_fraction=-1.0,
                       test_frac=0.2, random_state=42, 
                       y_stratify_col='toxicity_class'):
    
    # --- 1. Load Data ---
    all_df = pd.read_csv(os.path.join(path_to_folders, 'all_data.csv'))
    split_df = pd.read_csv(os.path.join(path_to_folders, 'crossval_split_specs', split_spec_fname))
    col_types = pd.read_csv(os.path.join(path_to_folders, 'col_type.csv'))

    # --- 2. Setup Directories ---
    split_name = split_spec_fname.replace('.csv', '')
    split_path = os.path.join(path_to_folders, 'crossval_splits', split_name)
    if ultra_held_out_fraction > 0:
        split_path += '_with_uho'
    
    path_if_none(split_path)
    path_if_none(os.path.join(split_path, 'test'))
    if ultra_held_out_fraction > 0:
        path_if_none(os.path.join(split_path, 'ultra_held_out'))

    # --- 3. Partition Data into Pools ---
    perma_train, standard_pool, context_pool = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for _, row in split_df.iterrows():
        dtypes = [d.strip() for d in row['Data_types_for_component'].split(',')]
        vals = [v.strip() for v in row['Values'].split(',')]
        
        subset = all_df.copy()
        for dtype, val in zip(dtypes, vals):
            subset = subset[subset[dtype].astype(str) == str(val)]
        
        if subset.empty: continue

        mode = row['Train_or_split'].lower()
        if mode == 'train':
            perma_train = pd.concat([perma_train, subset])
        elif mode == 'split':
            standard_pool = pd.concat([standard_pool, subset])
        elif mode == 'split_context':
            context_pool = pd.concat([context_pool, subset])

    # --- 4. Perform Splitting ---
    # A) Process Context Data (Grouped by Lipid, with UHO)
    ctx_uho, ctx_test, ctx_cv_pool, ctx_folds = get_context_splits(
        context_pool, cv_fold, test_frac, ultra_held_out_fraction, y_stratify_col
    )
    
    # B) Process Standard Data (Row-level, with UHO)
    std_uho, std_test, std_train_val = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if not standard_pool.empty:
        # Standard UHO split
        if ultra_held_out_fraction > 0:
            std_cv_test, std_uho = train_test_split(
                standard_pool, test_size=ultra_held_out_fraction, 
                random_state=random_state, stratify=standard_pool[y_stratify_col]
            )
        else:
            std_cv_test = standard_pool

        # Standard Test split
        adj_test_size = test_frac / (1 - ultra_held_out_fraction) if ultra_held_out_fraction > 0 else test_frac
        std_train_val, std_test = train_test_split(
            std_cv_test, test_size=adj_test_size, 
            random_state=random_state, stratify=std_cv_test[y_stratify_col]
        )

    # --- 5. Save Results ---
    # 5a. Save Ultra Held Out
    final_uho = pd.concat([std_uho, ctx_uho], ignore_index=True)
    if not final_uho.empty:
        y, x, w, m = split_df_by_col_type(final_uho, col_types)
        yxwm_to_csvs(y, x, w, m, os.path.join(split_path, 'ultra_held_out'), 'test')

    # 5b. Save Global Test Set
    final_test = pd.concat([std_test, ctx_test], ignore_index=True)
    y, x, w, m = split_df_by_col_type(final_test, col_types)
    yxwm_to_csvs(y, x, w, m, os.path.join(split_path, 'test'), 'test')

    # 5c. Save CV Folds
    skf = StratifiedKFold(n_splits=cv_fold, shuffle=True, random_state=random_state)
    std_fold_gen = list(skf.split(std_train_val, std_train_val[y_stratify_col])) if not std_train_val.empty else []

    for i in range(cv_fold):
        path_if_none(os.path.join(split_path, f'cv_{i}'))
        
        # Build training set: Perma + Std Fold Train + Ctx Fold Train
        f_train = pd.concat([
            perma_train,
            std_train_val.iloc[std_fold_gen[i][0]] if std_fold_gen else pd.DataFrame(),
            ctx_cv_pool.loc[ctx_folds[i][0]] if ctx_folds else pd.DataFrame()
        ], ignore_index=True)
        
        # Build validation set: Std Fold Val + Ctx Fold Val
        f_val = pd.concat([
            std_train_val.iloc[std_fold_gen[i][1]] if std_fold_gen else pd.DataFrame(),
            ctx_cv_pool.loc[ctx_folds[i][1]] if ctx_folds else pd.DataFrame()
        ], ignore_index=True)

        # Save Fold
        for df, name in [(f_train, 'train'), (f_val, 'valid')]:
            y_f, x_f, w_f, m_f = split_df_by_col_type(df, col_types)
            yxwm_to_csvs(y_f, x_f, w_f, m_f, os.path.join(split_path, f'cv_{i}'), name)

    print(f"Full stratified split finished at {split_path}")


# Helper functions for cv split
def get_context_splits(df, cv_fold, test_frac, uho_frac, y_col, group_col='Lipid_name', random_state=42):
    """
    Groups lipids and performs standard fold splitting (1/K validation).
    Uses the MINIMUM class value (most potent) for stratification.
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    # 1. Identify "worst-case" class per lipid across all conditions/datasets
    group_repr = df.groupby(group_col)[y_col].min().reset_index()
    group_repr.rename(columns={y_col: 'strat_label'}, inplace=True)

    # 2. Safety: Identify classes with fewer members than cv_fold
    # These are the cause of the UserWarning. We move them to a forced training pool.
    counts = group_repr['strat_label'].value_counts()
    small_classes = counts[counts < cv_fold].index.tolist()
    
    forced_train_lipids = group_repr[group_repr['strat_label'].isin(small_classes)][group_col].tolist()
    # The pool we can safely stratify
    strat_pool = group_repr[~group_repr['strat_label'].isin(small_classes)].copy()

    # 3. Stage 1: Separate Ultra-Held-Out (UHO)
    uho_df = pd.DataFrame()
    if uho_frac > 0:
        cv_test_ids, uho_ids = train_test_split(
            strat_pool[group_col], test_size=uho_frac, 
            random_state=random_state, stratify=strat_pool['strat_label']
        )
        uho_df = df[df[group_col].isin(uho_ids)].copy()
        strat_pool = strat_pool[strat_pool[group_col].isin(cv_test_ids)]

    # 4. Stage 2: Separate Test Set
    # Scale test_frac to the remainder
    adj_test_size = test_frac / (1 - (uho_frac if uho_frac > 0 else 0))
    cv_ids, test_ids = train_test_split(
        strat_pool[group_col], test_size=adj_test_size, 
        random_state=random_state, stratify=strat_pool['strat_label']
    )
    test_df = df[df[group_col].isin(test_ids)].copy()
    
    # The CV Pool includes the stratified lipids + the lipids from small classes
    cv_pool_df = df[df[group_col].isin(cv_ids) | df[group_col].isin(forced_train_lipids)].copy()
    cv_pool_repr = strat_pool[strat_pool[group_col].isin(cv_ids)]

    # 5. Stage 3: Generate CV Folds (Standard 1/K Split)
    skf = StratifiedKFold(n_splits=cv_fold, shuffle=True, random_state=random_state)
    
    fold_map = {}
    # We only stratify the lipids that have enough members to fill all folds
    for fold_idx, (_, val_idx) in enumerate(skf.split(cv_pool_repr[group_col], cv_pool_repr['strat_label'])):
        for lip in cv_pool_repr.iloc[val_idx][group_col].values:
            fold_map[lip] = fold_idx

    cv_folds = []
    for i in range(cv_fold):
        # Validation: Only contains lipids from the stratified pool assigned to this fold
        val_idx = cv_pool_df[cv_pool_df[group_col].map(fold_map) == i].index
        
        # Training: Contains lipids from other folds + ALL forced_train_lipids
        train_idx = cv_pool_df[(cv_pool_df[group_col].map(fold_map) != i) | 
                              (cv_pool_df[group_col].isin(forced_train_lipids))].index
        
        cv_folds.append((train_idx, val_idx))

    return uho_df, test_df, cv_pool_df, cv_folds


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
    cv_num = 5
    if len(argv)>3:
        for i, arg in enumerate(argv):
            if arg.replace('–', '-') == '--cv':
                cv_num = int(argv[i+1])
                print('this many folds: ',str(cv_num))
    cv_split_stratified(split, cv_fold=cv_num, ultra_held_out_fraction = ultra_held_out)

    
if __name__ == '__main__':
    main(sys.argv)