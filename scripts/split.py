import numpy as np
import os
import pandas as pd
import random
import sys
from helpers import path_if_none
from sklearn.model_selection import StratifiedKFold, train_test_split, StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import gaussian_kde

# --- New Imports for Butina Clustering ---
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.ML.Cluster import Butina

def assign_butina_clusters(df, smiles_col='smiles', cutoff=0.2, fp_radius=2, fp_bits=1024):
    """
    Generates Butina clusters for unique SMILES in the dataframe and 
    assigns a 'cluster_id' column to the dataframe.
    """
    print("Generating fingerprints and computing Butina clusters...")
    
    # 1. Get unique SMILES to avoid redundant computation
    unique_smiles = df[smiles_col].dropna().unique()
    mols = [Chem.MolFromSmiles(s) for s in unique_smiles]
    
    # Filter out invalid SMILES
    valid_indices = [i for i, m in enumerate(mols) if m is not None]
    valid_smiles = [unique_smiles[i] for i in valid_indices]
    fps = [AllChem.GetMorganFingerprintAsBitVect(mols[i], fp_radius, nBits=fp_bits) for i in valid_indices]
    
    # 2. Compute Distance Matrix (1 - Tanimoto Similarity)
    dists = []
    n_fps = len(fps)
    for i in range(1, n_fps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])
    
    # 3. Run Butina Clustering
    clusters = Butina.ClusterData(dists, n_fps, cutoff, isDistData=True)
    
    # 4. Map back to SMILES
    smiles_to_cluster = {}
    for cluster_id, idx_tuple in enumerate(clusters):
        for idx in idx_tuple:
            smiles_to_cluster[valid_smiles[idx]] = cluster_id
            
    # 5. Apply to DataFrame
    df['cluster_id'] = df[smiles_col].map(smiles_to_cluster)
    
    # Fill NA with unique negative IDs
    na_mask = df['cluster_id'].isna()
    df.loc[na_mask, 'cluster_id'] = range(-1, -1 - sum(na_mask), -1)
    
    print(f"Clustering complete. Found {len(clusters)} clusters for {len(valid_smiles)} unique SMILES.")
    return df

def get_custom_class_label(val):
    """
    Maps toxicity value to 0, 1, or 2 based on:
    0: < 0.7
    1: 0.7 <= x < 0.8
    2: >= 0.8
    """
    if pd.isna(val): return -1
    if val < 0.7: return 0
    if val < 0.8: return 1
    return 2

def stratified_group_split_custom(df, group_col, target_col, test_size, min_samples_per_class=10, random_state=42):
    """
    Custom splitter that:
    1. Respects groups (clusters cannot be split).
    2. Enforces at least `min_samples_per_class` for defined toxicity ranges in the Test set.
    3. Target ranges: [<0.7, 0.7-0.8, >=0.8].
    """
    if test_size <= 0:
        return df, pd.DataFrame()

    df = df.copy()
    # Assign custom class bin
    df['custom_bin'] = df[target_col].apply(get_custom_class_label)
    
    # Summarize content of each cluster
    # We need to know how many Low, Mid, and High samples are in each cluster
    cluster_stats = df.groupby(group_col)['custom_bin'].value_counts().unstack(fill_value=0)
    
    # Ensure all columns 0, 1, 2 exist
    for c in [0, 1, 2]:
        if c not in cluster_stats.columns:
            cluster_stats[c] = 0
            
    # Get total size per cluster
    cluster_sizes = df.groupby(group_col).size()
    
    all_clusters = list(cluster_sizes.index)
    total_samples = len(df)
    target_test_samples = int(total_samples * test_size)
    
    # Initialize Test Sets
    test_clusters = []
    current_test_counts = {0: 0, 1: 0, 2: 0}
    current_test_size = 0
    
    rng = np.random.RandomState(random_state)
    rng.shuffle(all_clusters)
    
    remaining_clusters = set(all_clusters)
    
    # --- PHASE 1: Satisfy Minimum Class Counts ---
    # We iterate through required classes (0, 1, 2). If a class count < 10,
    # we hunt for a cluster that contains that class and move it to Test.
    
    for target_class in [0, 1, 2]:
        while current_test_counts[target_class] < min_samples_per_class:
            
            # Find candidate clusters that have at least 1 sample of this target_class
            # and are not yet in test_clusters
            candidates = [c for c in remaining_clusters if cluster_stats.loc[c, target_class] > 0]
            
            if not candidates:
                print(f"Warning: Not enough clusters containing class {target_class} to satisfy minimum requirement of {min_samples_per_class}.")
                break
                
            # Pick a random candidate
            chosen = rng.choice(candidates)
            
            # Move to Test
            test_clusters.append(chosen)
            remaining_clusters.remove(chosen)
            
            # Update counts
            counts_in_cluster = cluster_stats.loc[chosen]
            for c in [0, 1, 2]:
                current_test_counts[c] += counts_in_cluster[c]
            current_test_size += cluster_sizes[chosen]

    # --- PHASE 2: Fill to Target Test Size ---
    # Now we just fill up the rest of the bucket to reach test_frac
    # Convert set back to list for consistent iteration/shuffling
    remaining_clusters_list = list(remaining_clusters)
    rng.shuffle(remaining_clusters_list)
    
    for cluster in remaining_clusters_list:
        if current_test_size >= target_test_samples:
            break
            
        test_clusters.append(cluster)
        current_test_size += cluster_sizes[cluster]
        
        # Update class counts (just for tracking)
        counts_in_cluster = cluster_stats.loc[cluster]
        for c in [0, 1, 2]:
            current_test_counts[c] += counts_in_cluster[c]

    # --- Final Split ---
    test_df = df[df[group_col].isin(test_clusters)].copy()
    train_df = df[~df[group_col].isin(test_clusters)].copy()
    
    # Cleanup temp column
    test_df.drop(columns=['custom_bin'], inplace=True, errors='ignore')
    train_df.drop(columns=['custom_bin'], inplace=True, errors='ignore')
    
    print("--- Custom Split Statistics ---")
    print(f"Test Set Total: {len(test_df)} samples")
    print(f"Class < 0.7:   {current_test_counts[0]} (Req: {min_samples_per_class})")
    print(f"Class 0.7-0.8: {current_test_counts[1]} (Req: {min_samples_per_class})")
    print(f"Class >= 0.8:  {current_test_counts[2]} (Req: {min_samples_per_class})")
    
    return train_df, test_df


# ==========================================
#      MAIN SPLIT FUNCTION (BUTINA)
# ==========================================

def cv_split_butina(split_spec_fname, path_to_folders='../data',
                    cv_fold=5, ultra_held_out_fraction=-1.0,
                    test_frac=0.2, random_state=42, 
                    y_target_col='quantified_toxicity',
                    smiles_col='smiles',
                    butina_cutoff=0.2): 
    
    # --- 1. Load Data ---
    all_df = pd.read_csv(os.path.join(path_to_folders, 'all_data_regression.csv'))
    split_df = pd.read_csv(os.path.join(path_to_folders, 'crossval_split_specs', split_spec_fname))
    col_types = pd.read_csv(os.path.join(path_to_folders, 'col_types_regression.csv'))

    # --- 2. Setup Directories ---
    split_name = split_spec_fname.replace('.csv', '') + '_butina'
    split_path = os.path.join(path_to_folders, 'crossval_splits', split_name)
    if ultra_held_out_fraction > 0:
        split_path += '_with_uho'
    
    path_if_none(split_path)
    path_if_none(os.path.join(split_path, 'test'))
    if ultra_held_out_fraction > 0:
        path_if_none(os.path.join(split_path, 'ultra_held_out'))

    # --- 3. Partition Data into Pools ---
    perma_train = pd.DataFrame()
    splittable_pool = pd.DataFrame() 

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
        elif mode in ['split', 'split_context']:
            splittable_pool = pd.concat([splittable_pool, subset])

    if splittable_pool.empty:
        print("No data available for splitting.")
        return

    # --- 4. Assign Clusters ---
    splittable_pool = assign_butina_clusters(splittable_pool, smiles_col=smiles_col, cutoff=butina_cutoff)

    # --- 5. Perform Hierarchical Splitting ---
    
    # A) Ultra Held Out (Cluster-based)
    # Note: UHO uses standard stratified group split, or you can switch to custom if UHO also needs min counts
    uho_df = pd.DataFrame()
    if ultra_held_out_fraction > 0:
        # We use a simplified binning for UHO unless specifically requested otherwise
        # Just ensuring groups stay together
        splittable_pool['strat_bin_uho'] = pd.cut(splittable_pool[y_target_col], bins=5, labels=False)
        try:
            train_grps, uho_grps = train_test_split(
                splittable_pool['cluster_id'].unique(),
                test_size=ultra_held_out_fraction,
                random_state=random_state
            )
            # Simple group split fallback for UHO to preserve pool for the strict Test split
            uho_df = splittable_pool[splittable_pool['cluster_id'].isin(uho_grps)].copy()
            splittable_pool = splittable_pool[splittable_pool['cluster_id'].isin(train_grps)].copy()
        except:
             print("UHO Split Warning: Falling back to random row split due to group constraints")

    # B) Test Set (Cluster-based + Strict Class Counts)
    # Adjust test size because pool is smaller after UHO removal
    adj_test_frac = test_frac / (1 - (ultra_held_out_fraction if ultra_held_out_fraction > 0 else 0))
    
    print("\nGenerating Test Set with strict class requirements...")
    cv_pool, test_df = stratified_group_split_custom(
        splittable_pool, 
        group_col='cluster_id', 
        target_col=y_target_col, 
        test_size=adj_test_frac, 
        min_samples_per_class=10, # <--- ENFORCED REQUIREMENT
        random_state=random_state
    )

    # --- 6. Save Fixed Sets ---
    if not uho_df.empty:
        y, x, w, m = split_df_by_col_type(uho_df, col_types)
        yxwm_to_csvs(y, x, w, m, os.path.join(split_path, 'ultra_held_out'), 'test')

    if not test_df.empty:
        y, x, w, m = split_df_by_col_type(test_df, col_types)
        yxwm_to_csvs(y, x, w, m, os.path.join(split_path, 'test'), 'test')

    # --- 7. Generate CV Folds (Cluster-based) ---
    # Create bin for CV stratification
    cv_pool = cv_pool.copy()
    try:
        cv_pool['strat_bin'] = pd.qcut(cv_pool[y_target_col], q=5, labels=False, duplicates='drop')
    except:
        cv_pool['strat_bin'] = pd.cut(cv_pool[y_target_col], bins=5, labels=False)

    sgkf = StratifiedGroupKFold(n_splits=cv_fold, shuffle=True, random_state=random_state)
    
    fold_idx = 0
    # Split provides indices. Arguments: X, y (for stratification), groups (to keep together)
    for train_idxs, val_idxs in sgkf.split(cv_pool, cv_pool['strat_bin'], cv_pool['cluster_id']):
        path_if_none(os.path.join(split_path, f'cv_{fold_idx}'))
        
        # Extract Fold Data
        fold_train = cv_pool.iloc[train_idxs]
        fold_val = cv_pool.iloc[val_idxs]
        
        # Add Perma Train to every training fold
        final_train = pd.concat([perma_train, fold_train], ignore_index=True)
        final_val = fold_val.copy()
        
        # Recalculate Weights using the existing method
        final_train = generate_weights_gkde(final_train)
        
        # Save
        for df, name in [(final_train, 'train'), (final_val, 'valid')]:
            y_f, x_f, w_f, m_f = split_df_by_col_type(df, col_types)
            yxwm_to_csvs(y_f, x_f, w_f, m_f, os.path.join(split_path, f'cv_{fold_idx}'), name)
            
        print(f"Saved Fold {fold_idx}: Train {len(final_train)}, Val {len(final_val)}")
        fold_idx += 1

    print(f"Full Butina stratified split finished at {split_path}")

    
def cv_split_stratified(split_spec_fname, path_to_folders='../data',
                       cv_fold=5, ultra_held_out_fraction=-1.0,
                       test_frac=0.2, random_state=42, 
                       y_target_col='quantified_toxicity'): 
    
    # --- 1. Load Data ---
    all_df = pd.read_csv(os.path.join(path_to_folders, 'all_data_regression.csv'))
    split_df = pd.read_csv(os.path.join(path_to_folders, 'crossval_split_specs', split_spec_fname))
    col_types = pd.read_csv(os.path.join(path_to_folders, 'col_types_regression.csv'))

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

    # --- Helper: Create Bins for Continuous Stratification ---
    # We create a temporary column 'strat_bin' to allow StratifiedKFold to work on regression data
    def add_strat_bins(df, target_col, n_bins=5):
        if df.empty: return df
        # drop na targets
        df = df.dropna(subset=[target_col]).copy()
        try:
            # Try qcut first for equal-sized bins
            df['strat_bin'] = pd.qcut(df[target_col], q=n_bins, labels=False, duplicates='drop')
        except:
            # Fallback to cut if qcut fails (e.g., too many identical values)
            df['strat_bin'] = pd.cut(df[target_col], bins=n_bins, labels=False)
        return df

    # --- 4. Perform Splitting ---
    
    # A) Process Context Data (Grouped by Lipid, with UHO)
    # Note: get_context_splits handles its own binning internally
    ctx_uho, ctx_test, ctx_cv_pool, ctx_folds = get_context_splits(
        context_pool, cv_fold, test_frac, ultra_held_out_fraction, y_target_col
    )
    
    # B) Process Standard Data (Row-level, with UHO)
    std_uho, std_test, std_train_val = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    if not standard_pool.empty:
        # Add bins for stratification
        standard_pool = add_strat_bins(standard_pool, y_target_col)
        
        # Standard UHO split
        if ultra_held_out_fraction > 0:
            std_cv_test, std_uho = train_test_split(
                standard_pool, test_size=ultra_held_out_fraction, 
                random_state=random_state, stratify=standard_pool['strat_bin']
            )
        else:
            std_cv_test = standard_pool

        # Standard Test split
        adj_test_size = test_frac / (1 - ultra_held_out_fraction) if ultra_held_out_fraction > 0 else test_frac
        std_train_val, std_test = train_test_split(
            std_cv_test, test_size=adj_test_size, 
            random_state=random_state, stratify=std_cv_test['strat_bin']
        )
        
        # Clean up temporary bin column
        for df in [std_uho, std_test, std_train_val]:
            if 'strat_bin' in df.columns:
                df.drop(columns=['strat_bin'], inplace=True)

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
    # Re-calculate bins for the training/val pool specifically for CV
    if not std_train_val.empty:
        std_train_val = add_strat_bins(std_train_val, y_target_col)
        skf = StratifiedKFold(n_splits=cv_fold, shuffle=True, random_state=random_state)
        # Stratify using the bin
        std_fold_gen = list(skf.split(std_train_val, std_train_val['strat_bin']))
        std_train_val.drop(columns=['strat_bin'], inplace=True) # clean up
    else:
        std_fold_gen = []

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
        
        f_train = generate_weights_gkde(f_train)
        # Save Fold
        for df, name in [(f_train, 'train'), (f_val, 'valid')]:
            y_f, x_f, w_f, m_f = split_df_by_col_type(df, col_types)
            yxwm_to_csvs(y_f, x_f, w_f, m_f, os.path.join(split_path, f'cv_{i}'), name)

    print(f"Full stratified split finished at {split_path}")


# Helper functions for cv split
def get_context_splits(df, cv_fold, test_frac, uho_frac, y_col, group_col='smiles', random_state=42):
    """
    Groups lipids and performs stratified splitting based on MEAN toxicity per lipid.
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    # 1. Identify representative value (mean toxicity) per lipid
    group_repr = df.groupby(group_col)[y_col].mean().reset_index()
    
    # 2. Bin the continuous values for stratification
    # We use qcut to create 5 bins (quintiles) of toxicity
    try:
        group_repr['strat_label'] = pd.qcut(group_repr[y_col], q=5, labels=False, duplicates='drop')
    except:
        group_repr['strat_label'] = pd.cut(group_repr[y_col], bins=5, labels=False)

    # 3. Safety: Identify bins with fewer members than cv_fold
    counts = group_repr['strat_label'].value_counts()
    small_classes = counts[counts < cv_fold].index.tolist()
    
    forced_train_lipids = group_repr[group_repr['strat_label'].isin(small_classes)][group_col].tolist()
    # The pool we can safely stratify
    strat_pool = group_repr[~group_repr['strat_label'].isin(small_classes)].copy()

    # 4. Stage 1: Separate Ultra-Held-Out (UHO)
    uho_df = pd.DataFrame()
    if uho_frac > 0:
        cv_test_ids, uho_ids = train_test_split(
            strat_pool[group_col], test_size=uho_frac, 
            random_state=random_state, stratify=strat_pool['strat_label']
        )
        uho_df = df[df[group_col].isin(uho_ids)].copy()
        strat_pool = strat_pool[strat_pool[group_col].isin(cv_test_ids)]

    # 5. Stage 2: Separate Test Set
    adj_test_size = test_frac / (1 - (uho_frac if uho_frac > 0 else 0))
    cv_ids, test_ids = train_test_split(
        strat_pool[group_col], test_size=adj_test_size, 
        random_state=random_state, stratify=strat_pool['strat_label']
    )
    test_df = df[df[group_col].isin(test_ids)].copy()
    
    # The CV Pool includes the stratified lipids + the lipids from small classes
    cv_pool_df = df[df[group_col].isin(cv_ids) | df[group_col].isin(forced_train_lipids)].copy()
    cv_pool_repr = strat_pool[strat_pool[group_col].isin(cv_ids)]

    # 6. Stage 3: Generate CV Folds (Standard 1/K Split)
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


# weighting functions 
def generate_weights_bin(all_df):
    """
    Adjusts the 'Sample_weight' column based on class balance
    of 'toxicity_class' (0, 1, 2).

    New_Weight = Old_Weight * Class_Balancing_Multiplier
    Also reports Effective Sample Size (ESS).
    """
    # Ensure weight column exists
    if 'Sample_weight' not in all_df.columns:
        all_df['Sample_weight'] = 1.0

    # Only use rows with labels to compute class weights
    mask = ~all_df['toxicity_class'].isna()
    y = all_df.loc[mask, 'toxicity_class'].values
    unique_classes = np.unique(y)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=y
    )
    weight_dict = dict(zip(unique_classes, class_weights))

    # Apply class multipliers
    class_multipliers = all_df['toxicity_class'].map(weight_dict)
    all_df['Sample_weight'] *= class_multipliers.fillna(1.0)

    # ---- ESS computation (only labeled rows) ----
    w = all_df.loc[mask, 'Sample_weight'].values
    ess = (w.sum() ** 2) / np.sum(w ** 2)

    print(
        f"Binned weighting complete | "
        f"ESS = {ess:.1f} / {mask.sum()} "
        f"({ess / mask.sum():.2%} effective)"
    )

    return all_df


def generate_weights_gkde(
    all_df,
    target_col='unnormalized_toxicity',
    power=0.85,
    bandwidth='silverman',      
    bandwidth_adjust=0.5,       # NEW: Multiplier to tighten the KDE (prevent smoothing out rare items)
    clip_quantile=0.995,         # NEW: Dynamic clipping (e.g., 0.98) instead of hard max
    clip_max=50.0,              
    lower_bound=0.0,            
    upper_bound=1.0,
    reflect_frac=2.0,           
    verbose=True
):
    mask = ~all_df[target_col].isna() & ~np.isinf(all_df[target_col])
    y = all_df.loc[mask, target_col].to_numpy(dtype=float)

    if len(y) < 2:
        return all_df

    # -----------------------------


    y_std = np.std(y)
    
    # Upper Reflection
    upper_mask = (upper_bound - y) < (reflect_frac * y_std)
    y_upper = 2 * upper_bound - y[upper_mask]
    
    # Lower Reflection (NEW)
    lower_mask = (y - lower_bound) < (reflect_frac * y_std)
    y_lower = 2 * lower_bound - y[lower_mask]

    y_combined = np.concatenate([y, y_upper, y_lower])

    kde = gaussian_kde(y_combined, bw_method=bandwidth)
    
    # Manually tighten the bandwidth
    # This reduces "leakage" from the majority class into the minority class
    kde.set_bandwidth(kde.factor * bandwidth_adjust)

    densities = kde(y)
    
    # Prevent divide by zero / extreme explosions
    densities = np.maximum(densities, 1e-8)

    weights = 1.0 / (densities ** power)

    # Normalize to mean=1 first so 'clip_max' makes relative sense
    weights /= weights.mean()

    # Dynamic Clipping (Quantile)
    if clip_quantile is not None:
        limit = np.quantile(weights, clip_quantile)
        if verbose: print(f"Clipping weights at {clip_quantile} quantile: {limit:.2f}")
        weights = np.clip(weights, a_min=None, a_max=limit)
    else:
        # Hard Cap
        weights = np.clip(weights, a_min=None, a_max=clip_max)

    weights /= weights.mean()

    ess = (weights.sum() ** 2) / np.sum(weights ** 2)
    
    if verbose:
        print(f"--- KDE Weighting Diagnostics ---")
        print(f"BW Factor used: {kde.factor:.4f}")
        print(f"Weights: Min={weights.min():.2f}, Max={weights.max():.2f}, Mean={weights.mean():.2f}")
        print(f"ESS: {ess:.1f} / {len(y)} ({(ess/len(y))*100:.1f}%)")
        
        # Check if minority is actually getting boosted
        # Assuming y < 0.7 is minority
        minority_mask = y < 0.7
        if minority_mask.sum() > 0:
            avg_min_wt = weights[minority_mask].mean()
            avg_maj_wt = weights[~minority_mask].mean()
            print(f"Avg Weight (y < 0.7): {avg_min_wt:.2f}")
            print(f"Avg Weight (y >= 0.7): {avg_maj_wt:.2f}")
            print(f"Boost Factor: {avg_min_wt / avg_maj_wt:.1f}x")

    # Map back to DataFrame
    kde_series = pd.Series(weights, index=all_df.loc[mask].index)
    kde_multipliers = all_df.index.map(kde_series).fillna(1.0)

    if 'Sample_weight' not in all_df.columns:
        all_df['Sample_weight'] = 1.0

    all_df['Sample_weight'] *= kde_multipliers

    return all_df

def main(argv):
    split = argv[1]
    cv_num = 5
    if len(argv)>3:
        for i, arg in enumerate(argv):
            if arg.replace('–', '-') == '--cv':
                cv_num = int(argv[i+1])
                print('this many folds: ',str(cv_num))
    cv_split_butina(split, cv_fold=cv_num)

    
if __name__ == '__main__':
    main(sys.argv)