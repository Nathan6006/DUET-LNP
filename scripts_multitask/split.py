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

# ==========================================
#      HELPER FUNCTIONS
# ==========================================

def adjust_col_types_in_memory(col_types_df, active_target_col):
    """
    Modifies col_types dataframe in memory.
    1. Ensures active_target_col is 'Y_val'.
    2. Demotes other potential targets to 'Metadata' to prevent leakage/confusion in output files.
    """
    col_types = col_types_df.copy()
    
    # Potential targets
    known_targets = ['quantified_toxicity', 'quantified_delivery', 'unnormalized_toxicity', 'unnormalized_delivery']
    
    for t in known_targets:
        if t == active_target_col:
            # Set active target to Y_val
            col_types.loc[col_types['Column_name'] == t, 'Type'] = 'Y_val'
        else:
            # If it was previously Y_val, demote it to Metadata
            mask = (col_types['Column_name'] == t) & (col_types['Type'] == 'Y_val')
            if mask.any():
                col_types.loc[mask, 'Type'] = 'Metadata'
                
    return col_types

def get_bin_label(val, thresholds):
    """
    Maps value to 0, 1, or 2 based on provided thresholds [low_cut, high_cut].
    0: < low_cut
    1: low_cut <= x < high_cut
    2: >= high_cut
    """
    if pd.isna(val): return -1
    if val < thresholds[0]: return 0
    if val < thresholds[1]: return 1
    return 2

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
    
    if not fps:
        print("Warning: No valid fingerprints generated.")
        return df

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

def stratified_group_split_custom(df, group_col, target_col, test_size, thresholds, min_samples_per_class=10, random_state=42):
    """
    Custom splitter that respects groups and enforces minimum samples per class bin.
    """
    if test_size <= 0:
        return df, pd.DataFrame()

    df = df.copy()
    # Assign custom class bin based on passed thresholds
    df['custom_bin'] = df[target_col].apply(lambda x: get_bin_label(x, thresholds))
    
    # Summarize content of each cluster
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
    for target_class in [0, 1, 2]:
        while current_test_counts[target_class] < min_samples_per_class:
            
            candidates = [c for c in remaining_clusters if cluster_stats.loc[c, target_class] > 0]
            
            if not candidates:
                print(f"Warning: Not enough clusters containing class {target_class} to satisfy minimum requirement.")
                break
                
            chosen = rng.choice(candidates)
            test_clusters.append(chosen)
            remaining_clusters.remove(chosen)
            
            counts_in_cluster = cluster_stats.loc[chosen]
            for c in [0, 1, 2]:
                current_test_counts[c] += counts_in_cluster[c]
            current_test_size += cluster_sizes[chosen]

    # --- PHASE 2: Fill to Target Test Size ---
    remaining_clusters_list = list(remaining_clusters)
    rng.shuffle(remaining_clusters_list)
    
    for cluster in remaining_clusters_list:
        if current_test_size >= target_test_samples:
            break
            
        test_clusters.append(cluster)
        current_test_size += cluster_sizes[cluster]
        
        counts_in_cluster = cluster_stats.loc[cluster]
        for c in [0, 1, 2]:
            current_test_counts[c] += counts_in_cluster[c]

    # --- Final Split ---
    test_df = df[df[group_col].isin(test_clusters)].copy()
    train_df = df[~df[group_col].isin(test_clusters)].copy()
    
    test_df.drop(columns=['custom_bin'], inplace=True, errors='ignore')
    train_df.drop(columns=['custom_bin'], inplace=True, errors='ignore')
    
    print("--- Custom Split Statistics ---")
    print(f"Test Set Total: {len(test_df)} samples")
    print(f"Class Low (<{thresholds[0]}):   {current_test_counts[0]}")
    print(f"Class Mid ({thresholds[0]}-{thresholds[1]}): {current_test_counts[1]}")
    print(f"Class High (>={thresholds[1]}):  {current_test_counts[2]}")
    
    return train_df, test_df


# ==========================================
#      MAIN SPLIT FUNCTION (BUTINA)
# ==========================================

def cv_split_butina(split_spec_fname, mode_flag, path_to_folders='../data',
                    cv_fold=5, ultra_held_out_fraction=-1.0,
                    test_frac=0.2, random_state=42, 
                    y_target_col='quantified_toxicity',
                    smiles_col='smiles',
                    butina_cutoff=0.2): 
    
    # --- 1. Load Data ---
    all_df = pd.read_csv(os.path.join(path_to_folders, 'all_data.csv'))
    
    # Filter for valid targets immediately
    print(f"Filtering data for target: {y_target_col}")
    all_df = all_df.dropna(subset=[y_target_col]).copy()
    
    split_df = pd.read_csv(os.path.join(path_to_folders, 'crossval_split_specs', split_spec_fname))
    col_types_raw = pd.read_csv(os.path.join(path_to_folders, 'col_types.csv'))
    
    # Adjust Col Types for the specific target
    col_types = adjust_col_types_in_memory(col_types_raw, y_target_col)

    # --- 2. Setup Directories ---
    # Naming convention: {split_spec_file}_{mode}_B
    split_base = split_spec_fname.replace('.csv', '')
    split_name = f"{split_base}_{mode_flag}_B"
    
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

    # --- Determine Thresholds for Custom Splitter ---
    if mode_flag == 'tox':
        # Legacy thresholds for toxicity
        split_thresholds = [0.7, 0.8]
    else:
        # Dynamic thresholds for delivery (or others) using 33rd/66th percentile
        q33 = splittable_pool[y_target_col].quantile(0.33)
        q66 = splittable_pool[y_target_col].quantile(0.66)
        split_thresholds = [q33, q66]
        print(f"Calculated dynamic thresholds for {mode_flag}: {split_thresholds}")

    # --- 5. Perform Hierarchical Splitting ---
    
    # A) Ultra Held Out (Cluster-based)
    uho_df = pd.DataFrame()
    if ultra_held_out_fraction > 0:
        try:
            # Simple stratification bin
            splittable_pool['strat_bin_uho'] = pd.qcut(splittable_pool[y_target_col], q=5, labels=False, duplicates='drop')
        except:
            splittable_pool['strat_bin_uho'] = pd.cut(splittable_pool[y_target_col], bins=5, labels=False)

        try:
            train_grps, uho_grps = train_test_split(
                splittable_pool['cluster_id'].unique(),
                test_size=ultra_held_out_fraction,
                random_state=random_state
            )
            uho_df = splittable_pool[splittable_pool['cluster_id'].isin(uho_grps)].copy()
            splittable_pool = splittable_pool[splittable_pool['cluster_id'].isin(train_grps)].copy()
        except:
             print("UHO Split Warning: Falling back to random row split due to group constraints")

    # B) Test Set (Cluster-based + Strict Class Counts)
    adj_test_frac = test_frac / (1 - (ultra_held_out_fraction if ultra_held_out_fraction > 0 else 0))
    
    print("\nGenerating Test Set with strict class requirements...")
    cv_pool, test_df = stratified_group_split_custom(
        splittable_pool, 
        group_col='cluster_id', 
        target_col=y_target_col, 
        test_size=adj_test_frac, 
        thresholds=split_thresholds,
        min_samples_per_class=10,
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
    cv_pool = cv_pool.copy()
    try:
        cv_pool['strat_bin'] = pd.qcut(cv_pool[y_target_col], q=5, labels=False, duplicates='drop')
    except:
        cv_pool['strat_bin'] = pd.cut(cv_pool[y_target_col], bins=5, labels=False)

    sgkf = StratifiedGroupKFold(n_splits=cv_fold, shuffle=True, random_state=random_state)
    
    fold_idx = 0
    for train_idxs, val_idxs in sgkf.split(cv_pool, cv_pool['strat_bin'], cv_pool['cluster_id']):
        path_if_none(os.path.join(split_path, f'cv_{fold_idx}'))
        
        fold_train = cv_pool.iloc[train_idxs]
        fold_val = cv_pool.iloc[val_idxs]
        
        final_train = pd.concat([perma_train, fold_train], ignore_index=True)
        final_val = fold_val.copy()
        
        # Weighted on the specific target
        # final_train = generate_weights_gkde(final_train, target_col=y_target_col)
        
        for df, name in [(final_train, 'train'), (final_val, 'valid')]:
            y_f, x_f, w_f, m_f = split_df_by_col_type(df, col_types)
            yxwm_to_csvs(y_f, x_f, w_f, m_f, os.path.join(split_path, f'cv_{fold_idx}'), name)
            
        print(f"Saved Fold {fold_idx}: Train {len(final_train)}, Val {len(final_val)}")
        fold_idx += 1

    print(f"Full Butina stratified split finished at {split_path}")

# ==========================================
#      MAIN SPLIT FUNCTION (STRATIFIED)
# ==========================================

def cv_split_stratified(split_spec_fname, mode_flag, path_to_folders='../data',
                       cv_fold=5, ultra_held_out_fraction=-1.0,
                       test_frac=0.2, random_state=42, 
                       y_target_col='quantified_toxicity'): 
    
    # --- 1. Load Data ---
    all_df = pd.read_csv(os.path.join(path_to_folders, 'all_data.csv'))
    
    # Filter for valid targets
    print(f"Filtering data for target: {y_target_col}")
    all_df = all_df.dropna(subset=[y_target_col]).copy()

    split_df = pd.read_csv(os.path.join(path_to_folders, 'crossval_split_specs', split_spec_fname))
    col_types_raw = pd.read_csv(os.path.join(path_to_folders, 'col_types.csv'))
    
    # Adjust Col Types for the specific target
    col_types = adjust_col_types_in_memory(col_types_raw, y_target_col)

    # --- 2. Setup Directories ---
    # Naming convention: {split_spec_file}_{mode}_S
    split_base = split_spec_fname.replace('.csv', '')
    split_name = f"{split_base}_{mode_flag}_S"

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
    def add_strat_bins(df, target_col, n_bins=5):
        if df.empty: return df
        df = df.dropna(subset=[target_col]).copy()
        try:
            df['strat_bin'] = pd.qcut(df[target_col], q=n_bins, labels=False, duplicates='drop')
        except:
            df['strat_bin'] = pd.cut(df[target_col], bins=n_bins, labels=False)
        return df

    # --- 4. Perform Splitting ---
    
    # A) Process Context Data (Grouped by Lipid, with UHO)
    ctx_uho, ctx_test, ctx_cv_pool, ctx_folds = get_context_splits(
        context_pool, cv_fold, test_frac, ultra_held_out_fraction, y_target_col
    )
    
    # B) Process Standard Data (Row-level, with UHO)
    std_uho, std_test, std_train_val = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    if not standard_pool.empty:
        standard_pool = add_strat_bins(standard_pool, y_target_col)
        
        if ultra_held_out_fraction > 0:
            std_cv_test, std_uho = train_test_split(
                standard_pool, test_size=ultra_held_out_fraction, 
                random_state=random_state, stratify=standard_pool['strat_bin']
            )
        else:
            std_cv_test = standard_pool

        adj_test_size = test_frac / (1 - ultra_held_out_fraction) if ultra_held_out_fraction > 0 else test_frac
        std_train_val, std_test = train_test_split(
            std_cv_test, test_size=adj_test_size, 
            random_state=random_state, stratify=std_cv_test['strat_bin']
        )
        
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
    if not std_train_val.empty:
        std_train_val = add_strat_bins(std_train_val, y_target_col)
        skf = StratifiedKFold(n_splits=cv_fold, shuffle=True, random_state=random_state)
        std_fold_gen = list(skf.split(std_train_val, std_train_val['strat_bin']))
        std_train_val.drop(columns=['strat_bin'], inplace=True)
    else:
        std_fold_gen = []

    for i in range(cv_fold):
        path_if_none(os.path.join(split_path, f'cv_{i}'))
        
        f_train = pd.concat([
            perma_train,
            std_train_val.iloc[std_fold_gen[i][0]] if std_fold_gen else pd.DataFrame(),
            ctx_cv_pool.loc[ctx_folds[i][0]] if ctx_folds else pd.DataFrame()
        ], ignore_index=True)
        
        f_val = pd.concat([
            std_train_val.iloc[std_fold_gen[i][1]] if std_fold_gen else pd.DataFrame(),
            ctx_cv_pool.loc[ctx_folds[i][1]] if ctx_folds else pd.DataFrame()
        ], ignore_index=True)
        
        f_train = generate_weights_gkde(f_train, target_col=y_target_col)
        
        for df, name in [(f_train, 'train'), (f_val, 'valid')]:
            y_f, x_f, w_f, m_f = split_df_by_col_type(df, col_types)
            yxwm_to_csvs(y_f, x_f, w_f, m_f, os.path.join(split_path, f'cv_{i}'), name)

    print(f"Full stratified split finished at {split_path}")


# Helper functions for cv split
def get_context_splits(df, cv_fold, test_frac, uho_frac, y_col, group_col='smiles', random_state=42):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    group_repr = df.groupby(group_col)[y_col].mean().reset_index()
    
    try:
        group_repr['strat_label'] = pd.qcut(group_repr[y_col], q=5, labels=False, duplicates='drop')
    except:
        group_repr['strat_label'] = pd.cut(group_repr[y_col], bins=5, labels=False)

    counts = group_repr['strat_label'].value_counts()
    small_classes = counts[counts < cv_fold].index.tolist()
    
    forced_train_lipids = group_repr[group_repr['strat_label'].isin(small_classes)][group_col].tolist()
    strat_pool = group_repr[~group_repr['strat_label'].isin(small_classes)].copy()

    uho_df = pd.DataFrame()
    if uho_frac > 0:
        cv_test_ids, uho_ids = train_test_split(
            strat_pool[group_col], test_size=uho_frac, 
            random_state=random_state, stratify=strat_pool['strat_label']
        )
        uho_df = df[df[group_col].isin(uho_ids)].copy()
        strat_pool = strat_pool[strat_pool[group_col].isin(cv_test_ids)]

    adj_test_size = test_frac / (1 - (uho_frac if uho_frac > 0 else 0))
    cv_ids, test_ids = train_test_split(
        strat_pool[group_col], test_size=adj_test_size, 
        random_state=random_state, stratify=strat_pool['strat_label']
    )
    test_df = df[df[group_col].isin(test_ids)].copy()
    
    cv_pool_df = df[df[group_col].isin(cv_ids) | df[group_col].isin(forced_train_lipids)].copy()
    cv_pool_repr = strat_pool[strat_pool[group_col].isin(cv_ids)]

    skf = StratifiedKFold(n_splits=cv_fold, shuffle=True, random_state=random_state)
    
    fold_map = {}
    for fold_idx, (_, val_idx) in enumerate(skf.split(cv_pool_repr[group_col], cv_pool_repr['strat_label'])):
        for lip in cv_pool_repr.iloc[val_idx][group_col].values:
            fold_map[lip] = fold_idx

    cv_folds = []
    for i in range(cv_fold):
        val_idx = cv_pool_df[cv_pool_df[group_col].map(fold_map) == i].index
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
def generate_weights_gkde(
    all_df,
    target_col,
    power=0.85,
    bandwidth='silverman',
    bandwidth_adjust=0.5,
    clip_quantile=0.995,
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

    y_std = np.std(y)

    upper_mask = (upper_bound - y) < (reflect_frac * y_std)
    y_upper = 2 * upper_bound - y[upper_mask]

    lower_mask = (y - lower_bound) < (reflect_frac * y_std)
    y_lower = 2 * lower_bound - y[lower_mask]

    y_combined = np.concatenate([y, y_upper, y_lower])

    kde = gaussian_kde(y_combined, bw_method=bandwidth)
    kde.set_bandwidth(kde.factor * bandwidth_adjust)

    densities = kde(y)
    densities = np.maximum(densities, 1e-8)

    weights = 1.0 / (densities ** power)
    weights /= weights.mean()

    if clip_quantile is not None:
        limit = np.quantile(weights, clip_quantile)
        weights = np.clip(weights, a_min=None, a_max=limit)
    else:
        weights = np.clip(weights, a_min=None, a_max=clip_max)

    weights /= weights.mean()

    # ---- ESS + boost factor ----
    ess = (weights.sum() ** 2) / np.sum(weights ** 2)
    boost_factor = weights.max()

    if verbose:
        print(f"GKDE weighting for '{target_col}':")
        print(f"  N samples      : {len(weights)}")
        print(f"  ESS            : {ess:.1f} ({ess / len(weights):.2%} of N)")
        print(f"  Boost factor   : {boost_factor:.2f}×")

    # Map back to DataFrame
    kde_series = pd.Series(weights, index=all_df.loc[mask].index)
    kde_multipliers = all_df.index.map(kde_series).fillna(1.0)

    if 'Sample_weight' not in all_df.columns:
        all_df['Sample_weight'] = 1.0

    all_df['Sample_weight'] *= kde_multipliers

    return all_df

def main(argv):
    if len(argv) < 3:
        print("Usage: python script.py <split_spec> <mode: del or tox> [--cv <folds>]")
        sys.exit(1)

    split = argv[1]
    mode_arg = argv[2].lower()
    
    # Determine target based on mode
    if mode_arg == 'del':
        target_col = 'quantified_delivery'
        mode_flag = 'del'
    elif mode_arg == 'tox':
        target_col = 'quantified_toxicity'
        mode_flag = 'tox'
    else:
        print(f"Error: Unknown mode '{mode_arg}'. Use 'del' or 'tox'.")
        sys.exit(1)

    cv_num = 5
    if len(argv) > 3:
        for i, arg in enumerate(argv):
            if arg.replace('–', '-') == '--cv':
                cv_num = int(argv[i+1])
                print('this many folds: ', str(cv_num))
    
    cv_split_butina(split, mode_flag=mode_flag, cv_fold=cv_num, y_target_col=target_col)

    
if __name__ == '__main__':
    main(sys.argv)