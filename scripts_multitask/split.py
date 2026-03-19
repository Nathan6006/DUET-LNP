import numpy as np
import os
import pandas as pd
import random
import sys
# from helpers import path_if_none
from sklearn.model_selection import StratifiedKFold, train_test_split, StratifiedGroupKFold, GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import gaussian_kde
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdFingerprintGenerator
from rdkit.ML.Cluster import Butina

# ==========================================
#       HELPER FUNCTIONS
# ==========================================

def path_if_none(path):
    if not os.path.exists(path):
        os.makedirs(path)

def adjust_col_types_in_memory(col_types_df, active_target_col):
    col_types = col_types_df.copy()
    known_targets = ['quantified_toxicity', 'quantified_delivery', 'unnormalized_toxicity', 'unnormalized_delivery']
    for t in known_targets:
        if t == active_target_col:
            col_types.loc[col_types['Column_name'] == t, 'Type'] = 'Y_val'
        else:
            mask = (col_types['Column_name'] == t) & (col_types['Type'] == 'Y_val')
            if mask.any():
                col_types.loc[mask, 'Type'] = 'Metadata'
    return col_types

def get_bin_label(val, thresholds):
    if pd.isna(val): return -1
    if val < thresholds[0]: return 0
    if val < thresholds[1]: return 1
    return 2

def assign_butina_clusters(df, smiles_col='smiles', cutoff=0.2, fp_radius=2, fp_bits=1024):
    print("Generating fingerprints and computing Butina clusters...")
    
    unique_smiles = df[smiles_col].dropna().unique()
    mols = [Chem.MolFromSmiles(s) for s in unique_smiles]
    
    # Filter out invalid molecules
    valid_indices = [i for i, m in enumerate(mols) if m is not None]
    valid_smiles = [unique_smiles[i] for i in valid_indices]
    
    # --- UPDATED CODE START ---
    # Create the generator once (more efficient)
    # GetFingerprint() on this generator returns an ExplicitBitVect (same as before)
    mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=fp_radius, fpSize=fp_bits)
    fps = [mfgen.GetFingerprint(mols[i]) for i in valid_indices]
    # --- UPDATED CODE END ---

    if not fps:
        print("Warning: No valid fingerprints generated.")
        return df

    # Calculate distance matrix (1 - similarity)
    # Note: Butina expects the lower triangle of the distance matrix
    dists = []
    n_fps = len(fps)
    for i in range(1, n_fps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])
    
    # Run Clustering
    clusters = Butina.ClusterData(dists, n_fps, cutoff, isDistData=True)
    
    # Map cluster IDs back to SMILES
    smiles_to_cluster = {}
    for cluster_id, idx_tuple in enumerate(clusters):
        for idx in idx_tuple:
            smiles_to_cluster[valid_smiles[idx]] = cluster_id
            
    # Assign to DataFrame
    df['cluster_id'] = df[smiles_col].map(smiles_to_cluster)
    
    # Fill NaN clusters (molecules that failed conversion) with unique negative IDs
    na_mask = df['cluster_id'].isna()
    df.loc[na_mask, 'cluster_id'] = range(-1, -1 - sum(na_mask), -1)
    
    print(f"Clustering complete. Found {len(clusters)} clusters for {len(valid_smiles)} unique SMILES.")
    return df

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

def optimized_balanced_group_split(df, group_col, n_folds, n_iter=20, random_state=42):
    """
    Distributes clusters (groups) into folds to minimize size variance.
    """
    group_sizes = df.groupby(group_col).size()
    groups = list(group_sizes.index)
    
    best_assignment = None
    best_score = float('inf')
    
    if not groups:
        return {}

    print(f"  > Optimizing Class 0 balance over {n_iter} iterations...")

    for i in range(n_iter):
        rng = np.random.RandomState(random_state + i)
        rng.shuffle(groups)
        
        sorted_groups = sorted(groups, key=lambda g: group_sizes[g], reverse=True)
        
        fold_assignment = {}
        counts_per_fold = [0] * n_folds
        
        for g in sorted_groups:
            size = group_sizes[g]
            min_val = min(counts_per_fold)
            candidates = [idx for idx, c in enumerate(counts_per_fold) if c == min_val]
            best_fold = rng.choice(candidates)
            
            fold_assignment[g] = best_fold
            counts_per_fold[best_fold] += size
            
        diff = max(counts_per_fold) - min(counts_per_fold)
        
        if diff < best_score:
            best_score = diff
            best_assignment = fold_assignment.copy()
            
    print(f"  > Best split found (Imbalance: {best_score} items)")
    return best_assignment

def get_context_splits(df, cv_fold, test_frac_adjusted, y_col, group_col='smiles', random_state=42):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), []
    group_repr = df.groupby(group_col)[y_col].mean().reset_index()
    
    try:
        group_repr['strat_label'] = pd.qcut(group_repr[y_col], q=5, labels=False, duplicates='drop')
    except:
        group_repr['strat_label'] = pd.cut(group_repr[y_col], bins=5, labels=False)

    counts = group_repr['strat_label'].value_counts()
    small_classes = counts[counts < cv_fold].index.tolist()
    
    forced_train_lipids = group_repr[group_repr['strat_label'].isin(small_classes)][group_col].tolist()
    strat_pool = group_repr[~group_repr['strat_label'].isin(small_classes)].copy()

    cv_ids, test_ids = train_test_split(
        strat_pool[group_col], test_size=test_frac_adjusted, 
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

    return test_df, cv_pool_df, cv_folds


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
    """
    Calculates sample weights based on inverse density (GKDE) of the target variable.
    Applied in-place to the 'Sample_weight' column (multiplicative if exists).
    """
    mask = ~all_df[target_col].isna() & ~np.isinf(all_df[target_col])
    y = all_df.loc[mask, target_col].to_numpy(dtype=float)

    if len(y) < 2:
        return all_df

    y_std = np.std(y)

    # Reflection for boundary correction
    upper_mask = (upper_bound - y) < (reflect_frac * y_std)
    y_upper = 2 * upper_bound - y[upper_mask]

    lower_mask = (y - lower_bound) < (reflect_frac * y_std)
    y_lower = 2 * lower_bound - y[lower_mask]

    y_combined = np.concatenate([y, y_upper, y_lower])

    try:
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
            print(f"  N samples       : {len(weights)}")
            print(f"  ESS             : {ess:.1f} ({ess / len(weights):.2%} of N)")
            print(f"  Boost factor    : {boost_factor:.2f}x")

        # Map back to DataFrame
        kde_series = pd.Series(weights, index=all_df.loc[mask].index)
        
        # Multiply existing weights (or 1.0 if new)
        if 'Sample_weight' not in all_df.columns:
            all_df['Sample_weight'] = 1.0
            
        # We align the calculated weights to the dataframe index
        kde_multipliers = all_df.index.map(kde_series).fillna(1.0)
        all_df['Sample_weight'] *= kde_multipliers

    except Exception as e:
        print(f"Warning: GKDE Weighting failed ({e}). Returning default weights.")
        if 'Sample_weight' not in all_df.columns:
            all_df['Sample_weight'] = 1.0

    return all_df



def cv_split_tox_butina(split_spec_fname, mode_flag, path_to_folders='../data',
                        cv_fold=5, ultra_held_out_fraction=-1.0,
                        test_frac=0.2, random_state=42, 
                        y_target_col='quantified_toxicity',
                        smiles_col='smiles',
                        butina_cutoff=0.2):
    
    print("\n=== Running Hybrid TOX Split (Anti-Leakage Grouped Mode) ===")
    
    # 1. Load Data
    all_df = pd.read_csv(os.path.join(path_to_folders, 'all_data.csv'))
    
    if 'toxicity_class' not in all_df.columns:
        raise ValueError("Error: 'toxicity_class' column not found in all_data.csv")

    all_df = all_df.dropna(subset=[y_target_col, 'toxicity_class']).copy()
    split_df = pd.read_csv(os.path.join(path_to_folders, 'crossval_split_specs', split_spec_fname))
    col_types = adjust_col_types_in_memory(pd.read_csv(os.path.join(path_to_folders, 'col_types.csv')), y_target_col)
    
    split_name = f"{split_spec_fname.replace('.csv', '')}_{mode_flag}_B"
    split_path = os.path.join(path_to_folders, 'crossval_splits', split_name)
    if ultra_held_out_fraction > 0: split_path += '_with_uho'
    path_if_none(split_path)
    path_if_none(os.path.join(split_path, 'test'))
    if ultra_held_out_fraction > 0: path_if_none(os.path.join(split_path, 'ultra_held_out'))

    # 2. Filter Pool & Perma Train
    perma_train_df, splittable_pool_df = pd.DataFrame(), pd.DataFrame()
    for _, row in split_df.iterrows():
        subset = all_df.copy()
        for dtype, val in zip(row['Data_types_for_component'].split(','), row['Values'].split(',')):
            subset = subset[subset[dtype.strip()].astype(str) == str(val.strip())]
        if row['Train_or_split'].lower() == 'train': perma_train_df = pd.concat([perma_train_df, subset])
        elif row['Train_or_split'].lower() in ['split', 'split_context']: splittable_pool_df = pd.concat([splittable_pool_df, subset])

    # 3. GROUP BY SMILES
    unique_pool = splittable_pool_df.groupby(smiles_col)['toxicity_class'].max().reset_index()
    
    # 4. SEPARATE CLASSES
    maj_unique = unique_pool[unique_pool['toxicity_class'] == 0].copy()
    min_unique = unique_pool[unique_pool['toxicity_class'].isin([1, 2])].copy()

    print(f"Unique Molecules: Class 0 (Majority) = {len(maj_unique)} | Class 1&2 (Minority) = {len(min_unique)}")

    maj_unique = assign_butina_clusters(maj_unique, smiles_col=smiles_col, cutoff=butina_cutoff)
    smiles_assignment = {}

    maj_uho_smiles, min_uho_smiles = [], []
    
    if ultra_held_out_fraction > 0:
        # Use GroupShuffleSplit for UHO to be safer on sizing
        gss_uho = GroupShuffleSplit(n_splits=1, test_size=ultra_held_out_fraction, random_state=random_state)
        for tr_idx, uh_idx in gss_uho.split(maj_unique, groups=maj_unique['cluster_id']):
            maj_uho_smiles = maj_unique.iloc[uh_idx][smiles_col].tolist()
            maj_unique = maj_unique.iloc[tr_idx].copy()

        if len(min_unique) > 1/ultra_held_out_fraction:
            tr_idx, uh_idx = train_test_split(min_unique.index, test_size=ultra_held_out_fraction, 
                                              stratify=min_unique['toxicity_class'], random_state=random_state)
            min_uho_smiles = min_unique.loc[uh_idx, smiles_col].tolist()
            min_unique = min_unique.loc[tr_idx].copy()

    for s in maj_uho_smiles + min_uho_smiles:
        smiles_assignment[s] = 'uho'

    # ==========================
    # TEST SET (Exact Sizing Fix)
    # ==========================
    n_perma_rows = len(perma_train_df)
    n_current_pool_rows = len(splittable_pool_df) 
    
    # Estimate remaining pool rows
    uho_smiles_set = set(maj_uho_smiles + min_uho_smiles)
    remaining_pool_mask = ~splittable_pool_df[smiles_col].isin(uho_smiles_set)
    n_remaining_pool_rows = splittable_pool_df[remaining_pool_mask].shape[0]

    total_data_rows = n_perma_rows + n_current_pool_rows
    target_test_rows = int(total_data_rows * test_frac)
    
    if n_remaining_pool_rows > 0:
        frac_of_pool_needed = target_test_rows / n_remaining_pool_rows
    else:
        frac_of_pool_needed = 0.0

    frac_of_pool_needed = min(frac_of_pool_needed, 0.99)

    print(f"Adjusted Test Split: Target={target_test_rows} rows. "
          f"Taking {frac_of_pool_needed*100:.2f}% of remaining pool molecules to satisfy this.")

    maj_available_count = len(maj_unique)
    min_available_count = len(min_unique)
    
    # Target Molecule Counts
    target_maj_molecules = int(maj_available_count * frac_of_pool_needed)
    target_min_molecules = int(min_available_count * frac_of_pool_needed)

    if target_min_molecules < 10: target_min_molecules = min(10, min_available_count)

    maj_test_smiles, min_test_smiles = [], []
    maj_cv_unique = maj_unique.copy()
    min_cv_unique = min_unique.copy()

    # 2. Split Minority
    if min_available_count > 0:
        if target_min_molecules == min_available_count:
            min_test_smiles = min_unique[smiles_col].tolist()
            min_cv_unique = pd.DataFrame(columns=min_unique.columns)
        else:
            tr_idx, ts_idx = train_test_split(min_unique.index, 
                                              test_size=target_min_molecules, 
                                              stratify=min_unique['toxicity_class'], 
                                              random_state=random_state)
            min_test_smiles = min_unique.loc[ts_idx, smiles_col].tolist()
            min_cv_unique = min_unique.loc[tr_idx].copy()

    # 3. Split Majority (Using GroupShuffleSplit for better sizing)
    if maj_available_count > 0:
        target_maj_frac = target_maj_molecules / maj_available_count
        target_maj_frac = min(target_maj_frac, 0.99)
        
        if target_maj_frac > 0:
            # Replaces manual cluster shuffling with GroupShuffleSplit which handles float ratios better
            gss_maj = GroupShuffleSplit(n_splits=1, test_size=target_maj_frac, random_state=random_state)
            for keep_idx, test_idx in gss_maj.split(maj_unique, groups=maj_unique['cluster_id']):
                maj_test_smiles = maj_unique.iloc[test_idx][smiles_col].tolist()
                maj_cv_unique = maj_unique.iloc[keep_idx].copy()

    for s in maj_test_smiles + min_test_smiles:
        smiles_assignment[s] = 'test'

    # ==========================
    # CV FOLDS ASSIGNMENT
    # ==========================
    maj_fold_map_clusters = optimized_balanced_group_split(maj_cv_unique, group_col='cluster_id', n_folds=cv_fold, random_state=random_state)
    
    for _, row in maj_cv_unique.iterrows():
        s = row[smiles_col]
        c_id = row['cluster_id']
        f_id = maj_fold_map_clusters.get(c_id, 0)
        smiles_assignment[s] = f'cv_val_{f_id}'

    skf = StratifiedKFold(n_splits=cv_fold, shuffle=True, random_state=random_state)
    min_cv_unique = min_cv_unique.reset_index(drop=True)
    if not min_cv_unique.empty:
        for fold_idx, (_, val_idx) in enumerate(skf.split(min_cv_unique, min_cv_unique['toxicity_class'])):
            val_smiles = min_cv_unique.loc[val_idx, smiles_col].tolist()
            for s in val_smiles:
                smiles_assignment[s] = f'cv_val_{fold_idx}'

    # ==========================
    # CONSTRUCT FINAL DATAFRAMES
    # ==========================
    splittable_pool_df['split_assignment'] = splittable_pool_df[smiles_col].map(smiles_assignment)
    
    if ultra_held_out_fraction > 0:
        uho_df = splittable_pool_df[splittable_pool_df['split_assignment'] == 'uho'].copy()
        y, x, w, m = split_df_by_col_type(uho_df, col_types)
        yxwm_to_csvs(y, x, w, m, os.path.join(split_path, 'ultra_held_out'), 'test')

    test_df = splittable_pool_df[splittable_pool_df['split_assignment'] == 'test'].copy()
    y, x, w, m = split_df_by_col_type(test_df, col_types)
    yxwm_to_csvs(y, x, w, m, os.path.join(split_path, 'test'), 'test')

    for i in range(cv_fold):
        path_if_none(os.path.join(split_path, f'cv_{i}'))
        val_df = splittable_pool_df[splittable_pool_df['split_assignment'] == f'cv_val_{i}'].copy()
        train_mask = (
            (splittable_pool_df['split_assignment'].str.startswith('cv_val_')) & 
            (splittable_pool_df['split_assignment'] != f'cv_val_{i}')
        )
        train_pool_df = splittable_pool_df[train_mask].copy()
        full_train_df = pd.concat([perma_train_df, train_pool_df], ignore_index=True)

        cols_to_drop = ['split_assignment', 'cluster_id']
        full_train_df.drop(columns=cols_to_drop, errors='ignore', inplace=True)
        val_df.drop(columns=cols_to_drop, errors='ignore', inplace=True)

        full_train_df = generate_weights_gkde(full_train_df, target_col=y_target_col)
        for df, name in [(full_train_df, 'train'), (val_df, 'valid')]:
            y_f, x_f, w_f, m_f = split_df_by_col_type(df, col_types)
            yxwm_to_csvs(y_f, x_f, w_f, m_f, os.path.join(split_path, f'cv_{i}'), name)

    print(f"Hybrid Split Complete at {split_path}")


def optimize_group_split(df, group_col, stratify_col, target_fraction, n_trials=50, random_state=42):
    """
    Optimizes the split to match target size AND distribution using Monte Carlo trials.
    """
    best_train_idx, best_test_idx = None, None
    best_score = float('inf')
    
    ideal_dist = df[stratify_col].value_counts(normalize=True).sort_index()
    target_size = int(len(df) * target_fraction)
    if target_size == 0: target_size = 1
    
    print(f"   > Optimizing split... Target Rows: {target_size}")

    for i in range(n_trials):
        seed = random_state + i
        gss = GroupShuffleSplit(n_splits=1, test_size=target_fraction, random_state=seed)
        
        for train_idx, test_idx in gss.split(df, df[stratify_col], df[group_col]):
            current_size = len(test_idx)
            if current_size == 0: continue
            
            # 1. Size Penalty
            size_error = abs(current_size - target_size) / target_size
            
            # 2. Distribution Penalty
            test_subset = df.iloc[test_idx]
            test_dist = test_subset[stratify_col].value_counts(normalize=True).sort_index()
            combined_dist = pd.DataFrame({'ideal': ideal_dist, 'test': test_dist}).fillna(0)
            dist_error = np.sum((combined_dist['ideal'] - combined_dist['test']) ** 2)
            
            # Weighted Score (Distribution is priority)
            total_penalty = size_error + (10 * dist_error)
            
            if total_penalty < best_score:
                best_score = total_penalty
                best_train_idx, best_test_idx = train_idx, test_idx
                
    return best_train_idx, best_test_idx

def cv_split_butina(split_spec_fname, mode_flag, path_to_folders='../data',
                    cv_fold=5,                # <--- 5 Folds = ~16.5% Val (Balanced with Test)
                    test_frac=0.175,          # <--- 17.5% Test Set
                    ultra_held_out_fraction=-1.0,
                    random_state=42, 
                    y_target_col='quantified_toxicity',
                    smiles_col='smiles',
                    butina_cutoff=0.32): 
    
    # 1. Load Data
    all_df = pd.read_csv(os.path.join(path_to_folders, 'all_data.csv'))
    all_df = all_df.dropna(subset=[y_target_col]).copy()
    
    split_df = pd.read_csv(os.path.join(path_to_folders, 'crossval_split_specs', split_spec_fname))
    col_types = adjust_col_types_in_memory(pd.read_csv(os.path.join(path_to_folders, 'col_types.csv')), y_target_col)

    split_name = f"{split_spec_fname.replace('.csv', '')}_{mode_flag}_B"
    split_path = os.path.join(path_to_folders, 'crossval_splits', split_name)
    if ultra_held_out_fraction > 0: split_path += '_with_uho'
    
    path_if_none(split_path)
    path_if_none(os.path.join(split_path, 'test'))
    if ultra_held_out_fraction > 0: path_if_none(os.path.join(split_path, 'ultra_held_out'))

    # 2. Separate Data
    perma_train, splittable_pool = pd.DataFrame(), pd.DataFrame() 
    for _, row in split_df.iterrows():
        subset = all_df.copy()
        for dtype, val in zip(row['Data_types_for_component'].split(','), row['Values'].split(',')):
            subset = subset[subset[dtype.strip()].astype(str) == str(val.strip())]
        
        if row['Train_or_split'].lower() == 'train': perma_train = pd.concat([perma_train, subset])
        elif row['Train_or_split'].lower() in ['split', 'split_context']: splittable_pool = pd.concat([splittable_pool, subset])

    # 3. Butina & Binning
    splittable_pool = assign_butina_clusters(splittable_pool, smiles_col=smiles_col, cutoff=butina_cutoff)
    
    if mode_flag == 'tox': split_thresholds = [0.7, 0.8]
    else: split_thresholds = [splittable_pool[y_target_col].quantile(0.33), splittable_pool[y_target_col].quantile(0.66)]
    
    splittable_pool['custom_bin'] = splittable_pool[y_target_col].apply(lambda x: get_bin_label(x, split_thresholds))

    # --- Print Strategy ---
    total_data = len(splittable_pool) + len(perma_train)
    remainder = 1.0 - test_frac
    val_percent = remainder / cv_fold
    train_percent = 1.0 - test_frac - val_percent
    
    print(f"--- SPLIT STRATEGY ---")
    print(f"Total Rows: {total_data}")
    print(f"Target Test: {test_frac*100}%")
    print(f"Target Val:  ~{val_percent*100:.1f}% ({cv_fold}-Fold CV)")
    print(f"Target Train: ~{train_percent*100:.1f}%")
    print(f"----------------------")

    # 4. Ultra Held Out (Optional)
    uho_df = pd.DataFrame()
    if ultra_held_out_fraction > 0:
        train_idx, uho_idx = optimize_group_split(
            splittable_pool, 'cluster_id', 'custom_bin', ultra_held_out_fraction, n_trials=20, random_state=random_state
        )
        uho_df = splittable_pool.iloc[uho_idx].copy()
        splittable_pool = splittable_pool.iloc[train_idx].copy()
        y, x, w, m = split_df_by_col_type(uho_df, col_types)
        yxwm_to_csvs(y, x, w, m, os.path.join(split_path, 'ultra_held_out'), 'test')

    # 5. Optimized Test Split
    total_relevant_data = len(perma_train) + len(splittable_pool)
    target_test_size = int(total_relevant_data * test_frac)
    pool_size = len(splittable_pool)

    if pool_size > 0:
        fraction_of_pool_needed = target_test_size / pool_size
        fraction_of_pool_needed = min(fraction_of_pool_needed, 0.95)
    else:
        fraction_of_pool_needed = 0.0
    
    print(f"Extracting {target_test_size} rows for Test Set ({fraction_of_pool_needed*100:.1f}% of pool)")
    
    keep_idx, test_idx = optimize_group_split(
        splittable_pool, 
        group_col='cluster_id', 
        stratify_col='custom_bin', 
        target_fraction=fraction_of_pool_needed, 
        n_trials=50, 
        random_state=random_state
    )

    test_df = splittable_pool.iloc[test_idx].copy()
    cv_pool = splittable_pool.iloc[keep_idx].copy()
    
    if not test_df.empty:
        y, x, w, m = split_df_by_col_type(test_df, col_types)
        yxwm_to_csvs(y, x, w, m, os.path.join(split_path, 'test'), 'test')

    # 6. CV Split
    cv_pool['custom_bin'] = cv_pool[y_target_col].apply(lambda x: get_bin_label(x, split_thresholds))
    
    sgkf = StratifiedGroupKFold(n_splits=cv_fold, shuffle=True, random_state=random_state)
    
    fold_idx = 0
    for train_idxs, val_idxs in sgkf.split(cv_pool, cv_pool['custom_bin'], cv_pool['cluster_id']):
        path_if_none(os.path.join(split_path, f'cv_{fold_idx}'))
        
        fold_train = cv_pool.iloc[train_idxs]
        fold_val = cv_pool.iloc[val_idxs]
        final_train = pd.concat([perma_train, fold_train], ignore_index=True)
        final_val = fold_val.copy()
        
        if 'custom_bin' in final_train.columns: final_train = final_train.drop(columns=['custom_bin'])
        if 'custom_bin' in final_val.columns: final_val = final_val.drop(columns=['custom_bin'])
        
        for df, name in [(final_train, 'train'), (final_val, 'valid')]:
            y_f, x_f, w_f, m_f = split_df_by_col_type(df, col_types)
            yxwm_to_csvs(y_f, x_f, w_f, m_f, os.path.join(split_path, f'cv_{fold_idx}'), name)
            
        print(f"  > Fold {fold_idx}: Train={len(final_train)}, Val={len(final_val)}")
        fold_idx += 1

    print(f"Full Butina Split Complete at {split_path}")

def cv_split_stratified(split_spec_fname, mode_flag, path_to_folders='../data',
                       cv_fold=5, ultra_held_out_fraction=-1.0,
                       test_frac=0.2, random_state=42, 
                       y_target_col='quantified_toxicity'): 
     
    all_df = pd.read_csv(os.path.join(path_to_folders, 'all_data.csv'))
    all_df = all_df.dropna(subset=[y_target_col]).copy()

    split_df = pd.read_csv(os.path.join(path_to_folders, 'crossval_split_specs', split_spec_fname))
    col_types = adjust_col_types_in_memory(pd.read_csv(os.path.join(path_to_folders, 'col_types.csv')), y_target_col)

    split_name = f"{split_spec_fname.replace('.csv', '')}_{mode_flag}_S"
    split_path = os.path.join(path_to_folders, 'crossval_splits', split_name)
    if ultra_held_out_fraction > 0: split_path += '_with_uho'
    
    path_if_none(split_path)
    path_if_none(os.path.join(split_path, 'test'))
    if ultra_held_out_fraction > 0: path_if_none(os.path.join(split_path, 'ultra_held_out'))

    perma_train, standard_pool, context_pool = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for _, row in split_df.iterrows():
        subset = all_df.copy()
        for dtype, val in zip(row['Data_types_for_component'].split(','), row['Values'].split(',')):
            subset = subset[subset[dtype.strip()].astype(str) == str(val.strip())]
        
        if row['Train_or_split'].lower() == 'train': perma_train = pd.concat([perma_train, subset])
        elif row['Train_or_split'].lower() == 'split': standard_pool = pd.concat([standard_pool, subset])
        elif row['Train_or_split'].lower() == 'split_context': context_pool = pd.concat([context_pool, subset])

    def add_strat_bins(df, target_col, n_bins=5):
        if df.empty: return df
        df = df.dropna(subset=[target_col]).copy()
        try: df['strat_bin'] = pd.qcut(df[target_col], q=n_bins, labels=False, duplicates='drop')
        except: df['strat_bin'] = pd.cut(df[target_col], bins=n_bins, labels=False)
        return df

    # --- UHO LOGIC ---
    std_uho, ctx_uho = pd.DataFrame(), pd.DataFrame()
    
    if ultra_held_out_fraction > 0:
        if not standard_pool.empty:
            standard_pool = add_strat_bins(standard_pool, y_target_col)
            standard_pool, std_uho = train_test_split(standard_pool, test_size=ultra_held_out_fraction, 
                                                      random_state=random_state, stratify=standard_pool['strat_bin'])
            if 'strat_bin' in std_uho.columns: std_uho.drop(columns=['strat_bin'], inplace=True)
            if 'strat_bin' in standard_pool.columns: standard_pool.drop(columns=['strat_bin'], inplace=True)
            
        if not context_pool.empty:
            context_pool = add_strat_bins(context_pool, y_target_col)
            context_pool, ctx_uho = train_test_split(context_pool, test_size=ultra_held_out_fraction, 
                                                     random_state=random_state, stratify=context_pool['strat_bin'])
            if 'strat_bin' in ctx_uho.columns: ctx_uho.drop(columns=['strat_bin'], inplace=True)
            if 'strat_bin' in context_pool.columns: context_pool.drop(columns=['strat_bin'], inplace=True)

    final_uho = pd.concat([std_uho, ctx_uho], ignore_index=True)
    if not final_uho.empty:
        y, x, w, m = split_df_by_col_type(final_uho, col_types)
        yxwm_to_csvs(y, x, w, m, os.path.join(split_path, 'ultra_held_out'), 'test')

    # --- UPDATED TEST SPLIT LOGIC ---
    total_data_count = len(perma_train) + len(standard_pool) + len(context_pool) + len(final_uho)
    target_test_count = int(total_data_count * test_frac)
    
    remaining_pool_count = len(standard_pool) + len(context_pool)
    
    if remaining_pool_count > 0:
        fraction_of_pool_needed = target_test_count / remaining_pool_count
        fraction_of_pool_needed = min(fraction_of_pool_needed, 0.99)
    else:
        fraction_of_pool_needed = 0.0
        
    print(f"Adjusted Test Split: Target={target_test_count} rows. "
          f"Taking {fraction_of_pool_needed*100:.2f}% of remaining pool to satisfy this.")

    ctx_test, ctx_cv_pool, ctx_folds = get_context_splits(context_pool, cv_fold, fraction_of_pool_needed, y_target_col)
    
    std_test, std_train_val = pd.DataFrame(), pd.DataFrame()
    if not standard_pool.empty and fraction_of_pool_needed > 0:
        standard_pool = add_strat_bins(standard_pool, y_target_col)
        std_train_val, std_test = train_test_split(standard_pool, test_size=fraction_of_pool_needed, 
                                                   random_state=random_state, stratify=standard_pool['strat_bin'])
        if 'strat_bin' in std_test.columns: std_test.drop(columns=['strat_bin'], inplace=True)
        if 'strat_bin' in std_train_val.columns: std_train_val.drop(columns=['strat_bin'], inplace=True)
    elif not standard_pool.empty:
        std_train_val = standard_pool.copy()

    final_test = pd.concat([std_test, ctx_test], ignore_index=True)
    y, x, w, m = split_df_by_col_type(final_test, col_types)
    yxwm_to_csvs(y, x, w, m, os.path.join(split_path, 'test'), 'test')

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
        
        # if y_target_col == "quantified_toxicity":
        #     f_train = generate_weights_gkde(f_train, target_col=y_target_col)
        
        f_train = generate_weights_gkde(f_train, target_col=y_target_col)

        for df, name in [(f_train, 'train'), (f_val, 'valid')]:
            y_f, x_f, w_f, m_f = split_df_by_col_type(df, col_types)
            yxwm_to_csvs(y_f, x_f, w_f, m_f, os.path.join(split_path, f'cv_{i}'), name)

    print(f"Full Stratified Split Complete at {split_path}")

# ==========================================
#      MAIN
# ==========================================

def main(argv):
    if len(argv) < 4:
        print("Usage: python split.py <split_spec> <mode: del or tox> <method: B or S> [--cv <folds>]")
        sys.exit(1)

    split = argv[1]
    mode_arg = argv[2].lower()
    method_arg = argv[3].upper()
    
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
    if len(argv) > 4:
        for i, arg in enumerate(argv):
            if arg.replace('–', '-') == '--cv':
                cv_num = int(argv[i+1])
                print(f'Using {cv_num} folds')

    if mode_flag == 'tox' and method_arg == 'B':
        cv_split_tox_butina(split, mode_flag=mode_flag, cv_fold=cv_num, y_target_col=target_col)
    elif method_arg == 'B':
        cv_split_butina(split, mode_flag=mode_flag, cv_fold=cv_num, y_target_col=target_col)
    elif method_arg == 'S':
        cv_split_stratified(split, mode_flag=mode_flag, cv_fold=cv_num, y_target_col=target_col)
    else:
        print("Error: Unknown split method. Use 'B' for Butina or 'S' for Stratified.")

if __name__ == '__main__':
    main(sys.argv)