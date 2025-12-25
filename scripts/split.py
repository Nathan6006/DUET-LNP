import os
import pandas as pd
import sys
import numpy as np
import random
from sklearn.model_selection import train_test_split, StratifiedKFold
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
from helpers import path_if_none
from sklearn.cluster import AgglomerativeClustering

class LogicalUnit:
    def __init__(self, smiles, indices, labels):
        self.smiles = smiles
        self.indices = indices  # List of original DF indices
        self.size = len(indices)
        # Rule: The group is treated as the class of the LOWEST value in the group
        self.effective_class = min(labels) 
        self.original_labels = labels

    def __repr__(self):
        return f"Unit(Cls={self.effective_class}, Sz={self.size})"

def group_into_logical_units(df, smiles_col, y_col):
    """
    Groups dataframe rows by SMILES. 
    Returns a list of LogicalUnit objects.
    """
    units = []
    # Group by SMILES to find identicals
    grouped = df.groupby(smiles_col)
    
    for s, group in grouped:
        indices = group.index.tolist()
        labels = group[y_col].tolist()
        units.append(LogicalUnit(s, indices, labels))
        
    return units

# ==========================================
# 3. CORE STRATIFICATION LOGIC
# ==========================================

def get_fingerprints_and_matrix(units):
    """
    Computes distance matrix for Logical Units based on their SMILES.
    """
    mols = [Chem.MolFromSmiles(u.smiles) for u in units]
    # Filter out invalid mols (though should be rare if data is clean)
    valid_data = [(i, m) for i, m in enumerate(mols) if m is not None]
    
    # Map back which unit corresponds to which valid mol
    valid_indices = [x[0] for x in valid_data]
    valid_mols = [x[1] for x in valid_data]
    
    fpgen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)
    fps = [fpgen.GetFingerprint(m) for m in valid_mols]
    
    n_fps = len(fps)
    dist_matrix = np.zeros((n_fps, n_fps))
    
    # Calculate Distances
    for i in range(1, n_fps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        for j, sim in enumerate(sims):
            dist = 1.0 - sim
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist 
            
    return valid_indices, dist_matrix

def shatter_cluster_into_units(cluster_obj, units_map):
    """
    Breaks a Cluster (group of similar Units) back down into individual Logical Units.
    We CANNOT split a Logical Unit, so this is the finest granularity we have.
    """
    unit_indices = cluster_obj['unit_indices']
    broken_pieces = []
    
    for u_idx in unit_indices:
        unit = units_map[u_idx]
        broken_pieces.append({
            'type': 'unit', # Mark as a single indivisible unit
            'unit_indices': [u_idx],
            'counts': {unit.effective_class: unit.size},
            'total': unit.size
        })
    return broken_pieces

def perform_grouped_stratification(df, smiles_col, y_col, 
                                   cv_folds=5, test_frac=0.175, uho_frac=0.0):
    
    if df.empty: return df
    print(f"-> Grouping Identical Contexts...")
    
    # 1. Create Indivisible Units
    all_units = group_into_logical_units(df, smiles_col, y_col)
    print(f"   Reduced {len(df)} rows to {len(all_units)} unique structural units.")

    # 2. Compute Clustering on Units
    valid_ptr, dist_matrix = get_fingerprints_and_matrix(all_units)
    
    # Map matrix indices back to all_units list indices
    # valid_ptr[matrix_index] = all_units_index
    
    print("   Clustering Units...")
    lt_dists = []
    n_fps = len(valid_ptr)
    for i in range(1, n_fps):
        lt_dists.extend(dist_matrix[i, :i].tolist())
    
    raw_clusters = Butina.ClusterData(lt_dists, n_fps, 0.4, isDistData=True)
    
    # 3. Build Pool of Clusters
    pool = []
    for c_idx, member_matrix_indices in enumerate(raw_clusters):
        # Convert matrix indices -> unit list indices
        unit_indices = [valid_ptr[mi] for mi in member_matrix_indices]
        
        c_counts = {}
        total_size = 0
        
        for u_idx in unit_indices:
            u = all_units[u_idx]
            c_counts[u.effective_class] = c_counts.get(u.effective_class, 0) + u.size
            total_size += u.size
            
        pool.append({
            'type': 'cluster',
            'id': f"c_{c_idx}",
            'unit_indices': unit_indices,
            'counts': c_counts,
            'total': total_size
        })

    # 4. Define Requirements
    # Note: We are now counting "effective class" sizes
    test_min = {0: 100, 1: 10, 2: 10}
    cv_min   = {0: 100, 1: 7, 2: 10}
    classes = sorted(list(set(df[y_col])))

    bins = []
    # Test Bin
    bins.append({
        'name': 'test',
        'min': test_min,
        'current': {k: 0 for k in classes},
        'items': [] # Stores unit indices
    })
    
    # CV Bins
    cv_bins = []
    for i in range(cv_folds):
        cv_bins.append({
            'name': f'cv_{i}',
            'min': cv_min,
            'current': {k: 0 for k in classes},
            'items': []
        })
    bins.extend(cv_bins)

    final_assignments = {} # smiles -> bin_name

    # ==========================================
    # PHASE 1: TEST SET (Strict Requirement)
    # ==========================================
    print("   Phase 1: Filling Test Set...")
    test_bin = bins[0]
    
    for p_class in [1, 2, 0]: # Rare first
        while test_bin['current'].get(p_class, 0) < test_bin['min'].get(p_class, 0):
            needed = test_bin['min'][p_class] - test_bin['current'][p_class]
            
            # Find best cluster/unit
            best_idx = -1
            best_score = -1
            
            for i, item in enumerate(pool):
                cnt = item['counts'].get(p_class, 0)
                if cnt > 0:
                    # Density heuristic
                    score = cnt / item['total']
                    if score > best_score:
                        best_score = score
                        best_idx = i
            
            if best_idx == -1:
                print(f"      WARNING: Ran out of Class {p_class} for Test Set!")
                break
            
            cand = pool.pop(best_idx)
            
            # Check for massive overflow/waste
            # If cand is a single Unit, we MUST take it or leave it (indivisible)
            wasteful = cand['counts'][p_class] > (needed + 5)
            too_heavy = cand['total'] > (needed + 10)
            
            if (wasteful or too_heavy) and cand['type'] == 'cluster':
                # Shatter Cluster -> Logical Units
                frags = shatter_cluster_into_units(cand, all_units)
                pool.extend(frags)
                # Restart search with smaller pieces
            else:
                # Assign
                test_bin['items'].extend(cand['unit_indices'])
                for k, v in cand['counts'].items(): test_bin['current'][k] += v

    # ==========================================
    # PHASE 2: CV MINIMUMS
    # ==========================================
    print("   Phase 2: Filling CV Minimums...")
    
    for p_class in [1, 2, 0]:
        while True:
            # Who needs it?
            neediest_bin = None
            max_deficit = 0
            for b in cv_bins:
                defic = b['min'].get(p_class, 0) - b['current'].get(p_class, 0)
                if defic > max_deficit:
                    max_deficit = defic
                    neediest_bin = b
            
            if neediest_bin is None: break 
            
            # Find candidate
            best_idx = -1
            best_score = -1
            for i, item in enumerate(pool):
                cnt = item['counts'].get(p_class, 0)
                if cnt > 0:
                    score = cnt / item['total']
                    if score > best_score:
                        best_score = score
                        best_idx = i
            
            if best_idx == -1: break
            
            cand = pool.pop(best_idx)
            needed = neediest_bin['min'][p_class] - neediest_bin['current'][p_class]
            
            # Fit Check
            wasteful = cand['counts'][p_class] > (needed + 5)
            too_heavy = cand['total'] > (needed + 20)
            
            if (wasteful or too_heavy) and cand['type'] == 'cluster':
                frags = shatter_cluster_into_units(cand, all_units)
                pool.extend(frags)
            else:
                neediest_bin['items'].extend(cand['unit_indices'])
                for k, v in cand['counts'].items(): neediest_bin['current'][k] += v

    # ==========================================
    # PHASE 3: CLEANUP (Force Assign All Remaining)
    # ==========================================
    print("   Phase 3: Distributing Remaining Units...")
    
    # Sort remaining pool by size (descending) to place large rocks first
    pool.sort(key=lambda x: x['total'], reverse=True)
    
    while pool:
        cand = pool.pop(0)
        
        # Assign to the CV fold that is smallest (by total count)
        # We exclude Test set from this phase to prevent over-stuffing it
        target_bin = sorted(cv_bins, key=lambda b: sum(b['current'].values()))[0]
        
        target_bin['items'].extend(cand['unit_indices'])
        for k, v in cand['counts'].items():
            target_bin['current'][k] += v

    # ==========================================
    # MAPPING BACK TO DATAFRAME
    # ==========================================
    # Assign bin names to SMILES
    smiles_to_bin = {}
    for b in bins:
        for u_idx in b['items']:
            u = all_units[u_idx]
            smiles_to_bin[u.smiles] = b['name']
            
    df['split_group'] = df[smiles_col].map(smiles_to_bin).fillna('skipped')
    
    print("\n  Final Stratified Report (Groups + Cleanup):")
    print(f"    {'BIN':<10} | {'ACTUAL (Rows)':<30} | {'MIN REQUIRED':<25}")
    print("-" * 75)
    for b in bins:
        print(f"    {b['name']:<10} | {b['current']} | {b['min']}")

    return df

# ==========================================
# 4. MAIN WRAPPER
# ==========================================

def cv_split_butina(split_spec_fname, path_to_folders='../data',
                    cv_fold=5, ultra_held_out_fraction=-1.0,
                    test_frac=0.175, random_state=42, 
                    y_stratify_col='toxicity_class'):
    
    print(f"Starting Process: {split_spec_fname}")

    # 1. Load Data
    all_df = pd.read_csv(os.path.join(path_to_folders, 'all_data.csv'))
    split_df = pd.read_csv(os.path.join(path_to_folders, 'crossval_split_specs', split_spec_fname))
    col_types = pd.read_csv(os.path.join(path_to_folders, 'col_type.csv'))

    split_name = split_spec_fname.replace('.csv', '')
    split_path = os.path.join(path_to_folders, 'crossval_splits', split_name)
    if ultra_held_out_fraction > 0: split_path += '_with_uho'
    
    os.makedirs(os.path.join(split_path, 'test'), exist_ok=True)
    if ultra_held_out_fraction > 0: 
        os.makedirs(os.path.join(split_path, 'ultra_held_out'), exist_ok=True)

    # 2. Filter & Merge
    perma_train = pd.DataFrame()
    split_pool = pd.DataFrame() 

    for _, row in split_df.iterrows():
        dtypes = [d.strip() for d in row['Data_types_for_component'].split(',')]
        vals = [v.strip() for v in row['Values'].split(',')]
        subset = all_df.copy()
        
        for dtype, val in zip(dtypes, vals):
            subset = subset[subset[dtype].astype(str) == str(val)]
        if subset.empty: continue

        # Handle 'split_context' explicitly as the pool for grouping
        role = row['Train_or_split'].lower()
        if role == 'train':
            perma_train = pd.concat([perma_train, subset])
        elif role == 'split_context' or role == 'split':
            split_pool = pd.concat([split_pool, subset])

    # 3. RUN STRATIFICATION
    if not split_pool.empty:
        split_pool = perform_grouped_stratification(
            split_pool, 
            smiles_col='smiles', 
            y_col=y_stratify_col,
            cv_folds=cv_fold, 
            test_frac=test_frac, 
            uho_frac=ultra_held_out_fraction
        )

    # 4. Save Logic
    def save_group(df_subset, save_path, set_name):
        if df_subset.empty: return
        y, x, w, m = split_df_by_col_type(df_subset, col_types)
        yxwm_to_csvs(y, x, w, m, save_path, set_name)

    # A. UHO
    if 'split_group' in split_pool.columns:
        uho = split_pool[split_pool['split_group'] == 'ultra_held_out']
        save_group(uho, os.path.join(split_path, 'ultra_held_out'), 'test')

        # B. Test
        test = split_pool[split_pool['split_group'] == 'test']
        save_group(test, os.path.join(split_path, 'test'), 'test')

        # C. CV Folds
        for i in range(cv_fold):
            fold_dir = os.path.join(split_path, f'cv_{i}')
            os.makedirs(fold_dir, exist_ok=True)
            
            # Validation
            val_df = split_pool[split_pool['split_group'] == f'cv_{i}']
            
            # Train
            train_groups = [f'cv_{j}' for j in range(cv_fold) if j != i]
            pool_train = split_pool[split_pool['split_group'].isin(train_groups)]
            
            full_train = pd.concat([perma_train, pool_train], ignore_index=True)
            
            save_group(full_train, fold_dir, 'train')
            save_group(val_df, fold_dir, 'valid')

    print(f"Done. Saved to {split_path}")




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