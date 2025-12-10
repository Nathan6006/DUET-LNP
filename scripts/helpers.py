import os 
import pandas as pd
from chemprop import data 
from rdkit import Chem
import numpy as np
from rdkit.Chem import rdFingerprintGenerator, DataStructs
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

def generate_scaffolds(df, smiles_col='SMILES'):
    """Generates Murcko Scaffolds for a dataframe."""
    scaffolds = defaultdict(list)
    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row[smiles_col])
        if mol:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            scaffolds[scaffold].append(idx)
        else:
            # Handle invalid SMILES by grouping them together or dropping
            scaffolds['INVALID'].append(idx)
    
    # Sort scaffolds by size (largest groups first) to ensure balanced filling
    scaffold_sets = sorted(scaffolds.values(), key=lambda x: len(x), reverse=True)
    return scaffold_sets

def perform_scaffold_split(df, smiles_col, n_splits=5, test_frac=0.1, random_state=42):
    """
    Splits data based on scaffolds. 
    1. separates a Test set.
    2. Returns a KFold-like iterator for the Train/Valid set.
    """
    scaffold_sets = generate_scaffolds(df, smiles_col)
    
    # Flatten structure slightly to manage allocation
    # We simply want to assign every index to a fold [0, 1, ... n_splits]
    # Fold -1 will be the Test set
    
    total_len = len(df)
    test_size = int(total_len * test_frac)
    train_valid_size = total_len - test_size
    
    # Buckets: Index -1 is Test, 0 to n_splits-1 are CV folds
    fold_indices = defaultdict(list)
    fold_sizes = defaultdict(int)
    
    # Greedy implementation: Assign largest scaffolds first to the emptiest buckets
    # We reserve a specific bucket for 'test' to ensure it's chemically distinct
    
    for group_indices in scaffold_sets:
        # Check if test set is full
        if fold_sizes[-1] < test_size:
            fold_indices[-1].extend(group_indices)
            fold_sizes[-1] += len(group_indices)
        else:
            # Assign to the smallest CV fold to keep them balanced
            # We look at folds 0 to n_splits-1
            smallest_fold = min(range(n_splits), key=lambda k: fold_sizes[k])
            fold_indices[smallest_fold].extend(group_indices)
            fold_sizes[smallest_fold] += len(group_indices)

    # 1. Create Train/Valid DF and Test DF
    test_idx = fold_indices[-1]
    cv_idx = []
    for i in range(n_splits):
        cv_idx.extend(fold_indices[i])
        
    test_df = df.loc[test_idx]
    train_valid_df = df.loc[cv_idx].reset_index(drop=True)
    
    # 2. Create the iterator for K-Fold
    # We must re-map the original dataframe indices to the new reset_index of train_valid_df
    # This is complex, so we simply re-run the scaffold group assignment on the subset
    # OR, better yet, we pre-calculated the folds above.
    
    # Let's organize the pre-calculated folds for the iterator
    # We need to map the global indices in fold_indices to the new 0..N indices of train_valid_df
    old_to_new_map = {old_idx: new_idx for new_idx, old_idx in enumerate(cv_idx)}
    
    custom_cv_folds = []
    for i in range(n_splits):
        val_fold_indices = [old_to_new_map[idx] for idx in fold_indices[i]]
        # Train indices are everything in train_valid that isn't in valid
        all_indices = set(range(len(train_valid_df)))
        train_fold_indices = list(all_indices - set(val_fold_indices))
        custom_cv_folds.append((train_fold_indices, val_fold_indices))
        
    return train_valid_df, test_df, custom_cv_folds


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

def change_column_order(path, all_df, first_cols = ['smiles','quantified_toxicity','unnormalized_toxicity']):
    other_cols = [col for col in all_df.columns if col not in first_cols]
    all_df = all_df[first_cols + other_cols]
    all_df.to_csv(path, index=False)



def load_datapoints_tox_only(smiles_csv, extra_csv, smiles_column='smiles', target_columns = ["quantified_toxicity"]):
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

def load_datapoints_rf(smiles_csv, extra_csv, smiles_column='smiles',
                       target_columns=["quantified_toxicity"]):
    df_smi = pd.read_csv(smiles_csv)
    df_extra = pd.read_csv(extra_csv)

    smis = df_smi[smiles_column].values
    ys = df_smi[target_columns].values
    extra_features = df_extra.to_numpy(dtype=float)

    datapoints = []
    for smi, y, xf in zip(smis, ys, extra_features):
        datapoints.append({
            "smiles": smi,
            "y": y,
            "x_d": xf
        })
    return datapoints


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
