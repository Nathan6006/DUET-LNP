import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
import os

def get_count_fingerprints(smiles_series):
    """
    Uses Count-Based Morgan Fingerprints via the modern Generator API.
    
    - includeChirality=True: Distinguishes stereoisomers (Cis vs Trans).
    - GetCountFingerprint: Returns counts (SparseIntVect) rather than bits.
      This is crucial for lipids to distinguish chain lengths (e.g. C12 vs C14).
    """
    # Instantiate the modern generator once
    # Radius=2 is standard (equivalent to ECFP4)
    mfgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, 
        includeChirality=True
    )

    fps = []
    valid_indices = []

    for idx, smile in enumerate(smiles_series):
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            # Generate Count-Based Fingerprint (SparseIntVect)
            # This captures the frequency of substructures, not just presence/absence
            fp = mfgen.GetCountFingerprint(mol)
            fps.append(fp)
            valid_indices.append(idx)

    return fps

def calculate_similarity_metrics(query_fps, reference_fps):
    """
    Calculates Tanimoto similarity on Count Vectors.
    RDKit handles SparseIntVect Tanimoto internally (Intersection / Union).
    """
    max_sims = []
    mean_sims = []
    
    for q_fp in query_fps:
        # Calculate bulk similarity (1 query vs all references)
        sims = DataStructs.BulkTanimotoSimilarity(q_fp, reference_fps)
        
        if sims:
            max_sims.append(max(sims))      # Nearest neighbor score
            mean_sims.append(np.mean(sims)) # Average similarity
        else:
            max_sims.append(0.0)
            mean_sims.append(0.0)

    return max_sims, mean_sims

def find_duplicates(test_smiles, train_smiles):
    """
    Strict duplicate check using Canonical SMILES strings.
    """
    train_canon = set()
    for s in train_smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            train_canon.add(Chem.MolToSmiles(mol, canonical=True))
            
    duplicates = []
    for s in test_smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            canon_s = Chem.MolToSmiles(mol, canonical=True)
            if canon_s in train_canon:
                duplicates.append(s)
                
    return duplicates

def main(argv):
    # --- 1. SETUP ---
    # Update paths to match your actual files
    df_train = pd.read_csv(f'../data/crossval_splits/{argv[1]}/cv_2/train.csv')
    df_test = pd.read_csv(f'../data/crossval_splits/{argv[1]}/test/test.csv')
    print(f"Loaded {len(df_train)} training and {len(df_test)} test molecules.")

    # --- 2. DUPLICATE CHECK (String Match) ---
    print("\nChecking for strict duplicates (Canonical SMILES)...")
    dups = find_duplicates(df_test['smiles'], df_train['smiles'])
    
    if dups:
        print(f"WARNING: Found {len(dups)} strict duplicates (Identical strings).")
    else:
        print("SUCCESS: No strict duplicates found.")

    # --- 3. SIMILARITY METRICS (Count-Based Morgan) ---
    print("\nCalculating similarity metrics (Count-Based Morgan + Chirality)...")
    train_fps = get_count_fingerprints(df_train['smiles'])
    test_fps = get_count_fingerprints(df_test['smiles'])

    test_max_sims, test_mean_sims = calculate_similarity_metrics(
        test_fps, train_fps
    )

    # --- 4. PRINT RESULTS ---
    print("\n--- SIMILARITY ANALYSIS RESULTS ---")
    print("Method: Morgan Generator (Count-Based, Radius=2, with Chirality)")
    
    print("\nMETRIC 1: Max Similarity (Nearest Neighbor)")
    print(f"  Average: {np.mean(test_max_sims):.4f}")
    print(f"  Range:   {np.min(test_max_sims):.4f} - {np.max(test_max_sims):.4f}")
    
    print("\nMETRIC 2: Mean Similarity (Global Similarity)")
    print(f"  Average: {np.mean(test_mean_sims):.4f}")
    print(f"  Range:   {np.min(test_mean_sims):.4f} - {np.max(test_mean_sims):.4f}")

if __name__ == "__main__":
    main(sys.argv)