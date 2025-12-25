import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

def get_count_fingerprints(smiles_series):
    """
    Generates Count-Based Morgan Fingerprints using the modern MorganGenerator.
    """
    # 1. Instantiate the MorganGenerator
    # radius=2 is standard (ECFP4 equivalent)
    # includeChirality=True is critical for stereoisomers
    mfgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, 
        includeChirality=True
    )

    fps = []
    valid_indices = []

    for idx, smile in enumerate(smiles_series):
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            # 2. Use GetCountFingerprint (returns SparseIntVect)
            # This preserves count info (e.g. 2 vs 4 carbons), crucial for lipids
            fp = mfgen.GetCountFingerprint(mol)
            fps.append(fp)
            valid_indices.append(idx)

    return fps, valid_indices

def calculate_corrected_metrics(test_fps, train_fps, test_smiles, train_smiles):
    """
    Calculates Tanimoto similarity with a 'Safety Patch' for isomers.
    
    Problem: Morgan fingerprints sometimes score Cis/Trans isomers as 1.0.
    Fix: If Score=1.0 but SMILES strings are different, force Score=0.9999.
    """
    max_sims = []
    mean_sims = []
    
    # Pre-compute unique Canonical SMILES in training set for fast lookup
    train_canon_set = set()
    for s in train_smiles:
        mol = Chem.MolFromSmiles(s)
        if mol: 
            train_canon_set.add(Chem.MolToSmiles(mol, canonical=True))
    
    # Iterate through each test molecule
    for i, q_fp in enumerate(test_fps):
        # Calculate bulk similarity against all training fingerprints
        sims = DataStructs.BulkTanimotoSimilarity(q_fp, train_fps)
        
        if not sims:
            max_sims.append(0.0)
            mean_sims.append(0.0)
            continue
            
        raw_max = max(sims)
        mean_val = np.mean(sims)
        
        # --- THE SAFETY PATCH ---
        final_max = raw_max
        
        # If the fingerprint says "Perfect Match" (1.0)
        if raw_max >= 0.999999: 
            # Check the actual string
            test_mol = Chem.MolFromSmiles(test_smiles[i])
            if test_mol:
                test_canon = Chem.MolToSmiles(test_mol, canonical=True)
                
                # If the string is NOT in the training set, it's an isomer collision.
                # Cap the score to indicate "Very similar, but not identical"
                if test_canon not in train_canon_set:
                    final_max = 0.9999 
        
        max_sims.append(final_max)
        mean_sims.append(mean_val)

    return max_sims, mean_sims

def main():
    # --- 1. SETUP ---
    df_train = pd.read_csv('../data/crossval_splits/butina/cv_2/train.csv')
    df_test = pd.read_csv('../data/crossval_splits/butina/test/test.csv')

    print(f"Loaded {len(df_train)} training and {len(df_test)} test molecules.")

    # --- 2. GENERATE FINGERPRINTS (MorganGenerator) ---
    print("Generating fingerprints (MorganGenerator, Count-based)...")
    train_fps, train_valid_idx = get_count_fingerprints(df_train['smiles'])
    test_fps, test_valid_idx = get_count_fingerprints(df_test['smiles'])

    # Align SMILES lists with valid fingerprints (in case of RDKit failures)
    train_clean_smiles = df_train['smiles'].iloc[train_valid_idx].tolist()
    test_clean_smiles = df_test['smiles'].iloc[test_valid_idx].tolist()

    # --- 3. CALCULATE METRICS ---
    print("Calculating metrics (with Isomer Correction)...")
    test_max_sims, test_mean_sims = calculate_corrected_metrics(
        test_fps, 
        train_fps,
        test_clean_smiles,
        train_clean_smiles
    )

    # --- 4. PRINT RESULTS ---
    print("\n--- SIMILARITY ANALYSIS RESULTS ---")
    print("Method: MorganGenerator (Count, Radius=2) + String Verification")
    
    print("\nMETRIC 1: Max Similarity (Nearest Neighbor)")
    print(f"  Average: {np.mean(test_max_sims):.4f}")
    # The max should now be 0.9999 instead of 1.0000 for isomers
    print(f"  Range:   {np.min(test_max_sims):.4f} - {np.max(test_max_sims):.4f}")
    
    print("\nMETRIC 2: Mean Similarity (Global Similarity)")
    print(f"  Average: {np.mean(test_mean_sims):.4f}")
    print(f"  Range:   {np.min(test_mean_sims):.4f} - {np.max(test_mean_sims):.4f}")

if __name__ == "__main__":
    main()