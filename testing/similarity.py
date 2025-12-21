import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

def get_fingerprints(smiles_series, radius=2, nBits=2048):
    """
    Converts a pandas Series of SMILES strings to a list of Morgan Fingerprints
    using the modern rdFingerprintGenerator API.
    """
    # Create the generator once (more efficient)
    mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    
    fps = []
    
    for smile in smiles_series:
        try:
            mol = Chem.MolFromSmiles(smile)
            if mol is not None:
                fp = mfgen.GetFingerprint(mol)
                fps.append(fp)
        except:
            pass
            
    return fps

def calculate_similarity_metrics(query_fps, reference_fps):
    """
    For each fingerprint in query_fps:
      1. Calculates similarity to ALL reference_fps.
      2. Extracts the MAX similarity (nearest neighbor).
      3. Extracts the MEAN similarity (average distance to set).
    """
    if not query_fps or not reference_fps:
        return [], []

    max_sims = []
    mean_sims = []
    
    # Iterate through each molecule in the query set (Test Set)
    for q_fp in query_fps:
        # Calculate similarity against ALL training molecules
        # BulkTanimotoSimilarity returns a list of scores
        sims = DataStructs.BulkTanimotoSimilarity(q_fp, reference_fps)
        
        # Metric 1: Max Similarity (Did we memorize a specific molecule?)
        max_sims.append(max(sims))
        
        # Metric 2: Mean Similarity (How distinct is this molecule from the whole set?)
        mean_sims.append(np.mean(sims))
        
    return max_sims, mean_sims

def main():
    # 1. Load the datasets
    try:
        df_train = pd.read_csv('../data/crossval_splits/three/cv_0/train.csv')
        df_valid = pd.read_csv('../data/crossval_splits/three/cv_0/valid.csv')
        df_test = pd.read_csv('../data/crossval_splits/three/test/test.csv')
    except FileNotFoundError as e:
        print(f"Error: Could not find file. {e}")
        return

    train_fps = get_fingerprints(df_train['smiles'])
    test_fps = get_fingerprints(df_test['smiles'])

    test_max_sims, test_mean_sims = calculate_similarity_metrics(test_fps, train_fps)

    # 4. Aggregating Results
    avg_max_sim = np.mean(test_max_sims)
    
    avg_mean_sim = np.mean(test_mean_sims)
    min_mean_sim = np.min(test_mean_sims)
    max_mean_sim = np.max(test_mean_sims)

    # 5. Output Report
    print(f"SIMILARITY ANALYSIS RESULTS")
    
    print(f"\nMETRIC 1: Nearest Neighbor Analysis (Leakage Check)")
    print(f"Average Maximum Similarity: {avg_max_sim:.4f}")

    print(f"\nMETRIC 2: Mean Similarity Analysis (Distinctness Check)")
    print(f"Average Mean Similarity:    {avg_mean_sim:.4f}")
    print(f"Range of Mean Similarities: {min_mean_sim:.4f} - {max_mean_sim:.4f}")
    

if __name__ == "__main__":
    main()