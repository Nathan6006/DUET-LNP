import numpy as np
from rdkit.Chem import rdFingerprintGenerator, DataStructs, MolFromSmiles, Descriptors


def dataset_to_numpy(datapoints, smiles_column="smiles", method="morgan"):
    X, y = [], []
    
    for i, dp in enumerate(datapoints):
        # Safety check for missing extra features
        extra_features = dp.get("x_d", [])
        if extra_features is None: 
            extra_features = []
        if method == "morgan":
            embed = morgan_fingerprint(dp[smiles_column], use_counts=True)
        elif method == "rdkit":
            embed = rdkit_descriptors(dp[smiles_column])
            embed = np.nan_to_num(embed, nan=0.0, posinf=0.0, neginf=0.0)
            embed = np.clip(embed, -1e6, 1e6)
        elif method == "rdkit_morgan":
            morgan = morgan_fingerprint(dp[smiles_column], use_counts=True)
            rdkit = rdkit_descriptors(dp[smiles_column])
            rdkit = np.nan_to_num(rdkit, nan=0.0, posinf=0.0, neginf=0.0)
            rdkit = np.clip(rdkit, -1e6, 1e6)
            embed = np.concatenate([morgan, rdkit])
        # Ensure dimensionality matches
        feats = np.concatenate([embed, np.array(extra_features)])
        X.append(feats)
        
        # specific handling for target existence
        target = dp.get("y", [None])[0]
        y.append(target)

    return np.array(X), np.array(y)

def morgan_fingerprint(smiles, radius=2, n_bits=2048, use_counts=True):
    """
    Convert a SMILES string into a Morgan fingerprint.
    
    Args:
        use_counts (bool): If True, returns count vector (ECFP-Counts). 
                           If False, returns bit vector (ECFP/Morgan).
    """
    mol = MolFromSmiles(smiles)
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

def rdkit_descriptors(smiles):
    """
    Convert a SMILES string into the full RDKit descriptor vector (~200 features).

    Returns:
        np.ndarray: 1D array of RDKit descriptors in a fixed order
    """
    mol = MolFromSmiles(smiles)
    n_desc = len(Descriptors._descList)

    if mol is None:
        return np.zeros(n_desc)

    values = []
    for _, func in Descriptors._descList:
        try:
            values.append(func(mol))
        except Exception:
            values.append(0.0)

    return np.array(values, dtype=float)  



