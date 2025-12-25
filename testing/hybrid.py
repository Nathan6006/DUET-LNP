import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from rdkit.Chem import rdFingerprintGenerator, Descriptors

# --- Config ---
DATA_PATH = '../data/all_data.csv'
SMILES_COL = 'smiles'
TARGET_COL = 'toxicity_class'
WEIGHT_COL = 'Sample_weight'
CLUSTERING_THRESHOLD = 0.7 

def get_maccs_and_physics(df):
    """
    Generates MACCS Keys (167 bits) + Physics (6 features).
    Total Features = 173.
    """
    X = []
    y = []
    weights = []
    has_weights = WEIGHT_COL in df.columns
    
    print("Generating MACCS Keys + Physics...")
    
    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row[SMILES_COL])
        if mol:
            # 1. MACCS Keys (Structural dictionary)
            # MACCSkeys.GenMACCSKeys(mol) returns a bit vector of length 167
            maccs = MACCSkeys.GenMACCSKeys(mol)
            maccs_arr = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(maccs, maccs_arr)
            
            # 2. Physics (Universal properties)
            phys = np.array([
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.MolWt(mol),
                Chem.GetFormalCharge(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumHDonors(mol) + Descriptors.NumHAcceptors(mol)
            ])
            
            # 3. Combine
            combined = np.concatenate([maccs_arr, phys])
            
            X.append(combined)
            y.append(row[TARGET_COL])
            weights.append(row[WEIGHT_COL] if has_weights else 1.0)
            
    return np.array(X), np.array(y), np.array(weights)

def generate_clusters_butina(df, threshold=0.7):
    # Same clustering as before to keep comparison fair
    print(f"Clustering (Threshold={threshold})...")
    mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mols, fps, valid_indices = [], [], []
    for idx, smi in enumerate(df[SMILES_COL]):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mols.append(mol)
            fps.append(mfgen.GetFingerprint(mol))
            valid_indices.append(idx)
    dists = []
    n_fps = len(fps)
    for i in range(1, n_fps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])
    clusters = Butina.ClusterData(dists, n_fps, 1.0 - threshold, isDistData=True)
    cluster_map = {}
    for c_id, indices in enumerate(clusters):
        for idx in indices:
            cluster_map[valid_indices[idx]] = c_id
    df['cluster_id'] = df.index.map(cluster_map)
    return df.dropna(subset=['cluster_id']).copy()

def train_maccs_physics(df):
    groups = df['cluster_id'].values
    X_all, y_all, w_all = get_maccs_and_physics(df)
    
    sgkf = StratifiedGroupKFold(n_splits=5)
    fold_scores = []
    
    print("\n" + "="*50)
    print(f"STARTING MACCS + PHYSICS TRAINING (Features: {X_all.shape[1]})")
    print("="*50)

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X_all, y_all, groups)):
        
        X_train, y_train, w_train = X_all[train_idx], y_all[train_idx], w_all[train_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]
        
        clf = xgb.XGBClassifier(
            objective='multi:softmax', num_class=3,
            n_estimators=100, max_depth=6,
            eval_metric='mlogloss', n_jobs=-1, random_state=42
        )
        clf.fit(X_train, y_train, sample_weight=w_train)
        
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        print(f"FOLD {fold+1}: F1 Macro={f1:.4f}")
        fold_scores.append(f1)
        
        # Check Feature Importance on Fold 1
        if fold == 0:
            print("\n--- Feature Importance Snapshot (Fold 1) ---")
            # Last 6 features are Physics
            phys_imp = np.sum(clf.feature_importances_[-6:])
            # First 167 are MACCS
            maccs_imp = np.sum(clf.feature_importances_[:-6])
            print(f"Physics Contribution: {phys_imp*100:.2f}%")
            print(f"MACCS (Structure) Contribution: {maccs_imp*100:.2f}%")
            
            # Identify Top 3 MACCS Keys (What structure matters?)
            maccs_indices = np.argsort(clf.feature_importances_[:-6])[::-1][:3]
            print(f"Top 3 Important MACCS Keys: {maccs_indices}")
            print("-" * 30)

    print("\n" + "-"*40)
    print(f"AVERAGE F1 MACRO: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
    print("-" * 40)

def main():
    df = pd.read_csv(DATA_PATH)
    cols = {c.lower(): c for c in df.columns}
    if 'smiles' in cols: df = df.rename(columns={cols['smiles']: SMILES_COL})
    df = generate_clusters_butina(df, threshold=CLUSTERING_THRESHOLD)
    train_maccs_physics(df)

if __name__ == "__main__":
    main()