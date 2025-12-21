import os
import sys
import warnings

# --- 1. Suppress Warnings & Configure Environment ---
# Filter specific UMAP warnings that are just informational
warnings.filterwarnings("ignore", message="gradient function is not yet implemented")
warnings.filterwarnings("ignore", message="n_jobs value 1 overridden")

# Suppress Intel OpenMP (OMP) status messages
os.environ["KMP_WARNINGS"] = "0"
os.environ["OMP_DISPLAY_ENV"] = "FALSE" 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import DataStructs
from rdkit.Chem import rdFingerprintGenerator
import umap

DATA_PATH = '../data/all_data.csv'
SMILES_COL = 'smiles'       
TARGET_COL = 'toxicity_class'
OUTPUT_DIR = '../testing/scaffold_diagnostics'

def generate_scaffolds(df):
    """Adds a 'scaffold' column to the dataframe."""
    scaffolds = []
    print("Generating Murcko Scaffolds...")
    for smi in df[SMILES_COL]:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)
                scaffolds.append(scaffold)
            else:
                scaffolds.append("INVALID")
        except:
            scaffolds.append("INVALID")
    df['scaffold'] = scaffolds
    return df[df['scaffold'] != "INVALID"]

def plot_target_distribution_by_scaffold(df, top_n=15):
    """
    Creates a STACKED BAR CHART for Top N Scaffolds.
    """
    scaffold_counts = df['scaffold'].value_counts().head(top_n)
    top_scaffolds = scaffold_counts.index.tolist()
    
    subset = df[df['scaffold'].isin(top_scaffolds)].copy()
    
    scaffold_to_label = {scaf: f"S{i+1}" for i, scaf in enumerate(top_scaffolds)}
    subset['Scaffold_Label'] = subset['scaffold'].map(scaffold_to_label)
    
    key_data = []
    for scaf, label in scaffold_to_label.items():
        scaf_subset = subset[subset['scaffold'] == scaf]
        # Use mode safely
        modes = scaf_subset[TARGET_COL].mode()
        dominant_class = modes[0] if not modes.empty else "N/A"
        
        key_data.append({
            'Label': label,
            'Count': scaffold_counts[scaf],
            'Dominant_Class': dominant_class,
            'Scaffold_SMILES': scaf
        })
    
    key_df = pd.DataFrame(key_data)
    key_df.to_csv(os.path.join(OUTPUT_DIR, 'scaffold_legend.csv'), index=False)

    counts = subset.groupby(['Scaffold_Label', TARGET_COL]).size().unstack(fill_value=0)
    order = [f"S{i+1}" for i in range(len(top_scaffolds))]
    counts = counts.reindex(order)
    
    counts.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='viridis')
    
    plt.title(f"Class Distribution for Top {top_n} Scaffolds")
    plt.xlabel("Scaffold ID (See scaffold_legend.csv)")
    plt.ylabel("Count of Molecules")
    plt.legend(title='Toxicity Class')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'scaffold_class_distribution.png'))
    plt.close()
    print(f"Saved stacked bar chart to {OUTPUT_DIR}/scaffold_class_distribution.png")

def get_morgan_fingerprints(df):
    """Helper to generate numpy array of fingerprints."""
    print("Calculating Morgan Fingerprints...")
    mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    
    fps = []
    valid_indices = []
    
    for idx, smi in enumerate(df[SMILES_COL]):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = mfgen.GetFingerprint(mol)
            arr = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
            valid_indices.append(idx)
            
    return np.array(fps), valid_indices

def plot_umap_space(df):
    """
    Generates a UMAP plot of the entire dataset.
    """
    X, valid_indices = get_morgan_fingerprints(df)
    df_subset = df.iloc[valid_indices].copy()
    
    if len(df_subset) < 10: # UMAP needs reasonable sample size
        print("Not enough data for UMAP (needs >10 samples).")
        return

    print("Running UMAP (2D projection)...")
    # Using 'jaccard' metric is better for binary fingerprints
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='jaccard', random_state=42)
    embedding = reducer.fit_transform(X)
    
    df_subset['umap_1'] = embedding[:, 0]
    df_subset['umap_2'] = embedding[:, 1]
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_subset,
        x='umap_1', y='umap_2',
        hue=TARGET_COL,         
        palette='viridis',      
        alpha=0.6,
        s=20
    )
    plt.title("Chemical Space (UMAP) colored by Toxicity Class")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'chemical_space_umap.png'))
    plt.close()
    print(f"Saved UMAP plot to {OUTPUT_DIR}/chemical_space_umap.png")

def analyze_scaffold_purity(df):
    """
    Calculates how 'pure' the scaffolds are.
    """
    scaffold_groups = df.groupby('scaffold')[TARGET_COL]
    
    purities = []
    sizes = []
    
    for name, group in scaffold_groups:
        if len(group) > 2: 
            most_common_count = group.value_counts().max()
            purity = most_common_count / len(group)
            purities.append(purity)
            sizes.append(len(group))
            
    avg_purity = np.mean(purities) if purities else 0
    weighted_purity = np.average(purities, weights=sizes) if purities else 0
    
    report_data = [
        {'Metric': 'Total Molecules', 'Value': len(df)},
        {'Metric': 'Total Scaffolds', 'Value': df['scaffold'].nunique()},
        {'Metric': 'Avg Scaffold Purity (Unweighted)', 'Value': round(avg_purity, 4)},
        {'Metric': 'Weighted Scaffold Purity', 'Value': round(weighted_purity, 4)},
        {'Metric': 'Interpretation', 'Value': 'High Purity = Scaffolds determine Class (Easy)' if weighted_purity > 0.9 else 'Mixed'}
    ]
    
    report_df = pd.DataFrame(report_data)
    csv_path = os.path.join(OUTPUT_DIR, 'scaffold_purity_report.csv')
    report_df.to_csv(csv_path, index=False)
    
    print("\n" + "="*40)
    print("SCAFFOLD PURITY REPORT")
    print("="*40)
    print(f"Weighted Purity: {weighted_purity:.4f}")
    print("(1.0 = Every scaffold contains only one class of toxicity)")
    print("(0.33 = Random mix of classes within scaffolds)")
    print("="*40)

def main():
    if not os.path.exists(DATA_PATH):
        print(f"File not found: {DATA_PATH}")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Loading {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Rename smiles column
    cols = {c.lower(): c for c in df.columns}
    if 'smiles' in cols:
        df = df.rename(columns={cols['smiles']: SMILES_COL})
    
    if TARGET_COL not in df.columns:
        print(f"Error: Column '{TARGET_COL}' not found. Check your CSV.")
        return

    # Ensure target is categorical string for plotting
    df[TARGET_COL] = df[TARGET_COL].astype(str)

    df = generate_scaffolds(df)
    
    analyze_scaffold_purity(df)
    plot_target_distribution_by_scaffold(df)
    plot_umap_space(df)

if __name__ == "__main__":
    main()