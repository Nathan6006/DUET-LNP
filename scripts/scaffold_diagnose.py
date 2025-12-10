import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem import DataStructs
    # Modern Fingerprint Generator
    from rdkit.Chem import rdFingerprintGenerator
except ImportError:
    print("Error: RDKit is required. Install via 'pip install rdkit'.")
    sys.exit(1)

# --- Configuration ---
DATA_PATH = '../data/all_data.csv'
SMILES_COL = 'smiles'       # Case sensitive in input file
TARGET_COL = 'quantified_toxicity'    # The column you are predicting
OUTPUT_DIR = '../scripts/scaffold_diagnostics'

def generate_scaffolds(df):
    """Adds a 'scaffold' column to the dataframe."""
    scaffolds = []
    print("Generating Murcko Scaffolds...")
    for smi in df[SMILES_COL]:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
                scaffolds.append(scaffold)
            else:
                scaffolds.append("INVALID")
        except:
            scaffolds.append("INVALID")
    df['scaffold'] = scaffolds
    return df[df['scaffold'] != "INVALID"]

def plot_target_distribution_by_scaffold(df, top_n=15):
    """
    Creates a HORIZONTAL boxplot.
    Truncates long SMILES strings so they don't crush the graph.
    """
    # 1. Select Top Scaffolds
    scaffold_counts = df['scaffold'].value_counts()
    top_scaffolds = scaffold_counts.head(top_n).index.tolist()
    subset = df[df['scaffold'].isin(top_scaffolds)].copy()
    
    # 2. Truncate SMILES for display (keep first 15 chars)
    # This prevents the text from taking up the whole image
    subset['display_label'] = subset['scaffold'].apply(
        lambda x: x[:15] + '...' if len(x) > 15 else x
    )
    
    # 3. Sort by Median Target Value (puts highest toxicity at top)
    # Note: We sort by the SHORT label now
    order = subset.groupby('display_label')[TARGET_COL].median().sort_values().index
    
    # 4. Plot Horizontal
    plt.figure(figsize=(12, 8)) # Taller figure to fit list
    sns.boxplot(
        data=subset,
        y='display_label',   # Put Text on Y-axis
        x=TARGET_COL,        # Put Data on X-axis
        order=order, 
        palette="viridis"
    )
    
    plt.title(f"Target Distribution for Top {top_n} Scaffolds")
    plt.xlabel(f"Target Value ({TARGET_COL})")
    plt.ylabel("Scaffold (Truncated)")
    
    # Add grid lines for easier reading
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'scaffold_target_distribution.png'))
    plt.close()
    print(f"Saved readable distribution plot to {OUTPUT_DIR}")

def plot_chemical_space(df):
    """
    Generates a t-SNE plot using the modern MorganGenerator.
    """
    print("Calculating Morgan Fingerprints (using MorganGenerator)...")
    
    # Radius 2 ~= ECFP4, 1024 bits
    mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    
    fps = []
    valid_indices = []
    
    for idx, smi in enumerate(df[SMILES_COL]):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            # Generate Folded Bit Vector
            fp = mfgen.GetFingerprint(mol)
            
            # Convert to numpy
            arr = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
            valid_indices.append(idx)
            
    if not fps:
        print("No valid fingerprints generated.")
        return

    X = np.array(fps)
    df_subset = df.iloc[valid_indices].copy()

    print("Running PCA + t-SNE...")
    # PCA to reduce to 50 dims first (standard practice for t-SNE on bits)
    pca = PCA(n_components=min(50, X.shape[1]))
    X_pca = pca.fit_transform(X)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_embedded = tsne.fit_transform(X_pca)
    
    df_subset['tsne_1'] = X_embedded[:, 0]
    df_subset['tsne_2'] = X_embedded[:, 1]
    
    # Label top 5 scaffolds
    top_5 = df_subset['scaffold'].value_counts().head(5).index
    df_subset['Scaffold_Label'] = df_subset['scaffold'].apply(lambda x: x if x in top_5 else 'Other')
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_subset.sort_values('Scaffold_Label'), 
        x='tsne_1', y='tsne_2', 
        hue='Scaffold_Label', 
        alpha=0.7,
        palette='tab10'
    )
    plt.title("Chemical Space (t-SNE) Colored by Top Scaffolds")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'chemical_space_tsne.png'))
    plt.close()

def analyze_scaffold_stats(df):
    """Calculates diagnostics and saves to CSV."""
    scaffold_counts = df['scaffold'].value_counts()
    n_scaffolds = len(scaffold_counts)
    n_singletons = (scaffold_counts == 1).sum()
    
    # Variance Analysis
    global_std = df[TARGET_COL].std()
    
    # Scaffolds with > 3 items
    large_scaffolds = scaffold_counts[scaffold_counts > 3].index
    intra_scaffold_stds = df[df['scaffold'].isin(large_scaffolds)].groupby('scaffold')[TARGET_COL].std()
    avg_intra_std = intra_scaffold_stds.mean() if not intra_scaffold_stds.empty else 0
    
    variance_ratio = avg_intra_std / global_std if global_std > 0 else 0
    
    # Interpretation logic
    if variance_ratio < 0.5:
        interpretation = "High Bias (Model memorizes scaffolds)"
    elif variance_ratio > 0.9:
        interpretation = "Low Bias (Model must learn features)"
    else:
        interpretation = "Balanced"

    # --- SAVE TO CSV LOGIC ---
    report_data = [
        {'Metric': 'Total Molecules', 'Value': len(df)},
        {'Metric': 'Total Unique Scaffolds', 'Value': n_scaffolds},
        {'Metric': 'Singleton Scaffolds (Count)', 'Value': n_singletons},
        {'Metric': 'Singleton Scaffolds (%)', 'Value': round(n_singletons/n_scaffolds, 4)},
        {'Metric': 'Largest Scaffold Size', 'Value': scaffold_counts.iloc[0]},
        {'Metric': 'Global Target Std Dev', 'Value': round(global_std, 4)},
        {'Metric': 'Avg Intra-Scaffold Std Dev', 'Value': round(avg_intra_std, 4)},
        {'Metric': 'Variance Ratio (Intra/Global)', 'Value': round(variance_ratio, 4)},
        {'Metric': 'Scaffold Interpretation', 'Value': interpretation}
    ]
    
    report_df = pd.DataFrame(report_data)
    csv_path = os.path.join(OUTPUT_DIR, 'scaffold_report.csv')
    report_df.to_csv(csv_path, index=False)
    
    print("\n" + "="*40)
    print("SCAFFOLD DIAGNOSTICS")
    print("="*40)
    print(report_df.to_string(index=False))
    print("="*40)
    print(f"Full report saved to: {csv_path}")

def main():
    if not os.path.exists(DATA_PATH):
        print(f"File not found: {DATA_PATH}")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Loading {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Normalize column names
    real_smiles = [c for c in df.columns if c.lower() == SMILES_COL.lower()]
    if not real_smiles:
        print(f"Column '{SMILES_COL}' not found.")
        return
    df = df.rename(columns={real_smiles[0]: SMILES_COL})

    # Auto-detect target
    global TARGET_COL
    if TARGET_COL not in df.columns:
        numerics = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numerics) > 0:
            TARGET_COL = numerics[-1] 
            print(f"Assuming target is: {TARGET_COL}")
        else:
            print("No numeric target found.")
            return

    df = generate_scaffolds(df)
    analyze_scaffold_stats(df)
    plot_target_distribution_by_scaffold(df)
    plot_chemical_space(df)

if __name__ == "__main__":
    main()