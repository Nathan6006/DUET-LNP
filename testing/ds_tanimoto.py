import os
import itertools
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np

BASE_DIR = "../data_files"
META_PATH = os.path.join(BASE_DIR, "experiment_metadata.csv")
OUT_PATH = os.path.join(BASE_DIR, "tanimoto_similarity_report.csv")

def smiles_to_fps(smiles_list, radius=2, nBits=2048):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        fps.append(fp)
    return fps

def mean_and_max_tanimoto(fps1, fps2):
    if len(fps1) == 0 or len(fps2) == 0:
        return np.nan, np.nan

    sims = []
    for fp1 in fps1:
        sims.extend(DataStructs.BulkTanimotoSimilarity(fp1, fps2))

    sims = np.array(sims)
    return float(sims.mean()), float(sims.max())

# --- Load experiment IDs ---
meta_df = pd.read_csv(META_PATH)
experiment_ids = meta_df["Experiment_ID"].dropna().unique().tolist()

# --- Load fingerprints for each experiment ---
experiment_fps = {}

for exp_id in experiment_ids:
    data_path = os.path.join(BASE_DIR, exp_id, "main_data.csv")
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found, skipping")
        continue

    df = pd.read_csv(data_path)
    if "smiles" not in df.columns:
        print(f"Warning: smiles column missing in {exp_id}, skipping")
        continue

    fps = smiles_to_fps(df["smiles"].dropna().tolist())
    experiment_fps[exp_id] = fps

# --- Pairwise similarities ---
rows = []

for exp1, exp2 in itertools.combinations(experiment_fps.keys(), 2):
    mean_sim, max_sim = mean_and_max_tanimoto(
        experiment_fps[exp1],
        experiment_fps[exp2]
    )

    rows.append({
        "experiment_1": exp1,
        "experiment_2": exp2,
        "mean_tanimoto": mean_sim,
        "max_tanimoto": max_sim
    })

report_df = pd.DataFrame(rows)

# --- Overall averages ---
overall = {
    "experiment_1": "ALL",
    "experiment_2": "ALL",
    "mean_tanimoto": report_df["mean_tanimoto"].mean(),
    "max_tanimoto": report_df["max_tanimoto"].mean()
}

report_df = pd.concat([report_df, pd.DataFrame([overall])], ignore_index=True)

# --- Save ---
report_df.to_csv(OUT_PATH, index=False)

print(f"Saved Tanimoto similarity report to {OUT_PATH}")