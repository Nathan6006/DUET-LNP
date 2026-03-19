import sys
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.DataStructs import TanimotoSimilarity, BulkTanimotoSimilarity
import Levenshtein
from tqdm import tqdm

# ----------------------------
# Parse arguments
# ----------------------------
if len(sys.argv) != 2:
    print("Usage: python cliff.py {del or tox}")
    sys.exit(1)

mode = sys.argv[1]

if mode not in ["del", "tox"]:
    print("Argument must be 'del' or 'tox'")
    sys.exit(1)

# ----------------------------
# Mode configuration
# ----------------------------
min_sim = 0.9
if mode == "del":
    spec_file = "crossval_split_specs/zhu.csv"
    target_col = "quantified_delivery"
    target_diff = 1.5
else:
    spec_file = "crossval_split_specs/tox.csv"
    target_col = "quantified_toxicity"
    target_diff = 0.2

output_file = f"cliffs_{mode}.csv"

# ----------------------------
# Load experiment IDs
# ----------------------------
spec_df = pd.read_csv(spec_file)
valid_ids = set(spec_df["Values"])

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("all_data.csv")
df = df[df["Experiment_ID"].isin(valid_ids)].copy()

# ----------------------------
# Precompute chemistry objects
# ----------------------------
print("Preparing molecules...")

df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
df = df[df["mol"].notnull()].copy()

df["ecfp"] = df["mol"].apply(lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))
df["maccs"] = df["mol"].apply(lambda m: MACCSkeys.GenMACCSKeys(m))
df["smiles_len"] = df["smiles"].apply(len)

# ----------------------------
# Similarity helpers
# ----------------------------
def smiles_similarity(s1, s2, l1, l2):
    dist = Levenshtein.distance(s1, s2)
    return 1 - dist / max(l1, l2)

# ----------------------------
# Cliff detection
# ----------------------------
results = []

for exp_id, group in tqdm(df.groupby("Experiment_ID"), desc="Processing experiments"):

    group = group.reset_index(drop=True)

    ecfps = list(group["ecfp"])

    for i in range(len(group)):

        fp_i = ecfps[i]

        # Fast vectorized similarity to all later molecules
        sims = BulkTanimotoSimilarity(fp_i, ecfps[i+1:])

        for offset, sub in enumerate(sims):

            j = i + 1 + offset

            # Early pruning
            if sub < min_sim-0.05:
                continue

            scaffold = TanimotoSimilarity(group.loc[i,"maccs"], group.loc[j,"maccs"])

            if (sub + scaffold)/2 < min_sim-0.05:
                continue

            smiles_sim = smiles_similarity(
                group.loc[i,"smiles"],
                group.loc[j,"smiles"],
                group.loc[i,"smiles_len"],
                group.loc[j,"smiles_len"]
            )

            structure_sim = (sub + scaffold + smiles_sim) / 3

            if structure_sim > min_sim:

                t1 = group.loc[i, target_col]
                t2 = group.loc[j, target_col]

                if abs(t1 - t2) > target_diff:

                    results.append({
                        "similarity_score": structure_sim,
                        "target_difference": abs(t1 - t2),
                        "target1": t1,
                        "target2": t2,
                        "smiles1": group.loc[i,"smiles"],
                        "smiles2": group.loc[j,"smiles"],
                        "Experiment_ID": exp_id
                    })

# ----------------------------
# Save output
# ----------------------------
cliffs_df = pd.DataFrame(results)
cliffs_df.to_csv(output_file, index=False)

print(f"Saved {len(cliffs_df)} cliffs to {output_file}")