import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# calcuate weight to weight ratio given molar ratio

# === Constants ===
MW_mRNA = 1375250.0   # Cas9 mRNA molecular weight in g/mol (1375.25 kDa)
R = 2500                # Fixed molar ratio lipid:nucleic acid

# === Load CSVs ===
main_data = pd.read_csv("main_data.csv")
metadata = pd.read_csv("individual_metadata.csv")
formulations = pd.read_csv("formulations.csv")

# === Step 1: Copy Dosage column into formulations ===
formulations["Dosage"] = metadata["Dosage"]

# === Step 2: Compute lipid molecular weight from SMILES ===
main_data["MW_lipid"] = main_data["smiles"].apply(
    lambda s: Descriptors.MolWt(Chem.MolFromSmiles(s))
)

# === Step 3: Compute lipid:mRNA weight ratio ===
main_data["Cationic_Lipid_to_mRNA_weight_ratio"] = (
    R * main_data["MW_lipid"] / MW_mRNA
)

# === Step 4: Add ratio column into formulations ===
formulations["Cationic_Lipid_to_mRNA_weight_ratio"] = main_data[
    "Cationic_Lipid_to_mRNA_weight_ratio"
]

# === Step 5: Save updated formulations ===
formulations.to_csv("formulations_final.csv", index=False)

print("Updated formulations saved to formulations_final.csv")