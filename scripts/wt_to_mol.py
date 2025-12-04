import pandas as pd
import numpy as np

folder = "../data/data_files_to_merge/Han_a3/"
formulations = pd.DataFrame()         # currently only has Dosage
metadata = pd.read_csv(f"{folder}individual_metadata.csv")       # has Lipid_name and MolWt

MW_DOPE = 744.0        # dioleoylphosphatidylethanolamine
MW_CHOL = 386.65       # cholesterol
MW_PEG = 2500.0        # DMG-PEG2000 (average MW ~2500)
MW_mRNA = 340.0        # average per nucleotide

weight_ratios = {
    "cationic": 16.0,
    "dope": 10.0,
    "chol": 10.0,
    "peg": 3.0,
    "mrna": 1.6
}

mol_ratios = []
for idx, row in metadata.iterrows():
    mw_cationic = row["MolWt"]  # from individual_metadata.csv
    
    # Convert weight ratios to molar ratios
    mol_cationic = weight_ratios["cationic"] / mw_cationic
    mol_dope     = weight_ratios["dope"]     / MW_DOPE
    mol_chol     = weight_ratios["chol"]     / MW_CHOL
    mol_peg      = weight_ratios["peg"]      / MW_PEG
    
    ratios = np.array([mol_cationic, mol_dope, mol_chol, mol_peg])
    ratios = ratios / mol_cationic  # normalize relative to cationic lipid
    
    max_val = ratios.max()
    scale_factor = 40 / max_val if max_val > 0 else 1
    scaled = ratios * scale_factor
    
    mol_ratios.append({
        "Cationic_Lipid_Mol_Ratio": scaled[0],
        "Phospholipid_Mol_Ratio": scaled[1],
        "Cholesterol_Mol_Ratio": scaled[2],
        "PEG_Lipid_Mol_Ratio": scaled[3]
    })

formulations["Formulation"] = ["Han_a3"] * len(mol_ratios)
formulations["Cationic_Lipid_Mol_Ratio"] = [r["Cationic_Lipid_Mol_Ratio"] for r in mol_ratios]
formulations["Phospholipid_Mol_Ratio"] = [r["Phospholipid_Mol_Ratio"] for r in mol_ratios]
formulations["Cholesterol_Mol_Ratio"] = [r["Cholesterol_Mol_Ratio"] for r in mol_ratios]
formulations["PEG_Lipid_Mol_Ratio"] = [r["PEG_Lipid_Mol_Ratio"] for r in mol_ratios]
formulations["Helper_lipid_ID"] = ["DOPE"] * len(mol_ratios)
formulations["Cationic_Lipid_to_mRNA_weight_ratio"] = [10] * len(mol_ratios)
formulations["Dosage"] = [15] * len(mol_ratios)
 
formulations.to_csv(f"{folder}formulations.csv", index=False)

print("formulations.csv updated with scaled molar ratios (range ~10–40).")