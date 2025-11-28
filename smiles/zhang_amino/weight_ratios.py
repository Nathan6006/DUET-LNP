import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# --- Parameters ---
# Define the molar ratio (CLS:siRNA or CLS:mRNA)
CLS_to_mRNA_molar_ratio = 50  

# Define the molecular weight of your nucleic acid (adjust as needed)
# Example: siLuc duplex ~13,860 Da, siFVII duplex ~19,000 Da
MW_mRNA = 13860  

# --- Read input CSV ---
df = pd.read_csv("main_data.csv")
form = pd.read_csv("formulations.csv")

# --- Calculate MW for each lipid and weight ratio ---
mw_list = []
ratio_list = []

for smiles in df["smiles"]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        mw = Descriptors.MolWt(mol)
        mw_list.append(mw)
        
        # Calculate weight ratio: (molar ratio × MW_lipid) / MW_mRNA
        weight_ratio = (CLS_to_mRNA_molar_ratio * mw) / MW_mRNA
        ratio_list.append(weight_ratio)
    except Exception as e:
        mw_list.append(None)
        ratio_list.append(None)

# --- Add results to dataframe ---
form["Cationic_Lipid_to_mRNA_weight_ratio"] = ratio_list

# --- Save to new CSV ---
form.to_csv("formulations_.csv", index=False)