from rdkit import Chem
import pandas as pd

def build_aifa_lipid(monomer_smiles, k_repeat):
    """
    Robust assembly using string concatenation followed by RDKit graph validation.
    """
    # 1. Start with Acetyl Cap (from Acetic Anhydride step)
    n_cap = "CC(=O)"
    
    # 2. C-terminal Amide (from TFA cleavage of Rink Amide Resin)
    c_term = "N"
    
    # 3. Concatenate: Cap + (Repeat Units) + End
    full_string = n_cap + (monomer_smiles * k_repeat) + c_term
    
    # 4. Double Check: Convert to RDKit object to validate chemistry
    mol = Chem.MolFromSmiles(full_string)
    if mol is None:
        raise ValueError(f"Invalid chemical structure generated for: {full_string}")
    
    # Sanitize to ensure correct valency and aromaticity
    Chem.SanitizeMol(mol)
    
    return Chem.MolToSmiles(mol)

# Load the blocks
df_blocks = pd.read_csv("parts.csv")

# Create full lipids for each block (example: k=4 as seen in a12K4)
full_lipids = []
for k in range(1,6):
    for index, row in df_blocks.iterrows():
        full_name = f"{row['part']}-{k}"
        try:
            lipid_smiles = build_aifa_lipid(row['smiles'], k)
            full_lipids.append({"full_name": full_name, "smiles": lipid_smiles})
        except Exception as e:
            print(f"Error building {full_name}: {e}")

# Save to new CSV
df_full = pd.DataFrame(full_lipids)
df_full.to_csv("full_lipids_library.csv", index=False)
print("Saved full_lipids_library.csv")