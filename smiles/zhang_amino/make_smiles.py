import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from smiles.combine import attach_tails_to_core
import pandas as pd

# Load the CSV files
cell_df = pd.read_csv("cell_viability.csv")   # columns: cell_viability, lipid_name
parts_df = pd.read_csv("parts.csv")           # columns: part, smiles

# Create lookup dictionary for parts -> smiles
parts_dict = dict(zip(parts_df["part"], parts_df["smiles"]))

# Function to process each lipid_name
def process_lipid(lipid_name):
    core = lipid_name[:2]      # first two characters
    tail = lipid_name[-3:]     # last three characters
    
    core_smiles = parts_dict.get(core, "")
    tail_smiles = parts_dict.get(tail, "")
    
    if core_smiles and tail_smiles:
        return attach_tails_to_core(core_smiles, tail_smiles)
    else:
        return None

# Apply to each row
cell_df["smiles"] = cell_df["Lipid_name"].apply(process_lipid)

# Save updated CSV
cell_df.to_csv("cell_viability_with_smiles.csv", index=False)

print("Updated file saved as cell_viability_with_smiles.csv")