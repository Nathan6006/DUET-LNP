import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from smiles.combine import attach_tails_to_core

# Load the two input files
cell_viability = pd.read_csv("cell_viability.csv")  # contains Lipid_name, quantified_toxicity
parts = pd.read_csv("parts.csv")  # contains parts, smiles

# Create a dictionary for quick lookup of smiles by part name
parts_dict = dict(zip(parts["parts"], parts["smiles"]))

# Function to parse Lipid_name into prefix (A/B/C) and part (e.g., "1-2")
def parse_lipid_name(lipid_name):
    prefix = lipid_name[0]        # first character (A, B, or C)
    part = lipid_name[1:]         # rest of the string (e.g., "1-2")
    return prefix, part

# Build the main dataset
rows = []
for _, row in cell_viability.iterrows():
    lipid_name = row["Lipid_name"]
    toxicity = row["quantified_toxicity"]

    # Parse lipid name
    prefix, part = parse_lipid_name(lipid_name)

    # Look up smiles fragments
    tail_smiles = parts_dict.get(prefix)
    core_smiles = parts_dict.get(part)
    print(tail_smiles, core_smiles)

    if tail_smiles is None or core_smiles is None:
        raise ValueError(f"Missing SMILES for {lipid_name}: tail={prefix}, core={part}")

    # Combine using attach_tails_to_core
    full_smiles = attach_tails_to_core(tail_smiles, core_smiles)

    # Append to results
    rows.append({
        "smiles": full_smiles,
        "quantified_toxicity": toxicity
    })

# Convert to DataFrame
main_data = pd.DataFrame(rows)

# Save to CSV
main_data.to_csv("main_data.csv", index=False)