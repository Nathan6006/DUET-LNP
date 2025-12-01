import pandas as pd

# Load the main data file
main_df = pd.read_csv("main_data.csv")

# --- Step 1: Create individual_metadata.csv ---
# Extract Lipid_name
metadata_df = main_df[["Lipid_name"]].copy()

# Add Num_tails = 2 for all rows
metadata_df["Num_tails"] = 2

# Parse Lipid_name to determine Num_carbon_in_tail
def parse_num_carbon(lipid_name):
    if "-" in lipid_name:
        parts = lipid_name.split("-")
        return max(int(parts[0]), int(parts[1]))
    else:
        if "MC3" == lipid_name:
            return 19
        else:
            return int(lipid_name)

metadata_df["Num_carbon_in_tail"] = metadata_df["Lipid_name"].apply(parse_num_carbon)

# Save metadata file
metadata_df.to_csv("individual_metadata.csv", index=False)

# --- Step 2: Create formulations.csv ---
formulations_df = main_df[["Dosage"]].copy()
formulations_df.to_csv("formulations.csv", index=False)

# --- Step 3: Clean main_data.csv to only keep smiles + quantified_toxicity ---
main_cleaned = main_df[["smiles", "quantified_toxicity"]].copy()
main_cleaned.to_csv("main_data_.csv", index=False)

print("Files created: individual_metadata.csv, formulations.csv, and updated main_data.csv")