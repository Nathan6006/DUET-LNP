import pandas as pd

# Load both CSVs
meta = pd.read_csv("individual_metadata.csv")
main = pd.read_csv("main_data.csv")

# Substrings to filter out
bad_patterns = ["Iso62DC18", "Iso92DC18", "Iso52DC18"]

# Identify rows in metadata where Lipid_name contains any of the patterns
mask_bad = meta["Lipid_name"].astype(str).str.contains("|".join(bad_patterns), regex=True)

# Get the indices to drop
indices_to_drop = meta.index[mask_bad]

print("Dropping rows at indices:", list(indices_to_drop))

# Drop those rows from both dataframes
meta_clean = meta.drop(indices_to_drop).reset_index(drop=True)
main_clean = main.drop(indices_to_drop).reset_index(drop=True)

# Save cleaned versions
meta_clean.to_csv("individual_metadata_cleaned.csv", index=False)
main_clean.to_csv("main_data_cleaned.csv", index=False)

print("Cleaning complete. Saved individual_metadata_cleaned.csv and main_data_cleaned.csv.")
