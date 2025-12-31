import pandas as pd
import re

# Load CSVs
main_del = pd.read_csv("main_del.csv")
meta_del = pd.read_csv("meta_del.csv")
main_data = pd.read_csv("main_data.csv")
individual_metadata = pd.read_csv("individual_metadata.csv")

def normalize(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())

# Normalize lipid names
meta_del["_norm_name"] = meta_del["Lipid_name"].apply(normalize)
individual_metadata["_norm_name"] = individual_metadata["Lipid_name"].apply(normalize)

quantified_values = []

for _, row in individual_metadata.iterrows():
    ind_name = row["_norm_name"]

    match = meta_del[meta_del["_norm_name"].str.contains(ind_name, na=False)]

    if len(match) == 0:
        quantified_values.append(None)
    else:
        idx = match.index[0]
        quantified_values.append(main_del.loc[idx, "quantified_delivery"])

# Add new column by row alignment
main_data["quantified_delivery"] = quantified_values

# Save output
main_data.to_csv("main_data_.csv", index=False)