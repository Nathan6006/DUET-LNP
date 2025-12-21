import pandas as pd
import re

# Read input file
lipids_df = pd.read_csv("all_lipids.csv")

# Create new dataframe with renamed column
meta_df = pd.DataFrame()
meta_df["Lipid_name"] = lipids_df["full_name"]

# Extract the repetition number (after the hyphen)
meta_df["Num_tails"] = (
    meta_df["Lipid_name"]
    .str.split("-")
    .str[-1]
    .astype(int) * 2
)

# Extract numeric part of the building block name
meta_df["Num_carbon_in_tail"] = (
    meta_df["Lipid_name"]
    .str.split("-")
    .str[0]
    .str.extract(r"(\d+)")
    .astype(int)
)

# Save output CSV
meta_df.to_csv("individual_metadata.csv", index=False)