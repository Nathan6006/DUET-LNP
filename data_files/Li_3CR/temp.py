import pandas as pd
import re

# Load the metadata file
df = pd.read_csv("individual_metadata.csv")

# Mapping from z → Num_carbon_in_tail
z_to_carbon = {
    1: 18,
    2: 18,
    3: 10,
    4: 18,
    5: 18,
    6: 12,
    7: 9,
    8: 18
}

# Extract the z value from patterns like Ax_Iy_Tz
# Example: A3_I12_T7 → z = 7
df["z_value"] = (
    df["Lipid_name"]
    .str.extract(r"_T(\d+)$")      # capture digits after _T
    .astype(int)
)

# Map z → Num_carbon_in_tail
df["Num_carbon_in_tail"] = df["z_value"].map(z_to_carbon)

# Num_tails is always 2
df["Num_tails"] = 2

# (Optional) drop helper column
df = df.drop(columns=["z_value"])

# Save updated file
df.to_csv("individual_metadata_parsed.csv", index=False)
