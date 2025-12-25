import pandas as pd
import re

# Read input CSV
df = pd.read_csv("full_smiles.csv")

# ---- Create main_data.csv ----
main_data = df[["smiles", "quantified_toxicity"]]
main_data.to_csv("main_data.csv", index=False)

# ---- Extract metadata from Lipid_name ----
def extract_num_tails(lipid_name):
    # number after 'A'
    match = re.search(r"A(\d+)", lipid_name)
    if not match:
        return None

    num_tails = int(match.group(1))

    # Special handling when extracted value is 1
    if num_tails == 1:
        # first digit(s) before 'A'
        prefix_match = re.match(r"(\d+)", lipid_name)
        if not prefix_match:
            return num_tails

        first_num = int(prefix_match.group(1))

        if 1 <= first_num <= 6:
            num_tails += 1
        elif 7 <= first_num <= 13:
            num_tails += 2
        elif 14 <= first_num <= 18:
            num_tails += 3

    return num_tails

def extract_num_carbon_in_tail(lipid_name):
    # Digits after '-P'
    match = re.search(r"-P(\d+)", lipid_name)
    return int(match.group(1)) if match else None

df["Num_tails"] = df["Lipid_name"].apply(extract_num_tails)
df["Num_carbon_in_tail"] = df["Lipid_name"].apply(extract_num_carbon_in_tail)

# Optional: save metadata if you want it separately
metadata = df[["Lipid_name", "Num_tails", "Num_carbon_in_tail"]]
metadata.to_csv("lipid_metadata.csv", index=False)