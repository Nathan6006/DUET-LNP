import pandas as pd

# Load the CSV
df = pd.read_csv("full_smiles.csv")

# Split Lipid_name into amine and tail
df[["amine", "tail"]] = df["Lipid_name"].str.split("-", expand=True)

# Assign num_tails based on amine
def get_num_tails(amine):
    if amine == "20T":
        return 3
    elif amine == "22T":
        return 3
    elif amine == "25T":
        return 4
    else:
        return 2

df["Num_tails"] = df["amine"].apply(get_num_tails)

# Extract number from tail (e.g., O9 -> 9, O16 -> 16)
df["Num_carbon_in_tail"] = df["tail"].str.extract(r"O(\d+)").astype(int)

# Keep only the requested columns
final_df = df[["Lipid_name", "Num_tails", "Num_carbon_in_tail"]]

# Save to new CSV
final_df.to_csv("individual_metadata.csv", index=False)

print("main_data.csv created with Lipid_name, Num_tails, and Num_carbon_in_tail.")