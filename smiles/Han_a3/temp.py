import pandas as pd 

df = pd.read_csv("cell_viability.csv")

new = pd.DataFrame()
new["Lipid_name"] = df["Lipid_name"]

new["Num_tails"] = df["Lipid_name"].str[-1:].astype(int)

# Map first character (A/B/C) to carbon counts
map = {"A": 10, "B": 12, "C": 13}
new["Num_carbon_in_tail"] = new["Lipid_name"].str[0].map(map)

new.to_csv("individual_metadata.csv", index=False)