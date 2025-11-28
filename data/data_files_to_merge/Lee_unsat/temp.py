import pandas as pd
df = pd.read_csv("individual_metadata_.csv")

df.drop(columns=["MolWt"], inplace=True)
df.to_csv("individual_metadata.csv", index=False)