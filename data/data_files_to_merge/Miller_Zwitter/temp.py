import pandas as pd
df = pd.read_csv("formulations.csv")
df["Helper_lipid_ID"] = 'None'
df.to_csv("formulations_.csv", index=False)