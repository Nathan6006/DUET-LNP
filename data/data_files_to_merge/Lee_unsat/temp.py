import pandas as pd
df = pd.read_csv("formulations.csv")
df["Dosage"] = 25
df.to_csv("formulation.csv", index=False)