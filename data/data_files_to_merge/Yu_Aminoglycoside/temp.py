import pandas as pd

df = pd.read_csv("main_data.csv")
df["quantified_toxicity"] = df["quantified_toxicity"] * 100

df.to_csv("main_data_.csv", index=False)