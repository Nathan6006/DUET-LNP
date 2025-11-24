import pandas as pd

df = pd.read_csv("main_data_.csv")

new=df["smiles"]
x=df["quantified_toxicity"]

new = pd.concat([new, x], axis=1)
new.to_csv("main_data__.csv", index=False)