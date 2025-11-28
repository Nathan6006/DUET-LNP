import pandas as pd 
df = pd.read_csv("main_data.csv")
main = df.drop(columns=["Lipid_name", "Dosage"])
meta = df.drop(columns=["smiles", "quantified_toxicity"])

main.to_csv("main_data_.csv", index=False)
meta.to_csv("individual_metadata.csv", index=False)