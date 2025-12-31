import pandas as pd
df = pd.read_csv("main_data.csv")
d=pd.read_csv("main_dat.csv")
df["quantified_delivery"] = d["quantified_delivery"]
df.to_csv("main_data_.csv", index=False)