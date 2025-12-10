import pandas as pd
df = pd.read_csv("main_data.csv")
df.drop(columns=["quantified_delivery"], inplace=True)
df.to_csv("main_data_.csv", index=False)