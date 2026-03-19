import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv("main_data.csv")

# Convert log2(raw) → log10(raw)
# log10(raw) = x * log10(2)
df["quantified_delivery"] = df["quantified_delivery"] * np.log10(2)

# Save back to CSV
df.to_csv("main_data.csv", index=False)