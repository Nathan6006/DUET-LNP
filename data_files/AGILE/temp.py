import pandas as pd
import re

# Read the CSV
df = pd.read_csv("individual_metadata.csv")

# Extract the number after 'B' at the end of Lipid_name
df.drop(columns=["Lipid/Cells"], inplace=True)

# Optional: save back to CSV
df.to_csv("individual_metadata.csv", index=False)