import pandas as pd

# Read the toxicity file
toxicity_df = pd.read_csv("toxicity.csv")

# Filter rows where identifier == "a3"
filtered_df = toxicity_df[toxicity_df["identifier"].str.contains("a3", na=False)]

# Keep the entire row (all columns)
filtered_df.to_csv("3_tails.csv", index=False)

print("✅ Full rows with identifier 'a3' saved to 3_tails.csv")