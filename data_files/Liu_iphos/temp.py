import pandas as pd

# Load the data
df = pd.read_csv("main_data.csv")

# Define the mapping
mapping = {
    0: 2.69,
    1: 3.69,
    2: 4.69,
    3: 5.69
}

# Apply the mapping to the quantified_delivery column
df["quantified_delivery"] = df["quantified_delivery"].map(mapping)

# Save to new file
df.to_csv("main_data_.csv", index=False)