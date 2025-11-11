import pandas as pd
import sys

def normalize_toxicity(input_file="3_tails.csv", output_file="3_tails_normalized.csv"):
    """
    Reads a CSV file, normalizes the 'toxicity' column to [0,1] using min-max scaling,
    and writes the result to a new CSV file.
    """
    # Load the data
    df = pd.read_csv(input_file)

    # Check that the column exists
    if "toxicity" not in df.columns:
        raise ValueError("Column 'toxicity' not found in the input file.")

    # Perform min-max normalization
    min_val = df["toxicity"].min()
    max_val = df["toxicity"].max()

    if min_val == max_val:
        # Avoid division by zero: all values are the same
        df["toxicity_normalized"] = 0.0
    else:
        df["toxicity_normalized"] = (df["toxicity"] - min_val) / (max_val - min_val)

    # Save to new file
    df.to_csv(output_file, index=False)

    print(f"✅ Normalized toxicity values saved to {output_file}")


normalize_toxicity()