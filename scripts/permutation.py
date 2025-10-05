import pandas as pd
import numpy as np

def make_permutation_csv(input_file="toxicity.csv", output_file="permutation.csv", seed=42):
    """
    Reads a CSV with 'toxicity' column, shuffles the values, 
    and writes a new CSV with the same structure but randomized labels.
    """
    # Load original data
    df = pd.read_csv(input_file)
    
    if "toxicity" not in df.columns:
        raise ValueError("Input file must contain a 'toxicity' column.")
    
    # Copy and shuffle toxicity values
    shuffled_df = df.copy()
    shuffled_df["toxicity"] = np.random.RandomState(seed).permutation(df["toxicity"].values)
    
    # Save shuffled dataset
    shuffled_df.to_csv(output_file, index=False)
    print(f"Permutation dataset saved to {output_file}")

make_permutation_csv("toxicity.csv", "permutation.csv")
