import pandas as pd

def keep_smiles_toxicity(input_file="3_tails.csv", output_file="3_tails_filtered.csv"):
    """
    Reads 3_tails.csv, keeps only 'smiles' and 'toxicity' columns,
    and saves to a new CSV file.
    """
    # Read the CSV
    df = pd.read_csv(input_file)

    # Keep only the desired columns
    filtered_df = df[['smiles', 'toxicity']]

    # Save to new file
    filtered_df.to_csv(output_file, index=False)
    print(f"Filtered file saved to {output_file}")

# Example usage:
keep_smiles_toxicity("3_tails.csv", "3_tails_filtered.csv")