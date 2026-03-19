import pandas as pd

def find_smiles_duplicates(input_file, output_file):
    # Read data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows from {input_file}")

    # Reset index so original row indices are preserved
    df = df.reset_index()

    # Group by SMILES and aggregate required info
    agg = df.groupby("smiles").agg({
        "index": list,              # indices where this SMILES appears
        "Lipid_name": list,         # corresponding lipid names
        "Experiment_ID": list       # corresponding experiment IDs
    }).reset_index()

    # Keep only actual duplicates
    duplicates = agg[agg["index"].map(len) > 1].copy()

    if duplicates.empty:
        print("No duplicate SMILES found.")
        return

    # Rename columns for clarity
    duplicates.rename(columns={
        "smiles": "unique_smiles",
        "index": "duplicate_indices",
        "Lipid_name": "lipid_names",
        "Experiment_ID": "experiment_ids"
    }, inplace=True)

    # Save to CSV
    duplicates.to_csv(output_file, index=False)

    print(f"Found {len(duplicates)} duplicated SMILES entries.")
    print(f"Results saved to {output_file}")


# ---- CONFIG ----
INPUT_CSV = "all_data.csv"
OUTPUT_CSV = "smiles_duplicates.csv"

if __name__ == "__main__":
    find_smiles_duplicates(INPUT_CSV, OUTPUT_CSV)