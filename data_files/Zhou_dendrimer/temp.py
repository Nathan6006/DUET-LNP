import pandas as pd

def fix_sc3_smiles(main_data_path, metadata_path, output_path):
    # 1. Load the datasets
    print("Loading data...")
    try:
        df_main = pd.read_csv(main_data_path)
        df_meta = pd.read_csv(metadata_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. Verify alignment
    # The prompt implies the files are row-aligned (index 0 in main matches index 0 in meta)
    if len(df_main) != len(df_meta):
        print(f"Warning: Files have different lengths! (Main: {len(df_main)}, Meta: {len(df_meta)})")
        print("Proceeding based on index alignment, but please verify your data integrity.")

    # 3. Define the patterns
    # The incorrect pattern found in SC3 lipids (looks like SC2)
    wrong_pattern = "SCCC(C)C" 
    # The correct pattern for SC3 lipids (based on your instruction "instead of SCCC")
    correct_pattern = "SCCC"

    # 4. Find indices where Lipid_name contains "SC3"
    # We create a boolean mask from the metadata file
    sc3_mask = df_meta['Lipid_name'].astype(str).str.contains('SC3', na=False)
    
    # Calculate how many rows will be affected
    affected_count = sc3_mask.sum()
    print(f"Found {affected_count} rows containing 'SC3'.")

    if affected_count > 0:
        # 5. Apply the fix only to the identified rows in the main_data
        # We use regex=False to ensure parentheses are treated as literal characters
        df_main.loc[sc3_mask, 'smiles'] = df_main.loc[sc3_mask, 'smiles'].str.replace(
            wrong_pattern, 
            correct_pattern, 
            regex=False
        )
        print("Corrections applied.")
    else:
        print("No SC3 lipids found to correct.")

    # 6. Save the corrected data
    df_main.to_csv(output_path, index=False)
    print(f"Success! Corrected data saved to '{output_path}'")

# Execute the function
if __name__ == "__main__":
    fix_sc3_smiles(
        main_data_path='main_data.csv',
        metadata_path='individual_metadata.csv',
        output_path='main_data_fixed.csv'
    )