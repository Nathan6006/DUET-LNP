import pandas as pd
from rdkit import Chem

def canonicalize_csv(input_file, output_file):
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)

    if 'smiles' not in df.columns:
        print("Error: 'smiles' column not found.")
        return

    def get_canonical(smiles):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol:
            # Returns the unique RDKit canonical string
            return Chem.MolToSmiles(mol)
        else:
            # Returns None if the SMILES is invalid
            return None

    print("Canonicalizing SMILES...")
    # Apply the function to the smiles column
    df['smiles'] = df['smiles'].apply(get_canonical)

    # Count invalid entries before dropping them
    initial_count = len(df)
    df = df.dropna(subset=['smiles'])
    final_count = len(df)

    # Save the cleaned and canonicalized data
    df.to_csv(output_file, index=False)
    
    print("-" * 30)
    print(f"Success! Processed {initial_count} rows.")
    if initial_count > final_count:
        print(f"Dropped {initial_count - final_count} invalid SMILES strings.")
    print(f"Canonicalized file saved as: {output_file}")

# Execute the update
canonicalize_csv('all_data.csv', 'all_data_canonical.csv')