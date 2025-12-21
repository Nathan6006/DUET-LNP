import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

def calculate_molecular_weight(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Descriptors.ExactMolWt(mol)
        else:
            return None
    except:
        return None

def add_molecular_weights(input_csv, output_csv):
    # Read the input CSV file
    df = pd.read_csv(input_csv)

    # Check if 'smiles' column exists in the DataFrame
    if 'smiles' not in df.columns:
        raise ValueError("'smiles' column not found in the input CSV file")

    # Calculate molecular weights and add to a new column
    df['molecular_weight'] = df['smiles'].apply(calculate_molecular_weight)

    # Write the output CSV file
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    input_csv = 'input_file.csv'  # Replace with your input CSV file name
    output_csv = 'output_file.csv'  # Replace with your desired output CSV file name

    add_molecular_weights(input_csv, output_csv)