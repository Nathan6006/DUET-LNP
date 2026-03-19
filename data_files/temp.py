import os
import pandas as pd
from rdkit import Chem

EXPERIMENT_METADATA_PATH = "experiment_metadata.csv"
SMILES_COLUMN = "smiles"


def canonicalize_smiles(smiles):
    if pd.isna(smiles):
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles  # leave invalid SMILES unchanged
    return Chem.MolToSmiles(mol, canonical=True)


def main():
    metadata_df = pd.read_csv(EXPERIMENT_METADATA_PATH)

    for exp_id in metadata_df["Experiment_ID"]:
        main_data_path = os.path.join(str(exp_id), "main_data.csv")

        if not os.path.exists(main_data_path):
            print(f"Skipping {exp_id}: main_data.csv not found")
            continue

        df = pd.read_csv(main_data_path)

        if SMILES_COLUMN not in df.columns:
            print(f"Skipping {exp_id}: '{SMILES_COLUMN}' column not found")
            continue

        df[SMILES_COLUMN] = df[SMILES_COLUMN].apply(canonicalize_smiles)
        df.to_csv(main_data_path, index=False)

        print(f"Updated canonical SMILES for experiment {exp_id}")


if __name__ == "__main__":
    main()