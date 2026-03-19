#!/usr/bin/env python3

import os
import pandas as pd
from rdkit import Chem

base_folder = "../data/crossval_splits/zhu_del_B"


def canonicalize_smiles(s):
    if pd.isna(s):
        return s
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        print(f"[WARN] Invalid SMILES: {s}")
        return s
    return Chem.MolToSmiles(mol, canonical=True)


def process_file(path):
    print(f"Processing: {path}")
    df = pd.read_csv(path)

    if "smiles" not in df.columns:
        raise ValueError(f"'smiles' column not found in {path}")

    df["smiles"] = df["smiles"].apply(canonicalize_smiles)
    df.to_csv(path, index=False)


def main():
    # Cross-validation splits
    for i in range(5):
        cv_folder = os.path.join(base_folder, f"cv_{i}")
        for split in ["train.csv", "valid.csv"]:
            file_path = os.path.join(cv_folder, split)
            process_file(file_path)

    # Test set
    test_path = os.path.join(base_folder, "test", "test.csv")
    process_file(test_path)


if __name__ == "__main__":
    main()