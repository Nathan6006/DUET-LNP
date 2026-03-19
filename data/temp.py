#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

# Paths
zhou_path = Path("../data_files/Zhou_dendrimer")
lee_path = Path("../data_files/Lee_unsat")

zhou_main_file = zhou_path / "main_data.csv"
zhou_metadata_file = zhou_path / "individual_metadata.csv"
zhou_formulations_file = zhou_path / "formulations.csv"

lee_main_file = lee_path / "main_data.csv"

# Read SMILES columns
zhou_main = pd.read_csv(zhou_main_file)
lee_main = pd.read_csv(lee_main_file)

# Find duplicates
duplicates = zhou_main['smiles'].isin(lee_main['smiles'])

if duplicates.any():
    print(f"Found {duplicates.sum()} duplicates. Removing from Zhou files...")

    # Remove duplicates from Zhou files
    zhou_main_clean = zhou_main.loc[~duplicates].reset_index(drop=True)
    zhou_metadata = pd.read_csv(zhou_metadata_file)
    zhou_metadata_clean = zhou_metadata.loc[~duplicates].reset_index(drop=True)
    zhou_formulations = pd.read_csv(zhou_formulations_file)
    zhou_formulations_clean = zhou_formulations.loc[~duplicates].reset_index(drop=True)

    # Save cleaned files
    zhou_main_clean.to_csv(zhou_main_file, index=False)
    zhou_metadata_clean.to_csv(zhou_metadata_file, index=False)
    zhou_formulations_clean.to_csv(zhou_formulations_file, index=False)

    print("Cleaning complete.")
else:
    print("No duplicates found. No changes made.")