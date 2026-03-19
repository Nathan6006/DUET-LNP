import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

"""
script to turn values for the molar ratios section of the data from weight ratios 
to the correct molar ratios. 
"""

# List of folders to process
# folders = [
#     "../data/data_files_to_merge/Liu_iphos"
#     # "../data/data_files_to_merge/Lin_peptide"
#     # "../data/data_files_to_merge/Han_a3,"
#     # "../data/data_files_to_merge/Han_branched,"
#     # "../data/data_files_to_merge/Han_amidine,"
#     # "../data/data_files_to_merge/Miller_Zwitter,"
#     # "../data/data_files_to_merge/3_tails",
#     # "../data/data_files_to_merge/Lee_unsat"
#     # "../data/data_files_to_merge/Farbiak_dendrimer_Hek",
#     # "../data/data_files_to_merge/Farbiak_dendrimer_HeLa",
#     # "../data/data_files_to_merge/Farbiak_dendrimer_igrov1",
#     # "../data/data_files_to_merge/Xue_CAD_LNP",
#     # "../data/data_files_to_merge/Yu_Aminoglycoside",
#     # "../data/data_files_to_merge/Zhang_Aminoglycoside"
# ]

folders = [
"../data_files/Whitehead_siRNA"
]

def calculate_molwt(smiles):
    """Calculate molecular weight from a SMILES string using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Descriptors.MolWt(mol)
        else:
            return None
    except Exception:
        return None

for folder in folders:
    print(f"Processing folder: {folder}")
    
    main_data_path = os.path.join(folder, "main_data.csv")
    metadata_path = os.path.join(folder, "individual_metadata.csv")
    output_path = os.path.join(folder, "individual_metadata.csv")
    
    # Load data
    main_df = pd.read_csv(main_data_path)
    metadata_df = pd.read_csv(metadata_path)
    
    # Calculate molecular weights for each SMILES
    molwts = main_df["smiles"].apply(calculate_molwt)
    
    # Add MolWt column to metadata (without overwriting existing columns)
    metadata_df["MolWt"] = molwts
    
    # Save new file
    metadata_df.to_csv(output_path, index=False)
    print(f"Saved updated metadata to {output_path}")