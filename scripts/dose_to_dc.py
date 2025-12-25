import os
import pandas as pd

# List of folders to process (FIXED commas)
folders = [
    "../data/data_files_to_merge/Liu_iphos"
    # "../data/data_files_to_merge/Lin_peptide",
    # "../data/data_files_to_merge/Han_a3",
    # "../data/data_files_to_merge/Han_branched",
    # "../data/data_files_to_merge/Han_amidine",
    # "../data/data_files_to_merge/Miller_Zwitter",
    # # "../data/data_files_to_merge/3_tails",
    # "../data/data_files_to_merge/Lee_unsat",
    # "../data/data_files_to_merge/Farbiak_dendrimer_Hek",
    # "../data/data_files_to_merge/Farbiak_dendrimer_HeLa",
    # "../data/data_files_to_merge/Farbiak_dendrimer_igrov1"
    # "../data/data_files_to_merge/Xue_CAD_LNP",
    # "../data/data_files_to_merge/Yu_Aminoglycoside",
    # "../data/data_files_to_merge/Zhang_Aminoglycoside"
]

# Cell densities aligned by folder index
# REPORTED IN N THOUSAND CELLS 
cell_den = [
    10       # Liu_iphos
    # 10,     # Lin_peptide
    # 5,   # Han_a3
    # 5,   # Han_branched
    # 5,   # Han_amidine
    # 10   # Miller_Zwitter
    # 5,   # 3_tails
    # 10,   # Lee_unsat
    #  10,   # Farbiak_dendrimer_Hek
    #  4,   # Farbiak_dendrimer_HeLa
    #  4,   # Farbiak_dendrimer_igrov1
    # 5,   # Xue_CAD_LNP
    # 10,   # Yu_Aminoglycoside
    # 15    # Zhang_Aminoglycoside
]

for i, folder in enumerate(folders):
    print(f"Processing folder: {folder}")
    
    metadata_path = os.path.join(folder, "formulations.csv")
    output_path = os.path.join(folder, "formulations.csv")
    
    df = pd.read_csv(metadata_path)
    
    # # Select correct cell density
    curr_cell_den = cell_den[i]
    
    df["Dose/Cells"] = df["Dosage"] / curr_cell_den
    df.drop(columns=["Dosage"], inplace=True)

    df["Dose/Cells"] = df["Dose/Cells"] * df["Ionizable_Lipid_to_mRNA_weight_ratio"]

    
    df.to_csv(output_path, index=False)
    print(f"Saved updated metadata to {output_path}")