import os
import pandas as pd

# List of folders to process 
folders = [
    "../data_files/AGILE"
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

dataset_info = {
    # "AGILE": {"cell_den": 10, "ng_nucleic_acid": 100},
    # "Liu_iphos": {"cell_den": 10, "ng_nucleic_acid": 50},
    # "Lin_peptide": {"cell_den": 10, "ng_nucleic_acid": 80},
    # "Han_a3": {"cell_den": 5, "ng_nucleic_acid": None},
    # "Han_branched": {"cell_den": 5, "ng_nucleic_acid": 15},
    # "Han_amidine": {"cell_den": 5, "ng_nucleic_acid": 15},
    # "Miller_Zwitter": {"cell_den": 10, "ng_nucleic_acid": 100},
    "Lee_unsat": {"cell_den": 10, "ng_nucleic_acid": 25},
    # "Farbiak_dendrimer_Hek": {"cell_den": 10, "ng_nucleic_acid": None},
    # "Farbiak_dendrimer_HeLa": {"cell_den": 4, "ng_nucleic_acid": None},
    # "Farbiak_dendrimer_igrov1": {"cell_den": 4, "ng_nucleic_acid": None},
    "Xue_CAD_LNP": {"cell_den": 5, "ng_nucleic_acid": 10},
    "Yu_Aminoglycoside": {"cell_den": 10, "ng_nucleic_acid": 25},
    "Zhang_Aminoglycoside": {"cell_den": 15, "ng_nucleic_acid": 50},
}

for i, folder in enumerate(folders):
    print(f"Processing folder: {folder}")
    
    metadata_path = os.path.join(folder, "formulations.csv")
    output_path = os.path.join(folder, "formulations.csv")
    
    df = pd.read_csv(metadata_path)
    
    # # Select correct cell density
    curr_cell_den = cell_den[i]
    
    df["Dose/Cells"] = df["Dosage"] / curr_cell_den
    df.drop(columns=["Dosage"], inplace=True)

    df["Lipid/Cells"] = df["Lipid/Cells"] * df["Ionizable_Lipid_to_mRNA_weight_ratio"]

    
    df.to_csv(output_path, index=False)
    print(f"Saved updated metadata to {output_path}")