import pandas as pd
import os
from rdkit import Chem

# --- Feature Engineering Logic ---

def get_lnp_features(smiles):

    # Default values for invalid SMILES
    default_vals = {'num_unsaturated_cc_bonds': 0, 'num_protonatable_nitrogens': 0}
    
    if pd.isna(smiles) or smiles == "":
        return default_vals

    mol = Chem.MolFromSmiles(str(smiles))
    
    if mol is None:
        return default_vals

    # Pattern: C=C
    # C=C
    cc_double = Chem.MolFromSmarts('[CX3]=[CX3]')
    # C#C
    cc_triple = Chem.MolFromSmarts('[CX2]#[CX2]')

    num_double = len(mol.GetSubstructMatches(cc_double))
    num_triple = len(mol.GetSubstructMatches(cc_triple))

    num_unsat_cc = num_double + num_triple
    # 2. Protonatable Nitrogens (Basic Amines)
    # Pattern: Trivalent Nitrogen, Not Amide, Not Thio/Phospho-amide, Not Aromatic
    basic_nitrogen_pattern = Chem.MolFromSmarts(
        '[N;!+;!$(NC=[O,S]);!$(N-[S,P]=[O,S]);!$(N=C-N);!$(N=C-O)]'
    )
    matches = mol.GetSubstructMatches(basic_nitrogen_pattern)
    num_nitrogens = len(matches)

    return {
        'num_unsaturated_cc_bonds': num_unsat_cc,
        'num_protonatable_nitrogens': num_nitrogens
    }

# --- Pipeline Logic ---

def process_experiments(base_path="../data_files"):
    
    # 1. Load the master experiment list
    master_file_path = os.path.join(base_path, "experiment_metadata.csv")
    
    if not os.path.exists(master_file_path):
        print(f"Error: Master file not found at {master_file_path}")
        return

    master_df = pd.read_csv(master_file_path)
    
    # Check if Experiment_ID column exists
    if 'Experiment_ID' not in master_df.columns:
        print("Error: 'Experiment_ID' column missing in experiment_metadata.csv")
        return

    # 2. Iterate through each experiment
    for exp_id in master_df['Experiment_ID']:
        print(f"Processing Experiment: {exp_id}...")
        
        # Define file paths
        exp_dir = os.path.join(base_path, str(exp_id))
        main_data_path = os.path.join(exp_dir, "main_data.csv")
        indiv_meta_path = os.path.join(exp_dir, "individual_metadata.csv")
        output_path = os.path.join(exp_dir, "individual_metadata.csv")

        # Check if files exist
        if not os.path.exists(main_data_path) or not os.path.exists(indiv_meta_path):
            print(f"  -> Skipping {exp_id}: Missing main_data.csv or individual_metadata.csv")
            continue

        try:
            # Load Data
            main_df = pd.read_csv(main_data_path)
            meta_df = pd.read_csv(indiv_meta_path)

            # Check if row counts match (crucial for index alignment)
            if len(main_df) != len(meta_df):
                print(f"  -> Warning: Row count mismatch in {exp_id}. Main: {len(main_df)}, Meta: {len(meta_df)}. Using index alignment.")

            # Check for smiles column
            # (Case-insensitive check for robustness)
            smiles_col = next((col for col in main_df.columns if col.lower() == 'smiles'), None)
            
            if not smiles_col:
                print(f"  -> Skipping {exp_id}: 'smiles' column not found in main_data.csv")
                continue

            # 3. Calculate Features
            # We apply the function row by row
            features_df = main_df[smiles_col].apply(lambda x: pd.Series(get_lnp_features(x)))

            # 4. Augment Metadata
            # This aligns by index automatically. 
            # We explicitly copy meta_df to avoid SettingWithCopy warnings
            new_meta_df = meta_df.copy()
            new_meta_df['num_unsaturated_cc_bonds'] = features_df['num_unsaturated_cc_bonds']
            new_meta_df['num_protonatable_nitrogens'] = features_df['num_protonatable_nitrogens']

            # 5. Save Output
            new_meta_df.to_csv(output_path, index=False)
            print(f"  -> Success: Saved to {output_path}")

        except Exception as e:
            print(f"  -> Error processing {exp_id}: {e}")

if __name__ == "__main__":
    process_experiments()