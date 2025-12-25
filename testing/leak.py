import sys
import os
import pandas as pd

def check_leakage_and_export(split_root_path, all_data_path, metadata_col='Lipid_name'):
    """
    1. Checks for data leakage (Train vs Test, Train vs Valid).
    2. If leakage is found, extracts the full details of the leaked lipids.
    3. Saves a 'leakage_report.csv' with columns: 
       [Lipid_name, smiles, Experiement_ID, location]
    """
    print(f"--- Checking for Data Leakage in: {split_root_path} ---")
    
    # Load Source Data to get SMILES and Experiment_IDs later
    print(f"Loading reference data from {all_data_path}...")
    all_df = pd.read_csv(all_data_path)
    
    # Store leaked IDs and where they were found
    # format: { 'Lipid_Name': [ 'test', 'cv_0_train', 'cv_0_valid', ... ] }
    leaked_tracker = {}
    
    # --- 1. Load Global Test Set ---
    test_path = os.path.join(split_root_path, 'test')
    test_meta_file = os.path.join(test_path, 'test_metadata.csv')
    test_ids = set()
    
    if os.path.exists(test_meta_file):
        test_df = pd.read_csv(test_meta_file)
        test_ids = set(test_df[metadata_col].unique())
        
        # Log location for these IDs
        for lip in test_ids:
            leaked_tracker.setdefault(lip, []).append('test_set')
    else:
        print("No Global Test set found.")

    # --- 2. Iterate through CV Folds ---
    folds = [d for d in os.listdir(split_root_path) if d.startswith('cv_')]
    
    leaking_lipids_found = set()
    
    for fold in sorted(folds):
        fold_path = os.path.join(split_root_path, fold)
        
        try:
            train_df = pd.read_csv(os.path.join(fold_path, 'train_metadata.csv'))
            valid_df = pd.read_csv(os.path.join(fold_path, 'valid_metadata.csv'))
        except FileNotFoundError:
            continue

        train_ids = set(train_df[metadata_col].unique())
        valid_ids = set(valid_df[metadata_col].unique())

        # Log locations
        for lip in train_ids:
            leaked_tracker.setdefault(lip, []).append(f'{fold}_train')
        for lip in valid_ids:
            leaked_tracker.setdefault(lip, []).append(f'{fold}_valid')
            
        # CHECK 1: Train vs Validation (Inside Fold)
        tv_leak = train_ids.intersection(valid_ids)
        if tv_leak:
            print(f"[{fold}] CRITICAL: {len(tv_leak)} lipids overlapping Train/Valid")
            leaking_lipids_found.update(tv_leak)

        # CHECK 2: Train vs Global Test
        tt_leak = train_ids.intersection(test_ids)
        if tt_leak:
            print(f"[{fold}] CRITICAL: {len(tt_leak)} lipids overlapping Train/Test")
            leaking_lipids_found.update(tt_leak)

    # --- 3. Generate Report ---
    if not leaking_lipids_found:
        print("\nSUCCESS: No leakage detected.")
        return

    print(f"\nGeneratng report for {len(leaking_lipids_found)} unique leaked lipids...")
    
    report_rows = []
    
    # Filter all_data for only the leaked lipids
    leaked_subset = all_df[all_df[metadata_col].isin(leaking_lipids_found)].copy()
    
    for _, row in leaked_subset.iterrows():
        lip_name = row[metadata_col]
        
        # Get all locations where this lipid appeared in the split
        locations = leaked_tracker.get(lip_name, [])
        
        # We want one row per location found? Or just list all locations?
        # The prompt implies "the location... from where the train was from"
        # usually means we list the row for every place it appeared.
        
        # However, 'all_df' is the source. The split files are subsets.
        # If a lipid is in 'all_df', it might appear multiple times there too.
        # We will list the locations where it was FOUND in the split.
        
        location_str = "; ".join(sorted(set(locations)))
        
        report_rows.append({
            'Lipid_name': lip_name,
            'smiles': row.get('smiles', 'N/A'),
            'Experiment_ID': row.get('Experiment_ID', 'N/A'), # Ensure spelling matches your CSV
            'location': location_str
        })
        
    report_df = pd.DataFrame(report_rows)
    
    # Deduplicate if source had multiple rows for same lipid but same info
    report_df = report_df.drop_duplicates()
    
    out_file = os.path.join('leakage_report.csv')
    report_df.to_csv(out_file, index=False)
    print(f"Report saved to: {out_file}")

# --- USAGE ---

# --- USAGE ---
# Update this path to where your splits were saved
def main(argv):
    split = argv[1] 
    split_folder = f'../data/crossval_splits/{split}'
    all_data = '../data/all_data.csv'

    check_leakage_and_export(split_folder, all_data)

if __name__ == "__main__":
    main(sys.argv)