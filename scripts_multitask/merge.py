import numpy as np 
import os
import pandas as pd  
from rdkit import Chem 
from helpers import path_if_none, change_column_order

def z_score_normalize(df, col_name):
    if col_name in df.columns:
        df[f'unnormalized_{col_name.replace("quantified_", "")}'] = df[col_name]
        series = df[col_name]
        mean, std = series.mean(), series.std()
        
        if pd.isna(std) or std == 0:
            df[col_name] = 0.0
        else:
            df[col_name] = round(((series - mean) / std), 5)
    return df

def merge_datasets(experiment_list, path_to_folders='../data_files', write_path='../data'): 
    all_df = pd.DataFrame({})
    col_type = {'Column_name':[],'Type':[]}
    
    experiment_df = pd.read_csv(os.path.join(path_to_folders, 'experiment_metadata.csv'))
    if experiment_list is None:
        experiment_list = list(experiment_df.Experiment_ID)
    
    for folder in experiment_list:
        print("Processing:", folder)
        try:
            main_path = os.path.join(path_to_folders, folder, 'main_data.csv')
            main_temp = pd.read_csv(main_path)
        except FileNotFoundError:
            continue

        if 'Unnamed' in str(main_temp.columns):
            print(f'Warning: Unnamed columns in {folder}')

        # Normalize delivery per dataset
        main_temp = z_score_normalize(main_temp, 'quantified_delivery')
            
        data_n = len(main_temp)
        form_path = os.path.join(path_to_folders, folder, 'formulations.csv')
        formulation_temp = pd.read_csv(form_path)

        try:
            ind_path = os.path.join(path_to_folders, folder, 'individual_metadata.csv')
            individual_temp = pd.read_csv(ind_path)
        except FileNotFoundError:
            individual_temp = pd.DataFrame({})

        if len(formulation_temp) == 1:
            formulation_temp = pd.concat([formulation_temp]*data_n, ignore_index=True)
        elif len(formulation_temp) != data_n:
            raise ValueError(f'Formulation length mismatch in {folder}')
        
        if len(individual_temp) == 1:
            individual_temp = pd.concat([individual_temp]*data_n, ignore_index=True)
        if not individual_temp.empty and len(individual_temp) != data_n:
            raise ValueError(f'Individual metadata length mismatch in {folder}')

        experiment_temp = experiment_df[experiment_df.Experiment_ID == folder]
        experiment_temp = pd.concat([experiment_temp]*data_n, ignore_index=True).reset_index(drop=True)
        
        cols_to_drop = [c for c in experiment_temp.columns if c in individual_temp.columns]
        experiment_temp = experiment_temp.drop(columns=cols_to_drop)

        folder_df = pd.concat([main_temp, formulation_temp, individual_temp, experiment_temp], axis=1).reset_index(drop=True)
        
        if 'Sample_weight' not in folder_df.columns:
            folder_df['Sample_weight'] = [float(folder_df.Experiment_weight[i]) for i in range(len(folder_df))]
            
        all_df = pd.concat([all_df, folder_df], ignore_index=True)

    # Replacements
    replacements = {
        'im': 'intramuscular', 'iv': 'intravenous', 'a549': 'lung_epithelium',
        'bdmc': 'macrophage', 'bmdm': 'dendritic_cell', 'hela': 'generic_cell',
        'hek': 'generic_cell', 'igrov1': 'generic_cell'
    }
    all_df = all_df.replace(replacements)
    all_df['Model_type'] = all_df['Model_type'].replace('muscle', 'Mouse')

    # OHE and Column setup
    extra_x_variables = ['Ionizable_Lipid_Mol_Ratio','Phospholipid_Mol_Ratio','Cholesterol_Mol_Ratio',
                         'PEG_Lipid_Mol_Ratio','Ionizable_Lipid_to_mRNA_weight_ratio', 'Num_tails', 
                         'Num_carbon_in_tail', 'MolWt',  'num_unsaturated_cc_bonds', 'num_protonatable_nitrogens'] #'Lipid/Cells', 'mRNA/Cells']
    
    extra_x_categorical = ['Helper_lipid_ID','Cargo_type','Model_type']

    for x_cat in extra_x_categorical:
        dummies = pd.get_dummies(all_df[x_cat], prefix=x_cat)
        all_df = pd.concat([all_df, dummies], axis=1)
        extra_x_variables.extend(dummies.columns)

    # Generate Classes
    tox_classes, del_classes = generate_classes(all_df)
    all_df['toxicity_class'] = tox_classes
    all_df['delivery_class'] = del_classes

    # Finalize Targets
    if 'quantified_toxicity' in all_df.columns:
        all_df["unnormalized_toxicity"] = (all_df['quantified_toxicity']/100).clip(upper=1.0).round(5)
        all_df['quantified_toxicity'] = (all_df['quantified_toxicity'] / 100.0).clip(upper=1.0, lower=0.3).round(5)
    
    # def hinged_power(y, t=0.7, p=0.3):
    #     y = np.asarray(y)
    #     out = y.copy()
    #     mask = y < t
    #     out[mask] = t * (y[mask] / t) ** p
    #     return out

    # all_df["quantified_toxicity_hinged"] = hinged_power(
    #     all_df["quantified_toxicity"].values, t=0.7, p=0.5)

    y_val_cols = ["quantified_toxicity", "quantified_delivery", "smiles"]

    for column in all_df.columns:
        col_type['Column_name'].append(column)
        if column in y_val_cols:
            col_type['Type'].append('Y_val')
        elif column in extra_x_variables:
            col_type['Type'].append('X_val')
        elif column in extra_x_categorical:
            col_type['Type'].append('Metadata')
        elif column == 'Sample_weight':
            col_type['Type'].append('Sample_weight')
        else:
            col_type['Type'].append('Metadata')

    all_df = all_df.where(all_df != True, 1.0).where(all_df != False, 0.0)
    all_df["MolWt"] = np.log1p(all_df["MolWt"])
    all_df["Lipid/Cells"] = np.log1p(all_df["Lipid/Cells"])
    all_df["mRNA/Cells"] = np.log1p(all_df["mRNA/Cells"])

    print("Creating all_data.csv")
    change_column_order(os.path.join(write_path, 'all_data.csv'), all_df, first_cols=['quantified_toxicity', 'quantified_delivery','smiles'])
    pd.DataFrame(col_type).to_csv(os.path.join(write_path, 'col_types.csv'), index=False)

def generate_classes(all_df):
    """
    Generates integer classes for weighting both Toxicity and Delivery.
    """
    tox_classes = []
    del_classes = []
    
    for _, row in all_df.iterrows():
        # Toxicity Logic (Absolute thresholds)
        try:
            tox = row.get('quantified_toxicity', np.nan)
            if pd.isna(tox):
                tox_class = np.nan
            elif tox > 80: tox_class = 0
            elif tox > 70: tox_class = 1
            else: tox_class = 2
            tox_classes.append(tox_class)
        except:
            tox_classes.append(np.nan)

        # Delivery Logic (Z-score thresholds)
        # Class 0: High (> 1 std), Class 1: Med (-1 to 1 std), Class 2: Low (< -1 std)
        try:
            delivery = row.get('quantified_delivery', np.nan)
            if pd.isna(delivery):
                del_class = np.nan
            elif delivery > 1.0: del_class = 2
            elif delivery >= -1.0: del_class = 1
            else: del_class = 0
            del_classes.append(del_class)
        except:
            del_classes.append(np.nan)

    return tox_classes, del_classes

def main():
    merge_datasets(None)

if __name__ == '__main__':
    main()