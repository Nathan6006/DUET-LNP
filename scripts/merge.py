import numpy as np 
import os
import pandas as pd  
from rdkit import Chem 
from helpers import path_if_none, change_column_order


"""
script for merging data from individual folders into all_data_regression.csv
"""

def merge_datasets(experiment_list, path_to_folders = '../data/data_files_to_merge', write_path = '../data'): 
 
    """
    Based on these files, Merge_datasets will merge all the datasets into one dataset. 
    Outputs 2 files:
        all_data_regression.csv: merged data with normalized regression targets
        col_types_regression.csv: column type definitions
    """
    
    all_df = pd.DataFrame({})
    col_type = {'Column_name':[],'Type':[]}
    experiment_df = pd.read_csv(path_to_folders + '/experiment_metadata.csv')
    if experiment_list == None:
        experiment_list = list(experiment_df.Experiment_ID)
    
    for folder in experiment_list:
        print("Creating folder:", folder)
        contin = False
        try:
            main_temp = pd.read_csv(path_to_folders + '/' + folder + '/main_data.csv')
            contin = True
        except:
            pass
        if contin:
            for col in main_temp.columns:
                if 'Unnamed' in col:
                    print('\n\n\nTHERE IS AN UNNAMED COLUMN IN FOLDER: ',folder,'\n\n')
            data_n = len(main_temp)
            formulation_temp = pd.read_csv(path_to_folders + '/' + folder + '/formulations.csv')

            try:
                individual_temp = pd.read_csv(path_to_folders + '/' + folder + '/individual_metadata.csv')
            except:
                individual_temp = pd.DataFrame({})
            if len(formulation_temp) == 1:
                formulation_temp = pd.concat([formulation_temp]*data_n,ignore_index = True)
            elif len(formulation_temp) != data_n:
                to_raise = 'For experiment ID: ',folder,': Length of formulation file (', str(len(formulation_temp))#, ') doesn\'t match length of main datafile (',str(data_n),')'
                raise ValueError(to_raise)
            
            if len(individual_temp) == 1:
                individual_temp = pd.concat([individual_temp]*data_n,ignore_index = True)

        
            if len(individual_temp) != data_n:
                raise ValueError('For experiment ID: ',folder,': Length of individual metadata file  (',len(individual_temp), ') doesn\'t match length of main datafile (',data_n,')')
            experiment_temp = experiment_df[experiment_df.Experiment_ID == folder]
            experiment_temp = pd.concat([experiment_temp]*data_n, ignore_index = True).reset_index(drop = True)
            to_drop = []
            for col in experiment_temp.columns:
                if col in individual_temp.columns:
                    print('Column ',col,' in experiment ID ',folder,'is being provided for each individual lipid.')
                    to_drop.append(col)
            experiment_temp = experiment_temp.drop(columns = to_drop)
            folder_df = pd.concat([main_temp, formulation_temp, individual_temp], axis = 1).reset_index(drop = True)
            folder_df = pd.concat([folder_df, experiment_temp], axis = 1)
            if 'Sample_weight' not in folder_df.columns:
                folder_df['Sample_weight'] = [float(folder_df.Experiment_weight[i]) for i,smile in enumerate(folder_df.smiles)]
            all_df = pd.concat([all_df,folder_df], ignore_index = True)
            
    # Make changes:
    all_df = all_df.replace('im','intramuscular')
    all_df = all_df.replace('iv','intravenous')
    all_df = all_df.replace('a549','lung_epithelium')
    all_df = all_df.replace('bdmc','macrophage')
    all_df = all_df.replace('bmdm','dendritic_cell')
    all_df = all_df.replace('hela','generic_cell')
    all_df = all_df.replace('hek','generic_cell')
    all_df = all_df.replace('igrov1','generic_cell')
    all_df = all_df.replace({'Model_type':'muscle'},'Mouse')


    # Make the column type dict
    extra_x_variables = ['Ionizable_Lipid_Mol_Ratio','Phospholipid_Mol_Ratio','Cholesterol_Mol_Ratio','PEG_Lipid_Mol_Ratio','Ionizable_Lipid_to_mRNA_weight_ratio', 'Num_tails', 'Num_carbon_in_tail', 'Dose/Cells', 'MolWt'] 
    extra_x_categorical = ['Helper_lipid_ID','Cargo_type','Model_type']

    for x_cat in extra_x_categorical:
        dummies = pd.get_dummies(all_df[x_cat], prefix = x_cat)
        all_df = pd.concat([all_df, dummies], axis = 1)
        extra_x_variables = extra_x_variables + list(dummies.columns)

    split_names, class_values = generate_classes(all_df)
        
    # Keep the integer class for weighting purposes
    all_df['toxicity_class'] = class_values 

    # 2. Handle the Regression Target
    # Normalize: divide by 100
    all_df["unnormalized_toxicity"] = (all_df['quantified_toxicity']/100).clip(upper=1.0)
    all_df['quantified_toxicity'] = (all_df['quantified_toxicity'] / 100.0).clip(upper=1.0, lower=0.3)

    y_val_cols = ["quantified_toxicity", "smiles"]

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

    col_type_df = pd.DataFrame(col_type)
  
    all_df = all_df.where(all_df != True, 1.0).where(all_df != False, 0.0)
    all_df["MolWt"] = np.log1p(all_df["MolWt"])
    all_df["Dose/Cells"] = np.log1p(all_df["Dose/Cells"])

    path = write_path + '/all_data_regression.csv'
    print("creating all_data_regression")
    change_column_order(path, all_df)
    col_type_df.to_csv(write_path + '/col_types_regression.csv', index = False)

def generate_classes(all_df, split_variables = ['Experiment_ID']):
    """
    Generates integer classes for weighting, but does NOT return OHE dataframe.
    """
    split_names = []
    for _, row in all_df.iterrows():
        split_name = ''
        for vbl in split_variables:
            split_name = split_name + str(row[vbl])+'_'
        split_names.append(split_name[:-1])

    classified_toxicity = []
    for i, row in all_df.iterrows():
        try:
            tox = row['quantified_toxicity']
            if pd.isna(tox):
                classified_toxicity.append(np.nan)
            else:
                if tox > 80: class_value = 0
                elif 80 >= tox > 70: class_value = 1
                else:
                    class_value = 2
                
                classified_toxicity.append(class_value)
        except Exception:
            classified_toxicity.append(np.nan)
    
    # Only return split names and the integer list for weighting
    return split_names, classified_toxicity


def main():
    merge_datasets(None)

if __name__ == '__main__':
    main()