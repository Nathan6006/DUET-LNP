import numpy as np 
import os
import pandas as pd  
from rdkit import Chem 
from rdkit.Chem import Descriptors 
from helpers import path_if_none, change_column_order
from sklearn.utils.class_weight import compute_class_weight

"""
script for merging data from individual folders into all_data.csv
"""

def merge_datasets(experiment_list, path_to_folders = '../data/data_files_to_merge', write_path = '../data'): 
 
    """Each folder contains the following files: 
    main_data.csv: a csv file with columns: 'smiles', which should contain the SMILES of the ionizable lipid, the activity measurements for that measurement
    If the same ionizable lipid is measured multiple times (i.e. for different properties, or transfection in vitro and in vivo) make separate rows, one for each measurement
    formulations.csv: a csv file with columns:
        Ionizable_Lipid_Mol_Ratio
        Phospholipid_Mol_Ratio
        Cholesterol_Mol_Ratio
        PEG_Lipid_mol_ratio
        Ionizable_Lipid_to_mRNA_weight_ratio
        Helper_lipid_ID
        If the dataset contains only 1 formulation in it: still provide the formulations data thing but with only one row; the model will copy it
        Otherwise match the row to the data in formulations.csv
    individual_metadata.csv: metadata that contains as many rows as main_data, each row is certain metadata for each lipid
        For example, could contain the identity (SMILES) of the amine to be used in training/test splits, or contain a dosage if the dataset includes varying dosage
        Either includes a column called "Sample_weight" with weight for each sample (each ROW, that is; weight for a kind of experiment will be determined separately)
            alternatively, default sample weight of 1
    experiment_metadata.csv: contains metadata about particular dataset. This includes:
        Experiment_ID: each experiment will be given a unique ID.

    Based on these files, Merge_datasets will merge all the datasets into one dataset. In particular, it will output 2 files:
        all_merged.csv: each row  will contain all the data for a measurement (SMILES, info on dose/formulation/etc, metadata, sample weights, activity value)
        col_type.csv: two columns, column name and type. Four types: Y_val, X_val, X_val_cat (categorical X value), Metadata, Sample_weight

    Some metadata columns that should be held consistent, in terms of names:
        Purity ("Pure" or "Crude")
        ng_dose (for the dose, duh)
        Sample_weight
        Amine_SMILES
        Tail_SMILES
        Library_ID
        Experimenter_ID
        Experiment_ID
        Cargo (siRNA, DNA, mRNA, RNP are probably the relevant 4 options)
        Model_type (either the cell type or the name of the animal (probably "mouse"))"""
    
    all_df = pd.DataFrame({})
    col_type = {'Column_name':[],'Type':[]}
    experiment_df = pd.read_csv(path_to_folders + '/experiment_metadata.csv')
    if experiment_list == None:
        experiment_list = list(experiment_df.Experiment_ID)
    y_val_cols = []
    helper_mol_weights = pd.read_csv(path_to_folders + '/Component_molecular_weights.csv')

    for folder in experiment_list:
        print("Creating folder:", folder)
        contin = False
        try:
            main_temp = pd.read_csv(path_to_folders + '/' + folder + '/main_data.csv')
            contin = True
        except:
            pass
        if contin:
            # y_val_cols = y_val_cols + list(main_temp.columns)
            y_val_cols = ["smiles", "toxicity_class"]
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

            # Change formulations from mass to molar ratio
            form_cols = formulation_temp.columns
            mass_ratio_variables = ['Ionizable_Lipid_Mass_Ratio','Phospholipid_Mass_Ratio','Cholesterol_Mass_Ratio','PEG_Lipid_Mass_Ratio']
            molar_ratio_variables = ['Ionizable_Lipid_Mol_Ratio','Phospholipid_Mol_Ratio','Cholesterol_Mol_Ratio','PEG_Lipid_Mol_Ratio']
            mass_count = 0
            molar_count = 0
            for col in form_cols:
                if col in mass_ratio_variables:
                    mass_count += 1
                elif col in molar_ratio_variables:
                    molar_count += 1
            if mass_count>0 and molar_count>0:
                raise ValueError('For experiment ID: ',folder,': Formulation information includes both mass and molar ratios.')
            elif mass_count<4 and molar_count<4:
                raise ValueError('For experiment ID: ',folder,': Formulation not completely specified, mass count: ',mass_count,', molar count: ',molar_count)
            elif mass_count == 4:
                cat_lip_mol_fracs = []
                phos_mol_fracs = []
                chol_mol_fracs = []
                peg_lip_mol_fracs = []
                # Change mass ratios to weight ratios
                for i in range(len(formulation_temp)):
                    phos_id = formulation_temp['Helper_lipid_ID'][i]
                    ion_lipid_mol = Chem.MolFromSmiles(main_temp['smiles'][i])
                    ion_lipid_mol_weight = Chem.Descriptors.MolWt(ion_lipid_mol)
                    phospholipid_mol_weight = helper_mol_weights[phos_id][0]
                    cholesterol_mol_weight = helper_mol_weights['Cholesterol']
                    PEG_lipid_mol_weight = helper_mol_weights['C14-PEG2000']
                    ion_lipid_moles = formulation_temp['Ionizable_Lipid_Mass_Ratio'][i]/ion_lipid_mol_weight
                    phospholipid_moles = formulation_temp['Phospholipid_Mass_Ratio'][i]/phospholipid_mol_weight
                    cholesterol_moles = formulation_temp['Cholesterol_Mass_Ratio'][i]/cholesterol_mol_weight
                    PEG_lipid_moles = formulation_temp['PEG_Lipid_Mass_Ratio'][i]/PEG_lipid_mol_weight
                    mol_sum = ion_lipid_moles+phospholipid_moles+cholesterol_moles+PEG_lipid_moles
                    cat_lip_mol_fracs.append(float(ion_lipid_moles/mol_sum*100))
                    phos_mol_fracs.append(float(phospholipid_moles/mol_sum*100))
                    chol_mol_fracs.append(float(cholesterol_moles/mol_sum*100))
                    peg_lip_mol_fracs.append(float(PEG_lipid_moles/mol_sum*100))
                formulation_temp['Ionizable_Lipid_Mol_Ratio'] = cat_lip_mol_fracs
                formulation_temp['Phospholipid_Mol_Ratio'] = phos_mol_fracs
                formulation_temp['Cholesterol_Mol_Ratio'] = chol_mol_fracs
                formulation_temp['PEG_Lipid_Mol_Ratio'] = peg_lip_mol_fracs

        
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
                # folder_df['Sample_weight'] = [float(folder_df.Experiment_weight[i])/list(folder_df.smiles).count(smile) for i,smile in enumerate(folder_df.smiles)]
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
    extra_x_variables = ['Ionizable_Lipid_Mol_Ratio','Phospholipid_Mol_Ratio','Cholesterol_Mol_Ratio','PEG_Lipid_Mol_Ratio','Ionizable_Lipid_to_mRNA_weight_ratio', 'Num_tails', 'Num_carbon_in_tail', 'Dose/Cells', 'MolWt'] # no exposure time for now since its always 24

    # extra_x_categorical = ['Delivery_target','Helper_lipid_ID','Route_of_administration','Batch_or_individual_or_barcoded','Cargo_type','Model_type']
    extra_x_categorical = ['Helper_lipid_ID','Cargo_type','Model_type']

    # other_x_vals = ['Target_organ']
    # form_variables.append('Helper_lipid_ID')

    for x_cat in extra_x_categorical:
        dummies = pd.get_dummies(all_df[x_cat], prefix = x_cat)
        all_df = pd.concat([all_df, dummies], axis = 1)
        extra_x_variables = extra_x_variables + list(dummies.columns)

    # Update the unpacking to include the new ohe_df
    split_names, class_values, ohe_df = generate_classes(all_df)
    
    all_df['split_name_for_normalization'] = split_names
    all_df.rename(columns = {'quantified_toxicity':'exact_toxicity'}, inplace = True)
    
    # Add the OHE columns to the main dataframe
    all_df = pd.concat([all_df, ohe_df], axis=1)
    
    # Keep the original integer class for weighting purposes
    all_df['toxicity_class'] = class_values 
    
    # Register the OHE columns as Y values
    y_val_cols = list(ohe_df.columns) 
    y_val_cols.append("smiles")
    all_df = generate_weights(all_df)

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

    
    # all_df = all_df.replace({True: 1.0, False: 0.0})    
    all_df = all_df.where(all_df != True, 1.0).where(all_df != False, 0.0)
    all_df["MolWt"] = np.log1p(all_df["MolWt"])
    all_df["Dose/Cells"] = np.log1p(all_df["Dose/Cells"])


    path = write_path + '/all_data.csv'
    print("creating all_data")
    change_column_order(path, all_df)
    col_type_df.to_csv(write_path + '/col_type.csv', index = False)

def generate_classes(all_df, split_variables = ['Experiment_ID']):
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
                # if tox > 90: class_value = 0
                # elif 90 >= tox > 80: class_value = 1
                # elif 80 >= tox > 70: class_value = 2
                # else:
                #     class_value = 3
                if tox > 80: class_value = 0
                elif 80 >= tox > 70: class_value = 1
                else:
                    class_value = 2
                
                classified_toxicity.append(class_value)
        except Exception:
            classified_toxicity.append(np.nan)

    # Create dummies with prefix so we can identify them later
    ohe_df = pd.get_dummies(classified_toxicity, prefix='class', dummy_na=False)
    # Convert True/False to 1.0/0.0
    ohe_df = ohe_df.astype(float) 
    
    # We still return classified_toxicity as a list/series for the weight calculation
    return split_names, classified_toxicity, ohe_df

def generate_weights(all_df):
    """
    Adjusts the 'weights' column in the dataframe based on the distribution
    of 'toxicity_class' (0, 1, 2).
        New_Weight = Old_Weight * Class_Balancing_Multiplier
    """
    unique_classes = np.unique(all_df['toxicity_class'])
    y = all_df['toxicity_class'].values
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=unique_classes, 
        y=y
    )
    weight_dict = dict(zip(unique_classes, class_weights))
    class_multipliers = all_df['toxicity_class'].map(weight_dict)
    all_df['Sample_weight'] = all_df['Sample_weight'] * class_multipliers

    return all_df

def main():
    merge_datasets(None)



    
if __name__ == '__main__':
    main()