import numpy as np 
import os
import pandas as pd  
from rdkit import Chem 
from rdkit.Chem import Descriptors 
import sys
import random
from sklearn.model_selection import train_test_split
from helpers import path_if_none, change_column_order, load_datapoints_tox_only


def merge_datasets(experiment_list, path_to_folders = '../data/data_files_to_merge', write_path = '../data'): 
    """Each folder contains the following files: 
    main_data.csv: a csv file with columns: 'smiles', which should contain the SMILES of the ionizable lipid, the activity measurements for that measurement
    If the same ionizable lipid is measured multiple times (i.e. for different properties, or transfection in vitro and in vivo) make separate rows, one for each measurement
    formulations.csv: a csv file with columns:
        Cationic_Lipid_Mol_Ratio
        Phospholipid_Mol_Ratio
        Cholesterol_Mol_Ratio
        PEG_Lipid_mol_ratio
        Cationic_Lipid_to_mRNA_weight_ratio
        Helper_lipid_ID
        If the dataset contains only 1 formulation in it: still provide the formulations data thing but with only one row; the model will copy it
        Otherwise match the row to the data in formulations.csv
    individual_metadata.csv: metadata that contains as many rows as main_data, each row is certain metadata for each lipid
        For example, could contain the identity (SMILES) of the amine to be used in training/test splits, or contain a dosage if the dataset includes varying dosage
        Either includes a column called "Sample_weight" with weight for each sample (each ROW, that is; weight for a kind of experiment will be determined separately)
            alternatively, default sample weight of 1
    experiment_metadata.csv: contains metadata about particular dataset. This includes:
        Experiment_ID: each experiment will be given a unique ID.
        There will be two ROWS and any number of columns

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
        print("370")
        experiment_list = list(experiment_df.Experiment_ID)
    y_val_cols = []
    helper_mol_weights = pd.read_csv(path_to_folders + '/Component_molecular_weights.csv')

    for folder in experiment_list:
        print("folder", folder)
        contin = False
        try:
            main_temp = pd.read_csv(path_to_folders + '/' + folder + '/main_data.csv')
            contin = True
        except:
            pass
        if contin:
            y_val_cols = y_val_cols + list(main_temp.columns)
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
                print(len(formulation_temp))
                to_raise = 'For experiment ID: ',folder,': Length of formulation file (', str(len(formulation_temp))#, ') doesn\'t match length of main datafile (',str(data_n),')'
                raise ValueError(to_raise)
            
            if len(individual_temp) == 1:
                individual_temp = pd.concat([individual_temp]*data_n,ignore_index = True)

            # Change formulations from mass to molar ratio
            form_cols = formulation_temp.columns
            mass_ratio_variables = ['Cationic_Lipid_Mass_Ratio','Phospholipid_Mass_Ratio','Cholesterol_Mass_Ratio','PEG_Lipid_Mass_Ratio']
            molar_ratio_variables = ['Cationic_Lipid_Mol_Ratio','Phospholipid_Mol_Ratio','Cholesterol_Mol_Ratio','PEG_Lipid_Mol_Ratio']
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
                    ion_lipid_moles = formulation_temp['Cationic_Lipid_Mass_Ratio'][i]/ion_lipid_mol_weight
                    phospholipid_moles = formulation_temp['Phospholipid_Mass_Ratio'][i]/phospholipid_mol_weight
                    cholesterol_moles = formulation_temp['Cholesterol_Mass_Ratio'][i]/cholesterol_mol_weight
                    PEG_lipid_moles = formulation_temp['PEG_Lipid_Mass_Ratio'][i]/PEG_lipid_mol_weight
                    mol_sum = ion_lipid_moles+phospholipid_moles+cholesterol_moles+PEG_lipid_moles
                    cat_lip_mol_fracs.append(float(ion_lipid_moles/mol_sum*100))
                    phos_mol_fracs.append(float(phospholipid_moles/mol_sum*100))
                    chol_mol_fracs.append(float(cholesterol_moles/mol_sum*100))
                    peg_lip_mol_fracs.append(float(PEG_lipid_moles/mol_sum*100))
                formulation_temp['Cationic_Lipid_Mol_Ratio'] = cat_lip_mol_fracs
                formulation_temp['Phospholipid_Mol_Ratio'] = phos_mol_fracs
                formulation_temp['Cholesterol_Mol_Ratio'] = chol_mol_fracs
                formulation_temp['PEG_Lipid_Mol_Ratio'] = peg_lip_mol_fracs

        
            if len(individual_temp) != data_n:
                print(len(individual_temp))
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
            # print(folder_df.columns)
            if 'Sample_weight' not in folder_df.columns:
                # print(folder)
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
    extra_x_variables = ['Cationic_Lipid_Mol_Ratio','Phospholipid_Mol_Ratio','Cholesterol_Mol_Ratio','PEG_Lipid_Mol_Ratio','Cationic_Lipid_to_mRNA_weight_ratio', 'Num_tails', 'Num_carbon_in_tail', 'Dosage', 'Exposure_time']
    # ADD HELPER LIPID ID
    # extra_x_categorical = ['Delivery_target','Helper_lipid_ID','Route_of_administration','Batch_or_individual_or_barcoded','screen_id']
    extra_x_categorical = ['Delivery_target','Helper_lipid_ID','Route_of_administration','Batch_or_individual_or_barcoded','Cargo_type','Model_type']

    # other_x_vals = ['Target_organ']
    # form_variables.append('Helper_lipid_ID')

    for x_cat in extra_x_categorical:
        dummies = pd.get_dummies(all_df[x_cat], prefix = x_cat)
        all_df = pd.concat([all_df, dummies], axis = 1)
        extra_x_variables = extra_x_variables + list(dummies.columns)

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

    norm_split_names, norm_del, norm_tox = generate_normalized_data(all_df)
    all_df['split_name_for_normalization'] = norm_split_names
    all_df.rename(columns = {'quantified_delivery':'unnormalized_delivery'}, inplace = True)
    all_df['quantified_delivery'] = norm_del
    all_df.rename(columns = {'quantified_toxicity':'unnormalized_toxicity'}, inplace = True)
    all_df['quantified_toxicity'] = norm_tox
    
    all_df = all_df.replace({True: 1.0, False: 0.0})
    path = write_path + '/all_data.csv'
    print("creating all_data")
    change_column_order(path, all_df)
    col_type_df.to_csv(write_path + '/col_type.csv', index = False)


def cv_split(split_spec_fname, path_to_folders='../data',
                       is_morgan=False, cv_fold=2, ultra_held_out_fraction=-1.0,
                       min_unique_vals=2.0, test_is_valid=False,
                       train_frac=0.7, valid_frac=.125, test_frac=0.175,
                       random_state=42):
    """
    Splits the dataset according to the specifications in split_spec_fname.
    Uses sklearn to create a single fixed test set and splits the rest into train/valid.
    Supports ultra held-out sets and maintains folder structure.

    Parameters:
        split_spec_fname: CSV specifying split/train rules
        path_to_folders: folder containing all_data.csv, crossval_split_specs, etc.
        is_morgan: whether to include Morgan fingerprints
        cv_fold: number of CV folds (1-5)
        ultra_held_out_fraction: fraction to hold out from all CV splits
        min_unique_vals: minimum unique values for splitting
        test_is_valid: if True, validation = test fold (used for in-silico screening)
        train_frac, valid_frac, test_frac: fractions for train/valid/test (must sum to 1)
        random_state: random seed for reproducibility
    """
    all_df = pd.read_csv(os.path.join(path_to_folders, 'all_data.csv'))
    split_df = pd.read_csv(os.path.join(path_to_folders, 'crossval_split_specs', split_spec_fname))
    col_types = pd.read_csv(os.path.join(path_to_folders, 'col_type.csv'))

    split_path = os.path.join(path_to_folders, 'crossval_splits', split_spec_fname[:-4])
    if ultra_held_out_fraction > 0:
        split_path += '_with_uho'
    if is_morgan:
        split_path += '_morgan'
    if test_is_valid:
        split_path += '_for_iss'

    if ultra_held_out_fraction > 0:
        path_if_none(os.path.join(split_path, 'ultra_held_out'))
    
    for i in range(cv_fold):
        path_if_none(os.path.join(split_path, f'cv_{i}'))

    perma_train = pd.DataFrame({})
    ultra_held_out = pd.DataFrame({})
    cv_data = pd.DataFrame({})

    for _, row in split_df.iterrows():
        dtypes = row['Data_types_for_component'].split(',')
        vals = row['Values'].split(',')
        df_to_concat = all_df.copy()

        for i, dtype in enumerate(dtypes):
            df_to_concat = df_to_concat[df_to_concat[dtype.strip()] == vals[i].strip()].reset_index(drop=True)

        values_to_split = df_to_concat[row['Data_type_for_split']]
        unique_values_to_split = list(set(values_to_split))

        if row['Train_or_split'].lower() == 'train':
            perma_train = pd.concat([perma_train, df_to_concat])
        elif row['Train_or_split'].lower() == 'split':
            cv_split_values, ultra_held_out_values = split_for_cv(unique_values_to_split, cv_fold, ultra_held_out_fraction)
            ultra_held_out = pd.concat([ultra_held_out, df_to_concat[df_to_concat[row['Data_type_for_split']].isin(ultra_held_out_values)]])
            cv_data = pd.concat([cv_data, df_to_concat[df_to_concat[row['Data_type_for_split']].isin(sum(cv_split_values, []))]])

    if ultra_held_out_fraction >= 0 and not ultra_held_out.empty:
        y, x, w, m = split_df_by_col_type(ultra_held_out, col_types)
        yxwm_to_csvs(y, x, w, m, split_path + '/ultra_held_out', 'test')

    if abs(train_frac + valid_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train_frac + valid_frac + test_frac must sum to 1.0")

    train_valid_df, test_df = train_test_split(
        cv_data, test_size=test_frac, random_state=random_state, shuffle=True
    )
    y, x, w, m = split_df_by_col_type(test_df, col_types)
    path_if_none(split_path + '/test')
    yxwm_to_csvs(y, x, w, m, split_path + '/test', 'test')
    
    valid_size = valid_frac / (train_frac + valid_frac)
    train_df, valid_df = train_test_split(
        train_valid_df, test_size=valid_size, random_state=random_state, shuffle=True
    )

    if not perma_train.empty:
        train_df = pd.concat([train_df, perma_train]).drop_duplicates().reset_index(drop=True)

    for i in range(cv_fold):
        for df, split_type in zip([valid_df, train_df], ['valid', 'train']):
            y, x, w, m = split_df_by_col_type(df, col_types)
            yxwm_to_csvs(y, x, w, m, split_path + '/cv_' +str(i), split_type)

def generate_normalized_data_(all_df, split_variables = ['Experiment_ID','Library_ID','Delivery_target','Model_type','Route_of_administration']):
    split_names = []
    norm_dict_del = {}
    norm_dict_tox = {}
    for index, row in all_df.iterrows():
        split_name = ''
        for vbl in split_variables:
            split_name = split_name + str(row[vbl])+'_'
        split_names.append(split_name[:-1])
    unique_split_names = set(split_names)
    for split_name in unique_split_names:
        data_subset = all_df[[spl==split_name for spl in split_names]]
        norm_dict_del[split_name] = (np.mean(data_subset['quantified_delivery']), np.std(data_subset['quantified_delivery']))
        norm_dict_tox[split_name] = (np.mean(data_subset['quantified_toxicity']), np.std(data_subset['quantified_toxicity']))
    norm_delivery = []
    norm_toxicity = []
    for i, row in all_df.iterrows():
        deli = row['quantified_delivery']
        split_del = split_names[i]
        std_del = norm_dict_del[split_del][1]
        mn_del = norm_dict_del[split_del][0]
        if pd.isna(deli):
            norm_delivery.append(np.nan)
        else:
            norm_delivery.append((float(deli)-mn_del)/std_del)

        tox = row['quantified_toxicity']
        # split_tox = split_names[i]
        # std_tox = norm_dict_tox[split_tox][1]
        # mn_tox = norm_dict_tox[split_tox][0]
        if pd.isna(tox):
            norm_toxicity.append(np.nan)
        else:
            normarlized = float(tox)/100
            if normarlized > 1:
                normarlized=1
            norm_toxicity.append(normarlized)

    return split_names, norm_delivery, norm_toxicity

def generate_normalized_data(all_df, split_variables = ['Experiment_ID','Library_ID','Delivery_target','Model_type','Route_of_administration']):
    split_names = []
    norm_dict_del = {}

    for _, row in all_df.iterrows():
        split_name = ''
        for vbl in split_variables:
            split_name = split_name + str(row[vbl])+'_'
        split_names.append(split_name[:-1])
    unique_split_names = set(split_names)

    for split_name in unique_split_names:
        data_subset = all_df[[spl==split_name for spl in split_names]]
        try:
            norm_dict_del[split_name] = (
                np.mean(data_subset['quantified_delivery']),
                np.std(data_subset['quantified_delivery'])
            )
        except Exception:
            norm_dict_del[split_name] = (np.nan, np.nan)

    norm_delivery = []
    norm_toxicity = []

    # Normalize row by row
    for i, row in all_df.iterrows():
        split_name = split_names[i]

        try:
            deli = row['quantified_delivery']
            mn_del, std_del = norm_dict_del[split_name]
            if pd.isna(deli) or pd.isna(std_del) or std_del == 0:
                norm_delivery.append(np.nan)
            else:
                norm_delivery.append((float(deli)-mn_del)/std_del)
        except Exception:
            print("no delivery", split_name)
            norm_delivery.append(np.nan)

        try:
            tox = row['quantified_toxicity']
            if pd.isna(tox):
                norm_toxicity.append(np.nan)
            else:
                normalized = float(tox)/100
                if normalized > 1:
                    normalized = 1
                norm_toxicity.append(normalized)
        except Exception:
            norm_toxicity.append(np.nan)

    return split_names, norm_delivery, norm_toxicity

# these functions only used in cv_split
def split_df_by_col_type(df,col_types):
    # Splits into 4 dataframes: y_vals, x_vals, sample_weights, metadata
    y_vals_cols = col_types.Column_name[col_types.Type == 'Y_val']
    x_vals_cols = col_types.Column_name[col_types.Type == 'X_val']
    xvals_df = df[x_vals_cols]
    weight_cols = col_types.Column_name[col_types.Type == 'Sample_weight']
    metadata_cols = col_types.Column_name[col_types.Type.isin(['Metadata','X_val_categorical'])]
    return df[y_vals_cols], xvals_df ,df[weight_cols] ,df[metadata_cols]

def yxwm_to_csvs(y, x, w, m, path,settype):
    # y is y values,  x is x values, w is weights, m is metadata
    # set_type is either train, valid, or test
    y.to_csv(path+'/'+settype+'.csv', index = False)
    x.to_csv(path + '/' + settype + '_extra_x.csv', index = False)
    w.to_csv(path + '/' + settype + '_weights.csv', index = False)
    m.to_csv(path + '/' + settype + '_metadata.csv', index = False)

def split_for_cv(vals,cv_fold, held_out_fraction):
    # randomly splits vals into cv_fold groups, plus held_out_fraction of vals are completely held out. So for example split_for_cv(vals,5,0.1) will hold out 10% of data and randomly put 18% into each of 5 folds
    random.seed(42)
    random.shuffle(vals)
    held_out_vals = vals[:int(held_out_fraction*len(vals))]
    cv_vals = vals[int(held_out_fraction*len(vals)):]
    return [cv_vals[i::cv_fold] for i in range(cv_fold)], held_out_vals


def main(argv):
    task_type = argv[1] 
    if task_type == 'split':
        split = argv[2]
        ultra_held_out = float(argv[3])
        is_morgan = False
        in_silico_screen = False
        cv_num = 2
        if len(argv)>4:
            for i, arg in enumerate(argv):
                if arg.replace('–', '-') == '--cv':
                    cv_num = int(argv[i+1])
                    print('this many folds: ',str(cv_num))
                if arg.replace('–', '-') == '--morgan':
                    is_morgan = True
                if arg.replace('–', '-') == '--in_silico':
                    in_silico_screen = True
        cv_split(split, cv_fold=cv_num, ultra_held_out_fraction = ultra_held_out, is_morgan = is_morgan, test_is_valid = in_silico_screen)
    
    elif task_type == 'merge':
        print("merge")
        merge_datasets(None)



    
if __name__ == '__main__':
    main(sys.argv)