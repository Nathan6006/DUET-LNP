import os 
import pandas as pd
from chemprop import data 


# general helper functions 
def path_if_none(newpath):
    if not os.path.exists(newpath):
        os.makedirs(newpath)

def load_datapoints(smiles_csv, extra_csv, smiles_column='smiles', target_columns = ["quantified_delivery", "quantified_toxicity"]):
    df_smi = pd.read_csv(smiles_csv)
    df_extra = pd.read_csv(extra_csv)

    smis = df_smi[smiles_column].values
    ys = df_smi[target_columns].values
    extra_features = df_extra.to_numpy(dtype=float)

    datapoints = [
        data.MoleculeDatapoint.from_smi(smi, y, x_d=xf)
        for smi, y, xf in zip(smis, ys, extra_features)
    ]
    return datapoints

def change_column_order(path, all_df, first_cols = ['smiles','quantified_delivery','unnormalized_delivery','quantified_toxicity','unnormalized_toxicity']):
    other_cols = [col for col in all_df.columns if col not in first_cols]
    all_df = all_df[first_cols + other_cols]
    all_df.to_csv(path, index=False)



def load_datapoints_tox_only(smiles_csv, extra_csv, smiles_column='smiles', target_columns = ["quantified_toxicity"]):
    df_smi = pd.read_csv(smiles_csv)
    df_extra = pd.read_csv(extra_csv)

    smis = df_smi[smiles_column].values
    ys = df_smi[target_columns].values
    extra_features = df_extra.to_numpy(dtype=float)

    datapoints = [
        data.MoleculeDatapoint.from_smi(smi, y, x_d=xf)
        for smi, y, xf in zip(smis, ys, extra_features)
    ]
    return datapoints

def load_datapoints_rf(smiles_csv, extra_csv, smiles_column='smiles',
                       target_columns=["quantified_toxicity"]):
    df_smi = pd.read_csv(smiles_csv)
    df_extra = pd.read_csv(extra_csv)

    smis = df_smi[smiles_column].values
    ys = df_smi[target_columns].values
    extra_features = df_extra.to_numpy(dtype=float)

    datapoints = []
    for smi, y, xf in zip(smis, ys, extra_features):
        datapoints.append({
            "smiles": smi,
            "y": y,
            "x_d": xf
        })
    return datapoints
