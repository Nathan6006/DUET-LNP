import numpy as np 
import os
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_curve, roc_auc_score
from train_multitask import train_multitask_model, get_base_args, optimize_hyperparameters, train_hyperparam_optimized_model
from predict_multitask_from_json import predict_multitask_from_json, get_base_predict_args
from rdkit import Chem
import matplotlib.pyplot as plt
import scipy.stats
import json
import sys

def add_mass(fname):
	exp_df = pd.read_csv(fname)
	all_weights = []
	for smile in exp_df.smiles:
		smile = smile.replace('H','')
		mol = Chem.MolFromSmiles(smile)
		all_weights.append(Chem.Descriptors.MolWt(mol))
	exp_df['MolWt'] = all_weights
	exp_df.to_csv(fname[:-4]+'_with_mass.csv', index = False)

add_mass('Data/Multitask_data/All_datasets/Luke_Raj_Branched_ester/Structure_generation/in_silico_screen_structures.csv')