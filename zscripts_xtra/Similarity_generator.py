import numpy as np 
import os
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_curve, roc_auc_score
from train_multitask import train_multitask_model, get_base_args, optimize_hyperparameters, train_hyperparam_optimized_model
from predict_multitask_from_json import predict_multitask_from_json, get_base_predict_args, predict_multitask_from_json_cv
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
# from rdkit.DataStructs import cDataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import Draw
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import scipy.stats
import json
import sys
import random

# Run conda activate chemprop before running this


eb66_smiles = 'CCCCNC(C(CCCC(O[C@H]1C[C@@H](C)CC[C@@H]1C(C)=C)=O)(N(CCCCN(C)C)C(CCCCC(OC(CCCCCCCC)CCCCCCCC)=O)=O)CCCC(O[C@H]2C[C@@H](C)CC[C@@H]2C(C)=C)=O)=O' #This is EB66
best_eb66_match = 'N(C(CCCCCOC(=O)CCCCCCC/C=C\CCCCCCCC)C(=O)NC(C)(C)C)(C(=O)CCCCC(=O)OC(CCCCCCCC)CCCCCCCC)CCCN(C)C'
fo32_smiles = 'O=C(CCCC(CCCC(OCCC#CCCCCCC)=O)(C(NCCCC)=O)N(CCCN(C)C)C(CCCCC(OC(CCCCCCCC)CCCCCCCC)=O)=O)OCCC#CCCCCCC' #This is FO-32
best_fo32_match = 'N(C(CCCCCOC(=O)CCCCCCCCCCCCCCC)C(=O)NC(C)(C)C)(C(=O)CCCCC(=O)OC(CCCCCCCC)CCCCCCCC)CCCN(C)C'
fo35_smiles = 'CCCCNC(C(CCCC(OCC1=CCC(C(C)=C)CC1)=O)(N(CCCCN(CC)CC)C(CCCCC(OC(CCCCCCCC)CCCCCCCC)=O)=O)CCCC(OCC2=CCC(C(C)=C)CC2)=O)=O' #This is FO-35
best_fo35_match = 'N(C(CCCCCOC(=O)CCCCCCC/C=C\C/C=C\CCCCC)C(=O)NCCCC)(C(=O)CCCCC(=O)OC(CCCCCCCC)CCCCCCCC)CCCN1CCCCC1CC'
fo35_ketone_smiles = 'OCC1=CCC(C(C)=C)CC1' #The FO-35 ketone
best_fo35_ketone_match = 'O=C(OCCCCCCCC/C=C\CC(CCCCCC)OC(CCCC1CCCCC1)=O)CCN(CCC(OCCCCCCCC/C=C\CC(CCCCCC)OC(CCCC2CCCCC2)=O)=O)CCCN(CC)CC'
eb66_ketone_smiles = 'O[C@H]1C[C@@H](C)CC[C@@H]1C(C)=C' #The EB66 ketone
best_eb66_ketone_match = 'CCCCCCCC/C=C\CCCCCCCC(OCCCCCC(C(CNCCCO)CCCCOC(CCCCCCC/C=C\CCCCCCCC)=O)O)=O'
fo32_ketone_smiles = 'OCCC#CCCCCCC' #The FO-32 ketone
best_fo32_ketone_match = 'CN(CCC(=O)OCCCCCCCC/C=C\CC(CCCCCC)OC(OCCC#CCCCCCC)=O)CCO'

def get_closest_match(smiles,other_smiles):
	fpgen = rdFingerprintGenerator.GetRDKitFPGenerator()
	target_fp = fpgen.GetFingerprint(Chem.MolFromSmiles(smiles_to_check))
	fps = [fpgen.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in other_smiles]
	similarities = [DataStructs.TanimotoSimilarity(target_fp,fp) for fp in fps]

	ind = np.argmax(similarities)
	print('Tanimoto similarity of: ',similarities[ind])
	print('SMILES: ',other_smiles[ind])
	print('Index: ',ind)

def get_closest_distance_equivalent(target_sim,all_smiles):
	fpgen = rdFingerprintGenerator.GetRDKitFPGenerator()
	fps = [fpgen.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in all_smiles]

	closest_pair = (None, None)
	best_sim = 1
	closest_distance = float('inf')
	n = len(fps)

	for i in range(n):
		for j in range(i+1, n):
			sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
			distance = abs(sim - target_sim)
            
			if distance < closest_distance:
				closest_distance = distance
				best_sim = sim
				closest_pair = (all_smiles[i], all_smiles[j])
	print('Distance for closest pair ',best_sim)
	return closest_pair


def plot_best_alignment(smiles1,smiles2,fname):
	# Load your molecules
	# For the example, let's use Benzene and Pyridine
	mol1 = Chem.MolFromSmiles(smiles1) 
	mol2 = Chem.MolFromSmiles(smiles2)

	# Compute 2D coordinates
	rdDepictor.Compute2DCoords(mol1)
	rdDepictor.Compute2DCoords(mol2)

	# Get substructure match
	match = mol2.GetSubstructMatch(mol1)
	# Align to the first molecule
	rdDepictor.GenerateDepictionMatching2DStructure(mol2,mol1,match)

	# Draw molecules
	drawer = rdMolDraw2D.MolDraw2DCairo(500, 500) # Or MolDraw2DSVG to generate SVGs
	drawer.DrawMolecule(mol1,highlightAtoms=[])
	drawer.DrawMolecule(mol2,highlightAtoms=match)
	drawer.FinishDrawing()

	# Save and display the image
	with open(fname+'_alignment.png', 'wb') as f:
	    f.write(drawer.GetDrawingText())

	Image(filename=fname+'_alignment.png')

# For EB66, plot best alignment:
plot_best_alignment(eb66_smiles,best_eb66_match,'Alignments_for_paper_writeup/EB66')



# other_smiles = pd.read_csv('Data/Multitask_data/All_datasets/all_data_old.csv')['smiles']
# # print(len(other_smiles))
# other_smiles = list(set(other_smiles))
# # print(len(other_smiles))
# # print(other_smiles[:5])

# print('For EB66: ',get_closest_distance_equivalent(0.6456996148908858,other_smiles)) #For EB66
# print('For FO-32: ',get_closest_distance_equivalent(0.7488855869242199,other_smiles)) #For FO-32
# print('For FO-35: ',get_closest_distance_equivalent(0.6563706563706564,other_smiles)) #For FO-35

# get_closest_match('OCC1=CCC(C(C)=C)CC1',other_smiles)


# Closest equivalents to similarity of 0.6456996148908858 (EB66):
# Distance of: 
# SMILES 1: CCN(CC)CCN(CCC(=O)OCCCCCCCC/C=C\\CC(CCCCCC)OC(OCCCCCCCC/C=C\\C/C=C\\CCCCC)=O)CCN(CC)CC
# SMILES 2: OC[C@@H]1CCCN(CCC(=O)OCCCCCCCC/C=C\\CC(CCCCCC)OC(OCCC#CCCCCCC)=O)1

# Closest equivalents to similarity of 0.7488855869242199 (FO-32):
# Distance of: 
# SMILES 1: OCCOCCN1CCN(CCC(=O)OCCCCCCCC/C=C\\CC(CCCCCC)OC(OCCC#CCCCCCC)=O)CC1
# SMILES 2: N(CCC(=O)OCCCCCCCC/C=C\\CC(CCCCCC)OC(OCC#CCCC)=O)(CCC(=O)OCCCCCCCC/C=C\\CC(CCCCCC)OC(OCC#CCCC)=O)CCO

# Closest equivalents to similarity of 0.6563706563706564 (FO-35):
# Distance of: 
# SMILES 1: N(CCN(CCOP(=O)([O-])OCCCCCCCCCCCC)C)(CCN(CCOP(=O)([O-])OCCCCCCCCCCCC)C)(CCN([H])C)
# SMILES 2: N(CCOP(=O)([O-])OCCCCCCCCCCCCCC)CCCN([H])CCCCN([H])CCCN(CCOP(=O)([O-])OCCCCCCCCCCCCCC)

# EB66:
# Tanimoto similarity of:  0.6456996148908858
# SMILES:  N(C(CCCCCOC(=O)CCCCCCC/C=C\CCCCCCCC)C(=O)NC(C)(C)C)(C(=O)CCCCC(=O)OC(CCCCCCCC)CCCCCCCC)CCCN(C)C

# EB-66 ketone:
# Tanimoto similarity of:  0.3151515151515151
# SMILES:  CCCCCCCC/C=C\CCCCCCCC(OCCCCCC(C(CNCCCO)CCCCOC(CCCCCCC/C=C\CCCCCCCC)=O)O)=O

# FO-32:
# Tanimoto similarity of:  0.7488855869242199
# SMILES:  N(C(CCCCCOC(=O)CCCCCCCCCCCCCCC)C(=O)NC(C)(C)C)(C(=O)CCCCC(=O)OC(CCCCCCCC)CCCCCCCC)CCCN(C)C

# FO-32 ketone:
# Tanimoto similarity of:  0.13636363636363635
# SMILES:   CN(CCC(=O)OCCCCCCCC/C=C\CC(CCCCCC)OC(OCCC#CCCCCCC)=O)CCO

# FO-35:
# Tanimoto similarity of:  0.6563706563706564
# SMILES:  N(C(CCCCCOC(=O)CCCCCCC/C=C\C/C=C\CCCCC)C(=O)NCCCC)(C(=O)CCCCC(=O)OC(CCCCCCCC)CCCCCCCC)CCCN1CCCCC1CC

# FO-35 ketone:
# Tanimoto similarity of:  0.22815533980582525
# SMILES:  O=C(OCCCCCCCC/C=C\CC(CCCCCC)OC(CCCC1CCCCC1)=O)CCN(CCC(OCCCCCCCC/C=C\CC(CCCCCC)OC(CCCC2CCCCC2)=O)=O)CCCN(CC)CC




