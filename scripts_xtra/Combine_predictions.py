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
import seaborn as sns

def path_if_none(newpath):
	if not os.path.exists(newpath):
		os.makedirs(newpath)

def combine_predictions(header, splits, addon, conds, path_to_splits = 'Data/Multitask_data/All_datasets/Splits/', save_path = 'Data/Multitask_data/All_datasets/Pred_analysis/', toy = False):
	all_combined = pd.DataFrame({})
	all_averaged_combined = pd.DataFrame({})
	train_averaged_combined = pd.DataFrame({})
	# split_averaged_combined = pd.DataFrame({})
	header_path = save_path+header+addon
	path_if_none(header_path)
	pred_col_names = ['m'+str(i)+'_pred_quantified_delivery' for i in range(5)]
	# cmap_all = sns.color_palette('coolwarm')
	cmap_all = 'coolwarm'
	figsize = 9
	meta_done = False
	meta_cols = ['Lipid_name', 'Amine', 'Ketone', 'Isocyanide', 'Carboxylic_acid', 'smiles']
	if '4cr_a_' in header:
		meta_cols = ['Lipid_name', 'Amine', 'Aldehyde', 'Isocyanide', 'Carboxylic_acid', 'smiles']
	meta_df = pd.DataFrame({})
	for split in splits:
		all_split_df = pd.DataFrame({})
		averaged_split_df = pd.DataFrame({})
		split_path = header_path+'/'+split
		path_if_none(split_path)
		pred_files = []
		for fname in os.listdir(path_to_splits+split):
			if fname.startswith('Predicted_vs_actual'+header):
				for cond in conds:
					if cond in fname:
						pred_files.append(fname)
		# pred_files = [fname for fname in os.listdir(path_to_splits+split) if fname.startswith('Predicted_vs_actual'+header)]
		for fname in pred_files:
			split_plus_con_path = split_path + '/' + fname[19:]
			path_if_none(split_plus_con_path)
			to_add = pd.read_csv(path_to_splits+split+'/'+fname)
			if not meta_done:
				meta_done = True
				meta_df = to_add[meta_cols]
			to_add = to_add[pred_col_names]
			# print(to_add.head())
			if toy:
				to_add = to_add[:1000]
			# if not meta_done:
				# meta_df = to_add[meta_cols]
			preds_header = split+'_'+fname[19:]+'_'
			rename_dict = {}
			for i,pcn in enumerate(pred_col_names):
				rename_dict[pcn] = preds_header+str(i)
			to_add.rename(columns = rename_dict, inplace = True)
			plt.figure(figsize = (figsize, figsize))
			hm = sns.heatmap(to_add.corr(), annot = True, cmap = cmap_all, vmin = -1, vmax = 1, square = True)
			hm.figure.tight_layout()
			hm.set(xlabel='Model', ylabel='Model', title = 'Different training runs, '+split_plus_con_path[len(save_path)+len(header)+1:])
			plt.savefig(split_plus_con_path+'/all_split_corr.png')
			plt.close()
			all_split_df = pd.concat([all_split_df, to_add], axis = 1)
			averaged_split_df['Avg_'+split_plus_con_path[len(save_path)+len(header)+1:]] = to_add.mean(axis = 1)
		train_averaged_combined = pd.concat([train_averaged_combined,averaged_split_df], axis = 1)
		plt.figure(figsize = (figsize, figsize))
		plt.subplots_adjust(left = 0.5)
		# plt.tight_layout()
		hm = sns.heatmap(all_split_df.corr(), annot = False, cmap = cmap_all, vmin = -1, vmax = 1, square = True)
		hm.figure.tight_layout()
		hm.set(xlabel='Model', ylabel='Model', title = 'All training runs, split'+split_path[len(save_path):])
		plt.savefig(split_path+'/all_split_corr.png')
		plt.close()
		plt.figure(figsize = (figsize, figsize))
		hm = sns.heatmap(averaged_split_df.corr(), annot = True, cmap = cmap_all, vmin = -1, vmax = 1, square = True)
		hm.figure.tight_layout()
		hm.set(xlabel='Model', ylabel='Model', title = 'Averaged over training runs, split'+split_path[len(save_path):])
		plt.savefig(split_path+'/avg_split_corr.png')
		plt.close()
		all_combined = pd.concat([all_combined, all_split_df], axis = 1)
		all_averaged_combined[split+'_avg'] = all_split_df.mean(axis = 1)
	plt.figure(figsize = (figsize, figsize))
	hm = sns.heatmap(all_combined.corr(), annot = False, cmap = cmap_all, vmin = -1, vmax = 1, square = True)
	hm.figure.tight_layout()
	hm.set(xlabel='Model', ylabel='Model', title = 'All_training_runs, dataset '+header_path[len(save_path):])
	plt.savefig(header_path+'/all_model_corr.png')
	plt.close()
	plt.figure(figsize = (figsize, figsize))
	hm = sns.heatmap(train_averaged_combined.corr(), annot = False, cmap = cmap_all, vmin = -1, vmax = 1, square = True)
	hm.figure.tight_layout()
	hm.set(xlabel='Model', ylabel='Model', title = 'Split and condition comparison, dataset '+header_path[len(save_path):])
	plt.savefig(header_path+'/split_cond_avg_corr.png')
	plt.close()
	plt.figure(figsize = (figsize, figsize))
	hm = sns.heatmap(all_averaged_combined.corr(), annot = True, cmap = cmap_all, vmin = -1, vmax = 1, square = True)
	hm.figure.tight_layout()
	hm.set(xlabel='Model', ylabel='Model', title = 'Split comparison, dataset '+header_path[len(save_path):])
	plt.savefig(header_path+'/split_avg_corr.png')
	plt.close()
	sizedf = pd.DataFrame({'Size':[len(all_combined),len(all_combined.columns)]})
	sizedf.to_csv(header_path+'/dataset_size.csv', index = False)
	meta_df['ens_avg_pred'] = all_combined.mean(axis = 1)
	meta_df.sort_values(by = 'ens_avg_pred', ascending = False, inplace = True)
	top_1000 = meta_df.head(1000)
	meta_df.to_csv(header_path+'/metadata_and_avg.csv',index = False)
	top_1000.to_csv(header_path+'/metadata_and_avg_top_100.csv', index = False)
	all_combined.to_csv(header_path + '/all_combined_preds.csv', index = False)
	return all_combined
	# meta_df contains metadata (smiles, etc)
	# combined contains all the combined data

def combine_predictions_cv(header, splits, addon, conds, path_to_splits = 'Data/Multitask_data/All_datasets/crossval_splits/', save_path = 'Data/Multitask_data/All_datasets/cv_pred_analysis/', toy = False, crossval_splits = 5):
	all_combined = pd.DataFrame({})
	all_averaged_combined = pd.DataFrame({})
	train_averaged_combined = pd.DataFrame({})
	# split_averaged_combined = pd.DataFrame({})
	header_path = save_path+header+addon
	path_if_none(header_path)
	pred_col_names = ['m'+str(i)+'_pred_quantified_delivery' for i in range(crossval_splits)]
	# cmap_all = sns.color_palette('coolwarm')
	cmap_all = 'coolwarm'
	figsize = 9
	meta_done = False
	meta_cols = ['Lipid_name', 'Amine', 'Ketone', 'Isocyanide', 'Carboxylic_acid', 'smiles']
	if '4cr_a_' in header:
		meta_cols = ['Lipid_name', 'Amine', 'Aldehyde', 'Isocyanide', 'Carboxylic_acid', 'smiles']
	meta_df = pd.DataFrame({})
	for split in splits:
		all_split_df = pd.DataFrame({})
		averaged_split_df = pd.DataFrame({})
		split_save_path = header_path+'/'+split
		path_if_none(split_save_path)
		split_data_path = path_to_splits+split+'/in_silico_screen_results'
		pred_files = []
		for fname in os.listdir(split_data_path):
			if fname.startswith(header):
				for cond in conds:
					if cond in fname:
						pred_files.append(fname)
		# pred_files = [fname for fname in os.listdir(path_to_splits+split) if fname.startswith('Predicted_vs_actual'+header)]
		for fname in pred_files:
			# split_plus_con_path = split_path + '/' + fname[19:]
			split_plus_con_save_path = split_save_path+'/'+fname[:-4]
			path_if_none(split_plus_con_save_path)
			to_add = pd.read_csv(split_data_path+'/'+fname)
			# split_save_path = save_path+
			if not meta_done:
				meta_done = True
				meta_df = to_add[meta_cols]
			to_add = to_add[pred_col_names]
			# print(to_add.head())
			if toy:
				to_add = to_add[:1000]
				meta_df = meta_df[:1000]
			# if not meta_done:
				# meta_df = to_add[meta_cols]
			preds_header = split+'_'+fname[:-4]+'_'
			rename_dict = {}
			for i,pcn in enumerate(pred_col_names):
				rename_dict[pcn] = preds_header+str(i)
			to_add.rename(columns = rename_dict, inplace = True)
			plt.figure(figsize = (figsize, figsize))
			hm = sns.heatmap(to_add.corr(), annot = True, cmap = cmap_all, vmin = -1, vmax = 1, square = True)
			hm.figure.tight_layout()
			hm.set(xlabel='Model', ylabel='Model', title = 'Different training runs, '+split_plus_con_save_path[len(split_save_path)+1:])
			plt.savefig(split_plus_con_save_path+'/all_split_corr.png')
			plt.close()
			all_split_df = pd.concat([all_split_df, to_add], axis = 1)
			averaged_split_df['Avg_'+split_plus_con_save_path[len(split_save_path)+1:]] = to_add.mean(axis = 1)
			train_averaged_combined = pd.concat([train_averaged_combined,averaged_split_df], axis = 1)
		plt.figure(figsize = (figsize, figsize))
		plt.subplots_adjust(left = 0.5)
		# plt.tight_layout()
		hm = sns.heatmap(all_split_df.corr(), annot = False, cmap = cmap_all, vmin = -1, vmax = 1, square = True)
		hm.figure.tight_layout()
		hm.set(xlabel='Model', ylabel='Model', title = 'All training runs, split'+split)
		plt.savefig(split_save_path+'/all_split_corr.png')
		plt.close()
		plt.figure(figsize = (figsize, figsize))
		hm = sns.heatmap(averaged_split_df.corr(), annot = True, cmap = cmap_all, vmin = -1, vmax = 1, square = True)
		hm.figure.tight_layout()
		hm.set(xlabel='Model', ylabel='Model', title = 'Averaged over crossval runs, split'+split)
		plt.savefig(split_save_path+'/avg_split_corr.png')
		plt.close()
		all_combined = pd.concat([all_combined, all_split_df], axis = 1)
		all_averaged_combined[split+'_avg'] = all_split_df.mean(axis = 1)
	plt.figure(figsize = (figsize, figsize))
	hm = sns.heatmap(all_combined.corr(), annot = False, cmap = cmap_all, vmin = -1, vmax = 1, square = True)
	hm.figure.tight_layout()
	hm.set(xlabel='Model', ylabel='Model', title = 'All_training_runs, dataset '+header_path[len(save_path):])
	plt.savefig(header_path+'/all_model_corr.png')
	plt.close()
	plt.figure(figsize = (figsize, figsize))
	hm = sns.heatmap(train_averaged_combined.corr(), annot = False, cmap = cmap_all, vmin = -1, vmax = 1, square = True)
	hm.figure.tight_layout()
	hm.set(xlabel='Model', ylabel='Model', title = 'Split and condition comparison, dataset '+header+addon)
	plt.savefig(header_path+'/split_cond_avg_corr.png')
	plt.close()
	plt.figure(figsize = (figsize, figsize))
	hm = sns.heatmap(all_averaged_combined.corr(), annot = True, cmap = cmap_all, vmin = -1, vmax = 1, square = True)
	hm.figure.tight_layout()
	hm.set(xlabel='Model', ylabel='Model', title = 'Split comparison, dataset '+header+addon)
	plt.savefig(header_path+'/split_avg_corr.png')
	plt.close()
	sizedf = pd.DataFrame({'Size':[len(all_combined),len(all_combined.columns)]})
	sizedf.to_csv(header_path+'/dataset_size.csv', index = False)
	meta_df['ens_avg_pred'] = all_combined.mean(axis = 1)
	all_combined = pd.concat([meta_df, all_combined], axis = 1)
	meta_df.sort_values(by = 'ens_avg_pred', ascending = False, inplace = True)
	top_1000 = meta_df.head(1000)
	meta_df.to_csv(header_path+'/metadata_and_avg.csv',index = False)
	top_1000.to_csv(header_path+'/metadata_and_avg_top_1000.csv', index = False)
	all_combined.to_csv(header_path + '/all_combined_preds.csv', index = False)
	all_combined.sort_values(by = 'ens_avg_pred', ascending = False, inplace=True)
	all_combined_top_1000 = all_combined.head(1000)
	all_combined_top_1000.to_csv(header_path+'/all_preds_top_1000.csv', index = False)
	return all_combined
	# meta_df contains metadata (smiles, etc)
	# combined contains all the combined data



def main(argv):
	# args = sys.argv[1:]
	task_type = argv[1]
	if task_type == 'combine_predictions':
		header = argv[2]
		addon = argv[3]
		settings = argv[4:]
		cutoff = settings.index('cond')
		all_splits = settings[:cutoff]
		all_conds = settings[cutoff+1:]
		# all_splits = argv[4:]
		df = combine_predictions(header, all_splits, addon, all_conds)
	if task_type == 'combine_predictions_cv':
		header = argv[2]
		addon = argv[3]
		settings = argv[4:]
		cutoff = settings.index('cond')
		all_splits = settings[:cutoff]
		all_conds = settings[cutoff+1:]
		# all_splits = argv[4:]
		df = combine_predictions_cv(header, all_splits, addon, all_conds)



if __name__ == '__main__':
	main(sys.argv)


