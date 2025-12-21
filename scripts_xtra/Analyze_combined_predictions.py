import numpy as np 
import os
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_curve, roc_auc_score
# from train_multitask import train_multitask_model, get_base_args, optimize_hyperparameters, train_hyperparam_optimized_model
# from predict_multitask_from_json import predict_multitask_from_json, get_base_predict_args
# from rdkit import Chem
import matplotlib.pyplot as plt
import scipy.stats
import json
import sys
# import seaborn as sns

def do_simple_component_analysis(header, is_simple, preds_path = 'Data/Multitask_data/All_datasets/Pred_analysis/', components = ['Amine', 'Ketone', 'Isocyanide', 'Carboxylic_acid']):
	# For each component: find mean and sigma 
	df = pd.read_csv(preds_path+header+'/metadata_and_avg.csv')
	picks_per_component = int(float(desired_screen_size)**(1.0/len(components)))
	print('picks per com: ',picks_per_component)
	comdf = pd.DataFrame({})
	for component in components:
		coms = []
		means = []
		stds = []
		ns = []
		all_possible = set(list(df[component]))
		for com in all_possible:
			subset_df = df[df[component] == com]
			n = len(subset_df)
			if n>30:
				coms.append(com)
				means.append(np.mean(subset_df.ens_avg_pred))
				# print(com)
				# print(means)
				stds.append(np.std(subset_df.ens_avg_pred))
				ns.append(n)
		to_concat = pd.DataFrame({})
		to_concat[component+'_name'] = pd.Series(coms)
		to_concat[component+'_mean'] = pd.Series(means)
		to_concat[component+'_stds'] = pd.Series(stds)
		to_concat[component+'_n'] = pd.Series(ns)
		to_concat = to_concat.sort_values(by = [component+'_mean'], ascending = False).reset_index(drop = True)
		comdf = pd.concat([comdf,to_concat], axis = 1)
		print('done with: ',component)
	comdf.to_csv(preds_path+header+'/component_analysis.csv',index = False)

def complex_name_breakdown(name):
	# interprets things like 'h27&c15' or just 'h5'
	only = 'na'
	linker = 'na'
	term = 'na'
	if '&' in name:
		linker = name[:name.find('&')+1]
		term = name[name.find('&'):]
	else:
		only = name
	return linker, term, only

def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs>costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

def generate_pareto_efficient_lipids(header, is_ketone, preds_path = 'Data/Multitask_data/All_datasets/crossval_pred_analysis/',colnames = ['ens_avg_pred_no_morgan','ens_avg_morgan']):
	df = pd.read_csv(preds_path+header+'/metadata_and_avg.csv')

	preds_for_pareto = df[colnames].to_numpy()
	is_efficient = is_pareto_efficient(preds_for_pareto,return_mask = True)
	efficient_subset = df[is_efficient]

	plt.figure()
	plt.scatter(df[colnames[0]], df[colnames[1]], color = 'gray')
	plt.scatter(efficient_subset[colnames[0]], efficient_subset[colnames[1]], color = 'black')
	plt.xlabel(colnames[0])
	plt.ylabel(colnames[1])
	# plt.legend(loc = 'lower right')
	plt.savefig(preds_path+header + '/morgan_Pareto_frontier.png')
	plt.close()
	efficient_subset.to_csv(preds_path+header + '/morgan_Pareto_frontier.csv', index = True)


def do_complex_component_analysis(header,is_ketone, preds_path = 'Data/Multitask_data/All_datasets/crossval_pred_analysis/'):
	# For each component: find mean and sigma 
	simple_components = ['Amine','Isocyanide']
	ket_components = ['OH_only', 'OH_linker','COOH_term']
	ald_components = ['al_COOH_only','al_COOH_linker','al_OH_term']
	cooh_components = ['COOH_only','COOH_linker','OH_term']
	df = pd.read_csv(preds_path+header+'/metadata_and_avg.csv')
	comdf = pd.DataFrame({})
	oh_onlys = []
	oh_linkers = []
	cooh_terms = []
	cooh_onlys = []
	cooh_linkers = []
	oh_terms = []
	al_cooh_onlys = []
	al_cooh_linkers = []
	al_oh_terms = []
	if is_ketone:
		for ket in df.Ketone:
			# print(ket)
			split = ket.split('+')
			l, t, o = complex_name_breakdown(split[0])
			if not o == 'na':
				oh_onlys.append(o)
			if not t == 'na':
				cooh_terms.append(t)
			if not l == 'na':
				oh_linkers.append(l)
			# split = ket.split('+')
			# l, t, o = complex_name_breakdown(split[1])
			# if not o == 'na':
			# 	oh_onlys.append(o)
			# if not t == 'na':
			# 	cooh_terms.append(t)
			# if not l == 'na':
			# 	oh_linkers.append(l)
	else:
		for cooh in df.Aldehyde:
			l, t, o = complex_name_breakdown(cooh)
			if not o == 'na':
				al_cooh_onlys.append(o)
			if not t == 'na':
				al_cooh_linkers.append(l)
			if not l == 'na':
				al_oh_terms.append(t)
	
	for cooh in df.Carboxylic_acid:
		l, t, o = complex_name_breakdown(cooh)
		if not o == 'na':
			cooh_onlys.append(o)
		if not t == 'na':
			cooh_linkers.append(l)
		if not l == 'na':
			oh_terms.append(t)
	
	cooh_onlys = set(cooh_onlys)
	cooh_linkers = set(cooh_linkers)
	oh_terms = set(oh_terms)
	
	# print(oh_onlys)
	# print(oh_linkers)
	# print(cooh_terms)
	# print(cooh_onlys)
	# print(cooh_linkers)
	# print(oh_terms)
	if is_ketone:
		oh_onlys = set(oh_onlys)
		oh_linkers = set(oh_linkers)
		cooh_terms = set(cooh_terms)
		building_blocks = (oh_onlys, oh_linkers, cooh_terms, cooh_onlys, cooh_linkers, oh_terms)
		block_names = ['OH_only','OH_linker','COOH_terminus','COOH_only','COOH_linker','OH_terminus']
		block_comps = ['Ketone','Ketone','Ketone','Carboxylic_acid','Carboxylic_acid','Carboxylic_acid']
	else:
		al_cooh_onlys = set(al_cooh_onlys)
		al_cooh_linkers = set(al_cooh_linkers)
		al_oh_terms = set(al_oh_terms)
		building_blocks = (al_cooh_onlys, al_cooh_linkers, al_oh_terms, cooh_onlys, cooh_linkers, oh_terms)
		block_names = ['al_COOH_only','al_COOH_linker','al_OH_terminus','COOH_only','COOH_linker','OH_terminus']
		block_comps = ['Aldehyde','Aldehyde','Aldehyde','Carboxylic_acid','Carboxylic_acid','Carboxylic_acid']
	for component in simple_components:
		coms = []
		means = []
		stds = []
		ns = []
		all_possible = set(list(df[component]))
		for com in all_possible:
			subset_df = df[df[component] == com]
			n = len(subset_df)
			if n>10:
				coms.append(com)
				means.append(np.mean(subset_df.ens_avg_pred))
				# print(com)
				# print(means)
				stds.append(np.std(subset_df.ens_avg_pred))
				ns.append(n)
		to_concat = pd.DataFrame({})
		to_concat[component+'_name'] = pd.Series(coms)
		to_concat[component+'_mean'] = pd.Series(means)
		to_concat[component+'_stds'] = pd.Series(stds)
		to_concat[component+'_n'] = pd.Series(ns)
		to_concat = to_concat.sort_values(by = [component+'_mean'], ascending = False).reset_index(drop = True)
		comdf = pd.concat([comdf,to_concat], axis = 1)
		print('done with: ',component)
	for i, block_name in enumerate(block_names):
		coms = []
		means = []
		stds = []
		ns = []
		all_possible = building_blocks[i]
		for com in all_possible:
			mask = [has_component(name,com) for name in df[block_comps[i]]]
			subset_df = df[mask]
			n = len(subset_df)
			if n>50:
				coms.append(com)
				means.append(np.mean(subset_df.ens_avg_pred))
				stds.append(np.std(subset_df.ens_avg_pred))
				ns.append(n)
		to_concat = pd.DataFrame({})
		to_concat[block_name+'_name'] = pd.Series(coms)
		to_concat[block_name+'_mean'] = pd.Series(means)
		to_concat[block_name+'_stds'] = pd.Series(stds)
		to_concat[block_name+'_n'] = pd.Series(ns)
		to_concat = to_concat.sort_values(by = [block_name+'_mean'], ascending = False).reset_index(drop = True)
		comdf = pd.concat([comdf,to_concat], axis = 1)

	comdf.to_csv(preds_path+header+'/component_analysis.csv',index = False)

def find_all(st, substr):
	return [i for i in range(len(st)) if st.startswith(substr,i)]

def is_instance(lname, com, idx):
	right_invalid = [str(i) for i in range(10)]+['&']
	left_invalid = ['&']
	if idx == 0 or lname[idx-1] not in left_invalid:
		if idx+len(com)==len(lname) or lname[idx+len(com)] not in right_invalid:
			return True
	return False

def has_component(lipid_name, component):
	if component in lipid_name:
		candidates = find_all(lipid_name, component)
		for idx in candidates:
			if is_instance(lipid_name, component, idx):
				return True
	return False


def main(argv):
	# args = sys.argv[1:]
	task_type = argv[1]
	if task_type == 'select_subscreen':
		header = argv[3]
		is_simple = bool(argv[4])
		do_component_analysis(header,is_simple)
	if task_type == 'do_complex_component_analysis':
		header = argv[2]
		is_ketone = argv[3]=='ketone'
		do_complex_component_analysis(header, is_ketone)
	if task_type == 'pareto':
		header = argv[2]
		is_ketone = argv[3]=='ketone'
		generate_pareto_efficient_lipids(header, is_ketone)



if __name__ == '__main__':
	main(sys.argv)


