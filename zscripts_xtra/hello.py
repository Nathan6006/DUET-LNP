from Analyze_predictions import evaluate_multiple_models
from predict_from_json import predict_from_json, get_base_predict_args
from train_from_json import train_model, get_base_args
from chemprop.train.transfer_train import run_transfer_training
import os

def run_pipeline(train = True, predict = True, analyze = True, train_base = True, train_morgan = True, train_rdkit = True, train_morgan_plus_rdkit = True,
	base_model_save_dir = None, split_name = 'Random_split',
	scaffold_split = False, train_path = None, val_path = None, test_path = None, save_smiles_splits = True,
	tune_path = None, in_vivo_test_path = None, layer_list = ([1]*10+[2]*10+[3]*10+[4]*10+[5]*10), lr_ratio = 1,
	activity_name = 'Trunc_luc_activity', xlabel='Predicted Rel Luc Activity', ylabel='Experimental Rel Luc Activity',
	use_formulations = False):
	base_model_dir = base_model_save_dir + '/' + split_name
	if train:
		if train_base:
			save_dir = base_model_dir + '/Base_model_only'
			try:
				os.makedirs(save_dir)
			except:
				pass
			train_model(get_base_args(),save_dir = save_dir, generator = None, save_smiles_splits = save_smiles_splits,
				train_path = train_path, val_path = val_path, test_path = test_path, scaffold_split = scaffold_split, use_formulations = use_formulations)
			if not tune_path is None:
				run_transfer_training(save_dir, tune_path, in_vivo_test_path, layer_list = layer_list, lr_ratio = lr_ratio, use_formulations = use_formulations)
		if train_morgan:
			save_dir = base_model_dir + '/Morgan_count'
			try:
				os.makedirs(save_dir)
			except:
				pass
			train_model(get_base_args(),save_dir = save_dir, generator = ['morgan_count'], save_smiles_splits = save_smiles_splits,
				train_path = train_path, val_path = val_path, test_path = test_path, scaffold_split = scaffold_split, use_formulations = use_formulations)
			if not tune_path is None:
				run_transfer_training(save_dir, tune_path, in_vivo_test_path, layer_list = layer_list, lr_ratio = lr_ratio, use_formulations = use_formulations)
		if train_rdkit:
			save_dir = base_model_dir + '/rdkit_2d'
			try:
				os.makedirs(save_dir)
			except:
				pass
			train_model(get_base_args(),save_dir = save_dir, generator = ['rdkit_2d'], save_smiles_splits = save_smiles_splits,
				train_path = train_path, val_path = val_path, test_path = test_path, scaffold_split = scaffold_split, use_formulations = use_formulations)
			if not tune_path is None:
				run_transfer_training(save_dir, tune_path, in_vivo_test_path, layer_list = layer_list, lr_ratio = lr_ratio, use_formulations = use_formulations)
		if train_morgan_plus_rdkit:
			save_dir = base_model_dir + '/Morgan_plus_rdkit'
			try:
				os.makedirs(save_dir)
			except:
				pass
			train_model(get_base_args(),save_dir = save_dir, generator = ['morgan_count','rdkit_2d'], save_smiles_splits = save_smiles_splits,
				train_path = train_path, val_path = val_path, test_path = test_path, scaffold_split = scaffold_split, use_formulations = use_formulations)
			if not tune_path is None:
				run_transfer_training(save_dir, tune_path, in_vivo_test_path, layer_list = layer_list, lr_ratio = lr_ratio, use_formulations = use_formulations)
	if predict:
		predict_args = get_base_predict_args()
		try:
			os.makedirs(base_model_dir + '/predictions')
			os.makedirs(base_model_dir + '/untuned_in_vivo_predictions')
			os.makedirs(base_model_dir + '/tuned_in_vivo_predictions')
		except:
			pass
			# print('\n\n\n\n\n\n\n\nERROR' + base_model_dir + '/predictions\n\n\n\n\n\n\n\n')
		models_to_analyze = []
		for model_type in ['Base_model_only', 'Morgan_count', 'rdkit_2d', 'Morgan_plus_rdkit']:
			model_dir = base_model_dir + '/' + model_type
			if os.path.isdir(model_dir):
				models_to_analyze.append(model_type)
				predict_from_json(predict_args, preds_path = base_model_dir + '/predictions/' + model_type + '.csv', checkpoint_dir = model_dir, test_data_path = test_path, use_formulations = use_formulations)
				if not in_vivo_test_path is None:
					predict_from_json(predict_args, preds_path = base_model_dir + '/untuned_in_vivo_predictions/' + model_type + '.csv', checkpoint_dir = model_dir, test_data_path = in_vivo_test_path, use_formulations = use_formulations)
				if not tune_path is None:
					predict_from_json(predict_args, preds_path = base_model_dir + '/tuned_in_vivo_predictions/' + model_type + '.csv', checkpoint_dir = model_dir, test_data_path = in_vivo_test_path, is_transfer = True, use_formulations = use_formulations)
	if analyze:
		if not predict:
			models_to_analyze = []
			for model_type in ['Base_model_only', 'Morgan_count', 'rdkit_2d', 'Morgan_plus_rdkit']:
				model_dir = base_model_dir + '/' + model_type
				if os.path.isdir(model_dir):
					models_to_analyze.append(model_type)
		preds_dir = base_model_dir + '/predictions/'
		evaluate_multiple_models([mod+'.csv' for mod in models_to_analyze], test_path, preds_dir, cutoff_type = 'in_vitro', include_classification_analysis = False, activity_name = activity_name)
		if not in_vivo_test_path is None:
			preds_dir = base_model_dir + '/untuned_in_vivo_predictions/'
			evaluate_multiple_models([mod + '.csv' for mod in models_to_analyze], in_vivo_test_path, preds_dir, cutoff_type = 'in_vivo', include_classification_analysis = True, activity_name = activity_name)
			preds_dir = base_model_dir + '/tuned_in_vivo_predictions/'
			evaluate_multiple_models([mod + '.csv' for mod in models_to_analyze], in_vivo_test_path, preds_dir, cutoff_type = 'in_vivo', include_classification_analysis = True, activity_name = activity_name)


# run_pipeline(base_model_save_dir = 'Anti_nCoV_results', train_base = False, train_morgan = True, train_rdkit = True, train_morgan_plus_rdkit = True, train = False, predict = False,
# 	test_path = 'Data/Anti_nCoV_splits/test_full.csv', train_path = 'Data/Anti_nCoV_splits/train_full.csv',
# 	val_path = 'Data/Anti_nCoV_splits/val_full.csv', split_name = 'Scaffold_split', scaffold_split = True, save_smiles_splits = False, activity_name = 'Inh_index',
# 	xlabel = 'Predicted inhibition index', ylabel = 'Experimental inhibition index')

# run_pipeline(base_model_save_dir = 'Whitehead_results', train_base = False, train_morgan = True, train_rdkit = True, train_morgan_plus_rdkit = True, train = True,
	# test_path = 'Whitehead_results/Random_split/Base_model_only/fold_0/test_full.csv')

# run_pipeline(base_model_save_dir = 'Whitehead_results', train_base = False, train_morgan = True, train_rdkit = True, train_morgan_plus_rdkit = True, train = True,
# 	test_path = 'Data/Whitehead_in_vitro_splits/Amine_split_500s/test.csv', train_path = 'Data/Whitehead_in_vitro_splits/Amine_split_500s/train.csv',
# 	val_path = 'Data/Whitehead_in_vitro_splits/Amine_split_500s/val.csv', tune_path = 'Data/Whitehead_in_vivo_splits/Amine_split_500s_in_vivo/tune.csv',
# 	in_vivo_test_path = 'Data/Whitehead_in_vivo_splits/Amine_split_500s_in_vivo/test.csv', split_name = 'Amine_split_500s', save_smiles_splits = False)

# run_pipeline(base_model_save_dir = 'Whitehead_results', train_base = False, train_morgan = True, train_rdkit = True, train_morgan_plus_rdkit = True, train = True,
# 	test_path = 'Data/Whitehead_in_vitro_splits/Amine_split_test_train_val/test.csv', train_path = 'Data/Whitehead_in_vitro_splits/Amine_split_test_train_val/train.csv',
# 	val_path = 'Data/Whitehead_in_vitro_splits/Amine_split_test_train_val/val.csv', split_name = 'Amine_split', save_smiles_splits = False)

# run_pipeline(base_model_save_dir = 'Whitehead_results', train_base = True, train_morgan = False, train_rdkit = False, train_morgan_plus_rdkit = False, train = True, predict = True, analyze = True,
# 	test_path = 'Data/Whitehead_in_vitro_splits/Amine_split_test_train_val_subsampled_formulations/test.csv', train_path = 'Data/Whitehead_in_vitro_splits/Amine_split_test_train_val_subsampled_formulations/train.csv',
# 	val_path = 'Data/Whitehead_in_vitro_splits/Amine_split_test_train_val_subsampled_formulations/val.csv', split_name = 'Amine_split_subsampled_formulations', save_smiles_splits = False, use_formulations = True)

# run_pipeline(base_model_save_dir = 'Whitehead_results', train_base = True, train_morgan = False, train_rdkit = False, train_morgan_plus_rdkit = False, train = True, predict = True, analyze = True,
# 	test_path = 'Data/Whitehead_in_vitro_splits/Amine_split_test_train_val_subsampled_formulations_noerror/test.csv', train_path = 'Data/Whitehead_in_vitro_splits/Amine_split_test_train_val_subsampled_formulations_noerror/train.csv',
# 	val_path = 'Data/Whitehead_in_vitro_splits/Amine_split_test_train_val_subsampled_formulations_noerror/val.csv', split_name = 'Amine_split_subsampled_formulations_noerror', save_smiles_splits = False, use_formulations = True)

# run_pipeline(base_model_save_dir = 'Whitehead_results', train_base = True, train_morgan = False, train_rdkit = False, train_morgan_plus_rdkit = False, train = True, predict = True, analyze = True,
# 	test_path = 'Data/Whitehead_in_vitro_splits/Random_split_subsampled_formulations_noerror/test.csv', train_path = 'Data/Whitehead_in_vitro_splits/Random_split_subsampled_formulations_noerror/train.csv',
# 	val_path = 'Data/Whitehead_in_vitro_splits/Random_split_subsampled_formulations_noerror/val.csv', split_name = 'Random_split_subsampled_formulations_noerror', save_smiles_splits = False, use_formulations = True)


# run_pipeline(base_model_save_dir = 'Whitehead_results', train_base = True, train_morgan = True, train_rdkit = False, train_morgan_plus_rdkit = False, train = True, predict = True, analyze = True,
	# test_path = 'Data/Whitehead_in_vitro_splits/Formulation_split_subsampled_formulations_noerror/test_short.csv', train_path = 'Data/Whitehead_in_vitro_splits/Formulation_split_subsampled_formulations_noerror/train.csv',
	# val_path = 'Data/Whitehead_in_vitro_splits/Formulation_split_subsampled_formulations_noerror/val.csv', split_name = 'Subsampled_formulation_split_noerror', save_smiles_splits = False, use_formulations = True)

# run_pipeline(base_model_save_dir = 'Whitehead_results', train_base = True, train_morgan = False, train_rdkit = False, train_morgan_plus_rdkit = False, train = True, predict = True, analyze = True,
	# test_path = 'Data/Whitehead_in_vitro_splits/Formulation_split_subsampled_formulations_noerror/test_short.csv', train_path = 'Data/Whitehead_in_vitro_splits/Formulation_split_subsampled_formulations_noerror/train.csv',
	# val_path = 'Data/Whitehead_in_vitro_splits/Formulation_split_subsampled_formulations_noerror/val.csv', split_name = 'Subsampled_formulation_split_noerror', save_smiles_splits = False, use_formulations = True,
	# tune_path = 'Data/Whitehead_in_vivo_splits/Amine_split_500s_in_vivo_with_formulations/tune.csv', in_vivo_test_path = 'Data/Whitehead_in_vivo_splits/Amine_split_500s_in_vivo_with_formulations/test.csv')

print('running')
# run_pipeline(base_model_save_dir = 'Cancer_results', train_base = False, train_morgan = False, train_rdkit = False, train_morgan_plus_rdkit = False, train = False, predict = True, analyze = True,
	# test_path = 'Data/Cancer/test_full.csv', train_path = 'Data/Cancer/train_full.csv',
	# val_path = 'Data/Cancer/val_full.csv', split_name = 'Scaffold_split', save_smiles_splits = False, use_formulations = False)

