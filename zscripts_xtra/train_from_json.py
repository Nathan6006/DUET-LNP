"""Trains a model on a dataset."""

from chemprop.parsing import parse_train_args
from chemprop.train import cross_validate
from chemprop.utils import create_logger
import json
from argparse import Namespace

# source activate chemprop
# then, to run: for example, python JW_train_script.py --data_path Data/in_vitro_data_for_chemprop.csv --dataset_type regression --save_dir Whitehead_results --features_generator morgan_count --split_sizes 0.675 0.075 0.25

# python JW_train_script.py --data_path Data/in_vitro_data_for_chemprop.csv --dataset_type regression --save_dir Whitehead_results --features_generator morgan_count --split_sizes 0.675 0.075 0.25 --save_smiles_splits

# python JW_train_script.py --data_path Data/Anti_nCoV_for_Chemprop.csv --dataset_type regression --save_dir Anti_nCoV_results --features_generator morgan_count --split_sizes 0.6 0.02 0.2 --save_smiles_splits

# if __name__ == '__main__':
	# Load base args
	# base_args = json.load(open('base_train_args.json','r'))
	# args = Namespace(**base_args)
	# print('args: ',args)

	# Set args to what I actually want
	# args.save_dir = 'Whitehead_results/rdkit_2d_normalized'
	# args.features_generator = ['morgan_count']
	# args.use_input_features = args.features_generator


	# Save args so I can have them for the future
	# json.dump(vars(args),open(my_args['save_dir']+'/local_train_args.json','w'))
	# Train!
	# cross_validate(args, logger)

def get_base_args():
	base_args = json.load(open('base_train_args.json','r'))
	return Namespace(**base_args)

def train_model(args,save_dir = 'Unfiled_results',generator=None, scaffold_split = False, train_path = None, val_path = None, test_path = None, save_smiles_splits = True, use_formulations = False, num_folds = 1, splits_only = False, epochs = 40):
	args.epochs = epochs
	args.save_dir = save_dir
	if not generator is None:
		args.features_generator = generator
		args.use_input_features = generator
	if scaffold_split:
		args.split_type = 'scaffold_balanced'
	if not train_path is None:
		args.data_path = train_path
	if not val_path  is None:
		args.separate_val_path = val_path
	if not test_path  is None:
		args.separate_test_path = test_path
	args.num_folds = num_folds
	if splits_only:
		args.epochs = 1
		args.split_sizes = [1.0 - 2.0/num_folds, 1.0/num_folds, 1.0/num_folds]
	if use_formulations:
		# args.batch_size = 128
		# args.epochs = 20
		# args.hidden_size = 100
		# args.ffn_hidden_size = 100
		# args.no_features_scaling = True
		args.features_scaling = False
		args.features_path = [train_path[:-4] + '_formulations.csv']
		args.separate_val_features_path = [val_path[:-4] + '_formulations.csv']
		args.separate_test_features_path = [test_path[:-4] + '_formulations.csv']
		if generator is None:
			args.use_input_features = True
	args.save_smiles_splits = save_smiles_splits
	logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
	json.dump(vars(args),open(save_dir+'/local_train_args.json','w'))
	cross_validate(args,logger)

def train_model_random_split():
	train_model(get_base_args(),save_dir = 'Whitehead_results/Base_model_only')
	train_model(get_base_args(),save_dir = 'Whitehead_results/Morgan_counts',generator = ['morgan_count'])
	train_model(get_base_args(),save_dir = 'Whitehead_results/rdkit_2d',generator = ['rdkit_2d'])
	train_model(get_base_args(),save_dir = 'Whitehead_results/Morgan_plus_rdkit',generator = ['morgan_count','rdkit_2d'])


def train_model_scaffold_split():
	train_model(get_base_args(),save_dir = 'Whitehead_results/Scaffold_split/Base_model_only', scaffold_split = True)
	train_model(get_base_args(),save_dir = 'Whitehead_results/Scaffold_split/Morgan_counts',generator = ['morgan_count'], scaffold_split = True)
	# train_model(get_base_args(),save_dir = 'Whitehead_results/Scaffold_split/rdkit_2d',generator = ['rdkit_2d'], scaffold_split = True)
	# train_model(get_base_args(),save_dir = 'Whitehead_results/Scaffold_split/Morgan_plus_rdkit',generator = ['morgan_count','rdkit_2d'], scaffold_split = True)

def train_model_amine_split():
	test_path = 'Data/Whitehead_in_vitro_splits/Amine_split_test_train_val/test.csv'
	val_path = 'Data/Whitehead_in_vitro_splits/Amine_split_test_train_val/val.csv'
	train_path = 'Data/Whitehead_in_vitro_splits/Amine_split_test_train_val/train.csv'
	save_prefix = 'Whitehead_results/in_vitro_only/Amine_split/'
	# train_model(get_base_args(),save_dir = save_prefix + 'Base_model_only', train_path = train_path, val_path = val_path, test_path = test_path, save_smiles_splits = False)
	# train_model(get_base_args(),save_dir = save_prefix + 'Morgan_counts',generator = ['morgan_count'], train_path = train_path, val_path = val_path, test_path = test_path, save_smiles_splits = False)
	train_model(get_base_args(),save_dir = save_prefix + 'rdkit_2d',generator = ['rdkit_2d'], train_path = train_path, val_path = val_path, test_path = test_path, save_smiles_splits = False)
	train_model(get_base_args(),save_dir = save_prefix + 'Morgan_plus_rdkit',generator = ['morgan_count','rdkit_2d'], train_path = train_path, val_path = val_path, test_path = test_path, save_smiles_splits = False)

def train_model_500s_split():
	test_path = 'Data/Whitehead_in_vitro_splits/Amine_split_500s/test.csv'
	val_path = 'Data/Whitehead_in_vitro_splits/Amine_split_500s/val.csv'
	train_path = 'Data/Whitehead_in_vitro_splits/Amine_split_500s/train.csv'
	save_prefix = 'Whitehead_results/in_vitro_only/Amine_split_500s/'
	train_model(get_base_args(),save_dir = save_prefix + 'Base_model_only', train_path = train_path, val_path = val_path, test_path = test_path, save_smiles_splits = False)
	train_model(get_base_args(),save_dir = save_prefix + 'Morgan_counts',generator = ['morgan_count'], train_path = train_path, val_path = val_path, test_path = test_path, save_smiles_splits = False)
	train_model(get_base_args(),save_dir = save_prefix + 'rdkit_2d',generator = ['rdkit_2d'], train_path = train_path, val_path = val_path, test_path = test_path, save_smiles_splits = False)
	train_model(get_base_args(),save_dir = save_prefix + 'Morgan_plus_rdkit',generator = ['morgan_count','rdkit_2d'], train_path = train_path, val_path = val_path, test_path = test_path, save_smiles_splits = False)

def train_cancer_model_scaffold_split():
	train_path = 'Data/Cancer/gi50_for_chemprop_nonames.csv'
	train_model(get_base_args(), train_path = train_path, save_dir = 'Cancer_results/Scaffold_split/Base_model_only', scaffold_split = True)
	# train_model(get_base_args(),save_dir = 'Cancer_results/Scaffold_split/Morgan_counts',generator = ['morgan_count'], scaffold_split = True)

def train_mers_model_scaffold_split():
	train_path = 'Data/MERS_splits/MERS_for_chemprop.csv'
	train_model(get_base_args(), train_path = train_path, save_dir = 'MERS_results/Scaffold_split/Base_model_only', scaffold_split = True)

def train_heiser_model_scaffold_split():
	train_path = 'Data/Heiser_nCoV_splits/heiser_ncov_for_chemprop.csv'
	train_model(get_base_args(), train_path = train_path, save_dir = 'Heiser_results/Scaffold_split/Base_model_only', scaffold_split = True)

def all_screens_scaffold_split():
	train_path = 'Data/All_antivirus_combined/All_screens_combined_for_chemprop_with_nans_replaced.csv'
	train_model(get_base_args(), train_path = train_path, save_dir = 'All_combined_results/Scaffold_split/Crossval/Base_model_only', scaffold_split = True, num_folds = 5, splits_only = True)

def luke_screen_split(split_type = 'Scaffold'):
	train_path = 'Data/Luke_bioactive_screen/luke_activity_data.csv'
	train_model(get_base_args(), train_path = train_path, save_dir = 'Luke_bioactive_screen_results/'+split_type+'_split/Crossval/Base_model_only',scaffold_split = (split_type=='Scaffold'), num_folds = 5, splits_only = True)

# luke_screen_split('Scaffold')
# luke_screen_split('Random')

# all_screens_scaffold_split()

# train_heiser_model_scaffold_split()

# train_cancer_model_scaffold_split()

# train_model_500s_split()

# train_mers_model_scaffold_split()

