"""Trains a model on a dataset."""

from parsing import parse_train_args
from chemprop.train import cross_validate
from chemprop.utils import create_logger
import json
import os
from argparse import Namespace
# from hyperparameter_optimization import grid_search
# Run conda activate chemprop before running this


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
	base_args = json.load(open('../data/args_files/base_train_args.json','r'))
	# print(base_args)
	return Namespace(**base_args)


def train_hyperparam_optimized_model(args,path_to_splits, depth, dropout, ffn_num_layers, hidden_size, generator=None, epochs = 40,pyseed=42):
	args.epochs = epochs
	savepath = path_to_splits + '/trained_model'
	if not os.path.exists(savepath):
		os.makedirs(savepath)
	args.save_dir = savepath

	if not generator is None:
		args.features_generator = generator
		args.use_input_features = generator
	args.ffn_num_layers = ffn_num_layers
	args.depth = depth
	args.dropout = dropout
	args.hidden_size = hidden_size
	args.data_path = path_to_splits + '/train.csv'
	args.data_weights_path = path_to_splits + '/train_weights.csv'
	args.separate_val_path = path_to_splits + '/valid.csv'
	args.separate_test_path = path_to_splits + '/test.csv'
	args.features_path = [path_to_splits + '/train_extra_x.csv']
	args.separate_val_features_path = [path_to_splits + '/valid_extra_x.csv']
	args.separate_test_features_path = [path_to_splits + '/test_extra_x.csv']
	args.smiles_columns =  ['smiles']
	args.target_columns = ['quantified_delivery']
	args.ignore_columns = None
	args.loss_function = 'mse'
	args.quantile_loss_alpha = 0.0
	args.num_tasks = 1
	args.explicit_h = False
	args.adding_h = None
	args.reaction = False
	args.keeping_atom_map = None
	args.reaction_solvent = None
	args.phase_features_path = None
	args.atom_descriptors_path = None
	args.bond_descriptors_path = None
	args.atom_descriptors = None
	args.bond_descriptors = None
	args.overwrite_default_atom_features = False
	args.overwrite_default_bond_features = False
	args.separate_test_atom_descriptors_path = None
	args.separate_test_bond_descriptors_path = None
	args.separate_test_phase_features_path = None
	args.separate_test_constraints_path = None
	args.separate_val_atom_descriptors_path = None
	args.separate_val_bond_descriptors_path = None
	args.separate_val_phase_features_path = None
	args.separate_val_constraints_path = None
	args.atom_descriptor_scaling = None
	args.bond_descriptor_scaling = None
	args.is_atom_bond_targets = None
	args.cache_cutoff = 30000
	args.class_balance = None
	args.device = None
	args.mpn_shared = None
	args.number_of_molecules = 1
	args.target_weights = None
	args.resume_experiment = False
	args.constraints_path = None
	args.checkpoint_frzn = None
	args.atom_descriptors_size = 0
	args.spectra_activation = None
	args.aggregation = None
	args.aggregation_norm = 100
	args.pytorch_seed = pyseed

	if generator is None:
		args.use_input_features = True
	args.save_smiles_splits = False
	logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
	json.dump(vars(args),open(savepath+'/local_train_args.json','w'))
	cross_validate(args,logger)

def train_multitask_model(args,path_to_splits, generator=None, epochs = 40):
	args.epochs = epochs
	savepath = path_to_splits + '/trained_model'
	if not os.path.exists(savepath):
		os.makedirs(savepath)
	args.save_dir = savepath

	if not generator is None:
		args.features_generator = generator
		args.use_input_features = generator
	args.ffn_num_layers = 3
	args.data_path = path_to_splits + '/train.csv'
	args.separate_val_path = path_to_splits + '/valid.csv'
	args.separate_test_path = path_to_splits + '/test.csv'
	args.features_path = [path_to_splits + '/train_extra_x.csv']
	args.separate_val_features_path = [path_to_splits + '/valid_extra_x.csv']
	args.separate_test_features_path = [path_to_splits + '/test_extra_x.csv']
	if generator is None:
		args.use_input_features = True
	args.save_smiles_splits = False
	logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
	json.dump(vars(args),open(savepath+'/local_train_args.json','w'))
	cross_validate(args,logger)

