"""Trains a model on a dataset."""

from chemprop.parsing import parse_train_args
from chemprop.train import cross_validate
from chemprop.utils import create_logger
import pickle
import json

# source activate chemprop
# then, to run: for example, python JW_train_script.py --data_path Data/in_vitro_data_for_chemprop.csv --dataset_type regression --save_dir Whitehead_results --features_generator morgan_count --split_sizes 0.675 0.075 0.25

# python JW_train_script.py --data_path Data/in_vitro_data_for_chemprop.csv --dataset_type regression --save_dir Whitehead_results --features_generator morgan_count --split_sizes 0.675 0.075 0.25 --save_smiles_splits

# python JW_train_script.py --data_path Data/Anti_nCoV_for_Chemprop.csv --dataset_type regression --save_dir Anti_nCoV_results --features_generator morgan_count --split_sizes 0.6 0.02 0.2 --save_smiles_splits

if __name__ == '__main__':
    args = pickle.load(open('args.pkl','rb')) 
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
    print('args: ',args)
    json.dump(vars(args),open('test_args.json','w'))
    # import pdb; pdb.set_trace()
    cross_validate(args, logger)
