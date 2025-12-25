import sys
import os
from distribution_ds import save_dist_to_csv

def path_if_none(newpath):
    if not os.path.exists(newpath):
        os.makedirs(newpath)

def main(argv):
    split_folder = argv[1]
    path_if_none(split_folder)

    for i in range(5):
        save_dist_to_csv(
            output_filename=f'{split_folder}/train_cv_{i}.csv',
            folder= f'../data/crossval_splits/{split_folder}/cv_{i}/train_metadata.csv',
            bins=[0,70,80,200]
        )

        save_dist_to_csv(
            output_filename=f'{split_folder}/valid_cv_{i}.csv',
            folder= f'../data/crossval_splits/{split_folder}/cv_{i}/valid_metadata.csv',
            bins=[0,70,80,200]
        )

    save_dist_to_csv(
        output_filename=f'{split_folder}/test.csv',
        folder= f'../data/crossval_splits/{split_folder}/test/test_metadata.csv',
        bins=[0,70,80,200]
    )

if __name__ == "__main__":
    main(sys.argv)