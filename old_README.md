# Artificial intelligence-guided design of lipid nanoparticles for pulmonary gene therapy

## Description

This repository contains the data processing code and analysis discussed in the paper "Artificial intelligence-guided design of lipid nanoparticles for pulmonary gene therapy".

_____

# Table of contents

- [Repository structure](#repository-structure)
- [System requirements](#system-requirements)
- [Installation guide](#installation-guide)
- [Demo and Instructions](#demo-and-instructions)

_____

# Repository structure

The repository is organized into three main folders - `/data`, `/scripts`, and `/results`. 

# System requirements

## Hardware requirements

To run the analysis code in this repository, it is recommended to have a computer with enough RAM (> 8 GB) to support in-memory operations. Run times will be significantly more efficient with a CUDA GPU, but everything can run with just a CPU.

## Software requirements

This code has been implemented with `Python` version 3.8.19. See below for creating an appropriate conda environment.

### OS requirements

This code has been tested on the following systems, although it should work generally for other OS types (e.g., Windows).

- macOS: Ventura 13.2.1

- Linux: Ubuntu 22.04

# Installation guide

## Install repository

```
git clone https://github.com/jswitten/LNP_ML.git
```

## Install Python dependencies

Create a conda environnment (`lnp_ml`) as follows:

```
conda create -n lnp_ml python=3.8
conda activate lnp_ml
pip install chemprop==1.7.0
```

# Demo and Instructions

For all commands shown below, run them from the `/scripts` folder.

### Making a new dataset (optional)

In order to make a new dataset (and incorporate, for example, your own internal datasets), add a folder in the `data/data_files_to_merge` directory. The folder should have three files:

-	`main_data.csv`: This should contain two columns, `smiles` and `quantified_delivery`, containing the SMILES and measured delivery z-score, respectively, for each row (i.e., LNP formulation).

-	`formulations.csv`: This should contain the columns `Formulation` (name for the formulation), `Cationic_Lipid_Mol_Ratio`, `Phospholipid_Mol_Ratio`, `Cholesterol_Mol_Ratio`, `PEG_Lipid_Mol_Ratio`, `Helper_lipid_ID`, `Cationic_Lipid_to_mRNA_weight_ratio`, and `Form_comment` (which can be left blank if there are no relevant notes about it). The molar ratio columns should sum to 100 for each row. You can also provide mass ratios (`Phospholipid_Mass_Ratio` etc), and, as long as the helper lipid molecular weights are recognized (if not, add an entry in the `Component_molecular_weights.csv` file), the code will compute molar ratios.

-	`individual_metadata.csv`: This should provide all the other optional data about a measurement: target cell type, organ, etc. See some of the other metadata files for more information.

The rows in these three .csv files should line up, i.e., row #17 for should correspond to the same measurement for all three files. The one exception is that if you have the exact same formulation parameters for each row, you can just specify the formulation parameters once, and the code will assume the formulation is the same for every row.

Also, you will need to add a row in the `experiment_metadata.csv` file corresponding to your new dataset. The `Experiment_ID` must match the folder name for your new data folder. The other columns can be specified in `experiment_metadata.csv` if the value is the same for all measurements in your folder. If not (for example, if you have a dataset with both liver delivery and HeLa cell delivery), the value for that column must be specified in the `individual_metadata.csv` file.

To incorporate all the data into your new dataset, run the command:

```
python main_script.py merge_datasets
```

As a result, `all_data.csv` in the `/data` folder will now include your updated dataset. Note that `all_data.csv` already included in this repository features all the LNP data in this repository. For the subset of data analyzed in the paper, see `all_data_for_paper.csv`.

### Generating train & test splits (optional)

In order to make a train & test split (besides the ones are are already present in the repository, for example, if you are incorporating new data), make a .csv file in `data/crossval_split_specs` entitled `all_amine_split.csv`. It should have a format like the current file in the repository:

| Data_types_for_component | Values | Data_type_for_split | Train_or_split |
| ------------------------ | ------ | ------------------- | -------------- |
| Experiment_ID	| Whitehead_siRNA | Amine | split |
| Experiment_ID |	LM_3CR	| Amine | split |
| Library_ID,Delivery_target | RM_carbonate, generic_cell | Amine	| split |
| Experiment_ID,Library_ID | A549_form_screen,IR_Reductive_amination | Amine | split |
| Experiment_ID,Library_ID | A549_form_screen,other | smiles | train |
| Experiment_ID | Luke_Raj_133_3_form | Random_id | train |
| Library_ID | RM_Michael_addition_branched | Amine | split |
| Experiment_ID | IR_BL_AG_4CR | Amine | split |
| Experiment_ID | IR_BL_AG_3CR_sting | Amine | split |
| Experiment_ID | Akinc_Michael_addition | Amine | split |
| Experiment_ID | Li_Thiolyne | Amine | split |
| Experiment_ID	| Liu_Phospholipids	| Amine	| split |
| Experiment_ID	| Miller_zwitter_lipids	| Amine	| split |
| Experiment_ID	| Zhou_dendrimer	| Amine	| split |
| Experiment_ID	| Lee_unsat_dendrimer	| Amine	| split |

Example can be found in `data/crossval_split_specs`.

The columns for `all_amine_split.csv` are as follows:

-	`Data_types_for_component`: the data column (in `all_data.csv`) used to specify a particular subset of the train data.

-	`Values`: values that specify that particular subset (i.e., column identified in `Data_types_for_component`) of the train data.

-	`Data_type_for_split`: the data column (in `all_data.csv`) used as the variable along which the split is performed. For example, if you specify `Amine`, all rows with a particular value for Amine will be grouped together during the train/test/validation splitting. If you specify `smiles`, all rows with a particular SMILES will be grouped together (there may be multiple rows with the same SMILES string, for example formulation screening with one or more ionizable lipids).

-	`Train_or_split`: specify either `train` or `split`. If `split` is specified, the dataset will be split per the usual train/test/validation approach. If `train` is specified, all selected data will be put into the train set. This is useful if a data subset is relatively small so measuring and reporting performance on that particular subset is not possible or useful; there is no harm in at least training on these small datasets as it may help the model generalize across different libraries.

To perform a split, run the following command, ensuring to insert the appropriate values for items in {}:

```
python main_script.py split {split filename} ultra-held-out-fraction {either "-1" or “morgan”} {either nothing or “in_silico_screen_split”}
```

For example,

```
python main_script.py split all_amine_split_for_paper.csv -1
```

This will generate a five-fold cross-validation split according to the split specified in the given split filename. The `ultra-held-out-fraction` is the fraction of data that you want held out of all of the cross-validation sets. This is useful for reporting error with predictions based on an ensemble average of all the cross-validation splits. If you do not want an ultra-held-out dataset, set it to `-1`. If you specify `morgan` then binary Morgan fingerprints (radius 2, 2048 bits) will be included. If you select `in_silico_screen_split`, then the train and validation sets will be the same, i.e. each model will be trained on 80% of the data with the remaining 20% used as a validation set and also to report (slightly biased) performance of the resulting model. This option should be selected for the actual model used to do in silico screening, since it trains on 80% instead of 60% of the data.

The output will be five folders `cv_0`, `cv_1`, `cv_2`, `cv_3`, `cv_4` and an `ultra_held_out` folder if that option was specified, within the `data/crossval_splits` folder.


### Training a model

To run a model, run the following command, ensuring to insert the appropriate values for items in {}:

```
python main_script.py train {name of split folder on which to train}
```

For example,

```
python main_script.py train all_amine_split_for_paper
```

The number of epochs for training can also be specified with `--epochs {number of epochs}`

If the folder name ends in Morgan: the model will include binary Morgan fingerprints of radius 2, 2048 bits. 


### Testing model performance

Run the following command, ensuring to insert the appropriate values for items in {}:

```
python main_script.py analyze {folder with trained model}
```

For example,

```
python main_script.py analyze all_amine_split_for_paper
```

This will create a folder in `results/crossval_splits` with the name of the folder containing the splits and trained model that contains analyzed results on the five test sets from cross-validation. The folder will have six sub-folders:

-	`crossval_performance` folder will report kendall, spearman, and pearson correlations between predicted and experimental data on the test set, for each test set, and for each individual data subset. It will leave a spot blank if there were < 10 datapoints to reference. It will also report the p value for a significant (Pearson) correlation for each predicted-experimental comparision, along with root mean squared error (RMSE) and n (number of points for that data subset).

-	`cv_i` for `i` in 0, 1, 2, 3, 4 will contain:
    - `predicted_vs_actual.csv`, which contains all measurement metadata on the test set along with the columns `quantified_delivery` and `cv_i_pred_quantified_delivery` which are the experimental and predicted delivery, respectively.

    - A separate folder for each data subset, which in turn will contain `pred_vs_actual_data.csv`, the predicted and experimental delivery that particular data subset, and `pred_vs_actual.png` (a plot of predicted versus experimental delivery).

    - If an ultra-held-out test set was created, `results/crossval_splits/{split name}` will contain another folder, `ultra_held_out`. This will contain an `individual_dataset_results` folder that has individual predicted-experimental plots and data for each data subset, along with `ultra_held_out_results.csv` which will contain all the statistical characterization for each data subset for the ultra-held-out test set.
    

### Running an in silico screen

Run the following command, ensuring to insert the appropriate values for items in {}:

```
python main_script.py predict {folder with trained model} {folder with LNPs to screen}
```

For example,

```
python main_script.py predict all_amine_split test_screen
```

The folder with list of SMILES to screen should be in `data/libraries/{folder}` and should have three files, `test_screen.csv`, `test_screen_extra_x.csv`, and `test_screen_metadata.csv` which hold the SMILES you are screening, the extra_x values (i.e. formulation parameters) and metadata for individual rows of the screen. Note that to run a screen, the formulation parameters must be specified. At this time, we recommend as a baseline selecting the KK formulation (35:16:46.5:2.5 ionizable lipid:DOPE:Cholesterol:PEG-lipid molar ratios) in HeLa cells as this tends to produce the most stable predictions though this may change in the future. If you are doing an expansion of an existing dataset, you should match the conditions from that dataset.

The output will be in `results/screen_results/{folder with trained model}/{folder with LNPs to screen}`. `cv_x_preds.csv` will contain predictions for each individual model, while `pred_file.csv` will contain the averaged predictions along with all the relevant metadata.


### Putting it all together

So, an example of a full run-though with small, toy datasets from generating a split to training and analyzing a model to running an in silico screen would be:

```
python main_script.py split small_test_split.csv 0.2 in_silico_screen_split
```

```
python main_script.py train small_test_split_with_ultra_held_out_for_in_silico_screen –epochs 3
```

```
python main_script.py analyze small_test_split_with_ultra_held_out_for_in_silico_screen
```

```
python main_script.py predict small_test_split_with_ultra_held_out_for_in_silico_screen test_screen
```

Running all code for training a model based on all data in this repository can take ~3 hours with CUDA GPU or ~24+ hours with CPU only, depending on your computer specifications. The smaller toy dataset discussed immediately above should take significantly less time at ~10 minutes to run with CUDA GPU or ~3 hours with CPU only.
