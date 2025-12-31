"""Loads a trained model checkpoint and makes predictions on a dataset."""

from parsing import parse_predict_args
from chemprop.train import make_predictions
import json
from argparse import Namespace

# python predict.py --test_path Whitehead_results/No_Morgan_counts/fold_0/test_full.csv --checkpoint_path Whitehead_results/No_Morgan_counts/fold_0/model_0/model.pt --preds_path Whitehead_results/Predictions/no_morgan_counts.csv

# python predict.py --test_path Anti_nCoV_results/Scaffold_split/fold_0/test_full.csv --checkpoint_path Anti_nCoV_results/Scaffold_split/fold_0/model_0/model.pt --preds_path Anti_nCoV_results/Scaffold_split/Predictions/morgan_count_predictions.csv --features_generator morgan_count

# if __name__ == '__main__':
#     my_args = json.load(open('predict_args.json','r'))
#     args = Namespace(**my_args)
#     make_predictions(args)




def predict_multitask_from_json_cv(args, model_path, path_to_new_test = '', ensemble_number = -1):
	# print(checkpoint_dir)
	# print(preds_path)
	model_prefix = '/cv_'+str(ensemble_number)+'/trained_model/'
	train_args_path = model_path + model_prefix + 'local_train_args.json'

	checkpoint_path = model_path + model_prefix + 'fold_0/model_0/model.pt'
	my_train_args = json.load(open(train_args_path,'r'))
	features_path = model_path + '/cv_'+str(ensemble_number) + '/test_extra_x.csv'
	test_data_path = model_path + '/cv_'+str(ensemble_number) + '/test.csv'
	if not path_to_new_test=='':
		test_data_path = model_path + '/in_silico_screens/'+path_to_new_test+'/test.csv'
		features_path = model_path + '/in_silico_screens/'+path_to_new_test+'/test_extra_x.csv'

	args.test_path = test_data_path
	args.features_path = [features_path]

	args.checkpoint_paths = [checkpoint_path]
	args.preds_path = model_path + model_prefix + 'Predictions/test_predictions.csv'
	if not path_to_new_test=='':
		args.preds_path = model_path + model_prefix + 'Predictions/test_predictions_' + path_to_new_test + '.csv'
	args.features_generator = my_train_args['features_generator']
	# print('\n\nmaking predictions!\n\n')
	make_predictions(args)

def predict_multitask_from_json(args, model_path, path_to_new_test = '', ensemble_number = -1):
	# print(checkpoint_dir)
	# print(preds_path)
	model_prefix = '/trained_model/'
	if ensemble_number > -0.5:
		model_prefix = '/trained_model_'+str(ensemble_number)+'/'
	train_args_path = model_path + model_prefix + 'local_train_args.json'

	test_data_path = model_path + '/test.csv'
	checkpoint_path = model_path + model_prefix + 'fold_0/model_0/model.pt'
	my_train_args = json.load(open(train_args_path,'r'))
	features_path = model_path + '/test_extra_x.csv'
	if not path_to_new_test=='':
		test_data_path = model_path + '/in_silico_screens/'+path_to_new_test+'/test.csv'
		features_path = model_path + '/in_silico_screens/'+path_to_new_test+'/test_extra_x.csv'

	args.test_path = test_data_path
	args.features_path = [features_path]

	args.checkpoint_paths = [checkpoint_path]
	args.preds_path = model_path + model_prefix + 'Predictions/test_predictions.csv'
	if not path_to_new_test=='':
		args.preds_path = model_path + model_prefix + 'Predictions/test_predictions_' + path_to_new_test + '.csv'
	args.features_generator = my_train_args['features_generator']
	# print('\n\nmaking predictions!\n\n')
	make_predictions(args)

def get_base_predict_args():
	base_predict_args = json.load(open('base_predict_args.json','r'))
	return Namespace(**base_predict_args)

def make_predictions_from_model(preds_dir, base = True, morgan = True, rdkit = True, morgan_plus_rdkit = True, test_data_path = None, is_in_vivo = False, is_transfer = False):
	predictions_dir = preds_dir + '/'
	if is_transfer:
		predictions_dir += 'Transfer_learning/'
	if is_in_vivo:
		predictions_dir += 'in_vivo_'
	if base:
		predict_multitask_from_json(get_base_predict_args(), preds_path = predictions_dir+'Predictions/base_only.csv',
			checkpoint_dir = preds_dir + '/Base_model_only', test_data_path = test_data_path, is_transfer = is_transfer)
	if morgan:
		predict_multitask_from_json(get_base_predict_args(), preds_path = predictions_dir + 'Predictions/morgan_counts.csv',
			checkpoint_dir = preds_dir + '/Morgan_counts', test_data_path = test_data_path, is_transfer = is_transfer)
	if rdkit:
		predict_multitask_from_json(get_base_predict_args(), preds_path = predictions_dir + 'Predictions/rdkit_2d.csv',
			checkpoint_dir = preds_dir + '/rdkit_2d', test_data_path = test_data_path, is_transfer = is_transfer)
	if morgan_plus_rdkit:
		predict_multitask_from_json(get_base_predict_args(), preds_path = predictions_dir + 'Predictions/morgan_plus_rdkit.csv',
			checkpoint_dir = preds_dir + '/Morgan_plus_rdkit', test_data_path = test_data_path, is_transfer = is_transfer)



# make_predictions('Whitehead_results/in_vitro_only/Random_split/Predictions')
# make_predictions_from_model('Whitehead_results/in_vitro_only/Scaffold_split',rdkit = False, morgan_plus_rdkit = False)
# make_predictions_from_model('Whitehead_results/in_vitro_only/Amine_split',rdkit = True,
# 	morgan_plus_rdkit = True,test_data_path = 'Data/Whitehead_in_vitro_splits/Amine_split_test_train_val/test.csv')
# make_predictions_from_model('Whitehead_results/in_vitro_only/Amine_split_500s',rdkit = True,
	# morgan_plus_rdkit = True,test_data_path = 'Data/Whitehead_in_vitro_splits/Amine_split_500s/test.csv')
# make_predictions_from_model('Whitehead_results/in_vitro_only/Amine_split_500s',rdkit = True,
# 	morgan_plus_rdkit = True,test_data_path = 'Data/Whitehead_in_vivo_splits/Amine_split_500s_in_vivo/test.csv', is_in_vivo = True)
# make_predictions_from_model('Whitehead_results/in_vitro_only/Amine_split_500s',rdkit = True,
	# morgan_plus_rdkit = True,test_data_path = 'Data/Whitehead_in_vivo_splits/Amine_split_500s_in_vivo/test.csv', is_in_vivo = True, is_transfer = True)



