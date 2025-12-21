"""Loads a trained model checkpoint and makes predictions on a dataset."""

from chemprop.parsing import parse_predict_args
from chemprop.train import make_predictions
import json

# python predict.py --test_path Whitehead_results/No_Morgan_counts/fold_0/test_full.csv --checkpoint_path Whitehead_results/No_Morgan_counts/fold_0/model_0/model.pt --preds_path Whitehead_results/Predictions/no_morgan_counts.csv

# python predict.py --test_path Anti_nCoV_results/Scaffold_split/fold_0/test_full.csv --checkpoint_path Anti_nCoV_results/Scaffold_split/fold_0/model_0/model.pt --preds_path Anti_nCoV_results/Scaffold_split/Predictions/morgan_count_predictions.csv --features_generator morgan_count

if __name__ == '__main__':
    args = parse_predict_args()
    # json.dump(vars(args),open('predict_args.json','w'))
    pickle.dump(args,open('predict_args.pkl','wb'))
    # make_predictions(args)
