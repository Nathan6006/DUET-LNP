import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, mode, spearmanr
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from sklearn.metrics import roc_curve, roc_auc_score


CLASSIFICATION_CUTOFF = 0.5
IN_VIVO_CLASS_CUTOFF = 0.2

def eval_test_set(y_pred,y_test, plot=False, verbose=True, xlabel='Predicted Rel Luc Activity', ylabel='Experimental Rel Luc Activity'):
    if plot:
        plt.plot(y_pred,y_test,'.')
        plt.xlabel(xlabel, fontsize = 20)
        plt.ylabel(ylabel, fontsize = 32)
        plt.ylim(0,1)
        plt.xlim(min(0,min(y_pred)*1.05),max(1,max(y_pred)*1.05))
    
    tau,pval = kendalltau(y_pred,y_test)
    rho = np.corrcoef(y_pred,y_test)[0][1]
    mse = np.average([(y_pred[i]-y_test[i])**2 for i in range(len(y_pred))])
    if verbose:
        print('\tPearson correlation: '+repr(rho))
        print('\tKendall tau: '+repr(tau))
        print('\tMean squared error: '+repr(mse))
    return tau,rho,mse

def evaluate_regression_model(y_pred,full_y_test,cutoff = CLASSIFICATION_CUTOFF,plot=True):
    print('Results for full test set:')
    eval_test_set(y_pred,full_y_test,plot=True)
    trunc_y_pred = []
    trunc_y_test = []
    for i in range(len(full_y_test)):
        if full_y_test[i] < cutoff:
            trunc_y_pred.append(y_pred[i])
            trunc_y_test.append(full_y_test[i])
    print('Results for positives only in test set:')
    eval_test_set(trunc_y_pred,trunc_y_test,plot=plot)
    plt.show()

def evaluate_model_predictions(pred_fname,test_fname,activity_name = 'Trunc_luc_activity'):
	predictions = pd.read_csv(pred_fname)
	results = pd.read_csv(test_fname)
	y_pred = predictions[activity_name]
	y_test = results[activity_name]
	evaluate_regression_model(y_pred,y_test)

def evaluate_multiple_models(pred_fnames,test_fname, preds_dir, activity_name = 'Trunc_luc_activity', xlabel='Predicted Rel Luc Activity', ylabel='Experimental Rel Luc Activity',
    cutoff_type = 'in_vitro', include_classification_analysis = False, eval_formulations = False, analyze_good_and_bad = False):
    if cutoff_type == 'in_vitro':
        cutoff = CLASSIFICATION_CUTOFF
    elif cutoff_type == 'in_vivo':
        cutoff = IN_VIVO_CLASS_CUTOFF
    elif cutoff_type == 'ncov':
        cutoff = 1
    row_names = ['Sample_size','RMSE','r','tau','RMSE (positives only)','r (positives only)', 'tau (positives only)']
    if include_classification_analysis:
        row_names.append('ROC (cutoff '+repr(cutoff)+')')
        fprs = []
        tprs = []
        legend_names = []
    df = pd.DataFrame({'Attribute':row_names})
    results = pd.read_csv(test_fname)
    y_test = results[activity_name]
    for pred_fname in pred_fnames:
        full_fname = preds_dir + pred_fname
        pred_type = pred_fname[:-4]
        if analyze_good_and_bad:
            good_predictions_direc = preds_dir + pred_type + '_struc_analysis/Good_predictions'
            bad_predictions_direc = preds_dir + pred_type + '_struc_analysis/Bad_predictions'
            # os.mkdir(preds_dir + pred_type + '_struc_analysis')
            if not os.path.exists(good_predictions_direc):
                os.makedirs(good_predictions_direc)
            if not os.path.exists(bad_predictions_direc):
                os.makedirs(bad_predictions_direc)

        predictions = pd.read_csv(full_fname)
        
        y_pred = predictions[activity_name]
        all_smiles = predictions['smiles']
        sorted_preds = sorted(y_pred)

        tau, pval = kendalltau(y_pred,y_test)
        rho = np.corrcoef(y_pred,y_test)[0][1]
        rmse = np.sqrt(np.average([(y_pred[i]-y_test[i])**2 for i in range(len(y_pred))]))

        trunc_y_pred = []
        trunc_y_test = []
        if analyze_good_and_bad:
            for i in range(len(y_test)):
                if y_test[i] < cutoff:
                    if float(sorted_preds.index(y_pred[i]))/len(sorted_preds)<0.25:
                        Draw.MolToFile(Chem.MolFromSmiles(all_smiles[i]), good_predictions_direc+'/'+all_smiles[i].replace('/','&')+'.png', size = (600,600))
                    elif float(sorted_preds.index(y_pred[i]))/len(sorted_preds) >= 0.25:
                        Draw.MolToFile(Chem.MolFromSmiles(all_smiles[i]), bad_predictions_direc+'/'+all_smiles[i].replace('/','&')+'.png', size = (600,600))
                    trunc_y_pred.append(y_pred[i])
                    trunc_y_test.append(y_test[i])

        trunc_tau, pval = kendalltau(trunc_y_pred,trunc_y_test)
        trunc_rho = np.corrcoef(trunc_y_pred,trunc_y_test)[0][1]
        trunc_rmse = np.sqrt(np.average([(trunc_y_pred[i]-trunc_y_test[i])**2 for i in range(len(trunc_y_pred))]))

        plt.figure()
        plt.plot(y_pred, y_test, '.')
        # plt.plot(trunc_y_pred, trunc_y_test, '.')
        plt.xlabel(xlabel, fontsize = 16)
        plt.ylabel(ylabel, fontsize = 16)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.ylim(min(0,min(y_test)*1.05),max(1,max(y_test)*1.05))
        # plt.title(pred_type)
        plt.xlim(min(0,min(y_pred)*1.05),max(1,max(y_pred)*1.05))
        plt.tight_layout()
        plt.savefig(preds_dir+pred_type+'.png')

        if include_classification_analysis:
            class_y_test = [val<cutoff for val in y_test]
            class_y_pred = [-val for val in y_pred]
            fpr, tpr, thresholds = roc_curve(class_y_test,class_y_pred)
            fprs.append(fpr)
            tprs.append(tpr)
            # thresholds.append(threshold)
            df[pred_type] = [len(y_pred), rmse, rho, tau, trunc_rmse, trunc_rho, trunc_tau,roc_auc_score(class_y_test,class_y_pred)]
            legend_names.append(pred_type)
        else:
            df[pred_type] = [len(y_pred), rmse, rho, tau, trunc_rmse, trunc_rho, trunc_tau]
    df.to_csv(preds_dir+'results_vs_test_set.csv')
    if include_classification_analysis:
        plt.figure()
        for i in range(len(fprs)):
            plt.plot(fprs[i],tprs[i])
        plt.xlabel('False positive rate')
        plt.ylabel('False negative rate')
        plt.legend(legend_names)
        plt.savefig(preds_dir+'ROC_plot.png')

# preds_dir = 'chemprop-master/Whitehead_results/in_vitro_only/Amine_split_500s/Transfer_learning/in_vivo_Predictions/'
# pred_fnames = []
# test_fname = 'chemprop-master/Data/Whitehead_in_vivo_splits/Amine_split_500s_in_vivo/test.csv'
# all_preds = ['base_only.csv', 'morgan_counts.csv','rdkit_2d.csv','morgan_plus_rdkit.csv']
# evaluate_multiple_models(all_preds,test_fname, preds_dir,cutoff = IN_VIVO_CLASS_CUTOFF, include_classification_analysis = True)

# preds_dir = 'chemprop-master/Whitehead_results/in_vitro_only/Amine_split_500s/in_vivo_Predictions/'
# pred_fnames = []
# test_fname = 'chemprop-master/Data/Whitehead_in_vivo_splits/Amine_split_500s_in_vivo/test.csv'
# all_preds = ['base_only.csv', 'morgan_counts.csv', 'rdkit_2d.csv','morgan_plus_rdkit.csv']
# evaluate_multiple_models(all_preds,test_fname, preds_dir,cutoff = IN_VIVO_CLASS_CUTOFF, include_classification_analysis = True)

# preds_dir = 'chemprop-master/Whitehead_results/in_vitro_only/Amine_split_500s/Predictions/'
# pred_fnames = []
# test_fname = 'chemprop-master/Data/Whitehead_in_vitro_splits/Amine_split_500s/test.csv'
# all_preds = ['base_only.csv', 'morgan_counts.csv', 'rdkit_2d.csv','morgan_plus_rdkit.csv']
# evaluate_multiple_models(all_preds,test_fname, preds_dir)

# preds_dir = 'chemprop-master/Whitehead_results/in_vitro_only/Amine_split/Predictions/'
# pred_fnames = []
# test_fname = 'chemprop-master/Data/Whitehead_in_vitro_splits/Amine_split_test_train_val/test.csv'
# all_preds = ['base_only.csv', 'morgan_counts.csv', 'rdkit_2d.csv','morgan_plus_rdkit.csv']
# evaluate_multiple_models(all_preds,test_fname, preds_dir)

# preds_dir = 'chemprop-master/Whitehead_results/in_vitro_only/Scaffold_split/Predictions/'
# pred_fnames = []
# test_fname = 'chemprop-master/Whitehead_results/in_vitro_only/Scaffold_split/Base_model_only/fold_0/test_full.csv'
# all_preds = ['base_only.csv', 'morgan_counts.csv']
# evaluate_multiple_models(all_preds,test_fname, preds_dir)

# preds_dir = 'chemprop-master/Whitehead_results/in_vitro_only/Predictions/'
# pred_fnames = []
# test_fname = 'chemprop-master/Whitehead_results/in_vitro_only/Base_model_only/fold_0/test_full.csv'
# all_preds = ['base_only.csv', 'morgan_counts.csv', 'morgan_plus_rdkit.csv', 'rdkit_2d.csv']
# evaluate_multiple_models(all_preds,test_fname, preds_dir)
