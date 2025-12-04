import sys
from training_funcs import train_basic, train_cm
from analyze_funcs import make_pred_vs_actual_tvt, analyze_predictions_cv_tvt

#sys.argv = ['tox_testing.py', 'xg_1.1', '--basic', '--cv', '5']

def main(argv):
    split_folder = argv[1]
    epochs = 50
    cv_num = 2
    basic=True
    for i, arg in enumerate(argv):
        if arg.replace('–', '-') == '--epochs':
            epochs = int(argv[i+1])
            print('this many epochs: ',str(epochs))
        if arg.replace('–', '-') == '--cv':
            cv_num = int(argv[i+1])
            print('this many folds: ',str(cv_num))
        if arg.replace('–', '-') == '--basic':
            print("using basic model")
            basic = True
    if basic:
        for cv in range(cv_num):
            split_dir = '../data/crossval_splits/'+split_folder+'/cv_'+str(cv)
            save_dir = split_dir+'/model_'+str(cv)
            train_basic(split_dir=split_dir, save_dir=save_dir, model_type="xg")
    else:      
        for cv in range(cv_num):
            split_dir = '../data/crossval_splits/'+split_folder+'/cv_'+str(cv)
            save_dir = split_dir+'/model_'+str(cv)
            train_cm(split_dir=split_dir,epochs=epochs, save_dir=save_dir)
        
    
    test_dir = split_folder
    model_dir = test_dir
    s = False
    to_eval = ["test", "train", "valid"]
    for i, arg in enumerate(argv):
        if arg.replace('–', '-') == '--diff_model':
            model_dir = argv[i+1]
        if arg.replace('–', '-') == '--standardize':
            s = True 
            print('standardize')
    for tvt in to_eval:
        print("make pva")
        make_pred_vs_actual_tvt(test_dir, model_dir, ensemble_size = cv_num, standardize_predictions= s, tvt=tvt, rf=basic)
        print("analyze preds")
        print(cv)
        analyze_predictions_cv_tvt(test_dir, ensemble_number= cv_num, tvt=tvt)
        print("done with:", tvt)


    
if __name__ == '__main__':
    main(sys.argv)
