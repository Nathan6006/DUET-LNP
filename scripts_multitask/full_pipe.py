import sys
from train_ import train_basic, train_cm, PossMSEObjective
from analyze import make_pred_vs_actual_tvt, analyze_predictions_cv_tvt

def main(argv):
    split_folder = argv[1]
    epochs = 50
    cv_num = 5
    basic = True
    
    # Regression target
    mode_arg = split_folder.split("_")[1]
    if mode_arg == 'del':
        target_cols = ['quantified_delivery']
        print("Testing for Delivery (quantified_delivery)")
        # Generic quartiles for delivery since scale is unknown (0-100 or 0-1)
        # Using [0, 0.25, 0.5, 0.75, 1.0] assuming 0-1 norm, or adapt if you expect 0-100
        analysis_bins = [-10, -1, 1, 10] 
    elif mode_arg == 'tox':
        target_cols = ['quantified_toxicity']
        print("Testing for Toxicity (quantified_toxicity)")
        analysis_bins = [0.0, 0.7, 0.8, 1.1]
    else:
        print(f"Error: Unknown mode '{mode_arg}'. Use 'del' or 'tox'.")
        sys.exit(1)

    for i, arg in enumerate(argv):
        if arg.replace('–', '-') == '--epochs':
            epochs = int(argv[i+1])
            print('this many epochs: ', str(epochs))
        if arg.replace('–', '-') == '--cv':
            cv_num = int(argv[i+1])
            print('this many folds: ', str(cv_num))
        if arg.replace('–', '-') == '--basic':
            print("using basic model")
            basic = True
            
    if basic:
        for cv in range(cv_num):
            split_dir = '../data/crossval_splits/' + split_folder + '/cv_' + str(cv)
            save_dir = split_dir + '/model_' + str(cv)
            # Pass regression target to training function
            train_basic(
                split_dir=split_dir, 
                save_dir=save_dir, 
                cv_fold=cv, 
                target_columns=target_cols
            )
    else:      
        for cv in range(cv_num):
            split_dir = '../data/crossval_splits/' + split_folder + '/cv_' + str(cv)
            save_dir = split_dir + '/model_' + str(cv)
            # Assuming train_cm is also updated or agnostic to target columns in your implementation
            # If train_cm needs explicit targets, add target_columns=target_cols here as well
            train_cm(split_dir=split_dir, epochs=epochs, save_dir=save_dir)
    
    cv_num = 5        
    test_dir = argv[1]
    model_dir = test_dir
    to_eval = ["test", "train", "valid"]
    
    for i, arg in enumerate(argv):
        if arg.replace('–', '-') == '--diff_model':
            model_dir = argv[i+1]

    for tvt in to_eval:
        print(f"Making predictions vs actual for: {tvt}")
        make_pred_vs_actual_tvt(
            test_dir, 
            model_dir, 
            ensemble_size=cv_num, 
            tvt=tvt,
            target_columns=target_cols
        )
        
        print(f"Analyzing predictions for: {tvt}")
        analyze_predictions_cv_tvt(
            test_dir, 
            ensemble_number=cv_num, 
            tvt=tvt,
            target_columns=target_cols, 
            class_bins=analysis_bins
        )
        print("Done with:", tvt)
        
    


if __name__ == '__main__':
    main(sys.argv)