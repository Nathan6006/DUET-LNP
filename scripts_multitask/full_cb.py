import sys
from train_cb import train_hybrid 
from analyze_cb import make_pred_vs_actual_tvt, analyze_predictions_cv_tvt

def main(argv):
    if len(argv) < 2:
        print("Usage: python train_cb.py {split_name} [--epochs N] [--cv N]")
        sys.exit(1)

    split_folder = argv[1]
    
    if 'del' in split_folder.lower():
        target_cols = ['quantified_delivery'] 
        print("Training for Delivery (delivery)")
    elif 'tox' in split_folder.lower():
        target_cols = ['quantified_toxicity'] 
        print("Training for Toxicity (viability)")
    else:
        print(f"Error: Could not infer mode (del/tox) from '{split_folder}'.")
        sys.exit(1)

    epochs = 3 
    cv_num = 5
    
    for i, arg in enumerate(argv):
        if arg.replace('–', '-') == '--epochs':
            epochs = int(argv[i+1])
            print('this many epochs: ', str(epochs))
        if arg.replace('–', '-') == '--cv':
            cv_num = int(argv[i+1])
            print('this many folds: ', str(cv_num))

    for cv in range(cv_num):
        split_dir = f'../data/crossval_splits/{split_folder}/cv_{cv}'
        save_dir = f'{split_dir}/model_{cv}'
        
        train_hybrid(
            split_dir=split_dir, 
            save_dir=save_dir, 
            cv_fold=cv, 
            target_columns=target_cols,
            epochs=epochs
        )

    test_dir = argv[1]
    
    # Check mode (del/tox)
    if 'del' in test_dir.lower():
        target_cols = ['quantified_delivery']
        print("Testing for Delivery (quantified_delivery)")
        # Adjust bins for delivery (assumed range, e.g. -10 to 10 for Z-scores or similar)
        analysis_bins = [-10, -1, 1, 10] 
    elif 'tox' in test_dir.lower():
        target_cols = ['quantified_toxicity']
        print("Testing for Toxicity (quantified_toxicity)")
        analysis_bins = [0.0, 0.7, 0.8, 1.1]
    else:
        # Default fallback
        print("Warning: Could not infer 'del' or 'tox' from name. Defaulting to 'quantified_toxicity'.")
        target_cols = ['quantified_toxicity']
        analysis_bins = [0.0, 0.7, 0.8, 1.1]

    cv_num = 5        
    model_dir = test_dir
    to_eval = ["test", "train", "valid"]
    
    # Parse optional args
    for i, arg in enumerate(argv):
        if arg.replace('–', '-') == '--diff_model':
            model_dir = argv[i+1]
        if arg.replace('–', '-') == '--cv':
            cv_num = int(argv[i+1])

    for tvt in to_eval:
        print(f"\n=== Processing: {tvt} ===")
        make_pred_vs_actual_tvt(
            test_dir, 
            model_dir, 
            ensemble_size=cv_num, 
            tvt=tvt,
            target_columns=target_cols
        )
        
        print(f"--- Analyzing: {tvt} ---")
        analyze_predictions_cv_tvt(
            test_dir, 
            ensemble_number=cv_num, 
            tvt=tvt,
            target_columns=target_cols,
            class_bins=analysis_bins
        )
        print(f"Done with {tvt}.")

if __name__ == '__main__':
    main(sys.argv)

