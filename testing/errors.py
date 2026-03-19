import pandas as pd
import argparse
import os

def analyze_errors(split_folder):
    # Base path
    base_path = os.path.join('..', 'results', 'crossval_splits', split_folder, 'test')

    # Output directory
    output_dir = f"{split_folder}_errors"
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Starting Error Analysis for: {split_folder} ---")
    print(f"Reading from: {base_path}")
    print(f"Saving to:    {output_dir}/\n")

    # Iterate through CV folds
    for i in range(5):
        fold_name = f"cv_{i}"
        file_path = os.path.join(base_path, fold_name, 'predicted_vs_actual.csv')

        if not os.path.exists(file_path):
            print(f"[WARNING] File not found: {file_path}. Skipping fold {i}.")
            continue

        try:
            df = pd.read_csv(file_path)

            # ---- Infer active task from non-NaN prediction column ----
            pred_delivery = f"{fold_name}_pred_quantified_delivery"
            pred_toxicity = f"{fold_name}_pred_quantified_toxicity"

            delivery_valid = (
                pred_delivery in df.columns and
                df[pred_delivery].notna().any()
            )

            toxicity_valid = (
                pred_toxicity in df.columns and
                df[pred_toxicity].notna().any()
            )

            if delivery_valid and not toxicity_valid:
                mode = 'delivery'
                pred_col = pred_delivery
                actual_col = 'quantified_delivery'
            elif toxicity_valid and not delivery_valid:
                mode = 'toxicity'
                pred_col = pred_toxicity
                actual_col = 'quantified_toxicity'
            elif delivery_valid and toxicity_valid:
                raise ValueError(
                    f"Both delivery and toxicity predictions contain values in fold {i}. "
                    "Expected only one active target."
                )
            else:
                raise ValueError(
                    f"No valid prediction values found for fold {i}. "
                    "Both prediction columns are empty or missing."
                )

            # ---- Calculate absolute error ----
            df['abs_error'] = (df[pred_col] - df[actual_col]).abs()

            # ---- Extract top 20 worst errors ----
            worst_offenders = (
                df.sort_values(by='abs_error', ascending=False)
                  .head(20)
            )

            # ---- Save results ----
            output_filename = os.path.join(
                output_dir, f"{fold_name}_worst_errors_{mode}.csv"
            )
            worst_offenders.to_csv(output_filename, index=False)

            print(
                f"[Fold {i} | {mode}] "
                f"Max Error: {worst_offenders['abs_error'].max():.4f} "
                f"-> Saved to {output_filename}"
            )

        except Exception as e:
            print(f"[ERROR] Failed processing fold {i}: {e}")

    print(f"\nDone. Check the '{output_dir}' folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract worst model predictions per fold (delivery or toxicity)."
    )
    parser.add_argument(
        "split_folder",
        type=str,
        help="The name of the split/experiment folder to analyze."
    )

    args = parser.parse_args()
    analyze_errors(args.split_folder)