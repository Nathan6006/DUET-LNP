import pandas as pd
import numpy as np
import sys

def save_dist_to_csv(
    output_filename="dataset_data_dist.csv",
    folder='../data/all_data.csv',
    bins=None):
    """
    bins: list of bin edges (must be increasing), e.g. [0, 75, 90, 100]
          Highest-value bin will be labeled Class 0
    """

    if bins is None:
        raise ValueError("You must provide bin edges via the `bins` argument.")

    # Load data
    try:
        df = pd.read_csv(folder)
    except FileNotFoundError:
        print(f"Error: File {folder} not found.")
        return

    # Ensure numeric
    df["unnormalized_toxicity"] = pd.to_numeric(df["unnormalized_toxicity"], errors="coerce")

    # Get unique experiments
    experiments = df["Experiment_ID"].unique()

    num_classes = len(bins) - 1
    results = []

    for exp_id in experiments:
        dataset = df[df["Experiment_ID"] == exp_id]
        tox = dataset["unnormalized_toxicity"].dropna()
        total_count = len(tox)

        if total_count == 0:
            counts = np.zeros(num_classes, dtype=int)
        else:
            counts, _ = np.histogram(tox, bins=bins)

        # Reverse so highest bin → Class 0
        counts = counts[::-1]

        row = {
            "Dataset": str(exp_id),
            "Total": total_count
        }

        for i in range(num_classes):
            row[f"Class_{i}"] = counts[i]

        results.append(row)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Dataset")

    # ---- ADD TOTAL ROW ----
    total_row = {"Dataset": "Total"}
    for col in results_df.columns:
        if col != "Dataset":
            total_row[col] = results_df[col].sum()

    results_df = pd.concat(
        [results_df, pd.DataFrame([total_row])],
        ignore_index=True
    )
    # -----------------------

    results_df.to_csv(output_filename, index=False)
    print(f"Successfully saved distribution data to '{output_filename}'")
    print("-" * 30)
    print(results_df.to_string(index=False))


def main(argv):
    if len(argv)>1:
        save_dist_to_csv(
            folder=argv[1],
            bins=[0, 70, 80, 90, 200]
        )
    else:
        save_dist_to_csv(
            bins=[0, 70, 80, 90, 200]
        )

if __name__ == "__main__":
    main(sys.argv)