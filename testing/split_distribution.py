import argparse
from pathlib import Path
import pandas as pd


def collect_split_distributions(split_folder, out_csv="split_distributions.csv"):
    base_dir = Path("../data/crossval_splits") / split_folder
    rows = []

    for i in range(5):
        split_dir = base_dir / f"cv_{i}"
        test_path = base_dir / "test" / "test.csv"

        train_path = split_dir / "train.csv"
        valid_path = split_dir / "valid.csv"

        if not train_path.exists() or not valid_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"Missing files in {split_dir} or test folder")

        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)
        test_df = pd.read_csv(test_path)

        rows.append({
            "split": f"cv_{i}",
            "train_size": len(train_df),
            "valid_size": len(valid_df),
            "test_size": len(test_df)
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved distributions to {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_folder", required=True, help="Name of crossval split folder")
    parser.add_argument("--out_csv", default="split_distributions.csv", help="Output CSV path")

    args = parser.parse_args()

    collect_split_distributions(args.split_folder, args.out_csv)


if __name__ == "__main__":
    main()