import os
import pandas as pd

base_dir = "../data_files"
folders = [
    "Farbiak_Dendrimer_HeLa",
    "Farbiak_Dendrimer_igrov1",
    "Han_a3",
]

for folder in folders:
    csv_path = os.path.join(base_dir, folder, "formulations.csv")

    if not os.path.exists(csv_path):
        print(f"Missing: {csv_path}")
        continue

    df = pd.read_csv(csv_path)

    # Create new column
    df["mRNA/Cells"] = (
        df["Lipid/Cells"] / df["Ionizable_lipid_to_mRNA_weight_ratio"]
    )

    # Save back to the same file
    df.to_csv(csv_path, index=False)

    print(f"Updated: {csv_path}")