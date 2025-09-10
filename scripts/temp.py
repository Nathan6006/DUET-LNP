import pandas as pd

temp = pd.read_csv("../data/data_files_to_merge/Xue_CAD_LNP/individual_metadata.csv")
keep = temp["Lipid_name"]
keep.to_csv("../data/data_files_to_merge/Xue_CAD_LNP/individual_metadata.csv", index=False)