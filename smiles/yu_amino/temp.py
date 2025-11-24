import pandas as pd
import numpy as np

main = pd.read_csv("main_data_.csv")

main["quantified_toxicity"] = main["quantified_toxicity"] * 100


main.to_csv("main_data_.csv", index=False)