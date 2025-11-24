import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from smiles.combine import attach_tails_to_core
import pandas as pd

cores = pd.read_csv("cores.csv")
tails = pd.read_csv("tails.csv")
ep_tails = tails.head(6)
Ac_tails = tails.tail(5)
data = []
for i, core in cores.iterrows(): 
    for h, ep_tail in ep_tails.iterrows():
         identifier = core["Name"] + "-" + ep_tail ["Name"]
         smile = attach_tails_to_core(core["smiles_r"], ep_tail["smiles_r"])
         data.append({"identifier": identifier, "smiles": smile})

for i, core in cores.iterrows(): 
    for h, Ac_tail in Ac_tails.iterrows():
         identifier = core["Name"] + "-" + Ac_tail ["Name"]
         smile = attach_tails_to_core(core["smiles_r"], Ac_tail["smiles_r"])
         data.append({"identifier": identifier, "smiles": smile})

df = pd.DataFrame(data)
df.to_csv("data.csv", index=False)