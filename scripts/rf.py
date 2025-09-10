import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from rdkit import Chem
from rdkit.Chem import AllChem

df = pd.read_csv("toxicity_data.csv") #has smiles, toxicity

def smiles_to_fp(smiles, radius=2, n_bits=512):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    else:
        return None

fps = [smiles_to_fp(s) for s in df["smiles"]]
fps = [list(fp) for fp in fps if fp is not None]

X = pd.DataFrame(fps)
y = df["toxicity"].iloc[:len(X)] 

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Random Forest baseline:")
print(f"R²: {r2:.3f}")
print(f"MSE: {mse:.3f}")

scores = cross_val_score(rf, X, y, cv=5, scoring="r2")
print("5-fold CV R²:", scores.mean())