from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

def smiles_to_morgan(smiles, radius=2, nBits=512):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def main(alpha):
    #regularization value
    alpha_value = alpha
    print("\n\n\n")
    print("alpha value:", alpha_value)
    ridge = Ridge(alpha=alpha_value)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    ridge_cv_r2 = cross_val_score(ridge, X_fingerprints, y_values, cv=kf, scoring='r2')
    ridge.fit(X_fingerprints, y_values)
    ridge_pred = ridge.predict(X_fingerprints)
    print("Ridge Regression:")
    print("R²:", r2_score(y_values, ridge_pred))
    print("MSE:", mean_squared_error(y_values, ridge_pred))
    print("5-fold CV R²:", ridge_cv_r2.mean())

    lasso = Lasso(alpha=alpha_value, max_iter=10000)
    lasso_cv_r2 = cross_val_score(lasso, X_fingerprints, y_values, cv=kf, scoring='r2')
    lasso.fit(X_fingerprints, y_values)
    lasso_pred = lasso.predict(X_fingerprints)
    print("\nLasso Regression:")
    print("R²:", r2_score(y_values, lasso_pred))
    print("MSE:", mean_squared_error(y_values, lasso_pred))
    print("5-fold CV R²:", lasso_cv_r2.mean())

if __name__ == '__main__':
    print("here")
    tox = pd.read_csv("delivery.csv")
    X = list(tox["smiles"])
    Y = list(tox["delivery"])
    smiles_list = X 
    y_values = Y    
    X_fingerprints = np.array([smiles_to_morgan(s) for s in smiles_list])

    alpha = [0.001, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.5]
    for i in alpha:
        main(i)