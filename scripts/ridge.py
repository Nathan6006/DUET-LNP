from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator




def smiles_to_morgan(smiles, radius=2, nBits=512):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits)
    morgan_gen = GetMorganGenerator(radius=radius, fpSize=nBits)  # set radius and size
    fp = morgan_gen.GetFingerprint(mol)    
    arr = np.zeros((nBits,), dtype=int)
    from rdkit import DataStructs
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def run_models(alpha):
    # Ridge Regression
    ridge = Ridge(alpha=alpha)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    ridge_cv_r2 = cross_val_score(ridge, X_fingerprints, y_values, cv=kf, scoring='r2')
    ridge.fit(X_fingerprints, y_values)
    ridge_pred = ridge.predict(X_fingerprints)
    print(f"\nalpha value: {alpha}")
    print("Ridge Regression:")
    print("Train R²:", r2_score(y_values, ridge_pred))
    print("Train MSE:", mean_squared_error(y_values, ridge_pred))
    print("5-fold CV R²:", ridge_cv_r2.mean())

    # Lasso Regression
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso_cv_r2 = cross_val_score(lasso, X_fingerprints, y_values, cv=kf, scoring='r2')
    lasso.fit(X_fingerprints, y_values)
    lasso_pred = lasso.predict(X_fingerprints)
    print("\nLasso Regression:")
    print("Train R²:", r2_score(y_values, lasso_pred))
    print("Train MSE:", mean_squared_error(y_values, lasso_pred))
    print("5-fold CV R²:", lasso_cv_r2.mean())

    # Random Forest Regression
    rf = RandomForestRegressor(
        n_estimators=200,   # number of trees
        max_depth=None,     # let trees expand fully
        random_state=42,
        n_jobs=-1
    )
    rf_cv_r2 = cross_val_score(rf, X_fingerprints, y_values, cv=kf, scoring='r2')
    rf.fit(X_fingerprints, y_values)
    rf_pred = rf.predict(X_fingerprints)
    print("\nRandom Forest Regression:")
    print("Train R²:", r2_score(y_values, rf_pred))
    print("Train MSE:", mean_squared_error(y_values, rf_pred))
    print("5-fold CV R²:", rf_cv_r2.mean())


if __name__ == '__main__':
    tox = pd.read_csv("toxicity.csv")
    scaler = MinMaxScaler(feature_range=(0, 1))
    tox["toxicity_normalized"] = scaler.fit_transform(tox[["toxicity"]])
    X = list(tox["smiles"])
    Y = list(tox["toxicity_normalized"])

    smiles_list = X
    y_values = Y
    X_fingerprints = np.array([smiles_to_morgan(s) for s in smiles_list])

    alpha_values = [0.001, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.5]
    for alpha in alpha_values:
        run_models(alpha)
