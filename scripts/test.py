import torch, pickle, numpy as np, pandas as pd
from pathlib import Path

def single_sample_forward_check(model_ckpt_path, scaler_path, csv_row, extra_x_row, smiles_col='smiles'):
    # 1) Build MoleculeDatapoint, featurizer and dataset exactly as you do in train & predict
    from chemprop import data, featurizers, models  # replace import path to your module
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    dp_train = data.MoleculeDatapoint.from_smi(csv_row[smiles_col], y=csv_row.get('quantified_delivery', None), x_d=extra_x_row)
    dp_predict = data.MoleculeDatapoint.from_smi(csv_row[smiles_col], x_d=extra_x_row)

    # 2) Make identical datasets
    d_train = data.MoleculeDataset([dp_train], featurizer=featurizer)
    d_predict = data.MoleculeDataset([dp_predict], featurizer=featurizer)

    # 3) Print raw X_d and shapes (before scaling)
    print("raw train.x_d:", d_train[0].x_d, "shape", getattr(d_train[0], "x_d", None).shape)
    print("raw predict.x_d:", d_predict[0].x_d, "shape", getattr(d_predict[0], "x_d", None).shape)

    # 4) Load scaler and normalize both (if scaler exists)
    if Path(scaler_path).exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        d_train.normalize_inputs("X_d", scaler)
        d_predict.normalize_inputs("X_d", scaler)
        print("after scaling train.x_d:", d_train[0].x_d)
        print("after scaling predict.x_d:", d_predict[0].x_d)
    else:
        print("scaler not found at", scaler_path)

    # 5) Load model from checkpoint and run one forward pass on both
    model = models.MPNN.load_from_checkpoint(str(model_ckpt_path))
    model.eval()
    with torch.inference_mode():
        # you may need to create a dataloader or call model.forward with tensors exactly like your library does
        out_train = model.forward(d_train[0].to_batch()) if hasattr(d_train[0], "to_batch") else model(d_train[0])
        out_predict = model.forward(d_predict[0].to_batch()) if hasattr(d_predict[0], "to_batch") else model(d_predict[0])

    print("model output train:", out_train)
    print("model output predict:", out_predict)

# Example usage:
df = pd.read_csv('../data/crossval_splits/cm/test/test.csv')
extra = pd.read_csv('../data/crossval_splits/cm/test/test_extra_x.csv')
single_sample_forward_check('../data/crossval_splits/cm/cv_0/model_0/best.ckpt',
                            '../data/crossval_splits/cm/cv_0/model_0/extra_features_scaler.pkl',
                            df.iloc[0].to_dict(),
                            extra.iloc[0].to_numpy(dtype=float))
