import pandas as pd
import numpy as np
from typing import Dict
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig, OutputType, Normalization

def _synthetic_data_lasso(m=100, n=20, sigma=5, density=0.2) -> pd.DataFrame:
    np.random.seed(1)
    beta_star = np.random.randn(n)
    idxs = np.random.choice(range(n), int((1-density)*n), replace=False)
    for idx in idxs:
        beta_star[idx] = 0
    X = np.random.randn(m,n)
    Y = X.dot(beta_star) + np.random.normal(0, sigma, size=m)
    return X, Y, beta_star

def _synthetic_data_atomic(n = 1000, time_period = 30) -> pd.DataFrame:
    df = pd.DataFrame(np.random.random(size=(n,time_period)))
    df.columns = pd.date_range("2022-01-01", periods=time_period)
    model = DGAN(DGANConfig(
    max_sequence_len=time_period,
    sample_len=2,
    batch_size=n,
    epochs=10,
    ))
    model.train_dataframe(
    df
    )
    synthetic_df = model.generate_dataframe(time_period)
    df = synthetic_df.T
    df.columns = [f"x_{ix}" for ix in range(synthetic_df.shape[0])]
    df['yt'] = np.random.uniform(0,1, size = df.shape[0])
    df.index.name = "date"
    return df

def _synthetic_data_pooled(n_models = 2, n = 1000, time_period = 30) -> pd.DataFrame:
    dc = []
    for ix in range(n_models):
        df = _synthetic_data_atomic(n = n, time_period = time_period)
        df[["model_id"]] = ix
        dc.append(df)
    df_ = pd.concat(dc)
    df_.index.name = "date"
    return df_

def generate_synthetic_data(model_name: str = "lasso", synthetic_data_config: Dict = None) -> pd.DataFrame:
    if not synthetic_data_config:
        raise ValueError("Synthetic data config cannot be empty")
    if model_name == "lasso":
        X, Y, _ =  _synthetic_data_lasso(**synthetic_data_config)
        df = pd.DataFrame(X)
        df.columns = [f"x_{ix}" for ix in range(X.shape[1])]
        df["y"] = Y
        return df
    elif model_name == "atomic":
        return _synthetic_data_atomic(**synthetic_data_config)
    elif model_name == "pooled":
        return _synthetic_data_pooled(**synthetic_data_config)
    else:
        raise ValueError(f"Unknown model name {model_name}")