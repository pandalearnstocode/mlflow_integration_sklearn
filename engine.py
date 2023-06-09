from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import cvxpy as cp
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from joblib import Parallel, delayed
from loguru import logger


class LassoRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def _loss_fn(self, X, y, beta):
        return cp.norm2(X @ beta - y) ** 2

    def _regularizer(self, beta):
        return cp.norm1(beta)

    def _obj_fn(self, X, y, beta, lambd):
        return self._loss_fn(X, y, beta) + lambd * self._regularizer(beta)

    def _mse(self, X, Y, beta):
        return (1.0 / X.shape[0]) * self._loss_fn(X, Y, beta).value

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        n = X.shape[1]
        beta = cp.Variable(n)
        lambd = cp.Parameter(nonneg=True)
        problem = cp.Problem(cp.Minimize(self._obj_fn(X, y, beta, lambd)))
        lambd.value = self.alpha
        problem.solve()
        self.coeff_ = beta.value
        self.intercept_ = 0.0
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return X @ self.coeff_


class HoltWintersAtomic(BaseEstimator, RegressorMixin):
    def __init__(
        self, trend=None, damped_trend=False, seasonal=None, seasonal_periods=None, date_col = "date",date_freq = "D", target_col = "yt"
    ):
        self.trend = trend
        self.damped_trend = damped_trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.model_ = None
        self.date_col = date_col
        self.date_freq = date_freq
        self.target_col = "yt"

    def fit(self, X, y=None):
        X = X.set_index(self.date_col)
        X.index.freq = self.date_freq
        X = X[[self.target_col]]
        self.model_ = ExponentialSmoothing(
            X,
            trend=self.trend,
            damped_trend=self.damped_trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
        ).fit(method="SLSQP", optimized=True, remove_bias=True, use_brute=True)
        return self

    def predict(self, X):
        X = X.set_index(self.date_col)
        X.index.freq = self.date_freq
        df = self.model_.predict(start=X.index[0], end=X.index[-1]).to_frame(name = "yt_hat")
        df.index.name = self.date_col
        return df

    def score(self, X, y):
        pred_df = X[[self.date_col, self.target_col]].merge(self.predict(X), on=self.date_col, how="left")
        return (
            np.mean(
                np.abs(
                    (pred_df[self.target_col].to_numpy() - pred_df["yt_hat"].to_numpy())
                    / pred_df[self.target_col].to_numpy()
                )
            )
            * 100
        )


class HoltWintersPooled(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        trend=None,
        damped_trend=False,
        seasonal=None,
        seasonal_periods=None,
        n_jobs=1,
        date_col = "date",
        date_freq = "D",
        target_col = "yt",
        model_id_col = "model_id"
    ):
        self.trend = trend
        self.damped_trend = damped_trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.models = {}
        self.training_model_ids_ = []
        self.n_jobs = n_jobs
        self.date_col = date_col
        self.date_freq = date_freq
        self.target_col = target_col
        self.model_id_col = model_id_col

    def _fit(self, data):
        X_ = data[[self.date_col,self.target_col]]
        y_ = data[[self.date_col, self.target_col]]
        inner_model = HoltWintersAtomic(
            trend=self.trend,
            damped_trend=self.damped_trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
        )
        return inner_model.fit(X_, y_)

    def fit(self, X, y):
        self.training_model_ids = X[self.model_id_col].unique().tolist()
        if self.n_jobs == -1:
            self.models = Parallel(n_jobs=-1, verbose=1)(
                delayed(self._fit)(data) for _, data in X.groupby(self.model_id_col)
            )
        else:
            for model_id, data in X.groupby(self.model_id_col):
                X_ = data[[self.date_col, self.target_col]]
                y_ = data[[self.date_col, self.target_col]]
                self.models[model_id] = HoltWintersAtomic(
                    trend=self.trend,
                    damped_trend=self.damped_trend,
                    seasonal=self.seasonal,
                    seasonal_periods=self.seasonal_periods,
                )
                self.models[model_id].fit(X_, y_)
        return self

    def predict(self, X):
        self.predicting_model_ids = X[self.model_id_col].unique().tolist()
        self.model_ids_in_fit_not_in_prediction = list(
            set(self.training_model_ids) - set(self.predicting_model_ids)
        )
        self.model_ids_in_prediction_not_in_fit = list(
            set(self.predicting_model_ids) - set(self.training_model_ids)
        )
        if self.model_ids_in_fit_not_in_prediction:
            logger.info(
                "Model ids in fit but not in predict: ",
                ", ".join(map(str, self.model_ids_in_fit_not_in_prediction)),
            )
        if self.model_ids_in_prediction_not_in_fit:
            logger.critical(
                "Model ids in predict but not in fit: ",
                ", ".join(map(str, self.model_ids_in_prediction_not_in_fit)),
            )
        preds = []
        for model_id, data in X.groupby(self.model_id_col):
            X_ = data
            pred = self.models[model_id].predict(X_)
            pred[self.model_id_col] = model_id
            pred.index.name = self.date_col
            preds.append(pred)
        return (
            pd.concat(preds).reset_index().set_index([self.model_id_col, self.date_col]).sort_index()
        )

    def score(self, X, y, sample_weight=None):
        pred_df = self.predict(X).join(y.set_index([self.model_id_col, self.date_col]).sort_index())
        return (
            np.mean(
                np.abs(
                    (pred_df[self.target_col].to_numpy() - pred_df["yt_hat"].to_numpy())
                    / pred_df[self.target_col].to_numpy()
                )
            )
            * 100
        )
