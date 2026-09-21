from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from . import config


def fit_ridge_cv(
    X: np.ndarray,
    Y: np.ndarray,
    alphas=config.RIDGE_ALPHAS,
    n_folds: int = config.CV_FOLDS,
    seed: int = config.RANDOM_SEED,
):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_mse = {}
    for a in alphas:
        errs = []
        for tr, va in kf.split(X):
            m = Ridge(alpha=a, fit_intercept=False)
            m.fit(X[tr], Y[tr])
            pred = m.predict(X[va])
            errs.append(float(np.mean((pred - Y[va]) ** 2)))
        cv_mse[a] = float(np.mean(errs))
    best_alpha = min(cv_mse, key=cv_mse.get)
    model = Ridge(alpha=best_alpha, fit_intercept=False)
    model.fit(X, Y)
    return model, best_alpha, cv_mse


def protein_wise_corr(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    yt = Y_true - Y_true.mean(axis=0, keepdims=True)
    yp = Y_pred - Y_pred.mean(axis=0, keepdims=True)
    num = (yt * yp).sum(axis=0)
    den = np.sqrt((yt**2).sum(axis=0) * (yp**2).sum(axis=0))
    return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)


def patient_wise_corr(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    yt = Y_true - Y_true.mean(axis=1, keepdims=True)
    yp = Y_pred - Y_pred.mean(axis=1, keepdims=True)
    num = (yt * yp).sum(axis=1)
    den = np.sqrt((yt**2).sum(axis=1) * (yp**2).sum(axis=1))
    return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)


def r2_vs_null(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    ss_res = float(((Y_true - Y_pred) ** 2).sum())
    ss_tot = float((Y_true**2).sum())
    return 1.0 - ss_res / ss_tot


def summarize(Y_true: np.ndarray, Y_pred: np.ndarray, label: str) -> pd.Series:
    pw = protein_wise_corr(Y_true, Y_pred)
    tw = patient_wise_corr(Y_true, Y_pred)
    return pd.Series(
        {
            "protein_corr_median": float(np.nanmedian(pw)),
            "protein_corr_mean": float(np.nanmean(pw)),
            "frac_protein_corr_pos": float(np.nanmean(pw > 0)),
            "patient_corr_median": float(np.nanmedian(tw)),
            "R2_vs_null": r2_vs_null(Y_true, Y_pred),
        },
        name=label,
    )
