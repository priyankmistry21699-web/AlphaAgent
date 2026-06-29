"""
AlphaAgent — Quantile Regression for Asymmetric Tail Prediction

OLS regression predicts the conditional mean. Quantile regression predicts
arbitrary conditional quantiles — directly delivering asymmetric upside/
downside estimates.

For trading:
  - 5th percentile prediction → worst-case loss expectation
  - 50th percentile (median)  → robust central estimate
  - 95th percentile           → upside potential

Reference: Koenker & Bassett (1978).
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class QuantileResult:
    median_pred: float          # 50th percentile prediction
    p05_pred: float             # 5th percentile (tail loss)
    p25_pred: float
    p75_pred: float
    p95_pred: float             # 95th percentile (upside)
    asymmetry: float            # upside_dist - downside_dist (positive = bullish skew)
    skew_premium: float         # (p95 - median) - (median - p05)
    n_obs: int


def quantile_regression(X: np.ndarray, y: np.ndarray,
                         quantile: float = 0.5,
                         max_iter: int = 1000,
                         tol: float = 1e-6) -> Optional[np.ndarray]:
    """
    Solve quantile regression via iteratively re-weighted least squares (IRLS).

    Returns coefficients (1D array of length X.shape[1] + 1, with intercept).
    """
    try:
        n, p = X.shape
        X_aug = np.column_stack([np.ones(n), X])
        # OLS starting point
        beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)

        for _ in range(max_iter):
            resid = y - X_aug @ beta
            # IRLS weights for quantile loss
            w = np.where(resid >= 0, quantile, 1 - quantile)
            w = w / (np.maximum(np.abs(resid), tol))
            W = np.diag(w)
            try:
                # Weighted least squares
                XtWX = X_aug.T @ W @ X_aug
                XtWy = X_aug.T @ W @ y
                beta_new = np.linalg.solve(XtWX, XtWy)
            except np.linalg.LinAlgError:
                break
            if np.max(np.abs(beta_new - beta)) < tol:
                beta = beta_new
                break
            beta = beta_new
        return beta
    except Exception as e:
        logger.warning(f"Quantile regression IRLS failed: {e}")
        return None


def quantile_forecast(features_history: pd.DataFrame,
                       target_history: pd.Series,
                       features_today: pd.Series,
                       quantiles: List[float] = None) -> Optional[QuantileResult]:
    """
    Fit quantile regressions at multiple quantiles and predict today's outcome
    distribution.

    features_history : NxK DataFrame of historical predictors
    target_history   : N-length series of historical returns (or target)
    features_today   : K-length series of current feature values
    """
    if quantiles is None:
        quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    try:
        # Align
        df = features_history.copy()
        df["__y__"] = target_history.values
        df = df.dropna()
        if len(df) < 50:
            return None
        X = df.drop(columns=["__y__"]).values
        y = df["__y__"].values
        x_today = features_today.values.reshape(1, -1)
        x_today_aug = np.column_stack([np.ones(1), x_today])

        preds = {}
        for q in quantiles:
            beta = quantile_regression(X, y, quantile=q)
            if beta is None:
                continue
            preds[q] = float(x_today_aug @ beta)

        if 0.5 not in preds:
            return None
        median = preds[0.5]
        p05    = preds.get(0.05, median)
        p25    = preds.get(0.25, median)
        p75    = preds.get(0.75, median)
        p95    = preds.get(0.95, median)
        asymmetry = (p95 - median) - (median - p05)
        skew_premium = asymmetry

        return QuantileResult(
            median_pred=round(median, 5),
            p05_pred=round(p05, 5),
            p25_pred=round(p25, 5),
            p75_pred=round(p75, 5),
            p95_pred=round(p95, 5),
            asymmetry=round(asymmetry, 5),
            skew_premium=round(skew_premium, 5),
            n_obs=len(df),
        )
    except Exception as e:
        logger.warning(f"Quantile forecast failed: {e}")
        return None
