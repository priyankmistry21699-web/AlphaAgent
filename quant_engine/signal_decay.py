"""
AlphaAgent — Signal Decay & Half-Life Engine

Uses the Ornstein-Uhlenbeck process to determine whether a time series is
mean-reverting (ranging) or trending, and calculates the exact mathematical
half-life of a trading signal.

An Augmented Dickey-Fuller (ADF) test first confirms stationarity before
accepting any OLS-derived half-life — without it, a non-stationary series
can spuriously produce a negative lambda.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DecayResult:
    is_mean_reverting: bool
    half_life_days: float
    optimal_hold_min: int
    optimal_hold_max: int
    adf_pvalue: float        # ADF p-value (< 0.05 = stationary at 95% conf)
    adf_stationary: bool     # True if ADF rejects unit root at 10% level


class SignalDecay:
    """
    Calculates signal decay using the Ornstein-Uhlenbeck process.

    ADF gate: if the log-price series cannot reject the unit-root null
    (p-value > 0.10), we classify the series as trending regardless of the
    OLS lambda sign, preventing spurious half-life estimates.
    """

    ADF_SIGNIFICANCE = 0.10   # reject unit root at 10% level

    def __init__(self, prices: pd.Series):
        self.prices = prices.dropna()

    def calculate_half_life(self) -> DecayResult:
        if len(self.prices) < 30:
            return DecayResult(
                is_mean_reverting=False, half_life_days=0.0,
                optimal_hold_min=1, optimal_hold_max=5,
                adf_pvalue=1.0, adf_stationary=False,
            )

        log_prices = np.log(self.prices)

        # ── 1. ADF stationarity gate ──────────────────────────────────────────
        try:
            adf_stat, adf_pval, _, _, _, _ = adfuller(log_prices.values, maxlag=1, autolag=None)
            adf_stationary = adf_pval < self.ADF_SIGNIFICANCE
        except Exception:
            adf_pval = 1.0
            adf_stationary = False

        # ── 2. OLS for OU mean-reversion speed ───────────────────────────────
        dy = log_prices.diff().dropna()
        y_lag = log_prices.shift(1).dropna()
        min_len = min(len(dy), len(y_lag))
        dy = dy.iloc[-min_len:]
        y_lag = y_lag.iloc[-min_len:]

        try:
            X = sm.add_constant(y_lag.values)
            model = sm.OLS(dy.values, X)
            res = model.fit()
            lam = float(res.params[1])
        except Exception:
            lam = 0.0

        # Series is mean-reverting only when ADF confirms stationarity AND
        # lambda is negative (the necessary OU condition).
        is_mean_reverting = adf_stationary and (lam < 0)

        if is_mean_reverting:
            half_life = -np.log(2) / lam
            half_life = min(252.0, max(1.0, half_life))

            min_hold = max(1, int(half_life * 0.5))
            max_hold = max(min_hold + 1, int(half_life))
            min_hold = min(126, min_hold)
            max_hold = min(252, max_hold)
        else:
            half_life = 0.0
            min_hold = 10
            max_hold = 30

        return DecayResult(
            is_mean_reverting=is_mean_reverting,
            half_life_days=round(half_life, 1),
            optimal_hold_min=min_hold,
            optimal_hold_max=max_hold,
            adf_pvalue=round(float(adf_pval), 4),
            adf_stationary=adf_stationary,
        )
