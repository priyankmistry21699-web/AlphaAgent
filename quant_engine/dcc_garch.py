"""
AlphaAgent — DCC-GARCH (Dynamic Conditional Correlation)

Estimates time-varying correlations between a stock and SPY using the
Engle (2002) DCC-GARCH approach:
  1. Fit univariate GARCH(1,1) to each series → standardised residuals
  2. Fit DCC(1,1) to the residuals → Q_t → R_t (correlation matrix)
  3. Return current correlation, 20-day rolling vs baseline, and crisis flag

Usage:
    from quant_engine.dcc_garch import DCCGarch, DCCResult
    result = DCCGarch().fit(stock_returns, spy_returns)
    # result.current_corr, result.corr_trend, result.crisis_flag
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DCCResult:
    current_corr: float        # current dynamic correlation (−1 to +1)
    baseline_corr: float       # 252-day unconditional correlation
    corr_trend: float          # current vs 20-day avg (positive = rising corr)
    corr_percentile: float     # percentile vs full history (0–100)
    crisis_flag: bool          # True if corr spike >0.15 above 20d avg
    q_diag: float              # DCC Q matrix off-diagonal (raw)
    n_obs: int


class DCCGarch:
    """
    Simplified DCC-GARCH(1,1) estimator using moment-based parameter init.
    Full MLE is expensive; this uses a two-step Engle approach with
    fixed DCC parameters (a=0.05, b=0.93) calibrated to equity markets.
    """

    def __init__(self, a: float = 0.05, b: float = 0.93):
        self.a = a    # DCC shock coefficient
        self.b = b    # DCC persistence coefficient

    def _garch_standardise(self, returns: np.ndarray) -> np.ndarray:
        """
        Fit GARCH(1,1) via moment matching and return standardised residuals.
        omega = (1 - alpha - beta) * var(returns)
        alpha = 0.10, beta = 0.85 (typical equity calibration)
        """
        alpha, beta = 0.10, 0.85
        omega = (1 - alpha - beta) * np.var(returns)
        omega = max(omega, 1e-8)

        n = len(returns)
        h = np.full(n, np.var(returns))
        for t in range(1, n):
            h[t] = omega + alpha * returns[t - 1] ** 2 + beta * h[t - 1]
            h[t] = max(h[t], 1e-8)

        std_resid = returns / np.sqrt(h)
        return std_resid

    def fit(self, stock_returns: pd.Series,
            spy_returns: pd.Series) -> Optional[DCCResult]:
        """
        Fit DCC-GARCH to stock + SPY return series.
        Both series must be daily pct_change (not multiplied by 100).
        Returns None if insufficient data (<60 observations).
        """
        try:
            s1 = stock_returns.dropna().values.astype(float)
            s2 = spy_returns.dropna().values.astype(float)

            # Align lengths
            n = min(len(s1), len(s2))
            if n < 60:
                return None
            s1, s2 = s1[-n:], s2[-n:]

            # Step 1: Univariate GARCH standardisation
            e1 = self._garch_standardise(s1)
            e2 = self._garch_standardise(s2)

            # Step 2: DCC(1,1) recursion
            # Q_bar = unconditional covariance of standardised residuals
            # RMT-cleaned for noise reduction (Marchenko-Pastur)
            try:
                from quant_engine.rmt import clean_correlation_matrix
                _resid_df = pd.DataFrame({"e1": e1, "e2": e2})
                _cleaned = clean_correlation_matrix(_resid_df)
                if _cleaned is not None:
                    Q_bar = _cleaned.values
                else:
                    Q_bar = np.cov(np.stack([e1, e2], axis=0))
            except Exception:
                Q_bar = np.cov(np.stack([e1, e2], axis=0))
            Q_bar = np.maximum(Q_bar, np.eye(2) * 1e-8)

            Q = Q_bar.copy()
            corr_series = np.zeros(n)
            a, b = self.a, self.b

            for t in range(1, n):
                e_outer = np.array([[e1[t - 1] ** 2, e1[t - 1] * e2[t - 1]],
                                    [e1[t - 1] * e2[t - 1], e2[t - 1] ** 2]])
                Q = (1 - a - b) * Q_bar + a * e_outer + b * Q
                Q = np.maximum(Q, np.eye(2) * 1e-8)
                # Correlation from Q
                denom = np.sqrt(Q[0, 0] * Q[1, 1])
                corr_series[t] = Q[0, 1] / denom if denom > 0 else 0.0

            corr_series = np.clip(corr_series[1:], -0.999, 0.999)

            current_corr   = float(corr_series[-1])
            baseline_corr  = float(np.corrcoef(s1, s2)[0, 1])
            rolling_20     = float(np.mean(corr_series[-20:])) if len(corr_series) >= 20 else current_corr
            corr_trend     = current_corr - rolling_20
            corr_pct       = float(np.mean(corr_series <= current_corr) * 100)
            crisis_flag    = bool(corr_trend > 0.15)

            return DCCResult(
                current_corr=round(current_corr, 4),
                baseline_corr=round(baseline_corr, 4),
                corr_trend=round(corr_trend, 4),
                corr_percentile=round(corr_pct, 1),
                crisis_flag=crisis_flag,
                q_diag=round(float(Q[0, 1]), 6),
                n_obs=n,
            )

        except Exception as e:
            logger.warning(f"DCC-GARCH failed: {e}")
            return None
