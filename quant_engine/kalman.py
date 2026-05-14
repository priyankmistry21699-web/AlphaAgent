"""
AlphaAgent — Kalman Filter for Dynamic Beta Estimation

Uses a state-space model to estimate a time-varying beta coefficient
for any stock relative to the market. Unlike static OLS beta, Kalman
beta updates in real-time as new data arrives.

Model:
    State: x_t = [alpha_t, beta_t]  (evolves as a random walk)
    Observation: r_stock_t = alpha_t + beta_t * r_market_t + epsilon_t
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class KalmanResult:
    """Output from the Kalman Filter beta estimation."""
    beta: float             # Current dynamic beta
    alpha: float            # Current dynamic alpha (idiosyncratic return)
    spy_correlation: float  # Rolling 60-day correlation with market
    beta_series: list = field(default_factory=list)
    alpha_series: list = field(default_factory=list)


class KalmanBeta:
    """
    Estimates dynamic (time-varying) beta using a Kalman Filter.

    Usage:
        >>> model = KalmanBeta(stock_returns, spy_returns)
        >>> result = model.estimate()
        >>> print(f"Dynamic Beta: {result.beta:.2f}")
    """

    def __init__(
        self,
        stock_returns: pd.Series,
        market_returns: pd.Series,
        delta: float = 1e-4,
    ):
        """
        Args:
            stock_returns:  Daily log returns of the stock
            market_returns: Daily log returns of SPY (or any benchmark)
            delta:          State transition noise — lower = smoother beta evolution
        """
        self.stock_returns = stock_returns.dropna()
        self.market_returns = market_returns.dropna()
        self.delta = delta

    def estimate(self) -> KalmanResult:
        """Run the Kalman Filter and return the latest beta/alpha estimates."""
        # Align the two series on their common dates
        aligned = pd.concat(
            [self.stock_returns, self.market_returns], axis=1
        ).dropna()

        if len(aligned) < 30:
            logger.warning("Insufficient data for Kalman beta — returning beta=1.")
            return KalmanResult(beta=1.0, alpha=0.0, spy_correlation=0.0)

        y = aligned.iloc[:, 0].values   # stock returns
        x = aligned.iloc[:, 1].values   # market returns
        n = len(y)

        # State-transition noise covariance Q (2x2 identity scaled by delta)
        Q = (self.delta / (1.0 - self.delta)) * np.eye(2)

        # Observation noise R — estimated from first 20 residuals
        R = float(np.var(y[:20])) if n >= 20 else 1e-4

        # Initial state: alpha=0, beta=1 (neutral prior)
        theta = np.array([0.0, 1.0])
        P = np.eye(2) * 100.0   # Large initial uncertainty

        betas, alphas = [], []

        for t in range(n):
            H = np.array([[1.0, x[t]]])   # Observation matrix [1, r_market_t]

            # ── Prediction ──
            theta_pred = theta             # Random walk: state doesn't drift
            P_pred = P + Q

            # ── Innovation ──
            innovation = y[t] - float(H @ theta_pred)
            S = float(H @ P_pred @ H.T) + R

            # ── Kalman Gain ──
            K = (P_pred @ H.T) / S        # Shape (2,1) / scalar

            # ── Update ──
            theta = theta_pred + K.ravel() * innovation
            P = (np.eye(2) - np.outer(K.ravel(), H)) @ P_pred

            betas.append(float(theta[1]))
            alphas.append(float(theta[0]))

        # Rolling 60-day correlation
        corr_window = min(60, n)
        spy_corr = float(
            np.corrcoef(y[-corr_window:], x[-corr_window:])[0, 1]
        ) if corr_window >= 5 else 0.0

        return KalmanResult(
            beta=betas[-1],
            alpha=alphas[-1],
            spy_correlation=spy_corr,
            beta_series=betas,
            alpha_series=alphas,
        )
