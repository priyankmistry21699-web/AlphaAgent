"""
AlphaAgent — Black-Litterman Portfolio Construction

Combines market-equilibrium implied returns (CAPM reverse-optimisation)
with active views from the AlphaAgent agent council to produce
posterior expected returns and an optimal portfolio weight vector.

Reference: Black & Litterman (1992), He & Litterman (1999).

Usage:
    from quant_engine.black_litterman import BlackLitterman
    weights = BlackLitterman().optimize(
        market_caps={"AAPL": 3e12, "MSFT": 2.8e12, ...},
        cov_matrix=cov_df,
        views={"AAPL": 0.10, "MSFT": -0.05},  # agent-implied 1y returns
        view_confidence={"AAPL": 0.8, "MSFT": 0.4},
    )
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BLResult:
    posterior_returns: Dict[str, float]
    weights: Dict[str, float]
    implied_returns: Dict[str, float]
    risk_aversion: float
    n_assets: int


class BlackLitterman:
    """
    Black-Litterman portfolio construction.

    risk_aversion δ : market price of risk (typically 2.5)
    tau            : scaling for prior covariance (typically 0.025–0.05)
    """

    def __init__(self, risk_aversion: float = 2.5, tau: float = 0.05):
        self.delta = risk_aversion
        self.tau   = tau

    def optimize(self,
                 market_caps: Dict[str, float],
                 cov_matrix: pd.DataFrame,
                 views: Dict[str, float],
                 view_confidence: Optional[Dict[str, float]] = None
                 ) -> Optional[BLResult]:
        """
        market_caps      : ticker → market cap (used for equilibrium weights)
        cov_matrix       : NxN return covariance matrix (annualised)
        views            : ticker → expected 1y return (e.g., 0.10 = 10%)
        view_confidence  : ticker → confidence ∈ (0, 1) used to set view variance
        """
        try:
            assets = list(cov_matrix.columns)
            n = len(assets)
            if n < 2:
                return None

            # 1. Market-cap equilibrium weights
            caps = np.array([market_caps.get(a, 1e9) for a in assets], dtype=float)
            w_mkt = caps / caps.sum()

            Sigma = cov_matrix.values
            # 2. Reverse-optimise: implied equilibrium returns
            pi = self.delta * Sigma @ w_mkt   # length n

            # 3. Build views matrix P, q, Omega
            valid_views = {a: v for a, v in views.items() if a in assets}
            if not valid_views:
                # No views → just use equilibrium weights
                weights = w_mkt
                posterior = pi
            else:
                k = len(valid_views)
                P = np.zeros((k, n))
                q = np.zeros(k)
                Omega_diag = np.zeros(k)
                view_assets = list(valid_views.keys())
                for i, a in enumerate(view_assets):
                    j = assets.index(a)
                    P[i, j] = 1.0
                    q[i] = valid_views[a]
                    # View variance from confidence
                    conf = (view_confidence or {}).get(a, 0.5)
                    conf = max(0.05, min(0.95, conf))
                    # Less confident → larger variance
                    Omega_diag[i] = (1.0 - conf) * float(P[i] @ Sigma @ P[i])
                Omega = np.diag(np.maximum(Omega_diag, 1e-8))

                # 4. Posterior mean (He-Litterman closed form)
                tau_Sigma = self.tau * Sigma
                try:
                    M_inv = np.linalg.inv(tau_Sigma) + P.T @ np.linalg.inv(Omega) @ P
                    M = np.linalg.inv(M_inv)
                    posterior = M @ (np.linalg.inv(tau_Sigma) @ pi + P.T @ np.linalg.inv(Omega) @ q)
                except np.linalg.LinAlgError:
                    posterior = pi

                # 5. Optimal weights from posterior
                try:
                    weights = np.linalg.inv(self.delta * Sigma) @ posterior
                except np.linalg.LinAlgError:
                    weights = w_mkt

                # Long-only normalisation (clip negatives, renormalise)
                weights = np.maximum(weights, 0)
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                else:
                    weights = w_mkt

            return BLResult(
                posterior_returns={a: float(posterior[i]) for i, a in enumerate(assets)},
                weights={a: float(weights[i]) for i, a in enumerate(assets)},
                implied_returns={a: float(pi[i]) for i, a in enumerate(assets)},
                risk_aversion=self.delta,
                n_assets=n,
            )
        except Exception as e:
            logger.warning(f"Black-Litterman failed: {e}")
            return None
