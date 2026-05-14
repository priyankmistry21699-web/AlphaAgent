"""
AlphaAgent — Portfolio Optimizer

Markowitz mean-variance optimization and risk-parity allocation across
multiple assets. Requires scipy for SLSQP minimization; falls back to
equal-weight allocation if scipy is unavailable.

Supported methods:
  max_sharpe     — maximize Sharpe ratio on the efficient frontier
  min_variance   — minimize portfolio variance (lowest-risk portfolio)
  risk_parity    — equal risk contribution (Maillard et al., 2010)
  signal_weighted — weight proportional to |signal - 0.5| conviction
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    weights: Dict[str, float]       # ticker → portfolio weight (sums to 1)
    expected_return_ann: float      # % annualized portfolio expected return
    expected_vol_ann: float         # % annualized portfolio volatility
    sharpe_ratio: float
    method: str


class PortfolioOptimizer:
    """
    Optimizes portfolio weights given a list of tickers.

    Parameters
    ----------
    risk_free_rate : annualized risk-free rate (default 5%)
    max_weight     : maximum allocation per ticker (default 40%)
    min_weight     : minimum allocation per ticker (default 2%)
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        max_weight: float = 0.40,
        min_weight: float = 0.02,
    ):
        self.rf = risk_free_rate
        self.max_w = max_weight
        self.min_w = min_weight

    def optimize(
        self,
        tickers: List[str],
        signals: Optional[Dict[str, float]] = None,
        method: str = "max_sharpe",
        period: str = "1y",
    ) -> OptimizationResult:
        """
        Downloads returns for tickers and optimizes weights.

        Parameters
        ----------
        tickers : list of ticker symbols
        signals : optional dict of ticker → probability_up (for signal_weighted)
        method  : 'max_sharpe' | 'min_variance' | 'risk_parity' | 'signal_weighted'
        period  : yfinance period string for return download
        """
        import yfinance as yf

        # Download returns
        prices_dict: Dict[str, pd.Series] = {}
        for t in tickers:
            try:
                df = yf.download(t, period=period, interval="1d",
                                 auto_adjust=True, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if not df.empty:
                    prices_dict[t] = df["Close"]
            except Exception as e:
                logger.warning(f"[PortfolioOptimizer] Failed to download {t}: {e}")

        valid_tickers = list(prices_dict.keys())
        if len(valid_tickers) < 2:
            logger.warning("[PortfolioOptimizer] Fewer than 2 valid tickers — using equal weight.")
            return self._equal_weight(tickers or valid_tickers, method)

        returns_df = pd.DataFrame(prices_dict).pct_change().dropna()
        mu = returns_df.mean() * 252         # annualized mean returns (decimal)
        sigma = returns_df.cov() * 252       # annualized covariance matrix
        n = len(valid_tickers)

        if method == "signal_weighted" and signals:
            return self._signal_weighted(valid_tickers, signals, mu, sigma)

        if method == "risk_parity":
            return self._risk_parity(valid_tickers, sigma)

        # scipy SLSQP for max_sharpe / min_variance
        try:
            from scipy.optimize import minimize
        except ImportError:
            logger.warning("[PortfolioOptimizer] scipy not installed — using equal weight.")
            return self._equal_weight(valid_tickers, method)

        w0 = np.full(n, 1.0 / n)
        bounds = [(self.min_w, self.max_w)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        if method == "max_sharpe":
            def neg_sharpe(w: np.ndarray) -> float:
                port_ret = float(np.dot(w, mu))
                port_vol = float(np.sqrt(w @ sigma.values @ w))
                return -(port_ret - self.rf) / port_vol if port_vol > 1e-10 else 0.0

            res = minimize(neg_sharpe, w0, method="SLSQP",
                           bounds=bounds, constraints=constraints,
                           options={"maxiter": 1000, "ftol": 1e-9})
        else:  # min_variance
            def port_variance(w: np.ndarray) -> float:
                return float(w @ sigma.values @ w)

            res = minimize(port_variance, w0, method="SLSQP",
                           bounds=bounds, constraints=constraints,
                           options={"maxiter": 1000, "ftol": 1e-9})

        weights = np.clip(res.x, 0.0, None)
        weights /= weights.sum()

        return self._build_result(valid_tickers, weights, mu, sigma, method)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _signal_weighted(
        self,
        tickers: List[str],
        signals: Dict[str, float],
        mu: pd.Series,
        sigma: pd.DataFrame,
    ) -> OptimizationResult:
        convictions = np.array([abs(signals.get(t, 0.5) - 0.5) for t in tickers])
        total = convictions.sum()
        weights = convictions / total if total > 0 else np.full(len(tickers), 1.0 / len(tickers))
        weights = np.clip(weights, self.min_w, self.max_w)
        weights /= weights.sum()
        return self._build_result(tickers, weights, mu, sigma, "signal_weighted")

    def _risk_parity(
        self,
        tickers: List[str],
        sigma: pd.DataFrame,
    ) -> OptimizationResult:
        try:
            from scipy.optimize import minimize
        except ImportError:
            return self._equal_weight(tickers, "risk_parity")

        n = len(tickers)
        w0 = np.full(n, 1.0 / n)
        sigma_val = sigma.values

        def risk_budget_obj(w: np.ndarray) -> float:
            port_vol = float(np.sqrt(w @ sigma_val @ w))
            if port_vol < 1e-10:
                return 0.0
            marginal = sigma_val @ w / port_vol
            risk_contrib = w * marginal
            target = port_vol / n
            return float(np.sum((risk_contrib - target) ** 2))

        bounds = [(self.min_w, self.max_w)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        res = minimize(risk_budget_obj, w0, method="SLSQP",
                       bounds=bounds, constraints=constraints)
        weights = np.clip(res.x, 0.0, None)
        weights /= weights.sum()

        # mu not available here — compute approximate from sigma diagonal
        mu_approx = pd.Series(np.zeros(n), index=tickers)
        return self._build_result(tickers, weights, mu_approx, sigma, "risk_parity")

    def _build_result(
        self,
        tickers: List[str],
        weights: np.ndarray,
        mu: pd.Series,
        sigma: pd.DataFrame,
        method: str,
    ) -> OptimizationResult:
        mu_vals = mu.reindex(tickers).fillna(0.0).values
        sigma_vals = sigma.reindex(index=tickers, columns=tickers).fillna(0.0).values
        port_ret = float(np.dot(weights, mu_vals))
        port_vol = float(np.sqrt(np.clip(weights @ sigma_vals @ weights, 0.0, None)))
        sharpe = (port_ret - self.rf) / port_vol if port_vol > 1e-10 else 0.0
        return OptimizationResult(
            weights={t: round(float(w), 4) for t, w in zip(tickers, weights)},
            expected_return_ann=round(port_ret * 100, 2),
            expected_vol_ann=round(port_vol * 100, 2),
            sharpe_ratio=round(sharpe, 3),
            method=method,
        )

    def _equal_weight(self, tickers: List[str], method: str) -> OptimizationResult:
        n = len(tickers)
        w = 1.0 / n if n > 0 else 1.0
        return OptimizationResult(
            weights={t: round(w, 4) for t in tickers},
            expected_return_ann=0.0,
            expected_vol_ann=0.0,
            sharpe_ratio=0.0,
            method=f"equal_weight (fallback — {method})",
        )
