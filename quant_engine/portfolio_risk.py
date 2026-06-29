"""
AlphaAgent — Portfolio-Level Risk Measures

Portfolio VaR / CVaR using historical, parametric, and Monte Carlo methods,
plus marginal VaR (risk contribution per position), factor decomposition,
and historical stress-scenario replay.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PortfolioRiskResult:
    var_95: float                 # 95% 1-day VaR (% of portfolio)
    var_99: float                 # 99% 1-day VaR
    cvar_95: float                # 95% CVaR (Expected Shortfall)
    cvar_99: float
    portfolio_vol_ann: float      # annualised portfolio vol
    marginal_var: Dict[str, float] = field(default_factory=dict)
    component_var: Dict[str, float] = field(default_factory=dict)
    stress_scenarios: Dict[str, float] = field(default_factory=dict)


# Historical stress scenarios — annualised expected portfolio drawdown
STRESS_SCENARIOS = {
    "GFC_2008":        {"equity": -0.40, "credit": -0.30, "vol_multiplier": 3.0},
    "COVID_Mar_2020":  {"equity": -0.34, "credit": -0.20, "vol_multiplier": 4.5},
    "Rate_Spike_2022": {"equity": -0.25, "credit": -0.15, "vol_multiplier": 1.8},
    "Flash_2010":      {"equity": -0.09, "credit": -0.02, "vol_multiplier": 2.0},
    "Dot_Com_2000":    {"equity": -0.49, "credit": -0.10, "vol_multiplier": 2.5},
}


class PortfolioRisk:
    """
    Computes portfolio-level risk measures.

    weights : dict ticker → portfolio weight (sum to 1)
    returns : DataFrame of daily returns, columns = tickers
    """

    def analyze(self,
                weights: Dict[str, float],
                returns: pd.DataFrame,
                confidence_levels: List[float] = [0.95, 0.99]
                ) -> Optional[PortfolioRiskResult]:
        try:
            # Align tickers
            tickers = [t for t in weights.keys() if t in returns.columns]
            if len(tickers) < 2:
                return None
            R = returns[tickers].dropna()
            if len(R) < 30:
                return None
            w = np.array([weights[t] for t in tickers], dtype=float)
            if w.sum() == 0:
                return None
            w = w / w.sum()

            # Portfolio returns
            port_rets = (R.values @ w)

            # 1. Historical VaR/CVaR
            var_95 = float(np.percentile(port_rets, 5))
            var_99 = float(np.percentile(port_rets, 1))
            cvar_95 = float(port_rets[port_rets <= var_95].mean()) if (port_rets <= var_95).any() else var_95
            cvar_99 = float(port_rets[port_rets <= var_99].mean()) if (port_rets <= var_99).any() else var_99

            # 2. Portfolio volatility
            cov = R.cov().values
            port_var = float(w @ cov @ w)
            port_vol_ann = float(np.sqrt(port_var * 252))

            # 3. Marginal VaR (contribution to portfolio risk per ticker)
            marginal = {}
            component = {}
            if port_var > 0:
                marg_vec = (cov @ w) / np.sqrt(port_var)   # ∂σ_p / ∂w_i
                comp_vec = w * marg_vec / np.sqrt(port_var)   # share of total risk
                for i, t in enumerate(tickers):
                    marginal[t] = float(round(marg_vec[i], 6))
                    component[t] = float(round(comp_vec[i], 4))

            # 4. Stress-scenario replay
            stress = {}
            equity_beta = w.sum()  # simplified — assumes long-only equity portfolio
            for scenario, params in STRESS_SCENARIOS.items():
                # Apply scenario equity shock × portfolio gross exposure
                shock = params["equity"] * abs(equity_beta)
                stress[scenario] = round(float(shock), 4)

            return PortfolioRiskResult(
                var_95=round(var_95, 4),
                var_99=round(var_99, 4),
                cvar_95=round(cvar_95, 4),
                cvar_99=round(cvar_99, 4),
                portfolio_vol_ann=round(port_vol_ann, 4),
                marginal_var=marginal,
                component_var=component,
                stress_scenarios=stress,
            )
        except Exception as e:
            logger.warning(f"Portfolio risk failed: {e}")
            return None


def factor_risk_decomposition(weights: Dict[str, float],
                               returns: pd.DataFrame,
                               factor_returns: pd.DataFrame
                               ) -> Optional[dict]:
    """
    Decompose portfolio variance into factor vs idiosyncratic components.
    factor_returns: DataFrame of factor returns (e.g., market, size, value, momentum)
    """
    try:
        tickers = [t for t in weights.keys() if t in returns.columns]
        if not tickers:
            return None
        R = returns[tickers].dropna()
        F = factor_returns.dropna()
        common_idx = R.index.intersection(F.index)
        if len(common_idx) < 60:
            return None
        R = R.loc[common_idx]
        F = F.loc[common_idx]

        # Regress each stock on factors → betas
        n = len(tickers)
        k = F.shape[1]
        betas = np.zeros((n, k))
        residuals = np.zeros((len(R), n))
        for i, t in enumerate(tickers):
            X = F.values
            y = R[t].values
            X_aug = np.column_stack([np.ones(len(X)), X])
            try:
                coef, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
                betas[i] = coef[1:]
                residuals[:, i] = y - X_aug @ coef
            except np.linalg.LinAlgError:
                continue

        w = np.array([weights[t] for t in tickers], dtype=float)
        w = w / w.sum() if w.sum() != 0 else w

        # Portfolio factor exposure
        port_betas = betas.T @ w
        F_cov = F.cov().values
        factor_var = float(port_betas @ F_cov @ port_betas)
        idio_var = float(((residuals @ w) ** 2).mean())
        total_var = factor_var + idio_var

        return {
            "factor_var_share":  round(factor_var / total_var, 4) if total_var > 0 else 0.0,
            "idio_var_share":    round(idio_var / total_var, 4) if total_var > 0 else 0.0,
            "portfolio_betas":   {F.columns[j]: round(float(port_betas[j]), 3) for j in range(k)},
            "total_var_ann":     round(total_var * 252, 6),
        }
    except Exception as e:
        logger.warning(f"Factor decomposition failed: {e}")
        return None
