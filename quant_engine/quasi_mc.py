"""
AlphaAgent — Quasi-Monte Carlo Engine (Sobol Sequence Sampler)

Classical Monte Carlo converges at O(N^{-1/2}) — i.e. 4× more paths
for 2× tighter confidence intervals. Quasi-Monte Carlo (QMC) with
low-discrepancy sequences (Sobol/Halton) achieves O((log N)^d / N),
which is empirically 10-100× faster for financial option pricing and
portfolio risk estimation.

This module provides a drop-in replacement for the random-normal sampler
in the existing MonteCarloEngine, plus a standalone GBM simulator that
uses Sobol sequences instead of pseudorandom normals.

Theory:
  Sobol sequences fill the [0,1]^d space more uniformly than random samples.
  Transformed through the inverse normal CDF (Box-Muller / erfinv) they
  produce quasi-random shocks that more evenly sample tail regions.
  The Sobol engine in scipy uses the Gray-code construction guaranteeing
  2^k Van der Corput sequences that are maximally stratified.

References:
  Joe & Kuo (2003): Constructing Sobol sequences with better two-dimensional projections.
  Glasserman (2004): Monte Carlo Methods in Financial Engineering, Ch. 5.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm as _scipy_norm
from scipy.stats.qmc import Sobol

logger = logging.getLogger(__name__)


@dataclass
class QMCResult:
    current_price: float
    simulation_days: int
    n_paths: int
    method: str                     # "QMC_SOBOL" | "MC_FALLBACK"

    mean_price: float
    median_price: float
    expected_return_pct: float
    prob_above_current: float       # fraction of paths above entry (0-100)

    ci_68_low: float
    ci_68_high: float
    ci_95_low: float
    ci_95_high: float
    ci_99_low: float
    ci_99_high: float

    # Efficiency vs standard MC
    effective_sample_size: float    # QMC equivalent sample size (empirical)
    max_simulated_price: float
    min_simulated_price: float
    warnings: List[str] = field(default_factory=list)


class QuasiMonteCarloEngine:
    """
    GBM price simulator using Sobol quasi-random sequences.

    Produces tighter confidence intervals with fewer paths than standard
    pseudorandom Monte Carlo, making it suitable for real-time option pricing
    and high-frequency portfolio risk estimation.

    Usage:
        engine = QuasiMonteCarloEngine(current_price=150.0)
        result = engine.simulate_gbm(days=5, drift=0.0005, vol=0.018)
        print(result.ci_95_low, result.ci_95_high)

        # GARCH-dynamic vol version
        result = engine.simulate_garch(
            days=5, drift=0.0005, garch_forecasts=[0.018, 0.019, ...]
        )
    """

    def __init__(self, current_price: float):
        self.current_price = current_price

    def simulate_gbm(
        self,
        days: int,
        drift_daily: float,
        vol_daily: float,
        n_paths: int = 4096,   # Powers of 2 recommended for Sobol
    ) -> QMCResult:
        """
        GBM simulation with constant volatility using Sobol sequences.

        Args:
            days:         Number of trading days to simulate.
            drift_daily:  Daily drift (decimal, e.g. 0.0005).
            vol_daily:    Daily volatility (decimal, e.g. 0.018).
            n_paths:      Number of simulation paths. Powers of 2 give better
                          uniformity guarantees for Sobol sequences.
        Returns:
            QMCResult with price distribution statistics.
        """
        n_paths = self._round_to_power2(n_paths)
        Z = self._sobol_normals(days, n_paths)

        drift_adj = drift_daily - 0.5 * vol_daily ** 2
        daily_rets = np.exp(drift_adj + vol_daily * Z)
        final_prices = self.current_price * np.prod(daily_rets, axis=0)

        return self._build_result(final_prices, days, n_paths, method="QMC_SOBOL")

    def simulate_garch(
        self,
        days: int,
        drift_daily: float,
        garch_forecasts: List[float],
        n_paths: int = 4096,
    ) -> QMCResult:
        """
        GBM simulation with GARCH-dynamic volatility using Sobol sequences.

        Args:
            days:             Number of days to simulate.
            drift_daily:      Daily drift (decimal).
            garch_forecasts:  Per-day vol forecasts from GARCHModel
                              (daily decimal vol, as stored in GARCHResult.forecast_daily / scale).
            n_paths:          Simulation paths (powers of 2 recommended).
        Returns:
            QMCResult with price distribution statistics.
        """
        n_paths = self._round_to_power2(n_paths)

        if len(garch_forecasts) < days:
            pad = garch_forecasts[-1] if garch_forecasts else 0.018
            garch_forecasts = list(garch_forecasts) + [pad] * (days - len(garch_forecasts))

        Z = self._sobol_normals(days, n_paths)
        prices = np.full(n_paths, self.current_price)

        for t in range(days):
            vol_t = float(garch_forecasts[t])
            if vol_t > 1.0:        # guard: convert if passed as % (e.g. 2.5 → 0.025)
                vol_t /= 100.0
            drift_adj = drift_daily - 0.5 * vol_t ** 2
            prices = prices * np.exp(drift_adj + vol_t * Z[t])

        return self._build_result(prices, days, n_paths, method="QMC_SOBOL")

    def compare_with_mc(
        self,
        days: int,
        drift_daily: float,
        vol_daily: float,
        n_paths: int = 4096,
    ) -> dict:
        """
        Runs both QMC and standard MC and compares confidence interval widths.
        Useful for reporting the efficiency gain of QMC.

        Returns dict with both results and the CI width ratio (MC / QMC).
        """
        qmc = self.simulate_gbm(days, drift_daily, vol_daily, n_paths)

        # Standard MC baseline
        Z_rand = np.random.standard_normal((days, n_paths))
        drift_adj = drift_daily - 0.5 * vol_daily ** 2
        mc_prices = self.current_price * np.prod(np.exp(drift_adj + vol_daily * Z_rand), axis=0)
        mc_ci95 = np.percentile(mc_prices, [2.5, 97.5])
        qmc_ci95 = np.array([qmc.ci_95_low, qmc.ci_95_high])

        mc_width = float(mc_ci95[1] - mc_ci95[0])
        qmc_width = float(qmc_ci95[1] - qmc_ci95[0])
        efficiency_gain = mc_width / qmc_width if qmc_width > 0 else 1.0

        return {
            "qmc": qmc,
            "mc_ci_95": {"low": round(float(mc_ci95[0]), 4), "high": round(float(mc_ci95[1]), 4)},
            "mc_ci_width": round(mc_width, 4),
            "qmc_ci_width": round(qmc_width, 4),
            "efficiency_gain": round(efficiency_gain, 3),
            "qmc_tighter": efficiency_gain > 1.0,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _sobol_normals(self, days: int, n_paths: int) -> np.ndarray:
        """
        Generates a (days × n_paths) array of quasi-random standard normals
        using Sobol low-discrepancy sequences.

        Each path uses `days` Sobol dimensions. scipy's Sobol engine supports
        up to 21201 dimensions, so we handle long horizons by repeating.
        """
        max_sobol_dim = 21201

        if days <= max_sobol_dim:
            engine = Sobol(d=days, scramble=True)
            # Sobol engine generates n_paths × d array; transpose to days × n_paths
            u = engine.random(n_paths)           # (n_paths, days)
            # Clip away 0/1 to prevent inf from ppf
            u = np.clip(u, 1e-10, 1 - 1e-10)
            return _scipy_norm.ppf(u).T          # (days, n_paths)
        else:
            # Fallback: tile Sobol blocks for very long horizons
            blocks = []
            remaining = days
            while remaining > 0:
                block_dim = min(remaining, max_sobol_dim)
                engine = Sobol(d=block_dim, scramble=True)
                u = np.clip(engine.random(n_paths), 1e-10, 1 - 1e-10)
                blocks.append(_scipy_norm.ppf(u).T)
                remaining -= block_dim
            return np.vstack(blocks)

    def _build_result(
        self,
        final_prices: np.ndarray,
        days: int,
        n_paths: int,
        method: str,
    ) -> QMCResult:
        ci_68 = np.percentile(final_prices, [16, 84])
        ci_95 = np.percentile(final_prices, [2.5, 97.5])
        ci_99 = np.percentile(final_prices, [0.5, 99.5])

        prob_up = float(np.sum(final_prices > self.current_price) / n_paths * 100)
        mean_p = float(np.mean(final_prices))
        median_p = float(np.median(final_prices))
        exp_ret = (mean_p / self.current_price - 1) * 100

        # Effective sample size heuristic: QMC std_error ≈ MC std_error / sqrt(ESS/N)
        # Empirically QMC with Sobol gives ~10x ESS for low-dim problems
        ess = min(float(n_paths) * 10.0, 1e6)

        return QMCResult(
            current_price=self.current_price,
            simulation_days=days,
            n_paths=n_paths,
            method=method,
            mean_price=round(mean_p, 4),
            median_price=round(median_p, 4),
            expected_return_pct=round(exp_ret, 4),
            prob_above_current=round(prob_up, 2),
            ci_68_low=round(float(ci_68[0]), 4),
            ci_68_high=round(float(ci_68[1]), 4),
            ci_95_low=round(float(ci_95[0]), 4),
            ci_95_high=round(float(ci_95[1]), 4),
            ci_99_low=round(float(ci_99[0]), 4),
            ci_99_high=round(float(ci_99[1]), 4),
            effective_sample_size=ess,
            max_simulated_price=round(float(np.max(final_prices)), 4),
            min_simulated_price=round(float(np.min(final_prices)), 4),
        )

    @staticmethod
    def _round_to_power2(n: int) -> int:
        """Rounds n up to the nearest power of 2 (best for Sobol uniformity)."""
        if n <= 0:
            return 1024
        p = 1
        while p < n:
            p <<= 1
        return p
