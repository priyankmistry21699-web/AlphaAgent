"""
AlphaAgent — Extreme Value Theory (EVT) Engine (Hardened)

Models the extreme tails of the return distribution (Black Swan events)
using two complementary EVT approaches:

  1. POT (Peaks-Over-Threshold) — Generalized Pareto Distribution (GPD)
     Fitted to daily losses exceeding a threshold u.
     Formula: VaR_p = u + (σ/ξ) * [((1−p)/F̄(u))^(−ξ) − 1]
     Automatic threshold selection via Mean Excess Function stability.

  2. Block Maxima (GEV) — Generalized Extreme Value Distribution
     Fitted to the worst loss in each non-overlapping monthly block.
     Provides independent tail estimate useful as cross-check.

  3. Tail Dependence (lower tail index with SPY)
     Measures co-crash probability: P(X < VaR | SPY < VaR).
     High tail dependence → stock crashes with the market.

Hardening additions vs the original EVT engine:
  - Automatic threshold u* selection (mean excess plot stability criterion)
  - GEV block maxima as second VaR/CVaR estimate
  - Tail dependence coefficient λ_L with the benchmark
  - All four VaR/CVaR estimates returned (GPD + GEV, 95% + 99%)
  - Robust fallback chain: GPD → GEV → Historical

References:
  Embrechts, Klüppelberg, Mikosch (1997): Modelling Extremal Events.
  de Haan & Ferreira (2006): Extreme Value Theory: An Introduction.
"""

import numpy as np
import pandas as pd
from scipy.stats import genpareto, genextreme
from scipy.optimize import minimize_scalar
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from config.settings_manager import settings

logger = logging.getLogger(__name__)


@dataclass
class EVTResult:
    # POT / GPD estimates
    var_95: float                    # GPD 95% VaR (negative = loss)
    var_99: float                    # GPD 99% VaR
    cvar_95: float                   # GPD 95% CVaR (Expected Shortfall)
    cvar_99: float                   # GPD 99% CVaR

    # GEV / Block Maxima estimates (cross-check)
    gev_var_95: float = 0.0          # GEV 95% VaR
    gev_var_99: float = 0.0          # GEV 99% VaR

    # Distributional shape
    tail_index: float = 0.0          # GPD shape ξ (xi): ξ > 0 → fat tails
    threshold: float = 0.0           # POT threshold u* (auto-selected)

    # Cross-asset tail dependence
    tail_dependence: float = 0.0     # Lower tail dependence λ_L with SPY (0=indep, 1=perfect cocrash)

    # Model diagnostics
    n_tail_events: int = 0           # Exceedances used for GPD fit
    gpd_converged: bool = False      # Whether GPD MLE converged
    gev_converged: bool = False      # Whether GEV block maxima converged

    warnings: List[str] = field(default_factory=list)


class ExtremeValueModel:
    """
    Hardened EVT engine with automatic threshold selection,
    GEV block-maxima cross-check, and tail dependence estimation.

    Usage:
        evt = ExtremeValueModel(returns)
        result = evt.fit_and_calculate()
        print(f"GPD VaR99: {result.var_99:.2%}")
        print(f"GEV VaR99: {result.gev_var_99:.2%}")
        print(f"Tail dep:  {result.tail_dependence:.3f}")
    """

    @property
    def MIN_OBSERVATIONS(self):
        return settings.get("evt.min_observations", 100)

    @property
    def MIN_TAIL_EVENTS(self):
        return settings.get("evt.min_tail_events", 10)

    @property
    def BLOCK_SIZE(self):
        return settings.get("evt.block_size", 21)

    def __init__(
        self,
        returns: pd.Series,
        threshold_percentile: float = 0.10,
        benchmark_returns: Optional[pd.Series] = None,
    ):
        """
        Args:
            returns:               Daily return series.
            threshold_percentile:  Fallback threshold if auto-selection fails
                                   (worst X% of returns define the tail).
            benchmark_returns:     SPY returns for tail dependence estimation.
        """
        if threshold_percentile == 0.10:
            threshold_percentile = settings.get("evt.threshold_percentile", 0.10)
        self.returns = returns.dropna()
        self.threshold_percentile = threshold_percentile
        self.benchmark_returns = (
            benchmark_returns.dropna() if benchmark_returns is not None else None
        )

    def fit_and_calculate(self) -> EVTResult:
        warnings: List[str] = []

        if len(self.returns) < self.MIN_OBSERVATIONS:
            warnings.append(
                f"Only {len(self.returns)} observations (need {self.MIN_OBSERVATIONS}+). "
                "EVT estimates may be unstable."
            )

        # Losses are positive values (we flip sign of negative returns)
        losses = -self.returns[self.returns < 0]

        if len(losses) < self.MIN_TAIL_EVENTS:
            logger.error("Not enough negative returns for EVT. Using historical VaR fallback.")
            return self._fallback_var(warnings)

        # ── 1. Auto-select POT threshold ──────────────────────────────────────
        threshold = self._select_threshold(losses)
        excesses = losses[losses > threshold] - threshold
        n_tail = len(excesses)

        if n_tail < self.MIN_TAIL_EVENTS:
            threshold = float(losses.quantile(1.0 - self.threshold_percentile))
            excesses = losses[losses > threshold] - threshold
            n_tail = len(excesses)
            warnings.append(
                f"Auto-threshold gave too few events — fell back to "
                f"{self.threshold_percentile:.0%} percentile threshold."
            )

        # ── 2. GPD / POT fitting ──────────────────────────────────────────────
        var_95, var_99, cvar_95, cvar_99, gpd_shape, gpd_converged = (
            self._fit_gpd(excesses, losses, threshold)
        )

        if gpd_shape > 0.5:
            warnings.append(
                f"Extremely fat tails (ξ={gpd_shape:.3f} > 0.5). "
                "High Black Swan risk — tail events may be much worse than modelled."
            )

        if not gpd_converged:
            warnings.append("GPD fit did not converge cleanly. Estimates may be imprecise.")

        # ── 3. GEV Block Maxima ───────────────────────────────────────────────
        gev_var_95, gev_var_99, gev_converged = self._fit_gev_block_maxima(losses)

        if not gev_converged:
            warnings.append("GEV block maxima fit did not converge — GEV estimates unreliable.")

        # ── 4. Tail dependence with benchmark ─────────────────────────────────
        tail_dep = 0.0
        if self.benchmark_returns is not None:
            tail_dep = self._estimate_tail_dependence(
                self.returns, self.benchmark_returns, q=0.05
            )
            if tail_dep > 0.5:
                warnings.append(
                    f"High lower tail dependence with benchmark (λ_L={tail_dep:.3f}). "
                    "Stock likely to crash with the market during stress events."
                )

        return EVTResult(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            gev_var_95=gev_var_95,
            gev_var_99=gev_var_99,
            tail_index=round(float(gpd_shape), 4),
            threshold=round(float(threshold), 6),
            tail_dependence=round(float(tail_dep), 4),
            n_tail_events=n_tail,
            gpd_converged=gpd_converged,
            gev_converged=gev_converged,
            warnings=warnings,
        )

    # ── Method 1: POT / GPD ───────────────────────────────────────────────────

    def _fit_gpd(
        self,
        excesses: pd.Series,
        losses: pd.Series,
        threshold: float,
    ) -> Tuple[float, float, float, float, float, bool]:
        """
        Fits GPD to excesses over threshold. Returns
        (var_95, var_99, cvar_95, cvar_99, shape_xi, converged).
        All VaR/CVaR returned as negative decimals (loss convention).
        """
        try:
            shape, _, scale = genpareto.fit(excesses.values, floc=0)
            converged = True
        except Exception as e:
            logger.error(f"GPD fit failed: {e}")
            hist = self._fallback_var([])
            return hist.var_95, hist.var_99, hist.cvar_95, hist.cvar_99, 0.0, False

        p_exceed = len(excesses) / len(losses) * (len(losses) / max(len(self.returns), 1))

        def gpd_var(conf: float) -> float:
            p_tail = 1.0 - conf
            if abs(shape) < 1e-8:
                v = threshold + scale * np.log(p_tail / max(p_exceed, 1e-8))
            else:
                v = threshold + (scale / shape) * ((p_tail / max(p_exceed, 1e-8)) ** (-shape) - 1)
            return -float(v)

        def gpd_cvar(var_val: float) -> float:
            if shape >= 1:
                return -1.0
            numerator = (-var_val) + scale - shape * threshold
            denominator = 1 - shape
            return -float(numerator / denominator)

        var_95 = gpd_var(0.95)
        var_99 = gpd_var(0.99)
        cvar_95 = gpd_cvar(var_95)
        cvar_99 = gpd_cvar(var_99)

        return var_95, var_99, cvar_95, cvar_99, float(shape), converged

    # ── Method 2: GEV Block Maxima ────────────────────────────────────────────

    def _fit_gev_block_maxima(
        self,
        losses: pd.Series,
    ) -> Tuple[float, float, bool]:
        """
        Fits Generalized Extreme Value (GEV) distribution to block maxima.
        Returns (var_95, var_99, converged) as negative decimals.
        """
        arr = losses.values
        n_blocks = len(arr) // self.BLOCK_SIZE
        if n_blocks < 5:
            return 0.0, 0.0, False

        # Extract the maximum loss in each non-overlapping block
        block_maxima = np.array([
            arr[i * self.BLOCK_SIZE:(i + 1) * self.BLOCK_SIZE].max()
            for i in range(n_blocks)
        ])

        try:
            shape, loc, scale = genextreme.fit(block_maxima)

            def gev_var(conf: float) -> float:
                # GEV quantile: loc + scale/shape * ((-log(conf))^(-shape) - 1)
                if abs(shape) < 1e-8:
                    q = loc - scale * np.log(-np.log(conf))
                else:
                    q = loc + scale / shape * ((-np.log(conf)) ** (-shape) - 1)
                return -float(q)

            return gev_var(0.95), gev_var(0.99), True
        except Exception as e:
            logger.warning(f"GEV block maxima fit failed: {e}")
            return 0.0, 0.0, False

    # ── Automatic Threshold Selection ─────────────────────────────────────────

    def _select_threshold(self, losses: pd.Series) -> float:
        """
        Selects the optimal POT threshold u* via Mean Excess Function (MEF)
        stability: finds the lowest threshold where mean excess is approximately
        linear (GPD assumption holds).

        Falls back to the 90th percentile if stability criterion fails.
        """
        arr = np.sort(losses.values)[::-1]  # descending
        n = len(arr)

        # Candidate thresholds: 5th to 25th percentile of losses (worst 5-25%)
        low_idx = max(1, int(0.05 * n))
        high_idx = max(low_idx + 1, int(0.25 * n))
        candidates = arr[low_idx:high_idx]

        if len(candidates) < 3:
            return float(losses.quantile(0.90))

        # Compute mean excess e(u) = E[X - u | X > u] for each candidate
        def mean_excess(u: float) -> float:
            ex = arr[arr > u] - u
            return float(ex.mean()) if len(ex) > 0 else 0.0

        me_vals = np.array([mean_excess(u) for u in candidates])

        # Stability criterion: find region where MEF is most linear (min curvature)
        if len(me_vals) < 4:
            return float(candidates.mean())

        # Second-order differences as curvature proxy
        curvature = np.abs(np.diff(me_vals, n=2))
        # Lowest curvature = most linear region → best GPD fit
        best_idx = int(np.argmin(curvature)) + 1  # offset for second-diff
        best_idx = min(best_idx, len(candidates) - 1)

        return float(candidates[best_idx])

    # ── Tail Dependence Estimation ────────────────────────────────────────────

    @staticmethod
    def _estimate_tail_dependence(
        returns: pd.Series,
        benchmark: pd.Series,
        q: float = 0.05,
    ) -> float:
        """
        Estimates the lower tail dependence coefficient λ_L:
          λ_L = P(X < VaR_q | B < VaR_q)

        Uses the empirical (non-parametric) estimator:
          λ̂_L = #{X_t < Q_q(X) and B_t < Q_q(B)} / #{B_t < Q_q(B)}

        Args:
            returns:    Return series of the asset.
            benchmark:  Return series of the benchmark (e.g. SPY).
            q:          Quantile threshold (0.05 = worst 5%).
        Returns:
            Empirical lower tail dependence in [0, 1].
        """
        try:
            aligned = pd.concat([returns, benchmark], axis=1).dropna()
            if len(aligned) < 50:
                return 0.0
            x = aligned.iloc[:, 0].values
            b = aligned.iloc[:, 1].values

            q_x = np.quantile(x, q)
            q_b = np.quantile(b, q)

            b_tail = b < q_b
            n_b_tail = b_tail.sum()
            if n_b_tail == 0:
                return 0.0

            both_tail = ((x < q_x) & b_tail).sum()
            return float(both_tail / n_b_tail)
        except Exception:
            return 0.0

    # ── Historical Fallback ───────────────────────────────────────────────────

    def _fallback_var(self, warnings: List[str]) -> EVTResult:
        """Returns historical simulation VaR when GPD fitting fails."""
        var_95 = float(self.returns.quantile(0.05)) if len(self.returns) > 0 else -0.05
        var_99 = float(self.returns.quantile(0.01)) if len(self.returns) > 0 else -0.10
        cvar_95 = float(self.returns[self.returns <= var_95].mean()) if len(self.returns) > 0 else -0.07
        cvar_99 = float(self.returns[self.returns <= var_99].mean()) if len(self.returns) > 0 else -0.12
        return EVTResult(
            var_95=var_95, var_99=var_99,
            cvar_95=cvar_95, cvar_99=cvar_99,
            gev_var_95=0.0, gev_var_99=0.0,
            tail_index=0.0, threshold=0.0, tail_dependence=0.0,
            n_tail_events=0, gpd_converged=False, gev_converged=False,
            warnings=warnings + ["EVT GPD fit failed. Using historical/empirical VaR."],
        )
