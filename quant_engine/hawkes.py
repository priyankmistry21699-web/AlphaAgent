"""
AlphaAgent — Hawkes Process Engine

Models self-exciting point processes in financial return data to detect
trade clustering, dark pool footprints, and institutional cascade risk.

The exponential Hawkes process models the conditional intensity λ(t):
  λ(t) = μ + α * Σ_{t_i < t} exp(-β * (t - t_i))

where:
  μ (mu)    = background arrival rate (baseline event frequency)
  α (alpha) = excitation coefficient (each event's triggering strength)
  β (beta)  = decay rate (how fast excitement decays)
  α/β       = branching ratio (< 1 for stationarity; → 1 = cascade risk)

Financial interpretations:
  High branching ratio (α/β → 0.9) → one large trade triggers many more
    → dark pool footprint, institutional cascade
  High μ → high baseline activity → news-driven or high-volume regime
  High α relative to β → fat-tailed event clustering (HFT-like behavior)

The events are defined as returns that exceed a vol-scaled threshold,
representing significant market moves (spikes above 1.5σ).
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from scipy.optimize import minimize

from config.settings_manager import settings

logger = logging.getLogger(__name__)

def _MIN_EVENTS():     return settings.get("hawkes.min_events",        10)
def _BRANCHING_WARN(): return settings.get("hawkes.branching_warn",    0.80)
def _BRANCHING_CRIT(): return settings.get("hawkes.branching_critical", 0.95)


@dataclass
class HawkesResult:
    mu: float                    # Background intensity (events/day)
    alpha: float                 # Excitation magnitude
    beta: float                  # Decay rate
    branching_ratio: float       # α/β — must be < 1 for stationarity
    n_events: int                # Total exceedance events observed
    event_rate: float            # Events per day (empirical)
    log_likelihood: float        # MLE objective (higher = better fit)
    converged: bool              # Whether MLE optimizer converged
    cascade_risk: str            # "LOW" | "MODERATE" | "HIGH" | "CRITICAL"
    clustering_score: float      # [0, 1]: 0 = Poisson, 1 = fully self-exciting
    dark_pool_signal: float      # [0, 1]: likelihood of institutional footprint
    warnings: List[str] = field(default_factory=list)


class HawkesProcess:
    """
    Fits an exponential Hawkes process to return exceedance events.

    Usage:
        hp = HawkesProcess(threshold_sigma=1.5)
        result = hp.fit(returns_series)
        print(result.branching_ratio, result.cascade_risk)
    """

    def __init__(self, threshold_sigma: float = 1.5):
        """
        Args:
            threshold_sigma: Events are returns exceeding this many std devs.
        """
        self.threshold_sigma = threshold_sigma

    def fit(self, returns: pd.Series) -> HawkesResult:
        """
        Identifies exceedance events and fits the Hawkes process via MLE.

        Args:
            returns: Daily return series (decimals, e.g. 0.01 = 1%).
        """
        returns = returns.dropna()
        warnings = []

        if len(returns) < 30:
            return self._empty_result(warnings=["Insufficient data for Hawkes (need 30+ observations)."])

        # ── Event extraction: returns exceeding ±threshold_sigma * vol ────
        vol = float(returns.std())
        if vol < 1e-8:
            return self._empty_result(warnings=["Return series has zero variance."])

        threshold = self.threshold_sigma * vol
        abs_ret = np.abs(returns.values)
        event_idx = np.where(abs_ret > threshold)[0]
        n_events = len(event_idx)

        if n_events < _MIN_EVENTS():
            warnings.append(
                f"Only {n_events} exceedance events found (need {_MIN_EVENTS()}+). "
                "Hawkes fit may be unreliable."
            )
            if n_events < 3:
                return self._empty_result(
                    warnings=warnings + ["Too few events for MLE."]
                )

        # Convert event indices to continuous "time" in trading days
        event_times = event_idx.astype(float)
        T = float(len(returns))   # observation window length
        event_rate = n_events / T

        # ── MLE fitting ────────────────────────────────────────────────────
        mu_init = max(0.01, event_rate * 0.5)
        alpha_init = 0.5
        beta_init = 2.0

        mu, alpha, beta, loglik, converged = self._fit_mle(
            event_times, T, mu_init, alpha_init, beta_init
        )

        if not converged:
            warnings.append("Hawkes MLE optimizer did not converge — estimates may be noisy.")

        branching_ratio = alpha / beta if beta > 1e-8 else float("inf")

        # ── Cascade risk classification ────────────────────────────────────
        if branching_ratio >= _BRANCHING_CRIT():
            cascade_risk = "CRITICAL"
            warnings.append(
                f"CRITICAL: Branching ratio {branching_ratio:.3f} ≥ {_BRANCHING_CRIT()}. "
                "Event cascade is near-explosive — institutional domino risk very high."
            )
        elif branching_ratio >= _BRANCHING_WARN():
            cascade_risk = "HIGH"
            warnings.append(
                f"HIGH cascade risk: branching ratio {branching_ratio:.3f}. "
                f"Trade clustering is severe — possible dark pool or HFT activity."
            )
        elif branching_ratio >= 0.5:
            cascade_risk = "MODERATE"
        else:
            cascade_risk = "LOW"

        # ── Clustering score: how far above Poisson? ──────────────────────
        # Purely Poisson process has branching ratio → 0
        # Fully self-exciting approaches branching ratio → 1
        clustering_score = float(np.clip(branching_ratio, 0, 1))

        # ── Dark pool proxy ───────────────────────────────────────────────
        # High clustering + above-average event rate → institutional footprint
        rate_z = (event_rate - 0.05) / 0.03  # z-score against 5% baseline
        dark_pool_signal = float(np.clip(
            0.5 * clustering_score + 0.5 * float(np.clip(rate_z / 2.0, 0, 1)),
            0, 1
        ))

        return HawkesResult(
            mu=round(float(mu), 6),
            alpha=round(float(alpha), 6),
            beta=round(float(beta), 6),
            branching_ratio=round(float(branching_ratio), 4),
            n_events=n_events,
            event_rate=round(event_rate, 4),
            log_likelihood=round(float(loglik), 4),
            converged=converged,
            cascade_risk=cascade_risk,
            clustering_score=round(clustering_score, 3),
            dark_pool_signal=round(dark_pool_signal, 3),
            warnings=warnings,
        )

    # ── MLE Implementation ────────────────────────────────────────────────────

    @staticmethod
    def _fit_mle(
        times: np.ndarray,
        T: float,
        mu0: float,
        alpha0: float,
        beta0: float,
    ) -> Tuple[float, float, float, float, bool]:
        """
        Fits Hawkes parameters via log-likelihood maximization.

        Log-likelihood of exponential Hawkes (Ozaki 1979):
          LL = -μT + Σ_i log(λ(t_i)) + (α/β) * Σ_i (exp(-β*(T-t_i)) - 1)

        Returns (mu, alpha, beta, loglik, converged).
        """
        def neg_loglik(params: np.ndarray) -> float:
            mu, alpha, beta = params
            if mu <= 0 or alpha <= 0 or beta <= 0 or alpha >= beta:
                return 1e10

            n = len(times)
            # Compute λ(t_i) for each event using recursive formula
            # A_i = Σ_{j < i} exp(-β * (t_i - t_j))
            A = np.zeros(n)
            for i in range(1, n):
                A[i] = np.exp(-beta * (times[i] - times[i-1])) * (1 + A[i-1])

            lam = mu + alpha * A   # conditional intensity at each event

            if np.any(lam <= 0):
                return 1e10

            # Integral of λ over [0, T]:
            # ∫λ dt = μT + (α/β) * Σ_i (1 - exp(-β*(T - t_i)))
            integral = mu * T + (alpha / beta) * np.sum(1 - np.exp(-beta * (T - times)))

            ll = np.sum(np.log(lam)) - integral
            return -ll

        x0 = np.array([mu0, alpha0, beta0])
        bounds = [(1e-6, None), (1e-6, None), (1e-4, None)]
        constraints = [{"type": "ineq", "fun": lambda p: p[2] - p[1] - 1e-4}]  # β > α

        try:
            res = minimize(
                neg_loglik, x0, method="SLSQP",
                bounds=bounds, constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-9},
            )
            mu, alpha, beta = res.x
            return float(mu), float(alpha), float(beta), float(-res.fun), bool(res.success)
        except Exception as e:
            logger.warning(f"[Hawkes] MLE failed: {e}")
            return float(mu0), float(alpha0), float(beta0), 0.0, False

    @staticmethod
    def _empty_result(warnings: Optional[List[str]] = None) -> HawkesResult:
        return HawkesResult(
            mu=0.0, alpha=0.0, beta=1.0, branching_ratio=0.0,
            n_events=0, event_rate=0.0, log_likelihood=0.0,
            converged=False, cascade_risk="LOW",
            clustering_score=0.0, dark_pool_signal=0.0,
            warnings=warnings or [],
        )
