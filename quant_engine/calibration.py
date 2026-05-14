"""
AlphaAgent — Probability Calibrator

Calibrates raw agent probabilities via Platt scaling (logistic regression
on log-odds). Computes Brier score and a reliability diagram to measure
how well probabilities match observed frequencies.

Usage:
    cal = ProbabilityCalibrator()
    cal.fit(probs=[0.65, 0.48, 0.72], outcomes=[1, 0, 1])
    calibrated = cal.transform([0.60, 0.55, 0.80])
    report = cal.evaluate()
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

_CONVERGENCE_TOL = 1e-6
_MAX_ITER = 500
_LR = 0.05


@dataclass
class CalibrationResult:
    brier_score: float
    log_loss: float
    reliability_bins: List[dict]
    slope: float
    intercept: float
    converged: bool = True
    n_iter: int = 0


class ProbabilityCalibrator:
    """
    Fits Platt scaling on (probability, binary_outcome) pairs.
    Uses Newton's-method logistic regression with early stopping
    when the gradient magnitude drops below _CONVERGENCE_TOL.
    """

    def __init__(self):
        self._slope: float = 1.0
        self._intercept: float = 0.0
        self._fitted: bool = False
        self._train_probs: List[float] = []
        self._train_outcomes: List[int] = []

    def fit(self, probs: List[float], outcomes: List[int]) -> "ProbabilityCalibrator":
        if len(probs) < 10:
            logger.warning("Too few samples (%d) for calibration — using identity.", len(probs))
            self._train_probs = list(probs)
            self._train_outcomes = list(outcomes)
            return self

        p = np.clip(probs, 1e-7, 1 - 1e-7)
        log_odds = np.log(p / (1 - p))
        y = np.array(outcomes, dtype=float)

        slope, intercept = 1.0, 0.0
        converged = False
        n_iter = 0

        for n_iter in range(1, _MAX_ITER + 1):
            z = slope * log_odds + intercept
            sig = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            err = sig - y

            grad_s = float(np.mean(err * log_odds))
            grad_i = float(np.mean(err))

            # Early stopping: both gradient components are tiny
            if abs(grad_s) < _CONVERGENCE_TOL and abs(grad_i) < _CONVERGENCE_TOL:
                converged = True
                break

            h_s = float(np.mean(sig * (1 - sig) * log_odds ** 2)) + 1e-8
            h_i = float(np.mean(sig * (1 - sig))) + 1e-8
            slope -= _LR * grad_s / h_s
            intercept -= _LR * grad_i / h_i

        if not converged:
            logger.debug("Platt scaling did not fully converge in %d iterations.", _MAX_ITER)

        self._slope = slope
        self._intercept = intercept
        self._fitted = True
        self._converged = converged
        self._n_iter = n_iter
        self._train_probs = list(probs)
        self._train_outcomes = list(outcomes)
        return self

    def transform(self, probs: List[float]) -> List[float]:
        """Applies Platt calibration. Returns identity if not fitted."""
        if not self._fitted:
            return list(probs)
        p = np.clip(probs, 1e-7, 1 - 1e-7)
        log_odds = np.log(p / (1 - p))
        z = self._slope * log_odds + self._intercept
        calibrated = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return [float(c) for c in calibrated]

    def evaluate(
        self,
        probs: Optional[List[float]] = None,
        outcomes: Optional[List[int]] = None,
    ) -> CalibrationResult:
        p = np.clip(
            probs if probs is not None else self._train_probs,
            1e-7, 1 - 1e-7,
        )
        y = np.array(
            outcomes if outcomes is not None else self._train_outcomes,
            dtype=float,
        )

        if len(p) == 0:
            return CalibrationResult(
                brier_score=0.25, log_loss=0.693, reliability_bins=[],
                slope=self._slope, intercept=self._intercept,
            )

        p = np.asarray(p)
        brier = float(np.mean((p - y) ** 2))
        log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

        bins = []
        edges = np.arange(0.0, 1.01, 0.10)
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (p >= lo) & (p < hi)
            if mask.sum() > 0:
                bins.append({
                    "bin": f"{lo:.1f}–{hi:.1f}",
                    "mean_predicted": round(float(p[mask].mean()), 3),
                    "actual_freq": round(float(y[mask].mean()), 3),
                    "n": int(mask.sum()),
                })

        return CalibrationResult(
            brier_score=round(brier, 4),
            log_loss=round(log_loss, 4),
            reliability_bins=bins,
            slope=round(self._slope, 4),
            intercept=round(self._intercept, 4),
            converged=getattr(self, "_converged", True),
            n_iter=getattr(self, "_n_iter", 0),
        )
