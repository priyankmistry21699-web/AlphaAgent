"""
AlphaAgent — Structural Break Detection (CUSUM)

CUSUM test for online parameter drift detection. Used to flag when
agent IC has shifted significantly from baseline, indicating that
model recalibration is needed.

Reference: Page (1954), Brown-Durbin-Evans (1975).

Usage:
    from quant_engine.structural_break import CUSUMDetector
    result = CUSUMDetector().detect(time_series)
    # result.break_detected, result.break_index, result.cumsum_max
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CUSUMResult:
    break_detected: bool
    break_index: int           # -1 if no break
    cumsum_max: float
    cumsum_threshold: float
    n_obs: int
    direction: str             # "up" / "down" / "none"


class CUSUMDetector:
    """
    CUSUM (CUmulative SUM) chart for detecting mean shifts in a time series.

    threshold_k : sensitivity (typically 0.5 σ — half the smallest shift to detect)
    threshold_h : decision threshold (typically 4–5 σ — alarm level)
    """

    def __init__(self, threshold_k: float = 0.5, threshold_h: float = 5.0):
        self.k = threshold_k
        self.h = threshold_h

    def detect(self, series: pd.Series) -> Optional[CUSUMResult]:
        """
        Detect mean-shift break in a time series.
        Returns CUSUMResult with break_detected flag and approximate break_index.
        """
        try:
            x = pd.Series(series).dropna().values.astype(float)
            n = len(x)
            if n < 20:
                return None

            mu = float(np.mean(x))
            sigma = float(np.std(x))
            if sigma == 0:
                return CUSUMResult(False, -1, 0.0, self.h * sigma, n, "none")

            # Standardised CUSUM
            z = (x - mu) / sigma
            S_pos = np.zeros(n)
            S_neg = np.zeros(n)
            for t in range(1, n):
                S_pos[t] = max(0.0, S_pos[t - 1] + z[t] - self.k)
                S_neg[t] = min(0.0, S_neg[t - 1] + z[t] + self.k)

            max_pos = float(np.max(S_pos))
            min_neg = float(np.min(S_neg))
            cumsum_max = max(max_pos, abs(min_neg))

            break_detected = cumsum_max > self.h
            if max_pos > self.h and abs(min_neg) <= self.h:
                idx = int(np.argmax(S_pos))
                direction = "up"
            elif abs(min_neg) > self.h and max_pos <= self.h:
                idx = int(np.argmin(S_neg))
                direction = "down"
            elif break_detected:
                idx = int(np.argmax(np.abs([max_pos, min_neg]) == cumsum_max))
                idx = int(np.argmax(S_pos)) if max_pos > abs(min_neg) else int(np.argmin(S_neg))
                direction = "up" if max_pos > abs(min_neg) else "down"
            else:
                idx = -1
                direction = "none"

            return CUSUMResult(
                break_detected=bool(break_detected),
                break_index=idx,
                cumsum_max=round(cumsum_max, 3),
                cumsum_threshold=round(self.h, 3),
                n_obs=n,
                direction=direction,
            )
        except Exception as e:
            logger.warning(f"CUSUM failed: {e}")
            return None


class RollingCUSUM:
    """
    Rolling window CUSUM — useful for monitoring agent IC over time.
    Returns a series of break flags for each window.
    """

    def __init__(self, window: int = 60, threshold_k: float = 0.5, threshold_h: float = 5.0):
        self.window = window
        self.detector = CUSUMDetector(threshold_k, threshold_h)

    def monitor(self, series: pd.Series) -> pd.Series:
        """Returns a series of break-detection flags aligned to input index."""
        series = pd.Series(series).dropna()
        flags = pd.Series(False, index=series.index)
        if len(series) < self.window:
            return flags
        for i in range(self.window, len(series)):
            window_data = series.iloc[i - self.window:i]
            r = self.detector.detect(window_data)
            if r and r.break_detected:
                flags.iloc[i] = True
        return flags
