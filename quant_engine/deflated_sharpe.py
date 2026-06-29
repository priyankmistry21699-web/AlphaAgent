"""
AlphaAgent — Deflated Sharpe Ratio + FDR Correction

Bailey & López de Prado (2014): the Deflated Sharpe Ratio adjusts the
observed Sharpe for the number of independent trials run during backtest
calibration — answering "is this Sharpe statistically distinguishable
from zero given the search effort?"

Also includes Benjamini-Hochberg False Discovery Rate correction for
multiple testing of factor significance.
"""

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DSRResult:
    observed_sharpe: float
    deflated_sharpe: float
    expected_max_sharpe: float
    probability_skill: float       # P(SR > 0 | observed SR)
    n_trials: int
    n_obs: int
    is_significant: bool           # at 95% confidence


def deflated_sharpe_ratio(observed_sharpe: float,
                          n_obs: int,
                          n_trials: int,
                          skewness: float = 0.0,
                          kurtosis: float = 3.0) -> Optional[DSRResult]:
    """
    Compute Deflated Sharpe Ratio.

    observed_sharpe : annualised Sharpe from backtest
    n_obs           : number of return observations
    n_trials        : number of strategy variants tested (e.g. v1→v9 = 9)
    skewness        : sample skewness of returns (default 0 = normal)
    kurtosis        : sample kurtosis (default 3 = normal)
    """
    try:
        if n_trials < 1 or n_obs < 30:
            return None

        # Expected max Sharpe under null (Bailey-Prado 2014)
        gamma = 0.5772156649  # Euler-Mascheroni
        expected_max = (
            (1 - gamma) * stats.norm.ppf(1 - 1.0 / n_trials) +
            gamma * stats.norm.ppf(1 - 1.0 / (n_trials * math.e))
        )

        # Standard error of Sharpe with non-normal adjustment
        se_sr = np.sqrt(
            (1 - skewness * observed_sharpe + ((kurtosis - 1) / 4.0) * observed_sharpe ** 2)
            / max(n_obs - 1, 1)
        )

        # Deflated Sharpe = (SR - E[max SR]) / SE
        deflated = (observed_sharpe - expected_max) / se_sr if se_sr > 0 else 0.0

        # Probability skill exists (Z-statistic → probability)
        prob_skill = float(stats.norm.cdf(deflated))
        is_significant = bool(prob_skill > 0.95)

        return DSRResult(
            observed_sharpe=round(observed_sharpe, 4),
            deflated_sharpe=round(float(deflated), 4),
            expected_max_sharpe=round(float(expected_max), 4),
            probability_skill=round(prob_skill, 4),
            n_trials=n_trials,
            n_obs=n_obs,
            is_significant=is_significant,
        )
    except Exception as e:
        logger.warning(f"Deflated Sharpe failed: {e}")
        return None


def benjamini_hochberg_fdr(p_values: List[float],
                            alpha: float = 0.10) -> List[bool]:
    """
    Benjamini-Hochberg FDR correction for multiple testing.
    Controls expected proportion of false discoveries at level α.

    Returns list of bools — True = reject null (significant after FDR).
    """
    try:
        p_arr = np.asarray(p_values, dtype=float)
        n = len(p_arr)
        if n == 0:
            return []
        order = np.argsort(p_arr)
        sorted_p = p_arr[order]
        thresholds = (np.arange(1, n + 1) / n) * alpha
        # Find largest k where sorted_p[k] <= threshold[k]
        below = sorted_p <= thresholds
        if not below.any():
            return [False] * n
        k_star = np.max(np.where(below)[0])
        cutoff = sorted_p[k_star]
        return [bool(p <= cutoff) for p in p_arr]
    except Exception as e:
        logger.warning(f"BH-FDR failed: {e}")
        return [False] * len(p_values)


def bonferroni_correction(p_values: List[float],
                           alpha: float = 0.05) -> List[bool]:
    """
    Bonferroni correction — most conservative multiple-testing adjustment.
    Reject null only if p < alpha / n.
    """
    n = len(p_values)
    if n == 0:
        return []
    threshold = alpha / n
    return [bool(p < threshold) for p in p_values]


def probabilistic_sharpe_ratio(observed_sharpe: float,
                                n_obs: int,
                                benchmark_sharpe: float = 0.0,
                                skewness: float = 0.0,
                                kurtosis: float = 3.0) -> Optional[float]:
    """
    PSR: probability that the true Sharpe exceeds a benchmark.
    Bailey & López de Prado (2012).
    """
    try:
        if n_obs < 30:
            return None
        se = np.sqrt(
            (1 - skewness * observed_sharpe + ((kurtosis - 1) / 4.0) * observed_sharpe ** 2)
            / max(n_obs - 1, 1)
        )
        z = (observed_sharpe - benchmark_sharpe) / se if se > 0 else 0.0
        return float(stats.norm.cdf(z))
    except Exception:
        return None
