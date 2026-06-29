"""
AlphaAgent — Random Matrix Theory (RMT) Covariance Cleaning

Marchenko-Pastur eigenvalue clipping for correlation/covariance matrices.
Eigenvalues below the MP upper bound are statistically indistinguishable
from noise; clipping them recovers a more stable estimate of the true
underlying structure.

Reference: Marchenko & Pastur (1967), Bouchaud & Potters (2009).

Usage:
    from quant_engine.rmt import clean_correlation_matrix
    clean_corr = clean_correlation_matrix(returns_df)
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def marchenko_pastur_bounds(T: int, N: int) -> Tuple[float, float]:
    """
    Marchenko-Pastur eigenvalue bounds for a correlation matrix.
      lambda_min = (1 - sqrt(N/T))²
      lambda_max = (1 + sqrt(N/T))²
    Eigenvalues outside [lambda_min, lambda_max] contain real signal.
    """
    q = float(N) / float(T)
    if q <= 0 or q >= 1:
        return (1.0, 1.0)
    lam_min = (1 - np.sqrt(q)) ** 2
    lam_max = (1 + np.sqrt(q)) ** 2
    return float(lam_min), float(lam_max)


def clean_correlation_matrix(returns: pd.DataFrame,
                              method: str = "clip") -> Optional[pd.DataFrame]:
    """
    Apply MP eigenvalue cleaning to correlation matrix.

    method:
      "clip"     : replace bulk eigenvalues with their mean (default)
      "shrink"   : Ledoit-Wolf style shrinkage toward identity
      "discard"  : zero out bulk eigenvalues (aggressive)

    returns: DataFrame of returns (rows=time, cols=assets)
    """
    try:
        R = returns.dropna(axis=0).dropna(axis=1)
        T, N = R.shape
        if T < N + 5 or N < 2:
            return None

        # Sample correlation matrix
        C = R.corr().values
        assets = list(R.columns)

        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(C)
        sort_idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sort_idx]
        eigvecs = eigvecs[:, sort_idx]

        # MP bounds
        lam_min, lam_max = marchenko_pastur_bounds(T, N)

        # Identify bulk (noise) vs signal eigenvalues
        bulk_mask = eigvals <= lam_max

        if method == "clip":
            bulk_mean = np.mean(eigvals[bulk_mask]) if bulk_mask.any() else 0.0
            cleaned_eigvals = np.where(bulk_mask, bulk_mean, eigvals)
        elif method == "shrink":
            cleaned_eigvals = (eigvals + 1.0) / 2.0   # toward identity
        elif method == "discard":
            cleaned_eigvals = np.where(bulk_mask, 0.0, eigvals)
        else:
            cleaned_eigvals = eigvals

        # Rebuild correlation matrix
        C_clean = eigvecs @ np.diag(cleaned_eigvals) @ eigvecs.T

        # Renormalise diagonal to 1 (preserve correlation property)
        D = np.sqrt(np.diag(C_clean))
        D = np.where(D > 0, D, 1.0)
        C_clean = C_clean / np.outer(D, D)
        np.fill_diagonal(C_clean, 1.0)

        return pd.DataFrame(C_clean, index=assets, columns=assets)
    except Exception as e:
        logger.warning(f"RMT cleaning failed: {e}")
        return None


def clean_covariance_matrix(returns: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Clean covariance matrix by cleaning correlation + restoring individual variances.
    """
    try:
        clean_corr = clean_correlation_matrix(returns)
        if clean_corr is None:
            return None
        vols = returns.std()
        D = np.diag(vols.values)
        cov_clean = D @ clean_corr.values @ D
        return pd.DataFrame(cov_clean, index=clean_corr.index, columns=clean_corr.columns)
    except Exception as e:
        logger.warning(f"RMT covariance cleaning failed: {e}")
        return None


def signal_to_noise_ratio(returns: pd.DataFrame) -> Optional[dict]:
    """
    Diagnostic: how much of the correlation eigenvalue mass is signal vs noise?
    """
    try:
        R = returns.dropna(axis=0).dropna(axis=1)
        T, N = R.shape
        if T < N + 5 or N < 2:
            return None
        C = R.corr().values
        eigvals = np.linalg.eigvalsh(C)
        eigvals = np.sort(eigvals)[::-1]
        _, lam_max = marchenko_pastur_bounds(T, N)
        signal_mass = float(np.sum(eigvals[eigvals > lam_max]))
        noise_mass  = float(np.sum(eigvals[eigvals <= lam_max]))
        return {
            "T": T, "N": N,
            "mp_upper": round(lam_max, 4),
            "signal_eigenvalues":    int((eigvals > lam_max).sum()),
            "noise_eigenvalues":     int((eigvals <= lam_max).sum()),
            "signal_mass_fraction":  round(signal_mass / (signal_mass + noise_mass + 1e-12), 4),
            "largest_eigenvalue":    round(float(eigvals[0]), 4),
        }
    except Exception:
        return None
