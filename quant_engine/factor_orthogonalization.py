"""
AlphaAgent — Factor Orthogonalization

Removes redundant information from a set of correlated factors via:
  - Gram-Schmidt orthogonalization (residualize on prior factors)
  - PCA rotation (uncorrelated principal components)
  - Symmetric orthogonalization (Löwdin, preserves rank ordering)

This prevents over-counting when combining factors that measure the
same underlying concept (e.g. RSI + Stochastic + Williams %R).

Reference: Klein & Chow (2013), "Orthogonalized Factors and Systematic Risk".
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OrthogonalResult:
    orthogonal_factors: pd.DataFrame
    correlation_before: pd.DataFrame
    correlation_after: pd.DataFrame
    method: str
    redundancy_reduction: float          # max off-diagonal corr before vs after


def gram_schmidt_orthogonalize(factors: pd.DataFrame,
                                 priority_order: Optional[List[str]] = None
                                 ) -> Optional[OrthogonalResult]:
    """
    Gram-Schmidt: orthogonalize factors one at a time by residualizing
    on all prior factors. Order matters — earlier factors keep more variance.

    priority_order : list of factor names ranked by importance
                     (default = column order)
    """
    try:
        df = factors.dropna(axis=0)
        if df.shape[0] < df.shape[1] + 5:
            return None
        cols = priority_order if priority_order else list(df.columns)
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return None

        ortho = pd.DataFrame(index=df.index)
        ortho[cols[0]] = df[cols[0]] - df[cols[0]].mean()   # de-mean
        for c in cols[1:]:
            y = df[c] - df[c].mean()
            X = ortho.values
            # OLS: y = Xβ + ε; residuals are orthogonal to X
            beta, *_ = np.linalg.lstsq(X, y.values, rcond=None)
            resid = y.values - X @ beta
            ortho[c] = resid

        corr_before = df.corr().abs()
        corr_after  = ortho.corr().abs()
        max_off_before = float((corr_before.values - np.eye(len(cols))).max())
        max_off_after  = float((corr_after.values - np.eye(len(cols))).max())
        reduction = max_off_before - max_off_after

        return OrthogonalResult(
            orthogonal_factors=ortho,
            correlation_before=corr_before,
            correlation_after=corr_after,
            method="gram_schmidt",
            redundancy_reduction=round(reduction, 4),
        )
    except Exception as e:
        logger.warning(f"Gram-Schmidt failed: {e}")
        return None


def pca_orthogonalize(factors: pd.DataFrame,
                      n_components: Optional[int] = None,
                      explained_var_threshold: float = 0.95
                      ) -> Optional[OrthogonalResult]:
    """
    PCA: rotate factors into uncorrelated principal components.
    Keeps components explaining `explained_var_threshold` of total variance.
    """
    try:
        df = factors.dropna(axis=0)
        if df.shape[0] < df.shape[1] + 5:
            return None
        X = df.values
        X_centered = X - X.mean(axis=0)
        # Standardise to give equal voice to each factor
        std = X_centered.std(axis=0)
        std = np.where(std > 0, std, 1.0)
        X_norm = X_centered / std

        cov = np.cov(X_norm, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Sort descending
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # Choose # components
        if n_components is None:
            cum_var = np.cumsum(eigvals) / eigvals.sum()
            n_components = int(np.searchsorted(cum_var, explained_var_threshold)) + 1
        n_components = min(n_components, X.shape[1])

        components = X_norm @ eigvecs[:, :n_components]
        pc_cols = [f"PC{i+1}" for i in range(n_components)]
        ortho = pd.DataFrame(components, index=df.index, columns=pc_cols)

        corr_before = df.corr().abs()
        corr_after  = ortho.corr().abs()
        max_off_before = float((corr_before.values - np.eye(df.shape[1])).max())
        max_off_after  = float((corr_after.values - np.eye(n_components)).max())
        reduction = max_off_before - max_off_after

        return OrthogonalResult(
            orthogonal_factors=ortho,
            correlation_before=corr_before,
            correlation_after=corr_after,
            method=f"pca({n_components} components)",
            redundancy_reduction=round(reduction, 4),
        )
    except Exception as e:
        logger.warning(f"PCA orthogonalization failed: {e}")
        return None


def symmetric_orthogonalize(factors: pd.DataFrame) -> Optional[OrthogonalResult]:
    """
    Löwdin symmetric orthogonalization: F* = F · (F'F)^(-1/2)
    Equal treatment of all factors — no ordering bias.
    Closest orthogonal set to the original (minimum rotation).
    """
    try:
        df = factors.dropna(axis=0)
        if df.shape[0] < df.shape[1] + 5:
            return None
        X = df.values
        X_centered = X - X.mean(axis=0)
        # Inverse square root of correlation matrix
        corr = np.corrcoef(X_centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.maximum(eigvals, 1e-10)
        sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        ortho_vals = X_centered @ sqrt_inv
        ortho = pd.DataFrame(ortho_vals, index=df.index, columns=df.columns)

        corr_before = df.corr().abs()
        corr_after  = ortho.corr().abs()
        max_off_before = float((corr_before.values - np.eye(df.shape[1])).max())
        max_off_after  = float((corr_after.values - np.eye(df.shape[1])).max())
        reduction = max_off_before - max_off_after

        return OrthogonalResult(
            orthogonal_factors=ortho,
            correlation_before=corr_before,
            correlation_after=corr_after,
            method="symmetric_lowdin",
            redundancy_reduction=round(reduction, 4),
        )
    except Exception as e:
        logger.warning(f"Symmetric orthogonalization failed: {e}")
        return None


def factor_redundancy_score(factor_dict: Dict[str, float]) -> float:
    """
    Quick proxy: given a dict of {factor_name: score}, estimate redundancy
    by clustering similar factor names. Returns 0 (no redundancy) to 1 (full).
    """
    if len(factor_dict) < 2:
        return 0.0
    # Simple heuristic: similar keywords → redundant
    families = {
        "momentum": ["rsi", "stochastic", "williams", "macd", "trix", "roc",
                     "momentum", "velocity"],
        "trend":    ["sma", "ema", "trend", "adx", "ichimoku", "supertrend"],
        "vol":      ["vol", "bollinger", "atr", "garch", "iv", "vix", "skew",
                     "kurtosis", "variance"],
        "value":    ["pe", "pb", "ev_ebitda", "pcf", "ps", "graham", "dcf"],
        "quality":  ["roe", "roa", "margin", "piotroski", "altman", "quality"],
    }
    counts = {fam: 0 for fam in families}
    for name in factor_dict.keys():
        nl = name.lower()
        for fam, kws in families.items():
            if any(kw in nl for kw in kws):
                counts[fam] += 1
                break
    # Redundancy = how clustered the factors are
    n = len(factor_dict)
    if n == 0:
        return 0.0
    max_cluster = max(counts.values()) if counts else 0
    return round(min(1.0, max_cluster / n), 4)
