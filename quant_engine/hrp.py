"""
AlphaAgent — Hierarchical Risk Parity (HRP)

López de Prado (2016) HRP allocation. Uses hierarchical clustering
on the correlation matrix to build a quasi-diagonal covariance and
recursively bisect to allocate risk-parity weights.

Advantages over MVO:
  - No covariance matrix inversion (numerically stable at N>50)
  - Robust to estimation error in correlation matrix
  - Cluster-aware allocation

Usage:
    from quant_engine.hrp import HierarchicalRiskParity
    weights = HierarchicalRiskParity().allocate(returns_df)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class HRPResult:
    weights: Dict[str, float]
    cluster_order: List[str]
    n_assets: int


class HierarchicalRiskParity:
    """
    HRP allocation via single-linkage clustering on correlation distance.
    """

    def _correl_dist(self, corr: np.ndarray) -> np.ndarray:
        """Convert correlation to distance: d_ij = sqrt(0.5 * (1 - rho_ij))."""
        return np.sqrt(0.5 * (1 - corr))

    def _get_quasi_diag(self, link: np.ndarray) -> List[int]:
        """Sort items so that cluster-adjacent items are close in the order."""
        n = link.shape[0] + 1
        sorted_idx = list(map(int, link[-1, :2]))
        clusters = {}
        for i in range(link.shape[0]):
            clusters[n + i] = list(map(int, link[i, :2]))

        def expand(node):
            if node < n:
                return [node]
            return expand(clusters[node][0]) + expand(clusters[node][1])

        flat = []
        for c in sorted_idx:
            flat.extend(expand(c))
        return flat

    def _get_recursive_bisection(self, cov: np.ndarray, sort_idx: List[int]) -> np.ndarray:
        """Allocate inversely proportional to cluster variance via recursive bisection."""
        n = cov.shape[0]
        w = np.ones(n)
        clusters = [sort_idx]
        while clusters:
            new_clusters = []
            for c in clusters:
                if len(c) <= 1:
                    continue
                split = len(c) // 2
                c1, c2 = c[:split], c[split:]
                # Variances of each sub-cluster (inverse-variance weighted)
                v1 = self._cluster_var(cov, c1)
                v2 = self._cluster_var(cov, c2)
                if v1 + v2 == 0:
                    alpha = 0.5
                else:
                    alpha = 1 - v1 / (v1 + v2)
                w[c1] *= alpha
                w[c2] *= 1 - alpha
                new_clusters.extend([c1, c2])
            clusters = new_clusters
        return w

    def _cluster_var(self, cov: np.ndarray, idx: List[int]) -> float:
        sub = cov[np.ix_(idx, idx)]
        diag = np.diag(sub)
        # Inverse-variance weights within cluster
        ivp = 1.0 / np.maximum(diag, 1e-10)
        ivp = ivp / ivp.sum()
        return float(ivp @ sub @ ivp)

    def allocate(self, returns: pd.DataFrame) -> Optional[HRPResult]:
        """
        returns: DataFrame of daily returns, columns = tickers.
        Returns HRP weights dict (sum to 1).
        """
        try:
            from scipy.cluster.hierarchy import linkage
            from scipy.spatial.distance import squareform

            returns = returns.dropna(axis=1, how="all").dropna(axis=0)
            assets = list(returns.columns)
            n = len(assets)
            if n < 2:
                return None

            corr = returns.corr().values
            cov  = returns.cov().values
            d    = self._correl_dist(corr)
            np.fill_diagonal(d, 0)
            d = (d + d.T) / 2   # ensure symmetric for squareform

            link = linkage(squareform(d, checks=False), method="single")
            sort_idx = self._get_quasi_diag(link)
            w = self._get_recursive_bisection(cov, sort_idx)
            w = w / w.sum()

            return HRPResult(
                weights={assets[i]: float(w[i]) for i in range(n)},
                cluster_order=[assets[i] for i in sort_idx],
                n_assets=n,
            )
        except Exception as e:
            logger.warning(f"HRP allocation failed: {e}")
            return None
