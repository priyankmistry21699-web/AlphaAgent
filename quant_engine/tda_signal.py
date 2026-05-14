"""
AlphaAgent — Topological Data Analysis (TDA) Signal Engine

Applies persistent homology to market return series to detect topological
features in price trajectories that standard statistics miss.

Pipeline:
  1. Takens delay embedding: reconstruct high-dim attractor from 1D returns
  2. Build Vietoris-Rips complex across filtration scales
  3. Compute persistent H0 (connected components) and H1 (loops) barcodes
  4. Extract features: max persistence, persistence entropy, Betti numbers,
     total persistence

Signal interpretation:
  H1 max persistence high → cyclic / mean-reverting regime (loops in attractor)
  H0 entropy low + H1 low → coherent trending regime
  H0 entropy high → fragmented, regime-switching market
  Total H1 persistence → complexity budget for cyclicity

Uses `ripser` (fast C++ persistent homology) if installed;
falls back to scipy-based single-linkage H0 approximation otherwise.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple

logger = logging.getLogger(__name__)

try:
    from ripser import ripser as _ripser_fn
    _HAS_RIPSER = True
except ImportError:
    _HAS_RIPSER = False
    logger.info("[TDA] ripser not installed — falling back to scipy H0 approximation.")


@dataclass
class TDAResult:
    h0_max_persistence: float        # Max lifetime of H0 features (components)
    h1_max_persistence: float        # Max lifetime of H1 features (loops)
    h0_persistence_entropy: float    # Shannon entropy of H0 persistence diagram
    h1_persistence_entropy: float    # Shannon entropy of H1 persistence diagram
    betti_0: int                     # H0 features surviving at mid-scale threshold
    betti_1: int                     # H1 features surviving at mid-scale threshold
    total_h1_persistence: float      # Sum of all H1 lifetimes (cyclic complexity)
    tda_signal: float                # [-1, +1]: +1 = trending, -1 = mean-reverting
    regime_label: str                # "TRENDING" | "CYCLIC" | "FRAGMENTED" | "NEUTRAL"
    warnings: List[str] = field(default_factory=list)


class TDASignalEngine:
    """
    Computes persistent homology of market return series to detect
    the topological 'shape' of market dynamics.

    Usage:
        engine = TDASignalEngine()
        result = engine.analyze(price_series)
        print(result.regime_label, result.tda_signal)
    """

    EMBED_DIM = 3       # Takens embedding dimension (2-4 typical for financial data)
    EMBED_DELAY = 2     # Lag for delay coordinates (autocorrelation-motivated)
    MIN_POINTS = 60     # Minimum price series length
    MAX_POINTS = 200    # Subsample beyond this for computational efficiency

    def __init__(self, embed_dim: int = EMBED_DIM, embed_delay: int = EMBED_DELAY):
        self.embed_dim = embed_dim
        self.embed_delay = embed_delay

    def analyze(self, prices: pd.Series) -> TDAResult:
        """
        Computes TDA features from a closing price series.

        Args:
            prices: Daily closing price series (pd.Series).
        Returns:
            TDAResult with topology-derived trading signal.
        """
        prices = prices.dropna()
        if len(prices) < self.MIN_POINTS:
            return TDAResult(
                h0_max_persistence=0.0, h1_max_persistence=0.0,
                h0_persistence_entropy=0.0, h1_persistence_entropy=0.0,
                betti_0=1, betti_1=0, total_h1_persistence=0.0,
                tda_signal=0.0, regime_label="NEUTRAL",
                warnings=["Insufficient data for TDA (need 60+ points)."],
            )

        log_ret = np.log(prices / prices.shift(1)).dropna().values

        # Subsample for efficiency while keeping the most recent window
        if len(log_ret) > self.MAX_POINTS:
            log_ret = log_ret[-self.MAX_POINTS:]

        point_cloud = self._takens_embedding(log_ret)

        if len(point_cloud) < 10:
            return TDAResult(
                h0_max_persistence=0.0, h1_max_persistence=0.0,
                h0_persistence_entropy=0.0, h1_persistence_entropy=0.0,
                betti_0=1, betti_1=0, total_h1_persistence=0.0,
                tda_signal=0.0, regime_label="NEUTRAL",
                warnings=["Takens embedding produced too few points."],
            )

        try:
            if _HAS_RIPSER:
                diagrams = self._compute_homology_ripser(point_cloud)
            else:
                diagrams = self._compute_homology_fallback(point_cloud)
        except Exception as e:
            logger.warning(f"[TDA] Homology computation failed: {e}")
            return TDAResult(
                h0_max_persistence=0.0, h1_max_persistence=0.0,
                h0_persistence_entropy=0.0, h1_persistence_entropy=0.0,
                betti_0=1, betti_1=0, total_h1_persistence=0.0,
                tda_signal=0.0, regime_label="NEUTRAL",
                warnings=[f"TDA computation error: {e}"],
            )

        h0_bars = diagrams[0]
        h1_bars = diagrams[1] if len(diagrams) > 1 else np.empty((0, 2))

        h0_life = self._finite_lifetimes(h0_bars)
        h1_life = self._finite_lifetimes(h1_bars)

        h0_max = float(np.max(h0_life)) if len(h0_life) > 0 else 0.0
        h1_max = float(np.max(h1_life)) if len(h1_life) > 0 else 0.0
        h0_ent = self._persistence_entropy(h0_life)
        h1_ent = self._persistence_entropy(h1_life)
        total_h1 = float(np.sum(h1_life))

        # Betti numbers: features alive at 50% of max H0 filtration scale
        threshold = 0.5 * h0_max if h0_max > 0 else 0.01
        betti_0 = max(1, int(np.sum(h0_life > threshold)))
        betti_1 = int(np.sum(h1_life > threshold))

        tda_signal, label = self._derive_signal(h1_max, h0_ent, betti_1)

        return TDAResult(
            h0_max_persistence=round(h0_max, 6),
            h1_max_persistence=round(h1_max, 6),
            h0_persistence_entropy=round(h0_ent, 4),
            h1_persistence_entropy=round(h1_ent, 4),
            betti_0=betti_0,
            betti_1=betti_1,
            total_h1_persistence=round(total_h1, 6),
            tda_signal=round(tda_signal, 3),
            regime_label=label,
        )

    def _takens_embedding(self, signal: np.ndarray) -> np.ndarray:
        """
        Constructs a delay-coordinate embedding from a 1D signal.

        Each row is a point in R^{embed_dim} formed by taking values
        at lags [0, delay, 2*delay, ..., (dim-1)*delay].
        """
        n = len(signal) - (self.embed_dim - 1) * self.embed_delay
        if n <= 0:
            return np.empty((0, self.embed_dim))

        indices = np.arange(self.embed_dim) * self.embed_delay
        points = np.array([signal[i + indices] for i in range(n)])

        # Normalize each dimension to [0,1] for scale-invariant distances
        for col in range(points.shape[1]):
            lo, hi = points[:, col].min(), points[:, col].max()
            if hi - lo > 1e-10:
                points[:, col] = (points[:, col] - lo) / (hi - lo)

        return points

    @staticmethod
    def _compute_homology_ripser(points: np.ndarray) -> List[np.ndarray]:
        """Persistent H0 and H1 via the ripser library (fast Rips complex)."""
        result = _ripser_fn(points, maxdim=1)
        return result["dgms"]

    @staticmethod
    def _compute_homology_fallback(points: np.ndarray) -> List[np.ndarray]:
        """
        Fallback H0 via scipy single-linkage hierarchical clustering.
        H0 birth-death pairs are the merge distances from the dendrogram.
        H1 is approximated as empty (no loop detection without ripser).
        """
        from scipy.spatial.distance import pdist, squareform
        from scipy.cluster.hierarchy import linkage

        dists = squareform(pdist(points, metric="euclidean"))
        Z = linkage(dists, method="single")
        # Each row of Z records a merge at distance Z[i,2]
        births = np.zeros(len(Z))
        deaths = Z[:, 2]
        h0_bars = np.column_stack([births, deaths])
        return [h0_bars, np.empty((0, 2))]

    @staticmethod
    def _finite_lifetimes(bars: np.ndarray) -> np.ndarray:
        """Returns persistence lifetimes for all finite bars (death < inf)."""
        if len(bars) == 0:
            return np.array([])
        finite = np.isfinite(bars[:, 1])
        lives = bars[finite, 1] - bars[finite, 0]
        return lives[lives > 0]

    @staticmethod
    def _persistence_entropy(lifetimes: np.ndarray) -> float:
        """Shannon entropy of the persistence diagram (normalized lifetimes)."""
        if len(lifetimes) == 0 or lifetimes.sum() < 1e-12:
            return 0.0
        p = lifetimes / lifetimes.sum()
        return float(-np.sum(p * np.log(p + 1e-12)))

    @staticmethod
    def _derive_signal(
        h1_max: float,
        h0_entropy: float,
        betti_1: int,
    ) -> Tuple[float, str]:
        """
        Maps TDA features to a directional signal and regime label.

        Logic:
          High H1 max → significant looping → cyclic / mean-reverting
          Low H1 max + low entropy → coherent, trending
          High entropy, low H1 → fragmented / regime change
        """
        # Normalize H1 max against an empirical scale (30% of normalized range)
        h1_norm = float(np.clip(h1_max / 0.30, 0, 1))
        ent_norm = float(np.clip(h0_entropy / 2.0, 0, 1))

        if h1_norm > 0.6 or betti_1 >= 3:
            # Strong cyclic structure → mean-reverting bias
            signal = -h1_norm
            return signal, "CYCLIC"
        elif ent_norm > 0.7 and h1_norm < 0.3:
            # High entropy, no loops → fragmented market
            return 0.0, "FRAGMENTED"
        elif h1_norm < 0.2 and ent_norm < 0.4:
            # Low complexity → coherent trending
            return 1.0, "TRENDING"
        else:
            # Mixed signal — interpolate
            signal = float((0.5 - h1_norm) * 2.0)
            return float(np.clip(signal, -1, 1)), "NEUTRAL"
