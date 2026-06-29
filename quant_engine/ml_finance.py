"""
AlphaAgent — López de Prado ML Finance Framework

Three pillars of "Advances in Financial Machine Learning" (Prado 2018):
  1. Fractional Differentiation — make price series stationary
     without losing memory (d ∈ (0, 1))
  2. Triple Barrier Labeling — label outcomes via profit-take /
     stop-loss / time barriers (not fixed-horizon returns)
  3. Purged K-Fold Cross-Validation — remove observations adjacent
     to test set to prevent leakage from overlapping labels
  4. Walk-Forward Validation — out-of-sample evaluation framework
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
#  1. Fractional Differentiation (Prado, Chapter 5)
# ════════════════════════════════════════════════════════════════════════

def _get_weights_ffd(d: float, threshold: float = 1e-4) -> np.ndarray:
    """Get FFD weights until cumulative magnitude < threshold."""
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
        k += 1
        if k > 10_000:
            break
    return np.array(w[::-1])


def fractional_difference(series: pd.Series,
                          d: float = 0.4,
                          threshold: float = 1e-4) -> Optional[pd.Series]:
    """
    Fixed-window fractional differentiation.
    d=1 → first differences (returns)
    d=0 → original series
    d∈(0,1) → preserves long memory while inducing stationarity
    """
    try:
        w = _get_weights_ffd(d, threshold)
        width = len(w) - 1
        if width >= len(series):
            return None
        out = pd.Series(index=series.index, dtype=float)
        s = series.dropna()
        for i in range(width, len(s)):
            window = s.iloc[i - width:i + 1].values
            out.iloc[i] = float(np.dot(w, window))
        return out.dropna()
    except Exception as e:
        logger.warning(f"Fractional diff failed: {e}")
        return None


def optimal_d_value(series: pd.Series,
                    d_grid: np.ndarray = None,
                    p_value_threshold: float = 0.05) -> Optional[float]:
    """
    Find smallest d that achieves stationarity (ADF p < threshold).
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        if d_grid is None:
            d_grid = np.linspace(0.05, 0.95, 19)
        for d in d_grid:
            ffd = fractional_difference(series, d=float(d))
            if ffd is None or len(ffd) < 30:
                continue
            try:
                p = adfuller(ffd, maxlag=1, regression="c", autolag=None)[1]
                if p < p_value_threshold:
                    return float(d)
            except Exception:
                continue
        return 1.0
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════════
#  2. Triple Barrier Labeling (Prado, Chapter 3)
# ════════════════════════════════════════════════════════════════════════

@dataclass
class TripleBarrierLabel:
    label: int          # 1 = profit-take hit, -1 = stop-loss hit, 0 = time expired
    barrier_time: int   # index at which a barrier was hit
    barrier_type: str   # "PT", "SL", "TIME"
    realized_return: float


def triple_barrier_label(prices: pd.Series,
                          start_idx: int,
                          profit_take_pct: float,
                          stop_loss_pct: float,
                          time_barrier_days: int) -> Optional[TripleBarrierLabel]:
    """
    Label a position entered at prices[start_idx] using profit-take, stop-loss,
    and time barriers. Returns -1/0/+1 based on which barrier hits first.
    """
    try:
        if start_idx >= len(prices) - 1:
            return None
        entry = float(prices.iloc[start_idx])
        if entry <= 0:
            return None
        pt = entry * (1 + profit_take_pct)
        sl = entry * (1 - stop_loss_pct)
        end_idx = min(start_idx + time_barrier_days, len(prices) - 1)
        for i in range(start_idx + 1, end_idx + 1):
            p = float(prices.iloc[i])
            if p >= pt:
                return TripleBarrierLabel(1, i, "PT", (p / entry - 1))
            if p <= sl:
                return TripleBarrierLabel(-1, i, "SL", (p / entry - 1))
        # Time barrier hit
        p_end = float(prices.iloc[end_idx])
        return TripleBarrierLabel(0, end_idx, "TIME", (p_end / entry - 1))
    except Exception as e:
        logger.warning(f"Triple barrier failed: {e}")
        return None


def label_dataset(prices: pd.Series,
                   entry_signals: pd.Series,
                   profit_take_pct: float = 0.03,
                   stop_loss_pct: float = 0.02,
                   time_barrier_days: int = 10) -> pd.DataFrame:
    """
    Apply triple-barrier labeling to a series of entry signals.
    entry_signals : pd.Series of bools or ints (1 = enter LONG)
    Returns DataFrame indexed by entry timestamps with label/return/barrier.
    """
    rows = []
    for ts, sig in entry_signals.items():
        if not sig:
            continue
        idx = prices.index.get_loc(ts) if ts in prices.index else None
        if idx is None:
            continue
        lbl = triple_barrier_label(prices, idx, profit_take_pct,
                                    stop_loss_pct, time_barrier_days)
        if lbl is not None:
            rows.append({
                "entry_time":    ts,
                "label":         lbl.label,
                "barrier_type":  lbl.barrier_type,
                "barrier_idx":   lbl.barrier_time,
                "realized_return": round(lbl.realized_return, 4),
            })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════
#  3. Purged K-Fold Cross-Validation (Prado, Chapter 7)
# ════════════════════════════════════════════════════════════════════════

def purged_k_fold_splits(n_samples: int,
                          n_splits: int = 5,
                          embargo_pct: float = 0.01
                          ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Purged K-Fold CV with embargo period to prevent leakage.

    n_samples   : total observations
    n_splits    : number of folds
    embargo_pct : fraction of n_samples to embargo after each test fold
    """
    splits = []
    fold_size = n_samples // n_splits
    embargo = int(embargo_pct * n_samples)
    for k in range(n_splits):
        test_start = k * fold_size
        test_end = min((k + 1) * fold_size, n_samples)
        test_idx = np.arange(test_start, test_end)
        # Train = all - test - embargo
        embargo_end = min(test_end + embargo, n_samples)
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[test_start:embargo_end] = False
        train_idx = np.where(train_mask)[0]
        splits.append((train_idx, test_idx))
    return splits


# ════════════════════════════════════════════════════════════════════════
#  4. Walk-Forward Validation
# ════════════════════════════════════════════════════════════════════════

def walk_forward_splits(n_samples: int,
                         initial_train: int,
                         test_window: int,
                         step: int = 0,
                         anchored: bool = True
                         ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate walk-forward train/test splits.

    initial_train : initial training set size
    test_window   : test set size per fold
    step          : step size for sliding window (default = test_window)
    anchored      : True = expand training set, False = rolling fixed window
    """
    if step <= 0:
        step = test_window
    splits = []
    train_start = 0
    train_end = initial_train
    while train_end + test_window <= n_samples:
        if anchored:
            train_idx = np.arange(0, train_end)
        else:
            train_idx = np.arange(train_start, train_end)
        test_idx = np.arange(train_end, train_end + test_window)
        splits.append((train_idx, test_idx))
        train_end += step
        train_start += step
    return splits
