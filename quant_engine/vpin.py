"""
AlphaAgent — VPIN (Volume-Synchronized Probability of Informed Trading)

Theory: Easley, López de Prado & O'Hara (2012).
  Measures the toxicity of order flow — the probability that a trade
  comes from an informed (not noise) trader.  VPIN spikes precede large
  price moves and are a leading indicator of volatility cascades.

Algorithm:
  1. Divide daily volume into equal-size volume buckets (V buckets).
  2. Classify each bucket as buy-initiated or sell-initiated using the
     bulk volume classification (BVC) method: if price rises during the
     bucket → buy volume, else → sell volume.
  3. OI (Order Imbalance) per bucket = |buys - sells| / bucket_size
  4. VPIN = rolling mean of OI over the last n buckets

Score mapping (0-100):
  VPIN < 0.25 : 65 (low toxicity — safe liquidity)
  VPIN 0.25-0.50 : 55
  VPIN 0.50-0.65 : 45 (elevated — caution)
  VPIN > 0.65  : 25 (high toxicity — informed trading likely)

Trend:
  Rising VPIN → additional bearish adjustment (informed selling)
  Falling VPIN → slight bullish (pressure releasing)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_N_BUCKETS    = 50     # rolling window: last 50 buckets
_BUCKET_MULT  = 0.01   # bucket size = 1% of avg daily volume


@dataclass
class VPINResult:
    vpin: float               # current VPIN value 0-1
    vpin_trend: float         # VPIN now vs 5-bucket MA (positive = rising)
    score: float              # 0-100 factor score
    prob_adjustment: float    # direct probability nudge
    regime: str               # LOW / MEDIUM / ELEVATED / HIGH
    bucket_size: float
    n_buckets_used: int


def compute_vpin(
    ohlcv: pd.DataFrame,
    n_buckets: int = _N_BUCKETS,
) -> Optional[VPINResult]:
    """
    Compute VPIN from daily OHLCV bars using bulk volume classification.
    ohlcv must have columns: Open, High, Low, Close, Volume
    Returns None if insufficient data.
    """
    try:
        if ohlcv is None or ohlcv.empty or len(ohlcv) < 30:
            return None

        df = ohlcv[["Open", "High", "Low", "Close", "Volume"]].copy().dropna()
        if len(df) < 30:
            return None

        closes  = df["Close"].values.astype(float)
        volumes = df["Volume"].values.astype(float)
        opens   = df["Open"].values.astype(float)

        # ── 1. Bulk Volume Classification ────────────────────────────────
        # Z-score method: classify each bar's volume as buy/sell by price change
        price_changes = closes - opens
        bar_sigma     = np.std(price_changes[price_changes != 0]) or 1e-6

        buy_fracs  = np.array([_norm_cdf(dp / bar_sigma) for dp in price_changes])
        sell_fracs = 1.0 - buy_fracs

        buy_vols  = volumes * buy_fracs
        sell_vols = volumes * sell_fracs

        # ── 2. Bucket volume ─────────────────────────────────────────────
        avg_daily_vol = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        bucket_size   = max(1.0, avg_daily_vol * _BUCKET_MULT)

        # Build buckets by accumulating volume
        oi_list   = []
        cum_buy   = 0.0
        cum_sell  = 0.0
        cum_vol   = 0.0

        for bv, sv in zip(buy_vols, sell_vols):
            remaining_b = bv
            remaining_s = sv

            while remaining_b + remaining_s > 0:
                space = bucket_size - cum_vol
                fill  = min(remaining_b + remaining_s, space)
                frac  = fill / (remaining_b + remaining_s) if (remaining_b + remaining_s) > 0 else 0
                cum_buy  += remaining_b  * frac
                cum_sell += remaining_s  * frac
                cum_vol  += fill
                remaining_b = remaining_b  * (1 - frac)
                remaining_s = remaining_s  * (1 - frac)

                if cum_vol >= bucket_size * 0.999:
                    if bucket_size > 0:
                        oi_list.append(abs(cum_buy - cum_sell) / bucket_size)
                    cum_buy = cum_sell = cum_vol = 0.0

        if len(oi_list) < max(5, n_buckets // 2):
            return None

        oi_arr = np.array(oi_list)

        # ── 3. VPIN = rolling mean of last n_buckets ──────────────────────
        window  = min(n_buckets, len(oi_arr))
        vpin    = float(np.mean(oi_arr[-window:]))

        # Trend: compare last 5 buckets vs prior 15
        if len(oi_arr) >= 20:
            vpin_recent = float(np.mean(oi_arr[-5:]))
            vpin_prior  = float(np.mean(oi_arr[-20:-5]))
            vpin_trend  = vpin_recent - vpin_prior
        else:
            vpin_trend = 0.0

        # ── 4. Score ──────────────────────────────────────────────────────
        if vpin < 0.25:
            score   = 65.0
            regime  = "LOW"
            prob_adj = 0.02
        elif vpin < 0.50:
            score   = 55.0
            regime  = "MEDIUM"
            prob_adj = 0.0
        elif vpin < 0.65:
            score   = 40.0
            regime  = "ELEVATED"
            prob_adj = -0.03
        else:
            score   = 22.0
            regime  = "HIGH"
            prob_adj = -0.06

        # Trend adjustment
        if vpin_trend > 0.05:
            score    -= 8.0
            prob_adj -= 0.02
        elif vpin_trend < -0.05:
            score    += 5.0
            prob_adj += 0.01

        score = float(np.clip(score, 10.0, 90.0))

        logger.debug(
            f"[VPIN] vpin={vpin:.3f} regime={regime} "
            f"trend={vpin_trend:+.3f} score={score:.1f}"
        )
        return VPINResult(
            vpin=round(vpin, 4),
            vpin_trend=round(vpin_trend, 4),
            score=round(score, 1),
            prob_adjustment=round(prob_adj, 4),
            regime=regime,
            bucket_size=round(bucket_size, 0),
            n_buckets_used=window,
        )

    except Exception as e:
        logger.warning(f"[VPIN] compute error: {e}")
        return None


def _norm_cdf(z: float) -> float:
    """Fast approximation of normal CDF — avoids scipy import overhead."""
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))
