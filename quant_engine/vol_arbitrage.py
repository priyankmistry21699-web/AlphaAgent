"""
AlphaAgent — Volatility Arbitrage Signal (VRP Trade)

Variance Risk Premium = IV² − RV² > 0 on average; investors systematically
overpay for options as insurance. Selling rich vol vs realising cheap vol
is one of the most consistent risk premia in markets.

Translates the existing VRP factor into an explicit trade signal:
  - VRP elevated → SHORT vol (sell straddles, calendar spreads)
  - VRP compressed → LONG vol (buy options, expect realisation > IV)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class VolArbSignal:
    ticker: str
    iv_annual: float                # implied vol (ann)
    rv_annual: float                # realised vol (ann)
    vrp: float                       # IV² - RV² (annualised variance points)
    vrp_z_score: float               # z-score vs 60d distribution
    signal: str                      # "SHORT_VOL" / "LONG_VOL" / "NEUTRAL"
    expected_edge_bps: float         # estimated edge in basis points
    confidence: float


def compute_vol_arbitrage(ticker: str,
                          iv_pct: float,
                          ohlcv: pd.DataFrame,
                          lookback_days: int = 60) -> Optional[VolArbSignal]:
    """
    iv_pct  : current implied volatility as a decimal (e.g. 0.22 = 22%)
    ohlcv   : DataFrame with at least Close column for realised vol
    """
    try:
        if ohlcv is None or len(ohlcv) < lookback_days + 5:
            return None
        if iv_pct <= 0:
            return None

        rets = ohlcv["Close"].pct_change().dropna()
        if len(rets) < lookback_days:
            return None

        rv_ann = float(rets.tail(lookback_days).std() * np.sqrt(252))
        vrp_current = (iv_pct ** 2 - rv_ann ** 2) * 100   # variance points × 100

        # Build VRP history for z-score
        vrp_series = []
        for end_i in range(lookback_days, len(rets)):
            window_rv = float(rets.iloc[end_i - lookback_days:end_i].std() * np.sqrt(252))
            # Approximation: use scaling of iv_pct (we don't have historical IV per day)
            vrp_series.append((iv_pct ** 2 - window_rv ** 2) * 100)
        vrp_series = pd.Series(vrp_series)
        mu = float(vrp_series.mean())
        sigma = float(vrp_series.std())
        z = (vrp_current - mu) / sigma if sigma > 0 else 0.0

        # Signal classification
        if z > 1.0:
            signal = "SHORT_VOL"     # vol rich
        elif z < -1.0:
            signal = "LONG_VOL"      # vol cheap
        else:
            signal = "NEUTRAL"

        # Estimated edge: VRP in bps of underlying
        edge_bps = abs(vrp_current) * 0.5 * 100   # rough straddle P&L proxy
        confidence = float(min(1.0, abs(z) / 2.5))

        return VolArbSignal(
            ticker=ticker.upper(),
            iv_annual=round(float(iv_pct), 4),
            rv_annual=round(rv_ann, 4),
            vrp=round(vrp_current, 3),
            vrp_z_score=round(z, 2),
            signal=signal,
            expected_edge_bps=round(edge_bps, 1),
            confidence=round(confidence, 3),
        )
    except Exception as e:
        logger.warning(f"Vol arbitrage failed for {ticker}: {e}")
        return None
