"""
AlphaAgent — Google Trends Search Volume Signal

Free retail interest proxy via Google Trends. Spikes in search volume
for a brand/product correlate with retail buying activity and earnings
surprises 1-3 weeks ahead.

Reference: Da, Engelberg & Gao (2011), "In Search of Attention".

Uses pytrends (free, unofficial Google Trends API). Gracefully degrades
if pytrends is unavailable or rate-limited.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TrendsSignal:
    keyword: str
    current_value: float
    avg_value: float
    z_score: float
    trend_4w: float                  # 4-week trend slope
    signal: str                      # "BULLISH_ATTENTION" / "BEARISH_FADE" / "NEUTRAL"
    confidence: float


# Map common tickers to their primary brand keyword
TICKER_TO_KEYWORD = {
    "AAPL":  "iPhone",
    "TSLA":  "Tesla",
    "AMZN":  "Amazon",
    "GOOGL": "Google",
    "NVDA":  "Nvidia",
    "META":  "Facebook",
    "NFLX":  "Netflix",
    "MSFT":  "Microsoft",
    "DIS":   "Disney+",
    "UBER":  "Uber",
    "BA":    "Boeing",
    "F":     "Ford",
    "GM":    "GM",
    "WMT":   "Walmart",
    "TGT":   "Target",
    "COST":  "Costco",
    "MCD":   "McDonalds",
    "SBUX":  "Starbucks",
    "NKE":   "Nike",
    "HD":    "Home Depot",
    "LOW":   "Lowes",
}


def get_trends_signal(ticker: str,
                       timeframe: str = "today 3-m") -> Optional[TrendsSignal]:
    """
    Pull Google Trends for the ticker's primary keyword and compute z-score.
    """
    keyword = TICKER_TO_KEYWORD.get(ticker.upper())
    if not keyword:
        return None
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl="en-US", tz=360, timeout=(5, 10))
        pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo="", gprop="")
        df = pytrends.interest_over_time()
        if df is None or df.empty:
            return None
        series = df[keyword].dropna()
        if len(series) < 10:
            return None

        current = float(series.iloc[-1])
        avg = float(series.mean())
        std = float(series.std())
        z = (current - avg) / std if std > 0 else 0.0

        # 4-week trend slope
        recent = series.tail(28) if len(series) >= 28 else series
        x = np.arange(len(recent))
        slope, _ = np.polyfit(x, recent.values, 1)
        slope_pct = float(slope / max(avg, 1) * 100)

        if z > 1.5 and slope_pct > 0:
            signal = "BULLISH_ATTENTION"
        elif z > 2.5:
            signal = "BEARISH_FADE"   # extreme attention often peaks
        elif z < -1.0:
            signal = "BEARISH_FADE"   # interest collapsing
        else:
            signal = "NEUTRAL"

        return TrendsSignal(
            keyword=keyword,
            current_value=round(current, 2),
            avg_value=round(avg, 2),
            z_score=round(z, 3),
            trend_4w=round(slope_pct, 2),
            signal=signal,
            confidence=round(min(1.0, abs(z) / 3.0), 3),
        )
    except ImportError:
        logger.debug("pytrends not installed; Google Trends signal unavailable")
        return None
    except Exception as e:
        logger.debug(f"Google Trends failed for {ticker}: {e}")
        return None


_TRENDS_CACHE: dict = {}
_TRENDS_TTL = 12 * 3600   # 12 hours (rate-limit friendly)


def get_trends_signal_cached(ticker: str) -> Optional[TrendsSignal]:
    key = ticker.upper()
    cached = _TRENDS_CACHE.get(key)
    if cached and time.time() < cached[1]:
        return cached[0]
    sig = get_trends_signal(ticker)
    _TRENDS_CACHE[key] = (sig, time.time() + _TRENDS_TTL)
    return sig
