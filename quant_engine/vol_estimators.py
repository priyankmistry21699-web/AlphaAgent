"""
AlphaAgent — High-Efficiency Volatility Estimators

Range-based volatility estimators that are 5–8x more statistically
efficient than close-to-close vol using only daily OHLC data.

Estimators:
  Parkinson (1980)        : High-Low range → 5x efficiency
  Garman-Klass (1980)     : OHLC → 7.4x efficiency
  Rogers-Satchell (1991)  : OHLC, drift-unbiased → 8x efficiency
  Yang-Zhang (2000)       : OHLC + overnight gap → best overall

All return annualised volatility (× sqrt(252)).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> Optional[float]:
    """Parkinson estimator (high-low range)."""
    try:
        h = np.log(high.astype(float))
        l = np.log(low.astype(float))
        rs = (1.0 / (4.0 * np.log(2.0))) * (h - l) ** 2
        return float(np.sqrt(rs.tail(window).mean() * 252))
    except Exception as e:
        logger.warning(f"Parkinson failed: {e}")
        return None


def garman_klass_vol(open_: pd.Series, high: pd.Series, low: pd.Series,
                     close: pd.Series, window: int = 20) -> Optional[float]:
    """Garman-Klass estimator (full OHLC)."""
    try:
        o = np.log(open_.astype(float))
        h = np.log(high.astype(float))
        l = np.log(low.astype(float))
        c = np.log(close.astype(float))
        term1 = 0.5 * (h - l) ** 2
        term2 = (2.0 * np.log(2.0) - 1.0) * (c - o) ** 2
        rs = term1 - term2
        return float(np.sqrt(rs.tail(window).mean() * 252))
    except Exception as e:
        logger.warning(f"Garman-Klass failed: {e}")
        return None


def rogers_satchell_vol(open_: pd.Series, high: pd.Series, low: pd.Series,
                        close: pd.Series, window: int = 20) -> Optional[float]:
    """Rogers-Satchell estimator (drift-unbiased OHLC)."""
    try:
        o = np.log(open_.astype(float))
        h = np.log(high.astype(float))
        l = np.log(low.astype(float))
        c = np.log(close.astype(float))
        rs = (h - c) * (h - o) + (l - c) * (l - o)
        return float(np.sqrt(rs.tail(window).mean() * 252))
    except Exception as e:
        logger.warning(f"Rogers-Satchell failed: {e}")
        return None


def yang_zhang_vol(open_: pd.Series, high: pd.Series, low: pd.Series,
                   close: pd.Series, window: int = 20) -> Optional[float]:
    """
    Yang-Zhang estimator (best overall; handles overnight gaps).
    σ²_YZ = σ²_overnight + k·σ²_open + (1-k)·σ²_RS
    """
    try:
        o = np.log(open_.astype(float))
        h = np.log(high.astype(float))
        l = np.log(low.astype(float))
        c = np.log(close.astype(float))

        # Overnight (close→open)
        overnight = o - c.shift(1)
        sigma_ov2 = overnight.tail(window).var()

        # Open→close
        oc = c - o
        sigma_oc2 = oc.tail(window).var()

        # Rogers-Satchell intraday
        rs = (h - c) * (h - o) + (l - c) * (l - o)
        sigma_rs2 = rs.tail(window).mean()

        n = window
        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        sigma_yz2 = sigma_ov2 + k * sigma_oc2 + (1 - k) * sigma_rs2
        return float(np.sqrt(max(sigma_yz2, 0) * 252))
    except Exception as e:
        logger.warning(f"Yang-Zhang failed: {e}")
        return None


def all_estimators(ohlcv: pd.DataFrame, window: int = 20) -> dict:
    """
    Compute all 4 range-based vol estimators + close-to-close baseline.
    ohlcv must have columns Open, High, Low, Close.
    """
    try:
        cc = float(ohlcv["Close"].pct_change().tail(window).std() * np.sqrt(252))
    except Exception:
        cc = None
    return {
        "close_to_close": cc,
        "parkinson":      parkinson_vol(ohlcv["High"], ohlcv["Low"], window),
        "garman_klass":   garman_klass_vol(ohlcv["Open"], ohlcv["High"],
                                            ohlcv["Low"], ohlcv["Close"], window),
        "rogers_satchell": rogers_satchell_vol(ohlcv["Open"], ohlcv["High"],
                                                ohlcv["Low"], ohlcv["Close"], window),
        "yang_zhang":     yang_zhang_vol(ohlcv["Open"], ohlcv["High"],
                                          ohlcv["Low"], ohlcv["Close"], window),
    }
