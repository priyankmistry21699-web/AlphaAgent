"""
AlphaAgent — Commodity Roll Yield

Commodity futures must be rolled monthly/quarterly from the expiring front
contract to the next active month. This produces a "roll yield":
  - Contango : front < next month → negative roll (drag)
  - Backwardation : front > next month → positive roll (gain)

Roll yield explains 30-40% of long-run commodity ETF performance.
USO, UNG, and DBC are notorious contango victims.

Usage:
    from quant_engine.commodity_roll_yield import compute_roll_yield
    rr = compute_roll_yield("CL")
    # rr.roll_yield_annual, rr.curve_state, rr.signal
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RollYieldResult:
    symbol: str
    front_price: float
    next_price: float
    roll_yield_monthly: float       # (front/next - 1)
    roll_yield_annual: float        # annualised
    curve_state: str                # "CONTANGO" / "BACKWARDATION" / "FLAT"
    signal: str                     # "AVOID_ETF" / "FAVORABLE_ETF" / "NEUTRAL"
    severity: str                   # "mild" / "moderate" / "severe"


# Commodity futures roll calendars (Yahoo Finance symbol → next contract pattern)
# Yahoo doesn't expose future contracts directly; we use spot + ETF proxy
COMMODITY_ROLL_PROXIES = {
    "CL": {"front": "CL=F", "etf": "USO", "name": "Crude Oil"},
    "NG": {"front": "NG=F", "etf": "UNG", "name": "Natural Gas"},
    "GC": {"front": "GC=F", "etf": "GLD", "name": "Gold"},
    "SI": {"front": "SI=F", "etf": "SLV", "name": "Silver"},
    "HG": {"front": "HG=F", "etf": "CPER", "name": "Copper"},
    "ZW": {"front": "ZW=F", "etf": "WEAT", "name": "Wheat"},
    "ZC": {"front": "ZC=F", "etf": "CORN", "name": "Corn"},
    "ZS": {"front": "ZS=F", "etf": "SOYB", "name": "Soybeans"},
}


def compute_roll_yield(symbol: str,
                       lookback_days: int = 60) -> Optional[RollYieldResult]:
    """
    Estimate roll yield by comparing ETF performance vs spot futures performance.
    The gap (ETF underperformance vs spot) is approximately the roll yield.

    This is the practical "tracking error" approach commonly used by quants
    when full futures curves aren't available.
    """
    info = COMMODITY_ROLL_PROXIES.get(symbol.upper())
    if not info:
        return None
    try:
        import yfinance as yf

        df_etf = yf.download(info["etf"], period="6mo", interval="1d",
                              auto_adjust=True, progress=False)
        df_spot = yf.download(info["front"], period="6mo", interval="1d",
                               auto_adjust=True, progress=False)
        if df_etf.empty or df_spot.empty:
            return None

        s_etf = df_etf["Close"].squeeze().dropna()
        s_spot = df_spot["Close"].squeeze().dropna()
        df = pd.concat([s_etf, s_spot], axis=1, join="inner").dropna()
        if len(df) < 30:
            return None
        df.columns = ["etf", "spot"]

        # Returns over lookback
        period = min(lookback_days, len(df) - 1)
        etf_ret = float(df["etf"].iloc[-1] / df["etf"].iloc[-period] - 1)
        spot_ret = float(df["spot"].iloc[-1] / df["spot"].iloc[-period] - 1)
        gap = etf_ret - spot_ret   # negative gap = roll yield drag

        # Annualise the monthly gap (roll yield)
        monthly_roll = gap / (period / 21.0)
        annual_roll = monthly_roll * 12

        # Curve state
        if annual_roll < -0.10:
            curve_state = "CONTANGO"
            severity = "severe" if annual_roll < -0.25 else "moderate" if annual_roll < -0.15 else "mild"
            signal = "AVOID_ETF"
        elif annual_roll > 0.10:
            curve_state = "BACKWARDATION"
            severity = "severe" if annual_roll > 0.25 else "moderate" if annual_roll > 0.15 else "mild"
            signal = "FAVORABLE_ETF"
        else:
            curve_state = "FLAT"
            severity = "mild"
            signal = "NEUTRAL"

        return RollYieldResult(
            symbol=symbol.upper(),
            front_price=round(float(df["spot"].iloc[-1]), 3),
            next_price=round(float(df["etf"].iloc[-1]), 3),
            roll_yield_monthly=round(monthly_roll, 4),
            roll_yield_annual=round(annual_roll, 4),
            curve_state=curve_state,
            signal=signal,
            severity=severity,
        )
    except Exception as e:
        logger.warning(f"Roll yield failed for {symbol}: {e}")
        return None


_ROLL_CACHE: dict = {}
_ROLL_TTL = 21600   # 6 hours


def compute_roll_yield_cached(symbol: str) -> Optional[RollYieldResult]:
    key = symbol.upper()
    cached = _ROLL_CACHE.get(key)
    if cached and time.time() < cached[1]:
        return cached[0]
    r = compute_roll_yield(symbol)
    _ROLL_CACHE[key] = (r, time.time() + _ROLL_TTL)
    return r
