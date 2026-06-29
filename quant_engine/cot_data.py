"""
AlphaAgent — CFTC Commitment of Traders (COT) Data

Free weekly positioning data from CFTC. When commercials (actual producers
and consumers — the "smart money") are heavily net-long a commodity, it's
a strong buy signal historically.

Categories tracked:
  - Commercials (hedgers / smart money)
  - Non-commercials (speculators / momentum)
  - Non-reportables (small specs)

Usage:
    from quant_engine.cot_data import get_cot_signal
    signal = get_cot_signal("CL")   # crude oil
    # signal.commercials_net, signal.commercials_z, signal.signal
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import urlencode

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# CFTC commodity codes → futures contracts symbols
COT_COMMODITY_MAP: Dict[str, dict] = {
    "CL":   {"code": "067651", "name": "Crude Oil", "etf": "USO"},
    "NG":   {"code": "023651", "name": "Natural Gas", "etf": "UNG"},
    "GC":   {"code": "088691", "name": "Gold", "etf": "GLD"},
    "SI":   {"code": "084691", "name": "Silver", "etf": "SLV"},
    "HG":   {"code": "085692", "name": "Copper", "etf": "CPER"},
    "ZW":   {"code": "001602", "name": "Wheat", "etf": "WEAT"},
    "ZC":   {"code": "002602", "name": "Corn", "etf": "CORN"},
    "ZS":   {"code": "005602", "name": "Soybeans", "etf": "SOYB"},
    "ES":   {"code": "13874A", "name": "S&P 500", "etf": "SPY"},
    "EUR":  {"code": "099741", "name": "EUR/USD", "etf": "FXE"},
    "JPY":  {"code": "097741", "name": "JPY/USD", "etf": "FXY"},
}

# CFTC Socrata Open Data API (free, no key required)
CFTC_API_URL = "https://publicreporting.cftc.gov/resource/jun7-fc8e.json"


@dataclass
class COTSignal:
    symbol: str
    name: str
    report_date: str
    commercials_long: int
    commercials_short: int
    commercials_net: int         # long - short
    commercials_net_pct: float   # net as % of open interest
    commercials_z: float         # z-score vs 52-week distribution
    speculators_net: int
    open_interest: int
    signal: str                  # "BUY_SMART_MONEY" / "SELL_SMART_MONEY" / "NEUTRAL"


def get_cot_signal(symbol: str, weeks: int = 52) -> Optional[COTSignal]:
    """
    Fetch latest CFTC COT report for a commodity and compute z-scored
    commercial positioning signal.
    """
    info = COT_COMMODITY_MAP.get(symbol.upper())
    if not info:
        return None
    try:
        import requests
        params = {
            "$where": f"cftc_contract_market_code='{info['code']}'",
            "$order": "report_date_as_yyyy_mm_dd DESC",
            "$limit": str(weeks),
        }
        url = CFTC_API_URL + "?" + urlencode(params)
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data:
            return None

        df = pd.DataFrame(data)
        # Cast numerics
        for col in ["comm_positions_long_all", "comm_positions_short_all",
                    "noncomm_positions_long_all", "noncomm_positions_short_all",
                    "open_interest_all"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["comm_net"] = df["comm_positions_long_all"] - df["comm_positions_short_all"]
        df["spec_net"] = df["noncomm_positions_long_all"] - df["noncomm_positions_short_all"]
        df["comm_net_pct"] = df["comm_net"] / df["open_interest_all"].replace(0, np.nan) * 100

        latest = df.iloc[0]
        comm_net_series = df["comm_net"].dropna()
        if len(comm_net_series) < 10:
            return None

        mu = float(comm_net_series.mean())
        sigma = float(comm_net_series.std())
        z = (float(latest["comm_net"]) - mu) / sigma if sigma > 0 else 0.0

        # Signal classification — extreme commercial net positions tend to mean revert
        if z > 1.5:
            signal = "BUY_SMART_MONEY"   # commercials extremely long → bullish
        elif z < -1.5:
            signal = "SELL_SMART_MONEY"  # commercials extremely short → bearish
        else:
            signal = "NEUTRAL"

        return COTSignal(
            symbol=symbol.upper(),
            name=info["name"],
            report_date=str(latest.get("report_date_as_yyyy_mm_dd", "")),
            commercials_long=int(latest["comm_positions_long_all"]),
            commercials_short=int(latest["comm_positions_short_all"]),
            commercials_net=int(latest["comm_net"]),
            commercials_net_pct=round(float(latest["comm_net_pct"] or 0), 2),
            commercials_z=round(z, 3),
            speculators_net=int(latest["spec_net"]),
            open_interest=int(latest["open_interest_all"] or 0),
            signal=signal,
        )
    except Exception as e:
        logger.warning(f"COT fetch failed for {symbol}: {e}")
        return None


_COT_CACHE: dict = {}
_COT_TTL = 86400   # 1 day


def get_cot_signal_cached(symbol: str) -> Optional[COTSignal]:
    """Cached COT fetch (1-day TTL — CFTC publishes weekly)."""
    key = symbol.upper()
    cached = _COT_CACHE.get(key)
    if cached and time.time() < cached[1]:
        return cached[0]
    sig = get_cot_signal(symbol)
    _COT_CACHE[key] = (sig, time.time() + _COT_TTL)
    return sig
