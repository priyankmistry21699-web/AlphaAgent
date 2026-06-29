"""
AlphaAgent — Polygon.io REST Client (P0 Data-Quality Fix)

Free-tier (no API key) endpoints used:
  - Previous close         : /v2/aggs/ticker/{sym}/prev
  - Daily OHLCV bars       : /v2/aggs/ticker/{sym}/range/1/day/{from}/{to}
  - Ticker details (sector): /v3/reference/tickers/{sym}
  - Snapshot (live quote)  : /v2/snapshot/locale/us/markets/stocks/tickers/{sym}

API key is optional — unauthenticated requests are rate-limited to 5/min
but sufficient as a yfinance fallback.  Set POLYGON_API_KEY env var or
pass api_key= to PolygonClient() for higher limits.
"""

import os
import logging
import time
from datetime import date, timedelta
from typing import Optional

import requests
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

_BASE = "https://api.polygon.io"
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "AlphaAgent/1.0"})

_RATE_DELAY = 12.5   # seconds between unauthenticated calls (5/min limit)
_LAST_CALL: float = 0.0


def _get(path: str, params: dict, api_key: str = "") -> Optional[dict]:
    global _LAST_CALL
    if api_key:
        params["apiKey"] = api_key
    else:
        elapsed = time.time() - _LAST_CALL
        if elapsed < _RATE_DELAY:
            time.sleep(_RATE_DELAY - elapsed)
    try:
        r = _SESSION.get(f"{_BASE}{path}", params=params, timeout=10)
        _LAST_CALL = time.time()
        if r.status_code == 200:
            return r.json()
        logger.debug(f"[Polygon] {r.status_code} for {path}")
    except Exception as e:
        logger.debug(f"[Polygon] request error: {e}")
    return None


class PolygonClient:
    """
    Lightweight Polygon.io client used as a fallback when yfinance is rate-limited.
    All methods return None on failure — callers must handle gracefully.
    """

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY", "")

    def _q(self, path: str, params: dict = None) -> Optional[dict]:
        return _get(path, params or {}, self.api_key)

    # ── OHLCV bars ────────────────────────────────────────────────────────

    def get_daily_ohlcv(
        self, ticker: str,
        from_date: Optional[date] = None,
        to_date:   Optional[date] = None,
        days: int = 365,
    ) -> Optional[pd.DataFrame]:
        """Return daily OHLCV DataFrame or None."""
        to_date   = to_date   or date.today()
        from_date = from_date or (to_date - timedelta(days=days))
        path = (f"/v2/aggs/ticker/{ticker.upper()}/range/1/day"
                f"/{from_date.isoformat()}/{to_date.isoformat()}")
        data = self._q(path, {"adjusted": "true", "sort": "asc", "limit": 5000})
        if not data or not data.get("results"):
            return None
        rows = data["results"]
        df = pd.DataFrame({
            "Open":   [r["o"] for r in rows],
            "High":   [r["h"] for r in rows],
            "Low":    [r["l"] for r in rows],
            "Close":  [r["c"] for r in rows],
            "Volume": [r["v"] for r in rows],
        }, index=pd.to_datetime([r["t"] for r in rows], unit="ms", utc=True))
        df.index.name = "Date"
        logger.info(f"[Polygon] {ticker} OHLCV: {len(df)} bars")
        return df

    def get_prev_close(self, ticker: str) -> Optional[dict]:
        """Return {open, high, low, close, volume, vwap} for previous session."""
        data = self._q(f"/v2/aggs/ticker/{ticker.upper()}/prev",
                       {"adjusted": "true"})
        if not data or not data.get("results"):
            return None
        r = data["results"][0]
        return {
            "open":   r.get("o"),
            "high":   r.get("h"),
            "low":    r.get("l"),
            "close":  r.get("c"),
            "volume": r.get("v"),
            "vwap":   r.get("vw"),
        }

    # ── Live snapshot ─────────────────────────────────────────────────────

    def get_snapshot(self, ticker: str) -> Optional[dict]:
        """Return latest quote snapshot (requires API key for real-time)."""
        data = self._q(
            f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker.upper()}"
        )
        if not data:
            return None
        snap = data.get("ticker", {})
        day  = snap.get("day", {})
        prev = snap.get("prevDay", {})
        last = snap.get("lastTrade", {})
        return {
            "price":      last.get("p") or day.get("c"),
            "open":       day.get("o"),
            "high":       day.get("h"),
            "low":        day.get("l"),
            "close":      day.get("c"),
            "volume":     day.get("v"),
            "vwap":       day.get("vw"),
            "prev_close": prev.get("c"),
            "change":     (day.get("c", 0) - prev.get("c", 0)) if prev.get("c") else None,
            "change_pct": snap.get("todaysChangePerc"),
        }

    # ── Ticker details (sector / industry) ───────────────────────────────

    def get_ticker_details(self, ticker: str) -> Optional[dict]:
        data = self._q(f"/v3/reference/tickers/{ticker.upper()}")
        if not data:
            return None
        r = data.get("results", {})
        return {
            "name":        r.get("name"),
            "sector":      r.get("sic_description"),
            "market_cap":  r.get("market_cap"),
            "shares_out":  r.get("share_class_shares_outstanding"),
            "description": r.get("description"),
            "homepage":    r.get("homepage_url"),
            "exchange":    r.get("primary_exchange"),
            "currency":    r.get("currency_name"),
        }

    # ── Intraday 1-min bars ───────────────────────────────────────────────

    def get_intraday_bars(
        self, ticker: str,
        session_date: Optional[date] = None,
        multiplier: int = 1,
        timespan: str = "minute",
    ) -> Optional[pd.DataFrame]:
        """Return 1-min bars for a given session date (default today)."""
        d = session_date or date.today()
        path = (f"/v2/aggs/ticker/{ticker.upper()}/range/{multiplier}"
                f"/{timespan}/{d.isoformat()}/{d.isoformat()}")
        data = self._q(path, {"adjusted": "true", "sort": "asc", "limit": 50000})
        if not data or not data.get("results"):
            return None
        rows = data["results"]
        df = pd.DataFrame({
            "Open":   [r["o"] for r in rows],
            "High":   [r["h"] for r in rows],
            "Low":    [r["l"] for r in rows],
            "Close":  [r["c"] for r in rows],
            "Volume": [r["v"] for r in rows],
        }, index=pd.to_datetime([r["t"] for r in rows], unit="ms", utc=True))
        df.index.name = "Date"
        return df


# ── Module-level convenience singleton ───────────────────────────────────────

_client: Optional[PolygonClient] = None


def get_client() -> PolygonClient:
    global _client
    if _client is None:
        _client = PolygonClient()
    return _client


def ohlcv_fallback(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """
    Drop-in fallback: called when yfinance returns empty OHLCV.
    Maps yfinance period strings to Polygon day counts.
    """
    days_map = {
        "5d": 7, "1mo": 35, "3mo": 100, "6mo": 190,
        "1y": 370, "2y": 740, "3y": 1100, "5y": 1830,
    }
    days = days_map.get(period, 370)
    try:
        return get_client().get_daily_ohlcv(ticker, days=days)
    except Exception as e:
        logger.debug(f"[Polygon] ohlcv_fallback failed for {ticker}: {e}")
        return None


def quote_fallback(ticker: str) -> Optional[dict]:
    """Drop-in fallback for live price when yfinance fast_info fails."""
    try:
        snap = get_client().get_snapshot(ticker)
        if snap:
            return snap
        prev = get_client().get_prev_close(ticker)
        if prev:
            return {"price": prev["close"], **prev}
    except Exception as e:
        logger.debug(f"[Polygon] quote_fallback failed for {ticker}: {e}")
    return None
