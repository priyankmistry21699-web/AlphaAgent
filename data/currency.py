"""
AlphaAgent — Currency Data Layer

Fetches FX rates and DXY (US Dollar Index) data via yfinance.

Tickers used:
  DX-Y.NYB  — US Dollar Index (USDX)
  EURUSD=X  — EUR/USD
  JPY=X     — USD/JPY (quoted as USD per JPY, but yfinance returns units correctly)
  CNY=X     — USD/CNY
  GBP=X     — USD/GBP
  UUP       — Invesco DB USD Index (DXY ETF proxy, more liquid)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

from data.cache import DataCache

logger = logging.getLogger(__name__)


@dataclass
class CurrencySnapshot:
    """Latest FX rates and derived metrics."""
    dxy: float = 0.0                  # Dollar Index level
    dxy_1m_change: float = 0.0        # 1-month % change in DXY
    dxy_3m_change: float = 0.0
    eurusd: float = 0.0
    usdjpy: float = 0.0
    usdcny: float = 0.0
    gbpusd: float = 0.0
    em_currency_stress: float = 0.0   # Average USD strength vs EM proxies
    warnings: list = field(default_factory=list)


class CurrencyData:
    """
    Fetches and caches currency / FX data from yfinance.
    """

    CURRENCY_TTL = 3600  # 1 hour

    FX_TICKERS = {
        "dxy":    "DX-Y.NYB",
        "eurusd": "EURUSD=X",
        "usdjpy": "JPY=X",
        "usdcny": "CNY=X",
        "gbpusd": "GBPUSD=X",
    }

    # EM currency proxies (ETFs for liquidity)
    EM_TICKERS = {
        "usd_mxn": "MXN=X",
        "usd_brl": "BRL=X",
        "usd_inr": "INR=X",
        "usd_zar": "ZAR=X",
    }

    def __init__(self, cache: Optional[DataCache] = None):
        self.cache = cache or DataCache()

    def _fetch_series(self, ticker: str, period: str = "3mo") -> pd.Series:
        """Fetch closing price series for a yfinance ticker."""
        try:
            data = yf.download(ticker, period=period, interval="1d",
                               auto_adjust=True, progress=False)
            if data.empty:
                return pd.Series(dtype=float)
            return data["Close"].squeeze().dropna()
        except Exception as e:
            logger.warning(f"[CurrencyData] Failed to fetch {ticker}: {e}")
            return pd.Series(dtype=float)

    def get_snapshot(self) -> CurrencySnapshot:
        """Fetch the latest FX snapshot with change metrics."""
        cache_key = "currency_snapshot"
        cached = self.cache.get("currency", "FX", cache_key)
        if cached is not None:
            return CurrencySnapshot(**cached)

        snap = CurrencySnapshot()

        # DXY — fallback to UUP ETF if DX-Y.NYB returns garbage
        dxy_series = self._fetch_series("DX-Y.NYB", period="6mo")
        dxy_val = float(dxy_series.iloc[-1]) if len(dxy_series) >= 2 else 0.0
        if not (70.0 <= dxy_val <= 130.0):
            dxy_series = self._fetch_series("UUP", period="6mo")
            dxy_val = float(dxy_series.iloc[-1]) if len(dxy_series) >= 2 else 0.0
            if not (20.0 <= dxy_val <= 40.0):
                dxy_series = pd.Series(dtype=float)
        if len(dxy_series) >= 2:
            snap.dxy = float(dxy_series.iloc[-1])
            snap.dxy_1m_change = float(
                (dxy_series.iloc[-1] / dxy_series.iloc[-22] - 1) * 100
                if len(dxy_series) >= 22 else 0.0
            )
            snap.dxy_3m_change = float(
                (dxy_series.iloc[-1] / dxy_series.iloc[-66] - 1) * 100
                if len(dxy_series) >= 66 else 0.0
            )
        else:
            snap.warnings.append("DXY data unavailable")

        # Individual pairs — with sanity bounds to reject corrupted yfinance data
        _FX_BOUNDS = {
            "eurusd":  (0.70, 1.60),
            "usdjpy":  (80.0, 200.0),
            "usdcny":  (5.50, 9.00),
            "gbpusd":  (0.90, 1.70),
        }
        for attr, ticker in [
            ("eurusd", "EURUSD=X"),
            ("usdjpy", "USDJPY=X"),
            ("usdcny", "USDCNY=X"),
            ("gbpusd", "GBPUSD=X"),
        ]:
            s = self._fetch_series(ticker, period="5d")
            if not s.empty:
                val = float(s.iloc[-1])
                lo, hi = _FX_BOUNDS.get(attr, (0.0, 1e9))
                if lo <= val <= hi:
                    setattr(snap, attr, val)
                else:
                    snap.warnings.append(f"{ticker} value {val:.4f} out of expected range [{lo},{hi}] — skipped")

        # EM currency stress: average 1-month change in USD vs EM FX
        em_changes = []
        for name, ticker in self.EM_TICKERS.items():
            s = self._fetch_series(ticker, period="2mo")
            if len(s) >= 22:
                chg = (s.iloc[-1] / s.iloc[-22] - 1) * 100
                em_changes.append(float(chg))
        snap.em_currency_stress = float(sum(em_changes) / len(em_changes)) if em_changes else 0.0

        # Cache
        try:
            self.cache.set(
                "currency", "FX", cache_key,
                {
                    "dxy": snap.dxy,
                    "dxy_1m_change": snap.dxy_1m_change,
                    "dxy_3m_change": snap.dxy_3m_change,
                    "eurusd": snap.eurusd,
                    "usdjpy": snap.usdjpy,
                    "usdcny": snap.usdcny,
                    "gbpusd": snap.gbpusd,
                    "em_currency_stress": snap.em_currency_stress,
                    "warnings": snap.warnings,
                },
                ttl_seconds=self.CURRENCY_TTL,
            )
        except Exception:
            pass

        return snap

    def get_dxy_regime(self) -> str:
        """
        Classify the USD regime based on 3-month momentum.
          STRONG_USD  : DXY 3m change > +2%
          WEAK_USD    : DXY 3m change < -2%
          NEUTRAL     : otherwise
        """
        snap = self.get_snapshot()
        if snap.dxy_3m_change > 2.0:
            return "STRONG_USD"
        if snap.dxy_3m_change < -2.0:
            return "WEAK_USD"
        return "NEUTRAL_USD"
