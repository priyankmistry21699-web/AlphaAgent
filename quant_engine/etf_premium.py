"""
AlphaAgent — ETF Premium/Discount to NAV Signal

Detects when an ETF trades meaningfully above or below its NAV.
Mean-reversion signal with known convergence timeline (next trading day).

For commodity ETFs (USO, GLD, SLV, UNG) the gap also reveals supply/demand
pressure on the underlying creation/redemption process.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ETFPremiumResult:
    ticker: str
    price: float
    nav: float
    premium_pct: float            # (price/nav - 1) × 100
    z_score: float                # premium vs 60d distribution
    signal: str                   # "BUY_DISCOUNT" / "SELL_PREMIUM" / "NEUTRAL"
    confidence: float


# Approximate NAV proxy by tracked underlying (use as nav substitute when
# real iNAV is not available from yfinance)
ETF_NAV_PROXY = {
    "SPY":  "^GSPC",     # S&P 500
    "QQQ":  "^NDX",      # Nasdaq 100
    "IWM":  "^RUT",      # Russell 2000
    "DIA":  "^DJI",      # Dow
    "GLD":  "GC=F",      # Gold futures
    "SLV":  "SI=F",      # Silver futures
    "USO":  "CL=F",      # Crude oil
    "UNG":  "NG=F",      # Natural gas
    "TLT":  "^TYX",      # 30Y yield (inverse proxy)
    "AGG":  "^TNX",      # 10Y yield
}


def etf_premium_discount(etf_ticker: str,
                          lookback_days: int = 60
                          ) -> Optional[ETFPremiumResult]:
    """
    Compute ETF price vs NAV-proxy premium/discount and z-score it
    against the lookback window distribution.

    Returns ETFPremiumResult with mean-reversion signal classification.
    """
    try:
        import yfinance as yf
        proxy = ETF_NAV_PROXY.get(etf_ticker.upper())
        if not proxy:
            return None

        df_etf = yf.download(etf_ticker, period="6mo", interval="1d",
                              auto_adjust=True, progress=False)
        df_nav = yf.download(proxy, period="6mo", interval="1d",
                              auto_adjust=True, progress=False)
        if df_etf.empty or df_nav.empty:
            return None

        s_etf = df_etf["Close"].squeeze().dropna()
        s_nav = df_nav["Close"].squeeze().dropna()
        df = pd.concat([s_etf, s_nav], axis=1, join="inner").dropna()
        if len(df) < 30:
            return None
        df.columns = ["etf", "nav"]

        # Premium = (ETF / NAV) - long-run ratio
        ratio = df["etf"] / df["nav"]
        # Use rolling baseline as 'fair' ratio
        baseline = ratio.rolling(lookback_days, min_periods=20).median()
        premium_series = (ratio / baseline - 1) * 100

        current_premium = float(premium_series.iloc[-1])
        std = float(premium_series.tail(lookback_days).std())
        z = current_premium / std if std > 0 else 0.0

        # Classification
        if z > 1.5:
            signal = "SELL_PREMIUM"
        elif z < -1.5:
            signal = "BUY_DISCOUNT"
        else:
            signal = "NEUTRAL"

        confidence = min(1.0, abs(z) / 3.0)

        return ETFPremiumResult(
            ticker=etf_ticker.upper(),
            price=float(df["etf"].iloc[-1]),
            nav=float(df["nav"].iloc[-1]),
            premium_pct=round(current_premium, 3),
            z_score=round(z, 2),
            signal=signal,
            confidence=round(confidence, 3),
        )
    except Exception as e:
        logger.warning(f"ETF premium failed for {etf_ticker}: {e}")
        return None


def scan_etf_universe(tickers: Optional[list] = None) -> dict:
    """Scan a universe of ETFs for premium/discount signals."""
    universe = tickers or list(ETF_NAV_PROXY.keys())
    out = {}
    for t in universe:
        r = etf_premium_discount(t)
        if r is not None:
            out[t] = r
    return out
