"""
AlphaAgent — Alternative Data Layer

Computes alternative market sentiment signals:
  1. Fear & Greed approximation (VIX + momentum + safe-haven flows)
  2. Copper / Gold ratio (industrial demand signal)
  3. Market breadth (SPY vs equal-weight RSP)
  4. Baltic Dry proxy (BDRY shipping ETF)
  5. Junk bond spread proxy (HYG vs LQD)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from data.cache import DataCache

logger = logging.getLogger(__name__)


@dataclass
class AlternativeSnapshot:
    """Computed alternative market signals."""
    fear_greed_score: float = 50.0       # 0=max fear, 100=max greed
    fear_greed_label: str = "Neutral"
    copper_gold_ratio: float = 0.0
    copper_gold_1m_change: float = 0.0   # % change (positive = industrial demand rising)
    breadth_score: float = 50.0          # Market breadth 0–100
    junk_spread_pct: float = 0.0         # HYG/LQD ratio 1m change (negative = stress)
    bdi_proxy_change: float = 0.0        # BDRY 1-month change %
    warnings: list = field(default_factory=list)


class AlternativeData:
    """
    Computes alternative sentiment signals from liquid ETF prices.
    No special API keys required — all via yfinance.
    """

    ALT_TTL = 3600 * 4  # 4 hours

    def __init__(self, cache: Optional[DataCache] = None):
        self.cache = cache or DataCache()

    def _fetch(self, ticker: str, period: str = "3mo") -> pd.Series:
        try:
            df = yf.download(ticker, period=period, interval="1d",
                             auto_adjust=True, progress=False)
            if df.empty:
                return pd.Series(dtype=float)
            return df["Close"].squeeze().dropna()
        except Exception as e:
            logger.warning(f"[AltData] {ticker}: {e}")
            return pd.Series(dtype=float)

    def _safe_pct_change(self, s: pd.Series, lookback: int) -> float:
        if len(s) >= lookback + 1:
            return float((s.iloc[-1] / s.iloc[-lookback - 1] - 1) * 100)
        return 0.0

    # ── Fear & Greed ──────────────────────────────────────────────────────
    def compute_fear_greed(self) -> tuple[float, str]:
        """
        Approximates the CNN Fear & Greed Index from 5 sub-components:
          1. VIX level  (low VIX = greed)
          2. SPY 125d momentum  (rising = greed)
          3. SPY vs 125d MA (Junk Spread sub)
          4. HYG vs LQD (credit risk appetite)
          5. GLD vs SPY safe haven demand
        """
        scores = []

        # 1. VIX (VIXCLS via FRED is slow; use ^VIX via yfinance)
        vix = self._fetch("^VIX", period="1y")
        if len(vix) > 10:
            vix_now = float(vix.iloc[-1])
            # Map: VIX 10->85, VIX 40->10 (linear)
            vix_score = max(10.0, min(90.0, 95.0 - (vix_now - 10) * 2.1))
            scores.append(vix_score)

        # 2. SPY 125-day momentum
        spy = self._fetch("SPY", period="1y")
        if len(spy) > 130:
            spy_mom = (spy.iloc[-1] / spy.iloc[-126] - 1) * 100
            # Map: -20%->10, +20%->90
            mom_score = max(10.0, min(90.0, 50.0 + spy_mom * 2.0))
            scores.append(mom_score)

        # 3. SPY vs 125d MA
        if len(spy) > 130:
            ma125 = float(spy.iloc[-126:].mean())
            above_ma = spy.iloc[-1] > ma125
            scores.append(70.0 if above_ma else 30.0)

        # 4. Junk bond demand: HYG vs LQD ratio change
        hyg = self._fetch("HYG", period="3mo")
        lqd = self._fetch("LQD", period="3mo")
        if len(hyg) > 22 and len(lqd) > 22:
            ratio_now = float(hyg.iloc[-1] / lqd.iloc[-1])
            ratio_prev = float(hyg.iloc[-22] / lqd.iloc[-22])
            ratio_chg = (ratio_now / ratio_prev - 1) * 100
            junk_score = max(10.0, min(90.0, 50.0 + ratio_chg * 10.0))
            scores.append(junk_score)

        # 5. Gold vs SPY (safe haven: high GLD/SPY = fear)
        gld = self._fetch("GLD", period="3mo")
        if len(gld) > 22 and len(spy) > 22:
            gs_now = float(gld.iloc[-1] / spy.iloc[-1])
            gs_prev = float(gld.iloc[-22] / spy.iloc[-22])
            gs_chg = (gs_now / gs_prev - 1) * 100
            # Rising gold relative to SPY = fear
            haven_score = max(10.0, min(90.0, 50.0 - gs_chg * 5.0))
            scores.append(haven_score)

        if not scores:
            return 50.0, "Neutral"

        fg = float(np.mean(scores))

        if fg >= 75:
            label = "Extreme Greed"
        elif fg >= 60:
            label = "Greed"
        elif fg <= 25:
            label = "Extreme Fear"
        elif fg <= 40:
            label = "Fear"
        else:
            label = "Neutral"

        return fg, label

    # ── Copper / Gold Ratio ───────────────────────────────────────────────
    def compute_copper_gold(self) -> tuple[float, float]:
        """
        Copper/Gold ratio: rising = industrial demand (risk-on / bullish).
        Uses COPX (copper miner ETF) and GLD.
        Returns (ratio, 1m_change_pct).
        """
        copper = self._fetch("COPX", period="3mo")
        gold = self._fetch("GLD", period="3mo")

        if copper.empty or gold.empty:
            return 0.0, 0.0

        ratio_now = float(copper.iloc[-1] / gold.iloc[-1])
        ratio_prev = float(copper.iloc[-22] / gold.iloc[-22]) if len(copper) > 22 else ratio_now
        change_pct = (ratio_now / ratio_prev - 1) * 100 if ratio_prev != 0 else 0.0
        return ratio_now, change_pct

    # ── Market Breadth ────────────────────────────────────────────────────
    def compute_breadth(self) -> float:
        """
        Market breadth via SPY (cap-weighted) vs RSP (equal-weighted).
        If RSP > SPY on 1-month basis: broad rally (score 70+).
        If SPY > RSP: narrow/mega-cap-driven rally (score <50, warning).
        """
        spy = self._fetch("SPY", period="3mo")
        rsp = self._fetch("RSP", period="3mo")

        if len(spy) < 22 or len(rsp) < 22:
            return 50.0

        spy_chg = self._safe_pct_change(spy, 22)
        rsp_chg = self._safe_pct_change(rsp, 22)
        diff = rsp_chg - spy_chg  # positive = broad market participation

        # Map -5% to +5% difference onto 10–90 score
        breadth = max(10.0, min(90.0, 50.0 + diff * 8.0))
        return breadth

    # ── Junk Bond Spread ─────────────────────────────────────────────────
    def compute_junk_spread(self) -> float:
        """
        HYG/LQD ratio 1-month change as credit stress proxy.
        Negative = credit stress (spreads widening).
        """
        hyg = self._fetch("HYG", period="3mo")
        lqd = self._fetch("LQD", period="3mo")
        if len(hyg) < 22 or len(lqd) < 22:
            return 0.0
        r_now = float(hyg.iloc[-1] / lqd.iloc[-1])
        r_prev = float(hyg.iloc[-22] / lqd.iloc[-22])
        return (r_now / r_prev - 1) * 100 if r_prev != 0 else 0.0

    # ── BDI Proxy ─────────────────────────────────────────────────────────
    def compute_bdi_proxy(self) -> float:
        """Baltic Dry Index proxy via BDRY (Breakwave Dry Bulk Shipping ETF)."""
        bdry = self._fetch("BDRY", period="3mo")
        return self._safe_pct_change(bdry, 22)

    # ── Full Snapshot ─────────────────────────────────────────────────────
    def get_snapshot(self) -> AlternativeSnapshot:
        cache_key = "alt_snapshot"
        cached = self.cache.get("alternative", "ALT", cache_key)
        if cached is not None:
            return AlternativeSnapshot(**cached)

        snap = AlternativeSnapshot()
        snap.fear_greed_score, snap.fear_greed_label = self.compute_fear_greed()
        snap.copper_gold_ratio, snap.copper_gold_1m_change = self.compute_copper_gold()
        snap.breadth_score = self.compute_breadth()
        snap.junk_spread_pct = self.compute_junk_spread()
        snap.bdi_proxy_change = self.compute_bdi_proxy()

        try:
            self.cache.set(
                "alternative", "ALT", cache_key,
                {
                    "fear_greed_score": snap.fear_greed_score,
                    "fear_greed_label": snap.fear_greed_label,
                    "copper_gold_ratio": snap.copper_gold_ratio,
                    "copper_gold_1m_change": snap.copper_gold_1m_change,
                    "breadth_score": snap.breadth_score,
                    "junk_spread_pct": snap.junk_spread_pct,
                    "bdi_proxy_change": snap.bdi_proxy_change,
                    "warnings": snap.warnings,
                },
                ttl_seconds=self.ALT_TTL,
            )
        except Exception:
            pass

        return snap
