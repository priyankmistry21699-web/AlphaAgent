"""
AlphaAgent — Factor Exposure Analysis

Computes portfolio-level and per-position exposure to the five canonical
Fama-French / AQR style factors:

  Value      — P/B, P/E, EV/EBITDA (low multiples = high value loading)
  Momentum   — 12M-1M JT momentum (Jegadeesh-Titman)
  Quality    — ROE, profit margin, low leverage (high = quality tilt)
  Size       — log(market cap) — large cap has negative size factor loading
  Low-Vol    — beta to SPY and realized volatility (low = defensive tilt)

Portfolio factor scores are market-value-weighted averages.
Each score is normalised to a [-1, +1] range where:
  +1 = strong positive factor tilt
   0 = market-neutral
  -1 = strong negative factor tilt (factor-adverse)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class PositionFactors:
    ticker: str
    value_score: float       # +1 cheap, -1 expensive
    momentum_score: float    # +1 strong momentum, -1 reversal
    quality_score: float     # +1 high quality, -1 low quality
    size_score: float        # +1 small-cap, -1 large-cap
    low_vol_score: float     # +1 low vol/beta, -1 high vol/beta
    weight: float = 0.0      # portfolio allocation weight


@dataclass
class PortfolioFactorExposure:
    value: float             # aggregate portfolio factor scores
    momentum: float
    quality: float
    size: float
    low_vol: float
    positions: List[PositionFactors] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class FactorExposureAnalyzer:
    """
    Computes style-factor exposures for a list of portfolio holdings.

    Usage:
        analyzer = FactorExposureAnalyzer()
        exposure = analyzer.analyze(holdings)
    """

    LOOKBACK = "1y"

    def analyze(self, holdings: List[Dict]) -> PortfolioFactorExposure:
        """
        Args:
            holdings: list of dicts with keys: ticker, market_value.
        """
        if not holdings:
            return PortfolioFactorExposure(0.0, 0.0, 0.0, 0.0, 0.0,
                                           warnings=["Portfolio is empty."])

        total_value = sum(h.get("market_value", 0.0) for h in holdings)
        tickers = [h["ticker"] for h in holdings]

        # Batch-download 1-year prices for momentum and low-vol
        prices_df = self._fetch_prices(tickers)
        spy_ret = self._fetch_spy_returns()

        position_factors: List[PositionFactors] = []
        warnings = []

        for h in holdings:
            ticker = h["ticker"]
            weight = h.get("market_value", 0.0) / total_value if total_value > 0 else 0.0
            try:
                pf = self._score_position(ticker, weight, prices_df, spy_ret)
                position_factors.append(pf)
            except Exception as e:
                logger.warning(f"[FactorExposure] {ticker} scoring failed: {e}")
                position_factors.append(PositionFactors(
                    ticker=ticker, weight=weight,
                    value_score=0.0, momentum_score=0.0, quality_score=0.0,
                    size_score=0.0, low_vol_score=0.0,
                ))

        # ── Portfolio-level weighted average ──────────────────────────────────
        def wt_avg(attr: str) -> float:
            wts = np.array([p.weight for p in position_factors])
            vals = np.array([getattr(p, attr) for p in position_factors])
            return round(float(np.dot(wts, vals)), 3) if wts.sum() > 0 else 0.0

        port_value    = wt_avg("value_score")
        port_momentum = wt_avg("momentum_score")
        port_quality  = wt_avg("quality_score")
        port_size     = wt_avg("size_score")
        port_low_vol  = wt_avg("low_vol_score")

        # Diagnostic warnings
        if abs(port_value) > 0.6:
            bias = "value" if port_value > 0 else "growth"
            warnings.append(f"Portfolio has a strong {bias} tilt (score={port_value:.2f}).")
        if abs(port_momentum) > 0.6:
            bias = "momentum" if port_momentum > 0 else "reversal"
            warnings.append(f"Portfolio has a strong {bias} bias (score={port_momentum:.2f}).")
        if port_low_vol < -0.5:
            warnings.append(f"Portfolio is high-beta/high-vol (low_vol_score={port_low_vol:.2f}). Consider hedging.")

        return PortfolioFactorExposure(
            value=port_value,
            momentum=port_momentum,
            quality=port_quality,
            size=port_size,
            low_vol=port_low_vol,
            positions=position_factors,
            warnings=warnings,
        )

    def _score_position(
        self,
        ticker: str,
        weight: float,
        prices_df: Optional[pd.DataFrame],
        spy_ret: Optional[pd.Series],
    ) -> PositionFactors:
        info = yf.Ticker(ticker).info

        # ── VALUE FACTOR ──────────────────────────────────────────────────────
        pe = info.get("trailingPE") or info.get("forwardPE")
        pb = info.get("priceToBook")
        ev_ebitda = info.get("enterpriseToEbitda")
        value_score = self._score_value(pe, pb, ev_ebitda)

        # ── MOMENTUM FACTOR (12M-1M) ──────────────────────────────────────────
        momentum_score = 0.0
        if prices_df is not None and ticker in prices_df.columns:
            p = prices_df[ticker].dropna()
            if len(p) >= 252:
                ret_12m = float(p.iloc[-1] / p.iloc[-252] - 1)
                ret_1m = float(p.iloc[-1] / p.iloc[-21] - 1)
                jt = ret_12m - ret_1m
                momentum_score = float(np.clip(jt / 0.30, -1.0, 1.0))  # normalise to ±30% range

        # ── QUALITY FACTOR ─────────────────────────────────────────────────────
        roe = info.get("returnOnEquity")
        profit_margin = info.get("profitMargins")
        debt_equity = info.get("debtToEquity")
        quality_score = self._score_quality(roe, profit_margin, debt_equity)

        # ── SIZE FACTOR ────────────────────────────────────────────────────────
        mkt_cap = info.get("marketCap") or 0
        size_score = self._score_size(mkt_cap)

        # ── LOW-VOLATILITY FACTOR ─────────────────────────────────────────────
        low_vol_score = 0.0
        beta = info.get("beta")
        if beta and isinstance(beta, (int, float)) and 0 < beta < 5:
            low_vol_score = float(np.clip(-(beta - 1.0) / 1.5, -1.0, 1.0))
        elif prices_df is not None and ticker in prices_df.columns and spy_ret is not None:
            p = prices_df[ticker].dropna()
            if len(p) >= 60:
                ret = p.pct_change().dropna()
                aligned = ret.align(spy_ret, join="inner")
                r, s = aligned[0].dropna(), aligned[1].reindex(aligned[0].index).dropna()
                r = r.reindex(s.index)
                if len(r) >= 30 and s.var() > 0:
                    beta_est = float(np.cov(r, s)[0, 1] / s.var())
                    low_vol_score = float(np.clip(-(beta_est - 1.0) / 1.5, -1.0, 1.0))

        return PositionFactors(
            ticker=ticker,
            weight=round(weight, 4),
            value_score=round(value_score, 3),
            momentum_score=round(momentum_score, 3),
            quality_score=round(quality_score, 3),
            size_score=round(size_score, 3),
            low_vol_score=round(low_vol_score, 3),
        )

    @staticmethod
    def _score_value(
        pe: Optional[float],
        pb: Optional[float],
        ev_ebitda: Optional[float],
    ) -> float:
        scores = []
        if pe and pe > 0:
            if pe < 12:   scores.append(1.0)
            elif pe < 18: scores.append(0.5)
            elif pe < 25: scores.append(0.0)
            elif pe < 35: scores.append(-0.5)
            else:         scores.append(-1.0)
        if pb and pb > 0:
            if pb < 1.5:  scores.append(1.0)
            elif pb < 3:  scores.append(0.3)
            elif pb < 5:  scores.append(-0.3)
            else:         scores.append(-1.0)
        if ev_ebitda and ev_ebitda > 0:
            if ev_ebitda < 8:   scores.append(1.0)
            elif ev_ebitda < 15: scores.append(0.3)
            elif ev_ebitda < 25: scores.append(-0.3)
            else:                scores.append(-1.0)
        return float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def _score_quality(
        roe: Optional[float],
        margin: Optional[float],
        debt_eq: Optional[float],
    ) -> float:
        scores = []
        if roe is not None:
            if roe > 0.25:   scores.append(1.0)
            elif roe > 0.15: scores.append(0.5)
            elif roe > 0.05: scores.append(0.0)
            elif roe > 0:    scores.append(-0.5)
            else:            scores.append(-1.0)
        if margin is not None:
            if margin > 0.25:   scores.append(1.0)
            elif margin > 0.15: scores.append(0.5)
            elif margin > 0.05: scores.append(0.0)
            else:               scores.append(-0.5)
        if debt_eq is not None:
            if debt_eq < 20:   scores.append(1.0)
            elif debt_eq < 50: scores.append(0.5)
            elif debt_eq < 100: scores.append(-0.3)
            else:               scores.append(-1.0)
        return float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def _score_size(mkt_cap: float) -> float:
        """Small-cap = +1, mega-cap = -1."""
        if mkt_cap <= 0:
            return 0.0
        log_cap = np.log10(mkt_cap)
        # $300M ≈ 8.5, $10B ≈ 10, $1T ≈ 12
        score = -(log_cap - 10.0) / 1.5   # centres at $10B, ±1.5 orders of magnitude
        return float(np.clip(score, -1.0, 1.0))

    def _fetch_prices(self, tickers: List[str]) -> Optional[pd.DataFrame]:
        try:
            raw = yf.download(tickers, period=self.LOOKBACK, interval="1d",
                              auto_adjust=True, progress=False, threads=True)
            if raw.empty:
                return None
            if isinstance(raw.columns, pd.MultiIndex):
                return raw["Close"]
            return raw[["Close"]] if "Close" in raw.columns else raw
        except Exception as e:
            logger.warning(f"[FactorExposure] Price download failed: {e}")
            return None

    def _fetch_spy_returns(self) -> Optional[pd.Series]:
        try:
            spy = yf.download("SPY", period=self.LOOKBACK, interval="1d",
                              auto_adjust=True, progress=False)
            if spy.empty:
                return None
            return spy["Close"].squeeze().pct_change().dropna()
        except Exception as e:
            logger.warning(f"[FactorExposure] SPY download failed: {e}")
            return None
