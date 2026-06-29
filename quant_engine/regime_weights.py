"""
AlphaAgent — Regime-Conditional Agent Weights

Theory:
  Different agents have radically different IC (Information Coefficient)
  in different market regimes.  Technical momentum works in trending
  bull markets; Macro/Fundamental dominates in bear/crisis; Insider is
  regime-agnostic.  Using the same weights regardless of regime dilutes
  the strongest signal.

Regimes (4):
  BULL_TREND   : SPY > 200d SMA, VIX < 20, realised vol < 15%
  BULL_CHOPPY  : SPY near 200d, VIX 15-25, sideways price action
  BEAR         : SPY < 200d SMA, VIX 20-35
  CRISIS       : VIX > 35 or SPY –10% in 20d

Regime-conditional weights (rows = regime, cols = agent):
  Based on empirical academic literature + common quant-fund practice.
  Weights are normalised per regime (sum to 1, risk agent always 0).

Usage:
  from quant_engine.regime_weights import get_regime_weights, detect_regime
  regime = detect_regime()
  weights = get_regime_weights(regime)
"""

import logging
import time
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    BULL_TREND  = "BULL_TREND"
    BULL_CHOPPY = "BULL_CHOPPY"
    BEAR        = "BEAR"
    CRISIS      = "CRISIS"


# ── Regime-conditional weight table ─────────────────────────────────────────
# Rows   : MarketRegime
# Columns: agent name (must match orchestrator keys)
# Values : un-normalised weights (normalised in get_regime_weights)
#
# Rationale:
#  - BULL_TREND  : Technical momentum, Sentiment, Insider dominant
#  - BULL_CHOPPY : Fundamental value, Macro conditions, Insider
#  - BEAR        : Macro, Fundamental quality, Risk signals matter most
#  - CRISIS      : Macro + Geopolitical dominate; Technical unreliable
#
_RAW_WEIGHTS: Dict[MarketRegime, Dict[str, float]] = {
    MarketRegime.BULL_TREND: {
        "technical":    0.32,
        "fundamental":  0.05,   # poor short-horizon predictor; empirical IC ~18%
        "sentiment":    0.20,
        "macro":        0.12,
        "currency":     0.08,
        "geopolitical": 0.06,
        "insider":      0.15,
        "volatility":   0.02,
    },
    MarketRegime.BULL_CHOPPY: {
        "technical":    0.20,
        "fundamental":  0.07,   # poor short-horizon predictor; empirical IC ~18%
        "sentiment":    0.13,
        "macro":        0.24,
        "currency":     0.08,
        "geopolitical": 0.07,
        "insider":      0.18,
        "volatility":   0.03,
    },
    MarketRegime.BEAR: {
        "technical":    0.10,
        "fundamental":  0.20,
        "sentiment":    0.08,
        "macro":        0.26,
        "currency":     0.10,
        "geopolitical": 0.10,
        "insider":      0.12,
        "volatility":   0.04,
    },
    MarketRegime.CRISIS: {
        "technical":    0.06,
        "fundamental":  0.10,
        "sentiment":    0.06,
        "macro":        0.28,
        "currency":     0.12,
        "geopolitical": 0.22,
        "insider":      0.10,
        "volatility":   0.06,
    },
}

# Pre-normalise
_WEIGHTS: Dict[MarketRegime, Dict[str, float]] = {}
for _regime, _raw in _RAW_WEIGHTS.items():
    _total = sum(_raw.values())
    _WEIGHTS[_regime] = {k: round(v / _total, 4) for k, v in _raw.items()}


# ── Regime correlation penalties ────────────────────────────────────────────
# How much to de-correlate overlapping agents per regime
_CORRELATION_MAP: Dict[MarketRegime, Dict[str, float]] = {
    MarketRegime.BULL_TREND: {
        "sentiment":    0.15,
        "volatility":   0.35,
        "geopolitical": 0.20,
        "currency":     0.15,
    },
    MarketRegime.BULL_CHOPPY: {
        "sentiment":    0.20,
        "volatility":   0.40,
        "geopolitical": 0.25,
        "currency":     0.20,
    },
    MarketRegime.BEAR: {
        "sentiment":    0.25,
        "volatility":   0.30,
        "geopolitical": 0.15,
        "currency":     0.20,
    },
    MarketRegime.CRISIS: {
        "sentiment":    0.30,
        "volatility":   0.20,
        "geopolitical": 0.10,   # less penalty — geo IS the signal in crisis
        "currency":     0.15,
    },
}


def get_regime_weights(regime: MarketRegime) -> Dict[str, float]:
    """Return normalised agent weights for the given regime."""
    return dict(_WEIGHTS.get(regime, _WEIGHTS[MarketRegime.BULL_CHOPPY]))


def get_regime_correlations(regime: MarketRegime) -> Dict[str, float]:
    """Return correlation de-duplication penalties for the given regime."""
    return dict(_CORRELATION_MAP.get(regime, _CORRELATION_MAP[MarketRegime.BULL_CHOPPY]))


# ── Regime detection ─────────────────────────────────────────────────────────

_REGIME_CACHE: Optional[Tuple[MarketRegime, float]] = None
_CACHE_TTL = 3600   # 1-hour cache


def detect_regime() -> MarketRegime:
    """
    Detect current market regime from SPY price vs moving averages + VIX.
    Cached for 1 hour to avoid redundant API calls.
    """
    global _REGIME_CACHE
    if _REGIME_CACHE and time.time() < _REGIME_CACHE[1]:
        return _REGIME_CACHE[0]

    regime = _compute_regime()
    _REGIME_CACHE = (regime, time.time() + _CACHE_TTL)
    logger.info(f"[RegimeWeights] Detected regime: {regime.value}")
    return regime


def _compute_regime() -> MarketRegime:
    try:
        import yfinance as yf

        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period="1y", interval="1d")
        if spy_hist is None or spy_hist.empty or len(spy_hist) < 50:
            return MarketRegime.BULL_CHOPPY

        closes = spy_hist["Close"].dropna()
        price  = float(closes.iloc[-1])
        sma200 = float(closes.tail(200).mean()) if len(closes) >= 200 else float(closes.mean())
        sma50  = float(closes.tail(50).mean())

        # 20-day return for crash detection
        ret_20d = (price / float(closes.iloc[-21]) - 1) if len(closes) >= 21 else 0.0

        # Realised vol (20d annualised)
        log_rets = np.log(closes / closes.shift(1)).dropna()
        rv_20d   = float(log_rets.tail(20).std() * np.sqrt(252)) if len(log_rets) >= 20 else 0.15

        # VIX
        vix_level = 20.0
        try:
            vix_hist = yf.Ticker("^VIX").history(period="5d")
            if not vix_hist.empty:
                vix_level = float(vix_hist["Close"].iloc[-1])
        except Exception:
            pass

        # Classification
        if vix_level > 35 or ret_20d < -0.10:
            return MarketRegime.CRISIS

        if price < sma200 * 0.97 or (vix_level > 25 and rv_20d > 0.20):
            return MarketRegime.BEAR

        # Bull sub-regimes
        above_sma50  = price > sma50 * 1.005
        above_sma200 = price > sma200 * 1.005
        low_vix      = vix_level < 20
        low_rv       = rv_20d < 0.14

        if above_sma50 and above_sma200 and low_vix and low_rv:
            return MarketRegime.BULL_TREND

        return MarketRegime.BULL_CHOPPY

    except Exception as e:
        logger.warning(f"[RegimeWeights] Regime detection error: {e}")
        return MarketRegime.BULL_CHOPPY
