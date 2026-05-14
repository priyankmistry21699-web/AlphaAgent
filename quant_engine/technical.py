"""
AlphaAgent — Technical Indicators Engine

Computes all core technical indicators used by the Technical Agent.
Uses the `ta` library for standard indicators + custom implementations
for AlphaAgent-specific signals.

Factors computed (Phase 1 core set):
  - RSI (14-period)
  - MACD (12/26/9) + signal line + histogram
  - Bollinger Bands (20/2) + %B + bandwidth
  - SMA 50, SMA 200 + golden/death cross detection
  - EMA 9, EMA 21 + crossover detection
  - ADX (14) — trend strength
  - ATR (14) — average true range (volatility proxy)
  - OBV — on-balance volume (accumulation/distribution)
  - VWAP — volume-weighted average price
  - Stochastic %K/%D (14/3)
  - Volume ratio — current vs 20-day average
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)


@dataclass
class IndicatorResult:
    """Container for all computed technical indicators."""
    
    # ── RSI ──────────────────────────────────────────────
    rsi: float = 0.0                       # 0-100 scale
    rsi_signal: str = "neutral"            # oversold/neutral/overbought
    
    # ── MACD ─────────────────────────────────────────────
    macd_line: float = 0.0                 # MACD line value
    macd_signal_line: float = 0.0          # Signal line value
    macd_histogram: float = 0.0            # Histogram value
    macd_crossover: str = "none"           # bullish_cross/bearish_cross/none
    
    # ── Bollinger Bands ──────────────────────────────────
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_pct_b: float = 0.0                  # %B: where price is within bands (0-1)
    bb_bandwidth: float = 0.0              # Band width (volatility measure)
    bb_signal: str = "neutral"             # squeeze/breakout_up/breakout_down/neutral
    
    # ── Moving Averages ──────────────────────────────────
    sma_50: float = 0.0
    sma_200: float = 0.0
    ema_9: float = 0.0
    ema_21: float = 0.0
    price_vs_sma50: str = "above"          # above/below
    price_vs_sma200: str = "above"         # above/below
    golden_cross: bool = False             # SMA50 > SMA200 (bullish)
    death_cross: bool = False              # SMA50 < SMA200 (bearish)
    ema_crossover: str = "none"            # bullish_cross/bearish_cross/none
    
    # ── Trend Strength ───────────────────────────────────
    adx: float = 0.0                       # 0-100: >25 = strong trend
    adx_signal: str = "no_trend"           # no_trend/trending/strong_trend
    plus_di: float = 0.0                   # +DI (bullish directional)
    minus_di: float = 0.0                  # -DI (bearish directional)
    trend_direction: str = "neutral"       # bullish/bearish/neutral
    
    # ── Volatility ───────────────────────────────────────
    atr: float = 0.0                       # Average True Range (absolute $)
    atr_pct: float = 0.0                   # ATR as % of price
    
    # ── Volume ───────────────────────────────────────────
    obv: float = 0.0                       # On-Balance Volume
    obv_trend: str = "neutral"             # rising/falling/neutral
    volume_ratio: float = 0.0             # Current volume / 20-day avg
    volume_signal: str = "normal"          # low/normal/high/spike
    vwap: float = 0.0                      # Volume-Weighted Average Price
    price_vs_vwap: str = "above"           # above/below
    
    # ── Stochastic ───────────────────────────────────────
    stoch_k: float = 0.0                   # %K (fast)
    stoch_d: float = 0.0                   # %D (slow)
    stoch_signal: str = "neutral"          # oversold/neutral/overbought
    
    # ── Composite Score ──────────────────────────────────
    bullish_count: int = 0                 # How many indicators are bullish
    bearish_count: int = 0                 # How many indicators are bearish
    neutral_count: int = 0                 # How many are neutral
    composite_score: float = 50.0          # 0-100 (50 = neutral)
    
    # ── Metadata ─────────────────────────────────────────
    current_price: float = 0.0
    indicators_computed: int = 0
    warnings: list = field(default_factory=list)


def compute_indicators(df: pd.DataFrame) -> IndicatorResult:
    """
    Compute all technical indicators from OHLCV data.
    
    Args:
        df: DataFrame with columns [Open, High, Low, Close, Volume]
            Must have at least 200 rows for SMA200.
            
    Returns:
        IndicatorResult with all computed values
    """
    result = IndicatorResult()
    
    if df is None or len(df) < 26:
        result.warnings.append("Insufficient data for indicators (need 26+ rows)")
        return result
    
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    current_price = float(close.iloc[-1])
    result.current_price = current_price
    
    # ═══════════════════════════════════════════════════════
    #  RSI (Relative Strength Index)
    # ═══════════════════════════════════════════════════════
    try:
        rsi_indicator = ta.momentum.RSIIndicator(close, window=14)
        result.rsi = float(rsi_indicator.rsi().iloc[-1])
        
        if result.rsi < 30:
            result.rsi_signal = "oversold"        # Bullish signal
        elif result.rsi > 70:
            result.rsi_signal = "overbought"      # Bearish signal
        else:
            result.rsi_signal = "neutral"
        
        result.indicators_computed += 1
    except Exception as e:
        result.warnings.append(f"RSI error: {e}")
    
    # ═══════════════════════════════════════════════════════
    #  MACD (Moving Average Convergence Divergence)
    # ═══════════════════════════════════════════════════════
    try:
        macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        macd_line = macd.macd()
        signal_line = macd.macd_signal()
        histogram = macd.macd_diff()
        
        result.macd_line = float(macd_line.iloc[-1])
        result.macd_signal_line = float(signal_line.iloc[-1])
        result.macd_histogram = float(histogram.iloc[-1])
        
        # Detect crossover (last 2 bars)
        if len(macd_line) >= 2 and len(signal_line) >= 2:
            prev_diff = float(macd_line.iloc[-2] - signal_line.iloc[-2])
            curr_diff = float(macd_line.iloc[-1] - signal_line.iloc[-1])
            
            if prev_diff < 0 and curr_diff > 0:
                result.macd_crossover = "bullish_cross"
            elif prev_diff > 0 and curr_diff < 0:
                result.macd_crossover = "bearish_cross"
        
        result.indicators_computed += 1
    except Exception as e:
        result.warnings.append(f"MACD error: {e}")
    
    # ═══════════════════════════════════════════════════════
    #  Bollinger Bands
    # ═══════════════════════════════════════════════════════
    try:
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        result.bb_upper = float(bb.bollinger_hband().iloc[-1])
        result.bb_middle = float(bb.bollinger_mavg().iloc[-1])
        result.bb_lower = float(bb.bollinger_lband().iloc[-1])
        result.bb_pct_b = float(bb.bollinger_pband().iloc[-1])
        result.bb_bandwidth = float(bb.bollinger_wband().iloc[-1])
        
        # Squeeze detection (bandwidth < 20th percentile = tight squeeze)
        bw_series = bb.bollinger_wband().dropna()
        if len(bw_series) > 20:
            bw_percentile = (bw_series < result.bb_bandwidth).mean() * 100
            if bw_percentile < 20:
                result.bb_signal = "squeeze"
            elif current_price > result.bb_upper:
                result.bb_signal = "breakout_up"
            elif current_price < result.bb_lower:
                result.bb_signal = "breakout_down"
        
        result.indicators_computed += 1
    except Exception as e:
        result.warnings.append(f"Bollinger error: {e}")
    
    # ═══════════════════════════════════════════════════════
    #  Moving Averages (SMA 50/200, EMA 9/21)
    # ═══════════════════════════════════════════════════════
    try:
        # SMA
        result.sma_50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else 0.0
        result.sma_200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else 0.0
        
        # EMA
        result.ema_9 = float(close.ewm(span=9, adjust=False).mean().iloc[-1])
        result.ema_21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
        
        # Price vs SMA
        if result.sma_50 > 0:
            result.price_vs_sma50 = "above" if current_price > result.sma_50 else "below"
        if result.sma_200 > 0:
            result.price_vs_sma200 = "above" if current_price > result.sma_200 else "below"
        
        # Golden / Death cross
        if result.sma_50 > 0 and result.sma_200 > 0:
            result.golden_cross = result.sma_50 > result.sma_200
            result.death_cross = result.sma_50 < result.sma_200
        
        # EMA crossover detection
        if len(close) >= 22:
            ema9 = close.ewm(span=9, adjust=False).mean()
            ema21 = close.ewm(span=21, adjust=False).mean()
            prev_diff = float(ema9.iloc[-2] - ema21.iloc[-2])
            curr_diff = float(ema9.iloc[-1] - ema21.iloc[-1])
            
            if prev_diff < 0 and curr_diff > 0:
                result.ema_crossover = "bullish_cross"
            elif prev_diff > 0 and curr_diff < 0:
                result.ema_crossover = "bearish_cross"
        
        result.indicators_computed += 1
    except Exception as e:
        result.warnings.append(f"MA error: {e}")
    
    # ═══════════════════════════════════════════════════════
    #  ADX (Average Directional Index) — Trend Strength
    # ═══════════════════════════════════════════════════════
    try:
        adx_indicator = ta.trend.ADXIndicator(high, low, close, window=14)
        result.adx = float(adx_indicator.adx().iloc[-1])
        result.plus_di = float(adx_indicator.adx_pos().iloc[-1])
        result.minus_di = float(adx_indicator.adx_neg().iloc[-1])
        
        if result.adx < 20:
            result.adx_signal = "no_trend"
        elif result.adx < 40:
            result.adx_signal = "trending"
        else:
            result.adx_signal = "strong_trend"
        
        # Trend direction from DI
        if result.plus_di > result.minus_di and result.adx > 20:
            result.trend_direction = "bullish"
        elif result.minus_di > result.plus_di and result.adx > 20:
            result.trend_direction = "bearish"
        else:
            result.trend_direction = "neutral"
        
        result.indicators_computed += 1
    except Exception as e:
        result.warnings.append(f"ADX error: {e}")
    
    # ═══════════════════════════════════════════════════════
    #  ATR (Average True Range) — Volatility
    # ═══════════════════════════════════════════════════════
    try:
        atr_indicator = ta.volatility.AverageTrueRange(high, low, close, window=14)
        result.atr = float(atr_indicator.average_true_range().iloc[-1])
        result.atr_pct = (result.atr / current_price) * 100 if current_price > 0 else 0.0
        
        result.indicators_computed += 1
    except Exception as e:
        result.warnings.append(f"ATR error: {e}")
    
    # ═══════════════════════════════════════════════════════
    #  OBV (On-Balance Volume)
    # ═══════════════════════════════════════════════════════
    try:
        obv = ta.volume.OnBalanceVolumeIndicator(close, volume)
        obv_series = obv.on_balance_volume()
        result.obv = float(obv_series.iloc[-1])
        
        # OBV trend (compare to 20-day SMA of OBV)
        if len(obv_series) >= 20:
            obv_sma = float(obv_series.rolling(20).mean().iloc[-1])
            if result.obv > obv_sma * 1.02:
                result.obv_trend = "rising"       # Accumulation
            elif result.obv < obv_sma * 0.98:
                result.obv_trend = "falling"      # Distribution
            else:
                result.obv_trend = "neutral"
        
        result.indicators_computed += 1
    except Exception as e:
        result.warnings.append(f"OBV error: {e}")
    
    # ═══════════════════════════════════════════════════════
    #  Volume Analysis
    # ═══════════════════════════════════════════════════════
    try:
        avg_vol_20 = float(volume.rolling(20).mean().iloc[-1])
        current_vol = float(volume.iloc[-1])
        
        if avg_vol_20 > 0:
            result.volume_ratio = current_vol / avg_vol_20
        
        if result.volume_ratio < 0.5:
            result.volume_signal = "low"
        elif result.volume_ratio < 1.5:
            result.volume_signal = "normal"
        elif result.volume_ratio < 3.0:
            result.volume_signal = "high"
        else:
            result.volume_signal = "spike"        # Whale alert!
        
        result.indicators_computed += 1
    except Exception as e:
        result.warnings.append(f"Volume error: {e}")
    
    # ═══════════════════════════════════════════════════════
    #  VWAP (Volume-Weighted Average Price)
    # ═══════════════════════════════════════════════════════
    try:
        vwap = ta.volume.VolumeWeightedAveragePrice(high, low, close, volume)
        result.vwap = float(vwap.volume_weighted_average_price().iloc[-1])
        result.price_vs_vwap = "above" if current_price > result.vwap else "below"
        
        result.indicators_computed += 1
    except Exception as e:
        result.warnings.append(f"VWAP error: {e}")
    
    # ═══════════════════════════════════════════════════════
    #  Stochastic Oscillator
    # ═══════════════════════════════════════════════════════
    try:
        stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
        result.stoch_k = float(stoch.stoch().iloc[-1])
        result.stoch_d = float(stoch.stoch_signal().iloc[-1])
        
        if result.stoch_k < 20:
            result.stoch_signal = "oversold"
        elif result.stoch_k > 80:
            result.stoch_signal = "overbought"
        else:
            result.stoch_signal = "neutral"
        
        result.indicators_computed += 1
    except Exception as e:
        result.warnings.append(f"Stochastic error: {e}")
    
    # ═══════════════════════════════════════════════════════
    #  Composite Score (0-100)
    # ═══════════════════════════════════════════════════════
    _compute_composite_score(result)
    
    logger.info(f"Computed {result.indicators_computed} indicators | "
                f"Composite: {result.composite_score:.1f}/100 | "
                f"Bullish: {result.bullish_count} | Bearish: {result.bearish_count}")

    return result


def compute_seasonality_score(date) -> dict:
    """
    Compute calendar-based seasonality factor scores (0–100).

    Based on well-documented empirical anomalies:
      - Day-of-week: Mon bearish, Fri bullish
      - Month-of-year: Jan/Dec strong, Sep weakest
      - Quarter-end window dressing (last 5 trading days)
      - Turn-of-month effect (last 2 + first 3 days)
    """
    if hasattr(date, "to_pydatetime"):
        date = date.to_pydatetime()

    scores: dict = {}

    # Day-of-week effect  (0=Mon … 4=Fri, weekends skipped)
    dow_map = {0: 35.0, 1: 48.0, 2: 52.0, 3: 55.0, 4: 62.0}
    scores["day_of_week"] = dow_map.get(date.weekday(), 50.0)

    # Month-of-year effect
    month_map = {
        1: 65.0, 2: 52.0, 3: 52.0, 4: 58.0,
        5: 40.0, 6: 43.0, 7: 50.0, 8: 43.0,
        9: 35.0, 10: 52.0, 11: 58.0, 12: 65.0,
    }
    scores["month_of_year"] = month_map.get(date.month, 50.0)

    # Quarter-end window dressing (last 5 days of Mar/Jun/Sep/Dec)
    qend_months = {3: 31, 6: 30, 9: 30, 12: 31}
    if date.month in qend_months:
        days_remaining = qend_months[date.month] - date.day
        scores["quarter_end"] = 68.0 if days_remaining <= 5 else 50.0
    else:
        scores["quarter_end"] = 50.0

    # Turn-of-month effect (last 2 + first 3 days)
    scores["turn_of_month"] = 65.0 if (date.day <= 3 or date.day >= 28) else 50.0

    return scores


def _compute_composite_score(r: IndicatorResult):
    """
    Compute a composite bullish/bearish score from all indicators.
    
    Each indicator votes: +1 (bullish), -1 (bearish), or 0 (neutral).
    The composite score maps to 0-100 where 50 = neutral.
    """
    votes = []
    
    # RSI
    if r.rsi_signal == "oversold":
        votes.append(("RSI", +1))
    elif r.rsi_signal == "overbought":
        votes.append(("RSI", -1))
    else:
        votes.append(("RSI", 0))
    
    # MACD
    if r.macd_histogram > 0:
        votes.append(("MACD", +1))
    elif r.macd_histogram < 0:
        votes.append(("MACD", -1))
    else:
        votes.append(("MACD", 0))
    
    # MACD Crossover (strong signal)
    if r.macd_crossover == "bullish_cross":
        votes.append(("MACD_Cross", +1))
    elif r.macd_crossover == "bearish_cross":
        votes.append(("MACD_Cross", -1))
    else:
        votes.append(("MACD_Cross", 0))
    
    # Bollinger %B
    if r.bb_pct_b < 0.2:
        votes.append(("BB", +1))       # Near lower band = oversold
    elif r.bb_pct_b > 0.8:
        votes.append(("BB", -1))       # Near upper band = overbought
    else:
        votes.append(("BB", 0))
    
    # Price vs SMA 50
    votes.append(("SMA50", +1 if r.price_vs_sma50 == "above" else -1))
    
    # Price vs SMA 200
    if r.sma_200 > 0:
        votes.append(("SMA200", +1 if r.price_vs_sma200 == "above" else -1))
    
    # Golden/Death Cross
    if r.golden_cross:
        votes.append(("GoldenCross", +1))
    elif r.death_cross:
        votes.append(("DeathCross", -1))
    
    # EMA Crossover
    if r.ema_crossover == "bullish_cross":
        votes.append(("EMA_Cross", +1))
    elif r.ema_crossover == "bearish_cross":
        votes.append(("EMA_Cross", -1))
    else:
        votes.append(("EMA_Cross", 0))
    
    # ADX Trend Direction
    if r.trend_direction == "bullish":
        votes.append(("ADX", +1))
    elif r.trend_direction == "bearish":
        votes.append(("ADX", -1))
    else:
        votes.append(("ADX", 0))
    
    # OBV Trend
    if r.obv_trend == "rising":
        votes.append(("OBV", +1))
    elif r.obv_trend == "falling":
        votes.append(("OBV", -1))
    else:
        votes.append(("OBV", 0))
    
    # Price vs VWAP
    if r.vwap > 0:
        votes.append(("VWAP", +1 if r.price_vs_vwap == "above" else -1))
    
    # Stochastic
    if r.stoch_signal == "oversold":
        votes.append(("Stoch", +1))
    elif r.stoch_signal == "overbought":
        votes.append(("Stoch", -1))
    else:
        votes.append(("Stoch", 0))
    
    # Count votes
    r.bullish_count = sum(1 for _, v in votes if v > 0)
    r.bearish_count = sum(1 for _, v in votes if v < 0)
    r.neutral_count = sum(1 for _, v in votes if v == 0)
    
    # Composite: map total votes to 0-100 scale
    total_votes = len(votes)
    if total_votes > 0:
        net_score = sum(v for _, v in votes)
        # Map from [-total, +total] to [0, 100]
        r.composite_score = 50 + (net_score / total_votes) * 50
    else:
        r.composite_score = 50.0
