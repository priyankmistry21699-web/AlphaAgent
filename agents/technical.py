"""
AlphaAgent — Technical Agent

Reads price action and volume. Computes:
  - Core indicators (RSI, MACD, Bollinger, SMA/EMA, ADX, OBV, VWAP, Stochastic)
  - Momentum (Hurst exponent, 12M-1M Jegadeesh-Titman factor)
  - HMM market regime (Bull / Bear / Crisis)
  - Seasonality / calendar factors
"""

from typing import Any

from agents.base import BaseAgent
from agents.state import AgentResult, FactorScore
from quant_engine.technical import compute_indicators, compute_seasonality_score
from quant_engine.momentum import MomentumEngine
from quant_engine.hmm import RegimeDetector
from config.settings_manager import settings


class TechnicalAgent(BaseAgent):
    """
    Analyzes historical price data using technical indicators, momentum,
    regime detection, and calendar seasonality.
    """
    name = "technical"

    def _run_analysis(self, ticker: str, data: Any, **kwargs) -> AgentResult:
        period = settings.get("data.technical_period", "1y")
        min_days = settings.get("technical.min_data_days", 50)
        min_momentum_days = settings.get("technical.min_momentum_days", 100)
        ohlcv = data.get_ohlcv(period=period)

        if len(ohlcv) < min_days:
            return AgentResult(
                agent_name=self.name,
                probability_up=0.5,
                confidence=0.0,
                reasoning=f"Insufficient data for technical analysis (need {min_days}+ days).",
                warnings=["Insufficient history"],
            )

        # ── 1. Core indicators ───────────────────────────────────────────
        indicators = compute_indicators(ohlcv)

        # ── 2. Momentum (Hurst + 12M-1M) ────────────────────────────────
        momentum_result = None
        try:
            momentum_engine = MomentumEngine(ohlcv["Close"])
            momentum_result = momentum_engine.analyze()
        except Exception:
            pass

        # ── 3. HMM Market Regime ─────────────────────────────────────────
        regime_result = None
        try:
            returns = ohlcv["Close"].pct_change().dropna()
            if len(returns) >= min_momentum_days:
                detector = RegimeDetector(n_states=3)
                regime_result = detector.fit_predict(returns)
        except Exception:
            pass

        # ── 4. Seasonality ───────────────────────────────────────────────
        seasonality_scores: dict = {}
        try:
            seasonality_scores = compute_seasonality_score(ohlcv.index[-1])
        except Exception:
            pass

        # ── 5. Build FactorScores ────────────────────────────────────────
        factor_scores = {
            "rsi": FactorScore(
                name="RSI (14)",
                value=indicators.rsi,
                score=(100.0 if indicators.rsi_signal == "oversold"
                       else 0.0 if indicators.rsi_signal == "overbought"
                       else 50.0),
                interpretation=f"RSI {indicators.rsi:.1f} ({indicators.rsi_signal})",
            ),
            "macd": FactorScore(
                name="MACD Histogram",
                value=indicators.macd_histogram,
                score=100.0 if indicators.macd_histogram > 0 else 0.0,
                interpretation=f"Histogram {indicators.macd_histogram:+.3f} | cross: {indicators.macd_crossover}",
            ),
            "trend": FactorScore(
                name="ADX Trend",
                value=indicators.adx,
                score=(100.0 if indicators.trend_direction == "bullish"
                       else 0.0 if indicators.trend_direction == "bearish"
                       else 50.0),
                interpretation=f"ADX {indicators.adx:.1f} — {indicators.trend_direction}",
            ),
            "moving_averages": FactorScore(
                name="Golden / Death Cross",
                value=1.0 if indicators.golden_cross else -1.0 if indicators.death_cross else 0.0,
                score=100.0 if indicators.golden_cross else 0.0 if indicators.death_cross else 50.0,
                interpretation=f"SMA50 ${indicators.sma_50:.2f} | SMA200 ${indicators.sma_200:.2f}",
            ),
            "bollinger": FactorScore(
                name="Bollinger %B",
                value=indicators.bb_pct_b,
                score=(100.0 if indicators.bb_pct_b < 0.2
                       else 0.0 if indicators.bb_pct_b > 0.8
                       else 50.0),
                interpretation=f"%B {indicators.bb_pct_b:.2f} ({indicators.bb_signal})",
            ),
            "volume_obv": FactorScore(
                name="OBV Trend",
                value=indicators.volume_ratio,
                score=(70.0 if indicators.obv_trend == "rising"
                       else 30.0 if indicators.obv_trend == "falling"
                       else 50.0),
                interpretation=f"Volume ratio {indicators.volume_ratio:.1f}x | OBV {indicators.obv_trend}",
            ),
            "vwap": FactorScore(
                name="VWAP Position",
                value=1.0 if indicators.price_vs_vwap == "above" else -1.0,
                score=65.0 if indicators.price_vs_vwap == "above" else 35.0,
                interpretation=f"Price {'above' if indicators.price_vs_vwap == 'above' else 'below'} VWAP ${indicators.vwap:.2f}",
            ),
            "stochastic": FactorScore(
                name="Stochastic %K",
                value=indicators.stoch_k,
                score=(80.0 if indicators.stoch_signal == "oversold"
                       else 20.0 if indicators.stoch_signal == "overbought"
                       else 50.0),
                interpretation=f"%K {indicators.stoch_k:.1f} ({indicators.stoch_signal})",
            ),
        }

        # ── New: 52-Week Range Position ──────────────────────────────────
        try:
            high_52w = float(ohlcv["High"].rolling(252).max().iloc[-1])
            low_52w  = float(ohlcv["Low"].rolling(252).min().iloc[-1])
            last_price = float(ohlcv["Close"].iloc[-1])
            if high_52w > low_52w:
                pos_52w = (last_price - low_52w) / (high_52w - low_52w) * 100
                w52_score = (80.0 if pos_52w > 75 else 55.0 if pos_52w > 40 else 30.0)
                factor_scores["range_52w"] = FactorScore(
                    name="52-Week Range Position",
                    value=pos_52w,
                    score=w52_score,
                    interpretation=f"Price at {pos_52w:.1f}% of 52W range (L${low_52w:.2f}–H${high_52w:.2f})",
                )
        except Exception:
            pass

        # ── New: ATR-Normalized Momentum ─────────────────────────────────
        try:
            atr = indicators.atr if hasattr(indicators, 'atr') and indicators.atr else None
            if atr and atr > 0:
                close = float(ohlcv["Close"].iloc[-1])
                close_20 = float(ohlcv["Close"].iloc[-21])
                atr_mom = (close - close_20) / (atr * 20)
                atr_score = max(10.0, min(90.0, 50.0 + atr_mom * 15))
                factor_scores["atr_momentum"] = FactorScore(
                    name="ATR-Normalized Momentum",
                    value=round(atr_mom, 3),
                    score=atr_score,
                    interpretation=f"Vol-adj 20d move: {atr_mom:+.2f} ATR units",
                )
        except Exception:
            pass

        # ── New: Price vs EMA-200 (Long-Term Trend) ──────────────────────
        try:
            ema200 = float(ohlcv["Close"].ewm(span=200, adjust=False).mean().iloc[-1])
            last_p  = float(ohlcv["Close"].iloc[-1])
            ema200_gap = (last_p - ema200) / ema200 * 100
            ema200_score = (75.0 if ema200_gap > 0 else 25.0)
            factor_scores["ema200_trend"] = FactorScore(
                name="Price vs EMA-200",
                value=round(ema200_gap, 2),
                score=ema200_score,
                interpretation=f"Price {ema200_gap:+.1f}% vs EMA-200 (${ema200:.2f})",
            )
        except Exception:
            pass

        # ── New: Volume Surge (Breakout Confirmation) ─────────────────────
        try:
            vol_series = ohlcv["Volume"].dropna()
            if len(vol_series) > 20:
                avg_vol = float(vol_series.iloc[-20:].mean())
                last_vol = float(vol_series.iloc[-1])
                vol_surge = last_vol / avg_vol if avg_vol > 0 else 1.0
                vsurge_score = (80.0 if vol_surge > 2.0 else 65.0 if vol_surge > 1.5 else 50.0 if vol_surge > 0.8 else 35.0)
                factor_scores["volume_surge"] = FactorScore(
                    name="Volume Surge Ratio",
                    value=round(vol_surge, 2),
                    score=vsurge_score,
                    interpretation=f"Today's volume {vol_surge:.2f}x 20-day avg (breakout confirmation)",
                )
        except Exception:
            pass

        # ── New: Williams %R ─────────────────────────────────────────────
        try:
            high14 = float(ohlcv["High"].rolling(14).max().iloc[-1])
            low14  = float(ohlcv["Low"].rolling(14).min().iloc[-1])
            last_c = float(ohlcv["Close"].iloc[-1])
            if high14 > low14:
                williams_r = (high14 - last_c) / (high14 - low14) * -100
                wr_score = (80.0 if williams_r < -80 else 20.0 if williams_r > -20 else 50.0)
                factor_scores["williams_r"] = FactorScore(
                    name="Williams %R (14)",
                    value=round(williams_r, 1),
                    score=wr_score,
                    interpretation=f"W%R {williams_r:.1f} ({'oversold' if williams_r < -80 else 'overbought' if williams_r > -20 else 'neutral'})",
                )
        except Exception:
            pass

        # Momentum factors
        if momentum_result:
            factor_scores["hurst"] = FactorScore(
                name="Hurst Exponent",
                value=momentum_result.hurst_exponent,
                score=(70.0 if momentum_result.hurst_exponent > 0.55
                       else 30.0 if momentum_result.hurst_exponent < 0.45
                       else 50.0),
                interpretation=f"H={momentum_result.hurst_exponent:.3f} ({momentum_result.regime_type})",
            )
            mom_12m1m = momentum_result.momentum_12m_1m
            factor_scores["momentum_12m_1m"] = FactorScore(
                name="12M-1M Momentum",
                value=mom_12m1m,
                score=(70.0 if mom_12m1m > 0.05
                       else 30.0 if mom_12m1m < -0.05
                       else 50.0),
                interpretation=f"Jegadeesh-Titman: {mom_12m1m * 100:+.1f}%",
            )

        # HMM regime factor
        if regime_result:
            regime_score = {"BULL": 70.0, "BEAR": 30.0, "CRISIS": 10.0}.get(
                regime_result.current_regime, 50.0
            )
            factor_scores["hmm_regime"] = FactorScore(
                name="HMM Market Regime",
                value=regime_result.probabilities.get("bull", 0.5),
                score=regime_score,
                interpretation=(
                    f"Regime: {regime_result.current_regime} | "
                    f"P(Bull)={regime_result.probabilities.get('bull', 0):.2f}"
                ),
            )

        # Seasonality factors
        for key, score in seasonality_scores.items():
            factor_scores[f"seasonal_{key}"] = FactorScore(
                name=f"Seasonality: {key.replace('_', ' ').title()}",
                value=float(score),
                score=float(score),
                interpretation=f"Calendar factor: {score:.0f}/100",
            )

        # ── 6. Composite Probability ─────────────────────────────────────
        # Blend core composite (60%) with all factor scores (40%)
        all_scores = [fs.score for fs in factor_scores.values()]
        avg_all = sum(all_scores) / len(all_scores) if all_scores else 50.0
        blended = 0.6 * indicators.composite_score + 0.4 * avg_all
        prob_up = self._map_score_to_probability(blended, min_val=0, max_val=100)

        # ── 7. Confidence ────────────────────────────────────────────────
        confidence = 0.4
        if indicators.adx > 25:
            confidence += 0.15
        if indicators.volume_signal in ["high", "spike"]:
            confidence += 0.10
        if regime_result and regime_result.current_regime in ("BULL", "BEAR", "CRISIS"):
            confidence += 0.10
        if momentum_result and momentum_result.hurst_exponent > 0.6:
            confidence += 0.10
        confidence = min(max(confidence, 0.0), 1.0)

        # ── 8. Reasoning ─────────────────────────────────────────────────
        direction = "BULLISH" if prob_up > 0.55 else "BEARISH" if prob_up < 0.45 else "NEUTRAL"
        reasoning = (
            f"Technical outlook is {direction} ({prob_up * 100:.1f}% probability). "
            f"Price ${indicators.current_price:.2f}. "
            f"Trend: {indicators.adx_signal} (ADX {indicators.adx:.1f}). "
        )

        if indicators.golden_cross:
            reasoning += "Golden Cross (SMA50 > SMA200) — long-term bullish. "
        elif indicators.death_cross:
            reasoning += "Death Cross (SMA50 < SMA200) — long-term bearish. "

        if indicators.rsi_signal == "oversold":
            reasoning += f"RSI oversold ({indicators.rsi:.1f}) — bounce likely. "
        elif indicators.rsi_signal == "overbought":
            reasoning += f"RSI overbought ({indicators.rsi:.1f}) — pullback risk. "

        if momentum_result:
            reasoning += (
                f"Hurst={momentum_result.hurst_exponent:.3f} "
                f"({momentum_result.regime_type}). "
            )
        if regime_result:
            reasoning += f"HMM Regime: {regime_result.current_regime}. "

        return AgentResult(
            agent_name=self.name,
            probability_up=prob_up,
            confidence=confidence,
            reasoning=reasoning,
            factor_scores=factor_scores,
            warnings=indicators.warnings,
        )
