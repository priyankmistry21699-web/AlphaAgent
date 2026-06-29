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
        indicators = compute_indicators(ohlcv, ticker=ticker)

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

        # ── New: EMA 9/21 Crossover (Short-Term Trend) ──────────────────────
        try:
            ema9  = float(ohlcv["Close"].ewm(span=9,  adjust=False).mean().iloc[-1])
            ema21 = float(ohlcv["Close"].ewm(span=21, adjust=False).mean().iloc[-1])
            ema9_prev  = float(ohlcv["Close"].ewm(span=9,  adjust=False).mean().iloc[-2])
            ema21_prev = float(ohlcv["Close"].ewm(span=21, adjust=False).mean().iloc[-2])
            cross_now  = ema9 > ema21
            cross_prev = ema9_prev > ema21_prev
            if cross_now and not cross_prev:
                ema_cross_signal = "bullish_crossover"
                ema_score = 85.0
            elif not cross_now and cross_prev:
                ema_cross_signal = "bearish_crossover"
                ema_score = 15.0
            elif cross_now:
                ema_cross_signal = "above"
                ema_score = 68.0
            else:
                ema_cross_signal = "below"
                ema_score = 32.0
            factor_scores["ema_9_21"] = FactorScore(
                name="EMA 9/21 Crossover",
                value=round(ema9 - ema21, 3),
                score=ema_score,
                interpretation=f"EMA9 ${ema9:.2f} vs EMA21 ${ema21:.2f} — {ema_cross_signal}",
            )
        except Exception:
            pass

        # ── New: Bollinger Bandwidth (Volatility Squeeze) ────────────────────
        try:
            if hasattr(indicators, 'bb_bandwidth') and indicators.bb_bandwidth > 0:
                bw = indicators.bb_bandwidth
                _bw_narrow = settings.get("technical.bw_narrow", 20)
                _bw_wide = settings.get("technical.bw_wide", 40)
                bw_score = 70.0 if indicators.bb_signal == "squeeze" else 50.0 if bw < _bw_narrow else 35.0 if bw > _bw_wide else 50.0
                factor_scores["bb_bandwidth"] = FactorScore(
                    name="Bollinger Bandwidth",
                    value=round(bw, 2),
                    score=bw_score,
                    interpretation=f"BB width: {bw:.1f}% ({indicators.bb_signal}) — {'squeeze (breakout pending)' if indicators.bb_signal == 'squeeze' else 'expanded volatility' if bw > 40 else 'normal'}",
                )
        except Exception:
            pass

        # ── New: Momentum Acceleration (3M change in 3M momentum) ───────────
        try:
            if len(ohlcv) >= 130:
                mom_now   = float(ohlcv["Close"].iloc[-1] / ohlcv["Close"].iloc[-63] - 1) * 100
                mom_prior = float(ohlcv["Close"].iloc[-63] / ohlcv["Close"].iloc[-126] - 1) * 100
                mom_accel = mom_now - mom_prior
                accel_score = max(10.0, min(90.0, 50.0 + mom_accel * 1.2))
                factor_scores["momentum_acceleration"] = FactorScore(
                    name="Momentum Acceleration",
                    value=round(mom_accel, 2),
                    score=accel_score,
                    interpretation=f"3M mom: {mom_now:+.1f}% vs prior: {mom_prior:+.1f}% (Δ {mom_accel:+.1f}%) — {'accelerating' if mom_accel > 5 else 'decelerating' if mom_accel < -5 else 'stable'}",
                )
        except Exception:
            pass

        # ── New: Variance Risk Premium (IV - Realized Vol) ───────────────────
        try:
            import yfinance as yf
            tkr = yf.Ticker(ticker)
            exps = tkr.options
            if exps:
                chain = tkr.option_chain(exps[0])
                atm_iv = float(chain.calls["impliedVolatility"].dropna().median()) * 100
                # 30-day realized vol (annualized)
                ret30 = ohlcv["Close"].pct_change().dropna().iloc[-30:]
                rv30  = float(ret30.std() * (252 ** 0.5) * 100)
                vrp   = atm_iv - rv30
                vrp_score = (60.0 if vrp > 0 else 75.0)  # Negative VRP = options cheap = good
                factor_scores["variance_risk_premium"] = FactorScore(
                    name="Variance Risk Premium",
                    value=round(vrp, 2),
                    score=vrp_score,
                    interpretation=f"IV {atm_iv:.1f}% vs RV30 {rv30:.1f}% → VRP {vrp:+.1f}% ({'options expensive' if vrp > 5 else 'options cheap/fair'})",
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
            # ── New: P(Regime Change) from HMM Transition Matrix ─────────
            try:
                _curr = regime_result.current_regime
                _probs = regime_result.probabilities
                _trans = getattr(regime_result, 'transition_matrix', None)
                if _trans is not None:
                    _state_map = {"BULL": 0, "BEAR": 1, "CRISIS": 2}
                    _si = _state_map.get(_curr, 0)
                    _p_change = 1.0 - float(_trans[_si][_si]) if _si < len(_trans) else None
                else:
                    _p_curr = _probs.get(_curr.lower(), 1 / 3)
                    _p_change = 1.0 - _p_curr
                if _p_change is not None and 0 <= _p_change <= 1:
                    _prc_score = max(10.0, min(90.0, 100.0 * (1.0 - _p_change)))
                    factor_scores["p_regime_change"] = FactorScore(
                        name="P(Regime Change) - HMM",
                        value=round(_p_change, 3),
                        score=_prc_score,
                        interpretation=f"P(leave {_curr}): {_p_change*100:.1f}% ({'high transition risk' if _p_change > 0.5 else 'stable regime' if _p_change < 0.2 else 'moderate transition risk'})",
                    )
            except Exception:
                pass

        # Seasonality factors
        for key, score in seasonality_scores.items():
            factor_scores[f"seasonal_{key}"] = FactorScore(
                name=f"Seasonality: {key.replace('_', ' ').title()}",
                value=float(score),
                score=float(score),
                interpretation=f"Calendar factor: {score:.0f}/100",
            )

        # ── 5b. TDA (Topological Data Analysis) ─────────────────────────────
        try:
            from quant_engine.tda_signal import TDASignalEngine
            tda_engine = TDASignalEngine()
            tda_result = tda_engine.analyze(ohlcv["Close"])
            # Map TDA signal [-1,+1] to score [0,100]
            tda_score = 50.0 + tda_result.tda_signal * 40.0
            factor_scores["tda_regime"] = FactorScore(
                name="TDA Persistent Homology",
                value=round(tda_result.tda_signal, 3),
                score=round(tda_score, 1),
                interpretation=(
                    f"Regime: {tda_result.regime_label} | "
                    f"H0 entropy: {tda_result.h0_persistence_entropy:.3f} | "
                    f"H1 max: {tda_result.h1_max_persistence:.4f} | "
                    f"Betti-1: {tda_result.betti_1} "
                    f"({'trending — ride momentum' if tda_result.regime_label == 'TRENDING' else 'cyclic — mean-revert' if tda_result.regime_label == 'CYCLIC' else 'fragmented — regime shift' if tda_result.regime_label == 'FRAGMENTED' else 'neutral'})"
                ),
            )
        except Exception:
            pass

        # ── NEW: 52-Week High Proximity (IBD CANSLIM breakout signal) ──────
        try:
            _w52_high  = float(ohlcv["High"].rolling(252, min_periods=100).max().iloc[-1])
            _w52_low   = float(ohlcv["Low"].rolling(252, min_periods=100).min().iloc[-1])
            _w52_price = float(ohlcv["Close"].iloc[-1])
            _pct_from_high = (_w52_price / _w52_high - 1) * 100
            _w52_range_pct = (_w52_price - _w52_low) / max(_w52_high - _w52_low, 1e-9) * 100
            _w52_score = (85.0 if _pct_from_high > -5
                          else 68.0 if _pct_from_high > -15
                          else 45.0 if _pct_from_high > -30
                          else 25.0)
            factor_scores["week52_proximity"] = FactorScore(
                name="52-Week High Proximity",
                value=round(_pct_from_high, 2),
                score=_w52_score,
                interpretation=(
                    f"{_pct_from_high:.1f}% from 52W high ${_w52_high:.2f} | "
                    f"{_w52_range_pct:.0f}% of 52W range | "
                    f"{'near breakout' if _pct_from_high > -5 else 'oversold bounce candidate' if _pct_from_high < -30 else 'mid-range'}"
                ),
            )
        except Exception:
            pass

        # ── NEW: Adaptive RSI (thresholds widen in high-fear markets) ───────
        try:
            import yfinance as _yf_vix_rsi
            _vix_rsi = float(_yf_vix_rsi.Ticker("^VIX").history(period="3d")["Close"].dropna().iloc[-1])
            _rsi_os  = 38 if _vix_rsi > 30 else 32 if _vix_rsi > 20 else 26
            _rsi_ob  = 62 if _vix_rsi > 30 else 68 if _vix_rsi > 20 else 74
            _rsi_val = indicators.rsi
            _adap_score = (100.0 if _rsi_val < _rsi_os
                           else 0.0  if _rsi_val > _rsi_ob
                           else 50.0 + (_rsi_os - _rsi_val) / max(_rsi_ob - _rsi_os, 1) * 50.0)
            factor_scores["adaptive_rsi"] = FactorScore(
                name="Adaptive RSI (VIX-adjusted)",
                value=round(_rsi_val, 1),
                score=round(max(5.0, min(95.0, _adap_score)), 1),
                interpretation=(
                    f"RSI {_rsi_val:.1f} | VIX {_vix_rsi:.1f} → "
                    f"oversold<{_rsi_os} overbought>{_rsi_ob} | "
                    f"{'oversold' if _rsi_val < _rsi_os else 'overbought' if _rsi_val > _rsi_ob else 'neutral'}"
                ),
            )
        except Exception:
            pass

        # ── NEW: Signal strength vs ATR (momentum vs noise floor) ────────
        try:
            import numpy as _np_atr
            _hi_atr = ohlcv["High"].iloc[-22:]; _lo_atr = ohlcv["Low"].iloc[-22:]
            _cl_atr = ohlcv["Close"].iloc[-22:]
            _cl_prev = _cl_atr.shift(1).fillna(_cl_atr.iloc[0])
            _tr_atr  = _np_atr.maximum(
                (_hi_atr - _lo_atr).values,
                _np_atr.maximum(abs((_hi_atr - _cl_prev).values), abs((_lo_atr - _cl_prev).values))
            )
            _atr_pct  = float(_tr_atr.mean()) / float(_cl_atr.iloc[-1]) * 100
            _mom5d    = (float(_cl_atr.iloc[-1]) / float(ohlcv["Close"].iloc[-6]) - 1) * 100
            _strength = abs(_mom5d) / max(_atr_pct, 0.1)
            _atr_score = (78.0 if _strength > 2.5 else 63.0 if _strength > 1.5
                          else 50.0 if _strength > 0.8 else 38.0)
            factor_scores["signal_vs_atr"] = FactorScore(
                name="Signal Strength vs ATR",
                value=round(_strength, 2),
                score=_atr_score,
                interpretation=(
                    f"5d move {_mom5d:+.1f}% vs ATR {_atr_pct:.2f}%/day → "
                    f"{_strength:.1f}x ATR | "
                    f"{'strong directional signal' if _strength > 2.5 else 'moderate' if _strength > 1.5 else 'within noise'}"
                ),
            )
        except Exception:
            pass

        # ── Options-Based Factors (IV Skew, GEX, Max Pain, Implied Corr) ─
        try:
            import yfinance as _yf_opt
            import numpy as _np_opt
            _tkr_opt = _yf_opt.Ticker(ticker)
            _exps = _tkr_opt.options
            if _exps:
                _chain = _tkr_opt.option_chain(_exps[0])
                _calls = _chain.calls.dropna(subset=["strike", "impliedVolatility", "openInterest"])
                _puts  = _chain.puts.dropna(subset=["strike", "impliedVolatility", "openInterest"])
                _spot  = float(ohlcv["Close"].iloc[-1])

                # ── IV Skew (25Δ Put-Call) ────────────────────────────────
                try:
                    _call_otm = _calls[_calls["strike"].between(_spot * 1.04, _spot * 1.15)]
                    _put_otm  = _puts[_puts["strike"].between(_spot * 0.85, _spot * 0.96)]
                    if not _call_otm.empty and not _put_otm.empty:
                        _iv_call = float(_call_otm["impliedVolatility"].mean()) * 100
                        _iv_put  = float(_put_otm["impliedVolatility"].mean()) * 100
                        _iv_skew = _iv_put - _iv_call
                        _skew_score = max(10.0, min(90.0, 60.0 - _iv_skew * 2.0))
                        factor_scores["iv_skew"] = FactorScore(
                            name="IV Skew (25Δ Put-Call)",
                            value=round(_iv_skew, 2),
                            score=_skew_score,
                            interpretation=f"Put IV {_iv_put:.1f}% vs Call IV {_iv_call:.1f}% → Skew {_iv_skew:+.1f}% ({'bearish tail-risk hedging' if _iv_skew > 5 else 'bullish/complacent skew' if _iv_skew < -2 else 'neutral'})",
                        )
                except Exception:
                    pass

                # ── Gamma Exposure (GEX) ──────────────────────────────────
                try:
                    _T_gex   = 21 / 365
                    _two_pi  = float(_np_opt.sqrt(2 * _np_opt.pi))
                    _gex_net = 0.0
                    try:
                        from data.macro import MacroData as _MD_gex
                        _rf_s = _MD_gex().get_series("FEDFUNDS", years_back=1)
                        _rf_gex = float(_rf_s.iloc[-1].iloc[0]) / 100 if _rf_s is not None and len(_rf_s) > 0 else 0.045
                    except Exception:
                        _rf_gex = 0.045
                    for _, _row in _calls.iterrows():
                        try:
                            _K = float(_row["strike"]); _iv = float(_row["impliedVolatility"]); _oi = float(_row["openInterest"])
                            if _iv > 0 and _K > 0:
                                _d1 = (_np_opt.log(_spot / _K) + (_rf_gex + 0.5 * _iv ** 2) * _T_gex) / (_iv * _np_opt.sqrt(_T_gex))
                                _gamma = float(_np_opt.exp(-0.5 * _d1 ** 2) / (_two_pi * _spot * _iv * _np_opt.sqrt(_T_gex)))
                                _gex_net += _gamma * _oi * 100 * _spot
                        except Exception:
                            pass
                    for _, _row in _puts.iterrows():
                        try:
                            _K = float(_row["strike"]); _iv = float(_row["impliedVolatility"]); _oi = float(_row["openInterest"])
                            if _iv > 0 and _K > 0:
                                _d1 = (_np_opt.log(_spot / _K) + (_rf_gex + 0.5 * _iv ** 2) * _T_gex) / (_iv * _np_opt.sqrt(_T_gex))
                                _gamma = float(_np_opt.exp(-0.5 * _d1 ** 2) / (_two_pi * _spot * _iv * _np_opt.sqrt(_T_gex)))
                                _gex_net -= _gamma * _oi * 100 * _spot
                        except Exception:
                            pass
                    _gex_m = _gex_net / 1e6
                    factor_scores["gamma_exposure"] = FactorScore(
                        name="Gamma Exposure (GEX)",
                        value=round(_gex_m, 2),
                        score=65.0 if _gex_net > 0 else 35.0,
                        interpretation=f"Net GEX: ${_gex_m:+.1f}M ({'dealers long γ — vol suppression/mean-reversion' if _gex_net > 0 else 'dealers short γ — vol amplification/trending'})",
                    )
                except Exception:
                    pass

                # ── Max Pain ──────────────────────────────────────────────
                try:
                    _all_strikes = sorted(set(list(_calls["strike"]) + list(_puts["strike"])))
                    if len(_all_strikes) >= 5:
                        _pain = {}
                        for _s in _all_strikes:
                            _cp = float(sum(max(0.0, float(_s) - float(row["strike"])) * float(row["openInterest"])
                                           for _, row in _calls.iterrows()))
                            _pp = float(sum(max(0.0, float(row["strike"]) - float(_s)) * float(row["openInterest"])
                                           for _, row in _puts.iterrows()))
                            _pain[_s] = _cp + _pp
                        _mp = min(_pain, key=_pain.get)
                        _mp_gap = (float(_mp) - _spot) / _spot * 100
                        _mp_score = (65.0 if _mp_gap > 2 else 35.0 if _mp_gap < -2 else 50.0)
                        factor_scores["max_pain"] = FactorScore(
                            name="Options Max Pain",
                            value=round(float(_mp), 2),
                            score=_mp_score,
                            interpretation=f"Max pain: ${float(_mp):.2f} | Spot: ${_spot:.2f} | Gap: {_mp_gap:+.1f}% ({'upside gravity to max pain' if _mp_gap > 2 else 'downside gravity' if _mp_gap < -2 else 'pinned near max pain'})",
                        )
                except Exception:
                    pass

                # ── Implied Correlation proxy (SPY IV / Stock IV) ─────────
                try:
                    _spy_tkr  = _yf_opt.Ticker("SPY")
                    _spy_exps = _spy_tkr.options
                    if _spy_exps:
                        _spy_chain = _spy_tkr.option_chain(_spy_exps[0])
                        _spy_iv  = float(_spy_chain.calls["impliedVolatility"].dropna().median()) * 100
                        _stk_iv  = float(_calls["impliedVolatility"].dropna().median()) * 100
                        if _stk_iv > 0:
                            _impl_corr = _spy_iv / _stk_iv
                            _ic_score  = max(10.0, min(90.0, 70.0 - _impl_corr * 30.0))
                            factor_scores["implied_correlation"] = FactorScore(
                                name="Implied Correlation (SPY/Stock IV)",
                                value=round(_impl_corr, 3),
                                score=_ic_score,
                                interpretation=f"SPY IV {_spy_iv:.1f}% / Stock IV {_stk_iv:.1f}% = {_impl_corr:.2f} ({'macro-driven — limited alpha' if _impl_corr > 0.8 else 'idiosyncratic — alpha opportunity' if _impl_corr < 0.4 else 'mixed systemic/idiosyncratic'})",
                            )
                except Exception:
                    pass

        except Exception:
            pass

        # ── New: Fibonacci Retracement Levels (52W range) ────────────────
        try:
            _fib_hi = float(ohlcv["High"].rolling(252).max().iloc[-1])
            _fib_lo = float(ohlcv["Low"].rolling(252).min().iloc[-1])
            _fib_cl = float(ohlcv["Close"].iloc[-1])
            if _fib_hi > _fib_lo:
                _fib_rng = _fib_hi - _fib_lo
                _fibs = {0.236: _fib_hi - 0.236 * _fib_rng,
                         0.382: _fib_hi - 0.382 * _fib_rng,
                         0.500: _fib_hi - 0.500 * _fib_rng,
                         0.618: _fib_hi - 0.618 * _fib_rng,
                         0.786: _fib_hi - 0.786 * _fib_rng}
                _retrace_pos = (_fib_cl - _fib_lo) / _fib_rng          # 0=at low, 1=at high
                _nearest = min(_fibs.items(), key=lambda x: abs(_fib_cl - x[1]))
                _near_ratio, _near_price = _nearest
                _dist_pct = abs(_fib_cl - _near_price) / _near_price * 100
                if _retrace_pos > 0.618:
                    _fib_score = 78.0                                   # above golden ratio = strong bull
                elif _retrace_pos > 0.500:
                    _fib_score = 65.0
                elif _retrace_pos > 0.382:
                    _fib_score = 54.0
                elif _dist_pct < 1.5:                                   # sitting on fib support
                    _fib_score = 60.0
                else:
                    _fib_score = 32.0                                   # deep retracement
                factor_scores["fibonacci_levels"] = FactorScore(
                    name="Fibonacci Retracement",
                    value=round(_retrace_pos * 100, 1),
                    score=_fib_score,
                    interpretation=(
                        f"{_retrace_pos*100:.0f}% of 52W range — nearest Fib {_near_ratio*100:.1f}% level "
                        f"at ${_near_price:.2f} ({_dist_pct:.1f}% away)"
                    ),
                )
        except Exception:
            pass

        # ── New: Classic Pivot Points (Daily + Weekly) ───────────────────
        try:
            if len(ohlcv) >= 7:
                _ph = float(ohlcv["High"].iloc[-2])
                _pl = float(ohlcv["Low"].iloc[-2])
                _pc = float(ohlcv["Close"].iloc[-2])
                _curr = float(ohlcv["Close"].iloc[-1])
                _pp   = (_ph + _pl + _pc) / 3.0
                _r1   = 2 * _pp - _pl
                _r2   = _pp + (_ph - _pl)
                _s1   = 2 * _pp - _ph
                _s2   = _pp - (_ph - _pl)
                _whi  = float(ohlcv["High"].iloc[-6:-1].max())
                _wlo  = float(ohlcv["Low"].iloc[-6:-1].min())
                _wpp  = (_whi + _wlo + float(ohlcv["Close"].iloc[-2])) / 3.0
                _wr1  = 2 * _wpp - _wlo
                _ws1  = 2 * _wpp - _whi
                if _curr > _r2:
                    _piv_score = 88.0; _piv_lbl = f"Above R2 (${_r2:.2f}) — strong breakout"
                elif _curr > _r1:
                    _piv_score = 78.0; _piv_lbl = f"Above R1 (${_r1:.2f}) — bullish"
                elif _curr > _pp:
                    _piv_score = 63.0; _piv_lbl = f"Above daily pivot (${_pp:.2f})"
                elif _curr > _s1:
                    _piv_score = 40.0; _piv_lbl = f"Between S1 (${_s1:.2f}) and pivot"
                else:
                    _piv_score = 22.0; _piv_lbl = f"Below S1 (${_s1:.2f}) — bearish"
                if _curr > _wr1:
                    _piv_score = min(92.0, _piv_score + 6.0)
                elif _curr < _ws1:
                    _piv_score = max(10.0, _piv_score - 6.0)
                factor_scores["pivot_points"] = FactorScore(
                    name="Pivot Points (Daily/Weekly)",
                    value=round(_pp, 2),
                    score=_piv_score,
                    interpretation=f"{_piv_lbl} | PP=${_pp:.2f} R1=${_r1:.2f} S1=${_s1:.2f} | W.PP=${_wpp:.2f}",
                )
        except Exception:
            pass

        # ── New: Multi-Timeframe RSI + MACD Confluence (Daily vs Weekly) ─
        try:
            import yfinance as _yf_mtf
            _wkly = _yf_mtf.Ticker(ticker).history(period="2y", interval="1wk", auto_adjust=True)
            if len(_wkly) >= 26:
                _wc = _wkly["Close"].squeeze()
                _wd   = _wc.diff()
                _wg   = _wd.clip(lower=0).rolling(14).mean()
                _wl   = (-_wd.clip(upper=0)).rolling(14).mean()
                _wrsi = float((100 - 100 / (1 + _wg / _wl.replace(0, float("nan")))).iloc[-1])
                _we12 = _wc.ewm(span=12, adjust=False).mean()
                _we26 = _wc.ewm(span=26, adjust=False).mean()
                _whist = float((_we12 - _we26 - (_we12 - _we26).ewm(span=9, adjust=False).mean()).iloc[-1])
                _d_rsi  = indicators.rsi if hasattr(indicators, "rsi") else 50.0
                _d_hist = indicators.macd_histogram if hasattr(indicators, "macd_histogram") else 0.0
                _sigs = [
                    1 if _wrsi > 55 else (-1 if _wrsi < 45 else 0),
                    1 if _whist > 0 else -1,
                    1 if _d_rsi > 55 else (-1 if _d_rsi < 45 else 0),
                    1 if _d_hist > 0 else -1,
                ]
                _net = sum(_sigs)
                _mtf_score = max(10.0, min(90.0, 50.0 + _net * 12.0))
                _mtf_lbl = ("strong bull alignment" if _net >= 3 else
                            "mild bullish bias" if _net >= 1 else
                            "mixed / neutral" if _net == 0 else
                            "mild bearish bias" if _net >= -2 else
                            "strong bear alignment")
                factor_scores["multi_timeframe"] = FactorScore(
                    name="Multi-Timeframe Confluence (D/W)",
                    value=float(_net),
                    score=_mtf_score,
                    interpretation=f"Daily RSI {_d_rsi:.0f} + Weekly RSI {_wrsi:.0f} + MACD D/W → {_mtf_lbl} ({_net:+d}/4 aligned)",
                )
        except Exception:
            pass

        # ── New: Order Flow Imbalance (Close Location Value) ─────────────────
        try:
            import numpy as _np_ofi
            _ofi_h = ohlcv["High"].squeeze().dropna()
            _ofi_l = ohlcv["Low"].squeeze().dropna()
            _ofi_c = ohlcv["Close"].squeeze().dropna()
            _ofi_v = ohlcv["Volume"].squeeze().dropna()
            if len(_ofi_c) >= 20:
                _hl = (_ofi_h - _ofi_l).replace(0, float("nan"))
                _clv = (((_ofi_c - _ofi_l) - (_ofi_h - _ofi_c)) / _hl).fillna(0)
                _mfv = _clv * _ofi_v
                _clv_recent = float(_clv.iloc[-5:].mean())
                _mfv10 = float(_mfv.iloc[-10:].sum())
                _mfv20_abs = abs(float(_mfv.iloc[-20:].sum())) + 1
                _ofi_ratio = _mfv10 / _mfv20_abs
                _ofi_score = max(10.0, min(90.0, 50.0 + _clv_recent * 40.0))
                _ofi_lbl = ("strong buying pressure" if _clv_recent > 0.3 else
                            "mild buying" if _clv_recent > 0 else
                            "mild selling" if _clv_recent > -0.3 else
                            "strong selling pressure")
                factor_scores["order_flow_imbalance"] = FactorScore(
                    name="Order Flow Imbalance (CLV)",
                    value=round(_clv_recent, 3),
                    score=_ofi_score,
                    interpretation=f"5d avg CLV: {_clv_recent:+.3f} → {_ofi_lbl} | MFV 10d/20d ratio: {_ofi_ratio:+.2f}",
                )
        except Exception:
            pass

        # ── New: Cross-Sectional Momentum Rank (vs Sector Peers) ─────────────
        try:
            import yfinance as _yf_csr
            _SECTOR_PEERS = {
                "Technology": ["AAPL", "MSFT", "NVDA", "META", "GOOGL"],
                "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "T"],
                "Consumer Discretionary": ["AMZN", "TSLA", "HD", "NKE", "MCD"],
                "Consumer Staples": ["PG", "KO", "PEP", "WMT", "COST"],
                "Health Care": ["JNJ", "UNH", "PFE", "ABBV", "MRK"],
                "Industrials": ["CAT", "GE", "HON", "UPS", "RTX"],
                "Materials": ["LIN", "APD", "SHW", "NEM", "FCX"],
                "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
                "Financials": ["JPM", "BAC", "WFC", "GS", "MS"],
                "Real Estate": ["AMT", "PLD", "CCI", "EQIX", "PSA"],
                "Utilities": ["NEE", "DUK", "SO", "D", "AEP"],
            }
            _info_csr = {}
            try:
                _info_csr = data.get_info() or {}
            except Exception:
                _info_csr = _yf_csr.Ticker(ticker).info or {}
            _sector_csr = (_info_csr.get("sector", "") if isinstance(_info_csr, dict) else "")
            _peers = [p for p in _SECTOR_PEERS.get(_sector_csr, []) if p.upper() != ticker.upper()][:settings.get("backtest.cross_sectional_peers", 5)]
            if _peers:
                _lkbk_csr = settings.get_section("technical").get("momentum_lookback", 22)
                _rets = {}
                for _t in [ticker] + _peers:
                    try:
                        _s = _yf_csr.download(_t, period="3mo", interval="1d", auto_adjust=True, progress=False)
                        _c = _s["Close"].squeeze().dropna()
                        if len(_c) >= _lkbk_csr:
                            _rets[_t] = float((_c.iloc[-1] / _c.iloc[-_lkbk_csr] - 1) * 100)
                    except Exception:
                        pass
                if ticker in _rets and len(_rets) >= 2:
                    _sorted_r = sorted(_rets.values(), reverse=True)
                    _rank = _sorted_r.index(_rets[ticker]) + 1
                    _n_total = len(_sorted_r)
                    _csr_score = max(10.0, min(90.0, 100.0 * (1 - (_rank - 1) / _n_total)))
                    _peer_str = " | ".join(f"{t}: {v:+.0f}%" for t, v in list(_rets.items())[:4] if t != ticker)
                    factor_scores["cross_sectional_rank"] = FactorScore(
                        name=f"Cross-Sectional Rank ({_sector_csr or 'Sector'})",
                        value=float(_rank),
                        score=_csr_score,
                        interpretation=f"{ticker} #{_rank}/{_n_total} in {_sector_csr} | 22d: {_rets[ticker]:+.1f}% | Peers: {_peer_str}",
                    )
        except Exception:
            pass

        # ── New: Signal Backtest Calibration (Conditional Historical Hit Rate) ─
        try:
            import numpy as _np_bt
            _bt_lkbk = settings.get("backtest.calibration_lookback_days", 126)
            _bt_min  = settings.get("backtest.calibration_min_similar", 5)
            _bt_rsi_w = settings.get("backtest.calibration_rsi_window", 15)
            if len(ohlcv) >= _bt_lkbk + 20:
                _bt_cl  = ohlcv["Close"].squeeze().dropna()
                _bt_d   = _bt_cl.diff()
                _bt_g   = _bt_d.clip(lower=0).rolling(14).mean()
                _bt_l   = (-_bt_d.clip(upper=0)).rolling(14).mean()
                _bt_rsi = (100 - 100 / (1 + _bt_g / _bt_l.replace(0, float("nan")))).dropna()
                _bt_mom = _bt_cl.pct_change(22).dropna()
                _bt_fwd = _bt_cl.pct_change(5).shift(-5).dropna()
                _common = _bt_rsi.index.intersection(_bt_mom.dropna().index).intersection(_bt_fwd.dropna().index)
                _common = _common[-_bt_lkbk:] if len(_common) > _bt_lkbk else _common
                if len(_common) >= _bt_min:
                    _rsi_arr = _bt_rsi.reindex(_common).values
                    _mom_arr = _bt_mom.reindex(_common).values
                    _fwd_arr = _bt_fwd.reindex(_common).values
                    _cur_rsi = float(_bt_rsi.iloc[-1])
                    _cur_mom = float(_bt_mom.iloc[-1])
                    _mask = (
                        (_rsi_arr >= _cur_rsi - _bt_rsi_w) &
                        (_rsi_arr <= _cur_rsi + _bt_rsi_w) &
                        (_np_bt.sign(_mom_arr) == _np_bt.sign(_cur_mom)) &
                        _np_bt.isfinite(_fwd_arr)
                    )
                    _sim_fwd = _fwd_arr[_mask]
                    if len(_sim_fwd) >= _bt_min:
                        _hit_rate = float((_sim_fwd > 0).sum() / len(_sim_fwd) * 100)
                        _avg_fwd  = float(_np_bt.mean(_sim_fwd) * 100)
                        _bt_score = max(10.0, min(90.0, _hit_rate))
                        factor_scores["signal_backtest"] = FactorScore(
                            name="Signal Backtest Calibration",
                            value=round(_hit_rate, 1),
                            score=_bt_score,
                            interpretation=f"Past {_bt_lkbk}d similar RSI/momentum: {len(_sim_fwd)} cases | hit rate {_hit_rate:.0f}% up | avg 5d fwd: {_avg_fwd:+.2f}%",
                        )
        except Exception:
            pass

        # ── New: Chart Pattern Recognition (Double Top / Double Bottom) ─────
        try:
            import numpy as _np_cp
            _cp_cl = ohlcv["Close"].squeeze().dropna()
            if len(_cp_cl) >= 60:
                _w = _cp_cl.values[-60:]
                _peaks, _troughs = [], []
                for _i in range(2, len(_w) - 2):
                    if _w[_i] > _w[_i-1] and _w[_i] > _w[_i+1] and _w[_i] > _w[_i-2] and _w[_i] > _w[_i+2]:
                        _peaks.append((_i, float(_w[_i])))
                    if _w[_i] < _w[_i-1] and _w[_i] < _w[_i+1] and _w[_i] < _w[_i-2] and _w[_i] < _w[_i+2]:
                        _troughs.append((_i, float(_w[_i])))
                _pattern, _cp_score = "No clear pattern", 50.0
                _cur_cp = float(_cp_cl.iloc[-1])
                if len(_peaks) >= 2:
                    _p1_i, _p1_v = _peaks[-2]
                    _p2_i, _p2_v = _peaks[-1]
                    if abs(_p1_v - _p2_v) / max(_p1_v, _p2_v) < 0.03 and _p2_i - _p1_i >= 5:
                        _mid_tr = [t for t in _troughs if _p1_i < t[0] < _p2_i]
                        if _mid_tr:
                            _neck_v = min(_mid_tr, key=lambda x: x[1])[1]
                            if _cur_cp < _neck_v * 1.01:
                                _pattern, _cp_score = "DOUBLE TOP — bearish breakdown", 18.0
                            else:
                                _pattern, _cp_score = "Double Top forming at resistance", 35.0
                if len(_troughs) >= 2 and "forming" not in _pattern and "TOP" not in _pattern:
                    _t1_i, _t1_v = _troughs[-2]
                    _t2_i, _t2_v = _troughs[-1]
                    if abs(_t1_v - _t2_v) / max(_t1_v, _t2_v) < 0.03 and _t2_i - _t1_i >= 5:
                        _mid_pk = [p for p in _peaks if _t1_i < p[0] < _t2_i]
                        if _mid_pk:
                            _neck_v = max(_mid_pk, key=lambda x: x[1])[1]
                            if _cur_cp > _neck_v * 0.99:
                                _pattern, _cp_score = "DOUBLE BOTTOM — bullish breakout", 82.0
                            else:
                                _pattern, _cp_score = "Double Bottom forming at support", 65.0
                factor_scores["chart_pattern"] = FactorScore(
                    name="Chart Pattern Recognition",
                    value=float(len(_peaks) + len(_troughs)),
                    score=_cp_score,
                    interpretation=f"60-bar scan: {_pattern} | {len(_peaks)} peaks, {len(_troughs)} troughs",
                )
                if "bearish" in _pattern.lower():
                    warnings.append(f"Chart pattern: {_pattern}")
        except Exception:
            pass

        # ── New: Volume Profile — Point of Control (60-bar) ──────────────────
        try:
            import numpy as _np_vp
            _vp_cl  = ohlcv["Close"].squeeze().dropna().values[-60:]
            _vp_vol = ohlcv["Volume"].squeeze().dropna().values[-60:]
            if len(_vp_cl) >= 20 and len(_vp_cl) == len(_vp_vol):
                _vp_lo = float(_vp_cl.min())
                _vp_hi = float(_vp_cl.max())
                _vp_bins = _np_vp.linspace(_vp_lo, _vp_hi, 21)
                _vp_hist = _np_vp.zeros(20, dtype=float)
                for _px, _vol in zip(_vp_cl, _vp_vol):
                    _bi = min(int((_px - _vp_lo) / (_vp_hi - _vp_lo + 1e-9) * 20), 19)
                    _vp_hist[_bi] += float(_vol)
                _poc_idx   = int(_np_vp.argmax(_vp_hist))
                _poc_price = float((_vp_bins[_poc_idx] + _vp_bins[_poc_idx + 1]) / 2)
                _cur_vp    = float(ohlcv["Close"].iloc[-1])
                _poc_dist  = (_cur_vp - _poc_price) / _poc_price * 100
                _vp_score  = (72.0 if _poc_dist > 3 else 58.0 if _poc_dist > 0
                               else 42.0 if _poc_dist > -3 else 28.0)
                _tot_vol   = _vp_hist.sum()
                _cumvol, _va_bins = 0.0, 0
                for _hv in sorted(_vp_hist, reverse=True):
                    _cumvol += _hv
                    _va_bins += 1
                    if _cumvol >= 0.70 * _tot_vol:
                        break
                factor_scores["volume_profile"] = FactorScore(
                    name="Volume Profile (POC)",
                    value=round(_poc_price, 2),
                    score=_vp_score,
                    interpretation=f"POC: ${_poc_price:.2f} | Price {_poc_dist:+.1f}% {'above — strong support' if _poc_dist > 0 else 'below — overhead resistance'} | Value Area: {_va_bins}/20 bins",
                )
        except Exception:
            pass

        # ── New: Idiosyncratic Volatility Anomaly (Ang et al. 2006) ──────────
        # High idio vol → systematic underperformance
        try:
            import numpy as _np_iv
            import yfinance as _yf_iv
            _spy_iv = _yf_iv.download("SPY", period="3mo", interval="1d",
                                      auto_adjust=True, progress=False)
            if not _spy_iv.empty:
                _spy_r = _spy_iv["Close"].squeeze().pct_change().dropna()
                _tkr_r = ohlcv["Close"].pct_change().dropna()
                _df_iv = pd.concat([_tkr_r, _spy_r], axis=1, join="inner").dropna()
                _df_iv.columns = ["t", "m"]
                if len(_df_iv) >= 30:
                    _x = _df_iv["m"].values
                    _y = _df_iv["t"].values
                    _beta = float(_np_iv.cov(_x, _y)[0, 1] / _np_iv.var(_x))
                    _alpha = float(_np_iv.mean(_y) - _beta * _np_iv.mean(_x))
                    _resid = _y - (_alpha + _beta * _x)
                    _idio_vol = float(_np_iv.std(_resid)) * _np_iv.sqrt(252) * 100
                    _idio_score = (25.0 if _idio_vol > 60 else
                                   40.0 if _idio_vol > 40 else
                                   65.0 if _idio_vol > 20 else 75.0)
                    factor_scores["idio_volatility"] = FactorScore(
                        name="Idiosyncratic Volatility (Ang)",
                        value=round(_idio_vol, 1),
                        score=_idio_score,
                        interpretation=(
                            f"Idio vol (ann): {_idio_vol:.1f}% — "
                            f"{'extreme idio vol (Ang anomaly: bearish)' if _idio_vol > 60 else 'elevated idio vol' if _idio_vol > 40 else 'moderate' if _idio_vol > 20 else 'low idio vol'}"
                        ),
                    )
        except Exception:
            pass

        # ── New: MAX Anomaly (Bali et al. 2011) ───────────────────────────────
        # Stocks with highest single-day return in past month underperform (lottery demand)
        try:
            _rets_max = ohlcv["Close"].pct_change().dropna().tail(22)
            if len(_rets_max) >= 15:
                _max_d = float(_rets_max.max()) * 100
                _max_score = (30.0 if _max_d > 12 else
                              40.0 if _max_d > 7  else
                              60.0 if _max_d > 3  else 65.0)
                factor_scores["max_anomaly"] = FactorScore(
                    name="MAX Anomaly (Bali)",
                    value=round(_max_d, 2),
                    score=_max_score,
                    interpretation=(
                        f"Max 1d return (22d): {_max_d:.1f}% — "
                        f"{'lottery stock (bearish next month per Bali)' if _max_d > 7 else 'modest spike' if _max_d > 3 else 'no lottery feature'}"
                    ),
                )
        except Exception:
            pass

        # ── New: Short-Run Reversal (Jegadeesh 1990) ──────────────────────────
        # Past 1-week losers outperform next week
        try:
            _rets_1w = ohlcv["Close"].pct_change().dropna().tail(5)
            if len(_rets_1w) >= 5:
                _r1w = float((1 + _rets_1w).prod() - 1) * 100
                _sr_score = (75.0 if _r1w < -5 else      # losers → bullish
                             60.0 if _r1w < -2 else
                             40.0 if _r1w > 5  else
                             25.0 if _r1w > 8  else 50.0)
                factor_scores["short_run_reversal"] = FactorScore(
                    name="1-Week Short-Run Reversal",
                    value=round(_r1w, 2),
                    score=_sr_score,
                    interpretation=(
                        f"1w return: {_r1w:+.2f}% — "
                        f"{'strong loser → reversal expected (bullish)' if _r1w < -5 else 'mild loser → mean reversion' if _r1w < -2 else 'strong winner → reversal (bearish)' if _r1w > 5 else 'no reversal signal'}"
                    ),
                )
        except Exception:
            pass

        # ── New: Long-Run Reversal (DeBondt-Thaler) ───────────────────────────
        try:
            _closes_lr = ohlcv["Close"].dropna()
            if len(_closes_lr) >= 700:
                _r3y = float(_closes_lr.iloc[-1] / _closes_lr.iloc[-700] - 1) * 100
                _lr_score = (70.0 if _r3y < -30 else
                             55.0 if _r3y < -10 else
                             45.0 if _r3y > 50  else
                             35.0 if _r3y > 100 else 50.0)
                factor_scores["long_run_reversal"] = FactorScore(
                    name="3-Year Long-Run Reversal",
                    value=round(_r3y, 1),
                    score=_lr_score,
                    interpretation=(
                        f"3y return: {_r3y:+.1f}% — "
                        f"{'long-term loser → reversal expected' if _r3y < -30 else 'long-term winner → reversal risk' if _r3y > 100 else 'no extreme'}"
                    ),
                )
        except Exception:
            pass

        # ── New: 52-Week High Momentum (George-Hwang 2004) ────────────────────
        try:
            _hl_52 = ohlcv["Close"].dropna().tail(252)
            if len(_hl_52) >= 100:
                _p = float(_hl_52.iloc[-1])
                _h52 = float(_hl_52.max())
                _prox = _p / _h52
                _gh_score = (80.0 if _prox > 0.97 else   # near 52W high → momentum
                             65.0 if _prox > 0.90 else
                             50.0 if _prox > 0.80 else
                             40.0 if _prox > 0.70 else 30.0)
                factor_scores["week52_high_momentum"] = FactorScore(
                    name="52W High Momentum (George-Hwang)",
                    value=round(_prox, 3),
                    score=_gh_score,
                    interpretation=(
                        f"Price/52W high: {_prox:.3f} ({(1-_prox)*100:.1f}% below) — "
                        f"{'near 52W high: momentum buy signal' if _prox > 0.97 else 'close to high' if _prox > 0.90 else 'mid-range' if _prox > 0.70 else 'deep below high'}"
                    ),
                )
        except Exception:
            pass

        # ── New: Overnight vs Intraday Return Decomposition ───────────────────
        try:
            _od_df = ohlcv.dropna().tail(60)
            if len(_od_df) >= 30:
                _over = (_od_df["Open"] / _od_df["Close"].shift(1) - 1).dropna()
                _intr = (_od_df["Close"] / _od_df["Open"] - 1).dropna()
                _over_mean = float(_over.mean()) * 100
                _intr_mean = float(_intr.mean()) * 100
                _ratio = _over_mean - _intr_mean
                _od_score = (65.0 if _over_mean > 0.1 else
                             55.0 if _over_mean > 0    else
                             45.0 if _over_mean > -0.1 else 35.0)
                factor_scores["overnight_intraday"] = FactorScore(
                    name="Overnight vs Intraday Return",
                    value=round(_ratio, 3),
                    score=_od_score,
                    interpretation=(
                        f"Overnight: {_over_mean:+.3f}%/day | Intraday: {_intr_mean:+.3f}%/day — "
                        f"{'overnight bias (after-hours info bullish)' if _over_mean > _intr_mean + 0.05 else 'intraday bias' if _intr_mean > _over_mean + 0.05 else 'balanced'}"
                    ),
                )
        except Exception:
            pass

        # ── New: Commodity Roll Yield (contango/backwardation drag) ──────────
        try:
            from quant_engine.commodity_roll_yield import (
                compute_roll_yield_cached, COMMODITY_ROLL_PROXIES,
            )
            _commodity_etfs = {info["etf"]: sym for sym, info in COMMODITY_ROLL_PROXIES.items()}
            if ticker.upper() in _commodity_etfs:
                _sym = _commodity_etfs[ticker.upper()]
                _ry = compute_roll_yield_cached(_sym)
                if _ry is not None:
                    _ry_score = (25.0 if _ry.signal == "AVOID_ETF" else
                                 75.0 if _ry.signal == "FAVORABLE_ETF" else 50.0)
                    factor_scores["commodity_roll_yield"] = FactorScore(
                        name=f"Roll Yield ({_sym})",
                        value=round(_ry.roll_yield_annual * 100, 2),
                        score=_ry_score,
                        interpretation=(
                            f"Annual roll: {_ry.roll_yield_annual*100:+.1f}% | curve: {_ry.curve_state} ({_ry.severity}) — "
                            f"{'severe contango drag — avoid this ETF for buy-and-hold' if _ry.signal == 'AVOID_ETF' else 'backwardation tailwind — favorable for ETF' if _ry.signal == 'FAVORABLE_ETF' else 'neutral curve'}"
                        ),
                    )
                    if _ry.signal == "AVOID_ETF" and _ry.severity in ("moderate", "severe"):
                        warnings.append(f"Roll yield drag on {ticker}: {_ry.roll_yield_annual*100:+.1f}% annualised contango cost.")
        except Exception:
            pass

        # ── New: COT Commercials Net Position (smart money) ─────────────────
        try:
            from quant_engine.cot_data import get_cot_signal_cached, COT_COMMODITY_MAP
            _cot_etfs = {info["etf"]: sym for sym, info in COT_COMMODITY_MAP.items()}
            if ticker.upper() in _cot_etfs:
                _csym = _cot_etfs[ticker.upper()]
                _cot = get_cot_signal_cached(_csym)
                if _cot is not None:
                    _cot_score = (75.0 if _cot.signal == "BUY_SMART_MONEY" else
                                  25.0 if _cot.signal == "SELL_SMART_MONEY" else 50.0)
                    factor_scores["cot_commercials"] = FactorScore(
                        name=f"COT Commercials Net ({_csym})",
                        value=_cot.commercials_z,
                        score=_cot_score,
                        interpretation=(
                            f"COT commercials z={_cot.commercials_z:+.2f} | net={_cot.commercials_net:,} | OI={_cot.open_interest:,} | report: {_cot.report_date[:10]} — "
                            f"{'smart money extreme long: bullish' if _cot.signal == 'BUY_SMART_MONEY' else 'smart money extreme short: bearish' if _cot.signal == 'SELL_SMART_MONEY' else 'no extreme positioning'}"
                        ),
                    )
                    if _cot.signal in ("BUY_SMART_MONEY", "SELL_SMART_MONEY"):
                        reasoning.append(f"COT: commercials {_cot.signal.lower()} ({_cot.commercials_z:+.2f}σ).")
        except Exception:
            pass

        # ── New: Weather Anomaly (natgas / energy sensitivity) ──────────────
        try:
            from quant_engine.weather_factor import get_weather_signal_cached
            _energy_etfs = {"UNG", "USO", "XLE", "XOP"}
            if ticker.upper() in _energy_etfs:
                _wx = get_weather_signal_cached()
                if _wx is not None:
                    _wx_score = (70.0 if _wx.signal == "BULLISH_NATGAS" and ticker.upper() in ("UNG", "XLE") else
                                 30.0 if _wx.signal == "BEARISH_NATGAS" and ticker.upper() in ("UNG", "XLE") else 50.0)
                    factor_scores["weather_anomaly"] = FactorScore(
                        name="Weather Anomaly (US)",
                        value=_wx.anomaly_f,
                        score=_wx_score,
                        interpretation=(
                            f"US avg temp: {_wx.current_temp_f}°F | seasonal: {_wx.seasonal_avg_f}°F | anomaly: {_wx.anomaly_f:+.1f}°F | "
                            f"HDD anom: {_wx.hdd_anomaly:+.1f} | CDD anom: {_wx.cdd_anomaly:+.1f} — "
                            f"{'extreme heating/cooling demand: bullish nat gas' if _wx.signal == 'BULLISH_NATGAS' else 'mild weather: bearish nat gas' if _wx.signal == 'BEARISH_NATGAS' else 'normal weather'}"
                        ),
                    )
        except Exception:
            pass

        # ── New: EIA Petroleum Inventory Signal (energy ETFs) ────────────────
        try:
            from quant_engine.eia_petroleum import get_eia_signal_cached
            _energy_map = {"USO": "CRUDE", "XLE": "CRUDE", "XOP": "CRUDE",
                           "UNG": "NATGAS", "UGA": "GASOLINE"}
            if ticker.upper() in _energy_map:
                _eia = get_eia_signal_cached(_energy_map[ticker.upper()])
                if _eia is not None:
                    _eia_score = (70.0 if _eia.signal == "BULLISH_OIL" else
                                  30.0 if _eia.signal == "BEARISH_OIL" else 50.0)
                    factor_scores["eia_inventory"] = FactorScore(
                        name=f"EIA {_eia.series} Inventory",
                        value=_eia.change_z_score,
                        score=_eia_score,
                        interpretation=(
                            f"EIA proxy z={_eia.change_z_score:+.2f} | 10d move: {_eia.change_wow_mb:+.2f}% — "
                            f"{'inventory drawing → bullish oil/energy' if _eia.signal == 'BULLISH_OIL' else 'inventory building → bearish' if _eia.signal == 'BEARISH_OIL' else 'neutral inventory dynamics'}"
                        ),
                    )
        except Exception:
            pass

        # ── New: Google Trends Search Volume (retail attention) ──────────────
        try:
            from quant_engine.google_trends import get_trends_signal_cached
            _gt = get_trends_signal_cached(ticker)
            if _gt is not None:
                _gt_score = (70.0 if _gt.signal == "BULLISH_ATTENTION" else
                             30.0 if _gt.signal == "BEARISH_FADE" else 50.0)
                factor_scores["google_trends"] = FactorScore(
                    name="Google Trends Attention",
                    value=_gt.z_score,
                    score=_gt_score,
                    interpretation=(
                        f"Trends '{_gt.keyword}' z={_gt.z_score:+.2f} | 4w slope: {_gt.trend_4w:+.1f}% — "
                        f"{'rising retail attention: bullish' if _gt.signal == 'BULLISH_ATTENTION' else 'fading/peaked attention: bearish' if _gt.signal == 'BEARISH_FADE' else 'normal attention'}"
                    ),
                )
        except Exception:
            pass

        # ── New: ETF Premium/Discount to NAV (mean-reversion signal) ──────────
        try:
            from quant_engine.etf_premium import etf_premium_discount, ETF_NAV_PROXY
            if ticker.upper() in ETF_NAV_PROXY:
                _ep = etf_premium_discount(ticker.upper())
                if _ep is not None:
                    _ep_score = (75.0 if _ep.signal == "BUY_DISCOUNT" else
                                 25.0 if _ep.signal == "SELL_PREMIUM" else 50.0)
                    factor_scores["etf_premium"] = FactorScore(
                        name="ETF Premium/Discount to NAV",
                        value=_ep.premium_pct,
                        score=_ep_score,
                        interpretation=(
                            f"ETF: ${_ep.price:.2f} | NAV-proxy: {_ep.nav:.2f} | premium: {_ep.premium_pct:+.2f}% | z={_ep.z_score:+.2f} — "
                            f"{'discount → mean-reversion BUY' if _ep.signal == 'BUY_DISCOUNT' else 'premium → mean-reversion SELL' if _ep.signal == 'SELL_PREMIUM' else 'no NAV anomaly'}"
                        ),
                    )
                    if _ep.signal in ("BUY_DISCOUNT", "SELL_PREMIUM"):
                        reasoning.append(f"ETF NAV anomaly: {_ep.signal} (z={_ep.z_score:+.2f}) — known convergence to NAV.")
        except Exception:
            pass

        # ── New: Momentum Crash Risk (Daniel-Moskowitz) ───────────────────────
        # Momentum strategies crash when market bounces off a bear bottom
        try:
            import yfinance as _yf_mc
            _spy_mc = _yf_mc.download("SPY", period="6mo", interval="1d",
                                      auto_adjust=True, progress=False)
            if not _spy_mc.empty:
                _spy_c = _spy_mc["Close"].squeeze().dropna()
                _spy_max_dd = float((_spy_c / _spy_c.cummax() - 1).min()) * 100
                _spy_recent = float(_spy_c.iloc[-1] / _spy_c.iloc[-22] - 1) * 100 if len(_spy_c) >= 22 else 0.0
                # Crash risk = recent bear (-15%+ DD) + sharp bounce (>8% in 1M)
                _crash_risk = (_spy_max_dd < -15) and (_spy_recent > 8)
                _mc_score = 25.0 if _crash_risk else 60.0
                factor_scores["momentum_crash_risk"] = FactorScore(
                    name="Momentum Crash Risk (Daniel-Moskowitz)",
                    value=round(_spy_recent, 1),
                    score=_mc_score,
                    interpretation=(
                        f"SPY 6mo DD: {_spy_max_dd:.1f}% | 1M return: {_spy_recent:+.1f}% — "
                        f"{'MOMENTUM CRASH RISK: post-bear bounce, momentum strategies vulnerable' if _crash_risk else 'no crash setup'}"
                    ),
                )
                if _crash_risk:
                    warnings.append(f"Momentum crash setup: -{abs(_spy_max_dd):.0f}% DD followed by +{_spy_recent:.0f}% bounce.")
        except Exception:
            pass

        # ── PCA of Factor Scores (Signal Consensus Quality) ───────────────
        try:
            import numpy as _np_pca
            _scores_arr = _np_pca.array([fs.score for fs in factor_scores.values()], dtype=float)
            if len(_scores_arr) >= 5:
                _norm            = (_scores_arr - 50.0) / 50.0
                _mean_signal     = float(_np_pca.mean(_norm))
                _signal_strength = float(_np_pca.abs(_norm).mean())
                _dispersion      = float(_np_pca.std(_norm))
                _consensus_score = _signal_strength * max(0.0, 1.0 - _dispersion / 1.5)
                _pca_score = max(10.0, min(90.0, 50.0 + _mean_signal * 40.0))
                factor_scores["pca_signal_quality"] = FactorScore(
                    name="PCA Signal Quality",
                    value=round(_consensus_score, 3),
                    score=round(_pca_score, 1),
                    interpretation=(
                        f"Factor consensus: {_mean_signal:+.2f} | Strength: {_signal_strength:.2f} | σ: {_dispersion:.2f} — "
                        f"{'high conviction' if _consensus_score > 0.4 else 'mixed/uncertain signal' if _consensus_score < 0.2 else 'moderate consensus'}"
                    ),
                )
        except Exception:
            pass

        # ── New: Ichimoku Cloud ──────────────────────────────────────────
        try:
            if len(ohlcv) >= 52:
                _h = ohlcv["High"]
                _l = ohlcv["Low"]
                _c = ohlcv["Close"]
                _tenkan  = (_h.rolling(9).max()  + _l.rolling(9).min())  / 2
                _kijun   = (_h.rolling(26).max() + _l.rolling(26).min()) / 2
                _senkou_a = ((_tenkan + _kijun) / 2).shift(26)
                _senkou_b = ((_h.rolling(52).max() + _l.rolling(52).min()) / 2).shift(26)
                _price_now = float(_c.iloc[-1])
                _tk = float(_tenkan.iloc[-1])
                _kj = float(_kijun.iloc[-1])
                _sa = float(_senkou_a.dropna().iloc[-1]) if _senkou_a.dropna().size else _price_now
                _sb = float(_senkou_b.dropna().iloc[-1]) if _senkou_b.dropna().size else _price_now
                _cloud_top = max(_sa, _sb)
                _cloud_bot = min(_sa, _sb)
                _above_cloud = _price_now > _cloud_top
                _below_cloud = _price_now < _cloud_bot
                _tk_bull = _tk > _kj
                _ichi_score = (80.0 if _above_cloud and _tk_bull else 65.0 if _above_cloud else 35.0 if _below_cloud else 50.0)
                factor_scores["ichimoku"] = FactorScore(
                    name="Ichimoku Cloud",
                    value=round(_price_now, 2),
                    score=_ichi_score,
                    interpretation=(
                        f"Price ${_price_now:.2f} | Cloud {_cloud_bot:.2f}-{_cloud_top:.2f} | "
                        f"{'above cloud (bullish)' if _above_cloud else 'below cloud (bearish)' if _below_cloud else 'inside cloud (neutral)'} | "
                        f"TK {'bull' if _tk_bull else 'bear'}"
                    ),
                )
        except Exception:
            pass

        # ── New: Chaikin Money Flow (CMF 14) ─────────────────────────────
        try:
            if len(ohlcv) >= 14 and "Volume" in ohlcv.columns:
                _cmf_h = ohlcv["High"]
                _cmf_l = ohlcv["Low"]
                _cmf_c = ohlcv["Close"]
                _cmf_v = ohlcv["Volume"]
                _hl_rng = (_cmf_h - _cmf_l).replace(0, float("nan"))
                _mfv = ((_cmf_c - _cmf_l) - (_cmf_h - _cmf_c)) / _hl_rng * _cmf_v
                _cmf_val = float(_mfv.iloc[-14:].sum() / _cmf_v.iloc[-14:].sum())
                _cmf_score = max(10.0, min(90.0, 50.0 + _cmf_val * 150.0))
                factor_scores["chaikin_mf"] = FactorScore(
                    name="Chaikin Money Flow (CMF-14)",
                    value=round(_cmf_val, 4),
                    score=_cmf_score,
                    interpretation=(
                        f"CMF: {_cmf_val:+.3f} — "
                        f"{'accumulation (buying pressure)' if _cmf_val > 0.1 else 'distribution (selling pressure)' if _cmf_val < -0.1 else 'neutral money flow'}"
                    ),
                )
        except Exception:
            pass

        # ── New: TRIX Oscillator (15-period) ─────────────────────────────
        try:
            if len(ohlcv) >= 50:
                _trix_c = ohlcv["Close"]
                _ema1 = _trix_c.ewm(span=15, adjust=False).mean()
                _ema2 = _ema1.ewm(span=15, adjust=False).mean()
                _ema3 = _ema2.ewm(span=15, adjust=False).mean()
                _trix = _ema3.pct_change() * 100
                _trix_now = float(_trix.iloc[-1])
                _trix_prev = float(_trix.iloc[-2])
                _trix_score = max(10.0, min(90.0, 50.0 + _trix_now * 800.0))
                factor_scores["trix"] = FactorScore(
                    name="TRIX Oscillator (15-period)",
                    value=round(_trix_now, 5),
                    score=_trix_score,
                    interpretation=(
                        f"TRIX: {_trix_now:+.5f} | signal cross: {'bullish' if _trix_now > 0 and _trix_prev <= 0 else 'bearish' if _trix_now < 0 and _trix_prev >= 0 else 'trending'} — "
                        f"{'upward momentum' if _trix_now > 0 else 'downward momentum'}"
                    ),
                )
        except Exception:
            pass

        # ── New: Parabolic SAR ────────────────────────────────────────────
        try:
            if len(ohlcv) >= 30:
                _sar_h = ohlcv["High"].values
                _sar_l = ohlcv["Low"].values
                _sar_c = ohlcv["Close"].values
                _af_step, _af_max = 0.02, 0.20
                _sar = _sar_c[0]
                _ep = _sar_h[0]
                _af = _af_step
                _bull = True
                for _i in range(1, len(_sar_c)):
                    _prev_sar = _sar
                    if _bull:
                        _sar = _prev_sar + _af * (_ep - _prev_sar)
                        _sar = min(_sar, _sar_l[_i - 1], _sar_l[max(0, _i - 2)])
                        if _sar_l[_i] < _sar:
                            _bull = False
                            _sar = _ep
                            _ep = _sar_l[_i]
                            _af = _af_step
                        else:
                            if _sar_h[_i] > _ep:
                                _ep = _sar_h[_i]
                                _af = min(_af + _af_step, _af_max)
                    else:
                        _sar = _prev_sar + _af * (_ep - _prev_sar)
                        _sar = max(_sar, _sar_h[_i - 1], _sar_h[max(0, _i - 2)])
                        if _sar_h[_i] > _sar:
                            _bull = True
                            _sar = _ep
                            _ep = _sar_h[_i]
                            _af = _af_step
                        else:
                            if _sar_l[_i] < _ep:
                                _ep = _sar_l[_i]
                                _af = min(_af + _af_step, _af_max)
                _psar_score = 72.0 if _bull else 28.0
                factor_scores["parabolic_sar"] = FactorScore(
                    name="Parabolic SAR",
                    value=round(_sar, 4),
                    score=_psar_score,
                    interpretation=(
                        f"SAR: {_sar:.4f} | Price: {float(_sar_c[-1]):.2f} — "
                        f"{'bullish (price above SAR, uptrend)' if _bull else 'bearish (price below SAR, downtrend)'}"
                    ),
                )
        except Exception:
            pass

        # ── New: Supply / Demand Zones ───────────────────────────────────
        try:
            if len(ohlcv) >= 60:
                import numpy as _np_sd
                _sd_h   = ohlcv["High"].values
                _sd_l   = ohlcv["Low"].values
                _sd_c   = ohlcv["Close"].values
                _sd_v   = ohlcv["Volume"].values if "Volume" in ohlcv.columns else _np_sd.ones(len(_sd_c))
                _sd_cur = float(_sd_c[-1])
                _n_sd   = min(len(_sd_c) - 1, 60)
                _supply_zones, _demand_zones = [], []
                for _i in range(2, _n_sd - 2):
                    _local_h = float(_sd_h[_i])
                    _local_l = float(_sd_l[_i])
                    _vol_i   = float(_sd_v[_i])
                    _avg_vol = float(_np_sd.mean(_sd_v[max(0, _i - 10):_i + 10]))
                    _vol_spike = _vol_i > _avg_vol * 1.5
                    if _vol_spike and _sd_h[_i] > _sd_h[_i - 1] and _sd_h[_i] > _sd_h[_i + 1]:
                        _supply_zones.append(_local_h)
                    if _vol_spike and _sd_l[_i] < _sd_l[_i - 1] and _sd_l[_i] < _sd_l[_i + 1]:
                        _demand_zones.append(_local_l)
                _nearest_sup = min((_z for _z in _supply_zones if _z > _sd_cur), default=None)
                _nearest_dem = max((_z for _z in _demand_zones if _z < _sd_cur), default=None)
                if _nearest_sup and _nearest_dem:
                    _range_sd = _nearest_sup - _nearest_dem
                    _pos_sd   = (_sd_cur - _nearest_dem) / _range_sd if _range_sd > 0 else 0.5
                    _sd_score = max(10.0, min(90.0, (1.0 - _pos_sd) * 80.0 + 10.0))
                    factor_scores["supply_demand_zones"] = FactorScore(
                        name="Supply / Demand Zones",
                        value=round(_sd_cur, 2),
                        score=_sd_score,
                        interpretation=(
                            f"Price ${_sd_cur:.2f} | Demand zone: ${_nearest_dem:.2f} | Supply zone: ${_nearest_sup:.2f} — "
                            f"{'near demand — potential support' if _pos_sd < 0.3 else 'near supply — potential resistance' if _pos_sd > 0.7 else 'mid-range'}"
                        ),
                    )
        except Exception:
            pass

        # ── New: Short-Term Momentum (5-day) ────────────────────────────
        # Catches stocks that started moving 3-5 days before the 22d signal
        # fires. Specifically captures sector-rotation pre-move.
        try:
            _c5 = ohlcv["Close"].dropna()
            if len(_c5) >= 6:
                _mom5 = (_c5.iloc[-1] / _c5.iloc[-6] - 1.0) * 100
                _mom5_score = (
                    88.0 if _mom5 >  4.0 else
                    76.0 if _mom5 >  2.0 else
                    65.0 if _mom5 >  0.5 else
                    50.0 if _mom5 > -0.5 else
                    38.0 if _mom5 > -2.0 else
                    25.0 if _mom5 > -4.0 else
                    15.0
                )
                factor_scores["momentum_5d"] = FactorScore(
                    name="5-Day Price Momentum",
                    value=round(_mom5, 2),
                    score=_mom5_score,
                    interpretation=f"5d return: {_mom5:+.2f}% ({'building' if _mom5 > 0.5 else 'fading' if _mom5 < -0.5 else 'flat'})",
                )
        except Exception:
            pass

        # ── New: Sector ETF Momentum ─────────────────────────────────────
        # When the sector ETF is in a 5-day uptrend, stocks in that sector
        # get a tailwind boost even if their own signal is mildly bullish.
        # This directly fixes the missed CRWD/PANW/ZS/DDOG rotation.
        _SECTOR_ETF = {
            # Cybersecurity
            "CRWD": "CIBR", "PANW": "CIBR", "ZS": "CIBR", "FTNT": "CIBR",
            "NET": "CIBR", "OKTA": "CIBR", "CYBR": "CIBR",
            # Cloud / Software
            "SNOW": "IGV",  "DDOG": "IGV",  "MDB": "IGV",  "TEAM": "IGV",
            "NOW":  "IGV",  "WDAY": "IGV",  "HUBS": "IGV",
            # Semiconductors
            "AMAT": "SMH",  "LRCX": "SMH",  "KLAC": "SMH",
            "MU":   "SMH",  "NVDA": "SMH",  "AMD":  "SMH",  "AVGO": "SMH",
            "INTC": "SMH",  "QCOM": "SMH",  "TXN":  "SMH",
            # Broad Tech
            "AAPL": "XLK",  "MSFT": "XLK",  "GOOGL": "XLK", "META": "XLK",
            # Financials
            "JPM":  "XLF",  "BAC": "XLF",   "GS":  "XLF",   "MS":  "XLF",
            # Energy
            "XOM":  "XLE",  "CVX": "XLE",   "COP": "XLE",   "PSX": "XLE",
            # Healthcare
            "JNJ":  "XLV",  "UNH": "XLV",   "LLY": "XLV",
            # Consumer Staples
            "WMT":  "XLP",  "COST":"XLP",
            # Consumer Discretionary
            "AMZN": "XLY",  "TSLA":"XLY",   "HD":  "XLY",
            # Industrials
            "CAT":  "XLI",  "DE":  "XLI",   "GE":  "XLI",   "HON": "XLI",
        }
        try:
            _sector_etf = _SECTOR_ETF.get(ticker.upper())
            if _sector_etf:
                import yfinance as _yf_sec
                _sec_hist = _yf_sec.download(
                    _sector_etf, period="15d", interval="1d",
                    auto_adjust=True, progress=False
                )
                _sec_c = _sec_hist["Close"].squeeze().dropna()
                if len(_sec_c) >= 6:
                    _sec_mom5 = (_sec_c.iloc[-1] / _sec_c.iloc[-6] - 1.0) * 100
                    _sec_mom1 = (_sec_c.iloc[-1] / _sec_c.iloc[-2] - 1.0) * 100
                    _sec_above_ema10 = _sec_c.iloc[-1] > _sec_c.ewm(span=10).mean().iloc[-1]
                    if _sec_mom5 > 3.0 or (_sec_mom5 > 1.5 and _sec_above_ema10):
                        _sec_score = 80.0
                        _sec_interp = f"sector {_sector_etf} +{_sec_mom5:.1f}% (5d) — strong tailwind"
                    elif _sec_mom5 > 1.0:
                        _sec_score = 68.0
                        _sec_interp = f"sector {_sector_etf} +{_sec_mom5:.1f}% (5d) — mild tailwind"
                    elif _sec_mom5 < -3.0 or (_sec_mom5 < -1.5 and not _sec_above_ema10):
                        _sec_score = 22.0
                        _sec_interp = f"sector {_sector_etf} {_sec_mom5:.1f}% (5d) — sector headwind"
                    elif _sec_mom5 < -1.0:
                        _sec_score = 36.0
                        _sec_interp = f"sector {_sector_etf} {_sec_mom5:.1f}% (5d) — mild headwind"
                    else:
                        _sec_score = 50.0
                        _sec_interp = f"sector {_sector_etf} {_sec_mom5:+.1f}% (5d) — neutral"
                    factor_scores["sector_etf_momentum"] = FactorScore(
                        name=f"Sector ETF Momentum ({_sector_etf})",
                        value=round(_sec_mom5, 2),
                        score=_sec_score,
                        interpretation=_sec_interp,
                    )
        except Exception:
            pass

        # ── 6. Composite Probability ─────────────────────────────────────
        # Blend core composite (60%) with all factor scores (40%).
        # sector_etf_momentum and momentum_5d get 1.5x weight in the avg
        # because they are leading indicators for intraday rotation.
        _weighted_scores, _weight_total = 0.0, 0.0
        _LEADING_FACTORS = {"sector_etf_momentum", "momentum_5d", "momentum"}
        for _fk, _fv in factor_scores.items():
            _w = 1.5 if _fk in _LEADING_FACTORS else 1.0
            _weighted_scores += _fv.score * _w
            _weight_total    += _w
        avg_all = _weighted_scores / _weight_total if _weight_total else 50.0
        blended = 0.6 * indicators.composite_score + 0.4 * avg_all
        prob_up = self._map_score_to_probability(blended, min_val=0, max_val=100)

        # ── VPIN microstructure overlay (P2a) ────────────────────────────
        vpin_tag = ""
        try:
            from quant_engine.vpin import compute_vpin
            vpin_res = compute_vpin(ohlcv)
            if vpin_res:
                prob_up = max(0.01, min(0.99, prob_up + vpin_res.prob_adjustment))
                factor_scores["vpin"] = FactorScore(
                    name="VPIN Microstructure",
                    value=round(vpin_res.vpin, 4),
                    score=vpin_res.score,
                    interpretation=(
                        f"VPIN={vpin_res.vpin:.3f} trend={vpin_res.vpin_trend:+.3f} "
                        f"regime={vpin_res.regime} | bucket={vpin_res.bucket_size:.0f}"
                    ),
                )
                vpin_tag = f"VPIN={vpin_res.vpin:.3f} ({vpin_res.regime}). "
        except Exception:
            pass

        # ── 0DTE options flow overlay (P2b) ─────────────────────────────
        zero_dte_tag = ""
        try:
            from quant_engine.zero_dte import compute_zero_dte
            zt_res = compute_zero_dte(
                ticker, current_price=float(indicators.current_price)
            )
            if zt_res:
                if zt_res.regime != "NEUTRAL":
                    prob_up = max(0.01, min(0.99, prob_up + zt_res.prob_adjustment))
                factor_scores["zero_dte_flow"] = FactorScore(
                    name="0DTE Options Flow",
                    value=round(zt_res.call_put_premium_ratio, 3),
                    score=zt_res.score,
                    interpretation=(
                        f"DTE={zt_res.days_to_expiry} regime={zt_res.regime} "
                        f"GEX={zt_res.net_gex:+.1f} CP={zt_res.call_put_premium_ratio:.2f} "
                        f"strikes={zt_res.n_strikes}"
                    ),
                )
                zero_dte_tag = (
                    f"0DTE: {zt_res.regime} (GEX={zt_res.net_gex:+.1f}, "
                    f"CP={zt_res.call_put_premium_ratio:.2f}). "
                )
        except Exception:
            pass

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
        direction = "BULLISH" if prob_up > self.long_threshold else "BEARISH" if prob_up < self.short_threshold else "NEUTRAL"
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
        reasoning += vpin_tag + zero_dte_tag

        return AgentResult(
            agent_name=self.name,
            probability_up=prob_up,
            confidence=confidence,
            reasoning=reasoning,
            factor_scores=factor_scores,
            warnings=indicators.warnings,
        )
