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
                bw_score = 70.0 if indicators.bb_signal == "squeeze" else 50.0 if bw < 20 else 35.0 if bw > 40 else 50.0
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
                    for _, _row in _calls.iterrows():
                        try:
                            _K = float(_row["strike"]); _iv = float(_row["impliedVolatility"]); _oi = float(_row["openInterest"])
                            if _iv > 0 and _K > 0:
                                _d1 = (_np_opt.log(_spot / _K) + (0.045 + 0.5 * _iv ** 2) * _T_gex) / (_iv * _np_opt.sqrt(_T_gex))
                                _gamma = float(_np_opt.exp(-0.5 * _d1 ** 2) / (_two_pi * _spot * _iv * _np_opt.sqrt(_T_gex)))
                                _gex_net += _gamma * _oi * 100 * _spot
                        except Exception:
                            pass
                    for _, _row in _puts.iterrows():
                        try:
                            _K = float(_row["strike"]); _iv = float(_row["impliedVolatility"]); _oi = float(_row["openInterest"])
                            if _iv > 0 and _K > 0:
                                _d1 = (_np_opt.log(_spot / _K) + (0.045 + 0.5 * _iv ** 2) * _T_gex) / (_iv * _np_opt.sqrt(_T_gex))
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

        return AgentResult(
            agent_name=self.name,
            probability_up=prob_up,
            confidence=confidence,
            reasoning=reasoning,
            factor_scores=factor_scores,
            warnings=indicators.warnings,
        )
