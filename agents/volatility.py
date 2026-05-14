"""
AlphaAgent — Volatility Agent

Forecasts volatility regime and votes on direction based on:
  1. GARCH(1,1) volatility regime (LOW / NORMAL / HIGH / EXTREME)
  2. Options Put/Call ratio (hedging vs speculation)
  3. IV vs Realized Vol ratio (variance risk premium)
  4. Kalman Filter dynamic beta + SPY correlation
"""

import numpy as np
import pandas as pd
from typing import Any

from agents.base import BaseAgent
from agents.state import AgentResult, Direction, FactorScore
from quant_engine.garch import GARCHModel
from quant_engine.options_intel import analyze_options
from quant_engine.kalman import KalmanBeta
from config.settings_manager import settings


class VolatilityAgent(BaseAgent):
    name = "volatility"

    def _run_analysis(self, ticker: str, data: Any, **kwargs) -> AgentResult:
        returns = data.get_returns("1y")
        ticker_obj = data.get_yfinance_ticker()

        if returns is None or len(returns) == 0 or ticker_obj is None:
            return self._build_error_result("Missing returns or ticker object")

        prob_up = settings.get("agent_defaults.neutral_probability", 0.5)
        confidence = 0.4
        reasoning = []
        factor_scores = {}
        sv = settings.get_section("volatility")

        # ── 1. GARCH Vol Regime ──────────────────────────────────────────
        garch = GARCHModel(returns)
        garch_res = garch.fit_and_forecast(horizon=5)

        regime_score_map = {"LOW": 70, "NORMAL": 50, "HIGH": 30, "EXTREME": 10}
        if garch_res.vol_regime == "LOW":
            prob_up += 0.10
            confidence += 0.10
            reasoning.append("GARCH: LOW vol regime — bullish drift expected.")
        elif garch_res.vol_regime == "HIGH":
            prob_up -= 0.15
            confidence += 0.15
            reasoning.append("GARCH: HIGH vol regime — bearish pressure.")
        elif garch_res.vol_regime == "EXTREME":
            prob_up -= 0.20
            confidence += 0.20
            reasoning.append("GARCH: EXTREME vol — danger zone.")
        else:
            reasoning.append("GARCH: NORMAL vol regime.")

        factor_scores["garch_regime"] = FactorScore(
            name="GARCH Vol Regime",
            value=garch_res.vol_1day,
            score=float(regime_score_map.get(garch_res.vol_regime, 50)),
            interpretation=(
                f"1-day vol: {garch_res.vol_1day:.1f}% ann. | "
                f"Regime: {garch_res.vol_regime} "
                f"({garch_res.vol_percentile:.0f}th pctl)"
            ),
        )

        # ── 2. Options Put/Call Ratio ────────────────────────────────────
        opt_res = analyze_options(ticker_obj)

        pc_bearish = sv.get("put_call_overbought", 1.2)
        pc_bullish = sv.get("put_call_bearish", 0.7)
        if opt_res.put_call_ratio > pc_bearish:
            prob_up -= 0.15
            confidence += 0.10
            reasoning.append(f"High P/C ratio ({opt_res.put_call_ratio:.2f}) — heavy hedging (bearish).")
        elif opt_res.put_call_ratio < pc_bullish:
            prob_up += 0.10
            confidence += 0.10
            reasoning.append(f"Low P/C ratio ({opt_res.put_call_ratio:.2f}) — bullish speculation.")
        else:
            reasoning.append(f"Neutral P/C ratio ({opt_res.put_call_ratio:.2f}).")

        factor_scores["put_call_ratio"] = FactorScore(
            name="Put/Call Ratio",
            value=opt_res.put_call_ratio,
            score=(30.0 if opt_res.put_call_ratio > pc_bearish
                   else 70.0 if opt_res.put_call_ratio < pc_bullish
                   else 50.0),
            interpretation=f"P/C OI Ratio: {opt_res.put_call_ratio:.2f}",
        )

        # ── 3. IV vs Realized Vol (Variance Risk Premium) ───────────────
        # implied_move_pct is the ATM straddle / price over `expiry_days` calendar days.
        # Scale GARCH daily vol to the same horizon so units match.
        if opt_res.implied_move_pct > 0 and garch_res.vol_1day_daily > 0:
            horizon = max(1, opt_res.expiry_days)
            trading_days = max(1, round(horizon * 5 / 7))  # calendar → trading days
            rv_horizon = garch_res.vol_1day_daily * np.sqrt(trading_days)
            iv_rv_ratio = opt_res.implied_move_pct / rv_horizon

            iv_rv_over = sv.get("iv_rv_overpriced", 1.5)
            iv_rv_under = sv.get("iv_rv_underpriced", 0.7)
            if iv_rv_ratio > iv_rv_over:
                prob_up += 0.05
                reasoning.append(
                    f"IV >> RV ({iv_rv_ratio:.2f}x) — options overpriced, vol crush likely (slight bullish)."
                )
            elif iv_rv_ratio < iv_rv_under:
                reasoning.append(
                    f"IV << RV ({iv_rv_ratio:.2f}x) — options cheap, volatility expansion expected."
                )
            else:
                reasoning.append(f"IV/RV balanced ({iv_rv_ratio:.2f}x).")

            factor_scores["iv_vs_rv"] = FactorScore(
                name="IV vs Realized Vol",
                value=iv_rv_ratio,
                score=60.0 if iv_rv_ratio > iv_rv_over else 40.0 if iv_rv_ratio < iv_rv_under else 50.0,
                interpretation=f"IV/RV ratio: {iv_rv_ratio:.2f}x",
            )

        # ── 4. Kalman Dynamic Beta + SPY Correlation ─────────────────────
        try:
            from data.market import MarketData
            spy_data = MarketData("SPY")
            spy_returns = spy_data.get_returns("1y")

            if len(spy_returns) > 30:
                kalman = KalmanBeta(returns, spy_returns)
                k_res = kalman.estimate()

                reasoning.append(
                    f"Dynamic Beta: {k_res.beta:.2f} | "
                    f"SPY Correlation: {k_res.spy_correlation:.2f}."
                )

                factor_scores["dynamic_beta"] = FactorScore(
                    name="Kalman Dynamic Beta",
                    value=k_res.beta,
                    score=50.0,   # Beta predicts magnitude, not direction
                    interpretation=(
                        f"Beta: {k_res.beta:.2f} | "
                        f"SPY Corr: {k_res.spy_correlation:.2f}"
                    ),
                )

                factor_scores["spy_correlation"] = FactorScore(
                    name="SPY Correlation (60d)",
                    value=k_res.spy_correlation,
                    score=50.0,
                    interpretation=f"Rolling 60d corr with SPY: {k_res.spy_correlation:.2f}",
                )
        except Exception as e:
            reasoning.append(f"Kalman beta estimation skipped ({e}).")

        # ── New: VVIX (Vol of Vol) ────────────────────────────────────────
        try:
            import yfinance as yf
            vvix_extreme = sv.get("vvix_extreme", 120)
            vvix_elevated = sv.get("vvix_elevated", 90)
            vvix_period = settings.get("data.vvix_period", "1mo")
            vvix_df = yf.download("^VVIX", period=vvix_period, interval="1d", auto_adjust=True, progress=False)
            if not vvix_df.empty:
                vvix_val = float(vvix_df["Close"].squeeze().iloc[-1])
                vvix_score = max(10.0, min(90.0, 100.0 - (vvix_val - 60) * 1.5))
                factor_scores["vvix"] = FactorScore(
                    name="VVIX (Vol of Vol Index)",
                    value=round(vvix_val, 1),
                    score=vvix_score,
                    interpretation=f"VVIX: {vvix_val:.1f} ({'extreme uncertainty' if vvix_val > vvix_extreme else 'elevated' if vvix_val > vvix_elevated else 'normal'})",
                )
                if vvix_val > vvix_extreme:
                    reasoning.append(f"VVIX {vvix_val:.1f} — extreme vol-of-vol: options market pricing in major move.")
        except Exception:
            pass

        # ── New: VIX Term Structure (VIX3M - VIX contango/backwardation) ─
        try:
            import yfinance as yf
            short_period = settings.get("data.short_period", "5d")
            vix3m_df = yf.download("^VIX3M", period=short_period, interval="1d", auto_adjust=True, progress=False)
            vix_df   = yf.download("^VIX",   period=short_period, interval="1d", auto_adjust=True, progress=False)
            if not vix3m_df.empty and not vix_df.empty:
                vix3m = float(vix3m_df["Close"].squeeze().iloc[-1])
                vix_spot = float(vix_df["Close"].squeeze().iloc[-1])
                term_struct = vix3m - vix_spot   # positive = contango (calm), negative = backwardation (panic)
                ts_score = (75.0 if term_struct > 1.0 else 50.0 if term_struct > -1.0 else 25.0)
                factor_scores["vix_term_structure"] = FactorScore(
                    name="VIX Term Structure (3M-Spot)",
                    value=round(term_struct, 2),
                    score=ts_score,
                    interpretation=f"VIX 3M-Spot: {term_struct:+.2f} ({'contango/calm' if term_struct > 0 else 'backwardation/stress'})",
                )
                if term_struct < -2.0:
                    reasoning.append(f"VIX in backwardation ({term_struct:.1f}) — acute market stress signal.")
        except Exception:
            pass

        # ── New: Realized Vol Trend (10d vs 30d) ─────────────────────────
        try:
            rv_short = sv.get("rv_short_window", 10)
            rv_long = sv.get("rv_long_window", 30)
            rv_calm = sv.get("rv_ratio_calm", 0.8)
            rv_spiking = sv.get("rv_ratio_spiking", 1.2)
            if returns is not None and len(returns) >= rv_long:
                rv_short_val = float(returns.iloc[-rv_short:].std() * (252**0.5) * 100)
                rv_long_val = float(returns.iloc[-rv_long:].std() * (252**0.5) * 100)
                rv_ratio = rv_short_val / rv_long_val if rv_long_val > 0 else 1.0
                rvt_score = (70.0 if rv_ratio < rv_calm else 50.0 if rv_ratio < rv_spiking else 25.0)
                factor_scores["realized_vol_trend"] = FactorScore(
                    name=f"Realized Vol Trend ({rv_short}d/{rv_long}d)",
                    value=round(rv_ratio, 3),
                    score=rvt_score,
                    interpretation=f"{rv_short}d RV {rv_short_val:.1f}% vs {rv_long}d RV {rv_long_val:.1f}% (ratio {rv_ratio:.2f}) — {'calming' if rv_ratio < rv_calm else 'spiking' if rv_ratio > rv_spiking else 'stable'}",
                )
        except Exception:
            pass

        # ── New: SKEW Index (Tail Risk Premium) ──────────────────────────
        try:
            import yfinance as yf
            skew_extreme = sv.get("skew_extreme", 140)
            skew_elevated = sv.get("skew_elevated", 130)
            skew_df = yf.download("^SKEW", period=short_period, interval="1d", auto_adjust=True, progress=False)
            if not skew_df.empty:
                skew_val = float(skew_df["Close"].squeeze().iloc[-1])
                skew_score = max(10.0, min(90.0, skew_elevated - skew_val))
                factor_scores["skew_index"] = FactorScore(
                    name="CBOE SKEW Index",
                    value=round(skew_val, 1),
                    score=skew_score,
                    interpretation=f"SKEW: {skew_val:.1f} ({'high tail risk' if skew_val > skew_extreme else 'elevated' if skew_val > skew_elevated else 'normal'})",
                )
        except Exception:
            pass

        # ── Clamp and vote ───────────────────────────────────────────────
        prob_up = max(0.01, min(0.99, prob_up))
        confidence = max(0.0, min(1.0, confidence))

        vote = (Direction.LONG if prob_up > self.long_threshold
                else Direction.SHORT if prob_up < self.short_threshold
                else Direction.HOLD)

        return AgentResult(
            agent_name=self.name,
            vote=vote,
            probability_up=prob_up,
            confidence=confidence,
            reasoning=" ".join(reasoning),
            factor_scores=factor_scores,
            warnings=opt_res.warnings,
        )
