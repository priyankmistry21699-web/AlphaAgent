"""
AlphaAgent — Risk & Circuit Breaker Agent

The Chief Risk Officer. Evaluates current volatility, runs EVT for Black Swan
risk, calculates position sizing using the Kelly Criterion, and enforces
hard overrides under catastrophic conditions.

Override System:
  1. BLACK SWAN (>5σ move detected)  → HALT trading (prob_up=0.1)
  2. FLASH CRASH (>7% single-day drop) → HALT trading
  3. WAR/GEO SHOCK (extreme gold/oil + VIX > 35) → cap at 35% size
  4. CARRY UNWIND (JPY surge detected) → force SHORT bias
  5. EXTREME GARCH vol → CRITICAL RISK circuit breaker
  6. HIGH GARCH vol → HIGH RISK, half-size
"""

import logging
import numpy as np
from typing import Dict, Any

import yfinance as yf

from agents.base import BaseAgent
from agents.state import AgentResult, Direction, FactorScore
from quant_engine.garch import GARCHModel
from quant_engine.monte_carlo import MonteCarloEngine
from quant_engine.evt import ExtremeValueModel
from quant_engine.kelly import KellyCriterion
from config.settings_manager import settings

logger = logging.getLogger(__name__)


def _yf_latest_change(ticker: str) -> float:
    """Get the most recent 1-day % change for a ticker."""
    try:
        import yfinance as yf
        df = yf.download(ticker, period="5d", interval="1d",
                         auto_adjust=True, progress=False)
        s = df["Close"].squeeze().dropna()
        if len(s) >= 2:
            return float((s.iloc[-1] / s.iloc[-2] - 1) * 100)
    except Exception:
        pass
    return 0.0


class RiskAgent(BaseAgent):
    name = "risk"

    def _run_analysis(self, ticker: str, data: Any, **kwargs) -> AgentResult:
        prices = data.get_ohlcv("1y")["Close"]
        returns = data.get_returns("1y")
        current_price = data.get_current_price()

        if prices.empty or len(returns) == 0:
            return self._build_error_result("Missing price data")

        # ── 1. GARCH Volatility ───────────────────────────────────────────
        garch = GARCHModel(returns)
        garch_res = garch.fit_and_forecast(horizon=5)

        # ── 2. Monte Carlo — paths scale with GARCH vol regime ────────────
        _PATHS_BY_REGIME = {"LOW": 3_000, "NORMAL": 5_000, "HIGH": 8_000, "EXTREME": 10_000}
        num_paths = _PATHS_BY_REGIME.get(garch_res.vol_regime, 5_000)
        mc = MonteCarloEngine(current_price)
        mc_res = mc.simulate_garch(
            days=5,
            drift_daily=0.0005,
            garch_forecasts=garch_res.forecast_daily,
            num_paths=num_paths,
        )

        # ── 3. Extreme Value Theory ────────────────────────────────────────
        evt = ExtremeValueModel(returns, threshold_percentile=0.10)
        evt_res = evt.fit_and_calculate()

        # ── 4. Kelly Criterion — inputs derived from Monte Carlo result ────
        # prob_win = fraction of MC paths ending above current price
        # win/loss = symmetric 1-std-dev move from 68% CI
        # prob_above_current is in [0, 100] (percentage) — convert to fraction
        mc_prob_win = float(mc_res.prob_above_current) / 100.0
        mc_expected_win = max(0.001, (mc_res.ci_68.high - current_price) / current_price)
        mc_expected_loss = max(0.001, abs(mc_res.ci_68.low - current_price) / current_price)

        kelly = KellyCriterion(
            prob_win=mc_prob_win,
            expected_win_pct=mc_expected_win,
            expected_loss_pct=mc_expected_loss,
        )
        kelly_res = kelly.calculate(current_volatility=garch_res.vol_1day_daily,
                                    vol_regime=garch_res.vol_regime)

        prob_up = 0.5
        confidence = 1.0
        multiplier = 1.0
        reasoning = []
        warnings = []
        factor_scores = {}
        override_triggered = None

        # ── Base Vol Assessment ────────────────────────────────────────────
        reasoning.append(f"Risk Regime: {garch_res.vol_regime}.")
        reasoning.append(f"Expected 1-day Vol: {garch_res.vol_1day:.1f}% (annualized).")

        worst_case_drop_pct = (mc_res.ci_95.low - current_price) / current_price * 100
        reasoning.append(
            f"Monte Carlo 95% CI low (5d): ${mc_res.ci_95.low:.2f} ({worst_case_drop_pct:.1f}%)."
        )
        reasoning.append(f"EVT 99% VaR (1-in-100 day): {evt_res.var_99 * 100:+.1f}%.")

        factor_scores["garch_vol"] = FactorScore(
            name="GARCH Volatility Regime",
            value=garch_res.vol_1day,
            score={"LOW": 75.0, "NORMAL": 50.0, "HIGH": 25.0, "EXTREME": 5.0}.get(
                garch_res.vol_regime, 50.0
            ),
            interpretation=f"{garch_res.vol_regime} regime | {garch_res.vol_1day:.1f}% ann.",
        )

        factor_scores["evt_var"] = FactorScore(
            name="EVT Tail Risk (99% VaR)",
            value=evt_res.var_99 * 100,
            score=(80.0 if evt_res.var_99 > -0.02
                   else 40.0 if evt_res.var_99 > -0.05
                   else 10.0),
            interpretation=f"VaR99: {evt_res.var_99 * 100:+.1f}% | CVaR99: {evt_res.cvar_99 * 100:+.1f}%",
        )

        factor_scores["monte_carlo"] = FactorScore(
            name="Monte Carlo 95% CI (5d)",
            value=worst_case_drop_pct,
            score=(75.0 if worst_case_drop_pct > -3
                   else 40.0 if worst_case_drop_pct > -7
                   else 10.0),
            interpretation=f"5d 95% low: ${mc_res.ci_95.low:.2f} ({worst_case_drop_pct:.1f}%)",
        )

        factor_scores["kelly"] = FactorScore(
            name="Kelly Position Size",
            value=kelly_res.half_kelly * 100,
            score=50.0,
            interpretation=f"Half-Kelly: {kelly_res.half_kelly * 100:.1f}% | Vol-adj: {kelly_res.vol_adjusted_kelly * 100:.1f}%",
        )

        # ── Override 1: BLACK SWAN (>Nσ daily move) ───────────────────────
        try:
            _sr = settings.get_section("risk")
            _bs_sigma = _sr.get("black_swan_sigma", 5.0)
            recent_returns = returns.iloc[-5:]
            vol = float(returns.std())
            if vol > 0:
                z_scores = recent_returns / vol
                max_z = float(abs(z_scores).max())
                if max_z > _bs_sigma:
                    override_triggered = "BLACK_SWAN"
                    prob_up = _sr.get("black_swan_prob_up", 0.10)
                    multiplier = 0.0
                    confidence = 1.0
                    msg = f"BLACK SWAN DETECTED: {max_z:.1f}σ move in recent 5 sessions. HALTING all trades."
                    reasoning.append(msg)
                    warnings.append(msg)
                    factor_scores["black_swan"] = FactorScore(
                        name="Black Swan Detection",
                        value=max_z,
                        score=0.0,
                        interpretation=f"Max |Z-score| in 5d: {max_z:.2f}σ",
                    )
        except Exception as e:
            reasoning.append(f"Black swan detection error ({e}).")

        # ── Override 2: FLASH CRASH (vol-normalized: 3× daily realized vol) ────
        if override_triggered is None:
            try:
                # Dynamic threshold: 3× realized daily vol (not fixed %)
                # In low-vol markets a 3% drop IS a flash crash; in high-vol it's noise
                _fc_ticker_static = _sr.get("flash_crash_ticker_pct", 7.0)
                _fc_spy_static    = _sr.get("flash_crash_spy_pct", 5.0)
                try:
                    import numpy as _np_fc
                    import yfinance as _yf_fc
                    _fc_hist = _yf_fc.download(ticker, period="60d", interval="1d",
                                               auto_adjust=True, progress=False)
                    if not _fc_hist.empty and len(_fc_hist) >= 20:
                        _fc_rets = _fc_hist["Close"].squeeze().dropna().pct_change().dropna()
                        _fc_daily_vol = float(_fc_rets.tail(20).std()) * 100
                        _fc_ticker = max(3.0, min(_fc_ticker_static, _fc_daily_vol * 3))
                    else:
                        _fc_ticker = _fc_ticker_static
                    _spy_hist = _yf_fc.download("SPY", period="60d", interval="1d",
                                                auto_adjust=True, progress=False)
                    if not _spy_hist.empty and len(_spy_hist) >= 20:
                        _spy_rets = _spy_hist["Close"].squeeze().dropna().pct_change().dropna()
                        _spy_daily_vol = float(_spy_rets.tail(20).std()) * 100
                        _fc_spy = max(2.0, min(_fc_spy_static, _spy_daily_vol * 3))
                    else:
                        _fc_spy = _fc_spy_static
                except Exception:
                    _fc_ticker = _fc_ticker_static
                    _fc_spy    = _fc_spy_static

                spy_1d = _yf_latest_change("SPY")
                ticker_1d = _yf_latest_change(ticker)
                if ticker_1d < -_fc_ticker or spy_1d < -_fc_spy:
                    override_triggered = "FLASH_CRASH"
                    prob_up = 0.15
                    multiplier = 0.0
                    msg = (
                        f"FLASH CRASH SIGNAL: {ticker} 1d={ticker_1d:.1f}% | "
                        f"SPY 1d={spy_1d:.1f}%. HALTING trades."
                    )
                    reasoning.append(msg)
                    warnings.append(msg)
                    factor_scores["flash_crash"] = FactorScore(
                        name="Flash Crash Detection",
                        value=ticker_1d,
                        score=0.0,
                        interpretation=f"{ticker} 1d: {ticker_1d:.1f}% | SPY 1d: {spy_1d:.1f}%",
                    )
            except Exception as e:
                reasoning.append(f"Flash crash detection error ({e}).")

        # ── Override 3: WAR/GEO SHOCK (VIX > 35 + gold surging) ──────────
        if override_triggered is None:
            try:
                vix_df = yf.download("^VIX", period="5d", interval="1d",
                                     auto_adjust=True, progress=False)
                vix_now = float(vix_df["Close"].squeeze().dropna().iloc[-1])
                gold_chg = _yf_latest_change("GLD")
                oil_chg = _yf_latest_change("XLE")

                geo_stress = (vix_now > _sr.get("geo_shock_vix", 35.0)) and (gold_chg > _sr.get("geo_shock_gold_pct", 2.0) or oil_chg > _sr.get("geo_shock_oil_pct", 5.0))
                if geo_stress:
                    override_triggered = "GEO_SHOCK"
                    multiplier = _sr.get("geo_shock_multiplier", 0.35)
                    prob_up = _sr.get("geo_shock_prob_up", 0.30)
                    msg = (
                        f"WAR/GEO SHOCK: VIX={vix_now:.1f} + Gold {gold_chg:+.1f}%/Oil {oil_chg:+.1f}%. "
                        "Position size capped at 35%."
                    )
                    reasoning.append(msg)
                    warnings.append(msg)
                    factor_scores["geo_shock"] = FactorScore(
                        name="Geopolitical Shock Signal",
                        value=vix_now,
                        score=5.0,
                        interpretation=f"VIX={vix_now:.1f} | Gold {gold_chg:+.1f}% | Oil {oil_chg:+.1f}%",
                    )
            except Exception as e:
                reasoning.append(f"Geo shock detection error ({e}).")

        # ── Override 4: CARRY UNWIND (JPY < 125) ──────────────────────────
        if override_triggered is None:
            try:
                jpy_df = yf.download("JPY=X", period="5d", interval="1d",
                                     auto_adjust=True, progress=False)
                usdjpy = float(jpy_df["Close"].squeeze().dropna().iloc[-1])
                jpy_1d = _yf_latest_change("JPY=X")

                if usdjpy < _sr.get("carry_unwind_usdjpy", 125.0) or jpy_1d < -_sr.get("carry_unwind_yen_pct", 1.5):
                    override_triggered = "CARRY_UNWIND"
                    prob_up = _sr.get("geo_shock_prob_up", 0.30)
                    multiplier = _sr.get("carry_unwind_multiplier", 0.50)
                    msg = (
                        f"CARRY TRADE UNWIND: USD/JPY={usdjpy:.1f} (1d: {jpy_1d:.1f}%) — "
                        "yen surging, forced SHORT bias."
                    )
                    reasoning.append(msg)
                    warnings.append(msg)
                    factor_scores["carry_unwind"] = FactorScore(
                        name="Carry Trade Unwind",
                        value=usdjpy,
                        score=10.0,
                        interpretation=f"USD/JPY: {usdjpy:.1f} | 1d change: {jpy_1d:.1f}%",
                    )
            except Exception as e:
                reasoning.append(f"Carry unwind detection error ({e}).")

        # ── Standard GARCH Circuit Breakers ───────────────────────────────
        if override_triggered is None:
            if garch_res.vol_regime == "EXTREME" or evt_res.var_99 < -0.08:
                prob_up = _sr.get("extreme_vol_prob_up", 0.20)
                multiplier = 0.25   # reduced from 0.0 — still very conservative but allows high-conviction signals
                reasoning.append("CRITICAL RISK: Extreme volatility. Position sizes reduced to 25%.")
                warnings.append("CRITICAL RISK: Extreme volatility — positions capped at 25%.")
            elif garch_res.vol_regime == "HIGH" or evt_res.var_95 < -0.05:
                prob_up = _sr.get("high_vol_prob_up", 0.40)
                multiplier = _sr.get("high_vol_multiplier", 0.50)
                reasoning.append("HIGH RISK: Volatility elevated. Reducing position sizes by 50%.")
            else:
                prob_up = 0.50
                multiplier = 1.0
                reasoning.append("Risk environment normal. Full position sizing permitted.")

        # ── New: KL Divergence (Return Distribution Shift) ───────────────
        try:
            if returns is not None and len(returns) >= 60:
                import numpy as _np
                # Compare recent 20-day return distribution to 252-day baseline via KL divergence
                all_ret  = returns.dropna().values
                recent   = all_ret[-20:]
                baseline = all_ret[-252:] if len(all_ret) >= 252 else all_ret

                # Discretize into 20 equal-width bins spanning min/max
                lo, hi = float(_np.min(baseline)), float(_np.max(baseline))
                bins   = _np.linspace(lo, hi, 21)
                p_recent   = _np.histogram(recent,   bins=bins, density=False)[0].astype(float) + 1e-10
                p_baseline = _np.histogram(baseline, bins=bins, density=False)[0].astype(float) + 1e-10
                p_recent   /= p_recent.sum()
                p_baseline /= p_baseline.sum()

                kl_div = float(_np.sum(p_recent * _np.log(p_recent / p_baseline)))
                # High KL = current regime very different from historical = regime shift risk
                kl_score = max(10.0, min(90.0, 70.0 - kl_div * 30.0))
                _kl_thresh = settings.get("risk.kl_divergence_threshold", 1.0)
                factor_scores["kl_divergence"] = FactorScore(
                    name="KL Divergence (Regime Shift)",
                    value=round(kl_div, 4),
                    score=kl_score,
                    interpretation=(
                        f"KL(recent‖baseline): {kl_div:.3f} "
                        f"({'REGIME SHIFT — return distribution abnormal' if kl_div > _kl_thresh else 'elevated distributional drift' if kl_div > 0.4 else 'stable regime'})"
                    ),
                )
                if kl_div > _kl_thresh:
                    warnings.append(f"KL Divergence {kl_div:.2f} — return distribution has shifted dramatically from baseline (regime break risk).")
        except Exception:
            pass

        # ── New: Hawkes Process (Self-Exciting Jump Intensity) ────────────
        try:
            from quant_engine.hawkes import HawkesProcess
            if returns is not None and len(returns) >= 60:
                hwk = HawkesProcess()
                hwk_res = hwk.fit(returns.dropna())
                br = hwk_res.branching_ratio   # α/β: closer to 1 = cascade risk
                hwk_score = max(10.0, min(90.0, 70.0 - br * 60.0))
                factor_scores["hawkes_intensity"] = FactorScore(
                    name="Hawkes Branching Ratio (Jump Cascade)",
                    value=round(br, 4),
                    score=hwk_score,
                    interpretation=(
                        f"α/β={br:.3f} | μ={hwk_res.mu:.4f} | events={hwk_res.n_events} "
                        f"({'near-critical: cascade risk' if br > 0.9 else 'self-exciting regime' if br > 0.7 else 'stable'})"
                    ),
                )
                if br > 0.9:
                    warnings.append(f"Hawkes branching ratio {br:.2f} — near-critical: one large move likely to cascade into more (HFT/dark pool activity).")
        except Exception:
            pass

        # ── New: Quasi-Monte Carlo (Sobol Sequence VaR) ───────────────────
        try:
            from quant_engine.quasi_mc import QuasiMonteCarloEngine
            qmc_engine = QuasiMonteCarloEngine(current_price)
            qmc_res = qmc_engine.simulate_gbm(
                days=5,
                drift_daily=0.0005,
                vol_daily=garch_res.vol_1day_daily,
                n_paths=4096,
            )
            qmc_worst = (qmc_res.ci_95_low - current_price) / current_price * 100
            qmc_score = (75.0 if qmc_worst > -3 else 40.0 if qmc_worst > -7 else 10.0)
            factor_scores["quasi_mc_var"] = FactorScore(
                name="Quasi-MC VaR (Sobol)",
                value=round(qmc_worst, 2),
                score=qmc_score,
                interpretation=f"QMC 5d 95% low: ${qmc_res.ci_95_low:.2f} ({qmc_worst:.1f}%) — Sobol low-discrepancy sampling",
            )
        except Exception:
            pass

        # ── New: Liquidity Risk (Volume vs 20D Avg) ───────────────────────
        try:
            import yfinance as yf
            sr = settings.get_section("risk")
            liq_high = sr.get("liquidity_ratio_high", 1.5)
            liq_low = sr.get("liquidity_ratio_low", 0.7)
            min_rolling = settings.get("data.min_rolling_days", 20)
            medium_period = settings.get("data.medium_period", "3mo")
            hist = yf.Ticker(ticker).history(period=medium_period, interval="1d")
            if not hist.empty and len(hist) >= min_rolling:
                avg_vol = float(hist["Volume"].iloc[-min_rolling:].mean())
                last_vol = float(hist["Volume"].iloc[-1])
                liq_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0
                liq_score = (80.0 if liq_ratio > liq_high else 55.0 if liq_ratio > liq_low else 30.0)
                factor_scores["liquidity_risk"] = FactorScore(
                    name="Liquidity Risk (Vol Ratio)",
                    value=round(liq_ratio, 3),
                    score=liq_score,
                    interpretation=f"Volume {liq_ratio:.2f}x {min_rolling}D avg — {'liquid' if liq_ratio > 1.0 else 'thin/illiquid'}",
                )
        except Exception:
            pass

        # ── New: Drawdown from ATH ────────────────────────────────────────
        try:
            import yfinance as yf
            dd_warn_pct = sr.get("drawdown_warning_pct", 30)
            long_period = settings.get("data.long_period", "2y")
            hist2 = yf.Ticker(ticker).history(period=long_period, interval="1d")
            if not hist2.empty:
                ath = float(hist2["High"].max())
                last_c = float(hist2["Close"].iloc[-1])
                dd_from_ath = (last_c - ath) / ath * 100
                dd_score = max(10.0, min(90.0, 90.0 + dd_from_ath * 1.5))
                factor_scores["drawdown_ath"] = FactorScore(
                    name="Drawdown from ATH",
                    value=round(dd_from_ath, 2),
                    score=dd_score,
                    interpretation=f"{dd_from_ath:.1f}% below {long_period} ATH (${ath:.2f})",
                )
                if dd_from_ath < -dd_warn_pct:
                    warnings.append(f"Stock is {abs(dd_from_ath):.0f}% below {long_period} ATH — deep drawdown.")
        except Exception:
            pass

        # ── New: Tail Ratio (Upside/Downside Asymmetry) ───────────────────
        try:
            tr_positive = sr.get("tail_ratio_positive", 1.2)
            tr_negative = sr.get("tail_ratio_negative", 0.8)
            min_tail_days = settings.get("data.min_tail_days", 60)
            if returns is not None and len(returns) >= min_tail_days:
                ret_arr = returns.dropna().values
                p95 = float(np.percentile(ret_arr, 95))
                p5  = abs(float(np.percentile(ret_arr, 5)))
                tail_ratio = p95 / p5 if p5 > 0 else 1.0
                tr_score = (80.0 if tail_ratio > tr_positive else 50.0 if tail_ratio > tr_negative else 25.0)
                factor_scores["tail_ratio"] = FactorScore(
                    name="Tail Ratio (Up/Down Asymmetry)",
                    value=round(tail_ratio, 3),
                    score=tr_score,
                    interpretation=f"95th/5th percentile: {tail_ratio:.2f} ({'positive skew' if tail_ratio > 1.0 else 'negative skew'})",
                )
        except Exception:
            pass

        # ── New: Correlation Regime (Rolling 60D vs 252D) ─────────────────
        try:
            import yfinance as yf
            corr_min = sr.get("correlation_min_days", 60)
            corr_long = sr.get("correlation_long_days", 252)
            corr_delta_thresh = sr.get("correlation_delta_threshold", 0.1)
            spy_hist = yf.download("SPY", period=long_period, interval="1d", auto_adjust=True, progress=False)
            tkr_hist = yf.download(ticker, period=long_period, interval="1d", auto_adjust=True, progress=False)
            if not spy_hist.empty and not tkr_hist.empty:
                spy_ret = spy_hist["Close"].squeeze().pct_change().dropna()
                tkr_ret = tkr_hist["Close"].squeeze().pct_change().dropna()
                aligned = tkr_ret.align(spy_ret, join="inner")[0]
                spy_aligned = spy_ret.reindex(aligned.index)
                if len(aligned) >= corr_min:
                    corr_60  = float(aligned.iloc[-corr_min:].corr(spy_aligned.iloc[-corr_min:]))
                    corr_252 = float(aligned.corr(spy_aligned)) if len(aligned) >= corr_long else corr_60
                    corr_delta = corr_60 - corr_252
                    creg_score = (70.0 if corr_delta < -corr_delta_thresh else 50.0 if abs(corr_delta) < corr_delta_thresh else 30.0)
                    factor_scores["correlation_regime"] = FactorScore(
                        name="Correlation Regime (vs SPY)",
                        value=round(corr_60, 3),
                        score=creg_score,
                        interpretation=f"{corr_min}D corr: {corr_60:.2f} vs {corr_long}D: {corr_252:.2f} (Δ{corr_delta:+.2f}) — {'decorrelating' if corr_delta < -corr_delta_thresh else 'correlating' if corr_delta > corr_delta_thresh else 'stable'}",
                    )
        except Exception:
            pass

        # ── New: Yang-Zhang Efficient Vol (OHLC-based, 7-8x more efficient) ──
        try:
            from quant_engine.vol_estimators import yang_zhang_vol
            _ohlcv_yz = data.get_ohlcv(period="3mo")
            if _ohlcv_yz is not None and len(_ohlcv_yz) >= 25:
                _yz_vol = yang_zhang_vol(_ohlcv_yz["Open"], _ohlcv_yz["High"],
                                         _ohlcv_yz["Low"], _ohlcv_yz["Close"], 20)
                _cc_vol = float(_ohlcv_yz["Close"].pct_change().tail(20).std() * (252**0.5))
                if _yz_vol and _cc_vol > 0:
                    _yz_gap = (_yz_vol / _cc_vol - 1) * 100
                    _yz_score = (35.0 if _yz_vol > 0.50 else
                                 50.0 if _yz_vol > 0.30 else
                                 65.0 if _yz_vol > 0.15 else 70.0)
                    factor_scores["yang_zhang_efficient_vol"] = FactorScore(
                        name="Yang-Zhang Efficient Vol (OHLC)",
                        value=round(_yz_vol * 100, 2),
                        score=_yz_score,
                        interpretation=(
                            f"Yang-Zhang vol: {_yz_vol*100:.1f}% | Close-to-close: {_cc_vol*100:.1f}% | gap: {_yz_gap:+.1f}% — "
                            f"{'OHLC reveals higher vol than CC suggests' if _yz_gap > 5 else 'CC vol overstates' if _yz_gap < -5 else 'estimators agree'}"
                        ),
                    )
                    if _yz_vol > 0.50:
                        warnings.append(f"Yang-Zhang vol {_yz_vol*100:.1f}% — extreme volatility from OHLC range.")
        except Exception:
            pass

        # ── New: CUSUM Structural-Break Detector on Returns ──────────────────
        try:
            from quant_engine.structural_break import CUSUMDetector
            if returns is not None and len(returns) >= 60:
                _cu_res = CUSUMDetector(threshold_k=0.5, threshold_h=5.0).detect(returns.tail(120))
                if _cu_res is not None:
                    _cu_score = 25.0 if _cu_res.break_detected else 60.0
                    factor_scores["cusum_structural_break"] = FactorScore(
                        name="CUSUM Structural Break",
                        value=_cu_res.cumsum_max,
                        score=_cu_score,
                        interpretation=(
                            f"CUSUM max={_cu_res.cumsum_max:.2f} (threshold {_cu_res.cumsum_threshold:.1f}) | "
                            f"direction: {_cu_res.direction} — "
                            f"{'STRUCTURAL BREAK detected: model parameters may have shifted' if _cu_res.break_detected else 'parameters stable'}"
                        ),
                    )
                    if _cu_res.break_detected:
                        warnings.append(f"CUSUM detected return-distribution break ({_cu_res.direction}, max={_cu_res.cumsum_max:.2f}).")
        except Exception:
            pass

        # ── New: Correlation Network Centrality (systemic linkage) ───────────
        # High centrality stocks crash hardest in selloffs (think GE/C in 2008)
        try:
            import numpy as _np_cn
            import yfinance as _yf_cn
            _peers = ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLY", "XLI", ticker]
            _hist_cn = _yf_cn.download(_peers, period="6mo", interval="1d",
                                       auto_adjust=True, progress=False, group_by="ticker")
            _rets_mat = []
            _names = []
            for _p in _peers:
                try:
                    _c = _hist_cn[_p]["Close"].dropna() if len(_peers) > 1 else _hist_cn["Close"].dropna()
                    _r = _c.pct_change().dropna()
                    if len(_r) >= 60:
                        _rets_mat.append(_r.tail(60).values)
                        _names.append(_p)
                except Exception:
                    continue
            if len(_rets_mat) >= 5 and ticker in _names:
                _min_len = min(len(_r) for _r in _rets_mat)
                _M = _np_cn.array([r[-_min_len:] for r in _rets_mat])
                _corr = _np_cn.corrcoef(_M)
                _corr = _np_cn.nan_to_num(_corr, nan=0.0)
                # Eigenvector centrality: dominant eigenvector of abs(corr)
                _abs_corr = _np_cn.abs(_corr)
                _eigvals, _eigvecs = _np_cn.linalg.eigh(_abs_corr)
                _centrality = _np_cn.abs(_eigvecs[:, -1])   # dominant eigenvector
                _t_idx = _names.index(ticker)
                _tk_cent = float(_centrality[_t_idx] / _centrality.max())
                _cn_score = (30.0 if _tk_cent > 0.85 else   # high centrality = systemic risk
                             45.0 if _tk_cent > 0.65 else
                             65.0 if _tk_cent > 0.40 else 70.0)
                factor_scores["network_centrality"] = FactorScore(
                    name="Correlation Network Centrality",
                    value=round(_tk_cent, 3),
                    score=_cn_score,
                    interpretation=(
                        f"Network centrality: {_tk_cent:.3f} (vs {len(_names)} peers) — "
                        f"{'highly central: crashes hardest in selloff' if _tk_cent > 0.85 else 'above-avg systemic linkage' if _tk_cent > 0.65 else 'moderate' if _tk_cent > 0.40 else 'peripheral: idiosyncratic'}"
                    ),
                )
                if _tk_cent > 0.85:
                    warnings.append(f"High network centrality ({_tk_cent:.2f}) — amplified systemic crash risk.")
        except Exception:
            pass

        # ── New: DCC-GARCH Dynamic Correlation ───────────────────────────────
        try:
            from quant_engine.dcc_garch import DCCGarch
            import yfinance as _yf_dcc
            _spy_dcc = _yf_dcc.download("SPY", period="1y", interval="1d",
                                        auto_adjust=True, progress=False)
            _tkr_dcc = _yf_dcc.download(ticker, period="1y", interval="1d",
                                        auto_adjust=True, progress=False)
            if not _spy_dcc.empty and not _tkr_dcc.empty:
                _spy_r = _spy_dcc["Close"].squeeze().pct_change().dropna()
                _tkr_r = _tkr_dcc["Close"].squeeze().pct_change().dropna()
                _dcc_res = DCCGarch().fit(_tkr_r, _spy_r)
                if _dcc_res is not None:
                    _dcc_score = (30.0 if _dcc_res.crisis_flag else
                                  40.0 if _dcc_res.corr_trend > 0.08 else
                                  60.0 if _dcc_res.corr_trend < -0.05 else 50.0)
                    factor_scores["dcc_garch_corr"] = FactorScore(
                        name="DCC-GARCH Dynamic Correlation",
                        value=round(_dcc_res.current_corr, 3),
                        score=_dcc_score,
                        interpretation=(
                            f"DCC corr: {_dcc_res.current_corr:.3f} (baseline {_dcc_res.baseline_corr:.3f}) | "
                            f"trend: {_dcc_res.corr_trend:+.3f} | pct: {_dcc_res.corr_percentile:.0f}th — "
                            f"{'CRISIS: corr spike, systematic risk rising' if _dcc_res.crisis_flag else 'corr rising: systemic linkage increasing' if _dcc_res.corr_trend > 0.08 else 'corr falling: decorrelating, idiosyncratic' if _dcc_res.corr_trend < -0.05 else 'stable correlation'}"
                        ),
                    )
                    if _dcc_res.crisis_flag:
                        warnings.append(f"DCC-GARCH: correlation spike ({_dcc_res.corr_trend:+.3f}) — systemic risk contagion signal.")
        except Exception:
            pass

        # ── New: Rolling Sharpe + Sortino Ratio (63-day) ─────────────────
        try:
            _ohlcv_ss = data.get_ohlcv(period="1y")
            _ret_ss   = _ohlcv_ss["Close"].pct_change().dropna()
            if len(_ret_ss) >= 63:
                _r63      = _ret_ss.iloc[-63:]
                _mu63     = float(_r63.mean())
                _std63    = float(_r63.std())
                _neg_ret  = _r63[_r63 < 0]
                _dsv63    = float(_neg_ret.std()) if len(_neg_ret) > 1 else _std63
                _ann      = float(np.sqrt(252))
                _sharpe   = (_mu63 / _std63 * _ann) if _std63 > 0 else 0.0
                _sortino  = (_mu63 / _dsv63 * _ann) if _dsv63 > 0 else 0.0
                _sh_score = (85.0 if _sharpe > 1.5 else
                             70.0 if _sharpe > 0.5 else
                             50.0 if _sharpe > 0.0 else
                             35.0 if _sharpe > -0.5 else
                             15.0)
                factor_scores["rolling_sharpe"] = FactorScore(
                    name="Rolling Sharpe (63d)",
                    value=round(_sharpe, 3),
                    score=_sh_score,
                    interpretation=(
                        f"63d Sharpe {_sharpe:.2f} | Sortino {_sortino:.2f} — "
                        f"{'strong risk-adj return' if _sharpe > 1.5 else 'adequate' if _sharpe > 0.5 else 'negative — risk not rewarded' if _sharpe < 0 else 'marginal'}"
                    ),
                )
                _so_score = (85.0 if _sortino > 2.0 else
                             70.0 if _sortino > 1.0 else
                             52.0 if _sortino > 0.0 else
                             28.0)
                factor_scores["rolling_sortino"] = FactorScore(
                    name="Rolling Sortino (63d)",
                    value=round(_sortino, 3),
                    score=_so_score,
                    interpretation=(
                        f"63d Sortino {_sortino:.2f} (downside σ {_dsv63*100:.2f}%/day) — "
                        f"{'excellent downside protection' if _sortino > 2.0 else 'good' if _sortino > 1.0 else 'average' if _sortino > 0.5 else 'poor'}"
                    ),
                )
        except Exception:
            pass

        # ── New: Vanna / Charm Exposure (Options Greek Surface) ──────────────
        try:
            import math as _math_vc
            _tkr_vc  = yf.Ticker(ticker)
            _vc_exps = _tkr_vc.options
            if _vc_exps:
                _chain_vc = _tkr_vc.option_chain(_vc_exps[0])
                _calls_vc = _chain_vc.calls
                import datetime as _dt_vc
                _days_exp = max(1, (_dt_vc.datetime.strptime(_vc_exps[0], "%Y-%m-%d") - _dt_vc.datetime.now()).days)
                _T_vc = _days_exp / 365.0
                _r_vc = settings.get("backtest.risk_free_rate", 0.05)
                _S_vc = float(current_price)
                _total_vanna, _total_charm, _n_vc = 0.0, 0.0, 0
                for _, _row_vc in (_calls_vc.iterrows() if _calls_vc is not None and not _calls_vc.empty else iter([])):
                    _K  = float(_row_vc.get("strike", 0) or 0)
                    _IV = float(_row_vc.get("impliedVolatility", 0.3) or 0.3)
                    _OI = float(_row_vc.get("openInterest", 0) or 0)
                    if _K <= 0 or _IV < 0.01 or _OI < 10:
                        continue
                    _d1 = (_math_vc.log(_S_vc / _K) + (_r_vc + 0.5 * _IV ** 2) * _T_vc) / (_IV * _math_vc.sqrt(_T_vc))
                    _d2 = _d1 - _IV * _math_vc.sqrt(_T_vc)
                    _pdf1 = _math_vc.exp(-0.5 * _d1 ** 2) / _math_vc.sqrt(2 * _math_vc.pi)
                    _vanna_c = -_pdf1 * _d2 / _IV
                    _charm_c = (-_pdf1 * (2 * _r_vc * _T_vc - _d2 * _IV * _math_vc.sqrt(_T_vc))
                                / (2 * _T_vc * _IV * _math_vc.sqrt(_T_vc)))
                    _total_vanna += _vanna_c * _OI
                    _total_charm += _charm_c * _OI
                    _n_vc += 1
                if _n_vc > 0:
                    _vc_score = (65.0 if _total_vanna > 0 else 35.0)
                    factor_scores["vanna_charm"] = FactorScore(
                        name="Vanna/Charm Exposure",
                        value=round(_total_vanna, 4),
                        score=_vc_score,
                        interpretation=(
                            f"Net Vanna: {_total_vanna:+.2f} | Net Charm: {_total_charm:+.2f} "
                            f"over {_n_vc} strikes — "
                            f"{'dealer hedging supports price on IV drop (positive vanna)' if _total_vanna > 0 else 'negative vanna — dealer selling amplifies any vol spike'}"
                        ),
                    )
                    if _total_vanna < -_sr.get("vanna_negative_threshold", 0.5) and _total_charm < 0:
                        warnings.append("Negative vanna + charm: dealer hedging flows may amplify downside on vol expansion.")
        except Exception:
            pass

        # ── New: Return Distribution Skewness & Excess Kurtosis ─────────
        try:
            import numpy as _np_sk
            if returns is not None and len(returns) >= 60:
                _ret_arr = _np_sk.array(returns.dropna(), dtype=float)
                _n_sk    = len(_ret_arr)
                _mean_sk = float(_np_sk.mean(_ret_arr))
                _std_sk  = float(_np_sk.std(_ret_arr))
                if _std_sk > 0:
                    _skew = float(_np_sk.mean((_ret_arr - _mean_sk) ** 3) / _std_sk ** 3)
                    _kurt = float(_np_sk.mean((_ret_arr - _mean_sk) ** 4) / _std_sk ** 4 - 3)
                    _sk_score = (65.0 if _skew > 0 else 40.0 if _skew > -1.0 else 20.0)
                    factor_scores["skew_kurtosis"] = FactorScore(
                        name="Return Skewness & Kurtosis",
                        value=round(_skew, 3),
                        score=_sk_score,
                        interpretation=(
                            f"Skew: {_skew:+.3f} | Excess kurt: {_kurt:+.3f} — "
                            f"{'positive skew (right tail)' if _skew > 0.5 else 'negative skew — left tail risk' if _skew < -0.5 else 'near-symmetric'}"
                            f"{' | FAT TAILS (leptokurtic)' if _kurt > 3.0 else ''}"
                        ),
                    )
                    if _skew < -1.0:
                        warnings.append(f"Negative skew {_skew:.2f} — distribution has heavy left tail (crash risk).")
                    if _kurt > 5.0:
                        warnings.append(f"Excess kurtosis {_kurt:.1f} — fat tails, extreme moves more probable than normal.")
        except Exception:
            pass

        # ── New: Options GEX Wall (Gamma Exposure) ───────────────────────
        try:
            import yfinance as _yf_gex
            import math as _math_gex
            _opt_gex = _yf_gex.Ticker(ticker)
            _exp_gex = (_opt_gex.options or [])[:2]
            _S_gex   = float((_opt_gex.history(period="1d")["Close"].iloc[-1]))
            _r_gex   = settings.get("backtest.risk_free_rate", 0.05)
            _total_gex, _n_gex = 0.0, 0
            for _exp in _exp_gex:
                try:
                    _chain = _opt_gex.option_chain(_exp)
                    for _df_g, _is_call in [(_chain.calls, True), (_chain.puts, False)]:
                        for _, _row_g in _df_g.iterrows():
                            _K  = float(_row_g.get("strike", 0))
                            _IV = float(_row_g.get("impliedVolatility", 0))
                            _OI = float(_row_g.get("openInterest", 0) or 0)
                            if _K <= 0 or _IV <= 0 or _OI <= 0:
                                continue
                            _T_g = max(0.01, (len(_exp) and 0.1))
                            _d1_g = (_math_gex.log(_S_gex / _K) + (_r_gex + 0.5 * _IV ** 2) * _T_g) / (_IV * _math_gex.sqrt(_T_g))
                            _pdf_g = _math_gex.exp(-0.5 * _d1_g ** 2) / _math_gex.sqrt(2 * _math_gex.pi)
                            _gamma_g = _pdf_g / (_S_gex * _IV * _math_gex.sqrt(_T_g))
                            _sign_g = 1.0 if _is_call else -1.0
                            _total_gex += _sign_g * _gamma_g * _OI * 100 * _S_gex ** 2 / 1e8
                            _n_gex += 1
                except Exception:
                    pass
            if _n_gex > 0:
                _gex_score = (70.0 if _total_gex > 0 else 35.0)
                factor_scores["gex_wall"] = FactorScore(
                    name="Options GEX Wall",
                    value=round(_total_gex, 2),
                    score=_gex_score,
                    interpretation=(
                        f"Net GEX: {_total_gex:+.2f}M over {_n_gex} strikes — "
                        f"{'positive GEX — dealers absorbing vol, pinning near spot' if _total_gex > 0 else 'negative GEX — dealers amplify moves, elevated realized vol expected'}"
                    ),
                )
                if _total_gex < -5.0:
                    warnings.append(f"Negative GEX {_total_gex:.1f}M — dealer short gamma, realized vol likely to spike.")
        except Exception:
            pass

        # ── New: Pairs Trading Spread Z-score ────────────────────────────
        try:
            import yfinance as _yf_pt
            import numpy as _np_pt
            _PAIRS = {
                "Technology": ("MSFT", "GOOGL"),
                "Finance":    ("JPM", "BAC"),
                "Energy":     ("XOM", "CVX"),
                "Consumer":   ("AMZN", "WMT"),
            }
            _best_pair = None
            _best_corr  = 0.0
            _tick_upper = ticker.upper()
            for _sector, (_a, _b) in _PAIRS.items():
                if _tick_upper in (_a, _b):
                    _best_pair = (_a, _b)
                    _best_sector = _sector
                    break
            if _best_pair is None:
                _best_pair = ("SPY", "QQQ")
                _best_sector = "Market"
            _pa, _pb = _best_pair
            _da = _yf_pt.download(_pa, period="6mo", interval="1d", auto_adjust=True, progress=False)["Close"].squeeze().dropna()
            _db = _yf_pt.download(_pb, period="6mo", interval="1d", auto_adjust=True, progress=False)["Close"].squeeze().dropna()
            if len(_da) >= 60 and len(_db) >= 60:
                import pandas as _pd_pt
                _aligned = _pd_pt.concat([_da, _db], axis=1).dropna()
                _aligned.columns = ["A", "B"]
                _spread = _np_pt.log(_aligned["A"].values) - _np_pt.log(_aligned["B"].values)
                _sp_mean = float(_np_pt.mean(_spread))
                _sp_std  = float(_np_pt.std(_spread))
                if _sp_std > 0:
                    _z = (_spread[-1] - _sp_mean) / _sp_std
                    _pt_score = max(10.0, min(90.0, 50.0 - _z * 15.0))
                    factor_scores["pairs_zscore"] = FactorScore(
                        name=f"Pairs Spread Z-score ({_pa}/{_pb})",
                        value=round(float(_z), 3),
                        score=_pt_score,
                        interpretation=(
                            f"{_pa}/{_pb} spread Z: {_z:+.2f} (μ={_sp_mean:.3f}, σ={_sp_std:.3f}) — "
                            f"{'spread wide — {_pa} cheap vs {_pb} (mean-reversion buy signal)' if _z < -1.5 else 'spread narrow — {_pa} expensive vs {_pb}' if _z > 1.5 else 'spread near historical mean'}"
                        ),
                    )
        except Exception:
            pass

        reasoning.append(f"Kelly half-size: {kelly_res.half_kelly * 100:.1f}% | Multiplier: {multiplier:.1f}x.")

        # ── CDS spread proxy (P3b) ─────────────────────────────────────────
        # Use HYG/LQD ratio as a credit-market stress proxy for the sector.
        # When credit spreads widen rapidly vs historical → equity downside risk.
        try:
            import yfinance as _yf_cds
            import numpy as _np_cds
            hyg = _yf_cds.Ticker("HYG").history(period="3mo", interval="1d", progress=False)
            lqd = _yf_cds.Ticker("LQD").history(period="3mo", interval="1d", progress=False)
            if not hyg.empty and not lqd.empty and len(hyg) >= 20:
                hyg_c = hyg["Close"].dropna()
                lqd_c = lqd["Close"].dropna().reindex(hyg_c.index, method="ffill").dropna()
                common = hyg_c.index.intersection(lqd_c.index)
                if len(common) >= 20:
                    ratio = hyg_c.loc[common] / lqd_c.loc[common]
                    ratio_1m = float(ratio.pct_change(20).iloc[-1]) * 100
                    ratio_5d = float(ratio.pct_change(5).iloc[-1])  * 100
                    if ratio_1m < -3.0:
                        # HYG underperforms LQD — credit stress rising → bearish
                        prob_up = max(0.20, prob_up - 0.05)
                        cds_score = 28.0
                        cds_signal = f"HYG/LQD spread widening {ratio_1m:.1f}% 1M — credit stress ELEVATED"
                    elif ratio_1m < -1.5 or ratio_5d < -1.0:
                        prob_up = max(0.25, prob_up - 0.02)
                        cds_score = 38.0
                        cds_signal = f"HYG/LQD slightly negative {ratio_1m:.1f}% 1M — credit caution"
                    elif ratio_1m > 2.0:
                        prob_up = min(0.80, prob_up + 0.03)
                        cds_score = 68.0
                        cds_signal = f"HYG/LQD tightening {ratio_1m:+.1f}% 1M — credit healthy"
                    else:
                        cds_score = 52.0
                        cds_signal = f"HYG/LQD stable ({ratio_1m:+.1f}% 1M)"
                    factor_scores["cds_proxy"] = FactorScore(
                        name="CDS Spread Proxy (HYG/LQD)",
                        value=round(ratio_1m, 2),
                        score=cds_score,
                        interpretation=cds_signal,
                    )
                    reasoning.append(f"Credit: {cds_signal}.")
        except Exception:
            pass

        # ── Vote ───────────────────────────────────────────────────────────
        if prob_up < 0.35:
            vote = Direction.SHORT
        elif prob_up > 0.60:
            vote = Direction.LONG
        else:
            vote = Direction.HOLD

        return AgentResult(
            agent_name=self.name,
            vote=vote,
            probability_up=prob_up,
            confidence=confidence,
            reasoning=" ".join(reasoning),
            factor_scores=factor_scores,
            warnings=warnings,
        )
