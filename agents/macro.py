"""
AlphaAgent — Macro Agent

Analyzes broad economic conditions using FRED data to determine if the
overall market environment is safe for investing.

Factors:
  1.  Yield curve (10Y-2Y spread)
  2.  Fed Funds Rate level
  3.  VIX fear index
  4.  Unemployment rate
  5.  CPI / inflation (CPIAUCSL)
  6.  Credit spreads proxy (HYG vs LQD)
  7.  Cross-asset: Copper/Gold ratio
  8.  Cross-asset: BTC as risk-on proxy
  9.  Global equity breadth (ACWI vs SPY)
  10. Dollar index regime (DXY momentum)
  11. M2 money supply trend (FRED M2SL)
"""

import logging
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import yfinance as yf

from agents.base import BaseAgent
from agents.state import AgentResult, FactorScore
from data.macro import MacroData
from data.currency import CurrencyData
from quant_engine.macro import analyze_macro_environment
from config.settings_manager import settings

logger = logging.getLogger(__name__)

# Module-level cache for cross-asset yfinance series (15 min TTL)
_CROSS_ASSET_CACHE: dict = {}
_CROSS_ASSET_TTL = 900   # 15 minutes


def _yf_close_series(ticker: str, period: str = "3mo") -> "pd.Series":
    """Cached close price series for a yfinance ticker."""
    import pandas as pd
    key = f"{ticker}_{period}"
    entry = _CROSS_ASSET_CACHE.get(key)
    if entry and _time.time() < entry[1]:
        return entry[0]
    try:
        df = yf.download(ticker, period=period, interval="1d",
                         auto_adjust=True, progress=False)
        s = df["Close"].squeeze().dropna()
    except Exception:
        s = pd.Series(dtype=float)
    _CROSS_ASSET_CACHE[key] = (s, _time.time() + _CROSS_ASSET_TTL)
    return s


def _yf_momentum(ticker: str, lookback: int = 22, period: str = "3mo") -> float:
    s = _yf_close_series(ticker, period)
    if len(s) > lookback:
        return float((s.iloc[-1] / s.iloc[-lookback - 1] - 1) * 100)
    return 0.0


def _yf_rel_strength(t1: str, t2: str, lookback: int = 22, period: str = "3mo") -> float:
    s1 = _yf_close_series(t1, period)
    s2 = _yf_close_series(t2, period)
    if len(s1) > lookback and len(s2) > lookback:
        r1 = (s1.iloc[-1] / s1.iloc[-lookback - 1] - 1) * 100
        r2 = (s2.iloc[-1] / s2.iloc[-lookback - 1] - 1) * 100
        return float(r1 - r2)
    return 0.0


def _prefetch_cross_asset():
    """Pre-warm cross-asset series in parallel (call once per agent invocation)."""
    tickers = ["HYG", "LQD", "CPER", "GLD", "BTC-USD", "ACWI", "SPY", "DX-Y.NYB", "IWM"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_yf_close_series, t, "3mo"): t for t in tickers}
        for extra in [("SPY", "1y"), ("^VIX", "1mo"), ("^VIX", "2y")]:
            futs[pool.submit(_yf_close_series, *extra)] = f"{extra[0]}_{extra[1]}"
        for f in as_completed(futs):
            try: f.result(timeout=10)
            except Exception: pass


class MacroAgent(BaseAgent):
    """
    Evaluates Federal Reserve Economic Data (FRED) and cross-asset signals
    to determine recession probability and macro regime.
    """
    name = "macro"

    def _run_analysis(self, ticker: str, data: Any, **kwargs) -> AgentResult:
        _prefetch_cross_asset()   # parallel pre-warm cross-asset series
        macro_data = MacroData()
        snapshot = macro_data.get_macro_snapshot()

        if not snapshot:
            return AgentResult(
                agent_name=self.name,
                probability_up=0.5,
                confidence=0.0,
                reasoning="Failed to fetch Federal Reserve Economic Data (FRED).",
                warnings=["FRED API Error"],
            )

        result = analyze_macro_environment(snapshot)

        factor_scores = {}
        reasoning = []
        warnings = []
        sm = settings.get_section("macro")

        # ── 1. Recession Probability ──────────────────────────────────────
        rec_low = sm.get("recession_prob_low", 0.2)
        rec_cutoff = sm.get("recession_prob_cutoff", 0.6)
        factor_scores["recession_risk"] = FactorScore(
            name="Recession Probability",
            value=result.recession_probability * 100,
            score=(100.0 if result.recession_probability < rec_low
                   else 0.0 if result.recession_probability > rec_cutoff
                   else 50.0),
            interpretation=f"Recession risk: {result.recession_probability * 100:.0f}%",
        )

        # ── 2. Yield Curve ────────────────────────────────────────────────
        yc_pos = sm.get("yield_curve_positive", 0.5)
        yc_inv = sm.get("yield_curve_inverted", -0.5)
        factor_scores["yield_curve"] = FactorScore(
            name="Yield Curve (10Y-2Y)",
            value=result.yield_curve,
            score=(100.0 if result.yield_curve > yc_pos
                   else 0.0 if result.yield_curve < 0
                   else 50.0),
            interpretation=f"Spread: {result.yield_curve:.2f}%",
        )
        if result.yield_curve < yc_inv:
            reasoning.append(f"Yield curve deeply inverted ({result.yield_curve:.2f}%) — recession warning.")
        elif result.yield_curve < 0:
            reasoning.append(f"Yield curve inverted ({result.yield_curve:.2f}%) — caution.")
        else:
            reasoning.append(f"Yield curve positive ({result.yield_curve:.2f}%) — no imminent recession signal.")

        # ── 3. Fed Funds Rate ─────────────────────────────────────────────
        ffr_easy = sm.get("fed_rate_easy", 2.0)
        ffr_tight = sm.get("fed_rate_tight", 4.0)
        factor_scores["fed_funds"] = FactorScore(
            name="Fed Funds Rate",
            value=result.fed_funds_rate,
            score=(100.0 if result.fed_funds_rate < ffr_easy
                   else 0.0 if result.fed_funds_rate > ffr_tight * 1.25
                   else 50.0),
            interpretation=f"Rate: {result.fed_funds_rate:.2f}%",
        )

        # ── 4. CPI / Inflation ────────────────────────────────────────────
        try:
            cpi_target = sm.get("cpi_target", 2.5)
            cpi_comfort = sm.get("inflation_comfort", 4.0)
            cpi_elevated = sm.get("cpi_elevated", 5.0)
            min_yoy = settings.get("data.min_yoy_months", 13)
            cpi_series = macro_data.get_series("CPIAUCSL", years_back=2)
            if len(cpi_series) >= min_yoy:
                cpi_now = float(cpi_series.iloc[-1].iloc[0])
                cpi_year_ago = float(cpi_series.iloc[-min_yoy].iloc[0])
                cpi_yoy = (cpi_now / cpi_year_ago - 1) * 100

                cpi_score = (80.0 if cpi_yoy < cpi_target
                             else 50.0 if cpi_yoy < cpi_comfort
                             else 20.0)
                factor_scores["cpi_inflation"] = FactorScore(
                    name="CPI Inflation (YoY)",
                    value=cpi_yoy,
                    score=cpi_score,
                    interpretation=f"CPI YoY: {cpi_yoy:.1f}%",
                )
                if cpi_yoy > cpi_elevated:
                    warnings.append(f"High inflation: CPI {cpi_yoy:.1f}% YoY — Fed tightening risk.")
                    reasoning.append(f"Inflation elevated ({cpi_yoy:.1f}% YoY) — rate hike risk.")
                elif cpi_yoy < cpi_target:
                    reasoning.append(f"Inflation benign ({cpi_yoy:.1f}% YoY) — Fed has room to cut.")
                else:
                    reasoning.append(f"Inflation moderate ({cpi_yoy:.1f}% YoY).")
        except Exception as e:
            reasoning.append(f"CPI data unavailable ({e}).")

        # ── 5. VIX — rolling 2-year percentile score ─────────────────────
        if "vix" in snapshot:
            vix = snapshot["vix"]
            try:
                vix_hist = _yf_close_series("^VIX", period="2y")
                if len(vix_hist) >= 30:
                    pct_rank = float((vix_hist < vix).sum()) / len(vix_hist)
                    # High VIX percentile → fearful market → bearish score
                    vix_score = max(10.0, min(90.0, 100.0 * (1.0 - pct_rank)))
                    vix_label = (f"{vix:.1f} — {pct_rank*100:.0f}th pct of 2Y range"
                                 f" (dynamic, not fixed threshold)")
                else:
                    raise ValueError("insufficient history")
            except Exception:
                vix_score = max(10.0, min(90.0, 90.0 - (vix - 10) * 2.2))
                vix_label = f"VIX: {vix:.1f}"
            factor_scores["vix"] = FactorScore(
                name="VIX Fear Index",
                value=vix,
                score=vix_score,
                interpretation=vix_label,
            )

        # ── 6. Credit Spreads (HYG/LQD proxy) ────────────────────────────
        try:
            hyg_mom = _yf_momentum("HYG", lookback=22)
            lqd_mom = _yf_momentum("LQD", lookback=22)
            credit_chg = hyg_mom - lqd_mom

            credit_score = max(10.0, min(90.0, 50.0 + credit_chg * 5.0))
            factor_scores["credit_spreads"] = FactorScore(
                name="Credit Spreads (HYG/LQD)",
                value=credit_chg,
                score=credit_score,
                interpretation=f"Junk vs IG spread change (1m): {credit_chg:+.2f}%",
            )
            if credit_chg < -2.0:
                reasoning.append(f"Credit spreads widening ({credit_chg:.2f}%) — financial stress rising.")
            elif credit_chg > 1.0:
                reasoning.append(f"Credit markets healthy: HYG outperforming LQD ({credit_chg:+.2f}%).")
            else:
                reasoning.append(f"Credit spreads stable ({credit_chg:+.2f}%).")
        except Exception as e:
            reasoning.append(f"Credit spread data unavailable ({e}).")

        # ── 7. Copper/Gold Ratio ──────────────────────────────────────────
        try:
            copper_mom = _yf_momentum("COPX", lookback=22)
            gold_mom = _yf_momentum("GLD", lookback=22)
            cg_signal = copper_mom - gold_mom

            cg_score = max(10.0, min(90.0, 50.0 + cg_signal * 2.0))
            factor_scores["copper_gold"] = FactorScore(
                name="Copper/Gold Ratio (Growth Signal)",
                value=cg_signal,
                score=cg_score,
                interpretation=f"Copper vs Gold 1m: {cg_signal:+.2f}%",
            )
            if cg_signal > 5.0:
                reasoning.append(f"Copper outperforming gold ({cg_signal:+.1f}%) — industrial/growth regime.")
            elif cg_signal < -5.0:
                reasoning.append(f"Gold outperforming copper ({cg_signal:.1f}%) — defensive/stagflation signal.")
            else:
                reasoning.append(f"Copper/Gold ratio neutral ({cg_signal:+.1f}%).")
        except Exception as e:
            reasoning.append(f"Copper/Gold data unavailable ({e}).")

        # ── 8. BTC as Risk-On Proxy ───────────────────────────────────────
        try:
            btc_mom = _yf_momentum("BTC-USD", lookback=22, period="3mo")
            btc_score = max(10.0, min(90.0, 50.0 + btc_mom * 0.5))
            factor_scores["btc_risk_on"] = FactorScore(
                name="BTC Risk-On Signal",
                value=btc_mom,
                score=btc_score,
                interpretation=f"BTC 1m: {btc_mom:+.1f}%",
            )
            if btc_mom > 20:
                reasoning.append(f"BTC surging ({btc_mom:+.1f}% 1m) — risk appetite strong.")
            elif btc_mom < -20:
                reasoning.append(f"BTC collapsing ({btc_mom:.1f}% 1m) — risk-off pressure.")
            else:
                reasoning.append(f"BTC momentum neutral ({btc_mom:+.1f}% 1m).")
        except Exception as e:
            reasoning.append(f"BTC data unavailable ({e}).")

        # ── 9. Global Equity Breadth (ACWI vs SPY) ───────────────────────
        try:
            acwi_rs = _yf_rel_strength("ACWI", "SPY", lookback=22)
            breadth_score = max(10.0, min(90.0, 50.0 + acwi_rs * 5.0))
            factor_scores["global_breadth"] = FactorScore(
                name="Global Equity Breadth (ACWI vs SPY)",
                value=acwi_rs,
                score=breadth_score,
                interpretation=f"ACWI vs SPY RS: {acwi_rs:+.2f}%",
            )
            if acwi_rs > 2.0:
                reasoning.append(f"Global equities outperforming US ({acwi_rs:+.1f}%) — broad macro recovery.")
            elif acwi_rs < -3.0:
                reasoning.append(f"Global equities underperforming ({acwi_rs:.1f}%) — US-specific strength or global concern.")
        except Exception as e:
            reasoning.append(f"Global breadth unavailable ({e}).")

        # ── 10. DXY Regime ────────────────────────────────────────────────
        try:
            fx_data = CurrencyData()
            fx_snap = fx_data.get_snapshot()
            dxy_regime = fx_data.get_dxy_regime()
            dxy_score = (30.0 if dxy_regime == "STRONG_USD"
                         else 70.0 if dxy_regime == "WEAK_USD"
                         else 50.0)
            factor_scores["dxy_regime"] = FactorScore(
                name="Dollar Index Regime (DXY)",
                value=fx_snap.dxy_1m_change,
                score=dxy_score,
                interpretation=f"DXY {dxy_regime} | 1m: {fx_snap.dxy_1m_change:+.1f}%",
            )
            reasoning.append(f"DXY regime: {dxy_regime} ({fx_snap.dxy_1m_change:+.1f}% 1m).")
        except Exception as e:
            reasoning.append(f"DXY regime unavailable ({e}).")

        # ── 11. M2 Money Supply Trend ─────────────────────────────────────
        try:
            m2 = macro_data.get_series("M2SL", years_back=2)
            if len(m2) >= 13:
                m2_now = float(m2.iloc[-1].iloc[0])
                m2_prev = float(m2.iloc[-13].iloc[0])
                m2_growth = (m2_now / m2_prev - 1) * 100
                m2_score = (75.0 if m2_growth > 5.0
                            else 50.0 if m2_growth > 0
                            else 30.0)
                factor_scores["m2_money_supply"] = FactorScore(
                    name="M2 Money Supply Growth",
                    value=m2_growth,
                    score=m2_score,
                    interpretation=f"M2 YoY: {m2_growth:+.1f}%",
                )
                if m2_growth > 10:
                    reasoning.append(f"M2 growing {m2_growth:+.1f}% YoY — liquidity supportive.")
                elif m2_growth < 0:
                    reasoning.append(f"M2 contracting {m2_growth:.1f}% YoY — liquidity tightening.")
        except Exception as e:
            reasoning.append(f"M2 data unavailable ({e}).")

        # ── 12. ISM PMI Manufacturing (FRED: MANEMP proxy via NAPM) ──────
        try:
            pmi_expansion = sm.get("pmi_expansion", 50)
            pmi_strong = sm.get("pmi_strong", 55)
            pmi_weak = sm.get("pmi_weak", 48)
            pmi_series = macro_data.get_series("NAPM", years_back=1)
            if pmi_series is not None and len(pmi_series) >= 2:
                pmi_val = float(pmi_series.iloc[-1].iloc[0])
                pmi_score = (85.0 if pmi_val > pmi_strong else 60.0 if pmi_val > pmi_expansion else 35.0 if pmi_val > 45 else 15.0)
                factor_scores["ism_pmi"] = FactorScore(
                    name="ISM Manufacturing PMI",
                    value=pmi_val,
                    score=pmi_score,
                    interpretation=f"ISM PMI: {pmi_val:.1f} ({'expansion' if pmi_val > pmi_expansion else 'contraction'})",
                )
                if pmi_val < pmi_weak:
                    reasoning.append(f"ISM PMI deeply contractionary ({pmi_val:.1f}) — manufacturing recession signal.")
                elif pmi_val > pmi_strong:
                    reasoning.append(f"ISM PMI expansionary ({pmi_val:.1f}) — manufacturing growth strong.")
        except Exception as e:
            reasoning.append(f"ISM PMI unavailable ({e}).")

        # ── 13. Consumer Sentiment (FRED: UMCSENT) ────────────────────────
        try:
            cs_strong = sm.get("consumer_sentiment_strong", 90)
            cs_moderate = sm.get("consumer_sentiment_moderate", 70)
            cs_weak = sm.get("consumer_sentiment_weak", 55)
            umcs = macro_data.get_series("UMCSENT", years_back=1)
            if umcs is not None and len(umcs) >= 2:
                cs_now  = float(umcs.iloc[-1].iloc[0])
                cs_prev = float(umcs.iloc[-2].iloc[0])
                cs_change = cs_now - cs_prev
                cs_score = (80.0 if cs_now > cs_strong else 60.0 if cs_now > cs_moderate else 40.0 if cs_now > cs_weak else 20.0)
                factor_scores["consumer_sentiment"] = FactorScore(
                    name="UMich Consumer Sentiment",
                    value=cs_now,
                    score=cs_score,
                    interpretation=f"UMich Sentiment: {cs_now:.1f} ({cs_change:+.1f} MoM)",
                )
                if cs_now < cs_weak:
                    reasoning.append(f"Consumer sentiment very weak ({cs_now:.1f}) — demand destruction risk.")
                elif cs_now > cs_strong:
                    reasoning.append(f"Consumer sentiment strong ({cs_now:.1f}) — consumption tailwind.")
        except Exception as e:
            reasoning.append(f"Consumer sentiment unavailable ({e}).")

        # ── 14. Initial Jobless Claims (FRED: ICSA) ───────────────────────
        try:
            claims_low = sm.get("claims_low", 220000)
            claims_elevated = sm.get("claims_elevated", 260000)
            claims_high = sm.get("claims_high", 350000)
            icsa = macro_data.get_series("ICSA", years_back=1)
            if icsa is not None and len(icsa) >= 4:
                claims_now = float(icsa.iloc[-1].iloc[0])
                claims_4wa = float(icsa.iloc[-4:].mean().iloc[0])   # 4-week avg
                claims_score = (80.0 if claims_now < claims_low else 60.0 if claims_now < claims_elevated else 35.0 if claims_now < claims_high else 15.0)
                factor_scores["jobless_claims"] = FactorScore(
                    name="Initial Jobless Claims",
                    value=claims_now,
                    score=claims_score,
                    interpretation=f"Claims: {claims_now/1000:.0f}K (4W avg: {claims_4wa/1000:.0f}K)",
                )
                if claims_now > claims_high:
                    reasoning.append(f"Jobless claims elevated ({claims_now/1000:.0f}K) — labor deterioration.")
        except Exception as e:
            reasoning.append(f"Jobless claims unavailable ({e}).")

        # ── 15. 10Y TIPS Breakeven Inflation (FRED: T10YIE) ──────────────
        try:
            tips_low = sm.get("tips_breakeven_low", 1.5)
            tips_high = sm.get("tips_breakeven_high", 2.5)
            tips_danger = sm.get("tips_breakeven_danger", 3.0)
            tips = macro_data.get_series("T10YIE", years_back=1)
            if tips is not None and len(tips) >= 2:
                tips_val = float(tips.iloc[-1].iloc[0])
                tips_score = (70.0 if tips_low < tips_val < tips_high else 50.0 if 1.0 < tips_val <= tips_low else 30.0 if tips_val > tips_danger else 40.0)
                factor_scores["tips_breakeven"] = FactorScore(
                    name="10Y TIPS Breakeven Inflation",
                    value=tips_val,
                    score=tips_score,
                    interpretation=f"Breakeven: {tips_val:.2f}% ({'anchored' if tips_low < tips_val < tips_high else 'elevated' if tips_val > tips_high else 'deflation risk'})",
                )
                if tips_val > tips_danger:
                    reasoning.append(f"TIPS breakeven {tips_val:.2f}% — market pricing in persistent inflation.")
        except Exception as e:
            reasoning.append(f"TIPS breakeven unavailable ({e}).")

        # ── 16. Amihud Illiquidity Ratio (Market Microstructure) ─────────
        try:
            import pandas as pd
            spy_amihud = yf.download("SPY", period="3mo", interval="1d",
                                     auto_adjust=True, progress=False)
            if not spy_amihud.empty and len(spy_amihud) >= 20:
                spy_c = spy_amihud["Close"].squeeze().dropna()
                spy_v = spy_amihud["Volume"].squeeze().dropna()
                spy_ret_abs = spy_c.pct_change().abs().dropna()
                dollar_vol  = (spy_c * spy_v).reindex(spy_ret_abs.index).replace(0, float("nan"))
                amihud_raw  = (spy_ret_abs / dollar_vol).dropna()
                amihud_val  = float(amihud_raw.iloc[-20:].mean()) * 1e9  # scale to readable units
                amihud_hist = float(amihud_raw.mean()) * 1e9
                amihud_rel  = amihud_val / amihud_hist if amihud_hist > 0 else 1.0
                # Low amihud = liquid (good); high amihud = illiquid (bad)
                _amihud_thresh = sm.get("amihud_stress_threshold", 2.0)
                amihud_score = max(10.0, min(90.0, 70.0 - (amihud_rel - 1.0) * 30.0))
                factor_scores["amihud_illiquidity"] = FactorScore(
                    name="Amihud Illiquidity Ratio (SPY)",
                    value=round(amihud_rel, 3),
                    score=amihud_score,
                    interpretation=(
                        f"Market illiquidity: {amihud_rel:.2f}x avg "
                        f"({'drying up — flash crash risk' if amihud_rel > _amihud_thresh else 'elevated' if amihud_rel > 1.3 else 'normal liquidity'})"
                    ),
                )
                if amihud_rel > _amihud_thresh:
                    reasoning.append(f"Amihud illiquidity {amihud_rel:.1f}x above normal — market depth severely impaired.")
                    warnings.append(f"ILLIQUIDITY WARNING: Amihud ratio >{_amihud_thresh}x — large trades will move prices significantly.")
        except Exception as e:
            reasoning.append(f"Amihud illiquidity unavailable ({e}).")

        # ── 16b. Bond-Equity Correlation (TLT/SPY) ────────────────────────
        try:
            import pandas as pd
            d_tlt = yf.download("TLT", period="6mo", interval="1d", auto_adjust=True, progress=False)
            d_spy = yf.download("SPY", period="6mo", interval="1d", auto_adjust=True, progress=False)
            tlt = d_tlt["Close"].squeeze().dropna()
            spy = d_spy["Close"].squeeze().dropna()
            if len(tlt) > 60 and len(spy) > 60:
                tlt_ret = tlt.pct_change().dropna()
                spy_ret = spy.pct_change().dropna()
                aligned = pd.concat([tlt_ret, spy_ret], axis=1).dropna()
                if len(aligned) >= 60:
                    be_corr = float(aligned.iloc[-60:].corr().iloc[0, 1])
                    # Negative = normal (bonds up when stocks down); both falling = systemic stress
                    _be_thresh = sm.get("bond_equity_stress_corr", 0.3)
                    be_score = (30.0 if be_corr > _be_thresh else 55.0 if be_corr > -0.2 else 70.0)
                    factor_scores["bond_equity_corr"] = FactorScore(
                        name="Bond-Equity Correlation (TLT/SPY)",
                        value=round(be_corr, 3),
                        score=be_score,
                        interpretation=f"60d rolling TLT-SPY corr: {be_corr:+.2f} ({'systemic stress' if be_corr > _be_thresh else 'flight-to-safety intact' if be_corr < -0.2 else 'neutral'})",
                    )
                    if be_corr > _be_thresh:
                        reasoning.append(f"Bond-equity correlation positive ({be_corr:.2f}) — both assets falling: systemic risk event.")
        except Exception as e:
            reasoning.append(f"Bond-equity correlation unavailable ({e}).")

        # ── 17. Semiconductor (SOX) Relative Performance ──────────────────
        try:
            sox_rs = _yf_rel_strength("SOXX", "SPY", lookback=22)
            sox_score = max(10.0, min(90.0, 50.0 + sox_rs * 3.0))
            factor_scores["sox_relative"] = FactorScore(
                name="Semiconductor Index (SOX vs SPY)",
                value=sox_rs,
                score=sox_score,
                interpretation=f"SOXX vs SPY RS: {sox_rs:+.1f}% — {'tech leading (growth regime)' if sox_rs > 3 else 'tech lagging (caution)' if sox_rs < -3 else 'neutral'}",
            )
            if sox_rs > 5:
                reasoning.append(f"Semiconductors leading market ({sox_rs:+.1f}%) — tech growth regime confirmed.")
            elif sox_rs < -5:
                reasoning.append(f"Semiconductors underperforming ({sox_rs:.1f}%) — growth slowdown signal.")
        except Exception as e:
            reasoning.append(f"SOX data unavailable ({e}).")

        # ── 18. Global Pre-Market Signal (Asia/Europe ETFs) ───────────────
        try:
            asia_etfs = {"EWJ": "Japan", "EWZ": "Brazil", "EWY": "Korea"}
            eu_etfs   = {"EWG": "Germany", "EWU": "UK", "EWQ": "France"}
            all_premarket = {}
            for sym, name in {**asia_etfs, **eu_etfs}.items():
                mom = _yf_momentum(sym, lookback=5, period="1mo")
                all_premarket[name] = mom
            if all_premarket:
                avg_global = sum(all_premarket.values()) / len(all_premarket)
                pm_score = max(10.0, min(90.0, 50.0 + avg_global * 3.0))
                top_regions = sorted(all_premarket.items(), key=lambda x: x[1], reverse=True)[:2]
                bot_regions = sorted(all_premarket.items(), key=lambda x: x[1])[:2]
                factor_scores["global_premarket"] = FactorScore(
                    name="Global Pre-Market Signal (Asia/EU)",
                    value=round(avg_global, 2),
                    score=pm_score,
                    interpretation=f"Avg 5d return: {avg_global:+.1f}% | Leading: {top_regions[0][0]} {top_regions[0][1]:+.1f}% | Lagging: {bot_regions[0][0]} {bot_regions[0][1]:+.1f}%",
                )
                if avg_global < -3:
                    reasoning.append(f"Global equity weakness (avg {avg_global:.1f}%) — contagion risk to US markets.")
                elif avg_global > 3:
                    reasoning.append(f"Global equity strength (avg {avg_global:+.1f}%) — risk-on globally.")
        except Exception as e:
            reasoning.append(f"Global pre-market data unavailable ({e}).")

        # ── 19. Fed Balance Sheet (FRED WALCL) ────────────────────────────
        try:
            walcl = macro_data.get_series("WALCL", years_back=1)
            if walcl is not None and len(walcl) >= 8:
                walcl_now  = float(walcl.iloc[-1].iloc[0])
                walcl_prev = float(walcl.iloc[-9].iloc[0])   # ~2 months ago
                walcl_chg_pct = (walcl_now / walcl_prev - 1) * 100
                # QE (expanding) = bullish liquidity; QT (shrinking) = tighter
                bs_score = (75.0 if walcl_chg_pct > 1.0 else 50.0 if walcl_chg_pct > -0.5 else 30.0)
                factor_scores["fed_balance_sheet"] = FactorScore(
                    name="Fed Balance Sheet (WALCL)",
                    value=round(walcl_chg_pct, 2),
                    score=bs_score,
                    interpretation=f"Fed assets 2M change: {walcl_chg_pct:+.2f}% ({'QE / expanding' if walcl_chg_pct > 1 else 'QT / shrinking' if walcl_chg_pct < -0.5 else 'flat'})",
                )
                if walcl_chg_pct < -1.0:
                    reasoning.append(f"Fed balance sheet contracting ({walcl_chg_pct:.2f}%) — liquidity withdrawal, tightening bias.")
                elif walcl_chg_pct > 1.0:
                    reasoning.append(f"Fed balance sheet expanding ({walcl_chg_pct:+.2f}%) — liquidity supportive.")
        except Exception as e:
            reasoning.append(f"Fed balance sheet data unavailable ({e}).")

        # ── 20. Growth vs Value Rotation (IWF/IWD) ────────────────────────
        try:
            gv_rs = _yf_rel_strength("IWF", "IWD", lookback=22)
            gv_score = max(10.0, min(90.0, 50.0 + gv_rs * 3.0))
            factor_scores["growth_vs_value"] = FactorScore(
                name="Growth vs Value Rotation",
                value=round(gv_rs, 2),
                score=gv_score,
                interpretation=f"IWF vs IWD RS: {gv_rs:+.1f}% — {'growth regime (risk-on)' if gv_rs > 3 else 'value regime (defensive)' if gv_rs < -3 else 'neutral'}",
            )
        except Exception as e:
            reasoning.append(f"Growth/value rotation data unavailable ({e}).")

        # ── 21. Repo Market Stress (SOFR vs FEDFUNDS Spread) ─────────────
        try:
            sofr_s = macro_data.get_series("SOFR", years_back=1)
            ffr_s  = macro_data.get_series("FEDFUNDS", years_back=1)
            if sofr_s is not None and len(sofr_s) >= 2 and ffr_s is not None and len(ffr_s) >= 2:
                sofr_now   = float(sofr_s.iloc[-1].iloc[0])
                ffr_now    = float(ffr_s.iloc[-1].iloc[0])
                sofr_spread = sofr_now - ffr_now  # in percentage points (e.g. 0.03 = 3 bps)
                # SOFR trades close to FEDFUNDS; widening > 25 bps = repo stress
                repo_score = (70.0 if abs(sofr_spread) < 0.10
                              else 50.0 if abs(sofr_spread) < 0.25
                              else 30.0)
                factor_scores["sofr_spread"] = FactorScore(
                    name="Repo Market Stress (SOFR Spread)",
                    value=round(sofr_spread, 4),
                    score=repo_score,
                    interpretation=(
                        f"SOFR {sofr_now:.2f}% vs Fed Funds {ffr_now:.2f}% → spread {sofr_spread:+.4f}% "
                        f"({'normal — repo market calm' if abs(sofr_spread) < 0.10 else 'stressed — funding squeeze' if abs(sofr_spread) > 0.25 else 'slight tension'})"
                    ),
                )
                _sofr_thresh = sm.get("sofr_critical_spread", 0.50)
                if abs(sofr_spread) > _sofr_thresh:
                    warnings.append(f"REPO STRESS: SOFR spread to Fed Funds {sofr_spread:+.4f}% — funding market stress detected.")
                    reasoning.append(f"Repo market stress: SOFR-FF spread {sofr_spread:+.4f}% — interbank funding tightening.")
        except Exception as e:
            reasoning.append(f"SOFR spread unavailable ({e}).")

        # ── 22. Fed Rate Change Direction ─────────────────────────────────
        try:
            _ffr_series = macro_data.get_series("FEDFUNDS", years_back=1)
            if _ffr_series is not None and len(_ffr_series) >= 4:
                _ffr_now_r = float(_ffr_series.iloc[-1].iloc[0])
                _ffr_3mo_r = float(_ffr_series.iloc[-4].iloc[0])
                _ffr_delta = _ffr_now_r - _ffr_3mo_r
                if _ffr_delta > 0.24:
                    _rcd_label = "HIKING"
                    _rcd_score = 25.0
                elif _ffr_delta < -0.24:
                    _rcd_label = "CUTTING"
                    _rcd_score = 75.0
                else:
                    _rcd_label = "ON HOLD"
                    _rcd_score = 55.0
                factor_scores["rate_change_direction"] = FactorScore(
                    name="Fed Rate Change Direction",
                    value=round(_ffr_delta, 2),
                    score=_rcd_score,
                    interpretation=f"Fed: {_rcd_label} | FFR Δ3M: {_ffr_delta:+.2f}% — {'tightening cycle (bearish)' if _rcd_label == 'HIKING' else 'easing cycle (bullish)' if _rcd_label == 'CUTTING' else 'on hold — data dependent'}",
                )
                if _rcd_label == "CUTTING":
                    reasoning.append(f"Fed cutting rates ({_ffr_delta:+.2f}% in 3M) — accommodative pivot, equities positive.")
                elif _rcd_label == "HIKING":
                    reasoning.append(f"Fed hiking rates ({_ffr_delta:+.2f}% in 3M) — tightening cycle headwind.")
        except Exception:
            pass

        # ── 23. VIX 5-Day Change ──────────────────────────────────────────
        try:
            _vix_1m = _yf_close_series("^VIX", period="1mo")
            if len(_vix_1m) >= 6:
                _vix_now_v = float(_vix_1m.iloc[-1])
                _vix_5d_v  = float(_vix_1m.iloc[-6])
                _vix_delta = _vix_now_v - _vix_5d_v
                _v5d_score = max(10.0, min(90.0, 50.0 - _vix_delta * 4.0))
                factor_scores["vix_5d_change"] = FactorScore(
                    name="VIX 5-Day Change",
                    value=round(_vix_delta, 2),
                    score=_v5d_score,
                    interpretation=f"VIX Δ5d: {_vix_delta:+.1f} ({_vix_now_v:.1f} now) — {'fear spiking' if _vix_delta > 3 else 'fear dissipating' if _vix_delta < -3 else 'stable volatility'}",
                )
                if _vix_delta > 5:
                    reasoning.append(f"VIX spiked +{_vix_delta:.1f} in 5 days — volatility regime shift warning.")
        except Exception:
            pass

        # ── 24. SPY vs SMA(200) — Bull/Bear Market Regime ─────────────────
        try:
            _spy_1y = _yf_close_series("SPY", period="1y")
            if len(_spy_1y) >= 200:
                _spy_now_p  = float(_spy_1y.iloc[-1])
                _spy_sma200 = float(_spy_1y.iloc[-200:].mean())
                _pct_above  = (_spy_now_p / _spy_sma200 - 1) * 100
                _sma200_score = (80.0 if _pct_above > 5 else 60.0 if _pct_above > 0 else 35.0 if _pct_above > -5 else 15.0)
                factor_scores["spy_sma200"] = FactorScore(
                    name="SPY vs SMA(200)",
                    value=round(_pct_above, 2),
                    score=_sma200_score,
                    interpretation=f"SPY {_pct_above:+.1f}% {'above' if _pct_above > 0 else 'below'} 200-day SMA — {'bull market regime' if _pct_above > 3 else 'bear market' if _pct_above < -5 else 'near trend support'}",
                )
                if _pct_above < -5:
                    reasoning.append(f"SPY {_pct_above:.1f}% below 200-SMA — bear market regime confirmed.")
                    warnings.append(f"BEAR MARKET: SPY {_pct_above:.1f}% below 200-day moving average.")
                elif _pct_above > 5:
                    reasoning.append(f"SPY {_pct_above:+.1f}% above 200-SMA — bull market regime intact.")
        except Exception:
            pass

        # ── 25. Large vs Small Cap Rotation (SPY/IWM) ─────────────────────
        try:
            _lsc_rs = _yf_rel_strength("SPY", "IWM", lookback=22)
            _lsc_score = max(10.0, min(90.0, 50.0 - _lsc_rs * 2.0))
            factor_scores["large_vs_small_cap"] = FactorScore(
                name="Large vs Small Cap (SPY/IWM)",
                value=round(_lsc_rs, 2),
                score=_lsc_score,
                interpretation=f"SPY vs IWM RS: {_lsc_rs:+.1f}% — {'large-cap dominance (late cycle / defensive)' if _lsc_rs > 4 else 'small-cap leading (early cycle / risk-on)' if _lsc_rs < -4 else 'neutral — no size rotation signal'}",
            )
            if _lsc_rs < -5:
                reasoning.append(f"Small caps outperforming by {abs(_lsc_rs):.1f}% — risk-on early cycle signal.")
            elif _lsc_rs > 5:
                reasoning.append(f"Large caps dominating by {_lsc_rs:.1f}% — defensive rotation, late cycle signal.")
        except Exception:
            pass

        # ── 26. Sector Relative Strength (Ticker's Sector ETF vs SPY) ─────
        try:
            _sector_etf_map_m = {
                "Technology": "XLK", "Communication Services": "XLC",
                "Consumer Discretionary": "XLY", "Consumer Staples": "XLP",
                "Health Care": "XLV", "Industrials": "XLI", "Materials": "XLB",
                "Energy": "XLE", "Financials": "XLF", "Real Estate": "XLRE", "Utilities": "XLU",
            }
            _tkr_info_m = {}
            try:
                _tkr_info_m = data.get_info() or {}
            except Exception:
                import yfinance as _yf_m
                _tkr_info_m = _yf_m.Ticker(ticker).info or {}
            _sector_m = _tkr_info_m.get("sector", "") if isinstance(_tkr_info_m, dict) else ""
            _sec_etf_m = _sector_etf_map_m.get(_sector_m)
            if _sec_etf_m:
                _sec_rs_m = _yf_rel_strength(_sec_etf_m, "SPY", lookback=22)
                _sec_rs_score = max(10.0, min(90.0, 50.0 + _sec_rs_m * 3.0))
                factor_scores["sector_rs"] = FactorScore(
                    name=f"Sector RS ({_sector_m})",
                    value=round(_sec_rs_m, 2),
                    score=_sec_rs_score,
                    interpretation=f"{_sec_etf_m} vs SPY RS: {_sec_rs_m:+.1f}% — {'sector leading (momentum tailwind)' if _sec_rs_m > 3 else 'sector lagging (rotation headwind)' if _sec_rs_m < -3 else 'neutral sector rotation'}",
                )
                if _sec_rs_m > 5:
                    reasoning.append(f"{_sector_m} sector outperforming SPY by {_sec_rs_m:.1f}% — sector tailwind for {ticker}.")
                elif _sec_rs_m < -5:
                    reasoning.append(f"{_sector_m} sector underperforming SPY by {abs(_sec_rs_m):.1f}% — sector rotation headwind.")
        except Exception:
            pass

        # ── 27. Contagion Correlation Spike (Cross-Asset Co-Movement) ─────
        try:
            import pandas as _pd_cc
            _cc_series = {}
            for _sym_cc in ["SPY", "TLT", "GLD", "HYG"]:
                _s_cc = _yf_close_series(_sym_cc, "3mo")
                if len(_s_cc) >= 30:
                    _cc_series[_sym_cc] = _s_cc.pct_change().dropna()
            if len(_cc_series) >= 3:
                _cc_df = _pd_cc.DataFrame(_cc_series).dropna()
                if len(_cc_df) >= 30:
                    _n_cc = len(_cc_series)
                    _corr_rec  = _cc_df.iloc[-20:].corr()
                    _corr_hist = _cc_df.corr()
                    _avg_rec  = float((_corr_rec.sum().sum()  - _n_cc) / (_n_cc * (_n_cc - 1)))
                    _avg_hist = float((_corr_hist.sum().sum() - _n_cc) / (_n_cc * (_n_cc - 1)))
                    _corr_spike = _avg_rec - _avg_hist
                    _contagion_score = max(10.0, min(90.0, 70.0 - _corr_spike * 100))
                    factor_scores["contagion_correlation"] = FactorScore(
                        name="Contagion Correlation Spike",
                        value=round(_corr_spike, 3),
                        score=_contagion_score,
                        interpretation=f"Cross-asset corr (20d): {_avg_rec:.2f} vs hist {_avg_hist:.2f} → spike {_corr_spike:+.2f} — {'CRISIS CONTAGION — diversification failing' if _corr_spike > 0.15 else 'normal diversification' if _corr_spike < 0 else 'slightly elevated co-movement'}",
                    )
                    if _corr_spike > 0.20:
                        warnings.append(f"CONTAGION ALERT: Cross-asset correlation +{_corr_spike:.2f} above normal — crisis-like co-movement.")
                        reasoning.append(f"Cross-asset correlations spiking ({_corr_spike:+.2f}) — diversification failing, contagion risk.")
        except Exception:
            pass

        # ── 28. Real Interest Rate (DGS10 − T10YIE) ──────────────────────
        try:
            _tnx_s = macro_data.get_series("DGS10", years_back=1)
            _tie_s = macro_data.get_series("T10YIE", years_back=1)
            if (_tnx_s is not None and len(_tnx_s) >= 1 and
                    _tie_s is not None and len(_tie_s) >= 1):
                _nominal_r  = float(_tnx_s.iloc[-1].iloc[0])
                _breakeven  = float(_tie_s.iloc[-1].iloc[0])
                _real_rate  = _nominal_r - _breakeven
                _rr_m_score = (70.0 if _real_rate > 1.5 else 55.0 if _real_rate > 0.5
                               else 45.0 if _real_rate > 0 else 30.0 if _real_rate > -1.0 else 15.0)
                factor_scores["real_interest_rate"] = FactorScore(
                    name="Real Interest Rate (10Y)",
                    value=round(_real_rate, 3),
                    score=_rr_m_score,
                    interpretation=f"Real rate: {_real_rate:+.2f}% (Nominal {_nominal_r:.2f}% − BE {_breakeven:.2f}%) — {'positive real yield — equity multiple compression risk' if _real_rate > 1.0 else 'near-zero real rate' if abs(_real_rate) < 0.5 else 'negative real rate — financial repression, equity/gold supportive'}",
                )
                if _real_rate < -1.0:
                    reasoning.append(f"Deeply negative real rates ({_real_rate:.2f}%) — financial repression, equity-positive.")
                elif _real_rate > 2.0:
                    reasoning.append(f"Real rates elevated ({_real_rate:.2f}%) — equity multiple compression risk, rate burden rising.")
        except Exception:
            pass

        # ── New: PCE Inflation (FRED PCEPI — Fed's Preferred Gauge) ─────────
        try:
            _pce = macro_data.get_series("PCEPI", years_back=2)
            if _pce is not None and len(_pce) >= 13:
                _pce_now  = float(_pce.iloc[-1].iloc[0])
                _pce_yago = float(_pce.iloc[-13].iloc[0])
                _pce_yoy  = (_pce_now / _pce_yago - 1) * 100
                _pce_3m   = float(_pce.iloc[-4].iloc[0]) if len(_pce) >= 4 else _pce_yago
                _pce_3m_ann = ((_pce_now / _pce_3m) ** 4 - 1) * 100
                _pce_tgt   = sm.get("pce_target", 2.0)
                _pce_near  = sm.get("pce_near_target", 2.5)
                _pce_elev  = sm.get("pce_elevated", 3.5)
                _pce_score = (75.0 if _pce_yoy < _pce_tgt else 55.0 if _pce_yoy < _pce_near
                              else 35.0 if _pce_yoy < _pce_elev else 15.0)
                factor_scores["pce_inflation"] = FactorScore(
                    name="PCE Inflation (YoY)",
                    value=round(_pce_yoy, 2),
                    score=_pce_score,
                    interpretation=(
                        f"PCE YoY: {_pce_yoy:.1f}% | 3M ann: {_pce_3m_ann:.1f}% — "
                        f"{'at/below Fed 2% target' if _pce_yoy < 2.0 else 'near target' if _pce_yoy < 2.5 else 'above target — tightening pressure' if _pce_yoy < 3.5 else 'elevated — Fed hawkish bias'}"
                    ),
                )
                if _pce_yoy > _pce_elev:
                    warnings.append(f"PCE inflation {_pce_yoy:.1f}% — well above Fed {_pce_tgt:.1f}% target.")
                    reasoning.append(f"PCE inflation elevated ({_pce_yoy:.1f}% YoY, 3M ann {_pce_3m_ann:.1f}%) — Fed's preferred measure signals persistent price pressure.")
                elif _pce_yoy < _pce_tgt:
                    reasoning.append(f"PCE inflation benign ({_pce_yoy:.1f}% YoY) — below Fed target, rate cuts remain possible.")
                else:
                    reasoning.append(f"PCE inflation near Fed target ({_pce_yoy:.1f}% YoY).")
        except Exception:
            pass

        # ── New: Retail Sales MoM (FRED RSXFS) ──────────────────────────
        try:
            _rs_series = macro_data.get_series("RSXFS", years_back=1)
            if _rs_series is not None and len(_rs_series) >= 2:
                _rs_now  = float(_rs_series.iloc[-1].iloc[0])
                _rs_prev = float(_rs_series.iloc[-2].iloc[0])
                _rs_mom  = (_rs_now / _rs_prev - 1) * 100
                _rs_score = (75.0 if _rs_mom > 1.0 else 60.0 if _rs_mom > 0 else 40.0 if _rs_mom > -1.0 else 25.0)
                factor_scores["retail_sales"] = FactorScore(
                    name="Retail Sales MoM (ex-Auto)",
                    value=round(_rs_mom, 2),
                    score=_rs_score,
                    interpretation=(
                        f"Retail sales MoM: {_rs_mom:+.2f}% (${_rs_now / 1e3:.0f}B) — "
                        f"{'strong consumer spending' if _rs_mom > 1.0 else 'positive' if _rs_mom > 0 else 'slight contraction' if _rs_mom > -1.0 else 'consumer weakness'}"
                    ),
                )
                if _rs_mom < -1.5:
                    reasoning.append(f"Retail sales contracting ({_rs_mom:+.1f}% MoM) — consumer weakness signal.")
                elif _rs_mom > 1.5:
                    reasoning.append(f"Retail sales strong ({_rs_mom:+.1f}% MoM) — consumer resilience confirmed.")
        except Exception:
            pass

        # ── New: Leading Economic Index (FRED USSLIND) ───────────────────
        try:
            _lei_s = macro_data.get_series("USSLIND", years_back=1)
            if _lei_s is not None and len(_lei_s) >= 4:
                _lei_now   = float(_lei_s.iloc[-1].iloc[0])
                _lei_3m    = float(_lei_s.iloc[-4].iloc[0])
                _lei_trend = _lei_now - _lei_3m
                _lei_score = (72.0 if _lei_trend > 0.5 else 55.0 if _lei_trend > 0 else 40.0 if _lei_trend > -0.5 else 22.0)
                factor_scores["leading_index"] = FactorScore(
                    name="Leading Economic Index (LEI)",
                    value=round(_lei_trend, 2),
                    score=_lei_score,
                    interpretation=(
                        f"LEI: {_lei_now:.1f} | 3M change: {_lei_trend:+.2f} — "
                        f"{'expansion accelerating' if _lei_trend > 0.5 else 'mild expansion' if _lei_trend > 0 else 'slowdown signal' if _lei_trend > -0.5 else 'recession signal'}"
                    ),
                )
                if _lei_trend < -1.0:
                    warnings.append(f"LEI declining {_lei_trend:.2f} — leading indicator points to economic contraction.")
                    reasoning.append(f"Leading Economic Index falling ({_lei_trend:+.2f}) — forward-looking recession indicator.")
        except Exception:
            pass

        # ── New: HY Credit Spread (FRED BAMLH0A0HYM2) ───────────────────
        try:
            _hy_s = macro_data.get_series("BAMLH0A0HYM2", years_back=1)
            if _hy_s is not None and len(_hy_s) >= 2:
                _hy_now  = float(_hy_s.iloc[-1].iloc[0])
                _hy_prev = float(_hy_s.iloc[-5].iloc[0]) if len(_hy_s) >= 5 else _hy_now
                _hy_chg  = _hy_now - _hy_prev
                _hy_score = (72.0 if _hy_now < 3.0 else 55.0 if _hy_now < 5.0 else 35.0 if _hy_now < 7.0 else 18.0)
                factor_scores["hy_credit_spread"] = FactorScore(
                    name="HY Credit Spread (OAS)",
                    value=round(_hy_now, 2),
                    score=_hy_score,
                    interpretation=(
                        f"HY OAS: {_hy_now:.2f}% | 1W chg: {_hy_chg:+.2f}% — "
                        f"{'tight — risk-on, credit conditions easy' if _hy_now < 3.0 else 'normal' if _hy_now < 5.0 else 'wide — credit stress, risk-off' if _hy_now < 7.0 else 'DISTRESS — near-crisis spreads'}"
                    ),
                )
                if _hy_now > 7.0:
                    warnings.append(f"HY spreads at {_hy_now:.1f}% — near-crisis credit conditions.")
                    reasoning.append(f"High-yield credit spreads {_hy_now:.1f}% (OAS) — distress-level, systemic risk signal.")
                elif _hy_chg > 1.0:
                    reasoning.append(f"HY spreads widening {_hy_chg:+.1f}% in 1W — risk-off credit signal.")
        except Exception:
            pass

        # ── New: NFIB Small Business Optimism (FRED NFCI / BOPTIMISM) ───
        try:
            _nfib_s = macro_data.get_series("NFCI", years_back=1)
            if _nfib_s is None or len(_nfib_s) < 2:
                _nfib_s = macro_data.get_series("DRTSCILM", years_back=1)
            if _nfib_s is not None and len(_nfib_s) >= 4:
                _nfib_now  = float(_nfib_s.iloc[-1].iloc[0])
                _nfib_3m   = float(_nfib_s.iloc[-4].iloc[0])
                _nfib_trend = _nfib_now - _nfib_3m
                _nfib_neutral = 0.0
                _nfib_score = (70.0 if _nfib_now < _nfib_neutral and _nfib_trend < 0 else
                               55.0 if _nfib_now < _nfib_neutral else
                               40.0 if _nfib_trend > 0 else 50.0)
                factor_scores["nfib_conditions"] = FactorScore(
                    name="Financial Conditions Index (NFCI)",
                    value=round(_nfib_now, 3),
                    score=_nfib_score,
                    interpretation=(
                        f"NFCI: {_nfib_now:.3f} | 3M trend: {_nfib_trend:+.3f} — "
                        f"{'tight financial conditions — tightening bias' if _nfib_now > 0 else 'easy conditions — accommodative'} "
                        f"({'tightening' if _nfib_trend > 0 else 'easing'})"
                    ),
                )
                if _nfib_now > 0.5:
                    warnings.append(f"NFCI at {_nfib_now:.2f} — financial conditions tightening significantly.")
                    reasoning.append(f"Financial conditions index elevated ({_nfib_now:.2f}) — credit tightening, growth headwind.")
        except Exception:
            pass

        # ── New: Equity Risk Premium (E/P − Real Rate) ───────────────────
        try:
            import yfinance as _yf_erp
            _sp_info = _yf_erp.Ticker("^GSPC").info or {}
            _fwd_pe  = _sp_info.get("forwardPE", None) or _sp_info.get("trailingPE", None)
            _tnx_erp = macro_data.get_series("DGS10", years_back=1)
            _tie_erp = macro_data.get_series("T10YIE", years_back=1)
            if _fwd_pe and _tnx_erp is not None and _tie_erp is not None:
                _ep_ratio   = (1.0 / float(_fwd_pe)) * 100
                _real_r_erp = float(_tnx_erp.iloc[-1].iloc[0]) - float(_tie_erp.iloc[-1].iloc[0])
                _erp        = _ep_ratio - _real_r_erp
                _erp_score  = (75.0 if _erp > 4.0 else 60.0 if _erp > 2.0 else 45.0 if _erp > 0 else 25.0)
                factor_scores["equity_risk_premium"] = FactorScore(
                    name="Equity Risk Premium (E/P − Real Rate)",
                    value=round(_erp, 2),
                    score=_erp_score,
                    interpretation=(
                        f"ERP: {_erp:+.2f}% (E/P {_ep_ratio:.1f}% − real rate {_real_r_erp:.2f}%) — "
                        f"{'attractive equity valuation' if _erp > 4.0 else 'fair' if _erp > 2.0 else 'compressed premium' if _erp > 0 else 'equities expensive vs bonds'}"
                    ),
                )
                if _erp < 0:
                    reasoning.append(f"ERP negative ({_erp:.2f}%) — bonds offer better risk-adjusted return than equities.")
                elif _erp > 4.0:
                    reasoning.append(f"ERP at {_erp:.2f}% — equities attractively valued vs real bonds.")
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

        # ── Probability & Confidence ──────────────────────────────────────
        prob_up = 1.0 - result.recession_probability

        all_scores = [fs.score for fs in factor_scores.values()]
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 50.0
        score_prob = self._map_score_to_probability(avg_score, min_val=0, max_val=100)
        prob_up = 0.5 * prob_up + 0.5 * score_prob

        confidence = 0.5
        if result.yield_curve < -0.5:
            confidence = 0.90
        elif result.yield_curve > 1.0 and result.unemployment < 5.0:
            confidence = 0.80

        prob_up = max(0.01, min(0.99, prob_up))
        confidence = max(0.0, min(1.0, confidence))

        direction = "BULLISH" if prob_up > self.long_threshold else "BEARISH" if prob_up < self.short_threshold else "NEUTRAL"
        summary = (
            f"Macroeconomic environment is {result.regime} ({direction}). "
            f"Recession probability: {result.recession_probability * 100:.0f}%. "
            f"Fed Funds: {result.fed_funds_rate:.2f}%. "
            f"10Y-2Y Yield Curve: {result.yield_curve:.2f}%. "
        )
        reasoning.insert(0, summary)

        return AgentResult(
            agent_name=self.name,
            probability_up=prob_up,
            confidence=confidence,
            reasoning=" ".join(reasoning),
            factor_scores=factor_scores,
            warnings=warnings + result.warnings,
        )
