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
from typing import Any

import yfinance as yf

from agents.base import BaseAgent
from agents.state import AgentResult, FactorScore
from data.macro import MacroData
from data.currency import CurrencyData
from quant_engine.macro import analyze_macro_environment
from config.settings_manager import settings

logger = logging.getLogger(__name__)


def _yf_momentum(ticker: str, lookback: int = 22, period: str = "3mo") -> float:
    """1-month momentum % for a yfinance ticker."""
    try:
        df = yf.download(ticker, period=period, interval="1d",
                         auto_adjust=True, progress=False)
        s = df["Close"].squeeze().dropna()
        if len(s) > lookback:
            return float((s.iloc[-1] / s.iloc[-lookback - 1] - 1) * 100)
    except Exception:
        pass
    return 0.0


def _yf_rel_strength(t1: str, t2: str, lookback: int = 22, period: str = "3mo") -> float:
    """Relative strength: t1 vs t2 return difference over lookback days."""
    try:
        import yfinance as yf
        d1 = yf.download(t1, period=period, interval="1d", auto_adjust=True, progress=False)
        d2 = yf.download(t2, period=period, interval="1d", auto_adjust=True, progress=False)
        s1 = d1["Close"].squeeze().dropna()
        s2 = d2["Close"].squeeze().dropna()
        if len(s1) > lookback and len(s2) > lookback:
            r1 = (s1.iloc[-1] / s1.iloc[-lookback - 1] - 1) * 100
            r2 = (s2.iloc[-1] / s2.iloc[-lookback - 1] - 1) * 100
            return float(r1 - r2)
    except Exception:
        pass
    return 0.0


class MacroAgent(BaseAgent):
    """
    Evaluates Federal Reserve Economic Data (FRED) and cross-asset signals
    to determine recession probability and macro regime.
    """
    name = "macro"

    def _run_analysis(self, ticker: str, data: Any, **kwargs) -> AgentResult:
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
        factor_scores["recession_risk"] = FactorScore(
            name="Recession Probability",
            value=result.recession_probability * 100,
            score=(100.0 if result.recession_probability < rec_low
                   else 0.0 if result.recession_probability > 0.6
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
            cpi_series = macro_data.get_series("CPIAUCSL", years_back=2)
            if len(cpi_series) >= 13:
                cpi_now = float(cpi_series.iloc[-1].iloc[0])
                cpi_year_ago = float(cpi_series.iloc[-13].iloc[0])
                cpi_yoy = (cpi_now / cpi_year_ago - 1) * 100

                cpi_score = (80.0 if cpi_yoy < 2.5
                             else 50.0 if cpi_yoy < 4.0
                             else 20.0)
                factor_scores["cpi_inflation"] = FactorScore(
                    name="CPI Inflation (YoY)",
                    value=cpi_yoy,
                    score=cpi_score,
                    interpretation=f"CPI YoY: {cpi_yoy:.1f}%",
                )
                if cpi_yoy > 5.0:
                    warnings.append(f"High inflation: CPI {cpi_yoy:.1f}% YoY — Fed tightening risk.")
                    reasoning.append(f"Inflation elevated ({cpi_yoy:.1f}% YoY) — rate hike risk.")
                elif cpi_yoy < 2.0:
                    reasoning.append(f"Inflation benign ({cpi_yoy:.1f}% YoY) — Fed has room to cut.")
                else:
                    reasoning.append(f"Inflation moderate ({cpi_yoy:.1f}% YoY).")
        except Exception as e:
            reasoning.append(f"CPI data unavailable ({e}).")

        # ── 5. VIX ────────────────────────────────────────────────────────
        if "vix" in snapshot:
            vix = snapshot["vix"]
            vix_score = max(10.0, min(90.0, 90.0 - (vix - 10) * 2.2))
            factor_scores["vix"] = FactorScore(
                name="VIX Fear Index",
                value=vix,
                score=vix_score,
                interpretation=f"VIX: {vix:.1f}",
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
            pmi_series = macro_data.get_series("NAPM", years_back=1)
            if pmi_series is not None and len(pmi_series) >= 2:
                pmi_val = float(pmi_series.iloc[-1].iloc[0])
                pmi_score = (85.0 if pmi_val > 55 else 60.0 if pmi_val > 50 else 35.0 if pmi_val > 45 else 15.0)
                factor_scores["ism_pmi"] = FactorScore(
                    name="ISM Manufacturing PMI",
                    value=pmi_val,
                    score=pmi_score,
                    interpretation=f"ISM PMI: {pmi_val:.1f} ({'expansion' if pmi_val > 50 else 'contraction'})",
                )
                if pmi_val < 48:
                    reasoning.append(f"ISM PMI deeply contractionary ({pmi_val:.1f}) — manufacturing recession signal.")
                elif pmi_val > 55:
                    reasoning.append(f"ISM PMI expansionary ({pmi_val:.1f}) — manufacturing growth strong.")
        except Exception as e:
            reasoning.append(f"ISM PMI unavailable ({e}).")

        # ── 13. Consumer Sentiment (FRED: UMCSENT) ────────────────────────
        try:
            umcs = macro_data.get_series("UMCSENT", years_back=1)
            if umcs is not None and len(umcs) >= 2:
                cs_now  = float(umcs.iloc[-1].iloc[0])
                cs_prev = float(umcs.iloc[-2].iloc[0])
                cs_change = cs_now - cs_prev
                cs_score = (80.0 if cs_now > 90 else 60.0 if cs_now > 70 else 40.0 if cs_now > 55 else 20.0)
                factor_scores["consumer_sentiment"] = FactorScore(
                    name="UMich Consumer Sentiment",
                    value=cs_now,
                    score=cs_score,
                    interpretation=f"UMich Sentiment: {cs_now:.1f} ({cs_change:+.1f} MoM)",
                )
                if cs_now < 55:
                    reasoning.append(f"Consumer sentiment very weak ({cs_now:.1f}) — demand destruction risk.")
                elif cs_now > 90:
                    reasoning.append(f"Consumer sentiment strong ({cs_now:.1f}) — consumption tailwind.")
        except Exception as e:
            reasoning.append(f"Consumer sentiment unavailable ({e}).")

        # ── 14. Initial Jobless Claims (FRED: ICSA) ───────────────────────
        try:
            icsa = macro_data.get_series("ICSA", years_back=1)
            if icsa is not None and len(icsa) >= 4:
                claims_now = float(icsa.iloc[-1].iloc[0])
                claims_4wa = float(icsa.iloc[-4:].mean().iloc[0])   # 4-week avg
                claims_score = (80.0 if claims_now < 220000 else 60.0 if claims_now < 260000 else 35.0 if claims_now < 350000 else 15.0)
                factor_scores["jobless_claims"] = FactorScore(
                    name="Initial Jobless Claims",
                    value=claims_now,
                    score=claims_score,
                    interpretation=f"Claims: {claims_now/1000:.0f}K (4W avg: {claims_4wa/1000:.0f}K)",
                )
                if claims_now > 350000:
                    reasoning.append(f"Jobless claims elevated ({claims_now/1000:.0f}K) — labor deterioration.")
        except Exception as e:
            reasoning.append(f"Jobless claims unavailable ({e}).")

        # ── 15. 10Y TIPS Breakeven Inflation (FRED: T10YIE) ──────────────
        try:
            tips = macro_data.get_series("T10YIE", years_back=1)
            if tips is not None and len(tips) >= 2:
                tips_val = float(tips.iloc[-1].iloc[0])
                tips_score = (70.0 if 1.5 < tips_val < 2.5 else 50.0 if 1.0 < tips_val <= 1.5 else 30.0 if tips_val > 3.0 else 40.0)
                factor_scores["tips_breakeven"] = FactorScore(
                    name="10Y TIPS Breakeven Inflation",
                    value=tips_val,
                    score=tips_score,
                    interpretation=f"Breakeven: {tips_val:.2f}% ({'anchored' if 1.5 < tips_val < 2.5 else 'elevated' if tips_val > 2.5 else 'deflation risk'})",
                )
                if tips_val > 3.0:
                    reasoning.append(f"TIPS breakeven {tips_val:.2f}% — market pricing in persistent inflation.")
        except Exception as e:
            reasoning.append(f"TIPS breakeven unavailable ({e}).")

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

        direction = "BULLISH" if prob_up > 0.55 else "BEARISH" if prob_up < 0.45 else "NEUTRAL"
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
