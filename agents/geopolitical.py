"""
AlphaAgent — Geopolitical Agent

Monitors macro/geopolitical stress signals via liquid ETF proxies.
No geopolitical API required — everything is derived from price data.

Factors tracked:
  1.  Oil shock (XLE vs SPY relative strength)
  2.  Gold safe-haven demand (GLD price momentum)
  3.  EM stress (EEM vs SPY spread)
  4.  Defense sector RS (ITA vs SPY)
  5.  VIX regime
  6.  Copper/Gold ratio (industrial demand)
  7.  Global equity breadth (ACWI vs SPY)
  8.  USD safe-haven (DXY momentum from CurrencyData)
  9.  Credit stress (HYG/LQD spread)
  10. Sector rotation signal (XLF, XLK, XLE vs SPY)
  11. Put/Call ratio (from options intel if available)
"""

import logging
from typing import Any

import numpy as np
import yfinance as yf

from agents.base import BaseAgent
from agents.state import AgentResult, Direction, FactorScore
from data.currency import CurrencyData

logger = logging.getLogger(__name__)


def _fetch(ticker: str, period: str = "3mo") -> "pd.Series":
    import pandas as pd
    try:
        df = yf.download(ticker, period=period, interval="1d",
                         auto_adjust=True, progress=False)
        if df.empty:
            return pd.Series(dtype=float)
        return df["Close"].squeeze().dropna()
    except Exception as e:
        logger.warning(f"[GeoAgent] {ticker}: {e}")
        import pandas as pd
        return pd.Series(dtype=float)


def _rel_strength(ticker: str, bench: str = "SPY", period: str = "3mo",
                  lookback: int = 63) -> float:
    """
    Compute relative strength of ticker vs benchmark over lookback days.
    Returns % difference: positive = outperforming.
    """
    t = _fetch(ticker, period)
    b = _fetch(bench, period)
    if len(t) < lookback or len(b) < lookback:
        return 0.0
    t_ret = (t.iloc[-1] / t.iloc[-lookback] - 1) * 100
    b_ret = (b.iloc[-1] / b.iloc[-lookback] - 1) * 100
    return float(t_ret - b_ret)


def _momentum(ticker: str, period: str = "3mo", lookback: int = 22) -> float:
    """1-month price momentum %."""
    s = _fetch(ticker, period)
    if len(s) < lookback + 1:
        return 0.0
    return float((s.iloc[-1] / s.iloc[-lookback - 1] - 1) * 100)


class GeopoliticalAgent(BaseAgent):
    """
    Detects geopolitical and macro-systemic risk using ETF price signals.
    High geo-risk → bearish pressure + position size override.
    """

    name = "geopolitical"

    def _run_analysis(self, ticker: str, data: Any, **kwargs) -> AgentResult:
        prob_up = 0.5
        confidence = 0.35
        reasoning = []
        factor_scores = {}
        warnings = []

        # ── 1. Oil Shock — XLE vs SPY ─────────────────────────────────────
        oil_rs = _rel_strength("XLE", "SPY", lookback=22)
        if oil_rs > 5.0:
            prob_up -= 0.08
            reasoning.append(f"Oil shock: XLE outperforming SPY by {oil_rs:.1f}% — energy stress (bearish).")
        elif oil_rs < -5.0:
            prob_up += 0.03
            reasoning.append(f"Oil calm: XLE underperforming SPY by {abs(oil_rs):.1f}% — benign energy environment.")
        else:
            reasoning.append(f"Oil: neutral relative to market (RS {oil_rs:+.1f}%).")

        factor_scores["oil_shock"] = FactorScore(
            name="Oil Shock (XLE vs SPY)",
            value=oil_rs,
            score=max(10.0, min(90.0, 50.0 - oil_rs * 2.0)),
            interpretation=f"XLE vs SPY RS: {oil_rs:+.1f}%",
        )

        # ── 2. Gold Safe-Haven Demand ─────────────────────────────────────
        gold_mom = _momentum("GLD", lookback=22)
        if gold_mom > 5.0:
            prob_up -= 0.10
            confidence += 0.05
            reasoning.append(f"Gold surging +{gold_mom:.1f}% — safe-haven demand elevated (bearish geo signal).")
        elif gold_mom < -3.0:
            prob_up += 0.05
            reasoning.append(f"Gold falling {gold_mom:.1f}% — risk-off pressure easing.")
        else:
            reasoning.append(f"Gold: neutral momentum ({gold_mom:+.1f}%).")

        factor_scores["gold_haven"] = FactorScore(
            name="Gold Safe-Haven (GLD)",
            value=gold_mom,
            score=max(10.0, min(90.0, 50.0 - gold_mom * 3.0)),
            interpretation=f"GLD 1m momentum: {gold_mom:+.1f}%",
        )

        # ── 3. EM Stress — EEM vs SPY ─────────────────────────────────────
        em_rs = _rel_strength("EEM", "SPY", lookback=22)
        if em_rs < -5.0:
            prob_up -= 0.06
            confidence += 0.05
            reasoning.append(f"EM stress: EEM underperforming SPY by {abs(em_rs):.1f}% (bearish global signal).")
        elif em_rs > 3.0:
            prob_up += 0.04
            reasoning.append(f"EM outperforming SPY by {em_rs:.1f}% — global risk appetite healthy.")
        else:
            reasoning.append(f"EM relative strength neutral ({em_rs:+.1f}%).")

        factor_scores["em_stress"] = FactorScore(
            name="EM Stress (EEM vs SPY)",
            value=em_rs,
            score=max(10.0, min(90.0, 50.0 + em_rs * 2.5)),
            interpretation=f"EEM vs SPY RS: {em_rs:+.1f}%",
        )

        # ── 4. Defense Sector Relative Strength ───────────────────────────
        def_rs = _rel_strength("ITA", "SPY", lookback=22)
        if def_rs > 5.0:
            prob_up -= 0.07
            reasoning.append(f"Defense ETF (ITA) outperforming by {def_rs:.1f}% — war/conflict risk priced in.")
        else:
            reasoning.append(f"Defense RS: {def_rs:+.1f}% — no escalation signal.")

        factor_scores["defense_rs"] = FactorScore(
            name="Defense RS (ITA vs SPY)",
            value=def_rs,
            score=max(10.0, min(90.0, 50.0 - def_rs * 2.0)),
            interpretation=f"ITA vs SPY RS: {def_rs:+.1f}%",
        )

        # ── 5. VIX Regime ─────────────────────────────────────────────────
        vix_series = _fetch("^VIX", period="1mo")
        if not vix_series.empty:
            vix = float(vix_series.iloc[-1])
            if vix > 30:
                prob_up -= 0.10
                confidence += 0.10
                warnings.append(f"VIX={vix:.1f} — extreme fear zone.")
                reasoning.append(f"VIX={vix:.1f} (EXTREME). Crisis-level fear.")
            elif vix > 20:
                prob_up -= 0.05
                reasoning.append(f"VIX={vix:.1f} (elevated).")
            else:
                prob_up += 0.03
                reasoning.append(f"VIX={vix:.1f} (calm).")

            factor_scores["vix_regime"] = FactorScore(
                name="VIX Fear Index",
                value=vix,
                score=max(10.0, min(90.0, 90.0 - (vix - 10) * 2.5)),
                interpretation=f"VIX: {vix:.1f}",
            )

        # ── 6. Copper/Gold Ratio ──────────────────────────────────────────
        copper = _fetch("COPX", period="3mo")
        gold = _fetch("GLD", period="3mo")
        if len(copper) > 22 and len(gold) > 22:
            cg_now = float(copper.iloc[-1] / gold.iloc[-1])
            cg_prev = float(copper.iloc[-22] / gold.iloc[-22])
            cg_chg = (cg_now / cg_prev - 1) * 100 if cg_prev != 0 else 0.0
            if cg_chg > 3.0:
                prob_up += 0.05
                reasoning.append(f"Copper/Gold +{cg_chg:.1f}% — industrial demand signals recovery.")
            elif cg_chg < -3.0:
                prob_up -= 0.05
                reasoning.append(f"Copper/Gold {cg_chg:.1f}% — demand contraction, defensive posture.")
            else:
                reasoning.append(f"Copper/Gold ratio neutral ({cg_chg:+.1f}%).")

            factor_scores["copper_gold"] = FactorScore(
                name="Copper/Gold Ratio",
                value=cg_chg,
                score=max(10.0, min(90.0, 50.0 + cg_chg * 5.0)),
                interpretation=f"CG ratio 1m change: {cg_chg:+.1f}%",
            )

        # ── 7. Global Equity Breadth — ACWI vs SPY ───────────────────────
        acwi_rs = _rel_strength("ACWI", "SPY", lookback=22)
        if acwi_rs < -3.0:
            prob_up -= 0.04
            reasoning.append(f"Global breadth weakening: ACWI vs SPY RS {acwi_rs:+.1f}%.")
        elif acwi_rs > 2.0:
            prob_up += 0.03
            reasoning.append(f"Global breadth healthy: ACWI vs SPY RS {acwi_rs:+.1f}%.")
        else:
            reasoning.append(f"Global breadth neutral ({acwi_rs:+.1f}%).")

        factor_scores["global_breadth"] = FactorScore(
            name="Global Breadth (ACWI vs SPY)",
            value=acwi_rs,
            score=max(10.0, min(90.0, 50.0 + acwi_rs * 5.0)),
            interpretation=f"ACWI vs SPY RS: {acwi_rs:+.1f}%",
        )

        # ── 8. USD Safe-Haven ─────────────────────────────────────────────
        try:
            fx_data = CurrencyData()
            fx_snap = fx_data.get_snapshot()
            dxy_chg = fx_snap.dxy_1m_change
            if dxy_chg > 2.0:
                prob_up -= 0.06
                reasoning.append(f"USD strengthening ({dxy_chg:+.1f}% 1m) — safe-haven flight, risk-off.")
            elif dxy_chg < -2.0:
                prob_up += 0.04
                reasoning.append(f"USD weakening ({dxy_chg:+.1f}% 1m) — risk-on environment.")
            else:
                reasoning.append(f"USD neutral ({dxy_chg:+.1f}% 1m).")

            factor_scores["usd_haven"] = FactorScore(
                name="USD Safe-Haven (DXY)",
                value=dxy_chg,
                score=max(10.0, min(90.0, 50.0 - dxy_chg * 5.0)),
                interpretation=f"DXY 1m change: {dxy_chg:+.1f}%",
            )
        except Exception as e:
            reasoning.append(f"DXY data skipped ({e}).")

        # ── 9. Credit Stress — HYG/LQD ───────────────────────────────────
        hyg = _fetch("HYG", period="3mo")
        lqd = _fetch("LQD", period="3mo")
        if len(hyg) > 22 and len(lqd) > 22:
            ratio_now = float(hyg.iloc[-1] / lqd.iloc[-1])
            ratio_prev = float(hyg.iloc[-22] / lqd.iloc[-22])
            credit_chg = (ratio_now / ratio_prev - 1) * 100 if ratio_prev != 0 else 0.0
            if credit_chg < -2.0:
                prob_up -= 0.07
                confidence += 0.05
                reasoning.append(f"Credit stress: HYG/LQD ratio {credit_chg:.1f}% (spreads widening).")
            elif credit_chg > 1.0:
                prob_up += 0.04
                reasoning.append(f"Credit markets healthy: HYG/LQD ratio {credit_chg:+.1f}%.")
            else:
                reasoning.append(f"Credit spreads neutral ({credit_chg:+.1f}%).")

            factor_scores["credit_stress"] = FactorScore(
                name="Credit Stress (HYG/LQD)",
                value=credit_chg,
                score=max(10.0, min(90.0, 50.0 + credit_chg * 8.0)),
                interpretation=f"HYG/LQD ratio 1m: {credit_chg:+.1f}%",
            )

        # ── 10. Sector Rotation ───────────────────────────────────────────
        xlf_rs = _rel_strength("XLF", "SPY", lookback=22)  # Financials
        xlk_rs = _rel_strength("XLK", "SPY", lookback=22)  # Tech
        xle_rs = _rel_strength("XLE", "SPY", lookback=22)  # Energy

        # Tech leadership = growth regime (bullish); energy/financials = stagflation risk
        rotation_score = xlk_rs * 1.5 - xle_rs * 1.0 - xlf_rs * 0.5
        if rotation_score > 5.0:
            prob_up += 0.03
            reasoning.append("Sector rotation bullish: tech leading (growth regime).")
        elif rotation_score < -5.0:
            prob_up -= 0.04
            reasoning.append("Sector rotation defensive: energy/financials leading (inflation/recession risk).")
        else:
            reasoning.append(f"Sector rotation neutral (score {rotation_score:+.1f}).")

        factor_scores["sector_rotation"] = FactorScore(
            name="Sector Rotation Signal",
            value=rotation_score,
            score=max(10.0, min(90.0, 50.0 + rotation_score * 2.0)),
            interpretation=f"XLK RS {xlk_rs:+.1f}% | XLE RS {xle_rs:+.1f}% | XLF RS {xlf_rs:+.1f}%",
        )

        # ── New: Shipping / Supply Chain Stress (Baltic Dry Index proxy) ──
        try:
            import yfinance as yf
            # BDRY = Breakwave Dry Bulk Shipping ETF (proxy for Baltic Dry Index)
            bdry = yf.download("BDRY", period="3mo", interval="1d", auto_adjust=True, progress=False)
            if not bdry.empty and len(bdry) > 20:
                bdry_close = bdry["Close"].squeeze().dropna()
                bdry_mom = (float(bdry_close.iloc[-1]) / float(bdry_close.iloc[-21]) - 1) * 100
                sc_score = max(10.0, min(90.0, 50.0 + bdry_mom * 1.5))
                factor_scores["supply_chain"] = FactorScore(
                    name="Supply Chain Stress (Shipping)",
                    value=round(bdry_mom, 2),
                    score=sc_score,
                    interpretation=f"Dry bulk shipping 1M: {bdry_mom:+.1f}% ({'supply chain easing' if bdry_mom > 5 else 'tightening' if bdry_mom < -5 else 'stable'})",
                )
                if bdry_mom < -20:
                    reasoning.append(f"Shipping index collapsing ({bdry_mom:.1f}%) — demand destruction / supply chain stress.")
        except Exception:
            pass

        # ── New: Commodity Concentration Risk (Energy Shock Proxy) ────────
        try:
            import yfinance as yf
            xle = yf.download("XLE", period="3mo", interval="1d", auto_adjust=True, progress=False)
            xlb = yf.download("XLB", period="3mo", interval="1d", auto_adjust=True, progress=False)
            if not xle.empty and not xlb.empty:
                xle_mom = (float(xle["Close"].squeeze().iloc[-1]) / float(xle["Close"].squeeze().iloc[-21]) - 1) * 100
                xlb_mom = (float(xlb["Close"].squeeze().iloc[-1]) / float(xlb["Close"].squeeze().iloc[-21]) - 1) * 100
                comm_signal = (xle_mom + xlb_mom) / 2
                comm_score = max(10.0, min(90.0, 50.0 + comm_signal * 1.2))
                factor_scores["commodity_shock"] = FactorScore(
                    name="Commodity Shock (Energy+Materials)",
                    value=round(comm_signal, 2),
                    score=comm_score,
                    interpretation=f"XLE+XLB avg 1M: {comm_signal:+.1f}% ({'commodity inflation' if comm_signal > 8 else 'demand cooling' if comm_signal < -8 else 'neutral'})",
                )
        except Exception:
            pass

        # ── New: EM Contagion Risk (EM Bond Stress) ───────────────────────
        try:
            import yfinance as yf
            emb = yf.download("EMB", period="3mo", interval="1d", auto_adjust=True, progress=False)
            if not emb.empty and len(emb) > 20:
                emb_mom = (float(emb["Close"].squeeze().iloc[-1]) / float(emb["Close"].squeeze().iloc[-21]) - 1) * 100
                emb_score = max(10.0, min(90.0, 50.0 + emb_mom * 5))
                factor_scores["em_contagion"] = FactorScore(
                    name="EM Contagion Risk (EMB)",
                    value=round(emb_mom, 2),
                    score=emb_score,
                    interpretation=f"EM bonds 1M: {emb_mom:+.1f}% ({'EM stress spreading' if emb_mom < -3 else 'stable'})",
                )
        except Exception:
            pass

        # ── 11. Geopolitical Override Logic ───────────────────────────────
        all_scores = [fs.score for fs in factor_scores.values()]
        avg_score = np.mean(all_scores) if all_scores else 50.0

        if avg_score < 25.0:
            warnings.append("GEOPOLITICAL OVERRIDE: Extreme risk — cap position size at 35%.")

        # Clamp
        prob_up = max(0.05, min(0.95, prob_up))
        confidence = max(0.0, min(1.0, confidence))

        from agents.state import Direction
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
            warnings=warnings,
        )
