"""
AlphaAgent — Currency Agent

Evaluates FX market dynamics and their impact on US-listed stocks.

Factors:
  1. DXY regime (strong dollar = headwind for multinationals)
  2. EUR/USD momentum
  3. USD/JPY (carry trade signal — JPY strengthen = risk-off)
  4. USD/CNY (China trade/macro stress)
  5. GBP/USD stability
  6. EM currency stress (broad EM FX vs USD)
  7. Revenue-weighted FX impact (estimated from ticker's sector)
  8. Carry trade risk (JPY cross momentum)
  9. Petro-currency signal (CAD/AUD momentum vs oil)
"""

import logging
from typing import Any

import yfinance as yf

from agents.base import BaseAgent
from agents.state import AgentResult, Direction, FactorScore
from data.currency import CurrencyData

logger = logging.getLogger(__name__)

# Sector → international revenue exposure estimates (approx % of non-US revenue)
SECTOR_FX_EXPOSURE = {
    "Technology": 0.55,
    "Communication Services": 0.40,
    "Consumer Discretionary": 0.35,
    "Consumer Staples": 0.45,
    "Health Care": 0.40,
    "Industrials": 0.40,
    "Materials": 0.50,
    "Energy": 0.45,
    "Financials": 0.30,
    "Real Estate": 0.10,
    "Utilities": 0.10,
}


class CurrencyAgent(BaseAgent):
    """
    Analyzes FX trends and their projected impact on a stock.
    Strong USD is generally bearish for multinationals; weak USD is bullish.
    """

    name = "currency"

    def __init__(self):
        super().__init__()
        self.fx_data = CurrencyData()

    def _run_analysis(self, ticker: str, data: Any, **kwargs) -> AgentResult:
        prob_up = 0.5
        confidence = 0.30
        reasoning = []
        factor_scores = {}
        warnings = []

        # Fetch FX snapshot
        try:
            snap = self.fx_data.get_snapshot()
        except Exception as e:
            return self._build_error_result(f"Currency data unavailable: {e}")

        # Get sector for revenue exposure estimate
        sector = "Unknown"
        try:
            info = data.get_info()
            sector = info.get("sector", "Unknown")
        except Exception:
            pass

        fx_exposure = SECTOR_FX_EXPOSURE.get(sector, 0.35)

        # ── 1. DXY Regime ─────────────────────────────────────────────────
        dxy_1m = snap.dxy_1m_change
        dxy_3m = snap.dxy_3m_change

        if dxy_1m > 2.0:
            # Strong dollar — headwind for multinationals
            impact = -dxy_1m * fx_exposure * 0.3
            prob_up += impact
            confidence += 0.05
            reasoning.append(
                f"USD strengthened {dxy_1m:+.1f}% (1m). "
                f"With ~{fx_exposure*100:.0f}% intl revenue exposure, "
                f"estimated earnings headwind ~{abs(impact)*100:.1f}% (bearish)."
            )
        elif dxy_1m < -2.0:
            impact = abs(dxy_1m) * fx_exposure * 0.3
            prob_up += impact
            reasoning.append(
                f"USD weakened {dxy_1m:+.1f}% (1m). "
                f"FX tailwind for intl revenue (~{fx_exposure*100:.0f}% exposure)."
            )
        else:
            reasoning.append(f"DXY {dxy_1m:+.1f}% (1m) — neutral USD movement.")

        factor_scores["dxy_regime"] = FactorScore(
            name="US Dollar Index (DXY)",
            value=dxy_1m,
            score=max(10.0, min(90.0, 50.0 - dxy_1m * 4.0)),
            interpretation=f"DXY 1m: {dxy_1m:+.1f}% | 3m: {dxy_3m:+.1f}%",
        )

        # ── 2. EUR/USD Momentum ───────────────────────────────────────────
        if snap.eurusd > 0:
            # Strong EUR = weak USD = bullish for US multinationals
            eur_score = 65.0 if snap.eurusd > 1.05 else 35.0 if snap.eurusd < 0.95 else 50.0
            reasoning.append(f"EUR/USD: {snap.eurusd:.4f}.")
            factor_scores["eurusd"] = FactorScore(
                name="EUR/USD Rate",
                value=snap.eurusd,
                score=eur_score,
                interpretation=f"EUR/USD: {snap.eurusd:.4f}",
            )

        # ── 3. USD/JPY — Carry Trade Signal ─────────────────────────────
        if snap.usdjpy > 0:
            # High USD/JPY = yen weak = carry trade on = risk appetite
            # JPY surge (USD/JPY drops) = carry unwind = risk-off
            jpy_score = 65.0 if snap.usdjpy > 140 else 30.0 if snap.usdjpy < 120 else 50.0

            if snap.usdjpy < 120:
                prob_up -= 0.08
                confidence += 0.08
                warnings.append("CARRY UNWIND RISK: JPY surging — risk-off signal.")
                reasoning.append(f"USD/JPY={snap.usdjpy:.1f} — yen surging, carry trade unwinding (bearish).")
            elif snap.usdjpy > 150:
                prob_up += 0.03
                reasoning.append(f"USD/JPY={snap.usdjpy:.1f} — yen weak, carry trade intact (neutral-bullish).")
            else:
                reasoning.append(f"USD/JPY={snap.usdjpy:.1f} — carry trade stable.")

            factor_scores["usdjpy_carry"] = FactorScore(
                name="USD/JPY Carry Signal",
                value=snap.usdjpy,
                score=jpy_score,
                interpretation=f"USD/JPY: {snap.usdjpy:.2f}",
            )

        # ── 4. USD/CNY — China Macro Stress ──────────────────────────────
        if snap.usdcny > 0:
            # Rising USD/CNY = CNY weakening = China stress
            cny_stress = snap.usdcny > 7.3
            if cny_stress:
                prob_up -= 0.04
                reasoning.append(f"USD/CNY={snap.usdcny:.3f} — CNY under pressure (China macro stress).")
            else:
                reasoning.append(f"USD/CNY={snap.usdcny:.3f} — CNY stable.")

            factor_scores["usdcny"] = FactorScore(
                name="USD/CNY (China Stress)",
                value=snap.usdcny,
                score=30.0 if snap.usdcny > 7.3 else 60.0,
                interpretation=f"USD/CNY: {snap.usdcny:.3f}",
            )

        # ── 5. GBP/USD ───────────────────────────────────────────────────
        if snap.gbpusd > 0:
            factor_scores["gbpusd"] = FactorScore(
                name="GBP/USD Rate",
                value=snap.gbpusd,
                score=65.0 if snap.gbpusd > 1.25 else 40.0,
                interpretation=f"GBP/USD: {snap.gbpusd:.4f}",
            )

        # ── 6. EM Currency Stress ─────────────────────────────────────────
        em_stress = snap.em_currency_stress
        if em_stress > 3.0:
            prob_up -= 0.06
            confidence += 0.05
            warnings.append(f"EM currency stress elevated: avg USD gain vs EM = {em_stress:+.1f}%")
            reasoning.append(f"EM FX stress: USD gained {em_stress:+.1f}% vs EM basket (risk-off).")
        elif em_stress < -2.0:
            prob_up += 0.04
            reasoning.append(f"EM FX supportive: USD {em_stress:+.1f}% vs EM (risk appetite).")
        else:
            reasoning.append(f"EM FX stress neutral ({em_stress:+.1f}%).")

        factor_scores["em_fx_stress"] = FactorScore(
            name="EM Currency Stress",
            value=em_stress,
            score=max(10.0, min(90.0, 50.0 - em_stress * 5.0)),
            interpretation=f"USD vs EM basket (1m avg): {em_stress:+.1f}%",
        )

        # ── 7. Revenue-Weighted FX Impact ─────────────────────────────────
        # Estimated translation impact from DXY and sector exposure
        translation_impact = -dxy_1m * fx_exposure * 0.25
        factor_scores["fx_translation"] = FactorScore(
            name="FX Translation Impact",
            value=translation_impact,
            score=max(10.0, min(90.0, 50.0 + translation_impact * 10.0)),
            interpretation=(
                f"Est. FX translation: {translation_impact:+.1f}% | "
                f"Sector: {sector} | Intl exposure: {fx_exposure*100:.0f}%"
            ),
        )

        # ── 8. Petro-Currency Signal — CAD / AUD ─────────────────────────
        try:
            import yfinance as yf
            import pandas as pd
            cad_s = yf.download("CAD=X", period="3mo", interval="1d",
                                auto_adjust=True, progress=False)["Close"].squeeze().dropna()
            aud_s = yf.download("AUD=X", period="3mo", interval="1d",
                                auto_adjust=True, progress=False)["Close"].squeeze().dropna()

            petro_signals = []
            for s, name in [(cad_s, "CAD"), (aud_s, "AUD")]:
                if len(s) >= 22:
                    chg = (s.iloc[-1] / s.iloc[-22] - 1) * 100
                    petro_signals.append(float(chg))

            if petro_signals:
                avg_petro = sum(petro_signals) / len(petro_signals)
                # Rising commodity FX = risk-on signal
                if avg_petro > 2.0:
                    prob_up += 0.03
                    reasoning.append(f"Petro-currencies bullish (avg {avg_petro:+.1f}% 1m).")
                elif avg_petro < -2.0:
                    prob_up -= 0.03
                    reasoning.append(f"Petro-currencies bearish (avg {avg_petro:+.1f}% 1m).")

                factor_scores["petro_fx"] = FactorScore(
                    name="Petro-Currency Signal (CAD/AUD)",
                    value=avg_petro,
                    score=max(10.0, min(90.0, 50.0 + avg_petro * 5.0)),
                    interpretation=f"CAD+AUD avg 1m: {avg_petro:+.1f}%",
                )
        except Exception:
            pass

        # ── New: Carry Trade Attractiveness (Rate Differential) ───────────
        try:
            from data.macro import MacroData
            macro_data = MacroData()
            fed_series = macro_data.get_series("FEDFUNDS", years_back=1)
            if fed_series is not None and len(fed_series) >= 1:
                us_rate = float(fed_series.iloc[-1].iloc[0])
                ecb_series = macro_data.get_series("ECBDFR", years_back=1)
                global_rate = float(ecb_series.iloc[-1].iloc[0]) if ecb_series is not None and len(ecb_series) >= 1 else us_rate
                rate_diff = us_rate - global_rate
                carry_score = (70.0 if rate_diff > 1.0 else 50.0 if rate_diff > 0 else 35.0)
                factor_scores["carry_trade"] = FactorScore(
                    name="Carry Trade (USD Rate Advantage)",
                    value=round(rate_diff, 2),
                    score=carry_score,
                    interpretation=f"Fed {us_rate:.2f}% vs ECB {global_rate:.2f}% → diff {rate_diff:+.2f}% — {'USD carry attractive' if rate_diff > 1 else 'narrowing carry' if rate_diff < 0 else 'moderate carry'}",
                )
        except Exception:
            pass

        # ── New: Real Interest Rate (Nominal - Breakeven) ─────────────────
        try:
            from data.macro import MacroData
            macro_data2 = MacroData()
            tips_be = macro_data2.get_series("T10YIE", years_back=1)
            tnx = macro_data2.get_series("DGS10", years_back=1)
            if tips_be is not None and tnx is not None and len(tips_be) >= 1 and len(tnx) >= 1:
                nominal = float(tnx.iloc[-1].iloc[0])
                breakeven = float(tips_be.iloc[-1].iloc[0])
                real_rate = nominal - breakeven
                rr_score = (65.0 if real_rate > 1.0 else 50.0 if real_rate > 0 else 40.0 if real_rate > -1.0 else 25.0)
                factor_scores["real_interest_rate"] = FactorScore(
                    name="Real Interest Rate (10Y)",
                    value=round(real_rate, 3),
                    score=rr_score,
                    interpretation=f"Real rate: {real_rate:+.2f}% (Nominal {nominal:.2f}% - Breakeven {breakeven:.2f}%) — {'positive real yield (USD +)' if real_rate > 0 else 'negative real rate (USD -)'}",
                )
        except Exception:
            pass

        # ── New: EM FX Pressure Index (DXY vs EM FX basket) ──────────────
        try:
            import yfinance as yf
            # EEM as proxy for EM equity + FX health
            eem = yf.download("EEM", period="3mo", interval="1d", auto_adjust=True, progress=False)
            if not eem.empty and len(eem) > 20:
                eem_mom = (float(eem["Close"].squeeze().iloc[-1]) / float(eem["Close"].squeeze().iloc[-21]) - 1) * 100
                emfx_score = max(10.0, min(90.0, 50.0 + eem_mom * 2))
                factor_scores["em_fx_pressure"] = FactorScore(
                    name="EM FX Pressure (EEM Proxy)",
                    value=round(eem_mom, 2),
                    score=emfx_score,
                    interpretation=f"EEM 1M momentum: {eem_mom:+.1f}% — {'EM strength' if eem_mom > 5 else 'EM weakness/capital flight' if eem_mom < -5 else 'neutral'}",
                )
        except Exception:
            pass

        # ── New: DXY vs SMA(50) — Trend Direction ─────────────────────────
        try:
            dxy_s = yf.download("DX-Y.NYB", period="6mo", interval="1d",
                                auto_adjust=True, progress=False)
            if not dxy_s.empty and len(dxy_s) >= 50:
                dxy_c = dxy_s["Close"].squeeze().dropna()
                dxy_now_p = float(dxy_c.iloc[-1])
                dxy_sma50 = float(dxy_c.iloc[-50:].mean())
                dxy_pct_above = (dxy_now_p / dxy_sma50 - 1) * 100
                dxy_sma_score = max(10.0, min(90.0, 50.0 - dxy_pct_above * 5.0))
                factor_scores["dxy_sma50"] = FactorScore(
                    name="DXY vs SMA(50)",
                    value=round(dxy_pct_above, 2),
                    score=dxy_sma_score,
                    interpretation=f"DXY {dxy_pct_above:+.1f}% {'above' if dxy_pct_above > 0 else 'below'} 50d SMA — {'USD uptrend (multi-national headwind)' if dxy_pct_above > 1 else 'USD downtrend (earnings tailwind)' if dxy_pct_above < -1 else 'DXY at trend support/resistance'}",
                )
                if dxy_pct_above > 3:
                    warnings.append(f"DXY {dxy_pct_above:.1f}% above 50-SMA — strong dollar trend, earnings headwind for exporters.")
        except Exception:
            pass

        # ── 9. Carry Trade Risk Summary ───────────────────────────────────
        carry_risk = em_stress > 4.0 or (snap.usdjpy > 0 and snap.usdjpy < 125)
        if carry_risk:
            warnings.append("Carry trade unwind risk elevated — monitor yen and EM FX closely.")

        # Clamp
        prob_up = max(0.05, min(0.95, prob_up))
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
            warnings=warnings,
        )
