"""
AlphaAgent — Fundamental Agent

Analyzes a company's financial health by processing income statements,
balance sheets, and cash flows. Computes:
  - Piotroski F-Score (9-point profitability/leverage/efficiency system)
  - Altman Z-Score (bankruptcy predictor)
  - Beneish M-Score (earnings manipulation detector)
  - Valuation multiples: P/E, P/B, PEG, EV/EBITDA, P/S, FCF yield
  - DCF-implied upside
  - Revenue / earnings growth, ROE
"""

from typing import Any

from agents.base import BaseAgent
from agents.state import AgentResult, FactorScore
from quant_engine.scoring import compute_fundamental_scores
from config.settings_manager import settings


class FundamentalAgent(BaseAgent):
    """
    Analyzes SEC financial data (quarterly/annual) to determine
    company health, bankruptcy risk, and valuation attractiveness.
    """
    name = "fundamental"

    def _run_analysis(self, ticker: str, data: Any, **kwargs) -> AgentResult:
        info = data.get_info()
        financials = data.get_financials()

        inc = financials.get("income")
        bal = financials.get("balance")
        cf = financials.get("cashflow")

        if inc is None or bal is None or cf is None or inc.empty or bal.empty or cf.empty:
            return AgentResult(
                agent_name=self.name,
                probability_up=0.5,
                confidence=0.0,
                reasoning="Missing quarterly financial statements from data provider.",
                warnings=["Missing financials"],
            )

        scores = compute_fundamental_scores(inc, bal, cf, info)

        if "Error" in scores.data_quality or "Missing" in scores.data_quality or "Insufficient" in scores.data_quality:
            return AgentResult(
                agent_name=self.name,
                probability_up=0.5,
                confidence=0.0,
                reasoning=f"Data quality issue: {scores.data_quality}",
                warnings=[scores.data_quality],
            )

        warnings = []
        sf = settings.get_section("fundamental")

        # ── Factor Scores ────────────────────────────────────────────────
        factor_scores = {
            "piotroski": FactorScore(
                name="Piotroski F-Score",
                value=float(scores.piotroski_score),
                score=(scores.piotroski_score / 9.0) * 100.0,
                interpretation=f"Score: {scores.piotroski_score}/9 ({scores.f_score_interpretation})",
            ),
            "altman": FactorScore(
                name="Altman Z-Score",
                value=scores.altman_z_score,
                score=(100.0 if scores.altman_z_score > sf.get("altman_z_quality", 3.0)
                       else (scores.altman_z_score / sf.get("altman_z_quality", 3.0)) * 100 if scores.altman_z_score > 0
                       else 0.0),
                interpretation=f"Z-Score: {scores.altman_z_score:.2f} ({scores.z_score_interpretation})",
            ),
            "beneish": FactorScore(
                name="Beneish M-Score",
                value=scores.beneish_mscore,
                score=20.0 if scores.beneish_interpretation == "Likely Manipulator" else 75.0,
                interpretation=f"M-Score: {scores.beneish_mscore:.2f} ({scores.beneish_interpretation})",
            ),
            "margin": FactorScore(
                name="Gross Margin",
                value=scores.gross_margin,
                score=100.0 if scores.gross_margin > 40 else (scores.gross_margin / 40) * 100,
                interpretation=f"Margin: {scores.gross_margin:.1f}%",
            ),
            "debt": FactorScore(
                name="Debt-to-Equity",
                value=scores.debt_to_equity,
                score=(100.0 if scores.debt_to_equity < 1.0
                       else 0.0 if scores.debt_to_equity > sf.get("debt_to_equity_max", 3.0)
                       else 50.0),
                interpretation=f"D/E Ratio: {scores.debt_to_equity:.2f}",
            ),
        }

        # ── Valuation Factors ────────────────────────────────────────────
        if scores.pe_ratio > 0:
            pe_score = (100.0 if scores.pe_ratio < 15
                        else 60.0 if scores.pe_ratio < 25
                        else 30.0 if scores.pe_ratio < 40
                        else 10.0)
            factor_scores["pe_ratio"] = FactorScore(
                name="P/E Ratio (Trailing)",
                value=scores.pe_ratio,
                score=pe_score,
                interpretation=f"P/E: {scores.pe_ratio:.1f}x",
            )

        pb_uv = sf.get("pb_undervalued", 1.5)
        pb_fair = sf.get("pb_fair", 3.0)
        if scores.pb_ratio > 0:
            pb_score = (100.0 if scores.pb_ratio < pb_uv
                        else 65.0 if scores.pb_ratio < pb_fair
                        else 35.0 if scores.pb_ratio < pb_fair * 2
                        else 10.0)
            factor_scores["pb_ratio"] = FactorScore(
                name="Price/Book Ratio",
                value=scores.pb_ratio,
                score=pb_score,
                interpretation=f"P/B: {scores.pb_ratio:.2f}x",
            )

        peg_uv = sf.get("peg_undervalued", 1.0)
        peg_fair = sf.get("peg_fair", 2.0)
        if scores.peg_ratio > 0:
            peg_score = (100.0 if scores.peg_ratio < peg_uv
                         else 65.0 if scores.peg_ratio < peg_fair
                         else 30.0)
            factor_scores["peg_ratio"] = FactorScore(
                name="PEG Ratio",
                value=scores.peg_ratio,
                score=peg_score,
                interpretation=f"PEG: {scores.peg_ratio:.2f} ({'undervalued' if scores.peg_ratio < peg_uv else 'fair' if scores.peg_ratio < peg_fair else 'expensive'})",
            )

        ev_cheap = sf.get("ev_ebitda_cheap", 10)
        ev_fair = sf.get("ev_ebitda_fair", 20)
        ev_exp = sf.get("ev_ebitda_expensive", 30)
        if scores.ev_ebitda > 0:
            ev_score = (100.0 if scores.ev_ebitda < ev_cheap
                        else 60.0 if scores.ev_ebitda < ev_fair
                        else 30.0 if scores.ev_ebitda < ev_exp
                        else 10.0)
            factor_scores["ev_ebitda"] = FactorScore(
                name="EV/EBITDA",
                value=scores.ev_ebitda,
                score=ev_score,
                interpretation=f"EV/EBITDA: {scores.ev_ebitda:.1f}x",
            )

        ps_cheap = sf.get("ps_cheap", 2.0)
        ps_fair = sf.get("ps_fair", 4.0)
        if scores.ps_ratio > 0:
            ps_score = (100.0 if scores.ps_ratio < ps_cheap
                        else 60.0 if scores.ps_ratio < ps_fair * 1.25
                        else 30.0)
            factor_scores["ps_ratio"] = FactorScore(
                name="Price/Sales Ratio",
                value=scores.ps_ratio,
                score=ps_score,
                interpretation=f"P/S: {scores.ps_ratio:.2f}x",
            )

        fcf_good = sf.get("fcf_yield_good", 2.0)
        fcf_great = sf.get("fcf_yield_great", 5.0)
        if scores.fcf_yield != 0:
            fcf_score = (90.0 if scores.fcf_yield > fcf_great
                         else 60.0 if scores.fcf_yield > fcf_good
                         else 40.0 if scores.fcf_yield > 0
                         else 20.0)
            factor_scores["fcf_yield"] = FactorScore(
                name="FCF Yield",
                value=scores.fcf_yield,
                score=fcf_score,
                interpretation=f"FCF Yield: {scores.fcf_yield:.1f}%",
            )

        dcf_strong = sf.get("dcf_upside_strong", 30)
        dcf_mod = sf.get("dcf_upside_moderate", 10)
        dcf_lim = sf.get("dcf_downside_limit", -20)
        if scores.dcf_implied_upside != 0:
            dcf_score = (90.0 if scores.dcf_implied_upside > dcf_strong
                         else 65.0 if scores.dcf_implied_upside > dcf_mod
                         else 50.0 if scores.dcf_implied_upside > 0
                         else 30.0 if scores.dcf_implied_upside > dcf_lim
                         else 10.0)
            factor_scores["dcf_upside"] = FactorScore(
                name="DCF Implied Upside",
                value=scores.dcf_implied_upside,
                score=dcf_score,
                interpretation=f"DCF upside vs current: {scores.dcf_implied_upside:+.1f}%",
            )

        rg_strong = sf.get("revenue_growth_strong", 20)
        rg_mod = sf.get("revenue_growth_moderate", 10)
        eg_strong = sf.get("earnings_growth_strong", 20)
        eg_mod = sf.get("earnings_growth_moderate", 10)
        roe_strong = sf.get("roe_strong", 20)
        roe_mod = sf.get("roe_moderate", 10)

        if scores.revenue_growth_yoy != 0:
            rgrow_score = (90.0 if scores.revenue_growth_yoy > rg_strong
                           else 65.0 if scores.revenue_growth_yoy > rg_mod
                           else 50.0 if scores.revenue_growth_yoy > 0
                           else 25.0)
            factor_scores["revenue_growth"] = FactorScore(
                name="Revenue Growth (YoY)",
                value=scores.revenue_growth_yoy,
                score=rgrow_score,
                interpretation=f"Revenue growth: {scores.revenue_growth_yoy:+.1f}%",
            )

        if scores.earnings_growth_yoy != 0:
            egrow_score = (90.0 if scores.earnings_growth_yoy > eg_strong
                           else 65.0 if scores.earnings_growth_yoy > eg_mod
                           else 50.0 if scores.earnings_growth_yoy > 0
                           else 25.0)
            factor_scores["earnings_growth"] = FactorScore(
                name="Earnings Growth (YoY)",
                value=scores.earnings_growth_yoy,
                score=egrow_score,
                interpretation=f"Earnings growth: {scores.earnings_growth_yoy:+.1f}%",
            )

        if scores.roe != 0:
            roe_score = (90.0 if scores.roe > roe_strong
                         else 65.0 if scores.roe > roe_mod
                         else 40.0 if scores.roe > 0
                         else 15.0)
            factor_scores["roe"] = FactorScore(
                name="Return on Equity (ROE)",
                value=scores.roe,
                score=roe_score,
                interpretation=f"ROE: {scores.roe:.1f}%",
            )

        # ── New: Return on Assets ─────────────────────────────────────────
        try:
            roa = getattr(scores, 'return_on_assets', None)
            if roa is None:
                # Compute from info
                roa = info.get('returnOnAssets')
                if roa is not None:
                    roa = float(roa) * 100
            if roa is not None and roa != 0:
                roa_score = (90.0 if roa > 15 else 65.0 if roa > 8 else 45.0 if roa > 3 else 20.0)
                factor_scores["roa"] = FactorScore(
                    name="Return on Assets (ROA)",
                    value=round(roa, 2),
                    score=roa_score,
                    interpretation=f"ROA: {roa:.1f}%",
                )
        except Exception:
            pass

        # ── New: Operating Margin ─────────────────────────────────────────
        try:
            op_margin = info.get('operatingMargins')
            if op_margin is not None:
                op_margin = float(op_margin) * 100
                op_score = (90.0 if op_margin > 25 else 65.0 if op_margin > 15 else 45.0 if op_margin > 5 else 20.0 if op_margin > 0 else 5.0)
                factor_scores["operating_margin"] = FactorScore(
                    name="Operating Margin",
                    value=round(op_margin, 2),
                    score=op_score,
                    interpretation=f"Op. Margin: {op_margin:.1f}%",
                )
        except Exception:
            pass

        # ── New: Current Ratio (Liquidity) ────────────────────────────────
        try:
            current_ratio = info.get('currentRatio')
            if current_ratio is not None:
                cr = float(current_ratio)
                cr_score = (85.0 if cr > 2.0 else 60.0 if cr > 1.5 else 40.0 if cr > 1.0 else 15.0)
                factor_scores["current_ratio"] = FactorScore(
                    name="Current Ratio",
                    value=round(cr, 2),
                    score=cr_score,
                    interpretation=f"Current Ratio: {cr:.2f}x ({'liquid' if cr > 1.5 else 'adequate' if cr > 1.0 else 'tight liquidity'})",
                )
        except Exception:
            pass

        # ── New: Interest Coverage Ratio ──────────────────────────────────
        try:
            ebit = None; interest = None
            if inc is not None and not inc.empty:
                ebit_row = [r for r in inc.index if 'ebit' in str(r).lower() and 'ebitda' not in str(r).lower()]
                int_row  = [r for r in inc.index if 'interest' in str(r).lower() and 'expense' in str(r).lower()]
                if ebit_row:
                    ebit = float(inc.loc[ebit_row[0]].iloc[0])
                if int_row:
                    interest = abs(float(inc.loc[int_row[0]].iloc[0]))
            if ebit is not None and interest and interest > 0:
                icr = ebit / interest
                icr_score = (90.0 if icr > 10 else 70.0 if icr > 5 else 45.0 if icr > 2 else 15.0)
                factor_scores["interest_coverage"] = FactorScore(
                    name="Interest Coverage Ratio",
                    value=round(icr, 2),
                    score=icr_score,
                    interpretation=f"ICR: {icr:.1f}x ({'strong' if icr > 10 else 'adequate' if icr > 3 else 'weak'})",
                )
        except Exception:
            pass

        # ── New: Asset Turnover ───────────────────────────────────────────
        try:
            asset_turnover = info.get('assetTurnover') or info.get('totalAssets')
            # compute from financials if possible
            if inc is not None and not inc.empty and bal is not None and not bal.empty:
                rev_row = [r for r in inc.index if 'total revenue' in str(r).lower() or 'revenue' in str(r).lower()]
                ast_row = [r for r in bal.index if 'total assets' in str(r).lower()]
                if rev_row and ast_row:
                    rev = abs(float(inc.loc[rev_row[0]].iloc[0]))
                    ast = abs(float(bal.loc[ast_row[0]].iloc[0]))
                    if ast > 0:
                        at = rev / ast
                        at_score = (85.0 if at > 1.0 else 60.0 if at > 0.5 else 40.0)
                        factor_scores["asset_turnover"] = FactorScore(
                            name="Asset Turnover",
                            value=round(at, 3),
                            score=at_score,
                            interpretation=f"Asset Turnover: {at:.2f}x (efficiency of asset use)",
                        )
        except Exception:
            pass

        # ── New: EPS Surprise (Beat/Miss) ─────────────────────────────────
        try:
            eps_actual   = info.get('trailingEps')
            eps_estimate = info.get('forwardEps') or info.get('epsForward')
            eps_surprise_pct = info.get('earningsSurprise') or info.get('earningsQuarterlyGrowth')
            if eps_surprise_pct is not None:
                eps_surprise_pct = float(eps_surprise_pct) * 100
                eps_score = (85.0 if eps_surprise_pct > 10 else 60.0 if eps_surprise_pct > 2 else 50.0 if eps_surprise_pct > -2 else 25.0)
                factor_scores["eps_surprise"] = FactorScore(
                    name="EPS Surprise / Growth",
                    value=round(eps_surprise_pct, 2),
                    score=eps_score,
                    interpretation=f"EPS trend: {eps_surprise_pct:+.1f}%",
                )
        except Exception:
            pass

        # ── Composite Probability ─────────────────────────────────────────
        # Base from Piotroski (0–9 → 0.1 to 0.9)
        base_prob = (scores.piotroski_score / 9.0) * 0.8 + 0.1

        # Adjust for bankruptcy risk
        if scores.z_score_interpretation == "Distress":
            base_prob -= 0.30
            warnings.append("High bankruptcy risk (Altman Z-Score in Distress zone).")

        # Adjust for earnings manipulation
        if scores.beneish_interpretation == "Likely Manipulator":
            base_prob -= 0.15
            warnings.append("Earnings manipulation signal detected (Beneish M-Score > -1.78).")

        # Adjust for valuation
        if scores.dcf_implied_upside > 20:
            base_prob += 0.05
        elif scores.dcf_implied_upside < -20:
            base_prob -= 0.05

        if scores.revenue_growth_yoy > 15:
            base_prob += 0.04
        elif scores.revenue_growth_yoy < -5:
            base_prob -= 0.04

        # Blend valuation factor scores
        val_scores = [fs.score for k, fs in factor_scores.items()
                      if k in ("pe_ratio", "ev_ebitda", "fcf_yield", "dcf_upside")]
        if val_scores:
            val_avg = sum(val_scores) / len(val_scores)
            val_prob = self._map_score_to_probability(val_avg, min_val=0, max_val=100)
            base_prob = 0.65 * base_prob + 0.35 * val_prob

        prob_up = max(0.01, min(0.99, base_prob))

        # ── Confidence ────────────────────────────────────────────────────
        confidence = 0.5
        if scores.piotroski_score >= 7 and scores.z_score_interpretation == "Safe":
            confidence = 0.85
        elif scores.piotroski_score <= 3 and scores.z_score_interpretation == "Distress":
            confidence = 0.85
        if scores.beneish_interpretation == "Likely Manipulator":
            confidence = min(confidence, 0.40)  # cap if manipulation suspected

        # ── Reasoning ─────────────────────────────────────────────────────
        direction = "BULLISH" if prob_up > 0.55 else "BEARISH" if prob_up < 0.45 else "NEUTRAL"
        reasoning = (
            f"Fundamental outlook is {direction} ({prob_up * 100:.1f}% probability). "
            f"Piotroski F-Score {scores.piotroski_score}/9 ({scores.f_score_interpretation}). "
        )

        if scores.altman_z_score > 0:
            reasoning += (
                f"Altman Z: {scores.altman_z_score:.2f} ({scores.z_score_interpretation}). "
            )

        if scores.beneish_interpretation == "Likely Manipulator":
            reasoning += f"WARNING: Beneish M-Score={scores.beneish_mscore:.2f} (earnings manipulation risk). "

        if scores.pe_ratio > 0:
            reasoning += f"P/E: {scores.pe_ratio:.1f}x. "

        if scores.revenue_growth_yoy != 0:
            reasoning += f"Revenue growth: {scores.revenue_growth_yoy:+.1f}% YoY. "

        if scores.dcf_implied_upside != 0:
            reasoning += f"DCF implies {scores.dcf_implied_upside:+.1f}% upside. "

        return AgentResult(
            agent_name=self.name,
            probability_up=prob_up,
            confidence=confidence,
            reasoning=reasoning,
            factor_scores=factor_scores,
            warnings=warnings,
        )
