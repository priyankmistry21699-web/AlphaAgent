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
        pe_cheap = sf.get("pe_cheap", 15)
        pe_fair = sf.get("pe_fair", 25)
        pe_expensive = sf.get("pe_expensive", 40)
        if scores.pe_ratio > 0:
            pe_score = (100.0 if scores.pe_ratio < pe_cheap
                        else 60.0 if scores.pe_ratio < pe_fair
                        else 30.0 if scores.pe_ratio < pe_expensive
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
            roa_strong = sf.get("roa_strong", 15)
            roa_moderate = sf.get("roa_moderate", 8)
            roa_weak = sf.get("roa_weak", 3)
            roa = getattr(scores, 'return_on_assets', None)
            if roa is None:
                roa = info.get('returnOnAssets')
                if roa is not None:
                    roa = float(roa) * 100
            if roa is not None and roa != 0:
                roa_score = (90.0 if roa > roa_strong else 65.0 if roa > roa_moderate else 45.0 if roa > roa_weak else 20.0)
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
            om_strong = sf.get("operating_margin_strong", 25)
            om_moderate = sf.get("operating_margin_moderate", 15)
            om_weak = sf.get("operating_margin_weak", 5)
            op_margin = info.get('operatingMargins')
            if op_margin is not None:
                op_margin = float(op_margin) * 100
                op_score = (90.0 if op_margin > om_strong else 65.0 if op_margin > om_moderate else 45.0 if op_margin > om_weak else 20.0 if op_margin > 0 else 5.0)
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
            cr_liquid = sf.get("current_ratio_liquid", 2.0)
            cr_adequate = sf.get("current_ratio_adequate", 1.5)
            cr_tight = sf.get("current_ratio_tight", 1.0)
            current_ratio = info.get('currentRatio')
            if current_ratio is not None:
                cr = float(current_ratio)
                cr_score = (85.0 if cr > cr_liquid else 60.0 if cr > cr_adequate else 40.0 if cr > cr_tight else 15.0)
                factor_scores["current_ratio"] = FactorScore(
                    name="Current Ratio",
                    value=round(cr, 2),
                    score=cr_score,
                    interpretation=f"Current Ratio: {cr:.2f}x ({'liquid' if cr > cr_adequate else 'adequate' if cr > cr_tight else 'tight liquidity'})",
                )
        except Exception:
            pass

        # ── New: Interest Coverage Ratio ──────────────────────────────────
        try:
            icr_strong = sf.get("interest_coverage_strong", 10)
            icr_adequate = sf.get("interest_coverage_adequate", 5)
            icr_weak = sf.get("interest_coverage_weak", 2)
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
                icr_score = (90.0 if icr > icr_strong else 70.0 if icr > icr_adequate else 45.0 if icr > icr_weak else 15.0)
                factor_scores["interest_coverage"] = FactorScore(
                    name="Interest Coverage Ratio",
                    value=round(icr, 2),
                    score=icr_score,
                    interpretation=f"ICR: {icr:.1f}x ({'strong' if icr > icr_strong else 'adequate' if icr > icr_adequate else 'weak'})",
                )
        except Exception:
            pass

        # ── New: Asset Turnover ───────────────────────────────────────────
        try:
            at_good = sf.get("asset_turnover_good", 1.0)
            at_weak = sf.get("asset_turnover_weak", 0.5)
            if inc is not None and not inc.empty and bal is not None and not bal.empty:
                rev_row = [r for r in inc.index if 'total revenue' in str(r).lower() or 'revenue' in str(r).lower()]
                ast_row = [r for r in bal.index if 'total assets' in str(r).lower()]
                if rev_row and ast_row:
                    rev = abs(float(inc.loc[rev_row[0]].iloc[0]))
                    ast = abs(float(bal.loc[ast_row[0]].iloc[0]))
                    if ast > 0:
                        at = rev / ast
                        at_score = (85.0 if at > at_good else 60.0 if at > at_weak else 40.0)
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
            eps_beat = sf.get("eps_surprise_beat", 10)
            eps_miss = sf.get("eps_surprise_miss", -2)
            eps_surprise_pct = info.get('earningsSurprise') or info.get('earningsQuarterlyGrowth')
            if eps_surprise_pct is not None:
                eps_surprise_pct = float(eps_surprise_pct) * 100
                eps_score = (85.0 if eps_surprise_pct > eps_beat else 60.0 if eps_surprise_pct > 2 else 50.0 if eps_surprise_pct > eps_miss else 25.0)
                factor_scores["eps_surprise"] = FactorScore(
                    name="EPS Surprise / Growth",
                    value=round(eps_surprise_pct, 2),
                    score=eps_score,
                    interpretation=f"EPS trend: {eps_surprise_pct:+.1f}%",
                )
        except Exception:
            pass

        # ── New: Accruals Ratio (Earnings Quality) ────────────────────────
        try:
            ni_row  = [r for r in inc.index if 'net income' in str(r).lower()]
            ocf_row = [r for r in cf.index  if 'operating' in str(r).lower() and 'cash' in str(r).lower()]
            tas_row = [r for r in bal.index  if 'total assets' in str(r).lower()]
            if ni_row and ocf_row and tas_row:
                ni  = float(inc.loc[ni_row[0]].iloc[0])
                ocf = float(cf.loc[ocf_row[0]].iloc[0])
                tas = float(bal.loc[tas_row[0]].iloc[0])
                if tas != 0:
                    accruals_ratio = (ni - ocf) / abs(tas)
                    ar_score = (80.0 if accruals_ratio < -0.05
                                else 60.0 if accruals_ratio < 0.05
                                else 35.0 if accruals_ratio < 0.15
                                else 15.0)
                    factor_scores["accruals_ratio"] = FactorScore(
                        name="Accruals Ratio",
                        value=round(accruals_ratio, 4),
                        score=ar_score,
                        interpretation=f"Accruals: {accruals_ratio:+.3f} ({'high-quality earnings (cash-backed)' if accruals_ratio < 0 else 'accrual-heavy' if accruals_ratio > 0.1 else 'normal'})",
                    )
        except Exception:
            pass

        # ── New: Forward P/E ──────────────────────────────────────────────
        try:
            fwd_pe = info.get("forwardPE")
            if fwd_pe and float(fwd_pe) > 0:
                fpe = float(fwd_pe)
                pe_cheap = sf.get("pe_cheap", 15)
                pe_fair  = sf.get("pe_fair", 25)
                pe_exp   = sf.get("pe_expensive", 40)
                fpe_score = (90.0 if fpe < pe_cheap else 65.0 if fpe < pe_fair else 35.0 if fpe < pe_exp else 10.0)
                factor_scores["forward_pe"] = FactorScore(
                    name="Forward P/E",
                    value=round(fpe, 2),
                    score=fpe_score,
                    interpretation=f"Forward P/E: {fpe:.1f}x ({'cheap' if fpe < pe_cheap else 'fair' if fpe < pe_fair else 'expensive'})",
                )
        except Exception:
            pass

        # ── New: Net Margin Trend ─────────────────────────────────────────
        try:
            if inc is not None and not inc.empty and len(inc.columns) >= 2:
                rev_row = [r for r in inc.index if 'total revenue' in str(r).lower() or ('revenue' in str(r).lower() and 'cost' not in str(r).lower())]
                ni_row  = [r for r in inc.index if 'net income' in str(r).lower()]
                if rev_row and ni_row:
                    rev_now  = float(inc.loc[rev_row[0]].iloc[0])
                    rev_prev = float(inc.loc[rev_row[0]].iloc[1])
                    ni_now   = float(inc.loc[ni_row[0]].iloc[0])
                    ni_prev  = float(inc.loc[ni_row[0]].iloc[1])
                    if rev_now != 0 and rev_prev != 0:
                        nm_now  = (ni_now  / rev_now)  * 100
                        nm_prev = (ni_prev / rev_prev) * 100
                        nm_delta = nm_now - nm_prev
                        nm_score = (80.0 if nm_now > 15 and nm_delta > 0
                                    else 65.0 if nm_now > 8 and nm_delta >= 0
                                    else 50.0 if nm_now > 0 and nm_delta >= 0
                                    else 30.0 if nm_now > 0
                                    else 10.0)
                        factor_scores["net_margin_trend"] = FactorScore(
                            name="Net Margin Trend",
                            value=round(nm_delta, 2),
                            score=nm_score,
                            interpretation=f"Net margin: {nm_now:.1f}% (vs {nm_prev:.1f}% prev) Δ{nm_delta:+.1f}pp — {'expanding' if nm_delta > 0.5 else 'contracting' if nm_delta < -0.5 else 'stable'}",
                        )
        except Exception:
            pass

        # ── New: Dividend Cut Probability ─────────────────────────────────
        try:
            div_yield = info.get("dividendYield") or 0
            payout_ratio = info.get("payoutRatio") or 0
            if div_yield and div_yield > 0:
                div_yield_pct  = float(div_yield) * 100
                payout_pct     = float(payout_ratio) * 100
                # High payout + negative FCF = cut risk
                cut_risk = payout_pct > 90 or (payout_pct > 70 and scores.fcf_yield < 0)
                dcp_score = (25.0 if cut_risk else 70.0 if payout_pct < 50 else 50.0)
                factor_scores["dividend_cut_prob"] = FactorScore(
                    name="Dividend Cut Probability",
                    value=round(payout_pct, 1),
                    score=dcp_score,
                    interpretation=f"Yield: {div_yield_pct:.1f}% | Payout: {payout_pct:.0f}% — {'CUT RISK' if cut_risk else 'sustainable' if payout_pct < 50 else 'elevated payout'}",
                )
        except Exception:
            pass

        # ── New: Shares Buyback Signal (Float Reduction) ──────────────────
        try:
            shares_now  = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
            shares_old  = info.get("floatShares")
            if not (shares_now and shares_old):
                # Use quarterly income shares trend if available
                import yfinance as yf
                yf_tkr = yf.Ticker(ticker)
                bs_q = yf_tkr.quarterly_balance_sheet
                if bs_q is not None and not bs_q.empty:
                    share_rows = [r for r in bs_q.index if 'share' in str(r).lower() and 'issued' not in str(r).lower()]
                    if share_rows and len(bs_q.columns) >= 2:
                        s_now = float(bs_q.loc[share_rows[0]].iloc[0])
                        s_old = float(bs_q.loc[share_rows[0]].iloc[-1])
                        if s_old > 0:
                            share_chg_pct = (s_now - s_old) / s_old * 100
                            bb_score = (85.0 if share_chg_pct < -3 else 65.0 if share_chg_pct < 0 else 45.0 if share_chg_pct < 2 else 20.0)
                            factor_scores["buyback_signal"] = FactorScore(
                                name="Shares Outstanding Trend",
                                value=round(share_chg_pct, 2),
                                score=bb_score,
                                interpretation=f"Shares chg: {share_chg_pct:+.1f}% ({'buyback underway' if share_chg_pct < -1 else 'dilution risk' if share_chg_pct > 2 else 'flat'})",
                            )
        except Exception:
            pass

        # ── New: Fama-French 5-Factor Exposures ──────────────────────────
        try:
            import numpy as _np

            ff_info = info  # already fetched
            pe    = ff_info.get("trailingPE") or ff_info.get("forwardPE")
            pb    = ff_info.get("priceToBook")
            ev_eb = ff_info.get("enterpriseToEbitda")
            roe   = ff_info.get("returnOnEquity")
            margin = ff_info.get("profitMargins")
            debt_eq = ff_info.get("debtToEquity")
            mkt_cap = ff_info.get("marketCap") or 0
            beta_val = ff_info.get("beta")

            # ── Value factor (HML) ─────────────────────────
            val_scores = []
            if pe and pe > 0:
                val_scores.append(1.0 if pe < 12 else 0.5 if pe < 18 else 0.0 if pe < 25 else -0.5 if pe < 35 else -1.0)
            if pb and pb > 0:
                val_scores.append(1.0 if pb < 1.5 else 0.3 if pb < 3 else -0.3 if pb < 5 else -1.0)
            if ev_eb and ev_eb > 0:
                val_scores.append(1.0 if ev_eb < 8 else 0.3 if ev_eb < 15 else -0.3 if ev_eb < 25 else -1.0)
            if val_scores:
                val = float(_np.mean(val_scores))
                val_score = (val + 1) / 2 * 100
                factor_scores["ff_value"] = FactorScore(
                    name="Fama-French: Value (HML)",
                    value=round(val, 3),
                    score=round(val_score, 1),
                    interpretation=f"Value loading: {val:+.2f} ({'deep value' if val > 0.5 else 'growth' if val < -0.5 else 'blend'})",
                )

            # ── Size factor (SMB) ─────────────────────────
            if mkt_cap > 0:
                log_cap = _np.log10(mkt_cap)
                size_loading = float(_np.clip(-(log_cap - 10.0) / 1.5, -1.0, 1.0))
                size_score = (size_loading + 1) / 2 * 100
                cap_label = ("mega-cap" if log_cap > 11.5 else "large-cap" if log_cap > 10.5
                             else "mid-cap" if log_cap > 9.5 else "small-cap")
                factor_scores["ff_size"] = FactorScore(
                    name="Fama-French: Size (SMB)",
                    value=round(size_loading, 3),
                    score=round(size_score, 1),
                    interpretation=f"Size loading: {size_loading:+.2f} ({cap_label}, mktcap ${mkt_cap/1e9:.1f}B)",
                )

            # ── Quality / Profitability factor (RMW) ─────
            qual_scores = []
            if roe is not None:
                qual_scores.append(1.0 if roe > 0.25 else 0.5 if roe > 0.15 else 0.0 if roe > 0.05 else -0.5 if roe > 0 else -1.0)
            if margin is not None:
                qual_scores.append(1.0 if margin > 0.25 else 0.5 if margin > 0.15 else 0.0 if margin > 0.05 else -0.5)
            if debt_eq is not None:
                qual_scores.append(1.0 if debt_eq < 20 else 0.5 if debt_eq < 50 else -0.3 if debt_eq < 100 else -1.0)
            if qual_scores:
                qual = float(_np.mean(qual_scores))
                qual_score = (qual + 1) / 2 * 100
                factor_scores["ff_quality"] = FactorScore(
                    name="Fama-French: Quality (RMW)",
                    value=round(qual, 3),
                    score=round(qual_score, 1),
                    interpretation=f"Quality loading: {qual:+.2f} ({'high quality' if qual > 0.4 else 'low quality' if qual < -0.4 else 'average'})",
                )

            # ── Low-Volatility / Beta factor ─────────────
            if beta_val and isinstance(beta_val, (int, float)) and 0 < float(beta_val) < 5:
                beta_f = float(beta_val)
                lowvol_loading = float(_np.clip(-(beta_f - 1.0) / 1.5, -1.0, 1.0))
                lv_score = (lowvol_loading + 1) / 2 * 100
                factor_scores["ff_low_vol"] = FactorScore(
                    name="Fama-French: Low-Vol (BAB)",
                    value=round(lowvol_loading, 3),
                    score=round(lv_score, 1),
                    interpretation=f"Beta: {beta_f:.2f} | Low-vol loading: {lowvol_loading:+.2f} ({'defensive' if beta_f < 0.8 else 'aggressive' if beta_f > 1.3 else 'market-neutral'})",
                )
        except Exception:
            pass

        # ── New: CAPM Jensen's Alpha (1-Year Excess Return) ──────────────
        try:
            import yfinance as _yf_capm
            import numpy as _np_capm
            _beta = info.get("beta")
            if _beta and 0 < float(_beta) < 5:
                _beta_f = float(_beta)
                from data.macro import MacroData as _MD
                _mdata = _MD()
                _ffr_s = _mdata.get_series("FEDFUNDS", years_back=1)
                _rf = float(_ffr_s.iloc[-1].iloc[0]) / 100 if _ffr_s is not None and len(_ffr_s) > 0 else 0.045
                _spy_h = _yf_capm.download("SPY", period="1y", interval="1d", auto_adjust=True, progress=False)
                _spy_c = _spy_h["Close"].squeeze().dropna()
                _rm = float(_spy_c.iloc[-1] / _spy_c.iloc[0] - 1) if len(_spy_c) >= 200 else 0.15
                _stk_h = _yf_capm.download(ticker, period="1y", interval="1d", auto_adjust=True, progress=False)
                _stk_c = _stk_h["Close"].squeeze().dropna()
                _ri = float(_stk_c.iloc[-1] / _stk_c.iloc[0] - 1) if len(_stk_c) >= 200 else 0.0
                _capm_exp = _rf + _beta_f * (_rm - _rf)
                _alpha = _ri - _capm_exp
                _alpha_pct = _alpha * 100
                _alpha_score = max(10.0, min(90.0, 50.0 + _alpha_pct * 1.5))
                factor_scores["capm_alpha"] = FactorScore(
                    name="CAPM Jensen's Alpha (1Y)",
                    value=round(_alpha_pct, 2),
                    score=_alpha_score,
                    interpretation=f"α={_alpha_pct:+.1f}% | CAPM expected: {_capm_exp*100:.1f}% | Actual: {_ri*100:.1f}% | β={_beta_f:.2f} — {'outperformer' if _alpha > 0.05 else 'underperformer' if _alpha < -0.05 else 'market-neutral'}",
                )
        except Exception:
            pass

        # ── New: Dividend Growth Rate (5-Year CAGR) ───────────────────────
        try:
            import yfinance as _yf_div
            _divs = _yf_div.Ticker(ticker).dividends
            if _divs is not None and len(_divs) >= 4:
                _divs_ann = _divs.resample("YE").sum()
                _divs_ann = _divs_ann[_divs_ann > 0]
                if len(_divs_ann) >= 2:
                    _n_years = min(5, len(_divs_ann) - 1)
                    _div_now  = float(_divs_ann.iloc[-1])
                    _div_then = float(_divs_ann.iloc[-_n_years - 1])
                    if _div_then > 0:
                        _div_cagr = ((_div_now / _div_then) ** (1 / _n_years) - 1) * 100
                        _dgr_score = (85.0 if _div_cagr > 10 else 65.0 if _div_cagr > 5 else 50.0 if _div_cagr > 0 else 25.0)
                        factor_scores["dividend_growth_rate"] = FactorScore(
                            name=f"Dividend Growth Rate ({_n_years}Y CAGR)",
                            value=round(_div_cagr, 2),
                            score=_dgr_score,
                            interpretation=f"Dividend CAGR: {_div_cagr:+.1f}%/yr over {_n_years}y ({'dividend compounder' if _div_cagr > 8 else 'slow grower' if _div_cagr > 0 else 'dividend cut trend'})",
                        )
        except Exception:
            pass

        # ── New: Lockup Expiration Proxy (IPO Supply Overhang) ────────────
        try:
            import datetime as _dt_lk
            _ftd_ms = info.get("firstTradeDateMilliseconds") or info.get("firstTradeDateEpochUtc")
            if _ftd_ms:
                _ftd_date = _dt_lk.datetime.fromtimestamp(float(_ftd_ms) / 1000).date()
                _days_ipo = ((_dt_lk.date.today()) - _ftd_date).days
                if 0 <= _days_ipo <= 180:
                    _lk_score = 25.0
                    _lk_label = "inside 180-day lockup window — insider sell risk"
                elif 180 < _days_ipo <= 365:
                    _lk_score = 45.0
                    _lk_label = "post-lockup: insider selling may linger"
                else:
                    _lk_score = 65.0
                    _lk_label = f"IPO'd {_days_ipo // 365}yr ago — lockup long expired"
                factor_scores["lockup_expiry"] = FactorScore(
                    name="Lockup Expiration Proxy",
                    value=float(_days_ipo),
                    score=_lk_score,
                    interpretation=f"Days since IPO: {_days_ipo} | {_lk_label}",
                )
                if 150 <= _days_ipo <= 200:
                    warnings.append(f"LOCKUP EXPIRY: IPO was {_days_ipo}d ago — potential insider sell pressure at 180d lockup window.")
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
        direction = "BULLISH" if prob_up > self.long_threshold else "BEARISH" if prob_up < self.short_threshold else "NEUTRAL"
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
