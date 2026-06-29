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
                score=100.0 if scores.gross_margin > sf.get("gross_margin_strong", 40) else (scores.gross_margin / sf.get("gross_margin_strong", 40)) * 100,
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
        # Dynamic P/E threshold: cheap_PE = 1 / (risk_free_rate + equity_premium)
        # When 10Y yield is 4.5% + 4% ERP = earnings yield floor of 8.5% → P/E ≈ 11.8
        # When 10Y yield is 2.0% + 4% ERP = earnings yield floor of 6.0% → P/E ≈ 16.7
        try:
            import yfinance as _yf_rf
            _tnx = _yf_rf.Ticker("^TNX").history(period="3d")["Close"].dropna()
            _rf_rate = float(_tnx.iloc[-1]) / 100 if len(_tnx) else 0.045
            pe_cheap     = max(10, min(22, round(1 / (_rf_rate + 0.04))))
            pe_fair      = pe_cheap + 10
            pe_expensive = pe_cheap + 25
        except Exception:
            pe_cheap     = sf.get("pe_cheap", 15)
            pe_fair      = sf.get("pe_fair", 25)
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

        # ── New: P/E vs Sector Median ─────────────────────────────────────
        try:
            import yfinance as _yf_sec
            _sector_etf_map = {
                "Technology": "XLK", "Communication Services": "XLC",
                "Consumer Discretionary": "XLY", "Consumer Staples": "XLP",
                "Health Care": "XLV", "Industrials": "XLI", "Materials": "XLB",
                "Energy": "XLE", "Financials": "XLF", "Real Estate": "XLRE", "Utilities": "XLU",
            }
            _sector_name = info.get("sector", "")
            _sec_etf = _sector_etf_map.get(_sector_name, "SPY")
            _sec_info = _yf_sec.Ticker(_sec_etf).info
            _sector_pe = _sec_info.get("trailingPE") or _sec_info.get("forwardPE")
            _stock_pe  = info.get("trailingPE") or info.get("forwardPE")
            if _sector_pe and _stock_pe and float(_sector_pe) > 0 and float(_stock_pe) > 0:
                _pe_rel = float(_stock_pe) / float(_sector_pe)
                _pe_vs_sec_score = (85.0 if _pe_rel < 0.8 else 65.0 if _pe_rel < 1.0 else 45.0 if _pe_rel < 1.3 else 20.0)
                factor_scores["pe_vs_sector"] = FactorScore(
                    name="P/E vs Sector Median",
                    value=round(_pe_rel, 3),
                    score=_pe_vs_sec_score,
                    interpretation=f"Stock P/E {float(_stock_pe):.1f}x vs {_sec_etf} median {float(_sector_pe):.1f}x → ratio {_pe_rel:.2f}x ({'cheap vs sector' if _pe_rel < 0.9 else 'expensive vs sector' if _pe_rel > 1.2 else 'fair value vs peers'})",
                )
        except Exception:
            pass

        # ── New: P/E vs 5-Year Historical Average ─────────────────────────
        try:
            import yfinance as _yf_pe5
            import numpy as _np_pe5
            import pandas as _pd_pe5
            _pe_now5 = info.get("trailingPE")
            _shares5 = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding") or 1
            if _pe_now5 and float(_pe_now5) > 0 and not inc.empty:
                _ni_row5 = [r for r in inc.index if 'net income' in str(r).lower()]
                if _ni_row5 and len(inc.columns) >= 4:
                    _ph5 = _yf_pe5.download(ticker, period="5y", interval="3mo",
                                             auto_adjust=True, progress=False)
                    if not _ph5.empty:
                        _close5 = _ph5["Close"].squeeze().dropna()
                        _ni_ser5 = inc.loc[_ni_row5[0]].dropna()
                        _pe_hist5 = []
                        for _ci5 in range(min(8, len(_ni_ser5))):
                            _col_dt5 = _ni_ser5.index[_ci5]
                            _ni_val5 = float(_ni_ser5.iloc[_ci5])
                            _eps5 = _ni_val5 / _shares5
                            if _eps5 > 0:
                                try:
                                    _idx5 = _close5.index.get_indexer(
                                        [_pd_pe5.Timestamp(_col_dt5)], method="nearest"
                                    )[0]
                                    _p5 = float(_close5.iloc[_idx5])
                                    _pe_hist5.append(_p5 / _eps5)
                                except Exception:
                                    pass
                        if len(_pe_hist5) >= 3:
                            _avg_pe5 = float(_np_pe5.median(_pe_hist5))
                            _pe_vs5 = float(_pe_now5) / _avg_pe5 if _avg_pe5 > 0 else 1.0
                            _pev5_score = (80.0 if _pe_vs5 < 0.85 else 60.0 if _pe_vs5 < 1.0 else 40.0 if _pe_vs5 < 1.3 else 15.0)
                            factor_scores["pe_vs_5yr_avg"] = FactorScore(
                                name="P/E vs 5-Year Average",
                                value=round(_pe_vs5, 3),
                                score=_pev5_score,
                                interpretation=f"Current P/E {float(_pe_now5):.1f}x vs 5Y median {_avg_pe5:.1f}x → {(_pe_vs5-1)*100:+.0f}% relative ({'historically cheap' if _pe_vs5 < 0.85 else 'historically expensive' if _pe_vs5 > 1.2 else 'near historical norm'})",
                            )
        except Exception:
            pass

        # ── New: Earnings Consistency (Coefficient of Variation of EPS) ───
        try:
            import numpy as _np_ec
            _ni_rows_ec = [r for r in inc.index if 'net income' in str(r).lower()]
            _shares_ec  = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding") or 1
            if _ni_rows_ec and len(inc.columns) >= 4 and _shares_ec:
                _ni_ec = [float(inc.loc[_ni_rows_ec[0]].iloc[i]) / _shares_ec
                          for i in range(min(8, len(inc.columns)))]
                _ni_ec = [v for v in _ni_ec if not _np_ec.isnan(v)]
                if len(_ni_ec) >= 4:
                    _std_ec  = float(_np_ec.std(_ni_ec))
                    _mean_ec = abs(float(_np_ec.mean(_ni_ec)))
                    _cov_ec  = _std_ec / _mean_ec if _mean_ec > 0 else 99.0
                    _ec_score = (85.0 if _cov_ec < 0.2 else 65.0 if _cov_ec < 0.4 else 45.0 if _cov_ec < 0.7 else 20.0)
                    factor_scores["earnings_consistency"] = FactorScore(
                        name="Earnings Consistency (CoV)",
                        value=round(_cov_ec, 3),
                        score=_ec_score,
                        interpretation=f"Quarterly EPS CoV: {_cov_ec:.2f} ({'highly consistent' if _cov_ec < 0.2 else 'volatile earnings' if _cov_ec > 0.7 else 'moderate consistency'}) — latest EPS ${_ni_ec[0]:.2f}",
                    )
        except Exception:
            pass

        # ── New: CEO/CFO Departure Signal (EDGAR 8-K Search) ─────────────
        try:
            import requests as _req_dep
            import datetime as _dt_dep
            _dep_start = (_dt_dep.date.today() - _dt_dep.timedelta(days=90)).isoformat()
            _dep_url = (
                f"https://efts.sec.gov/LATEST/search-index?"
                f"q=%22{ticker}%22+%22departure%22&forms=8-K"
                f"&dateRange=custom&startdt={_dep_start}"
            )
            _dep_resp = _req_dep.get(
                _dep_url, timeout=5,
                headers={"User-Agent": "AlphaAgent alphaagent@research.example.com"}
            )
            if _dep_resp.ok:
                _dep_data = _dep_resp.json()
                _dep_hits = _dep_data.get("hits", {}).get("total", {})
                _dep_count = _dep_hits.get("value", 0) if isinstance(_dep_hits, dict) else int(_dep_hits)
                _dep_score = (20.0 if _dep_count > 0 else 65.0)
                factor_scores["ceo_cfoo_departure"] = FactorScore(
                    name="CEO/CFO Departure (8-K Signal)",
                    value=float(_dep_count),
                    score=_dep_score,
                    interpretation=f"8-K departure filings (90d): {_dep_count} ({'leadership change detected' if _dep_count > 0 else 'no departure signal'})",
                )
                if _dep_count > 0:
                    warnings.append(f"CEO/CFO DEPARTURE: {_dep_count} 8-K filing(s) mentioning departure in last 90 days.")
        except Exception:
            pass

        # ── New: M&A Activity Signal (EDGAR 8-K EFTS) ────────────────────
        try:
            import requests as _req_ma
            import datetime as _dt_ma
            _ma_start = (_dt_ma.date.today() - _dt_ma.timedelta(days=180)).isoformat()
            _ma_total = 0
            for _ma_kw in ["merger", "acquisition"]:
                try:
                    _ma_url = (
                        f"https://efts.sec.gov/LATEST/search-index?"
                        f"q=%22{ticker}%22+%22{_ma_kw}%22&forms=8-K"
                        f"&dateRange=custom&startdt={_ma_start}"
                    )
                    _ma_resp = _req_ma.get(
                        _ma_url, timeout=5,
                        headers={"User-Agent": "AlphaAgent alphaagent@research.example.com"}
                    )
                    if _ma_resp.ok:
                        _ma_data = _ma_resp.json()
                        _ma_hits = _ma_data.get("hits", {}).get("total", {})
                        _ma_count = _ma_hits.get("value", 0) if isinstance(_ma_hits, dict) else int(_ma_hits)
                        _ma_total += _ma_count
                except Exception:
                    pass
            _ma_score = (75.0 if _ma_total > 2 else 65.0 if _ma_total > 0 else 50.0)
            factor_scores["ma_activity"] = FactorScore(
                name="M&A Activity (EDGAR 8-K)",
                value=float(_ma_total),
                score=_ma_score,
                interpretation=f"8-K filings mentioning merger/acquisition (180d): {_ma_total} — {'active deal-making' if _ma_total > 2 else 'some M&A activity' if _ma_total > 0 else 'no M&A signal'}",
            )
            if _ma_total > 0:
                warnings.append(f"M&A ACTIVITY: {_ma_total} 8-K filing(s) mentioning merger/acquisition — potential catalyst.")
        except Exception:
            pass

        # ── New: Earnings Revision Momentum (EPS Estimate Trend) ─────────
        try:
            import yfinance as _yf_erm
            _erm_trend = _yf_erm.Ticker(ticker).eps_trend
            if _erm_trend is not None and not _erm_trend.empty:
                for _hz in ["0y", "+1y", "0q", "+1q"]:
                    if _hz not in _erm_trend.index:
                        continue
                    _row = _erm_trend.loc[_hz]
                    _c   = _row.get("current");   _a30 = _row.get("30daysAgo")
                    _a60 = _row.get("60daysAgo")
                    if _c is None or _a30 is None:
                        continue
                    _c = float(_c); _a30 = float(_a30)
                    if _a30 == 0:
                        continue
                    _rev30 = (_c - _a30) / abs(_a30) * 100
                    _rev60 = ((_c - float(_a60)) / abs(float(_a60)) * 100
                               if _a60 is not None and float(_a60) != 0 else 0.0)
                    _erm_score = (88.0 if _rev30 > 5 else
                                  72.0 if _rev30 > 2 else
                                  60.0 if _rev30 > 0 else
                                  40.0 if _rev30 > -2 else
                                  25.0 if _rev30 > -5 else
                                  12.0)
                    factor_scores["earnings_revision"] = FactorScore(
                        name="Earnings Revision Momentum",
                        value=round(_rev30, 2),
                        score=_erm_score,
                        interpretation=(
                            f"EPS est ({_hz}): {_rev30:+.1f}% (30d) / {_rev60:+.1f}% (60d) — "
                            f"{'rising — analysts upgrading' if _rev30 > 0 else 'falling — analysts cutting'}"
                        ),
                    )
                    if _rev30 > 3:
                        warnings.append(f"ESTIMATE UPGRADE: EPS ({_hz}) raised {_rev30:+.1f}% in 30d — bullish revision momentum.")
                    elif _rev30 < -3:
                        warnings.append(f"ESTIMATE CUT: EPS ({_hz}) lowered {_rev30:+.1f}% in 30d — bearish revision momentum.")
                    break
        except Exception:
            pass

        # ── New: Graham Number (Intrinsic Value Floor) ────────────────────
        try:
            import math as _math_gn
            _gn_eps  = info.get("trailingEps") or info.get("epsTrailingTwelveMonths")
            _gn_bvps = info.get("bookValue")
            _gn_px   = info.get("currentPrice") or info.get("regularMarketPrice")
            if _gn_eps and _gn_bvps and _gn_px:
                _eps = float(_gn_eps); _bvps = float(_gn_bvps); _px = float(_gn_px)
                if _eps > 0 and _bvps > 0 and _px > 0:
                    _graham = _math_gn.sqrt(22.5 * _eps * _bvps)
                    _gn_upside = (_graham - _px) / _px * 100
                    _gn_score  = (90.0 if _gn_upside > 30 else
                                  75.0 if _gn_upside > 15 else
                                  60.0 if _gn_upside > 0 else
                                  42.0 if _gn_upside > -20 else
                                  20.0)
                    factor_scores["graham_number"] = FactorScore(
                        name="Graham Number (Intrinsic Floor)",
                        value=round(_graham, 2),
                        score=_gn_score,
                        interpretation=(
                            f"Graham √(22.5×EPS×BVPS) = ${_graham:.2f} vs price ${_px:.2f} → "
                            f"{_gn_upside:+.1f}% {'upside (undervalued)' if _gn_upside > 0 else 'overvalued vs intrinsic floor'}"
                        ),
                    )
        except Exception:
            pass

        # ── New: Earnings Call NLP (Gemini — Quarterly Results Tone) ─────────
        try:
            import os as _os_ec
            import yfinance as _yf_ec
            _gemini_key = _os_ec.getenv("GEMINI_API_KEY")
            if _gemini_key:
                import google.genai as _genai_ec
                _tkr_ec = _yf_ec.Ticker(ticker)
                _qfin   = _tkr_ec.quarterly_financials
                _ec_parts = []
                if _qfin is not None and not _qfin.empty:
                    _ec_parts.append(f"Quarterly financials (last 2 periods):\n{_qfin.iloc[:8, :2].to_string()}")
                _target_p = info.get("targetMeanPrice")
                _price_p  = info.get("currentPrice") or info.get("regularMarketPrice")
                if _target_p and _price_p:
                    _ec_parts.append(f"Analyst target ${float(_target_p):.2f} vs price ${float(_price_p):.2f} ({(float(_target_p)/float(_price_p)-1)*100:+.1f}% upside)")
                _rec_key = info.get("recommendationKey", "")
                if _rec_key:
                    _ec_parts.append(f"Consensus: {_rec_key}")
                for _n in (_tkr_ec.news or [])[:4]:
                    _nt = _n.get("title", "")
                    if any(kw in _nt.lower() for kw in ["earn", "revenue", "eps", "quarter", "q1", "q2", "q3", "q4", "beat", "miss", "guid"]):
                        _ec_parts.append(f"News: {_nt}")
                if _ec_parts:
                    _ec_client = _genai_ec.Client(api_key=_gemini_key)
                    _ec_prompt = (
                        f"Analyze earnings quality for {ticker}:\n\n"
                        + "\n".join(_ec_parts)
                        + "\n\nRespond ONLY in this exact format:\n"
                        "EARNINGS_QUALITY: [STRONG/ADEQUATE/WEAK]\n"
                        "REVENUE_TREND: [ACCELERATING/STABLE/DECELERATING]\n"
                        "GUIDANCE_TONE: [RAISED/MAINTAINED/LOWERED/ABSENT]\n"
                        "RED_FLAGS: [YES/NO]\n"
                        "EARNINGS_SCORE: [0-100]\n"
                    )
                    _ec_resp = _ec_client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=_ec_prompt,
                        config=_genai_ec.types.GenerateContentConfig(temperature=0.0),
                    )
                    _ec_parsed = {}
                    for _line in _ec_resp.text.split("\n"):
                        if ":" in _line:
                            _ek, _ev = _line.split(":", 1)
                            _ec_parsed[_ek.strip()] = _ev.strip()
                    _eq  = _ec_parsed.get("EARNINGS_QUALITY", "ADEQUATE")
                    _rt  = _ec_parsed.get("REVENUE_TREND", "STABLE")
                    _gt  = _ec_parsed.get("GUIDANCE_TONE", "MAINTAINED")
                    _rf  = _ec_parsed.get("RED_FLAGS", "NO")
                    _es  = min(100.0, max(0.0, float(_ec_parsed.get("EARNINGS_SCORE", "50"))))
                    _ec_score = max(10.0, min(90.0, _es))
                    factor_scores["earnings_call_nlp"] = FactorScore(
                        name="Earnings Call NLP (Gemini)",
                        value=round(_es / 100.0, 3),
                        score=_ec_score,
                        interpretation=f"Quality: {_eq} | Revenue: {_rt} | Guidance: {_gt} | Red flags: {_rf} (score: {_es:.0f}/100)",
                    )
                    if _rf == "YES":
                        warnings.append(f"Earnings NLP: red flags detected in latest quarterly results.")
                    if _gt == "RAISED" and _eq == "STRONG":
                        reasoning.append(f"Earnings NLP: strong quality + raised guidance — positive catalyst.")
                    elif _eq == "WEAK" and _rf == "YES":
                        reasoning.append(f"Earnings NLP: weak earnings quality with red flags — negative fundamental signal.")
        except Exception:
            pass

        # ── New: Net Debt / EBITDA (Leverage Ratio) ──────────────────────
        try:
            import yfinance as _yf_nd
            _tick_nd = _yf_nd.Ticker(ticker)
            _bs_nd   = _tick_nd.balance_sheet
            _is_nd   = _tick_nd.income_stmt
            if _bs_nd is not None and not _bs_nd.empty and _is_nd is not None and not _is_nd.empty:
                def _row(df, *keys):
                    for k in keys:
                        matches = [c for c in df.index if k.lower() in c.lower()]
                        if matches:
                            v = df.loc[matches[0]].iloc[0]
                            if v != 0:
                                return float(v)
                    return None
                _total_debt = _row(_bs_nd, "Total Debt", "Long Term Debt") or 0.0
                _cash       = _row(_bs_nd, "Cash And Cash Equivalents", "Cash") or 0.0
                _ebitda     = _row(_is_nd, "EBITDA", "Normalized EBITDA") or None
                if _ebitda and _ebitda != 0:
                    _net_debt    = _total_debt - _cash
                    _nd_ebitda   = _net_debt / abs(_ebitda)
                    _nd_score    = (75.0 if _nd_ebitda < 1.0 else 60.0 if _nd_ebitda < 2.5 else 40.0 if _nd_ebitda < 4.0 else 20.0)
                    factor_scores["net_debt_ebitda"] = FactorScore(
                        name="Net Debt / EBITDA",
                        value=round(_nd_ebitda, 2),
                        score=_nd_score,
                        interpretation=(
                            f"Net debt: ${_net_debt / 1e9:.2f}B | EBITDA: ${_ebitda / 1e9:.2f}B | "
                            f"Leverage: {_nd_ebitda:.2f}x — "
                            f"{'low leverage' if _nd_ebitda < 1.0 else 'moderate' if _nd_ebitda < 2.5 else 'elevated' if _nd_ebitda < 4.0 else 'high leverage risk'}"
                        ),
                    )
                    if _nd_ebitda > 4.0:
                        warnings.append(f"High leverage: Net Debt/EBITDA {_nd_ebitda:.1f}x — debt servicing risk.")
        except Exception:
            pass

        # ── New: Dividend Yield + Payout Ratio ───────────────────────────
        try:
            import yfinance as _yf_div
            _tick_dv = _yf_div.Ticker(ticker)
            _info_dv = _tick_dv.info or {}
            _div_yield   = _info_dv.get("dividendYield", None) or _info_dv.get("trailingAnnualDividendYield", None)
            _payout      = _info_dv.get("payoutRatio", None)
            if _div_yield is not None:
                _div_pct = float(_div_yield) * 100
                _dv_score = (70.0 if 2.0 <= _div_pct <= 5.0 else 55.0 if _div_pct > 5.0 else 45.0 if _div_pct > 0 else 40.0)
                _payout_str = f" | payout: {float(_payout)*100:.0f}%" if _payout else ""
                if _payout and float(_payout) > 0.90:
                    _dv_score = min(_dv_score, 35.0)
                    warnings.append(f"Payout ratio {float(_payout)*100:.0f}% — dividend sustainability risk.")
                factor_scores["dividend_yield"] = FactorScore(
                    name="Dividend Yield & Payout",
                    value=round(_div_pct, 2),
                    score=_dv_score,
                    interpretation=(
                        f"Yield: {_div_pct:.2f}%{_payout_str} — "
                        f"{'attractive income' if 2.0 <= _div_pct <= 5.0 else 'potentially unsustainable yield' if _div_pct > 5.0 else 'low yield / growth stock' if _div_pct > 0 else 'no dividend'}"
                    ),
                )
        except Exception:
            pass

        # ── NEW: Earnings Date Proximity (blackout + pre-earnings drift) ────
        try:
            import yfinance as _yf_ep
            import pandas as _pd_ep
            _cal_ep = _yf_ep.Ticker(ticker).calendar
            if _cal_ep is not None and not _cal_ep.empty:
                _earn_col = [c for c in _cal_ep.columns if "Earnings" in str(c)]
                if _earn_col:
                    _earn_dt = _pd_ep.to_datetime(_cal_ep[_earn_col[0]].iloc[0])
                    _days_earn = int((_earn_dt - _pd_ep.Timestamp.now(tz=_earn_dt.tzinfo)).days)
                    if 0 <= _days_earn <= 5:
                        factor_scores["earnings_proximity"] = FactorScore(
                            name="Earnings Proximity",
                            value=float(_days_earn),
                            score=20.0,
                            interpretation=f"Earnings in {_days_earn}d — BLACKOUT: high event risk, avoid new position",
                        )
                        warnings.append(f"EARNINGS in {_days_earn} day(s) — high binary risk. Consider skipping.")
                    elif 5 < _days_earn <= 14:
                        factor_scores["earnings_proximity"] = FactorScore(
                            name="Earnings Proximity",
                            value=float(_days_earn),
                            score=55.0,
                            interpretation=f"Earnings in {_days_earn}d — pre-earnings drift window (historically +2-3%)",
                        )
                    else:
                        factor_scores["earnings_proximity"] = FactorScore(
                            name="Earnings Proximity",
                            value=float(_days_earn),
                            score=65.0,
                            interpretation=f"Earnings in {_days_earn}d — no near-term event risk",
                        )
        except Exception:
            pass

        # ── NEW: FCF Quality Score (FCF > Net Income = real earnings) ────────
        try:
            import yfinance as _yf_fcfq
            _cfstmt = _yf_fcfq.Ticker(ticker).cashflow
            _incstmt = _yf_fcfq.Ticker(ticker).financials
            if _cfstmt is not None and not _cfstmt.empty and _incstmt is not None and not _incstmt.empty:
                _fcf_rows = [r for r in _cfstmt.index if "Free Cash" in str(r) or ("Operating" in str(r) and "Capital" not in str(r))]
                _capex_rows = [r for r in _cfstmt.index if "Capital" in str(r) and "Expenditure" in str(r)]
                _ni_rows  = [r for r in _incstmt.index if "Net Income" in str(r)]
                if _fcf_rows and _ni_rows:
                    _fcf_val = float(_cfstmt.loc[_fcf_rows[0]].iloc[0])
                    if _capex_rows:
                        _fcf_val -= abs(float(_cfstmt.loc[_capex_rows[0]].iloc[0]))
                    _ni_val  = float(_incstmt.loc[_ni_rows[0]].iloc[0])
                    _fcf_ratio = _fcf_val / max(abs(_ni_val), 1)
                    _fcf_score = (88.0 if _fcf_ratio > 1.1 else 68.0 if _fcf_ratio > 0.8
                                  else 45.0 if _fcf_ratio > 0.5 else 18.0)
                    factor_scores["fcf_quality"] = FactorScore(
                        name="FCF Quality (FCF / Net Income)",
                        value=round(_fcf_ratio, 3),
                        score=_fcf_score,
                        interpretation=(
                            f"FCF/NI = {_fcf_ratio:.2f}x — "
                            f"{'high quality: cash > accruals' if _fcf_ratio > 1.1 else 'good quality' if _fcf_ratio > 0.8 else 'watch accruals' if _fcf_ratio > 0.5 else 'accruals dominate — earnings quality risk'}"
                        ),
                    )
        except Exception:
            pass

        # ── NEW: Analyst Revision Momentum (bull% trending up = quality) ─────
        try:
            import yfinance as _yf_arm
            _recom = _yf_arm.Ticker(ticker).recommendations_summary
            if _recom is not None and not _recom.empty and len(_recom) >= 2:
                def _bull_pct(row):
                    _sb = float(row.get("strongBuy", 0) or 0)
                    _b  = float(row.get("buy", 0) or 0)
                    _tot = float(row.sum()) if hasattr(row, "sum") else 1
                    return (_sb + _b) / max(_tot, 1) * 100
                _bp_now  = _bull_pct(_recom.iloc[0])
                _bp_prev = _bull_pct(_recom.iloc[1])
                _rev_delta = _bp_now - _bp_prev
                _arm_score = (80.0 if _rev_delta > 8 else 65.0 if _rev_delta > 2
                              else 45.0 if _rev_delta > -5 else 25.0)
                factor_scores["analyst_revision_momentum"] = FactorScore(
                    name="Analyst Revision Momentum",
                    value=round(_rev_delta, 1),
                    score=_arm_score,
                    interpretation=(
                        f"Bull% now {_bp_now:.0f}% vs prev period {_bp_prev:.0f}% → "
                        f"Δ{_rev_delta:+.0f}pp | "
                        f"{'upgrades accelerating' if _rev_delta > 8 else 'mild upgrade trend' if _rev_delta > 2 else 'stable coverage' if _rev_delta > -5 else 'downgrades increasing'}"
                    ),
                )
        except Exception:
            pass

        # ── New: Asset Growth Anomaly (Cooper et al. 2008) ───────────────────
        try:
            _ag = getattr(scores, "asset_growth_yoy", None)
            if _ag is None:
                _ag = scores.data.get("asset_growth_yoy") if hasattr(scores, "data") else None
            if _ag is not None:
                _ag_score = (25.0 if _ag > 25 else   # high asset growth → underperforms
                             40.0 if _ag > 15 else
                             60.0 if _ag > 5  else
                             75.0 if _ag > -5 else 65.0)
                factor_scores["asset_growth"] = FactorScore(
                    name="Asset Growth Anomaly",
                    value=round(float(_ag), 1),
                    score=_ag_score,
                    interpretation=(
                        f"Asset growth YoY: {_ag:+.1f}% — "
                        f"{'over-investment signal (bearish)' if _ag > 25 else 'rapid expansion, monitor' if _ag > 15 else 'normal growth' if _ag > 5 else 'asset shrinkage (mixed)'}"
                    ),
                )
                if _ag > 25:
                    warnings.append(f"Rapid asset growth ({_ag:.0f}% YoY) — Cooper anomaly: over-investors underperform.")
        except Exception:
            pass

        # ── New: Gross Profitability / Novy-Marx (2013) ───────────────────────
        try:
            _gp = getattr(scores, "gross_profitability", None)
            if _gp is None:
                _gp = scores.data.get("gross_profitability") if hasattr(scores, "data") else None
            if _gp is not None:
                _gp_score = (80.0 if _gp > 40 else
                             65.0 if _gp > 25 else
                             50.0 if _gp > 15 else 35.0)
                factor_scores["gross_profitability"] = FactorScore(
                    name="Gross Profitability (Novy-Marx)",
                    value=round(float(_gp), 1),
                    score=_gp_score,
                    interpretation=(
                        f"Gross profit / Assets: {_gp:.1f}% — "
                        f"{'highly profitable (quality buy)' if _gp > 40 else 'above-avg profitability' if _gp > 25 else 'moderate' if _gp > 15 else 'low gross profitability'}"
                    ),
                )
        except Exception:
            pass

        # ── New: Investment-to-Assets / q-factor ─────────────────────────────
        try:
            _ia = getattr(scores, "investment_to_assets", None)
            if _ia is None:
                _ia = scores.data.get("investment_to_assets") if hasattr(scores, "data") else None
            if _ia is not None:
                _ia_score = (30.0 if _ia > 15 else   # over-investment → underperformance
                             50.0 if _ia > 8  else
                             70.0 if _ia > 3  else 65.0)
                factor_scores["investment_to_assets"] = FactorScore(
                    name="Investment-to-Assets (q-factor)",
                    value=round(float(_ia), 1),
                    score=_ia_score,
                    interpretation=(
                        f"CapEx/Assets: {_ia:.1f}% — "
                        f"{'over-investment (bearish per q-factor)' if _ia > 15 else 'heavy investment phase' if _ia > 8 else 'normal capex' if _ia > 3 else 'asset-light model'}"
                    ),
                )
        except Exception:
            pass

        # ── New: Net Stock Issuance ───────────────────────────────────────────
        try:
            _ni = getattr(scores, "net_issuance_pct", None)
            if _ni is None:
                _ni = scores.data.get("net_issuance_pct") if hasattr(scores, "data") else None
            if _ni is not None:
                _ni_score = (25.0 if _ni > 5   else   # dilution → bearish
                             40.0 if _ni > 1   else
                             60.0 if _ni > -1  else
                             80.0)                     # buyback → bullish
                factor_scores["net_issuance"] = FactorScore(
                    name="Net Stock Issuance",
                    value=round(float(_ni), 2),
                    score=_ni_score,
                    interpretation=(
                        f"Shares change: {_ni:+.2f}% — "
                        f"{'significant dilution (bearish)' if _ni > 5 else 'mild dilution' if _ni > 1 else 'flat' if _ni > -1 else 'buyback program (bullish)'}"
                    ),
                )
                if _ni > 5:
                    warnings.append(f"Share issuance {_ni:.1f}% — dilution signal, firms issue at tops.")
        except Exception:
            pass

        # ── New: R&D Anomaly (Chan et al.) ────────────────────────────────────
        # High R&D / Market Cap → outperformance (capitalized intangibles)
        try:
            info_rd = data.get_info() or {}
            _mcap = float(info_rd.get("marketCap", 0) or 0)
            _rd = 0.0
            try:
                _is = financials.get("income")
                if _is is not None and not _is.empty:
                    for k in ["Research And Development", "Research Development", "Research & Development"]:
                        if k in _is.index:
                            _rd = abs(float(_is.loc[k].iloc[0] or 0))
                            break
            except Exception:
                pass
            if _mcap > 0 and _rd > 0:
                _rd_intensity = _rd / _mcap * 100
                _rd_score = (78.0 if _rd_intensity > 8 else
                             65.0 if _rd_intensity > 4 else
                             55.0 if _rd_intensity > 1 else 50.0)
                factor_scores["rd_anomaly"] = FactorScore(
                    name="R&D Anomaly (Chan)",
                    value=round(_rd_intensity, 2),
                    score=_rd_score,
                    interpretation=(
                        f"R&D/MCap: {_rd_intensity:.2f}% — "
                        f"{'high R&D intensity (Chan anomaly: bullish)' if _rd_intensity > 8 else 'meaningful R&D' if _rd_intensity > 4 else 'modest R&D' if _rd_intensity > 1 else 'low R&D'}"
                    ),
                )
        except Exception:
            pass

        # ── New: QMJ Composite — Quality Minus Junk (AQR / Asness) ────────────
        # Composite of profitability + growth + safety - matches the AQR formulation
        try:
            _profitability = (scores.roe / 25.0 if hasattr(scores, "roe") and scores.roe > 0 else 0) + \
                             (scores.gross_margin / 50.0 if hasattr(scores, "gross_margin") and scores.gross_margin > 0 else 0)
            _safety = (1.0 - min(scores.debt_to_equity / 3.0, 1.0)) if hasattr(scores, "debt_to_equity") else 0.5
            _growth = (scores.revenue_growth_yoy / 30.0 if hasattr(scores, "revenue_growth_yoy") else 0)
            _qmj_raw = (_profitability + _safety + _growth) / 3.0   # 0 to ~2
            _qmj_norm = max(0.0, min(1.0, _qmj_raw / 1.5))
            _qmj_score = round(_qmj_norm * 100, 1)
            factor_scores["qmj_composite"] = FactorScore(
                name="QMJ Composite (AQR)",
                value=round(_qmj_raw, 3),
                score=_qmj_score,
                interpretation=(
                    f"QMJ score: {_qmj_norm:.2f} (Prof: {_profitability:.2f} | Safety: {_safety:.2f} | Growth: {_growth:.2f}) — "
                    f"{'high quality (Q): bullish' if _qmj_norm > 0.65 else 'mid-quality' if _qmj_norm > 0.40 else 'junk (J): bearish'}"
                ),
            )
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
            _qv_blend = sf.get("quality_value_blend", 0.65)
            base_prob = _qv_blend * base_prob + (1.0 - _qv_blend) * val_prob

        # ── PEAD overlay ──────────────────────────────────────────────────
        pead_tag = ""
        try:
            from quant_engine.pead import compute_pead
            pead = compute_pead(ticker)
            if pead and pead.decay_factor > 0 and pead.direction != "NEUTRAL":
                base_prob = max(0.01, min(0.99, base_prob + pead.prob_adjustment))
                factor_scores["pead_drift"] = FactorScore(
                    name="PEAD Drift (Post-Earnings Anomaly)",
                    value=round(pead.sue, 3),
                    score=pead.score,
                    interpretation=(
                        f"{pead.direction} | {pead.days_since_earnings}d since earnings | "
                        f"SUE={pead.sue:+.2f} | decay={pead.decay_factor:.0%}"
                    ),
                )
                pead_tag = (
                    f"PEAD: {pead.direction} ({pead.days_since_earnings}d ago, "
                    f"SUE={pead.sue:+.2f}, decay={pead.decay_factor:.0%}). "
                )
        except Exception:
            pass

        # ── SEC 10-K language-shift NLP (P3a) ─────────────────────────────
        sec_nlp_tag = ""
        try:
            from quant_engine.sec_nlp import compute_10k_shift
            sec_res = compute_10k_shift(ticker)
            if sec_res is not None:
                base_prob = max(0.01, min(0.99, base_prob + sec_res.prob_adjustment))
                factor_scores["sec_10k_nlp"] = FactorScore(
                    name="SEC 10-K Language Shift",
                    value=round(sec_res.cosine_similarity, 3),
                    score=sec_res.score,
                    interpretation=f"Regime: {sec_res.regime} | cosine_sim={sec_res.cosine_similarity:.3f}",
                )
                sec_nlp_tag = f"SEC 10-K: {sec_res.regime}. "
        except Exception:
            pass

        # ── Sector-conditional threshold adjustment (P3c) ─────────────────
        try:
            sector = info.get("sector", "")
            from config.settings_manager import settings as _cfg
            sector_cfg = _cfg.get("sector_thresholds", {})
            norm_sector = sector.replace(" ", "_").replace("/", "_") if sector else ""
            for key, val in sector_cfg.items():
                if key.lower() in norm_sector.lower() or norm_sector.lower() in key.lower():
                    # Adjust pe_fair threshold based on sector
                    sf["pe_fair"] = val.get("pe_fair", sf.get("pe_fair", 25))
                    sf["pe_expensive"] = val.get("pe_expensive", sf.get("pe_expensive", 40))
                    sf["pb_fair"] = val.get("pb_fair", sf.get("pb_fair", 3.0))
                    break
        except Exception:
            pass

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
        reasoning += pead_tag + sec_nlp_tag

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
