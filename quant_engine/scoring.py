"""
AlphaAgent — Fundamental Scoring Engine

Computes robust financial health metrics from raw accounting data.
These scores cut through accounting tricks to reveal the true
health of a company.

Key Models:
  1. Piotroski F-Score: 9-point system measuring profitability,
     leverage/liquidity, and operating efficiency. (Higher is better)
  2. Altman Z-Score: Predicts the probability of bankruptcy within
     2 years. (Higher is safer)
  3. Beneish M-Score: 8-variable earnings manipulation detector.
     (M-Score > -1.78 → likely manipulator)
  4. Valuation Scores: P/E, P/B, PEG, EV/EBITDA, P/S, FCF yield, DCF.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FundamentalScores:
    """Output container for fundamental scoring models."""

    # Piotroski F-Score (0-9)
    piotroski_score: int = 0
    f_score_interpretation: str = "Poor"  # Poor (0-3), Fair (4-6), Strong (7-9)
    f_score_details: dict = None

    # Altman Z-Score (Continuous)
    altman_z_score: float = 0.0
    z_score_interpretation: str = "Distress"  # Distress (<1.81), Grey (1.81-2.99), Safe (>2.99)

    # Beneish M-Score
    beneish_mscore: float = -3.0             # < -1.78 = unlikely manipulator
    beneish_interpretation: str = "Clean"    # Clean / Manipulator

    # Valuation Scores
    pe_ratio: float = 0.0
    pb_ratio: float = 0.0
    peg_ratio: float = 0.0
    ev_ebitda: float = 0.0
    ps_ratio: float = 0.0
    fcf_yield: float = 0.0                   # % free cash flow yield
    dcf_implied_upside: float = 0.0          # % upside vs current price

    # Revenue & Earnings Growth
    revenue_growth_yoy: float = 0.0          # %
    earnings_growth_yoy: float = 0.0         # %
    roe: float = 0.0                         # Return on Equity %

    # Key Financial Metrics
    current_ratio: float = 0.0
    debt_to_equity: float = 0.0
    return_on_assets: float = 0.0
    gross_margin: float = 0.0

    data_quality: str = "Good"               # If data is missing, we note it here

    def __post_init__(self):
        if self.f_score_details is None:
            self.f_score_details = {}


def compute_fundamental_scores(
    income_stmt: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    cash_flow: pd.DataFrame,
    info: dict
) -> FundamentalScores:
    """
    Compute F-Score and Z-Score from financial statements.
    
    Args:
        income_stmt: Quarterly income statement from yfinance
        balance_sheet: Quarterly balance sheet from yfinance
        cash_flow: Quarterly cash flow from yfinance
        info: Company info dict from yfinance (for market cap)
        
    Returns:
        FundamentalScores object
    """
    result = FundamentalScores()
    
    # ── Data Validation ───────────────────────────────────
    if income_stmt is None or balance_sheet is None or cash_flow is None:
        result.data_quality = "Missing Statements"
        return result
        
    if income_stmt.empty or balance_sheet.empty or cash_flow.empty:
        result.data_quality = "Empty Statements"
        return result

    # We need at least 2 periods (current quarter and previous quarter/year)
    # yfinance quarterly data columns are dates.
    if len(income_stmt.columns) < 2 or len(balance_sheet.columns) < 2 or len(cash_flow.columns) < 2:
        result.data_quality = "Insufficient History (need 2+ quarters)"
        return result

    try:
        # Extract current (0) and previous (1) periods
        col_curr = income_stmt.columns[0]
        col_prev = income_stmt.columns[1]
        
        bs_curr = balance_sheet.columns[0]
        bs_prev = balance_sheet.columns[1]
        
        cf_curr = cash_flow.columns[0]
        
        # ── 1. PIOTROSki F-SCORE ──────────────────────────
        # Profitability
        # 1. ROA: Net Income / Total Assets > 0
        net_income = _safe_get(income_stmt, "Net Income", col_curr)
        total_assets = _safe_get(balance_sheet, "Total Assets", bs_curr)
        roa = net_income / total_assets if total_assets > 0 else 0
        f1 = 1 if roa > 0 else 0
        
        # 2. CFO: Operating Cash Flow > 0
        cfo = _safe_get(cash_flow, "Operating Cash Flow", cf_curr)
        f2 = 1 if cfo > 0 else 0
        
        # 3. Change in ROA: ROA(current) > ROA(previous)
        net_income_prev = _safe_get(income_stmt, "Net Income", col_prev)
        total_assets_prev = _safe_get(balance_sheet, "Total Assets", bs_prev)
        roa_prev = net_income_prev / total_assets_prev if total_assets_prev > 0 else 0
        f3 = 1 if roa > roa_prev else 0
        
        # 4. Accruals: CFO > Net Income
        f4 = 1 if cfo > net_income else 0
        
        # Leverage, Liquidity, and Source of Funds
        # 5. Change in Leverage: Long Term Debt ratio decreased
        ltd_curr = _safe_get(balance_sheet, "Long Term Debt", bs_curr)
        ltd_prev = _safe_get(balance_sheet, "Long Term Debt", bs_prev)
        leverage_curr = ltd_curr / total_assets if total_assets > 0 else 0
        leverage_prev = ltd_prev / total_assets_prev if total_assets_prev > 0 else 0
        f5 = 1 if leverage_curr < leverage_prev else 0
        
        # 6. Change in Current Ratio: Increased
        curr_assets_curr = _safe_get(balance_sheet, "Current Assets", bs_curr)
        curr_liab_curr = _safe_get(balance_sheet, "Current Liabilities", bs_curr)
        curr_ratio_curr = curr_assets_curr / curr_liab_curr if curr_liab_curr > 0 else 0
        
        curr_assets_prev = _safe_get(balance_sheet, "Current Assets", bs_prev)
        curr_liab_prev = _safe_get(balance_sheet, "Current Liabilities", bs_prev)
        curr_ratio_prev = curr_assets_prev / curr_liab_prev if curr_liab_prev > 0 else 0
        
        f6 = 1 if curr_ratio_curr > curr_ratio_prev else 0
        
        # 7. Change in Shares: No new shares issued
        shares_curr = _safe_get(balance_sheet, "Ordinary Shares Number", bs_curr)
        shares_prev = _safe_get(balance_sheet, "Ordinary Shares Number", bs_prev)
        f7 = 1 if shares_curr <= shares_prev else 0
        
        # Operating Efficiency
        # 8. Change in Gross Margin: Increased
        gross_profit_curr = _safe_get(income_stmt, "Gross Profit", col_curr)
        revenue_curr = _safe_get(income_stmt, "Total Revenue", col_curr)
        margin_curr = gross_profit_curr / revenue_curr if revenue_curr > 0 else 0
        
        gross_profit_prev = _safe_get(income_stmt, "Gross Profit", col_prev)
        revenue_prev = _safe_get(income_stmt, "Total Revenue", col_prev)
        margin_prev = gross_profit_prev / revenue_prev if revenue_prev > 0 else 0
        
        f8 = 1 if margin_curr > margin_prev else 0
        
        # 9. Change in Asset Turnover: Increased
        turnover_curr = revenue_curr / total_assets if total_assets > 0 else 0
        turnover_prev = revenue_prev / total_assets_prev if total_assets_prev > 0 else 0
        f9 = 1 if turnover_curr > turnover_prev else 0
        
        total_f_score = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
        result.piotroski_score = total_f_score
        
        if total_f_score <= 3:
            result.f_score_interpretation = "Poor"
        elif total_f_score <= 6:
            result.f_score_interpretation = "Fair"
        else:
            result.f_score_interpretation = "Strong"
            
        result.f_score_details = {
            "ROA > 0": bool(f1),
            "CFO > 0": bool(f2),
            "ROA increased": bool(f3),
            "CFO > Net Income": bool(f4),
            "Leverage decreased": bool(f5),
            "Current Ratio increased": bool(f6),
            "No new shares": bool(f7),
            "Gross Margin increased": bool(f8),
            "Asset Turnover increased": bool(f9)
        }
        
        # ── 2. ALTMAN Z-SCORE ─────────────────────────────
        # Z-Score = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E
        # A = Working Capital / Total Assets
        working_capital = curr_assets_curr - curr_liab_curr
        A = working_capital / total_assets if total_assets > 0 else 0
        
        # B = Retained Earnings / Total Assets
        retained_earnings = _safe_get(balance_sheet, "Retained Earnings", bs_curr)
        B = retained_earnings / total_assets if total_assets > 0 else 0
        
        # C = EBIT / Total Assets
        ebit = _safe_get(income_stmt, "EBIT", col_curr)
        if ebit == 0: # Try alternative name
            ebit = _safe_get(income_stmt, "Operating Income", col_curr)
        C = ebit / total_assets if total_assets > 0 else 0
        
        # D = Market Value of Equity / Total Liabilities
        market_cap = info.get("marketCap", 0)
        total_liab = _safe_get(balance_sheet, "Total Liabilities Net Minority Interest", bs_curr)
        if total_liab == 0:
            total_liab = _safe_get(balance_sheet, "Total Liabilities", bs_curr)
            
        D = market_cap / total_liab if total_liab > 0 else 0
        
        # E = Sales / Total Assets
        E = revenue_curr / total_assets if total_assets > 0 else 0
        
        z_score = (1.2 * A) + (1.4 * B) + (3.3 * C) + (0.6 * D) + (1.0 * E)
        result.altman_z_score = float(z_score)
        
        if z_score < 1.81:
            result.z_score_interpretation = "Distress"
        elif z_score < 2.99:
            result.z_score_interpretation = "Grey"
        else:
            result.z_score_interpretation = "Safe"
            
        # ── Save basic metrics ────────────────────────────
        result.current_ratio = float(curr_ratio_curr)

        total_equity = _safe_get(balance_sheet, "Stockholders Equity", bs_curr)
        result.debt_to_equity = float(total_liab / total_equity) if total_equity > 0 else 0.0

        result.return_on_assets = float(roa * 100)  # As percentage
        result.gross_margin = float(margin_curr * 100)  # As percentage

        # ── Beneish M-Score ───────────────────────────────
        result.beneish_mscore, result.beneish_interpretation = compute_beneish_mscore(
            income_stmt, balance_sheet, cash_flow
        )

        # ── Valuation Scores ──────────────────────────────
        val = compute_valuation_scores(income_stmt, balance_sheet, cash_flow, info)
        result.pe_ratio = val["pe_ratio"]
        result.pb_ratio = val["pb_ratio"]
        result.peg_ratio = val["peg_ratio"]
        result.ev_ebitda = val["ev_ebitda"]
        result.ps_ratio = val["ps_ratio"]
        result.fcf_yield = val["fcf_yield"]
        result.dcf_implied_upside = val["dcf_implied_upside"]
        result.revenue_growth_yoy = val["revenue_growth_yoy"]
        result.earnings_growth_yoy = val["earnings_growth_yoy"]
        result.roe = val["roe"]

    except Exception as e:
        logger.error(f"Error computing fundamental scores: {e}")
        result.data_quality = f"Calculation Error: {str(e)}"

    return result


def _safe_get(df: pd.DataFrame, row_name: str, col_name: str) -> float:
    """Safely extract a value from a DataFrame, returning 0 if missing."""
    try:
        if row_name in df.index:
            val = df.loc[row_name, col_name]
            if pd.isna(val):
                return 0.0
            return float(val)
        return 0.0
    except KeyError:
        return 0.0
    except Exception:
        return 0.0


# ── Beneish M-Score ────────────────────────────────────────────────────────────

def compute_beneish_mscore(
    income_stmt: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    cash_flow: pd.DataFrame,
) -> tuple[float, str]:
    """
    Compute the Beneish M-Score for earnings manipulation detection.

    M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
             + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI

    M-Score > -1.78 → likely manipulator.
    Returns (m_score, interpretation).
    """
    if (income_stmt is None or balance_sheet is None or cash_flow is None
            or income_stmt.empty or balance_sheet.empty
            or len(income_stmt.columns) < 2 or len(balance_sheet.columns) < 2):
        return -3.0, "Insufficient data"

    try:
        ic = income_stmt.columns[0]
        ip = income_stmt.columns[1]
        bc = balance_sheet.columns[0]
        bp = balance_sheet.columns[1]
        cfc = cash_flow.columns[0]

        # Revenue / Receivables
        rev_c = _safe_get(income_stmt, "Total Revenue", ic)
        rev_p = _safe_get(income_stmt, "Total Revenue", ip)
        rec_c = _safe_get(balance_sheet, "Receivables", bc)
        rec_p = _safe_get(balance_sheet, "Receivables", bp)
        DSRI = (rec_c / rev_c) / (rec_p / rev_p) if (rev_c > 0 and rev_p > 0 and rec_p > 0) else 1.0

        # Gross Margin Index
        gp_c = _safe_get(income_stmt, "Gross Profit", ic)
        gp_p = _safe_get(income_stmt, "Gross Profit", ip)
        gm_c = gp_c / rev_c if rev_c > 0 else 0
        gm_p = gp_p / rev_p if rev_p > 0 else 0
        GMI = gm_p / gm_c if gm_c > 0 else 1.0

        # Asset Quality Index
        ta_c = _safe_get(balance_sheet, "Total Assets", bc)
        ta_p = _safe_get(balance_sheet, "Total Assets", bp)
        ca_c = _safe_get(balance_sheet, "Current Assets", bc)
        ca_p = _safe_get(balance_sheet, "Current Assets", bp)
        ppe_c = _safe_get(balance_sheet, "Net PPE", bc)
        ppe_p = _safe_get(balance_sheet, "Net PPE", bp)
        nca_c = (ta_c - ca_c - ppe_c) / ta_c if ta_c > 0 else 0
        nca_p = (ta_p - ca_p - ppe_p) / ta_p if ta_p > 0 else 0
        AQI = nca_c / nca_p if nca_p > 0 else 1.0

        # Sales Growth Index
        SGI = rev_c / rev_p if rev_p > 0 else 1.0

        # Depreciation Index
        dep_c = _safe_get(income_stmt, "Reconciled Depreciation", ic)
        if dep_c == 0:
            dep_c = _safe_get(cash_flow, "Depreciation And Amortization", cfc)
        dep_p = _safe_get(income_stmt, "Reconciled Depreciation", ip)
        dep_rate_c = dep_c / (dep_c + ppe_c) if (dep_c + ppe_c) > 0 else 0
        dep_rate_p = dep_p / (dep_p + ppe_p) if (dep_p + ppe_p) > 0 else 0
        DEPI = dep_rate_p / dep_rate_c if dep_rate_c > 0 else 1.0

        # SG&A Index
        sga_c = _safe_get(income_stmt, "Selling General Administrative", ic)
        sga_p = _safe_get(income_stmt, "Selling General Administrative", ip)
        sgai_c = sga_c / rev_c if rev_c > 0 else 0
        sgai_p = sga_p / rev_p if rev_p > 0 else 0
        SGAI = sgai_c / sgai_p if sgai_p > 0 else 1.0

        # Total Accruals to Total Assets (TATA)
        ni_c = _safe_get(income_stmt, "Net Income", ic)
        cfo_c = _safe_get(cash_flow, "Operating Cash Flow", cfc)
        TATA = (ni_c - cfo_c) / ta_c if ta_c > 0 else 0

        # Leverage Index
        ltd_c = _safe_get(balance_sheet, "Long Term Debt", bc)
        ltd_p = _safe_get(balance_sheet, "Long Term Debt", bp)
        lev_c = (ltd_c + _safe_get(balance_sheet, "Current Liabilities", bc)) / ta_c if ta_c > 0 else 0
        lev_p = (ltd_p + _safe_get(balance_sheet, "Current Liabilities", bp)) / ta_p if ta_p > 0 else 0
        LVGI = lev_c / lev_p if lev_p > 0 else 1.0

        m = (-4.84 + 0.920 * DSRI + 0.528 * GMI + 0.404 * AQI
             + 0.892 * SGI + 0.115 * DEPI - 0.172 * SGAI
             + 4.679 * TATA - 0.327 * LVGI)

        interpretation = "Likely Manipulator" if m > -1.78 else "Clean"
        return float(m), interpretation

    except Exception as e:
        logger.warning(f"Beneish M-Score failed: {e}")
        return -3.0, "Calculation Error"


# ── Valuation Scores ────────────────────────────────────────────────────────────

def compute_valuation_scores(
    income_stmt: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    cash_flow: pd.DataFrame,
    info: dict,
) -> Dict[str, float]:
    """
    Compute valuation multiples from financial statements + yfinance info.

    Returns a dict with keys:
      pe_ratio, pb_ratio, peg_ratio, ev_ebitda, ps_ratio,
      fcf_yield, dcf_implied_upside, revenue_growth_yoy,
      earnings_growth_yoy, roe
    """
    result = {
        "pe_ratio": 0.0,
        "pb_ratio": 0.0,
        "peg_ratio": 0.0,
        "ev_ebitda": 0.0,
        "ps_ratio": 0.0,
        "fcf_yield": 0.0,
        "dcf_implied_upside": 0.0,
        "revenue_growth_yoy": 0.0,
        "earnings_growth_yoy": 0.0,
        "roe": 0.0,
    }

    if info is None:
        return result

    # From yfinance info (most reliable for market-derived ratios)
    result["pe_ratio"] = float(info.get("trailingPE", 0) or 0)
    result["pb_ratio"] = float(info.get("priceToBook", 0) or 0)
    result["peg_ratio"] = float(info.get("trailingPegRatio", 0) or 0)
    result["ev_ebitda"] = float(info.get("enterpriseToEbitda", 0) or 0)
    result["ps_ratio"] = float(info.get("priceToSalesTrailing12Months", 0) or 0)
    result["roe"] = float((info.get("returnOnEquity", 0) or 0) * 100)
    result["earnings_growth_yoy"] = float((info.get("earningsGrowth", 0) or 0) * 100)
    result["revenue_growth_yoy"] = float((info.get("revenueGrowth", 0) or 0) * 100)

    # FCF Yield = FCF / Market Cap
    market_cap = float(info.get("marketCap", 0) or 0)
    if market_cap > 0 and not income_stmt.empty and not cash_flow.empty:
        try:
            cfc = cash_flow.columns[0]
            cfo = _safe_get(cash_flow, "Operating Cash Flow", cfc)
            capex = abs(_safe_get(cash_flow, "Capital Expenditure", cfc))
            fcf = cfo - capex
            result["fcf_yield"] = float((fcf / market_cap) * 100) if market_cap > 0 else 0.0
        except Exception:
            pass

    # ── New: Asset Growth Anomaly (Cooper et al. 2008) ─────────────────────
    # High asset growth predicts underperformance
    try:
        _ta_curr = abs(float(balance_sheet.iloc[:, 0].get("Total Assets", 0) or 0))
        _ta_prev = abs(float(balance_sheet.iloc[:, 1].get("Total Assets", 0) or 0)) if balance_sheet.shape[1] > 1 else 0
        if _ta_prev > 0:
            result["asset_growth_yoy"] = float((_ta_curr / _ta_prev - 1) * 100)
    except Exception:
        pass

    # ── New: Net Stock Issuance (Daniel & Titman) ────────────────────────────
    # Firms that issue shares underperform; share buybacks outperform
    try:
        _shares_curr = float(info.get("sharesOutstanding", 0) or 0)
        _shares_prev = float(info.get("sharesOutstanding", _shares_curr) or _shares_curr)
        # Use treasury stock change as proxy if available
        _treasury_curr = abs(float(balance_sheet.iloc[:, 0].get("Treasury Shares Number", 0) or 0))
        _treasury_prev = abs(float(balance_sheet.iloc[:, 1].get("Treasury Shares Number", 0) or 0)) if balance_sheet.shape[1] > 1 else _treasury_curr
        _net_issuance_pct = float((_shares_curr - _shares_prev) / max(_shares_prev, 1) * 100) if _shares_prev > 0 else 0.0
        result["net_issuance_pct"] = _net_issuance_pct
    except Exception:
        pass

    # ── New: Gross Profitability / Novy-Marx (2013) ──────────────────────────
    # (Revenue − COGS) / Total Assets — cleanest quality metric
    try:
        _rev = abs(float(income_stmt.iloc[:, 0].get("Total Revenue", 0) or 0))
        _cogs = abs(float(income_stmt.iloc[:, 0].get("Cost Of Revenue", 0) or
                          income_stmt.iloc[:, 0].get("Cost of Goods Sold", 0) or 0))
        _ta_gp = abs(float(balance_sheet.iloc[:, 0].get("Total Assets", 1) or 1))
        if _ta_gp > 0 and _rev > 0:
            result["gross_profitability"] = float((_rev - _cogs) / _ta_gp * 100)
    except Exception:
        pass

    # ── New: Investment-to-Assets / q-factor (Hou et al.) ───────────────────
    # High CapEx growth relative to assets → overinvestment → underperformance
    try:
        _capex_curr = abs(float(cash_flow.iloc[:, 0].get("Capital Expenditure", 0) or 0))
        _capex_prev = abs(float(cash_flow.iloc[:, 1].get("Capital Expenditure", 0) or 0)) if cash_flow.shape[1] > 1 else 0
        _ta_ia = abs(float(balance_sheet.iloc[:, 0].get("Total Assets", 1) or 1))
        if _ta_ia > 0:
            result["investment_to_assets"] = float(_capex_curr / _ta_ia * 100)
            if _capex_prev > 0:
                result["capex_growth_yoy"] = float((_capex_curr / _capex_prev - 1) * 100)
    except Exception:
        pass

    # ── DCF with Dynamic WACC (real-time risk-free + credit spread) ──────────
    try:
        current_price = float(info.get("currentPrice", 0) or 0)
        eps = float(info.get("trailingEps", 0) or 0)
        growth_rate = max(-0.15, min(result["earnings_growth_yoy"] / 100, 0.30))

        # Dynamic WACC: risk-free (10Y Treasury) + equity risk premium + credit spread
        discount_rate = 0.10   # fallback
        terminal_growth = 0.03
        try:
            import yfinance as _yf_wacc
            _tnx = _yf_wacc.download("^TNX", period="5d", interval="1d",
                                     auto_adjust=True, progress=False)
            if not _tnx.empty:
                _rf = float(_tnx["Close"].squeeze().dropna().iloc[-1]) / 100
                _erp = 0.055   # long-run equity risk premium ~5.5%
                _beta = float(info.get("beta", 1.0) or 1.0)
                _beta = max(0.5, min(2.5, _beta))
                discount_rate = _rf + _beta * _erp
                discount_rate = max(0.06, min(0.18, discount_rate))
                # Terminal growth = inflation proxy (10Y breakeven ~ 2.3%)
                terminal_growth = min(_rf - 0.015, 0.04)
                terminal_growth = max(0.01, terminal_growth)
        except Exception:
            pass

        if current_price > 0 and eps > 0 and discount_rate > terminal_growth:
            dcf_value = 0.0
            eps_t = eps
            for _ in range(5):
                eps_t *= (1 + growth_rate)
                dcf_value += eps_t / ((1 + discount_rate) ** (_ + 1))
            terminal_eps = eps_t * (1 + terminal_growth)
            tv = terminal_eps / (discount_rate - terminal_growth)
            dcf_value += tv / ((1 + discount_rate) ** 5)
            result["dcf_implied_upside"] = float((dcf_value / current_price - 1) * 100)
            result["dcf_wacc_used"] = round(discount_rate * 100, 2)
    except Exception:
        pass

    return result
