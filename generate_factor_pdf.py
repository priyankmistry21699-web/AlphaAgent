"""
Generate AlphaAgent Factor & Parameter Inventory PDF
Uses fpdf2 (already installed)
"""

from fpdf import FPDF
from datetime import date

TODAY = date.today().strftime("%B %d, %Y")
OUT   = "AlphaAgent_Factor_Inventory.pdf"


class PDF(FPDF):
    def header(self):
        self.set_fill_color(15, 23, 42)       # dark navy
        self.rect(0, 0, 210, 14, "F")
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(148, 163, 184)     # slate
        self.set_xy(10, 3)
        self.cell(0, 8, "AlphaAgent  *  Factor & Parameter Inventory", ln=False)
        self.set_xy(-60, 3)
        self.cell(50, 8, TODAY, align="R")
        self.set_text_color(0, 0, 0)
        self.ln(8)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 8, f"Page {self.page_no()} / {{nb}}  |  AlphaAgent Confidential", align="C")
        self.set_text_color(0, 0, 0)

    # ── helpers ────────────────────────────────────────────────────────────

    def cover(self):
        self.add_page()
        # dark hero block
        self.set_fill_color(15, 23, 42)
        self.rect(0, 14, 210, 80, "F")
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 28)
        self.set_xy(0, 35)
        self.cell(210, 12, "AlphaAgent", align="C", ln=True)
        self.set_font("Helvetica", "", 14)
        self.set_text_color(96, 165, 250)
        self.cell(210, 8, "Quantitative Factor & Parameter Inventory", align="C", ln=True)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(148, 163, 184)
        self.cell(210, 7, TODAY, align="C", ln=True)
        self.ln(8)
        self.set_text_color(0, 0, 0)

        # stats grid
        stats = [
            ("9",   "Active Agents"),
            ("223", "Total Factors"),
            ("67",  "Static Params"),
            ("156", "Dynamic Params"),
        ]
        self.set_y(105)
        box_w = 42
        start_x = (210 - box_w * 4) / 2
        for i, (val, label) in enumerate(stats):
            x = start_x + i * box_w
            self.set_fill_color(30, 41, 59)
            self.rect(x, 105, box_w - 2, 24, "F")
            self.set_font("Helvetica", "B", 18)
            self.set_text_color(96, 165, 250)
            self.set_xy(x, 108)
            self.cell(box_w - 2, 10, val, align="C", ln=False)
            self.set_font("Helvetica", "", 7)
            self.set_text_color(148, 163, 184)
            self.set_xy(x, 118)
            self.cell(box_w - 2, 6, label, align="C", ln=False)

        # summary table
        self.set_text_color(0, 0, 0)
        self.set_xy(10, 140)
        self.h2("Agent Summary")

        agents = [
            ("1", "Technical",    "40", "22", "18", "Price Action, TDA, HMM, Options Greeks"),
            ("2", "Fundamental",  "42", "28", "14", "Fama-French 5F, Piotroski, Altman, DCF"),
            ("3", "Sentiment",    "24",  "1", "23", "NLP/LLM, Behavioural Finance, Options Flow"),
            ("4", "Macro",        "36",  "0", "36", "FRED Macro, Yield Curve, Credit Cycle, Liquidity"),
            ("5", "Risk",         "21",  "8", "13", "EVT, GARCH, Hawkes, Monte Carlo, Kelly"),
            ("6", "Currency",     "12",  "6",  "6", "FX Carry, PPP, EM Risk, Petro-currencies"),
            ("7", "Geopolitical", "22",  "2", "20", "Risk Premium, Safe-Haven, Commodity Shock"),
            ("8", "Insider",      "17",  "0", "17", "EDGAR 4/13F, Kyle Lambda, Dark Pool, Congress"),
            ("9", "Volatility",    "9",  "0",  "9", "GARCH, Stochastic Vol, SKEW, VVIX"),
        ]
        cols = [8, 30, 18, 16, 18, 90]
        hdrs = ["#", "Agent", "Factors", "Static", "Dynamic", "Key Theories"]
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(30, 41, 59)
        self.set_text_color(255, 255, 255)
        for w, h in zip(cols, hdrs):
            self.cell(w, 7, h, border=0, fill=True, align="C")
        self.ln()

        self.set_font("Helvetica", "", 7.5)
        for i, row in enumerate(agents):
            self.set_fill_color(245, 247, 250) if i % 2 == 0 else self.set_fill_color(255, 255, 255)
            self.set_text_color(30, 41, 59)
            for w, val in zip(cols, row):
                self.cell(w, 6.5, val, border=0, fill=True, align="C" if w < 40 else "L")
            self.ln()

        # total row
        self.set_fill_color(15, 23, 42)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 8)
        totals = ["", "TOTAL", "223", "67", "156", "30% static  |  70% dynamic"]
        for w, val in zip(cols, totals):
            self.cell(w, 7, val, fill=True, align="C" if w < 40 else "L")
        self.ln()
        self.set_text_color(0, 0, 0)

    def h1(self, txt):
        self.ln(4)
        self.set_fill_color(15, 23, 42)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 9, f"  {txt}", fill=True, ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def h2(self, txt):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(30, 64, 175)
        self.cell(0, 7, txt, ln=True)
        self.set_draw_color(96, 165, 250)
        self.set_line_width(0.4)
        x = self.get_x()
        y = self.get_y()
        self.line(10, y, 200, y)
        self.set_draw_color(0, 0, 0)
        self.set_line_width(0.2)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def tag(self, txt, dynamic=True):
        """Inline D/S badge."""
        if dynamic:
            self.set_fill_color(220, 252, 231)
            self.set_text_color(21, 128, 61)
        else:
            self.set_fill_color(219, 234, 254)
            self.set_text_color(29, 78, 216)
        self.set_font("Helvetica", "B", 6.5)
        self.cell(8, 4.5, txt, fill=True, align="C")
        self.set_text_color(0, 0, 0)
        self.set_font("Helvetica", "", 8)

    def factor_table(self, headers, rows, col_widths):
        self.set_font("Helvetica", "B", 7.5)
        self.set_fill_color(30, 41, 59)
        self.set_text_color(255, 255, 255)
        for w, h in zip(col_widths, headers):
            self.cell(w, 6.5, h, border=0, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", 7)
        for i, row in enumerate(rows):
            if self.get_y() > 270:
                self.add_page()
            self.set_fill_color(245, 247, 250) if i % 2 == 0 else self.set_fill_color(255, 255, 255)
            self.set_text_color(30, 41, 59)
            *data_cells, ds = row
            for w, val in zip(col_widths[:-1], data_cells):
                self.cell(w, 6, str(val), border=0, fill=True, align="L")
            # badge cell
            is_d = ds.upper() == "D"
            if is_d:
                self.set_fill_color(220, 252, 231)
                self.set_text_color(21, 128, 61)
            else:
                self.set_fill_color(219, 234, 254)
                self.set_text_color(29, 78, 216)
            self.set_font("Helvetica", "B", 7)
            self.cell(col_widths[-1], 6, "DYNAMIC" if is_d else "STATIC", fill=True, align="C")
            self.set_text_color(30, 41, 59)
            self.set_font("Helvetica", "", 7)
            self.ln()
        self.set_text_color(0, 0, 0)

    def section_intro(self, txt):
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(71, 85, 105)
        self.multi_cell(0, 5, txt)
        self.set_text_color(0, 0, 0)
        self.ln(1)


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

TECHNICAL_FACTORS = [
    ("RSI (14)",                      "Oscillator",      "window=14, thresholds 30/70 equity 25/75 crypto",  "D"),
    ("MACD (12/26/9)",                "Signal Crossover","fast=12, slow=26, signal=9",                       "S"),
    ("Bollinger Bands (20/2)",        "Vol Compression", "window=20, std=2.0, bw_narrow=20, bw_wide=40",     "D"),
    ("SMA 50/200 + Golden/Death Cross","Trend Following","periods=50, 200",                                  "S"),
    ("EMA 9/21 Crossover",            "Momentum",        "periods=9, 21",                                    "S"),
    ("ADX (14)",                      "Trend Strength",  "window=14, thresholds 20/40",                      "S"),
    ("ATR (14)",                      "Volatility Unit", "window=14",                                        "S"),
    ("OBV + 20-SMA",                  "Vol Confirmation","threshold=+/-2%",                                  "S"),
    ("VWAP",                          "Fair Value",      "intraday level",                                   "S"),
    ("Stochastic %K/%D",              "Oscillator",      "window=14, smooth=3, thresholds 20/80",            "S"),
    ("52-Week Range Position",        "Momentum Context","<40% / <75% scoring",                              "S"),
    ("ATR-Normalised Momentum",       "Risk-Adj Momentum","20-day, score=50+mom*15",                         "S"),
    ("Bollinger Bandwidth Squeeze",   "Vol Regime",      "bw from settings (20/40)",                         "D"),
    ("Momentum Acceleration (3M delta)","Jerk Factor",   "3M now vs prior, score=50+accel*1.2",              "S"),
    ("Variance Risk Premium",         "Options Theory",  "ATM straddle IV vs 30d RV",                        "S"),
    ("Hurst Exponent",                "Fractal Markets", ">0.55=trend / <0.45=mean-revert",                  "S"),
    ("Jegadeesh-Titman 12M-1M",       "Cross-Sectional", "threshold>5% /<-5%",                               "S"),
    ("HMM Market Regime (3-state)",   "Hidden Markov",   "Bull / Bear / Crisis states",                      "D"),
    ("Seasonality Factors",           "Calendar Anomaly","month-based scoring",                              "D"),
    ("TDA Persistent Homology H0/H1", "Topological DA",  "Betti-1, H0 entropy",                              "D"),
    ("IV Skew (25-delta put-call)",   "Options",         "score=60-skew*2, clamped [10,90]",                 "S"),
    ("Gamma Exposure (GEX)",          "Options Greeks",  "T=21/365, Rf=4.5%",                                "D"),
    ("Max Pain Level",                "Options Theory",  "min(call+put intrinsic) across strikes",            "D"),
    ("Implied Correlation SPY/Stock", "Dispersion",      ">0.8 limited alpha / <0.4 idiosyncratic",          "S"),
    ("Fibonacci Retracement",         "Market Geometry", "23.6/38.2/50/61.8/78.6% levels",                   "S"),
    ("Pivot Points (daily + weekly)", "Price Structure", "PP=(H+L+C)/3, R1/R2, S1/S2",                       "S"),
    ("Multi-Timeframe Confluence",    "Signal Alignment","daily vs weekly RSI+MACD, max 4 signals",           "S"),
    ("Order Flow Imbalance (CLV)",    "Microstructure",  "5d avg, 10d/20d money-flow ratio",                 "S"),
    ("Cross-Sectional Momentum Rank", "Factor Investing","22d lookback, peers=5 from settings",               "D"),
    ("Signal Backtest Calibration",   "Walk-Forward",    "lookback=126d, min_similar=5",                     "D"),
    ("Double Top/Bottom Patterns",    "Chart Patterns",  "60-bar scan, +/-3% tolerance",                     "D"),
    ("Volume Profile / POC",          "Market Profile",  "60-bar, 20 bins, 70% value area",                  "D"),
    ("PCA Signal Quality",            "Factor Analysis", "consensus + dispersion across factors",             "D"),
    ("Ichimoku Cloud",                "Japanese TA",     "Tenkan=9, Kijun=26, Senkou=52",                    "S"),
    ("Chaikin Money Flow (14)",       "Vol/Price Flow",  "MFV 14-period",                                    "S"),
    ("TRIX (15)",                     "Triple EMA Osc",  "period=15, %change",                               "S"),
    ("Parabolic SAR",                 "Trend Reversal",  "AF_step=0.02, AF_max=0.20",                        "S"),
    ("Supply/Demand Zones",           "Order Flow",      "60-bar, vol spike >1.5x avg",                      "D"),
    ("Sector ETF Momentum",           "Relative Strength","5-day, 1.5x weighting",                           "D"),
    ("5-Day Price Momentum",          "Short-Term Mom",  "6-day return",                                     "S"),
]

FUNDAMENTAL_FACTORS = [
    ("Piotroski F-Score (9-point)",   "Quality Accounting",    "9 binary checks; <=3=Poor 4-6=Fair 7-9=Strong",  "S"),
    ("Altman Z-Score",                "Bankruptcy Prediction", "weights 1.2/1.4/3.3/0.6/1.0; <1.81 distress",   "S"),
    ("Beneish M-Score",               "Manipulation Detect",   "8-variable; >-1.78=likely manipulator",          "S"),
    ("P/E Ratio (trailing)",          "Value",                 "pe_cheap=15, pe_fair=25, pe_exp=40 (rate-adj)",  "D"),
    ("Forward P/E",                   "Value",                 "same thresholds as trailing",                    "D"),
    ("P/B Ratio",                     "Value",                 "pb_under=1.5, pb_fair=3.0",                      "S"),
    ("PEG Ratio",                     "Growth-Value",          "peg_under=1.0, peg_fair=2.0",                    "S"),
    ("EV/EBITDA",                     "Enterprise Value",      "cheap=10, fair=20, exp=30",                      "S"),
    ("P/S Ratio",                     "Revenue Multiple",      "cheap=2.0, fair=4.0",                            "S"),
    ("FCF Yield",                     "Cash Generation",       "good=2%, great=5%",                              "S"),
    ("DCF Implied Upside",            "Intrinsic Value",       "upside_strong=30%, moderate=10%",                "S"),
    ("Revenue Growth YoY",            "Growth",                "strong=20%, moderate=10%",                       "S"),
    ("Earnings Growth YoY",           "Growth",                "strong=20%, moderate=10%",                       "S"),
    ("ROE",                           "Profitability",         "strong=20%, moderate=10%",                       "S"),
    ("ROA",                           "Efficiency",            "strong=15%, moderate=8%, weak=3%",               "S"),
    ("Operating Margin",              "Quality",               "strong=25%, moderate=15%, weak=5%",              "S"),
    ("Current Ratio",                 "Liquidity",             "liquid=2.0, adequate=1.5, tight=1.0",            "S"),
    ("Interest Coverage",             "Solvency",              "strong=10x, adequate=5x, weak=2x",               "S"),
    ("Asset Turnover",                "Efficiency",            "good=1.0, weak=0.5",                             "S"),
    ("EPS Surprise",                  "Earnings Quality",      "beat=+10%, miss=-2%",                            "S"),
    ("Accruals Ratio",                "Earnings Quality",      "(NI-OCF)/Assets; <-0.05 / 0.05 / 0.15",         "S"),
    ("Net Margin Trend",              "Profitability Momentum","current vs prior net margin delta",               "S"),
    ("Dividend Cut Probability",      "Income Risk",           "payout>90% or (>70% and FCF<0)=cut risk",        "S"),
    ("Shares Buyback Signal",         "Capital Allocation",    "<-3% / <0% / +2% shares change",                 "S"),
    ("Fama-French HML (Value)",       "Factor Investing",      "PE+PB+EV blend",                                 "S"),
    ("Fama-French SMB (Size)",        "Factor Investing",      "market cap deciles",                             "S"),
    ("Fama-French RMW (Quality)",     "Factor Investing",      "ROE+margins+debt ratio",                         "S"),
    ("Fama-French BAB (Low Vol)",     "Factor Investing",      "beta regime scoring",                            "S"),
    ("CAPM Jensen's Alpha (1Y)",      "Asset Pricing",         "a=Ri-[Rf+B(Rm-Rf)], score=50+alpha*1.5",        "D"),
    ("Dividend Growth CAGR (5Y)",     "Income",                ">10% / >5% / >0% thresholds",                    "S"),
    ("Lockup Expiration Proxy",       "Supply Overhang",       "0-180d=25 / 180-365d=45 / >365d=65",            "S"),
    ("P/E vs Sector Median",          "Relative Value",        "<0.8=85 / <1.0=65 / <1.3=45",                   "D"),
    ("P/E vs 5Y Historical Avg",      "Mean Reversion",        "<0.85=80 / <1.0=60 / <1.3=40",                  "D"),
    ("Earnings Consistency (CoV)",    "Quality",               "CoV<0.2 / <0.4 / <0.7",                         "D"),
    ("CEO/CFO Departure (8-K NLP)",   "Event-Driven",          "90-day EDGAR window",                            "D"),
    ("M&A Activity (8-K NLP)",        "Event-Driven",          "180-day EDGAR window; >2=75 / >0=65",            "D"),
    ("Earnings Revision Momentum",    "Estimate Trend",        ">5% / >2% / >-2% (30d/60d)",                    "D"),
    ("Graham Number",                 "Benjamin Graham",       "sqrt(22.5 * EPS * BVPS); upside vs price",       "D"),
    ("Earnings Call NLP (Gemini)",    "LLM Analysis",          "quality+revenue+guidance+red-flags (4 dims)",    "D"),
    ("Net Debt / EBITDA",             "Leverage",              "<1.0=75 / <2.5=60 / <4.0=40",                   "D"),
    ("FCF Quality (FCF/NI)",          "Cash Conversion",       ">1.1=88 / >0.8=68 / >0.5=45",                   "D"),
    ("Analyst Revision Momentum",     "Estimate Sentiment",    "bull% change >8%=80 / >2%=65",                   "D"),
]

SENTIMENT_FACTORS = [
    ("News Sentiment (Gemini RAG)",   "NLP/LLM",             "0-100 scale (0=panic 100=euphoria)",              "D"),
    ("Short Interest % + DTC",        "Behavioural",         "short_interest_high=20%, DTC threshold",          "D"),
    ("Fear & Greed Index (contrarian)","Behavioural Finance", "<=25 bullish / >=75 bearish",                    "D"),
    ("Analyst Consensus %",           "Sell-Side Sentiment", ">70%=+5% / <30%=-4% probability",                 "D"),
    ("Market Breadth (RSP/SPY)",      "Breadth Analysis",    ">65=+3% / <35=-3%",                               "D"),
    ("Options Put/Call OI Ratio",     "Options Flow",        "<0.7=70 / 0.7-1.2=50 / >1.2=25",                 "D"),
    ("News/Social Momentum (3d)",     "Social Listening",    "base=40, mult=4.0, score=min(80,40+count*4)",     "D"),
    ("EPS Forward Revision",          "Estimate Trend",      ">15%=85 / >5%=60 / >-5%=50",                     "D"),
    ("Unusual Options Activity",      "Smart Money",         "call vol/OI>2.0; score=min(85,40+ratio*30)",      "D"),
    ("Transfer Entropy (news->price)","Information Theory",  "lag-1 cross-corr; causal signal if >0.3",         "D"),
    ("Signal Entropy (Shannon)",      "Information Theory",  "H=(1-entropy)*80+10",                             "D"),
    ("Analyst Price Target Upside",   "Sell-Side",           ">20%=85 / >5%=65 / >-5%=50",                     "D"),
    ("AAII Sentiment (contrarian)",   "Behavioural",         "bull-bear spread <-20=75 / >30=25",               "D"),
    ("Consumer Credit (margin proxy)","Macro-Sentiment",     "3M % change >2%=60 / <-1%=40",                   "D"),
    ("Short Interest Change MoM",     "Behavioural",         "<-10%=75 (covering) / >+10%=20 (rising)",         "D"),
    ("Sentiment Momentum (3d vel.)",  "Social Velocity",     "3d vs 3-6d delta; >2=70 / >0=55 / <-3=30",       "D"),
    ("Source Credibility Weight",     "Media Quality",       "WSJ/BBG=1.0, SeekingAlpha=0.70 (static tiers)",  "S"),
    ("Headline/Body Alignment (LLM)", "NLP",                 "Gemini dual-pass tone divergence check",          "D"),
    ("FinBERT-Style NLP (Gemini)",    "LLM",                 "6-dim: sentiment/revenue/mgmt/risk/catalyst/conf","D"),
    ("Reddit Sentiment (WSB+Stocks)", "Social Media",        ">60% bullish=72 / 45-60%=55 / <30%=35",           "D"),
    ("News Decay Model (exp)",        "Signal Decay",        "halflife=4.6 days; dynamic near earnings",        "D"),
    ("Unusual Options Sweeps",        "Smart Money",         ">3 calls: +4% prob / >3 puts: -4% prob",          "D"),
    ("Short-Squeeze Score",           "Technical/Sentiment", "short float% + days-to-cover; >20%+5d=80",        "D"),
    ("PCA Signal Quality",            "Factor Analysis",     "factor consensus + dispersion",                    "D"),
]

MACRO_FACTORS = [
    ("Recession Probability",         "Macro Regime",         "FRED-derived; low=0.2 / cutoff=0.6",             "D"),
    ("Yield Curve (10Y-2Y Spread)",   "Yield Curve",          ">0.5%=100 / inverted<0%=0; positive=0.5",        "D"),
    ("Fed Funds Rate",                "Monetary Policy",      "easy=2.0% / tight=4.0%",                         "D"),
    ("CPI Inflation YoY",             "Inflation",            "target=2.5% / elevated=5.0%",                    "D"),
    ("PCE Inflation YoY",             "Inflation (Fed gauge)", "target=2.0% / near_target=2.5% / elevated=3.5%","D"),
    ("VIX (2Y rolling percentile)",   "Fear Index",           "percentile-based; fallback 90-(VIX-10)*2.2",     "D"),
    ("Credit Spreads (HYG/LQD 1m)",   "Credit Cycle",         "score=50+delta*5",                               "D"),
    ("Copper/Gold Ratio (1m)",        "Growth Signal",        ">5%: +5% / <-5%: -5%",                           "D"),
    ("BTC Risk-On Signal (1m)",       "Risk Appetite",        ">20%: + / <-20%: -",                             "D"),
    ("Global Breadth (ACWI/SPY)",     "Global Macro",         ">2%: +3% / <-3%: -4%",                           "D"),
    ("Dollar Index Regime (DXY)",     "FX / Liquidity",       "STRONG=30 / WEAK=70",                            "D"),
    ("M2 Money Supply YoY",           "Liquidity",            "FRED M2SL; >5%=75 / >0%=50 / <0%=30",           "D"),
    ("ISM PMI Manufacturing",         "Business Cycle",       "FRED NAPM; expansion=50 / strong=55 / weak=48",  "D"),
    ("UMich Consumer Sentiment",      "Consumer Confidence",  "FRED UMCSENT; strong=90 / moderate=70 / weak=55","D"),
    ("Initial Jobless Claims (4W avg)","Labour Market",       "FRED ICSA; low=220K / elevated=260K / high=350K","D"),
    ("TIPS 10Y Breakeven Inflation",  "Inflation Expectation","FRED T10YIE; low=1.5% / high=2.5% / danger=3.0%","D"),
    ("Amihud Illiquidity Ratio",      "Market Microstructure","stress_threshold=2.0x 3M avg",                   "D"),
    ("Bond-Equity Corr (TLT/SPY 60d)","Regime Detection",    "stress_corr=0.3",                                 "D"),
    ("Semiconductor RS (SOXX/SPY 1m)","Sector Rotation",     ">5%: +5% / <-5%: -5%",                           "D"),
    ("Global Pre-Market (6 intl ETFs)","Global Macro",        "EWJ/EWZ/EWY/EWG/EWU/EWQ avg >3%",               "D"),
    ("Fed Balance Sheet (WALCL 2M)", "QE/QT Signal",          ">1%=75 (QE) / <-0.5%=30 (QT)",                  "D"),
    ("Growth vs Value Rotation",      "Factor Rotation",      "IWF/IWD 1m RS; >3%=growth / <-3%=value",        "D"),
    ("Repo Stress (SOFR-FF spread)",  "Funding Markets",      "critical=50bps",                                  "D"),
    ("Fed Rate Change Direction",     "Monetary Policy",      "3M delta >0.24pp=hiking / <-0.24pp=cutting",     "D"),
    ("VIX 5-Day Change",              "Vol Momentum",         "score=50-delta*4",                                "D"),
    ("SPY vs SMA(200)",               "Market Regime",        ">5%=bull(80) / <-5%=bear(15)",                   "D"),
    ("Large vs Small Cap (SPY/IWM)",  "Cycle Phase",          "1m RS; >4%=late cycle / <-4%=early cycle",       "D"),
    ("Sector RS (sector ETF/SPY)",    "Sector Rotation",      "1m RS",                                           "D"),
    ("Contagion Corr Spike",          "Systemic Risk",        "SPY/TLT/GLD/HYG 20d avg corr vs historical",     "D"),
    ("Real Interest Rate (10Y-TIPS)", "Real Returns",         ">1.5%=70 / >0.5%=55 / >0%=45 / >-1%=30",        "D"),
    ("Retail Sales ex-Auto MoM",      "Consumer Spending",    "FRED RSXFS; >1%=75 / >0%=60 / >-1%=40",         "D"),
    ("Leading Economic Index (LEI)",  "Forward-Looking",      "FRED USSLIND 3M change; >0.5=72",                "D"),
    ("HY Credit Spread OAS",          "Credit Cycle",         "FRED BAMLH0A0HYM2; dynamic vs 5Y median",        "D"),
    ("NFCI Financial Conditions",     "Liquidity",            "FRED NFCI; neutral=0 / tight>0 / easy<0",        "D"),
    ("Equity Risk Premium",           "Valuation",            "(E/P%) - Real Rate; >4%=75 / >2%=60",            "D"),
    ("PCA Signal Quality",            "Factor Analysis",      "macro factor consensus",                           "D"),
]

RISK_FACTORS = [
    ("GARCH Vol Regime (1,1)",        "GARCH Volatility",     "fitted a/B/omega; pct 25/75/95",                 "D"),
    ("EVT Tail Risk (99% VaR)",       "Extreme Value Theory", "min_obs=100, threshold_pct=10%, block=21d",      "D"),
    ("Monte Carlo 95% CI (5d)",       "Stochastic Sim",       "paths=5000, GBM",                                "D"),
    ("Kelly Criterion (half-Kelly)",  "Optimal Sizing",       "fraction=0.5, cap=25%",                          "D"),
    ("Black Swan Detection (>5-sigma)","Fat Tail",            "sigma=5.0 threshold",                            "D"),
    ("Flash Crash Detection",         "Circuit Breaker",      "ticker=8%, SPY=6.5% 1-day move",                 "S"),
    ("WAR/GEO Shock Override",        "Regime Filter",        "VIX=35 gold=2% oil=5% mult=0.35",               "S"),
    ("Carry Unwind Override (JPY)",   "Carry Trade Risk",     "USDJPY=125, multiplier=0.50",                    "S"),
    ("KL Divergence (return shift)",  "Distribution Change",  "20d vs 252d baseline; threshold=1.0",            "D"),
    ("Hawkes Process (jump cascade)", "Point Process",        "branching warn=0.8, critical=0.95",              "D"),
    ("Quasi-Monte Carlo VaR (Sobol)", "Low-Discrepancy",      "5d horizon, 95% CI",                             "D"),
    ("Liquidity Risk (vol ratio)",    "Market Micro",         "high=1.5x, low=0.7x vs 20d avg",                "S"),
    ("Drawdown from ATH",             "Drawdown Analysis",    "2Y high; warning=30%",                           "S"),
    ("Tail Ratio (95th/5th pct)",     "Return Asymmetry",     "positive=1.2, negative=0.8",                    "S"),
    ("Correlation Regime (vs SPY 60d)","Regime Detection",    "delta_threshold=0.1",                            "S"),
    ("Rolling Sharpe (63d)",          "Risk-Adj Return",      "annualised",                                     "D"),
    ("Rolling Sortino (63d)",         "Downside Risk",        "annualised",                                     "D"),
    ("Vanna/Charm Exposure",          "Options 2nd-order",    "vanna_neg_threshold=0.5",                        "D"),
    ("Return Skewness + Excess Kurtosis","Distribution Shape","skew<-1=20; kurt>5=warning",                     "D"),
    ("Options GEX Wall",              "Gamma Exposure",       ">0=absorb vol / <0=amplify",                     "D"),
    ("Pairs Trading Z-score",         "Statistical Arb",      "log-ratio of correlated pair; Z*15 scoring",     "D"),
]

CURRENCY_FACTORS = [
    ("DXY Regime + Sector FX Exposure","FX Carry",            "sector weights 0.10-0.55 (static) DXY dynamic",  "D"),
    ("EUR/USD Rate",                  "PPP / FX",             ">1.05=65 / <0.95=35",                            "S"),
    ("USD/JPY Carry Signal",          "Carry Trade",          "strong=140, weak=120",                           "S"),
    ("USD/CNY (China Stress)",        "EM Risk",              "stress=7.3",                                     "S"),
    ("GBP/USD Rate",                  "FX",                   ">1.25=65",                                       "S"),
    ("EM Currency Stress",            "EM Risk",              "USD 1m gain vs EM basket; >3%/-2%",              "D"),
    ("FX Translation Impact",         "Revenue Impact",       "-DXY_1m * sector_exp * 0.25",                    "S"),
    ("Petro-Currency (CAD/AUD 1m)",   "Commodity FX",         ">2%: +3% / <-2%: -3%",                          "D"),
    ("Carry Trade Attractiveness",    "Rate Differential",    "Fed - ECB; >1%=70 / >0%=50",                    "D"),
    ("Real Interest Rate (10Y-TIPS)", "Real Returns",         ">1%=65 / >0%=50 / >-1%=40",                     "D"),
    ("EM FX Pressure (EEM 1m)",       "EM Flow",              "momentum-based",                                 "D"),
    ("DXY vs SMA(50)",                "Trend Direction",      ">1% above=strong dollar headwind",               "D"),
]

GEO_FACTORS = [
    ("Oil Shock (XLE/SPY RS 1m)",     "Commodity Risk",       ">5%: -8% / <-5%: +3%",                          "D"),
    ("Gold Safe-Haven (GLD 1m)",      "Safe-Haven Flow",      ">5%: -10% / <-3%: +5%",                         "D"),
    ("EM Stress (EEM/SPY 1m)",        "EM Contagion",         "<-5%: -6% / >3%: +4%",                          "D"),
    ("Defense Sector RS (ITA/SPY 1m)","War Risk Premium",     ">5%: -7%",                                       "D"),
    ("VIX Fear Index",                "Fear Gauge",           "cap=80; >30: -10% / >20: -5%",                   "D"),
    ("Copper/Gold Ratio (1m)",        "Growth vs Safety",     ">3%: +5% / <-3%: -5%",                          "D"),
    ("Global Breadth (ACWI/SPY 1m)", "Global Risk",           "<-3%: -4% / >2%: +3%",                          "D"),
    ("USD Safe-Haven (DXY 1m)",       "Currency Risk",        ">2%: -6% / <-2%: +4%",                          "D"),
    ("Credit Stress (HYG/LQD 1m)",    "Credit Risk",          "<-2%: -7% / >1%: +4%",                          "D"),
    ("Sector Rotation Signal",        "Risk Rotation",        "XLK*1.5 - XLE - XLF*0.5",                      "D"),
    ("Shipping Index (BDRY 1m)",      "Trade Demand",         "<-20%=demand destruction",                       "D"),
    ("Commodity Shock (XLE+XLB 1m)",  "Inflation Risk",       ">8%: inflation / <-8%: demand cooling",          "D"),
    ("EM Contagion Risk (EMB 1m)",     "Contagion",            "<-3%=EM stress spreading",                       "D"),
    ("Oil Direct (BZ=F 1m)",          "Commodity",            ">15%: -5% / <-15%: +3%",                        "D"),
    ("Election Cycle Phase",          "Political Risk",       "2026 midterms / 2028 presidential (hardcoded)",  "S"),
    ("Transport Index (IYT/SPY 1m)",  "Dow Theory",           "Dow Theory transportation confirmation",          "D"),
    ("Commodity Index (PDBC 1m)",     "Inflation",            ">12%: -4% prob",                                 "D"),
    ("GPR Geopolitical Risk (FRED)",  "Geopolitical Risk",    ">2x hist: -8% / >1.5x: -4% / <0.75x: +3%",     "D"),
    ("Active Conflict Score (NLP)",   "Text Mining",          "10 keywords; >3 hits: -6%",                      "D"),
    ("Sanctions Risk (NLP)",          "Text Mining",          "8 keywords; >=2 hits: -5%",                      "D"),
    ("Tariff/Regulatory Risk (NLP)",  "Text Mining",          "combined >3 keywords: -4%",                      "D"),
    ("PCA Signal Quality",            "Factor Analysis",      "geopolitical factor consensus",                   "D"),
]

INSIDER_FACTORS = [
    ("EDGAR Form 4 Activity (60d)",   "Information Asymmetry","cluster threshold by market cap",                 "D"),
    ("Material Events (8-K, 30d)",    "Event-Driven",         ">=5=warning (30)",                               "D"),
    ("Insider Cluster Buys",          "Smart Money",          "score=min(85, 40+count*12)",                     "D"),
    ("13F Institutional Change QoQ",  "Institutional Flow",   ">2pp=75 (accum) / <-2pp=20 (distrib)",          "D"),
    ("Short Squeeze Potential",       "Short Float",          "short%+days-to-cover combined",                  "D"),
    ("Float Reduction (buyback)",     "Capital Allocation",   "<-3% / <0% / <2% shares YoY",                   "D"),
    ("Kyle's Lambda (price impact)",  "Microstructure",       "normalized >2=thin(10) / <0.5=liquid(70)",       "D"),
    ("Congressional Trading Signal",  "Political Intelligence","House Stock Watcher 180d; net>0=75",             "D"),
    ("FINRA Short Volume %",          "RegSHO Data",          "weekly; >55%=25 / <45%=70",                     "D"),
    ("ETF Flow Impact",               "Passive Flow",         "sector ETF price*vol proxy",                     "D"),
    ("Activist 13D/G Filing (90d)",   "Activist Catalyst",    "EDGAR; >0=80 (activist signal)",                 "D"),
    ("Top-10 Holder Concentration",   "Ownership Structure",  "HHI; 0.05-0.25=60 / >0.25=40",                 "D"),
    ("Dark Pool Print Ratio (FINRA)", "Institutional Flow",   "off-exchange%; >50%=72 / >35%=55",               "D"),
    ("Insider Cluster (30d Form 4)",  "Form 4 Analysis",      ">=threshold=75",                                 "D"),
    ("EDGAR Institutional Ownership", "Ownership",            "% from EDGAR 13F data",                          "D"),
    ("Net Insider Shares (buy/sell)", "Sentiment",            "net shares bought/sold",                         "D"),
    ("Analyst Price Target",          "Sell-Side",            "mean target vs current price",                   "D"),
]

VOLATILITY_FACTORS = [
    ("GARCH Vol Regime (1,1)",        "GARCH",                "fitted alpha/beta/omega; pct 25/75/95",          "D"),
    ("Put/Call OI Ratio",             "Options Theory",       "bearish=0.7, overbought=1.2",                    "D"),
    ("IV vs Realized Vol (VRP)",      "Vol Risk Premium",     "underpriced=0.7, overpriced=1.5",               "D"),
    ("Kalman Dynamic Beta",           "State-Space Model",    "Kalman filter vs SPY returns",                   "D"),
    ("SPY Correlation (60d rolling)", "Systematic Risk",      "rolling 60d window",                             "D"),
    ("VVIX (vol of vol)",             "Second-Order Vol",     "extreme=120, elevated=90",                       "D"),
    ("VIX Term Structure",            "Vol Surface",          "VIX3M - spot VIX; >1=contango / <-1=backwardation","D"),
    ("Realized Vol Trend (10d/30d)",  "Vol Momentum",         "calm=0.8, spiking=1.2 ratio",                   "D"),
    ("SKEW Index (tail risk)",        "Tail Risk Premium",    "extreme=140, elevated=130",                      "D"),
]

QUANT_MODULES = [
    ("bayesian.py",       "Bayesian Updating",         "prior=0.5, sensitivity=1.8, corr_penalty=(1-corr)"),
    ("garch.py",          "GARCH(1,1)",                "fitted a/B/omega per ticker; 1d/5d/10d forecasts"),
    ("technical.py",      "10 Core Indicators",        "composite 0-100, bullish/bearish count"),
    ("evt.py",            "Extreme Value / GPD",       "min_obs=100, threshold_pct=10%, block=21d"),
    ("hawkes.py",         "Hawkes Point Process",      "branching warn=0.8, critical=0.95"),
    ("scoring.py",        "Piotroski/Altman/Beneish",  "hardcoded academic formulas"),
    ("calibration.py",    "Walk-Forward Calibration",  "lookback=126d, rsi_window=15"),
    ("tda_signal.py",     "Topological Data Analysis", "Persistent Homology H0/H1, Betti numbers"),
    ("hmm.py",            "Hidden Markov Model",       "3-state: Bull/Bear/Crisis"),
    ("kalman.py",         "Kalman Filter",             "dynamic beta estimation"),
    ("kelly.py",          "Kelly Criterion",           "half-Kelly, vol-adjusted position size"),
    ("monte_carlo.py",    "GBM Monte Carlo",           "paths=5000, dt=1/252"),
    ("quasi_mc.py",       "Sobol Sequences (QMC)",     "low-discrepancy sampling"),
    ("copula.py",         "Gaussian/Clayton Copula",   "tail dependence modelling"),
    ("granger.py",        "Granger Causality",         "F-test, maxlag=10"),
    ("causal_engine.py",  "Causal Inference",          "transfer entropy"),
    ("heston.py",         "Heston Stochastic Vol",     "k/theta/sigma/rho fitted"),
    ("sabr.py",           "SABR Vol Surface",          "alpha/beta/rho/nu fitted"),
    ("rough_vol.py",      "Rough Heston",              "Hurst H<0.5 fractional Brownian"),
    ("multifractal.py",   "MF-DFA",                   "q-order multifractal moments"),
    ("quantum_finance.py","Quantum Amplitude Est.",    "portfolio optimization"),
    ("lob.py",            "Limit Order Book",          "microstructure metrics"),
    ("factor_exposure.py","Risk Factor Loadings",      "Fama-French 5-factor exposures"),
]

CONFIG_PARAMS = [
    ("technical_period",          "1y",    "Data lookback window",                      "S"),
    ("risk_period",               "5d",    "Forecast horizon",                          "S"),
    ("backtest_period",           "3y",    "Backtest window",                           "S"),
    ("rsi_window",                "14",    "RSI calculation window",                    "S"),
    ("bollinger_window / std",    "20/2.0","Bollinger window and multiplier",           "S"),
    ("adx_window / atr_window",   "14",    "ADX & ATR window",                          "S"),
    ("stoch_window / smooth",     "14/3",  "Stochastic window & smoothing",             "S"),
    ("bw_narrow / bw_wide",       "20/40", "Bollinger bandwidth thresholds",            "S"),
    ("num_paths",                 "5000",  "Monte Carlo simulation paths",              "S"),
    ("ci_level_1/2/3",            ".68/.95/.99","Confidence interval levels",           "S"),
    ("regime_low/normal/high_pct","25/75/95","GARCH percentile regime boundaries",      "S"),
    ("put_call_bearish/overbought","0.7/1.2","Put-call ratio thresholds",               "S"),
    ("iv_rv_underpriced/overpriced","0.7/1.5","IV vs RV thresholds",                   "S"),
    ("vix_elevated/extreme",      "20/30", "VIX level thresholds",                     "S"),
    ("vvix_extreme/elevated",     "120/90","VVIX thresholds",                          "S"),
    ("skew_extreme/elevated",     "140/130","SKEW index thresholds",                   "S"),
    ("branching_warn/critical",   "0.8/0.95","Hawkes branching ratio thresholds",      "S"),
    ("half_kelly_fraction",       "0.5",   "Kelly fraction applied",                   "S"),
    ("max_kelly_cap",             "0.25",  "Maximum Kelly position cap",               "S"),
    ("altman_z_quality",          "3.0",   "Altman Z threshold for quality",           "S"),
    ("pe_cheap/fair/expensive",   "15/25/40","P/E valuation thresholds",               "S"),
    ("peg_undervalued/fair",      "1.0/2.0","PEG ratio thresholds",                   "S"),
    ("roe_strong/moderate",       "20%/10%","ROE quality thresholds",                  "S"),
    ("quality_value_blend",       "0.65",  "65% quality 35% value in probability",    "S"),
    ("yield_curve_positive",      "0.50%", "Yield curve threshold (positive)",         "S"),
    ("yield_curve_inverted",      "-0.50%","Yield curve threshold (inverted)",         "S"),
    ("fed_rate_easy/tight",       "2%/4%", "Fed rate regime thresholds",              "S"),
    ("recession_prob_cutoff",     "0.6",   "Recession probability trigger",            "S"),
    ("cpi_target/elevated",       "2.5%/5%","CPI thresholds",                         "S"),
    ("pce_target/elevated",       "2%/3.5%","PCE thresholds",                         "S"),
    ("pmi_expansion/strong/weak", "50/55/48","ISM PMI levels",                         "S"),
    ("claims_low/high",           "220K/350K","Initial claims thresholds",             "S"),
    ("sofr_critical_spread",      "50bps", "Repo stress threshold",                   "S"),
    ("bond_equity_stress_corr",   "0.3",   "Bond-equity stress correlation",           "S"),
    ("amihud_stress_threshold",   "2.0",   "Illiquidity stress multiple",              "S"),
    ("black_swan_sigma",          "5.0",   "Black swan detection sigma",               "S"),
    ("flash_crash_ticker_pct",    "8%",    "Flash crash single stock threshold",       "S"),
    ("geo_shock_vix/gold/oil",    "35/2%/5%","Geo shock trigger thresholds",          "S"),
    ("geo_shock_multiplier",      "0.35",  "Position size cap during geo shock",       "S"),
    ("carry_unwind_usdjpy",       "125.0", "JPY carry unwind trigger",                 "S"),
    ("usdjpy_strong/weak",        "140/120","JPY regime thresholds",                   "S"),
    ("usdcny_stress",             "7.3",   "CNY stress threshold",                    "S"),
    ("neutral_probability",       "0.5",   "Bayesian prior (starting point)",         "S"),
    ("long_threshold",            "0.53",  "Probability to trigger LONG signal",       "S"),
    ("short_threshold",           "0.47",  "Probability to trigger SHORT signal",      "S"),
    ("bayesian_sensitivity",      "1.8",   "Signal amplification factor",              "S"),
    ("news_decay_halflife_days",  "4.6",   "News sentiment half-life",                 "S"),
    ("transaction_cost_bps",      "5",     "Simulated transaction cost",               "S"),
    ("short_interest_high",       "20%",   "High short interest threshold",            "S"),
    ("risk_free_rate",            "5%",    "Risk-free rate for calculations",          "S"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Build PDF
# ─────────────────────────────────────────────────────────────────────────────

def build():
    pdf = PDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.set_margins(10, 16, 10)

    # ── Cover ──────────────────────────────────────────────────────────────
    pdf.cover()

    # ── Helper to write an agent section ──────────────────────────────────
    def agent_section(num, name, count, static, dynamic, theories, intro, factors, col_widths, headers):
        pdf.add_page()
        pdf.set_fill_color(15, 23, 42)
        pdf.rect(0, 14, 210, 16, "F")
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_xy(10, 17)
        pdf.cell(100, 8, f"Agent {num}   -   {name}")
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(148, 163, 184)
        pdf.set_xy(120, 17)
        pdf.cell(0, 8, f"{count} factors  |  {static} static  |  {dynamic} dynamic  |  {theories}")
        pdf.set_text_color(0, 0, 0)
        pdf.set_y(33)
        pdf.section_intro(intro)
        pdf.factor_table(headers, factors, col_widths)

    # ── Agent sections ─────────────────────────────────────────────────────
    agent_section(
        1, "Technical Agent", 40, 22, 18,
        "Price Action * TDA * HMM * Options Greeks",
        "Covers 40 factors across classical technical analysis, options theory, topological data analysis, "
        "Hidden Markov regimes, calendar seasonality, and cross-sectional momentum. "
        "22 parameters are static (fixed academic formulas); 18 are dynamically computed from market data.",
        TECHNICAL_FACTORS,
        [48, 28, 88, 20],
        ["Factor / Indicator", "Theory", "Parameters", "Type"],
    )

    agent_section(
        2, "Fundamental Agent", 42, 28, 14,
        "Fama-French 5F * Piotroski * Altman * DCF * CAPM",
        "Covers 42 factors across valuation multiples, accounting quality (Piotroski F-Score, Beneish M-Score, "
        "Altman Z-Score), growth metrics, Fama-French factor exposures, CAPM alpha, and LLM earnings-call analysis. "
        "28 parameters use fixed academic thresholds; 14 are derived from live market/fundamental data.",
        FUNDAMENTAL_FACTORS,
        [48, 28, 88, 20],
        ["Factor / Indicator", "Theory", "Parameters", "Type"],
    )

    agent_section(
        3, "Sentiment Agent", 24, 1, 23,
        "NLP/LLM * Behavioural Finance * Options Flow * Social Media",
        "Covers 24 factors including Gemini RAG news sentiment, Reddit/social analysis, options flow (unusual "
        "activity, put/call skew), AAII contrarian survey, Fear & Greed, short-squeeze scoring, Transfer Entropy, "
        "and exponential news-decay modelling. 23 of 24 factors are fully dynamic.",
        SENTIMENT_FACTORS,
        [48, 28, 88, 20],
        ["Factor / Indicator", "Theory", "Parameters", "Type"],
    )

    agent_section(
        4, "Macro Agent", 36, 0, 36,
        "FRED Data * Yield Curve * Credit Cycle * Global Liquidity",
        "All 36 factors are dynamically sourced  -  primarily FRED economic series (CPI, PCE, TIPS, PMI, claims, "
        "M2, retail sales, LEI) plus market proxies (VIX, credit spreads, copper/gold, global ETFs). "
        "The most dynamic agent  -  zero static parameters.",
        MACRO_FACTORS,
        [45, 28, 88, 23],
        ["Factor / Indicator", "Theory", "Parameters", "Type"],
    )

    agent_section(
        5, "Risk Agent", 21, 8, 13,
        "EVT * GARCH * Hawkes * Monte Carlo * Kelly Criterion",
        "Covers 21 risk factors: GARCH(1,1) volatility regimes, Extreme Value Theory (GPD) tail risk, "
        "Monte Carlo + Quasi-MC VaR, Hawkes jump process, Kelly sizing, black-swan detection, "
        "and override circuits for geo-shocks and carry unwinds.",
        RISK_FACTORS,
        [48, 28, 88, 20],
        ["Factor / Indicator", "Theory", "Parameters", "Type"],
    )

    agent_section(
        6, "Currency Agent", 12, 6, 6,
        "FX Carry * PPP * EM Risk * Petro-currencies",
        "Covers 12 FX-driven factors: DXY regime with sector-weighted FX exposure, major pairs "
        "(EUR/USD, USD/JPY, USD/CNY, GBP/USD), carry-trade attractiveness, petro-currency signals "
        "(CAD/AUD), and EM currency stress via EEM proxy.",
        CURRENCY_FACTORS,
        [48, 28, 88, 20],
        ["Factor / Indicator", "Theory", "Parameters", "Type"],
    )

    agent_section(
        7, "Geopolitical Agent", 22, 2, 20,
        "Risk Premium * Safe-Haven Flows * Commodity Shock * NLP Text Mining",
        "Covers 22 geopolitical factors: commodity/safe-haven flows (oil, gold, copper), defense/EM stress, "
        "credit contagion, shipping demand, election cycle phase, GPR Index (FRED), and NLP keyword-scanning "
        "for active conflicts, sanctions, and tariff/regulatory risks.",
        GEO_FACTORS,
        [48, 28, 88, 20],
        ["Factor / Indicator", "Theory", "Parameters", "Type"],
    )

    agent_section(
        8, "Insider Agent", 17, 0, 17,
        "EDGAR 4/13F/13D * Kyle Lambda * Dark Pool * Congressional Trading",
        "All 17 factors are dynamic: EDGAR Form 4 cluster buys, 13F institutional change QoQ, "
        "activist 13D/G filings, Kyle's Lambda (price impact / microstructure), Congressional trading "
        "signal, FINRA dark pool print ratio, short-squeeze scoring, and ETF flow impact.",
        INSIDER_FACTORS,
        [50, 28, 86, 20],
        ["Factor / Indicator", "Theory", "Parameters", "Type"],
    )

    agent_section(
        9, "Volatility Agent", 9, 0, 9,
        "GARCH * Heston * Stochastic Vol * SKEW * VVIX",
        "All 9 volatility factors are dynamic: GARCH regime, VRP (IV vs RV), put/call ratio, "
        "Kalman dynamic beta, 60d SPY correlation, VVIX (vol-of-vol), VIX term structure "
        "(contango/backwardation), realised vol trend ratio, and SKEW index.",
        VOLATILITY_FACTORS,
        [52, 28, 84, 20],
        ["Factor / Indicator", "Theory", "Parameters", "Type"],
    )

    # ── Quant Engine modules ───────────────────────────────────────────────
    pdf.add_page()
    pdf.h1("Quant Engine Modules  (23 modules)")
    pdf.section_intro(
        "Shared infrastructure powering all agents. Each module is a standalone quantitative library "
        "callable by any agent or the orchestrator."
    )

    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_fill_color(30, 41, 59)
    pdf.set_text_color(255, 255, 255)
    for w, h in zip([50, 52, 88], ["Module (quant_engine/)", "Theory / Model", "Key Parameters"]):
        pdf.cell(w, 6.5, h, fill=True, align="C", border=0)
    pdf.ln()
    pdf.set_font("Helvetica", "", 7)
    for i, (mod, theory, params) in enumerate(QUANT_MODULES):
        pdf.set_fill_color(245, 247, 250) if i % 2 == 0 else pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(30, 41, 59)
        for w, val in zip([50, 52, 88], [mod, theory, params]):
            pdf.cell(w, 5.8, val, fill=True, align="L", border=0)
        pdf.ln()
    pdf.set_text_color(0, 0, 0)

    # ── Config parameters page ─────────────────────────────────────────────
    pdf.add_page()
    pdf.h1("settings.yaml  -  All Configurable Parameters  (50 entries)")
    pdf.section_intro(
        "All numeric thresholds, windows, and multipliers defined in config/settings.yaml. "
        "These are loaded at startup and override hardcoded defaults. "
        "All are STATIC (set once at deployment) unless overridden via the /api/v1/settings endpoint."
    )
    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_fill_color(30, 41, 59)
    pdf.set_text_color(255, 255, 255)
    for w, h in zip([55, 22, 90, 17], ["Parameter Key", "Value", "Purpose", "Type"]):
        pdf.cell(w, 6.5, h, fill=True, align="C", border=0)
    pdf.ln()
    pdf.set_font("Helvetica", "", 7)
    for i, (key, val, purpose, ds) in enumerate(CONFIG_PARAMS):
        if pdf.get_y() > 272:
            pdf.add_page()
        pdf.set_fill_color(245, 247, 250) if i % 2 == 0 else pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(30, 41, 59)
        for w, v in zip([55, 22, 90], [key, val, purpose]):
            pdf.cell(w, 5.8, v, fill=True, align="L", border=0)
        is_d = ds == "D"
        if is_d:
            pdf.set_fill_color(220, 252, 231)
            pdf.set_text_color(21, 128, 61)
        else:
            pdf.set_fill_color(219, 234, 254)
            pdf.set_text_color(29, 78, 216)
        pdf.set_font("Helvetica", "B", 7)
        pdf.cell(17, 5.8, "DYNAMIC" if is_d else "STATIC", fill=True, align="C", border=0)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(30, 41, 59)
        pdf.ln()
    pdf.set_text_color(0, 0, 0)

    # ── Final summary page ─────────────────────────────────────────────────
    pdf.add_page()
    pdf.h1("Grand Total Summary")

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(30, 41, 59)
    pdf.set_text_color(255, 255, 255)
    for w, h in zip([8, 35, 20, 20, 22, 85], ["#", "Agent", "Factors", "Static", "Dynamic", "Key Quant Theories"]):
        pdf.cell(w, 7, h, fill=True, align="C", border=0)
    pdf.ln()

    rows = [
        ("1", "Technical",    "40", "22", "18", "Price Action, TDA, HMM, Options Greeks, Fibonacci"),
        ("2", "Fundamental",  "42", "28", "14", "Fama-French 5F, Piotroski F-Score, Altman Z, CAPM, DCF"),
        ("3", "Sentiment",    "24",  "1", "23", "NLP/LLM, Behavioural Finance, Options Flow, Reddit"),
        ("4", "Macro",        "36",  "0", "36", "FRED Macro, Yield Curve, Credit Cycle, Global Liquidity"),
        ("5", "Risk",         "21",  "8", "13", "EVT, GARCH, Hawkes Process, Monte Carlo, Kelly Criterion"),
        ("6", "Currency",     "12",  "6",  "6", "FX Carry, PPP, EM Risk, Petro-currencies"),
        ("7", "Geopolitical", "22",  "2", "20", "Risk Premium, Safe-Haven Flows, Commodity Shock, NLP"),
        ("8", "Insider",      "17",  "0", "17", "EDGAR, Kyle Lambda, Dark Pool, Congressional Trading"),
        ("9", "Volatility",    "9",  "0",  "9", "GARCH, Stochastic Vol, Heston, SKEW, VVIX"),
    ]
    pdf.set_font("Helvetica", "", 7.5)
    for i, row in enumerate(rows):
        pdf.set_fill_color(245, 247, 250) if i % 2 == 0 else pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(30, 41, 59)
        for w, val in zip([8, 35, 20, 20, 22, 85], row):
            pdf.cell(w, 6.5, val, fill=True, align="C" if w <= 22 else "L", border=0)
        pdf.ln()

    pdf.set_fill_color(15, 23, 42)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 8)
    for w, val in zip([8, 35, 20, 20, 22, 85],
                      ["", "TOTAL", "223", "67", "156", "30% static  |  70% dynamic  |  23 quant modules"]):
        pdf.cell(w, 7.5, val, fill=True, align="C" if w <= 22 else "L", border=0)
    pdf.ln()
    pdf.set_text_color(0, 0, 0)

    # Static vs Dynamic breakdown
    pdf.ln(8)
    pdf.h2("Static vs Dynamic Parameter Classification")
    pdf.ln(2)

    legend = [
        ("STATIC (67 params / 30%)",
         "Fixed academic formulas and configurable thresholds. Examples: RSI window=14, "
         "MACD(12/26/9), Fibonacci levels (23.6/38.2/50/61.8/78.6%), Piotroski/Altman weights, "
         "Bollinger std=2.0, Parabolic SAR AF=0.02, Kelly cap=25%, Flash crash trigger=8%. "
         "Changed only via settings.yaml or /api/v1/settings."),
        ("DYNAMIC (156 params / 70%)",
         "Computed fresh for every signal from live data sources: FRED macro series, "
         "EDGAR filings (Form 4/13F/8-K/13D), yfinance OHLCV + options chains, "
         "GARCH-fitted parameters (alpha/beta/omega), LLM/Gemini outputs (news sentiment, "
         "earnings-call NLP), Reddit/social counts, congressional trading data, "
         "dark pool prints, options Greeks (vanna/charm/GEX), rolling correlations, "
         "distribution moments (skew/kurtosis), and all momentum/return computations."),
    ]
    for title, body in legend:
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(30, 64, 175)
        pdf.cell(0, 6, title, ln=True)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(51, 65, 85)
        pdf.multi_cell(0, 5, body)
        pdf.ln(3)

    pdf.set_text_color(0, 0, 0)

    # ── Save ───────────────────────────────────────────────────────────────
    pdf.output(OUT)
    print(f"PDF saved → {OUT}")


if __name__ == "__main__":
    build()
