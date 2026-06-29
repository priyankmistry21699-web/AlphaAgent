# AlphaAgent — Complete Factor & Theory Inventory

> Definitive enumeration of **every single factor, every theory, every formula, and every data source** implemented in AlphaAgent. Extracted directly from source code.

**Last extracted:** 2026-06-16 from `agents/*.py` + `quant_engine/*.py`
**Total verified factors:** 271
**Total quant engine modules:** 37
**Total agents:** 9

---

# PART I — FACTORS BY AGENT (Every Single One)

## 1. Technical Agent (52 factors) — `agents/technical.py`

### A. Classical Indicators (12)
| # | Factor | Theory / Reference |
|---|--------|---------------------|
| 1 | RSI (14) | Wilder (1978) |
| 2 | MACD Histogram | Appel (1979) |
| 3 | ADX Trend | Wilder (1978) |
| 4 | Golden / Death Cross | SMA50 vs SMA200 |
| 5 | Bollinger %B | Bollinger (1980); **VIX-adaptive std 1.75–2.5σ** |
| 6 | Bollinger Bandwidth | Squeeze detection |
| 7 | OBV Trend | On-Balance Volume — Granville (1963) |
| 8 | VWAP Position | Volume-Weighted Avg Price |
| 9 | Stochastic %K | Lane (1950s) |
| 10 | Williams %R (14) | Williams (1973) |
| 11 | EMA 9/21 Crossover | Short-term trend |
| 12 | Parabolic SAR | Wilder (1978) |

### B. Trend / Position (5)
| # | Factor | Theory |
|---|--------|--------|
| 13 | Price vs EMA-200 | Long-term trend gap |
| 14 | 52-Week Range Position | Where price sits in 52W |
| 15 | 52-Week High Proximity | Breakout signal |
| 16 | Ichimoku Cloud | Hosoda (1969) — 9/26/52 system |
| 17 | Fibonacci Retracement | 0.236/0.382/0.5/0.618/0.786 |

### C. Momentum & Volume (8)
| # | Factor | Theory |
|---|--------|--------|
| 18 | ATR-Normalized Momentum | 20d move / 20×ATR |
| 19 | Volume Surge Ratio | Today vol / 20d avg |
| 20 | Momentum Acceleration | 2nd derivative of momentum |
| 21 | Hurst Exponent | Long-range memory — Hurst (1951) |
| 22 | 12M-1M Momentum | Jegadeesh-Titman (1993) |
| 23 | 5-Day Price Momentum | Short-term factor |
| 24 | Sector ETF Momentum | Relative strength |
| 25 | Chaikin Money Flow (CMF-14) | Chaikin (1980) |
| 26 | TRIX Oscillator (15-period) | Hutson (1983) |

### D. Behavioural Finance Anomalies (7)
| # | Factor | Theory / Reference |
|---|--------|---------------------|
| 27 | Idiosyncratic Volatility (Ang) | Ang et al. (2006) — high idio vol underperforms |
| 28 | MAX Anomaly (Bali) | Bali et al. (2011) — lottery demand |
| 29 | 1-Week Short-Run Reversal | Jegadeesh (1990) |
| 30 | 3-Year Long-Run Reversal | DeBondt-Thaler (1985) |
| 31 | 52W High Momentum (George-Hwang) | George & Hwang (2004) |
| 32 | Overnight vs Intraday Return | Decomposition of autocorrelation |
| 33 | Momentum Crash Risk (Daniel-Moskowitz) | Daniel & Moskowitz (2016) |

### E. Microstructure (4)
| # | Factor | Theory |
|---|--------|--------|
| 34 | Order Flow Imbalance (CLV) | Close Location Value + Money Flow Volume |
| 35 | VPIN Microstructure | Easley-López-O'Hara (2012) |
| 36 | Volume Profile (POC) | Steidlmayer Market Profile |
| 37 | Supply / Demand Zones | Local high-vol extrema |

### F. Statistical / Regime (5)
| # | Factor | Theory |
|---|--------|--------|
| 38 | HMM Market Regime | Gaussian HMM — Baum-Welch |
| 39 | P(Regime Change) - HMM | Transition matrix risk |
| 40 | TDA Persistent Homology | Carlsson (2009) — H0/H1 barcodes |
| 41 | Adaptive RSI (VIX-adjusted) | Dynamic threshold |
| 42 | Signal Strength vs ATR | Signal/noise ratio |

### G. Options Layer (4)
| # | Factor | Theory |
|---|--------|--------|
| 43 | IV Skew (25Δ Put-Call) | Tail-hedging demand |
| 44 | Gamma Exposure (GEX) | Dealer hedging flow |
| 45 | Options Max Pain | Strike with max OI loss |
| 46 | Implied Correlation (SPY/Stock IV) | Systemic vs idiosyncratic |
| 47 | 0DTE Options Flow | Zero-day-to-expiry analytics |
| 48 | Variance Risk Premium | IV − RV component |

### H. Cross-Sectional (3)
| # | Factor | Theory |
|---|--------|--------|
| 49 | Multi-Timeframe Confluence (D/W) | Cross-timeframe consensus |
| 50 | Cross-Sectional Rank vs Peers | Relative momentum vs sector |
| 51 | Signal Backtest Calibration | Conditional hit-rate |

### I. Pattern Recognition (3)
| # | Factor | Theory |
|---|--------|--------|
| 52 | Chart Pattern Recognition | Double Top/Bottom (5-bar) |
| 53 | Pivot Points (Daily/Weekly) | R2/R1/PP/S1/S2 levels |

### J. Free Data Integrations (NEW, 5)
| # | Factor | Source |
|---|--------|--------|
| 54 | Commodity Roll Yield | ETF tracking error vs spot futures |
| 55 | COT Commercials Net | CFTC weekly Commitment of Traders |
| 56 | Weather Anomaly (US) | NOAA HDD/CDD anomaly |
| 57 | EIA Inventory | Petroleum inventory proxy |
| 58 | Google Trends Attention | pytrends retail search volume |
| 59 | ETF Premium/Discount to NAV | NAV-proxy z-score mean-reversion |

### K. Meta (1)
| 60 | PCA Signal Quality | Consensus mean/strength/dispersion |

---

## 2. Fundamental Agent (52 factors) — `agents/fundamental.py`

### A. Quality / Bankruptcy (5)
| # | Factor | Author / Year |
|---|--------|---------------|
| 1 | Piotroski F-Score | Piotroski (2000) — 9-point system |
| 2 | Altman Z-Score | Altman (1968) bankruptcy predictor |
| 3 | Beneish M-Score | Beneish (1999) earnings manipulation |
| 4 | Accruals Ratio | Sloan (1996) earnings quality |
| 5 | Earnings Consistency (CoV) | Quarterly EPS variation |

### B. Valuation (10)
| # | Factor | Notes |
|---|--------|-------|
| 6 | P/E Ratio (Trailing) | Dynamic thresholds based on 10Y yield |
| 7 | Forward P/E | Forward EPS-based |
| 8 | P/E vs Sector Median | Relative valuation |
| 9 | P/E vs 5-Year Average | Historical context |
| 10 | Price/Book Ratio | P/B |
| 11 | PEG Ratio | P/E ÷ growth |
| 12 | EV/EBITDA | Enterprise multiple |
| 13 | Price/Sales Ratio | P/S |
| 14 | FCF Yield | Free cash flow / market cap |
| 15 | DCF Implied Upside | **Dynamic WACC = rf + β × ERP** |
| 16 | Graham Number (Intrinsic Floor) | sqrt(22.5 × EPS × BVPS) |

### C. Profitability (5)
| # | Factor | Notes |
|---|--------|-------|
| 17 | Gross Margin | Revenue − COGS / Revenue |
| 18 | Return on Equity (ROE) | NI / Equity |
| 19 | Return on Assets (ROA) | NI / Assets |
| 20 | Operating Margin | EBIT / Revenue |
| 21 | Net Margin Trend | Quarter-over-quarter |
| 22 | FCF Quality (FCF / Net Income) | Cash quality |

### D. Leverage / Solvency (5)
| # | Factor | Notes |
|---|--------|-------|
| 23 | Debt-to-Equity | D/E |
| 24 | Net Debt / EBITDA | Leverage ratio |
| 25 | Current Ratio | Liquidity |
| 26 | Interest Coverage Ratio | EBIT / interest |
| 27 | Asset Turnover | Revenue / Assets |

### E. Growth & Earnings (5)
| # | Factor | Notes |
|---|--------|-------|
| 28 | Revenue Growth (YoY) | Year-over-year |
| 29 | Earnings Growth (YoY) | EPS YoY |
| 30 | EPS Surprise / Growth | Beat/miss magnitude |
| 31 | Earnings Revision Momentum | EPS estimate change |
| 32 | Analyst Revision Momentum | Bull % change Q/Q |

### F. Factor Zoo Anomalies (6)
| # | Factor | Author / Year |
|---|--------|---------------|
| 33 | Asset Growth Anomaly | Cooper-Gulen-Schill (2008) |
| 34 | Gross Profitability (Novy-Marx) | Novy-Marx (2013) — (Rev−COGS)/Assets |
| 35 | Investment-to-Assets (q-factor) | Hou-Xue-Zhang (2015) |
| 36 | Net Stock Issuance | Daniel & Titman dilution |
| 37 | R&D Anomaly (Chan) | Chan et al. — R&D / Market Cap |
| 38 | QMJ Composite (AQR) | Asness-Frazzini-Pedersen (2019) |

### G. Fama-French Loadings (5)
| # | Factor | Reference |
|---|--------|-----------|
| 39 | Fama-French: Value (HML) | Fama-French (1993) |
| 40 | Fama-French: Size (SMB) | Market cap factor |
| 41 | Fama-French: Quality (RMW) | Profitability factor |
| 42 | Fama-French: Low-Vol (BAB) | Frazzini-Pedersen Betting Against Beta |
| 43 | CAPM Jensen's Alpha (1Y) | Jensen (1968) |

### H. Dividends & Buybacks (3)
| # | Factor | Notes |
|---|--------|-------|
| 44 | Dividend Cut Probability | Payout ratio + FCF check |
| 45 | Shares Outstanding Trend | Buyback / dilution proxy |
| 46 | Dividend Yield & Payout | Income + sustainability |

### I. Event-Driven (4)
| # | Factor | Notes |
|---|--------|-------|
| 47 | Lockup Expiration Proxy | IPO 180-day window |
| 48 | CEO/CFO Departure (8-K Signal) | EDGAR 8-K parse |
| 49 | M&A Activity (EDGAR 8-K) | Merger/acquisition filings |
| 50 | Earnings Proximity | Days to next earnings |

### J. NLP / Anomalies (3)
| # | Factor | Notes |
|---|--------|-------|
| 51 | Earnings Call NLP (Gemini) | LLM tone analysis |
| 52 | PEAD Drift (Post-Earnings Anomaly) | Bernard-Thomas (1989) |
| 53 | SEC 10-K Language Shift | Cosine similarity language regime |

### K. Meta (1)
| 54 | PCA Signal Quality | Consensus indicator |

---

## 3. Macro Agent (44 factors) — `agents/macro.py`

### A. Rates & Yield Curve (6)
| # | Factor | Source |
|---|--------|--------|
| 1 | Recession Probability | FRED composite |
| 2 | Yield Curve (10Y-2Y) | DGS10 − DGS2 |
| 3 | Nelson-Siegel Yield Curve | Level/Slope/Curvature factors |
| 4 | Fed Funds Rate | DFEDTAR |
| 5 | Fed Rate Change Direction | 3-month delta |
| 6 | Real Interest Rate (10Y) | DGS10 − T10YIE |

### B. Inflation (4)
| # | Factor | FRED Series |
|---|--------|-------------|
| 7 | CPI Inflation (YoY) | CPIAUCSL |
| 8 | PCE Inflation (YoY) | PCEPI |
| 9 | 10Y TIPS Breakeven Inflation | T10YIE |
| 10 | Equity Risk Premium (E/P − Real Rate) | Earnings yield vs real rate |

### C. Volatility & Fear (5)
| # | Factor | Source |
|---|--------|--------|
| 11 | VIX Fear Index | ^VIX |
| 12 | VIX 5-Day Change | Recent momentum |
| 13 | VIX Term Structure (VIX vs VIX3M) | ^VIX vs ^VIX3M |
| 14 | MOVE Index (Bond Vol) | ^MOVE |
| 15 | Cross-Sectional Momentum Dispersion | Sector return dispersion |

### D. Credit & Funding (5)
| # | Factor | Source |
|---|--------|--------|
| 16 | Credit Spreads (HYG/LQD) | ETF proxy |
| 17 | HY Credit Spread (OAS) | FRED OAS |
| 18 | Repo Market Stress (SOFR Spread) | SOFR − Fed Funds |
| 19 | TED Spread (Funding Stress) | T-bill vs SOFR |
| 20 | Financial Conditions Index (NFCI) | Chicago Fed FCI |

### E. Cross-Asset (6)
| # | Factor | Notes |
|---|--------|-------|
| 21 | Copper/Gold Ratio (Growth Signal) | Industrial vs defensive |
| 22 | BTC Risk-On Signal | Crypto sentiment |
| 23 | Global Equity Breadth (ACWI vs SPY) | Global RS |
| 24 | Bond-Equity Correlation (TLT/SPY) | Systemic stress signal |
| 25 | Dollar Index Regime (DXY) | DXY momentum |
| 26 | Global Pre-Market Signal (Asia/EU) | EWJ/EWG/EWU/EWQ avg |

### F. Money Supply & Liquidity (4)
| # | Factor | FRED |
|---|--------|------|
| 27 | M2 Money Supply Growth | M2SL |
| 28 | Fed Balance Sheet (WALCL) | QE/QT signal |
| 29 | Amihud Illiquidity Ratio (SPY) | Microstructure stress |
| 30 | Contagion Correlation Spike | Cross-asset correlation delta |

### G. Real Economy (6)
| # | Factor | FRED |
|---|--------|------|
| 31 | ISM Manufacturing PMI | NAPM |
| 32 | UMich Consumer Sentiment | UMCSENT |
| 33 | Initial Jobless Claims | ICSA |
| 34 | Retail Sales MoM (ex-Auto) | RSXFS |
| 35 | Leading Economic Index (LEI) | USSLIND |
| 36 | Business Cycle Phase | Recovery/Expansion/Slowdown/Contraction composite |

### H. Equity Style Rotation (4)
| # | Factor | Notes |
|---|--------|-------|
| 37 | Semiconductor Index (SOX vs SPY) | Tech leading indicator |
| 38 | Growth vs Value Rotation | IWF vs IWD |
| 39 | SPY vs SMA(200) | Bull/bear regime |
| 40 | Large vs Small Cap (SPY/IWM) | Size rotation |

### I. Sector Rotation (1)
| 41 | Sector RS | Sector ETF vs SPY |

### J. Composite Nowcast (1, NEW)
| 42 | FRED Macro Nowcast | 6-component GDPNow-style composite |

### K. Meta (1)
| 43 | PCA Signal Quality | |

---

## 4. Sentiment Agent (25 factors) — `agents/sentiment.py`

### A. NLP / News (5)
| # | Factor | Source |
|---|--------|--------|
| 1 | News Sentiment (RAG) | Gemini 2.5 Flash + 8 articles |
| 2 | FinBERT-Style NLP (Gemini) | LLM classification |
| 3 | News Decay Model (Weighted) | **VIX-adaptive halflife 2–6d** |
| 4 | Headline/Body Alignment | Gemini dual-pass |
| 5 | Source Credibility Weight | WSJ 1.0 → Investopedia 0.5 |

### B. Retail / Crowd (4)
| # | Factor | Source |
|---|--------|--------|
| 6 | Reddit Sentiment (WSB+Stocks+Investing) | PRAW |
| 7 | News / Social Momentum | 3-day velocity |
| 8 | Sentiment Momentum (3-Day Trend) | (0-3d) vs (3-6d) delta |
| 9 | Fear & Greed Index | Alternative data |

### C. Positioning (4)
| # | Factor | Source |
|---|--------|--------|
| 10 | Short Interest | yfinance.info |
| 11 | Short Interest Change (MoM) | Trend |
| 12 | Short-Squeeze Score | SI% + days to cover |
| 13 | AAII Sentiment Survey | Bull − Bear spread |

### D. Options Sentiment (3)
| # | Factor | Notes |
|---|--------|-------|
| 14 | Options Put/Call Skew | Put vol / Call vol |
| 15 | Unusual Options Activity | Vol/OI > 2x threshold |
| 16 | Earnings Whisper (Implied vs Historical Move) | ATM straddle / hist |

### E. Analyst (4)
| # | Factor | Notes |
|---|--------|-------|
| 17 | Analyst Consensus | % buy/strong-buy |
| 18 | EPS Forward Revision | Forward EPS change |
| 19 | Analyst Price Target Upside | (Target − Price) / Price |
| 20 | Analyst Revision Direction | Bull % change Q/Q |
| 21 | Earnings Revision Momentum | EPS estimate change |

### F. Information Theory (2)
| # | Factor | Theory |
|---|--------|--------|
| 22 | Transfer Entropy (News→Price) | Causality lag-1 |
| 23 | Signal Entropy (Clarity Index) | Shannon entropy of factors |

### G. Macro Sentiment (1)
| 24 | Consumer Credit (Margin Proxy) | Leverage signal |

### H. Composite (1)
| 25 | Market Breadth (RSP vs SPY) | Equal-weight vs cap-weight |

---

## 5. Risk Agent (26 factors) — `agents/risk.py` (acts as circuit breaker)

### A. Volatility Regimes (3)
| # | Factor | Theory |
|---|--------|--------|
| 1 | GARCH Volatility Regime | Bollerslev (1986) GARCH(1,1) |
| 2 | Yang-Zhang Efficient Vol (OHLC) | Yang-Zhang (2000) — 7.4× efficiency |
| 3 | Rolling Sharpe / Sortino (63d) | Risk-adjusted performance |

### B. Tail Risk (5)
| # | Factor | Theory |
|---|--------|--------|
| 4 | EVT Tail Risk (99% VaR) | Pickands-Balkema-de Haan GPD |
| 5 | Monte Carlo 95% CI (5d) | GBM + GARCH-driven drift |
| 6 | Quasi-MC VaR (Sobol) | Low-discrepancy sequences (~10× efficient) |
| 7 | Return Skewness & Kurtosis | 3rd + 4th moments |
| 8 | Tail Ratio (Up/Down Asymmetry) | 95th / |5th percentile| |

### C. Circuit Breakers (4)
| # | Factor | Trigger |
|---|--------|---------|
| 9 | Black Swan Detection | \|Z\| > 5σ in 5d |
| 10 | Flash Crash Detection | **3 × daily vol** (vol-normalised) |
| 11 | Geopolitical Shock Signal | VIX>35 + gold/oil surge |
| 12 | Carry Trade Unwind | USD/JPY<125 + JPY surge |

### D. Stochastic Processes (2)
| # | Factor | Theory |
|---|--------|--------|
| 13 | Hawkes Branching Ratio (Jump Cascade) | Hawkes (1971) self-exciting |
| 14 | KL Divergence (Regime Shift) | Recent 20d vs baseline 252d |

### E. Sizing (1)
| 15 | Kelly Position Size | Kelly (1956), Half-Kelly capped 25% |

### F. Correlation (3, NEW DCC + Network) |
| # | Factor | Theory |
|---|--------|--------|
| 16 | DCC-GARCH Dynamic Correlation | Engle (2002) — RMT-cleaned |
| 17 | Correlation Network Centrality | Eigenvector centrality vs peer ETFs |
| 18 | Correlation Regime (vs SPY) | 60d vs 252d correlation delta |

### G. Options Risk (2)
| # | Factor | Theory |
|---|--------|--------|
| 19 | Vanna/Charm Exposure | Higher-order Greeks |
| 20 | Options GEX Wall | Gamma net long/short |

### H. Microstructure / Liquidity (2)
| # | Factor | Notes |
|---|--------|-------|
| 21 | Liquidity Risk (Vol Ratio) | Today vol / 20D avg |
| 22 | Drawdown from ATH | 2Y high vs current |

### I. Online Monitoring (1, NEW)
| 23 | CUSUM Structural Break | Page (1954) parameter drift |

### J. Credit (1)
| 24 | CDS Spread Proxy (HYG/LQD) | Credit risk proxy |

---

## 6. Insider Agent (19 factors) — `agents/insider.py`

| # | Factor | Source |
|---|--------|--------|
| 1 | Insider Sentiment | SEC Form 4 net buys |
| 2 | EDGAR Form 4 Activity | Filings count |
| 3 | Material Events (8-K) | Corporate actions |
| 4 | Insider Cluster Buys | Multi-insider same window |
| 5 | Insider Cluster (30-Day) | 30d clustering |
| 6 | Institutional Ownership | yfinance |
| 7 | Institutional Ownership % | Share % held |
| 8 | Institutional Ownership (EDGAR) | EDGAR cross-check |
| 9 | 13F Institutional Change (QoQ) | Quarterly delta |
| 10 | 13F Institutional Ownership | 13F snapshot |
| 11 | Top-10 Holder Concentration (HHI) | Herfindahl index |
| 12 | Activist 13D/G Filing | >5% stake disclosure |
| 13 | Congressional Trading Signal | Capitol Trades-style |
| 14 | Short Squeeze Potential | SI + days-to-cover combo |
| 15 | Float Reduction (Buyback/Dilution) | Float trend |
| 16 | FINRA Short Sale Volume % | Short volume % |
| 17 | Dark Pool Print Ratio (FINRA ADF) | Dark pool % |
| 18 | Dark Pool Proxy (Volume vs Avg) | Volume anomaly |
| 19 | Kyle's Lambda (Price Impact) | Kyle (1985) market impact |

---

## 7. Geopolitical Agent (20 factors) — `agents/geopolitical.py`

### A. Macro Stress (5)
| 1 | Oil Shock (XLE vs SPY) |
| 2 | Gold Safe-Haven (GLD) |
| 3 | EM Stress (EEM vs SPY) |
| 4 | Defense RS (ITA vs SPY) |
| 5 | VIX Fear Index |

### B. Cross-Asset (4)
| 6 | Copper/Gold Ratio |
| 7 | Global Breadth (ACWI vs SPY) |
| 8 | USD Safe-Haven (DXY) |
| 9 | Credit Stress (HYG/LQD) |

### C. Sector / Supply Chain (5)
| 10 | Sector Rotation Signal |
| 11 | Supply Chain Stress (Shipping) | Baltic Dry proxy |
| 12 | Commodity Shock (Energy+Materials) |
| 13 | EM Contagion Risk (EMB) |
| 14 | Oil Price (Brent BZ=F) |

### D. Geopolitical Risk Indices (4)
| 15 | Election Cycle Phase |
| 16 | Transport Index (IYT vs SPY) |
| 17 | Commodity Index (PDBC proxy) |
| 18 | Geopolitical Risk Index (FRED GPR) | Caldara-Iacoviello |

### E. Event Detection (3)
| 19 | GPR 30-Day Change |
| 20 | Active Conflict Score |
| 21 | Sanctions Risk |
| 22 | Tariff / Regulatory Risk |

### F. Meta (1)
| 23 | PCA Signal Quality |

---

## 8. Currency Agent (13 factors) — `agents/currency.py`

| # | Factor | Source |
|---|--------|--------|
| 1 | US Dollar Index (DXY) | DXY momentum |
| 2 | EUR/USD Rate | EUR=X |
| 3 | USD/JPY Carry Signal | JPY=X |
| 4 | USD/CNY (China Stress) | CNY=X |
| 5 | GBP/USD Rate | GBP=X |
| 6 | EM Currency Stress | EM FX basket |
| 7 | FX Translation Impact | Foreign revenue exposure |
| 8 | Petro-Currency Signal (CAD/AUD) | Oil-linked FX |
| 9 | Carry Trade (USD Rate Advantage) | Rate differential |
| 10 | Real Interest Rate (10Y) | Real rate driver |
| 11 | EM FX Pressure (EEM Proxy) | EM stress |
| 12 | DXY vs SMA(50) | Trend signal |

---

## 9. Volatility Agent (12 factors) — `agents/volatility.py`

| # | Factor | Theory |
|---|--------|--------|
| 1 | GARCH Vol Regime | Bollerslev (1986) |
| 2 | Put/Call Ratio | Options sentiment |
| 3 | IV vs Realized Vol | Ratio analysis |
| 4 | **Variance Risk Premium (IV²−RV²)** | Formal VRP — NEW |
| 5 | Kalman Dynamic Beta | Kalman (1960) — time-varying β |
| 6 | SPY Correlation (60d) | Systemic linkage |
| 7 | VVIX (Vol of Vol Index) | Vol-of-vol |
| 8 | VIX Term Structure (3M-Spot) | Contango/backwardation |
| 9 | Realized Skewness (60d) | NEW — crash risk |
| 10 | Vol Arbitrage (VRP Trade) | NEW — explicit SHORT_VOL signal |
| 11 | Yang-Zhang Vol (OHLC efficient) | NEW — 7.4× efficiency |
| 12 | CBOE SKEW Index | Tail-risk demand |

---

# PART II — QUANT ENGINE MODULES (All 37)

## A. Volatility & Stochastic Processes (8 modules)

| Module | Author / Theory | Formula / Output |
|--------|-----------------|------------------|
| `garch.py` | Bollerslev (1986) GARCH(1,1) | σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1} |
| `heston.py` | Heston (1993) | dv = κ(θ−v)dt + ξ√v·dW₂ |
| `sabr.py` | Hagan et al. (2002) | SABR vol smile σ(K,F,α,β,ρ,ν) |
| `rough_vol.py` | Bayer-Friz-Gatheral (2016) | σ_t = ξ₀·exp(η·W^H_t), H<0.5 |
| `vol_estimators.py` | Parkinson/Garman-Klass/Rogers-Satchell/Yang-Zhang | Range-based vol (5–8× efficiency) |
| `vol_arbitrage.py` | Variance Risk Premium trading | VRP z-score → SHORT/LONG_VOL signal |
| `dcc_garch.py` | Engle (2002) | Q_t = (1−a−b)Q̄ + a·ε_{t-1}ε'_{t-1} + b·Q_{t-1} |
| `quasi_mc.py` | Sobol (1967) | Low-discrepancy VaR |

## B. Risk / Tail (5 modules)

| Module | Theory | Output |
|--------|--------|--------|
| `evt.py` | Pickands-Balkema-de Haan | GPD VaR₉₉, CVaR₉₉, tail index ξ |
| `monte_carlo.py` | GBM with GARCH drift | 5,000 paths, 68/95/99% CIs |
| `hawkes.py` | Hawkes (1971) | λ(t) = μ + α·Σexp(−β(t−tᵢ)); branching ratio α/β |
| `copula.py` | Gaussian / Clayton / Gumbel | Tail co-dependence λ_L |
| `portfolio_risk.py` | Portfolio-level VaR | VaR/CVaR, marginal/component VaR, stress scenarios |

## C. Regime / State (4 modules)

| Module | Theory | Output |
|--------|--------|--------|
| `hmm.py` | Baum-Welch Gaussian HMM | Bull/Bear/Crisis state probabilities |
| `markov_regime.py` | Markov-switching | Alternative regime model |
| `regime_weights.py` | Regime-conditional weighting | 4×8 agent weight tables + soft blend |
| `structural_break.py` | Page (1954) CUSUM | Online break detection |

## D. Fusion / Decision (4 modules)

| Module | Theory | Output |
|--------|--------|--------|
| `bayesian.py` | Sequential Bayesian log-odds | Posterior P(up) with correlation penalty |
| `meta_learner.py` | LightGBM stacking | Blended prediction |
| `calibration.py` | Platt scaling + isotonic | Calibrated probability |
| `leaderboard.py` | Per-agent IC tracking | Rolling Spearman correlation |

## E. Portfolio Construction (3 modules)

| Module | Theory | Output |
|--------|--------|--------|
| `portfolio_optimizer.py` | Markowitz MVO | Mean-variance weights |
| `black_litterman.py` | Black & Litterman (1992) | π + posterior view-blended returns |
| `hrp.py` | López de Prado (2016) HRP | Cluster-based risk-parity allocation |

## F. Statistical Methods (5 modules)

| Module | Theory | Output |
|--------|--------|--------|
| `rmt.py` | Marchenko-Pastur (1967) | Eigenvalue-clipped covariance matrix |
| `factor_orthogonalization.py` | Gram-Schmidt / PCA / Löwdin | Orthogonal factor decomposition |
| `deflated_sharpe.py` | Bailey-Prado (2014) + BH-FDR (1995) | Adjusted Sharpe + multiple testing |
| `quantile_regression.py` | Koenker-Bassett (1978) | 5/25/50/75/95 quantile predictions |
| `ml_finance.py` | López de Prado (2018) AFML | Fractional Diff + Triple Barrier + Purged CV |

## G. Microstructure / Information Theory (5 modules)

| Module | Theory | Output |
|--------|--------|--------|
| `vpin.py` | Easley-López-O'Hara (2012) | Volume-Synchronized PIN |
| `lob.py` | LOB proxy | Bid-ask + market impact |
| `kalman.py` | Kalman (1960) | Time-varying β + correlation |
| `granger.py` | Granger (1969) | Causality F-test on VAR(p) |
| `causal_engine.py` | Pearl Do-Calculus | Causal DAG inference |

## H. Topology / Geometry (3 modules)

| Module | Theory | Output |
|--------|--------|--------|
| `tda_signal.py` | Carlsson (2009) TDA | H0/H1 persistent homology barcodes |
| `multifractal.py` | Kantelhardt (2002) MF-DFA | Generalized Hurst h(q) |
| `quantum_finance.py` | Quantum-inspired (research) | Amplitude estimation + QAOA |

## I. Cost / Execution (1 module)

| Module | Theory | Output |
|--------|--------|--------|
| `transaction_costs.py` | Almgren-Chriss (2000) | Bid-ask + √impact + commission + borrow |

## J. Signal Lifecycle (4 modules)

| Module | Theory | Output |
|--------|--------|--------|
| `signal_decay.py` | Ornstein-Uhlenbeck | Half-life → optimal hold |
| `kelly.py` | Kelly (1956) | Half-Kelly capped 25% |
| `momentum.py` | Cross-sectional momentum | Factor library |
| `factor_exposure.py` | Fama-French 5-factor | β_market, β_SMB, β_HML, β_RMW, β_CMA |

## K. Domain Signals (8 modules)

| Module | Domain | Output |
|--------|--------|--------|
| `etf_premium.py` | ETF NAV deviation | Mean-reversion signal |
| `commodity_roll_yield.py` | Futures contango/backwardation | Roll yield drag |
| `cot_data.py` | CFTC weekly | Commercials z-score |
| `weather_factor.py` | NOAA temperature | HDD/CDD anomaly |
| `eia_petroleum.py` | EIA inventory proxy | Crude pressure signal |
| `fred_nowcast.py` | FRED composite | GDPNow-style nowcast |
| `google_trends.py` | pytrends | Retail attention |
| `options_intel.py` | Options chain | GEX, max pain, IV skew |
| `zero_dte.py` | Zero-DTE flow | Same-day options analytics |
| `pead.py` | Earnings drift | Post-Earnings Announcement Drift |
| `sec_nlp.py` | SEC 10-K/10-Q | Language similarity |
| `scoring.py` | Fundamental scores | Piotroski/Altman/Beneish |
| `technical.py` | Technical indicators | RSI/MACD/Bollinger (VIX-adaptive) |
| `macro.py` | Macro environment | Recession probability |
| `insider.py` | Form 4 analytics | Cluster trades |

---

# PART III — THEORIES INDEX (Master List by Author)

## Volatility Models
- **Bollerslev (1986)** GARCH(1,1) → `garch.py`
- **Engle (1982)** ARCH → underlying `garch.py`
- **Engle (2002)** DCC-GARCH → `dcc_garch.py`
- **Heston (1993)** Stochastic Volatility → `heston.py`
- **Hagan et al. (2002)** SABR → `sabr.py`
- **Bayer-Friz-Gatheral (2016)** Rough Volatility → `rough_vol.py`
- **Parkinson (1980)** range-based vol → `vol_estimators.py`
- **Garman-Klass (1980)** OHLC vol → `vol_estimators.py`
- **Rogers-Satchell (1991)** drift-unbiased vol → `vol_estimators.py`
- **Yang-Zhang (2000)** best OHLC vol → `vol_estimators.py`

## Risk / Tail Models
- **Pickands-Balkema-de Haan** Generalized Pareto → `evt.py`
- **Sobol (1967)** Quasi-Monte Carlo → `quasi_mc.py`
- **Hawkes (1971)** self-exciting process → `hawkes.py`
- **Kelly (1956)** optimal sizing → `kelly.py`
- **Almgren-Chriss (2000)** market impact → `transaction_costs.py`

## Regime Models
- **Baum-Welch** Gaussian HMM → `hmm.py`
- **Page (1954)** CUSUM detection → `structural_break.py`

## Portfolio Construction
- **Markowitz (1952)** mean-variance → `portfolio_optimizer.py`
- **Black & Litterman (1992)** → `black_litterman.py`
- **López de Prado (2016)** HRP → `hrp.py`

## Statistical Methods
- **Marchenko-Pastur (1967)** eigenvalue distribution → `rmt.py`
- **Bailey & López de Prado (2014)** Deflated Sharpe → `deflated_sharpe.py`
- **Benjamini-Hochberg (1995)** FDR → `deflated_sharpe.py`
- **Bonferroni** family-wise error → `deflated_sharpe.py`
- **Koenker-Bassett (1978)** quantile regression → `quantile_regression.py`
- **López de Prado (2018)** AFML — Fractional Diff, Triple Barrier, Purged CV → `ml_finance.py`

## Information Theory
- **Granger (1969)** causality → `granger.py`
- **Pearl** do-calculus → `causal_engine.py`
- **Kalman (1960)** filtering → `kalman.py`
- **Easley-López-O'Hara (2012)** VPIN → `vpin.py`
- **Kyle (1985)** market impact lambda → `insider.py`

## Topology / Geometry
- **Carlsson (2009)** Topological Data Analysis → `tda_signal.py`
- **Kantelhardt (2002)** MF-DFA → `multifractal.py`
- **Hurst (1951)** long-range memory → technical agent

## Fundamental Anomalies / Quality
- **Piotroski (2000)** F-Score → `scoring.py`
- **Altman (1968)** Z-Score → `scoring.py`
- **Beneish (1999)** M-Score → `scoring.py`
- **Sloan (1996)** accruals anomaly → fundamental agent
- **Bernard-Thomas (1989)** PEAD → `pead.py`

## Fama-French / Factor Models
- **Fama-French (1993, 2015)** 3-factor → 5-factor → `factor_exposure.py`
- **Jensen (1968)** CAPM alpha → fundamental agent
- **Frazzini-Pedersen (2014)** Betting Against Beta → `factor_exposure.py`
- **Novy-Marx (2013)** Gross Profitability → fundamental agent
- **Cooper-Gulen-Schill (2008)** Asset Growth → fundamental agent
- **Hou-Xue-Zhang (2015)** q-factor → fundamental agent
- **Daniel & Titman** Net Issuance → fundamental agent
- **Chan et al.** R&D Anomaly → fundamental agent
- **Asness-Frazzini-Pedersen (2019)** QMJ → fundamental agent

## Behavioural Finance
- **Ang et al. (2006)** Idiosyncratic Vol Anomaly → technical agent
- **Bali et al. (2011)** MAX Anomaly → technical agent
- **Jegadeesh (1990)** 1-week reversal → technical agent
- **DeBondt-Thaler (1985)** Long-Run Reversal → technical agent
- **George-Hwang (2004)** 52-Week High Momentum → technical agent
- **Daniel-Moskowitz (2016)** Momentum Crash → technical agent
- **Jegadeesh-Titman (1993)** 12-1 Momentum → technical agent
- **Hirshleifer-Teoh** Limited Attention → Google Trends signal

## Macro
- **Nelson-Siegel (1987)** yield curve → macro agent
- **Bridgewater** Business Cycle → macro agent
- **Atlanta Fed GDPNow** nowcasting → `fred_nowcast.py`
- **Chicago Fed NFCI** → macro agent
- **Caldara-Iacoviello** GPR Index → geopolitical agent

## Sentiment / Information
- **Shannon entropy** → sentiment agent
- **Transfer entropy** → sentiment agent
- **AAII** retail sentiment → sentiment agent
- **CFTC COT** smart money positioning → `cot_data.py`

## Options
- **Black-Scholes** baseline → underlying for all options modules
- **Breeden-Litzenberger** options-implied PDF → (option for `options_intel.py`)
- **Variance Risk Premium** (Carr-Wu 2009) → vol agent

## Quantum / Research
- **Quantum Amplitude Estimation** → `quantum_finance.py`
- **QAOA** Quantum Approximate Optimization → `quantum_finance.py`

---

# PART IV — DATA SOURCES (All Free)

| Source | API | What we use |
|--------|-----|-------------|
| **yfinance** | unofficial | OHLCV, options, financials, news, insider, holdings, ^VIX, ^MOVE, ^IRX, ^TNX, ^TYX, ^FVX, ^VIX3M |
| **FRED** (St. Louis Fed) | official + key | DGS10/2/30, CPI, PCE, M2, claims, PMI, sentiment, WALCL, GPR, NFCI |
| **CFTC** | Open Data Socrata | Weekly Commitment of Traders |
| **NOAA** | NCEI Climate Data | Temperature anomalies (HDD/CDD) |
| **EIA** | proxy via futures | Crude inventory pressure |
| **SEC EDGAR** | public | 10-K/Q full-text, Form 4, 8-K, 13D/G, 13F |
| **Reddit** | PRAW | WSB, stocks, investing subreddits |
| **Gemini API** | Google AI | LLM analysis (news, earnings calls) |
| **pytrends** | unofficial | Google Trends search volume |
| **FINRA** | public | Short volume %, dark pool ADF |
| **Capitol Trades** style | scraped | Congressional disclosures |

---

# PART V — DYNAMIC PARAMETERS (What Adapts Automatically)

| Parameter | Adaptive Logic |
|-----------|----------------|
| DCF WACC | `rf + β × ERP` from live ^TNX yield |
| DCF terminal growth | `rf − 1.5%` capped [1%, 4%] |
| Flash crash threshold | `3 × daily_realised_vol` (not fixed −7%) |
| Bollinger Bands std | 2.5σ if VIX>25, 1.75σ if VIX<15, linear blend 15-25 |
| News halflife | 2d if VIX>30, 3.5d if VIX>20, 6d if VIX<14 |
| Bayesian direction gate | Entropy>0.85: 0.44/0.56; Entropy<0.40: 0.485/0.515 |
| Regime weights | Soft-blend across 4 regimes via HMM probabilities |
| HMM XLF gate | Block financials when transition risk > 20% |
| Sector gap threshold | BULL: 0.001; BEAR: 0.010; CRISIS: 0.015 |
| HMM size scalar | `max(0.25, min(1.0, bull_prob × 1.5))` |
| Position multiplier | Risk-regime-conditional [0, 1] |

---

# PART VI — CIRCUIT BREAKER HIERARCHY

Risk agent overrides Bayesian fusion in this strict priority order:

| Priority | Trigger | Action |
|----------|---------|--------|
| 1 | BLACK_SWAN (\|Z\|>5σ in 5d) | halt=true, multiplier=0 |
| 2 | FLASH_CRASH (3× daily vol breach) | halt=true, multiplier=0 |
| 3 | CRITICAL_RISK (EVT VaR_99<−8% or GARCH EXTREME) | halt=false, multiplier=0.25 |
| 4 | HIGH_RISK (EVT VaR_95<−5% or GARCH HIGH) | multiplier=0.5 |
| 5 | GEO_SHOCK (VIX>35 + commodity surge) | multiplier=0.35 |
| 6 | CARRY_UNWIND (USD/JPY<125 + JPY surge) | multiplier=0.5 |
| 7 | GEOPOLITICAL OVERRIDE | multiplier=min(0.35, current) |

---

# PART VII — TOTALS

| Category | Count |
|----------|-------|
| **Agents** | 9 (8 voters + 1 circuit breaker) |
| **Total factors across all agents** | 271 |
| Technical agent factors | 60 |
| Fundamental agent factors | 54 |
| Macro agent factors | 43 |
| Sentiment agent factors | 25 |
| Risk agent factors | 24 |
| Insider agent factors | 19 |
| Geopolitical agent factors | 23 |
| Currency agent factors | 13 |
| Volatility agent factors | 12 |
| **Quant Engine modules** | 37 |
| **Academic theories implemented** | 60+ |
| **Free data sources** | 11 |
| **Dynamic parameters** | 11 |
| **Circuit breakers** | 7 |

---

# PART VIII — IMPLEMENTATION LINEAGE (4 Passes)

### Pass 1 — Core Signals + Dynamic Parameters
**Added:** MOVE Index, TED Spread, VIX term structure, Cross-sectional momentum dispersion, Asset Growth Anomaly, Net Stock Issuance, Gross Profitability, Investment-to-Assets, VRP formula, DCC-GARCH module, dynamic WACC, vol-normalised flash crash, VIX-adaptive Bollinger/news halflife, entropy-adaptive Bayesian gate

### Pass 2 — Behavioural + Methodology
**Added:** Idiosyncratic Vol, MAX Anomaly, 1W/3Y Reversal, 52W High Momentum, Overnight/Intraday split, Momentum Crash Risk, R&D Anomaly, QMJ Composite, Realized Skewness, Nelson-Siegel, Business Cycle Phase, Correlation Network Centrality, Black-Litterman module, HRP module, CUSUM module

### Pass 3-4 — Validity + Prado Framework
**Added:** Yang-Zhang/Garman-Klass/Parkinson/Rogers-Satchell vol estimators, Transaction Cost Model (Almgren-Chriss), Marchenko-Pastur RMT cleaning, Portfolio VaR/CVaR/stress, Deflated Sharpe + BH-FDR, López de Prado ML finance suite (Fractional Diff, Triple Barrier, Purged CV, Walk-Forward), ETF Premium signal, Soft regime blending in orchestrator

### Final — Free Data + Tomorrow's Backtest
**Added:** COT Commitment of Traders, NOAA Weather, EIA Petroleum proxy, FRED Macro Nowcast, Google Trends pytrends, Factor Orthogonalization (Gram-Schmidt/PCA/Löwdin), explicit Vol Arbitrage signal, Quantile Regression module, Commodity Roll Yield

---

**End of Complete Inventory.**

This document enumerates every implementation. For architecture/data flow, see `PROJECT_AUDIT.md`. For user-facing docs, see `README.md`.
