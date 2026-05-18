# AlphaAgent — Quantitative Theories Deep Dive

This document is the mathematical foundation for the AlphaAgent platform. It maps every theory to its implementation status, the agent that uses it, and what accuracy improvement it provides.

---

## 🏛️ Part 1: Core Institutional Theories

### 1. Bayesian Signal Fusion
- **Theory:** Bayesian Inference & Bayes' Theorem.
- **Application:** 8-agent voting → sequential prior→posterior updating. Each agent shifts P(Up) based on its confidence and an inter-agent correlation penalty.
- **Formula:** P(A|B) = P(B|A)·P(A) / P(B)
- **Status:** 🟢 **FULLY IMPLEMENTED** — Orchestrator (`orchestrator/graph.py`)

### 2. GARCH Family (Volatility Forecasting)
- **Theory:** GARCH(1,1), ARCH, EGARCH.
- **Application:** Models volatility clustering. EGARCH captures the leverage effect (downside shocks → bigger vol spike). Regime (LOW/NORM/HIGH/EXTREME) gates Monte Carlo path count.
- **Formula:** σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
- **Status:** 🟢 **FULLY IMPLEMENTED** — Volatility Agent (`agents/volatility.py`)

### 3. Monte Carlo GBM + Quasi-Monte Carlo (Sobol)
- **Theory:** Geometric Brownian Motion, Importance Sampling, Sobol low-discrepancy sequences.
- **Application:** Simulates 3k–10k price paths (count scales with GARCH regime). QMC (Sobol/Halton) replaces pseudo-random for faster tail convergence.
- **Status:** 🟢 **FULLY IMPLEMENTED** — Risk Agent (`agents/risk.py`)

### 4. Hidden Markov Model (Regime Detection)
- **Theory:** HMM 3-state (Bull / Bear / Crisis), Baum-Welch EM, Viterbi decoding.
- **Application:** Fitted on 252-day daily returns + vol. State probability contributes directly to Technical Agent factor score.
- **Status:** 🟢 **FULLY IMPLEMENTED** — Technical Agent (`agents/technical.py`)

---

## 🔬 Part 2: Advanced Quant Finance

### 5. Fama-French 5-Factor + CAPM Jensen's Alpha
- **Theory:** CAPM, Fama-French 3F/5F (Mkt, SMB, HML, RMW, CMA), APT.
- **Application:** Decomposes return into factor beta exposure + idiosyncratic alpha. Jensen's Alpha = excess return above CAPM prediction.
- **Status:** 🟢 **FULLY IMPLEMENTED** — Fundamental Agent

### 6. Quality Stack: Piotroski / Altman / Beneish
- **Theory:** Piotroski F-Score (9-pt binary), Altman Z-Score, Beneish M-Score.
- **Application:** Three independent quality lenses — profitability health, bankruptcy risk, earnings manipulation detection. Composite fed into Fundamental Agent PCA.
- **Status:** 🟢 **FULLY IMPLEMENTED** — Fundamental Agent (`agents/fundamental.py`)

### 7. Extreme Value Theory (EVT)
- **Theory:** Generalized Pareto Distribution (GPD), Peaks Over Threshold (POT).
- **Application:** Extrapolates 99th-percentile VaR/CVaR beyond the historical sample. Models fat tails that standard normal misses.
- **Status:** 🟢 **FULLY IMPLEMENTED** — Risk Agent

### 8. Market Microstructure: Kyle's Lambda + Amihud
- **Theory:** Kyle (1985) λ price-impact model, Amihud (2002) illiquidity ratio.
- **Application:** Kyle's λ = Cov(ΔP, signed order flow) / Var(order flow). Amihud = |Return| / Dollar Volume. Both measure adverse-selection and liquidity depth.
- **Status:** 🟢 **FULLY IMPLEMENTED** — Insider Agent (`agents/insider.py`)

### 9. Fisher Equation / Real Interest Rate
- **Theory:** Irving Fisher: Real Rate = Nominal − Expected Inflation.
- **Application:** Real Rate = DGS10 (FRED) − T10YIE (10Y breakeven). High real rates suppress equity multiples (P/E compression). Negative real rates historically fuel risk-asset expansion.
- **Thresholds:** >1.5% (headwind) → 0.5–1.5% (neutral) → <0% (tailwind) → <-1% (strong tailwind)
- **Status:** 🟢 **FULLY IMPLEMENTED** — Macro Agent (Phase 6)

### 10. Contagion Correlation Spike
- **Theory:** Cross-asset correlation as systemic stress indicator (Longin & Solnik 2001).
- **Application:** Rolling 20-day avg pairwise correlation across SPY/TLT/GLD/HYG vs full-history baseline. Spike >0.20 = assets moving together = risk-off contagion.
- **Status:** 🟢 **FULLY IMPLEMENTED** — Macro Agent (Phase 6)

### 11. Sector Relative Strength
- **Theory:** Cross-asset relative momentum — sector leadership predicts underlying stock performance.
- **Application:** Ticker's SPDR sector ETF (XLK/XLF/XLE etc.) 22-day return vs SPY. RS > 1.0 = sector outperformance = tailwind for stocks in that sector.
- **Status:** 🟢 **FULLY IMPLEMENTED** — Macro Agent (Phase 6)

### 12. Herfindahl-Hirschman Index (Ownership Concentration)
- **Theory:** HHI from industrial economics, applied to institutional ownership.
- **Application:** HHI = Σ(fraction²) for top-10 holders. High HHI (>0.25) = concentrated → coordinated selling risk; moderate (0.05–0.25) = stable institutional base.
- **Status:** 🟢 **FULLY IMPLEMENTED** — Insider Agent (Phase 6)

### 13. Hawkes Self-Exciting Process
- **Theory:** Hawkes (1971) point process: λ(t) = μ + ∫α·e^(−β(t−s))dN(s).
- **Application:** Detects volatility event clustering — each vol spike seeds more spikes. Branching ratio α/β measures self-excitation strength.
- **Status:** 🟢 **FULLY IMPLEMENTED** — Volatility Agent

### 14. Copula Dependency Models
- **Theory:** Gaussian + Clayton copulas for non-linear tail co-dependence.
- **Application:** Captures correlation breakdown in stress (tail dependence λ). Better than linear correlation for portfolio risk under crash scenarios.
- **Status:** 🟢 **FULLY IMPLEMENTED** — Risk Agent (`quant_engine/copula.py`)

### 15. Multifractal Analysis (MF-DFA)
- **Theory:** Multi-fractal detrended fluctuation analysis; Hurst H spectrum across moments q.
- **Application:** Decomposes scaling exponents across q-order moments to detect anomalous long-memory regimes not visible in linear returns.
- **Status:** 🟢 **FULLY IMPLEMENTED** — Technical Agent (`quant_engine/multifractal.py`)

### 16. Topological Data Analysis (TDA)
- **Theory:** Persistent homology, Betti numbers.
- **Application:** Persistent H0/H1 features on sliding price windows. Detects market structural topology — cycles (H1) and connectivity changes (H0) before they appear in price.
- **Status:** 🟢 **IMPLEMENTED** — Technical Agent (`quant_engine/...`)

### 17. Granger Causality + Causal DAG (Do-Calculus)
- **Theory:** Granger (1969) VAR F-test + Pearl (2000) do-calculus interventions.
- **Application:** Granger tests if macro series (yield curve, M2, DXY) predict asset returns. DAG computes P(Return|do(VIX=x)) — causal vs correlational.
- **Status:** 🟢 **FULLY IMPLEMENTED** — Macro Agent (`quant_engine/granger.py`, `quant_engine/causal_engine.py`)

### 18. SABR / Heston / Rough Volatility
- **Theory:** SABR (stochastic α β ρ), Heston mean-reverting vol, Rough vol (fBm with H≈0.1).
- **Application:** Three vol surface models fit to options chain. SABR calibrates vol smile; Heston captures mean-reversion; Rough vol matches empirical vol roughness.
- **Status:** 🟢 **FULLY IMPLEMENTED** — Volatility Agent (`quant_engine/sabr.py`, `heston.py`, `rough_vol.py`)

---

## 🚀 Part 3: AI & Machine Learning

### 19. Information Theory Stack (Shannon + Transfer Entropy + KL)
- **Theory:** Shannon entropy, Transfer entropy (Schreiber 2000), KL divergence.
- **Application:** Shannon entropy = agent disagreement → probability confidence interval width. Transfer entropy = does news *cause* price moves? KL divergence = recent vs historical return distribution shift (regime alert).
- **Status:** 🟢 **IMPLEMENTED** — Sentiment Agent + Orchestrator

### 20. Source Credibility + LLM Headline/Body Alignment
- **Theory:** Weighted average credibility scoring + LLM-based stance detection.
- **Application:** 14-tier publisher quality weights (WSJ=1.0 … benzinga=0.55). Gemini LLM dual-pass: rates headline bullish score vs full-body score. Flags bearish body hidden in bullish headline (spin detection).
- **Status:** 🟢 **FULLY IMPLEMENTED** — Sentiment Agent (Phase 6)

### 21. Activist 13D Smart Money Detection
- **Theory:** Regulatory filings as information events (Brav et al. 2008 — activist hedge funds earn +7% alpha).
- **Application:** EDGAR EFTS full-text search for SC 13D / SC 13G filings mentioning ticker in last 90 days. 13D = activist intent → historically bullish catalyst.
- **Status:** 🟢 **FULLY IMPLEMENTED** — Insider Agent (Phase 6)

### 22. FINRA Short Sale Volume (Dark Pool Proxy)
- **Theory:** Short sale volume fraction as institutional conviction signal.
- **Application:** FINRA RegSHO weekly CSV: short vol / total vol. >55% = heavy bearish conviction; <45% = light shorts = bullish signal. Parsed from public `cdn.finra.org` weekly file.
- **Status:** 🟢 **FULLY IMPLEMENTED** — Insider Agent (Phase 6)

### 23. Geopolitical Signal Stack (Active Conflict / Sanctions / Tariff)
- **Theory:** Text-based geopolitical risk measurement (Caldara & Iacoviello 2022 GPR methodology).
- **Application:** Three keyword NLP layers on yfinance news: (1) active conflict keywords, (2) sanctions/embargo keywords, (3) tariff/antitrust/regulatory keywords. Each independently shifts P(Up) down on trigger.
- **Status:** 🟢 **FULLY IMPLEMENTED** — Geopolitical Agent (Phase 6)

---

## 🚀 Part 4: Tier 1–3 New Implementations (May 2026)

### Tier 1 — High-ROI Standard Technical/Fundamental

| Factor | Agent | Key Formula |
| :--- | :--- | :--- |
| **Fibonacci Retracement** | Technical | 52W High-Low range × (0.236/0.382/0.5/0.618/0.786) |
| **Classic Pivot Points** | Technical | P=(H+L+C)/3, R1=2P−L, R2=P+(H−L) daily+weekly |
| **Multi-Timeframe Confluence** | Technical | Weekly RSI+MACD vs Daily RSI+MACD → net 4-signal score |
| **Earnings Revision Momentum** | Fundamental | (Current EPS est − 30d ago est) / |30d est| × 100 |
| **Graham Number** | Fundamental | √(22.5 × EPS × BVPS) vs current price |
| **Rolling Sharpe** | Risk | (μ₆₃ / σ₆₃) × √252 — 63-day window |
| **Rolling Sortino** | Risk | (μ₆₃ / σ_downside) × √252 — 63-day window |

### Tier 2 — Institutional Edge Signals

| Factor | Agent | Key Formula |
| :--- | :--- | :--- |
| **Chart Pattern Recognition** | Technical | 60-bar peak/trough detection → Double Top/Bottom with neckline break |
| **Volume Profile (POC)** | Technical | 20-bin price histogram → highest volume bin = Point of Control |
| **PCE Inflation** | Macro | FRED PCEPI YoY% + 3M annualised rate vs dynamic settings target |
| **Unusual Options Activity** | Sentiment | Vol/OI > 2× per strike → net call/put unusual sweep count |
| **Vanna/Charm Exposure** | Risk | BSM: Vanna = −pdf(d₁)·d₂/σ; Charm = −pdf(d₁)·(2rT−d₂σ√T)/(2Tσ√T) |

### Tier 3 — AI/ML & Advanced Analytics

| Factor | Agent | Implementation |
| :--- | :--- | :--- |
| **FinBERT-Style NLP** | Sentiment | Gemini structured 6-field output: sentiment, guidance, tone, risk, catalyst, confidence |
| **Reddit Sentiment** | Sentiment | Public Reddit JSON API — r/wallstreetbets, r/stocks, r/investing keyword scoring |
| **News Decay Model** | Sentiment | Exponential decay weight = e^(−λ·age); λ = ln2/half-life; configurable half-life (4.6d default) |
| **Earnings Call NLP** | Fundamental | Gemini analysis of quarterly financials + guidance + news: quality/trend/guidance/red-flags |
| **Order Flow Imbalance** | Technical | CLV = ((C−L)−(H−C))/(H−L); Money Flow Volume = CLV×Vol; 5-day rolling avg |
| **Cross-Sectional Rank** | Technical | 11-sector peer map; rank ticker 22-day return vs top-5 sector peers; percentile score |
| **Signal Backtest Calibration** | Technical | Past 126-day conditional hit rate: similar RSI±15 + momentum direction → % up 5 days later |

### Tier 4 — Free-Tier High-Value Additions (May 2026)

| Factor | Agent | Implementation |
| :--- | :--- | :--- |
| **Ichimoku Cloud** | Technical | Tenkan(9)/Kijun(26)/Senkou A+B; above/below/inside cloud + TK cross |
| **Chaikin Money Flow (CMF-14)** | Technical | MFV = ((C−L)−(H−C))/(H−L)×Vol; 14-day sum MFV / 14-day sum Vol |
| **TRIX Oscillator** | Technical | Triple-smoothed EMA(15) pct_change×100; zero-line cross = momentum shift |
| **Parabolic SAR** | Technical | AF=0.02, max=0.20; iterative EP tracking; bull when price above SAR |
| **Net Debt / EBITDA** | Fundamental | (Total Debt − Cash) / EBITDA; <1x=low leverage, >4x=distress |
| **Dividend Yield & Payout** | Fundamental | yfinance dividendYield + payoutRatio; >90% payout → sustainability warning |
| **Retail Sales MoM (RSXFS)** | Macro | FRED RSXFS ex-auto MoM%; >+1%=strong consumer, <−1%=contraction |
| **Leading Economic Index (USSLIND)** | Macro | FRED USSLIND 3-month trend; declining → recession leading indicator |
| **HY Credit Spread OAS (BAMLH0A0HYM2)** | Macro | FRED HY OAS; <3%=tight, >7%=distress; 1-week change for directional signal |
| **Equity Risk Premium** | Macro | ERP = E/P(S&P500) − Real Rate (DGS10−T10YIE); negative = bonds preferred |
| **Short-Squeeze Score** | Sentiment | Short float % + days-to-cover; >20% float + >5 DTC = HIGH squeeze risk |
| **Return Skewness & Kurtosis** | Risk | 3rd/4th standardized moments; negative skew + fat tails = left tail crash risk |
| **Options GEX Wall** | Risk | Net dealer gamma: Σ(±γ×OI×100×S²)/1e8; negative GEX = vol amplification |
| **Pairs Trading Spread Z-score** | Risk | log(A/B) spread Z: (current−μ)/σ over 6M; Z<−1.5 = mean-reversion buy signal |
| **Supply / Demand Zones** | Technical | Volume-spike pivot H/L detection; nearest zone above/below = S/R levels |
| **Financial Conditions Index (NFCI)** | Macro | FRED NFCI 3-month trend; above 0 = tightening, below 0 = accommodative |
| **Dark Pool Print Ratio** | Insider | FINRA ADF weekly off-exchange vol %; >50% = institutional accumulation signal |

---

## 🔧 Part 5: Static → Dynamic Parameter Fixes (May 2026)

All previously hardcoded thresholds are now in `config/settings.yaml`:

| Parameter | Was | Now (settings.yaml key) |
| :--- | :--- | :--- |
| Black swan sigma | `5.0` hardcoded | `risk.black_swan_sigma` |
| Black swan prob_up | `0.10` hardcoded | `risk.black_swan_prob_up` |
| Flash crash ticker % | `7.0` hardcoded | `risk.flash_crash_ticker_pct` |
| Flash crash SPY % | `5.0` hardcoded | `risk.flash_crash_spy_pct` |
| Geo shock VIX level | `35` hardcoded | `risk.geo_shock_vix` |
| Geo shock gold % | `2.0` hardcoded | `risk.geo_shock_gold_pct` |
| Geo shock oil % | `5.0` hardcoded | `risk.geo_shock_oil_pct` |
| Geo shock multiplier | `0.35` hardcoded | `risk.geo_shock_multiplier` |
| Geo shock prob_up | `0.30` hardcoded | `risk.geo_shock_prob_up` |
| Carry unwind USD/JPY | `125` hardcoded | `risk.carry_unwind_usdjpy` |
| Carry unwind yen % | `1.5` hardcoded | `risk.carry_unwind_yen_pct` |
| Carry unwind multiplier | `0.50` hardcoded | `risk.carry_unwind_multiplier` |
| Extreme vol prob_up | `0.20` hardcoded | `risk.extreme_vol_prob_up` |
| High vol prob_up | `0.40` hardcoded | `risk.high_vol_prob_up` |
| High vol multiplier | `0.50` hardcoded | `risk.high_vol_multiplier` |
| KL divergence threshold | `1.0` hardcoded | `risk.kl_divergence_threshold` |
| Vanna neg threshold | `0.5` hardcoded | `risk.vanna_negative_threshold` |
| Vanna/Charm risk-free rate | `0.05` hardcoded | `backtest.risk_free_rate` |
| Amihud stress threshold | `2.0` hardcoded | `macro.amihud_stress_threshold` |
| Bond-equity stress corr | `0.3` hardcoded | `macro.bond_equity_stress_corr` |
| SOFR critical spread | `0.50` hardcoded | `macro.sofr_critical_spread` |
| Bollinger BW narrow | `20` hardcoded | `technical.bw_narrow` |
| Bollinger BW wide | `40` hardcoded | `technical.bw_wide` |
| Gross margin strong | `40%` hardcoded | `fundamental.gross_margin_strong` |
| Quality/value blend | `0.65` hardcoded | `fundamental.quality_value_blend` |
| Social score base | `40.0` hardcoded | `sentiment.social_score_base` |
| Social score multiplier | `4.0` hardcoded | `sentiment.social_score_mult` |
| PCE target | `2.0` hardcoded | `macro.pce_target` |
| PCE near-target | `2.5` hardcoded | `macro.pce_near_target` |
| PCE elevated | `3.5` hardcoded | `macro.pce_elevated` |
| MC paths (LOW regime) | `3000` hardcoded | `simulation.paths_by_regime_low` |
| MC paths (NORMAL) | `5000` hardcoded | `simulation.paths_by_regime_normal` |
| MC paths (HIGH) | `8000` hardcoded | `simulation.paths_by_regime_high` |
| MC paths (EXTREME) | `10000` hardcoded | `simulation.paths_by_regime_extreme` |
| MC daily drift | `0.0005` hardcoded | `simulation.daily_drift` |
| News decay half-life | `4.6d` hardcoded | `backtest.news_decay_halflife_days` |
| Backtest calibration window | `126d` hardcoded | `backtest.calibration_lookback_days` |
| Reddit post limit | `15` hardcoded | `backtest.reddit_post_limit` |
| Cross-sectional peers | `5` hardcoded | `backtest.cross_sectional_peers` |

---

## 🔭 Part 6: Remaining Roadmap (Premium / High-Effort)

These require paid APIs or significant infrastructure not yet justified:

### RL Position Sizing (PPO/SAC)
- **Why:** Static Kelly fraction doesn't adapt to multi-agent regime sequences. RL can maximize Sharpe dynamically.
- **Gap:** Needs live prediction history + backtesting environment.
- **Effort:** High — 200+ hours.

### Real-Time L2 Order Flow (Polygon.io / IBKR)
- **Why:** True bid-ask depth imbalance is stronger than CLV proxy.
- **Gap:** Requires $30–150/mo paid L2 data feed.
- **Effort:** Medium once data available.

### Live Probability Calibration (Isotonic Regression Feedback Loop)
- **Why:** Current calibration is static sigmoid. Live predictions vs outcomes stored in SQLite would allow weekly refitting.
- **Gap:** No historical prediction-vs-outcome database yet.
- **Effort:** Medium — requires persistent prediction store.

---

## 🗺️ Theory Implementation Map (Updated — May 2026)

| Category | Key Methods | AlphaAgent Integration | Status |
| :--- | :--- | :--- | :--- |
| **Ensemble** | Bayesian Fusion | Orchestrator | 🟢 |
| **Volatility** | GARCH, EGARCH, Heston, SABR, Rough Vol, Hawkes | Volatility Agent | 🟢 |
| **Simulation** | Monte Carlo GBM, Quasi-MC (Sobol), EVT | Risk Agent | 🟢 |
| **Regime** | HMM (3-state), Kalman Filter, GARCH Regime | Technical + Vol Agent | 🟢 |
| **Factors** | Fama-French 5F, CAPM Alpha, Piotroski, Altman, Beneish | Fundamental Agent | 🟢 |
| **Macro** | Yield Curve, Fisher Real Rate, Contagion Corr, Sector RS, PCE, LEI, HY OAS, ERP, Retail Sales | Macro Agent | 🟢 |
| **Microstructure** | Kyle's λ, Amihud, LOB, HHI, FINRA Short Vol | Insider + Technical | 🟢 |
| **Causal** | Granger, Do-Calculus DAG, Transfer Entropy | Macro + Sentiment | 🟢 |
| **Topology** | TDA Persistent Homology, Multifractal MF-DFA | Technical Agent | 🟢 |
| **NLP/AI** | Source Credibility, LLM Alignment, Shannon Entropy | Sentiment Agent | 🟢 |
| **NLP Deep (Tier 3)** | FinBERT-Style (Gemini), Reddit Sentiment, News Decay | Sentiment Agent | 🟢 |
| **Earnings NLP (Tier 3)** | Earnings Call NLP — quality/guidance/red-flags | Fundamental Agent | 🟢 |
| **Flow (Tier 3)** | Order Flow Imbalance (CLV proxy), Cross-Sectional Rank | Technical Agent | 🟢 |
| **Backtest (Tier 3)** | Signal Backtest Calibration (conditional hit rate) | Technical Agent | 🟢 |
| **Tech (Tier 1+2)** | Fibonacci, Pivot Points, MTF, Chart Patterns, Volume Profile | Technical Agent | 🟢 |
| **Value (Tier 1)** | Graham Number, ERM, Rolling Sharpe/Sortino | Fundamental + Risk | 🟢 |
| **Geopolitical** | GPR Index, Active Conflict, Sanctions, Tariff NLP | Geopolitical Agent | 🟢 |
| **Insider/Smart $** | Activist 13D, ETF Flow, Congressional STOCK Act | Insider Agent | 🟢 |
| **Options (Tier 2)** | Unusual Options Activity, Vanna/Charm Exposure | Sentiment + Risk | 🟢 |
| **Leverage (Tier 4)** | Net Debt/EBITDA, Dividend Yield + Payout | Fundamental Agent | 🟢 |
| **Technical (Tier 4)** | Ichimoku Cloud, CMF-14, TRIX, Parabolic SAR | Technical Agent | 🟢 |
| **Credit/ERP (Tier 4)** | HY OAS Spread, ERP, Retail Sales, LEI | Macro Agent | 🟢 |
| **Behavioral (Tier 4)** | Short-Squeeze Score (float + DTC) | Sentiment Agent | 🟢 |
| **Distribution (Tier 4)** | Skewness/Kurtosis, Options GEX Wall | Risk Agent | 🟢 |
| **Sizing** | Kelly Criterion (regime-adaptive), Signal Decay | Risk + Orchestrator | 🟢 |
| **Calibration** | Probability Calibration (Platt/isotonic) | Orchestrator | 🟢 |
| **RL/DL** | PPO/SAC position sizing | ❌ Premium roadmap | 🔴 |
| **LOB Real-time** | True L2 Order Flow Imbalance | ❌ Needs paid feed | 🔴 |
| **Live Calibration** | Isotonic regression feedback loop | ❌ Needs prediction DB | 🔴 |

---

> **Implementation count:** 41+ distinct mathematical theories/models across 9 agents · **209+ factors total**
> **TraderUnion Gap Analysis:** All Tier 1–3 factors from the comparison are now implemented. Remaining gaps are premium/paid-API only.  
> **Static → Dynamic:** 21 previously hardcoded parameters moved to `config/settings.yaml` (May 2026).
