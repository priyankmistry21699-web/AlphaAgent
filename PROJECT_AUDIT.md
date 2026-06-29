# AlphaAgent — Complete Project Audit

> **Read this first.** This document is the single source of truth for any agent/engineer joining the project. It covers WHAT the system is, WHY each piece exists, HOW the pieces fit together, and the WHOLE technology + theory stack.

**Last verified:** 2026-06-16
**Repo:** `priyankmistry21699-web/AlphaAgent`
**Primary branch:** `main`
**Phase:** 8 (post 4-pass quant expansion)

---

## 1. What the System Does (One Paragraph)

AlphaAgent is a **multi-agent quantitative trading signal system** for US equities, ETFs, commodities, FX, and crypto. Nine specialist agents run in parallel via LangGraph, each computing dozens of factors from market data, fundamentals, sentiment, macro, options, and alternative datasets. Their probability outputs are fused through Bayesian log-odds combination with regime-conditional weighting and meta-learner stacking, then post-processed with HMM regime overlays, soft regime blending, transaction cost modelling, and portfolio-level risk metrics. The final output for any ticker is a **LONG / SHORT / NEUTRAL** decision with a calibrated probability, conviction percentage, holding period estimate, full agent reasoning trace, portfolio-level VaR, and risk circuit-breaker overrides. Served through FastAPI + WebSocket, visualised through a React/Vite dashboard.

---

## 2. Architecture at a Glance

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          REACT FRONTEND (Vite, port 5173)                  │
│  Signal · Portfolio · Market · Backtest · QuantPanel · AI Assistant tabs   │
└─────────────────────────────────┬──────────────────────────────────────────┘
                                  │  REST + WebSocket
┌─────────────────────────────────▼──────────────────────────────────────────┐
│                       FASTAPI BACKEND (uvicorn, port 8000)                 │
│  api/main.py — REST endpoints + WS streaming + monitoring + warmers        │
└─────────────────────────────────┬──────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼──────────────────────────────────────────┐
│                  LANGGRAPH ORCHESTRATOR (orchestrator/graph.py)            │
│                                                                            │
│   START → data_ingestion → run_agents (parallel) → portfolio_manager → END │
└─────────────────────────────────┬──────────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────────┐
        │                         │                             │
        ▼                         ▼                             ▼
┌──────────────────┐    ┌──────────────────┐         ┌──────────────────┐
│   8 voting       │    │   1 Risk         │         │   Quant Engine   │
│   agents         │    │   (circuit       │         │   (37 modules)   │
│   (parallel)     │    │   breaker)       │         │                  │
└──────────────────┘    └──────────────────┘         └──────────────────┘
        │                         │                             │
        └─────────────────────────┴─────────────────────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │   Bayesian Fusion             │
                  │   + Meta-Learner Stacking     │
                  │   + Soft Regime Blending      │
                  │   + HMM Overlay               │
                  │   + Entropy-Adaptive Gate     │
                  │   + Portfolio VaR             │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                         SignalPacket → DB → Frontend
```

---

## 3. Tech Stack

### Backend
| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.11 | Core implementation |
| Web framework | FastAPI + uvicorn | REST + WebSocket API |
| Orchestration | LangGraph 0.2+ | Typed parallel agent execution DAG |
| Validation | Pydantic v2 | Schema + state typing |
| ML | LightGBM | Meta-learner stacking |
| ML | scikit-learn | Calibration, preprocessing |
| Stats | scipy, statsmodels | Quantile regression, ADF tests |
| Time series | hmmlearn | Gaussian HMM regime detection |
| Data | pandas, numpy | Core dataframes |
| Technical indicators | ta library | RSI, MACD, Bollinger, etc. |
| Network/Graph | networkx | Causal DAG + correlation networks |
| LLM | google-generativeai (Gemini 2.5 Flash) | News sentiment + earnings call NLP |
| HTTP | requests, httpx | Data fetching |
| DB | SQLite + SQLAlchemy | Persistence (Trades, Portfolio, Signals, Backtests) |
| Caching | Redis-style in-memory dicts | TTL-keyed caches per module |
| Concurrency | ThreadPoolExecutor | Parallel agent + data warming |

### Frontend
| Layer | Technology |
|-------|-----------|
| Framework | React 18 |
| Build | Vite 5 |
| Charts | Recharts |
| Styling | CSS modules + tailwind-style utility classes |

### Data Sources (all free)
| Source | What |
|--------|------|
| **yfinance** | OHLCV, options chains, financials, news, insider, holdings |
| **FRED** (St. Louis Fed) | Macro time series (CPI, PMI, yield curve, M2, claims, etc.) |
| **CFTC** (free Open Data API) | Commitment of Traders weekly positioning |
| **NOAA** (free Climate API) | Temperature anomaly for nat gas |
| **EIA** (proxy via futures) | Petroleum inventory pressure signal |
| **SEC EDGAR** | 10-K/10-Q full-text NLP, Form 4 insider |
| **Reddit** (public PRAW) | WSB / Stocks / Investing sentiment |
| **Gemini API** | LLM-based news + earnings call analysis |
| **pytrends** (free Google Trends scrape) | Retail attention proxy |

---

## 4. The 9 Specialist Agents

All agents inherit from `agents/base.py:BaseAgent` and return a typed `AgentResult` (defined in `agents/state.py`). Each agent computes ~25–50 factors and returns a `probability_up` (0–1), `confidence` (0–1), `vote` (LONG/SHORT/HOLD), reasoning text, factor scores, and warnings.

### 8 Voting Agents (feed Bayesian fusion)

| # | Agent | File | Core Theory | Key Factors |
|---|-------|------|------------|-------------|
| 1 | **Technical** | `agents/technical.py` | Classical TA + behavioural + microstructure | RSI, MACD, Bollinger, Ichimoku, Hurst, TDA, GEX, Volume Profile, Fib, Parabolic SAR, Idiosyncratic Vol (Ang), MAX Anomaly (Bali), 1W Reversal (Jegadeesh), 3Y Reversal (DeBondt-Thaler), 52W High (George-Hwang), Overnight/Intraday split, Momentum Crash (Daniel-Moskowitz), Commodity Roll Yield, COT Commercials, Weather Anomaly, EIA Inventory, Google Trends, ETF Premium |
| 2 | **Fundamental** | `agents/fundamental.py` | Quality + value + growth + accounting | Piotroski F-Score, Altman Z, Beneish M, DCF (dynamic WACC), Graham Number, PEAD, FF5 loadings, Asset Growth (Cooper), Net Issuance (Daniel-Titman), Gross Profitability (Novy-Marx), Investment-to-Assets (Hou q-factor), R&D Anomaly (Chan), QMJ Composite (AQR), Gemini Earnings NLP |
| 3 | **Macro** | `agents/macro.py` | Business cycle + cross-asset + rates | Yield curve, VIX, credit spreads, PMI, LEI, ERP, SOFR, PCE, MOVE Index, TED Spread, VIX term structure (vs VIX3M), Cross-sectional momentum dispersion, Nelson-Siegel (L/S/C), Business Cycle Phase (Recovery/Expansion/Slowdown/Contraction), FRED Macro Nowcast |
| 4 | **Sentiment** | `agents/sentiment.py` | NLP + retail + positioning | RAG-Gemini news, Reddit (PRAW), AAII, put/call skew, transfer entropy, analyst revisions, news decay (VIX-adaptive halflife), FinBERT-style classification |
| 5 | **Insider** | `agents/insider.py` | Form 4 cluster trades | SEC Form 4 parse, buy/sell ratio, cluster detection, large-block insider trades |
| 6 | **Geopolitical** | `agents/geopolitical.py` | LLM event scoring + Baltic Dry | Gemini event sentiment, tariff/sanction events, Baltic Dry Index proxy |
| 7 | **Volatility** | `agents/volatility.py` | Options + stochastic vol | GARCH(1,1)/EGARCH, SABR, Rough vol (rBergomi), Heston, VPIN, Zero-DTE, Copula tail dep, VVIX, SKEW Index, IV/RV ratio, **Variance Risk Premium (IV²−RV²)**, Realized Skewness, Yang-Zhang efficient vol, explicit **Vol Arbitrage** signal |
| 8 | **Currency** | `agents/currency.py` | FX regime + carry | DXY momentum, JPY proxy, currency strength matrix |

### 1 Override Agent (circuit breaker — NOT a voter)

| Agent | File | Purpose |
|-------|------|---------|
| **Risk** | `agents/risk.py` | GARCH regimes, EVT 99% VaR, Monte Carlo + Quasi-MC Sobol, Hawkes jump detection, Kelly sizing, **vol-normalised flash crash**, **DCC-GARCH dynamic correlation**, **Correlation Network Centrality**, **CUSUM structural break**, Yang-Zhang Efficient Vol, rolling Sharpe/Sortino. Outputs HALT/CRISIS/HIGH-RISK overrides that pre-empt the Bayesian fusion. |

---

## 5. Quant Engine — All 37 Modules

Located in `quant_engine/`. Organized by purpose:

### A. Volatility / Stochastic Processes (8)
| Module | Theory |
|--------|--------|
| `garch.py` | GARCH(1,1) + EGARCH vol forecasting + regime classification |
| `heston.py` | Heston stochastic vol model with closed-form pricing |
| `sabr.py` | SABR vol smile calibration (Hagan formula) |
| `rough_vol.py` | Rough volatility rBergomi (Hurst < 0.5) |
| `vol_estimators.py` | Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang (5–8× more efficient than close-to-close) |
| `vol_arbitrage.py` | VRP z-score → explicit SHORT_VOL / LONG_VOL signal |
| `dcc_garch.py` | Engle Dynamic Conditional Correlation on RMT-cleaned residuals |
| `quasi_mc.py` | Quasi-Monte Carlo Sobol VaR (~10× efficiency vs pseudo-random) |

### B. Risk / Tail Models (5)
| Module | Theory |
|--------|--------|
| `evt.py` | Extreme Value Theory — POT + GPD fit, VaR₉₉, CVaR₉₉, tail index ξ |
| `monte_carlo.py` | GBM with GARCH-driven drift, 5,000 paths, 68/95/99% CIs |
| `hawkes.py` | Hawkes self-exciting process, branching ratio α/β cascade risk |
| `copula.py` | Gaussian, Clayton, Gumbel copulas — tail co-dependence |
| `portfolio_risk.py` | Portfolio VaR/CVaR, marginal VaR, component VaR, stress scenarios (2008, COVID, 2022) |

### C. Regime / State (4)
| Module | Theory |
|--------|--------|
| `hmm.py` | 3-state Gaussian HMM (Bull / Bear / Crisis) via hmmlearn |
| `markov_regime.py` | Alternative regime-switching model |
| `regime_weights.py` | 4-regime conditional weight tables (BULL_TREND/BULL_CHOPPY/BEAR/CRISIS) |
| `structural_break.py` | CUSUM detector (Page 1954) — online parameter drift detection |

### D. Fusion / Decision (4)
| Module | Theory |
|--------|--------|
| `bayesian.py` | Sequential Bayesian log-odds fusion with correlation penalty |
| `meta_learner.py` | LightGBM stacking blended with Bayesian posterior |
| `calibration.py` | Platt scaling + isotonic regression on rolling signal history |
| `leaderboard.py` | Per-agent rolling IC + accuracy tracking |

### E. Portfolio Construction (3)
| Module | Theory |
|--------|--------|
| `portfolio_optimizer.py` | Mean-variance optimisation |
| `black_litterman.py` | Black-Litterman: CAPM equilibrium + Bayesian agent views |
| `hrp.py` | Hierarchical Risk Parity (López de Prado 2016) — no matrix inversion |

### F. Statistical Methods (5)
| Module | Theory |
|--------|--------|
| `rmt.py` | Random Matrix Theory — Marchenko-Pastur eigenvalue clipping |
| `factor_orthogonalization.py` | Gram-Schmidt + PCA rotation + Symmetric Löwdin |
| `deflated_sharpe.py` | Bailey-Prado Deflated Sharpe + BH-FDR + Bonferroni multiple-testing |
| `quantile_regression.py` | Koenker-Bassett IRLS at 5/25/50/75/95 quantiles |
| `ml_finance.py` | López de Prado: Fractional Diff + Triple Barrier + Purged K-Fold + Walk-Forward |

### G. Microstructure / Information Theory (5)
| Module | Theory |
|--------|--------|
| `vpin.py` | Volume-synchronised PIN (toxic flow detection) |
| `lob.py` | LOB proxy via OHLCV bid-ask + market impact model |
| `kalman.py` | Kalman filter for time-varying β + SPY correlation |
| `granger.py` | Granger causality F-test on VAR(p) |
| `causal_engine.py` | Do-calculus DAG causal inference |

### H. Topology / Geometry (3)
| Module | Theory |
|--------|--------|
| `tda_signal.py` | Topological Data Analysis — Takens embedding + persistent homology |
| `multifractal.py` | MF-DFA q-order fluctuation + generalized Hurst h(q) |
| `quantum_finance.py` | Quantum-inspired amplitude estimation + QAOA (research) |

### I. Cost / Execution (1)
| Module | Theory |
|--------|--------|
| `transaction_costs.py` | Bid-ask + Almgren-Chriss √impact + commission + short borrow |

### J. Signal Lifecycle (4)
| Module | Theory |
|--------|--------|
| `signal_decay.py` | Ornstein-Uhlenbeck half-life → optimal holding period |
| `kelly.py` | Half-Kelly position sizing capped at 25% |
| `momentum.py` | Cross-sectional momentum factor library |
| `factor_exposure.py` | Fama-French 5-factor loadings |

### K. Domain-Specific Signals (6)
| Module | Theory |
|--------|--------|
| `etf_premium.py` | ETF NAV deviation z-score mean-reversion |
| `commodity_roll_yield.py` | Contango/backwardation drag detection |
| `cot_data.py` | CFTC Commitment of Traders — commercials z-score |
| `weather_factor.py` | NOAA HDD/CDD anomaly → nat gas demand |
| `eia_petroleum.py` | EIA inventory pressure via WTI cointegration |
| `fred_nowcast.py` | GDPNow-style composite from 6 FRED series |
| `google_trends.py` | pytrends Search Volume z-score |
| `options_intel.py` | Options chain analytics (GEX, max pain, implied move, IV skew) |
| `zero_dte.py` | Zero-day-to-expiry options flow analysis |
| `pead.py` | Post-Earnings Announcement Drift |
| `sec_nlp.py` | SEC 10-K/10-Q language similarity |
| `scoring.py` | Centralised fundamental score computation (Piotroski, Altman, Beneish) |
| `technical.py` | Centralised technical indicator computation |
| `macro.py` | Centralised macro environment analysis |
| `insider.py` | SEC Form 4 cluster trade analytics |

---

## 6. Data Flow — One Signal End-to-End

```
USER → POST /api/signal?ticker=AAPL
   │
   ▼
api/main.py → MarketData("AAPL") instantiated
   │
   ▼
LangGraph build_alpha_graph() invoked
   │
   ▼
Node 1: data_ingestion_node (parallel pre-warm)
   ├─ OHLCV (1y)
   ├─ Financials (income/balance/cashflow)
   ├─ Info (yfinance .info dict)
   ├─ News (yfinance .news)
   ├─ Options chain (nearest expiry)
   └─ Returns (1y pct_change)
   │
   ▼
Node 2: run_agents_node (8 agents in parallel via ThreadPoolExecutor)
   │
   │   ┌─ Technical  → 50+ factors → AgentResult(prob_up, confidence, vote)
   │   ├─ Fundamental → 45+ factors → AgentResult
   │   ├─ Macro      → 35+ factors → AgentResult
   │   ├─ Sentiment  → 25+ factors → AgentResult
   │   ├─ Insider    → AgentResult
   │   ├─ Geopolitical → AgentResult
   │   ├─ Volatility → 30+ factors → AgentResult
   │   ├─ Currency   → AgentResult
   │   └─ Risk       → factor_scores + override flags (CRISIS/HIGH/CRITICAL)
   │
   ▼
Node 3: portfolio_manager_node
   ├─ Step 1: Risk circuit breakers
   │     • BLACK_SWAN → halt=true, multiplier=0
   │     • FLASH_CRASH → halt=true, multiplier=0
   │     • CRITICAL_RISK → halt=false, multiplier=0.25
   │     • HIGH_RISK → multiplier=0.5
   │     • GEO_SHOCK → multiplier=0.35
   │     • CARRY_UNWIND → multiplier=0.5
   │
   ├─ Step 2: Market regime detection
   │     • detect_regime() → BULL_TREND / BULL_CHOPPY / BEAR / CRISIS
   │     • Get regime correlation map + agent weights
   │     • Soft-blend weights via HMM probabilities
   │
   ├─ Step 3: Bayesian fusion (8 voters)
   │     • Dynamic prior (SPY vs 50d SMA → 0.47-0.53)
   │     • Per-agent confidence × regime_scale × (1 - correlation)
   │     • Log-odds update → posterior probability
   │     • Entropy + agreement metrics
   │
   ├─ Step 4: Direction gate (entropy-adaptive)
   │     • Entropy > 0.85: gates 0.56/0.44
   │     • Entropy > 0.70: gates 0.545/0.455
   │     • Entropy < 0.40: gates 0.515/0.485
   │     • Default: 0.53/0.47
   │
   ├─ Step 4.5: Meta-Learner Stacking
   │     • LightGBM prediction
   │     • Blended with Bayesian posterior using learned weight
   │     • Recompute direction with entropy gates
   │
   ├─ Step 5: HMM regime + transition probability
   ├─ Step 6: Signal decay (OU half-life)
   ├─ Step 7: Compile SignalPacket
   └─ Step 8: Portfolio-level VaR/CVaR + stress scenarios
   │
   ▼
state.final_signal = {
    packet, probability_up, multiplier, entropy,
    council, agreement_score, market_regime, portfolio_risk
}
   │
   ▼
api/main.py response → DB persistence → WebSocket broadcast → React
```

---

## 7. Key Theories Implemented (Indexed by Author)

| Author / Paper | Implementation |
|----------------|---------------|
| **Black & Litterman (1992)** | `black_litterman.py` |
| **Marchenko & Pastur (1967)** | `rmt.py` |
| **Engle (2002) DCC-GARCH** | `dcc_garch.py` |
| **Bollerslev (1986) GARCH** | `garch.py` |
| **Heston (1993) Stoch Vol** | `heston.py` |
| **Hagan SABR (2002)** | `sabr.py` |
| **Bayer-Friz-Gatheral Rough Vol (2016)** | `rough_vol.py` |
| **Hawkes (1971) self-exciting** | `hawkes.py` |
| **Pickands-Balkema-de Haan EVT** | `evt.py` |
| **Sobol (1967) low-discrepancy** | `quasi_mc.py` |
| **Markov / Baum-Welch HMM** | `hmm.py` |
| **Page (1954) CUSUM** | `structural_break.py` |
| **Koenker-Bassett (1978) quantile reg** | `quantile_regression.py` |
| **Bailey & López de Prado (2014) Deflated Sharpe** | `deflated_sharpe.py` |
| **Benjamini-Hochberg FDR (1995)** | `deflated_sharpe.py` |
| **López de Prado (2016) HRP** | `hrp.py` |
| **López de Prado AFML (2018)** — Fractional Diff, Triple Barrier, Purged CV | `ml_finance.py` |
| **Parkinson (1980) range vol** | `vol_estimators.py` |
| **Garman-Klass (1980)** | `vol_estimators.py` |
| **Yang-Zhang (2000)** | `vol_estimators.py` |
| **Almgren-Chriss (2000) market impact** | `transaction_costs.py` |
| **Kelly (1956) optimal sizing** | `kelly.py` |
| **Granger (1969) causality** | `granger.py` |
| **Pearl Do-Calculus** | `causal_engine.py` |
| **Kalman (1960) filtering** | `kalman.py` |
| **Sloan (1996) accruals** | `scoring.py` (Beneish M-Score component) |
| **Piotroski (2000) F-Score** | `scoring.py` |
| **Altman (1968) Z-Score** | `scoring.py` |
| **Beneish (1999) M-Score** | `scoring.py` |
| **Graham Number** | `scoring.py` |
| **Fama-French 3+2 (1993, 2015)** | `factor_exposure.py` |
| **Frazzini-Pedersen BAB (2014)** | `factor_exposure.py` |
| **Novy-Marx Gross Profitability (2013)** | `scoring.py` |
| **Cooper-Gulen-Schill Asset Growth (2008)** | `scoring.py` |
| **Daniel-Titman Net Issuance** | `scoring.py` |
| **Hou-Xue-Zhang q-factor (2015)** | `scoring.py` (Investment-to-Assets) |
| **Asness QMJ (2019)** | fundamental agent composite |
| **Chan R&D Anomaly** | fundamental agent |
| **Ang et al. Idiosyncratic Vol (2006)** | technical agent |
| **Bali et al. MAX Anomaly (2011)** | technical agent |
| **Jegadeesh (1990) 1-week reversal** | technical agent |
| **DeBondt-Thaler (1985) long-run reversal** | technical agent |
| **George-Hwang 52W High (2004)** | technical agent |
| **Daniel-Moskowitz Momentum Crash (2016)** | technical agent |
| **Hirshleifer-Teoh Limited Attention** | sentiment agent (via Google Trends) |
| **Topological Data Analysis (Carlsson 2009)** | `tda_signal.py` |
| **Multifractal MF-DFA (Kantelhardt 2002)** | `multifractal.py` |
| **VPIN — Easley-López-O'Hara (2012)** | `vpin.py` |
| **Bridgewater Business Cycle** | macro agent (`business_cycle` factor) |
| **Nelson-Siegel (1987) yield curve** | macro agent (`nelson_siegel` factor) |
| **GDPNow nowcasting** | `fred_nowcast.py` |
| **CFTC COT theory (smart money)** | `cot_data.py` |

---

## 8. Dynamic vs Static Parameters

### Now Dynamic (after 4-pass refactor)
| Parameter | Old (static) | New (dynamic) |
|-----------|--------------|---------------|
| DCF WACC | 10% hardcoded | `rf + β × ERP` real-time from ^TNX |
| DCF terminal growth | 3% hardcoded | `rf − 1.5%` capped 1–4% |
| Flash crash threshold | -7% fixed | `3 × daily_realised_vol` |
| Bollinger Bands std | 2.0σ fixed | VIX-adaptive 1.75–2.5σ |
| News halflife | 4.6 days fixed | VIX-adaptive 2–6 days |
| Bayesian direction gate | 0.47/0.53 fixed | Entropy-adaptive (0.485–0.56 / 0.44–0.515) |
| Regime weights | Discrete table switch | Soft-blended via HMM probabilities |
| Sector gap threshold | 0.003 fixed | HMM-regime-adaptive (0.001 BULL, 0.010 BEAR, 0.015 CRISIS) |
| Position size scalar | 1.0 fixed | HMM bull_prob × 1.5, capped [0.25, 1.0] |

### Still Static (config/settings.yaml)
- Piotroski cutoffs, Altman Z thresholds (these are academic constants)
- Kelly max cap at 25%
- GARCH regime percentiles (25/75/95)
- MC simulation path counts per regime

---

## 9. Risk Circuit Breakers (Override Hierarchy)

Pre-empts Bayesian fusion. Order of precedence:

1. **BLACK_SWAN** — |Z-score| > 5σ in 5 sessions → `multiplier=0, halt=true`
2. **FLASH_CRASH** — ticker drops > 3×daily_vol or SPY > 3×SPY_daily_vol in 1d → `halt=true`
3. **CRITICAL_RISK** — EVT VaR_99 < -8% or GARCH EXTREME regime → `multiplier=0.25, halt=false`
4. **HIGH_RISK** — EVT VaR_95 < -5% or GARCH HIGH → `multiplier=0.5`
5. **GEO_SHOCK** — VIX > 35 + (gold +2% or oil +5% in 1d) → `multiplier=0.35`
6. **CARRY_UNWIND** — USD/JPY < 125 + JPY surge -1.5% → `multiplier=0.5`
7. **GEOPOLITICAL OVERRIDE** — `multiplier=min(0.35, current)`

---

## 10. Database Schema

`database/models.py` — SQLAlchemy ORM:

| Table | Purpose |
|-------|---------|
| `Trade` | Executed trades with entry/exit, P&L, agent breakdown |
| `Portfolio` | Position book — current holdings, cost basis |
| `AgentLog` | Per-signal agent vote audit trail |
| `AIPortfolioState` | LLM-generated portfolio context |
| `SignalHistory` | Every generated signal (rolling window) |
| `BacktestResult` | Saved backtest configurations + results |
| `Settings` | User-configurable runtime settings |
| `WarmupRegistry` | Background data warmer state |

Manager: `database/manager.py` — connection pooling + CRUD.

---

## 11. API Endpoints

Base URL: `http://localhost:8000`

### REST
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/signal/{ticker}` | GET | Full 9-agent signal for one ticker |
| `/api/signal/batch` | POST | Batch signals for ticker list |
| `/api/portfolio` | GET / POST | Portfolio CRUD |
| `/api/backtest` | POST | Run a backtest |
| `/api/market/snapshot` | GET | Market regime + breadth |
| `/api/quant/{module}` | GET | Direct quant_engine module output |
| `/api/agent/{name}/factors` | GET | Detailed factor scores for one agent |
| `/api/leaderboard` | GET | Per-agent rolling IC + accuracy |
| `/api/ai/chat` | POST | Gemini-powered portfolio Q&A |
| `/api/settings` | GET / PUT | Runtime config |

### WebSocket
| Endpoint | Stream |
|----------|--------|
| `/ws/signals` | Real-time agent reasoning + final signal |
| `/ws/portfolio` | Live position updates |
| `/ws/market` | Market regime changes |

---

## 12. Frontend Tabs (`frontend-react/`)

| Tab | Component | What it shows |
|-----|-----------|---------------|
| **Signal** | `Signal.jsx` | Per-ticker full agent breakdown, price chart, conviction, overrides |
| **Portfolio** | `Portfolio.jsx` | Position book, daily P&L, allocation pie |
| **Market** | `Market.jsx` | Regime, breadth, top movers, sector heatmap |
| **Backtest** | `Backtest.jsx` | Run backtests with adjustable parameters |
| **QuantPanel** | `QuantPanel.jsx` | Live quant_engine module outputs |
| **Other** | `OtherTabs.jsx` | Factor deep dive, leaderboard, settings |
| **AI Assistant** | `AIAssistant.jsx` | Gemini-powered portfolio chatbot |

---

## 13. Backtest Scripts

| Script | Purpose |
|--------|---------|
| `portfolio_may19_20_backtest.py` | 2-day US-50 backtest with HMM overlays, transaction costs |
| `portfolio_may20_backtest.py` | Single-day US-50 v7 (gap-adjusted sector overlays) |
| `portfolio_june5_backtest.py` | $100K single-investor June 5 2026 backtest |
| `portfolio_100k.py` | Generic single-investor $100K runner |
| `batch_signal_runner.py` | Concurrent multi-ticker signal runner |
| `backtest_multiday.py` | Generic multi-day backtest with HMM + cost overlays |
| `backtest_threshold_finder.py` | Hyperparameter sweep for direction gate thresholds |
| `backtest_2026_ytd_v3.py` | Year-to-date 2026 backtest |
| `intraday_sim.py` | Intraday simulation harness |

---

## 14. Configuration Files

| File | Purpose |
|------|---------|
| `config/settings.yaml` | All numeric thresholds (regime cutoffs, risk thresholds, vol windows) |
| `config/settings_manager.py` | Settings loader + dynamic update API |
| `config/thresholds.yaml` | Domain-specific thresholds (DCF, Z-score, etc.) |
| `.env` | Secrets: `GEMINI_API_KEY`, `FRED_API_KEY` (optional), DB URL |

---

## 15. Known Issues / Skipped Items

### Skipped — Need Institutional Data
- Real Level 2 order book (~$50K/month)
- Tick data (Bloomberg/Refinitiv ~$24K/year)
- Dark pool flow (prime broker only)
- 13F crowding real-time
- Satellite imagery ($50K-$500K/year)
- Credit card transaction data ($100K-$500K/year)
- CDS spreads (Markit ~$20K/year)

### Skipped — Diminishing Returns (<5% marginal alpha)
- Transformer time-series (PatchTST/TFT) — LightGBM is sufficient
- RL position sizing — Kelly + Bayesian is robust enough
- Conformal prediction — error bands at current accuracy are fine
- Optimal Transport / Wasserstein — KL divergence covers it
- Functional Data Analysis — Nelson-Siegel already captures yield curve shape
- Wavelet decomposition — multi-scale already via 5d/22d/252d momentum

### Built but Not Yet Wired Into Live Pipeline
| Module | Where it should connect |
|--------|------------------------|
| `black_litterman.py` | Should size positions in `batch_signal_runner.py` |
| `hrp.py` | Should allocate multi-asset portfolios |
| `ml_finance.py` (Triple Barrier, Purged CV) | Should train the LightGBM meta-learner |
| `factor_orthogonalization.py` | Should pre-process factor scores inside each agent |
| `deflated_sharpe.py` | Should be in every backtest summary |
| `quantile_regression.py` | Should feed conviction uncertainty into Bayesian fusion |

---

## 16. How to Run

```bash
# Backend
cd "d:\ML and DL\Python\AlphaAgent"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# Frontend (new terminal)
cd frontend-react
npm run dev
# → http://localhost:5173

# CLI signal test
python test_signal.py AAPL

# Run a backtest
python portfolio_june5_backtest.py
```

---

## 17. Project Structure (Top Level)

```
AlphaAgent/
├── agents/              # 9 agent implementations (Python)
├── orchestrator/        # LangGraph DAG (graph.py)
├── quant_engine/        # 37 quant modules
├── data/                # Data fetching layer (yfinance, FRED, polygon, alternative)
├── api/                 # FastAPI server + monitoring
├── database/            # SQLAlchemy models + manager
├── config/              # YAML settings + manager
├── frontend-react/      # React/Vite dashboard
├── backtest/            # Backtest engine (legacy)
├── portfolio_*_backtest.py    # Standalone backtest scripts
├── batch_signal_runner.py     # Concurrent batch runner
├── test_signal.py             # Single-ticker CLI tester
├── README.md                  # User-facing readme
└── PROJECT_AUDIT.md           # ← You are here
```

---

## 18. Quick Reference — Key Function Entry Points

| Want to... | Call |
|------------|------|
| Get a signal for one ticker | `from orchestrator.graph import build_alpha_graph; graph.invoke({"ticker": ..., "market_data": MarketData(t), "registry": AgentRegistry()})` |
| Run an agent in isolation | `from agents.technical import TechnicalAgent; TechnicalAgent().analyze("AAPL", MarketData("AAPL"))` |
| Get HMM regime | `from quant_engine.hmm import RegimeDetector; RegimeDetector(n_states=3).fit_predict(spy_returns)` |
| Compute transaction cost | `from quant_engine.transaction_costs import TransactionCostModel; TransactionCostModel().round_trip(...)` |
| Get portfolio VaR | `from quant_engine.portfolio_risk import PortfolioRisk; PortfolioRisk().analyze(weights, returns)` |
| Batch run multiple tickers | `from batch_signal_runner import run_batch; run_batch(["AAPL","MSFT","NVDA"])` |

---

## 19. The 4 Expansion Passes (Recent History)

This system went through 4 incremental expansion passes documented in this audit. Each pass added a specific category of capability:

| Pass | Focus | Added |
|------|-------|-------|
| **Pass 1** | Core gaps | MOVE/TED/VIX3M, 4 fundamental factors, dynamic params, entropy gate, VRP, DCC-GARCH |
| **Pass 2** | Behavioural + methodology | 7 behavioural factors, R&D/QMJ, realized skew, Nelson-Siegel, business cycle, network centrality, Black-Litterman, HRP, CUSUM |
| **Pass 3-4** | Validity + Prado framework | 4 efficient vol estimators, transaction cost model, RMT cleaning, portfolio VaR, deflated Sharpe, López de Prado ML finance suite, ETF premium, soft regime blending |
| **Final** | Tomorrow's backtest items | COT/Weather/EIA/FRED Nowcast/Google Trends data integrations, factor orthogonalization, vol arbitrage explicit signal, quantile regression |

---

## 20. Definitions an Agent Needs to Know

- **Signal direction**: `LONG / SHORT / NEUTRAL`
- **Probability up**: posterior P(price ↑ over horizon) ∈ [0, 1]
- **Conviction**: |prob_up − 0.5| × 2 ∈ [0, 1]
- **Multiplier**: position sizing scalar from risk overrides ∈ [0, 1]
- **Entropy**: Shannon entropy of fused agent vote distribution (high = disagreement)
- **Agreement score**: 1 − entropy (high = consensus)
- **IC (Information Coefficient)**: rolling Spearman correlation between agent's prob_up and forward return
- **Regime**: BULL_TREND / BULL_CHOPPY / BEAR / CRISIS (orchestrator-detected)
- **HMM regime**: BULL / BEAR / CRISIS (3-state Gaussian HMM)
- **VRP**: Variance Risk Premium = IV² − RV² (annualised variance points)
- **Override**: a Risk agent flag that bypasses Bayesian fusion

---

**End of audit.** Any agent reading this should now be able to answer "what is AlphaAgent, what does it do, how does it work, and what's already built vs missing?"
