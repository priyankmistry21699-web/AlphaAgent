# AlphaAgent — Phase-Wise Implementation Plan

> 12 Agents | 154 Factors | 5 Phases | ~10 Weeks

---

## Project Structure (Final)

```
AlphaAgent/
├── config/
│   ├── agents.yaml
│   ├── factors.yaml
│   └── portfolio.yaml
├── quant_engine/
│   ├── __init__.py
│   ├── garch.py              # GARCH(1,1) + EGARCH
│   ├── monte_carlo.py        # GBM simulation
│   ├── hmm.py                # Hidden Markov Models
│   ├── bayesian.py           # Bayesian fusion
│   ├── evt.py                # Extreme Value Theory
│   ├── kelly.py              # Kelly Criterion
│   ├── kalman.py             # Kalman Filter
│   ├── technical.py          # RSI, MACD, Bollinger, etc.
│   ├── options_intel.py      # GEX, IV skew, max pain
│   ├── momentum.py           # Hurst, 12M-1M, acceleration
│   ├── signal_decay.py       # Holding period estimation
│   └── scoring.py            # Piotroski, Altman, Beneish
├── data/
│   ├── __init__.py
│   ├── market.py             # yfinance wrapper
│   ├── macro.py              # FRED API
│   ├── news.py               # NewsAPI + RSS
│   ├── institutional.py      # SEC EDGAR, FINRA
│   ├── currency.py           # FX data
│   ├── alternative.py        # BDI, Fear&Greed, AAII
│   ├── cache.py              # SQLite caching layer
│   └── validation.py         # Data cleaning
├── factors/
│   ├── __init__.py
│   ├── base.py               # BaseFactor abstract class
│   ├── registry.py           # Auto-discovery
│   └── (individual factor files)
├── agents/
│   ├── __init__.py
│   ├── base.py               # BaseAgent abstract class
│   ├── registry.py           # Auto-discovery
│   ├── state.py              # Pydantic schemas
│   ├── graph.py              # LangGraph state machine
│   ├── orchestrator.py
│   ├── technical.py
│   ├── sentiment.py
│   ├── fundamental.py
│   ├── macro.py
│   ├── volatility.py
│   ├── institutional.py
│   ├── geopolitical.py
│   ├── currency.py
│   ├── debate.py
│   ├── risk.py
│   └── portfolio.py
├── api/
│   ├── __init__.py
│   ├── main.py               # FastAPI app
│   ├── routes.py             # REST endpoints
│   ├── websocket.py          # WebSocket streaming
│   └── models.py             # API schemas
├── frontend/                  # React + Vite
│   ├── src/
│   │   ├── components/
│   │   │   ├── SignalCard.jsx
│   │   │   ├── MonteCarloChart.jsx
│   │   │   ├── AgentRadar.jsx
│   │   │   ├── PortfolioDashboard.jsx
│   │   │   ├── RiskDashboard.jsx
│   │   │   └── PerformanceChart.jsx
│   │   ├── pages/
│   │   ├── App.jsx
│   │   └── main.jsx
│   └── package.json
├── tests/
│   ├── __init__.py
│   ├── test_quant_engine.py
│   ├── test_agents.py
│   ├── test_data.py
│   ├── test_portfolio.py
│   └── test_backtest.py
├── backtest/
│   ├── __init__.py
│   ├── engine.py             # Walk-forward backtester
│   ├── calibration.py        # Probability calibration
│   └── stress_test.py        # Crisis scenario simulation
├── FACTOR_MAP.md
├── FEATURES.md
├── DESIGN_PHILOSOPHY.md
├── QUANT_TECHNIQUES.md
├── QUANT_THEORIES_DEEP_DIVE.md
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Phase 1: Foundation (Weeks 1-2) — ✅ COMPLETED


### Week 1: Quant Engine Core

**Day 1-2: Project Setup + Data Layer**
```
Files to create:
  config/agents.yaml          — Agent enable/disable + weights
  data/market.py              — yfinance wrapper with caching
  data/cache.py               — SQLite cache for API responses
  data/validation.py          — Handle NaN, splits, gaps
  agents/state.py             — Pydantic schemas for AgentResult, SignalPacket

Deliverable: 
  >>> from data.market import MarketData
  >>> md = MarketData("NVDA")
  >>> md.get_ohlcv(period="1y")  # Returns clean DataFrame
  >>> md.get_info()               # P/E, market cap, sector
  >>> md.get_financials()         # Income statement, balance sheet
```

**Day 3-4: Technical Indicators + GARCH**
```
Files to create:
  quant_engine/technical.py   — RSI, MACD, Bollinger, SMA/EMA, ADX, ATR, OBV
  quant_engine/garch.py       — GARCH(1,1) fit + forecast
  
Deliverable:
  >>> from quant_engine.technical import compute_indicators
  >>> indicators = compute_indicators(ohlcv_df)
  >>> indicators["rsi"]        # 42.3
  >>> indicators["macd_signal"] # "bullish_crossover"
  
  >>> from quant_engine.garch import GARCHModel
  >>> model = GARCHModel(returns)
  >>> model.forecast_vol(days=5)  # [1.2%, 1.3%, 1.4%, 1.3%, 1.5%]
```

**Day 5: Monte Carlo Simulation**
```
Files to create:
  quant_engine/monte_carlo.py — GBM + GARCH-driven simulation

Deliverable:
  >>> from quant_engine.monte_carlo import MonteCarloEngine
  >>> mc = MonteCarloEngine(price=135.0, vol=0.018, drift=0.0008)
  >>> result = mc.simulate(days=5, paths=10000)
  >>> result.ci_68    # (137.2, 142.8)
  >>> result.ci_95    # (133.1, 146.3)
  >>> result.ci_99    # (130.5, 149.1)
```

### Week 2: First 3 Agents + Debate

**Day 6-7: Agent Framework + Technical Agent**
```
Files to create:
  agents/base.py              — BaseAgent with analyze() interface
  agents/registry.py          — Auto-discovery system
  agents/technical.py         — 15 core factors (RSI, MACD, etc.)
  
Deliverable:
  >>> agent = TechnicalAgent()
  >>> result = agent.analyze("NVDA", data)
  >>> result.probability_up    # 0.72
  >>> result.confidence        # 0.81
  >>> result.reasoning         # "RSI oversold + MACD bullish..."
  >>> result.factor_scores     # {"rsi": 78, "macd": 72, ...}
```

**Day 8: Fundamental Agent (core factors)**
```
Files to create:
  agents/fundamental.py       — P/E, P/B, DCF, margins, ROE, growth
  quant_engine/scoring.py     — Piotroski F-Score, Altman Z-Score

Deliverable:
  >>> agent = FundamentalAgent()
  >>> result = agent.analyze("NVDA", data)
  >>> result.probability_up    # 0.58
  >>> result.factor_scores["pe_vs_sector"]  # 0.72 (undervalued)
  >>> result.factor_scores["piotroski"]     # 7 (strong)
```

**Day 9: Sentiment Agent (basic) + News Data**
```
Files to create:
  data/news.py                — NewsAPI + RSS fetcher
  agents/sentiment.py         — LLM-scored news + basic behavioral

Deliverable:
  >>> agent = SentimentAgent()
  >>> result = agent.analyze("NVDA", data)
  >>> result.probability_up    # 0.65
  >>> result.news_summary      # "5 articles: 3 bullish, 1 bearish, 1 neutral"
```

**Day 10: Debate Agent + CLI Interface**
```
Files to create:
  agents/debate.py            — Weighted average fusion (simple V1)
  agents/graph.py             — LangGraph pipeline: Tech→Sent→Fund→Debate
  cli.py                      — Command-line interface

Deliverable:
  $ python cli.py --ticker NVDA
  
  ═══════════════════════════════════════
  AlphaAgent Signal — NVDA
  ═══════════════════════════════════════
  Direction:  LONG
  Conviction: 65% 
  Confidence: MEDIUM
  
  Agent Scores:
    Technical:    72% bullish
    Fundamental:  58% bullish  
    Sentiment:    65% bullish
  
  Expected Move (5-day):
    68% CI: +1.2% to +4.8%
    95% CI: -2.1% to +7.3%
  ═══════════════════════════════════════
```

### Phase 1 Tests
```
tests/test_quant_engine.py:
  - test_rsi_overbought_oversold()
  - test_garch_fit_converges()
  - test_monte_carlo_ci_contains_mean()
  - test_piotroski_score_range()
  
tests/test_agents.py:
  - test_technical_agent_returns_valid_probability()
  - test_fundamental_agent_pe_scoring()
  - test_debate_agent_combines_signals()
```

---

## Phase 2: Advanced Quant + More Agents (Weeks 3-4) — ✅ COMPLETED


### Week 3: Advanced Math Modules

**Day 11-12: HMM Regime Detection + Bayesian Fusion**
```
Files to create:
  quant_engine/hmm.py         — 3-state HMM (Bull/Bear/Crisis)
  quant_engine/bayesian.py    — Full Bayesian posterior with correlation adj

Deliverable:
  >>> hmm = RegimeDetector(returns)
  >>> hmm.current_regime       # "BULL"
  >>> hmm.probabilities        # {"bull": 0.78, "bear": 0.16, "crisis": 0.06}
  >>> hmm.transition_risk      # {"to_bear": 0.05, "to_crisis": 0.01}
  
  >>> fusion = BayesianFusion(prior=0.52)
  >>> fusion.update(agent_prob=0.72, correlation=0.0)   # Technical
  >>> fusion.update(agent_prob=0.65, correlation=0.3)   # Sentiment
  >>> fusion.posterior          # 0.73
  >>> fusion.entropy            # 0.42
```

**Day 13: EVT + Kelly Criterion**
```
Files to create:
  quant_engine/evt.py         — GPD fitting, VaR, CVaR
  quant_engine/kelly.py       — Kelly + half-Kelly + vol-adjusted sizing

Deliverable:
  >>> evt = ExtremeValueModel(losses)
  >>> evt.var(0.99)            # -4.2%
  >>> evt.cvar(0.99)           # -5.8%
  
  >>> kelly = KellyCriterion(p_win=0.73, win_loss_ratio=2.0)
  >>> kelly.full_kelly         # 0.595
  >>> kelly.half_kelly         # 0.297
```

**Day 14: Signal Decay + Options Intelligence**
```
Files to create:
  quant_engine/signal_decay.py — Autocorrelation, OU half-life
  quant_engine/options_intel.py — IV skew, GEX, max pain
  quant_engine/momentum.py     — Hurst exponent, 12M-1M

Deliverable:
  >>> decay = SignalDecay(signal_history)
  >>> decay.half_life_days      # 5.2
  >>> decay.optimal_hold        # (3, 7)  — 3 to 7 days
```

### Week 4: Agents + FastAPI

**Day 15-16: Macro + Volatility + Risk Agents**
```
Files to create:
  data/macro.py               — FRED API wrapper
  agents/macro.py             — Rates, VIX, liquidity, cross-asset, contagion
  agents/volatility.py        — GARCH forecast, IV/RV, vol regime, beta
  agents/risk.py              — EVT + Kelly + guardrails + override system

Upgrade:
  agents/debate.py            — Replace weighted avg with full Bayesian fusion
  agents/graph.py             — Add Macro + Volatility → Debate → Risk
```

**Day 17-18: Sentiment RAG Pipeline**
```
Files to create:
  data/news.py                — Upgrade: ChromaDB embeddings
  requirements: chromadb, sentence-transformers
  
Upgrade:
  agents/sentiment.py         — RAG retrieval + LLM scoring + behavioral factors
```

**Day 19-20: FastAPI Backend**
```
Files to create:
  api/main.py                 — FastAPI app with CORS
  api/routes.py               — POST /analyze/{ticker}, GET /signal/{ticker}
  api/models.py               — SignalPacket response schema
  api/websocket.py            — /ws/analysis/{ticker} streaming

Deliverable:
  $ uvicorn api.main:app --reload
  
  POST /analyze/NVDA →
  {
    "signal": {"direction": "LONG", "conviction_pct": 73.2, "confidence": "HIGH"},
    "expected_move": {"ci_68": {"low": 1.2, "high": 4.8}, ...},
    "risk_metrics": {"var_95": -2.1, "cvar_99": -5.8, ...},
    "agent_scores": {...},
    "regime": {"current": "BULL", "probabilities": {...}}
  }
```

### Phase 2 Tests
```
  - test_hmm_detects_bull_market()
  - test_bayesian_fusion_correlation_penalty()
  - test_evt_var_reasonable_range()
  - test_risk_agent_override_on_crisis()
  - test_api_analyze_returns_valid_signal()
```

---

## Phase 3: Full Agent Roster (Weeks 5-6) — ✅ COMPLETED


### Week 5: Remaining Analysis Agents

**Day 21-22: Institutional Agent + Whale Tracking Data Layer**
```
Files to create:
  data/institutional.py       — All institutional data sources:
  
  SEC EDGAR Integration (FREE, no API key):
    - 13F Parser:    Track quarterly holdings of Buffett, Citadel, 
                     Bridgewater, Soros, BlackRock, ARK, etc.
    - Form 4 Parser: Insider buys/sells (CEO/CFO/Directors)
                     → Filed within 2 days = fastest legal signal
    - 13D Parser:    Activist positions (>5% stake with intent)
    - 8-K Parser:    Buybacks, M&A, CEO departures
    
  Congressional Trading (FREE):
    - Quiver Quantitative API: politician trades + lobbying data
    - Capitol Trades scraper: backup source
    - House/Senate Stock Watcher: cross-reference
    
  Dark Pool Data (FREE, 2-week delay):
    - FINRA ATS download + parser
    - Dark pool % = dark_volume / total_volume
    
  Whale Proxy Detection (FREE, real-time from yfinance):
    - Volume spike: volume / SMA(20) > 3× = whale alert
    - Block trades: single trade > 10K shares
    - VWAP absorption: price above VWAP all day = big buyer
    - OBV divergence: price flat + OBV rising = quiet accumulation
    - Options vol/OI > 5× = someone betting big
    
  agents/institutional.py     — Compute all 17 factors:
    Dark pool volume/sentiment, block trades, options flow,
    13F changes, insider ratio, ETF flows, short squeeze prob,
    float reduction, dilution, activist 13D, holder concentration,
    spoofing detection, wash trading, congressional trades

Deliverable:
  >>> agent = InstitutionalAgent()
  >>> result = agent.analyze("NVDA", data)
  >>> result.probability_up       # 0.71
  >>> result.whale_signals        # ["Buffett added 2M shares (13F)",
                                  #  "CEO bought $500K (Form 4)",
                                  #  "Dark pool volume 45% (high)"]
  >>> result.manipulation_risk    # "LOW"
  >>> result.congressional_trades # [{"name": "Pelosi", "action": "BUY CALL"}]
```

**Day 23-24: Geopolitical + Currency Agents**
```
Files to create:
  agents/geopolitical.py      — GPR index, conflict scoring, sanctions, trade policy
  agents/currency.py          — DXY, revenue-weighted FX impact, carry trade risk
  data/currency.py            — FX pairs via yfinance
  data/alternative.py         — Fear&Greed, AAII, BDI, commodity prices
```

**Day 25: Orchestrator + Full Pipeline**
```
Files to create:
  agents/orchestrator.py      — Routes queries, manages parallel execution
  
Upgrade:
  agents/graph.py             — Full 12-agent LangGraph pipeline
                                 Orchestrator → 8 parallel → Debate → Risk

Deliverable:
  All 8 analysis agents run in parallel → Debate fuses → Risk approves
  Full signal packet with all 154 factors computed
```

### Week 6: Override System + Seasonality

**Day 26-27: Override System**
```
Upgrade:
  agents/risk.py              — Hard-coded override triggers:
                                 War → BEARISH, cap 35%
                                 Carry unwind → BEARISH all
                                 Manipulation → discount technicals
                                 Flash crash → halt 30min
                                 Black swan → emergency mode
```

**Day 28-29: Seasonality + Calendar Factors**
```
Upgrade:
  agents/technical.py         — Add seasonality sub-module:
                                 Day-of-week, month, OpEx, quarter-end,
                                 earnings proximity, ex-div, turn-of-month
```

**Day 30: Integration Testing**
```
  Full pipeline test: "Analyze NVDA" end-to-end
  Verify all 12 agents execute
  Verify override triggers work
  Verify signal packet is complete
```

---

## Phase 4: Portfolio + Dashboard (Weeks 7-8) — ✅ COMPLETED


### Week 7: Portfolio Agent

**Day 31-32: Position Tracking + Database**
```
Files to create:
  agents/portfolio.py         — Portfolio agent
  data/database.py            — SQLite: positions, trades, signals, performance

Features:
  - Holdings tracker (ticker, qty, entry, P&L)
  - Realized + unrealized P&L
  - Trade history
```

**Day 33-34: Portfolio Optimization**
```
Upgrade:
  agents/portfolio.py         — Add optimization methods:
                                 Mean-Variance (Markowitz)
                                 Black-Litterman
                                 Risk Parity
                                 Minimum Variance
                                 
  Risk constraints:
    - Sector max 30%
    - Position max 10%
    - Correlation monitoring
    - Drawdown protection
    - Cash reserve minimum
```

**Day 35: Performance Analytics**
```
Upgrade:
  agents/portfolio.py         — Performance tracking:
                                 Sharpe, Sortino, Max Drawdown,
                                 Alpha, Beta, Win Rate, Calmar
                                 
  Portfolio API endpoints:
    GET /portfolio
    POST /portfolio/add
    GET /portfolio/performance
    GET /portfolio/optimize
```

### Week 8: React Dashboard

**Day 36-37: Dashboard Foundation**
```
Files to create:
  frontend/                   — npx create-vite@latest ./ --template react
  frontend/src/App.jsx        — Main layout with sidebar nav
  frontend/src/components/SignalCard.jsx       — Conviction meter, badges
  frontend/src/components/AgentRadar.jsx       — 8-agent radar chart
  frontend/src/components/MonteCarloChart.jsx  — Plotly fan chart

Deliverable:
  Signal analysis page with:
  - Ticker input → triggers analysis
  - Signal card (direction, conviction, confidence)
  - Agent radar chart
  - Monte Carlo fan chart with CIs
  - Bull vs Bear debate reasoning
```

**Day 38-39: Portfolio + Risk Dashboards**
```
Files to create:
  frontend/src/components/PortfolioDashboard.jsx  — Holdings, allocation pie
  frontend/src/components/RiskDashboard.jsx       — VaR, drawdown, regime
  frontend/src/components/PerformanceChart.jsx    — Equity curve vs SPY
  frontend/src/pages/PortfolioPage.jsx
  frontend/src/pages/MarketPage.jsx
```

**Day 40: WebSocket Integration**
```
  Agent reasoning streams live via WebSocket
  Portfolio value updates in real-time
  Alert system for override triggers
```

---

## Phase 5: Backtesting + Polish (Weeks 9-10) — ✅ COMPLETED


### Week 9: Backtesting Harness

**Day 41-42: Walk-Forward Backtester**
```
Files to create:
  backtest/engine.py          — Walk-forward testing engine
  
Features:
  - Train on window N, test on window N+1, roll forward
  - Transaction cost modeling (slippage + commissions)
  - Out-of-sample validation
  - Benchmark comparison (SPY, QQQ)
```

**Day 43-44: Calibration + Agent Evaluation**
```
Files to create:
  backtest/calibration.py     — Probability calibration
  
Features:
  - Calibration plot (predicted vs actual frequency)
  - Brier score per agent
  - Isotonic regression auto-calibration
  - Information Coefficient (IC) per agent
  - Auto-tune Bayesian weights from backtest results
```

**Day 45: Stress Testing**
```
Files to create:
  backtest/stress_test.py     — Crisis scenario simulation
  
Scenarios:
  - 2008 GFC replay
  - 2020 COVID crash
  - Rate shock (+200bps)
  - Sector collapse (-30%)
  - Black Monday (-20% single day)
  - Custom user-defined
```

### Week 10: Polish + Documentation

**Day 46-47: Factor Exposure + Dashboard Polish**
```
Features:
  - Factor exposure tracking (value, momentum, quality, size, vol)
  - Monthly returns heatmap
  - Trade journal with reasoning
  - Agent accuracy tracker over time
  - Rolling Sharpe chart
```

**Day 48-49: Paper Trading Mode**
```
Features:
  - Live paper trading against real market
  - Track virtual P&L
  - Signal accuracy logging
  - Daily summary generation
```

**Day 50: Final Integration + Documentation**
```
  - Full end-to-end test
  - Update all documentation
  - README with quick start guide
  - Video demo recording
```

## Phase 6: Advanced Theory Scaling — ✅ COMPLETED

> **Goal**: Integrate TDA, Extreme Value Theory hardening, and Quantum MC.

**Theories to Unlock:**
- **Topological Data Analysis (TDA):** Persistent homology for market shape detection.
- **Extreme Value Theory (EVT):** GPD/POT for -10% tail risk prediction.
- **Hawkes Processes:** Modeling trade clustering and dark pool footprints.
- **Reinforcement Learning (RL):** SAC/PPO for automated portfolio rebalancing.
- **Quantum Monte Carlo:** Quadratic speedup for high-frequency option pricing.

---

## Phase Summary

| Phase | Status | Agents | Factors | Key Deliverable |
|---|---|---|---|---|
| **1** | ✅ | 4 | ~35 | CLI signal output + GARCH + Monte Carlo |
| **2** | ✅ | 7 | ~80 | Bayesian fusion + HMM + EVT + FastAPI |
| **3** | ✅ | 12 | ~154 | Full agent roster + overrides |
| **4** | ✅ | 12 | ~154 | Portfolio management + Dashboard |
| **5** | ✅ | 12 | ~154 | Backtesting + calibration + stress testing |
| **6** | ✅ | 12+ | ~200+ | TDA + Quasi-MC (Sobol) + RL Portfolio |

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Agent Framework | LangGraph + LangChain |
| LLM | Gemini 2.0 Flash (free) / GPT-4o-mini |
| Quant Libraries | arch, hmmlearn, scipy, numpy, ta, statsmodels |
| Market Data | yfinance, fredapi |
| News/RAG | NewsAPI, ChromaDB, sentence-transformers |
| Backend | FastAPI + WebSocket + uvicorn |
| Frontend | React + Vite + Plotly |
| Database | SQLite (aiosqlite) |
| Testing | pytest, pytest-asyncio |

---

## Prerequisites Before Starting

1. **LLM API Key** — Gemini (free) or OpenAI
2. **NewsAPI Key** — Free tier: newsapi.org
3. **FRED API Key** — Free: fred.stlouisfed.org
4. **Python 3.11+** installed
5. **Node.js 18+** installed (for React dashboard)

---

## Definition of Done

The system is "done" when:

- [x] `python api/main.py` produces a full signal packet and streams via WebSocket
- [x] Dashboard shows signal card + Monte Carlo chart + agent radar
- [x] Portfolio tracks holdings with Sharpe, Alpha, Max Drawdown
- [x] Backtester shows positive alpha vs SPY over 1 year out-of-sample
- [x] Probability calibration: when we say 70%, it actually happens ~70% of the time
- [x] Stress test: portfolio survives 2008/2020 replay without >25% drawdown
- [x] All override triggers fire correctly on historical crisis data
- [x] Advanced theories (TDA, EVT) integrated into core risk engine
