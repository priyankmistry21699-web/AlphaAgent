<div align="center">

<img src="https://img.shields.io/badge/Phase-6%20Complete-22c55e?style=for-the-badge&logo=checkmarx&logoColor=white"/>
<img src="https://img.shields.io/badge/Agents-9%20Specialist-3b82f6?style=for-the-badge&logo=robot-framework&logoColor=white"/>
<img src="https://img.shields.io/badge/Factors-163%20Signals-8b5cf6?style=for-the-badge&logo=chartdotjs&logoColor=white"/>
<img src="https://img.shields.io/badge/LLM-Gemini%202.0%20Flash-f59e0b?style=for-the-badge&logo=google&logoColor=white"/>
<img src="https://img.shields.io/badge/Framework-LangGraph-ef4444?style=for-the-badge&logo=langchain&logoColor=white"/>

<br/><br/>

```
█████╗ ██╗     ██████╗ ██╗  ██╗ █████╗      █████╗  ██████╗ ███████╗███╗   ██╗████████╗
██╔══██╗██║     ██╔══██╗██║  ██║██╔══██╗    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
███████║██║     ██████╔╝███████║███████║    ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   
██╔══██║██║     ██╔═══╝ ██╔══██║██╔══██║    ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   
██║  ██║███████╗██║     ██║  ██║██║  ██║    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   
╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   
```

### **Production-Grade Multi-Agent Quantitative Trading Intelligence**
*9 specialist agents · 163 factors · 11 quant modules · real-time LLM reasoning · dynamic horizon weighting*

<br/>

> ⚠️ **Educational & Research Use Only — This is NOT financial advice.**

</div>

---

## 🗂️ Table of Contents

| | |
|---|---|
| [⚡ What is AlphaAgent?](#-what-is-alphagent) | [🏗️ System Architecture](#️-system-architecture) |
| [🤖 The 9 Specialist Agents](#-the-9-specialist-agents) | [📐 Quant Engine Modules](#-quant-engine-modules) |
| [⏱️ Dynamic Horizon Weighting](#️-dynamic-horizon-weighting) | [🛡️ Risk Circuit Breakers](#️-risk-circuit-breakers) |
| [🖥️ Dashboard Tabs](#️-dashboard-tabs) | [📡 API Reference](#-api-reference) |
| [🔌 Data Sources](#-data-sources) | [🚀 Quick Start](#-quick-start) |
| [📁 Project Structure](#-project-structure) | [📚 Technical Reference](#-technical-reference) |

---

## ⚡ What is AlphaAgent?

AlphaAgent is a **production-grade agentic AI trading signal system** that runs **9 specialist agents in parallel** using LangGraph, fuses their outputs through a **Bayesian correlation-adjusted blend**, and delivers a final **LONG / SHORT / NEUTRAL** probability signal — all with full reasoning transparency.

```
Every signal you get answers three questions:
  ① DIRECTION  →  LONG / SHORT / NEUTRAL
  ② CONFIDENCE →  probability score (0–100%) + conviction %
  ③ TIMING     →  holding period estimate (Ornstein-Uhlenbeck mean-reversion half-life)
```

**What makes it different from a simple screener:**

- 🧠 **LLM at inference time** — Gemini 2.0 Flash reads real news via RAG and scores sentiment live
- 📐 **Real quant math** — GARCH, EVT, Hawkes, TDA, Quasi-MC, Kalman — not just RSI thresholds
- ⚡ **Parallel agent execution** — all 9 agents run simultaneously via LangGraph DAG
- 🛡️ **Hard circuit breakers** — automatic halt on Black Swan (>5σ), Flash Crash, Carry Unwind
- ⏱️ **Horizon-aware** — agent weights shift automatically from 1D (technical-dominant) to 1Y (fundamental-dominant)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     USER INPUT  ──  ticker + horizon                    │
│                      1D · 1W · 1M · 3M · 6M · 1Y                      │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
               ┌────────────▼────────────┐
               │   🎯  ORCHESTRATOR       │
               │   LangGraph parallel    │
               │   agent execution       │
               └────────────┬────────────┘
                            │
    ┌───────────────────────┼───────────────────────┐
    ▼                       ▼                       ▼
┌─────────────┐   ┌─────────────────┐   ┌────────────────┐
│ 🔵 TECHNICAL │   │  🟢 FUNDAMENTAL │   │  🟡 MACRO      │
│  30 factors │   │   28 factors   │   │  26 factors   │
│─────────────│   │─────────────────│   │────────────────│
│ RSI · MACD  │   │ P/E · P/B · DCF │   │ CPI · Fed Rate │
│ BB · ADX    │   │ EPS · Revenue   │   │ Yield Curve    │
│ EMA 9/21    │   │ CAPM Alpha      │   │ VIX · ISM PMI  │
│ IV Skew 25Δ │   │ FF5-Factor      │   │ SOFR Spread    │
│ GEX · Pain  │   │ Div Growth      │   │ Bond-Equity ρ  │
│ Impl. Corr  │   │ Buyback Signal  │   │ M2 · WALCL     │
│ TDA Homology│   │ Accruals Ratio  │   │ Global Pre-Mkt │
│ Variance RP │   │ Altman Z-Score  │   │ SOX vs SPY     │
│ PCA Quality │   │ PCA Quality    │   │ PCA Quality   │
└─────────────┘   └─────────────────┘   └────────────────┘

┌─────────────┐   ┌─────────────────┐   ┌────────────────┐
│ 🟠 SENTIMENT │   │   🔴 INSIDER    │   │   🟣 RISK      │
│  14 factors │   │   11 factors   │   │  16 factors   │
│─────────────│   │─────────────────│   │────────────────│
│ News RAG LLM│   │ EDGAR Form 4   │   │ GARCH(1,1)     │
│ Short Int   │   │ Inst. Ownership │   │ EVT (99% VaR)  │
│ Fear & Greed│   │ Congressional  │   │ Monte Carlo    │
│ Analyst Con │   │ Kyle's Lambda   │   │ Kelly Criterion│
│ Transfer Ent│   │ Float Reduction │   │ Hawkes Process │
│ Shannon Ent │   │ Cluster 30-Day  │   │ Quasi-MC Sobol │
│ AAII Survey │   │ Short Squeeze  │   │ KL Divergence  │
│ Price Target│   │ Buyback Signal  │   │ Tail Ratio     │
│ UOA · VRP   │   │ 13F Net Change │   │ Flash Crash Det│
└─────────────┘   └─────────────────┘   └────────────────┘

┌─────────────┐   ┌─────────────────┐   ┌────────────────┐
│ 🌍 GEO-     │   │  💱 CURRENCY    │   │ 📊 VOLATILITY  │
│ POLITICAL   │   │   11 factors   │   │   9 factors   │
│  19 factors │   │─────────────────│   │────────────────│
│─────────────│   │ DXY Regime     │   │ GARCH Vol Reg  │
│ GPR Index   │   │ EUR/USD        │   │ Put/Call Ratio │
│ (Caldara)   │   │ USD/JPY Carry  │   │ IV vs Realized │
│ Oil (BZ=F)  │   │ USD/CNY Stress │   │ Kalman Beta    │
│ Election    │   │ EM FX Stress   │   │ VIX Term Struct│
│ Transport   │   │ Real Int. Rate │   │ VVIX           │
│ Commodity   │   │ GBP/USD        │   │ SKEW Index     │
│ PCA Quality │   │ Carry Trade    │   │ RV 10d/30d     │
└─────────────┘   └─────────────────┘   └────────────────┘
                            │
               ┌────────────▼────────────┐
               │  ⚖️  BAYESIAN FUSION     │
               │  Correlation-adjusted   │
               │  log-odds sequential    │
               │  probability blend      │
               └────────────┬────────────┘
                            │
               ┌────────────▼────────────┐
               │  ⏱️  HORIZON REWEIGHTING│
               │  1D → Technical ×3.0   │
               │  1Y → Fundamental ×3.0 │
               │  20% base anchoring     │
               └────────────┬────────────┘
                            │
               ┌────────────▼────────────┐
               │  🛡️  RISK CIRCUIT BREAK │
               │  Black Swan · EVT halt  │
               │  Kelly position sizing  │
               │  6-tier override cascade│
               └────────────┬────────────┘
                            │
               ┌────────────▼────────────┐
               │  ✅  FINAL SIGNAL       │
               │  LONG / SHORT / NEUTRAL │
               │  P(up) · Conviction %   │
               │  Holding Period (OU)    │
               └────────────────────────┘
```

---

## 🤖 The 9 Specialist Agents

| # | Agent | Color | Factors | Key Theories |
|---|-------|:-----:|:-------:|---|
| 1 | **Technical** | 🔵 | 30 | RSI, MACD, BB, ADX, EMA 9/21, TDA Persistent Homology, IV Skew, GEX, Max Pain, Variance Risk Premium, PCA Quality |
| 2 | **Fundamental** | 🟢 | 28 | DCF, Fama-French 5-Factor, CAPM Jensen's Alpha, Altman Z-Score, Accruals Ratio, PEG, FCF Yield, Dividend Growth |
| 3 | **Macro** | 🟡 | 26 | Yield Curve, CPI, Fed Funds, ISM PMI, WALCL, SOFR Spread, Bond-Equity Correlation, Amihud Illiquidity, GPR Index |
| 4 | **Sentiment** | 🟠 | 14 | News RAG (Gemini 2.0 Flash), Transfer Entropy, Shannon Entropy, AAII Contrarian, Price Target Upside, UOA |
| 5 | **Insider** | 🔴 | 11 | EDGAR Form 4, Kyle's Lambda, Float Reduction, Insider Cluster 30-Day, Short Squeeze, 13F Institutional |
| 6 | **Risk** | 🟣 | 16 | GARCH(1,1), EVT/GPD, Monte Carlo GBM, Kelly Criterion, Hawkes Process, Quasi-MC Sobol, KL Divergence |
| 7 | **Geopolitical** | 🌍 | 19 | Caldara-Iacoviello GPR (FRED), Oil Shock, Copper/Gold Ratio, Election Cycle, Transport (Dow Theory), ITA Defense RS |
| 8 | **Volatility** | 📊 | 9 | GARCH Regime, Put/Call Ratio, IV/RV (Variance RP), Kalman Dynamic Beta, VIX Term Structure, CBOE SKEW, VVIX |
| 9 | **Currency** | 💱 | 11 | DXY Regime, USD/JPY Carry Trade, USD/CNY Stress, EM FX Basket, Real Interest Rate, Petro-Currency (CAD/AUD) |

---

## 📐 Quant Engine Modules

| Module | Theory | Key Output |
|--------|--------|-----------|
| `garch.py` | **GARCH(1,1) / EGARCH** | Vol forecast, regime (LOW/NORMAL/HIGH/EXTREME), annualized σ |
| `hmm.py` | **Hidden Markov Models** | Bull / Bear / Crisis regime with transition probabilities |
| `monte_carlo.py` | **Geometric Brownian Motion** | 5000-path simulation, 68/95/99% CIs, prob_above_current |
| `quasi_mc.py` | **Quasi-Monte Carlo (Sobol)** | Low-discrepancy VaR — ~10× more efficient than pseudorandom MC |
| `bayesian.py` | **Bayesian Fusion** | Log-odds sequential update with correlation penalty → posterior P(up) |
| `evt.py` | **Extreme Value Theory (GPD)** | VaR99, CVaR99, tail index ξ, tail dependence λ_L |
| `kalman.py` | **Kalman Filter** | Time-varying dynamic beta, SPY correlation (state-space [α,β]) |
| `tda_signal.py` | **Topological Data Analysis** | Persistent homology (H0/H1) → TRENDING / CYCLIC / FRAGMENTED |
| `hawkes.py` | **Hawkes Self-Exciting Process** | Branching ratio α/β — cascade / near-critical / stable classification |
| `calibration.py` | **Platt Scaling** | Isotonic regression probability calibration |
| `technical.py` | **Classical Indicators** | RSI, MACD, Bollinger Bands, ADX, ATR, Stochastic |

---

## ⏱️ Dynamic Horizon Weighting

When you select a time horizon, **agent weights are automatically reblended** using:

```
P_final = 0.80 × P_horizon_weighted + 0.20 × P_base_orchestrator
```

| Horizon | 🔵 Technical | 📊 Volatility | 🟠 Sentiment | 🟡 Macro | 🟢 Fundamental | 🔴 Insider | 💱 Currency | 🌍 Geo |
|:-------:|:------------:|:-------------:|:------------:|:--------:|:--------------:|:----------:|:-----------:|:------:|
| **1D**  | ×3.0 | ×3.0 | ×2.0 | ×0.5 | ×0.2 | ×0.5 | ×0.5 | ×0.3 |
| **1W**  | ×2.5 | ×2.5 | ×1.8 | ×0.8 | ×0.3 | ×0.8 | ×0.8 | ×0.5 |
| **1M**  | ×1.0 | ×1.0 | ×1.0 | ×1.0 | ×1.0 | ×1.0 | ×1.0 | ×1.0 |
| **3M**  | ×0.5 | ×0.5 | ×0.7 | ×1.5 | ×2.0 | ×2.0 | ×1.2 | ×1.5 |
| **6M**  | ×0.3 | ×0.3 | ×0.5 | ×2.0 | ×2.5 | ×2.5 | ×1.5 | ×2.0 |
| **1Y**  | ×0.2 | ×0.2 | ×0.3 | ×2.5 | ×3.0 | ×3.0 | ×2.0 | ×2.5 |

> **Short horizons** → Technical & Volatility dominant (price momentum, options flow)
> **Long horizons** → Fundamental, Macro & Insider dominant (valuation, policy, smart money)

---

## 🛡️ Risk Circuit Breakers

The Risk Agent runs a **6-tier priority override cascade**. The first triggered override halts further checks:

| Priority | Override | Trigger Condition | Effect |
|:--------:|----------|-------------------|--------|
| 🔴 **1** | **BLACK SWAN** | Any \|Z-score\| in last 5 sessions **> 5σ** | `prob=0.10` · **Full halt** · size=0% |
| 🔴 **2** | **FLASH CRASH** | Ticker 1d **< −7%** OR SPY 1d **< −5%** | `prob=0.15` · **Full halt** · size=0% |
| 🟠 **3** | **GEO SHOCK** | VIX **> 35** AND (Gold **> +2%** OR Oil **> +5%**) | `prob=0.30` · size capped at **35%** |
| 🟡 **4** | **CARRY UNWIND** | USD/JPY **< 125** OR JPY surges **> 1.5%** in 1d | `prob=0.30` · SHORT bias · size=**50%** |
| 🟡 **5** | **GARCH EXTREME** | vol_regime=EXTREME OR VaR99 **< −5%** | `prob=0.20` · **Full halt** · size=0% |
| 🟢 **6** | **GARCH HIGH** | vol_regime=HIGH OR VaR95 **< −3%** | `prob=0.40` · **Half size** · size=50% |

---

## 🖥️ Dashboard Tabs

| Tab | Icon | Purpose |
|-----|:----:|---------|
| **Signal Analysis** | 🔵 | Full 9-agent deep research with 1D/1W/1M/3M/6M/1Y horizon selector, Top-50 quick-launch chips, smart company/ticker autocomplete, live TradingView chart |
| **Live Portfolio** | 🟢 | Real-time positions, unrealized P&L, portfolio composition chart, add/close positions |
| **Paper Trading** | 🟡 | Simulated order execution at live prices, trade log, running P&L — no real money |
| **Backtest** | 🟠 | Historical strategy simulation with Sharpe, Sortino, max drawdown, win rate, equity curve |
| **Walk-Forward** | 🔴 | Rolling out-of-sample validation — the real test of whether signals are predictive vs overfit |
| **Stress Test** | 🟣 | Scenario analysis: 2008 Crash · COVID-19 · 1987 Black Monday · Rate Shock · Oil Shock · Tech Bubble |
| **Quant Lab** | 🔵 | Interactive GARCH forecast · Monte Carlo paths · EVT tail risk · Quasi-MC Sobol comparison |
| **Leaderboard** | 🟢 | Agent accuracy ranking, confidence calibration scores, recent call history |
| **Optimizer** | 🟡 | Mean-variance portfolio weight optimization, efficient frontier visualization |

---

## 📡 API Reference

### Core Signal Endpoint
```http
GET /api/v1/signal/{ticker}?horizon=1m
```

| Parameter | Type | Values | Default |
|-----------|------|--------|---------|
| `ticker` | path | Any valid symbol (e.g. `AAPL`, `NVDA`) | required |
| `horizon` | query | `1d` `1w` `1m` `3m` `6m` `1y` | `1m` |

**Response:**
```jsonc
{
  "ticker":           "AAPL",
  "horizon":          "3m",
  "direction":        "LONG",           // LONG · SHORT · NEUTRAL
  "probability":      0.6312,           // horizon-reweighted P(up)
  "base_probability": 0.5891,           // raw Bayesian fusion output
  "conviction":       74.2,             // |prob - 0.5| × 200
  "multiplier":       0.85,             // position size multiplier from Risk Agent
  "entropy":          0.312,            // agent disagreement (0=unanimous, 1=random)
  "agents": [
    { "name": "technical", "vote": "LONG",
      "probability_up": 0.72, "confidence": 0.85,
      "reasoning": "...", "factor_scores": { ... } }
  ],
  "warnings":         [],
  "holding_period":   { "min_days": 3, "expected_days": 12, "max_days": 28 },
  "summary":          { "bull_agents": 6, "bear_agents": 2, "neutral_agents": 1 },
  "latency_ms":       42300
}
```

### All Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Dashboard (serves `frontend/index.html`) |
| `GET` | `/api/v1/signal/{ticker}?horizon=` | Full 9-agent signal |
| `GET` | `/api/v1/portfolio` | Live portfolio positions |
| `POST` | `/api/v1/paper/signal/{ticker}` | Signal + execute paper trade |
| `POST` | `/api/v1/backtest` | Historical backtest |
| `POST` | `/api/v1/chat` | LLM chat about a signal |
| `GET` | `/api/v1/quant/garch/{ticker}` | GARCH vol forecast only |
| `GET` | `/api/v1/quant/monte-carlo/{ticker}` | Monte Carlo simulation only |
| `GET` | `/api/v1/quant/evt/{ticker}` | EVT tail risk only |
| `GET` | `/api/v1/chart/history/{ticker}` | OHLCV bars (5m / 1h / 1d) |
| `GET` | `/api/v1/markets` | US + global index prices |
| `GET` | `/api/v1/leaderboard` | Agent accuracy rankings |
| `POST` | `/api/v1/optimizer` | Portfolio weight optimization |
| `POST` | `/api/v1/stress-test` | Scenario stress test |
| `WS` | `/ws/{ticker}` | Real-time agent reasoning stream |

---

## 🔌 Data Sources

| Source | Type | What It Provides |
|--------|:----:|-----------------|
| `yfinance` | 🔵 Market | OHLCV, options chain, fundamentals, insider transactions, institutional holders |
| `fredapi` | 🟡 Macro | CPI, SOFR, M2, WALCL, Fed Funds, TIPS breakeven, ISM PMI, AAII sentiment, GPR Index |
| `NewsAPI` | 🟠 News | Headlines for RAG sentiment pipeline (ChromaDB vector store) |
| `SEC EDGAR` | 🔴 Insider | Form 4 insider filings, 8-K material events, 13F institutional ownership |
| `House Stock Watcher` | 🟢 Congress | Congressional trading disclosures (House) — STOCK Act |
| `Senate Stock Watcher` | 🟢 Congress | Congressional trading disclosures (Senate) — STOCK Act |
| `Gemini 2.0 Flash` | 🟣 LLM | News RAG scoring (SCORE 0–100), signal chat assistant |
| `ChromaDB` | 🔵 Vector DB | Semantic search over news headlines for RAG retrieval |

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/priyankmistry21699-web/AlphaAgent.git
cd AlphaAgent

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Fill in your keys:
#   GEMINI_API_KEY=...
#   NEWSAPI_KEY=...
#   FRED_API_KEY=...

# 5. Start the server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8088

# 6. Open the dashboard
# → http://localhost:8088
```

### Required API Keys

| Key | Free Tier | Get It |
|-----|:---------:|--------|
| `GEMINI_API_KEY` | ✅ Yes | [aistudio.google.com](https://aistudio.google.com) |
| `NEWSAPI_KEY` | ✅ Yes (100 req/day) | [newsapi.org](https://newsapi.org) |
| `FRED_API_KEY` | ✅ Yes | [fred.stlouisfed.org/docs/api](https://fred.stlouisfed.org/docs/api/api_key.html) |

---

## 📁 Project Structure

```
AlphaAgent/
│
├── 🤖 agents/                        # 9 specialist agents
│   ├── technical.py                  # 30 factors — momentum, options flow, TDA
│   ├── fundamental.py                # 28 factors — valuation, quality, CAPM, FF5
│   ├── macro.py                      # 26 factors — rates, inflation, global macro
│   ├── sentiment.py                  # 14 factors — news RAG, social, options sentiment
│   ├── insider.py                    # 11 factors — EDGAR, congressional, microstructure
│   ├── risk.py                       # 16 factors — EVT, GARCH, Hawkes, KL divergence
│   ├── geopolitical.py               # 19 factors — GPR, oil, election cycle, Dow Theory
│   ├── volatility.py                 #  9 factors — IV surface, Kalman beta, VVIX, SKEW
│   ├── currency.py                   # 11 factors — DXY, carry trade, EM FX, real rates
│   └── base.py                       # BaseAgent: probability clamping, threshold logic
│
├── 🧮 quant_engine/                  # Mathematical core
│   ├── garch.py                      # GARCH(1,1) / EGARCH volatility forecasting
│   ├── hmm.py                        # Hidden Markov Model regime detection
│   ├── monte_carlo.py                # GBM stochastic path simulation (5000 paths)
│   ├── quasi_mc.py                   # Sobol quasi-random low-discrepancy VaR
│   ├── bayesian.py                   # Bayesian log-odds correlation-adjusted fusion
│   ├── evt.py                        # Extreme Value Theory (GPD + GEV)
│   ├── kalman.py                     # Kalman Filter dynamic beta estimation
│   ├── tda_signal.py                 # Topological Data Analysis (persistent homology)
│   ├── hawkes.py                     # Hawkes self-exciting jump process
│   └── calibration.py               # Platt scaling probability calibration
│
├── 🕸️ orchestrator/
│   └── graph.py                      # LangGraph DAG — parallel agent execution + fusion
│
├── 📊 backtest/
│   ├── engine.py                     # Historical backtesting with realistic cost model
│   ├── walk_forward.py               # Rolling out-of-sample walk-forward validation
│   └── stress_test.py               # 7 scenario stress tests (2008, COVID, 1987, ...)
│
├── 💹 trading/
│   ├── paper_trader.py               # Paper trading engine (SQLite-backed)
│   └── rl_rebalancer.py             # PPO reinforcement learning portfolio rebalancer
│
├── 🗄️ database/
│   └── manager.py                    # SQLite via SQLAlchemy (trades, portfolio, logs)
│
├── 📡 api/
│   └── main.py                       # FastAPI application — all REST + WebSocket endpoints
│
├── 🖥️ frontend/
│   └── index.html                    # Single-file HTML/CSS/JS dashboard (no build step)
│
├── ⚙️ config/
│   └── settings.yaml                 # 72 dynamic parameters (all thresholds configurable)
│
├── 📚 AlphaAgent_Technical_Reference.html   # 1978-line complete technical reference
├── 📚 AlphaAgent_Technical_Reference.pdf    # Print-ready PDF (MathJax-rendered equations)
└── .env                              # API keys (not committed)
```

---

## 📊 Key Performance Characteristics

| Metric | Value | Notes |
|--------|:-----:|-------|
| **Signal latency** | 30–60s | Dominated by LLM inference + market data fetch |
| **Agents running** | Parallel (9) | LangGraph concurrent DAG execution |
| **Factors evaluated** | 163 | Per signal, across all agents |
| **GARCH paths** | 5,000 | Monte Carlo simulation per signal |
| **Sobol paths** | 4,096 | Quasi-MC VaR (power-of-2 for uniformity) |
| **Backtest capital** | $100,000 | Default starting capital |
| **Transaction cost** | 5 bps one-way | Plus 2 bps bid-ask slippage |
| **Max position** | 20% of portfolio | Kelly-adjusted within this cap |

---

## 📚 Technical Reference

A **complete 300-page technical reference** is included covering every theory, formula, and parameter:

| Document | Format | Coverage |
|----------|--------|----------|
| `AlphaAgent_Technical_Reference.html` | Interactive HTML | All 30 chapters, MathJax equations, clickable TOC |
| `AlphaAgent_Technical_Reference.pdf` | Print-ready PDF | Same content, A4 layout with page numbers |

**Chapters include:** GARCH mathematics · Bayesian fusion log-odds · EVT/GPD tail theory · Kalman Filter state-space model · TDA persistent homology · Hawkes self-exciting process · Quasi-Monte Carlo Sobol · Kelly Criterion · all 9 agent factor breakdowns · settings reference · full API docs

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Agent Orchestration** | LangGraph + LangChain (parallel DAG) |
| **LLM Provider** | Google Gemini 2.0 Flash |
| **Quant Libraries** | `arch` (GARCH) · `hmmlearn` · `scipy` · `numpy` · `ta` · `gudhi` (TDA) |
| **Market Data** | `yfinance` · `fredapi` |
| **Options Data** | `yfinance` options chain (IV, OI, volume) |
| **Congressional Data** | House/Senate Stock Watcher API (STOCK Act, free) |
| **Vector DB** | ChromaDB (news RAG semantic search) |
| **Backend** | FastAPI + uvicorn |
| **Frontend** | Single-file HTML/CSS/JS — no build step, no npm |
| **Database** | SQLite via SQLAlchemy |
| **Deployment** | `uvicorn api.main:app --reload --port 8088` |

---

## ⚙️ Configuration

All 72 system parameters are configurable in `config/settings.yaml` without touching code:

```yaml
# Examples of configurable parameters
technical:
  rsi_window: 14          # RSI Wilder smoothing period
  bollinger_std: 2.0      # Bollinger Bands standard deviation

volatility:
  put_call_overbought: 1.2   # P/C ratio → bearish hedging signal
  vvix_extreme: 120          # VVIX → extreme vol-of-vol

fundamental:
  pe_cheap: 15            # P/E below → undervalued signal
  dcf_upside_strong: 30   # DCF upside % → strong bullish

backtest:
  initial_capital: 100000
  transaction_cost_bps: 5.0
  max_position_pct: 0.20

agent_defaults:
  long_threshold: 0.55    # prob_up above → LONG vote
  short_threshold: 0.45   # prob_up below → SHORT vote
```

---

<div align="center">

**Built for learning, research, and exploration of quantitative finance.**

⚠️ *This system is for educational purposes only. It is not a registered investment advisor and does not constitute financial advice. All trading involves risk of loss.*

<br/>

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.1+-FF6B35?style=flat-square&logo=langchain&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-3-003B57?style=flat-square&logo=sqlite&logoColor=white)
![License](https://img.shields.io/badge/License-Educational-gray?style=flat-square)

</div>
