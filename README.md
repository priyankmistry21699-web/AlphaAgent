# AlphaAgent — Multi-Agent Quantitative Trading Intelligence

> A production-grade agentic AI platform where **9 specialized agents** collaborate using **163 factors** across advanced quantitative finance theories, producing probability-scored trading signals with full reasoning transparency and **dynamic time-horizon weighting**.

⚠️ **Disclaimer**: Educational analysis only — NOT financial advice.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          USER INPUT  ─  ticker + horizon                     │
│                       1D · 1W · 1M · 3M · 6M · 1Y                          │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                   ┌────────────▼────────────┐
                   │   🎯  Orchestrator       │
                   │   LangGraph parallel     │
                   │   agent execution        │
                   └────────────┬────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
 ┌────────▼───────┐   ┌─────────▼──────┐   ┌─────────▼──────┐
 │  🔵 TECHNICAL  │   │  🟢 FUNDAMENTAL│   │  🟡 MACRO      │
 │  30 factors    │   │  28 factors    │   │  26 factors    │
 │                │   │                │   │                │
 │ RSI · MACD     │   │ P/E · P/B      │   │ CPI · Fed Rate │
 │ BB · ADX       │   │ DCF · EPS      │   │ Yield Curve    │
 │ EMA 9/21       │   │ CAPM α         │   │ VIX · ISM PMI  │
 │ IV Skew (25Δ)  │   │ FF5-Factor     │   │ SOFR Spread    │
 │ GEX · Max Pain │   │ Div Growth     │   │ Bond-Equity ρ  │
 │ Impl. Corr.    │   │ Lockup Expiry  │   │ M2 · WALCL     │
 │ TDA · Momentum │   │ Buyback Signal │   │ Global Pre-Mkt │
 │ Variance RP    │   │ Accruals Ratio │   │ SOX vs SPY     │
 │ PCA Quality    │   │ PCA Quality    │   │ PCA Quality    │
 └────────────────┘   └────────────────┘   └────────────────┘

 ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
 │  🟠 SENTIMENT  │   │  🔴 INSIDER    │   │  🟣 RISK       │
 │  18 factors    │   │  14 factors    │   │  16 factors    │
 │                │   │                │   │                │
 │ News RAG (LLM) │   │ EDGAR Form 4   │   │ GARCH(1,1)     │
 │ Short Interest │   │ Inst. Ownership│   │ EVT (99% VaR)  │
 │ Fear & Greed   │   │ Congressional  │   │ Monte Carlo    │
 │ Analyst Census │   │ Short Squeeze  │   │ Kelly Criterion│
 │ Transfer Ent.  │   │ Kyle's Lambda  │   │ Hawkes Process │
 │ Shannon Ent.   │   │ Float Reduction│   │ Quasi-MC Sobol │
 │ AAII Sentiment │   │ Cluster 30-Day │   │ KL Divergence  │
 │ Price Target   │   │ Buyback Signal │   │ Tail Ratio     │
 │ Consumer Credit│   │                │   │ Flash Crash Det│
 └────────────────┘   └────────────────┘   └────────────────┘

 ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
 │  🌍 GEO-       │   │  💱 CURRENCY   │   │  📊 VOLATILITY │
 │  POLITICAL     │   │  12 factors    │   │  12 factors    │
 │  15 factors    │   │                │   │                │
 │                │   │ DXY Regime     │   │ GARCH Vol      │
 │ GPR Index      │   │ EUR/USD        │   │ Put/Call Ratio │
 │ (Caldara-Iac.) │   │ USD/JPY Carry  │   │ IV vs Realized │
 │ Oil (BZ=F)     │   │ USD/CNY Stress │   │ Kalman Beta    │
 │ Election Cycle │   │ EM FX Stress   │   │ Term Structure │
 │ Transport Idx  │   │ Real Int. Rate │   │ VIX Regime     │
 │ Commodity Idx  │   │ GBP/USD        │   │ Skew Regime    │
 │ PCA Quality    │   │ Carry Trade    │   │                │
 └────────────────┘   └────────────────┘   └────────────────┘
                                │
                   ┌────────────▼────────────┐
                   │  ⚖️  Bayesian Fusion     │
                   │  Correlation-adjusted    │
                   │  probability blend       │
                   └────────────┬────────────┘
                                │
                   ┌────────────▼────────────┐
                   │  ⏱️  Horizon Reweighting │
                   │  1D→Technical dominant   │
                   │  1Y→Fundamental dominant │
                   └────────────┬────────────┘
                                │
                   ┌────────────▼────────────┐
                   │  🛡️  Risk Circuit Breaker│
                   │  Black Swan · EVT halt   │
                   │  Kelly position sizing   │
                   └────────────┬────────────┘
                                │
                   ┌────────────▼────────────┐
                   │  ✅  Final Signal        │
                   │  LONG / SHORT / NEUTRAL  │
                   │  P(up) · Conviction %    │
                   └─────────────────────────┘
```

---

## Quant Engine Modules

| Module | Color | Theory | Purpose |
|---|:---:|---|---|
| `garch.py` | 🔴 | GARCH(1,1) / EGARCH | Volatility forecasting & regime |
| `hmm.py` | 🔵 | Hidden Markov Models | Bull / Bear / Crisis regime detection |
| `monte_carlo.py` | 🟣 | Geometric Brownian Motion | Stochastic path simulation |
| `quasi_mc.py` | 🟣 | Quasi-Monte Carlo (Sobol) | Low-discrepancy VaR estimation |
| `bayesian.py` | 🔵 | Bayesian Fusion | Correlation-adjusted signal blend |
| `evt.py` | 🔴 | Extreme Value Theory (GPD) | Tail risk — VaR / CVaR |
| `kalman.py` | 🔵 | Kalman Filter | Dynamic beta / signal smoothing |
| `tda_signal.py` | 🟣 | Topological Data Analysis | Persistent homology price cycles |
| `hawkes.py` | 🟣 | Hawkes Self-Exciting Process | Jump / cascade branching ratio |
| `calibration.py` | 🔵 | Platt Scaling | Probability calibration |
| `technical.py` | 🟡 | RSI / MACD / BB / ADX | Classical technical indicators |

---

## Dynamic Horizon Weighting

When you select a time horizon, agent weights are reblended:

| Horizon | 🔵 Technical | 📊 Volatility | 🟠 Sentiment | 🟢 Fundamental | 🟡 Macro | 🔴 Insider | 🌍 Geo |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1D** | ×3.0 | ×3.0 | ×2.0 | ×0.2 | ×0.5 | ×0.5 | ×0.3 |
| **1W** | ×2.5 | ×2.5 | ×1.8 | ×0.3 | ×0.8 | ×0.8 | ×0.5 |
| **1M** | ×1.0 | ×1.0 | ×1.0 | ×1.0 | ×1.0 | ×1.0 | ×1.0 |
| **3M** | ×0.5 | ×0.5 | ×0.7 | ×2.0 | ×1.5 | ×2.0 | ×1.5 |
| **6M** | ×0.3 | ×0.3 | ×0.5 | ×2.5 | ×2.0 | ×2.5 | ×2.0 |
| **1Y** | ×0.2 | ×0.2 | ×0.3 | ×3.0 | ×2.5 | ×3.0 | ×2.5 |

Short horizons are dominated by technical/volatility signals. Long horizons are dominated by fundamentals, macro, and insider activity.

---

## Dashboard Tabs

| Tab | Color | Purpose |
|---|:---:|---|
| Signal Analysis | 🔵 | Full 9-agent deep research with horizon selector |
| Live Portfolio | 🟢 | Real-time portfolio positions and P&L |
| Paper Trading | 🟡 | Simulated order execution and tracking |
| Backtest | 🟠 | Historical strategy backtesting |
| Walk-Forward | 🔴 | Out-of-sample walk-forward validation |
| Stress Test | 🟣 | Scenario analysis (crash / rate shock / etc.) |
| Quant Lab | 🔵 | Interactive GARCH / Monte Carlo / EVT tools |
| Leaderboard | 🟢 | Agent performance ranking |
| Optimizer | 🟡 | Portfolio weight optimization |

---

## Data Sources

| Source | Color | What |
|---|:---:|---|
| `yfinance` | 🔵 | OHLCV, options chain, fundamentals, insider trades |
| `fredapi` | 🟡 | FRED macroeconomic series (CPI, SOFR, M2, WALCL, GPR) |
| `NewsAPI` | 🟠 | News headlines for RAG sentiment scoring |
| SEC EDGAR | 🔴 | Form 4 insider filings, 8-K material events |
| House Stock Watcher | 🟢 | Congressional trading disclosures (House) |
| Senate Stock Watcher | 🟢 | Congressional trading disclosures (Senate) |
| ChromaDB | 🔵 | Vector store for news RAG |
| Gemini 2.0 Flash | 🟣 | LLM for sentiment scoring & signal chat |

---

## Quick Start

```bash
# 1. Clone and create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Fill in: GEMINI_API_KEY, NEWSAPI_KEY, FRED_API_KEY

# 4. Start the server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8088

# 5. Open dashboard
# Navigate to http://localhost:8088
```

---

## API Reference

### `GET /api/v1/signal/{ticker}`

Run full 9-agent analysis on a ticker.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ticker` | path | required | Stock symbol (e.g. `AAPL`) |
| `horizon` | query | `1m` | Time horizon: `1d` `1w` `1m` `3m` `6m` `1y` |

**Response fields:**

```json
{
  "ticker": "AAPL",
  "horizon": "3m",
  "direction": "LONG",
  "probability": 0.6312,
  "base_probability": 0.5891,
  "conviction": 74.2,
  "multiplier": 0.85,
  "entropy": 0.312,
  "agents": [...],
  "warnings": [],
  "holding_period": {...},
  "summary": {...},
  "latency_ms": 42300
}
```

### Other endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/portfolio` | GET | Live portfolio positions |
| `/api/v1/paper/signal/{ticker}` | POST | Run signal + paper trade |
| `/api/v1/backtest` | POST | Run historical backtest |
| `/api/v1/chat` | POST | AI chat about a signal |
| `/api/v1/quant/garch/{ticker}` | GET | GARCH volatility forecast |
| `/api/v1/quant/monte-carlo/{ticker}` | GET | Monte Carlo simulation |
| `/api/v1/quant/evt/{ticker}` | GET | EVT tail risk |
| `/api/v1/health` | GET | Server health check |

---

## Project Structure

```
AlphaAgent/
├── agents/                  # 9 specialist agents
│   ├── technical.py         # 30 factors — momentum, options, TDA
│   ├── fundamental.py       # 28 factors — valuation, quality, CAPM
│   ├── macro.py             # 26 factors — rates, inflation, global
│   ├── sentiment.py         # 18 factors — news RAG, social, options flow
│   ├── insider.py           # 14 factors — EDGAR, congressional, microstructure
│   ├── risk.py              # 16 factors — EVT, GARCH, Hawkes, KL divergence
│   ├── geopolitical.py      # 15 factors — GPR, oil, election cycle
│   ├── volatility.py        # 12 factors — IV surface, Kalman beta
│   └── currency.py          # 12 factors — DXY, carry trade, EM FX
├── quant_engine/            # Mathematical core
│   ├── garch.py             # GARCH(1,1) / EGARCH
│   ├── hmm.py               # Hidden Markov Model
│   ├── monte_carlo.py       # GBM simulation
│   ├── quasi_mc.py          # Sobol quasi-random VaR
│   ├── bayesian.py          # Bayesian fusion
│   ├── evt.py               # Extreme Value Theory
│   ├── kalman.py            # Kalman filter
│   ├── tda_signal.py        # Topological Data Analysis
│   ├── hawkes.py            # Hawkes self-exciting process
│   └── calibration.py       # Platt scaling calibration
├── orchestrator/
│   └── graph.py             # LangGraph agent orchestration
├── backtest/
│   ├── engine.py            # Historical backtesting
│   └── walk_forward.py      # Walk-forward validation
├── trading/
│   ├── paper_trader.py      # Paper trading simulation
│   └── rl_rebalancer.py     # RL-based portfolio rebalancer
├── database/
│   └── manager.py           # SQLite via SQLAlchemy
├── data/
│   └── market.py            # MarketData wrapper (yfinance)
├── api/
│   └── main.py              # FastAPI application
├── frontend/
│   └── index.html           # Single-file HTML dashboard
├── settings.yaml            # 72 dynamic parameters
└── .env                     # API keys (not committed)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent Framework | LangGraph + LangChain |
| LLM Provider | Google Gemini 2.0 Flash |
| Quant Libraries | `arch` · `hmmlearn` · `scipy` · `numpy` · `ta` |
| Market Data | `yfinance` · `fredapi` |
| Options Data | `yfinance` options chain |
| Congressional Data | House/Senate Stock Watcher (STOCK Act, free) |
| Vector DB | ChromaDB |
| Backend | FastAPI + uvicorn |
| Frontend | Single-file HTML/CSS/JS (no build step) |
| Database | SQLite via SQLAlchemy |

---

## License

Educational use only. Not financial advice.
