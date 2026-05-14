# 📈 AlphaAgent — Multi-Agent Quantitative Trading Intelligence

> A production-grade agentic AI platform where **8 specialized agents** collaborate to analyze stocks using advanced quantitative finance theories, producing **probability-scored buy/sell/hold signals** with full reasoning transparency.

⚠️ **Disclaimer**: This system produces educational analysis only, NOT financial advice.

---

## Architecture

```
User: "Analyze NVDA"
    ↓
🎯 Orchestrator Agent
    ↓ (parallel)
┌─────────────┬──────────────┬─────────────────┬──────────────┐
│ 📊 Market   │ 📉 Technical │ 📰 Sentiment    │ 🏦 Fundamental│
│    Data     │   Analysis   │   (News RAG)    │   Analysis   │
│ (yfinance)  │ (Monte Carlo │ (ChromaDB +     │ (P/E, DCF)   │
│             │  GARCH, HMM) │  LLM scoring)   │              │
└──────┬──────┴──────┬───────┴────────┬────────┴──────┬───────┘
       │             │               │               │
       └─────────────┴───────┬───────┴───────────────┘
                             ↓
                    ⚖️ Debate Agent
                    (Bayesian Fusion)
                    Bull vs Bear case
                             ↓
                    🛡️ Risk Agent
                    (EVT, VaR, Kelly)
                    Guardrails + approval
                             ↓
                    ✅ Final Signal
                    BUY/SELL/HOLD @ 73% conviction
```

## Quant Engine

| Module | Theory | Purpose |
|---|---|---|
| `monte_carlo.py` | Geometric Brownian Motion | P(price > X at time T) |
| `garch.py` | GARCH(1,1) + EGARCH | Volatility forecasting |
| `hmm.py` | Hidden Markov Models | Regime detection (Bull/Bear/Crisis) |
| `bayesian.py` | Bayesian Fusion | Optimal signal combination |
| `evt.py` | Extreme Value Theory | Tail risk (VaR, CVaR) |
| `kalman.py` | Kalman Filter | Signal smoothing |
| `technical.py` | RSI, MACD, Bollinger | Technical indicators |

## Quick Start

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Set up API keys
cp .env.example .env
# Edit .env with your API keys

# 3. Run analysis (CLI)
python -m agents.graph --ticker AAPL

# 4. Start the API server
uvicorn api.main:app --reload

# 5. Start the dashboard
cd frontend && npm run dev
```

## Documentation

- [📋 Implementation Plan](./IMPLEMENTATION_PLAN.md) — Full build plan (8 phases)
- [🔬 Dominant Quant Techniques](./QUANT_TECHNIQUES.md) — What actually works at top funds
- [📚 Quant Theories Deep Dive](./QUANT_THEORIES_DEEP_DIVE.md) — Mathematical foundations

## Tech Stack

| Component | Technology |
|---|---|
| Agent Framework | LangGraph + LangChain |
| LLM | Gemini 2.0 Flash / OpenAI GPT-4o-mini |
| Quant Libraries | arch, hmmlearn, scipy, numpy, ta |
| Market Data | yfinance, fredapi |
| Vector DB | ChromaDB |
| Backend | FastAPI + WebSocket |
| Frontend | React + Vite + Plotly |
| Database | SQLite |

## License

Educational use only. Not financial advice.
