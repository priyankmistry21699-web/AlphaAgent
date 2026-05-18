# AlphaAgent — Complete Technical Reference Manual

> **Version 7.0 — Phase 6 + New Quant Theories Complete**  
> Updated: May 2026  
> Platform: Windows 11 · Python 3.11 · FastAPI · LangGraph  
> _This markdown is the authoritative updated reference. See `AlphaAgent_Technical_Reference.pdf` for the v6.1 baseline (pre-Phase 6 factor additions)._

---

## 1. System Overview

AlphaAgent is a multi-agent quantitative trading signal system combining **9 independent analytical perspectives** into a single probability-scored recommendation. The core design principles:

- **No black boxes** — every factor named, scored 0–100, and shown in the dashboard.
- **Correlation penalty** — agents sharing data sources are penalized in Bayesian fusion.
- **Hard circuit breakers** — Risk Agent overrides all agents with HALT (size=0) on Black Swan / Flash Crash / Hawkes cascade.
- **Dynamic confidence** — each agent's confidence weight (0–1) scales its evidence in Bayesian update.
- **Time-horizon awareness** — final blend reweighted by horizon map (1D→technical-dominant, 1Y→fundamental-dominant).

### Signal Pipeline

```
START
  ↓
[data_ingestion_node]  — pre-warms OHLCV(1y), financials, info, news (parallel)
  ↓
[run_agents_node]  — ThreadPoolExecutor(max_workers=9), all agents in parallel
  ├── TechnicalAgent      ├── FundamentalAgent    ├── MacroAgent
  ├── SentimentAgent      ├── InsiderAgent         ├── RiskAgent
  ├── GeopoliticalAgent   ├── VolatilityAgent      └── CurrencyAgent
  ↓
[portfolio_manager_node]
  — Risk circuit breaker check
  — BayesianFusion.update() × 8 agents (risk excluded from voting)
  — HMM regime detection
  — SignalDecay half-life
  — Horizon reweighting
  — Build SignalPacket
  ↓
END → HTTP Response: direction, P(up), conviction %, entropy, size multiplier
```

---

## 2. Factor Count by Agent (v7.0)

| Agent | Factors | Primary Theory | Key Data Sources |
|:------|:-------:|:--------------|:----------------|
| Technical | 30 | Price action, TDA, HMM, microstructure | yfinance OHLCV, options chain |
| Fundamental | **28** | DCF, FF5, CAPM, Piotroski, Altman, Beneish | yfinance financials, FRED |
| Macro | **28** | Yield curve, Fisher Real Rate, Sector RS, Contagion | FRED API, yfinance ETFs |
| Sentiment | **20** | NLP, Source Credibility, LLM Alignment, Entropy | NewsAPI, yfinance news, Gemini |
| Insider | **18** | Kyle's λ, Amihud, HHI, FINRA, Activist 13D | SEC EDGAR, yfinance, FINRA |
| Risk | 16 | EVT, GARCH, Hawkes, Quasi-MC, Kelly | yfinance OHLCV + options |
| Geopolitical | **18** | GPR, Active Conflict, Sanctions, Tariff NLP | FRED, yfinance, EDGAR |
| Volatility | **14** | GARCH, Heston, SABR, Rough Vol, Hawkes, MF-DFA | yfinance OHLCV + options |
| Currency | 12 | Carry trade, DXY regime, EM stress | yfinance FX pairs, FRED rates |
| **Total** | **~190** | | |

> **Bold** = updated/expanded in Phase 6. Previous total was ~163 (v6.1).

---

## 3. Phase 6 New Factors (14 additions)

### 3.1 Macro Agent — 3 new factors

| Factor | Method | Key Signal |
|:-------|:-------|:-----------|
| `sector_rs` | SPDR ETF (XLK/XLF/XLE…) 22d return vs SPY | RS > 1.0 = sector outperformance → bullish tailwind |
| `contagion_correlation` | Rolling 20d avg pairwise correlation of SPY/TLT/GLD/HYG vs historical | Spike >0.20 → cross-asset contagion → bearish |
| `real_interest_rate` | FRED DGS10 − T10YIE (Fisher real rate) | >1.5% headwind; <0% tailwind; <-1.0% strong tailwind |

**Dynamic data sources:** All three use FRED API (DGS10, T10YIE, NROU) and yfinance ETFs. No static values.

### 3.2 Fundamental Agent — 1 new factor

| Factor | Method | Key Signal |
|:-------|:-------|:-----------|
| `ma_activity` | EDGAR EFTS full-text search: 8-K filings with "merger"/"acquisition" keywords, 180d lookback | >2 hits → 75.0; 1-2 hits → 65.0; 0 hits → 50.0 |

**EDGAR endpoint:** `https://efts.sec.gov/LATEST/search-index?q="{ticker}"+"merger"&forms=8-K&dateRange=custom&startdt={180d_ago}`

### 3.3 Sentiment Agent — 2 new factors

| Factor | Method | Key Signal |
|:-------|:-------|:-----------|
| `source_credibility` | Tier-weighted average across last 20 news articles. WSJ/Bloomberg=1.0, Reuters=1.0, FT=1.0, Barrons=0.95, CNBC=0.85, MarketWatch=0.80, SeekingAlpha=0.70, MF=0.65, Forbes=0.65, BI=0.60, Benzinga=0.55, TheStreet=0.55, Zacks=0.60, Investopedia=0.50 | High-credibility coverage → more signal weight |
| `headline_body_alignment` | Gemini LLM dual-pass: HEADLINE_SCORE vs BODY_SCORE. Flags if body is >20pts more bearish than headline | ALIGNED+bullish→70; ALIGNED+neutral→50; DIVERGED+bearish body→30 |

**Note:** `headline_body_alignment` only activates if `GEMINI_API_KEY` env var is set.

### 3.4 Insider Agent — 4 new factors

| Factor | Method | Key Signal |
|:-------|:-------|:-----------|
| `finra_short_volume` | FINRA RegSHO weekly CSV `cdn.finra.org/equity/regsho/weekly/CNMSweekly{date}.txt`. Short vol / total vol | <45% → 70.0 bullish; 45–55% → 50.0; ≥55% → 25.0 bearish |
| `etf_flow_impact` | Sector ETF volume ratio (5d vs 17d avg) × 22d price momentum = flow signal | Positive momentum + volume surge = buying pressure |
| `activist_13d` | EDGAR EFTS SC 13D/SC 13G forms search, 90d lookback | ≥2 filings → 80.0; 1 filing → 65.0; 0 → 50.0 |
| `top10_concentration` | HHI = Σ(institutional_holder_fraction²) for top-10 holders from yfinance | 0.05–0.25 → 60.0 (healthy); >0.25 → 40.0 (concentrated risk) |

### 3.5 Geopolitical Agent — 3 new factors (single try-block)

| Factor | Keywords | Score Impact |
|:-------|:---------|:------------|
| `active_conflict` | war, conflict, military, invasion, attack, strike, troops, missile, combat, escalation | >3 hits → `prob_up -= 0.06` |
| `sanctions_risk` | sanction, embargo, ban, blacklist, ofac, export control, restricted, blocked entity | ≥2 hits → `prob_up -= 0.05` |
| `tariff_regulatory_risk` | tariff, trade war, import tax, quota, antidumping, protectionism + antitrust, DOJ probe, FTC investigation, SEC investigation, regulatory fine, GDPR, class action | combined >3 → `prob_up -= 0.04` |

**Data source:** `yf.Ticker(ticker).news` last 15 articles (free, no API key required).

---

## 4. Architecture — Key Files

```
AlphaAgent/
├── api/main.py              — FastAPI: /analyze/{ticker}, /chat, /health
├── orchestrator/graph.py    — LangGraph workflow: data→agents→portfolio_manager
├── agents/
│   ├── technical.py         — 30 factors: RSI, MACD, HMM, TDA, options microstructure
│   ├── fundamental.py       — 28 factors: FF5, CAPM, Piotroski, Altman, Beneish, M&A
│   ├── macro.py             — 28 factors: yield curve, FRED, Sector RS, Contagion, Real Rate
│   ├── sentiment.py         — 20 factors: NLP, Source Credibility, Gemini LLM, Entropy
│   ├── insider.py           — 18 factors: Form 4, Kyle λ, Amihud, HHI, FINRA, 13D
│   ├── risk.py              — 16 factors: MC VaR, CVaR, EVT, Kelly, Flash Crash, Hawkes
│   ├── geopolitical.py      — 18 factors: GPR, Oil, Active Conflict, Sanctions, Tariff
│   ├── volatility.py        — 14 factors: GARCH, EGARCH, SABR, Heston, Rough Vol
│   └── currency.py          — 12 factors: DXY, carry trade, EM stress, SOFR
├── quant_engine/
│   ├── bayesian.py          — Bayesian fusion engine
│   ├── causal_engine.py     — Causal DAG / do-calculus
│   ├── copula.py            — Gaussian + Clayton copulas
│   ├── granger.py           — Granger causality VAR F-test
│   ├── heston.py            — Heston stochastic volatility
│   ├── lob.py               — Limit order book microstructure
│   ├── multifractal.py      — MF-DFA multifractal analysis
│   ├── quantum_finance.py   — QAOA portfolio optimizer
│   ├── rough_vol.py         — Rough volatility (fBm, H≈0.1)
│   └── sabr.py              — SABR vol smile calibration
├── data/market.py           — MarketData: yfinance + FRED + ChromaDB + NewsAPI
├── database/manager.py      — SQLite trade/signal persistence
└── frontend-react/          — React dashboard
    └── src/
        ├── App.jsx           — Main app + routing
        ├── constants.js      — Agent theories, factor encyclopedia, ticker lists
        └── components/
            ├── Market.jsx    — Price chart + agent vote bars
            ├── Signal.jsx    — Main analysis dashboard
            ├── OtherTabs.jsx — Portfolio, History, Settings tabs
            └── QuantPanel.jsx — Quant Theory Index (34 theories, 190+ factors)
```

---

## 5. Bayesian Fusion Engine

**File:** `quant_engine/bayesian.py`

The Bayesian update equation applied for each agent:

```
P_new = (P_old × L_bullish) / ((P_old × L_bullish) + ((1−P_old) × L_bearish))
```

Where:
- `L_bullish = agent_prob_up × confidence_weight × (1 − correlation_penalty)`
- `L_bearish = (1 − agent_prob_up) × confidence_weight`
- `correlation_penalty` is applied when two agents share >60% of data sources

After all agents: isotonic regression calibration maps raw posterior → calibrated P(Up).

**Shannon entropy** of the final posterior distribution is computed and reported as `entropy` in SignalPacket. High entropy = agents disagree = wide confidence interval.

---

## 6. GARCH Volatility Model

**File:** `agents/volatility.py`

```python
GARCH(1,1): σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
EGARCH:     log(σ²_t) = ω + β·log(σ²_{t-1}) + α·|z_{t-1}| + γ·z_{t-1}
```

Regime thresholds (annualized vol):
| Regime | Vol Threshold | MC Paths | Kelly Cap |
|:-------|:------------:|:--------:|:---------:|
| LOW | < 15% | 3,000 | 20% |
| NORMAL | 15–30% | 5,000 | 15% |
| HIGH | 30–50% | 8,000 | 10% |
| EXTREME | > 50% | 10,000 | 5% |

---

## 7. Hidden Markov Model

**File:** `agents/technical.py`

3-state HMM (Bull / Bear / Crisis) fitted via Baum-Welch EM on 252 daily returns + realized vol. Viterbi decoding gives current state. Regime state probability feeds directly into Technical Agent factor score.

```python
States:  0=Bull,  1=Bear,  2=Crisis
Emissions: [daily_return, daily_vol_percentile]
Fit on: rolling 252-day window
```

---

## 8. Monte Carlo Simulation + EVT

**File:** `agents/risk.py`

GBM paths:
```
S(t+1) = S(t) · exp((μ - σ²/2)·Δt + σ·√Δt·Z)
```

where σ is the GARCH conditional vol and path count scales with vol regime.

**EVT (Generalized Pareto):** threshold = 95th percentile of negative returns. GPD fitted to exceedances. Extrapolates 99%/99.5% VaR beyond historical sample.

**Quasi-MC:** Sobol (2¹²=4,096) sequences replace pseudo-random for faster convergence. ~10× more efficient in filling probability space.

---

## 9. Factor Data Classes

```python
@dataclass
class FactorScore:
    name:           str     # Display name, e.g. "RSI (14)"
    value:          float   # Raw computed value
    score:          float   # Normalized 0–100  (50=neutral, 100=max bullish)
    interpretation: str     # Plain-English explanation

@dataclass
class AgentResult:
    agent_name:     str
    vote:           Direction    # LONG | SHORT | HOLD
    probability_up: float        # P(price up) ∈ [0.0, 1.0]
    confidence:     float        # ∈ [0.0, 1.0]
    reasoning:      str
    factor_scores:  dict[str, FactorScore]
    warnings:       list[str]
    computation_time_ms: float
```

Score convention: 0–100, where 50 = neutral, 100 = maximum bullish, 0 = maximum bearish.

---

## 10. Quant Theory Implementation Status (34 theories)

| # | Theory | Agent | Status |
|:--|:-------|:------|:------:|
| 1 | Bayesian Signal Fusion | Orchestrator | 🟢 |
| 2 | GARCH(1,1) + EGARCH | Volatility, Risk | 🟢 |
| 3 | Monte Carlo GBM + Quasi-MC (Sobol) | Risk | 🟢 |
| 4 | Hidden Markov Model (3-state) | Technical | 🟢 |
| 5 | Kalman Filter (dynamic beta) | Technical | 🟢 |
| 6 | Extreme Value Theory (GPD) | Risk | 🟢 |
| 7 | Kelly Criterion (regime-adaptive) | Risk | 🟢 |
| 8 | Topological Data Analysis (TDA) | Technical | 🟢 |
| 9 | Hawkes Self-Exciting Process | Volatility | 🟢 |
| 10 | Jegadeesh-Titman Momentum Engine | Technical | 🟢 |
| 11 | Options Intelligence (IV/skew/GEX) | Sentiment | 🟢 |
| 12 | Signal Decay (half-life model) | Orchestrator | 🟢 |
| 13 | Probability Calibration (isotonic) | Orchestrator | 🟢 |
| 14 | Factor Exposure (FF5 + CAPM alpha) | Fundamental | 🟢 |
| 15 | Quality Stack (Piotroski/Altman/Beneish) | Fundamental | 🟢 |
| 16 | SABR Volatility Model | Volatility | 🟢 |
| 17 | Heston Stochastic Volatility | Volatility | 🟢 |
| 18 | Rough Volatility (fBm, H≈0.1) | Volatility | 🟢 |
| 19 | Copula Dependency (Gaussian+Clayton) | Risk | 🟢 |
| 20 | Multifractal MF-DFA | Technical | 🟢 |
| 21 | Granger Causality (VAR F-test) | Macro | 🟢 |
| 22 | Causal DAG (do-calculus) | Macro | 🟢 |
| 23 | Limit Order Book (Kyle's λ, Amihud) | Technical, Insider | 🟢 |
| 24 | Quantum Finance (QAOA) | Portfolio optimizer | 🟢 |
| 25 | **Fisher Equation / Real Rate** | Macro | 🟢 |
| 26 | **Fama-French 5F + Quality Stack** | Fundamental | 🟢 |
| 27 | **Contagion Correlation Spike** | Macro | 🟢 |
| 28 | **Herfindahl-Hirschman Index (HHI)** | Insider | 🟢 |
| 29 | **FINRA Short Volume Proxy** | Insider | 🟢 |
| 30 | **Sector Relative Strength** | Macro | 🟢 |
| 31 | **Source Credibility + LLM Alignment** | Sentiment | 🟢 |
| 32 | **Activist Smart Money (13D)** | Insider | 🟢 |
| 33 | **Geopolitical Signal Stack** | Geopolitical | 🟢 |
| 34 | **Information Theory Stack** (Shannon+Transfer+KL) | Sentiment, Orchestrator | 🟢 |

> **Bold** = new in Phase 6 (May 2026). Previous count was 24 theories.

---

## 11. Static vs Dynamic Values Audit

All critical macro thresholds have been converted from static constants to dynamic FRED-sourced values:

| Parameter | Old (Static) | New (Dynamic) | FRED Series |
|:----------|:-------------|:-------------|:------------|
| NAIRU / natural unemployment | 4.5% (hardcoded) | `NROU` latest | NROU |
| r* neutral rate | 2.5% (hardcoded) | DFII10 + 2.0 | DFII10 |
| Risk-free rate (GEX) | 5.25% (hardcoded) | `FEDFUNDS` latest | FEDFUNDS |
| Real interest rate | computed from static 2% | DGS10 − T10YIE live | DGS10, T10YIE |
| M&A lookback | 180 days (constant) | `sf.get("ma_lookback_days", 180)` | — |
| Activist lookback | 90 days (constant) | configurable via settings | — |

---

## 12. API Reference

### POST /analyze/{ticker}

**Request:**
```json
{
  "horizon": "1w",
  "use_cache": true
}
```

**Response (SignalPacket):**
```json
{
  "ticker": "AAPL",
  "direction": "LONG",
  "conviction_pct": 42.3,
  "confidence": "HIGH",
  "probability_up": 0.711,
  "entropy": 0.83,
  "regime": "BULL",
  "holding_period": { "half_life_days": 5.2, "optimal_hold_min": 3, "optimal_hold_max": 10 },
  "override_active": false,
  "warnings": [],
  "agent_results": [
    {
      "agent_name": "TechnicalAgent",
      "vote": "LONG",
      "probability_up": 0.74,
      "confidence": 0.82,
      "factor_scores": { "rsi": { "name": "RSI (14)", "value": 63.2, "score": 68.0 } }
    }
  ]
}
```

### GET /health
Returns `{"status": "healthy", "agents": 9, "version": "7.0"}`.

### POST /chat
LangChain RAG chatbot with ChromaDB context. Accepts `{"message": "...", "ticker": "AAPL"}`.

---

## 13. Settings Configuration

Key settings in `config/settings.py` (or environment):

| Setting | Default | Description |
|:--------|:-------:|:-----------|
| `GEMINI_API_KEY` | `""` | Enables Gemini LLM in Sentiment Agent |
| `NEWSAPI_KEY` | `""` | Enables NewsAPI for news scoring |
| `FRED_API_KEY` | required | FRED data (yield curve, VIX, NAIRU, Real Rate) |
| `ma_lookback_days` | `180` | M&A activity EDGAR lookback |
| `activist_lookback_days` | `90` | Activist 13D EDGAR lookback |
| `gemini_model` | `"gemini-2.5-flash"` | Gemini model for headline/body alignment |
| `garch_vol_low` | `0.15` | GARCH LOW/NORMAL threshold |
| `garch_vol_high` | `0.30` | GARCH NORMAL/HIGH threshold |
| `garch_vol_extreme` | `0.50` | GARCH HIGH/EXTREME threshold |

---

## 14. Data Sources Summary

| Source | Free? | Used For |
|:-------|:-----:|:--------|
| yfinance | ✅ | OHLCV, financials, options, news, institutional holders |
| FRED API | ✅ | DGS10, T10YIE, FEDFUNDS, NROU, DFII10, WALCL, M2, CPI, etc. |
| SEC EDGAR EFTS | ✅ | Form 4, 8-K, SC 13D/G full-text search |
| FINRA RegSHO CSV | ✅ | Weekly short sale volume by ticker |
| NewsAPI (free tier) | ✅ | Sentiment Agent news articles |
| Gemini API | ✅ (free tier) | Headline/body alignment LLM |
| ChromaDB | ✅ | RAG vector store for chatbot |
| Bloomberg / FactSet | ❌ | GDP nowcast, earnings calendar, institutional positioning |
| Unusual Whales | ❌ | Premium options flow data |

---

## 15. Probability Accuracy Improvement Roadmap

Top 5 unimplemented theories by estimated accuracy gain:

1. **FinBERT NLP** (+15–20% sentiment accuracy) — replace keyword scan with `ProsusAI/finbert`.
2. **RL Position Sizing** (PPO/SAC) — learn dynamic multi-agent position sizes by maximizing Sharpe.
3. **Order Flow Imbalance** (real-time L2 LOB) — requires Polygon.io or IBKR L2 WebSocket feed.
4. **News Decay Model** — per-news-type half-life (earnings decay in hours, regulatory in weeks).
5. **Cross-Sectional Momentum Ranking** — rank ticker in top/bottom decile vs sector universe.

---

*Generated by Claude Code — AlphaAgent v7.0 Phase 6 Technical Reference*
