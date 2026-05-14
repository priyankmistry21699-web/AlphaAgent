# AlphaAgent vs Claude Trading Agents — Design for a Superior System

---

## Part 1: What Claude Trading Agents Actually Do

### The Current Landscape

There are several Claude-based trading agent projects:

| Project | What It Does | Limitation |
|---|---|---|
| **claude-trading-skills** | Modular "skills" for market review, trade planning, journaling | Process tool, no quant math |
| **claude-equity-research** | Generates analyst-style Buy/Sell/Hold reports | LLM opinion, not probabilistic |
| **cbt-framework** | AI-powered backtesting via Claude Code CLI | Backtesting only, no live signals |
| **AgentTrader** | Autonomous crypto Buy/Sell/Hold from price + news | Single-agent, no quant engine |

### How They Work (Architecture)

```
User: "Should I buy NVDA?"
        ↓
┌─────────────────────────────────────────┐
│         Claude LLM (Single Agent)       │
│                                         │
│  1. Fetch price data (yfinance/Alpaca)  │
│  2. Compute basic indicators (RSI, MA)  │
│  3. Read recent news headlines          │
│  4. LLM reasons about the data          │
│  5. Output: "BUY" / "SELL" / "HOLD"     │
│  6. Log decision to journal             │
└─────────────────────────────────────────┘
        ↓
"Based on the bullish MACD crossover and positive 
 earnings sentiment, I recommend BUY with moderate
 confidence. Consider a stop-loss at $120."
```

### Their Strengths
- ✅ Simple to set up (Claude Code + API keys)
- ✅ Natural language reasoning (explains WHY)
- ✅ Scheduled automation (cron routines)
- ✅ Journaling for self-improvement ("dreaming")
- ✅ MCP integration for live data

### Their Critical Weaknesses

| Weakness | Why It Matters |
|---|---|
| **No real math** | LLM "vibes" ≠ probability. Saying "moderate confidence" means nothing mathematically |
| **Single-agent** | One LLM doing everything → context drift, hallucination risk |
| **No regime awareness** | Same logic in bull markets and crashes |
| **No tail risk** | Can't model "what if it drops 15% tomorrow?" |
| **No calibrated probabilities** | "High confidence" from an LLM is not calibrated — it's often wrong when confident |
| **No holding period** | "Buy" but for how long? No mathematical basis |
| **No position sizing** | No Kelly Criterion, no volatility scaling |
| **No backtesting integration** | Can't validate if signals would have worked historically |
| **Overfits to narrative** | LLMs are storytellers — they construct plausible narratives, not statistical edges |

---

## Part 2: Why Your Instinct Is Right — Probabilities > Price Targets

### The Price Prediction Trap

```
❌ BAD: "NVDA will be $142.50 next week"
   → This is almost certainly wrong
   → Gives false precision
   → No actionable risk management

✅ GOOD: "73% probability NVDA moves up 2-5% within 5 days
          under current bull regime. 
          VaR(99%): max downside -4.2%.
          Optimal hold: 3-7 days.
          Kelly size: 8% of portfolio."
   → Actionable
   → Honest about uncertainty  
   → Includes risk limits
   → Tells you WHEN to exit
```

### What Professional Quants Actually Output

Real quant systems don't output prices. They output **decision packets**:

```
┌─────────────────────────────────────────────────────┐
│                 SIGNAL PACKET                        │
│                                                      │
│  Direction:     LONG                                │
│  Conviction:    73% (posterior probability)          │
│  Confidence:    HIGH (entropy = 0.42)               │
│                                                      │
│  Expected Move:                                      │
│    68% CI:  +1.2% to +4.8%  (1σ range)             │
│    95% CI:  -2.1% to +7.3%  (2σ range)             │
│    99% CI:  -4.2% to +9.1%  (EVT tail)             │
│                                                      │
│  Holding Period:                                     │
│    Optimal:  3-7 trading days                        │
│    Signal decay: 50% after 12 days                   │
│    Max hold: 15 days (signal expires)                │
│                                                      │
│  Risk Metrics:                                       │
│    VaR(95%):  -2.1%  ($4,200 on $200K position)     │
│    CVaR(99%): -4.2%  ($8,400 worst-case avg)        │
│    Max Drawdown (Monte Carlo): -6.8%                │
│                                                      │
│  Position Sizing:                                    │
│    Kelly Criterion:  12.4% of capital               │
│    Half-Kelly (safer): 6.2% of capital              │
│    Vol-adjusted: 5.8% of capital                     │
│                                                      │
│  Current Regime: BULL (HMM: P=0.78)                 │
│  Regime Risk: P(transition to Bear) = 0.06          │
│                                                      │
│  Signal Sources:                                     │
│    Technical:    +0.68 (bullish MACD + RSI)          │
│    Sentiment:    +0.45 (positive earnings tone)      │
│    Fundamental:  +0.22 (slightly undervalued)        │
│    Macro:        +0.55 (risk-on environment)         │
│    Combined α:   +0.52 (Bayesian posterior)          │
└─────────────────────────────────────────────────────┘
```

**This is what AlphaAgent should output.** Not "BUY" with a hand-wavy explanation.

---

## Part 3: How Each Output Component Is Computed

### 1. Direction + Conviction (Bayesian Posterior)

```
Inputs from each agent:
  Technical Agent → P(up | technicals) = 0.68
  Sentiment Agent → P(up | news) = 0.62  
  Fundamental Agent → P(up | fundamentals) = 0.55
  Macro Agent → P(up | macro) = 0.58

Bayesian Fusion:
  Prior: P(up) = 0.52 (historical base rate, slight bull bias)
  
  For each agent signal Sᵢ:
    Likelihood ratio: LRᵢ = P(Sᵢ | up) / P(Sᵢ | down)
    
  Combined posterior:
    Posterior odds = Prior odds × LR₁ × LR₂ × LR₃ × LR₄
    
  Adjustments:
    - Correlation penalty (signals from same data source are correlated)
    - Confidence weighting (high-entropy agents get less weight)
    - Regime adjustment (different priors in bull/bear)

Output: P(up) = 0.73, P(down) = 0.27
Direction: LONG
Conviction: 73%
```

### 2. Confidence (Entropy-Based)

```
Entropy: H = -P(up)·log₂(P(up)) - P(down)·log₂(P(down))

H = 0.0    → 100% confident (one signal dominates — suspicious)
H = 0.5    → moderate conflict between signals
H = 1.0    → maximum uncertainty (50/50 — don't trade)

Confidence mapping:
  H < 0.6  → HIGH confidence
  H < 0.8  → MEDIUM confidence  
  H < 0.9  → LOW confidence
  H ≥ 0.9  → NO SIGNAL (too uncertain, skip trade)
```

> [!IMPORTANT]
> Conviction (73%) and Confidence (HIGH) are DIFFERENT things.
> - **Conviction** = "how bullish?" (direction strength)
> - **Confidence** = "how sure are we about the conviction?" (agreement between agents)
> A 73% conviction with LOW confidence means the agents disagree — risky trade.

### 3. Expected Move (Monte Carlo + Options-Implied)

```
Method 1: Monte Carlo Simulation
  For i = 1 to 10,000:
    S(T) = S(0) × exp[(μ - σ²/2)·T + σ·√T·Z]
    where σ = GARCH forecast (not historical vol!)
    
  68% CI: [percentile_16(simulations), percentile_84(simulations)]
  95% CI: [percentile_2.5(simulations), percentile_97.5(simulations)]
  99% CI: [percentile_0.5(simulations), percentile_99.5(simulations)]

Method 2: Options-Implied Expected Move (if options data available)
  Expected_Move = Price × IV × √(Days/365)
  
  68% range: Price ± 1.0 × Expected_Move
  95% range: Price ± 1.96 × Expected_Move

Method 3: Regime-Conditional Monte Carlo (BEST)
  Use HMM regime probabilities to weight simulations:
    P(Bull)=0.78 → 78% of sims use bull parameters (μ=+0.08%, σ=0.8%)
    P(Bear)=0.16 → 16% of sims use bear parameters (μ=-0.05%, σ=1.5%)
    P(Crisis)=0.06 → 6% of sims use crisis parameters (μ=-0.3%, σ=3.5%)
```

### 4. Optimal Holding Period

```
Method 1: Signal Decay Analysis
  Compute autocorrelation of your alpha signal at different lags:
    AC(1 day) = 0.85  → signal still strong
    AC(3 days) = 0.62 → decaying
    AC(7 days) = 0.35 → weak
    AC(15 days) = 0.10 → expired
    
  Half-life = t where AC(t) = 0.5 × AC(0)
  Optimal hold ≈ half-life to 2× half-life

Method 2: Ornstein-Uhlenbeck Half-Life (for mean-reversion signals)
  dx = θ(μ - x)dt + σdW
  Half-life = ln(2) / θ
  
  If half-life = 5 days → close position around day 5-10

Method 3: Profit Target / Stop-Loss Based
  Given expected move and vol forecast:
    Profit target = Expected_Move × 0.7 (take 70% of expected move)
    Stop-loss = -VaR(95%)
    
  Average time to hit either = Monte Carlo simulation
    Run 10K paths, track when each path first hits target or stop
    Median time = optimal holding period
```

### 5. Risk Metrics (EVT + Kelly)

```
VaR (Value at Risk):
  Using EVT (Generalized Pareto Distribution):
    Fit GPD to historical losses exceeding threshold u
    VaR(α) = u + (σ/ξ) × [(n/Nᵤ × (1-α))^(-ξ) - 1]
    
  VaR(95%): "There's a 5% chance of losing more than X%"
  VaR(99%): "There's a 1% chance of losing more than X%"

CVaR (Expected Shortfall):
  CVaR(α) = VaR(α) / (1-ξ) + (σ - ξ·u) / (1-ξ)
  "IF the loss exceeds VaR, the average loss will be CVaR"

Kelly Criterion (Position Sizing):
  f* = (p·b - q) / b
  
  Where:
    p = probability of winning (from Bayesian posterior)
    q = 1 - p = probability of losing
    b = win/loss ratio (from expected move / stop-loss)
    
  Example: p=0.73, b=2.0 (2:1 reward/risk)
    f* = (0.73 × 2.0 - 0.27) / 2.0 = 0.595 = 59.5%
    Half-Kelly = 29.7% (much safer in practice)
    Vol-adjusted = f* × (target_vol / realized_vol)
```

### 6. Regime Detection (HMM)

```
Hidden Markov Model with 3-4 states:
  
  Current regime probabilities (from Forward algorithm):
    P(Bull)    = 0.78  → Use momentum signals
    P(Bear)    = 0.16  → Use defensive signals
    P(Crisis)  = 0.06  → Use tail hedging
    
  Transition forecast (from transition matrix):
    P(stay in Bull next week) = 0.94
    P(Bull → Bear) = 0.05
    P(Bull → Crisis) = 0.01
    
  Strategy adjustment by regime:
    Bull:   Full signal weight, larger positions
    Bear:   Invert momentum, reduce size, tighten stops
    Crisis: Minimum positions, max hedging, cash up
```

---

## Part 4: AlphaAgent vs Claude Trading Agents — Side by Side

| Feature | Claude Trading Agents | AlphaAgent (Ours) |
|---|---|---|
| **Architecture** | Single LLM agent | 8 specialized agents + quant engine |
| **Signal type** | "Buy/Sell/Hold" text | Calibrated probability (73% bullish) |
| **Confidence** | "Moderate" (meaningless) | Entropy-based (H=0.42, quantified) |
| **Expected move** | None | Monte Carlo CI: 68%, 95%, 99% ranges |
| **Holding period** | None | Signal decay analysis + OU half-life |
| **Position sizing** | None | Kelly Criterion + vol-adjusted |
| **Risk metrics** | None | VaR, CVaR via EVT |
| **Regime awareness** | None | HMM (Bull/Bear/Crisis) + strategy switching |
| **Volatility model** | None | GARCH(1,1) + EGARCH forecasts |
| **Tail risk** | None | Extreme Value Theory |
| **Signal combination** | LLM narrative | Bayesian fusion with correlation adjustment |
| **News analysis** | LLM reads headlines | RAG + embeddings + LLM sentiment scoring |
| **Backtesting** | Separate tool | Integrated backtest harness |
| **Transparency** | LLM explanation text | Full math: every number has a formula |
| **Self-improvement** | "Dreaming" (review journals) | Walk-forward optimization + accuracy tracking |
| **Hallucination risk** | HIGH (LLM generates plausible BS) | LOW (math produces numbers, LLM only reasons) |

---

## Part 5: The Key Design Philosophy

### What Claude Agents Get Wrong

```
Claude Agent thought process:
  "MACD is bullish → RSI is not overbought → Recent news positive → BUY"
  
Problems:
  1. No probability attached to "bullish MACD"
  2. No quantification of "positive" news  
  3. No risk management math
  4. No regime context
  5. LLM can construct equally convincing narratives for SELL
```

### What AlphaAgent Does Differently

```
AlphaAgent pipeline:
  1. GARCH → σ_tomorrow = 1.8% (quantified volatility)
  2. HMM → P(Bull) = 0.78 (quantified regime)
  3. Technical indicators → each scored 0-100
  4. News sentiment → P(Bullish|articles) = 0.65 (calibrated)
  5. Fundamentals → valuation z-score = +0.8 (undervalued)
  6. Macro → risk-on/risk-off score = +0.55
  7. Bayesian fusion → P(up) = 0.73 (posterior)
  8. Monte Carlo → 68% CI: [+1.2%, +4.8%] (expected range)
  9. EVT → VaR(99%) = -4.2% (tail risk)
  10. Kelly → position = 6.2% of capital (sizing)
  11. Signal decay → hold 3-7 days (timing)
  
  LLM's ONLY job: 
    - Explain the math in plain English
    - Highlight which factors are driving the signal
    - Flag any contradictions between agents
```

> [!TIP]
> **The LLM explains the math — it doesn't DO the math.** This is the critical difference. Numbers come from equations, not from LLM "reasoning."

---

## Part 6: Updated Signal Output Design

### The Final Signal Card

```json
{
  "ticker": "NVDA",
  "timestamp": "2026-05-12T15:30:00Z",
  "analysis_duration": "12.3s",
  
  "signal": {
    "direction": "LONG",
    "conviction_pct": 73.2,
    "confidence": "HIGH",
    "entropy": 0.42
  },
  
  "expected_move": {
    "horizon_days": 5,
    "ci_68": {"low_pct": 1.2, "high_pct": 4.8},
    "ci_95": {"low_pct": -2.1, "high_pct": 7.3},
    "ci_99": {"low_pct": -4.2, "high_pct": 9.1},
    "median_return_pct": 2.8
  },
  
  "holding_period": {
    "optimal_days": [3, 7],
    "signal_halflife_days": 5.2,
    "max_hold_days": 15,
    "decay_rate": 0.87
  },
  
  "risk_metrics": {
    "var_95_pct": -2.1,
    "var_99_pct": -4.2,
    "cvar_99_pct": -5.8,
    "max_drawdown_mc_pct": -6.8,
    "sharpe_forecast": 1.45
  },
  
  "position_sizing": {
    "kelly_full_pct": 12.4,
    "kelly_half_pct": 6.2,
    "vol_adjusted_pct": 5.8,
    "dollar_amount": 11600,
    "portfolio_value": 200000
  },
  
  "regime": {
    "current": "BULL",
    "probabilities": {"bull": 0.78, "bear": 0.16, "crisis": 0.06},
    "transition_risk": {"to_bear": 0.05, "to_crisis": 0.01}
  },
  
  "agent_scores": {
    "technical":    {"score": 72, "direction": "bullish", "weight": 0.30},
    "sentiment":    {"score": 65, "direction": "bullish", "weight": 0.20},
    "fundamental":  {"score": 58, "direction": "neutral", "weight": 0.20},
    "macro":        {"score": 63, "direction": "bullish", "weight": 0.15},
    "volatility":   {"forecast_pct": 1.8, "regime": "normal", "weight": 0.15}
  },
  
  "debate_summary": {
    "bull_case": "Strong MACD crossover + positive earnings sentiment + risk-on macro...",
    "bear_case": "RSI approaching overbought + elevated VIX + sector rotation risk...",
    "key_risk": "Fed meeting in 3 days could shift regime"
  },
  
  "guardrails": {
    "approved": true,
    "warnings": ["Fed meeting on 2026-05-15 may cause volatility spike"],
    "stop_loss_pct": -3.0,
    "take_profit_pct": 4.5
  }
}
```

---

## Part 7: What To Change in the Implementation Plan

The current IMPLEMENTATION_PLAN.md needs these updates to align with the probabilistic approach:

### Phase 1 Additions
- Add **signal decay analysis** module to quant_engine
- Add **Ornstein-Uhlenbeck** half-life estimator for holding periods
- Add **options-implied expected move** calculator

### Phase 5 Revisions (Debate + Risk)
- Debate Agent outputs the full **signal packet JSON** (not just Buy/Sell/Hold)
- Risk Agent computes **stop-loss and take-profit levels** from EVT
- Risk Agent estimates **optimal holding period** from signal decay

### Phase 6 Dashboard
- Signal Card shows **conviction meter** (0-100%) not just text
- Monte Carlo fan chart with **clickable confidence intervals**
- **Holding period timeline** with signal strength decay curve
- **Regime indicator** with transition probabilities
- **Position sizing calculator** based on portfolio value input

### New Phase (Optional): Walk-Forward Backtesting
- Run signal generation on historical data
- Track actual accuracy of probability estimates (calibration)
- Adjust agent weights based on backtested performance
- Generate **calibration plots** (predicted probability vs actual frequency)

---

## Summary: Why AlphaAgent Will Be Superior

```
Claude Agent:  "Buy NVDA. I'm fairly confident based on the technicals."
                → No math. No risk. No timing. No sizing. Vibes.

AlphaAgent:    "LONG NVDA @ 73% conviction (HIGH confidence).
                Expected +1.2% to +4.8% (68% CI) over 3-7 days.
                Max downside: -4.2% (99% VaR). 
                Size: 6.2% of portfolio (half-Kelly).
                Current regime: BULL (78%).
                Stop: -3.0%. Target: +4.5%.
                Risk: Fed meeting in 3 days."
                → Every number backed by math. Actionable. Honest.
```

The difference is **mathematical rigor** vs **LLM storytelling**.
