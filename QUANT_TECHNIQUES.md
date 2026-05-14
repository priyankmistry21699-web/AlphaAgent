# Dominant Quant Techniques for Stock Price Prediction

> An honest, no-BS breakdown of what **actually works** at the firms that consistently make money — Renaissance Technologies, Two Sigma, Citadel, DE Shaw, AQR, Jump Trading, Jane Street.

> [!CAUTION]
> No single technique "predicts" price reliably. The edge comes from **combining weak signals** at scale with proper risk management. Anyone claiming one magic formula is lying.

---

## The Brutal Truth First

Stock price prediction is NOT about finding one equation that outputs tomorrow's price. It's about:

1. **Finding tiny statistical edges** (signals that predict direction 51-53% of the time)
2. **Combining hundreds of weak signals** into one stronger composite signal
3. **Managing risk** so that when you're wrong (you will be ~47% of the time), losses are small
4. **Executing faster and cheaper** than competitors

The techniques below are organized by **how dominant they actually are** in the industry, not by how impressive they sound in a textbook.

---

## TIER 1 — The Absolute Core (Every quant fund uses these daily)

### 1. Factor Models (The King of Quant)

> [!IMPORTANT]
> This is the **single most important concept** in institutional quant finance. If you learn nothing else, learn this.

**What it is**: Stock returns are explained by exposure to common "factors" (risk drivers). The stock-specific return (alpha) is what you're trying to predict.

```
r_i = α_i + β₁·F₁ + β₂·F₂ + ... + βₖ·Fₖ + ε_i

Where:
  r_i = return of stock i
  α_i = stock-specific return (YOUR EDGE)
  F_j = return of factor j
  β_j = stock i's exposure to factor j
  ε_i = random noise
```

**The Fama-French factors (classic):**
| Factor | What it captures |
|---|---|
| Market (MKT) | Overall market direction |
| Size (SMB) | Small stocks outperform large |
| Value (HML) | Cheap stocks outperform expensive |
| Momentum (UMD) | Winners keep winning, losers keep losing |
| Quality (QMJ) | High-quality companies outperform |
| Low Volatility | Low-vol stocks outperform (anomaly) |

**Modern extensions (what Two Sigma/AQR actually use):**
- Short-term reversal (1-week)
- Earnings momentum (post-earnings drift)
- Analyst revision momentum
- Option-implied factors
- ESG factors
- 100+ proprietary factors

**Why it's dominant**: 
- Explains 60-90% of return variance
- Every $1T+ asset manager uses factor models
- AQR Capital literally built a $140B firm on this

**How to predict with it**: Construct a "composite alpha score" from many factor exposures → rank stocks → go long top decile, short bottom decile.

---

### 2. Statistical/ML Signal Combination (Alpha Combination)

**What it is**: Take 100-1000+ individual "signals" (each predicting direction weakly) and combine them optimally.

```
Combined_Alpha = w₁·Signal₁ + w₂·Signal₂ + ... + wₙ·Signalₙ

Where weights w are determined by:
  - Historical predictive power (IC = Information Coefficient)
  - Signal decay rate
  - Correlation between signals (avoid redundancy)
  - Turnover costs
```

**Signal examples (each individually weak, ~51-53% accuracy):**
- Price momentum (3-12 month)
- Earnings surprise magnitude
- Short interest changes
- Insider buying/selling
- Analyst revision direction
- Options market implied move
- News sentiment score
- Order flow imbalance
- Cross-sectional valuation rank
- Supply chain relationship signals

**How weights are chosen:**
```
Optimal: w* = Σ⁻¹ · IC_vector

Where:
  Σ = covariance matrix of signal returns
  IC = information coefficient (correlation of signal with future return)
```

**Why it's dominant**: This IS the business of quant hedge funds. Renaissance's Medallion Fund runs 10,000+ signals.

---

### 3. GARCH Family (Volatility Prediction)

**What it does**: Predicts **tomorrow's volatility** — not price direction, but how much the price will move.

```
GARCH(1,1):  σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁
EGARCH:      ln(σ²ₜ) = ω + β·ln(σ²ₜ₋₁) + α·|zₜ₋₁| + γ·zₜ₋₁
```

**Why it's dominant for prediction:**
- Volatility is **far more predictable** than price direction
- GARCH volatility forecast R² ≈ 30-50% (vs price direction R² ≈ 0.5-2%)
- Options are priced on volatility → predict vol, you can trade options profitably
- Risk management: know your position's true risk

**Practical use**: 
- If GARCH says vol will spike → reduce position size, buy puts
- If GARCH says vol will drop → sell options (collect premium)
- Every options desk on Wall Street runs GARCH

---

### 4. Mean-Reversion & Cointegration (Statistical Arbitrage)

**What it is**: Some price relationships are "stationary" — they deviate but always return to equilibrium. Trade the deviation.

```
Pairs Trading (Engle-Granger):
  Spread = Price_A - β·Price_B
  
  If spread > μ + 2σ → SHORT spread (sell A, buy B)
  If spread < μ - 2σ → LONG spread (buy A, sell B)
  
  Half-life = ln(2)/θ  (how fast it reverts)
```

**Ornstein-Uhlenbeck Process (mathematical model):**
```
dx = θ(μ - x)dt + σdW

θ = mean-reversion speed (higher = faster revert = better trade)
μ = equilibrium level
σ = noise
```

**Why it's dominant**: 
- Stat arb is a **$100B+ strategy class**
- Works because of market microstructure (ETF arbitrage, index rebalancing)
- DE Shaw, Citadel, Renaissance all run massive stat arb books
- More robust than directional prediction (you're betting on a relationship, not a direction)

---

### 5. Bayesian Updating (Real-Time Belief Revision)

**What it is**: Continuously update your probability estimate as new data arrives.

```
P(Bull | new_data) = P(new_data | Bull) × P(Bull) / P(new_data)

In practice with multiple signals:
  Prior:      P(up) = 0.52 (slight bullish base rate)
  Signal 1:   Positive earnings → P(up|earnings) ∝ P(earnings|up) × P(up)
  Signal 2:   Negative news → P(up|news) ∝ P(news|up) × P(up|earnings)
  Signal 3:   Bullish technicals → P(up|tech) ∝ P(tech|up) × P(up|news)
  ...
  Final:      P(up | all evidence) = posterior after all signals
```

**Why it's dominant**:
- It's the **mathematically optimal** way to combine information
- Every multi-signal system implicitly or explicitly uses this
- Allows you to properly weight conflicting signals
- Naturally handles uncertainty

---

## TIER 2 — Serious Edge (Used by most top-tier funds)

### 6. Hidden Markov Models (Regime Detection)

**What it does**: Detects which "hidden state" the market is in (bull/bear/crisis) and predicts transitions.

```
States: {Bull, Bear, Sideways, Crisis}

Each state has different:
  - Return distribution: Bull ~ N(+0.08%, 0.8%), Bear ~ N(-0.05%, 1.5%)
  - Transition probabilities: P(Bull→Bear) = 0.04, P(Bear→Crisis) = 0.05
  
Real-time output: 
  P(currently in Bull) = 72%
  P(transition to Bear next week) = 8%
```

**Why it's dominant**: 
- **The same strategy doesn't work in all regimes**. Momentum works in trends, mean-reversion in ranges
- HMM tells you which strategy to use RIGHT NOW
- Bridgewater's "All Weather" approach is regime-aware
- AQR uses regime models to adjust factor exposures

---

### 7. Monte Carlo Simulation (Probability Estimation)

**What it does**: Simulate 10,000+ possible future price paths → compute probability of ANY outcome.

```
For each simulation i = 1 to 10,000:
  For each day t = 1 to T:
    S(t) = S(t-1) × exp[(μ - σ²/2)·dt + σ·√dt·Z]  
    where Z ~ N(0,1)

Results:
  P(AAPL > $200 in 30 days) = count(paths where S(30) > 200) / 10,000
  5th percentile price = worst-case scenario
  Median path = expected trajectory
```

**Advanced variants used at top firms:**
- **Jump-diffusion Monte Carlo** — add Poisson jumps for crash scenarios
- **Stochastic vol Monte Carlo** — use Heston model instead of constant vol
- **Importance sampling** — oversample rare events for better tail estimates
- **Conditional Monte Carlo** — condition on macro regime from HMM

**Why it's dominant**: 
- Most flexible technique — can answer ANY probability question
- Used for options pricing, risk management, scenario analysis
- Every risk department runs Monte Carlo VaR daily

---

### 8. Kalman Filter (Signal Extraction)

**What it does**: Extracts the "true" signal from noisy market data in real-time.

```
Predict:  x̂ₜ|ₜ₋₁ = F·x̂ₜ₋₁         (model prediction)
          Pₜ|ₜ₋₁ = F·Pₜ₋₁·F' + Q   (uncertainty grows)

Update:   Kₜ = Pₜ|ₜ₋₁·H'·(H·Pₜ|ₜ₋₁·H' + R)⁻¹  (Kalman gain)
          x̂ₜ = x̂ₜ|ₜ₋₁ + Kₜ·(zₜ - H·x̂ₜ|ₜ₋₁)      (corrected estimate)
```

**Practical uses:**
- **Dynamic beta estimation**: How sensitive is a stock to the market RIGHT NOW? (beta changes over time)
- **Trend extraction**: Separate the trend from noise without lag (unlike moving averages)
- **Pairs trading**: Track the hedge ratio β in real-time (it drifts)
- **Alpha signal smoothing**: Reduce noise in your trading signals

**Why it's dominant**: It's the optimal linear filter. Every GPS uses it, every quant should too.

---

### 9. NLP/Sentiment Analysis (News as Alpha)

**What it is**: Extract predictive signal from text data (news, earnings calls, social media, SEC filings).

```
Pipeline:
  Raw text → Tokenize → Embedding → Sentiment score → Alpha signal

Modern approach:
  1. FinBERT/LLM processes article
  2. Output: P(Bullish), P(Bearish), P(Neutral)
  3. Aggregate across all articles for a stock
  4. Sentiment momentum = change in aggregate sentiment
  5. Combine with price signal → alpha
```

**Key finding**: The SPEED of sentiment processing matters more than accuracy. Being 10 minutes faster with 60% accuracy beats being 1 hour slower with 90% accuracy.

**Why it's dominant:**
- **Post-news drift is real** — prices don't fully adjust for hours/days
- Two Sigma and WorldQuant invest heavily in NLP
- Earnings call tone predicts next-quarter earnings (proven in academic papers)
- SEC filing changes (10-K diff) predict price moves

---

### 10. Extreme Value Theory (Tail Risk)

**What it does**: Models the probability of extreme events (crashes, melt-ups) that normal distributions wildly underestimate.

```
Generalized Pareto Distribution for tails:
  P(X > x | X > u) ≈ (1 + ξ(x-u)/σ)^(-1/ξ)

  ξ > 0: Heavy tail (stocks, crypto)
  ξ = 0: Light tail (bonds)

Value at Risk (VaR):  "Max loss at 99% confidence"
CVaR (Expected Shortfall): "Average loss in the worst 1% of cases"
```

**Why it's dominant:**
- Normal distribution says a 10σ event happens once per universe lifetime. In markets, it happens every few years.
- 2008, 2020, flash crashes — EVT models these correctly
- Required by Basel III regulations for bank risk capital
- Kelly Criterion for position sizing depends on tail estimates

---

## TIER 3 — Advanced Edge (Cutting-edge at elite firms)

### 11. Deep Learning for Price Prediction

**What works (and what doesn't):**

| Architecture | Works? | Why |
|---|---|---|
| Simple LSTM/GRU on prices | ❌ No | Prices are nearly random; overfits to noise |
| Transformer on multi-modal data | ✅ Yes | Combines price + volume + news + order flow |
| CNN on limit order book | ✅ Yes | Captures microstructure patterns |
| Graph Neural Network on stock relationships | ✅ Yes | Models sector/supply chain effects |
| Temporal Fusion Transformer | ✅ Yes | Best for multi-horizon forecasting |

**The key insight**: Deep learning works when you feed it **alternative data** (not just price). Price alone has nearly zero predictive signal at daily+ horizons.

**What top firms actually use:**
```
Input features (100+):
  - Price/volume features (50+)
  - News sentiment time series
  - Options market features (implied vol, put-call ratio, skew)
  - Order flow imbalance
  - Analyst estimates and revisions
  - Insider trading activity
  - Macroeconomic indicators
  - Cross-asset signals (bonds, commodities, FX)
  
Architecture: Transformer or Attention-based ensemble
Output: P(up), P(down) with calibrated probabilities
```

---

### 12. Reinforcement Learning for Trading

**What it does**: An agent learns to trade by trial-and-error in simulated markets.

```
State:   s = [prices, position, portfolio_value, indicators, regime]
Action:  a = continuous position size [-1, +1]
Reward:  r = risk_adjusted_PnL - transaction_costs

Best algorithms:
  - PPO (Proximal Policy Optimization) — most stable
  - SAC (Soft Actor-Critic) — best for continuous actions
  - Multi-agent RL — multiple agents with different time horizons
```

**Why it's cutting-edge:**
- Can discover strategies humans haven't thought of
- Naturally handles transaction costs, position limits
- JP Morgan, Goldman, Man AHL all have RL research teams
- **Warning**: Extremely hard to train without overfitting

---

### 13. Hawkes Processes (Order Flow Prediction)

**What it does**: Models how trades trigger more trades (self-excitation). Essential for intraday prediction.

```
Intensity: λ(t) = μ + Σᵢ α·e^(-β(t-tᵢ))

After each trade, the probability of the next trade SPIKES, 
then decays exponentially. 

Extend to "mutually exciting":
  λ_buy(t) = μ_buy + α_bb·(past buys) + α_bs·(past sells)
  λ_sell(t) = μ_sell + α_sb·(past buys) + α_ss·(past sells)
```

**Why it matters**: Short-term price impact is predictable. If a burst of buys occurs, the price will continue moving up for milliseconds-to-seconds. HFT firms exploit this.

---

### 14. Rough Volatility (State of the Art — 2024+)

**What it is**: Volatility follows a fractional Brownian motion with Hurst exponent H ≈ 0.1 (much rougher than classical models assume).

```
Classical: dv = mean_revert + vol_of_vol · dW     (H = 0.5)
Rough:     dv = mean_revert + vol_of_vol · dW^H   (H ≈ 0.1)

This means volatility has VERY short memory and extremely 
jagged paths — matching real market data perfectly.
```

**Why it's the frontier:**
- Explains the VIX smile that classical models can't
- Better options pricing than Heston/SABR
- rBergomi model is becoming the new industry standard
- Empirically validated across equities, FX, commodities

---

## TIER 4 — Specialized Niches

### 15. Copula-Based Dependency Models
For portfolio construction when you need to understand how correlations change during crashes (hint: everything becomes correlated in a crash).

### 16. Optimal Execution (Almgren-Chriss)
For minimizing market impact when trading large positions. Not prediction per se, but crucial for turning predictions into profits.

### 17. Topological Data Analysis (TDA)
Using persistent homology to detect market regime changes from the "shape" of price data. Very new, used at a few quant funds.

### 18. Causal Inference (Granger Causality, DoWhy)
Distinguish "X predicts Y" from "X causes Y". Critical for avoiding spurious signals.

---

## The Honest Ranking: What to Learn First

| Priority | Technique | Why First | Time to Learn |
|---|---|---|---|
| 1 | **Factor Models + Alpha Combination** | This IS quant finance | 2-3 weeks |
| 2 | **GARCH (Volatility Prediction)** | Most predictable thing in markets | 1 week |
| 3 | **Bayesian Updating** | How to combine all your signals | 1 week |
| 4 | **Monte Carlo Simulation** | Answer any probability question | 1 week |
| 5 | **Cointegration / Stat Arb** | Most proven strategy class | 2 weeks |
| 6 | **Hidden Markov Models** | Know your regime | 1-2 weeks |
| 7 | **NLP/Sentiment** | Alternative data edge | 2 weeks |
| 8 | **Kalman Filter** | Real-time signal extraction | 1-2 weeks |
| 9 | **Extreme Value Theory** | Know your tail risk | 1 week |
| 10 | **Deep Learning (Transformers)** | Combine everything at scale | 3-4 weeks |
| 11 | **Reinforcement Learning** | Autonomous trading agent | 4+ weeks |
| 12 | **Rough Volatility** | Cutting-edge options pricing | 3-4 weeks |

---

## What Renaissance Technologies Actually Does (Best Guess)

Nobody knows for sure, but from patents, papers by ex-employees, and industry analysis:

1. **Thousands of weak signals** (each 51-53% accurate) combined via optimized weighting
2. **Short holding periods** (hours to days) — signal decays fast
3. **Massive diversification** — trade 5,000+ instruments simultaneously
4. **Regime-aware** — different signal weights in different market states
5. **Non-linear signal combination** — probably kernel methods or neural nets
6. **Microstructure signals** — order book imbalance, trade clustering
7. **Alternative data** — satellite imagery, credit card data, web scraping
8. **Execution optimization** — minimize market impact with Almgren-Chriss style models
9. **Transaction cost awareness** — signals must overcome costs to be traded
10. **Aggressive risk management** — hard position limits, factor exposure limits, stop losses

The math is not the secret. The secret is **having better data** and **combining signals better** than everyone else.
