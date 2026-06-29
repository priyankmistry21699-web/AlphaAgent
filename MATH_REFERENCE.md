# AlphaAgent — Complete Mathematical & Factor Reference

> **The definitive math + factor + formula reference.** Every theory with its formula, every factor classified as dynamic or static, every constant and threshold. Nothing omitted.

**Companion to:** `PROJECT_AUDIT.md` (architecture), `COMPLETE_INVENTORY.md` (factor list), `PROJECT_GRAPH.json` (graph memory).

---

# PART I — MATHEMATICAL FUNDAMENTALS

## A. Probability & Statistics

### 1. Bayes' Theorem (foundational — used in `bayesian.py`)
```
P(H|E) = P(E|H) · P(H) / P(E)
```
In log-odds form (used for Bayesian fusion):
```
log-odds(posterior) = log-odds(prior) + log-likelihood-ratio
log-odds(p) = ln(p / (1-p))
sigmoid(x) = 1 / (1 + exp(-x))
```

### 2. Shannon Entropy (used in `bayesian.py` for fusion entropy)
```
H(X) = -Σᵢ pᵢ · log₂(pᵢ)
```
Used to measure agreement: high entropy = agents disagree.

### 3. KL Divergence (used in `risk.py` for regime shift detection)
```
D_KL(P || Q) = Σᵢ P(i) · log(P(i) / Q(i))
```
Recent 20d distribution vs baseline 252d.

### 4. Variance / Standard Deviation
```
σ² = E[(X - μ)²]
σ = √σ²
Sample: σ²_n = (1/(n-1)) · Σ(xᵢ - x̄)²
```

### 5. Skewness (used in `risk.py`, `volatility.py`)
```
Skew = E[(X - μ)³] / σ³
Sample: g₁ = (1/n) · Σ((xᵢ - x̄)/σ)³
```

### 6. Excess Kurtosis (used in `risk.py`)
```
ExcessKurt = E[(X - μ)⁴] / σ⁴ - 3
Sample: g₂ = (1/n) · Σ((xᵢ - x̄)/σ)⁴ - 3
```

### 7. Z-Score (used everywhere)
```
z = (x - μ) / σ
```

### 8. Pearson Correlation
```
ρ(X,Y) = Cov(X,Y) / (σ_X · σ_Y)
       = E[(X-μ_X)(Y-μ_Y)] / (σ_X · σ_Y)
```

### 9. Spearman Correlation (used in `leaderboard.py` for IC)
```
ρ_s = 1 - 6·Σdᵢ² / (n(n²-1))
where dᵢ = rank(xᵢ) - rank(yᵢ)
```

---

## B. Linear Algebra

### 10. Covariance Matrix
```
Σᵢⱼ = E[(Xᵢ - μᵢ)(Xⱼ - μⱼ)]
```

### 11. Eigendecomposition (used in `rmt.py`, `factor_orthogonalization.py`)
```
C · v = λ · v
C = V · Λ · Vᵀ
```
where V is orthonormal eigenvectors, Λ is diagonal eigenvalues.

### 12. Marchenko-Pastur Eigenvalue Bounds (used in `rmt.py`)
```
λ_max = (1 + √(N/T))²
λ_min = (1 - √(N/T))²
```
Eigenvalues outside [λ_min, λ_max] contain signal; inside is noise.

### 13. Inverse Square Root (Löwdin orthogonalization, `factor_orthogonalization.py`)
```
F* = F · (FᵀF)^(-1/2)
A^(-1/2) = V · diag(1/√λᵢ) · Vᵀ
```

### 14. Gram-Schmidt (used in `factor_orthogonalization.py`)
```
For each column fᵢ:
  Project on prior: pᵢ = Σⱼ<ᵢ (fᵢᵀ · fⱼ / fⱼᵀ · fⱼ) · fⱼ
  Orthogonal: f*ᵢ = fᵢ - pᵢ
```

### 15. OLS Regression (used in idiosyncratic vol calc)
```
β = (XᵀX)⁻¹ Xᵀy
α = ȳ - β·x̄
ε = y - (α + β·X)
```

### 16. Cholesky Decomposition (used in Monte Carlo for correlated draws)
```
Σ = L · Lᵀ
correlated_z = L · independent_z
```

---

## C. Time Series

### 17. Pct Change
```
r_t = (P_t / P_{t-1}) - 1
or log return: r_t = ln(P_t / P_{t-1})
```

### 18. Rolling Window Statistics
```
μ_t = (1/w) · Σᵢ₌ₜ₋ᵥ₊₁^t xᵢ
σ_t = √((1/(w-1)) · Σ(xᵢ - μ_t)²)
```

### 19. Exponentially Weighted Moving Average (EMA)
```
EMA_t = α·x_t + (1-α)·EMA_{t-1}
α = 2/(N+1)
```

### 20. Annualization (252 trading days)
```
σ_annual = σ_daily · √252
Sharpe = (μ_daily · 252) / σ_annual
```

### 21. ADF Test for Stationarity (used in `ml_finance.py`)
```
Δy_t = α + βt + γy_{t-1} + Σδᵢ·Δy_{t-i} + ε_t
H₀: γ = 0 (non-stationary)
```

### 22. Hurst Exponent (used in technical agent)
```
E[R/S]_n ~ c · n^H
H ≈ 0.5: random walk
H > 0.5: trending (long-range positive memory)
H < 0.5: mean-reverting (long-range negative memory)
```
Computed via R/S rescaled-range analysis.

---

## D. Stochastic Processes

### 23. Geometric Brownian Motion (used in `monte_carlo.py`)
```
dS_t = μ · S_t · dt + σ · S_t · dW_t
S_t = S_0 · exp((μ - σ²/2)·t + σ·W_t)
```

### 24. Ornstein-Uhlenbeck (used in `signal_decay.py`)
```
dX_t = θ(μ - X_t)·dt + σ·dW_t
Half-life: τ = ln(2) / θ
```

### 25. Itô's Lemma (foundation of Heston)
```
For dX_t = μ·dt + σ·dW_t, and f(X,t):
df = (∂f/∂t + μ·∂f/∂X + ½σ²·∂²f/∂X²) dt + σ·∂f/∂X · dW
```

### 26. Heston Stochastic Volatility (used in `heston.py`)
```
dS_t = μ·S_t·dt + √v_t·S_t·dW_1
dv_t = κ(θ - v_t)·dt + ξ·√v_t·dW_2
Cov(dW_1, dW_2) = ρ·dt
```
Parameters: κ (mean reversion), θ (long-run var), ξ (vol of vol), ρ (correlation).

### 27. SABR Vol Smile (used in `sabr.py`)
```
dF_t = α·F_t^β · dW_1
dα_t = ν·α_t · dW_2
Cov(dW_1, dW_2) = ρ·dt
```
Hagan's formula gives σ_BS(K, F) as a closed form.

### 28. Rough Volatility / rBergomi (used in `rough_vol.py`)
```
σ_t = ξ_0 · exp(η · W^H_t - η²·t^(2H)/2)
where W^H is fractional Brownian motion with Hurst H < 0.5
```

---

# PART II — QUANT THEORY FORMULAS (Every Module)

## E. Volatility Models

### 29. GARCH(1,1) — Bollerslev 1986 (`garch.py`)
```
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
Persistence: α + β (must be < 1)
Unconditional variance: ω / (1 - α - β)
```
Typical equity calibration: α≈0.10, β≈0.85.

### 30. EGARCH — Exponential GARCH
```
ln(σ²_t) = ω + α·(|z_{t-1}| - E|z|) + γ·z_{t-1} + β·ln(σ²_{t-1})
where z_t = ε_t / σ_t
```
Captures asymmetric leverage effect (negative returns increase vol more).

### 31. DCC-GARCH — Engle 2002 (`dcc_garch.py`)
```
Step 1: Univariate GARCH → standardised residuals ε*_i,t = ε_i,t / σ_i,t
Step 2: DCC(1,1) recursion on Q_t:
  Q_t = (1 - a - b)·Q̄ + a·ε*_{t-1}·ε*ᵀ_{t-1} + b·Q_{t-1}
Step 3: Conditional correlation:
  R_t = diag(Q_t)^(-½) · Q_t · diag(Q_t)^(-½)
```
Calibration: a=0.05, b=0.93 (equity standard). Q̄ is RMT-cleaned in our impl.

### 32. Variance Risk Premium (Carr-Wu 2009)
```
VRP = E^Q[σ²_RV] - σ²_RV_realized
    ≈ IV² - RV²    (in our impl)
```
In our system: `_vrp = (_iv_ann² - _rv_ann²) × 100`

### 33. Parkinson Vol Estimator (1980)
```
σ²_P = (1 / (4·ln(2)·n)) · Σ (ln(Hᵢ) - ln(Lᵢ))²
```
~5x more efficient than close-to-close.

### 34. Garman-Klass Vol (1980)
```
σ²_GK = (1/n) · Σ [ 0.5·(ln(Hᵢ/Lᵢ))² - (2·ln(2)-1)·(ln(Cᵢ/Oᵢ))² ]
```
~7.4x more efficient.

### 35. Rogers-Satchell Vol (1991)
```
σ²_RS = (1/n) · Σ [ ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O) ]
```
Drift-unbiased, ~8x efficient.

### 36. Yang-Zhang Vol (2000) — best overall
```
σ²_YZ = σ²_overnight + k·σ²_open + (1-k)·σ²_RS
where:
  σ²_overnight = Var(ln(O_t / C_{t-1}))
  σ²_open      = Var(ln(C_t / O_t))
  k = 0.34 / (1.34 + (n+1)/(n-1))
```
Handles overnight gaps. ~14x efficient overall.

---

## F. Risk / Tail Models

### 37. EVT — Generalized Pareto Distribution (`evt.py`)
```
F(x) = 1 - (1 + ξ·x/β)^(-1/ξ)  for x ≥ 0
```
Fit ξ (tail index) and β (scale) via MLE on exceedances above threshold u.

VaR estimation:
```
VaR_q = u + (β/ξ) · [(n/N_u · (1-q))^(-ξ) - 1]
CVaR_q = VaR_q / (1-ξ) + (β - ξ·u) / (1-ξ)
```

### 38. Monte Carlo VaR (`monte_carlo.py`)
```
For path k = 1..N:
  S_T^(k) = S_0 · exp((μ - σ²/2)·T + σ·√T·Z^(k))
  loss^(k) = max(0, S_0 - S_T^(k))
VaR_q = q-th quantile of losses
```
N=5000 paths typical, scaled by GARCH regime (3000 to 10000).

### 39. Quasi-MC / Sobol (`quasi_mc.py`)
Replaces pseudo-random Z with low-discrepancy Sobol sequences:
```
Z_sobol = Φ⁻¹(sobol_uniform)
```
~10x more efficient convergence.

### 40. Hawkes Process (`hawkes.py`)
```
λ(t) = μ + Σ_{tᵢ<t} α · exp(-β·(t - tᵢ))
```
- μ: baseline rate
- α: self-excitation strength
- β: decay rate
Branching ratio: n = α / β
- n < 1: stable
- n → 1: cascade risk
- n > 1: explosive

### 41. Kelly Criterion (`kelly.py`)
```
f* = (b·p - q) / b
where p = win prob, q = 1-p, b = odds
```
For continuous returns (with normal assumption):
```
f* = (μ - r_f) / σ²
Half-Kelly: f = 0.5 · f*  (used)
Capped at 0.25 max
```

### 42. Almgren-Chriss Market Impact (`transaction_costs.py`)
```
impact_bps = const · daily_vol_bps · √(participation_rate)
participation_rate = notional / ADV_dollar  (capped 30%)
```
Square-root law for permanent market impact.

### 43. Copula (`copula.py`)
Sklar's theorem: any joint dist H(x,y) = C(F(x), G(y)) for some copula C.

**Gaussian copula:**
```
C(u,v) = Φ_ρ(Φ⁻¹(u), Φ⁻¹(v))
```
**Clayton copula** (lower tail dependence):
```
C(u,v) = (u^(-θ) + v^(-θ) - 1)^(-1/θ)
λ_L = 2^(-1/θ)
```
**Gumbel copula** (upper tail dependence):
```
C(u,v) = exp(-((-ln u)^θ + (-ln v)^θ)^(1/θ))
λ_U = 2 - 2^(1/θ)
```

---

## G. Regime Detection

### 44. Hidden Markov Model — Baum-Welch (`hmm.py`)
Hidden state s_t ∈ {1, 2, 3} = {Bull, Bear, Crisis}.

**Forward algorithm:**
```
α_t(i) = P(O_1..t, s_t=i | λ)
α_t(j) = [Σᵢ α_{t-1}(i)·a_ij] · b_j(O_t)
```
**Viterbi for max-likely sequence:**
```
δ_t(j) = max_i [δ_{t-1}(i)·a_ij] · b_j(O_t)
```
States sorted by realised vol → BULL (lowest vol), BEAR (medium), CRISIS (highest).

Transition matrix:
```
A = [[a_11 a_12 a_13],
     [a_21 a_22 a_23],
     [a_31 a_32 a_33]]
```

### 45. CUSUM Page (1954) (`structural_break.py`)
Two-sided cumulative sum chart:
```
S_t^+ = max(0, S_{t-1}^+ + z_t - k)
S_t^- = min(0, S_{t-1}^- + z_t + k)
where z_t = (x_t - μ) / σ
```
Alarm when max(S^+, |S^-|) > h.
Standard: k = 0.5, h = 5 (detects 1σ shift).

---

## H. Fusion / Decision

### 46. Bayesian Log-Odds Fusion (`bayesian.py`)
```
posterior = prior + Σᵢ wᵢ · sensitivity · log(pᵢ / (1-pᵢ)) · (1 - corrᵢ)

final_prob = sigmoid(posterior_log_odds)
```
- wᵢ: agent weight × regime scale × confidence
- corrᵢ: correlation penalty (overlap with other agents)
- sensitivity: dynamic (1.4 default)

### 47. Meta-Learner Stacking (`meta_learner.py`)
LightGBM trained on:
- Per-agent probabilities
- Bayesian posterior
- Conviction
- Entropy
- Agreement score
- Regime indicator
Output: `lgbm_prob` blended with `bayesian_prob` via learned weight:
```
final_prob = w_lgbm · lgbm_prob + (1 - w_lgbm) · bayesian_prob
```

### 48. Platt Scaling (`calibration.py`)
Logistic regression on (raw_score, label):
```
P(y=1 | score) = 1 / (1 + exp(A·score + B))
```
A, B fit by maximum likelihood on historical signal/outcome pairs.

### 49. Information Coefficient (IC) — used in `leaderboard.py`
Rolling Spearman correlation between agent's probability_up and forward return:
```
IC_t = ρ_s(prob_up_{t-N..t}, fwd_return_{t-N..t})
```
N = 60 days standard.

---

## I. Portfolio Construction

### 50. Markowitz MVO (`portfolio_optimizer.py`)
```
maximize: μᵀw - (λ/2)·wᵀΣw
subject to: Σwᵢ = 1, wᵢ ≥ 0
Closed form (no constraints): w* = (1/λ) · Σ⁻¹ · μ
```

### 51. Black-Litterman (1992) (`black_litterman.py`)
**Equilibrium implied returns:**
```
π = δ · Σ · w_market
where δ = risk aversion (≈ 2.5), w_market from market caps
```
**View matrix:**
```
P = matrix of view rows (1s and 0s picking assets)
q = view expected returns
Ω = view covariance (uncertainty)
```
**Posterior mean (He-Litterman closed form):**
```
M⁻¹ = (τ·Σ)⁻¹ + Pᵀ·Ω⁻¹·P
posterior_μ = M · [(τ·Σ)⁻¹·π + Pᵀ·Ω⁻¹·q]
```
**Optimal weights:**
```
w* = (δ·Σ)⁻¹ · posterior_μ
```

### 52. Hierarchical Risk Parity — Prado 2016 (`hrp.py`)
Step 1: Distance matrix from correlation:
```
d_ij = √(0.5·(1 - ρ_ij))
```
Step 2: Single-linkage clustering → quasi-diagonal order.
Step 3: Recursive bisection — split clusters by inverse-variance:
```
α = 1 - v_1 / (v_1 + v_2)
where v_k = inverse_var_portfolio(cluster_k)
```
No matrix inversion needed → numerically stable at N>50.

---

## J. Statistical Methods

### 53. Random Matrix Theory — Marchenko-Pastur (`rmt.py`)
Given correlation eigenvalues λ:
- **Clip method:** Replace λᵢ < λ_max with mean of bulk:
```
λ*ᵢ = mean(λⱼ : λⱼ ≤ λ_max)  for λᵢ in bulk
λ*ᵢ = λᵢ                       for signal
```
- Reconstruct: `C* = V · diag(λ*) · Vᵀ`, renormalise diagonal to 1.

### 54. Deflated Sharpe Ratio — Bailey-Prado 2014 (`deflated_sharpe.py`)
**Expected max Sharpe under null:**
```
E[max SR] = (1-γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e))
where γ = Euler-Mascheroni ≈ 0.5772, N = # trials
```
**Standard error of Sharpe (non-normal):**
```
SE(SR) = √[(1 - skew·SR + ((kurt-1)/4)·SR²) / (n-1)]
```
**Deflated Sharpe:**
```
DSR = (SR_obs - E[max SR]) / SE(SR)
```
Significant if DSR > Φ⁻¹(0.95) ≈ 1.645.

### 55. Probabilistic Sharpe Ratio
```
PSR(SR*) = Φ(((SR_obs - SR*) · √(n-1)) / √(1 - skew·SR + ((kurt-1)/4)·SR²))
```

### 56. Benjamini-Hochberg FDR (1995)
Sort p-values ascending: p_(1), p_(2), ..., p_(n).
Reject H_(i) where:
```
p_(i) ≤ (i/n) · α
```
Largest such i defines cutoff.

### 57. Bonferroni Correction
```
Reject H_i if p_i < α / n
```
Conservative — controls family-wise error.

### 58. Quantile Regression (Koenker-Bassett 1978) (`quantile_regression.py`)
Minimize asymmetric absolute loss:
```
L_q(β) = Σ ρ_q(yᵢ - xᵢᵀβ)
ρ_q(u) = u · (q - 𝟙{u<0})
```
Solved via IRLS:
```
w_t = 1 / max(|residual_t|, tol)
β_new = (XᵀWX)⁻¹ · XᵀWy
```

### 59. Fractional Differentiation (Prado 2018) (`ml_finance.py`)
For d ∈ (0, 1):
```
Δ^d X_t = Σ_{k=0}^∞ wₖ · X_{t-k}
w_0 = 1
w_k = -w_{k-1} · (d - k + 1) / k
```
Optimal d found by smallest d giving stationary series (ADF p < 0.05).

### 60. Triple Barrier Labeling (`ml_finance.py`)
For position entered at price p_0:
- Profit-take barrier: p_0 · (1 + pt%)
- Stop-loss barrier:    p_0 · (1 - sl%)
- Time barrier: T days

Label:
- +1 if PT hit first
- -1 if SL hit first
- 0 if T expires

### 61. Purged K-Fold CV (`ml_finance.py`)
Standard K-fold with **embargo**:
```
For each fold k:
  test = samples in fold k
  embargo = next (n·embargo_pct) samples
  train = all samples NOT in (test ∪ embargo)
```
Prevents leakage from autocorrelated labels.

---

## K. Microstructure

### 62. VPIN — Easley-López-O'Hara 2012 (`vpin.py`)
**Bulk Volume Classification:**
```
V_buy_i = V_total_i · Φ((P_i - P_{i-1}) / σ)
V_sell_i = V_total_i - V_buy_i
```
**Order flow toxicity (PIN):**
```
VPIN = Σ |V_buy - V_sell| / Σ V_total   (over n bars)
```
High VPIN → informed trading present.

### 63. Kalman Filter (`kalman.py`)
State equation: `x_t = F·x_{t-1} + w_t`, observation: `y_t = H·x_t + v_t`.

**Predict:**
```
x̂_{t|t-1} = F · x̂_{t-1|t-1}
P_{t|t-1} = F · P_{t-1|t-1} · Fᵀ + Q
```
**Update:**
```
K_t = P_{t|t-1} · Hᵀ · (H·P_{t|t-1}·Hᵀ + R)⁻¹
x̂_{t|t} = x̂_{t|t-1} + K_t · (y_t - H·x̂_{t|t-1})
P_{t|t} = (I - K_t·H) · P_{t|t-1}
```
We use it for time-varying β between asset and SPY.

### 64. Kyle's Lambda (1985) — price impact
```
Δp = λ · order_flow_imbalance
λ = market_impact_per_unit_volume
```
Estimated by regressing price change on signed volume.

### 65. Amihud Illiquidity (1985)
```
ILLIQ_t = |r_t| / Dollar_Volume_t
ILLIQ_period = (1/D) · Σ_t ILLIQ_t
```

---

## L. Topology

### 66. TDA Persistent Homology (`tda_signal.py`)
**Takens embedding** of price series:
```
X_t = (P_t, P_{t-τ}, P_{t-2τ}, ..., P_{t-(d-1)τ})
```
**Vietoris-Rips complex** at scale ε:
```
Edges: {x_i, x_j} if ||x_i - x_j|| ≤ ε
Higher simplices similarly
```
**Persistence barcode:** birth/death of H_0 (connected components) and H_1 (loops) as ε grows.

Long H_1 bars → cyclic regime; long H_0 → fragmented; short bars → trending.

### 67. MF-DFA Multifractal (Kantelhardt 2002) (`multifractal.py`)
Profile: `Y(i) = Σ_{k=1}^i (x_k - x̄)`.
Detrended fluctuation:
```
F²(s, ν) = (1/s) · Σ (Y(ν·s + j) - y_fit(j))²
```
q-order generalized fluctuation:
```
F_q(s) = ((1/N_s) · Σ_ν [F²(s,ν)]^(q/2))^(1/q)
```
Generalized Hurst h(q) from `F_q(s) ~ s^h(q)`.

### 68. Hurst R/S (1951)
```
R(n) = max(cumulative deviation) - min(cumulative deviation)
S(n) = standard deviation
E[R/S] ~ c · n^H
```

---

## M. Behavioural Factor Formulas

### 69. Idiosyncratic Volatility (Ang et al. 2006)
```
r_stock = α + β · r_market + ε
idio_vol = σ(ε) · √252
```
High idio vol → bearish next month.

### 70. MAX Anomaly (Bali et al. 2011)
```
MAX_t = max(r_1, r_2, ..., r_22)   (max daily return in past month)
```
High MAX → bearish next month (lottery demand).

### 71. 1-Week Reversal (Jegadeesh 1990)
```
r_1week = Π_{i=1}^5 (1 + r_i) - 1
```
Negative correlation with next week's return.

### 72. 12-1 Momentum (Jegadeesh-Titman 1993)
```
mom_12_1 = r_{t-252:t-22} (12-month skip-most-recent)
```

### 73. Long-Run Reversal (DeBondt-Thaler 1985)
```
r_3yr = (P_t / P_{t-700}) - 1
```
3-5 year losers reverse upward.

### 74. 52-Week High Momentum (George-Hwang 2004)
```
prox_52w = P_t / max(P_{t-252:t})
```

### 75. Momentum Crash (Daniel-Moskowitz 2016)
```
crash_setup = (drawdown_6mo < -15%) AND (return_1mo > +8%)
```
Past bear bottom + sharp bounce → momentum strategies crash.

---

## N. Fundamental Score Formulas

### 76. Piotroski F-Score (2000)
9-point binary scoring across:
- ROA > 0
- CFO > 0
- ΔROA > 0
- Accruals: CFO > NI
- ΔLeverage < 0
- ΔCurrent Ratio > 0
- No share issuance
- ΔGross Margin > 0
- ΔAsset Turnover > 0

### 77. Altman Z-Score (1968)
```
Z = 1.2·X₁ + 1.4·X₂ + 3.3·X₃ + 0.6·X₄ + 1.0·X₅
X₁ = WC/TA
X₂ = RE/TA
X₃ = EBIT/TA
X₄ = MarketCap/TotalLiab
X₅ = Sales/TA
```
Z < 1.81: distress. Z > 2.99: safe.

### 78. Beneish M-Score (1999)
```
M = -4.84 + 0.92·DSRI + 0.528·GMI + 0.404·AQI + 0.892·SGI
       + 0.115·DEPI - 0.172·SGAI + 4.679·TATA - 0.327·LVGI
```
M > -1.78 → likely earnings manipulator.

### 79. Graham Number (intrinsic floor)
```
Graham = √(22.5 · EPS · BVPS)
```
Discount: undervalued if Price < Graham.

### 80. DCF — Gordon Growth (with **dynamic WACC**)
```
DCF = Σ_{i=1}^5 EPS·(1+g)^i / (1+WACC)^i + TV/(1+WACC)^5
TV = EPS_5 · (1+g_terminal) / (WACC - g_terminal)

WACC = r_f + β · ERP    (dynamic in our impl)
g_terminal = max(0.01, min(0.04, r_f - 0.015))
```

### 81. PEAD — Post-Earnings Drift
```
SUE = (EPS_actual - EPS_expected) / σ_EPS
PEAD signal: take long when SUE > 1, drift continues 60-90 days
```

### 82. Gross Profitability (Novy-Marx 2013)
```
GP = (Revenue - COGS) / Total Assets
```
High GP → outperformance.

### 83. Asset Growth Anomaly (Cooper et al. 2008)
```
AG = (Total Assets_t - Total Assets_{t-1}) / Total Assets_{t-1}
```
High AG > 20% → underperformance.

### 84. Net Issuance (Daniel-Titman)
```
NI% = (Shares_t - Shares_{t-1}) / Shares_{t-1}
```
Positive = dilution = bearish.

### 85. Investment-to-Assets (Hou-Xue-Zhang q-factor)
```
I/A = CapEx / Total Assets
```
High I/A → over-investment → underperformance.

### 86. QMJ Composite (Asness 2019)
```
Quality = (Profitability + Growth + Safety) / 3
Profitability ~ ROE + GrossMargin
Safety ~ 1 - debt/equity
Growth ~ revenue_growth
QMJ_score = standardize(Quality)
```

---

## O. Macro Formulas

### 87. Nelson-Siegel Yield Curve (1987)
```
y(τ) = β_0 + β_1·((1-e^(-λτ))/(λτ)) + β_2·((1-e^(-λτ))/(λτ) - e^(-λτ))
```
- β_0 = Level (long-term)
- β_1 = Slope (10Y - short)
- β_2 = Curvature (humping)

In our impl (simplified):
```
Level = average across 3M/2Y/5Y/10Y/30Y
Slope = 30Y - 3M
Curvature = 2·10Y - 3M - 30Y
```

### 88. Equity Risk Premium
```
ERP = E/P - r_real
where r_real = 10Y_nominal - 10Y_TIPS_breakeven
```

### 89. Real Interest Rate
```
r_real = DGS10 - T10YIE
```

### 90. Business Cycle Phase
```
Recovery:   Leading↑, Coincident↓
Expansion:  Leading↑, Coincident↑
Slowdown:   Leading↓, Coincident↑
Contraction: Leading↓, Coincident↓
```

### 91. FRED Nowcast Composite
```
Component_z = (current - mean) / std × sign
where sign inverts for inverted indicators (e.g. claims)
Composite_z = mean(component_z)
```

---

## P. Options

### 92. Black-Scholes (foundational)
```
Call = S·Φ(d_1) - K·e^(-rT)·Φ(d_2)
d_1 = (ln(S/K) + (r + σ²/2)·T) / (σ·√T)
d_2 = d_1 - σ·√T
```

### 93. Greeks (used in `options_intel.py`)
```
Δ = ∂C/∂S = Φ(d_1)
Γ = ∂²C/∂S² = φ(d_1) / (S·σ·√T)
Vega = ∂C/∂σ = S·φ(d_1)·√T
Θ = ∂C/∂t
Vanna = ∂Δ/∂σ
Charm = ∂Δ/∂t
```

### 94. Gamma Exposure (GEX)
```
GEX = Σ_strikes OI · Γ · contract_multiplier · sign(call/put)
```

### 95. IV Skew (25Δ Put-Call)
```
Skew = IV_25Δ_put - IV_25Δ_call
```
Positive skew = expensive put protection = bearish hedging.

### 96. Implied Move (Straddle)
```
implied_move% = (ATM_call_price + ATM_put_price) / S_0
```

### 97. Realized Skewness
```
Skew_realized = E[(r - μ)³] / σ³
```
Negative realized skew → crash risk.

---

## Q. Sentiment / NLP

### 98. Transfer Entropy (causality)
```
TE(X→Y) = Σ p(y_{t+1}, y_t, x_t) · log[p(y_{t+1}|y_t, x_t) / p(y_{t+1}|y_t)]
```
We use Pearson correlation lag-1 as proxy.

### 99. News Decay Model
```
weight_i = exp(-λ · age_in_days_i)
λ = ln(2) / halflife
sentiment_weighted = Σ wᵢ·polarityᵢ / Σ wᵢ
```
Halflife is **VIX-adaptive** in our impl.

### 100. Reddit Sentiment Score
```
bullish_score = #bullish_posts / total_posts
keywords scored via Gemini NLP
```

---

# PART III — ALL 271 FACTORS: DYNAMIC vs STATIC

## DYNAMIC FACTORS (adapt to market regime / data)

| # | Factor | What Makes It Dynamic |
|---|--------|----------------------|
| 1 | DCF Implied Upside | WACC = rf + β·ERP, live from ^TNX |
| 2 | DCF terminal growth | rf − 1.5%, capped [1%, 4%] |
| 3 | Flash Crash Detection | 3 × realised daily vol (not −7%) |
| 4 | Bollinger %B | std = 2.5 if VIX>25, 1.75 if VIX<15, linear blend 15-25 |
| 5 | Bollinger Bandwidth | Inherits VIX-adaptive std |
| 6 | News Decay Model | Halflife 2d/3.5d/4.6d/6d by VIX regime |
| 7 | Adaptive RSI (VIX-adjusted) | Thresholds 26/74 if VIX<20, 32/68 if VIX<30, 38/62 else |
| 8 | Bayesian Direction Gate | Entropy>0.85: 0.44/0.56; Entropy<0.40: 0.485/0.515 |
| 9 | Regime Weights | Soft-blend via HMM P(bull), P(bear), P(crisis) |
| 10 | HMM XLF Block | Triggered when transition_risk > 0.20 |
| 11 | Sector Gap Threshold | BULL: 0.001; BEAR: 0.010; CRISIS: 0.015 |
| 12 | Position Multiplier | Risk-regime conditional [0, 1] |
| 13 | HMM Size Scalar | max(0.25, min(1.0, bull_prob × 1.5)) |
| 14 | P/E Cheap/Fair/Expensive | Dynamic: 1/(rf + 4% ERP) → P/E cheap |
| 15 | Yang-Zhang Vol | k weight = 0.34 / (1.34 + (n+1)/(n-1)), window-adaptive |
| 16 | GARCH Vol Regime | Percentile-based (LOW/NORMAL/HIGH/EXTREME) on rolling 252d |
| 17 | EVT Tail Risk (99% VaR) | ξ and β re-fit each call from POT exceedances |
| 18 | Hawkes Branching Ratio | α, β re-estimated from recent event clusters |
| 19 | Kelly Position Size | f* = (μ-rf)/σ², σ from realised vol |
| 20 | DCC-GARCH Dynamic Correlation | ρ_t updates each step; Q_t recursion |
| 21 | Kalman Dynamic Beta | β_t evolves as random walk via Kalman update |
| 22 | Variance Risk Premium | IV² − RV², both update intraday |
| 23 | Vol Arbitrage Signal | Re-z-scored against rolling 60d VRP distribution |
| 24 | News Halflife (sentiment agent) | Shrinks 0–7d to earnings (1.5d), 7–21d (2.8d) |
| 25 | News Halflife (VIX adjust) | min(base, 2.0) if VIX>30, min(base, 3.5) if VIX>20, max(base, 6.0) if VIX<14 |
| 26 | Bayesian Sensitivity | Could decay as 1.0 + 0.4·(1 − entropy) (current default 1.4) |
| 27 | Credit Spread Multipliers | 5x default, intended to be regime-conditional |
| 28 | Signal Decay Half-life | Re-estimated per ticker via OU fit |
| 29 | HMM Regime Probabilities | bull_prob, bear_prob, crisis_prob update each invocation |
| 30 | Soft Regime Blend Weights | Σ(p_regime × W_regime) renormalised each signal |
| 31 | RMT Eigenvalue Cleaning | λ_max threshold depends on T/N ratio |
| 32 | Black-Litterman Posterior | π = δ·Σ·w_market, updates with market caps |
| 33 | HRP Weights | Recomputed via clustering on current corr matrix |
| 34 | Quantile Predictions | 5/25/50/75/95 quantiles re-fit per window |
| 35 | Fractional Diff d | Optimised per series via ADF p-value sweep |
| 36 | Triple Barrier (PT, SL) | Set as % of realised vol typically |
| 37 | CUSUM Threshold | k=0.5σ default but can be set per series |
| 38 | Transaction Cost Spread | 1/3/10/30 bps by ADV tier (auto-selected) |
| 39 | Market Impact | const · daily_vol · √participation, adaptive to ADV |
| 40 | Borrow Cost | annualised rate × holding_days |
| 41 | Stress Scenarios | Beta-scaled by current portfolio gross exposure |
| 42 | FRED Nowcast Composite | Updates with new monthly data releases |
| 43 | COT Smart Money Signal | Z-scored against 52-week commercials net |
| 44 | Weather Anomaly | HDD/CDD anomaly vs seasonal norms |
| 45 | EIA Inventory Pressure | z-scored 10d vs 60d return on WTI |
| 46 | Google Trends | z-scored vs 3-month attention distribution |
| 47 | ETF Premium/Discount | Z-scored against 60-day premium distribution |
| 48 | Commodity Roll Yield | Annualised gap = ETF return − spot return |
| 49 | Realized Skewness | Window of 60 daily returns |
| 50 | Realized Vol Trend (10d vs 30d) | Ratio always recomputed |
| 51 | Correlation Network Centrality | Eigenvector centrality of |corr| matrix |
| 52 | Idiosyncratic Volatility | β recomputed against SPY each call |
| 53 | Black-Litterman Risk Aversion | Could be dynamic (currently 2.5 default) |
| 54 | Meta-Learner Blend Weight | LightGBM-learned; updates when model retrains |
| 55 | Calibration A, B (Platt) | Re-fit on rolling signal history |
| 56 | Information Coefficient (IC) | Rolling 60d Spearman per agent |
| 57 | Forward P/E (sector-adjusted) | Could be adjusted to sector median (not yet wired) |
| 58 | Drawdown from ATH | 2y rolling high vs current price |
| 59 | Tail Ratio | 95th / |5th| percentile, period-adaptive |
| 60 | Rolling Sharpe / Sortino (63d) | Rolling 63d window |

## STATIC FACTORS (fixed thresholds / academic constants)

| # | Factor / Parameter | Value | Source |
|---|--------------------|-------|--------|
| 1 | RSI window | 14 | Wilder standard |
| 2 | MACD periods | 12/26/9 | Appel standard |
| 3 | ADX window | 14 | Wilder |
| 4 | ATR window | 14 | Wilder |
| 5 | Stochastic window | 14 | Lane |
| 6 | Stochastic smooth | 3 | Standard |
| 7 | Williams %R window | 14 | Standard |
| 8 | EMA periods | 9 / 21 / 50 / 200 | Common |
| 9 | Ichimoku Tenkan | 9 | Hosoda |
| 10 | Ichimoku Kijun | 26 | Hosoda |
| 11 | Ichimoku Senkou shift | 26 days forward | Hosoda |
| 12 | Ichimoku Senkou B | 52 | Hosoda |
| 13 | Bollinger window | 20 | Standard |
| 14 | Bollinger std (default) | 2.0 σ | Standard (overridden dynamically) |
| 15 | Parabolic SAR step | 0.02 | Wilder |
| 16 | Parabolic SAR max AF | 0.20 | Wilder |
| 17 | TRIX EMA span | 15 | Hutson |
| 18 | Chaikin Money Flow period | 14 | Chaikin |
| 19 | Hurst exponent regime | 0.55 trend, 0.45 mean-revert | Standard |
| 20 | Fibonacci levels | 0.236, 0.382, 0.500, 0.618, 0.786 | Math constants |
| 21 | Pivot lookback (weekly) | 7 days | Standard |
| 22 | Backtest calibration lookback | 126 days | Empirical |
| 23 | Chart pattern scan window | 60 bars | Empirical |
| 24 | Volume profile bins | 20 | Standard |
| 25 | Volume profile value area | 0.70 (70%) | Steidlmayer |
| 26 | GEX time-to-exp days | 21 | ~1 month approximation |
| 27 | GEX risk-free rate (fallback) | 4.5% | Should pull from ^TNX |
| 28 | Implied correlation thresholds | 0.8 macro / 0.4 idio | Empirical |
| 29 | Piotroski cutoff (good) | 6/9 | Piotroski |
| 30 | Altman distress threshold | 1.81 | Altman |
| 31 | Altman safe threshold | 2.99 | Altman |
| 32 | Beneish manipulation cutoff | -1.78 | Beneish |
| 33 | Graham multiplier | 22.5 | Graham (1949) |
| 34 | DCF projection years (Stage 1) | 5 | Standard |
| 35 | DCF max growth rate | 30% | Cap |
| 36 | DCF min growth rate | -15% | Floor |
| 37 | P/B undervalued | 1.5 | Empirical |
| 38 | P/B fair | 3.0 | Empirical |
| 39 | PEG undervalued | 1.0 | Peter Lynch |
| 40 | PEG fair | 2.0 | Common |
| 41 | EV/EBITDA cheap | 10 | Empirical |
| 42 | EV/EBITDA fair | 20 | Empirical |
| 43 | EV/EBITDA expensive | 30 | Empirical |
| 44 | P/S cheap | 2.0 | Empirical |
| 45 | P/S fair | 4.0 | Empirical |
| 46 | FCF yield good | 2.0% | Empirical |
| 47 | FCF yield great | 5.0% | Empirical |
| 48 | DCF upside strong | 30% | Empirical |
| 49 | DCF upside moderate | 10% | Empirical |
| 50 | Revenue growth strong | 20% | Empirical |
| 51 | Earnings growth strong | 20% | Empirical |
| 52 | ROE strong | 20% | Empirical |
| 53 | ROA strong | 15% | Empirical |
| 54 | Operating margin strong | 25% | Empirical |
| 55 | Current ratio liquid | 2.0 | Standard |
| 56 | Interest coverage strong | 10x | Standard |
| 57 | Asset turnover good | 1.0 | Standard |
| 58 | Gross margin strong | 40% | Empirical |
| 59 | EPS surprise beat | 10% | Common |
| 60 | Quality-value blend | 65% / 35% | Empirical |
| 61 | Accruals high-quality | < -0.05 | Sloan |
| 62 | Bankruptcy WACC fallback | 10% | Replaced dynamically |
| 63 | Dividend cut threshold | Payout > 90% | Common |
| 64 | IPO lockup window | 0-180 days | SEC standard |
| 65 | FF Value transitions | Multiple thresholds | Fama-French |
| 66 | Yield curve positive | +0.5% | Empirical |
| 67 | Yield curve inverted | −0.5% | Empirical |
| 68 | Fed rate easy | 2.0% | Empirical |
| 69 | Fed rate tight | 4.0% | Empirical (should be neutral r*) |
| 70 | CPI target | 2.5% | Fed |
| 71 | CPI elevated | 5.0% | Empirical |
| 72 | M2 growth supportive | > 5% | Empirical |
| 73 | M2 growth concerning | < 0% | Empirical |
| 74 | PMI expansion threshold | 50 | ISM standard |
| 75 | PMI strong | 55 | ISM |
| 76 | PMI weak | 48 | ISM |
| 77 | Consumer sentiment strong | 90 | UMich |
| 78 | Consumer sentiment weak | 55 | UMich |
| 79 | Claims low | 220K | Empirical |
| 80 | Claims elevated | 260K | Empirical |
| 81 | Claims high | 350K | Empirical |
| 82 | TIPS breakeven low | 1.5% | Fed target zone |
| 83 | TIPS breakeven high | 2.5% | Fed target zone |
| 84 | TIPS breakeven danger | 3.0% | Fed target zone |
| 85 | Amihud stress | 2.0× avg | Empirical |
| 86 | Bond-equity corr stress | 0.30+ | Empirical |
| 87 | SOFR critical spread | 0.50% (50bps) | Money market |
| 88 | Fed rate change hiking | > +0.24% in 3M | Empirical |
| 89 | Fed rate change cutting | < −0.24% in 3M | Empirical |
| 90 | VIX 5d spike | > +3 points | Empirical |
| 91 | VIX 5d collapse | < −3 points | Empirical |
| 92 | SPY 200-SMA bullish | > +5% | Empirical |
| 93 | SPY 200-SMA bearish | < −5% | Empirical |
| 94 | Real rate positive | > 1.5% | Empirical |
| 95 | Real rate elevated | > 2.0% | Empirical |
| 96 | PCE target (Fed) | 2.0% | Fed |
| 97 | NFCI tight | > 0.5 | Chicago Fed |
| 98 | ERP attractive | > 4.0% | Empirical |
| 99 | ERP compressed | < 0% | Empirical |
| 100 | Short interest high | > 20% float | Common |
| 101 | Short interest low | < 3% float | Common |
| 102 | Fear & Greed extreme low | ≤ 25 | CNN |
| 103 | Fear & Greed extreme high | ≥ 75 | CNN |
| 104 | Analyst bullish | > 70% buy | Empirical |
| 105 | Analyst bearish | > 30% sell | Empirical |
| 106 | Options P/C bullish | < 0.7 | Common |
| 107 | Options P/C bearish | > 1.2 | Common |
| 108 | EPS forward revision beat | > 15% | Empirical |
| 109 | Options unusual activity | Vol/OI > 1.5× | Common |
| 110 | Options sweep threshold | Vol/OI > 2.0× | Common |
| 111 | AAII extreme bullish | > +30% spread | AAII |
| 112 | AAII extreme bearish | < −20% spread | AAII |
| 113 | Consumer credit growing | > 2% (3M) | Empirical |
| 114 | News halflife (base) | 4.6 days | Empirical (overridden dynamically) |
| 115 | Reddit bullish threshold | > 60% of posts | Empirical |
| 116 | Reddit post processing limit | 15 / subreddit | Capacity |
| 117 | Short squeeze high-risk | SI > 20% + DTC > 5d | Empirical |
| 118 | Earnings whisper elevated | > 1.5× historical | Empirical |
| 119 | Headline/body divergence | > 20pp | Empirical |
| 120 | Bullish polarity (news) | 0.65 | Empirical |
| 121 | Bearish polarity (news) | 0.35 | Empirical |
| 122 | Black swan sigma | 5.0 σ | Empirical |
| 123 | Flash crash ticker (fallback) | −7% | Replaced dynamically |
| 124 | Flash crash SPY (fallback) | −5% | Replaced dynamically |
| 125 | Geo shock VIX trigger | 35.0 | Empirical |
| 126 | Geo shock gold | +2% in 1d | Empirical |
| 127 | Geo shock oil | +5% in 1d | Empirical |
| 128 | Carry unwind USD/JPY | 125 | Empirical |
| 129 | Carry unwind JPY surge | −1.5% in 1d | Empirical |
| 130 | EVT threshold percentile | 10 (bottom 10%) | Standard |
| 131 | EVT VaR level (95) | 0.95 | Standard |
| 132 | EVT VaR level (99) | 0.99 | Standard |
| 133 | Hawkes min events | 10 | Statistical floor |
| 134 | Hawkes warning branching | 0.80 | Empirical |
| 135 | Hawkes critical branching | 0.95 | Empirical |
| 136 | Kelly fraction | 0.5 (half-Kelly) | Conservative |
| 137 | Kelly max cap | 0.25 | Conservative |
| 138 | MC paths (LOW regime) | 3,000 | Capacity |
| 139 | MC paths (NORMAL) | 5,000 | Capacity |
| 140 | MC paths (HIGH) | 8,000 | Capacity |
| 141 | MC paths (EXTREME) | 10,000 | Capacity |
| 142 | MC drift daily | 0.0005 | Empirical |
| 143 | GARCH percentile LOW | < 25th | Standard |
| 144 | GARCH percentile NORMAL | 25-75th | Standard |
| 145 | GARCH percentile HIGH | 75-95th | Standard |
| 146 | GARCH percentile EXTREME | > 95th | Standard |
| 147 | KL divergence regime shift | 1.0 | Empirical |
| 148 | Liquidity ratio high | 1.5× | Empirical |
| 149 | Liquidity ratio low | 0.7× | Empirical |
| 150 | Drawdown warning | −30% from ATH | Empirical |
| 151 | Tail ratio bullish | 1.2+ | Empirical |
| 152 | Tail ratio bearish | 0.8 | Empirical |
| 153 | Correlation delta threshold | 0.10 | Empirical |
| 154 | Sharpe high | > 1.5 | Empirical |
| 155 | Sortino excellent | > 2.0 | Empirical |
| 156 | Vanna negative threshold | −0.5 | Empirical |
| 157 | Skew left-tail | < −1.0 | Statistical |
| 158 | Kurtosis fat tails | > 5.0 excess | Statistical |
| 159 | GEX critical negative | < −5.0M | Empirical |
| 160 | DCC-GARCH a | 0.05 | Engle calibration |
| 161 | DCC-GARCH b | 0.93 | Engle calibration |
| 162 | Univariate GARCH α (DCC) | 0.10 | Equity calibration |
| 163 | Univariate GARCH β (DCC) | 0.85 | Equity calibration |
| 164 | RMT clip method | mean of bulk | Standard |
| 165 | Marchenko-Pastur formula | (1±√(N/T))² | Math (1967) |
| 166 | CUSUM k (sensitivity) | 0.5 σ | Standard |
| 167 | CUSUM h (threshold) | 5.0 σ | Standard |
| 168 | Black-Litterman δ | 2.5 | Standard |
| 169 | Black-Litterman τ | 0.05 | Standard |
| 170 | HRP linkage method | "single" | Standard |
| 171 | Transaction cost commission/share | $0.005 | IBKR rate |
| 172 | Transaction cost spread (large) | 1 bp | ADV > $1B |
| 173 | Transaction cost spread (mid) | 3 bps | $100M-1B |
| 174 | Transaction cost spread (small) | 10 bps | $10M-100M |
| 175 | Transaction cost spread (micro) | 30 bps | < $10M |
| 176 | Market impact constant | 0.10 | Empirical |
| 177 | Market impact participation cap | 30% ADV | Liquidity floor |
| 178 | Short borrow annual | 100 bps | Default |
| 179 | Stress scenario GFC 2008 | −40% equity | Historical |
| 180 | Stress scenario COVID 2020 | −34% equity | Historical |
| 181 | Stress scenario Rate Spike 2022 | −25% equity | Historical |
| 182 | Stress scenario Flash 2010 | −9% equity | Historical |
| 183 | Stress scenario Dot Com 2000 | −49% equity | Historical |
| 184 | Vol of vol VVIX extreme | > 120 | Empirical |
| 185 | VIX term backwardation | VIX > VIX3M | Definition |
| 186 | VIX contango (calm) | VIX3M − VIX > +1 | Empirical |
| 187 | MOVE Index elevated | > 100 | Empirical |
| 188 | MOVE Index extreme | > 130 | Empirical |
| 189 | TED spread elevated | > 0.50% | Empirical |
| 190 | TED spread crisis | > 1.0% | Empirical |
| 191 | Asset growth high | > 20% | Cooper |
| 192 | Gross profitability high | > 40% | Novy-Marx |
| 193 | Investment-to-assets high | > 15% | q-factor |
| 194 | Net issuance bad | > 5% | Daniel-Titman |
| 195 | R&D anomaly high | > 8% | Chan |
| 196 | QMJ bullish | > 0.65 normalised | AQR |
| 197 | Idio vol extreme | > 60% annualised | Ang |
| 198 | MAX anomaly bearish | > 7% single-day | Bali |
| 199 | 1W reversal strong | < −5% past week | Jegadeesh |
| 200 | Long-run reversal | < −30% past 3y | DeBondt-Thaler |
| 201 | 52W high proximity bullish | > 97% | George-Hwang |
| 202 | Momentum crash setup | DD < −15% + 1M > +8% | Daniel-Moskowitz |
| 203 | Roll yield contango (avoid) | < −10% annual | Empirical |
| 204 | Roll yield backwardation | > +10% annual | Empirical |
| 205 | COT smart money z | > +1.5 or < −1.5 | Empirical |
| 206 | Weather HDD/CDD anomaly | > +3 | Empirical |
| 207 | EIA inventory z | > +1 or < −1 | Empirical |
| 208 | Google Trends z | > +1.5 / < −1.0 | Empirical |
| 209 | ETF premium signal | z > +1.5 (sell) / < −1.5 (buy) | Empirical |
| 210 | Vol arbitrage VRP z | > +1 SHORT / < −1 LONG | Empirical |
| 211 | Soft-blend bull weight | p_bull − 0.30 (chop) | Heuristic |

---

# PART IV — CIRCUIT BREAKER OVERRIDE FORMULAS

```
Priority 1: BLACK_SWAN
  Trigger:  max |Zₜ| in 5d > 5σ
            Zₜ = (rₜ − μ_252d) / σ_252d
  Action:   halt=True, multiplier=0, P(up)=0.10

Priority 2: FLASH_CRASH
  Trigger:  rₜ_ticker < −max(3·σ_ticker_daily, 3%)
            OR rₜ_SPY < −max(3·σ_SPY_daily, 2%)
  Action:   halt=True, multiplier=0, P(up)=0.15

Priority 3: CRITICAL_RISK
  Trigger:  GARCH regime == EXTREME
            OR EVT VaR_99 < −8%
  Action:   halt=False, multiplier=0.25, P(up)=0.20

Priority 4: HIGH_RISK
  Trigger:  GARCH regime == HIGH
            OR EVT VaR_95 < −5%
  Action:   multiplier=0.50, P(up)=0.40

Priority 5: GEO_SHOCK
  Trigger:  VIX > 35 AND (Gold +2% in 1d OR Oil +5% in 1d)
  Action:   multiplier=0.35, P(up)=0.30

Priority 6: CARRY_UNWIND
  Trigger:  USD/JPY level > 125 AND JPY surge < −1.5% in 1d
  Action:   multiplier=0.50

Priority 7: GEOPOLITICAL OVERRIDE
  Trigger:  Geo-political agent flags
  Action:   multiplier = min(0.35, current)
```

---

# PART V — DIRECTION GATE FORMULAS

```
Soft Regime Blend in Orchestrator:
  W_blended[agent] = p_bull · W_BULL_TREND[agent]
                   + max(0, p_bull − 0.30) · W_BULL_CHOPPY[agent]
                   + p_bear · W_BEAR[agent]
                   + p_crisis · W_CRISIS[agent]
  Renormalise: W[agent] / Σ W[agent]

Entropy-Adaptive Direction Gate:
  if entropy > 0.85:
      LONG_gate, SHORT_gate = 0.56, 0.44      (high disagreement → conservative)
  elif entropy > 0.70:
      LONG_gate, SHORT_gate = 0.545, 0.455
  elif entropy < 0.40:
      LONG_gate, SHORT_gate = 0.515, 0.485     (strong consensus → act on small edge)
  else:
      LONG_gate, SHORT_gate = 0.53, 0.47       (default)

  if final_prob > LONG_gate: direction = LONG
  elif final_prob < SHORT_gate: direction = SHORT
  else: direction = HOLD

Dynamic Prior (replaces flat 0.5):
  if SPY > SMA50 · 1.005:  prior = 0.53
  elif SPY < SMA50 · 0.995: prior = 0.47
  else: prior = 0.50

Conviction:
  conviction = |final_prob − 0.5| · 2.0     (0 to 1 scale)
  Confidence level:
    HIGH   if conviction > 0.4
    MEDIUM if conviction > 0.2
    LOW    otherwise
```

---

# PART VI — POSITION SIZING & P&L

```
Position Notional:
  notional = CAPITAL × multiplier × hmm_size_scalar × conviction_weight
  hmm_size_scalar = max(0.25, min(1.0, p_bull × 1.5))

Gross P&L:
  if direction == LONG:  gross_pnl = notional × (P_close / P_open − 1)
  elif direction == SHORT: gross_pnl = notional × (P_open / P_close − 1)

Transaction Costs (round trip):
  spread_bps        = 1 if ADV>$1B else 3 if ADV>$100M else 10 if ADV>$10M else 30
  spread_cost       = notional × (spread_bps / 10000)            (half-spread × 2 legs)
  impact_bps        = 0.10 × daily_vol_bps × √(min(notional/ADV, 0.30))
  impact_cost       = notional × (impact_bps / 10000)
  commission        = (notional / price) × $0.005                 (per share)
  borrow_cost (SHORT only) = notional × (100 bps / 10000) × (days / 252)
  tcm_total         = 2 × (spread_cost + impact_cost + commission) + borrow_cost

Net P&L:
  net_pnl = gross_pnl − tcm_total

Portfolio Return:
  port_return = Σ net_pnl_i / Σ notional_i
```

---

# PART VII — SUMMARY STATS

| Metric | Count |
|--------|-------|
| Mathematical formulas documented | 100+ |
| Quant theories with full equations | 60+ |
| Dynamic factors / parameters | 60 |
| Static factors / parameters | 211 |
| **Total factors** | 271 |
| Quant engine modules | 37 |
| Free data sources | 10 |
| Circuit breakers | 7 |
| Stress scenarios | 5 historical |

---

**End of Mathematical Reference.** Pair with:
- `PROJECT_AUDIT.md` for architecture
- `COMPLETE_INVENTORY.md` for factor list
- `PROJECT_GRAPH.json` for queryable graph
- `PROJECT_GRAPH.md` for visual diagrams
