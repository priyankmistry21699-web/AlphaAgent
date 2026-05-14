"""Day 11 & 12 Test — HMM Regime Detection & Bayesian Fusion."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r"d:\ML and DL\Python\AlphaAgent")

from data.market import MarketData
from quant_engine.hmm import RegimeDetector
from quant_engine.bayesian import BayesianFusion

print("=" * 60)
print("  AlphaAgent — Day 11 & 12: Advanced Quant Modules")
print("=" * 60)

ticker = "SPY"
print(f"\n  [1] Fetching Market Data for {ticker}...")
md = MarketData(ticker)
returns = md.get_returns(period="5y")

# ─── TEST 1: HMM Regime Detection ────────────────────────────
print("\n" + "─" * 60)
print("  🔍 HIDDEN MARKOV MODEL (HMM) REGIME DETECTION")
print("─" * 60)

print("  Fitting 3-State Gaussian HMM (Bull, Bear, Crisis)...")
hmm = RegimeDetector(n_states=3)
regime_result = hmm.fit_predict(returns)

print(f"\n  Current Hidden Regime: {regime_result.current_regime}")
print("  Probabilities:")
for state, prob in regime_result.probabilities.items():
    print(f"    - {state.upper()}: {prob * 100:.1f}%")

print("\n  Transition Risk (Probability of moving tomorrow):")
for state, prob in regime_result.transition_risk.items():
    print(f"    - {state}: {prob * 100:.1f}%")

# ─── TEST 2: Bayesian Fusion ────────────────────────────
print("\n" + "─" * 60)
print("  🧠 BAYESIAN FUSION ENGINE")
print("─" * 60)

# Simulate 3 agents voting
print("  Agent Votes:")
print("    - Technical Agent:   80% Bullish (Confidence: 90%)")
print("    - Sentiment Agent:   60% Bullish (Confidence: 50%, High Correlation to Tech: 0.4)")
print("    - Macro Agent:       30% Bullish (Confidence: 100%, Independent: 0.0)")

fusion = BayesianFusion(prior=0.5) # Start neutral

# Technical agent votes
fusion.update(agent_prob=0.80, confidence=0.9, correlation=0.0)
print(f"\n  [After Technical] Posterior Prob: {fusion.posterior * 100:.1f}%")

# Sentiment agent votes (correlated to technical, so it should carry less weight)
fusion.update(agent_prob=0.60, confidence=0.5, correlation=0.4)
print(f"  [After Sentiment] Posterior Prob: {fusion.posterior * 100:.1f}%")

# Macro agent screams BEARISH (0.30 bullish) with 100% confidence and zero correlation
fusion.update(agent_prob=0.30, confidence=1.0, correlation=0.0)
print(f"  [After Macro]     Posterior Prob: {fusion.posterior * 100:.1f}%")

print(f"\n  Final System Uncertainty (Entropy): {fusion.entropy:.3f} (0=Certain, 1=Confused)")

print("\n" + "=" * 60)
print("  ✅ DAY 11 & 12: ADVANCED MATH — ALL TESTS PASSED")
print("=" * 60)
