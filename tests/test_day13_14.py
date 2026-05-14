"""Day 13 & 14 Test — EVT, Kelly, Signal Decay, and Options Intel."""
import sys
import pandas as pd

sys.path.insert(0, r"d:\ML and DL\Python\AlphaAgent")

from data.market import MarketData
from quant_engine.evt import ExtremeValueModel
from quant_engine.kelly import KellyCriterion
from quant_engine.signal_decay import SignalDecay
from quant_engine.momentum import MomentumEngine
from quant_engine.options_intel import analyze_options

print("=" * 60)
print("  AlphaAgent — Day 13 & 14: Risk & Optimization Math")
print("=" * 60)

ticker = "SPY"
print(f"\n  [1] Fetching Data for {ticker}...")
md = MarketData(ticker)
returns = md.get_returns("5y")
prices = md.get_ohlcv("5y")['Close']
ticker_obj = md.get_yfinance_ticker()

# ─── TEST 1: Extreme Value Theory (EVT) ──────────────────
print("\n" + "─" * 60)
print("  🌪️ EXTREME VALUE THEORY (Black Swan Risk)")
print("─" * 60)
evt_model = ExtremeValueModel(returns, threshold_percentile=0.10)
evt_result = evt_model.fit_and_calculate()

print(f"  95% VaR:  {evt_result.var_95 * 100:+.2f}%")
print(f"  99% VaR:  {evt_result.var_99 * 100:+.2f}% (1-in-100 days worst loss)")
print(f"  99% CVaR: {evt_result.cvar_99 * 100:+.2f}% (Average loss if 99% VaR is breached)")
print(f"  Tail Index (Xi): {evt_result.tail_index:.3f}")
for w in evt_result.warnings:
    print(f"  ⚠️ {w}")


# ─── TEST 2: Kelly Criterion ────────────────────────────
print("\n" + "─" * 60)
print("  💰 KELLY CRITERION (Position Sizing)")
print("─" * 60)
# Let's pretend an agent says 65% win probability, with a 3% expected gain and 1.5% expected loss (Win/Loss = 2.0)
kelly = KellyCriterion(prob_win=0.65, expected_win_pct=0.03, expected_loss_pct=0.015)
kelly_res = kelly.calculate(current_volatility=0.015)

print("  Assumptions: 65% Win Probability, 2:1 Reward/Risk")
print(f"  Full Kelly (Aggressive):   {kelly_res.full_kelly * 100:.1f}% of Portfolio")
print(f"  Half Kelly (Standard):     {kelly_res.half_kelly * 100:.1f}% of Portfolio")
print(f"  Vol-Adjusted Kelly (Safe): {kelly_res.vol_adjusted_kelly * 100:.1f}% of Portfolio")


# ─── TEST 3: Signal Decay & Mean Reversion ──────────────
print("\n" + "─" * 60)
print("  ⏳ SIGNAL DECAY (Ornstein-Uhlenbeck)")
print("─" * 60)
decay = SignalDecay(prices)
decay_res = decay.calculate_half_life()

print(f"  Is Mean Reverting:  {decay_res.is_mean_reverting}")
print(f"  Signal Half-Life:   {decay_res.half_life_days:.1f} days")
print(f"  Optimal Hold Time:  {decay_res.optimal_hold_min} to {decay_res.optimal_hold_max} days")


# ─── TEST 4: Momentum & Hurst ───────────────────────────
print("\n" + "─" * 60)
print("  🚀 MOMENTUM (Hurst Exponent)")
print("─" * 60)
mom_engine = MomentumEngine(prices)
mom_res = mom_engine.analyze()

print(f"  Hurst Exponent:   {mom_res.hurst_exponent:.3f}")
print(f"  Market Regime:    {mom_res.regime_type}")
print(f"  12M-1M Momentum:  {mom_res.momentum_12m_1m * 100:+.2f}%")


# ─── TEST 5: Options Intelligence ───────────────────────
print("\n" + "─" * 60)
print("  📉 OPTIONS INTELLIGENCE (Put/Call Ratio)")
print("─" * 60)
opt_res = analyze_options(ticker_obj)

print(f"  Put/Call Ratio:   {opt_res.put_call_ratio:.2f}")
print(f"  Implied Move:     {opt_res.implied_move_pct * 100:.2f}%")
for w in opt_res.warnings:
    print(f"  ⚠️ {w}")


print("\n" + "=" * 60)
print("  ✅ DAY 13 & 14: ALL TESTS PASSED")
print("=" * 60)
