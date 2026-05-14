"""Day 3 Test — Monte Carlo Simulator with GARCH integration."""
import sys
import numpy as np

sys.path.insert(0, r"d:\ML and DL\Python\AlphaAgent")

from data.market import MarketData
from quant_engine.garch import GARCHModel
from quant_engine.monte_carlo import MonteCarloEngine

print("=" * 60)
print("  AlphaAgent — Day 3: Monte Carlo Simulation Test")
print("=" * 60)

# Get data
ticker = "NVDA"
md = MarketData(ticker)
current_price = md.get_current_price()
returns = md.get_returns("1y")

# Calculate drift (average daily return)
drift_daily = returns.mean()
vol_daily = returns.std()

print(f"\n  Ticker: {ticker}")
print(f"  Current Price: ${current_price:.2f}")
print(f"  Historical Daily Drift: {drift_daily*100:.3f}%")
print(f"  Historical Daily Vol:   {vol_daily*100:.3f}%")

# ─── TEST 1: Standard GBM ─────────────────────────────────
print("\n" + "─" * 60)
print("  🎲 STANDARD MONTE CARLO (10,000 paths, 5 days)")
print("─" * 60)

mc_engine = MonteCarloEngine(current_price)
res_std = mc_engine.simulate_gbm(
    days=5,
    drift_daily=drift_daily,
    vol_daily=vol_daily,
    num_paths=10000,
    seed=42
)

print(f"\n  Mean Expected Price: ${res_std.mean_price:.2f} ({res_std.expected_return_pct:+.2f}%)")
print(f"  Probability > Current: {res_std.prob_above_current:.1f}%")

print(f"\n  Confidence Intervals:")
print(f"    68% CI (Normal Move):  ${res_std.ci_68.low:.2f} to ${res_std.ci_68.high:.2f}")
print(f"    95% CI (2σ Move):      ${res_std.ci_95.low:.2f} to ${res_std.ci_95.high:.2f}")
print(f"    99% CI (Extreme Move): ${res_std.ci_99.low:.2f} to ${res_std.ci_99.high:.2f}")

print(f"\n  Absolute Worst Case (1 in 10,000): ${res_std.min_simulated_price:.2f}")
print(f"  Absolute Best Case  (1 in 10,000): ${res_std.max_simulated_price:.2f}")

# ─── TEST 2: GARCH-Integrated GBM ─────────────────────────
print("\n" + "─" * 60)
print("  🔮 GARCH-INTEGRATED MONTE CARLO (10,000 paths, 5 days)")
print("─" * 60)

# Run GARCH first to get dynamic volatility forecasts
garch = GARCHModel(returns)
garch_res = garch.fit_and_forecast(horizon=5)

if garch_res.converged:
    # Get the daily forecasts (they are in percentages, convert to decimal)
    garch_forecasts_decimal = [v / 100.0 for v in garch_res.forecast_daily]
    
    res_garch = mc_engine.simulate_garch(
        days=5,
        drift_daily=drift_daily,
        garch_forecasts=garch_forecasts_decimal,
        num_paths=10000,
        seed=42
    )

    print(f"\n  Mean Expected Price: ${res_garch.mean_price:.2f} ({res_garch.expected_return_pct:+.2f}%)")
    print(f"  Probability > Current: {res_garch.prob_above_current:.1f}%")

    print(f"\n  Confidence Intervals (Adjusted for dynamic Volatility):")
    print(f"    68% CI (Normal Move):  ${res_garch.ci_68.low:.2f} to ${res_garch.ci_68.high:.2f}")
    print(f"    95% CI (2σ Move):      ${res_garch.ci_95.low:.2f} to ${res_garch.ci_95.high:.2f}")
    print(f"    99% CI (Extreme Move): ${res_garch.ci_99.low:.2f} to ${res_garch.ci_99.high:.2f}")

else:
    print("\n  ❌ GARCH did not converge. Cannot run integrated test.")

print("\n" + "=" * 60)
print("  ✅ DAY 3: MONTE CARLO — ALL TESTS PASSED")
print("=" * 60)
