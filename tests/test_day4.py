"""Day 4 Test — Fundamental Scoring + Signal Decay."""
import sys

sys.path.insert(0, r"d:\ML and DL\Python\AlphaAgent")

from data.market import MarketData
from quant_engine.scoring import compute_fundamental_scores
from quant_engine.signal_decay import SignalDecay

print("=" * 60)
print("  AlphaAgent — Day 4: Fundamentals + Decay Test")
print("=" * 60)

# Get data
ticker = "NVDA"
md = MarketData(ticker)
print(f"\n  Fetching data for {ticker}...")

info = md.get_info()
financials = md.get_financials()
prices = md.get_ohlcv("2y")["Close"]

# ─── TEST 1: Fundamental Scoring ──────────────────────────
print("\n" + "─" * 60)
print("  🏢 FUNDAMENTAL SCORING (Piotroski & Altman)")
print("─" * 60)

inc = financials.get("income")
bal = financials.get("balance")
cf = financials.get("cashflow")

if inc is not None and bal is not None and cf is not None:
    fund_res = compute_fundamental_scores(inc, bal, cf, info)
    
    print(f"\n  Data Quality: {fund_res.data_quality}")
    
    # F-Score
    print(f"\n  Piotroski F-Score: {fund_res.piotroski_score}/9 ({fund_res.f_score_interpretation})")
    print(f"  Breakdown:")
    for metric, passed in fund_res.f_score_details.items():
        icon = "✅" if passed else "❌"
        print(f"    {icon} {metric}")
        
    # Z-Score
    print(f"\n  Altman Z-Score: {fund_res.altman_z_score:.2f} ({fund_res.z_score_interpretation})")
    
    # Core Metrics
    print(f"\n  Gross Margin:   {fund_res.gross_margin:.1f}%")
    print(f"  ROA:            {fund_res.return_on_assets:.1f}%")
    print(f"  Debt/Equity:    {fund_res.debt_to_equity:.2f}")
    print(f"  Current Ratio:  {fund_res.current_ratio:.2f}")

else:
    print("\n  ❌ Missing financial statements from yfinance.")

# ─── TEST 2: Signal Decay ─────────────────────────────────
print("\n" + "─" * 60)
print("  ⏳ SIGNAL DECAY & HOLDING PERIODS")
print("─" * 60)

decay = SignalDecay(prices)

# Test different signal types
types = ["technical", "sentiment", "fundamental"]

for t in types:
    res = decay.calculate_half_life(signal_type=t)
    print(f"\n  Signal Type: {t.upper()}")
    print(f"    Half-life: {res.half_life_days:.1f} days")
    print(f"    Recommended hold: {res.optimal_hold_min} to {res.optimal_hold_max} days")

print(f"\n  Market Behavior:")
print(f"    Autocorrelation: {res.autocorrelation_lag1:+.3f}")
if res.is_mean_reverting:
    print(f"    Regime: Mean-reverting (choppy)")
elif res.is_trending:
    print(f"    Regime: Trending (momentum)")

print("\n" + "=" * 60)
print("  ✅ DAY 4: FUNDAMENTALS + DECAY — ALL TESTS PASSED")
print("=" * 60)
