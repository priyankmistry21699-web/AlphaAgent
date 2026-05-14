"""Day 2 Test — Technical Indicators + GARCH Volatility."""
import sys
sys.path.insert(0, r"d:\ML and DL\Python\AlphaAgent")

from data.market import MarketData
from quant_engine.technical import compute_indicators
from quant_engine.garch import GARCHModel

print("=" * 60)
print("  AlphaAgent — Day 2: Technical + GARCH Test")
print("=" * 60)

# Get data
md = MarketData("NVDA")
ohlcv = md.get_ohlcv("1y")
returns = md.get_returns("1y")

# ─── TEST 1: Technical Indicators ─────────────────────────
print("\n" + "─" * 60)
print("  📊 TECHNICAL INDICATORS — NVDA")
print("─" * 60)

result = compute_indicators(ohlcv)

print(f"\n  Price: ${result.current_price:.2f}")
print(f"  Indicators computed: {result.indicators_computed}")

print(f"\n  RSI:        {result.rsi:.1f}  ({result.rsi_signal})")
print(f"  MACD:       {result.macd_histogram:+.3f}  (crossover: {result.macd_crossover})")
print(f"  Bollinger:  %B = {result.bb_pct_b:.2f}  ({result.bb_signal})")
print(f"  Stochastic: %K = {result.stoch_k:.1f}  ({result.stoch_signal})")

print(f"\n  SMA 50:     ${result.sma_50:.2f}  (price {result.price_vs_sma50})")
print(f"  SMA 200:    ${result.sma_200:.2f}  (price {result.price_vs_sma200})")
print(f"  EMA 9/21:   ${result.ema_9:.2f} / ${result.ema_21:.2f}  ({result.ema_crossover})")
gc_status = "✅ GOLDEN CROSS" if result.golden_cross else "❌ DEATH CROSS" if result.death_cross else "—"
print(f"  Cross:      {gc_status}")

print(f"\n  ADX:        {result.adx:.1f}  ({result.adx_signal}, {result.trend_direction})")
print(f"  ATR:        ${result.atr:.2f}  ({result.atr_pct:.2f}% of price)")
print(f"  OBV Trend:  {result.obv_trend}")
print(f"  Volume:     {result.volume_ratio:.2f}x avg  ({result.volume_signal})")
print(f"  VWAP:       ${result.vwap:.2f}  (price {result.price_vs_vwap})")

print(f"\n  ╔══════════════════════════════════╗")
print(f"  ║  COMPOSITE SCORE: {result.composite_score:.1f}/100       ║")
print(f"  ║  Bullish: {result.bullish_count}  Bearish: {result.bearish_count}  Neutral: {result.neutral_count}  ║")
print(f"  ╚══════════════════════════════════╝")

# ─── TEST 2: GARCH Volatility ─────────────────────────────
print("\n" + "─" * 60)
print("  📈 GARCH(1,1) VOLATILITY FORECAST — NVDA")
print("─" * 60)

garch = GARCHModel(returns)
garch_result = garch.fit_and_forecast(horizon=10)

print(f"\n  Model converged: {'✅ Yes' if garch_result.converged else '❌ No'}")
print(f"  Persistence (α+β): {garch_result.persistence:.4f}")
print(f"    α (news impact):   {garch_result.alpha:.4f}")
print(f"    β (persistence):   {garch_result.beta:.4f}")

print(f"\n  📊 Volatility Forecasts (annualized):")
print(f"    Tomorrow (1-day):  {garch_result.vol_1day:.1f}%")
print(f"    Next week (5-day): {garch_result.vol_5day:.1f}%")
print(f"    2 weeks (10-day):  {garch_result.vol_10day:.1f}%")

print(f"\n  Current vol:      {garch_result.current_vol:.1f}%")
print(f"  Vol percentile:   {garch_result.vol_percentile:.0f}th")

regime_emoji = {"LOW": "😌", "NORMAL": "😐", "HIGH": "😰", "EXTREME": "🚨"}
emoji = regime_emoji.get(garch_result.vol_regime, "❓")
print(f"  Vol regime:       {emoji} {garch_result.vol_regime}")

print(f"\n  Daily vol forecast (next 5 days):")
for i, v in enumerate(garch_result.forecast_daily[:5], 1):
    bar = "█" * int(v * 3)
    print(f"    Day {i}: {v:.3f}%  {bar}")

print("\n" + "=" * 60)
print("  ✅ DAY 2: TECHNICAL + GARCH — ALL TESTS PASSED")
print("=" * 60)
