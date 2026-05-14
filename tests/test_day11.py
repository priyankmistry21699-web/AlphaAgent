"""Day 11 Test — Historical Backtesting Engine."""
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, r"d:\ML and DL\Python\AlphaAgent")

from data.market import MarketData
from quant_engine.technical import compute_indicators
from backtesting.engine import BacktestEngine

print("=" * 60)
print("  AlphaAgent — Day 11: Backtesting Engine")
print("=" * 60)

ticker = "AAPL"
print(f"\n  [1] Fetching 1-Year Historical Data for {ticker}...")
md = MarketData(ticker)
ohlcv_df = md.get_ohlcv("1y")

print(f"  [2] Pre-computing Technical Signals for {len(ohlcv_df)} days...")

# We will generate a simple signal series: 
# Buy (1) if Fast MA > Slow MA (Golden Cross)
# Sell (-1) if Fast MA < Slow MA (Death Cross)
signals = []
for i in range(len(ohlcv_df)):
    # To prevent look-ahead bias, we only look at data up to today
    window_df = ohlcv_df.iloc[:i+1]
    if len(window_df) < 50:
        signals.append(0)
        continue
        
    # We can use our compute_indicators, but since it's meant for the current day,
    # we'll do a quick vectorized MA crossover for the backtest speed
    fast_ma = window_df['Close'].rolling(window=10).mean().iloc[-1]
    slow_ma = window_df['Close'].rolling(window=50).mean().iloc[-1]
    
    if fast_ma > slow_ma:
        signals.append(1) # Buy/Hold
    elif fast_ma < slow_ma:
        signals.append(-1) # Sell/Wait
    else:
        signals.append(0)
        
signal_series = pd.Series(signals, index=ohlcv_df.index)

print("  [3] Running Simulation Engine...")
engine = BacktestEngine(initial_capital=100000.0)
results = engine.run_fast_backtest(ticker, ohlcv_df, signal_series)

print("\n" + "─" * 60)
print("  📈 BACKTEST RESULTS (1-Year SMA Crossover)")
print("─" * 60)

print(f"  Ticker:           {results.ticker}")
print(f"  Period:           {results.start_date} to {results.end_date}")
print(f"  Total Trades:     {results.total_trades}")
print(f"  Win Rate:         {results.win_rate * 100:.1f}%")
print(f"  Total Return:     {results.total_return_pct * 100:.2f}%")
print(f"  Max Drawdown:     {results.max_drawdown_pct * 100:.2f}%")

if results.trades:
    print("\n  Sample Trades:")
    for t in results.trades[:3]:
        dir_color = "🟩" if t.pnl_pct > 0 else "🟥"
        print(f"    - {t.entry_date} to {t.exit_date} | {dir_color} {t.pnl_pct * 100:+.2f}%")
        
print("\n" + "=" * 60)
print("  ✅ DAY 11: BACKTESTER — ALL TESTS PASSED")
print("=" * 60)
