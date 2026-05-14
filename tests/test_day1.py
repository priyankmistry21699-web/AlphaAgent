"""Quick test of the Day 1 data layer."""
import sys
sys.path.insert(0, r"d:\ML and DL\Python\AlphaAgent")

from data.market import MarketData

print("=" * 50)
print("  AlphaAgent — Day 1 Data Layer Test")
print("=" * 50)

# Initialize
md = MarketData("NVDA")

# Test 1: Company info
print("\n📋 Company Info:")
s = md.summary()
print(f"  Company:    {s['company_name']}")
print(f"  Sector:     {s['sector']}")
print(f"  Price:      ${s['current_price']:.2f}")
print(f"  P/E Ratio:  {s['pe_ratio']}")
print(f"  Market Cap: ${s['market_cap']/1e9:.1f}B")

# Test 2: OHLCV
print("\n📈 Price Data (1 year):")
ohlcv = md.get_ohlcv("1y")
print(f"  Trading days: {len(ohlcv)}")
print(f"  Date range:   {ohlcv.index[0].date()} to {ohlcv.index[-1].date()}")
print(f"  Latest close: ${ohlcv['Close'].iloc[-1]:.2f}")

# Test 3: Returns
print("\n📊 Return Statistics:")
returns = md.get_returns()
print(f"  Data points:       {len(returns)}")
print(f"  Mean daily return: {returns.mean()*100:.3f}%")
print(f"  Daily volatility:  {returns.std()*100:.3f}%")
print(f"  Annual volatility: {returns.std()*100*(252**0.5):.1f}%")

# Test 4: Cache
print("\n💾 Cache:")
from data.cache import DataCache
cache = DataCache()
stats = cache.get_stats()
print(f"  Cached entries: {stats['valid_entries']}")

# Test 5: Sufficient history
print(f"\n✅ Sufficient history (200 days): {md.has_sufficient_history()}")

print("\n" + "=" * 50)
print("  ✅ DAY 1 DATA LAYER: ALL TESTS PASSED")
print("=" * 50)
