"""
AlphaAgent — High Volatility Forecast: XAUUSD (Gold)
Predicting probability for tomorrow at 10:00 AM using hourly data.
"""

import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.append('d:/ML and DL/Python/AlphaAgent')

from data.market import MarketData
from orchestrator.graph import build_alpha_graph
from agents.registry import AgentRegistry
from quant_engine.garch import GARCHModel
from quant_engine.monte_carlo import MonteCarloEngine

def forecast_gold():
    ticker = "GC=F" # Gold Futures (Most volatile and active)
    print("="*60)
    print(f"  ALPHAAGENT VOLATILITY FORECAST: {ticker}")
    print(f"  Target Horizon: Tomorrow 10:00 AM")
    print("="*60)

    # 1. Initialize Data (Hourly for precision)
    md = MarketData(ticker)
    registry = AgentRegistry()
    graph = build_alpha_graph()
    
    print(f"\n[1/3] Fetching Hourly Price Action for {ticker}...")
    # Fetch 730 days of hourly data (max allowed by yfinance for 1h is 730 days)
    hourly_data = md.get_ohlcv(period="730d", interval="1h")
    
    if hourly_data.empty:
        print("Error: Could not fetch hourly Gold data. Falling back to daily.")
        hourly_data = md.get_ohlcv(period="2y", interval="1d")

    # 2. GARCH Volatility Analysis
    returns = hourly_data['Close'].pct_change().dropna()
    garch = GARCHModel(returns)
    garch_res = garch.fit_and_forecast(horizon=12) # 12 hours to 10 AM
    
    print(f"\n[2/3] GARCH Volatility Outlook:")
    print(f"  - Volatility Regime: {garch_res.vol_regime}")
    print(f"  - Hourly Vol Forecast: {garch_res.vol_1day_daily:.4f}")
    
    # 3. Monte Carlo Simulation for 10 AM (12 hours away)
    mc = MonteCarloEngine(current_price=hourly_data['Close'].iloc[-1])
    # Assuming drift is slightly positive for gold in expansion
    mc_res = mc.simulate_garch(
        days=12, # Using hours as 'days' for the math
        drift_daily=0.0001, # Small hourly drift
        garch_forecasts=garch_res.forecast_daily,
        num_paths=10000
    )
    
    # 4. Run the Full Agent Boardroom
    print(f"\n[3/3] Running 7-Agent Signal Fusion...")
    initial_state = {
        "ticker": ticker,
        "market_data": md,
        "registry": registry
    }
    result = graph.invoke(initial_state)
    packet = result["final_signal"]["packet"]

    # 5. Output Results
    print(f"\n────────────────────────────────────────────────────────────")
    print(f"  PROBABILITY FOR TOMORROW 10:00 AM")
    print(f"────────────────────────────────────────────────────────────")
    print(f"  Directional Vote:  {packet.direction}")
    print(f"  Probability UP:    {result['final_signal']['probability_up']*100:.1f}%")
    print(f"  System Conviction: {packet.conviction_pct:.1f}%")
    print(f"  Expected Range:    ${mc_res.ci_95.low:,.2f} - ${mc_res.ci_95.high:,.2f}")
    
    print(f"\n[Agent Logic for Gold]")
    for res in packet.agent_results:
        if res.agent_name in ["technical", "volatility", "macro"]:
            print(f"  - {res.agent_name.upper()}: {res.reasoning[:120]}...")

    print("\n" + "="*60)

if __name__ == "__main__":
    forecast_gold()
