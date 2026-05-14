"""
AlphaAgent — Rigetti (RGTI) Deep Audit
Runs the full 7-agent pipeline and extracts institutional/insider details.
"""

import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append('d:/ML and DL/Python/AlphaAgent')

from data.market import MarketData
from orchestrator.graph import build_alpha_graph
from agents.registry import AgentRegistry

def run_deep_audit(ticker: str):
    print("="*60)
    print(f"  ALPHAAGENT DEEP AUDIT: {ticker}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # 1. Initialize Data
    md = MarketData(ticker)
    registry = AgentRegistry()
    graph = build_alpha_graph()

    # 2. Run LangGraph Pipeline
    print(f"\n[1/3] Executing 7-Agent Boardroom for {ticker}...")
    initial_state = {
        "ticker": ticker,
        "market_data": md,
        "registry": registry
    }
    result = graph.invoke(initial_state)
    packet = result["final_signal"]["packet"]

    # 3. Print Bayesian Probability
    print(f"\n[FINAL SIGNAL]")
    print(f"  Direction:   {packet.direction}")
    print(f"  Probability: {result['final_signal']['probability_up']*100:.1f}%")
    print(f"  Conviction:  {packet.conviction_pct:.1f}%")
    print(f"  Risk Regime: {packet.risk_metrics.position_size_pct*100 if packet.risk_metrics else 10.0:.1f}% max size")

    # 4. Insider & Institutional "Who is Holding?"
    print(f"\n[2/3] Institutional & Insider Ownership Audit")
    
    holders = md.get_institutional_holders()
    if holders is not None and not holders.empty:
        print(f"  Top Institutional Holders:")
        # Take top 5
        for i, row in holders.head(5).iterrows():
            print(f"    - {row.get('Holder', 'Unknown')}: {row.get('Shares', 0):,} shares")
    else:
        print("  - No institutional data found.")

    insiders = md.get_insider_transactions()
    if insiders is not None and not insiders.empty:
        print(f"\n  Recent Insider Activity:")
        # Take last 5
        for i, row in insiders.head(5).iterrows():
            print(f"    - {row.get('Insider', 'Unknown')} ({row.get('Position', '')}): {row.get('Text', '')}")
    else:
        print("\n  - No recent insider transactions.")

    # 5. Balance Sheet & Fundamentals
    print(f"\n[3/3] Balance Sheet & Fundamental Integrity")
    info = md.get_info()
    print(f"  Company: {info.get('longName', ticker)}")
    print(f"  Market Cap: ${info.get('marketCap', 0):,}")
    print(f"  P/E Ratio: {info.get('trailingPE', 'N/A')}")
    print(f"  Price/Book: {info.get('priceToBook', 'N/A')}")
    
    # 6. Agent Boardroom Votes
    print(f"\n[Agent Reasoning]")
    for res in packet.agent_results:
        print(f"  - {res.agent_name.upper()}: {res.reasoning[:150]}...")

    print("\n" + "="*60)

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "RGTI"
    run_deep_audit(ticker)
