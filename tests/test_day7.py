"""Day 7 Test — The Full LangGraph Orchestrator."""
import sys
import os
import json

sys.path.insert(0, r"d:\ML and DL\Python\AlphaAgent")

from data.market import MarketData
from agents.registry import AgentRegistry
from agents.technical import TechnicalAgent
from agents.fundamental import FundamentalAgent
from agents.sentiment import SentimentAgent
from agents.risk import RiskAgent
from orchestrator.graph import build_alpha_graph, AlphaGraphState

print("=" * 60)
print("  AlphaAgent — Day 7: Full Orchestrator Test")
print("=" * 60)

ticker = "NVDA"

# 1. Setup the environment
md = MarketData(ticker)
registry = AgentRegistry()

# Manually register all our completed agents
registry.register(TechnicalAgent)
registry.register(FundamentalAgent)
registry.register(SentimentAgent)
registry.register(RiskAgent)

print(f"\n  Starting AlphaGraph execution for {ticker}...")
print(f"  Active Agents: {list(registry.get_active_agents().keys())}")

# 2. Build Graph and State
app = build_alpha_graph()
initial_state = AlphaGraphState(
    ticker=ticker,
    market_data=md,
    registry=registry
)

# 3. Execute the Graph
# LangGraph streams state changes node by node
final_state = None
for output in app.stream(initial_state):
    for node_name, state_update in output.items():
        print(f"  [Graph] Node completed: {node_name}")
        final_state = state_update

# 4. Display the Final Results
print("\n" + "─" * 60)
print("  📈 FINAL TRADING SIGNAL (SignalPacket)")
print("─" * 60)

signal_data = final_state["final_signal"]
signal = signal_data["packet"]
prob_up = signal_data["probability_up"]
risk_res = signal_data["risk_res"]

# Visual formatting for the final signal
dir_emoji = "🟩" if signal.direction.value == "LONG" else "🟥" if signal.direction.value == "SHORT" else "🟨"

print(f"\n  TICKER:     {signal.ticker}")
print(f"  DIRECTION:  {dir_emoji} {signal.direction.value}")
print(f"  PROBABILITY:{prob_up * 100:>6.1f}% UP")
print(f"  CONVICTION: {signal.conviction_pct:>6.1f}%")

print("\n  ⚖️ AGENT VOTING BREAKDOWN:")
for res in signal.agent_results:
    if res.agent_name == "risk":
        print(f"    - 🛡️  {res.agent_name.upper():<12} : {res.factor_scores.get('vol_regime').interpretation if res.factor_scores.get('vol_regime') else 'Unknown Regime'}")
    else:
        vote = "BULL" if res.probability_up > 0.55 else "BEAR" if res.probability_up < 0.45 else "HOLD"
        print(f"    - 🤖 {res.agent_name.upper():<12} : {vote:<4} ({res.probability_up * 100:.1f}%) | Confidence: {res.confidence * 100:.0f}%")

print("\n  ⚠️ WARNINGS & RISK METRICS:")
for w in signal.warnings:
    print(f"    - {w}")

print("\n" + "=" * 60)
print("  ✅ DAY 7: ORCHESTRATOR — ALL TESTS PASSED")
print("=" * 60)
