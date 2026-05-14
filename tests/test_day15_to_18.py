"""Day 15-18 Test — Full LangGraph with Advanced Agents & RAG."""
import sys
import logging

sys.path.insert(0, r"d:\ML and DL\Python\AlphaAgent")

# Disable chromadb logging noise
logging.getLogger("chromadb").setLevel(logging.ERROR)

from data.market import MarketData
from agents.registry import AgentRegistry
from orchestrator.graph import build_alpha_graph

print("=" * 60)
print("  AlphaAgent — Day 15-18: Advanced Agents & RAG Orchestration")
print("=" * 60)

ticker = "AAPL"
print(f"\n  [1] Building State Graph and Registries...")
registry = AgentRegistry()
graph = build_alpha_graph()
md = MarketData(ticker)

initial_state = {
    "ticker": ticker,
    "market_data": md,
    "registry": registry
}

print(f"  [2] Executing LangGraph for {ticker} (Fetching News, RAG, Bayesian Fusion)...")
# This will take a moment as it fetches news, embeds it in ChromaDB, and runs all 7 agents
result_state = graph.invoke(initial_state)

signal_info = result_state["final_signal"]
packet = signal_info["packet"]
entropy = signal_info["entropy"]

print("\n" + "─" * 60)
print("  🧠 FINAL BAYESIAN ORCHESTRATION RESULT")
print("─" * 60)

print(f"  Direction:   {packet.direction.name}")
print(f"  Probability: {signal_info['probability_up'] * 100:.1f}%")
print(f"  Conviction:  {packet.conviction_pct:.1f}%")
print(f"  Entropy:     {entropy:.3f} (0=Certain, 1=Confused)")

print("\n  [Agent Boardroom Votes]")
for res in packet.agent_results:
    print(f"    - {res.agent_name.ljust(12)} | Vote: {res.vote.name.ljust(5)} | Prob: {res.probability_up*100:5.1f}% | Conf: {res.confidence*100:4.1f}%")

print("\n  [Agent Reasoning & Circuit Breakers]")
for res in packet.agent_results:
    if res.agent_name in ["Risk", "Volatility", "Sentiment"]:
        print(f"\n  >> {res.agent_name} Agent:")
        print(f"     {res.reasoning}")

if packet.warnings:
    print("\n  [Warnings]")
    for w in packet.warnings:
        print(f"    ⚠️ {w}")

print("\n" + "=" * 60)
print("  ✅ DAY 15-18: FULL AGENT RAG PIPELINE PASSED")
print("=" * 60)
