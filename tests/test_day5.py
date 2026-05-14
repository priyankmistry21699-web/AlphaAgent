"""Day 5 Test — Agent Framework & Technical Agent."""
import sys

sys.path.insert(0, r"d:\ML and DL\Python\AlphaAgent")

from data.market import MarketData
from agents.registry import AgentRegistry
from agents.technical import TechnicalAgent

print("=" * 60)
print("  AlphaAgent — Day 5: Agent Framework Test")
print("=" * 60)

ticker = "NVDA"
md = MarketData(ticker)

# ─── TEST 1: Agent Registry ───────────────────────────────
print("\n" + "─" * 60)
print("  📚 AGENT REGISTRY")
print("─" * 60)

registry = AgentRegistry()
registry.register(TechnicalAgent)

active_agents = registry.get_active_agents()
print(f"  Active Agents: {list(active_agents.keys())}")

tech_weight = registry.get_agent_weight("technical")
print(f"  Technical Agent Normal Weight: {tech_weight:.2f}")

tech_crisis_weight = registry.get_agent_weight("technical", regime="crisis")
print(f"  Technical Agent Crisis Weight: {tech_crisis_weight:.2f}")

# ─── TEST 2: Technical Agent Execution ────────────────────
print("\n" + "─" * 60)
print("  🤖 TECHNICAL AGENT RUN")
print("─" * 60)

agent = registry.get_agent("technical")
print(f"  Running {agent.name.capitalize()} Agent on {ticker}...")

result = agent.analyze(ticker, md)

print(f"\n  Result Class: {type(result).__name__}")
print(f"  Probability Up: {result.probability_up * 100:.1f}%")
print(f"  Confidence:     {result.confidence * 100:.1f}%")
print(f"  Compute Time:   {result.computation_time_ms:.1f} ms")

print(f"\n  Reasoning generated:")
print(f"  > {result.reasoning}")

print(f"\n  Factor Breakdown:")
for factor_key, factor_data in result.factor_scores.items():
    print(f"    - {factor_data.name}: Score {factor_data.score}/100")
    print(f"      Interpretation: {factor_data.interpretation}")

print("\n" + "=" * 60)
print("  ✅ DAY 5: AGENTS — ALL TESTS PASSED")
print("=" * 60)
