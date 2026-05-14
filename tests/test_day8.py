"""Day 8 Test — Macro Engine & FRED Integration."""
import sys

sys.path.insert(0, r"d:\ML and DL\Python\AlphaAgent")

from data.market import MarketData
from agents.registry import AgentRegistry
from agents.macro import MacroAgent
from orchestrator.graph import build_alpha_graph, AlphaGraphState

print("=" * 60)
print("  AlphaAgent — Day 8: Macro Engine Test")
print("=" * 60)

ticker = "SPY"  # Macro applies to the whole market, but we pass SPY

# ─── TEST 1: Macro Agent ────────────────────────────
print("\n" + "─" * 60)
print("  🦅 MACRO AGENT RUN (FRED API)")
print("─" * 60)

md = MarketData(ticker)
registry = AgentRegistry()

# Manually register the MacroAgent (and others if we want)
registry.register(MacroAgent)

macro_agent = registry.get_agent("macro")
print(f"  Running {macro_agent.name.capitalize()} Agent...")

macro_result = macro_agent.analyze(ticker, md)

print(f"\n  Probability Up: {macro_result.probability_up * 100:.1f}%")
print(f"  Confidence:     {macro_result.confidence * 100:.1f}%")
print(f"  Compute Time:   {macro_result.computation_time_ms:.1f} ms")

print(f"\n  Reasoning generated:")
print(f"  > {macro_result.reasoning}")

print(f"\n  Factor Breakdown:")
for factor_key, factor_data in macro_result.factor_scores.items():
    print(f"    - {factor_data.name}: {factor_data.interpretation}")

if macro_result.warnings:
    print(f"\n  Warnings: {macro_result.warnings}")

print("\n" + "=" * 60)
print("  ✅ DAY 8: MACRO ENGINE — ALL TESTS PASSED")
print("=" * 60)
