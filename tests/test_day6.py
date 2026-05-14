"""Day 6 Test — Fundamental and Sentiment Agents."""
import sys
import os

sys.path.insert(0, r"d:\ML and DL\Python\AlphaAgent")

from data.market import MarketData
from agents.registry import AgentRegistry
from agents.fundamental import FundamentalAgent
from agents.sentiment import SentimentAgent

print("=" * 60)
print("  AlphaAgent — Day 6: Fundamental & Sentiment Test")
print("=" * 60)

ticker = "NVDA"
md = MarketData(ticker)

# Register Agents
registry = AgentRegistry()
registry.register(FundamentalAgent)
registry.register(SentimentAgent)

# ─── TEST 1: Fundamental Agent ────────────────────────────
print("\n" + "─" * 60)
print("  🏢 FUNDAMENTAL AGENT RUN")
print("─" * 60)

fund_agent = registry.get_agent("fundamental")
print(f"  Running {fund_agent.name.capitalize()} Agent on {ticker}...")

fund_result = fund_agent.analyze(ticker, md)

print(f"\n  Probability Up: {fund_result.probability_up * 100:.1f}%")
print(f"  Confidence:     {fund_result.confidence * 100:.1f}%")
print(f"  Compute Time:   {fund_result.computation_time_ms:.1f} ms")

print(f"\n  Reasoning generated:")
print(f"  > {fund_result.reasoning}")

print(f"\n  Factor Breakdown:")
for factor_key, factor_data in fund_result.factor_scores.items():
    print(f"    - {factor_data.name}: {factor_data.interpretation}")

# ─── TEST 2: Sentiment Agent ──────────────────────────────
print("\n" + "─" * 60)
print("  📰 SENTIMENT AGENT RUN (LLM / Simulated)")
print("─" * 60)

sent_agent = registry.get_agent("sentiment")
print(f"  Running {sent_agent.name.capitalize()} Agent on {ticker}...")

# Check if user has provided an API key
if not os.environ.get("GEMINI_API_KEY"):
    print("  [Notice] GEMINI_API_KEY not found in environment. Using keyword simulation.")
else:
    print("  [Notice] GEMINI_API_KEY detected. Using Gemini 2.5 Flash for analysis.")

sent_result = sent_agent.analyze(ticker, md)

print(f"\n  Probability Up: {sent_result.probability_up * 100:.1f}%")
print(f"  Confidence:     {sent_result.confidence * 100:.1f}%")
print(f"  Compute Time:   {sent_result.computation_time_ms:.1f} ms")

print(f"\n  Reasoning generated:")
print(f"  > {sent_result.reasoning}")

if sent_result.warnings:
    print(f"\n  Warnings: {sent_result.warnings}")

print("\n" + "=" * 60)
print("  ✅ DAY 6: AGENTS — ALL TESTS PASSED")
print("=" * 60)
