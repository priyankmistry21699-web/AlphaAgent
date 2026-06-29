"""
AlphaAgent — Batch Signal Runner
Runs the full 8-agent pipeline (Macro + Sentiment + Technical + Fundamental +
Volatility + Insider + Geopolitical + Risk) for an arbitrary list of tickers
concurrently via ThreadPoolExecutor.

Usage:
    from batch_signal_runner import run_batch
    signals = run_batch(["AAPL", "MSFT", "NVDA"], horizon="1d", max_workers=4)
    # → {ticker: {direction, probability, conviction, agent_breakdown, ...}}

Or run standalone to test:
    python batch_signal_runner.py
"""

import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Logging — suppress agent noise during batch run ──────────────────────────
logging.basicConfig(level=logging.WARNING)
for _lg in [
    "agents", "data", "orchestrator", "quant_engine",
    "httpx", "urllib3", "langchain", "openai", "google",
    "chromadb", "sentence_transformers", "transformers",
]:
    logging.getLogger(_lg).setLevel(logging.ERROR)

# ── Lazy one-time init ────────────────────────────────────────────────────────
_GRAPH     = None
_REGISTRY  = None
_MarketData = None
_init_lock = __import__("threading").Lock()


def _ensure_init():
    global _GRAPH, _REGISTRY, _MarketData
    if _GRAPH is not None:
        return
    with _init_lock:
        if _GRAPH is not None:
            return
        print("  [BatchRunner] Initializing AlphaAgent pipeline (one-time)...")
        from data.market import MarketData as _MD
        from agents.registry import AgentRegistry as _AR
        from orchestrator.graph import build_alpha_graph as _BAG
        _REGISTRY   = _AR()
        _GRAPH      = _BAG()
        _MarketData = _MD
        print("  [BatchRunner] Pipeline ready.\n")


def _normalize_direction(raw: str) -> str:
    """Normalize Direction enum / string to LONG / SHORT / NEUTRAL."""
    s = str(raw).upper()
    for strip in ("DIRECTION.", "<DIRECTION.", "DIRECTION", ">"):
        s = s.replace(strip, "")
    s = s.strip()
    if s in ("LONG",):   return "LONG"
    if s in ("SHORT",):  return "SHORT"
    return "NEUTRAL"


def run_ticker(ticker: str, horizon: str = "1d") -> Dict[str, Any]:
    """
    Run the full AlphaAgent pipeline for ONE ticker.
    Thread-safe — multiple tickers can be in-flight simultaneously.
    """
    _ensure_init()

    try:
        md = _MarketData(ticker)

        result_state = _GRAPH.invoke({
            "ticker":      ticker,
            "market_data": md,
            "registry":    _REGISTRY,
        })

        final_info = result_state["final_signal"]
        packet     = final_info["packet"]

        direction = _normalize_direction(packet.direction)

        # ── Per-agent breakdown ───────────────────────────────────────────────
        agent_breakdown = []
        for a in packet.agent_results:
            agent_breakdown.append({
                "agent":      a.agent_name,
                "vote":       _normalize_direction(a.vote),
                "prob":       round(float(a.probability_up), 4),
                "confidence": round(float(a.confidence), 4),
                "reason":     (a.reasoning or "")[:300].replace("\n", " "),
            })

        return {
            "ticker":          ticker,
            "direction":       direction,
            "probability":     round(float(final_info["probability_up"]), 4),
            "conviction":      round(float(packet.conviction_pct), 2),
            "multiplier":      float(final_info.get("multiplier", 1.0)),
            "entropy":         round(float(final_info.get("entropy", 1.0)), 4),
            "market_regime":   str(final_info.get("market_regime", "UNKNOWN")),
            "agreement_score": round(float(final_info.get("agreement_score", 0.0)), 4),
            "override_active": bool(packet.override_active),
            "override_reason": str(packet.override_reason or ""),
            "warnings":        list(packet.warnings or []),
            "agent_breakdown": agent_breakdown,
            "agent_probs":     {a["agent"]: a["prob"] for a in agent_breakdown},
            "error":           None,
        }

    except Exception as e:
        return {
            "ticker":          ticker,
            "direction":       "NEUTRAL",
            "probability":     0.5,
            "conviction":      0.0,
            "multiplier":      0.0,
            "entropy":         1.0,
            "market_regime":   "UNKNOWN",
            "agreement_score": 0.0,
            "override_active": False,
            "override_reason": "",
            "warnings":        [],
            "agent_breakdown": [],
            "agent_probs":     {},
            "error":           str(e)[:300],
        }


def run_batch(
    tickers:      List[str],
    horizon:      str  = "1d",
    max_workers:  int  = 4,
    show_progress:bool = True,
    ticker_timeout: int = 150,
) -> Dict[str, Dict]:
    """
    Run the full AlphaAgent pipeline for a list of tickers concurrently.

    Args:
        tickers       : List of ticker symbols
        horizon       : Signal horizon — "1d", "1w", "1m", "3m", "6m", "1y"
        max_workers   : Max concurrent tickers (4 is safe for API rate limits)
        show_progress : Print live progress per ticker
        ticker_timeout: Seconds before a single ticker is abandoned (default 150)

    Returns:
        Dict mapping ticker → result dict
    """
    _ensure_init()

    results: Dict[str, Dict] = {}
    total = len(tickers)
    done  = 0
    t0    = time.time()

    if show_progress:
        print(f"  Running real 8-agent pipeline for {total} tickers "
              f"({max_workers} concurrent, horizon={horizon})")
        print(f"  Estimated time: {total // max_workers * 35}–{total // max_workers * 55}s")
        print(f"  {'─'*72}")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_ticker = {
            pool.submit(run_ticker, t, horizon): t
            for t in tickers
        }

        for future in as_completed(future_to_ticker, timeout=ticker_timeout * total):
            ticker = future_to_ticker[future]
            done  += 1
            elapsed = time.time() - t0

            try:
                result = future.result(timeout=ticker_timeout)
            except Exception as e:
                result = {
                    "ticker": ticker, "direction": "NEUTRAL",
                    "probability": 0.5, "conviction": 0.0,
                    "multiplier": 0.0, "entropy": 1.0,
                    "market_regime": "UNKNOWN", "agreement_score": 0.0,
                    "override_active": False, "override_reason": "",
                    "warnings": [], "agent_breakdown": [],
                    "agent_probs": {},
                    "error": f"TIMEOUT/ERROR: {e}",
                }

            results[ticker] = result

            if show_progress:
                pct  = done / total * 100
                d    = result["direction"]
                prob = result["probability"]
                conv = result["conviction"]
                regime = result["market_regime"]
                err_tag = f"  ⚠ {result['error'][:40]}" if result["error"] else ""
                eta  = (elapsed / done * (total - done)) if done > 0 else 0
                override_tag = " [OVERRIDE]" if result["override_active"] else ""
                print(
                    f"  [{pct:5.1f}%] {done:>2}/{total}  "
                    f"{ticker:<8}  {d:<8}  prob={prob:.3f}  "
                    f"conv={conv:5.1f}%  regime={regime:<12}"
                    f"{override_tag}{err_tag}  ETA {eta:.0f}s"
                )

    if show_progress:
        total_time = time.time() - t0
        n_ok  = sum(1 for r in results.values() if not r.get("error"))
        n_err = total - n_ok
        print(f"  {'─'*72}")
        print(f"  Completed {n_ok}/{total} tickers in {total_time:.1f}s  "
              f"({n_err} errors)\n")

    return results


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    TEST_TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]
    print("\n" + "="*72)
    print("  BatchRunner standalone test — 5 tickers, horizon=1d")
    print("="*72)

    results = run_batch(TEST_TICKERS, horizon="1d", max_workers=3)

    print(f"\n  {'TICKER':<8}  {'DIR':<8}  {'PROB':>6}  {'CONV':>7}  {'AGENTS':>8}")
    print(f"  {'─'*50}")
    for t, r in results.items():
        ab  = r.get("agent_breakdown", [])
        agents_summary = ", ".join(
            f"{a['agent'][:4]}={a['vote'][0]}" for a in ab if a["vote"] != "NEUTRAL"
        ) or "all NEUTRAL"
        err = f"  ERR: {r['error'][:40]}" if r["error"] else ""
        print(
            f"  {t:<8}  {r['direction']:<8}  {r['probability']:>6.3f}  "
            f"{r['conviction']:>6.1f}%  {agents_summary}{err}"
        )
    print()
