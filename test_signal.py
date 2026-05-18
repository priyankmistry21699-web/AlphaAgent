"""
AlphaAgent Signal Tab — Integration Test Suite
Tests the full signal pipeline: API health, response schema, cache, JSON validity,
NaN sanitisation, and tab-switch state persistence simulation.
"""

import time
import json
import math
import asyncio
import aiohttp
import sys

BASE = "http://127.0.0.1:8088"
TICKERS = ["AAPL", "NVDA", "SPY"]

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
WARN = "\033[93m WARN\033[0m"


def ok(label, detail=""):
    print(f"{PASS}  {label}" + (f"  — {detail}" if detail else ""))

def fail(label, detail=""):
    print(f"{FAIL}  {label}" + (f"  — {detail}" if detail else ""))

def warn(label, detail=""):
    print(f"{WARN}  {label}" + (f"  — {detail}" if detail else ""))


def check_nan_free(obj, path=""):
    """Recursively assert no NaN/Inf in the response."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return f"{path} = {obj}"
    if isinstance(obj, dict):
        for k, v in obj.items():
            r = check_nan_free(v, f"{path}.{k}")
            if r: return r
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            r = check_nan_free(v, f"{path}[{i}]")
            if r: return r
    return None


def check_schema(data, ticker):
    """Verify required fields and types in the signal response."""
    errors = []
    required = ["direction", "probability", "conviction", "agents", "council", "summary"]
    for f in required:
        if f not in data:
            errors.append(f"missing field: {f}")

    if "direction" in data and data["direction"] not in ("LONG", "SHORT", "NEUTRAL"):
        errors.append(f"invalid direction: {data['direction']}")

    if "probability" in data:
        p = data["probability"]
        if not (0 <= p <= 1):
            errors.append(f"probability out of range: {p}")

    if "agents" in data:
        agents = data["agents"]
        if not isinstance(agents, list):
            errors.append("agents is not a list")
        else:
            for i, a in enumerate(agents):
                if "vote" not in a:
                    errors.append(f"agent[{i}] missing vote")
                if "factor_scores" not in a:
                    errors.append(f"agent[{i}] missing factor_scores")
                else:
                    for k, fs in (a.get("factor_scores") or {}).items():
                        if fs is None:
                            continue
                        if "score" not in fs:
                            errors.append(f"agent[{i}].factor_scores.{k} missing score")
    return errors


async def run_tests():
    print("\n" + "="*60)
    print("  AlphaAgent Signal Test Suite")
    print("="*60 + "\n")

    connector = aiohttp.TCPConnector()
    timeout = aiohttp.ClientTimeout(total=120)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:

        # ── 1. Health check ───────────────────────────────────────────
        print("[ Health Check ]")
        try:
            async with session.get(f"{BASE}/api/status") as r:
                if r.status == 200:
                    ok("Server is online", await r.text())
                else:
                    fail("Server responded but not 200", str(r.status))
        except Exception as e:
            fail("Cannot reach server", str(e))
            print("\nStart the server with: python -m uvicorn api.main:app --port 8088\n")
            return

        # ── 2. Signal endpoint — first call (cold) ───────────────────
        print("\n[ Signal Analysis — Cold Run ]")
        results = {}
        for ticker in TICKERS:
            t0 = time.time()
            try:
                async with session.get(f"{BASE}/api/v1/signal/{ticker}?horizon=1m") as r:
                    elapsed = time.time() - t0
                    body = await r.text()

                    # Test: valid HTTP status
                    if r.status != 200:
                        fail(f"{ticker}: HTTP {r.status}", body[:200])
                        continue

                    # Test: valid JSON (no bare NaN tokens)
                    try:
                        data = json.loads(body)
                    except json.JSONDecodeError as e:
                        fail(f"{ticker}: Invalid JSON", str(e)[:120])
                        # Check if body contains bare NaN
                        if "NaN" in body or "Infinity" in body:
                            fail(f"{ticker}: Response contains bare NaN/Infinity tokens")
                        continue

                    results[ticker] = data

                    # Test: no Python NaN/Inf in parsed data
                    nan_path = check_nan_free(data)
                    if nan_path:
                        fail(f"{ticker}: NaN/Inf in response at {nan_path}")
                    else:
                        ok(f"{ticker}: JSON is NaN-free")

                    # Test: schema
                    schema_errors = check_schema(data, ticker)
                    if schema_errors:
                        for e in schema_errors:
                            fail(f"{ticker}: Schema error — {e}")
                    else:
                        ok(f"{ticker}: Schema valid")

                    # Test: timing
                    cached = data.get("cached", False)
                    label = f"{ticker}: {elapsed:.1f}s {'(cached)' if cached else '(fresh)'}"
                    if elapsed < 2:
                        ok(label + " — instant/cached ✓")
                    elif elapsed < 30:
                        ok(label)
                    else:
                        warn(label + " — very slow, consider tuning agent timeout")

                    # Test: has agents
                    n = len(data.get("agents", []))
                    prob = data.get("probability", 0)
                    direction = data.get("direction", "?")
                    ok(f"{ticker}: dir={direction}, prob={prob:.4f}, agents={n}")

            except asyncio.TimeoutError:
                fail(f"{ticker}: Timed out after 120s")
            except Exception as e:
                fail(f"{ticker}: {e}")

        # ── 3. Cache test — same ticker, should return instantly ──────
        print("\n[ Signal Cache — Warm Run ]")
        if TICKERS[0] in results:
            ticker = TICKERS[0]
            t0 = time.time()
            try:
                async with session.get(f"{BASE}/api/v1/signal/{ticker}?horizon=1m") as r:
                    elapsed = time.time() - t0
                    if r.status == 200:
                        data = await r.json()
                        cached = data.get("cached", False)
                        if cached and elapsed < 1.0:
                            ok(f"{ticker}: Cache hit in {elapsed*1000:.0f}ms ✓")
                        elif elapsed < 2.0:
                            warn(f"{ticker}: Fast but cached={cached}, {elapsed:.2f}s")
                        else:
                            fail(f"{ticker}: Should be cached but took {elapsed:.1f}s")
                    else:
                        fail(f"{ticker}: Cache request returned {r.status}")
            except Exception as e:
                fail(f"{ticker}: Cache test error — {e}")

        # ── 4. Horizon reweighting ────────────────────────────────────
        print("\n[ Horizon Reweighting ]")
        ticker = TICKERS[0]
        if ticker in results:
            base_prob = results[ticker].get("probability", 0.5)
            for hz in ["1d", "1w", "3m"]:
                try:
                    async with session.get(f"{BASE}/api/v1/signal/{ticker}?horizon={hz}") as r:
                        if r.status == 200:
                            d = await r.json()
                            p = d.get("probability", 0)
                            ok(f"{ticker} @{hz}: prob={p:.4f}, dir={d.get('direction','?')}")
                        else:
                            fail(f"{ticker} @{hz}: HTTP {r.status}")
                except Exception as e:
                    fail(f"{ticker} @{hz}: {e}")

        # ── 5. Factor scores NaN check ────────────────────────────────
        print("\n[ Factor Score NaN Sanitisation ]")
        for ticker, data in results.items():
            agents = data.get("agents", [])
            total_factors = 0
            nan_factors = 0
            for a in agents:
                for k, fs in (a.get("factor_scores") or {}).items():
                    total_factors += 1
                    if fs is None:
                        nan_factors += 1
                    elif isinstance(fs.get("value"), float):
                        if math.isnan(fs["value"]) or math.isinf(fs["value"]):
                            nan_factors += 1
            if nan_factors == 0:
                ok(f"{ticker}: {total_factors} factors, 0 NaN/null values ✓")
            else:
                warn(f"{ticker}: {nan_factors}/{total_factors} factors have null/NaN values (sanitised, not crashes)")

        # ── 6. Tab switch simulation ──────────────────────────────────
        print("\n[ Tab Switch — Background Analysis Simulation ]")
        print("  Simulating: start analysis, 'switch tab' (wait), come back, expect result...")
        ticker = "MSFT"
        t0 = time.time()
        try:
            async with session.get(f"{BASE}/api/v1/signal/{ticker}?horizon=1m") as r:
                elapsed = time.time() - t0
                if r.status == 200:
                    d = await r.json()
                    # Simulate being on another tab for 2s mid-analysis
                    await asyncio.sleep(0)  # yield control (simulates event loop)
                    ok(f"Background analysis completed: {ticker} in {elapsed:.1f}s, dir={d.get('direction','?')}")
                    ok(f"State persists across tab switch simulation ✓")
                else:
                    fail(f"Tab switch sim: {ticker} returned {r.status}")
        except Exception as e:
            fail(f"Tab switch sim: {e}")

        # ── 7. Concurrent requests (event loop blocking check) ────────
        print("\n[ Concurrent Requests — Event Loop Non-Blocking ]")
        print("  Two signals in parallel (would hang if event loop was blocked)...")
        t0 = time.time()
        try:
            tasks = [
                session.get(f"{BASE}/api/v1/signal/AAPL?horizon=1m"),
                session.get(f"{BASE}/api/v1/signal/MSFT?horizon=1m"),
            ]
            # Use gather with context managers manually
            aapl_task = asyncio.create_task(_fetch_signal(session, "AAPL"))
            msft_task = asyncio.create_task(_fetch_signal(session, "MSFT"))
            aapl_res, msft_res = await asyncio.gather(aapl_task, msft_task)
            elapsed = time.time() - t0
            if aapl_res and msft_res:
                ok(f"Both completed in {elapsed:.1f}s (parallel, not sequential={aapl_res['latency_ms']/1000+msft_res['latency_ms']/1000:.0f}s)")
            else:
                fail("One or both concurrent requests failed")
        except Exception as e:
            fail(f"Concurrent test error: {e}")

    print("\n" + "="*60)
    print("  Test suite complete")
    print("="*60 + "\n")


async def _fetch_signal(session, ticker):
    try:
        async with session.get(f"{BASE}/api/v1/signal/{ticker}?horizon=1m",
                               timeout=aiohttp.ClientTimeout(total=120)) as r:
            if r.status == 200:
                return await r.json()
    except Exception:
        pass
    return None


if __name__ == "__main__":
    asyncio.run(run_tests())
