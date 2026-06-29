"""
AlphaAgent — Intraday $100K Simulation  (2026-05-19)
─────────────────────────────────────────────────────
Strategy:
  1. Run AlphaAgent full analysis on top US stocks
  2. Select tickers where direction=LONG or SHORT with high conviction
  3. Allocate $100K by score weight
  4. Buy at today's actual 9:30 AM open, sell at 4:00 PM close
  5. Compare vs SPY / QQQ intraday returns
"""

import sys
import requests
import yfinance as yf
import pandas as pd
from datetime import date
import pytz
import time

CAPITAL      = 100_000.0
API_BASE     = "http://localhost:8000"
TODAY        = date.today().isoformat()
ET           = pytz.timezone("America/New_York")
HORIZON      = "1d"
MIN_SCORE    = 0.20    # probability * (conviction/100) threshold

# Top US large-cap + sector leaders to analyse
US_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "AVGO", "JPM", "AMD",
    "NFLX", "COST", "LLY",  "V",   "BRK-B",
]


def run_signal(ticker: str) -> dict | None:
    try:
        r = requests.get(
            f"{API_BASE}/api/v1/signal/{ticker}",
            params={"horizon": HORIZON},
            timeout=90,
        )
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"  [error] {ticker}: {e}")
    return None


def get_open_close(ticker: str) -> tuple[float | None, float | None]:
    """Return (open at first 9:30 bar, close at last ≤16:00 bar) for today."""
    try:
        t    = yf.Ticker(ticker)
        hist = t.history(period="1d", interval="1m", prepost=False)
        if hist is None or hist.empty:
            return None, None

        hist.index = hist.index.tz_convert(ET)
        open_bars  = hist[hist.index.time >= pd.Timestamp("09:30").time()]
        close_bars = hist[hist.index.time <= pd.Timestamp("16:00").time()]

        op = float(open_bars.iloc[0]["Open"])    if not open_bars.empty  else None
        cl = float(close_bars.iloc[-1]["Close"]) if not close_bars.empty else None
        return op, cl
    except Exception as e:
        print(f"  [price warn] {ticker}: {e}")
        return None, None


def benchmark_return(ticker: str) -> float | None:
    try:
        t    = yf.Ticker(ticker)
        hist = t.history(period="1d", interval="1m", prepost=False)
        if hist is None or hist.empty:
            return None
        hist.index = hist.index.tz_convert(ET)
        opens  = hist[hist.index.time >= pd.Timestamp("09:30").time()]
        closes = hist[hist.index.time <= pd.Timestamp("16:00").time()]
        if opens.empty or closes.empty:
            return None
        o = float(opens.iloc[0]["Open"])
        c = float(closes.iloc[-1]["Close"])
        return (c - o) / o * 100
    except Exception:
        return None


# ── Main ────────────────────────────────────────────────────────────────────

def run_simulation():
    print("=" * 72)
    print(f"  AlphaAgent  ·  Intraday $100K Simulation  ·  {TODAY}")
    print(f"  Horizon: {HORIZON}  |  Universe: {len(US_TICKERS)} US stocks")
    print("=" * 72)
    print(f"\nStep 1 — Running AlphaAgent multi-agent analysis…\n")

    signals = []
    for ticker in US_TICKERS:
        print(f"  {ticker:<8}", end="", flush=True)
        sig = run_signal(ticker)
        if sig:
            direction  = sig.get("direction", "NEUTRAL")
            prob       = sig.get("probability", 0.5)
            conviction = sig.get("conviction", 0.0)
            # conviction is 0-100 scale; normalise to 0-1 for scoring
            score = prob * (conviction / 100.0)
            agents_bullish = sig.get("agents_bullish", 0)
            agents_bearish = sig.get("agents_bearish", 0)
            print(f"→ {direction:<7} prob={prob:.3f}  conv={conviction:.1f}  "
                  f"score={score:.3f}  [{agents_bullish}▲ / {agents_bearish}▼]")
            if direction in ("LONG", "SHORT") and score >= MIN_SCORE:
                signals.append({
                    "ticker":    ticker,
                    "direction": direction,
                    "prob":      prob,
                    "conviction":conviction,
                    "score":     score,
                    "reasoning": sig.get("reasoning", "")[:70],
                })
        else:
            print("→ no response")
        time.sleep(0.2)

    print(f"\n  {len(signals)} actionable signals (score ≥ {MIN_SCORE})\n")
    if not signals:
        print("[error] No actionable signals. Is the server running?")
        sys.exit(1)

    # ── Allocation by score weight ─────────────────────────────────────────
    total_score = sum(s["score"] for s in signals)
    for s in signals:
        s["pct"]       = s["score"] / total_score * 100
        s["allocated"] = CAPITAL * s["pct"] / 100

    # ── Fetch actual intraday prices ───────────────────────────────────────
    print("Step 2 — Fetching intraday prices (9:30 open → 4:00 close)…\n")
    rows = []
    total_pnl = 0.0

    for s in signals:
        op, cl = get_open_close(s["ticker"])
        if op and cl and op > 0:
            if s["direction"] == "SHORT":
                ret_pct = (op - cl) / op * 100
            else:
                ret_pct = (cl - op) / op * 100
            pnl = s["allocated"] * ret_pct / 100
            total_pnl += pnl
        else:
            ret_pct = None
            pnl     = None

        rows.append({**s, "open": op, "close": cl, "ret_pct": ret_pct, "pnl": pnl})
        status = f"open={op:.2f}  close={cl:.2f}  ret={ret_pct:+.2f}%" if ret_pct is not None else "no data"
        print(f"  {s['ticker']:<8} {status}")

    # ── Benchmarks ─────────────────────────────────────────────────────────
    print("\nStep 3 — Fetching benchmark returns (SPY / QQQ)…\n")
    spy_ret = benchmark_return("SPY")
    qqq_ret = benchmark_return("QQQ")
    iwm_ret = benchmark_return("IWM")

    portfolio_ret = total_pnl / CAPITAL * 100

    # ── Results table ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"  {'TICKER':<8} {'DIR':<6} {'SCORE':>6} {'ALLOC%':>7} "
          f"{'ALLOC $':>9} {'OPEN':>9} {'CLOSE':>9} {'RET%':>7} {'P&L $':>10}")
    print("  " + "-" * 76)

    for r in sorted(rows, key=lambda x: abs(x["pnl"] or 0), reverse=True):
        ret_s = f"{r['ret_pct']:>+.2f}%" if r["ret_pct"] is not None else "      —"
        pnl_s = f"${r['pnl']:>+,.2f}"   if r["pnl"]     is not None else "         —"
        op_s  = f"{r['open']:>9.2f}"    if r["open"]    is not None else "        —"
        cl_s  = f"{r['close']:>9.2f}"   if r["close"]   is not None else "        —"
        print(f"  {r['ticker']:<8} {r['direction']:<6} {r['score']:>6.3f} "
              f"{r['pct']:>6.1f}% ${r['allocated']:>8,.0f} "
              f"{op_s} {cl_s} {ret_s:>8} {pnl_s:>10}")

    print("  " + "=" * 76)
    print(f"  {'PORTFOLIO':<8} {'':6} {'':6} {'100.0%':>7} "
          f"${CAPITAL:>8,.0f} {'':9} {'':9} "
          f"{portfolio_ret:>+7.3f}% ${total_pnl:>+10,.2f}")

    # ── Benchmark panel ─────────────────────────────────────────────────────
    print("\n── Benchmark Comparison ─────────────────────────────────────────────────")
    bmarks = [
        ("AlphaAgent (this sim)",  portfolio_ret, total_pnl),
        ("SPY  — S&P 500 ETF",     spy_ret,       (spy_ret * CAPITAL / 100) if spy_ret else None),
        ("QQQ  — Nasdaq-100 ETF",  qqq_ret,       (qqq_ret * CAPITAL / 100) if qqq_ret else None),
        ("IWM  — Russell 2000",    iwm_ret,       (iwm_ret * CAPITAL / 100) if iwm_ret else None),
    ]
    for name, ret, pnl in bmarks:
        if ret is None:
            print(f"  {name:<28}  no data")
            continue
        bar   = "█" * min(int(abs(ret) * 6), 30)
        arrow = "▲" if ret >= 0 else "▼"
        pnl_s = f"  ${pnl:>+,.2f}" if pnl is not None else ""
        sign  = "+" if ret >= 0 else "-"
        print(f"  {name:<28}  {arrow} {ret:>+6.3f}%{pnl_s}   {sign}{bar}")

    if spy_ret is not None:
        alpha = portfolio_ret - spy_ret
        beat  = "BEAT" if alpha > 0 else "LAGGED"
        print(f"\n  Alpha vs SPY : {alpha:>+.3f}%  (${alpha * CAPITAL / 100:>+,.2f})"
              f"  → AlphaAgent {beat} the market today")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"  Simulation date     : {TODAY}  (9:30 AM → 4:00 PM ET)")
    print(f"  Starting capital    : ${CAPITAL:,.2f}")
    print(f"  Tickers analysed    : {len(US_TICKERS)}")
    print(f"  Actionable signals  : {len(signals)}")
    with_data = sum(1 for r in rows if r["pnl"] is not None)
    print(f"  With intraday data  : {with_data}")
    print(f"  Ending capital      : ${CAPITAL + total_pnl:,.2f}")
    print(f"  Net P&L             : ${total_pnl:>+,.2f}")
    print(f"  Portfolio return    : {portfolio_ret:>+.3f}%")

    wins  = [r for r in rows if r["pnl"] is not None and r["pnl"] > 0]
    losses= [r for r in rows if r["pnl"] is not None and r["pnl"] <= 0]
    print(f"  Winners / Losers    : {len(wins)} / {len(losses)}")

    print("\n  Signal reasoning summary:")
    for r in rows:
        icon = "✓" if (r["ret_pct"] or 0) > 0 else "✗" if r["ret_pct"] is not None else "?"
        print(f"    [{icon}] {r['ticker']:<7}  {r['direction']:<5}  {r['reasoning']}")

    print("=" * 72)


if __name__ == "__main__":
    run_simulation()
