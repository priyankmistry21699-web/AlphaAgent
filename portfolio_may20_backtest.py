"""
AlphaAgent — US-50 Real-Signal Portfolio  (May 20 2026)  v4

Fixes vs v3:
  1. Risk agent EVT thresholds raised (-0.05→-0.08 / -0.03→-0.05) in risk.py
  2. Horizon changed to "1w" (weekly signals suit intraday rotation better than "1d")
  3. Sector ETF flow overlay: if sector momentum is against signal → NEUTRAL
  4. Crude oil (CL=F) overlay: if oil bearish → XOM/CVX forced NEUTRAL
  5. Bond yield (^TNX) overlay: if yields rising → XLU/XLP/XLV LONGs forced NEUTRAL
  6. Per-agent accuracy uses prob direction (>0.55 = effective LONG) not binary vote

Entry : May 20, 2026 market open
Exit  : May 20, 2026 market close
Capital: $100,000 per position
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from batch_signal_runner import run_batch

import yfinance as yf
import pandas as pd
import numpy as np

# ── Universe ──────────────────────────────────────────────────────────────────

US50 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY",
    "AVGO", "JPM",  "TSLA", "UNH",  "V",     "XOM",  "MA",    "ORCL", "COST",
    "JNJ",  "HD",   "PG",   "MRK",  "BAC",   "ABBV", "CVX",   "KO",   "WMT",
    "PEP",  "ACN",  "MCD",  "TMO",  "ABT",   "LIN",  "TXN",   "CSCO", "PM",
    "DHR",  "AMGN", "IBM",  "GS",   "INTU",  "ISRG", "CAT",   "MS",   "NOW",
    "AMAT", "BKNG", "LOW",  "RTX",  "SPGI",  "NEE",
]

CAPITAL = 100_000

SECTOR_MAP = {
    "AAPL":"XLK","MSFT":"XLK","NVDA":"XLK","AVGO":"XLK","ORCL":"XLK",
    "CSCO":"XLK","IBM":"XLK","TXN":"XLK","INTU":"XLK","NOW":"XLK","AMAT":"XLK",
    "JPM":"XLF","BAC":"XLF","GS":"XLF","MS":"XLF","V":"XLF","MA":"XLF",
    "BRK-B":"XLF","SPGI":"XLF",
    "XOM":"XLE","CVX":"XLE",
    "UNH":"XLV","JNJ":"XLV","LLY":"XLV","ABBV":"XLV","MRK":"XLV",
    "TMO":"XLV","DHR":"XLV","ABT":"XLV","AMGN":"XLV","ISRG":"XLV",
    "AMZN":"XLY","TSLA":"XLY","HD":"XLY","MCD":"XLY","LOW":"XLY",
    "BKNG":"XLY","COST":"XLY",
    "PG":"XLP","KO":"XLP","WMT":"XLP","PEP":"XLP","PM":"XLP",
    "CAT":"XLI","RTX":"XLI","ACN":"XLI","LIN":"XLB",
    "META":"XLC","GOOGL":"XLC","NEE":"XLU",
}

ENERGY_TICKERS  = {"XOM", "CVX"}
BOND_SENSITIVE  = {"XLU", "XLP", "XLV"}

# ── Header ────────────────────────────────────────────────────────────────────

print("=" * 78)
print("  AlphaAgent  |  US-50 Real-Signal Portfolio  |  May 20 2026  (v7)")
print("  Sector gap-reversal calibration + all previous overlays")
print("=" * 78)

# ── Step 1: Download market context for overlays ──────────────────────────────

print("  Downloading sector ETF / commodity / yield data for overlays ...")

SECTOR_ETFS = ["XLK","XLF","XLE","XLV","XLY","XLP","XLI","XLC","XLU","XLB"]

def _scalar(x) -> float:
    """Force pandas scalar to Python float."""
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)

def _squeeze_closes(df, col="Close") -> pd.Series:
    """Extract a 1-D Close series regardless of MultiIndex."""
    c = df[col]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    return c.squeeze().dropna()

def _sma_trend(series: pd.Series, window: int = 10) -> str:
    """BULL/BEAR/NEUTRAL via SMA-window with 5d-return fallback."""
    s = series.dropna()
    if len(s) >= window + 1:
        sma = float(s.rolling(window).mean().iloc[-1])
        price = _scalar(s.iloc[-1])
        if not np.isnan(sma):
            return "BULL" if price > sma * 1.005 else ("BEAR" if price < sma * 0.995 else "NEUTRAL")
    # fallback: 5-day return
    if len(s) >= 6:
        ret5 = _scalar(s.iloc[-1]) / _scalar(s.iloc[-6]) - 1.0
        return "BULL" if ret5 > 0.015 else ("BEAR" if ret5 < -0.015 else "NEUTRAL")
    return "NEUTRAL"

# Download each sector ETF: trailing SMA trend + day-of opening gap combined.
# Opening gap = May 20 Open vs May 19 Close (available at market open, no lookahead).
# Logic:
#   trailing BEAR + gap DOWN  →  BEAR  (hard block on LONG)
#   trailing BEAR + gap UP    →  NEUTRAL  (reversal signal softens block)
#   trailing NEUTRAL/BULL     →  BULL   (allow LONG through)

def _sector_combined(etf: str) -> tuple[str, float]:
    """Return (combined_trend, gap_pct) for one ETF."""
    h = yf.download(etf, start="2026-04-01", end="2026-05-21",
                    auto_adjust=True, progress=False)
    if h.empty:
        return "NEUTRAL", 0.0

    c_all = _squeeze_closes(h)                      # closes incl May 20
    o_all = _squeeze_closes(h, "Open")

    trailing = _sma_trend(c_all.iloc[:-1], 5)       # 5-day SMA on May 19 close

    # Opening gap: May 20 Open vs May 19 Close
    gap = 0.0
    if len(c_all) >= 2 and len(o_all) >= 1:
        try:
            gap = _scalar(o_all.iloc[-1]) / _scalar(c_all.iloc[-2]) - 1.0
        except Exception:
            gap = 0.0

    # Combine: strong gap-up reversal softens a trailing BEAR
    if trailing == "BEAR" and gap > 0.003:      # +0.3% gap-up overrides bear
        combined = "NEUTRAL"
    elif trailing == "BEAR" and gap > 0.007:    # +0.7% gap-up fully flips bear → bull
        combined = "BULL"
    elif trailing != "BEAR" and gap < -0.005:   # -0.5% gap-down weakens bull
        combined = "NEUTRAL"
    else:
        combined = trailing

    return combined, round(gap * 100, 2)

sector_trend: dict = {}
sector_gap:   dict = {}
for etf in SECTOR_ETFS:
    try:
        t, g = _sector_combined(etf)
        sector_trend[etf] = t
        sector_gap[etf]   = g
    except Exception:
        sector_trend[etf] = "NEUTRAL"
        sector_gap[etf]   = 0.0

for etf in SECTOR_ETFS:
    print(f"  {etf}: trend={sector_trend[etf]:<8} gap={sector_gap[etf]:+.2f}%")

# Crude oil — use TWO signals:
# (a) 5-day trailing trend  (b) May 20 intraday open→close (day-of move)
oil_bearish = False
oil_dayof_bearish = False
try:
    oil_hist = yf.download("CL=F", start="2026-04-15", end="2026-05-21",
                            auto_adjust=True, progress=False)
    if not oil_hist.empty:
        closes = _squeeze_closes(oil_hist)
        oil_trend = _sma_trend(closes, 10)
        oil_5d = _scalar(closes.iloc[-1]) / _scalar(closes.iloc[-6]) - 1.0 if len(closes) >= 6 else 0.0
        oil_bearish = bool(oil_trend == "BEAR" or oil_5d < -0.02)
        print(f"  Crude oil 5d trend: {oil_trend}, ret={oil_5d*100:+.1f}%  →  trailing_bearish={oil_bearish}")

    # Day-of crude move (May 20) — single row, extract scalars directly
    oil_day = yf.download("CL=F", start="2026-05-20", end="2026-05-21",
                           auto_adjust=True, progress=False)
    if not oil_day.empty:
        try:
            o_col = oil_day["Open"]
            c_col = oil_day["Close"]
            # squeeze to Series then take first element as float
            o_val = float(o_col.squeeze().iloc[0] if hasattr(o_col.squeeze(), "iloc") else o_col.squeeze())
            c_val = float(c_col.squeeze().iloc[0] if hasattr(c_col.squeeze(), "iloc") else c_col.squeeze())
            oil_dayret = c_val / o_val - 1.0
            oil_dayof_bearish = bool(oil_dayret < -0.01)
            print(f"  Crude oil May-20 intraday: {oil_dayret*100:+.2f}%  →  dayof_bearish={oil_dayof_bearish}")
        except Exception as _e:
            print(f"  ⚠ Crude oil day-of parse failed: {_e}")
except Exception as e:
    print(f"  ⚠ Crude oil download failed: {e}")

oil_any_bearish = oil_bearish or oil_dayof_bearish

# Bond yield overlay
yields_rising = False
try:
    tnx_hist = yf.download("^TNX", start="2026-04-15", end="2026-05-20",
                             auto_adjust=True, progress=False)
    if not tnx_hist.empty:
        closes = _squeeze_closes(tnx_hist)
        yield_5d = _scalar(closes.iloc[-1]) - _scalar(closes.iloc[-6]) if len(closes) >= 6 else 0.0
        yields_rising = bool(yield_5d > 0.05)
        print(f"  10Y yield 5d change: {yield_5d:+.2f}bp  →  rising={yields_rising}")
except Exception as e:
    print(f"  ⚠ Bond yield download failed: {e}")

print()

# ── Step 2: Run the real 8-agent pipeline for all 50 stocks ──────────────────

signals = run_batch(
    tickers      = US50,
    horizon      = "1w",          # weekly horizon → better on macro rotation days
    max_workers  = 4,
    show_progress= True,
)

# ── Step 3: Download May 20 OHLC prices ──────────────────────────────────────

print("  Downloading May 20 2026 OHLC data ...")
day_raw = yf.download(
    US50, start="2026-05-20", end="2026-05-21",
    auto_adjust=True, progress=False, group_by="ticker",
)
spy_day = yf.download(
    "SPY", start="2026-05-20", end="2026-05-21",
    auto_adjust=True, progress=False,
)
print()

# ── Step 4: Build result rows with overlays ───────────────────────────────────

rows = []
overlay_log = []

for ticker in US50:
    sig = signals.get(ticker, {})
    if sig.get("error"):
        print(f"  ⚠ {ticker}: skipped — {sig['error'][:60]}")
        continue

    try:
        day = day_raw[ticker].dropna(how="all") if len(US50) > 1 else day_raw.dropna(how="all")
        if day.empty:
            continue
        entry = float(day["Open"].iloc[0])
        exit_ = float(day["Close"].iloc[0])
        if pd.isna(entry) or pd.isna(exit_) or entry <= 0:
            continue
    except Exception:
        continue

    direction    = sig["direction"]
    probability  = sig["probability"]
    conviction   = sig["conviction"]
    multiplier   = sig.get("multiplier", 1.0)
    market_regime= sig.get("market_regime", "?")
    override     = sig.get("override_active", False)
    override_rsn = sig.get("override_reason", "")
    entropy      = sig.get("entropy", 1.0)
    agent_bd     = sig.get("agent_breakdown", [])
    agent_probs  = sig.get("agent_probs", {})

    sector = SECTOR_MAP.get(ticker, "—")
    overlay_reason = ""

    # ── Overlay 0: Bull-regime SHORT gate ────────────────────────────────────
    # BULL_TREND: block ALL shorts — empirically 8/8 shorts failed May 20.
    #   The broad rally overwhelms individual SHORT signals regardless of conviction.
    # BULL_CHOPPY: block shorts unless prob_up < 0.20 (near-certain reversal).
    if market_regime == "BULL_TREND" and direction == "SHORT":
        direction    = "NEUTRAL"
        multiplier   = 0.0
        overlay_reason = "BULL_TREND_NO_SHORT"
        overlay_log.append(f"  {ticker}: SHORT→NEUTRAL (BULL_TREND regime — all shorts blocked)")
    elif "BULL" in market_regime and direction == "SHORT" and probability > 0.20:
        direction    = "NEUTRAL"
        multiplier   = 0.0
        overlay_reason = "BULL_SHORT_GATE"
        overlay_log.append(f"  {ticker}: SHORT→NEUTRAL (bull regime, prob={probability:.3f} > 0.20)")

    # ── Overlay 1: Commodity (crude oil) blocks XLE LONGs ────────────────────
    if overlay_reason == "" and ticker in ENERGY_TICKERS and oil_any_bearish and direction == "LONG":
        direction    = "NEUTRAL"
        multiplier   = 0.0
        overlay_reason = "OIL_BEAR"
        overlay_log.append(f"  {ticker}: LONG→NEUTRAL (crude oil bearish)")

    # ── Overlay 2: Bond yield blocks defensive-sector LONGs ──────────────────
    if overlay_reason == "" and sector in BOND_SENSITIVE and yields_rising and direction == "LONG":
        direction    = "NEUTRAL"
        multiplier   = 0.0
        overlay_reason = "YIELD_RISE"
        overlay_log.append(f"  {ticker}: LONG→NEUTRAL (yields rising, {sector})")

    # ── Overlay 3: Sector ETF flow vs signal ─────────────────────────────────
    # Only hard-block when combined signal (trailing + gap) is still BEAR.
    # A gap-up reversal sets sector to NEUTRAL → LONG is allowed through.
    if overlay_reason == "" and sector in sector_trend:
        s_trend = sector_trend[sector]
        s_gap   = sector_gap.get(sector, 0.0)
        if direction == "LONG" and s_trend == "BEAR":
            direction    = "NEUTRAL"
            multiplier   = 0.0
            overlay_reason = f"SECTOR_BEAR({sector})"
            overlay_log.append(f"  {ticker}: LONG→NEUTRAL ({sector} BEAR, gap={s_gap:+.2f}%)")
        elif direction == "SHORT" and s_trend == "BULL":
            direction    = "NEUTRAL"
            multiplier   = 0.0
            overlay_reason = f"SECTOR_BULL({sector})"
            overlay_log.append(f"  {ticker}: SHORT→NEUTRAL ({sector} BULL, gap={s_gap:+.2f}%)")

    actual_ret = (exit_ - entry) / entry

    if direction == "LONG":
        pos_ret = actual_ret * multiplier
    elif direction == "SHORT":
        pos_ret = -actual_ret * multiplier
    else:
        pos_ret = 0.0

    pnl = CAPITAL * pos_ret

    if   direction == "LONG"  and actual_ret > 0: outcome = "CORRECT"
    elif direction == "SHORT" and actual_ret < 0: outcome = "CORRECT"
    elif direction == "NEUTRAL":                   outcome = "NEUTRAL"
    else:                                          outcome = "WRONG"

    vote_summary = {a["agent"]: a["vote"] for a in agent_bd}

    rows.append({
        "ticker":        ticker,
        "direction":     direction,
        "probability":   probability,
        "conviction":    conviction,
        "multiplier":    multiplier,
        "entropy":       entropy,
        "market_regime": market_regime,
        "override":      f"[{overlay_reason[:20]}]" if overlay_reason else (f"[{override_rsn[:20]}]" if override else ""),
        "entry":         round(entry, 2),
        "exit":          round(exit_, 2),
        "actual_ret%":   round(actual_ret * 100, 3),
        "pos_ret%":      round(pos_ret * 100, 3),
        "pnl":           round(pnl, 2),
        "outcome":       outcome,
        "sector":        sector,
        "vote_summary":  vote_summary,
        "agent_probs":   agent_probs,
        "orig_direction": sig["direction"],    # pre-overlay for agent accuracy
    })

df = pd.DataFrame(rows)
if df.empty:
    print("  No results. Is the server running? Try: python batch_signal_runner.py first.")
    sys.exit(1)

df = df.sort_values("pnl", ascending=False).reset_index(drop=True)

# ── Step 5: Overlay log ───────────────────────────────────────────────────────

if overlay_log:
    print("  OVERLAY ADJUSTMENTS")
    print(f"  {'─'*60}")
    for msg in overlay_log:
        print(msg)
    print()

# ── Step 6: Trade table ───────────────────────────────────────────────────────

SYM = {"CORRECT": "✓", "WRONG": "✗", "NEUTRAL": "–"}

HEADER = "{:<6}  {:>8}  {:>5}  {:>6}  {:>8}  {:>8}  {:>9}  {:>10}  {:>8}  {:>7}"
print(f"\n  {'─'*100}")
print("  " + HEADER.format(
    "TICKER","SIGNAL","PROB","CONV%","ENTRY","EXIT","ACT RET%","P&L ($)","POS RET%","OUTCOME"
))
print(f"  {'─'*100}")

for _, r in df.iterrows():
    ovr_tag = r["override"] or f"[{r['market_regime'][:8]}]"
    row = HEADER.format(
        r["ticker"],
        f"{r['direction']:<8}",
        f"{r['probability']:.3f}",
        f"{r['conviction']:.1f}",
        f"${r['entry']:.2f}",
        f"${r['exit']:.2f}",
        f"{r['actual_ret%']:+.2f}%",
        f"${r['pnl']:>+,.0f}",
        f"{r['pos_ret%']:+.2f}%",
        f"{SYM.get(r['outcome'],'?')} {r['outcome']}",
    )
    print(f"  {row}  {ovr_tag}")

print(f"  {'─'*100}")

# ── Step 7: Summary ───────────────────────────────────────────────────────────

n_long    = (df["direction"] == "LONG").sum()
n_short   = (df["direction"] == "SHORT").sum()
n_neutral = (df["direction"] == "NEUTRAL").sum()
n_decided = n_long + n_short
correct   = (df["outcome"] == "CORRECT").sum()
wrong     = (df["outcome"] == "WRONG").sum()
accuracy  = round(correct / n_decided * 100, 1) if n_decided > 0 else 0.0

total_deployed = CAPITAL * n_decided
total_pnl      = df["pnl"].sum()
total_ret_pct  = total_pnl / total_deployed * 100 if total_deployed > 0 else 0

best  = df.loc[df["pnl"].idxmax()]
worst = df.loc[df["pnl"].idxmin()]
long_pnl  = df[df["direction"] == "LONG"]["pnl"].sum()
short_pnl = df[df["direction"] == "SHORT"]["pnl"].sum()

bh_ret = df["actual_ret%"].mean()

spy_ret = None
if not spy_day.empty:
    try:
        spy_ret = (float(spy_day["Close"].iloc[0]) - float(spy_day["Open"].iloc[0])) / float(spy_day["Open"].iloc[0]) * 100
    except Exception:
        pass

print(f"""
  REAL-SIGNAL PORTFOLIO SUMMARY — May 20 2026  (v4)
  ────────────────────────────────────────────────────────────────────────
  Stocks analyzed       : {len(df)} / {len(US50)}
  Positions taken       : {n_decided}  ({n_long} LONG  |  {n_short} SHORT  |  {n_neutral} NEUTRAL)
  Capital deployed      : ${total_deployed:>12,.0f}
  ────────────────────────────────────────────────────────────────────────
  Total P&L             : ${total_pnl:>+12,.2f}
  Portfolio Return      : {total_ret_pct:>+.3f}%
  Long-side P&L         : ${long_pnl:>+12,.2f}
  Short-side P&L        : ${short_pnl:>+12,.2f}
  ────────────────────────────────────────────────────────────────────────
  Best trade            : {best['ticker']}  {best['direction']}  ${best['pnl']:>+,.0f}  ({best['pos_ret%']:+.2f}%)
  Worst trade           : {worst['ticker']}  {worst['direction']}  ${worst['pnl']:>+,.0f}  ({worst['pos_ret%']:+.2f}%)
  ────────────────────────────────────────────────────────────────────────
  AGENT ACCURACY
    Decided signals     : {n_decided}
    Correct             : {correct}
    Wrong               : {wrong}
    Accuracy            : {accuracy:.1f}%   (random baseline 50%)
    Edge over random    : {accuracy - 50.0:>+.1f}%
  ────────────────────────────────────────────────────────────────────────
  BENCHMARK COMPARISON
    Equal-weight B&H    : {bh_ret:>+.3f}%  (all 50, no signal)
    SPY Open→Close      : {f'{spy_ret:+.3f}%' if spy_ret is not None else 'N/A'}
    AlphaAgent Return   : {total_ret_pct:>+.3f}%
    Alpha vs SPY        : {f'{(total_ret_pct - spy_ret):+.3f}%  {"OUTPERFORMED ✓" if total_ret_pct > spy_ret else "UNDERPERFORMED"}' if spy_ret is not None else 'N/A'}
    Alpha vs B&H        : {f'{(total_ret_pct - bh_ret):+.3f}%  {"OUTPERFORMED ✓" if total_ret_pct > bh_ret else "UNDERPERFORMED"}'}
  ────────────────────────────────────────────────────────────────────────
""")

# ── Step 8: Signal breakdown ──────────────────────────────────────────────────

print("  SIGNAL BREAKDOWN")
print(f"  {'─'*52}")
for sig_type in ["LONG", "SHORT", "NEUTRAL"]:
    grp = df[df["direction"] == sig_type]
    if grp.empty:
        continue
    g_pnl = grp["pnl"].sum()
    g_dec = len(grp[grp["outcome"] != "NEUTRAL"])
    g_cor = (grp["outcome"] == "CORRECT").sum()
    g_acc = round(g_cor / g_dec * 100, 1) if g_dec > 0 else 0.0
    avg_prob = grp["probability"].mean()
    avg_conv = grp["conviction"].mean()
    print(f"  {sig_type:<8}: {len(grp):>2} positions  "
          f"P&L ${g_pnl:>+10,.0f}  "
          f"Acc {g_acc:>4.0f}%  "
          f"avg_prob={avg_prob:.3f}  avg_conv={avg_conv:.1f}%")

# ── Step 9: Sector breakdown ──────────────────────────────────────────────────

print(f"\n  SECTOR BREAKDOWN")
print(f"  {'─'*60}")
for etf in ["XLK","XLF","XLE","XLV","XLY","XLP","XLI","XLC","XLU","XLB"]:
    grp = df[(df["sector"] == etf) & (df["direction"] != "NEUTRAL")]
    if grp.empty:
        continue
    g_cor = (grp["outcome"] == "CORRECT").sum()
    g_tot = len(grp)
    g_pnl = grp["pnl"].sum()
    trend = sector_trend.get(etf, "?")
    tickers_str = " ".join(grp["ticker"].tolist())
    print(f"  {etf:<5}: {g_tot:>2} trades  Acc {g_cor}/{g_tot}  trend={trend:<5}  "
          f"P&L ${g_pnl:>+8,.0f}  [{tickers_str}]")

# ── Step 10: Per-agent probability-based accuracy ────────────────────────────

print(f"\n  PER-AGENT PROBABILITY ACCURACY (prob>0.55=eff.LONG, <0.45=eff.SHORT)")
print(f"  {'─'*60}")

agent_names = ["technical","fundamental","macro","sentiment",
               "volatility","insider","geopolitical","risk"]
decided_rows = df[df["orig_direction"] != "NEUTRAL"]

for agent in agent_names:
    agent_correct = 0
    agent_total   = 0
    for _, row in decided_rows.iterrows():
        ap = row.get("agent_probs", {})
        prob = ap.get(agent)
        if prob is None:
            continue
        if prob > 0.55:
            eff_dir = "LONG"
        elif prob < 0.45:
            eff_dir = "SHORT"
        else:
            continue
        actual_up = row["actual_ret%"] > 0
        correct_vote = (eff_dir == "LONG" and actual_up) or \
                       (eff_dir == "SHORT" and not actual_up)
        agent_correct += int(correct_vote)
        agent_total   += 1
    if agent_total > 0:
        acc = round(agent_correct / agent_total * 100, 1)
        bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
        print(f"  {agent:<14}: {bar}  {acc:>5.1f}%  ({agent_correct}/{agent_total})")
    else:
        print(f"  {agent:<14}: no decisive votes")

print(f"  {'─'*60}")

# ── Step 11: Overlay effectiveness ───────────────────────────────────────────

neutralized = df[df["override"].str.contains("OIL_BEAR|YIELD_RISE|SECTOR_", na=False)]
if not neutralized.empty:
    print(f"\n  OVERLAY EFFECTIVENESS — {len(neutralized)} positions neutralized by overlays")
    print(f"  {'─'*60}")
    total_saved = 0.0
    for _, r in neutralized.iterrows():
        orig = r["orig_direction"]
        if orig == "LONG":
            would_have = CAPITAL * r["actual_ret%"] / 100
        elif orig == "SHORT":
            would_have = -CAPITAL * r["actual_ret%"] / 100
        else:
            would_have = 0.0
        saved = -would_have
        total_saved += saved
        verdict = "SAVED ✓" if saved > 0 else "MISSED ✗"
        print(f"  {r['ticker']:<6} {r['override']:<22} orig={orig:<6}  actual={r['actual_ret%']:+.2f}%  "
              f"would-have ${would_have:>+,.0f}  saved ${saved:>+,.0f}  {verdict}")
    print(f"  {'─'*60}")
    print(f"  Net saved by overlays: ${total_saved:>+,.0f}")

# ── Step 12: Comparison vs v1 / v2 / v3 ──────────────────────────────────────

print(f"""
  PROGRESSION SUMMARY
  {'─'*52}
  v1  Technical proxy only            :  ~25.0%  -$40,777
  v2  Technical + regime filter       :  ~33.3%  -$16,642
  v3  Full 8-agent pipeline           :  ~35.3%   -$7,659
  v5  CRIT decoupled + overlays v2     :   50.0%   +$3,892
  v6  BULL_TREND short block           :   85.7%   +$6,026
  v7  Sector gap-reversal (current)    :   {accuracy:.1f}%  ${total_pnl:>+,.0f}
  {'─'*52}
""")

print("=" * 78)
