"""
AlphaAgent — June 5 2026 Single-Day Portfolio Backtest

Scenario: $100,000 TOTAL capital deployed on the morning of June 5 2026.
AlphaAgent generates signals on US-50 universe, picks positions, allocates
the $100K conviction-weighted across the chosen tickers, opens at 9:30 AM,
closes at 4:00 PM. Returns are compared to SPY Open→Close benchmark.

This is the realistic single-investor view (not $100K per position).
"""

import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from batch_signal_runner import run_batch
from quant_engine.transaction_costs import TransactionCostModel
import yfinance as yf
import pandas as pd
import numpy as np

# ── Universe ──────────────────────────────────────────────────────────────────

US50 = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","BRK-B","LLY",
    "AVGO","JPM","TSLA","UNH","V","XOM","MA","ORCL","COST",
    "JNJ","HD","PG","MRK","BAC","ABBV","CVX","KO","WMT",
    "PEP","ACN","MCD","TMO","ABT","LIN","TXN","CSCO","PM",
    "DHR","AMGN","IBM","GS","INTU","ISRG","CAT","MS","NOW",
    "AMAT","BKNG","LOW","RTX","SPGI","NEE",
]

CAPITAL_TOTAL = 100_000          # ← single-investor: $100K total
TRADE_DATE    = "2026-06-05"
LABEL         = "June 5 2026"

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

# ── Helpers ───────────────────────────────────────────────────────────────────

def _scalar(x) -> float:
    if hasattr(x, "item"): return float(x.item())
    return float(x)

def _squeeze(df, col="Close") -> pd.Series:
    c = df[col]
    if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
    r = c.squeeze()
    if isinstance(r, pd.Series): return r.dropna()
    return pd.Series([float(r)])

# ── Run HMM regime for the day ────────────────────────────────────────────────

def run_hmm(trade_date: str) -> dict:
    result = {
        "regime": "BULL", "bull_prob": 0.5, "trans_risk": 0.0,
        "size_scalar": 1.0, "xlf_block": False, "error": None,
    }
    try:
        from quant_engine.hmm import RegimeDetector
        h = yf.download("SPY", start="2025-01-01", end=trade_date,
                        auto_adjust=True, progress=False)
        if h.empty or len(h) < 105:
            result["error"] = "Insufficient SPY history"; return result
        rets = _squeeze(h, "Close").pct_change().dropna()
        hmm = RegimeDetector(n_states=3).fit_predict(rets)
        result["regime"] = hmm.current_regime
        result["bull_prob"] = hmm.probabilities["bull"]
        result["trans_risk"] = hmm.transition_risk["to_bear"] + hmm.transition_risk["to_crisis"]
        result["xlf_block"] = bool(result["trans_risk"] > 0.20)
        result["size_scalar"] = max(0.25, min(1.0, result["bull_prob"] * 1.5))
    except Exception as e:
        result["error"] = str(e)[:120]
    return result


# ═════════════════════════════════════════════════════════════════════════════
#  Step 1: Run signal pipeline once on US-50
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print(f"  AlphaAgent  |  $100K SINGLE-INVESTOR BACKTEST  |  {LABEL}")
print(f"  Total capital: ${CAPITAL_TOTAL:,}  |  Universe: US-50  |  Hold: Open→Close")
print("=" * 80)

signals = run_batch(US50, horizon="1d", max_workers=4, show_progress=True)

# ═════════════════════════════════════════════════════════════════════════════
#  Step 2: HMM regime + sector overlay computation
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'─'*80}")
print(f"  Computing HMM regime + market overlays for {LABEL} ...")
hmm = run_hmm(TRADE_DATE)
if hmm["error"]:
    print(f"  ⚠ HMM fallback: {hmm['error']}")
print(f"  HMM: regime={hmm['regime']}  bull_prob={hmm['bull_prob']:.3f}  "
      f"trans_risk={hmm['trans_risk']:.3f}  xlf_block={hmm['xlf_block']}  "
      f"size_scalar={hmm['size_scalar']:.3f}")

# ═════════════════════════════════════════════════════════════════════════════
#  Step 3: Download OHLC for trade date
# ═════════════════════════════════════════════════════════════════════════════

next_day = (pd.Timestamp(TRADE_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
day_raw  = yf.download(US50, start=TRADE_DATE, end=next_day,
                        auto_adjust=True, progress=False, group_by="ticker")
spy_day  = yf.download("SPY", start=TRADE_DATE, end=next_day,
                        auto_adjust=True, progress=False)

spy_ret = None
if not spy_day.empty:
    try:
        spy_ret = (_scalar(spy_day["Close"].iloc[0]) -
                   _scalar(spy_day["Open"].iloc[0])) / _scalar(spy_day["Open"].iloc[0]) * 100
    except Exception:
        pass

# ═════════════════════════════════════════════════════════════════════════════
#  Step 4: Filter signals — only directional (LONG/SHORT) with conviction
# ═════════════════════════════════════════════════════════════════════════════

candidates = []
for ticker in US50:
    sig = signals.get(ticker, {})
    if sig.get("error"): continue

    direction   = sig["direction"]
    if direction == "NEUTRAL": continue

    # HMM overlay: block all shorts in BULL regime
    if "BULL" in sig.get("market_regime", "") and direction == "SHORT":
        continue

    # HMM XLF block
    if hmm["xlf_block"] and SECTOR_MAP.get(ticker) == "XLF" and direction == "LONG":
        continue

    conviction = sig["conviction"]
    if conviction < 30: continue   # require >=30% conviction

    candidates.append({
        "ticker":      ticker,
        "direction":   direction,
        "probability": sig["probability"],
        "conviction":  conviction,
        "multiplier":  sig.get("multiplier", 1.0),
        "regime":      sig.get("market_regime", "?"),
        "agent_probs": sig.get("agent_probs", {}),
        "orig_dir":    sig["direction"],
    })

if not candidates:
    print("\n  ⚠ No qualifying signals on June 5 2026. Holding cash.")
    sys.exit(0)

print(f"\n  Qualifying signals: {len(candidates)}  "
      f"(LONG: {sum(1 for c in candidates if c['direction']=='LONG')} | "
      f"SHORT: {sum(1 for c in candidates if c['direction']=='SHORT')})")

# ═════════════════════════════════════════════════════════════════════════════
#  Step 5: Conviction-weighted allocation of $100K across candidates
# ═════════════════════════════════════════════════════════════════════════════

# Position weight = conviction × HMM size_scalar × multiplier
for c in candidates:
    c["raw_weight"] = c["conviction"] * c["multiplier"] * hmm["size_scalar"]

total_raw = sum(c["raw_weight"] for c in candidates)
for c in candidates:
    c["weight"]   = c["raw_weight"] / total_raw if total_raw > 0 else 1.0 / len(candidates)
    c["notional"] = CAPITAL_TOTAL * c["weight"]

# ═════════════════════════════════════════════════════════════════════════════
#  Step 6: Compute per-position P&L (with transaction costs)
# ═════════════════════════════════════════════════════════════════════════════

tcm = TransactionCostModel()
rows = []

for c in candidates:
    ticker = c["ticker"]
    try:
        day = day_raw[ticker].dropna(how="all") if len(US50) > 1 else day_raw.dropna(how="all")
        if day.empty: continue
        entry = float(day["Open"].iloc[0])
        exit_ = float(day["Close"].iloc[0])
        if pd.isna(entry) or pd.isna(exit_) or entry <= 0: continue
    except Exception:
        continue

    actual_ret = (exit_ - entry) / entry
    if c["direction"] == "LONG":  pos_ret = actual_ret
    else:                          pos_ret = -actual_ret
    gross_pnl = c["notional"] * pos_ret

    # Transaction cost — round trip
    tcm_res = tcm.round_trip(
        ticker=ticker, side=c["direction"], notional=c["notional"],
        adv_dollar=5e9, daily_vol=0.02, price=entry, holding_days=1,
    )
    net_pnl = gross_pnl - tcm_res.total_dollars

    if c["direction"] == "LONG"  and actual_ret > 0: outcome = "✓"
    elif c["direction"] == "SHORT" and actual_ret < 0: outcome = "✓"
    else:                                              outcome = "✗"

    rows.append({
        "ticker":       ticker,
        "direction":    c["direction"],
        "probability":  c["probability"],
        "conviction":   c["conviction"],
        "weight":       c["weight"],
        "notional":     c["notional"],
        "entry":        round(entry, 2),
        "exit":         round(exit_, 2),
        "actual_ret%":  round(actual_ret * 100, 3),
        "gross_pnl":    round(gross_pnl, 2),
        "tcm_cost":     round(tcm_res.total_dollars, 2),
        "net_pnl":      round(net_pnl, 2),
        "outcome":      outcome,
        "sector":       SECTOR_MAP.get(ticker, "—"),
        "regime":       c["regime"],
    })

if not rows:
    print("\n  ⚠ No tradeable positions (no data for June 5 2026).")
    sys.exit(0)

df = pd.DataFrame(rows).sort_values("net_pnl", ascending=False).reset_index(drop=True)

# ═════════════════════════════════════════════════════════════════════════════
#  Step 7: Print Trade Table
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n  {'─'*108}")
print(f"  TRADE TABLE — {LABEL}")
HFMT = "{:<6}  {:>6}  {:>5}  {:>5}  {:>10}  {:>8}  {:>8}  {:>9}  {:>10}  {:>8}  {:>9}  {:>3}"
print("  " + HFMT.format(
    "TICKER","DIR","PROB","WGT%","NOTIONAL","ENTRY","EXIT","ACT RET%",
    "GROSS P&L","TCM($)","NET P&L","OUT"
))
print(f"  {'─'*108}")
for _, r in df.iterrows():
    print("  " + HFMT.format(
        r["ticker"], r["direction"][:5], f"{r['probability']:.2f}",
        f"{r['weight']*100:.1f}", f"${r['notional']:>8,.0f}",
        f"${r['entry']:.2f}", f"${r['exit']:.2f}",
        f"{r['actual_ret%']:+.2f}%",
        f"${r['gross_pnl']:>+8,.0f}", f"${r['tcm_cost']:.0f}",
        f"${r['net_pnl']:>+8,.0f}", r["outcome"]
    ))
print(f"  {'─'*108}")

# ═════════════════════════════════════════════════════════════════════════════
#  Step 8: Summary + comparison vs SPY benchmark
# ═════════════════════════════════════════════════════════════════════════════

n_total       = len(df)
n_long        = (df["direction"] == "LONG").sum()
n_short       = (df["direction"] == "SHORT").sum()
n_correct     = (df["outcome"] == "✓").sum()
n_wrong       = (df["outcome"] == "✗").sum()
accuracy      = n_correct / n_total * 100 if n_total else 0
gross_pnl     = df["gross_pnl"].sum()
tcm_total     = df["tcm_cost"].sum()
net_pnl       = df["net_pnl"].sum()
portfolio_ret = net_pnl / CAPITAL_TOTAL * 100

# Best/worst
best  = df.loc[df["net_pnl"].idxmax()]
worst = df.loc[df["net_pnl"].idxmin()]

# What if equal-weighted instead?
ew_ret = df["actual_ret%"].mean()

# What if you just bought SPY?
if spy_ret is not None:
    spy_pnl = CAPITAL_TOTAL * (spy_ret / 100)
else:
    spy_pnl = None

print(f"""
  SUMMARY — {LABEL}
  ════════════════════════════════════════════════════════════════════════
  Capital deployed     : ${CAPITAL_TOTAL:>10,.0f}    (single account, all in)
  Number of positions  : {n_total:>10}    ({n_long} LONG | {n_short} SHORT)
  HMM regime           : {hmm['regime']:<10}    bull_prob={hmm['bull_prob']:.3f}
  ────────────────────────────────────────────────────────────────────────
  Gross P&L            : ${gross_pnl:>+10,.2f}
  Transaction costs    : ${tcm_total:>+10,.2f}    ({tcm_total/CAPITAL_TOTAL*100:.3f}% drag)
  NET P&L              : ${net_pnl:>+10,.2f}
  Portfolio return     : {portfolio_ret:>+10.3f}%
  ────────────────────────────────────────────────────────────────────────
  Accuracy             : {accuracy:>10.1f}%   ({n_correct}/{n_total} correct)
  Best position        : {best['ticker']} {best['direction']}  ${best['net_pnl']:>+,.0f} ({best['actual_ret%']:+.2f}%)
  Worst position       : {worst['ticker']} {worst['direction']}  ${worst['net_pnl']:>+,.0f} ({worst['actual_ret%']:+.2f}%)
  ════════════════════════════════════════════════════════════════════════
  Equal-weight basket  : {ew_ret:>+10.3f}%
  SPY Open→Close       : {f'{spy_ret:+.3f}%' if spy_ret is not None else '         N/A':>10}
  AlphaAgent return    : {portfolio_ret:>+10.3f}%""")

if spy_ret is not None:
    alpha = portfolio_ret - spy_ret
    spy_pnl_print = f'${spy_pnl:>+,.0f}'
    print(f"""  ────────────────────────────────────────────────────────────────────────
  Alpha vs SPY         : {alpha:>+10.3f}%   ({'✓ OUTPERFORMED' if alpha > 0 else '✗ UNDERPERFORMED'})
  SPY hypothetical P&L : {spy_pnl_print:>10}    (had you just bought SPY with $100K)
  AlphaAgent edge      : ${net_pnl - spy_pnl:>+10,.2f}""")

# Sector breakdown
print(f"\n  SECTOR BREAKDOWN")
print(f"  {'─'*65}")
for sec in sorted(df["sector"].unique()):
    sub = df[df["sector"] == sec]
    s_corr = (sub["outcome"] == "✓").sum()
    s_pnl  = sub["net_pnl"].sum()
    tks = ", ".join(sub["ticker"].tolist())
    print(f"  {sec:<5}: {len(sub):>2} positions  Acc {s_corr}/{len(sub)}  "
          f"NET P&L ${s_pnl:>+8,.0f}  [{tks}]")

# Per-agent decisive-vote accuracy
print(f"\n  PER-AGENT DECISIVE-VOTE ACCURACY")
print(f"  {'─'*60}")
for ag in ["technical","fundamental","macro","sentiment",
           "volatility","insider","geopolitical","risk"]:
    cor = tot = 0
    for _, r in df.iterrows():
        ap = signals.get(r["ticker"], {}).get("agent_probs", {})
        p = ap.get(ag)
        if p is None: continue
        if p > 0.55:   ev = "LONG"
        elif p < 0.45: ev = "SHORT"
        else:          continue
        correct = (ev == "LONG"  and r["actual_ret%"] > 0) or \
                  (ev == "SHORT" and r["actual_ret%"] < 0)
        cor += int(correct); tot += 1
    if tot:
        acc = cor / tot * 100
        bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
        print(f"  {ag:<14}: {bar}  {acc:>5.1f}%  ({cor}/{tot})")
    else:
        print(f"  {ag:<14}: no decisive votes")

print("\n" + "=" * 80)
print(f"  Result: ${CAPITAL_TOTAL:,} → ${CAPITAL_TOTAL + net_pnl:,.2f}  "
      f"({portfolio_ret:+.3f}%)  in 1 trading day")
print("=" * 80)
