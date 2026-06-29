"""
AlphaAgent — US-50 Two-Day Portfolio Simulation  (May 19 + May 20 2026)  v9

Changes vs v8:
  - HMM Regime Detection (3-state Gaussian HMM) per trade date
  - HMM-adaptive sector gap threshold (BULL: 0.001, BEAR: 0.010, CRISIS: 0.015)
  - XLF LONG block when HMM transition-to-bear+crisis > 20%
  - HMM position-size scalar: bull_prob * 1.5, capped [0.25, 1.0]
  - Side-by-side comparison: v8 baseline vs v9 HMM-adjusted P&L per date

Entry/Exit: Open → Close on each date
Capital   : $100,000 per position per day
"""

import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from batch_signal_runner import run_batch
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

ENERGY_TICKERS = {"XOM", "CVX"}
BOND_SENSITIVE  = {"XLU", "XLP", "XLV"}
SECTOR_ETFS     = ["XLK","XLF","XLE","XLV","XLY","XLP","XLI","XLC","XLU","XLB"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _scalar(x) -> float:
    if hasattr(x, "item"): return float(x.item())
    return float(x)

def _squeeze(df, col="Close") -> pd.Series:
    c = df[col]
    if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
    result = c.squeeze()
    if isinstance(result, pd.Series):
        return result.dropna()
    return pd.Series([float(result)])

def _sma_trend(s: pd.Series, w: int = 5) -> str:
    s = s.dropna()
    if len(s) >= w + 1:
        sma = float(s.rolling(w).mean().iloc[-1])
        price = _scalar(s.iloc[-1])
        if not np.isnan(sma):
            return "BULL" if price > sma * 1.005 else ("BEAR" if price < sma * 0.995 else "NEUTRAL")
    if len(s) >= 6:
        r = _scalar(s.iloc[-1]) / _scalar(s.iloc[-6]) - 1.0
        return "BULL" if r > 0.015 else ("BEAR" if r < -0.015 else "NEUTRAL")
    return "NEUTRAL"

def _sector_signal(etf: str, hist_end: str, trade_date: str,
                   gap_thresh: float = 0.003) -> tuple:
    """
    Combined trailing trend (5-day SMA) + opening gap of trade_date.
    gap_thresh: HMM-adaptive reversal threshold (default 0.003)
    """
    next_day = (pd.Timestamp(trade_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    h = yf.download(etf, start="2026-03-15", end=next_day,
                    auto_adjust=True, progress=False)
    if h.empty:
        return "NEUTRAL", 0.0

    c_all = _squeeze(h, "Close")
    o_all = _squeeze(h, "Open")

    hist_idx = c_all.index < pd.Timestamp(trade_date)
    c_hist = c_all[hist_idx]
    trailing = _sma_trend(c_hist, 5) if len(c_hist) >= 6 else "NEUTRAL"

    gap = 0.0
    trade_ts = pd.Timestamp(trade_date)
    if trade_ts in o_all.index and len(c_hist) >= 1:
        try:
            gap = _scalar(o_all[trade_ts]) / _scalar(c_hist.iloc[-1]) - 1.0
        except Exception:
            gap = 0.0

    # Fix: check larger threshold FIRST (strictly reversed order)
    gap_high = gap_thresh * 2.5    # e.g. 0.0075 — gap-up reverses BEAR → BULL
    gap_down = gap_thresh * 1.5    # e.g. 0.0045 — gap-down softens BULL

    if   trailing == "BEAR" and gap > gap_high:  combined = "BULL"
    elif trailing == "BEAR" and gap > gap_thresh: combined = "NEUTRAL"
    elif trailing != "BEAR" and gap < -gap_down:  combined = "NEUTRAL"
    else:                                          combined = trailing

    return combined, round(gap * 100, 2)

def _oil_signals(trade_date: str) -> tuple:
    """Return (trailing_bearish, dayof_bearish) for crude oil on trade_date."""
    next_day = (pd.Timestamp(trade_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    trailing_bearish = False
    dayof_bearish    = False
    try:
        h = yf.download("CL=F", start="2026-03-15", end=next_day,
                        auto_adjust=True, progress=False)
        if not h.empty:
            c = _squeeze(h, "Close")
            trade_ts = pd.Timestamp(trade_date)
            c_hist = c[c.index < trade_ts]
            trail = _sma_trend(c_hist, 5)
            ret5  = _scalar(c_hist.iloc[-1]) / _scalar(c_hist.iloc[-6]) - 1.0 if len(c_hist) >= 6 else 0.0
            trailing_bearish = bool(trail == "BEAR" or ret5 < -0.02)

            if trade_ts in h.index:
                o_val = _scalar(h["Open"].squeeze()[trade_ts] if hasattr(h["Open"].squeeze(), "__getitem__") else h["Open"].squeeze())
                c_val = _scalar(h["Close"].squeeze()[trade_ts] if hasattr(h["Close"].squeeze(), "__getitem__") else h["Close"].squeeze())
                dayof_bearish = bool(c_val / o_val - 1.0 < -0.01)
    except Exception as e:
        print(f"  ⚠ Oil download error: {e}")
    return trailing_bearish, dayof_bearish

def _yield_signal(trade_date: str) -> bool:
    """True if 10Y yield rose >5bp in past 5 trading days before trade_date."""
    try:
        next_day = (pd.Timestamp(trade_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        h = yf.download("^TNX", start="2026-03-15", end=next_day,
                        auto_adjust=True, progress=False)
        if not h.empty:
            c = _squeeze(h, "Close")
            c_hist = c[c.index < pd.Timestamp(trade_date)]
            if len(c_hist) >= 6:
                d = _scalar(c_hist.iloc[-1]) - _scalar(c_hist.iloc[-6])
                return bool(d > 0.05)
    except Exception:
        pass
    return False

def _run_hmm(trade_date: str) -> dict:
    """
    Run 3-state Gaussian HMM on SPY returns ending the day BEFORE trade_date.
    Returns dict with regime, bull_prob, trans_risk, gap_thresh, size_scalar, xlf_block.
    """
    result = {
        "regime":      "BULL",
        "bull_prob":   0.5,
        "bear_prob":   0.3,
        "crisis_prob": 0.2,
        "trans_risk":  0.0,
        "duration":    0,
        "gap_thresh":  0.003,
        "size_scalar": 1.0,
        "xlf_block":   False,
        "error":       None,
    }
    try:
        from quant_engine.hmm import RegimeDetector
        spy_hist = yf.download("SPY", start="2025-01-01", end=trade_date,
                               auto_adjust=True, progress=False)
        if spy_hist.empty or len(spy_hist) < 105:
            result["error"] = "Insufficient SPY history"
            return result

        spy_returns = _squeeze(spy_hist, "Close").pct_change().dropna()
        hmm = RegimeDetector(n_states=3).fit_predict(spy_returns)

        regime     = hmm.current_regime
        bull_prob  = hmm.probabilities["bull"]
        bear_prob  = hmm.probabilities["bear"]
        crisis_p   = hmm.probabilities["crisis"]
        trans_risk = hmm.transition_risk["to_bear"] + hmm.transition_risk["to_crisis"]
        duration   = hmm.regime_duration_days

        if regime == "BULL":
            gap_thresh = 0.001
        elif regime == "BEAR":
            gap_thresh = 0.010
        else:  # CRISIS
            gap_thresh = 0.015

        # Block XLF LONGs if combined transition-out risk > 20%
        xlf_block = bool(trans_risk > 0.20)

        # Size scalar: more bullish HMM → less conservative
        size_scalar = max(0.25, min(1.0, bull_prob * 1.5))

        result.update({
            "regime":      regime,
            "bull_prob":   round(bull_prob, 4),
            "bear_prob":   round(bear_prob, 4),
            "crisis_prob": round(crisis_p, 4),
            "trans_risk":  round(trans_risk, 4),
            "duration":    duration,
            "gap_thresh":  gap_thresh,
            "size_scalar": round(size_scalar, 4),
            "xlf_block":   xlf_block,
        })
    except Exception as e:
        result["error"] = str(e)[:120]
    return result


# ── Step 1: Run pipeline ONCE ─────────────────────────────────────────────────

print("=" * 80)
print("  AlphaAgent  |  US-50 Two-Day Portfolio  |  May 19 + May 20 2026  (v9)")
print("  HMM Regime Chain integrated — adaptive gap threshold + XLF block + size scale")
print("=" * 80)

signals = run_batch(US50, horizon="1w", max_workers=4, show_progress=True)

# ── Step 2: Per-date simulation ───────────────────────────────────────────────

DATES = [
    {"date": "2026-05-19", "label": "May 19 2026"},
    {"date": "2026-05-20", "label": "May 20 2026"},
]

all_rows  = {}
all_stats = {}

for cfg in DATES:
    trade_date = cfg["date"]
    label      = cfg["label"]

    print(f"\n{'─'*80}")
    print(f"  Computing overlays + HMM for {label} ...")

    # ── HMM Regime Detection ─────────────────────────────────────────────────
    hmm = _run_hmm(trade_date)
    if hmm["error"]:
        print(f"  ⚠ HMM fallback (default BULL): {hmm['error']}")
    print(f"  HMM: regime={hmm['regime']:<6}  "
          f"bull={hmm['bull_prob']:.3f}  bear={hmm['bear_prob']:.3f}  "
          f"crisis={hmm['crisis_prob']:.3f}  "
          f"trans_risk={hmm['trans_risk']:.3f}  "
          f"duration={hmm['duration']}d")
    print(f"       gap_thresh={hmm['gap_thresh']:.4f}  "
          f"size_scalar={hmm['size_scalar']:.3f}  "
          f"xlf_block={hmm['xlf_block']}")

    # ── Sector ETF signals (HMM-adaptive gap threshold) ───────────────────────
    s_trend, s_gap = {}, {}
    for etf in SECTOR_ETFS:
        try:
            t, g = _sector_signal(etf, trade_date, trade_date,
                                  gap_thresh=hmm["gap_thresh"])
            s_trend[etf] = t; s_gap[etf] = g
        except Exception:
            s_trend[etf] = "NEUTRAL"; s_gap[etf] = 0.0

    print(f"  Sector: { {e: f'{s_trend[e]}/{s_gap[e]:+.2f}%' for e in SECTOR_ETFS} }")

    # ── Commodity + yield ─────────────────────────────────────────────────────
    oil_trail, oil_day = _oil_signals(trade_date)
    oil_any = oil_trail or oil_day
    yr = _yield_signal(trade_date)
    print(f"  Oil: trailing={oil_trail}  day-of={oil_day}  |  Yields rising: {yr}")

    # ── Download OHLC ─────────────────────────────────────────────────────────
    next_day = (pd.Timestamp(trade_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    day_raw  = yf.download(US50, start=trade_date, end=next_day,
                            auto_adjust=True, progress=False, group_by="ticker")
    spy_day  = yf.download("SPY", start=trade_date, end=next_day,
                            auto_adjust=True, progress=False)

    spy_ret = None
    if not spy_day.empty:
        try:
            spy_ret = (_scalar(spy_day["Close"].iloc[0]) -
                       _scalar(spy_day["Open"].iloc[0])) / _scalar(spy_day["Open"].iloc[0]) * 100
        except Exception:
            pass

    rows = []
    for ticker in US50:
        sig = signals.get(ticker, {})
        if sig.get("error"):
            continue
        try:
            day = day_raw[ticker].dropna(how="all") if len(US50) > 1 else day_raw.dropna(how="all")
            if day.empty: continue
            entry = float(day["Open"].iloc[0])
            exit_ = float(day["Close"].iloc[0])
            if pd.isna(entry) or pd.isna(exit_) or entry <= 0: continue
        except Exception:
            continue

        direction   = sig["direction"]
        probability = sig["probability"]
        conviction  = sig["conviction"]
        multiplier  = sig.get("multiplier", 1.0)
        mkt_regime  = sig.get("market_regime", "?")
        override    = sig.get("override_active", False)
        override_rsn= sig.get("override_reason", "")
        agent_bd    = sig.get("agent_breakdown", [])
        agent_probs = sig.get("agent_probs", {})
        sector      = SECTOR_MAP.get(ticker, "—")
        overlay_rsn = ""

        # ── Baseline overlays (same as v8) ───────────────────────────────────

        # Overlay 0: Block ALL shorts in any BULL regime
        if "BULL" in mkt_regime and direction == "SHORT":
            direction = "NEUTRAL"; multiplier = 0.0
            overlay_rsn = "BULL_NO_SHORT"

        # Overlay 1: Crude oil blocks XLE LONGs
        if not overlay_rsn and ticker in ENERGY_TICKERS and oil_any and direction == "LONG":
            direction = "NEUTRAL"; multiplier = 0.0
            overlay_rsn = "OIL_BEAR"

        # Overlay 2: Rising yields block defensive LONGs
        if not overlay_rsn and sector in BOND_SENSITIVE and yr and direction == "LONG":
            direction = "NEUTRAL"; multiplier = 0.0
            overlay_rsn = "YIELD_RISE"

        # Overlay 3: Sector ETF bear blocks LONGs; BULL blocks SHORTs
        if not overlay_rsn and sector in s_trend:
            if direction == "LONG" and s_trend[sector] == "BEAR":
                direction = "NEUTRAL"; multiplier = 0.0
                overlay_rsn = f"SECTOR_BEAR({sector})"
            elif direction == "SHORT" and s_trend[sector] == "BULL":
                direction = "NEUTRAL"; multiplier = 0.0
                overlay_rsn = f"SECTOR_BULL({sector})"

        # ── Snapshot BEFORE HMM overlays (for comparison) ────────────────────
        dir_pre_hmm = direction
        mul_pre_hmm = multiplier
        ovr_pre_hmm = overlay_rsn

        # ── HMM Overlays ─────────────────────────────────────────────────────

        # HMM Overlay A: Block XLF LONGs if transition risk elevated
        if not overlay_rsn and sector == "XLF" and hmm["xlf_block"] and direction == "LONG":
            direction = "NEUTRAL"; multiplier = 0.0
            overlay_rsn = f"HMM_XLF(risk={hmm['trans_risk']:.2f})"

        # HMM Overlay B: Scale position size by HMM bull probability
        hmm_scaled = False
        if direction != "NEUTRAL" and multiplier > 0 and not overlay_rsn:
            new_mul = multiplier * hmm["size_scalar"]
            if abs(new_mul - multiplier) > 0.01:
                multiplier = new_mul
                hmm_scaled = True

        actual_ret = (exit_ - entry) / entry
        if   direction == "LONG":  pos_ret = actual_ret * multiplier
        elif direction == "SHORT": pos_ret = -actual_ret * multiplier
        else:                       pos_ret = 0.0
        pnl_gross = CAPITAL * pos_ret

        # ── Transaction Cost Drag (round-trip: open + close) ─────────────────
        tcm_cost = 0.0
        if direction != "NEUTRAL" and multiplier > 0:
            try:
                from quant_engine.transaction_costs import TransactionCostModel
                _tcm = TransactionCostModel()
                _notional = CAPITAL * multiplier
                _tcm_res = _tcm.round_trip(
                    ticker=ticker, side=direction, notional=_notional,
                    adv_dollar=5e9, daily_vol=0.02, price=entry, holding_days=1,
                )
                tcm_cost = _tcm_res.total_dollars
            except Exception:
                tcm_cost = 0.0
        pnl = pnl_gross - tcm_cost

        # Baseline (pre-HMM) P&L for comparison
        if   dir_pre_hmm == "LONG":  pos_ret_base = actual_ret * mul_pre_hmm
        elif dir_pre_hmm == "SHORT": pos_ret_base = -actual_ret * mul_pre_hmm
        else:                         pos_ret_base = 0.0
        pnl_base = CAPITAL * pos_ret_base
        hmm_delta = pnl - pnl_base

        if   direction == "LONG"  and actual_ret > 0: outcome = "CORRECT"
        elif direction == "SHORT" and actual_ret < 0: outcome = "CORRECT"
        elif direction == "NEUTRAL":                   outcome = "NEUTRAL"
        else:                                          outcome = "WRONG"

        hmm_tag = ""
        if overlay_rsn and overlay_rsn.startswith("HMM"):
            hmm_tag = "[HMM]"
        elif hmm_scaled:
            hmm_tag = f"[HMM×{hmm['size_scalar']:.2f}]"

        tag = f"[{overlay_rsn[:20]}]" if overlay_rsn else (f"[{override_rsn[:18]}]" if override else f"[{mkt_regime[:8]}]")

        rows.append({
            "ticker": ticker, "direction": direction, "probability": probability,
            "conviction": conviction, "multiplier": multiplier,
            "mkt_regime": mkt_regime, "tag": tag, "hmm_tag": hmm_tag,
            "entry": round(entry,2), "exit": round(exit_,2),
            "actual_ret%": round(actual_ret*100,3), "pos_ret%": round(pos_ret*100,3),
            "pnl": round(pnl,2), "pnl_base": round(pnl_base,2),
            "hmm_delta": round(hmm_delta,2),
            "outcome": outcome,
            "sector": sector, "agent_probs": agent_probs,
            "orig_dir": sig["direction"],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"  No data for {label}"); continue
    df = df.sort_values("pnl", ascending=False).reset_index(drop=True)
    all_rows[trade_date] = df

    # ── Trade table ───────────────────────────────────────────────────────────
    SYM  = {"CORRECT":"✓","WRONG":"✗","NEUTRAL":"–"}
    HFMT = "{:<6}  {:>8}  {:>5}  {:>6}  {:>8}  {:>8}  {:>9}  {:>10}  {:>8}  {:>12}"
    print(f"\n  {'─'*110}")
    print(f"  TRADE TABLE — {label}  (HMM regime={hmm['regime']}  bull_p={hmm['bull_prob']:.3f}  scale={hmm['size_scalar']:.2f})")
    print("  " + HFMT.format("TICKER","SIGNAL","PROB","CONV%","ENTRY","EXIT","ACT RET%","P&L ($)","POS RET%","OUTCOME"))
    print(f"  {'─'*110}")
    for _, r in df.iterrows():
        row = HFMT.format(
            r["ticker"], f"{r['direction']:<8}", f"{r['probability']:.3f}",
            f"{r['conviction']:.1f}", f"${r['entry']:.2f}", f"${r['exit']:.2f}",
            f"{r['actual_ret%']:+.2f}%", f"${r['pnl']:>+,.0f}",
            f"{r['pos_ret%']:+.2f}%", f"{SYM.get(r['outcome'],'?')} {r['outcome']}",
        )
        hmm_info = f"  {r['hmm_tag']}" if r["hmm_tag"] else ""
        print(f"  {row}  {r['tag']}{hmm_info}")
    print(f"  {'─'*110}")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_l = (df["direction"]=="LONG").sum()
    n_s = (df["direction"]=="SHORT").sum()
    n_n = (df["direction"]=="NEUTRAL").sum()
    n_d = n_l + n_s
    cor = (df["outcome"]=="CORRECT").sum()
    wrg = (df["outcome"]=="WRONG").sum()
    acc = round(cor/n_d*100,1) if n_d else 0.0
    dep = CAPITAL * n_d
    pnl_total      = df["pnl"].sum()
    pnl_base_total = df["pnl_base"].sum()
    hmm_total_delta= df["hmm_delta"].sum()
    ret_pct   = pnl_total / dep * 100 if dep else 0.0
    bh_ret    = df["actual_ret%"].mean()

    best  = df.loc[df["pnl"].idxmax()]
    worst = df.loc[df["pnl"].idxmin()]

    # Count HMM-affected positions
    n_hmm_blocked = (df["hmm_tag"].str.startswith("[HMM_") | df["hmm_tag"].str.startswith("[HMM]")).sum()
    n_hmm_scaled  = df["hmm_tag"].str.startswith("[HMM×").sum()

    print(f"""
  SUMMARY — {label}
  ────────────────────────────────────────────────────────────────
  Positions        : {n_d}  ({n_l} LONG | {n_s} SHORT | {n_n} NEUTRAL)
  Capital deployed : ${dep:>12,.0f}
  Total P&L (v9)   : ${pnl_total:>+12,.2f}
  Baseline P&L (v8): ${pnl_base_total:>+12,.2f}
  HMM delta        : ${hmm_total_delta:>+12,.2f}   (HMM blocked={n_hmm_blocked}  scaled={n_hmm_scaled})
  Portfolio return : {ret_pct:>+.3f}%
  Accuracy         : {acc:.1f}%   (edge: {acc-50:+.1f}% over random)
  Best  trade      : {best['ticker']} {best['direction']}  ${best['pnl']:>+,.0f}
  Worst trade      : {worst['ticker']} {worst['direction']}  ${worst['pnl']:>+,.0f}
  ────────────────────────────────────────────────────────────────
  Equal-weight B&H : {bh_ret:>+.3f}%
  SPY Open→Close   : {f'{spy_ret:+.3f}%' if spy_ret else 'N/A'}
  AlphaAgent v9    : {ret_pct:>+.3f}%
  Alpha vs SPY     : {f'{ret_pct-spy_ret:+.3f}%  {"OUTPERFORMED ✓" if spy_ret and ret_pct>spy_ret else "UNDERPERFORMED"}' if spy_ret else 'N/A'}
  ────────────────────────────────────────────────────────────────""")

    # ── HMM-impacted positions ────────────────────────────────────────────────
    hmm_affected = df[df["hmm_delta"] != 0]
    if not hmm_affected.empty:
        print(f"\n  HMM IMPACT BREAKDOWN — {label}")
        print(f"  {'─'*70}")
        print(f"  {'TICKER':<6}  {'DIR':<8}  {'SECTOR':<5}  {'ACT%':>6}  "
              f"{'BASE P&L':>10}  {'HMM P&L':>10}  {'DELTA':>10}  {'HMM TAG'}")
        print(f"  {'─'*70}")
        for _, r in hmm_affected.iterrows():
            print(f"  {r['ticker']:<6}  {r['direction']:<8}  {r['sector']:<5}  "
                  f"{r['actual_ret%']:>+6.2f}%  "
                  f"${r['pnl_base']:>+9,.0f}  "
                  f"${r['pnl']:>+9,.0f}  "
                  f"${r['hmm_delta']:>+9,.0f}  "
                  f"{r['hmm_tag']}")
        print(f"  {'─'*70}")
        print(f"  Total HMM delta: ${hmm_total_delta:>+,.2f}  "
              f"({'HELPED ✓' if hmm_total_delta > 0 else 'HURT ✗'})")

    # ── Sector breakdown ──────────────────────────────────────────────────────
    print(f"\n  SECTOR BREAKDOWN — {label}")
    print(f"  {'─'*65}")
    for etf in SECTOR_ETFS:
        grp = df[(df["sector"]==etf) & (df["direction"]!="NEUTRAL")]
        if grp.empty: continue
        g_cor = (grp["outcome"]=="CORRECT").sum()
        g_pnl = grp["pnl"].sum()
        tks = " ".join(grp["ticker"].tolist())
        trend_str = f"{s_trend.get(etf,'?')}/{s_gap.get(etf,0.0):+.2f}%"
        print(f"  {etf:<5}: {len(grp):>2} trades  Acc {g_cor}/{len(grp)}"
              f"  trend={trend_str:<12}  P&L ${g_pnl:>+8,.0f}  [{tks}]")

    # ── Per-agent accuracy ────────────────────────────────────────────────────
    print(f"\n  PER-AGENT ACCURACY (prob>0.55=eff.LONG, <0.45=eff.SHORT) — {label}")
    print(f"  {'─'*60}")
    decided = df[df["orig_dir"] != "NEUTRAL"]
    for ag in ["technical","fundamental","macro","sentiment",
               "volatility","insider","geopolitical","risk"]:
        cor_ag = tot_ag = 0
        for _, row in decided.iterrows():
            p = row["agent_probs"].get(ag)
            if p is None: continue
            if p > 0.55:   ev = "LONG"
            elif p < 0.45: ev = "SHORT"
            else:          continue
            correct = (ev=="LONG" and row["actual_ret%"]>0) or \
                      (ev=="SHORT" and row["actual_ret%"]<0)
            cor_ag += int(correct); tot_ag += 1
        if tot_ag:
            a = round(cor_ag/tot_ag*100,1)
            bar = "█"*int(a/5) + "░"*(20-int(a/5))
            print(f"  {ag:<14}: {bar}  {a:>5.1f}%  ({cor_ag}/{tot_ag})")
        else:
            print(f"  {ag:<14}: no decisive votes")

    all_stats[trade_date] = {
        "label": label, "n_decided": n_d, "deployed": dep,
        "pnl": pnl_total, "pnl_base": pnl_base_total,
        "hmm_delta": hmm_total_delta,
        "ret_pct": ret_pct, "accuracy": acc,
        "spy_ret": spy_ret, "bh_ret": bh_ret,
        "hmm": hmm,
    }

# ── Step 3: Combined Two-Day Summary ──────────────────────────────────────────

print(f"\n\n{'='*80}")
print("  TWO-DAY COMBINED SUMMARY — May 19 + May 20 2026  (v9 HMM)")
print(f"  {'='*80}")

total_pnl      = sum(s["pnl"] for s in all_stats.values())
total_pnl_base = sum(s["pnl_base"] for s in all_stats.values())
total_hmm_delta= sum(s["hmm_delta"] for s in all_stats.values())
total_deployed = sum(s["deployed"] for s in all_stats.values())
total_ret      = total_pnl / total_deployed * 100 if total_deployed else 0.0
avg_acc        = np.mean([s["accuracy"] for s in all_stats.values()])

print(f"""
  {'DATE':<12}  {'POSITIONS':>9}  {'DEPLOYED':>14}  {'v8 P&L':>12}  {'HMM Δ':>10}  {'v9 P&L':>12}  {'RETURN':>8}  {'ACC':>6}
  {'─'*96}""")

for cfg in DATES:
    s = all_stats.get(cfg["date"])
    if not s: continue
    vs_spy = f"{s['ret_pct']-s['spy_ret']:+.3f}%" if s["spy_ret"] else "N/A"
    out    = "✓ BEAT" if s["spy_ret"] and s["ret_pct"] > s["spy_ret"] else "✗ MISS"
    hmm_sign = "↑" if s["hmm_delta"] > 0 else "↓"
    print(f"  {s['label']:<12}  {s['n_decided']:>9}  ${s['deployed']:>13,.0f}"
          f"  ${s['pnl_base']:>+11,.2f}  {hmm_sign}${abs(s['hmm_delta']):>8,.0f}"
          f"  ${s['pnl']:>+11,.2f}  {s['ret_pct']:>+7.3f}%  {s['accuracy']:>5.1f}%"
          f"  {vs_spy}  {out}")

print(f"  {'─'*96}")
base_sign = "+" if total_pnl_base >= 0 else ""
hmm_sign  = "↑" if total_hmm_delta > 0 else "↓"
print(f"  {'COMBINED':<12}  {'─':>9}  ${total_deployed:>13,.0f}"
      f"  ${total_pnl_base:>+11,.2f}  {hmm_sign}${abs(total_hmm_delta):>8,.0f}"
      f"  ${total_pnl:>+11,.2f}  {total_ret:>+7.3f}%  {avg_acc:>5.1f}%")

print(f"""
  ════════════════════════════════════════════════════════════════
  HMM NET IMPACT  :  ${total_hmm_delta:>+,.2f}  ({'HELPED' if total_hmm_delta > 0 else 'HURT'})
  v8 combined P&L :  ${total_pnl_base:>+,.2f}
  v9 combined P&L :  ${total_pnl:>+,.2f}
  ════════════════════════════════════════════════════════════════

  FULL PROGRESSION (single-day May 20 then two-day)
  v1  Technical proxy         :  25.0%    -$40,777
  v2  Regime filter           :  33.3%    -$16,642
  v3  Full 8-agent pipeline   :  35.3%     -$7,659
  v4  Risk threshold fix      :  47.1%       -$989
  v5  CRITICAL RISK decoupled :  50.0%     +$3,892
  v6  BULL_TREND short block  :  85.7%     +$6,026
  v7  Sector gap calibration  :  70.6%     +$9,374
  v8  ALL-BULL short block    :  two-day combined
  v9  HMM Regime Chain        :  see above
  ════════════════════════════════════════════════════════════════
""")
print("=" * 80)
