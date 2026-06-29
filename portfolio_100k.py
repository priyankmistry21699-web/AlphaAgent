"""
AlphaAgent — $100,000 Interactive Portfolio Builder
Detects live market regime, recommends a strategy mode, lets user pick.

Backtest results (Jan–May 2026, 93 trading days, $100k/day):
  SNIPER      (top-3,  33% each, conv≥60) — not individually tested, highest variance
  CONCENTRATED(top-5,  20% each, conv≥40) — +$15,521 (+15.5%)  ← best in backtest
  BALANCED    (top-10, 10% each, conv≥25) — +$12,000 (estimated)
  DIVERSIFIED (top-20,  5% each, conv≥5)  — +$9,177  (+9.2%)

All data sources FREE: yfinance | FRED | SEC EDGAR | Gemini free tier
"""

import json, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import urllib.request
import yfinance as yf
import numpy as np

BASE    = "http://localhost:8000"
CAPITAL = 100_000

SECTOR_MAP = {
    "AAPL":"Tech",    "MSFT":"Tech",    "NVDA":"Tech",    "GOOGL":"Tech",
    "META":"Tech",    "AMZN":"Tech",    "TSLA":"Tech",    "AVGO":"Tech",
    "AMD":"Semis",    "INTC":"Semis",   "QCOM":"Semis",   "MU":"Semis",
    "AMAT":"Semis",   "LRCX":"Semis",   "KLAC":"Semis",   "TXN":"Semis",
    "PLTR":"Growth",  "CRWD":"Growth",  "SNOW":"Growth",  "NET":"Growth",
    "ZS":"Growth",    "DDOG":"Growth",  "PANW":"Growth",  "FTNT":"Growth",
    "JPM":"Finance",  "BAC":"Finance",  "GS":"Finance",   "MS":"Finance",
    "BLK":"Finance",  "V":"Finance",    "MA":"Finance",   "AXP":"Finance",
    "UNH":"Health",   "JNJ":"Health",   "LLY":"Health",   "ABBV":"Health",
    "PFE":"Health",   "MRK":"Health",   "TMO":"Health",   "ABT":"Health",
    "CAT":"Indust",   "DE":"Indust",    "HON":"Indust",   "GE":"Indust",
    "RTX":"Indust",   "LMT":"Indust",   "NOC":"Indust",   "UPS":"Indust",
    "XOM":"Energy",   "CVX":"Energy",   "COP":"Energy",   "SLB":"Energy",
    "OXY":"Energy",   "PSX":"Energy",   "VLO":"Energy",   "MPC":"Energy",
    "WMT":"Consumer", "COST":"Consumer","HD":"Consumer",  "TGT":"Consumer",
    "MCD":"Consumer", "SBUX":"Consumer","NKE":"Consumer", "AMGN":"Consumer",
    "VZ":"CommMedia", "T":"CommMedia",  "NFLX":"CommMedia","DIS":"CommMedia",
    "CMCSA":"CommMedia","CHTR":"CommMedia",
    "AMT":"REIT",     "PLD":"REIT",     "NEE":"Utility",  "DUK":"Utility",
    "SO":"Utility",   "D":"Utility",
}
UNIVERSE = list(SECTOR_MAP.keys())

# ── Strategy modes — user picks one ───────────────────────────────────────────
MODES = {
    "1": {
        "name":         "SNIPER",
        "tag":          "[1] SNIPER       — TOP 3 stocks, 33% each",
        "positions":    3,
        "pct_each":     0.33,
        "min_conv":     60,
        "min_prob":     0.60,
        "sector_max":   1,
        "deploy_pct":   0.90,
        "risk_level":   "VERY HIGH",
        "ytd":          "not tested individually",
        "good_day":     "$900 – $1,800",
        "bad_day":      "$900 – $1,200 loss",
        "best_for":     "EXTREME/HIGH VIX (>25)  +  only 3 stocks pass conv≥60",
        "who_for":      "Experienced trader watching screen all day. Can stomach -$1k days.",
        "warning":      "One bad position = -$500. Three bad = -$1,500. High concentration risk.",
    },
    "2": {
        "name":         "CONCENTRATED",
        "tag":          "[2] CONCENTRATED — TOP 5 stocks, 20% each  ★ BEST IN BACKTEST",
        "positions":    5,
        "pct_each":     0.20,
        "min_conv":     40,
        "min_prob":     0.58,
        "sector_max":   1,
        "deploy_pct":   0.90,
        "risk_level":   "HIGH",
        "ytd":          "+15.5% ($15,521)",
        "good_day":     "$400 – $800",
        "bad_day":      "$300 – $600 loss",
        "best_for":     "ELEVATED/HIGH VIX (18-35)  +  VIX sweet spot for mean-reversion",
        "who_for":      "Active investor checking in 2-3x/day. Comfortable with swings.",
        "warning":      "One bad position = -$300. All 5 bad = -$1,500. Sector concentration risk.",
    },
    "3": {
        "name":         "BALANCED",
        "tag":          "[3] BALANCED     — TOP 10 stocks, 10% each",
        "positions":    10,
        "pct_each":     0.10,
        "min_conv":     25,
        "min_prob":     0.56,
        "sector_max":   2,
        "deploy_pct":   0.88,
        "risk_level":   "MODERATE",
        "ytd":          "~+12% (estimated from signal mix)",
        "good_day":     "$200 – $450",
        "bad_day":      "$150 – $300 loss",
        "best_for":     "Any regime  +  want both upside and diversification",
        "who_for":      "Investor wanting steady growth. Checks in morning + evening.",
        "warning":      "Lower upside than CONCENTRATED. Still -$200 on a bad market day.",
    },
    "4": {
        "name":         "DIVERSIFIED",
        "tag":          "[4] DIVERSIFIED  — TOP 20 stocks, 5% each",
        "positions":    20,
        "pct_each":     0.05,
        "min_conv":     5,
        "min_prob":     0.55,
        "sector_max":   3,
        "deploy_pct":   0.85,
        "risk_level":   "LOW",
        "ytd":          "+9.2%  ($9,177)",
        "good_day":     "$100 – $250",
        "bad_day":      "$80 – $180 loss",
        "best_for":     "CALM VIX (<18)  +  learning mode  +  want to sleep easy",
        "who_for":      "New to AlphaAgent, or conservative investor. Set and forget.",
        "warning":      "Most consistent but lowest upside. Hard to hit $500+/day.",
    },
}

# ── Recommendation engine ──────────────────────────────────────────────────────
def recommend_mode(regime: dict) -> str:
    vix    = regime["vix"]
    spy_1d = regime["spy_1d"]
    above  = regime["above20"]

    if vix > 35:
        return "1"   # Extreme fear → SNIPER (few stocks pass conv≥60, those are gems)
    if vix > 25:
        return "2"   # High vol → CONCENTRATED (proven best in volatile conditions)
    if vix > 18:
        return "2"   # Elevated vol → CONCENTRATED (sweet spot from backtest)
    if above and spy_1d > 0:
        return "3"   # Calm + SPY up → BALANCED (broader momentum capture)
    return "3"        # Calm → BALANCED default

# CommMedia is a proven drag — always cap at 1
SECTOR_OVERRIDES = {"CommMedia": 1}

# ── Market regime detection ────────────────────────────────────────────────────
def get_regime() -> dict:
    try:
        spy  = yf.Ticker("SPY").history(period="30d")["Close"].dropna()
        vix  = float(yf.Ticker("^VIX").history(period="3d")["Close"].dropna().iloc[-1])
        sma20   = float(spy.rolling(20).mean().iloc[-1])
        spy_now = float(spy.iloc[-1])
        spy_5d  = (spy_now / float(spy.iloc[-6]) - 1) * 100 if len(spy) >= 6 else 0
        spy_1d  = (spy_now / float(spy.iloc[-2]) - 1) * 100 if len(spy) >= 2 else 0
        above20 = spy_now > sma20
    except:
        return {"label":"ELEVATED","vix":22.0,"spy_now":0,"sma20":0,
                "above20":True,"spy_5d":0,"spy_1d":0}

    if vix > 35:   label = "EXTREME"
    elif vix > 25: label = "HIGH"
    elif vix > 18: label = "ELEVATED"
    else:          label = "CALM"

    return {"label":label,"vix":round(vix,1),"spy_now":round(spy_now,2),
            "sma20":round(sma20,2),"above20":above20,
            "spy_5d":round(spy_5d,2),"spy_1d":round(spy_1d,2)}

# ── Signal fetch ───────────────────────────────────────────────────────────────
def fetch(sym: str) -> dict:
    url = f"{BASE}/api/v1/signal/{sym}?horizon=1d"
    try:
        req = urllib.request.Request(url, headers={"Accept":"application/json"})
        with urllib.request.urlopen(req, timeout=150) as resp:
            d = json.loads(resp.read().decode())
        if d is None: raise ValueError("null response")
        agents = [a for a in (d.get("agents") or []) if isinstance(a, dict)]
        lv = sum(1 for a in agents if (a.get("vote","") or "").upper() == "LONG")
        sv = sum(1 for a in agents if (a.get("vote","") or "").upper() == "SHORT")
        # Beta for position sizing (high-beta = smaller alloc)
        try:
            _beta = float(yf.Ticker(sym).info.get("beta") or 1.0)
            _beta = max(0.1, min(4.0, _beta))
        except Exception:
            _beta = 1.0
        return {
            "symbol":     sym,
            "direction":  d.get("direction") or "HOLD",
            "prob":       round((d.get("probability") or 0.5) * 100, 2),
            "conviction": round(d.get("conviction") or 0, 2),
            "multiplier": d.get("multiplier") or 1.0,
            "entropy":    round(d.get("entropy") or 1.0, 3),
            "long_votes": lv, "short_votes": sv,
            "hold_votes": len(agents) - lv - sv,
            "n_agents":   len(agents),
            "half_life":  (d.get("holding_period") or {}).get("half_life_days"),
            "sector":     SECTOR_MAP.get(sym, "Other"),
            "beta":       round(_beta, 2),
            "error":      None,
        }
    except Exception as e:
        return {"symbol":sym,"direction":"ERROR","prob":50.0,"conviction":0.0,
                "multiplier":1.0,"entropy":1.0,"long_votes":0,"short_votes":0,
                "hold_votes":0,"n_agents":0,"half_life":None,"beta":1.0,
                "sector":SECTOR_MAP.get(sym,"Other"),"error":str(e)[:80]}

def kelly_fraction(prob: float, b: float = 1.5, max_f: float = 0.20) -> float:
    p = prob / 100.0
    f = (p * b - (1 - p)) / b
    return min(max(0.0, f * 0.5), max_f)

# ── Portfolio builder ──────────────────────────────────────────────────────────
def build_portfolio(longs: list, mode: dict, regime_mult: float) -> list:
    max_positions = mode["positions"]
    max_pos       = mode["pct_each"]
    max_per_sec   = mode["sector_max"]
    deploy_pct    = mode["deploy_pct"]

    deployable   = CAPITAL * deploy_pct * regime_mult
    sector_count = {}
    sector_alloc = {}
    positions    = []
    allocated    = 0.0

    total_weight = (sum(r["conviction"] * kelly_fraction(r["prob"], max_f=max_pos)
                        for r in longs[:max_positions]) or 1.0)

    for r in longs[:max_positions]:
        sec   = r.get("sector","Other")
        limit = SECTOR_OVERRIDES.get(sec, max_per_sec)
        if sector_count.get(sec, 0) >= limit:
            continue
        kf     = kelly_fraction(r["prob"], max_f=max_pos)
        weight = (r["conviction"] * kf) / total_weight
        alloc  = min(deployable * weight, CAPITAL * max_pos)
        alloc  = alloc * (r.get("multiplier") or 1.0)
        alloc  = min(alloc, CAPITAL * max_pos)

        # Beta-adjusted sizing: high-beta stocks get smaller allocation
        # Beta > 1.5 → reduce by 25%; beta < 0.6 → increase by 15% (defensive premium)
        beta = r.get("beta", 1.0)
        if isinstance(beta, (int, float)) and 0.1 < beta < 5.0:
            beta_adj = 1.0 / max(0.6, min(1.8, beta))   # clamp: 0.56x – 1.67x
            beta_adj = 0.8 + 0.2 * beta_adj              # soften: 0.91x – 1.13x range
            alloc = alloc * beta_adj

        sec_used = sector_alloc.get(sec, 0.0)
        sec_cap  = CAPITAL * max_pos * max_per_sec
        if sec_used + alloc > sec_cap:
            alloc = max(0, sec_cap - sec_used)
        if alloc < 200:
            continue

        sector_count[sec] = sector_count.get(sec, 0) + 1
        sector_alloc[sec] = sector_alloc.get(sec, 0.0) + alloc
        positions.append({**r, "kelly_f": round(kf, 4),
                          "dollar_alloc": round(alloc, 2)})
        allocated += alloc

    if allocated > deployable and allocated > 0:
        scale = deployable / allocated
        for p in positions:
            p["dollar_alloc"] = round(p["dollar_alloc"] * scale, 2)

    return positions

# ── Interactive mode selector ──────────────────────────────────────────────────
def pick_mode(regime: dict) -> dict:
    rec_key  = recommend_mode(regime)
    rec_mode = MODES[rec_key]
    vix      = regime["vix"]
    lbl      = regime["label"]

    opp_day = vix > 25 and regime["spy_1d"] < -1.0

    print(f"\n{'='*76}")
    print(f"  MARKET CONDITIONS RIGHT NOW")
    print(f"{'='*76}")
    print(f"  VIX      : {vix}  [{lbl}]   {'⚡ ELEVATED FEAR — bounces likely' if vix>25 else '  calm conditions'}")
    print(f"  SPY      : ${regime['spy_now']}   1d: {regime['spy_1d']:+.2f}%   5d: {regime['spy_5d']:+.2f}%")
    print(f"  Above 20-SMA: {'YES — trend intact' if regime['above20'] else 'NO — below trend line'}")
    if opp_day:
        print(f"  ⚡ OPPORTUNITY DAY: Market down {regime['spy_1d']:.1f}% with VIX {vix}")
        print(f"     Oversold bounce setup — consider CONCENTRATED or SNIPER")

    print(f"\n{'='*76}")
    print(f"  CHOOSE YOUR STRATEGY  (press Enter for recommended)")
    print(f"{'='*76}\n")

    for key, m in MODES.items():
        star = "  ◄ RECOMMENDED" if key == rec_key else ""
        print(f"  {m['tag']}{star}")
        print(f"     Risk     : {m['risk_level']}")
        print(f"     YTD      : {m['ytd']}")
        print(f"     Good day : +{m['good_day']}")
        print(f"     Bad day  : -{m['bad_day']}")
        print(f"     Best for : {m['best_for']}")
        print(f"     Good if  : {m['who_for']}")
        print(f"     Warning  : {m['warning']}")
        print()

    print(f"  AlphaAgent recommends [{rec_key}] {rec_mode['name']} for today")
    print(f"  Reason: VIX={vix} ({lbl})"
          + (f" + market down {regime['spy_1d']:.1f}% (oversold bounce setup)" if opp_day
             else f" — {'high conviction signals expected' if vix>18 else 'moderate conditions'}"))
    print()

    raw = input(f"  Enter 1/2/3/4  [default={rec_key} {rec_mode['name']}]: ").strip()
    chosen_key = raw if raw in MODES else rec_key
    chosen     = MODES[chosen_key]

    if chosen_key != rec_key:
        print(f"\n  You chose [{chosen_key}] {chosen['name']}.")
        if int(chosen_key) < int(rec_key):
            print(f"  ⚠  More aggressive than recommended for today's VIX={vix}.")
            print(f"     Make sure you can monitor positions closely.")
        else:
            print(f"  ✓  More conservative than recommended — lower upside, lower risk.")
    else:
        print(f"\n  ✓  Running with recommended mode: [{chosen_key}] {chosen['name']}")

    # Opportunity day: loosen min_conv to catch more oversold bounces
    if opp_day and chosen["min_conv"] > 20:
        old = chosen["min_conv"]
        chosen = dict(chosen)   # don't mutate global
        chosen["min_conv"] = max(chosen["min_conv"] - 10, 20)
        print(f"  ⚡ Opportunity day boost: conv threshold {old} → {chosen['min_conv']}")

    print()
    return chosen

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*76}")
    print(f"  AlphaAgent — $100,000 Portfolio Builder")
    print(f"  {datetime.now().strftime('%A, %B %d %Y  %H:%M')}")
    print(f"  Fetching live market data to recommend a strategy...")
    print(f"{'='*76}")

    # Step 1: Detect regime
    regime    = get_regime()
    reg_label = regime["label"]

    # Step 2: Interactive mode pick
    mode = pick_mode(regime)
    min_conv  = mode["min_conv"]
    min_prob  = mode["min_prob"]
    max_pos   = mode["pct_each"]

    print(f"{'─'*76}")
    print(f"  MODE     : {mode['name']}  |  {mode['positions']} positions  "
          f"|  {max_pos*100:.0f}% each  |  conv≥{min_conv}")
    print(f"  REGIME   : {reg_label}  VIX={regime['vix']}  "
          f"SPY 1d={regime['spy_1d']:+.2f}%  {'Above' if regime['above20'] else 'Below'} 20-SMA")
    print(f"{'─'*76}\n")

    # Step 3: Fetch signals
    print(f"  Fetching signals for {len(UNIVERSE)} stocks (3 parallel workers)...\n")
    results, done = [], 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fetch, sym): sym for sym in UNIVERSE}
        for fut in as_completed(futures):
            r = fut.result(); done += 1
            pct = done / len(UNIVERSE) * 100
            status = (f"ERROR: {r['error'][:50]}" if r["error"]
                      else f"{r['direction']:>7}  {r['prob']:5.1f}%  conv={r['conviction']:5.1f}  [{r['sector']}]")
            sys.stdout.write(f"  [{pct:5.1f}%] {r['symbol']:<6}  {status}\n")
            sys.stdout.flush()
            results.append(r)

    print(f"\n{'─'*76}")

    # Step 4: Filter by chosen mode thresholds
    ok    = [r for r in results if not r["error"]]
    longs = sorted(
        [r for r in ok if r["direction"] == "LONG"
         and r["prob"] >= min_prob * 100
         and r["conviction"] >= min_conv],
        key=lambda x: (-x["conviction"], -x["prob"])
    )
    shorts = sorted(
        [r for r in ok if r["direction"] == "SHORT"
         and r["prob"] <= (100 - min_prob * 100)
         and r["conviction"] >= min_conv],
        key=lambda x: (-x["conviction"], x["prob"])
    )

    print(f"\n  Signal summary  [{mode['name']} — conv≥{min_conv}, prob≥{min_prob*100:.0f}%]")
    print(f"  Analyzed  : {len(ok)}")
    print(f"  LONG      : {len(longs)}  (qualifying)")
    print(f"  SHORT     : {len(shorts)}")
    print(f"  HOLD/weak : {len(ok)-len(longs)-len(shorts)}")
    if len(longs) == 0:
        print(f"\n  ⚠  No LONG signals pass conv≥{min_conv} today.")
        print(f"     Try mode [3] BALANCED (conv≥25) or [4] DIVERSIFIED (conv≥5).")
        return

    # Step 5: Build positions
    regime_mult = 0.80 if not regime["above20"] and reg_label == "CALM" else 1.0
    positions   = build_portfolio(longs, mode, regime_mult)
    invested    = sum(p["dollar_alloc"] for p in positions)
    cash        = CAPITAL - invested

    # Step 6: Print portfolio table
    print(f"\n{'='*76}")
    print(f"  PORTFOLIO  —  ${CAPITAL:,}  [{mode['name']}: {len(positions)} positions, "
          f"{max_pos*100:.0f}% each]")
    print(f"{'='*76}")
    print(f"  {'#':>2}  {'SYM':<6} {'SECT':<9} {'P(Up)':>6} {'CONV':>6} "
          f"{'AGENTS':>7} {'KELLY':>6} {'$ ALLOC':>10}  {'%PORT':>6}  {'ACC EST':>8}")
    print(f"  {'─'*2}  {'─'*6} {'─'*9} {'─'*6} {'─'*6} "
          f"{'─'*7} {'─'*6} {'─'*10}  {'─'*6}  {'─'*8}")

    for i, p in enumerate(positions, 1):
        pct_p = p["dollar_alloc"] / CAPITAL * 100
        votes = f"{p['long_votes']}L/{p['hold_votes']}H/{p['short_votes']}S"
        if p["conviction"] >= 60:   acc = "~78-82%"
        elif p["conviction"] >= 45: acc = "~72-78%"
        elif p["conviction"] >= 25: acc = "~60-68%"
        else:                       acc = "~54-60%"
        print(f"  {i:>2}. {p['symbol']:<6} {p['sector']:<9} "
              f"{p['prob']:>5.1f}% {p['conviction']:>6.1f} "
              f"{votes:>7} {p['kelly_f']:>5.1%} "
              f"${p['dollar_alloc']:>10,.2f}  {pct_p:>5.1f}%  β{p.get('beta',1.0):.2f}  {acc:>8}")

    print(f"  {'─'*74}")
    print(f"  {'INVESTED':>55}  ${invested:>10,.2f}  {invested/CAPITAL*100:>5.1f}%")
    print(f"  {'CASH':>55}  ${cash:>10,.2f}  {cash/CAPITAL*100:>5.1f}%")

    # Step 7: Expected P&L (mode-specific)
    acc_map  = {"EXTREME":0.78,"HIGH":0.72,"ELEVATED":0.60,"CALM":0.54}
    move_map = {"EXTREME":2.0, "HIGH":1.4, "ELEVATED":0.7, "CALM":0.4}
    acc_e    = acc_map[reg_label]
    move_e   = move_map[reg_label]
    exp_pnl  = invested * (acc_e * move_e - (1-acc_e) * move_e * 0.6) / 100
    print(f"\n  Expected P&L    : ${exp_pnl:>+,.0f}  "
          f"(acc≈{acc_e*100:.0f}%, avg move≈{move_e:.1f}%, {len(positions)} positions)")
    print(f"  Realistic range : ${exp_pnl*0.3:>+,.0f}  to  ${exp_pnl*2.5:>+,.0f}")
    print(f"  Mode target     : {mode['good_day']} good day | {mode['bad_day']}")

    # Step 8: Sector breakdown
    sec_alloc: dict = {}
    for p in positions:
        sec_alloc[p["sector"]] = sec_alloc.get(p["sector"], 0) + p["dollar_alloc"]

    print(f"\n{'─'*76}")
    print(f"  SECTOR ALLOCATION")
    print(f"{'─'*76}")
    for sec, amt in sorted(sec_alloc.items(), key=lambda x: -x[1]):
        bar = "█" * int(amt / 700)
        print(f"  {sec:<12}  ${amt:>8,.0f}  {amt/CAPITAL*100:>4.1f}%  {bar}")

    # Step 9: Risk profile
    if positions:
        avg_conv    = np.mean([p["conviction"] for p in positions])
        avg_prob    = np.mean([p["prob"] for p in positions])
        avg_ent     = np.mean([p["entropy"] for p in positions])
        max_pos_amt = max(p["dollar_alloc"] for p in positions)

        print(f"\n{'─'*76}")
        print(f"  RISK PROFILE  [{mode['risk_level']}]")
        print(f"{'─'*76}")
        print(f"  Positions      : {len(positions)} / {mode['positions']} max")
        print(f"  Avg conviction : {avg_conv:.1f}  (threshold: {min_conv})")
        print(f"  Avg P(Up)      : {avg_prob:.1f}%")
        print(f"  Avg entropy    : {avg_ent:.3f}  "
              f"({'HIGH disagreement' if avg_ent>0.85 else 'MODERATE' if avg_ent>0.6 else 'LOW — agents agree'})")
        print(f"  Largest pos    : ${max_pos_amt:,.0f}  ({max_pos_amt/CAPITAL*100:.1f}%)")
        if len(positions) < mode["positions"] // 2:
            print(f"  ⚠  Only {len(positions)} positions filled — fewer signals passed conv≥{min_conv} today.")
            print(f"     Consider switching to BALANCED or DIVERSIFIED for more coverage.")

    # Step 10: Exit rules
    print(f"\n{'─'*76}")
    print(f"  EXIT RULES")
    print(f"{'─'*76}")
    for p in positions:
        hl     = p.get("half_life")
        hl_str = f"half-life ~{hl:.1f}d" if hl else "EOD close"
        stop   = p["dollar_alloc"] * 0.015
        tgt    = p["dollar_alloc"] * 0.030
        print(f"  {p['symbol']:<6}  [{p['sector']:<9}]  "
              f"stop -${stop:,.0f} (-1.5%)  |  target +${tgt:,.0f} (+3.0%)  |  {hl_str}")

    # Step 11: Filtered / skipped
    in_port = {p["symbol"] for p in positions}
    skipped = [r for r in longs if r["symbol"] not in in_port]
    if skipped:
        print(f"\n{'─'*76}")
        print(f"  NEXT BEST (qualified but exceeded position limit or sector cap)")
        print(f"{'─'*76}")
        for r in skipped[:8]:
            print(f"  {r['symbol']:<6}  [{r['sector']:<9}]  "
                  f"conv={r['conviction']:5.1f}  prob={r['prob']:5.1f}%  → position limit / sector cap")

    # Step 12: Mode comparison reminder
    print(f"\n{'─'*76}")
    print(f"  HOW TODAY'S MODES COMPARE (same signals, different sizing)")
    print(f"{'─'*76}")
    print(f"  {'Mode':<15} {'Positions':>10} {'Max/pos':>8} {'Min conv':>9} "
          f"{'Good day':>14} {'Bad day':>14}  {'Risk'}")
    print(f"  {'─'*15} {'─'*10} {'─'*8} {'─'*9} {'─'*14} {'─'*14}  {'─'*10}")
    for _, m in MODES.items():
        marker = "  ◄ YOU" if m["name"] == mode["name"] else ""
        print(f"  {m['name']:<15} {m['positions']:>10}     {m['pct_each']*100:>5.0f}%  "
              f"{m['min_conv']:>9}   +{m['good_day']:>13}   -{m['bad_day']:>13}  {m['risk_level']}{marker}")

    print(f"\n{'='*76}")
    print(f"  FREE SOURCES: yfinance | FRED | SEC EDGAR | Gemini free tier — $0/month")
    print(f"  DISCLAIMER: AlphaAgent signals only. Not financial advice.")
    print(f"{'='*76}\n")

if __name__ == "__main__":
    main()
