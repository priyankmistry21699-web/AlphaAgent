"""
AlphaAgent — Comprehensive cross-asset signal tester.
Tests every asset class across all 6 horizons, reports direction/probability/conviction.
Uses ThreadPoolExecutor for parallel HTTP requests.
"""

import json, time, csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import urllib.request, urllib.error

BASE = "http://localhost:8000"
HORIZONS = ["1d", "1w", "1m", "3m", "6m", "1y"]

# ── Asset universe ──────────────────────────────────────────────────────────
ASSETS = {
    # US Large-Cap Stocks
    "US_LARGE": ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META", "TSLA",
                 "BRK-B", "JPM", "JNJ", "XOM", "V", "UNH", "WMT", "HD"],

    # US Mid / Small Cap
    "US_MID_SMALL": ["PLTR", "RKLB", "SMCI", "COIN", "RIVN", "MSTR"],

    # International Stocks (US-listed ADRs / cross-listed)
    "INTL_STOCKS": [
        "TM",     # Toyota (Japan)
        "TSM",    # TSMC (Taiwan)
        "BABA",   # Alibaba (China)
        "BIDU",   # Baidu (China)
        "SAP",    # SAP (Germany)
        "ASML",   # ASML (Netherlands)
        "NVO",    # Novo Nordisk (Denmark)
        "AZN",    # AstraZeneca (UK/Sweden)
        "INFY",   # Infosys (India)
        "HDB",    # HDFC Bank (India)
        "VALE",   # Vale (Brazil)
        "PBR",    # Petrobras (Brazil)
        "RY",     # Royal Bank of Canada
        "CNQ",    # Canadian Natural Resources
        "BHP",    # BHP (Australia)
        "SHOP",   # Shopify (Canada)
    ],

    # US Sector ETFs
    "ETF_SECTOR": ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE"],

    # Broad Market ETFs
    "ETF_BROAD": ["SPY", "QQQ", "IWM", "DIA", "VTI", "VOO"],

    # International / Country ETFs
    "ETF_INTL": ["EEM", "EFA", "VEA", "VWO",
                 "EWJ",   # Japan
                 "EWZ",   # Brazil
                 "FXI",   # China
                 "EWG",   # Germany
                 "EWC",   # Canada
                 "EWA",   # Australia
                 "INDA",  # India
                 "EWU",   # United Kingdom
                 "EWY",   # South Korea
                 "EWT",   # Taiwan
                 "EWS",   # Singapore
                 "EWI",   # Italy
                 "EWP",   # Spain
                 "EWQ",   # France
                 "EWN",   # Netherlands
                 "EWL",   # Switzerland
    ],

    # Bond ETFs
    "ETF_BONDS": ["TLT", "IEF", "SHY", "AGG", "BND", "HYG", "LQD", "TIP", "MUB", "VCSH"],

    # Commodity ETFs
    "ETF_COMMODITY": ["GLD", "SLV", "IAU", "PDBC", "DJP", "COPX", "USO", "UNG", "CORN", "WEAT"],

    # Forex Pairs (yfinance format)
    "FOREX": [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X",
        "NZDUSD=X", "USDCHF=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X",
        "USDCNY=X", "USDINR=X", "USDBRL=X", "USDMXN=X", "USDZAR=X",
        "USDSGD=X", "USDKRW=X", "USDTRY=X", "USDHKD=X", "USDTHB=X",
    ],

    # Metals (Futures)
    "METALS": ["GC=F", "SI=F", "HG=F", "PL=F", "PA=F"],

    # Energy Futures
    "ENERGY": ["CL=F", "BZ=F", "NG=F", "RB=F", "HO=F"],

    # Agricultural Futures
    "AGRI": ["ZC=F", "ZW=F", "ZS=F", "ZO=F", "KC=F", "SB=F", "CT=F", "CC=F"],

    # Crypto
    "CRYPTO": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
               "ADA-USD", "DOGE-USD", "AVAX-USD", "MATIC-USD", "LINK-USD"],

    # Market Indices
    "INDICES": ["^GSPC", "^IXIC", "^DJI", "^RUT", "^VIX", "^TNX",
                "^FTSE", "^GDAXI", "^FCHI", "^N225", "^HSI", "^AXJO",
                "^BSESN", "^STOXX50E"],

    # Mutual Funds (US, yfinance accessible)
    "MUTUAL_FUNDS": ["VFIAX", "FXAIX", "FCNTX", "VBTLX", "VGTSX",
                     "FSKAX", "VWELX", "PRGFX", "AGTHX", "DODGX"],

    # REITs
    "REITS": ["VNQ", "O", "AMT", "PLD", "EQIX", "SPG", "PSA", "DLR", "WELL", "AVB"],
}

ALL_SYMBOLS = [(cat, sym) for cat, syms in ASSETS.items() for sym in syms]
TOTAL = len(ALL_SYMBOLS) * len(HORIZONS)


def fetch_signal(cat: str, sym: str, horizon: str) -> dict:
    url = f"{BASE}/api/v1/signal/{sym}?horizon={horizon}"
    try:
        start = time.time()
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode())
        elapsed = time.time() - start
        return {
            "category":  cat,
            "symbol":    sym,
            "horizon":   horizon,
            "direction": data.get("direction", "?"),
            "prob":      round(data.get("probability", 0.5) * 100, 1),
            "base_prob": round(data.get("base_probability", data.get("probability", 0.5)) * 100, 1),
            "conviction":round(data.get("conviction", 0), 1),
            "entropy":   round(data.get("entropy", 1.0), 3),
            "agents":    len(data.get("agents", [])),
            "cached":    data.get("cached", False),
            "elapsed_s": round(elapsed, 1),
            "error":     None,
        }
    except Exception as e:
        return {
            "category": cat, "symbol": sym, "horizon": horizon,
            "direction": "ERROR", "prob": 50.0, "base_prob": 50.0,
            "conviction": 0.0, "entropy": 1.0, "agents": 0,
            "cached": False, "elapsed_s": 0.0,
            "error": str(e)[:120],
        }


def main():
    print(f"\n{'='*80}")
    print(f"AlphaAgent Cross-Asset Signal Test — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Testing {len(ALL_SYMBOLS)} symbols × {len(HORIZONS)} horizons = {TOTAL} requests")
    print(f"Parallelism: 10 concurrent | Timeout: 90s per request")
    print(f"{'='*80}\n")

    tasks = [(cat, sym, h) for (cat, sym) in ALL_SYMBOLS for h in HORIZONS]
    results = []
    done = 0

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(fetch_signal, cat, sym, h): (cat, sym, h)
                   for (cat, sym, h) in tasks}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            done += 1
            if r["error"]:
                status = f"ERROR: {r['error'][:55]}"
            else:
                status = f"{r['direction']:7s} {r['prob']:5.1f}%  conv={r['conviction']:4.1f}  ent={r['entropy']:.3f}"
            pct = done / TOTAL * 100
            print(f"[{pct:5.1f}%] {r['symbol']:15s} {r['horizon']:3s}  {status}  ({r['elapsed_s']}s)")

    # ── Sort by category, symbol, horizon order ─────────────────────────────
    H_ORDER = {"1d": 0, "1w": 1, "1m": 2, "3m": 3, "6m": 4, "1y": 5}
    results.sort(key=lambda r: (r["category"], r["symbol"], H_ORDER.get(r["horizon"], 9)))

    # ── Write CSV ─────────────────────────────────────────────────────────────
    csv_path = "signal_test_results.csv"
    fieldnames = ["category","symbol","horizon","direction","prob","base_prob",
                  "conviction","entropy","agents","cached","elapsed_s","error"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"\nResults saved → {csv_path}")

    # ── Summary analysis ──────────────────────────────────────────────────────
    ok    = [r for r in results if not r["error"]]
    longs  = [r for r in ok if r["direction"] == "LONG"]
    shorts = [r for r in ok if r["direction"] == "SHORT"]
    holds  = [r for r in ok if r["direction"] in ("HOLD","NEUTRAL")]
    errs   = [r for r in results if r["error"]]

    print(f"\n{'='*80}")
    print(f"SUMMARY  ({len(ok)} OK / {len(errs)} ERRORS / {TOTAL} total)")
    print(f"{'='*80}")
    print(f"  LONG   : {len(longs):4d}  ({len(longs)/max(len(ok),1)*100:.1f}%)")
    print(f"  SHORT  : {len(shorts):4d}  ({len(shorts)/max(len(ok),1)*100:.1f}%)")
    print(f"  HOLD   : {len(holds):4d}  ({len(holds)/max(len(ok),1)*100:.1f}%)")
    print(f"  ERROR  : {len(errs):4d}")

    # Per-category breakdown
    print(f"\n{'─'*80}")
    print(f"{'CATEGORY':<18} {'LONG':>5} {'SHORT':>6} {'HOLD':>5} {'ERR':>5} {'AVG%':>7} {'AVG_CONV':>9}")
    print(f"{'─'*80}")
    for cat in ASSETS:
        cat_r = [r for r in ok   if r["category"] == cat]
        cat_e = [r for r in errs if r["category"] == cat]
        cat_l = sum(1 for r in cat_r if r["direction"] == "LONG")
        cat_s = sum(1 for r in cat_r if r["direction"] == "SHORT")
        cat_h = sum(1 for r in cat_r if r["direction"] in ("HOLD","NEUTRAL"))
        avg_p = sum(r["prob"] for r in cat_r) / max(len(cat_r), 1)
        avg_c = sum(r["conviction"] for r in cat_r) / max(len(cat_r), 1)
        print(f"{cat:<18} {cat_l:>5} {cat_s:>6} {cat_h:>5} {len(cat_e):>5} {avg_p:>6.1f}%  {avg_c:>8.1f}")

    # Per-horizon breakdown
    print(f"\n{'─'*80}")
    print(f"{'HORIZON':<8} {'LONG':>5} {'SHORT':>6} {'HOLD':>5} {'AVG%':>7} {'AVG_CONV':>9}")
    print(f"{'─'*80}")
    for h in HORIZONS:
        h_r = [r for r in ok if r["horizon"] == h]
        h_l = sum(1 for r in h_r if r["direction"] == "LONG")
        h_s = sum(1 for r in h_r if r["direction"] == "SHORT")
        h_h = sum(1 for r in h_r if r["direction"] in ("HOLD","NEUTRAL"))
        avg_p = sum(r["prob"] for r in h_r) / max(len(h_r), 1)
        avg_c = sum(r["conviction"] for r in h_r) / max(len(h_r), 1)
        print(f"{h:<8} {h_l:>5} {h_s:>6} {h_h:>5} {avg_p:>6.1f}%  {avg_c:>8.1f}")

    # High conviction calls
    high_conv = sorted([r for r in ok if r["conviction"] >= 20], key=lambda r: -r["conviction"])
    if high_conv:
        print(f"\n{'─'*80}")
        print(f"TOP HIGH-CONVICTION CALLS (conviction ≥ 20):")
        print(f"{'─'*80}")
        for r in high_conv[:40]:
            print(f"  {r['symbol']:15s} {r['horizon']:3s}  {r['direction']:7s}  {r['prob']:5.1f}%  conv={r['conviction']:5.1f}")

    # Stuck at exactly 50%
    stuck = [r for r in ok if r["prob"] == 50.0]
    if stuck:
        print(f"\n{'─'*80}")
        print(f"STUCK AT 50.0% ({len(stuck)} results) — may indicate agent failures:")
        for r in stuck[:20]:
            print(f"  {r['symbol']:15s} {r['horizon']:3s}  agents={r['agents']}")

    # Error details
    if errs:
        print(f"\n{'─'*80}")
        print(f"ERRORS ({len(errs)}):")
        seen = set()
        for r in errs:
            key = (r["symbol"], r["error"][:40])
            if key not in seen:
                seen.add(key)
                print(f"  {r['symbol']:15s}  {r['error']}")

    print(f"\n{'='*80}")
    print("Done. Full results in signal_test_results.csv")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
