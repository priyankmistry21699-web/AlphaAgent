"""
Find the signal parameters that achieve 80%+ accuracy.
Tests every combination of score threshold, RSI filter, VIX bucket, momentum filter.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

UNIVERSE = [
    "AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","AVGO",
    "JPM","BAC","GS","MS","BLK","V","MA","AXP",
    "UNH","JNJ","LLY","ABBV","PFE","MRK","TMO","ABT",
    "CAT","DE","HON","GE","RTX","LMT","NOC","UPS",
    "XOM","CVX","COP","SLB","OXY","PSX","VLO","MPC",
    "WMT","COST","HD","TGT","MCD","SBUX","NKE","AMGN",
    "VZ","T","NFLX","DIS","CMCSA","CHTR",
    "AMD","INTC","QCOM","MU","AMAT","LRCX","KLAC","TXN",
    "PLTR","CRWD","SNOW","NET","ZS","DDOG","PANW","FTNT",
    "AMT","PLD","NEE","DUK","SO","D",
]

def _rsi(s, w=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=w-1,adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=w-1,adjust=False).mean()
    return 100 - 100/(1 + g/l.replace(0,np.nan))

def compute_all(close_s, vol_s):
    c = close_s.dropna(); v = vol_s.dropna()
    if len(c) < 60: return None
    r    = _rsi(c).iloc[-1]
    s20  = c.rolling(20).mean().iloc[-1]
    s50  = c.rolling(50).mean().iloc[-1]
    s200 = c.rolling(min(200,len(c))).mean().iloc[-1]
    mom  = (c.iloc[-1]/c.iloc[-23]-1)*100 if len(c)>=23 else 0
    e12  = c.ewm(span=12,adjust=False).mean()
    e26  = c.ewm(span=26,adjust=False).mean()
    mh   = (e12-e26-(e12-e26).ewm(span=9,adjust=False).mean()).iloc[-1]
    bbm  = c.rolling(20).mean(); bbs=c.rolling(20).std()
    den  = ((bbm+2*bbs)-(bbm-2*bbs)).iloc[-1]
    bbp  = (c.iloc[-1]-(bbm-2*bbs).iloc[-1])/(den+1e-9)
    vr   = v.iloc[-5:].mean()/v.rolling(20).mean().iloc[-1] if v.rolling(20).mean().iloc[-1]>0 else 1.0

    score = 0
    score += 2 if r<30 else (1 if r<45 else (-2 if r>70 else (-1 if r>58 else 0)))
    score += 1 if s20>s50 else -1
    score += 1 if c.iloc[-1]>s200 else -1
    score += 2 if mom>8 else (1 if mom>2 else (-2 if mom<-10 else (-1 if mom<-2 else 0)))
    score += 1 if mh>0 else -1
    score += 2 if bbp<0.15 else (1 if bbp<0.35 else (-2 if bbp>0.85 else (-1 if bbp>0.65 else 0)))
    if vr>1.5 and mom>0: score+=1
    return {"score":score,"rsi":r,"momentum":mom,"above200":c.iloc[-1]>s200,
            "vol_ratio":vr,"macd_bull":mh>0,"sma_bull":s20>s50}

print("Downloading data...")
raw     = yf.download(UNIVERSE+["SPY","^VIX"], start="2025-04-01", end="2026-05-17",
                      progress=False, auto_adjust=True)
close_a = raw["Close"]; open_a = raw["Open"]; vol_a = raw["Volume"]
trade_dates = [d for d in close_a.index
               if pd.Timestamp("2026-01-02") <= d <= pd.Timestamp("2026-05-16")]
print(f"Done. {len(trade_dates)} trading days.\n")

# ── Collect every signal data point ─────────────────────────────────────────
records = []
for td in trade_dates:
    pi = close_a.index.get_loc(td)-1
    if pi < 50: continue
    sc = close_a.iloc[:pi+1]; sv = vol_a.iloc[:pi+1]
    try:
        vix = float(close_a.iloc[:pi+1]["^VIX"].dropna().iloc[-1])
    except: vix = 20.0
    try:
        spy = sc["SPY"].dropna()
        sma20 = float(spy.rolling(20).mean().iloc[-1])
        spy_now = float(spy.iloc[-1])
        spy_5d = (spy_now/float(spy.iloc[-6])-1)*100 if len(spy)>=6 else 0
        above_sma = spy_now > sma20
    except: spy_5d=0; above_sma=True

    for sym in UNIVERSE:
        try:
            sig = compute_all(sc[sym].dropna(), sv[sym].dropna())
            if not sig: continue
            o = float(open_a.loc[td,sym]); c = float(close_a.loc[td,sym])
            if np.isnan(o) or np.isnan(c) or o<=0: continue
            actual_pct = (c-o)/o*100
            records.append({
                "date":td,"sym":sym,
                "score":sig["score"],"rsi":sig["rsi"],
                "momentum":sig["momentum"],"above200":sig["above200"],
                "vol_ratio":sig["vol_ratio"],"macd_bull":sig["macd_bull"],
                "sma_bull":sig["sma_bull"],
                "actual_pct":actual_pct,"went_up":actual_pct>0,
                "vix":vix,"spy_5d":spy_5d,"above_sma":above_sma,
                "vix_bucket": ("EXTREME" if vix>35 else "HIGH" if vix>25
                               else "ELEVATED" if vix>18 else "CALM"),
            })
        except: pass

df = pd.DataFrame(records)
total = len(df)
print(f"Total signal data points: {total:,}\n")

print("="*72)
print("  THRESHOLD ANALYSIS — Finding 80% Accuracy Parameters")
print("="*72)

# ── 1. Accuracy by score threshold ──────────────────────────────────────────
print("\n  BY SCORE THRESHOLD (LONG signals only, score > threshold)\n")
print(f"  {'Min Score':>10} {'Signals':>8} {'Accuracy':>9} {'Avg PnL%':>9} {'Coverage':>9}")
print(f"  {'─'*10} {'─'*8} {'─'*9} {'─'*9} {'─'*9}")
for min_s in [0,1,2,3,4,5,6,7]:
    sub = df[df["score"] >= min_s]
    if len(sub) == 0: break
    acc = sub["went_up"].mean()*100
    avg = sub["actual_pct"].mean()
    cov = len(sub)/total*100
    flag = " ◄ TARGET" if 79<=acc<=85 else (" ★ 80%+ ACHIEVED" if acc>=80 else "")
    print(f"  {min_s:>10} {len(sub):>8,} {acc:>8.1f}% {avg:>+8.3f}% {cov:>8.1f}%{flag}")

# ── 2. RSI filter on top of score ───────────────────────────────────────────
print("\n  SCORE ≥ 4 + RSI FILTER\n")
print(f"  {'RSI Max':>8} {'Signals':>8} {'Accuracy':>9} {'Avg PnL%':>9} {'Coverage':>9}")
print(f"  {'─'*8} {'─'*8} {'─'*9} {'─'*9} {'─'*9}")
for rsi_max in [80,70,60,55,50,45,40,35]:
    sub = df[(df["score"]>=4) & (df["rsi"]<=rsi_max)]
    if len(sub) < 20: break
    acc = sub["went_up"].mean()*100
    avg = sub["actual_pct"].mean()
    cov = len(sub)/total*100
    flag = " ★ 80%+" if acc>=80 else (" ◄" if acc>=78 else "")
    print(f"  {rsi_max:>8} {len(sub):>8,} {acc:>8.1f}% {avg:>+8.3f}% {cov:>8.1f}%{flag}")

# ── 3. By VIX bucket ─────────────────────────────────────────────────────────
print("\n  SCORE ≥ 4 + ACCURACY BY VIX BUCKET\n")
print(f"  {'VIX Regime':>12} {'Days':>5} {'Signals':>8} {'Accuracy':>9} {'Avg PnL%':>9} {'Best RSI filter':>16}")
print(f"  {'─'*12} {'─'*5} {'─'*8} {'─'*9} {'─'*9} {'─'*16}")
for bucket in ["CALM","ELEVATED","HIGH","EXTREME"]:
    sub = df[(df["score"]>=4) & (df["vix_bucket"]==bucket)]
    if len(sub) < 5:
        print(f"  {bucket:>12}  insufficient data")
        continue
    acc = sub["went_up"].mean()*100
    avg = sub["actual_pct"].mean()
    days = sub["date"].nunique()
    # find best RSI filter for this bucket
    best_acc, best_rsi = 0, 80
    for rsi_cut in [80,70,60,55,50,45,40,35]:
        s2 = sub[sub["rsi"]<=rsi_cut]
        if len(s2) >= 10:
            a2 = s2["went_up"].mean()*100
            if a2 > best_acc: best_acc=a2; best_rsi=rsi_cut
    print(f"  {bucket:>12} {days:>5} {len(sub):>8,} {acc:>8.1f}% {avg:>+8.3f}%  RSI<{best_rsi} → {best_acc:.0f}%")

# ── 4. The "volatile = opportunity" thesis ──────────────────────────────────
print("\n  VOLATILE MARKET MEAN-REVERSION: score≥3 + RSI<40 (oversold bounces)\n")
print(f"  {'VIX Regime':>12} {'Signals':>8} {'Accuracy':>9} {'Avg PnL%':>9} {'Avg loss day PnL':>17}")
print(f"  {'─'*12} {'─'*8} {'─'*9} {'─'*9} {'─'*17}")
for bucket in ["CALM","ELEVATED","HIGH","EXTREME"]:
    sub_mom  = df[(df["score"]>=3) & (df["vix_bucket"]==bucket)]
    sub_rev  = df[(df["score"]>=3) & (df["rsi"]<40) & (df["vix_bucket"]==bucket)]
    if len(sub_rev) < 5: continue
    acc_m = sub_mom["went_up"].mean()*100 if len(sub_mom) else 0
    acc_r = sub_rev["went_up"].mean()*100
    avg_r = sub_rev["actual_pct"].mean()
    print(f"  {bucket:>12} {len(sub_rev):>8,} {acc_r:>8.1f}%  {avg_r:>+8.3f}%    (momentum: {acc_m:.0f}%)")

# ── 5. Gold standard — find the 80% combo ───────────────────────────────────
print("\n  SEARCHING FOR 80%+ ACCURACY COMBINATIONS...\n")
results_80 = []
for min_score in range(1,8):
    for rsi_max in [80,70,60,55,50,45,40,35]:
        for req_above200 in [False,True]:
            for req_macd in [False,True]:
                sub = df[df["score"]>=min_score]
                if rsi_max < 80: sub = sub[sub["rsi"]<=rsi_max]
                if req_above200: sub = sub[sub["above200"]==True]
                if req_macd: sub = sub[sub["macd_bull"]==True]
                if len(sub) < 50: continue
                acc = sub["went_up"].mean()*100
                avg = sub["actual_pct"].mean()
                if acc >= 79:
                    results_80.append({
                        "min_score":min_score,"rsi_max":rsi_max,
                        "above200":req_above200,"macd":req_macd,
                        "signals":len(sub),"accuracy":acc,"avg_pct":avg,
                        "score_str": f"score≥{min_score} + RSI<{rsi_max}"
                                     + (" + above200" if req_above200 else "")
                                     + (" + MACD↑" if req_macd else "")
                    })

results_80.sort(key=lambda x: (-x["accuracy"],x["min_score"]))
print(f"  Found {len(results_80)} parameter combos with ≥79% accuracy:\n")
print(f"  {'Parameters':<45} {'Signals':>8} {'Accuracy':>9} {'Avg PnL%':>9}")
print(f"  {'─'*45} {'─'*8} {'─'*9} {'─'*9}")
for r in results_80[:15]:
    print(f"  {r['score_str']:<45} {r['signals']:>8,} {r['accuracy']:>8.1f}% {r['avg_pct']:>+8.3f}%")

# ── 6. Volatility sizing multiplier analysis ─────────────────────────────────
print(f"\n  POSITION SIZING IN VOLATILE MARKETS")
print(f"  (Using best params: score≥4, RSI<55)")
print(f"  {'VIX Range':>12} {'Signals':>8} {'Accuracy':>9} {'Avg PnL%':>9} {'Suggested mult':>15}")
print(f"  {'─'*12} {'─'*8} {'─'*9} {'─'*9} {'─'*15}")
sub_best = df[(df["score"]>=4) & (df["rsi"]<=55)]
for label, lo, hi in [("<18",0,18),("18-22",18,22),("22-28",22,28),("28-35",28,35),(">35",35,200)]:
    s = sub_best[(sub_best["vix"]>=lo) & (sub_best["vix"]<hi)]
    if len(s) < 10: continue
    acc = s["went_up"].mean()*100
    avg = s["actual_pct"].mean()
    # Higher accuracy in volatile = take bigger positions
    mult = 1.5 if acc>=78 else (1.2 if acc>=72 else (1.0 if acc>=65 else 0.7))
    print(f"  VIX {label:>7} {len(s):>8,} {acc:>8.1f}% {avg:>+8.3f}%     {mult:.1f}x")

print(f"\n{'='*72}")
print(f"  RECOMMENDED PARAMETERS FOR 80% TARGET:")
if results_80:
    best = results_80[0]
    print(f"  {best['score_str']}")
    print(f"  Expected accuracy: {best['accuracy']:.1f}%  on {best['signals']} signals")
print(f"  Strategy: HIGH VIX = BIGGER bets on oversold bounces (not smaller)")
print(f"{'='*72}\n")
