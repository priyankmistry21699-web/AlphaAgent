"""
AlphaAgent — 2026 YTD Daily Backtest v3
Volatility-Adaptive Strategy: HIGH VIX = BIGGER bets, STRICTER signals
Jan 2, 2026 → May 16, 2026

Regime thresholds (match portfolio_100k.py exactly):
  CALM     VIX<18  : score≥1 (conv≥10), top-15, 1.0x size, max 10% per pos
  ELEVATED VIX18-25: score≥2 (conv≥20), top-12, 1.2x size, max 12% per pos
  HIGH     VIX25-35: score≥4 (conv≥40), top-8,  1.5x size, max 15% per pos
  EXTREME  VIX>35  : score≥5 (conv≥50), top-5,  2.0x size, max 20% per pos

PLUS: Opportunity Day boost — if VIX>25 AND SPY down >1% → size_mult × 1.25

All free data (yfinance only for backtest).
"""
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

CAPITAL  = 100_000

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
# Sector caps (CommMedia = max 1)
SECTOR_LIMITS_BY_REGIME = {
    "CALM":     {"CommMedia":1, "default":3},
    "ELEVATED": {"CommMedia":1, "default":3},
    "HIGH":     {"CommMedia":1, "default":2},
    "EXTREME":  {"CommMedia":1, "default":1},
}
REGIMES = {
    # label: (min_score, max_pos_pct, max_positions, size_mult)
    "CALM":     (1, 0.10, 15, 1.0),
    "ELEVATED": (2, 0.12, 12, 1.2),
    "HIGH":     (4, 0.15,  8, 1.5),
    "EXTREME":  (5, 0.20,  5, 2.0),
}

def _rsi(s, w=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=w-1,adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=w-1,adjust=False).mean()
    return 100 - 100/(1 + g/l.replace(0,np.nan))

def compute_signal(close_s, vol_s):
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
    prob = min(75, max(25, 50+(score/10)*25))
    return {"score":score,"prob":prob,"conviction":min(100,abs(score)*10),
            "direction":"LONG" if score>0 else ("SHORT" if score<0 else "NEUTRAL"),
            "rsi":r,"momentum":mom,"above200":c.iloc[-1]>s200}

def kelly_frac(p, max_f=0.10, b=1.5):
    f=(p/100*b-(1-p/100))/b
    return min(max(0.0,f*0.5), max_f)

# ── Download ─────────────────────────────────────────────────────────────────
print("="*76)
print("  AlphaAgent — 2026 YTD Backtest v3  (Volatility-Adaptive)")
print("  HIGH VIX = BIGGER BETS on STRICTER signals")
print("="*76)
print("\nDownloading price history...")
raw = yf.download(UNIVERSE+["SPY","^VIX"], start="2025-04-01", end="2026-05-17",
                  progress=True, auto_adjust=True)
print("Done.\n")

close_all = raw["Close"]; open_all = raw["Open"]; vol_all = raw["Volume"]
trade_dates = [d for d in close_all.index
               if pd.Timestamp("2026-01-02") <= d <= pd.Timestamp("2026-05-16")]
print(f"Trading days: {len(trade_dates)}\n")

# v1 reference P&L (technical-only, no fixes)
v1_pnl = [
    664,1014,514,-672,446,1025,251,-691,677,-750,-100,-388,1054,-475,-237,
    -232,215,-191,-58,64,1660,21,-1261,250,2357,571,-131,233,-1395,505,
    -315,256,-87,676,241,385,-377,-260,1117,1059,386,162,-1235,245,1763,
    575,917,-171,-507,-178,-251,-670,90,-1287,-411,512,175,255,-913,-2030,
    1027,-864,-440,251,395,828,-1489,-213,-473,-326,281,581,814,-108,-130,
    -98,290,209,-24,-730,258,1365,207,-123,827,792,55,245,757,-422,429,274,-330
]

# ── Main loop ─────────────────────────────────────────────────────────────────
print(f"  {'DATE':<12} {'REG':<9} {'VIX':>5} {'N':>3} {'SIZE':>5} {'INVESTED':>10} "
      f"{'P&L v3':>10} {'P&L v1':>10} {'DELTA':>9} {'SPY%':>7} {'W/L':>6}")
print(f"  {'─'*12} {'─'*9} {'─'*5} {'─'*3} {'─'*5} {'─'*10} "
      f"{'─'*10} {'─'*10} {'─'*9} {'─'*7} {'─'*6}")

daily_results = []
monthly: dict = {}

for idx, td in enumerate(trade_dates):
    pi = close_all.index.get_loc(td)-1
    if pi < 50: continue
    sc = close_all.iloc[:pi+1]; oc = open_all.iloc[:pi+1]; sv = vol_all.iloc[:pi+1]

    # ── Regime detection ─────────────────────────────────────────────────────
    try:
        spy_s   = sc["SPY"].dropna()
        vix_s   = sc["^VIX"].dropna()
        vix     = float(vix_s.iloc[-1])
        spy_now = float(spy_s.iloc[-1])
        sma20   = float(spy_s.rolling(20).mean().iloc[-1])
        spy_1d  = (spy_now/float(spy_s.iloc[-2])-1)*100 if len(spy_s)>=2 else 0
    except: vix=20; spy_now=sma20=0; spy_1d=0

    if vix > 35:    reg = "EXTREME"
    elif vix > 25:  reg = "HIGH"
    elif vix > 18:  reg = "ELEVATED"
    else:           reg = "CALM"

    min_score, max_pos_pct, max_positions, size_mult = REGIMES[reg]
    sec_limits = SECTOR_LIMITS_BY_REGIME[reg]

    # Opportunity day boost: VIX>25 AND market down → more aggressive
    opp_day = vix > 25 and spy_1d < -1.0
    if opp_day:
        size_mult = min(size_mult * 1.25, 2.5)
        min_score = max(min_score - 1, 1)

    # ── Signals ──────────────────────────────────────────────────────────────
    signals = {}
    for sym in UNIVERSE:
        try:
            s = compute_signal(sc[sym].dropna(), sv[sym].dropna())
            if s and s["score"] >= min_score and s["direction"] == "LONG":
                signals[sym] = s
        except: pass

    prices = {}
    for sym in UNIVERSE:
        try:
            o=float(open_all.loc[td,sym]); c=float(close_all.loc[td,sym])
            if not(np.isnan(o) or np.isnan(c)) and o>0:
                prices[sym]={"open":o,"close":c,"pct":(c-o)/o*100}
        except: pass

    # ── Build candidates ─────────────────────────────────────────────────────
    candidates = sorted(
        [{"sym":s,**g,"sector":SECTOR_MAP.get(s,"Other")}
         for s,g in signals.items() if s in prices],
        key=lambda x: (-x["conviction"],-x["prob"])
    )

    # ── Sector cap & position cap ─────────────────────────────────────────────
    sec_count = {}; longs_final = []
    for r in candidates:
        sec   = r["sector"]
        limit = sec_limits.get(sec, sec_limits["default"])
        if sec_count.get(sec,0) >= limit: continue
        sec_count[sec] = sec_count.get(sec,0)+1
        longs_final.append(r)
        if len(longs_final) >= max_positions: break

    # ── Sizing ────────────────────────────────────────────────────────────────
    deployable   = CAPITAL * 0.90
    tw           = sum(s["conviction"]*kelly_frac(s["prob"],max_pos_pct) for s in longs_final) or 1.0
    positions    = []; allocated=0; sec_alloc={}

    for s in longs_final:
        kf   = kelly_frac(s["prob"], max_pos_pct)
        w    = s["conviction"]*kf/tw
        a    = min(deployable*w, CAPITAL*max_pos_pct) * size_mult
        a    = min(a, CAPITAL*max_pos_pct*size_mult)   # hard cap per position
        sec  = s["sector"]
        if sec_alloc.get(sec,0)+a > CAPITAL*0.35*size_mult:
            a = max(0, CAPITAL*0.35*size_mult - sec_alloc.get(sec,0))
        if a < 200: continue
        sec_alloc[sec] = sec_alloc.get(sec,0)+a
        positions.append({**s,"alloc":a,"px":prices[s["sym"]]})
        allocated += a

    if allocated > deployable*size_mult and allocated>0:
        scale = deployable*size_mult/allocated
        for p in positions: p["alloc"]*=scale

    # ── P&L ──────────────────────────────────────────────────────────────────
    day_pnl=wins=losses=0; invested=0
    pos_detail=[]
    for p in positions:
        px=p["px"]; sh=p["alloc"]/px["open"]; pl=sh*(px["close"]-px["open"])
        day_pnl+=pl; invested+=p["alloc"]
        if pl>0: wins+=1
        else: losses+=1
        pos_detail.append({"sym":p["sym"],"sector":p["sector"],"pnl":pl,
                           "pct":px["pct"],"alloc":p["alloc"],"correct":px["pct"]>0})

    try:
        so=float(open_all.loc[td,"SPY"]); sc2=float(close_all.loc[td,"SPY"])
        spy_day=(sc2-so)/so*100
    except: spy_day=0

    bh = sum((90000/len(prices))/v["open"]*(v["close"]-v["open"]) for v in prices.values()) if prices else 0

    v1 = v1_pnl[idx] if idx < len(v1_pnl) else 0
    delta = day_pnl-v1

    daily_results.append({
        "date":td,"pnl":day_pnl,"v1":v1,"bh":bh,
        "wins":wins,"losses":losses,"n_pos":len(positions),
        "invested":invested,"spy_day":spy_day,"regime":reg,"vix":vix,
        "size_mult":size_mult,"opp_day":opp_day,"positions":pos_detail
    })

    mkey = td.strftime("%Y-%m")
    if mkey not in monthly:
        monthly[mkey]={"pnl":0,"v1":0,"bh":0,"days":0,"wins":0,"losses":0,"prof_days":0,"regimes":{}}
    monthly[mkey]["pnl"]+=day_pnl; monthly[mkey]["v1"]+=v1; monthly[mkey]["bh"]+=bh
    monthly[mkey]["days"]+=1; monthly[mkey]["wins"]+=wins; monthly[mkey]["losses"]+=losses
    if day_pnl>0: monthly[mkey]["prof_days"]+=1
    monthly[mkey]["regimes"][reg] = monthly[mkey]["regimes"].get(reg,0)+1

    s3 = "+" if day_pnl>=0 else ""
    sv1= "+" if v1>=0 else ""
    sd = "+" if delta>=0 else ""
    opp= "⚡" if opp_day else "  "
    print(f"  {td.strftime('%Y-%m-%d'):<12} {reg:<9} {vix:>5.1f} {len(positions):>3} "
          f"{size_mult:>4.1f}x ${invested:>9,.0f} "
          f"{s3}${day_pnl:>8,.0f} {sv1}${v1:>8,.0f} "
          f"{sd}${delta:>8,.0f}  {spy_day:>+6.2f}% {wins}W/{losses}L {opp}")

# ── Summary ───────────────────────────────────────────────────────────────────
total_v3  = sum(d["pnl"] for d in daily_results)
total_v1  = sum(d["v1"]  for d in daily_results)
total_bh  = sum(d["bh"]  for d in daily_results)
tw3       = sum(d["wins"] for d in daily_results)
tl3       = sum(d["losses"] for d in daily_results)
prof_days = sum(1 for d in daily_results if d["pnl"]>0)
best      = max(daily_results, key=lambda x: x["pnl"])
worst     = min(daily_results, key=lambda x: x["pnl"])
avg_pos   = np.mean([d["n_pos"] for d in daily_results])

print(f"\n{'='*76}")
print(f"  2026 YTD — FULL COMPARISON  v1 vs v3")
print(f"{'='*76}")
print(f"  {'Metric':<38} {'v1 (technical)':>15} {'v3 (vol-adapt)':>15}")
print(f"  {'─'*38} {'─'*15} {'─'*15}")
print(f"  {'Total P&L':<38} ${total_v1:>+14,.0f} ${total_v3:>+14,.0f}")
print(f"  {'vs Equal-weight benchmark':<38} ${total_v1-total_bh:>+14,.0f} ${total_v3-total_bh:>+14,.0f}")
v1_prof_days = sum(1 for d in daily_results if d["v1"] > 0)
v1_best = max(d["v1"] for d in daily_results)
v1_worst = min(d["v1"] for d in daily_results)
print(f"  {'Profitable days':<38} {v1_prof_days:>14}/{len(daily_results)} {prof_days:>14}/{len(daily_results)}")
print(f"  {'Position win rate':<38} {'53%':>15} {tw3/(tw3+tl3)*100:>14.1f}%")
print(f"  {'Avg positions/day':<38} {'24.5':>15} {avg_pos:>14.1f}")
print(f"  {'Best day':<38} ${v1_best:>+14,.0f} ${best['pnl']:>+14,.0f}")
print(f"  {'Worst day':<38} ${v1_worst:>+14,.0f} ${worst['pnl']:>+14,.0f}")
print(f"  {'TOTAL IMPROVEMENT':<38} {'—':>15} ${total_v3-total_v1:>+14,.0f}")

# ── Monthly breakdown ─────────────────────────────────────────────────────────
print(f"\n  {'─'*76}")
print(f"  MONTHLY BREAKDOWN")
print(f"  {'─'*76}")
print(f"  {'Month':<8} {'Days':>5} {'v1 P&L':>11} {'v3 P&L':>11} {'Δ':>9} "
      f"{'Win%':>6} {'ProfDays':>9} {'Regimes'}")
print(f"  {'─'*8} {'─'*5} {'─'*11} {'─'*11} {'─'*9} {'─'*6} {'─'*9} {'─'*20}")
for m, d in sorted(monthly.items()):
    tw = d["wins"]+d["losses"]
    wr = d["wins"]/max(tw,1)*100
    delta = d["pnl"]-d["v1"]
    reg_str = "  ".join(f"{k}:{v}" for k,v in sorted(d["regimes"].items()))
    print(f"  {m:<8} {d['days']:>5} ${d['v1']:>+10,.0f} ${d['pnl']:>+10,.0f} "
          f"${delta:>+8,.0f} {wr:>5.0f}% {d['prof_days']:>4}/{d['days']}     {reg_str}")

# ── Accuracy by regime ────────────────────────────────────────────────────────
print(f"\n  {'─'*76}")
print(f"  ACCURACY BY REGIME  (v3 strategy)")
print(f"  {'─'*76}")
print(f"  {'Regime':<10} {'Days':>5} {'Pos':>5} {'Win%':>7} {'Avg P&L/day':>13} "
      f"{'Size':>6} {'ProfDays':>9}")
print(f"  {'─'*10} {'─'*5} {'─'*5} {'─'*7} {'─'*13} {'─'*6} {'─'*9}")
for reg in ["CALM","ELEVATED","HIGH","EXTREME"]:
    days = [d for d in daily_results if d["regime"]==reg]
    if not days: continue
    all_pos = [p for d in days for p in d["positions"]]
    w_pos   = sum(1 for p in all_pos if p["correct"])
    t_pos   = len(all_pos)
    win_pct = w_pos/max(t_pos,1)*100
    avg_pnl = np.mean([d["pnl"] for d in days])
    avg_siz = np.mean([d["size_mult"] for d in days])
    prof    = sum(1 for d in days if d["pnl"]>0)
    print(f"  {reg:<10} {len(days):>5} {t_pos:>5} {win_pct:>6.0f}% "
          f"${avg_pnl:>+12,.0f} {avg_siz:>5.1f}x  {prof}/{len(days)}")

# ── Opportunity days ─────────────────────────────────────────────────────────
opp_days = [d for d in daily_results if d["opp_day"]]
if opp_days:
    print(f"\n  ⚡ OPPORTUNITY DAYS (VIX>25 + SPY down>1%):  {len(opp_days)} days")
    for d in opp_days:
        print(f"     {d['date'].strftime('%Y-%m-%d')}  VIX={d['vix']:.1f}  "
              f"size={d['size_mult']:.2f}x  P&L=${d['pnl']:>+,.0f}  "
              f"SPY={d['spy_day']:>+.2f}%  {d['wins']}W/{d['losses']}L")

# ── Best / Worst days ─────────────────────────────────────────────────────────
print(f"\n  TOP 10 BEST DAYS (v3)")
print(f"  {'─'*65}")
for d in sorted(daily_results, key=lambda x: -x["pnl"])[:10]:
    print(f"  {d['date'].strftime('%Y-%m-%d')}  {d['regime']:<9} "
          f"VIX={d['vix']:>5.1f}  {d['size_mult']:.1f}x  "
          f"${d['pnl']:>+9,.0f}  SPY={d['spy_day']:>+5.2f}%  "
          f"{d['wins']}W/{d['losses']}L  n={d['n_pos']}")

print(f"\n  TOP 10 WORST DAYS (v3)")
print(f"  {'─'*65}")
for d in sorted(daily_results, key=lambda x: x["pnl"])[:10]:
    print(f"  {d['date'].strftime('%Y-%m-%d')}  {d['regime']:<9} "
          f"VIX={d['vix']:>5.1f}  {d['size_mult']:.1f}x  "
          f"${d['pnl']:>+9,.0f}  SPY={d['spy_day']:>+5.2f}%  "
          f"{d['wins']}W/{d['losses']}L  n={d['n_pos']}")

# ── Equity curve ─────────────────────────────────────────────────────────────
print(f"\n  CUMULATIVE P&L CURVE  (v1=░  v3=█)")
print(f"  {'─'*65}")
cum_v3=0; cum_v1=0
for d in daily_results:
    cum_v3 += d["pnl"]; cum_v1 += d["v1"]
    if d["date"].day <= 3 or d == daily_results[-1]:
        bar_v3 = "█" * int(max(0,cum_v3)/200)
        bar_v1 = "░" * int(max(0,cum_v1)/200)
        neg_v3 = "▼" * int(max(0,-cum_v3)/200)
        print(f"  {d['date'].strftime('%Y-%m-%d')}  "
              f"v3=${cum_v3:>+8,.0f}  v1=${cum_v1:>+8,.0f}  "
              f"{bar_v3}{neg_v3}")

print(f"\n{'='*76}")
print(f"  FINAL VERDICT")
print(f"{'='*76}")
print(f"  v3 Total P&L         : ${total_v3:>+,.0f}")
print(f"  v1 Total P&L         : ${total_v1:>+,.0f}")
print(f"  Improvement          : ${total_v3-total_v1:>+,.0f}")
print(f"  Position win rate    : {tw3/(tw3+tl3)*100:.1f}%  (v1: 53%)")
print(f"  Avg positions/day    : {avg_pos:.1f}  (v1: 24.5)")
print(f"  Worst day            : ${worst['pnl']:>+,.0f}  (v1: -$2,030)")
print(f"  Days traded          : {len(daily_results)}")
print(f"{'='*76}\n")
