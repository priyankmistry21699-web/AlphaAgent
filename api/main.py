"""
AlphaAgent — Production FastAPI Server

The high-performance entry point for the AlphaAgent system.
Features:
  - REST endpoints for trading signals
  - WebSocket streaming for real-time agent reasoning
  - Performance monitoring and telemetry
  - Automatic error handling and fallback
"""

import time
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
import uvicorn

from data.market import MarketData
from agents.registry import AgentRegistry
from orchestrator.graph import build_alpha_graph
from agents.state import SignalPacket
from api.monitoring import Monitor
from database.manager import DatabaseManager
from database.models import Trade, Portfolio, AgentLog, AIPortfolioState, SignalHistory, BacktestResult, Settings, WarmupRegistry
from backtest.engine import BacktestEngine
from quant_engine.leaderboard import AgentLeaderboard
from quant_engine.portfolio_optimizer import PortfolioOptimizer
from data.universe import UniverseManager, UNIVERSES
from agents.portfolio import PortfolioAgent
from backtest.stress_test import StressTester, SCENARIOS
from backtest.walk_forward import WalkForwardValidator
from quant_engine.factor_exposure import FactorExposureAnalyzer
from trading.paper_trader import PaperTrader
from quant_engine.tda_signal import TDASignalEngine
from quant_engine.hawkes import HawkesProcess
from quant_engine.quasi_mc import QuasiMonteCarloEngine
from quant_engine.evt import ExtremeValueModel as EVTModel
from trading.rl_rebalancer import RLRebalancer
from config.settings_manager import settings

# ─── Configuration ───────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AlphaAgent Engine",
    description="Agentic Quantitative Trading Signal API",
    version="2.0.0"
)

# Enable CORS for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Registry, Graph, DB, and Paper Trader
registry = AgentRegistry()
graph = build_alpha_graph()
db_manager = DatabaseManager()
paper_trader = PaperTrader()


@app.on_event("startup")
async def _warm_up_embeddings_in_background():
    """Warm the sentence-transformer embedding model in a daemon thread so the
    API is responsive immediately while the model loads in parallel."""
    import threading
    try:
        sentiment_agent = registry.get_agent("sentiment")
        news_db = getattr(sentiment_agent, "news_db", None)
        if news_db is not None and hasattr(news_db, "warm_up"):
            threading.Thread(target=news_db.warm_up, daemon=True, name="embed-warmup").start()
    except Exception:
        pass


async def _bg_quote_warmer():
    """Batch-fetch live quotes for all universe tickers every 60s.
    Includes equities, ETFs, indices, commodities, forex, crypto."""
    import yfinance as yf
    loop = asyncio.get_running_loop()

    def _batch_fetch(syms: list) -> None:
        if not syms:
            return
        # yfinance download up to 200 at a time
        chunk_size = 150
        for i in range(0, len(syms), chunk_size):
            chunk = syms[i:i + chunk_size]
            try:
                raw = yf.download(chunk, period="5d", interval="1d",
                                  progress=False, auto_adjust=True, group_by="ticker")
                now = time.time()
                for sym in chunk:
                    try:
                        if len(chunk) == 1:
                            df = raw
                        else:
                            df = raw[sym] if sym in raw.columns.get_level_values(0) else None
                        if df is None or df.empty:
                            continue
                        closes = df["Close"].dropna()
                        if len(closes) < 1:
                            continue
                        price = float(closes.iloc[-1])
                        prev  = float(closes.iloc[-2]) if len(closes) >= 2 else price
                        chg   = price - prev
                        pct   = chg / prev * 100 if prev != 0 else 0.0
                        vols  = df["Volume"].dropna()
                        vol   = int(vols.iloc[-1]) if not vols.empty else None
                        _LIVE_QUOTE_CACHE[sym.upper()] = {
                            "price":      round(price, 4),
                            "change":     round(chg, 4),
                            "change_pct": round(pct, 4),
                            "positive":   chg >= 0,
                            "volume":     vol,
                            "expires":    now + _QUOTE_TTL,
                        }
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"quote warmer chunk error: {e}")

    await asyncio.sleep(3)
    while True:
        try:
            all_tickers = list(_ALL_UNIVERSE_TICKERS)  # snapshot
            await loop.run_in_executor(None, _batch_fetch, all_tickers)
            logger.info(f"quote warmer: {len(_LIVE_QUOTE_CACHE)}/{len(all_tickers)} tickers cached")
        except Exception as e:
            logger.warning(f"quote warmer error: {e}")
        await asyncio.sleep(90)


_PRIORITY_CORE: list[str] = [
    # Trimmed priority: top 25 most-traded US large caps + 5 core ETFs.
    # Lets background warmer finish first phase in ~3 min so the
    # portfolio + AI portfolio tabs become usable quickly.
    # On-demand fallback covers anything outside this list.
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
    "JPM", "V", "MA", "UNH", "LLY", "XOM", "WMT", "BRK-B",
    "COST", "JNJ", "HD", "ORCL", "AMD", "NFLX", "PG", "BAC",
    "GS",
    # Core ETFs (regime detection + sector signals)
    "SPY", "QQQ", "IWM", "GLD", "TLT",
]


# Per-region priority list used by portfolio-scan + strategy-build for
# on-demand warming. Covers each market's most-traded names so non-US
# users get fast results too.
_REGIONAL_PRIORITY: dict[str, list[str]] = {
    "us":     _PRIORITY_CORE,
    "india":  [
        "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS","BHARTIARTL.NS",
        "BAJFINANCE.NS","HINDUNILVR.NS","KOTAKBANK.NS","LT.NS","AXISBANK.NS","ASIANPAINT.NS",
        "MARUTI.NS","TITAN.NS","WIPRO.NS","SBIN.NS","HCLTECH.NS","SUNPHARMA.NS",
        "ULTRACEMCO.NS","TATAMOTORS.NS","NESTLEIND.NS","TATASTEEL.NS","NTPC.NS",
        "POWERGRID.NS","ONGC.NS",
        "^NSEI","^BSESN","NIFTYBEES.NS","BANKBEES.NS","INDA",
    ],
    "europe": [
        "ASML","SAP","NVO","SHEL","AZN","HSBC","BP","UL","GSK","DEO",
        "NESN.SW","ROG.SW","NOVN.SW","ABBN.SW","UBSG.SW",
        "OR.PA","MC.PA","BNP.PA","AIR.PA","SAN.PA",
        "ADS.DE","BMW.DE","SIE.DE","ALV.DE","MBG.DE",
        "EFA","EWU","EWG","EWQ","VGK",
    ],
    "asia":   [
        "BABA","TCEHY","JD","BIDU","PDD","NTES","SONY","TM","HMC","NTDOY",
        "MUFG","SFTBY","NTT","FANUY","BHP","RIO",
        "005930.KS","000660.KS","9988.HK","0700.HK","2318.HK","0005.HK",
        "CBA.AX","ANZ.AX","WBC.AX",
        "EWJ","FXI","MCHI","EWY","EWH","VPL",
    ],
    "japan":  [
        "SONY","TM","HMC","NTDOY","SFTBY","MUFG","NTT","FANUY","KYCCF",
        "7203.T","6758.T","6861.T","9984.T","8306.T","9432.T","4063.T",
        "6501.T","6702.T","7267.T","4661.T","9613.T","6098.T","8031.T",
        "^N225","EWJ","DXJ","DBJP","HEWJ","SCJ",
    ],
    "china":  [
        "BABA","TCEHY","JD","BIDU","PDD","NIO","XPEV","LI","NTES","BEKE",
        "9988.HK","0700.HK","1299.HK","0005.HK","0941.HK","2318.HK",
        "3690.HK","1810.HK","9999.HK","2020.HK","0175.HK",
        "FXI","MCHI","CQQQ","KWEB","ASHR","^HSI","000001.SS",
    ],
    "global": [
        "AAPL","MSFT","NVDA","AMZN","TSLA","META","BABA","TM","ASML","SAP",
        "NVO","SHEL","RIO","BHP","VALE","RELIANCE.NS","TCS.NS","SONY",
        "SPY","EFA","VWO","GLD","TLT","VT","ACWI","IEMG","VEU","VXUS",
        "BTC-USD","ETH-USD",
    ],
}


# Per-region sector classification — used by strategy-build for sector
# diversification. US uses the rich _SECTOR map inside strategy-build;
# these are simplified for international markets.
_REGIONAL_SECTOR: dict[str, dict[str, str]] = {
    "india": {
        "RELIANCE.NS":"Energy","ONGC.NS":"Energy","COALINDIA.NS":"Energy",
        "TCS.NS":"Tech","INFY.NS":"Tech","WIPRO.NS":"Tech","HCLTECH.NS":"Tech","TECHM.NS":"Tech",
        "HDFCBANK.NS":"Finance","ICICIBANK.NS":"Finance","KOTAKBANK.NS":"Finance",
        "AXISBANK.NS":"Finance","SBIN.NS":"Finance","BAJFINANCE.NS":"Finance",
        "BAJAJFINSV.NS":"Finance","MUTHOOTFIN.NS":"Finance",
        "HINDUNILVR.NS":"Consumer","NESTLEIND.NS":"Consumer","BRITANNIA.NS":"Consumer",
        "TITAN.NS":"Consumer","ASIANPAINT.NS":"Consumer",
        "MARUTI.NS":"Auto","TATAMOTORS.NS":"Auto",
        "SUNPHARMA.NS":"Health","DRREDDY.NS":"Health","CIPLA.NS":"Health","DIVISLAB.NS":"Health",
        "LT.NS":"Indust","ULTRACEMCO.NS":"Indust","PIIND.NS":"Indust",
        "TATASTEEL.NS":"Metals","JSWSTEEL.NS":"Metals","HINDALCO.NS":"Metals",
        "NTPC.NS":"Utility","POWERGRID.NS":"Utility",
        "BHARTIARTL.NS":"Telecom","ADANIPORTS.NS":"Infra","HAVELLS.NS":"Consumer",
    },
    "europe": {
        "ASML":"Semis","SAP":"Tech","SIE.DE":"Indust","INGA.AS":"Finance",
        "AZN":"Health","NVO":"Health","GSK":"Health","ROG.SW":"Health","NOVN.SW":"Health",
        "SHEL":"Energy","BP":"Energy","EQNR":"Energy",
        "HSBC":"Finance","BNP.PA":"Finance","SAN":"Finance","UBSG.SW":"Finance",
        "ALV.DE":"Finance","MUV2.DE":"Finance",
        "UL":"Consumer","DEO":"Consumer","NESN.SW":"Consumer","MC.PA":"Consumer",
        "OR.PA":"Consumer","ADS.DE":"Consumer",
        "BMW.DE":"Auto","MBG.DE":"Auto","VOW3.DE":"Auto",
        "AIR.PA":"Indust","BAS.DE":"Indust","SU.PA":"Indust",
        "BTI":"Consumer","PHG":"Health","ABBN.SW":"Indust","CSGN.SW":"Finance",
    },
    "asia": {
        "BABA":"Tech","TCEHY":"Tech","JD":"Consumer","BIDU":"Tech","PDD":"Consumer",
        "NTES":"Tech","SONY":"Tech","NTDOY":"Tech","NTT":"Telecom","SFTBY":"Tech",
        "TM":"Auto","HMC":"Auto","FANUY":"Indust","MUFG":"Finance",
        "BHP":"Metals","RIO":"Metals","VALE":"Metals",
        "005930.KS":"Semis","000660.KS":"Semis","035420.KS":"Tech","035720.KS":"Tech",
        "051910.KS":"Chemicals","9988.HK":"Tech","0700.HK":"Tech","1299.HK":"Finance",
        "0005.HK":"Finance","0941.HK":"Telecom","2318.HK":"Finance",
        "CBA.AX":"Finance","ANZ.AX":"Finance","WBC.AX":"Finance","NAB.AX":"Finance",
        "WES.AX":"Consumer",
    },
    "japan": {
        "7203.T":"Auto","6758.T":"Tech","6861.T":"Indust","9984.T":"Tech",
        "8306.T":"Finance","9432.T":"Telecom","4063.T":"Chemicals","6501.T":"Indust",
        "6702.T":"Tech","7267.T":"Auto","4661.T":"Consumer","9613.T":"Tech",
        "6098.T":"Indust","8031.T":"Finance","SONY":"Tech","TM":"Auto","HMC":"Auto",
        "NTDOY":"Tech","SFTBY":"Tech","MUFG":"Finance","NTT":"Telecom","FANUY":"Indust",
    },
    "china": {
        "BABA":"Tech","TCEHY":"Tech","JD":"Consumer","BIDU":"Tech","PDD":"Consumer",
        "NTES":"Tech","NIO":"Auto","XPEV":"Auto","LI":"Auto","BEKE":"RealEstate",
        "9988.HK":"Tech","0700.HK":"Tech","1299.HK":"Finance","0005.HK":"Finance",
        "0941.HK":"Telecom","2318.HK":"Finance","3690.HK":"Consumer","1810.HK":"Tech",
        "9999.HK":"Tech","2020.HK":"Consumer","0175.HK":"Auto",
    },
    "global": {
        # Global mixes US/Asia/EU — fall back to broad sectors
        "AAPL":"Tech","MSFT":"Tech","NVDA":"Tech","AMZN":"Tech","TSLA":"Tech","META":"Tech",
        "BABA":"Tech","TM":"Auto","ASML":"Semis","SAP":"Tech","NVO":"Health","SHEL":"Energy",
        "RIO":"Metals","BHP":"Metals","VALE":"Metals","SONY":"Tech",
        "RELIANCE.NS":"Energy","TCS.NS":"Tech",
    },
}


def _get_regional_priority(region: str) -> list[str]:
    """Return the priority ticker list for a region (defaults to US)."""
    return _REGIONAL_PRIORITY.get((region or "us").lower(), _PRIORITY_CORE)


def _get_regional_universe(region: str, asset_type: str | None = None) -> list[str]:
    """Build a flat ticker list for a region + optional asset_type filter."""
    region = (region or "us").lower()
    rdata = _REGIONAL_UNIVERSES.get(region, _REGIONAL_UNIVERSES["us"])
    _type_map = {
        "stocks":      "equities",
        "etfs":        "etfs",
        "mutual_fund": "mutual_funds",
        "index":       "indices",
        "commodities": "commodities",
        "forex":       "forex",
        "crypto":      "crypto",
    }
    cat_key = _type_map.get((asset_type or "").lower())
    if cat_key:
        return list(rdata.get(cat_key, []))
    # All categories
    out: list[str] = []
    seen: set[str] = set()
    for cat in rdata.values():
        for t in cat:
            if t not in seen:
                seen.add(t); out.append(t)
    return out


async def _bg_signal_warmer():
    """Sequentially run AlphaAgent signals for all signable tickers.
    Phase 1: priority core (~60 tickers) at 2s gap → done in ~2 min.
    Phase 2: remaining tickers at 4s gap."""
    await asyncio.sleep(12)
    loop = asyncio.get_running_loop()

    def _warm_one(sym: str):
        md    = _get_market_data(sym.upper())
        state = {"ticker": sym.upper(), "market_data": md, "registry": registry}
        res   = graph.invoke(state)
        fi    = res.get("final_signal", {})
        pkt   = fi.get("packet")
        if not pkt:
            return
        prob_up   = fi.get("probability_up", 0.5)
        direction = "LONG" if prob_up >= 0.53 else "SHORT" if prob_up <= 0.47 else "NEUTRAL"
        _SIGNAL_CACHE[(sym.upper(), "1d")] = ({
            "direction":  direction,
            "conviction": float(getattr(pkt, "conviction", 0)),
            "probability": prob_up,
            "agents": [{"agent_name": a.agent_name, "vote": a.vote}
                       for a in getattr(pkt, "agent_results", [])],
        }, time.time() + _SIGNAL_TTL)

    def _ordered_signal_tickers() -> list[str]:
        global _USER_REGION
        seen: dict[str, bool] = {}
        # Priority core first (always)
        for t in _PRIORITY_CORE:
            if _is_signable(t):
                seen[t] = True
        # User's region
        region_data = _REGIONAL_UNIVERSES.get(_USER_REGION, {})
        for cat in ("equities", "etfs", "crypto"):
            for t in region_data.get(cat, []):
                if _is_signable(t):
                    seen[t] = True
        # Portfolio themes
        for tickers in _PORTFOLIO_UNIVERSES.values():
            for t in tickers:
                if _is_signable(t):
                    seen[t] = True
        # All remaining regions
        for rname, rdata in _REGIONAL_UNIVERSES.items():
            if rname == _USER_REGION:
                continue
            for cat in ("equities", "etfs", "crypto"):
                for t in rdata.get(cat, []):
                    if _is_signable(t):
                        seen[t] = True
        return list(seen.keys())

    priority_set = {t.upper() for t in _PRIORITY_CORE if _is_signable(t)}

    while True:
        signal_tickers = _ordered_signal_tickers()
        warmed = 0

        # Phase 1: priority core at 2s gap
        for sym in signal_tickers:
            if sym.upper() not in priority_set:
                continue
            key = (sym.upper(), "1d")
            if key in _SIGNAL_CACHE and time.time() < _SIGNAL_CACHE[key][1]:
                try:
                    db_manager.increment_warmup_cache(sym.upper(), hit=True)
                except Exception:
                    pass
                continue
            try:
                t0 = time.time()
                await loop.run_in_executor(None, _warm_one, sym)
                warmed += 1
                try:
                    db_manager.register_warmup(sym.upper(), status="done",
                                               duration_ms=round((time.time() - t0) * 1000))
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"signal warmer [priority] {sym}: {e}")
                try:
                    db_manager.register_warmup(sym.upper(), status="failed", error_msg=str(e))
                except Exception:
                    pass
            await asyncio.sleep(2)

        logger.info(f"signal warmer priority phase done: cache={len(_SIGNAL_CACHE)}")

        # Phase 2: remaining tickers at 4s gap
        for sym in signal_tickers:
            if sym.upper() in priority_set:
                continue
            key = (sym.upper(), "1d")
            if key in _SIGNAL_CACHE and time.time() < _SIGNAL_CACHE[key][1]:
                try:
                    db_manager.increment_warmup_cache(sym.upper(), hit=True)
                except Exception:
                    pass
                continue
            try:
                t0 = time.time()
                await loop.run_in_executor(None, _warm_one, sym)
                warmed += 1
                try:
                    db_manager.register_warmup(sym.upper(), status="done",
                                               duration_ms=round((time.time() - t0) * 1000))
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"signal warmer [extended] {sym}: {e}")
                try:
                    db_manager.register_warmup(sym.upper(), status="failed", error_msg=str(e))
                except Exception:
                    pass
            await asyncio.sleep(4)

        if warmed > 0:
            logger.info(f"signal warmer cycle done: warmed {warmed} | cache={len(_SIGNAL_CACHE)} | region={_USER_REGION}")
        await asyncio.sleep(30)


@app.on_event("startup")
async def _launch_background_warmers():
    """Launch quote + signal pre-warming tasks in the background."""
    asyncio.create_task(_bg_quote_warmer(), name="quote-warmer")
    asyncio.create_task(_bg_signal_warmer(), name="signal-warmer")
    logger.info(f"Background warmers launched — {len(_ALL_UNIVERSE_TICKERS)} tickers total")


# ─── On-demand signal generation (used by portfolio + AI portfolio when cache empty) ──

def _generate_signal_sync(sym: str) -> dict | None:
    """
    Run the AlphaAgent pipeline for ONE ticker. Synchronous (call from a thread).
    Writes to _SIGNAL_CACHE on success. Returns the cached signal dict or None.
    """
    try:
        sym = sym.upper()
        md    = _get_market_data(sym)
        state = {"ticker": sym, "market_data": md, "registry": registry}
        res   = graph.invoke(state)
        fi    = res.get("final_signal", {})
        pkt   = fi.get("packet")
        if not pkt:
            return None
        prob_up   = fi.get("probability_up", 0.5)
        direction = "LONG" if prob_up >= 0.53 else "SHORT" if prob_up <= 0.47 else "NEUTRAL"
        sig = {
            "direction":  direction,
            "conviction": float(getattr(pkt, "conviction", 0)),
            "probability": prob_up,
            "agents": [{"agent_name": a.agent_name, "vote": a.vote}
                       for a in getattr(pkt, "agent_results", [])],
            "holding_period": {
                "half_life_days": getattr(getattr(pkt, "holding_period", None),
                                          "half_life_days", None)
            } if getattr(pkt, "holding_period", None) else None,
            "entropy": float(fi.get("entropy", 1.0)),
        }
        _SIGNAL_CACHE[(sym, "1d")] = (sig, time.time() + _SIGNAL_TTL)
        return sig
    except Exception as e:
        logger.warning(f"on-demand signal {sym}: {e}")
        return None


async def _on_demand_warm_signals(
    tickers: list[str],
    max_workers: int = 4,
    per_ticker_timeout: int = 35,
    total_budget: int = 90,
    skip_if_cached: bool = True,
) -> dict[str, dict]:
    """
    Generate signals for `tickers` in parallel. Skips tickers already in cache
    (unless skip_if_cached=False). Bounded by total_budget seconds so the request
    can't hang forever. Returns dict[ticker -> signal] for whatever completed.
    """
    from concurrent.futures import ThreadPoolExecutor

    now = time.time()
    todo: list[str] = []
    out:  dict[str, dict] = {}

    for t in tickers:
        t_up = t.upper()
        if skip_if_cached:
            entry = _SIGNAL_CACHE.get((t_up, "1d"))
            if entry and now < entry[1]:
                out[t_up] = entry[0]
                continue
        todo.append(t_up)

    if not todo:
        return out

    loop = asyncio.get_running_loop()
    deadline = time.time() + total_budget
    workers = min(max_workers, max(1, len(todo)))

    def _run_batch():
        results: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_generate_signal_sync, sym): sym for sym in todo}
            for fut in futs:
                sym = futs[fut]
                remaining = max(1, deadline - time.time())
                try:
                    sig = fut.result(timeout=min(per_ticker_timeout, remaining))
                    if sig is not None:
                        results[sym] = sig
                except Exception as e:
                    logger.warning(f"on-demand batch {sym} timeout/err: {e}")
                if time.time() >= deadline:
                    break
        return results

    try:
        batch = await asyncio.wait_for(
            loop.run_in_executor(None, _run_batch),
            timeout=total_budget + 5,
        )
        out.update(batch)
    except asyncio.TimeoutError:
        logger.warning(f"on-demand warm batch exceeded total budget ({total_budget}s)")
    return out

# Module-level MarketData cache — reuse same object per ticker for 5 min so
# internal yfinance/options/FRED data isn't re-fetched on every signal request.
_MD_CACHE: dict[str, tuple] = {}   # ticker -> (MarketData, expire_epoch)
_MD_TTL = 300                       # seconds

def _get_market_data(ticker: str) -> MarketData:
    import time as _t
    ticker = ticker.upper()
    entry = _MD_CACHE.get(ticker)
    now = _t.time()
    if entry and now < entry[1]:
        return entry[0]
    md = MarketData(ticker)
    _MD_CACHE[ticker] = (md, now + _MD_TTL)
    return md

# Simple TTL caches for slow endpoints (summary=60s, earnings=1h, news=10min)
_SUMMARY_CACHE: dict = {}    # {"data": {...}, "expires": float}
_EARNINGS_CACHE: dict = {}   # {"data": [...], "expires": float}
_NEWS_CACHE: dict = {}       # {"data": [...], "expires": float}
_SUMMARY_TTL = 60
_EARNINGS_TTL = 3600
_NEWS_TTL = 600

# Signal result cache — same ticker+horizon within 5 min returns instantly
_SIGNAL_CACHE: dict[tuple, tuple] = {}   # (ticker, horizon) -> (result_dict, expire)
_SIGNAL_TTL = 300  # 5 minutes

# Live quote cache — batch-fetched for all universe tickers every 90s
_LIVE_QUOTE_CACHE: dict = {}   # ticker -> {price, change, change_pct, volume, expires}
_QUOTE_TTL = 180               # 3-min TTL; warmer runs every 90s so there's always a valid entry

# AI-managed live portfolio — loaded from DB on startup, written on every mutation
_AI_PORTFOLIO: dict = db_manager.load_ai_portfolio()

# ─── REST Endpoints ──────────────────────────────────────────────────────────

_REACT_DIST = Path(__file__).parent.parent / "frontend-react" / "dist"
_LEGACY_FRONTEND = Path(__file__).parent.parent / "frontend"

# Serve React build assets
if _REACT_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(_REACT_DIST / "assets")), name="react-assets")

@app.get("/")
async def root():
    if _REACT_DIST.exists() and (_REACT_DIST / "index.html").exists():
        return FileResponse(str(_REACT_DIST / "index.html"))
    return FileResponse(str(_LEGACY_FRONTEND / "index.html"))

@app.get("/legacy")
async def legacy():
    return FileResponse(str(_LEGACY_FRONTEND / "index.html"))

@app.get("/api/status")
async def status():
    return {"status": "online", "engine": "AlphaAgent V2", "phase": 6}


# ─── Market Data Endpoints ───────────────────────────────────────────────────

# US Markets
_US_SYMBOLS = {
    "S&P 500":      "^GSPC",
    "NASDAQ":       "^IXIC",
    "Dow Jones":    "^DJI",
    "Russell 2000": "^RUT",
    "VIX":          "^VIX",
}

# Global Markets
_GLOBAL_SYMBOLS = {
    "FTSE 100":     "^FTSE",
    "DAX":          "^GDAXI",
    "CAC 40":       "^FCHI",
    "Nikkei 225":   "^N225",
    "Hang Seng":    "^HSI",
    "Shanghai":     "000001.SS",
    "ASX 200":      "^AXJO",
    "Euro Stoxx 50":"^STOXX50E",
}

# Crypto & Commodities
_ASSET_SYMBOLS = {
    "Bitcoin":      "BTC-USD",
    "Ethereum":     "ETH-USD",
    "Gold":         "GC=F",
    "Crude Oil":    "CL=F",
    "Silver":       "SI=F",
    "Natural Gas":  "NG=F",
}

# FX Rates
_FX_SYMBOLS = {
    "EUR/USD":      "EURUSD=X",
    "GBP/USD":      "GBPUSD=X",
    "USD/JPY":      "JPY=X",
    "USD/CNY":      "CNY=X",
    "DXY Index":    "DX-Y.NYB",
}

_ALL_MARKET_SYMBOLS = {**_US_SYMBOLS, **_GLOBAL_SYMBOLS, **_ASSET_SYMBOLS, **_FX_SYMBOLS}

_NEWS_TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]

_RSS_FEEDS = [
    ("MarketWatch",    "https://feeds.marketwatch.com/marketwatch/topstories/"),
    ("Reuters Biz",    "https://feeds.reuters.com/reuters/businessNews"),
    ("CNBC Markets",   "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839135"),
    ("BBC Business",   "https://feeds.bbci.co.uk/news/business/rss.xml"),
    ("Yahoo Finance",  "https://finance.yahoo.com/news/rssindex"),
    ("Seeking Alpha",  "https://seekingalpha.com/market-news/index.xml"),
    ("Bloomberg",      "https://feeds.bloomberg.com/markets/news.rss"),
    ("FT Markets",     "https://www.ft.com/markets?format=rss"),
    ("WSJ Markets",    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"),
]

_EARNINGS_WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "INTC",
    "JPM", "BAC", "GS", "MS", "WMT", "HD", "DIS", "NFLX", "PYPL", "CRM",
    "BA", "CAT", "XOM", "CVX", "PFE", "JNJ", "MRK", "UNH",
]


def _parse_ts(ts_str) -> "datetime":
    """Parse an ISO timestamp string from the DB into a naive UTC datetime."""
    from datetime import datetime as _dt
    if isinstance(ts_str, _dt):
        return ts_str.replace(tzinfo=None) if getattr(ts_str, "tzinfo", None) else ts_str
    try:
        s = str(ts_str).replace("Z", "+00:00")
        dt = _dt.fromisoformat(s)
        return dt.replace(tzinfo=None)
    except Exception:
        return _dt.utcnow()


def _fetch_quote_fast(sym: str) -> dict | None:
    """Fetch live quote — checks _LIVE_QUOTE_CACHE first, falls back to yfinance fast_info."""
    import yfinance as yf
    cached = _LIVE_QUOTE_CACHE.get(sym.upper())
    if cached and time.time() < cached.get("expires", 0):
        return cached
    try:
        t = yf.Ticker(sym)
        fi = t.fast_info
        price = getattr(fi, "last_price", None) or getattr(fi, "regular_market_price", None)
        prev  = getattr(fi, "previous_close", None)
        if price is None or price == 0:
            # fallback: short history
            hist = t.history(period="5d", interval="1d", auto_adjust=True)
            if hist.empty:
                return None
            price = float(hist["Close"].dropna().iloc[-1])
            prev  = float(hist["Close"].dropna().iloc[-2]) if len(hist) >= 2 else price
        price = float(price)
        prev  = float(prev) if prev else price
        chg   = price - prev
        pct   = chg / prev * 100 if prev != 0 else 0.0
        # extra context
        day_high = getattr(fi, "day_high", None)
        day_low  = getattr(fi, "day_low",  None)
        volume   = getattr(fi, "three_month_average_volume", None) or getattr(fi, "regular_market_volume", None)
        return {
            "price":      round(price, 4),
            "change":     round(chg, 4),
            "change_pct": round(pct, 4),
            "positive":   chg >= 0,
            "day_high":   round(float(day_high), 4) if day_high else None,
            "day_low":    round(float(day_low),  4) if day_low  else None,
            "volume":     int(volume) if volume else None,
        }
    except Exception:
        return None


def _parse_rss(source_name: str, url: str, max_items: int = 8) -> list:
    """Parse an RSS/Atom feed and return a list of article dicts."""
    import urllib.request
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime
    import time as _time
    articles = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 AlphaAgent/2.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            raw = resp.read()
        root = ET.fromstring(raw)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        # RSS 2.0
        items = root.findall(".//item")
        for item in items[:max_items]:
            title = (item.findtext("title") or "").strip()
            link  = (item.findtext("link")  or "#").strip()
            pub   = item.findtext("pubDate") or ""
            try:
                ts = int(parsedate_to_datetime(pub).timestamp()) if pub else int(_time.time())
            except Exception:
                ts = int(_time.time())
            if title:
                articles.append({"title": title, "link": link, "publisher": source_name,
                                  "published_at": ts, "ticker": ""})

        # Atom feeds
        if not articles:
            for entry in root.findall("atom:entry", ns)[:max_items]:
                title = (entry.findtext("atom:title", namespaces=ns) or "").strip()
                link_el = entry.find("atom:link", ns)
                link = link_el.get("href", "#") if link_el is not None else "#"
                pub = entry.findtext("atom:published", namespaces=ns) or \
                      entry.findtext("atom:updated",   namespaces=ns) or ""
                try:
                    ts = int(__import__("datetime").datetime.fromisoformat(
                        pub.replace("Z", "+00:00")).timestamp()) if pub else int(_time.time())
                except Exception:
                    ts = int(_time.time())
                if title:
                    articles.append({"title": title, "link": link, "publisher": source_name,
                                      "published_at": ts, "ticker": ""})
    except Exception:
        pass
    return articles


@app.get("/api/v1/settings")
async def get_settings():
    """Return all user-configurable settings."""
    return settings.all()


@app.post("/api/v1/settings")
async def update_settings(patch: Dict[str, Any]):
    """
    Apply a partial patch to settings and persist to settings.yaml.
    Supports nested dicts: {"technical": {"rsi_window": 21}}
    Changes take effect immediately — no server restart needed.
    """
    try:
        settings.apply_patch(patch)
        settings.save()
        try:
            db_manager.bulk_save_settings(settings.all())
        except Exception:
            pass  # DB sync is best-effort; YAML is authoritative
        return {"status": "ok", "message": "Settings updated and saved."}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/v1/settings/reset")
async def reset_settings():
    """Reload settings from disk (discards any unsaved in-memory changes)."""
    settings.reload()
    return {"status": "ok", "message": "Settings reloaded from disk."}


# ── Ticker metadata (name + type + sector) for search autocomplete ────────────
_TICKER_NAME_MAP: dict[str, dict] = {
    # US Large Cap
    "AAPL":{"name":"Apple Inc.","sector":"Technology"},"MSFT":{"name":"Microsoft Corporation","sector":"Technology"},
    "NVDA":{"name":"NVIDIA Corporation","sector":"Semiconductors"},"AMZN":{"name":"Amazon.com Inc.","sector":"Consumer"},
    "GOOGL":{"name":"Alphabet Inc.","sector":"Technology"},"META":{"name":"Meta Platforms","sector":"Technology"},
    "TSLA":{"name":"Tesla Inc.","sector":"Automotive"},"JPM":{"name":"JPMorgan Chase","sector":"Finance"},
    "BAC":{"name":"Bank of America","sector":"Finance"},"V":{"name":"Visa Inc.","sector":"Finance"},
    "MA":{"name":"Mastercard","sector":"Finance"},"XOM":{"name":"ExxonMobil","sector":"Energy"},
    "CVX":{"name":"Chevron Corporation","sector":"Energy"},"UNH":{"name":"UnitedHealth Group","sector":"Healthcare"},
    "JNJ":{"name":"Johnson & Johnson","sector":"Healthcare"},"WMT":{"name":"Walmart Inc.","sector":"Consumer"},
    "PG":{"name":"Procter & Gamble","sector":"Consumer"},"COST":{"name":"Costco Wholesale","sector":"Consumer"},
    "HD":{"name":"Home Depot","sector":"Consumer"},"MCD":{"name":"McDonald's","sector":"Consumer"},
    "KO":{"name":"Coca-Cola","sector":"Consumer"},"PEP":{"name":"PepsiCo","sector":"Consumer"},
    "ABBV":{"name":"AbbVie Inc.","sector":"Healthcare"},"LLY":{"name":"Eli Lilly","sector":"Healthcare"},
    "MRK":{"name":"Merck & Co.","sector":"Healthcare"},"PFE":{"name":"Pfizer Inc.","sector":"Healthcare"},
    "AMD":{"name":"Advanced Micro Devices","sector":"Semiconductors"},"INTC":{"name":"Intel Corporation","sector":"Semiconductors"},
    "CRM":{"name":"Salesforce","sector":"Technology"},"ADBE":{"name":"Adobe Inc.","sector":"Technology"},
    "QCOM":{"name":"Qualcomm","sector":"Semiconductors"},"TXN":{"name":"Texas Instruments","sector":"Semiconductors"},
    "AVGO":{"name":"Broadcom Inc.","sector":"Semiconductors"},"ORCL":{"name":"Oracle Corporation","sector":"Technology"},
    "NFLX":{"name":"Netflix","sector":"Media"},"DIS":{"name":"Walt Disney Co.","sector":"Media"},
    "GS":{"name":"Goldman Sachs","sector":"Finance"},"MS":{"name":"Morgan Stanley","sector":"Finance"},
    "BLK":{"name":"BlackRock","sector":"Finance"},"BRK-B":{"name":"Berkshire Hathaway B","sector":"Finance"},
    "CAT":{"name":"Caterpillar","sector":"Industrial"},"BA":{"name":"Boeing","sector":"Industrial"},
    "GE":{"name":"GE Aerospace","sector":"Industrial"},"RTX":{"name":"Raytheon Technologies","sector":"Defense"},
    "NKE":{"name":"Nike Inc.","sector":"Consumer"},"SBUX":{"name":"Starbucks","sector":"Consumer"},
    "F":{"name":"Ford Motor","sector":"Automotive"},"GM":{"name":"General Motors","sector":"Automotive"},
    "PYPL":{"name":"PayPal Holdings","sector":"Finance"},"ABNB":{"name":"Airbnb","sector":"Travel"},
    "UBER":{"name":"Uber Technologies","sector":"Technology"},"LYFT":{"name":"Lyft","sector":"Technology"},
    "PLTR":{"name":"Palantir Technologies","sector":"Technology"},"SMCI":{"name":"Super Micro Computer","sector":"Technology"},
    "ANET":{"name":"Arista Networks","sector":"Technology"},"CRWD":{"name":"CrowdStrike","sector":"Cybersecurity"},
    "SNOW":{"name":"Snowflake","sector":"Technology"},"NET":{"name":"Cloudflare","sector":"Technology"},
    "DDOG":{"name":"Datadog","sector":"Technology"},"MDB":{"name":"MongoDB","sector":"Technology"},
    "ZS":{"name":"Zscaler","sector":"Cybersecurity"},"OKTA":{"name":"Okta","sector":"Cybersecurity"},
    "PANW":{"name":"Palo Alto Networks","sector":"Cybersecurity"},"FTNT":{"name":"Fortinet","sector":"Cybersecurity"},
    # ETFs
    "SPY":{"name":"SPDR S&P 500 ETF","sector":"ETF"},"QQQ":{"name":"Invesco QQQ (Nasdaq-100)","sector":"ETF"},
    "IWM":{"name":"iShares Russell 2000 ETF","sector":"ETF"},"VTI":{"name":"Vanguard Total Market ETF","sector":"ETF"},
    "VOO":{"name":"Vanguard S&P 500 ETF","sector":"ETF"},"DIA":{"name":"SPDR Dow Jones ETF","sector":"ETF"},
    "GLD":{"name":"SPDR Gold Shares","sector":"Commodity ETF"},"SLV":{"name":"iShares Silver Trust","sector":"Commodity ETF"},
    "TLT":{"name":"iShares 20+ Yr Treasury Bond","sector":"Bond ETF"},"HYG":{"name":"iShares High Yield Bond","sector":"Bond ETF"},
    "XLK":{"name":"Technology Select Sector SPDR","sector":"Sector ETF"},"XLF":{"name":"Financial Select Sector SPDR","sector":"Sector ETF"},
    "XLE":{"name":"Energy Select Sector SPDR","sector":"Sector ETF"},"XLV":{"name":"Health Select Sector SPDR","sector":"Sector ETF"},
    "XLI":{"name":"Industrial Select Sector SPDR","sector":"Sector ETF"},"XLY":{"name":"Consumer Discret. SPDR","sector":"Sector ETF"},
    "VNQ":{"name":"Vanguard Real Estate ETF","sector":"REIT ETF"},"ARKK":{"name":"ARK Innovation ETF","sector":"ETF"},
    "SCHD":{"name":"Schwab US Dividend ETF","sector":"Dividend ETF"},"VYM":{"name":"Vanguard High Dividend Yield","sector":"Dividend ETF"},
    "EFA":{"name":"iShares MSCI EAFE (Developed Markets)","sector":"Global ETF"},"VWO":{"name":"Vanguard Emerging Markets","sector":"Emerging ETF"},
    "INDA":{"name":"iShares MSCI India ETF","sector":"India ETF"},"EWJ":{"name":"iShares MSCI Japan ETF","sector":"Japan ETF"},
    "FXI":{"name":"iShares China Large-Cap ETF","sector":"China ETF"},"EEM":{"name":"iShares MSCI Emerging Markets","sector":"Emerging ETF"},
    "SMH":{"name":"VanEck Semiconductor ETF","sector":"Sector ETF"},"SOXX":{"name":"iShares Semiconductor ETF","sector":"Sector ETF"},
    "VGT":{"name":"Vanguard Information Technology","sector":"Sector ETF"},"SKYY":{"name":"First Trust Cloud Computing ETF","sector":"Sector ETF"},
    "BOTZ":{"name":"Global X Robotics & AI ETF","sector":"Theme ETF"},"IGV":{"name":"iShares Expanded Tech-Software","sector":"Sector ETF"},
    # Mutual Funds
    "VFIAX":{"name":"Vanguard 500 Index Fund Admiral","sector":"Mutual Fund"},"FXAIX":{"name":"Fidelity 500 Index Fund","sector":"Mutual Fund"},
    "VTSAX":{"name":"Vanguard Total Stock Market Admiral","sector":"Mutual Fund"},"FSKAX":{"name":"Fidelity Total Market Index","sector":"Mutual Fund"},
    # India
    "RELIANCE.NS":{"name":"Reliance Industries","sector":"Conglomerate"},"TCS.NS":{"name":"Tata Consultancy Services","sector":"IT Services"},
    "HDFCBANK.NS":{"name":"HDFC Bank","sector":"Banking"},"INFY.NS":{"name":"Infosys","sector":"IT Services"},
    "ICICIBANK.NS":{"name":"ICICI Bank","sector":"Banking"},"BAJFINANCE.NS":{"name":"Bajaj Finance","sector":"Finance"},
    "HINDUNILVR.NS":{"name":"Hindustan Unilever","sector":"FMCG"},"KOTAKBANK.NS":{"name":"Kotak Mahindra Bank","sector":"Banking"},
    "LT.NS":{"name":"Larsen & Toubro","sector":"Engineering"},"AXISBANK.NS":{"name":"Axis Bank","sector":"Banking"},
    "MARUTI.NS":{"name":"Maruti Suzuki","sector":"Automotive"},"TITAN.NS":{"name":"Titan Company","sector":"Consumer"},
    "WIPRO.NS":{"name":"Wipro Limited","sector":"IT Services"},"SBIN.NS":{"name":"State Bank of India","sector":"Banking"},
    "HCLTECH.NS":{"name":"HCL Technologies","sector":"IT Services"},"TATAMOTORS.NS":{"name":"Tata Motors","sector":"Automotive"},
    # Europe
    "ASML":{"name":"ASML Holding (Semis)","sector":"Semiconductors"},"SAP":{"name":"SAP SE (ERP Software)","sector":"Technology"},
    "AZN":{"name":"AstraZeneca","sector":"Healthcare"},"NVO":{"name":"Novo Nordisk","sector":"Healthcare"},
    "SHEL":{"name":"Shell plc","sector":"Energy"},"BP":{"name":"BP plc","sector":"Energy"},
    "HSBC":{"name":"HSBC Holdings","sector":"Banking"},"UL":{"name":"Unilever plc","sector":"FMCG"},
    # Asia
    "SONY":{"name":"Sony Group","sector":"Technology"},"TM":{"name":"Toyota Motor","sector":"Automotive"},
    "BABA":{"name":"Alibaba Group","sector":"E-commerce"},"TCEHY":{"name":"Tencent Holdings","sector":"Technology"},
    "JD":{"name":"JD.com","sector":"E-commerce"},"BIDU":{"name":"Baidu","sector":"Technology"},
    "PDD":{"name":"PDD Holdings (Temu/Pinduoduo)","sector":"E-commerce"},"HMC":{"name":"Honda Motor","sector":"Automotive"},
    "NTDOY":{"name":"Nintendo","sector":"Gaming"},"SFTBY":{"name":"SoftBank Group","sector":"Investment"},
    # Commodities (futures)
    "GC=F":{"name":"Gold Futures","sector":"Precious Metal"},"SI=F":{"name":"Silver Futures","sector":"Precious Metal"},
    "PL=F":{"name":"Platinum Futures","sector":"Precious Metal"},"PA=F":{"name":"Palladium Futures","sector":"Precious Metal"},
    "CL=F":{"name":"Crude Oil (WTI) Futures","sector":"Energy Commodity"},"BZ=F":{"name":"Brent Crude Futures","sector":"Energy Commodity"},
    "NG=F":{"name":"Natural Gas Futures","sector":"Energy Commodity"},"HG=F":{"name":"Copper Futures","sector":"Industrial Metal"},
    "ZW=F":{"name":"Wheat Futures","sector":"Agriculture"},"ZC=F":{"name":"Corn Futures","sector":"Agriculture"},
    "ZS=F":{"name":"Soybean Futures","sector":"Agriculture"},"CC=F":{"name":"Cocoa Futures","sector":"Agriculture"},
    "KC=F":{"name":"Coffee Futures","sector":"Agriculture"},"CT=F":{"name":"Cotton Futures","sector":"Agriculture"},
    # Forex
    "EURUSD=X":{"name":"Euro / US Dollar","sector":"Forex"},"GBPUSD=X":{"name":"British Pound / USD","sector":"Forex"},
    "JPY=X":{"name":"USD / Japanese Yen","sector":"Forex"},"AUDUSD=X":{"name":"Australian Dollar / USD","sector":"Forex"},
    "CADUSD=X":{"name":"Canadian Dollar / USD","sector":"Forex"},"CHFUSD=X":{"name":"Swiss Franc / USD","sector":"Forex"},
    "INR=X":{"name":"USD / Indian Rupee","sector":"Forex"},"CNY=X":{"name":"USD / Chinese Yuan","sector":"Forex"},
    "DX-Y.NYB":{"name":"US Dollar Index (DXY)","sector":"Forex"},"EURINR=X":{"name":"Euro / Indian Rupee","sector":"Forex"},
    # Crypto
    "BTC-USD":{"name":"Bitcoin","sector":"Crypto"},"ETH-USD":{"name":"Ethereum","sector":"Crypto"},
    "SOL-USD":{"name":"Solana","sector":"Crypto"},"BNB-USD":{"name":"BNB (Binance Coin)","sector":"Crypto"},
    "XRP-USD":{"name":"XRP (Ripple)","sector":"Crypto"},"ADA-USD":{"name":"Cardano","sector":"Crypto"},
    "AVAX-USD":{"name":"Avalanche","sector":"Crypto"},"DOGE-USD":{"name":"Dogecoin","sector":"Crypto"},
    "MATIC-USD":{"name":"Polygon (MATIC)","sector":"Crypto"},"DOT-USD":{"name":"Polkadot","sector":"Crypto"},
    "LINK-USD":{"name":"Chainlink","sector":"Crypto"},"LTC-USD":{"name":"Litecoin","sector":"Crypto"},
    # Indices
    "^GSPC":{"name":"S&P 500 Index","sector":"US Index"},"^IXIC":{"name":"NASDAQ Composite","sector":"US Index"},
    "^DJI":{"name":"Dow Jones Industrial Average","sector":"US Index"},"^RUT":{"name":"Russell 2000","sector":"US Index"},
    "^VIX":{"name":"CBOE Volatility Index (VIX)","sector":"Volatility"},"^FTSE":{"name":"FTSE 100 (UK)","sector":"Europe Index"},
    "^GDAXI":{"name":"DAX (Germany)","sector":"Europe Index"},"^FCHI":{"name":"CAC 40 (France)","sector":"Europe Index"},
    "^N225":{"name":"Nikkei 225 (Japan)","sector":"Asia Index"},"^HSI":{"name":"Hang Seng (HK)","sector":"Asia Index"},
    "^KS11":{"name":"KOSPI (South Korea)","sector":"Asia Index"},"^NSEI":{"name":"Nifty 50 (India)","sector":"India Index"},
    "^BSESN":{"name":"BSE Sensex (India)","sector":"India Index"},"000001.SS":{"name":"Shanghai Composite","sector":"China Index"},
    "^AXJO":{"name":"ASX 200 (Australia)","sector":"Asia Index"},
}

# Build flat ticker list with region + type metadata for search
def _build_ticker_meta() -> list[dict]:
    meta = {}
    for rname, rdata in _REGIONAL_UNIVERSES.items():
        for cat, tickers in rdata.items():
            for t in tickers:
                if t not in meta:
                    nm = _TICKER_NAME_MAP.get(t, {})
                    meta[t] = {
                        "ticker":  t,
                        "name":    nm.get("name", t),
                        "sector":  nm.get("sector", ""),
                        "type":    cat.rstrip("s").replace("equitie", "stock").replace("indice", "index"),
                        "regions": [rname],
                    }
                elif rname not in meta[t]["regions"]:
                    meta[t]["regions"].append(rname)
    return list(meta.values())

_TICKER_META_LIST: list[dict] = []   # populated on first search call (after universes are defined)


@app.get("/api/v1/market/search")
async def market_search(q: str = "", region: str = "all", asset_type: str = "all", limit: int = 10):
    """Autocomplete search across all universe tickers with live prices."""
    global _TICKER_META_LIST
    if not _TICKER_META_LIST:
        _TICKER_META_LIST = _build_ticker_meta()

    q = q.strip().lower()
    if len(q) < 1:
        return {"results": [], "query": q}

    now = time.time()
    results = []
    for item in _TICKER_META_LIST:
        # Region filter
        if region != "all" and region not in item.get("regions", []):
            continue
        # Asset type filter
        if asset_type != "all":
            itype = item.get("type", "")
            amap  = {
                "stocks": "stock", "etfs": "etf", "etf": "etf",
                "crypto": "crypto", "forex": "forex",
                "commodities": "commodity", "metals": "commodity",
                "indices": "index", "mutual_fund": "mutual_fund",
            }
            wanted = amap.get(asset_type, asset_type)
            if itype != wanted:
                continue
        # Query match on ticker or name
        ticker = item["ticker"]
        name   = item["name"]
        if q not in ticker.lower() and q not in name.lower():
            continue
        # Attach live price
        quote = _LIVE_QUOTE_CACHE.get(ticker.upper(), {})
        results.append({
            **item,
            "price":      quote.get("price"),
            "change_pct": quote.get("change_pct"),
            "positive":   quote.get("positive", True),
            "has_signal": (ticker.upper(), "1d") in _SIGNAL_CACHE,
            "signal_dir": (_SIGNAL_CACHE.get((ticker.upper(), "1d"), ({},))[0].get("direction") if (ticker.upper(),"1d") in _SIGNAL_CACHE else None),
        })
        if len(results) >= limit:
            break

    # Sort: exact ticker match first, then alphabetical
    results.sort(key=lambda x: (0 if x["ticker"].lower().startswith(q) else 1, x["ticker"]))
    return {"results": results[:limit], "query": q}


@app.post("/api/v1/market/set-region")
async def set_user_region(body: dict):
    """Set the user's detected region so the signal warmer prioritizes it."""
    global _USER_REGION
    region = body.get("region", "us").lower()
    if region in _REGIONAL_UNIVERSES:
        _USER_REGION = region
    return {"ok": True, "region": _USER_REGION}


@app.get("/api/v1/market/warmup-status")
async def get_warmup_status():
    """Returns pre-warming progress for quotes and signals, broken down by region."""
    now   = time.time()
    total = len(_ALL_UNIVERSE_TICKERS)
    quotes_ready  = sum(1 for t in _ALL_UNIVERSE_TICKERS
                        if t.upper() in _LIVE_QUOTE_CACHE and _LIVE_QUOTE_CACHE[t.upper()].get("expires", 0) > now)
    signals_ready = sum(1 for t in _ALL_UNIVERSE_TICKERS if _is_signable(t)
                        and (t.upper(), "1d") in _SIGNAL_CACHE and _SIGNAL_CACHE[(t.upper(), "1d")][1] > now)
    total_signable = sum(1 for t in _ALL_UNIVERSE_TICKERS if _is_signable(t))

    # Per-region breakdown
    region_status = {}
    for rname, rdata in _REGIONAL_UNIVERSES.items():
        rtickers = [t for cat in ("equities","etfs","crypto") for t in rdata.get(cat,[]) if _is_signable(t)]
        r_warm   = sum(1 for t in rtickers if (t.upper(),"1d") in _SIGNAL_CACHE and _SIGNAL_CACHE[(t.upper(),"1d")][1] > now)
        region_status[rname] = {"total": len(rtickers), "ready": r_warm, "pct": round(r_warm/len(rtickers)*100) if rtickers else 0}

    # Priority core progress
    priority_signable = [t for t in _PRIORITY_CORE if _is_signable(t)]
    priority_ready    = sum(1 for t in priority_signable
                            if (t.upper(), "1d") in _SIGNAL_CACHE and _SIGNAL_CACHE[(t.upper(), "1d")][1] > now)
    priority_pct      = round(priority_ready / len(priority_signable) * 100) if priority_signable else 0

    quotes = {t: _LIVE_QUOTE_CACHE[t.upper()] for t in _ALL_UNIVERSE_TICKERS if t.upper() in _LIVE_QUOTE_CACHE}
    return {
        "total_tickers":    total,
        "total_signable":   total_signable,
        "quotes_ready":     quotes_ready,
        "signals_ready":    signals_ready,
        "quotes_pct":       round(quotes_ready / total * 100) if total else 0,
        "signals_pct":      round(signals_ready / total_signable * 100) if total_signable else 0,
        "priority_total":   len(priority_signable),
        "priority_ready":   priority_ready,
        "priority_pct":     priority_pct,
        "current_region":   _USER_REGION,
        "region_status":    region_status,
        "quotes":           quotes,
    }


@app.get("/api/v1/market/summary")
async def get_market_summary():
    """Returns live prices, day-change for US, Global, Crypto, Commodities and FX."""
    import time as _time
    if _SUMMARY_CACHE.get("expires", 0) > _time.time():
        return _SUMMARY_CACHE["data"]

    def _sync_fetch():
        import yfinance as _yf
        import pandas as _pd

        sections = {
            "us":     _US_SYMBOLS,
            "global": _GLOBAL_SYMBOLS,
            "assets": _ASSET_SYMBOLS,
            "fx":     _FX_SYMBOLS,
        }

        # Build flat list of all symbols and fetch in one batch call
        all_syms = list(_ALL_MARKET_SYMBOLS.values())
        batch = {}
        try:
            raw = _yf.download(all_syms, period="5d", interval="1d",
                               auto_adjust=True, progress=False, group_by="ticker")
            now = time.time()
            for sym in all_syms:
                try:
                    df = raw[sym] if len(all_syms) > 1 else raw
                    if df is None or (hasattr(df, "empty") and df.empty):
                        continue
                    closes = df["Close"].dropna()
                    if len(closes) < 1:
                        continue
                    price = float(closes.iloc[-1])
                    prev  = float(closes.iloc[-2]) if len(closes) >= 2 else price
                    chg   = price - prev
                    pct   = chg / prev * 100 if prev != 0 else 0.0
                    batch[sym.upper()] = {
                        "price":      round(price, 4),
                        "change":     round(chg, 4),
                        "change_pct": round(pct, 4),
                        "positive":   chg >= 0,
                        "expires":    now + _QUOTE_TTL,
                    }
                    # Also update live quote cache so other endpoints benefit
                    _LIVE_QUOTE_CACHE[sym.upper()] = batch[sym.upper()]
                except Exception:
                    pass
        except Exception as _be:
            logger.warning(f"market/summary batch download failed: {_be}")
            # Fall back to individual fast_info calls
            for sym in all_syms:
                q = _fetch_quote_fast(sym)
                if q:
                    batch[sym.upper()] = q

        res = {}
        for name, syms in sections.items():
            out = []
            for label, sym in syms.items():
                q = batch.get(sym.upper())
                if q:
                    out.append({"label": label, "symbol": sym,
                                "price": q["price"], "change": q["change"],
                                "change_pct": q["change_pct"], "positive": q["positive"]})
            res[name] = out
        return res

    loop = asyncio.get_running_loop()
    try:
        import time as _time2
        data = await asyncio.wait_for(loop.run_in_executor(None, _sync_fetch), timeout=20.0)
        _SUMMARY_CACHE["data"] = data
        _SUMMARY_CACHE["expires"] = _time2.time() + _SUMMARY_TTL
    except (asyncio.TimeoutError, Exception) as _e:
        logger.warning(f"market/summary fetch failed ({_e}), returning cached/empty")
        data = _SUMMARY_CACHE.get("data") or {"us": [], "global": [], "assets": [], "fx": []}

    return data


@app.get("/api/v1/market/news")
async def get_market_news(limit: int = 40):
    """
    Returns recent global market news from RSS feeds + yfinance headlines.
    Sources: MarketWatch, Reuters, CNBC, BBC, Yahoo Finance, WSJ, Bloomberg.
    """
    import time as _time

    if _NEWS_CACHE.get("expires", 0) > _time.time():
        return {"news": _NEWS_CACHE["data"][:limit]}

    def _sync_fetch():
        import yfinance as yf
        import time as _t
        from concurrent.futures import ThreadPoolExecutor

        seen, articles = set(), []

        def fetch_rss(args):
            return _parse_rss(*args, max_items=10)

        with ThreadPoolExecutor(max_workers=6) as ex:
            rss_results = list(ex.map(fetch_rss, _RSS_FEEDS, timeout=10))

        for batch in rss_results:
            for a in batch:
                key = a["title"][:80]
                if key not in seen:
                    seen.add(key)
                    articles.append(a)

        for sym in _NEWS_TICKERS:
            if len(articles) >= limit * 2:
                break
            try:
                news = yf.Ticker(sym).news or []
                for item in news[:4]:
                    content = item.get("content", {}) or {}
                    title = (item.get("title") or content.get("title") or "").strip()
                    if not title:
                        continue
                    key = title[:80]
                    if key in seen:
                        continue
                    seen.add(key)
                    link = (item.get("link") or item.get("url")
                            or content.get("canonicalUrl", {}).get("url", "#"))
                    publisher = (item.get("publisher")
                                 or content.get("provider", {}).get("displayName", "Yahoo Finance"))
                    ts = item.get("providerPublishTime") or item.get("pubDate") or int(_t.time())
                    articles.append({"title": title, "publisher": publisher,
                                     "link": link, "published_at": int(ts), "ticker": sym})
            except Exception:
                continue

        articles.sort(key=lambda x: x.get("published_at", 0), reverse=True)
        return articles

    loop = asyncio.get_running_loop()
    try:
        articles = await asyncio.wait_for(loop.run_in_executor(None, _sync_fetch), timeout=15.0)
        _NEWS_CACHE["data"] = articles
        _NEWS_CACHE["expires"] = _time.time() + _NEWS_TTL
    except (asyncio.TimeoutError, Exception) as _e:
        logger.warning(f"market/news fetch failed ({_e}), returning cached/empty")
        articles = _NEWS_CACHE.get("data", [])

    return {"news": articles[:limit]}


@app.get("/api/v1/market/earnings")
async def get_upcoming_earnings():
    """Returns upcoming earnings dates for major stocks."""
    import time as _t

    if _EARNINGS_CACHE.get("expires", 0) > _t.time():
        return {"earnings": _EARNINGS_CACHE["data"]}

    def _sync_fetch():
        import yfinance as yf
        from datetime import date, timedelta
        import pandas as pd
        from concurrent.futures import ThreadPoolExecutor

        today = date.today()
        cutoff = today + timedelta(days=settings.get("data.earnings_horizon_days", 30))

        def fetch_one(sym):
            try:
                t = yf.Ticker(sym)
                cal = t.calendar
                if cal is None:
                    return None
                if isinstance(cal, pd.DataFrame):
                    cal = cal.to_dict()
                earnings_date = None
                if isinstance(cal, dict):
                    ed = cal.get("Earnings Date") or cal.get("earningsDate")
                    if ed:
                        if isinstance(ed, (list, tuple)) and len(ed) > 0:
                            earnings_date = pd.Timestamp(ed[0]).date()
                        elif hasattr(ed, "date"):
                            earnings_date = ed.date()
                        else:
                            try:
                                earnings_date = pd.Timestamp(str(ed)).date()
                            except Exception:
                                pass
                if earnings_date and today <= earnings_date <= cutoff:
                    fi = t.fast_info
                    q = _fetch_quote_fast(sym)
                    eps_est = cal.get("EPS Estimate") or cal.get("Earnings Average")
                    rev_est = cal.get("Revenue Estimate") or cal.get("Revenue Average")
                    return {
                        "ticker":             sym,
                        "company":            getattr(fi, "company_name", sym) if hasattr(fi, "company_name") else sym,
                        "earnings_date":      str(earnings_date),
                        "days_until":         (earnings_date - today).days,
                        "price":              q["price"] if q else None,
                        "change_pct":         q["change_pct"] if q else None,
                        "positive":           q["positive"] if q else True,
                        "eps_estimate":       round(float(eps_est), 2) if eps_est and str(eps_est) not in ("nan", "None") else None,
                        "revenue_estimate_b": round(float(rev_est) / 1e9, 2) if rev_est and str(rev_est) not in ("nan", "None") else None,
                    }
                return None
            except Exception:
                return None

        results = []
        with ThreadPoolExecutor(max_workers=12) as ex:
            futs = [ex.submit(fetch_one, sym) for sym in _EARNINGS_WATCHLIST]
            for fut in futs:
                try:
                    item = fut.result(timeout=8)
                    if item:
                        results.append(item)
                except Exception:
                    continue

        results.sort(key=lambda x: x["days_until"])
        return results

    loop = asyncio.get_running_loop()
    try:
        results = await asyncio.wait_for(loop.run_in_executor(None, _sync_fetch), timeout=15.0)
        _EARNINGS_CACHE["data"] = results
        _EARNINGS_CACHE["expires"] = _t.time() + _EARNINGS_TTL
    except (asyncio.TimeoutError, Exception) as _e:
        logger.warning(f"market/earnings fetch failed ({_e}), returning cached/empty")
        results = _EARNINGS_CACHE.get("data", [])

    return {"earnings": results}


@app.post("/api/v1/market/quotes")
async def get_quotes(body: dict):
    """
    Live price + day change for a list of tickers (watchlist + ETF sidebar).
    Body: {"tickers": ["AAPL", "MSFT", "NVDA"]}
    """
    from concurrent.futures import ThreadPoolExecutor
    tickers = [t.upper().strip() for t in body.get("tickers", []) if t]
    if not tickers:
        return {"quotes": {}}

    with ThreadPoolExecutor(max_workers=min(len(tickers), 10)) as ex:
        futures = {sym: ex.submit(_fetch_quote_fast, sym) for sym in tickers}
        quotes  = {sym: fut.result(timeout=10) for sym, fut in futures.items()
                   if fut.result(timeout=10) is not None}

    return {"quotes": {sym: q for sym, q in quotes.items() if q}}

@app.get("/api/v1/ticker/summary/{ticker}")
async def get_ticker_summary(ticker: str):
    """
    Returns rich market summary for the left info panel:
    price, change, day range, 52w range, market cap, P/E, beta, sector, volume.
    """
    import yfinance as yf
    import math

    t = yf.Ticker(ticker.upper())
    q = _fetch_quote_fast(ticker.upper()) or {}

    # fast_info for live data
    fi = t.fast_info
    # info dict for fundamentals (may be slow/cached)
    try:
        info = t.info or {}
    except Exception:
        info = {}

    def safe(val, fmt=None, divisor=1):
        try:
            v = float(val) / divisor
            if math.isnan(v) or math.isinf(v):
                return None
            return round(v, fmt) if fmt is not None else v
        except Exception:
            return None

    market_cap = safe(getattr(fi, "market_cap", None) or info.get("marketCap"), 0, 1)
    def fmt_cap(mc):
        if mc is None: return None
        if mc >= 1e12: return f"${mc/1e12:.2f}T"
        if mc >= 1e9:  return f"${mc/1e9:.2f}B"
        if mc >= 1e6:  return f"${mc/1e6:.2f}M"
        return f"${mc:,.0f}"

    return {
        "ticker":        ticker.upper(),
        "name":          info.get("longName") or info.get("shortName") or ticker.upper(),
        "sector":        info.get("sector") or info.get("quoteType", ""),
        "industry":      info.get("industry", ""),
        "exchange":      info.get("exchange", ""),
        "currency":      info.get("currency", "USD"),
        "price":         q.get("price"),
        "change":        q.get("change"),
        "change_pct":    q.get("change_pct"),
        "positive":      q.get("positive", True),
        "day_high":      q.get("day_high"),
        "day_low":       q.get("day_low"),
        "year_high":     safe(getattr(fi, "year_high", None) or info.get("fiftyTwoWeekHigh")),
        "year_low":      safe(getattr(fi, "year_low",  None) or info.get("fiftyTwoWeekLow")),
        "volume":        safe(getattr(fi, "last_volume", None) or info.get("regularMarketVolume"), 0),
        "avg_volume":    safe(info.get("averageVolume"), 0),
        "market_cap":    market_cap,
        "market_cap_fmt":fmt_cap(market_cap),
        "pe_trailing":   safe(info.get("trailingPE"), 2),
        "pe_forward":    safe(info.get("forwardPE"), 2),
        "eps_ttm":       safe(info.get("trailingEps"), 2),
        "dividend_yield":safe(info.get("dividendYield"), 4),
        "beta":          safe(info.get("beta"), 2),
        "shares_out":    safe(info.get("sharesOutstanding"), 0),
        "float_shares":  safe(info.get("floatShares"), 0),
        "short_pct":     safe(info.get("shortPercentOfFloat"), 4),
        "analyst_target":safe(info.get("targetMeanPrice"), 2),
        "recommendation":info.get("recommendationKey", ""),
        "description":   (info.get("longBusinessSummary") or "")[:300],
    }


@app.get("/api/v1/ticker/news/{ticker}")
async def get_ticker_news(ticker: str, limit: int = 10):
    """
    Returns top 10 news items for a ticker with focus on institutional,
    central bank, country-level, and major investor coverage.
    Scored by relevance: mentions of institutions, banks, sovereign funds, countries.
    """
    import yfinance as yf
    import time as _time
    import urllib.request, xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime

    INST_KEYWORDS = [
        # Central banks & monetary policy
        "federal reserve","fed","ecb","bank of japan","boj","pboc","bank of england",
        "central bank","monetary policy","interest rate","rate hike","rate cut","quantitative",
        # Major institutions & funds
        "blackrock","vanguard","fidelity","state street","jpmorgan","goldman sachs",
        "morgan stanley","warren buffett","berkshire","bridgewater","citadel","renaissance",
        "sovereign wealth","pension fund","hedge fund","institutional","endowment",
        # Country-level
        "china","united states","european union","japan","germany","united kingdom",
        "saudi arabia","norway","singapore","canada","australia","india",
        # Actions
        "stake","holding","position","acquires","divests","increases","reduces",
        "investment","buys","sells","portfolio","allocation","exposure",
        # Macro
        "inflation","recession","gdp","earnings","guidance","outlook","forecast",
    ]

    seen, articles = set(), []

    # 1. yfinance news
    try:
        news = yf.Ticker(ticker.upper()).news or []
        for item in news[:20]:
            content = item.get("content") or {}
            title = (item.get("title") or content.get("title") or "").strip()
            if not title or title[:60] in seen:
                continue
            seen.add(title[:60])
            link = (item.get("link") or item.get("url")
                    or content.get("canonicalUrl", {}).get("url", "#"))
            publisher = (item.get("publisher")
                         or content.get("provider", {}).get("displayName", "Yahoo Finance"))
            ts = int(item.get("providerPublishTime") or item.get("pubDate") or _time.time())
            title_lower = title.lower()
            score = sum(1 for kw in INST_KEYWORDS if kw in title_lower)
            articles.append({"title": title, "publisher": publisher, "link": link,
                              "published_at": ts, "score": score, "ticker": ticker.upper()})
    except Exception:
        pass

    # 2. Yahoo Finance RSS for this ticker
    rss_url = f"https://finance.yahoo.com/rss/headline?s={ticker.upper()}&region=US&lang=en-US"
    try:
        req = urllib.request.Request(rss_url, headers={"User-Agent": "Mozilla/5.0 AlphaAgent/2.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            raw = resp.read()
        root = ET.fromstring(raw)
        for item in root.findall(".//item")[:15]:
            title = (item.findtext("title") or "").strip()
            if not title or title[:60] in seen:
                continue
            seen.add(title[:60])
            link = (item.findtext("link") or "#").strip()
            pub = item.findtext("pubDate") or ""
            try:
                ts = int(parsedate_to_datetime(pub).timestamp())
            except Exception:
                ts = int(_time.time())
            title_lower = title.lower()
            score = sum(1 for kw in INST_KEYWORDS if kw in title_lower)
            articles.append({"title": title, "publisher": "Yahoo Finance RSS",
                              "link": link, "published_at": ts,
                              "score": score, "ticker": ticker.upper()})
    except Exception:
        pass

    # Sort: institutional-relevance first, then recency
    articles.sort(key=lambda x: (x.get("score", 0) * 1e9 + x.get("published_at", 0)), reverse=True)
    for a in articles:
        a.pop("score", None)

    return {"news": articles[:limit], "ticker": ticker.upper()}


@app.get("/api/v1/ticker/history/{ticker}")
async def get_ticker_history(ticker: str, period: str = "3mo"):
    """
    Returns OHLCV candlestick data for rendering charts client-side.
    Period: 1d (5m), 5d (1h), 1mo, 3mo, 6mo, 1y, 2y
    """
    import yfinance as yf
    import pandas as pd
    # Map period → yfinance interval
    period_interval = {
        "1d":  ("1d",  "5m"),
        "5d":  ("5d",  "1h"),
        "1mo": ("1mo", "1d"),
        "3mo": ("3mo", "1d"),
        "6mo": ("6mo", "1d"),
        "1y":  ("1y",  "1d"),
        "2y":  ("2y",  "1d"),
    }
    if period not in period_interval:
        period = "3mo"
    yf_period, yf_interval = period_interval[period]
    try:
        t = yf.Ticker(ticker.upper())
        hist = t.history(period=yf_period, interval=yf_interval, auto_adjust=True)
        if hist.empty:
            raise HTTPException(status_code=404, detail="No price history available")

        candles = []
        for ts, row in hist.iterrows():
            o = row.get("Open");  h = row.get("High")
            l = row.get("Low");   c = row.get("Close")
            v = row.get("Volume", 0)
            if any(pd.isna(x) for x in [o, h, l, c]):
                continue
            candles.append({
                "time":   int(ts.timestamp()),
                "open":   round(float(o), 4),
                "high":   round(float(h), 4),
                "low":    round(float(l), 4),
                "close":  round(float(c), 4),
                "volume": int(v) if not pd.isna(v) else 0,
            })

        return {"ticker": ticker.upper(), "period": period, "candles": candles}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _build_action_summary(
    ticker: str,
    direction: str,
    probability: float,
    conviction: float,
    entropy: float,
    multiplier: float,
    agents: list,
    warnings: list,
    holding_block: dict | None,
) -> dict:
    """
    Synthesises all agent outputs into a plain-English action summary.
    No LLM — pure rule-based synthesis from structured signal data.
    """
    prob_pct   = round(probability * 100, 1)
    conv_pct   = round(conviction, 1)
    dir_up     = direction == "LONG"
    dir_dn     = direction == "SHORT"
    is_neutral = direction in ("NEUTRAL", "HOLD")

    # ── Conviction label ─────────────────────────────────────────────────────
    if conv_pct >= 60:   conv_label = "Very High"
    elif conv_pct >= 40: conv_label = "High"
    elif conv_pct >= 25: conv_label = "Moderate"
    elif conv_pct >= 10: conv_label = "Low"
    else:                conv_label = "Very Low"

    # ── Agent vote tally ─────────────────────────────────────────────────────
    long_agents  = [a.agent_name for a in agents if getattr(a, "vote", "") == "LONG"]
    short_agents = [a.agent_name for a in agents if getattr(a, "vote", "") == "SHORT"]
    hold_agents  = [a.agent_name for a in agents if getattr(a, "vote", "") in ("HOLD", "NEUTRAL")]
    n_total = len(agents) or 1

    # ── Agent-specific insight extraction ────────────────────────────────────
    agent_map = {a.agent_name: a for a in agents}

    def agent_reasoning(name: str, max_chars: int = 120) -> str:
        a = agent_map.get(name)
        if not a: return ""
        r = (a.reasoning or "").strip()
        return r[:max_chars] + ("…" if len(r) > max_chars else "")

    def top_factor(name: str) -> tuple[str, float] | tuple[None, None]:
        """Returns (factor_interpretation, score) for the highest-scoring factor."""
        a = agent_map.get(name)
        if not a or not a.factor_scores: return None, None
        best = max(a.factor_scores.values(), key=lambda f: f.score)
        return best.interpretation, best.score

    def worst_factor(name: str) -> tuple[str, float] | tuple[None, None]:
        a = agent_map.get(name)
        if not a or not a.factor_scores: return None, None
        worst = min(a.factor_scores.values(), key=lambda f: f.score)
        return worst.interpretation, worst.score

    # ── Build bull / bear cases ───────────────────────────────────────────────
    bull_points, bear_points = [], []

    for ag_name in ["technical", "fundamental", "macro", "volatility",
                    "risk", "sentiment", "geopolitical", "currency", "insider"]:
        a = agent_map.get(ag_name)
        if not a: continue
        p = getattr(a, "probability_up", 0.5)
        interp, sc = top_factor(ag_name)
        worst_i, worst_sc = worst_factor(ag_name)

        if p >= 0.60 and interp:
            bull_points.append(f"**{ag_name.title()}**: {interp}")
        elif p <= 0.40 and interp:
            bear_points.append(f"**{ag_name.title()}**: {worst_i or interp}")

    # Cap at 4 each
    bull_points = bull_points[:4]
    bear_points = bear_points[:4]

    # ── Headline ─────────────────────────────────────────────────────────────
    if dir_up:
        headline = f"{ticker} — {conv_label} conviction LONG signal ({prob_pct}% probability up)"
    elif dir_dn:
        headline = f"{ticker} — {conv_label} conviction SHORT signal ({prob_pct}% probability down)"
    else:
        headline = f"{ticker} — Agents are mixed ({prob_pct}% probability up, {conv_label} conviction)"

    # ── Action recommendation ─────────────────────────────────────────────────
    size_pct = round(multiplier * 10, 1)   # rough portfolio % from multiplier
    if is_neutral or conv_pct < 15:
        action = "WAIT — conviction is too low. Monitor for a clearer setup before committing capital."
        action_color = "yellow"
    elif dir_up:
        action = (
            f"Consider a LONG position (~{size_pct}% of portfolio). "
            f"{len(long_agents)}/{n_total} agents are bullish."
        )
        action_color = "green"
    else:
        action = (
            f"Consider a SHORT position (~{size_pct}% of portfolio). "
            f"{len(short_agents)}/{n_total} agents are bearish."
        )
        action_color = "red"

    # ── Entry timing note ────────────────────────────────────────────────────
    tech = agent_map.get("technical")
    entry_notes = []
    if tech and tech.factor_scores:
        fs = tech.factor_scores
        if "rsi" in fs:
            rsi_score = fs["rsi"].score
            if rsi_score >= 70:   entry_notes.append("RSI oversold — good entry window")
            elif rsi_score <= 30: entry_notes.append("RSI overbought — wait for pullback")
        if "macd" in fs and fs["macd"].score >= 70:
            entry_notes.append("MACD positive crossover confirmed")
        if "ema200_trend" in fs and fs["ema200_trend"].score >= 70:
            entry_notes.append("Price above EMA-200 (long-term uptrend)")
        elif "ema200_trend" in fs and fs["ema200_trend"].score <= 30:
            entry_notes.append("Price below EMA-200 — caution on longs")
    if not entry_notes:
        entry_notes.append("Check intraday chart before entry")

    # ── Risk / stop guidance ──────────────────────────────────────────────────
    risk_notes = []
    risk_agent = agent_map.get("risk")
    if risk_agent and risk_agent.factor_scores:
        if "drawdown_ath" in risk_agent.factor_scores:
            risk_notes.append(risk_agent.factor_scores["drawdown_ath"].interpretation)
        if "tail_ratio" in risk_agent.factor_scores:
            risk_notes.append(risk_agent.factor_scores["tail_ratio"].interpretation)
    if entropy > 0.8:
        risk_notes.append(f"High signal entropy ({entropy:.2f}) — agents strongly disagree, size smaller")
    for w in warnings[:2]:
        risk_notes.append(w)
    risk_notes = risk_notes[:3]

    # ── Hold / exit guidance ──────────────────────────────────────────────────
    if holding_block:
        min_d = holding_block["optimal_hold_min"]
        max_d = holding_block["optimal_hold_max"]
        ex    = holding_block["expiry_max"]
        hold_guidance = (
            f"Hold {min_d}–{max_d} days. "
            f"{'Mean-reverting — exit by ' + ex if holding_block['half_life_days'] > 0 else 'Trending — use trailing stop, no fixed expiry.'}"
        )
    else:
        hold_guidance = "Holding period data unavailable — use standard risk management."

    # ── Agents agreeing / disagreeing ────────────────────────────────────────
    consensus = (
        f"{len(long_agents)} LONG, {len(short_agents)} SHORT, {len(hold_agents)} HOLD "
        f"out of {n_total} agents"
    )

    return {
        "headline":       headline,
        "action":         action,
        "action_color":   action_color,
        "conviction_label": conv_label,
        "bull_case":      bull_points,
        "bear_case":      bear_points,
        "entry_notes":    entry_notes,
        "risk_notes":     risk_notes,
        "hold_guidance":  hold_guidance,
        "consensus":      consensus,
        "agents_long":    long_agents,
        "agents_short":   short_agents,
        "agents_hold":    hold_agents,
    }


# Agent weight multipliers per horizon. Keys are agent names; higher = more influential.
_HORIZON_WEIGHTS: dict[str, dict[str, float]] = {
    "1d": {"technical": 3.0, "volatility": 3.0, "sentiment": 2.0, "macro": 0.5,
           "fundamental": 0.2, "insider": 0.5, "currency": 0.5, "geopolitical": 0.3},
    "1w": {"technical": 2.5, "volatility": 2.5, "sentiment": 1.8, "macro": 0.8,
           "fundamental": 0.3, "insider": 0.8, "currency": 0.8, "geopolitical": 0.5},
    "1m": {"technical": 1.0, "volatility": 1.0, "sentiment": 1.0, "macro": 1.0,
           "fundamental": 1.0, "insider": 1.0, "currency": 1.0, "geopolitical": 1.0},
    "3m": {"technical": 0.5, "volatility": 0.5, "sentiment": 0.7, "macro": 1.5,
           "fundamental": 2.0, "insider": 2.0, "currency": 1.2, "geopolitical": 1.5},
    "6m": {"technical": 0.3, "volatility": 0.3, "sentiment": 0.5, "macro": 2.0,
           "fundamental": 2.5, "insider": 2.5, "currency": 1.5, "geopolitical": 2.0},
    "1y": {"technical": 0.2, "volatility": 0.2, "sentiment": 0.3, "macro": 2.5,
           "fundamental": 3.0, "insider": 3.0, "currency": 2.0, "geopolitical": 2.5},
}

def _apply_horizon_weights(agent_results: list, horizon: str) -> float:
    """Reblend agent probability_up values using horizon-specific weights."""
    weights = _HORIZON_WEIGHTS.get(horizon, _HORIZON_WEIGHTS["1m"])
    weighted_sum = 0.0
    weight_total = 0.0
    for res in agent_results:
        name = getattr(res, "agent_name", None) or (res.get("agent_name") if isinstance(res, dict) else None)
        if not name or name == "risk":  # risk is circuit-breaker, exclude from blend
            continue
        p = getattr(res, "probability_up", None)
        if p is None and isinstance(res, dict):
            p = res.get("probability_up", 0.5)
        p = float(p or 0.5)
        w = weights.get(name, 1.0)
        weighted_sum += w * p
        weight_total += w
    if weight_total == 0:
        return 0.5
    return max(0.02, min(0.98, weighted_sum / weight_total))


@app.get("/api/v1/signal/{ticker}")
async def get_signal(ticker: str, horizon: str = "1m"):
    """
    Executes the full multi-agent orchestrator for a given ticker.
    horizon: 1d | 1w | 1m | 3m | 6m | 1y  — reweights agent blend for the chosen timeframe.
    """
    horizon = horizon.lower().strip()
    if horizon not in _HORIZON_WEIGHTS:
        horizon = "1m"

    start_time = time.time()
    logger.info(f"Received signal request for: {ticker} (horizon={horizon})")

    # Return cached result if available (avoids full re-analysis within 5 min)
    _cache_key = (ticker.upper(), horizon)
    _cached = _SIGNAL_CACHE.get(_cache_key)
    if _cached and time.time() < _cached[1]:
        cached_result = _cached[0].copy()
        cached_result["cached"] = True
        cached_result["latency_ms"] = 0.1
        _cq = _LIVE_QUOTE_CACHE.get(ticker.upper())
        if _cq:
            cached_result["current_price"] = _cq.get("price")
            cached_result["change_pct"]    = _cq.get("change_pct")
        logger.info(f"[{ticker}] Returning cached signal (horizon={horizon})")
        return cached_result

    try:
        md = _get_market_data(ticker)
        initial_state = {
            "ticker": ticker,
            "market_data": md,
            "registry": registry
        }

        # Run graph.invoke in a thread pool with an OUTER hard timeout so
        # rate-limited yfinance can never hang the request indefinitely.
        loop = asyncio.get_running_loop()
        try:
            result_state = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: graph.invoke(initial_state)),
                timeout=90.0,
            )
        except asyncio.TimeoutError:
            logger.warning(f"[{ticker}] Signal endpoint hit 90s outer timeout")
            raise HTTPException(
                status_code=504,
                detail=(
                    f"Signal generation for {ticker} exceeded 90s budget. "
                    "yfinance may be rate-limited — try again shortly."
                ),
            )

        final_info = result_state["final_signal"]
        packet = final_info["packet"]
        
        # Enrich packet with metadata
        latency_ms = (time.time() - start_time) * 1000
        packet.computation_time_ms = latency_ms
        packet.agents_used = len(packet.agent_results)

        # ── Horizon-weighted probability blend ──────────────────────────────
        base_prob = final_info["probability_up"]
        # Always apply horizon-specific agent reblending (including 1m which uses equal weights).
        # Anchored 80/20 toward the horizon blend to keep Bayesian direction signal.
        horizon_prob = _apply_horizon_weights(packet.agent_results, horizon)
        horizon_prob = 0.8 * horizon_prob + 0.2 * base_prob

        # Derive direction from horizon probability
        if horizon_prob >= 0.53:
            horizon_direction = "LONG"
        elif horizon_prob <= 0.47:
            horizon_direction = "SHORT"
        else:
            horizon_direction = "NEUTRAL"

        # Record metrics
        Monitor.record_request(latency_ms, success=True)
        for res in packet.agent_results:
            Monitor.record_agent_run(res.agent_name, res.computation_time_ms, success=True)

        # ── Persistent Storage ──
        signal_data = {
            "direction": horizon_direction,
            "probability": horizon_prob,
            "conviction": packet.conviction_pct,
            "multiplier": final_info.get("multiplier", 1.0)
        }
        trade_id = db_manager.record_signal(ticker, signal_data, packet.agent_results, horizon=horizon)

        # Build holding period / signal validity block
        hp = packet.holding_period
        from datetime import date, timedelta
        today = date.today()
        holding_block = None
        if hp:
            expiry_min = (today + timedelta(days=hp.optimal_hold_min)).isoformat()
            expiry_max = (today + timedelta(days=hp.optimal_hold_max)).isoformat()
            holding_block = {
                "half_life_days":   hp.half_life_days,
                "optimal_hold_min": hp.optimal_hold_min,
                "optimal_hold_max": hp.optimal_hold_max,
                "signal_strength":  round(hp.signal_strength, 3),
                "expiry_min":       expiry_min,
                "expiry_max":       expiry_max,
                "generated_on":     today.isoformat(),
            }

        summary = _build_action_summary(
            ticker      = ticker,
            direction   = horizon_direction,
            probability = horizon_prob,
            conviction  = packet.conviction_pct,
            entropy     = final_info.get("entropy", 0.0),
            multiplier  = final_info.get("multiplier", 1.0),
            agents      = packet.agent_results,
            warnings    = packet.warnings,
            holding_block = holding_block,
        )

        _quote = _LIVE_QUOTE_CACHE.get(ticker.upper()) or _fetch_quote_fast(ticker.upper()) or {}
        raw_dict = {
            "trade_id":       trade_id,
            "ticker":         ticker,
            "horizon":        horizon,
            "direction":      horizon_direction,
            "probability":    round(horizon_prob, 4),
            "base_probability": round(base_prob, 4),
            "conviction":     packet.conviction_pct,
            "multiplier":     final_info.get("multiplier", 1.0),
            "entropy":        final_info.get("entropy", 0.0),
            "agreement_score": final_info.get("agreement_score", 0.0),
            "market_regime":  final_info.get("market_regime", "UNKNOWN"),
            "council":        final_info.get("council", []),
            "agents":         packet.agent_results,
            "warnings":       packet.warnings,
            "holding_period": holding_block,
            "summary":        summary,
            "latency_ms":     round(latency_ms, 1),
            "cached":         False,
            "current_price":  _quote.get("price"),
            "change_pct":     _quote.get("change_pct"),
        }
        # Convert Pydantic models → plain dicts, then sanitise NaN/Inf so the
        # browser's JSON.parse() doesn't choke on bare NaN/Infinity tokens.
        result_dict = _nan_safe(jsonable_encoder(raw_dict))
        _SIGNAL_CACHE[_cache_key] = (result_dict, time.time() + _SIGNAL_TTL)
        return result_dict
        
    except Exception as e:
        logger.error(f"Signal generation failed for {ticker}: {e}", exc_info=True)
        Monitor.record_request(0, success=False)
        raise HTTPException(status_code=500, detail=str(e))

async def _call_ollama(messages: list, system: str = "") -> str:
    """Call local Ollama model. Returns text response."""
    import json as _jj, urllib.request as _uu
    host  = settings.get("ollama.host", "http://localhost")
    port  = settings.get("ollama.port", 11435)
    model = settings.get("ollama.model", "mistral:latest")
    msgs  = ([{"role": "system", "content": system}] if system else []) + list(messages)
    payload = _jj.dumps({"model": model, "messages": msgs, "stream": False}).encode()
    req = _uu.Request(f"{host}:{port}/api/chat", data=payload,
                      headers={"Content-Type": "application/json"})
    loop = asyncio.get_running_loop()
    def _fetch():
        with _uu.urlopen(req, timeout=120) as r:
            return _jj.loads(r.read())
    data = await loop.run_in_executor(None, _fetch)
    return (data.get("message") or {}).get("content", "")


async def _call_gemini(messages: list, system: str = "", max_tokens: int = 700) -> str:
    """Call Google Gemini via google-genai SDK. Returns text response."""
    import os as _os
    gemini_key = _os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY not set")
    from google import genai as _genai
    from google.genai import types as _gtypes
    loop = asyncio.get_running_loop()
    def _fetch():
        client = _genai.Client(api_key=gemini_key)
        contents = []
        for h in messages:
            role = "user" if h.get("role") == "user" else "model"
            contents.append(_gtypes.Content(role=role, parts=[_gtypes.Part(text=h.get("content", ""))]))
        cfg = _gtypes.GenerateContentConfig(
            system_instruction=system or None,
            max_output_tokens=max_tokens,
            temperature=0.7,
        )
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=cfg,
        )
        # resp.text may raise if no candidates — fall back gracefully
        try:
            return resp.text or ""
        except Exception:
            if resp.candidates:
                parts = resp.candidates[0].content.parts
                return "".join(p.text for p in parts if hasattr(p, "text")) or ""
            return ""
    try:
        return await asyncio.wait_for(loop.run_in_executor(None, _fetch), timeout=20.0)
    except asyncio.TimeoutError:
        raise TimeoutError("Gemini API timed out after 20s")


def _build_market_context_block() -> str:
    """Assemble a market-context string from AlphaAgent's live in-memory caches.
    Used to enrich every AI chat system prompt with real-time data server-side."""
    lines: list[str] = []

    # ── Live market prices from market-summary cache ─────────────────────────
    mkt = _SUMMARY_CACHE.get("data") or {}
    for section, label in [("us", "US Markets"), ("global", "Global"), ("assets", "Crypto/Commodities"), ("fx", "FX")]:
        items = mkt.get(section) or []
        if items:
            parts = [
                f"{i['label']} {'▲' if i.get('positive') else '▼'}{abs(i.get('change_pct', 0)):.2f}%"
                for i in items
            ]
            lines.append(f"{label}: {' | '.join(parts)}")

    # ── Top AlphaAgent signals from in-memory cache ───────────────────────────
    now_ts = time.time()
    sig_rows: list[tuple[str, str, float, float]] = []
    for (sym, hz), sig in list(_SIGNAL_CACHE.items()):
        if hz != "1m":
            continue
        sig_rows.append((
            sym,
            sig.get("direction", "NEUTRAL"),
            sig.get("probability_up", 0.5) * 100,
            sig.get("conviction", 0),
        ))
    sig_rows.sort(key=lambda x: -x[3])
    longs  = [f"{s}({p:.0f}%conv={c:.0f})" for s, d, p, c in sig_rows if d == "LONG"][:8]
    shorts = [f"{s}({p:.0f}%)" for s, d, p, c in sig_rows if d == "SHORT"][:5]
    if longs:
        lines.append(f"AlphaAgent LONG signals (conviction): {', '.join(longs)}")
    if shorts:
        lines.append(f"AlphaAgent SHORT signals: {', '.join(shorts)}")

    # ── Live quote snapshot for popular tickers ───────────────────────────────
    _POPULAR = ["AAPL", "NVDA", "MSFT", "TSLA", "AMZN", "META", "GOOGL", "AMD",
                "SPY", "QQQ", "BTC-USD", "ETH-USD", "GC=F", "CL=F"]
    quote_parts = []
    for sym in _POPULAR:
        q = _LIVE_QUOTE_CACHE.get(sym)
        if q and q.get("expires", 0) > now_ts:
            chg = q.get("change_pct", 0)
            quote_parts.append(f"{sym} ${q['price']:.2f}({'+'if chg>=0 else ''}{chg:.2f}%)")
    if quote_parts:
        lines.append(f"Live quotes: {' | '.join(quote_parts)}")

    return "\n".join(lines) if lines else ""


def _build_platform_context(
    region: str | None = None,
    include_signals: bool = True,
    include_portfolio: bool = True,
    include_quotes: bool = True,
    include_leaderboard: bool = False,
    include_backtests: bool = False,
    max_signals_per_region: int = 6,
) -> str:
    """
    Unified platform-state context block for ALL chat endpoints.

    Combines: live market regime, top cached AlphaAgent signals organised by
    region, committed AI portfolio, live quotes, optional agent leaderboard
    and recent backtests. Every chat request can pass this verbatim to Gemini
    so the model "sees" the full platform state, not just one ticker.
    """
    blocks: list[str] = []
    now_ts = time.time()

    # ── 1. Market regime (HMM + VIX) ─────────────────────────────────────────
    try:
        import yfinance as _yfp
        _spy_p = _yfp.Ticker("SPY").history(period="30d")["Close"].dropna()
        _vix_p = float(_yfp.Ticker("^VIX").history(period="3d")["Close"].dropna().iloc[-1])
        if len(_spy_p) >= 20:
            _sma20 = float(_spy_p.rolling(20).mean().iloc[-1])
            _now   = float(_spy_p.iloc[-1])
            _1d    = float((_now / _spy_p.iloc[-2] - 1) * 100) if len(_spy_p) >= 2 else 0.0
            _5d    = float((_now / _spy_p.iloc[-6] - 1) * 100) if len(_spy_p) >= 6 else 0.0
            _lbl   = ("EXTREME" if _vix_p > 35 else "HIGH" if _vix_p > 25 else
                      "ELEVATED" if _vix_p > 18 else "CALM")
            blocks.append(
                f"MARKET REGIME: {_lbl} | VIX={_vix_p:.1f} | SPY={_now:.2f} "
                f"({'above' if _now > _sma20 else 'below'} 20-SMA={_sma20:.2f}) | "
                f"SPY 1d={_1d:+.2f}% / 5d={_5d:+.2f}%"
            )
    except Exception:
        pass

    # ── 2. Top cached AlphaAgent signals, region-organised ──────────────────
    if include_signals:
        # Build region → ticker membership map
        region_lookup: dict[str, str] = {}
        for rname, rdata in _REGIONAL_UNIVERSES.items():
            for cat in ("equities", "etfs", "crypto"):
                for tk in rdata.get(cat, []):
                    region_lookup.setdefault(tk.upper(), rname)

        per_region: dict[str, list] = {}
        for (sym, hz), entry in list(_SIGNAL_CACHE.items()):
            if hz != "1d":
                continue
            sig, expires = entry
            if now_ts > expires:
                continue
            direction = sig.get("direction", "NEUTRAL")
            if direction not in ("LONG", "SHORT"):
                continue
            r_owner = region_lookup.get(sym.upper(), "global")
            per_region.setdefault(r_owner, []).append((
                sym, direction,
                float(sig.get("probability", 0.5)) * 100,
                float(sig.get("conviction", 0)),
            ))

        # Filter to requested region if specified
        if region and region.lower() != "all" and region.lower() in per_region:
            per_region = {region.lower(): per_region[region.lower()]}

        sig_lines: list[str] = []
        for rname in sorted(per_region.keys()):
            rows = sorted(per_region[rname], key=lambda x: -x[3])[:max_signals_per_region]
            if not rows:
                continue
            parts = [
                f"{sym}({d[0]} P={p:.0f}% conv={c:.0f})"
                for sym, d, p, c in rows
            ]
            sig_lines.append(f"  {rname.upper()}: {' | '.join(parts)}")
        if sig_lines:
            blocks.append("TOP CACHED SIGNALS (by region):\n" + "\n".join(sig_lines))

    # ── 3. Committed AI portfolio (if any) ──────────────────────────────────
    if include_portfolio:
        ai_port = _build_ai_portfolio_block()
        if ai_port and ai_port != "(no AI portfolio committed)":
            blocks.append("AI PORTFOLIO:\n" + ai_port)

    # ── 4. Live quote snapshot for popular tickers ──────────────────────────
    if include_quotes:
        pop = ["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "TSLA",
               "BTC-USD", "ETH-USD", "GC=F", "CL=F"]
        qparts = []
        for sym in pop:
            q = _LIVE_QUOTE_CACHE.get(sym)
            if q and q.get("expires", 0) > now_ts:
                chg = q.get("change_pct", 0)
                qparts.append(f"{sym} ${q['price']:.2f}({'+' if chg>=0 else ''}{chg:.2f}%)")
        if qparts:
            blocks.append("LIVE QUOTES: " + " | ".join(qparts))

    # ── 5. Optional: agent leaderboard ──────────────────────────────────────
    if include_leaderboard:
        try:
            lb_data = leaderboard.get_top_agents(window_days=30) if 'leaderboard' in globals() else None
            if lb_data:
                rows = [f"  {r.get('agent_name','?')}: IC={r.get('ic',0):.3f} acc={r.get('accuracy',0):.1%}"
                        for r in lb_data[:6]]
                blocks.append("AGENT LEADERBOARD (30d):\n" + "\n".join(rows))
        except Exception:
            pass

    # ── 6. Optional: recent backtests ───────────────────────────────────────
    if include_backtests:
        try:
            recent = db_manager.get_recent_backtests(limit=3) if hasattr(db_manager, "get_recent_backtests") else None
            if recent:
                rows = [f"  {bt.get('name','?')}: return={bt.get('return_pct',0):.2f}% Sharpe={bt.get('sharpe',0):.2f}"
                        for bt in recent]
                blocks.append("RECENT BACKTESTS:\n" + "\n".join(rows))
        except Exception:
            pass

    # ── 7. User region context ──────────────────────────────────────────────
    blocks.append(f"USER REGION: {_USER_REGION.upper()}")

    return "\n\n".join(blocks)


def _detect_invest_intent(message: str) -> dict | None:
    """
    Detect 'invest $X by [date] and tell me profit' intent from a chat message.

    Returns dict with parsed budget + exit timeline + region hint, or None.
    Handles: "invest 100k", "$50,000 by June 15", "put 25k into india tech by next month"
    """
    import re as _re
    from datetime import date as _date, timedelta as _td

    m = (message or "").lower().strip()
    if not m:
        return None

    # Budget extraction
    budget = None
    m_k = _re.search(r'\$?\s*(\d[\d,]*\.?\d*)\s*k\b', m)
    if m_k:
        budget = float(m_k.group(1).replace(',', '')) * 1000
    else:
        m_d = _re.search(r'\$\s*(\d[\d,]+)', message)
        if m_d:
            budget = float(m_d.group(1).replace(',', ''))
        else:
            m_w = _re.search(r'\b(\d[\d,]{3,})\b', message)
            if m_w:
                budget = float(m_w.group(1).replace(',', ''))

    if budget is None or budget < 100:
        return None

    invest_words = ("invest", "put", "deploy", "allocate", "buy", "create portfolio",
                    "build portfolio", "$100k", "100k", "50k", "25k")
    if not any(w in m for w in invest_words):
        return None

    # Exit timeline parsing
    today = _date.today()
    exit_date = None
    exit_days = None
    if "tomorrow" in m or "next day" in m or "end of day" in m or "eod" in m:
        exit_days = 1
    elif "next week" in m or "in a week" in m or "7 days" in m or "one week" in m:
        exit_days = 7
    elif "next month" in m or "in a month" in m or "30 days" in m or "one month" in m:
        exit_days = 30
    elif "3 months" in m or "quarter" in m or "90 days" in m:
        exit_days = 90
    elif "6 months" in m or "half year" in m or "180 days" in m:
        exit_days = 180
    elif "one year" in m or "1 year" in m or "annual" in m or "12 months" in m:
        exit_days = 365
    else:
        d_match = _re.search(r'(\d{1,3})\s*(?:days?|day)', m)
        if d_match:
            exit_days = int(d_match.group(1))
        w_match = _re.search(r'(\d{1,3})\s*(?:weeks?|week)', m)
        if w_match:
            exit_days = int(w_match.group(1)) * 7
        mo_match = _re.search(r'(\d{1,3})\s*(?:months?|month)', m)
        if mo_match:
            exit_days = int(mo_match.group(1)) * 30
        # ISO date match
        iso = _re.search(r'(\d{4}-\d{2}-\d{2})', message)
        if iso:
            exit_date = iso.group(1)
            try:
                exit_days = (_date.fromisoformat(exit_date) - today).days
            except Exception:
                exit_days = None

    if exit_days is not None and exit_date is None:
        exit_date = (today + _td(days=exit_days)).isoformat()

    # Region hint
    region_hint = None
    if any(k in m for k in ("india", "nifty", "sensex", ".ns")):
        region_hint = "india"
    elif any(k in m for k in ("europe", "european", "dax", "ftse")):
        region_hint = "europe"
    elif any(k in m for k in ("japan", "nikkei", "tokyo")):
        region_hint = "japan"
    elif any(k in m for k in ("china", "shanghai", "hk", "hong kong")):
        region_hint = "china"
    elif any(k in m for k in ("asia", "asian", "hang seng")):
        region_hint = "asia"
    elif any(k in m for k in ("global", "international", "world", "diversified")):
        region_hint = "global"
    else:
        region_hint = _USER_REGION or "us"

    # Mode hint
    mode_hint = None
    if "aggressive" in m or "concentrated" in m or "sniper" in m or "high conviction" in m:
        mode_hint = "CONCENTRATED"
    elif "diversified" in m or "spread" in m or "many stocks" in m or "safe" in m:
        mode_hint = "DIVERSIFIED"
    elif "balanced" in m:
        mode_hint = "BALANCED"

    return {
        "budget":     budget,
        "exit_date":  exit_date,
        "exit_days":  exit_days,
        "region":     region_hint,
        "mode":       mode_hint or "auto",
    }


async def _auto_strategy_build_for_chat(intent: dict) -> dict | None:
    """
    Run the strategy-build pipeline INTERNALLY from a chat message intent.

    Reuses the same logic as the /api/v1/portfolio/strategy-build endpoint
    but called directly so the chat assistant can return suggestions inline.
    """
    try:
        body = {
            "capital":    float(intent.get("budget") or 100_000),
            "mode":       intent.get("mode") or "auto",
            "region":     intent.get("region") or _USER_REGION or "us",
            "asset_type": "stocks",
        }
        result = await portfolio_strategy_build(body)
        return result
    except Exception as e:
        logger.warning(f"auto strategy-build for chat failed: {e}")
        return None


def _format_strategy_result_for_chat(result: dict, intent: dict) -> str:
    """Render strategy-build result as a chat context block."""
    if not result:
        return ""
    positions = result.get("positions") or []
    regime    = result.get("regime") or {}
    mode_info = result.get("mode_info") or {}
    exp_pnl   = result.get("expected_pnl") or 0
    allocated = result.get("allocated") or 0
    capital   = result.get("capital") or intent.get("budget") or 100000
    region    = result.get("region") or intent.get("region") or "us"
    exit_date = intent.get("exit_date") or "(not specified)"

    if not positions:
        return (
            f"\n=== AUTO-BUILT STRATEGY (no qualifying positions) ===\n"
            f"Region: {region.upper()} | Mode: {result.get('chosen_mode','?')} | "
            f"Regime: {regime.get('label','?')}\n"
            f"Reason: cache had {result.get('cache_coverage', 0)} signals but none "
            f"met conviction/probability thresholds.\n"
        )

    lines = [
        f"\n=== AUTO-BUILT STRATEGY (from user intent) ===",
        f"Capital: ${capital:,.0f} | Region: {region.upper()} | Exit target: {exit_date}",
        f"Regime: {regime.get('label','?')} (VIX {regime.get('vix','?')})  |  "
        f"Mode: {result.get('chosen_mode','?')} (recommended: {result.get('recommended_mode','?')})",
        f"Positions deployed: {len(positions)}  |  Allocated: ${allocated:,.0f}  |  "
        f"Cash: ${result.get('cash', 0):,.0f}",
        f"Expected P&L by next session: ~${exp_pnl:+,.0f} "
        f"(range ${result.get('exp_pnl_low',0):+,.0f} to ${result.get('exp_pnl_high',0):+,.0f})",
        "",
        "RECOMMENDED POSITIONS:",
    ]
    for p in positions[:15]:
        sym  = p.get("symbol", "?")
        conv = p.get("conviction", 0)
        prob = p.get("prob", 0)
        pct  = p.get("pct_portfolio", 0)
        sec  = p.get("sector", "?")
        dol  = p.get("dollar_alloc", 0)
        stop = p.get("stop")
        tgt  = p.get("target")
        line = f"  {sym} ({sec}): {pct:.1f}% = ${dol:,.0f} | conv={conv:.0f} P(up)={prob:.0f}%"
        if stop and tgt:
            line += f" | stop ${stop:.2f} target ${tgt:.2f}"
        lines.append(line)
    lines.append(
        "\nINSTRUCTION: Use these as concrete recommendations. "
        "Reference specific tickers, conviction scores, and the expected P&L range. "
        "Tell the user how to commit them via the AI Portfolio tab."
    )
    return "\n".join(lines)


def _detect_action_intent(
    message: str,
    history: list | None = None,
    portfolio: list | None = None,
) -> dict | None:
    """
    Parse an action intent from a chat message.

    Returns dict with keys:
      action  : "BUY" / "SELL" / "BUY_MORE" / "REBALANCE" / "COMMIT" / "EXIT_LOSERS" / "CONFIRM"
      ticker  : target symbol (uppercase) when applicable
      amount  : dollar amount when specified
      auto    : True if user explicitly asks AI to proceed without confirmation
      confirm : True if this message is confirming a previous proposal
    Returns None when no actionable intent is detected.
    """
    import re as _re

    m_raw = (message or "").strip()
    m = m_raw.lower()
    if not m:
        return None

    auto_words = ("auto", "automatic", "just do it", "do it", "go ahead",
                  "proceed", "execute", "run it", "without asking", "no need to ask")
    auto_flag  = any(w in m for w in auto_words)

    confirm_words = ("yes", "yep", "yeah", "ok", "okay", "sure", "confirmed",
                     "confirm", "execute", "do it", "go ahead", "proceed",
                     "approve", "approved")
    explicit_confirm = (m in confirm_words) or m.startswith(tuple(confirm_words))

    # If user is confirming a prior proposal, parse last assistant message
    if explicit_confirm and history:
        last_assist = next(
            (h for h in reversed(history) if h.get("role") == "assistant"), None
        )
        if last_assist and last_assist.get("content"):
            prior = last_assist["content"]
            # Try to recover the action from previous content
            _prior_common = {"A","I","THE","BUY","SELL","HOLD","ADD","CUT","ALL","NO",
                             "ON","IT","TO","AS","AT","BE","BY","DO","GO","IN","IS","OF",
                             "SO","UP","DOWN","ME","MY","NOW","NEW","RUN","TOP","WAY",
                             "OUR","OUT","CAN","WE","US","YES","AM","AN","OK","OR","AI",
                             "CEO","CFO","ETF","ETFS","USA","USD","EUR","GBP","JPY","INR"}
            for verb, action in (
                ("BUY MORE", "BUY_MORE"),
                ("BUY",      "BUY"),
                ("SELL",     "SELL"),
                ("EXIT",     "SELL"),
                ("COMMIT",   "COMMIT"),
                ("REBALANCE","REBALANCE"),
            ):
                if verb in prior.upper():
                    # Find first non-common ticker in prior message
                    prior_ticker = None
                    for mm in _re.finditer(
                        r"\b([A-Z]{1,5}(?:[-.][A-Z0-9]{1,5})?(?:\.NS|\.T|\.HK|\.KS|\.AX|\.SW|\.PA|\.DE|\.AS)?)\b",
                        prior,
                    ):
                        c = mm.group(1).upper()
                        if c not in _prior_common:
                            prior_ticker = c
                            break
                    return {
                        "action":  action,
                        "ticker":  prior_ticker,
                        "amount":  None,
                        "auto":    True,    # explicit confirm = execute
                        "confirm": True,
                    }

    # Extract a ticker symbol from current message (uppercase 1-5 chars optionally
    # with regional suffix). Find ALL candidates and pick first non-common one.
    ticker = None
    _common = {"A","I","THE","BUY","SELL","HOLD","ADD","CUT","ALL","NO","ON","IT","TO",
               "AS","AT","BE","BY","DO","GO","IN","IS","OF","SO","UP","DOWN","ME","MY",
               "NOW","NEW","RUN","TOP","WAY","OUR","OUT","CAN","WE","US","YES","AM","AN",
               "OK","OR","AI","CEO","CFO","ETF","ETFS","USA","USD","EUR","GBP","JPY","INR"}
    for tk_match in _re.finditer(
        r"\b([A-Z]{1,5}(?:[-.][A-Z0-9]{1,5})?(?:\.NS|\.T|\.HK|\.KS|\.AX|\.SW|\.PA|\.DE|\.AS)?)\b",
        m_raw,
    ):
        candidate = tk_match.group(1).upper()
        if candidate not in _common:
            ticker = candidate
            break

    # Amount extraction
    amount = None
    a_k = _re.search(r"\$?\s*(\d[\d,]*\.?\d*)\s*k\b", m)
    if a_k:
        amount = float(a_k.group(1).replace(",", "")) * 1000
    else:
        a_d = _re.search(r"\$\s*(\d[\d,]+)", m_raw)
        if a_d:
            amount = float(a_d.group(1).replace(",", ""))

    # ── Verb detection ──────────────────────────────────────────────────────
    # COMMIT a built strategy
    if any(p in m for p in ("commit portfolio", "commit the portfolio",
                            "commit it", "save portfolio", "lock in",
                            "deploy capital", "deploy portfolio",
                            "execute portfolio", "make it live", "commit this")):
        return {"action": "COMMIT", "ticker": None, "amount": amount,
                "auto": auto_flag, "confirm": False}

    # REBALANCE
    if any(p in m for p in ("rebalance", "reallocate", "rebalance my",
                            "redistribute", "re-balance")):
        return {"action": "REBALANCE", "ticker": None, "amount": None,
                "auto": auto_flag, "confirm": False}

    # EXIT_LOSERS
    if any(p in m for p in ("exit losers", "sell losers", "dump losers",
                            "cut losers", "exit my losses",
                            "sell underperformers")):
        return {"action": "EXIT_LOSERS", "ticker": None, "amount": None,
                "auto": auto_flag, "confirm": False}

    # BUY_MORE on existing position (verbs that imply adding)
    if any(p in m for p in ("buy more", "add more", "double down", "double up",
                            "increase", "add to my", "top up", "stack more")):
        if ticker:
            return {"action": "BUY_MORE", "ticker": ticker, "amount": amount,
                    "auto": auto_flag, "confirm": False}

    # SELL / EXIT a specific position
    if any(p in m for p in ("sell ", "exit ", "close ", "dump ", "liquidate ",
                            "get out of", "trim ", "reduce ")):
        if ticker:
            return {"action": "SELL", "ticker": ticker, "amount": amount,
                    "auto": auto_flag, "confirm": False}

    # BUY a new position
    if any(p in m for p in ("buy ", "long ", "add ", "pick up ", "open ",
                            "enter ", "go long", "purchase ", "i want to buy")):
        if ticker:
            # If portfolio already has this ticker, this should be BUY_MORE
            in_port = portfolio and any(
                (p.get("ticker") or "").upper() == ticker for p in portfolio
            )
            return {"action": "BUY_MORE" if in_port else "BUY",
                    "ticker": ticker, "amount": amount,
                    "auto": auto_flag, "confirm": False}

    return None


def _build_action_recommendation_block(intent: dict) -> str:
    """
    Run a signal for the action's target ticker and format a recommendation
    block for the Gemini system prompt.
    """
    action = intent.get("action")
    ticker = intent.get("ticker")
    amount = intent.get("amount")
    auto   = bool(intent.get("auto"))
    confirm = bool(intent.get("confirm"))

    header = f"\n=== USER ACTION INTENT: {action}"
    if ticker:
        header += f" {ticker}"
    if amount:
        header += f" ${amount:,.0f}"
    if confirm:
        header += " (CONFIRMING PRIOR PROPOSAL)"
    elif auto:
        header += " (AUTO-EXECUTE REQUESTED)"
    header += " ==="

    body_lines = [header]

    # For ticker-level actions: run/fetch a fresh signal
    if ticker and action in ("BUY", "SELL", "BUY_MORE"):
        sig = None
        cache_key = (ticker.upper(), "1d")
        cached = _SIGNAL_CACHE.get(cache_key)
        if cached and time.time() < cached[1]:
            sig = cached[0]
            body_lines.append(f"\nSIGNAL FOR {ticker} (cached):")
        else:
            try:
                sig = _generate_signal_sync(ticker)
                if sig:
                    body_lines.append(f"\nSIGNAL FOR {ticker} (fresh run):")
            except Exception as e:
                body_lines.append(f"\nSIGNAL FOR {ticker}: could not generate ({e})")

        if sig:
            direction  = sig.get("direction", "?")
            prob       = float(sig.get("probability", sig.get("prob_up", 0.5)))
            if prob > 1: prob /= 100.0
            conviction = float(sig.get("conviction", 0))
            entropy    = float(sig.get("entropy", 1.0))
            agents     = sig.get("agents") or []
            longs  = [a["agent_name"] for a in agents if a.get("vote") == "LONG"]
            shorts = [a["agent_name"] for a in agents if a.get("vote") == "SHORT"]
            body_lines.append(
                f"  Direction: {direction} | P(up)={prob*100:.1f}% | "
                f"Conviction={conviction:.1f}% | Entropy={entropy:.2f}"
            )
            if longs:
                body_lines.append(f"  LONG votes: {', '.join(longs[:6])}")
            if shorts:
                body_lines.append(f"  SHORT votes: {', '.join(shorts[:6])}")

            # ── Concrete recommendation rule based on the action vs signal ──
            rec = ""
            if action == "BUY" or action == "BUY_MORE":
                if direction == "LONG" and conviction > 40:
                    rec = f"✓ ALIGNED: Signal strongly supports {action} (conv {conviction:.0f}, P {prob*100:.0f}%)"
                elif direction == "LONG":
                    rec = f"✓ ALIGNED (modest): Signal supports {action} but conviction low ({conviction:.0f})"
                elif direction == "NEUTRAL":
                    rec = f"⚠ CAUTION: Signal is NEUTRAL — no edge to {action} right now"
                elif direction == "SHORT":
                    rec = f"✗ DO NOT {action}: Signal is SHORT (P-up only {prob*100:.0f}%) — buying here fights the model"
            elif action == "SELL":
                # Find this position in AI portfolio for P&L context
                ai_pos = next(
                    (p for p in _AI_PORTFOLIO.get("positions", [])
                     if (p.get("ticker") or "").upper() == ticker.upper()),
                    None
                )
                pl_str = ""
                if ai_pos:
                    q = _LIVE_QUOTE_CACHE.get(ticker.upper())
                    cur = (q["price"] if q else None) or ai_pos.get("entry_price") or 0
                    entry = ai_pos.get("entry_price") or cur
                    pl_pct = ((cur / entry) - 1) * 100 if entry else 0
                    pl_str = f" | current P&L {pl_pct:+.1f}%"
                if direction == "LONG" and conviction > 40:
                    rec = f"✗ DO NOT SELL: Signal still says LONG conv {conviction:.0f}{pl_str} — model wants you to hold"
                elif direction == "LONG":
                    rec = f"⚠ NEUTRAL: Signal mildly LONG{pl_str} — sell only if you need liquidity"
                elif direction == "NEUTRAL":
                    rec = f"✓ OK TO SELL: Signal is NEUTRAL{pl_str} — no upside expected"
                elif direction == "SHORT":
                    rec = f"✓ STRONG SELL: Signal is SHORT (P-up {prob*100:.0f}%){pl_str} — exit ASAP"
            body_lines.append(f"\n  RECOMMENDATION: {rec}")

    elif action == "REBALANCE":
        body_lines.append(
            "\nREBALANCE: Use current AI portfolio + top platform signals to suggest "
            "trims/adds. Specifically: 1) list positions where the signal has flipped "
            "to NEUTRAL/SHORT — recommend exit. 2) list new high-conviction LONG signals "
            "not yet in portfolio — recommend adding."
        )
    elif action == "EXIT_LOSERS":
        # Build a quick analysis of losers in AI portfolio
        losers = []
        for p in _AI_PORTFOLIO.get("positions", []):
            sym = (p.get("ticker") or "").upper()
            entry = p.get("entry_price") or 0
            q = _LIVE_QUOTE_CACHE.get(sym)
            cur = (q["price"] if q else None) or entry
            pl = ((cur / entry) - 1) * 100 if entry else 0
            if pl < 0:
                losers.append(f"{sym} ({pl:+.1f}%)")
        if losers:
            body_lines.append(f"\nLOSING POSITIONS: {', '.join(losers)}")
            body_lines.append(
                "RECOMMENDATION: Confirm each loser's signal before exiting. "
                "If signal still LONG with high conviction, hold. Otherwise exit."
            )
        else:
            body_lines.append("\nNo losing positions in AI portfolio currently.")

    elif action == "COMMIT":
        body_lines.append(
            "\nUser wants to commit the most recent auto-built strategy to the AI "
            "portfolio. Recommend confirming the positions, then tell them to click "
            "the explicit Commit button in the UI (or you can auto-execute if 'auto' set)."
        )

    # Confirmation / auto behaviour guidance for Gemini
    if confirm:
        body_lines.append(
            "\nUser is CONFIRMING the prior suggestion. Backend has already executed "
            "(see action_result in response). Tell user it's been done."
        )
    elif auto:
        body_lines.append(
            "\nUser asked for AUTO execution. Backend will execute now. "
            "Tell user the action has been performed."
        )
    else:
        body_lines.append(
            "\nIMPORTANT: Default mode = MANUAL. Give the user a SPECIFIC recommendation "
            "(BUY/SELL/HOLD with reasoning). End with: 'Reply Yes to proceed, or No to "
            "skip.' so they can confirm."
        )

    return "\n".join(body_lines)


def _execute_chat_action(intent: dict) -> dict:
    """
    Actually perform the action requested in the chat intent.
    Mutates the in-memory _AI_PORTFOLIO. Returns a result dict for the response.
    """
    from datetime import datetime as _dt, timezone as _tz

    action = intent.get("action")
    ticker = (intent.get("ticker") or "").upper()
    amount = intent.get("amount")

    result: dict = {"action": action, "ticker": ticker, "executed": False, "detail": ""}

    try:
        if action == "BUY":
            # Add a new position. Default amount = 10% of capital if not specified.
            capital = _AI_PORTFOLIO.get("capital", 100_000)
            amt = float(amount) if amount else capital * 0.10
            q = _LIVE_QUOTE_CACHE.get(ticker) or {}
            entry = q.get("price") or 0
            if entry <= 0:
                result["detail"] = f"No live price for {ticker} — could not execute"
                return result
            shares = round(amt / entry, 4)
            positions = _AI_PORTFOLIO.get("positions", [])
            existing = next((p for p in positions
                             if (p.get("ticker") or "").upper() == ticker), None)
            if existing:
                # Treat as BUY_MORE
                old_shares = existing.get("shares", 0)
                old_alloc  = existing.get("allocated", old_shares * entry)
                new_shares = old_shares + shares
                new_alloc  = old_alloc + amt
                wavg_entry = (
                    new_alloc / new_shares if new_shares > 0 else entry
                )
                existing["shares"]      = new_shares
                existing["allocated"]   = new_alloc
                existing["entry_price"] = round(wavg_entry, 4)
            else:
                positions.append({
                    "ticker":      ticker,
                    "shares":      shares,
                    "entry_price": entry,
                    "allocated":   amt,
                    "added_at":    _dt.now(_tz.utc).isoformat(),
                })
            _AI_PORTFOLIO["positions"] = positions
            try: db_manager.save_ai_portfolio(_AI_PORTFOLIO)
            except Exception: pass
            result.update({
                "executed": True,
                "detail":   f"BOUGHT {shares:.2f} shares of {ticker} @ ${entry:.2f} = ${amt:,.0f}",
                "shares":   shares,
                "entry":    entry,
                "amount":   amt,
            })

        elif action == "SELL":
            positions = _AI_PORTFOLIO.get("positions", [])
            before = len(positions)
            _AI_PORTFOLIO["positions"] = [
                p for p in positions
                if (p.get("ticker") or "").upper() != ticker
            ]
            sold = before - len(_AI_PORTFOLIO["positions"])
            try: db_manager.save_ai_portfolio(_AI_PORTFOLIO)
            except Exception: pass
            if sold:
                result.update({"executed": True,
                               "detail":   f"SOLD position {ticker} (removed from AI portfolio)"})
            else:
                result["detail"] = f"{ticker} not in AI portfolio — nothing to sell"

        elif action == "BUY_MORE":
            capital = _AI_PORTFOLIO.get("capital", 100_000)
            amt = float(amount) if amount else capital * 0.05
            q = _LIVE_QUOTE_CACHE.get(ticker) or {}
            cur_price = q.get("price")
            positions = _AI_PORTFOLIO.get("positions", [])
            existing = next((p for p in positions
                             if (p.get("ticker") or "").upper() == ticker), None)
            if not existing:
                result["detail"] = f"{ticker} not in AI portfolio — cannot buy more"
                return result
            entry = cur_price or existing.get("entry_price") or 0
            if entry <= 0:
                result["detail"] = f"No live price for {ticker}"
                return result
            extra_shares = round(amt / entry, 4)
            old_shares   = existing.get("shares", 0)
            old_alloc    = existing.get("allocated", old_shares * entry)
            new_shares   = old_shares + extra_shares
            new_alloc    = old_alloc + amt
            wavg_entry   = new_alloc / new_shares if new_shares > 0 else entry
            existing["shares"]      = new_shares
            existing["allocated"]   = new_alloc
            existing["entry_price"] = round(wavg_entry, 4)
            try: db_manager.save_ai_portfolio(_AI_PORTFOLIO)
            except Exception: pass
            result.update({
                "executed":     True,
                "detail":       f"ADDED {extra_shares:.2f} shares of {ticker} @ ${entry:.2f} = ${amt:,.0f}",
                "extra_shares": extra_shares,
                "amount":       amt,
            })

        elif action == "EXIT_LOSERS":
            positions = _AI_PORTFOLIO.get("positions", [])
            exited = []
            keep = []
            for p in positions:
                sym = (p.get("ticker") or "").upper()
                entry = p.get("entry_price") or 0
                q = _LIVE_QUOTE_CACHE.get(sym)
                cur = (q["price"] if q else None) or entry
                pl = ((cur / entry) - 1) * 100 if entry else 0
                if pl < 0:
                    exited.append(sym)
                else:
                    keep.append(p)
            _AI_PORTFOLIO["positions"] = keep
            try: db_manager.save_ai_portfolio(_AI_PORTFOLIO)
            except Exception: pass
            result.update({
                "executed": True,
                "detail":   f"Exited {len(exited)} losing positions: {', '.join(exited) or 'none'}",
                "exited":   exited,
            })

        else:
            result["detail"] = f"Action {action} requires UI confirmation (not auto-executable here)"

    except Exception as e:
        logger.warning(f"_execute_chat_action({action} {ticker}): {e}")
        result["detail"] = f"Execution error: {e}"

    return result


def _build_ai_portfolio_block() -> str:
    """Summarise the in-memory AI portfolio state for chat context."""
    positions = _AI_PORTFOLIO.get("positions") or []
    if not positions:
        return "(no AI portfolio committed)"
    capital = _AI_PORTFOLIO.get("capital", 100_000)
    now_ts  = time.time()
    rows = []
    for p in positions:
        sym   = p.get("ticker", "?")
        entry = p.get("entry_price") or 0
        shrs  = p.get("shares", 0)
        alloc = p.get("allocated", entry * shrs)
        q     = _LIVE_QUOTE_CACHE.get(sym)
        cur   = (q["price"] if q and q.get("expires", 0) > now_ts else None) or entry
        pnl   = round((cur - entry) * shrs, 2) if entry else 0
        pnl_p = round((cur - entry) / entry * 100, 1) if entry else 0
        rows.append(
            f"  {sym}: {shrs:.2f}sh @ ${entry:.2f} → ${cur:.2f} | PnL {'+' if pnl>=0 else ''}"
            f"${pnl:,.0f} ({'+' if pnl_p>=0 else ''}{pnl_p:.1f}%) | alloc ${alloc:,.0f}"
        )
    total_alloc = sum(p.get("allocated", 0) for p in positions)
    cash = capital - total_alloc
    return (
        f"Capital: ${capital:,.0f} | Invested: ${total_alloc:,.0f} | Cash: ${cash:,.0f}\n"
        + "\n".join(rows)
    )


@app.get("/api/v1/ticker/info/{ticker}")
async def get_ticker_info(ticker: str):
    """OHLCV, fundamentals, 60-day price history and company info for the Signal left panel."""
    import yfinance as _yf
    import math as _m

    sym = ticker.upper()

    def _fetch():
        t = _yf.Ticker(sym)

        def _clean(v, d=None):
            if v is None: return d
            try:
                fv = float(v)
                return d if (_m.isnan(fv) or _m.isinf(fv)) else fv
            except Exception:
                return v  # keep strings as-is

        # ── 1. Try t.info (full fundamentals) ─────────────────────────────
        info = {}
        try:
            info = t.info or {}
        except Exception:
            pass

        def _s(k, d=None):
            v = info.get(k, d)
            if isinstance(v, float) and (_m.isnan(v) or _m.isinf(v)):
                return d
            return v

        # ── 2. Always fetch fast_info — available even when rate-limited ──
        fi_price = fi_prev = fi_open = fi_high = fi_low = fi_vol = fi_mcap = None
        fi_52h = fi_52l = fi_exchange = fi_currency = None
        try:
            fi = t.fast_info
            fi_price    = _clean(getattr(fi, "last_price",              None)) or _clean(getattr(fi, "regular_market_price", None))
            fi_prev     = _clean(getattr(fi, "previous_close",          None))
            fi_open     = _clean(getattr(fi, "open",                    None))
            fi_high     = _clean(getattr(fi, "day_high",                None))
            fi_low      = _clean(getattr(fi, "day_low",                 None))
            fi_vol      = _clean(getattr(fi, "last_volume",             None)) or _clean(getattr(fi, "three_month_average_volume", None))
            fi_mcap     = _clean(getattr(fi, "market_cap",              None))
            fi_52h      = _clean(getattr(fi, "year_high",               None))
            fi_52l      = _clean(getattr(fi, "year_low",                None))
            fi_exchange = getattr(fi, "exchange", None)
            fi_currency = getattr(fi, "currency", None)
        except Exception:
            pass

        # ── 3. 1-year history: OHLCV + closes, 52w range, beta calculation ──
        closes, dates = [], []
        w52_high = w52_low = hist_beta = None
        hist_open = hist_high = hist_low = hist_vol = hist_prev = None
        try:
            hist = t.history(period="1y", interval="1d")
            if hist is not None and not hist.empty:
                c = hist["Close"].dropna()
                closes   = [round(float(v), 4) for v in c.tolist()[-60:]]
                dates    = [str(d.date()) for d in hist.index[-60:]]
                w52_high = round(float(c.max()), 4)
                w52_low  = round(float(c.min()), 4)

                # Pull today's OHLCV from the last history row
                last = hist.iloc[-1]
                if "Open"   in hist.columns: hist_open = _clean(last["Open"])
                if "High"   in hist.columns: hist_high = _clean(last["High"])
                if "Low"    in hist.columns: hist_low  = _clean(last["Low"])
                if "Volume" in hist.columns: hist_vol  = _clean(last["Volume"])
                # Previous close = second-to-last row's Close
                if len(hist) >= 2:
                    hist_prev = _clean(hist.iloc[-2]["Close"])

                # Compute beta vs SPY from parallel history
                if not _s("beta") and len(c) >= 60:
                    try:
                        spy = _yf.Ticker("SPY")
                        spy_hist = spy.history(period="1y", interval="1d")
                        if spy_hist is not None and not spy_hist.empty:
                            spy_c = spy_hist["Close"].dropna()
                            import pandas as _pd
                            merged = _pd.concat([c.rename("stk"), spy_c.rename("spy")], axis=1).dropna()
                            if len(merged) >= 30:
                                r_stk = merged["stk"].pct_change().dropna()
                                r_spy = merged["spy"].pct_change().dropna()
                                cov   = float(r_stk.cov(r_spy))
                                var   = float(r_spy.var())
                                hist_beta = round(cov / var, 3) if var > 0 else None
                    except Exception:
                        pass
        except Exception:
            pass

        # ── 4. Merge: info > fast_info > live cache > history fallbacks ───────
        cached_q   = _LIVE_QUOTE_CACHE.get(sym) or {}

        price      = _s("regularMarketPrice") or _s("currentPrice") or fi_price or cached_q.get("price")
        prev_close = _s("regularMarketPreviousClose") or _s("previousClose") or fi_prev or hist_prev
        if not price and closes:
            price      = closes[-1]
            prev_close = prev_close or (closes[-2] if len(closes) >= 2 else closes[-1])

        change     = _s("regularMarketChange")
        change_pct = _s("regularMarketChangePercent")
        if change is None and price and prev_close:
            change = round(float(price) - float(prev_close), 4)
        if change_pct is None and change and prev_close and float(prev_close) != 0:
            change_pct = round(float(change) / float(prev_close) * 100, 4)

        # OHLCV: info > fast_info > live cache (has day_high/low/vol) > history last row
        mkt_open   = _s("regularMarketOpen")    or _s("open")    or fi_open   or hist_open
        mkt_high   = _s("regularMarketDayHigh") or _s("dayHigh") or fi_high   or cached_q.get("day_high") or hist_high
        mkt_low    = _s("regularMarketDayLow")  or _s("dayLow")  or fi_low    or cached_q.get("day_low")  or hist_low
        volume     = _s("regularMarketVolume")  or _s("volume")  or fi_vol    or cached_q.get("volume")   or hist_vol
        market_cap = _s("marketCap") or fi_mcap
        w52h       = _s("fiftyTwoWeekHigh") or fi_52h or w52_high
        w52l       = _s("fiftyTwoWeekLow")  or fi_52l or w52_low
        beta       = _s("beta") or hist_beta
        exchange   = _s("exchange") or _s("fullExchangeName") or fi_exchange
        currency   = _s("currency") or fi_currency or "USD"

        return {
            "ticker":         sym,
            "short_name":     _s("shortName") or _s("longName") or sym,
            "sector":         _s("sector"),
            "industry":       _s("industry"),
            "exchange":       exchange,
            "currency":       currency,
            "price":          round(float(price), 4) if price else None,
            "open":           round(float(mkt_open), 4) if mkt_open else None,
            "high":           round(float(mkt_high), 4) if mkt_high else None,
            "low":            round(float(mkt_low),  4) if mkt_low  else None,
            "close":          round(float(prev_close), 4) if prev_close else None,
            "volume":         int(volume) if volume else None,
            "change":         change,
            "change_pct":     change_pct,
            "week52_high":    w52h,
            "week52_low":     w52l,
            "market_cap":     int(market_cap) if market_cap else None,
            "pe_trailing":    _s("trailingPE"),
            "pe_forward":     _s("forwardPE"),
            "eps":            _s("trailingEps"),
            "beta":           beta,
            "dividend_yield": _s("dividendYield"),
            "avg_volume":     _s("averageVolume") or _s("averageVolume10days"),
            "short_ratio":    _s("shortRatio"),
            "description":    (_s("longBusinessSummary") or "")[:320],
            "closes":         closes,
            "dates":          dates,
        }

    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, _fetch)
    except Exception as e:
        logger.warning(f"Ticker info failed for {sym}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ticker/chart/{ticker}")
async def get_ticker_chart(ticker: str, period: str = "3mo", interval: str = "1d"):
    """OHLCV time-series for the interactive price chart in the Signal left panel."""
    import yfinance as _yf
    import math as _m

    VALID_PERIODS   = {"1d","5d","1mo","3mo","6mo","1y","5y"}
    VALID_INTERVALS = {"1m","5m","15m","30m","1h","1d","1wk","1mo"}
    if period   not in VALID_PERIODS:   period   = "3mo"
    if interval not in VALID_INTERVALS: interval = "1d"

    sym = ticker.upper()

    def _fetch():
        t    = _yf.Ticker(sym)
        hist = t.history(period=period, interval=interval)
        if hist is None or hist.empty:
            return {"ticker": sym, "data": [], "period": period, "interval": interval}
        rows = []
        intraday = interval in {"1m","5m","15m","30m","1h"}
        for ts, row in hist.iterrows():
            def _f(k):
                v = row.get(k)
                if v is None: return None
                try:
                    f = float(v)
                    return None if (_m.isnan(f) or _m.isinf(f)) else round(f, 4)
                except Exception:
                    return None
            close = _f("Close")
            if close is None:
                continue
            vol = _f("Volume")
            rows.append({
                "date":   ts.strftime("%Y-%m-%d %H:%M") if intraday else str(ts.date()),
                "open":   _f("Open"),
                "high":   _f("High"),
                "low":    _f("Low"),
                "close":  close,
                "volume": int(vol) if vol else 0,
            })
        return {"ticker": sym, "data": rows, "period": period, "interval": interval}

    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, _fetch)
    except Exception as e:
        logger.warning(f"Chart data failed for {sym}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/chat")
async def signal_chat(body: dict):
    """
    AI chat endpoint for the Signal tab chatbot.
    Accepts user question + signal context, returns Claude's answer.
    Body: { "question": str, "ticker": str, "signal_context": dict }
    """
    import os
    question = (body.get("question") or "").strip()
    ticker   = (body.get("ticker") or "").upper()
    ctx      = body.get("signal_context") or {}

    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    # ── Discovery mode: no ticker loaded yet ────────────────────────────────
    if not ticker:
        system_prompt = """You are AlphaAgent AI — an expert quantitative trading assistant. The user has not loaded a signal yet. Help them discover what to analyze.

When the user's question implies they want to analyze, trade, or understand performance of a specific asset — end your response with a new line in EXACTLY this format:
ANALYZE:TICKER

Use the correct yfinance symbol. Reference guide:
Gold → ANALYZE:GC=F | Silver → ANALYZE:SI=F | Crude Oil → ANALYZE:CL=F | Natural Gas → ANALYZE:NG=F | Copper → ANALYZE:HG=F
Bitcoin → ANALYZE:BTC-USD | Ethereum → ANALYZE:ETH-USD | Solana → ANALYZE:SOL-USD | XRP → ANALYZE:XRP-USD | BNB → ANALYZE:BNB-USD
Apple → ANALYZE:AAPL | NVIDIA → ANALYZE:NVDA | Microsoft → ANALYZE:MSFT | Tesla → ANALYZE:TSLA | Amazon → ANALYZE:AMZN
Google/Alphabet → ANALYZE:GOOGL | Meta → ANALYZE:META | Netflix → ANALYZE:NFLX | AMD → ANALYZE:AMD
S&P 500 → ANALYZE:SPY | NASDAQ 100 → ANALYZE:QQQ | Dow Jones → ANALYZE:DIA | Russell 2000 → ANALYZE:IWM
Reliance → ANALYZE:RELIANCE.NS | TCS → ANALYZE:TCS.NS | HDFC Bank → ANALYZE:HDFCBANK.NS | Infosys → ANALYZE:INFY
ICICI Bank → ANALYZE:ICICIBANK.NS | Nifty 50 / India ETF → ANALYZE:INDA | SBI → ANALYZE:SBIN.NS
EUR/USD or Euro → ANALYZE:EURUSD=X | GBP/USD → ANALYZE:GBPUSD=X | USD/JPY → ANALYZE:USDJPY=X
GBP/JPY → ANALYZE:GBPJPY=X | USD/INR or Dollar-Rupee → ANALYZE:USDINR=X | EUR/INR → ANALYZE:EURINR=X
Alibaba → ANALYZE:BABA | Baidu → ANALYZE:BIDU | NIO → ANALYZE:NIO | Tencent → ANALYZE:TCEHY
Toyota → ANALYZE:7203.T | Sony → ANALYZE:6758.T | Nintendo → ANALYZE:7974.T | SoftBank → ANALYZE:9984.T
Gold ETF → ANALYZE:GLD | Silver ETF → ANALYZE:SLV | Oil ETF → ANALYZE:USO | ASML → ANALYZE:ASML | Novo Nordisk → ANALYZE:NVO

Rules:
- Only append ANALYZE: when user clearly wants to analyze/trade/invest in a specific asset
- Do NOT include ANALYZE: for general market education questions
- Keep answer ≤150 words
- Be conversational and helpful"""

        # Enrich discovery prompt with FULL platform context (regime, signals,
        # AI portfolio, leaderboard, quotes — across all regions)
        try:
            _d_plat = _build_platform_context(
                region=None,
                include_signals=True,
                include_portfolio=True,
                include_quotes=True,
                include_leaderboard=True,
                max_signals_per_region=5,
            )
            if _d_plat:
                system_prompt += (
                    f"\n\n=== ALPHAAGENT PLATFORM STATE (use this to give specific, "
                    f"data-driven answers) ===\n{_d_plat}"
                )

            # Detect "invest X by Y, give profit" intent → auto-build strategy
            _intent = _detect_invest_intent(question)
            if _intent:
                logger.info(f"discovery chat invest intent: {_intent}")
                _auto = await _auto_strategy_build_for_chat(_intent)
                if _auto:
                    _strat_block = _format_strategy_result_for_chat(_auto, _intent)
                    if _strat_block:
                        system_prompt += "\n" + _strat_block
                        system_prompt += (
                            "\nIMPORTANT: The user asked you to build a portfolio. "
                            "Use the AUTO-BUILT STRATEGY block above. Tell them the "
                            "specific tickers, the expected P&L range, and instruct "
                            "them to click 'Commit' on the AI Portfolio tab to deploy."
                        )
        except Exception as _pe:
            logger.warning(f"discovery platform context err: {_pe}")

        api_key       = os.environ.get("ANTHROPIC_API_KEY", "")
        gemini_key    = os.environ.get("GEMINI_API_KEY", "")
        ollama_on     = settings.get("ollama.enabled", False)
        prefer_ollama = settings.get("ollama.prefer_ollama", False)
        msgs_d        = [{"role": "user", "content": question}]

        async def _try_ollama_d():
            return await _call_ollama(msgs_d, system=system_prompt)

        if ollama_on and prefer_ollama:
            try:
                ans = await _try_ollama_d()
                return {"answer": ans, "ticker": "", "model": "ollama"}
            except Exception: pass
        if gemini_key:
            try:
                ans = await _call_gemini(msgs_d, system=system_prompt, max_tokens=400)
                return {"answer": ans, "ticker": "", "model": "gemini"}
            except Exception: pass
        if api_key:
            try:
                import anthropic as _an
                client = _an.Anthropic(api_key=api_key)
                msg = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=400,
                    system=system_prompt, messages=msgs_d)
                ans = msg.content[0].text if msg.content else ""
                return {"answer": ans, "ticker": "", "model": "claude"}
            except Exception: pass
        if ollama_on:
            try:
                ans = await _try_ollama_d()
                return {"answer": ans, "ticker": "", "model": "ollama"}
            except Exception: pass
        return {"answer": "Ask me about any stock, crypto, commodity, or forex pair and I'll help you analyze it!", "ticker": ""}

    # ── Build system prompt from signal context ─────────────────────────────
    direction  = ctx.get("direction", "NEUTRAL")
    prob       = ctx.get("probability", 0.5)
    conviction = ctx.get("conviction", 0)
    entropy    = ctx.get("entropy", 0)
    multiplier = ctx.get("multiplier", 1.0)
    warnings   = ctx.get("warnings") or []
    agents_raw = ctx.get("agents") or []
    summary    = ctx.get("summary") or {}
    holding    = ctx.get("holding_period") or {}

    agent_lines = []
    for a in agents_raw:
        if not isinstance(a, dict):
            try: a = a.__dict__
            except Exception: continue
        name  = a.get("agent_name") or a.get("name", "?")
        vote  = a.get("vote", "HOLD")
        p_up  = a.get("probability_up", 0.5)
        rsn   = (a.get("reasoning") or "")[:180]
        factors = a.get("factor_scores") or {}
        if isinstance(factors, dict):
            top = sorted(
                [(k, v) for k, v in factors.items() if isinstance(v, dict)],
                key=lambda x: x[1].get("score", 50), reverse=True
            )[:2]
            fstr = " | ".join(
                f"{v.get('name', k)}: {v.get('score', 0):.0f}/100 ({v.get('interpretation','')[:50]})"
                for k, v in top
            )
        else:
            fstr = ""
        line = f"  • {name}: {vote} ({p_up*100:.0f}%) — {rsn}"
        if fstr:
            line += f"\n    Best factors: {fstr}"
        agent_lines.append(line)

    bull   = summary.get("bull_case") or []
    bear   = summary.get("bear_case") or []
    entry  = summary.get("entry_notes") or []
    risk_n = summary.get("risk_notes") or []
    hl     = summary.get("half_life_days") or holding.get("half_life_days", 0)
    hmin   = holding.get("optimal_hold_min", "?")
    hmax   = holding.get("optimal_hold_max", "?")

    system_prompt = f"""You are AlphaAgent AI — an expert quantitative trading assistant embedded in AlphaAgent, a 9-agent quantitative trading system. A deep analysis of {ticker} has just been completed. Answer the user's questions using ONLY the data below. Be concise (≤200 words unless the user asks for detail), accurate, and use plain English.

═══ SIGNAL SNAPSHOT — {ticker} ═══
Direction:       {direction}
Probability Up:  {prob*100:.1f}%
Conviction:      {conviction:.1f}%
Position Size:   {multiplier:.2f}x (Kelly-sized)
Signal Entropy:  {entropy:.3f} ({'agents strongly disagree — high uncertainty' if entropy > 0.8 else 'agents reasonably aligned'})
Headline:        {summary.get('headline', 'N/A')}
Recommended:     {summary.get('action', 'N/A')}

═══ HOLDING PERIOD ═══
Optimal Hold:    {hmin}–{hmax} days
Half-Life:       {hl:.1f}d {'(mean-reverting signal — exit by expiry date)' if hl and hl > 0 else '(trending — no fixed expiry, use trailing stop)'}

═══ 9-AGENT VOTES ═══
{chr(10).join(agent_lines) or '  (no agent data)'}

═══ BULL CASE ═══
{chr(10).join(f'  + {b}' for b in bull) or '  (no strong bull factors)'}

═══ BEAR CASE ═══
{chr(10).join(f'  - {b}' for b in bear) or '  (no strong bear factors)'}

═══ ENTRY NOTES ═══
{chr(10).join(f'  → {e}' for e in entry) or '  (check intraday chart before entry)'}

═══ RISK NOTES ═══
{chr(10).join(f'  ⚠ {r}' for r in risk_n) or '  (no critical risk flags)'}

═══ ACTIVE WARNINGS ═══
{chr(10).join(f'  !! {w}' for w in warnings) or '  None'}

RULES: Answer only from this context. Do not invent numbers. If entropy > 0.8, note that agents strongly disagree. Use bullet points when listing multiple items. Keep answers ≤200 words unless the user explicitly asks for more detail.
If the user asks to analyze a DIFFERENT asset than {ticker}, end your response with: ANALYZE:TICKER (e.g., ANALYZE:GC=F for gold, ANALYZE:BTC-USD for bitcoin, ANALYZE:EURUSD=X for EUR/USD). Only do this when user explicitly asks to switch assets."""

    # ── Enrich system prompt with FULL platform context (signal + region signals
    # + AI portfolio + market regime + quotes) so Gemini can give specific,
    # cross-asset, data-driven answers — not just the single signal.
    try:
        _plat_block = _build_platform_context(
            region=None,
            include_signals=True,
            include_portfolio=True,
            include_quotes=True,
            include_leaderboard=False,
            max_signals_per_region=5,
        )
        if _plat_block:
            system_prompt += (
                f"\n\n═══ ALPHAAGENT PLATFORM STATE (use this to give specific, "
                f"data-driven answers across all markets) ═══\n{_plat_block}"
            )

        # Detect "invest $X by [date]" intent — auto-build strategy inline
        _intent = _detect_invest_intent(question)
        if _intent:
            logger.info(f"signal chat invest intent: {_intent}")
            _auto = await _auto_strategy_build_for_chat(_intent)
            if _auto:
                _strat_block = _format_strategy_result_for_chat(_auto, _intent)
                if _strat_block:
                    system_prompt += "\n" + _strat_block
                    system_prompt += (
                        "\nIMPORTANT: User asked you to build a portfolio. Use the "
                        "AUTO-BUILT STRATEGY block above with the specific tickers, "
                        "conviction scores, and expected P&L. Conclude by telling "
                        "them to commit via the AI Portfolio tab."
                    )
    except Exception as _pe:
        logger.warning(f"signal chat platform context err: {_pe}")

    # ── Determine AI backend (Gemini primary, fallback to Claude then Ollama) ──
    api_key       = os.environ.get("ANTHROPIC_API_KEY", "")
    gemini_key    = os.environ.get("GEMINI_API_KEY", "")
    ollama_on     = settings.get("ollama.enabled", False)
    prefer_ollama = settings.get("ollama.prefer_ollama", False)
    msgs_sc       = [{"role": "user", "content": question}]

    async def _try_ollama():
        return await _call_ollama(msgs_sc, system=system_prompt)

    if ollama_on and prefer_ollama:
        try:
            answer = await _try_ollama()
            return {"answer": answer, "ticker": ticker, "model": "ollama"}
        except Exception as oe:
            logger.debug(f"Ollama fallback for signal chat: {oe}")

    if gemini_key:
        try:
            answer = await _call_gemini(msgs_sc, system=system_prompt, max_tokens=512)
            return {"answer": answer, "ticker": ticker, "model": "gemini"}
        except Exception as ge:
            logger.debug(f"Gemini signal chat fallback: {ge}")

    if api_key:
        try:
            import anthropic as _anthropic
            client = _anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                system=system_prompt,
                messages=msgs_sc,
            )
            answer = msg.content[0].text if msg.content else "No response generated."
            return {"answer": answer, "ticker": ticker, "model": "claude"}
        except Exception as e:
            logger.error(f"Claude chat error: {e}")

    if ollama_on:
        try:
            answer = await _try_ollama()
            return {"answer": answer, "ticker": ticker, "model": "ollama"}
        except Exception as oe:
            logger.warning(f"Ollama chat failed: {oe}")

    return {"answer": (
        f"AI assistant is offline. Signal summary: {ticker} is **{direction}** "
        f"with {prob*100:.1f}% probability up and {conviction:.1f}% conviction. "
        f"Recommended: {summary.get('action', 'N/A')}"
    )}


@app.get("/api/v1/ticker/holders/{ticker}")
async def get_ticker_holders(ticker: str):
    """
    Returns institutional holders, mutual fund holders, major-holder percentages,
    insider transactions, and congressional trading activity for a given ticker.
    """
    import yfinance as yf
    import math, os

    sym = ticker.upper()

    def _fetch():
        t = yf.Ticker(sym)
        return t

    loop = asyncio.get_running_loop()
    t = await loop.run_in_executor(None, _fetch)

    result: dict = {
        "ticker":              sym,
        "major_holders":       {},
        "institutional":       [],
        "mutual_funds":        [],
        "insider_transactions": [],
        "congressional":       [],
        "congressional_note":  "",
        "rate_limited":        False,
    }

    def _safe(v):
        try:
            fv = float(v)
            return None if (math.isnan(fv) or math.isinf(fv)) else fv
        except Exception:
            return None

    def _pct(raw):
        v = _safe(raw)
        if v is None: return None
        return round(v * 100, 2) if v < 1.5 else round(v, 2)

    def _row(row, cols):
        """Try multiple possible column name variations."""
        for c in cols:
            if c in row.index:
                return row[c]
        return None

    # ── Major holders breakdown ───────────────────────────────────────
    try:
        mh = t.major_holders
        if mh is not None and not mh.empty:
            if "Value" in mh.columns:
                for label, val in mh["Value"].items():
                    v = _safe(val)
                    if v is not None:
                        label_s = str(label)
                        result["major_holders"][label_s] = round(v * 100, 2) if v < 1.5 else round(v, 2)
            else:
                vals   = mh.iloc[:, 0].tolist()
                labels = mh.iloc[:, 1].tolist() if mh.shape[1] > 1 else list(mh.index)
                for i in range(min(len(vals), len(labels))):
                    v = _safe(vals[i])
                    if v is not None:
                        result["major_holders"][str(labels[i])] = round(v * 100, 2) if v < 1.5 else round(v, 2)
    except Exception as _mhe:
        if "Too Many" in str(_mhe) or "Rate" in str(_mhe) or "401" in str(_mhe):
            result["rate_limited"] = True

    # ── Institutional holders ─────────────────────────────────────────
    try:
        ih = t.institutional_holders
        if ih is not None and not ih.empty:
            for _, row in ih.head(12).iterrows():
                name   = str(row.get("Holder", ""))
                shares = _safe(row.get("Shares"))
                value  = _safe(row.get("Value"))
                pct    = _safe(row.get("pctHeld") or row.get("% Out") or row.get("pct_out"))
                if name and name not in ("nan", "None", ""):
                    result["institutional"].append({
                        "name":     name,
                        "shares":   int(shares) if shares else None,
                        "value_m":  round(value / 1e6, 1) if value else None,
                        "pct_out":  round(pct * 100, 2) if pct and pct < 1.5 else (round(pct, 2) if pct else None),
                    })
    except Exception as _ihe:
        if "Too Many" in str(_ihe) or "Rate" in str(_ihe) or "401" in str(_ihe):
            result["rate_limited"] = True

    # ── Mutual fund holders ───────────────────────────────────────────
    try:
        mfh = t.mutualfund_holders
        if mfh is not None and not mfh.empty:
            for _, row in mfh.head(8).iterrows():
                name   = str(row.get("Holder", ""))
                shares = _safe(row.get("Shares"))
                value  = _safe(row.get("Value"))
                pct    = _safe(row.get("pctHeld") or row.get("% Out") or row.get("pct_out"))
                if name and name not in ("nan", "None", ""):
                    result["mutual_funds"].append({
                        "name":    name,
                        "shares":  int(shares) if shares else None,
                        "value_m": round(value / 1e6, 1) if value else None,
                        "pct_out": round(pct * 100, 2) if pct and pct < 1.5 else (round(pct, 2) if pct else None),
                    })
    except Exception:
        pass

    # ── Insider transactions (Form 4 — yfinance) ─────────────────────────────
    try:
        it = t.insider_transactions
        if it is not None and not it.empty:
            for _, row in it.head(15).iterrows():
                name   = str(row.get("Name", row.get("Insider", "")))
                title  = str(row.get("Title", row.get("Relationship", "")))
                txn    = str(row.get("Transaction", row.get("Type", "")))
                shares = _safe(row.get("Shares", row.get("Share", 0)))
                val    = _safe(row.get("Value", row.get("Total", 0)))
                date_s = str(row.get("Start Date", row.get("Date", "")))[:10]
                if name and name not in ("nan", "None", ""):
                    result["insider_transactions"].append({
                        "name":    name,
                        "title":   title,
                        "txn":     txn,
                        "shares":  int(shares) if shares else None,
                        "value_m": round(val / 1e6, 3) if val else None,
                        "date":    date_s,
                    })
    except Exception:
        pass

    # ── Congressional trades (Quiver Quant API or STOCK Act S3 fallbacks) ──
    try:
        import urllib.request as _ureq
        import json as _json

        _HEADERS = {"User-Agent": "AlphaAgent/2.0", "Accept": "application/json"}
        _qq_key = settings.get("api_keys.quiverquant", "")

        # ── Primary: US Senate EFTS (official government API, free, no key) ──────
        try:
            import urllib.parse as _urlparse
            from datetime import date as _date_cls
            _q = _urlparse.quote(f'"{sym}"')
            _today_s = _date_cls.today().isoformat()
            _efts_url = (
                "https://efts.senate.gov/LATEST/search-index"
                f"?q={_q}"
                "&dateRange=custom&fromDate=2021-01-01"
                f"&toDate={_today_s}"
                "&category=All+Transactions"
                "&results_count=20"
                "&sort=transaction_date&order=desc"
            )
            _efts_req = _ureq.Request(_efts_url, headers=_HEADERS)
            with _ureq.urlopen(_efts_req, timeout=10) as _efts_resp:
                _efts_data = _json.loads(_efts_resp.read())
            for _hit in _efts_data.get("hits", {}).get("hits", []):
                _src = _hit.get("_source", {})
                _fname = (_src.get("first_name") or "").strip()
                _lname = (_src.get("last_name") or "").strip()
                _sen_name = f"{_fname} {_lname}".strip() or _src.get("senator", "Unknown")
                _party = (_src.get("party") or "").upper()[:1]  # D / R / I
                for _tx in (_src.get("txs") or []):
                    _tx_ticker = (_tx.get("ticker") or "").upper().strip()
                    _tx_desc = (_tx.get("asset_description") or "").upper()
                    if _tx_ticker == sym or (not _tx_ticker and sym in _tx_desc):
                        _tx_date = (
                            _tx.get("transaction_date")
                            or _tx.get("date")
                            or _src.get("date", "")
                        )
                        result["congressional"].append({
                            "politician":  _sen_name,
                            "party":       _party,
                            "chamber":     "Senate",
                            "transaction": _tx.get("type", ""),
                            "range":       _tx.get("amount", ""),
                            "date":        str(_tx_date)[:10],
                        })
                        if len(result["congressional"]) >= 15:
                            break
                if len(result["congressional"]) >= 15:
                    break
        except Exception as _e:
            logger.debug(f"Senate EFTS fetch failed: {_e}")

        # ── Secondary: House + Senate Stock Watcher S3 (free, may be unavailable) ──
        if not result["congressional"]:
            for _s3_url, _chamber_name in [
                ("https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json", "House"),
                ("https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json", "Senate"),
            ]:
                try:
                    _sr = _ureq.Request(_s3_url, headers=_HEADERS)
                    with _ureq.urlopen(_sr, timeout=6) as _srr:
                        _s3_data = _json.loads(_srr.read())
                    _max = 6 if _chamber_name == "House" else 12
                    for _tr in _s3_data:
                        if str(_tr.get("ticker", "")).upper().strip() == sym:
                            _name_s3 = (
                                _tr.get("representative") or
                                f"{_tr.get('first_name','')} {_tr.get('last_name','')}".strip() or
                                _tr.get("senator", "Unknown")
                            )
                            result["congressional"].append({
                                "politician":  _name_s3,
                                "party":       _tr.get("party", ""),
                                "chamber":     _chamber_name,
                                "transaction": _tr.get("type", ""),
                                "range":       _tr.get("amount", ""),
                                "date":        _tr.get("transaction_date", ""),
                            })
                            if len(result["congressional"]) >= _max:
                                break
                except Exception as _e:
                    logger.debug(f"{_chamber_name} Stock Watcher S3 failed: {_e}")

        # ── Tertiary: Quiver Quant if API key is configured ─────────────────────
        if not result["congressional"] and _qq_key:
            try:
                _qq_url = f"https://api.quiverquant.com/beta/live/congresstrading/{sym}"
                _qq_req = _ureq.Request(
                    _qq_url,
                    headers={**_HEADERS, "Authorization": f"Token {_qq_key}"},
                )
                with _ureq.urlopen(_qq_req, timeout=10) as _qqr:
                    _qq_data = _json.loads(_qqr.read())
                for _tr in (_qq_data or [])[:15]:
                    _party = _tr.get("Party", "")
                    if _party.lower().startswith("rep"):
                        _party = "R"
                    elif _party.lower().startswith("dem"):
                        _party = "D"
                    result["congressional"].append({
                        "politician":  _tr.get("Representative", "Unknown"),
                        "party":       _party,
                        "chamber":     _tr.get("House", ""),
                        "transaction": _tr.get("Transaction", ""),
                        "range":       _tr.get("Range", ""),
                        "date":        (_tr.get("TransactionDate") or _tr.get("ReportDate", ""))[:10],
                    })
            except Exception as _e:
                logger.debug(f"Quiver Quant congressional fetch failed: {_e}")

        if result["congressional"]:
            result["congressional_note"] = (
                f"{len(result['congressional'])} trade(s) — STOCK Act disclosures "
                f"(Senate: {sum(1 for c in result['congressional'] if c['chamber']=='Senate')} · "
                f"House: {sum(1 for c in result['congressional'] if c['chamber']=='House')})"
            )
        else:
            result["congressional_note"] = "no_data"

    except Exception as _e:
        logger.warning(f"Congressional data fetch failed for {sym}: {_e}")
        result["congressional_note"] = "Congressional data temporarily unavailable"

    return result


@app.post("/api/v1/market-chat")
async def market_chat(body: dict):
    """
    Global Market AI assistant — answers news, earnings, and market questions.
    Detects deep-analysis requests and returns a structured intent for the frontend.
    Body: {"message": str, "history": [{"role": str, "content": str}]}
    Returns: {"answer": str, "intent": str, "ticker": str|null, "correction": dict|null}
    """
    import os
    import json as _json
    from datetime import date
    from concurrent.futures import ThreadPoolExecutor

    message = (body.get("message") or "").strip()
    history  = body.get("history") or []
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    # ── Quick context: news headlines (4s timeout) ────────────────
    news_headlines: list[str] = []
    try:
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(_parse_rss, src, url, 4) for src, url in _RSS_FEEDS[:4]]
            for fut in futs:
                try:
                    for a in fut.result(timeout=4):
                        h = f"[{a['publisher']}] {a['title']}"
                        if h not in news_headlines:
                            news_headlines.append(h)
                        if len(news_headlines) >= 10:
                            break
                except Exception:
                    pass
    except Exception:
        pass

    news_block     = "\n".join(f"- {h}" for h in news_headlines[:10]) or "- (headlines unavailable)"
    today_str      = date.today().isoformat()
    market_ctx     = body.get("market_context") or {}

    # ── Server-side market data from AlphaAgent caches (always fresh) ─────────
    srv_mkt_block = _build_market_context_block()

    # Merge frontend context (sectors, earnings, sentiment) with server-side data
    mkt_lines = [srv_mkt_block] if srv_mkt_block else []
    if market_ctx.get("sectors_up"):
        mkt_lines.append(f"Sectors UP: {market_ctx['sectors_up']}")
    if market_ctx.get("sectors_down"):
        mkt_lines.append(f"Sectors DOWN: {market_ctx['sectors_down']}")
    if market_ctx.get("upcoming_earnings"):
        mkt_lines.append(f"Upcoming earnings: {market_ctx['upcoming_earnings']}")
    if market_ctx.get("news_count"):
        mkt_lines.append(f"News sentiment: {market_ctx.get('bullish_news',0)} bullish, {market_ctx.get('bearish_news',0)} bearish out of {market_ctx['news_count']} headlines")
    # Fall back to frontend prices only if server cache empty
    if not srv_mkt_block:
        if market_ctx.get("us_markets"):
            mkt_lines.append(f"US Markets: {market_ctx['us_markets']}")
        if market_ctx.get("global_markets"):
            mkt_lines.append(f"Global: {market_ctx['global_markets']}")
        if market_ctx.get("assets"):
            mkt_lines.append(f"Crypto/Commodities: {market_ctx['assets']}")
        if market_ctx.get("fx"):
            mkt_lines.append(f"FX: {market_ctx['fx']}")
    prices_block = "\n".join(mkt_lines) or "(live prices not yet loaded)"

    # ── AI portfolio context (so Market AI knows what user holds) ─────────────
    ai_port_block = _build_ai_portfolio_block()

    system_prompt = (
        f"You are Market AI — an intelligent assistant embedded in AlphaAgent, a professional quantitative trading platform.\n\n"
        f"Today: {today_str}\n\n"
        "=== LIVE MARKET PRICES & ALPHAAGENT DATA ===\n"
        f"{prices_block}\n\n"
        "=== USER'S AI PORTFOLIO ===\n"
        f"{ai_port_block}\n\n"
        "=== LIVE MARKET NEWS ===\n"
        f"{news_block}\n\n"
        "=== YOUR CAPABILITIES ===\n"
        "1. General market info: latest news, earnings calendar, macro trends, sector moves, what's driving markets\n"
        "2. Deep analysis: if a user asks to analyze / research / get a signal on a stock, ETF, crypto, index, or commodity — set intent to deep_analysis and provide the ticker\n"
        "3. Ticker correction: if the user mentions an asset with a wrong/misspelled name, correct it and get confirmation\n\n"
        "=== KNOWN ASSETS (use exact ticker symbols) ===\n"
        "Stocks: AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AMD, INTC, JPM, BAC, GS, WMT, HD, DIS, NFLX, XOM, CVX, PFE, JNJ, MRK, UNH, CRM, PYPL, SBUX, NKE, V, MA, COST\n"
        "ETFs: SPY (S&P 500), QQQ (NASDAQ 100), IWM (Russell 2000), GLD (Gold), TLT (Bonds), VTI, VOO, ARKK, XLF, XLK, XLE, IBIT (Bitcoin ETF), ETHA (Ethereum ETF)\n"
        "Indices→ETF proxy: S&P 500→SPY, NASDAQ→QQQ, Dow→DIA, Russell→IWM\n"
        "Crypto: BTC-USD (Bitcoin), ETH-USD (Ethereum), BNB-USD, SOL-USD, XRP-USD\n"
        "Commodities: GC=F (Gold), CL=F (Crude Oil), SI=F (Silver), NG=F (Natural Gas) — or their ETF proxies\n"
        "Mutual Funds: use the ETF equivalent (e.g. Fidelity 500 Index → SPY or VOO)\n\n"
        "=== RESPONSE FORMAT ===\n"
        "Respond naturally and helpfully. Be concise (under 200 words unless detail is requested).\n"
        "Use bullet points for lists. Use **bold** for key terms.\n\n"
        "CRITICAL: End every response with exactly this block (no exceptions):\n"
        "---JSON---\n"
        '{"intent":"general","ticker":null,"correction":null}\n'
        "---END---\n\n"
        'Set "intent" to one of:\n'
        '- "general" — answering a general market/news/earnings question\n'
        '- "deep_analysis" — user wants a detailed quant signal on a specific asset; set "ticker" to the symbol (e.g. "AAPL", "BTC-USD", "GLD")\n'
        '- "confirm_ticker" — user mentioned an asset but the name is ambiguous or possibly misspelled; set "correction" to {"original":"what they said","suggested":"TICKER","suggested_name":"Full Name"}\n\n'
        "RULES:\n"
        '- For deep_analysis: resolve the best ticker. "Apple" → "AAPL", "Bitcoin" → "BTC-USD", "S&P 500" or "SPY" → "SPY", "gold" → "GLD".\n'
        '- For confirm_ticker: only use when genuinely uncertain (e.g. "Goggle stock", "Berkshire A or B?").\n'
        "- Never fabricate prices or returns — you only know today's news headlines above.\n"
        "- Be warm, direct, and actionable. If asked what you can do, explain briefly."
    )

    api_key       = os.environ.get("ANTHROPIC_API_KEY", "")
    gemini_key    = os.environ.get("GEMINI_API_KEY", "")
    ollama_on     = settings.get("ollama.enabled", False)
    prefer_ollama = settings.get("ollama.prefer_ollama", False)
    chat_msgs     = [{"role": h["role"], "content": h["content"]} for h in history[-12:]]
    chat_msgs.append({"role": "user", "content": message})

    raw = ""

    async def _try_ollama_market():
        return await _call_ollama(chat_msgs, system=system_prompt)

    if ollama_on and prefer_ollama:
        try:
            raw = await _try_ollama_market()
        except Exception as oe:
            logger.debug(f"Ollama market-chat fallback: {oe}")
            raw = ""

    if not raw and gemini_key:
        try:
            raw = await _call_gemini(chat_msgs, system=system_prompt, max_tokens=700)
        except Exception as ge:
            logger.debug(f"Gemini market-chat fallback: {ge}")
            raw = ""

    if not raw and api_key:
        try:
            import anthropic as _anthropic
            client = _anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=700,
                system=system_prompt,
                messages=chat_msgs,
            )
            raw = (msg.content[0].text if msg.content else "").strip()
        except Exception as e:
            logger.error(f"Claude market-chat error: {e}")
            raw = ""

    if not raw and ollama_on:
        try:
            raw = await _try_ollama_market()
        except Exception as oe:
            logger.warning(f"Ollama market-chat failed: {oe}")
            raw = ""

    if not raw:
        return {
            "answer": (
                "Market AI is offline — set GEMINI_API_KEY or ANTHROPIC_API_KEY in your .env file, "
                f"or ensure Ollama is running on port {settings.get('ollama.port', 11435)} "
                "and enable it in ⚙️ Settings."
            ),
            "intent": "general",
            "ticker": None,
            "correction": None,
        }

    try:

        # Parse the JSON action block
        intent     = "general"
        ticker     = None
        correction = None
        answer     = raw

        if "---JSON---" in raw:
            parts  = raw.split("---JSON---", 1)
            answer = parts[0].strip()
            tail   = parts[1]
            json_str = tail.split("---END---")[0].strip() if "---END---" in tail else tail.strip()
            try:
                parsed     = _json.loads(json_str)
                intent     = parsed.get("intent", "general")
                ticker     = parsed.get("ticker")
                correction = parsed.get("correction")
            except Exception:
                pass

        return {"answer": answer, "intent": intent, "ticker": ticker, "correction": correction}

    except Exception as e:
        logger.error(f"market_chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Market AI error: {str(e)}")


# ── Portfolio agent-scan helpers ─────────────────────────────────────────────

# ─ Regional comprehensive universes (equities, ETFs, indices, crypto, commodities, forex) ─
_REGIONAL_UNIVERSES: dict[str, dict[str, list]] = {
    "us": {
        "equities":    ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","JPM","BAC","V","MA","XOM","CVX",
                        "UNH","JNJ","WMT","PG","COST","HD","MCD","KO","PEP","ABBV","LLY","MRK","PFE",
                        "AMD","INTC","CRM","ADBE","QCOM","TXN","AVGO","ORCL","NFLX","DIS","GS","MS","BLK",
                        "BRK-B","CAT","BA","GE","RTX","NKE","SBUX","F","GM","PYPL","ABNB","UBER","LYFT",
                        "PLTR","SMCI","ANET","CRWD","SNOW","NET","DDOG","MDB","ZS","OKTA","PANW","FTNT"],
        "indices":     ["^GSPC","^IXIC","^DJI","^RUT","^VIX","^SOX","^NDX"],
        "etfs":        ["SPY","QQQ","IWM","VTI","VOO","DIA","GLD","SLV","TLT","IEF","LQD","HYG",
                        "XLK","XLF","XLE","XLV","XLI","XLY","XLB","XLU","XLP","XLRE",
                        "VNQ","ARKK","ARKG","SCHD","VYM","SPDW","IEMG","BND","AGG","VTIP",
                        "VGT","SOXX","SMH","SKYY","IGV","HACK","BOTZ","ROBO"],
        "commodities": ["GC=F","SI=F","PL=F","PA=F","CL=F","BZ=F","NG=F","RB=F","HO=F",
                        "ZW=F","ZC=F","ZS=F","CC=F","KC=F","CT=F","SB=F","HG=F","ALI=F","ZR=F"],
        "forex":       ["EURUSD=X","GBPUSD=X","JPY=X","AUDUSD=X","CADUSD=X","CHFUSD=X",
                        "NZDUSD=X","MXNUSD=X","DX-Y.NYB"],
        "crypto":      ["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","ADA-USD","AVAX-USD",
                        "DOGE-USD","MATIC-USD","DOT-USD","LINK-USD","UNI-USD","LTC-USD","ATOM-USD"],
        "mutual_funds":["VFIAX","FXAIX","VTSAX","FSKAX","VBTLX","FXNAX","VIMAX","VSMAX"],
    },
    "india": {
        "equities":    ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS","BAJFINANCE.NS",
                        "HINDUNILVR.NS","KOTAKBANK.NS","LT.NS","AXISBANK.NS","ASIANPAINT.NS",
                        "MARUTI.NS","TITAN.NS","WIPRO.NS","SBIN.NS","HCLTECH.NS","SUNPHARMA.NS",
                        "ULTRACEMCO.NS","TATAMOTORS.NS","TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS",
                        "ONGC.NS","BAJAJFINSV.NS","TECHM.NS","NTPC.NS","POWERGRID.NS","COALINDIA.NS",
                        "ADANIPORTS.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","BHARTIARTL.NS",
                        "NESTLEIND.NS","BRITANNIA.NS","HAVELLS.NS","PIIND.NS","MUTHOOTFIN.NS"],
        "indices":     ["^NSEI","^BSESN","^CNXAUTO","^CNXBANK","^CNXIT","^CNXPHARMA","^NSMIDCP100"],
        "etfs":        ["INDA","INDY","SMIN","FLIN","NIFTYBEES.NS","BANKBEES.NS","JUNIORBEES.NS"],
        "commodities": ["GC=F","SI=F","CL=F","NG=F","HG=F"],
        "forex":       ["INR=X","EURINR=X","GBPINR=X","JPYINR=X"],
        "crypto":      ["BTC-USD","ETH-USD","SOL-USD","XRP-USD","BNB-USD","ADA-USD"],
    },
    "europe": {
        "equities":    ["ASML","SAP","AZN","NVO","SHEL","BP","HSBC","UL","GSK","BTI","DEO","PHG",
                        "EQNR","SAN","ING","RY","TD","CNI","CP","BN","INGA.AS","AIR.PA",
                        "OR.PA","MC.PA","BNP.PA","SU.PA","CS.PA","SGO.PA","EDF.PA",
                        "ADS.DE","BMW.DE","MBG.DE","VOW3.DE","SIE.DE","BAS.DE","ALV.DE","MUV2.DE",
                        "NESN.SW","ROG.SW","NOVN.SW","ABBN.SW","UBSG.SW","CSGN.SW"],
        "indices":     ["^FTSE","^GDAXI","^FCHI","^STOXX50E","^AEX","^IBEX","^SSMI","^OSEAX"],
        "etfs":        ["EFA","FEZ","EWU","EWG","EWQ","EWI","EWP","EWD","EWN","IEUR","HEZU","VGK","DBEU"],
        "commodities": ["GC=F","SI=F","CL=F","NG=F","HG=F","ZW=F"],
        "forex":       ["EURUSD=X","GBPUSD=X","CHFUSD=X","SEKUSD=X","NOKUSD=X","DKKUSD=X","PLNUSD=X"],
        "crypto":      ["BTC-USD","ETH-USD","SOL-USD","XRP-USD","ADA-USD"],
    },
    "asia": {
        "equities":    ["SONY","TM","HMC","NTDOY","SFTBY","MUFG","NTT","FANUY","KYCCF",
                        "BABA","TCEHY","JD","BIDU","PDD","NIO","XPEV","LI","NTES","BEKE",
                        "005930.KS","000660.KS","035420.KS","035720.KS","051910.KS",
                        "9988.HK","0700.HK","1299.HK","0005.HK","0941.HK","2318.HK",
                        "BHP","RIO","CBA.AX","ANZ.AX","WBC.AX","NAB.AX","WES.AX"],
        "indices":     ["^N225","^HSI","^KS11","000001.SS","^AXJO","^STI","^TWII","^JKSE","^KLSE"],
        "etfs":        ["EWJ","FXI","MCHI","EWH","EWY","EWA","VPL","AAXJ","CQQQ","KWEB","ASHR"],
        "commodities": ["GC=F","SI=F","CL=F","NG=F","HG=F","ZS=F"],
        "forex":       ["JPY=X","CNY=X","AUDUSD=X","SGDUSD=X","HKDUSD=X","KRWUSD=X","TWDUSD=X","IDRUSD=X"],
        "crypto":      ["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","ADA-USD"],
    },
    "japan": {
        "equities":    ["SONY","TM","HMC","NTDOY","SFTBY","MUFG","NTT","FANUY","KYCCF",
                        "7203.T","6758.T","6861.T","9984.T","8306.T","9432.T","4063.T",
                        "6501.T","6702.T","7267.T","4661.T","9613.T","6098.T","8031.T"],
        "indices":     ["^N225","^N300","^TOPX"],
        "etfs":        ["EWJ","DXJ","DBJP","HEWJ","SCJ","JPXN"],
        "commodities": ["GC=F","SI=F","CL=F","NG=F"],
        "forex":       ["JPY=X","EURJPY=X","GBPJPY=X","AUDJPY=X"],
        "crypto":      ["BTC-USD","ETH-USD","SOL-USD","XRP-USD"],
    },
    "china": {
        "equities":    ["BABA","TCEHY","JD","BIDU","PDD","NIO","XPEV","LI","NTES","BEKE",
                        "9988.HK","0700.HK","1299.HK","0005.HK","0941.HK","2318.HK",
                        "3690.HK","1810.HK","9999.HK","2020.HK","0175.HK"],
        "indices":     ["000001.SS","^HSI","^HSCE","000300.SS","399001.SZ"],
        "etfs":        ["FXI","MCHI","CQQQ","KWEB","ASHR","CNXT","KURE","PGJ","GXC"],
        "commodities": ["GC=F","SI=F","CL=F","NG=F","HG=F"],
        "forex":       ["CNY=X","EURCNY=X","GBPCNY=X","JPYCNY=X","HKDUSD=X"],
        "crypto":      ["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD"],
    },
    "global": {
        "equities":    ["AAPL","MSFT","NVDA","AMZN","TSLA","META","BABA","TM","ASML","SAP","NVO",
                        "SHEL","RIO","BHP","VALE","RELIANCE.NS","TCS.NS","SONY","LVMH.PA"],
        "indices":     ["^GSPC","^IXIC","^FTSE","^N225","^HSI","^GDAXI","000001.SS","^AXJO"],
        "etfs":        ["SPY","EFA","VWO","GLD","TLT","VT","ACWI","IEMG","VEU","VXUS"],
        "commodities": ["GC=F","SI=F","CL=F","NG=F","HG=F","ZW=F","ZC=F","CC=F","KC=F"],
        "forex":       ["EURUSD=X","GBPUSD=X","JPY=X","AUDUSD=X","DX-Y.NYB","CNY=X","INR=X"],
        "crypto":      ["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","ADA-USD","AVAX-USD","DOGE-USD"],
    },
}

# Flat portfolio creation universe (by theme) — used by _detect_universe for chat
_PORTFOLIO_UNIVERSES = {
    "us_tech":   ["NVDA","MSFT","AAPL","AMZN","GOOGL","META","AMD","TSLA","CRM","ADBE","AVGO","ORCL"],
    "ai":        ["NVDA","MSFT","GOOGL","AMD","META","PLTR","ORCL","CRM","AVGO","SMCI","ANET","CRWD"],
    "us_large":  ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","JPM","V","UNH","XOM","JNJ","WMT","PG","COST"],
    "india":     ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS","BAJFINANCE.NS","HINDUNILVR.NS","TITAN.NS","WIPRO.NS","MARUTI.NS","SBIN.NS","LT.NS"],
    "finance":   ["JPM","BAC","GS","MS","BLK","V","MA","AXP","WFC","C","USB","PNC"],
    "health":    ["UNH","JNJ","LLY","ABBV","PFE","MRK","TMO","ABT","AMGN","MDT","CVS","CI"],
    "energy":    ["XOM","CVX","COP","SLB","OXY","PSX","VLO","MPC","EOG","BP"],
    "dividend":  ["SCHD","VYM","T","KO","JNJ","PG","XOM","CVX","MO","VZ","O","ABBV"],
    "crypto":    ["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","ADA-USD","AVAX-USD","DOGE-USD"],
    "global":    ["SPY","EFA","VWO","INDA","EWJ","EWZ","GLD","TLT","EEM","FXI"],
    "balanced":  ["SPY","QQQ","TLT","GLD","VTI","IWM","AAPL","MSFT","JPM","XOM"],
    "semis":     ["NVDA","AMD","INTC","AVGO","QCOM","MU","AMAT","LRCX","KLAC","TSM"],
    "consumer":  ["AMZN","TSLA","COST","WMT","HD","TGT","MCD","SBUX","NKE","LOW"],
    "europe":    ["ASML","SAP","AZN","NVO","SHEL","BP","HSBC","UL","GSK","DEO","EQNR","SAN"],
    "asia":      ["SONY","TM","BABA","TCEHY","JD","BIDU","HMC","NTDOY","BHP","RIO"],
    "metals":    ["GLD","SLV","GDX","GDXJ","PPLT","PALL","FCX","NEM","GOLD","WPM"],
    "commodities":["GLD","SLV","USO","UNG","DBA","DJP","PDBC","COMT","BCI","CRBQ"],
}

# Current user region (updated by set-region endpoint from frontend timezone detection)
_USER_REGION: str = "us"

# All unique tickers for background quote warming (from all regions, all asset classes)
def _build_all_tickers() -> list[str]:
    seen = {}
    for region_data in _REGIONAL_UNIVERSES.values():
        for cat_tickers in region_data.values():
            for t in cat_tickers:
                seen[t] = True
    # Also include portfolio universe tickers
    for tickers in _PORTFOLIO_UNIVERSES.values():
        for t in tickers:
            seen[t] = True
    return list(seen.keys())

_ALL_UNIVERSE_TICKERS: list[str] = _build_all_tickers()

def _is_signable(ticker: str) -> bool:
    """True if this ticker can run through graph.invoke() (equities, ETFs, crypto)."""
    t = ticker.upper()
    if t.startswith("^"):    return False   # market indices
    if t.endswith("=F"):     return False   # futures / commodities
    if t.endswith("=X"):     return False   # forex pairs
    if t.endswith(".NYB"):   return False   # NYB-suffixed (DXY)
    return True

def _detect_universe(message: str) -> list:
    m = message.lower()
    if any(x in m for x in ["india", "nifty", "sensex", "bse", ".ns"]):
        return _PORTFOLIO_UNIVERSES["india"]
    if any(x in m for x in ["crypto", "bitcoin", "btc", "ethereum", "eth", "defi"]):
        return _PORTFOLIO_UNIVERSES["crypto"]
    if any(x in m for x in ["semiconductor", "chip", "semi", "foundry"]):
        return _PORTFOLIO_UNIVERSES["semis"]
    if any(x in m for x in [" ai ", "artificial intelligence", "machine learning", "llm"]):
        return _PORTFOLIO_UNIVERSES["ai"]
    if any(x in m for x in ["tech", "technology", "software", "saas", "nasdaq"]):
        return _PORTFOLIO_UNIVERSES["us_tech"]
    if any(x in m for x in ["bank", "financ", "wall street", "fintech"]):
        return _PORTFOLIO_UNIVERSES["finance"]
    if any(x in m for x in ["health", "pharma", "biotech", "medical", "drug"]):
        return _PORTFOLIO_UNIVERSES["health"]
    if any(x in m for x in ["energy", "oil", "gas", "petroleum", "refin"]):
        return _PORTFOLIO_UNIVERSES["energy"]
    if any(x in m for x in ["dividend", "income", "yield", "payout"]):
        return _PORTFOLIO_UNIVERSES["dividend"]
    if any(x in m for x in ["global", "international", "world", "emerging", "europe", "japan"]):
        return _PORTFOLIO_UNIVERSES["global"]
    if any(x in m for x in ["consumer", "retail", "shop", "brand"]):
        return _PORTFOLIO_UNIVERSES["consumer"]
    if any(x in m for x in ["balanced", "diversif", "conservative", "moderate", "mix"]):
        return _PORTFOLIO_UNIVERSES["balanced"]
    return _PORTFOLIO_UNIVERSES["us_large"]

def _is_create_request(message: str) -> bool:
    m = message.lower()
    return any(x in m for x in ["create", "build", "make", "put", "construct", "portfolio", "invest", "allocat"])

async def _scan_signals_for_portfolio(tickers: list, total_timeout: float = 20.0) -> list:
    """Run AlphaAgent signals for up to 5 tickers. Uses cache when available, fresh otherwise.
    Entire scan runs in a thread executor so it never blocks the event loop."""
    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

    def _run_one(sym: str) -> dict | None:
        key = (sym.upper(), "1d")
        cached = _SIGNAL_CACHE.get(key)
        if cached and time.time() < cached[1]:
            r = cached[0]
            return {
                "ticker":     sym,
                "direction":  r.get("direction", "NEUTRAL"),
                "conviction": round(float(r.get("conviction", 0)), 1),
                "prob_up":    round(float(r.get("probability", 0.5)) * 100, 1),
                "cached":     True,
            }
        try:
            md    = _get_market_data(sym.upper())
            state = {"ticker": sym.upper(), "market_data": md, "registry": registry}
            res   = graph.invoke(state)
            fi    = res.get("final_signal", {})
            pkt   = fi.get("packet")
            if not pkt:
                return None
            prob_up = fi.get("probability_up", 0.5)
            direction = "LONG" if prob_up >= 0.53 else "SHORT" if prob_up <= 0.47 else "NEUTRAL"
            _SIGNAL_CACHE[key] = ({
                "direction": direction,
                "conviction": float(getattr(pkt, "conviction", 0)),
                "probability": prob_up,
                "agents": [{"agent_name": a.agent_name, "vote": a.vote} for a in getattr(pkt, "agent_results", [])],
            }, time.time() + _SIGNAL_TTL)
            return {
                "ticker":     sym,
                "direction":  direction,
                "conviction": round(float(getattr(pkt, "conviction", 0)), 1),
                "prob_up":    round(prob_up * 100, 1),
                "cached":     False,
            }
        except Exception as _e:
            logger.warning(f"signal scan {sym}: {_e}")
            return None

    def _sync_scan_all() -> list:
        # Separate cached vs uncached to avoid heavy work for cached hits
        cap = tickers[:8]
        cached_results = []
        to_fetch = []
        for sym in cap:
            key = (sym.upper(), "1d")
            c = _SIGNAL_CACHE.get(key)
            if c and time.time() < c[1]:
                r = c[0]
                cached_results.append({
                    "ticker": sym, "direction": r.get("direction", "NEUTRAL"),
                    "conviction": round(float(r.get("conviction", 0)), 1),
                    "prob_up": round(float(r.get("probability", 0.5)) * 100, 1),
                    "cached": True,
                })
            else:
                to_fetch.append(sym)

        # Run fresh signal calls for up to 3 uncached tickers, 2 workers max
        fresh = []
        if to_fetch:
            fetch_cap = to_fetch[:3]
            with ThreadPoolExecutor(max_workers=2) as ex:
                futs = {ex.submit(_run_one, t): t for t in fetch_cap}
                for fut in _as_completed(futs, timeout=total_timeout - 2):
                    try:
                        r = fut.result(timeout=2)
                        if r:
                            fresh.append(r)
                    except Exception:
                        pass
        return cached_results + fresh

    loop = asyncio.get_running_loop()
    try:
        results = await asyncio.wait_for(
            loop.run_in_executor(None, _sync_scan_all),
            timeout=total_timeout,
        )
    except asyncio.TimeoutError:
        logger.warning("signal scan timed out — returning empty")
        results = []
    return results


@app.post("/api/v1/portfolio-scan")
async def portfolio_scan(body: dict):
    """Scan pre-warmed signal cache and return top picks for given region/asset_type."""
    import json as _json

    region     = (body.get("region") or "all").lower()
    asset_type = (body.get("asset_type") or "all").lower()
    budget     = float(body.get("budget") or 100_000)
    exclude    = {t.upper() for t in (body.get("exclude") or [])}

    # Map frontend asset_type → universe category key
    _type_map = {
        "stocks":      "equities",
        "etfs":        "etfs",
        "mutual_fund": "mutual_funds",
        "index":       "indices",
        "commodities": "commodities",
        "forex":       "forex",
        "crypto":      "crypto",
    }
    cat_key = _type_map.get(asset_type)   # None means all categories

    # Collect tickers with region+category metadata
    universes = (
        {region: _REGIONAL_UNIVERSES[region]}
        if region in _REGIONAL_UNIVERSES
        else _REGIONAL_UNIVERSES
    )
    seen: dict[str, bool] = {}
    ticker_cat: dict[str, dict] = {}   # ticker → {region, category, label}
    _cat_labels = {
        "equities": "Stocks", "etfs": "ETFs", "crypto": "Crypto",
        "mutual_funds": "Mutual Funds", "indices": "Indices",
        "commodities": "Commodities", "forex": "Forex",
    }
    for _rname, _rdata in universes.items():
        cats = [cat_key] if cat_key else list(_rdata.keys())
        for cat in cats:
            for t in _rdata.get(cat, []):
                if t not in seen:
                    seen[t] = True
                    ticker_cat[t] = {
                        "region":   _rname,
                        "category": cat,
                        "label":    f"{_rname.upper()} {_cat_labels.get(cat, cat.title())}",
                    }
    tickers = [t for t in seen.keys() if t.upper() not in exclude]

    # ── On-demand fallback: if very few tickers cached, run signals now ────
    # Region-aware priority — uses India/Europe/Asia/Japan/China/Global priority
    # tickers when user is browsing a non-US market.
    # Bounded to keep response time reasonable: top 18 tickers, 4 workers, 90s budget.
    now_ts = time.time()
    cached_count = sum(
        1 for t in tickers
        if (_SIGNAL_CACHE.get((t.upper(), "1d"))
            and now_ts < _SIGNAL_CACHE[(t.upper(), "1d")][1])
    )
    if cached_count < 8 and tickers:
        ticker_set = {tk.upper() for tk in tickers}
        # Use region-specific priority if scanning a single region
        region_priority = _get_regional_priority(region) if region != "all" else _PRIORITY_CORE
        priority_order = [t for t in region_priority if t.upper() in ticker_set]
        priority_order += [t.upper() for t in tickers if t.upper() not in priority_order]
        await _on_demand_warm_signals(
            priority_order[:18],
            max_workers=4,
            per_ticker_timeout=35,
            total_budget=90,
        )
        now_ts = time.time()   # refresh after warm

    picks = []
    for ticker in tickers:
        t_upper = ticker.upper()
        sig_entry = (
            _SIGNAL_CACHE.get((t_upper, "1d")) or
            _SIGNAL_CACHE.get((t_upper, "1m")) or
            _SIGNAL_CACHE.get((ticker, "1d")) or
            _SIGNAL_CACHE.get((ticker, "1m"))
        )
        if not sig_entry or now_ts > sig_entry[1]:
            continue
        sig = sig_entry[0]
        direction = sig.get("direction", "NEUTRAL")
        if direction == "NEUTRAL":
            continue
        conviction  = float(sig.get("conviction", 0) or 0)
        probability = float(sig.get("probability", sig.get("prob_up", 50)) or 50)
        if probability > 1:
            probability = probability / 100.0

        # Derive conviction from probability strength when the packet didn't set it
        if conviction < 0.01:
            conviction = round(max(0.05, abs(probability - 0.5) * 2), 3)

        # Score: combine probability strength + conviction
        prob_strength = abs(probability - 0.5) * 2   # 0→1 scale

        quote    = _LIVE_QUOTE_CACHE.get(t_upper) or {}
        price    = quote.get("price") or 0
        meta     = _TICKER_NAME_MAP.get(ticker, {}) or _TICKER_NAME_MAP.get(t_upper, {})
        cat_info = ticker_cat.get(ticker, ticker_cat.get(t_upper, {}))

        # Reasoning from agent votes — clean up agent names
        _name_fix = {
            "technical": "Technical Analysis", "fundamental": "Fundamental Analysis",
            "macro": "Macro Outlook", "sentiment": "Market Sentiment",
            "geopolitical": "Geopolitical", "insider": "Insider Activity",
            "risk": "Risk Model", "bayesian": "Bayesian Model",
            "factor": "Factor Model", "volatility": "Volatility Model",
        }
        agents  = sig.get("agents", [])
        bullish = [_name_fix.get(a["agent_name"].replace("_agent","").lower(),
                   a["agent_name"].replace("_agent","").replace("_"," ").title())
                   for a in agents if a.get("vote") in ("LONG","BUY","bullish")]
        bearish = [_name_fix.get(a["agent_name"].replace("_agent","").lower(),
                   a["agent_name"].replace("_agent","").replace("_"," ").title())
                   for a in agents if a.get("vote") in ("SHORT","SELL","bearish")]

        if direction == "LONG":
            if bullish:
                reasoning = f"{len(bullish)} agents bullish: {', '.join(bullish[:4])}"
            else:
                reasoning = f"Quantitative: {round(probability*100)}% probability upside"
        else:
            if bearish:
                reasoning = f"{len(bearish)} agents bearish: {', '.join(bearish[:4])}"
            else:
                reasoning = f"Quantitative: {round(probability*100)}% probability downside"

        # Exit hint — use probability strength when conviction is low
        signal_strength = max(conviction, prob_strength)
        if direction == "LONG":
            if signal_strength >= 0.70:
                stop  = round(price * 0.95, 2) if price else None
                tgt   = round(price * 1.12, 2) if price else None
                exit_hint = f"Strong signal. Target ${tgt} (+12%). Stop ${stop} (−5%)."
            elif signal_strength >= 0.40:
                stop  = round(price * 0.97, 2) if price else None
                tgt   = round(price * 1.07, 2) if price else None
                exit_hint = f"Hold 2-4 weeks. Target ${tgt} (+7%). Stop ${stop} (−3%)."
            else:
                stop  = round(price * 0.97, 2) if price else None
                exit_hint = f"Moderate signal. Stop ${stop} (−3%). Exit on reversal."
        else:
            if signal_strength >= 0.70:
                tgt   = round(price * 0.90, 2) if price else None
                exit_hint = f"Strong short. Cover at ${tgt} (−10%). Stop on volume spike."
            elif signal_strength >= 0.40:
                tgt   = round(price * 0.95, 2) if price else None
                exit_hint = f"Short-term weakness. Cover ${tgt} (−5%). 1-2 week hold."
            else:
                exit_hint = "Weak short signal. Cover quickly on any bounce."

        picks.append({
            "ticker":         t_upper,
            "name":           meta.get("name", ticker),
            "sector":         meta.get("sector", ""),
            "asset_category": cat_info.get("label", ""),
            "region":         cat_info.get("region", region),
            "category":       cat_info.get("category", ""),
            "direction":      direction,
            "action":         "BUY" if direction == "LONG" else "SELL SHORT",
            "conviction":     round(conviction, 3),
            "probability":    round(probability, 3),
            "score":          round(prob_strength * 0.6 + conviction * 0.4, 4),
            "price":          quote.get("price"),
            "change_pct":     quote.get("change_pct"),
            "positive":       quote.get("positive", True),
            "reasoning":      reasoning,
            "exit_hint":      exit_hint,
        })

    longs  = sorted([p for p in picks if p["direction"] == "LONG"],  key=lambda x: -x["score"])
    shorts = sorted([p for p in picks if p["direction"] == "SHORT"], key=lambda x: -x["score"])
    # best 25 overall — keep at least 1 short if available, fill rest with longs
    n_short = min(len(shorts), 5)
    n_long  = min(len(longs),  25 - n_short)
    top     = sorted(longs[:n_long] + shorts[:n_short], key=lambda x: -x["score"])

    # ── Risk-based allocation ────────────────────────────────────────────────
    # Weight each pick by its score (prob_strength × 0.6 + conviction × 0.4)
    # Bounds: 5% min, 25% max per position, then renormalise to 100%
    if top:
        raw_scores = [max(p["score"], 0.02) for p in top]
        total_raw  = sum(raw_scores)
        # First pass: proportional weights clipped to [5%, 25%]
        clipped = [max(5.0, min(25.0, (s / total_raw) * 100)) for s in raw_scores]
        total_clipped = sum(clipped)
        for p, pct in zip(top, clipped):
            norm_pct = round(pct / total_clipped * 100, 1)
            p["suggested_pct"]     = norm_pct
            p["suggested_capital"] = round((norm_pct / 100) * budget)

    try:
        db_manager.record_scan_result(region, asset_type, budget, top)
    except Exception:
        pass  # non-critical — don't fail the scan if history write fails

    return {
        "picks":          top,
        "total_scanned":  len(tickers),
        "signals_cached": len(picks),
        "longs_found":    len(longs),
        "shorts_found":   len(shorts),
        "budget":         budget,
        "region":         region,
        "asset_type":     asset_type,
    }


@app.post("/api/v1/portfolio-chat")
async def portfolio_chat(body: dict):
    """Portfolio AI assistant."""
    import os as _os, json as _json
    from datetime import date as _date

    try:
        message        = (body.get("message") or "").strip()
        history        = body.get("history") or []
        portfolio      = body.get("portfolio") or []
        budget         = float(body.get("budget") or 100000)
        entry_date     = (body.get("entry_date") or "").strip() or "today"
        exit_date_pref = (body.get("exit_date_pref") or "").strip()  # empty = AI decides
    except Exception as _pe:
        logger.error(f"portfolio-chat param error: {_pe}\n{_tb.format_exc()}")
        raise HTTPException(status_code=400, detail=str(_pe))

    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    try:
        today_str = _date.today().isoformat()
    except Exception as _te:
        logger.error(f"portfolio-chat date error: {_te}")
        today_str = "2026-01-01"

    port_lines = ""
    if portfolio:
        port_lines = "\n".join(
            f"  {p.get('ticker','?')} — {p.get('name','?')} | ${p.get('allocated',0):,.0f} ({p.get('pct',0):.0f}%) | "
            f"Entry: ${p.get('entry',0):.2f} | Current: ${p.get('current',p.get('entry',0)):.2f} | "
            f"P&L: {'+' if p.get('pnl',0)>=0 else ''}${p.get('pnl',0):,.0f}"
            for p in portfolio
        )
    else:
        port_lines = "  (no positions yet)"

    # ── Run AlphaAgent signals when user is creating a portfolio ─────────────
    signal_block = ""
    if _is_create_request(message) and not portfolio:
        try:
            universe = _detect_universe(message)
            logger.info(f"portfolio-chat: scanning {len(universe)} tickers with AlphaAgent — {universe}")
            scan = await _scan_signals_for_portfolio(universe, total_timeout=25.0)
            longs  = [s for s in scan if s["direction"] == "LONG"]
            shorts = [s for s in scan if s["direction"] == "SHORT"]
            others = [s for s in scan if s["direction"] not in ("LONG", "SHORT")]
            longs.sort(key=lambda x: -x["conviction"])
            cached_count = sum(1 for s in scan if s.get("cached"))
            signal_block = (
                f"\n\n=== ALPHAAGENT LIVE SIGNAL SCAN ({len(scan)} stocks, {cached_count} cached) ===\n"
                "These are REAL quantitative signals from AlphaAgent's 9-agent ensemble (Technical, Fundamental, Macro, Sentiment, Geopolitical, Insider, Risk, Bayesian, Factor).\n\n"
                "LONG signals (strong BUY candidates):\n" +
                "\n".join(
                    f"  {s['ticker']}: conviction={s['conviction']:.1f}  P(up)={s['prob_up']:.1f}%"
                    + (" [CACHED]" if s.get("cached") else " [LIVE]")
                    for s in longs
                ) + ("\n\nSHORT signals (avoid or short):\n" +
                "\n".join(f"  {s['ticker']}: conviction={s['conviction']:.1f}  P(up)={s['prob_up']:.1f}%"
                          for s in shorts) if shorts else "") +
                ("\n\nNEUTRAL / no clear signal:\n" +
                "\n".join(f"  {s['ticker']}: P(up)={s['prob_up']:.1f}%"
                          for s in others) if others else "") +
                "\n\nINSTRUCTION: Build the portfolio ONLY from LONG signals. "
                "Size positions proportional to conviction score — higher conviction = larger allocation. "
                "Exclude SHORT and NEUTRAL stocks. If fewer than 3 LONG signals exist, use the highest P(up) stocks."
            )
            # Append live prices for the universe tickers from the pre-warmed cache
            now_ts = time.time()
            price_rows = []
            for s in scan:
                q = _LIVE_QUOTE_CACHE.get(s["ticker"].upper())
                if q and q.get("expires", 0) > now_ts:
                    chg = q.get("change_pct", 0)
                    price_rows.append(
                        f"  {s['ticker']}: ${q['price']:.2f}  ({'+' if chg>=0 else ''}{chg:.2f}% today)"
                    )
            if price_rows:
                signal_block += "\n\nLIVE PRICES (pre-fetched by AlphaAgent background warmer):\n" + "\n".join(price_rows)
            logger.info(f"portfolio-chat signal scan: {len(longs)} LONG, {len(shorts)} SHORT, {len(others)} NEUTRAL")
        except Exception as _se:
            logger.warning(f"portfolio-chat signal scan failed: {_se}")
            signal_block = ""

    try:
        _sp_test = f"test {today_str} {budget:,.0f} {port_lines[:10]}"
        logger.info(f"portfolio-chat sp_test OK: {_sp_test}")
    except Exception as _spe:
        logger.error(f"portfolio-chat fstring test error: {_spe}\n{_tb.format_exc()}")

    # Parse budget from message text if user mentioned a value (e.g. "$100k", "$50,000")
    import re as _re
    _bm = _re.search(r'\$\s*(\d[\d,]*\.?\d*)\s*k\b', message, _re.IGNORECASE)
    if _bm:
        budget = float(_bm.group(1).replace(',', '')) * 1000
    else:
        _bm2 = _re.search(r'\$\s*(\d[\d,]+)', message)
        if _bm2:
            budget = float(_bm2.group(1).replace(',', ''))

    exit_date_line = (
        f"User's preferred exit date: {exit_date_pref} — honour it unless fundamentally unsound."
        if exit_date_pref else
        "Exit date: NOT specified — YOU must recommend the optimal exit date for each position based on signal conviction, technical levels, and market conditions."
    )

    # ── Enrich with server-side AlphaAgent data ──────────────────────────────
    _live_mkt = _build_market_context_block()
    _ai_port  = _build_ai_portfolio_block()
    # FULL platform context (all regions, signals, leaderboard)
    try:
        _plat_block = _build_platform_context(
            region=None,
            include_signals=True,
            include_portfolio=False,   # already shown above
            include_quotes=False,      # already in _live_mkt
            include_leaderboard=True,
            max_signals_per_region=6,
        )
    except Exception:
        _plat_block = ""

    # Detect "invest $X by [date], give me profit" intent → run strategy build now
    auto_strategy_block = ""
    try:
        _intent = _detect_invest_intent(message)
        if _intent and not portfolio:   # don't auto-build if user already has positions
            logger.info(f"portfolio-chat invest intent: {_intent}")
            _auto = await _auto_strategy_build_for_chat(_intent)
            if _auto:
                auto_strategy_block = _format_strategy_result_for_chat(_auto, _intent)
    except Exception as _ie:
        logger.warning(f"portfolio-chat intent detect err: {_ie}")

    # ── Detect ACTION intent (buy / sell / add / rebalance / confirm) ───────
    # Runs the signal for the target ticker, formats a recommendation block,
    # and executes the action server-side if 'auto' or 'confirm' was detected.
    action_block  = ""
    action_result = None
    try:
        ai_port_positions = _AI_PORTFOLIO.get("positions", [])
        merged_port = list(portfolio or []) + list(ai_port_positions)
        a_intent = _detect_action_intent(message, history, merged_port)
        if a_intent:
            logger.info(f"portfolio-chat action intent: {a_intent}")
            action_block = _build_action_recommendation_block(a_intent)
            # If user confirmed or asked for auto execution → run it now
            if a_intent.get("auto") or a_intent.get("confirm"):
                action_result = _execute_chat_action(a_intent)
                if action_result:
                    action_block += f"\n\nEXECUTION RESULT: {action_result.get('detail','')}"
    except Exception as _ae:
        logger.warning(f"portfolio-chat action intent err: {_ae}")

    # Augment port_lines with fresh live prices from cache for existing positions
    try:
        now_ts2 = time.time()
        for p in portfolio:
            sym2 = p.get("ticker", "").upper()
            q2   = _LIVE_QUOTE_CACHE.get(sym2)
            if q2 and q2.get("expires", 0) > now_ts2 and not p.get("current"):
                p["current"] = q2["price"]
    except Exception:
        pass

    system_prompt = f"""You are Portfolio AI — an expert portfolio advisor embedded in AlphaAgent.
Today: {today_str}
User budget: ${budget:,.0f}
{exit_date_line}

=== LIVE PORTFOLIO (from user session) ===
{port_lines}

=== AI PORTFOLIO (committed positions tracked by AlphaAgent) ===
{_ai_port}

=== LIVE MARKET CONTEXT (AlphaAgent live caches) ===
{_live_mkt or '(market data loading — try again in a moment)'}

=== ALPHAAGENT PLATFORM STATE (signals across regions + leaderboard) ===
{_plat_block or '(no platform signals cached yet)'}
{auto_strategy_block}
{action_block}
=== YOUR ROLE ===
You are an ADVISOR, not a trade executor. Trades are placed only through the Signal Scan Results table.
Your job is to help the user understand their portfolio, make better decisions, and time entries/exits.

You can:
1. SUMMARIZE — P&L breakdown, allocation analysis, overall portfolio health
2. ADVISE ON EXIT — when/how to sell a specific position (give concrete price targets and dates)
3. ADVISE ON SIZING — how much of a position to trim or add to
4. RISK ANALYSIS — concentration risk, correlation, drawdown exposure, stop-loss review
5. MARKET CONTEXT — explain why a stock is moving, sector tailwinds/headwinds
6. REBALANCE — suggest reallocation between existing holdings
7. EXPLAIN SIGNALS — what the AlphaAgent signals mean for specific tickers

=== HOW TO GIVE ADVICE ===
- Always reference specific positions from the user's portfolio
- Give concrete numbers: "Sell 50% of NVDA at $280", "Stop loss at $195"
- If no portfolio yet, tell user to run Signal Scan and select stocks from the results
- Be direct and concise. No generic disclaimers.
- For exit timing: use entry price, current price, and conviction to recommend specific date/price

=== RESPONSE FORMAT ===
Start with the JSON block, then give your analysis.

---PORTFOLIO_JSON---
{{"action":"general"}}
---END---

Then write your advice in plain markdown (use **bold** for key numbers/dates). Max 150 words.
Keep it practical and specific to the user's actual positions."""

    chat_msgs = [{"role": h["role"], "content": h["content"]} for h in history[-10:]]
    chat_msgs.append({"role": "user", "content": message})

    gemini_key = _os.environ.get("GEMINI_API_KEY", "")
    api_key    = _os.environ.get("ANTHROPIC_API_KEY", "")
    ollama_on  = settings.get("ollama.enabled", False)
    raw = ""

    if gemini_key:
        try:
            raw = await _call_gemini(chat_msgs, system=system_prompt, max_tokens=4000)
        except Exception as ge:
            logger.warning(f"Gemini portfolio-chat error: {ge}", exc_info=True)

    if not raw and api_key:
        try:
            import anthropic as _an
            client = _an.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=4000,
                system=system_prompt, messages=chat_msgs,
            )
            raw = (msg.content[0].text if msg.content else "").strip()
        except Exception as e:
            logger.error(f"Claude portfolio-chat error: {e}")

    if not raw and ollama_on:
        try:
            raw = await _call_ollama(chat_msgs, system=system_prompt)
        except Exception: pass

    logger.info(f"portfolio-chat raw length: {len(raw)}, gemini_key_set: {bool(gemini_key)}")
    if not raw:
        return {"answer": "Portfolio AI is offline — set GEMINI_API_KEY in your .env file.", "action": "general", "positions": []}

    # Parse JSON block — JSON is now FIRST, explanation follows ---END---
    action    = "general"
    positions = []
    answer    = raw

    portfolios = []
    if "---PORTFOLIO_JSON---" in raw:
        pre, rest = raw.split("---PORTFOLIO_JSON---", 1)
        if "---END---" in rest:
            json_str, post = rest.split("---END---", 1)
            answer = post.strip() or pre.strip() or raw
        else:
            json_str = rest
            answer   = pre.strip() or raw
        try:
            parsed    = _json.loads(json_str.strip())
            action    = parsed.get("action", "general")
            positions = parsed.get("positions", [])
            portfolios = parsed.get("portfolios", [])
        except Exception:
            pass

    response = {"answer": answer, "action": action, "positions": positions, "portfolios": portfolios}
    # Surface the action-agent execution result (if any) so the frontend can
    # show "executed" status without requiring UI changes.
    if action_result:
        response["action_result"] = action_result
    return response


@app.post("/api/v1/ai-portfolio/commit")
async def commit_ai_portfolio(body: dict):
    """Receive AI-planned positions, fetch live entry prices, compute shares, store for live tracking."""
    positions = body.get("positions", [])
    capital   = float(body.get("capital", 100_000))
    if not positions:
        raise HTTPException(status_code=400, detail="positions list is empty")

    from datetime import datetime as _dt, timezone as _tz
    import yfinance as _yf2

    def _fetch_entries():
        from concurrent.futures import ThreadPoolExecutor
        committed = []
        now_entry = _dt.now(_tz.utc)
        with ThreadPoolExecutor(max_workers=min(len(positions), 8)) as ex:
            futs = {p["ticker"]: ex.submit(_fetch_quote_fast, p["ticker"]) for p in positions}
            for p in positions:
                try:
                    q = futs[p["ticker"]].result(timeout=8)
                except Exception:
                    q = None
                entry = (q["price"] if q else None)
                # Try to get day open from a quick history call
                open_price = None
                try:
                    t_obj = _yf2.Ticker(p["ticker"])
                    fi = t_obj.fast_info
                    open_price = getattr(fi, "open", None) or getattr(fi, "regular_market_open", None)
                    if open_price:
                        open_price = round(float(open_price), 4)
                except Exception:
                    pass
                alloc  = (p.get("pct", 0) / 100.0) * capital
                shares = round(alloc / entry, 6) if entry and entry > 0 else 0
                # target / stop from AI plan, or sensible defaults
                target_price  = p.get("target_price") or (round(entry * 1.12, 2) if entry else None)
                stop_loss     = p.get("stop_loss")    or (round(entry * 0.94, 2) if entry else None)
                committed.append({
                    "ticker":           p["ticker"],
                    "name":             p.get("name", p["ticker"]),
                    "pct":              p.get("pct", 0),
                    "thesis":           p.get("thesis", ""),
                    "trade_type":       p.get("trade_type", "Swing Trade"),
                    "allocated":        round(alloc, 2),
                    "shares":           shares,
                    "entry_price":      round(entry, 4) if entry else None,
                    "open_price":       open_price,
                    "target_price":     target_price,
                    "stop_loss":        stop_loss,
                    "exit_date":        p.get("exit_date"),
                    "hold_horizon":     p.get("hold_horizon", "30 days"),
                    "entry_datetime":   now_entry.isoformat(),
                    "direction":        "LONG",
                })
        return committed

    loop = asyncio.get_running_loop()
    try:
        committed = await asyncio.wait_for(loop.run_in_executor(None, _fetch_entries), timeout=25.0)
    except (asyncio.TimeoutError, Exception) as _e:
        logger.error(f"ai-portfolio commit error: {_e}")
        raise HTTPException(status_code=500, detail=str(_e))

    _AI_PORTFOLIO["positions"]     = committed
    _AI_PORTFOLIO["capital"]       = capital
    _AI_PORTFOLIO["created_at"]    = time.time()
    _AI_PORTFOLIO["daily_returns"] = []
    db_manager.save_ai_portfolio(_AI_PORTFOLIO)
    return {"ok": True, "positions": committed, "capital": capital, "positions_count": len(committed)}


@app.get("/api/v1/ai-portfolio/live")
async def get_ai_portfolio_live():
    """Return AI portfolio with live prices and computed P&L / portfolio metrics."""
    if not _AI_PORTFOLIO.get("positions"):
        return {
            "positions": [],
            "summary": {
                "portfolio_value": 0, "total_pnl": 0, "return_pct": 0,
                "win_rate": 0, "sharpe": 0, "positions_count": 0,
                "capital": _AI_PORTFOLIO.get("capital", 100_000),
                "created_at": None,
            },
        }

    positions = _AI_PORTFOLIO["positions"]
    capital   = _AI_PORTFOLIO.get("capital", 100_000)

    def _refresh():
        from concurrent.futures import ThreadPoolExecutor
        live = []
        with ThreadPoolExecutor(max_workers=min(len(positions), 12)) as ex:
            futs = {p["ticker"]: ex.submit(_fetch_quote_fast, p["ticker"]) for p in positions}
            for p in positions:
                try:
                    q = futs[p["ticker"]].result(timeout=8)
                except Exception:
                    q = None
                cur   = (q["price"] if q else None) or p.get("entry_price") or 0
                entry = p.get("entry_price") or cur
                shrs  = p.get("shares", 0)
                pnl   = round((cur - entry) * shrs, 2)
                pnl_p = round((cur - entry) / entry * 100, 4) if entry else 0
                mval  = round(cur * shrs, 2)
                live.append({
                    **p,
                    "current_price": round(cur, 4),
                    "pnl":           pnl,
                    "pnl_pct":       pnl_p,
                    "market_value":  mval,
                    "change_pct":    q.get("change_pct", 0) if q else 0,
                    "change":        q.get("change", 0) if q else 0,
                    "positive":      pnl >= 0,
                })
        return live

    loop = asyncio.get_running_loop()
    try:
        live = await asyncio.wait_for(loop.run_in_executor(None, _refresh), timeout=20.0)
    except (asyncio.TimeoutError, Exception) as _e:
        logger.error(f"ai-portfolio live error: {_e}")
        live = []

    total_value = sum(p["market_value"] for p in live)
    total_pnl   = sum(p["pnl"] for p in live)
    return_pct  = round(total_pnl / capital * 100, 4) if capital else 0
    winners     = [p for p in live if p["pnl"] > 0]
    win_rate    = round(len(winners) / len(live), 4) if live else 0

    rets = [p["pnl_pct"] / 100.0 for p in live]
    avg  = sum(rets) / len(rets) if rets else 0
    var  = sum((r - avg) ** 2 for r in rets) / len(rets) if rets else 0
    std  = var ** 0.5
    sharpe = round((avg / std) * (252 ** 0.5), 2) if std > 1e-9 else 0.0

    return {
        "positions": live,
        "summary": {
            "portfolio_value": round(total_value, 2),
            "total_pnl":       round(total_pnl, 2),
            "return_pct":      return_pct,
            "win_rate":        win_rate,
            "sharpe":          sharpe,
            "positions_count": len(live),
            "capital":         capital,
            "created_at":      _AI_PORTFOLIO.get("created_at"),
        },
    }


@app.delete("/api/v1/ai-portfolio")
async def reset_ai_portfolio():
    """Clear the AI portfolio and persist the reset."""
    _AI_PORTFOLIO["positions"]     = []
    _AI_PORTFOLIO["capital"]       = 100_000
    _AI_PORTFOLIO["created_at"]    = None
    _AI_PORTFOLIO["daily_returns"] = []
    db_manager.save_ai_portfolio(_AI_PORTFOLIO)
    return {"ok": True}


@app.delete("/api/v1/ai-portfolio/position/{ticker}")
async def sell_position(ticker: str):
    """Remove a single position from the portfolio (sell / exit)."""
    sym = ticker.upper()
    before = len(_AI_PORTFOLIO.get("positions", []))
    _AI_PORTFOLIO["positions"] = [p for p in _AI_PORTFOLIO.get("positions", []) if p["ticker"].upper() != sym]
    removed = before - len(_AI_PORTFOLIO["positions"])
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"{sym} not found in portfolio")
    db_manager.save_ai_portfolio(_AI_PORTFOLIO)
    return {"ok": True, "removed": sym, "positions_remaining": len(_AI_PORTFOLIO["positions"])}


@app.post("/api/v1/ai-portfolio/position/{ticker}/buy")
async def buy_more_position(ticker: str, body: dict = {}):
    """Buy additional shares of an existing position at the current live price."""
    sym    = ticker.upper()
    amount = float(body.get("amount", 0))
    if amount <= 0:
        raise HTTPException(status_code=400, detail="amount must be > 0")

    # Get live price
    loop = asyncio.get_running_loop()
    q = await loop.run_in_executor(None, _fetch_quote_fast, sym)
    if not q or not q.get("price"):
        raise HTTPException(status_code=502, detail=f"Could not fetch price for {sym}")

    price  = q["price"]
    shares = round(amount / price, 6)

    # Update existing position or add new
    positions = _AI_PORTFOLIO.get("positions", [])
    existing  = next((p for p in positions if p["ticker"].upper() == sym), None)
    if existing:
        old_shares = existing.get("shares", 0)
        old_alloc  = existing.get("allocated", 0)
        old_entry  = existing.get("entry_price") or price
        new_shares = old_shares + shares
        new_alloc  = old_alloc + amount
        # Weighted average entry price
        avg_entry  = round((old_entry * old_shares + price * shares) / new_shares, 4) if new_shares else price
        existing.update({
            "shares":      round(new_shares, 6),
            "allocated":   round(new_alloc, 2),
            "entry_price": avg_entry,
        })
        # Recalculate pct weights for all positions
        total_alloc = sum(p.get("allocated", 0) for p in positions)
        for p in positions:
            p["pct"] = round(p.get("allocated", 0) / total_alloc * 100) if total_alloc else p["pct"]
    else:
        from datetime import datetime as _dt2, timezone as _tz2
        positions.append({
            "ticker":         sym,
            "name":           sym,
            "pct":            0,
            "thesis":         "Manually added",
            "trade_type":     "Manual",
            "allocated":      round(amount, 2),
            "shares":         shares,
            "entry_price":    round(price, 4),
            "open_price":     q.get("price"),
            "target_price":   round(price * 1.10, 2),
            "stop_loss":      round(price * 0.95, 2),
            "exit_date":      None,
            "hold_horizon":   "—",
            "entry_datetime": _dt2.now(_tz2.utc).isoformat(),
            "direction":      "LONG",
        })
        # Recalculate pct weights
        total_alloc = sum(p.get("allocated", 0) for p in positions)
        for p in positions:
            p["pct"] = round(p.get("allocated", 0) / total_alloc * 100) if total_alloc else p["pct"]
        _AI_PORTFOLIO["positions"] = positions

    db_manager.save_ai_portfolio(_AI_PORTFOLIO)
    return {"ok": True, "ticker": sym, "added_shares": shares, "price": price, "amount": amount}


@app.get("/api/v1/metrics")
async def get_metrics():
    """Returns real-time telemetry for the dashboard."""
    return Monitor.get_summary()

@app.get("/api/v1/history")
async def get_history(limit: int = 50):
    """Retrieves past signals for auditing."""
    db = db_manager.SessionLocal()
    try:
        trades = db.query(Trade).order_by(Trade.timestamp.desc()).limit(limit).all()
        return trades
    finally:
        db.close()

@app.get("/api/v1/portfolio")
async def get_portfolio():
    """Returns current holdings."""
    db = db_manager.SessionLocal()
    try:
        holdings = db.query(Portfolio).all()
        return holdings
    finally:
        db.close()


# ─── Phase 4 Endpoints ───────────────────────────────────────────────────────

@app.get("/api/v1/backtest/{ticker}")
async def run_backtest(ticker: str, period: str = "3y"):
    """
    Runs the quantitative backtester on historical data.
    Returns performance metrics, trade log, and signal accuracy.
    """
    try:
        engine = BacktestEngine()
        result = engine.run(ticker.upper(), period=period)
        m = result.metrics
        response = {
            "ticker": result.ticker,
            "period": period,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "n_signals": result.n_signals,
            "signal_accuracy_pct": result.signal_accuracy,
            "metrics": {
                "total_return_pct": m.total_return_pct,
                "cagr_pct": m.cagr_pct,
                "sharpe_ratio": m.sharpe_ratio,
                "sortino_ratio": m.sortino_ratio,
                "calmar_ratio": m.calmar_ratio,
                "max_drawdown_pct": m.max_drawdown_pct,
                "max_drawdown_duration_days": m.max_drawdown_duration_days,
                "win_rate_pct": m.win_rate_pct,
                "profit_factor": m.profit_factor,
                "total_trades": m.total_trades,
                "alpha_vs_spy": m.alpha_vs_spy,
                "beta_vs_spy": m.beta_vs_spy,
                "information_ratio": m.information_ratio,
                "volatility_ann_pct": m.volatility_ann_pct,
            },
            "trades": [
                {
                    "date": t.date,
                    "direction": t.direction,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "return_pct": t.return_pct,
                    "probability": t.probability,
                    "position_pct": t.position_pct,
                    "regime": t.regime,
                }
                for t in result.trades[-50:]
            ],
        }
        try:
            db_manager.save_backtest(
                ticker=result.ticker,
                strategy="AlphaAgent",
                params={"period": period},
                metrics={
                    "total_return_pct": m.total_return_pct,
                    "sharpe": m.sharpe_ratio,
                    "max_drawdown_pct": m.max_drawdown_pct,
                    "win_rate_pct": m.win_rate_pct,
                    "n_trades": m.total_trades,
                    "cagr_pct": m.cagr_pct,
                },
            )
        except Exception:
            pass
        return response
    except Exception as e:
        logger.error(f"Backtest failed for {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/scan/{universe}")
async def scan_universe(universe: str, top_n: int = 5, direction: str = None):
    """
    Runs the full agent pipeline on every ticker in the named universe.
    Returns up to top_n signals sorted by conviction.
    Available universes: mega_cap, sp500_core, technology, financials, healthcare, energy, consumer
    """
    if universe not in UNIVERSES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown universe '{universe}'. Available: {list(UNIVERSES.keys())}"
        )
    try:
        um = UniverseManager()
        results = um.scan(
            universe=universe,
            top_n=top_n,
            registry=registry,
            direction_filter=direction,
        )
        return {
            "universe": universe,
            "tickers_scanned": len(um.get_universe(universe)),
            "results_returned": len(results),
            "signals": [
                {
                    "ticker": r.ticker,
                    "direction": r.direction,
                    "probability": r.probability,
                    "conviction_pct": r.conviction_pct,
                    "regime": r.regime,
                    "warnings": r.warnings,
                }
                for r in results
            ],
        }
    except Exception as e:
        logger.error(f"Universe scan failed for {universe}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/portfolio/optimize")
async def optimize_portfolio(body: dict):
    """
    Optimizes portfolio weights for a list of tickers.
    Body: {"tickers": ["AAPL", "MSFT"], "method": "max_sharpe", "period": "1y"}
    Methods: max_sharpe | min_variance | risk_parity | signal_weighted
    """
    tickers = body.get("tickers", [])
    method = body.get("method", "max_sharpe")
    period = body.get("period", "1y")
    signals = body.get("signals")  # optional: {"AAPL": 0.65, "MSFT": 0.55}

    if not tickers or len(tickers) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 tickers.")

    try:
        optimizer = PortfolioOptimizer()
        result = optimizer.optimize(tickers, signals=signals, method=method, period=period)
        return {
            "method": result.method,
            "weights": result.weights,
            "expected_return_ann_pct": result.expected_return_ann,
            "expected_vol_ann_pct": result.expected_vol_ann,
            "sharpe_ratio": result.sharpe_ratio,
        }
    except Exception as e:
        logger.error(f"Portfolio optimization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/portfolio/strategy-build")
async def portfolio_strategy_build(body: dict):
    """
    Regime-adaptive portfolio builder.
    Detects live market regime (VIX + SPY vs SMA20), recommends a strategy mode,
    and returns sized positions from the signal cache.

    Body: {
      "capital":   100000,          # total capital (default 100k)
      "mode":      "auto",          # "auto" | "SNIPER" | "CONCENTRATED" | "BALANCED" | "DIVERSIFIED"
      "universe":  ["AAPL", ...],   # optional ticker list; defaults to top US equities
    }
    """
    import yfinance as _yf
    import numpy as _np

    capital  = float(body.get("capital") or 100_000)
    req_mode = (body.get("mode") or "auto").upper()
    custom_univ = body.get("universe") or []
    region     = (body.get("region") or _USER_REGION or "us").lower()
    asset_type = (body.get("asset_type") or "all").lower()

    # ── Strategy mode definitions ───────────────────────────────────────────
    MODES = {
        "SNIPER": {
            "name": "SNIPER", "positions": 3, "pct_each": 0.33,
            "min_conv": 60, "min_prob": 0.60, "sector_max": 1, "deploy_pct": 0.90,
            "risk_level": "VERY HIGH",
            "ytd": "not individually tested",
            "good_day": "$900 – $1,800", "bad_day": "$900 – $1,200",
            "best_for": "EXTREME/HIGH VIX (>25) — few stocks pass conv≥60, those are gems",
            "who_for": "Experienced trader watching screen all day",
        },
        "CONCENTRATED": {
            "name": "CONCENTRATED", "positions": 5, "pct_each": 0.20,
            "min_conv": 40, "min_prob": 0.58, "sector_max": 1, "deploy_pct": 0.90,
            "risk_level": "HIGH",
            "ytd": "+15.5% ($15,521 on $100k, Jan–May 2026)",
            "good_day": "$400 – $800", "bad_day": "$300 – $600",
            "best_for": "ELEVATED/HIGH VIX (18–35) — proven best in volatile conditions",
            "who_for": "Active investor checking in 2–3x/day",
        },
        "BALANCED": {
            "name": "BALANCED", "positions": 10, "pct_each": 0.10,
            "min_conv": 25, "min_prob": 0.56, "sector_max": 2, "deploy_pct": 0.88,
            "risk_level": "MODERATE",
            "ytd": "~+12% (estimated from signal mix)",
            "good_day": "$200 – $450", "bad_day": "$150 – $300",
            "best_for": "Any regime — want both upside and diversification",
            "who_for": "Investor wanting steady growth, checks morning + evening",
        },
        "DIVERSIFIED": {
            "name": "DIVERSIFIED", "positions": 20, "pct_each": 0.05,
            "min_conv": 5, "min_prob": 0.55, "sector_max": 3, "deploy_pct": 0.85,
            "risk_level": "LOW",
            "ytd": "+9.2% ($9,177 on $100k, Jan–May 2026)",
            "good_day": "$100 – $250", "bad_day": "$80 – $180",
            "best_for": "CALM VIX (<18) — learning mode or conservative investor",
            "who_for": "New to AlphaAgent, or set-and-forget investor",
        },
    }

    # ── Regime detection ────────────────────────────────────────────────────
    regime = {"label": "ELEVATED", "vix": 22.0, "spy_now": 0.0, "sma20": 0.0,
              "above20": True, "spy_1d": 0.0, "spy_5d": 0.0}
    try:
        _spy  = _yf.Ticker("SPY").history(period="30d")["Close"].dropna()
        _vix  = float(_yf.Ticker("^VIX").history(period="3d")["Close"].dropna().iloc[-1])
        _sma  = float(_spy.rolling(20).mean().iloc[-1])
        _now  = float(_spy.iloc[-1])
        _1d   = float((_now / _spy.iloc[-2] - 1) * 100) if len(_spy) >= 2 else 0.0
        _5d   = float((_now / _spy.iloc[-6] - 1) * 100) if len(_spy) >= 6 else 0.0
        _lbl  = ("EXTREME" if _vix > 35 else "HIGH" if _vix > 25 else
                 "ELEVATED" if _vix > 18 else "CALM")
        regime = {"label": _lbl, "vix": round(_vix, 1), "spy_now": round(_now, 2),
                  "sma20": round(_sma, 2), "above20": _now > _sma,
                  "spy_1d": round(_1d, 2), "spy_5d": round(_5d, 2)}
    except Exception:
        pass

    # ── Recommend mode from regime ─────────────────────────────────────────
    def _recommend(reg: dict) -> str:
        v = reg["vix"]
        if v > 35:  return "SNIPER"
        if v > 18:  return "CONCENTRATED"
        if reg["above20"] and reg["spy_1d"] > 0: return "BALANCED"
        return "BALANCED"

    recommended = _recommend(regime)
    chosen_mode_key = req_mode if req_mode in MODES else recommended
    mode = MODES[chosen_mode_key]

    # Opportunity day: VIX>25 + market down → loosen conv
    opp_day = regime["vix"] > 25 and regime["spy_1d"] < -1.0
    if opp_day and mode["min_conv"] > 20:
        mode = dict(mode)
        mode["min_conv"] = max(mode["min_conv"] - 10, 20)

    # ── Sector map ─────────────────────────────────────────────────────────
    _SECTOR = {
        "AAPL":"Tech","MSFT":"Tech","NVDA":"Tech","GOOGL":"Tech","META":"Tech",
        "AMZN":"Tech","TSLA":"Tech","AVGO":"Tech","AMD":"Semis","INTC":"Semis",
        "QCOM":"Semis","MU":"Semis","AMAT":"Semis","LRCX":"Semis","KLAC":"Semis",
        "TXN":"Semis","PLTR":"Growth","CRWD":"Growth","SNOW":"Growth","NET":"Growth",
        "ZS":"Growth","DDOG":"Growth","PANW":"Growth","FTNT":"Growth",
        "JPM":"Finance","BAC":"Finance","GS":"Finance","MS":"Finance","BLK":"Finance",
        "V":"Finance","MA":"Finance","AXP":"Finance","UNH":"Health","JNJ":"Health",
        "LLY":"Health","ABBV":"Health","PFE":"Health","MRK":"Health","TMO":"Health",
        "ABT":"Health","CAT":"Indust","DE":"Indust","HON":"Indust","GE":"Indust",
        "RTX":"Indust","LMT":"Indust","NOC":"Indust","UPS":"Indust","XOM":"Energy",
        "CVX":"Energy","COP":"Energy","SLB":"Energy","OXY":"Energy","PSX":"Energy",
        "VLO":"Energy","MPC":"Energy","WMT":"Consumer","COST":"Consumer","HD":"Consumer",
        "TGT":"Consumer","MCD":"Consumer","SBUX":"Consumer","NKE":"Consumer",
        "AMGN":"Consumer","VZ":"CommMedia","T":"CommMedia","NFLX":"CommMedia",
        "DIS":"CommMedia","CMCSA":"CommMedia","CHTR":"CommMedia","AMT":"REIT",
        "PLD":"REIT","NEE":"Utility","DUK":"Utility","SO":"Utility","D":"Utility",
    }
    _SECTOR_CAP = {"CommMedia": 1}

    # ── Region-aware sector map + default universe ──────────────────────────
    # When region != us, swap to that market's universe + sector classification.
    if region != "us":
        # Merge regional sector with US (so cross-listed names still classify)
        _regional_sec = dict(_SECTOR)
        _regional_sec.update(_REGIONAL_SECTOR.get(region, {}))
        _SECTOR = _regional_sec
        default_univ = _get_regional_universe(region, asset_type if asset_type != "all" else None)
        # If asset_type filter returned nothing, fall back to equities
        if not default_univ:
            default_univ = _get_regional_universe(region, "stocks") or list(_SECTOR.keys())
    else:
        default_univ = list(_SECTOR.keys())
    universe = [t.upper() for t in custom_univ] if custom_univ else default_univ

    # ── On-demand fallback: if cache mostly empty, run pipeline now ────────
    # Region-aware: uses India/Europe/Asia/etc. priority when user is browsing
    # a non-US market — so non-US users get fast AI-portfolio results too.
    now_ts = time.time()
    cached_count = sum(
        1 for s in universe
        if (_SIGNAL_CACHE.get((s, "1d"))
            and now_ts < _SIGNAL_CACHE[(s, "1d")][1])
    )
    # Need enough candidates for the largest mode (DIVERSIFIED wants 20 LONGs)
    min_required = max(20, mode["positions"] * 3)
    if cached_count < min_required:
        univ_set = set(universe)
        region_priority = _get_regional_priority(region)
        # Build priority list: regional priority first, then rest of the universe
        priority_order = [t for t in region_priority if t in univ_set]
        priority_order += [t for t in universe if t not in priority_order]
        await _on_demand_warm_signals(
            priority_order[:30],
            max_workers=4,
            per_ticker_timeout=35,
            total_budget=120,
        )
        now_ts = time.time()

    # ── Pull from signal cache ──────────────────────────────────────────────
    candidates = []
    for sym in universe:
        sig_entry = (
            _SIGNAL_CACHE.get((sym, "1d")) or _SIGNAL_CACHE.get((sym, "1m"))
        )
        if not sig_entry or now_ts > sig_entry[1]:
            continue
        sig = sig_entry[0]
        if sig.get("direction", "NEUTRAL") != "LONG":
            continue
        conviction  = float(sig.get("conviction") or 0)
        prob        = float(sig.get("probability") or sig.get("prob_up") or 0.5)
        if prob > 1: prob /= 100.0
        if conviction < mode["min_conv"]: continue
        if prob < mode["min_prob"]:       continue
        agents  = sig.get("agents") or []
        lv = sum(1 for a in agents if (a.get("vote") or "").upper() == "LONG")
        quote = _LIVE_QUOTE_CACHE.get(sym) or {}
        candidates.append({
            "symbol":     sym,
            "sector":     _SECTOR.get(sym, "Other"),
            "conviction": round(conviction, 1),
            "prob":       round(prob * 100, 1),
            "long_votes": lv,
            "n_agents":   len(agents),
            "price":      quote.get("price"),
            "half_life":  (sig.get("holding_period") or {}).get("half_life_days"),
            "entropy":    round(float(sig.get("entropy") or 1.0), 3),
        })
    candidates.sort(key=lambda x: (-x["conviction"], -x["prob"]))

    # ── Apply sector cap + position limit ──────────────────────────────────
    sec_count: dict = {}
    selected = []
    for c in candidates:
        sec   = c["sector"]
        limit = _SECTOR_CAP.get(sec, mode["sector_max"])
        if sec_count.get(sec, 0) >= limit: continue
        sec_count[sec] = sec_count.get(sec, 0) + 1
        selected.append(c)
        if len(selected) >= mode["positions"]: break

    # ── Kelly-based sizing ──────────────────────────────────────────────────
    def _kelly(prob_pct: float, max_f: float, b: float = 1.5) -> float:
        p = prob_pct / 100.0
        f = (p * b - (1 - p)) / b
        return min(max(0.0, f * 0.5), max_f)

    deployable   = capital * mode["deploy_pct"]
    total_weight = sum(_kelly(s["prob"], mode["pct_each"]) * s["conviction"] for s in selected) or 1.0
    positions    = []
    allocated    = 0.0
    for s in selected:
        kf     = _kelly(s["prob"], mode["pct_each"])
        weight = (s["conviction"] * kf) / total_weight
        alloc  = min(deployable * weight, capital * mode["pct_each"])
        alloc  = round(alloc, 2)
        if alloc < 200: continue
        stop   = round(s["price"] * 0.985, 2) if s.get("price") else None
        target = round(s["price"] * 1.030, 2) if s.get("price") else None
        positions.append({**s, "dollar_alloc": alloc,
                          "pct_portfolio": round(alloc / capital * 100, 1),
                          "kelly_f": round(kf, 4),
                          "stop": stop, "target": target})
        allocated += alloc

    # Scale down if over budget
    if allocated > deployable and allocated > 0:
        scale = deployable / allocated
        for p in positions:
            p["dollar_alloc"] = round(p["dollar_alloc"] * scale, 2)
            p["pct_portfolio"] = round(p["dollar_alloc"] / capital * 100, 1)
        allocated = sum(p["dollar_alloc"] for p in positions)

    # ── Expected P&L ────────────────────────────────────────────────────────
    _acc  = {"EXTREME": 0.78, "HIGH": 0.72, "ELEVATED": 0.60, "CALM": 0.54}
    _move = {"EXTREME": 2.0,  "HIGH": 1.4,  "ELEVATED": 0.7,  "CALM": 0.4}
    acc_e  = _acc[regime["label"]]
    move_e = _move[regime["label"]]
    exp_pnl = allocated * (acc_e * move_e - (1 - acc_e) * move_e * 0.6) / 100

    return {
        "regime":           regime,
        "recommended_mode": recommended,
        "chosen_mode":      chosen_mode_key,
        "mode_info":        mode,
        "opportunity_day":  opp_day,
        "modes":            MODES,
        "positions":        positions,
        "n_candidates":     len(candidates),
        "n_positions":      len(positions),
        "allocated":        round(allocated, 2),
        "cash":             round(capital - allocated, 2),
        "capital":          capital,
        "region":           region,
        "asset_type":       asset_type,
        "universe_size":    len(universe),
        "expected_pnl":     round(exp_pnl, 0),
        "exp_pnl_low":      round(exp_pnl * 0.3, 0),
        "exp_pnl_high":     round(exp_pnl * 2.5, 0),
        "cache_coverage":   len([s for s in universe
                                  if _SIGNAL_CACHE.get((s, "1d")) or _SIGNAL_CACHE.get((s, "1m"))]),
    }


@app.get("/api/v1/signal-history")
async def get_signal_history(limit: int = 50):
    """Returns the most recent portfolio scan results stored in DB."""
    try:
        return {"history": db_manager.get_signal_history(limit=limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/backtest-history")
async def get_backtest_history(ticker: str = None, limit: int = 20):
    """Returns recent backtest runs from DB."""
    try:
        return {"history": db_manager.get_backtest_history(ticker=ticker, limit=limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/warmup-registry")
async def get_warmup_registry_endpoint():
    """Returns warmup state and cache stats for all tracked tickers."""
    try:
        return {"registry": db_manager.get_warmup_registry()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/leaderboard")
async def get_leaderboard(days: int = 90, ticker: str = None):
    """Agent performance rankings — batch-evaluated, with IC/IC_IR and weight suggestions."""
    try:
        lb = AgentLeaderboard(db_manager=db_manager)
        loop = asyncio.get_running_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: lb.evaluate_performance(ticker=ticker, window_days=days)),
                timeout=60.0,
            )
        except (asyncio.TimeoutError, Exception) as _le:
            logger.warning(f"Leaderboard evaluation slow/failed ({_le}), returning demo data")
            result = lb._demo_leaderboard(days)
        return {
            "evaluation_window_days": result.evaluation_window_days,
            "total_signals_evaluated": result.total_signals_evaluated,
            "is_demo": result.total_signals_evaluated == 0,
            "weight_suggestions": result.weight_suggestions,
            "rankings": [
                {
                    "rank": s.rank,
                    "agent": s.agent_name,
                    "n_predictions": s.n_predictions,
                    "directional_accuracy_pct": s.directional_accuracy,
                    "brier_score": s.brier_score,
                    "information_ratio": s.information_ratio,
                    "avg_confidence": s.avg_confidence,
                    "ic": s.ic,
                    "ic_ir": s.ic_ir,
                    "rolling_ic": s.rolling_ic,
                }
                for s in result.scores
            ],
        }
    except Exception as e:
        logger.error(f"Leaderboard failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/leaderboard/signal-runs")
async def get_signal_runs(limit: int = 100):
    """
    Returns recent signal runs with CORRECT / WRONG / PENDING outcome badges.
    Outcome is determined by comparing predicted direction vs actual 5-day price move.
    """
    try:
        trades = db_manager.get_recent_trades(limit=limit)
        if not trades:
            return {"runs": [], "total": 0, "correct": 0, "wrong": 0, "pending": 0}

        from datetime import datetime, timedelta
        import yfinance as yf
        import pandas as pd

        settle_cutoff = datetime.utcnow() - timedelta(days=5)

        # Separate settled vs pending
        settled = [t for t in trades if _parse_ts(t["timestamp"]) < settle_cutoff]
        pending_trades = [t for t in trades if _parse_ts(t["timestamp"]) >= settle_cutoff]

        # Batch fetch prices for settled trades
        price_cache: dict = {}
        if settled:
            unique_tickers = list({t["ticker"] for t in settled})
            oldest = min(_parse_ts(t["timestamp"]) for t in settled)
            start_date = (oldest - timedelta(days=3)).strftime("%Y-%m-%d")
            try:
                if len(unique_tickers) == 1:
                    df = yf.download(unique_tickers[0], start=start_date,
                                     auto_adjust=True, progress=False)
                    if not df.empty:
                        s = df["Close"].dropna()
                        if s.index.tz is not None:
                            s.index = s.index.tz_localize(None)
                        price_cache[unique_tickers[0]] = s
                else:
                    df = yf.download(unique_tickers, start=start_date,
                                     auto_adjust=True, progress=False, group_by="ticker")
                    for tkr in unique_tickers:
                        try:
                            s = df[tkr]["Close"].dropna()
                            if s.index.tz is not None:
                                s.index = s.index.tz_localize(None)
                            price_cache[tkr] = s
                        except Exception:
                            pass
            except Exception as e:
                logger.warning(f"[SignalRuns] Batch price fetch failed: {e}")

        def _outcome(trade: dict) -> dict:
            ts = _parse_ts(trade["timestamp"])
            if ts >= settle_cutoff:
                return {**trade, "outcome": "PENDING", "actual_return_pct": None}
            series = price_cache.get(trade["ticker"])
            if series is None or series.empty:
                return {**trade, "outcome": "PENDING", "actual_return_pct": None}
            try:
                ts_pd = pd.Timestamp(ts)
                pos = int(series.index.searchsorted(ts_pd))
                if pos + 5 >= len(series):
                    return {**trade, "outcome": "PENDING", "actual_return_pct": None}
                entry = float(series.iloc[pos])
                if entry <= 0:
                    return {**trade, "outcome": "PENDING", "actual_return_pct": None}
                ret = (float(series.iloc[pos + 5]) - entry) / entry
                direction = trade.get("direction", "LONG")
                correct = (direction == "LONG" and ret > 0) or (direction == "SHORT" and ret < 0)
                return {**trade, "outcome": "CORRECT" if correct else "WRONG",
                        "actual_return_pct": round(ret * 100, 2)}
            except Exception:
                return {**trade, "outcome": "PENDING", "actual_return_pct": None}

        runs = [_outcome(t) for t in trades]
        correct = sum(1 for r in runs if r["outcome"] == "CORRECT")
        wrong = sum(1 for r in runs if r["outcome"] == "WRONG")
        pending_count = sum(1 for r in runs if r["outcome"] == "PENDING")
        settled_count = correct + wrong
        accuracy = round(correct / settled_count * 100, 1) if settled_count > 0 else None

        return {
            "runs": runs,
            "total": len(runs),
            "correct": correct,
            "wrong": wrong,
            "pending": pending_count,
            "accuracy_pct": accuracy,
        }
    except Exception as e:
        logger.error(f"signal-runs failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/leaderboard/tune-weights")
async def tune_agent_weights(body: dict = None):
    """Derives optimal agent weights from IC_IR scores and writes to agents.yaml."""
    body = body or {}
    days = int(body.get("days", 90))
    try:
        lb = AgentLeaderboard(db_manager=db_manager)
        loop = asyncio.get_running_loop()
        new_weights = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: lb.tune_weights(window_days=days)),
            timeout=120.0,
        )
        if not new_weights:
            return {"status": "skipped", "reason": "insufficient data — weights unchanged", "weights": {}}
        return {"status": "updated", "weights": new_weights}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Evaluation timed out — try a shorter window")
    except Exception as e:
        logger.error(f"tune-weights failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Phase 4: Portfolio Management Endpoints ────────────────────────────────

@app.post("/api/v1/portfolio/add")
async def add_position(body: dict):
    """
    Adds or upserts a position in the paper portfolio.
    Body: {"ticker": "AAPL", "shares": 10, "avg_price": 175.0}
    If the ticker already exists, shares are blended (dollar-weighted avg entry).
    """
    ticker = body.get("ticker", "").upper()
    shares = body.get("shares")
    avg_price = body.get("avg_price")
    current_price = body.get("current_price")

    if not ticker or not shares or not avg_price:
        raise HTTPException(status_code=400, detail="ticker, shares, and avg_price are required.")
    if shares <= 0 or avg_price <= 0:
        raise HTTPException(status_code=400, detail="shares and avg_price must be positive.")

    try:
        position = db_manager.add_position(
            ticker=ticker,
            shares=float(shares),
            avg_price=float(avg_price),
            current_price=float(current_price) if current_price else None,
        )
        return {
            "ticker": position.ticker,
            "shares": position.shares,
            "avg_price": position.avg_price,
            "last_price": position.last_price,
            "market_value": position.market_value,
        }
    except Exception as e:
        logger.error(f"add_position failed for {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/portfolio/{ticker}")
async def close_position(ticker: str, exit_price: float):
    """
    Closes (removes) a position, recording realized P&L.
    Query param: exit_price=175.50
    """
    try:
        result = db_manager.close_position(ticker.upper(), exit_price)
        if result is None:
            raise HTTPException(status_code=404, detail=f"{ticker} not found in portfolio.")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"close_position failed for {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/portfolio/performance")
async def get_portfolio_performance():
    """
    Returns full portfolio analytics: P&L, allocation, win-rate, and risk metrics
    (beta, annualized vol, concentration, avg correlation).
    """
    try:
        perf = db_manager.get_portfolio_performance()
        if not perf["positions"]:
            return {**perf, "risk": None, "warnings": ["Portfolio is empty."]}

        agent = PortfolioAgent()
        view = agent.analyze(perf["positions"])

        return {
            **perf,
            "risk": {
                "portfolio_beta": view.portfolio_beta,
                "portfolio_vol_ann_pct": view.portfolio_vol_ann_pct,
                "max_drawdown_estimate_pct": view.max_drawdown_estimate_pct,
                "herfindahl_concentration": view.herfindahl_concentration,
                "avg_pairwise_correlation": view.avg_pairwise_correlation,
                "overweight_tickers": view.overweight_tickers,
                "underweight_tickers": view.underweight_tickers,
            },
            "warnings": view.warnings,
        }
    except Exception as e:
        logger.error(f"portfolio/performance failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Phase 4: Stress Testing Endpoints ──────────────────────────────────────

@app.get("/api/v1/stress/scenarios")
async def list_stress_scenarios():
    """Lists all available crisis stress scenarios."""
    return {
        "scenarios": [
            {
                "id": sid,
                "label": s["label"],
                "spy_shock_pct": s["spy_shock_pct"],
                "duration_days": s["duration_days"],
                "description": s["description"],
            }
            for sid, s in SCENARIOS.items()
        ]
    }


@app.post("/api/v1/stress/run")
async def run_stress_test(body: dict):
    """
    Runs one or all crisis stress scenarios against the current portfolio.
    Body: {"scenario_id": "2008_gfc"} — omit scenario_id to run all.
          {"scenario_id": "custom", "custom_shock_pct": -15.0}
          {"scenario_id": "sector_collapse", "target_sector": "Financial Services"}
    """
    try:
        perf = db_manager.get_portfolio_performance()
        holdings = perf.get("positions", [])

        if not holdings:
            raise HTTPException(status_code=400, detail="Portfolio is empty. Add positions first.")

        tester = StressTester()
        scenario_id = body.get("scenario_id")
        custom_shock = body.get("custom_shock_pct")
        target_sector = body.get("target_sector")

        if scenario_id:
            result = tester.run_scenario(
                scenario_id=scenario_id,
                holdings=holdings,
                custom_shock_pct=float(custom_shock) if custom_shock else None,
                target_sector=target_sector,
            )
            return {
                "scenario_id": result.scenario_id,
                "label": result.scenario_label,
                "description": result.description,
                "spy_shock_pct": result.spy_shock_pct,
                "duration_days": result.duration_days,
                "portfolio_before": result.portfolio_value_before,
                "portfolio_after": result.portfolio_value_after,
                "portfolio_pnl": result.portfolio_pnl,
                "portfolio_pnl_pct": result.portfolio_pnl_pct,
                "warnings": result.warnings,
                "position_impacts": [
                    {
                        "ticker": p.ticker,
                        "sector": p.sector,
                        "current_value": p.current_value,
                        "shocked_value": p.shocked_value,
                        "pnl": p.pnl,
                        "pnl_pct": p.pnl_pct,
                    }
                    for p in result.position_impacts
                ],
            }
        else:
            results = tester.run_all(holdings, include_custom=float(custom_shock) if custom_shock else None)
            return {
                "scenarios_run": len(results),
                "results": [
                    {
                        "scenario_id": r.scenario_id,
                        "label": r.scenario_label,
                        "spy_shock_pct": r.spy_shock_pct,
                        "portfolio_pnl_pct": r.portfolio_pnl_pct,
                        "portfolio_pnl": r.portfolio_pnl,
                        "warnings": r.warnings,
                    }
                    for r in results
                ],
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stress test failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Phase 5: Walk-Forward Validation ────────────────────────────────────────

@app.get("/api/v1/walkforward/{ticker}")
async def run_walkforward(ticker: str, n_folds: int = 4, train_years: float = 1.0, test_months: int = 3):
    """
    Runs anchored walk-forward cross-validation on the quant signal pipeline.
    Returns per-fold metrics + aggregate stats (Sharpe, MDD, accuracy, CAGR).
    """
    try:
        validator = WalkForwardValidator(n_folds=n_folds, train_years=train_years, test_months=test_months)
        result = validator.validate(ticker.upper())
        return {
            "ticker": result.ticker,
            "n_folds": result.n_folds,
            "avg_sharpe": result.avg_sharpe,
            "avg_max_drawdown_pct": result.avg_max_drawdown_pct,
            "avg_signal_accuracy_pct": result.avg_signal_accuracy,
            "avg_cagr_pct": result.avg_cagr_pct,
            "consistency_score_pct": result.consistency_score,
            "folds": [
                {
                    "fold": f.fold,
                    "train_period": f.train_period,
                    "test_period": f.test_period,
                    "n_trades": f.n_trades,
                    "signal_accuracy_pct": f.signal_accuracy,
                    "sharpe": f.metrics.sharpe_ratio,
                    "total_return_pct": f.metrics.total_return_pct,
                    "max_drawdown_pct": f.metrics.max_drawdown_pct,
                    "cagr_pct": f.metrics.cagr_pct,
                    "alpha_vs_spy": f.metrics.alpha_vs_spy,
                }
                for f in result.folds
            ],
        }
    except Exception as e:
        logger.error(f"Walk-forward failed for {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Phase 5: Calibration Diagnostics + Auto-Tune ────────────────────────────

@app.get("/api/v1/calibration")
async def get_calibration(days: int = 90, ticker: str = None):
    """
    Returns calibration diagnostics from the leaderboard (Brier score, IC, IC_IR)
    and a reliability diagram for each agent.
    """
    try:
        lb = AgentLeaderboard(db_manager=db_manager)
        result = lb.evaluate_performance(ticker=ticker, window_days=days)
        return {
            "evaluation_window_days": result.evaluation_window_days,
            "total_signals": result.total_signals_evaluated,
            "is_demo": result.total_signals_evaluated == 0,
            "weight_suggestions": result.weight_suggestions,
            "agents": [
                {
                    "agent": s.agent_name,
                    "rank": s.rank,
                    "n_predictions": s.n_predictions,
                    "directional_accuracy_pct": s.directional_accuracy,
                    "brier_score": s.brier_score,
                    "ic": s.ic,
                    "ic_ir": s.ic_ir,
                    "rolling_ic": s.rolling_ic,
                    "avg_confidence": s.avg_confidence,
                    "information_ratio": s.information_ratio,
                }
                for s in result.scores
            ],
        }
    except Exception as e:
        logger.error(f"Calibration failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/calibration/tune")
async def tune_agent_weights(days: int = 90):
    """
    Derives new agent weights from IC_IR rankings and writes them to
    config/agents.yaml. Returns the new weight mapping.
    The server must be restarted (or AgentRegistry reloaded) for changes to take effect.
    """
    try:
        lb = AgentLeaderboard(db_manager=db_manager)
        new_weights = lb.tune_weights(window_days=days)
        if not new_weights:
            raise HTTPException(
                status_code=422,
                detail="Insufficient signal history to derive weights. Run more signals first."
            )
        return {
            "status": "weights_updated",
            "config_path": "config/agents.yaml",
            "new_weights": new_weights,
            "note": "Restart the server or reload AgentRegistry for changes to take effect.",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Weight tuning failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Phase 5: Factor Exposure ─────────────────────────────────────────────────

@app.get("/api/v1/portfolio/factors")
async def get_factor_exposure():
    """
    Computes style-factor exposure for the current portfolio:
    Value, Momentum, Quality, Size, Low-Volatility.
    Returns portfolio-level scores (±1) and per-position breakdown.
    """
    try:
        perf = db_manager.get_portfolio_performance()
        holdings = perf.get("positions", [])
        if not holdings:
            raise HTTPException(status_code=400, detail="Portfolio is empty.")

        analyzer = FactorExposureAnalyzer()
        exposure = analyzer.analyze(holdings)
        return {
            "portfolio_factors": {
                "value": exposure.value,
                "momentum": exposure.momentum,
                "quality": exposure.quality,
                "size": exposure.size,
                "low_vol": exposure.low_vol,
            },
            "warnings": exposure.warnings,
            "positions": [
                {
                    "ticker": p.ticker,
                    "weight": p.weight,
                    "value_score": p.value_score,
                    "momentum_score": p.momentum_score,
                    "quality_score": p.quality_score,
                    "size_score": p.size_score,
                    "low_vol_score": p.low_vol_score,
                }
                for p in exposure.positions
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Factor exposure failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Phase 5: Monthly Returns Heatmap ────────────────────────────────────────

@app.get("/api/v1/backtest/{ticker}/heatmap")
async def get_monthly_heatmap(ticker: str, period: str = "3y"):
    """
    Returns monthly returns matrix for the backtest equity curve.
    Used by the dashboard to render a performance heatmap.
    Format: {"YYYY-MM": return_pct, ...}
    """
    try:
        engine = BacktestEngine()
        result = engine.run(ticker.upper(), period=period)
        return {
            "ticker": ticker.upper(),
            "period": period,
            "monthly_returns": result.metrics.monthly_returns,
            "var_95_pct": result.metrics.var_95_pct,
            "cvar_95_pct": result.metrics.cvar_95_pct,
            "omega_ratio": result.metrics.omega_ratio,
        }
    except Exception as e:
        logger.error(f"Monthly heatmap failed for {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Phase 5: Paper Trading ───────────────────────────────────────────────────

@app.post("/api/v1/paper/signal/{ticker}")
async def paper_trade_signal(ticker: str):
    """
    Runs the full agent pipeline for the ticker and immediately records
    a paper trade if direction is LONG or SHORT.
    Returns the paper trade ID and signal summary.
    """
    try:
        md = _get_market_data(ticker.upper())
        initial_state = {"ticker": ticker.upper(), "market_data": md, "registry": registry}
        result_state = graph.invoke(initial_state)
        final_info = result_state["final_signal"]
        packet = final_info["packet"]

        trade_id = paper_trader.record_signal(
            ticker.upper(), packet, probability_up=final_info["probability_up"]
        )

        return {
            "paper_trade_id": trade_id,
            "ticker": ticker.upper(),
            "direction": str(packet.direction),
            "conviction_pct": packet.conviction_pct,
            "probability_up": final_info["probability_up"],
            "recorded": trade_id > 0,
            "note": "Trade will be resolved after 5 trading days.",
        }
    except Exception as e:
        logger.error(f"Paper signal failed for {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/paper/resolve")
async def resolve_paper_trades():
    """
    Resolves all open paper trades whose hold period has elapsed.
    Fetches current prices and computes realized P&L.
    """
    try:
        resolved = paper_trader.resolve_open_trades()
        return {
            "resolved_count": len(resolved),
            "trades": resolved,
        }
    except Exception as e:
        logger.error(f"Paper trade resolution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/paper/summary")
async def get_paper_summary():
    """
    Returns the full paper trading performance summary:
    virtual capital, P&L, accuracy, Sharpe, Omega, trade journal, agent breakdown.
    """
    try:
        return paper_trader.summarize()
    except Exception as e:
        logger.error(f"Paper summary failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/paper/open")
async def get_open_paper_trades():
    """Lists all currently open (unresolved) paper positions."""
    try:
        return {"open_trades": paper_trader.get_open_trades()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/paper/report")
async def get_paper_daily_report():
    """Returns a plain-text daily summary of paper trading performance."""
    try:
        return {"report": paper_trader.get_daily_report()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Phase 6: TDA ────────────────────────────────────────────────────────────

@app.get("/api/v1/quant/tda/{ticker}")
async def get_tda_analysis(ticker: str, period: str = "1y"):
    """
    Applies Topological Data Analysis (persistent homology) to price data.
    Returns regime label, H0/H1 persistence metrics, and a [-1,+1] signal:
      +1 = trending (low topological complexity)
      -1 = mean-reverting (high cyclic structure in price attractor)
    """
    try:
        import yfinance as yf
        df = yf.download(ticker.upper(), period=period, interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        prices = df["Close"].dropna()
        if len(prices) < 60:
            raise HTTPException(status_code=400, detail="Need 60+ trading days of data.")

        engine = TDASignalEngine()
        result = engine.analyze(prices)

        return {
            "ticker": ticker.upper(),
            "regime_label": result.regime_label,
            "tda_signal": result.tda_signal,
            "h0_max_persistence": result.h0_max_persistence,
            "h1_max_persistence": result.h1_max_persistence,
            "h0_persistence_entropy": result.h0_persistence_entropy,
            "h1_persistence_entropy": result.h1_persistence_entropy,
            "betti_0": result.betti_0,
            "betti_1": result.betti_1,
            "total_h1_persistence": result.total_h1_persistence,
            "interpretation": {
                "TRENDING":   "Low topological complexity — momentum strategies favored.",
                "CYCLIC":     "High loop structure — mean-reversion strategies favored.",
                "FRAGMENTED": "High entropy, no dominant cycles — regime change likely.",
                "NEUTRAL":    "Mixed topology — no strong directional bias.",
            }.get(result.regime_label, ""),
            "warnings": result.warnings,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TDA failed for {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Phase 6: Hawkes Process ──────────────────────────────────────────────────

@app.get("/api/v1/quant/hawkes/{ticker}")
async def get_hawkes_analysis(ticker: str, period: str = "1y", threshold_sigma: float = 1.5):
    """
    Fits an exponential Hawkes self-exciting point process to return spikes.
    Detects trade clustering, institutional cascade risk, and dark pool footprints.

    Key metric: branching ratio α/β
      < 0.5  → Poisson-like, independent events (normal)
      0.5-0.8 → Moderate clustering (some institutional flow)
      0.8-0.95 → High clustering (probable dark pool / algo cascade)
      ≥ 0.95 → Near-explosive (systemic cascade risk)
    """
    try:
        md = MarketData(ticker.upper())
        returns = md.get_returns("1y")

        if returns is None or len(returns) < 30:
            raise HTTPException(status_code=400, detail="Need 30+ trading days of returns.")

        hp = HawkesProcess(threshold_sigma=threshold_sigma)
        result = hp.fit(returns)

        return {
            "ticker": ticker.upper(),
            "mu": result.mu,
            "alpha": result.alpha,
            "beta": result.beta,
            "branching_ratio": result.branching_ratio,
            "n_events": result.n_events,
            "event_rate_per_day": result.event_rate,
            "cascade_risk": result.cascade_risk,
            "clustering_score": result.clustering_score,
            "dark_pool_signal": result.dark_pool_signal,
            "converged": result.converged,
            "log_likelihood": result.log_likelihood,
            "interpretation": {
                "LOW":      "Event arrivals are near-Poisson. No significant clustering.",
                "MODERATE": "Moderate self-excitation. Some algorithmic or institutional flow.",
                "HIGH":     "Strong trade clustering. Possible dark pool or HFT cascade.",
                "CRITICAL": "Near-explosive branching. Systemic cascade risk — extreme caution.",
            }.get(result.cascade_risk, ""),
            "warnings": result.warnings,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hawkes failed for {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Phase 6: Quasi-Monte Carlo ───────────────────────────────────────────────

@app.get("/api/v1/quant/qmc/{ticker}")
async def get_qmc_simulation(
    ticker: str,
    days: int = 5,
    n_paths: int = 4096,
    compare: bool = False,
):
    """
    Runs Quasi-Monte Carlo (Sobol sequence) GBM price simulation.
    Provides tighter confidence intervals than standard MC with the same path count.

    Set compare=true to also run standard MC and report the CI width improvement.
    """
    try:
        md = MarketData(ticker.upper())
        price = md.get_current_price()
        returns = md.get_returns("1y")

        if price is None or price <= 0:
            raise HTTPException(status_code=400, detail="Could not fetch current price.")

        drift = float(returns.mean()) if returns is not None and len(returns) > 0 else 0.0005
        vol = float(returns.std()) if returns is not None and len(returns) > 0 else 0.018

        engine = QuasiMonteCarloEngine(current_price=price)

        if compare:
            comp = engine.compare_with_mc(days=days, drift_daily=drift, vol_daily=vol, n_paths=n_paths)
            result = comp["qmc"]
            return {
                "ticker": ticker.upper(),
                "method": result.method,
                "current_price": price,
                "simulation_days": days,
                "n_paths": n_paths,
                "expected_return_pct": result.expected_return_pct,
                "prob_above_current_pct": result.prob_above_current,
                "ci_68": {"low": result.ci_68_low, "high": result.ci_68_high},
                "ci_95": {"low": result.ci_95_low, "high": result.ci_95_high},
                "ci_99": {"low": result.ci_99_low, "high": result.ci_99_high},
                "comparison": {
                    "mc_ci_95_width": comp["mc_ci_width"],
                    "qmc_ci_95_width": comp["qmc_ci_width"],
                    "efficiency_gain": comp["efficiency_gain"],
                    "qmc_tighter": comp["qmc_tighter"],
                },
                "warnings": result.warnings,
            }
        else:
            result = engine.simulate_gbm(days=days, drift_daily=drift, vol_daily=vol, n_paths=n_paths)
            return {
                "ticker": ticker.upper(),
                "method": result.method,
                "current_price": price,
                "simulation_days": days,
                "n_paths": n_paths,
                "expected_return_pct": result.expected_return_pct,
                "prob_above_current_pct": result.prob_above_current,
                "ci_68": {"low": result.ci_68_low, "high": result.ci_68_high},
                "ci_95": {"low": result.ci_95_low, "high": result.ci_95_high},
                "ci_99": {"low": result.ci_99_low, "high": result.ci_99_high},
                "effective_sample_size": result.effective_sample_size,
                "warnings": result.warnings,
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"QMC failed for {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Phase 6: Hardened EVT ────────────────────────────────────────────────────

@app.get("/api/v1/quant/evt/{ticker}")
async def get_evt_analysis(ticker: str, period: str = "2y"):
    """
    Runs hardened Extreme Value Theory analysis:
      - GPD/POT with automatic threshold selection
      - GEV block maxima (cross-check)
      - Tail dependence coefficient with SPY
    Returns dual VaR/CVaR estimates at 95% and 99% confidence.
    """
    try:
        import yfinance as yf

        df = yf.download(ticker.upper(), period=period, interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        returns = df["Close"].pct_change().dropna()

        spy_df = yf.download("SPY", period=period, interval="1d",
                              auto_adjust=True, progress=False)
        if isinstance(spy_df.columns, pd.MultiIndex):
            spy_df.columns = spy_df.columns.get_level_values(0)
        spy_returns = spy_df["Close"].pct_change().dropna()

        if len(returns) < 100:
            raise HTTPException(status_code=400, detail="Need 100+ trading days for EVT.")

        evt = EVTModel(returns, benchmark_returns=spy_returns)
        result = evt.fit_and_calculate()

        return {
            "ticker": ticker.upper(),
            "period": period,
            "gpd": {
                "var_95_pct": round(result.var_95 * 100, 3),
                "var_99_pct": round(result.var_99 * 100, 3),
                "cvar_95_pct": round(result.cvar_95 * 100, 3),
                "cvar_99_pct": round(result.cvar_99 * 100, 3),
                "shape_xi": result.tail_index,
                "threshold": result.threshold,
                "n_tail_events": result.n_tail_events,
                "converged": result.gpd_converged,
            },
            "gev": {
                "var_95_pct": round(result.gev_var_95 * 100, 3),
                "var_99_pct": round(result.gev_var_99 * 100, 3),
                "converged": result.gev_converged,
            },
            "tail_dependence_with_spy": result.tail_dependence,
            "fat_tails": result.tail_index > 0.3,
            "warnings": result.warnings,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"EVT analysis failed for {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Phase 7: Advanced Quant Models ─────────────────────────────────────────

import math as _math

def _nan_safe(obj):
    """Recursively replace NaN/Inf with None so FastAPI can JSON-serialize."""
    if isinstance(obj, float):
        return None if (_math.isnan(obj) or _math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _nan_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_safe(v) for v in obj]
    return obj


@app.get("/api/v1/quant/heston/{ticker}")
async def get_heston(ticker: str):
    """Heston SV model: calibrate to options, return vol surface + fair-value signal."""
    from quant_engine.heston import analyze_heston
    import asyncio
    md = _get_market_data(ticker.upper())
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, analyze_heston, ticker.upper(), md)
    return _nan_safe({"ticker": ticker.upper(), **result.__dict__, "params": result.params.__dict__})

@app.get("/api/v1/quant/sabr/{ticker}")
async def get_sabr(ticker: str):
    """SABR model: vol smile fitting and skew analysis."""
    from quant_engine.sabr import analyze_sabr
    import asyncio
    md = _get_market_data(ticker.upper())
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, analyze_sabr, ticker.upper(), md)
    return _nan_safe({"ticker": ticker.upper(), **result.__dict__, "params": result.params.__dict__})

@app.get("/api/v1/quant/rough-vol/{ticker}")
async def get_rough_vol(ticker: str):
    """Rough Bergomi / rBergomi model with Hurst exponent estimation."""
    from quant_engine.rough_vol import analyze_rough_vol
    import asyncio
    md = _get_market_data(ticker.upper())
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, analyze_rough_vol, ticker.upper(), md)
    return _nan_safe({"ticker": ticker.upper(), **result.__dict__})

@app.post("/api/v1/quant/copula")
async def get_copula(body: dict):
    """Copula-based tail dependence between two assets."""
    import yfinance as yf
    import asyncio
    from quant_engine.copula import analyze_copula
    tickers_list = body.get("tickers", [])
    ticker_a = (tickers_list[0] if len(tickers_list) > 0 else body.get("ticker_a") or "SPY").upper()
    ticker_b = (tickers_list[1] if len(tickers_list) > 1 else body.get("ticker_b") or "QQQ").upper()
    period   = body.get("period", "1y")
    md_a = _get_market_data(ticker_a)
    md_b = _get_market_data(ticker_b)

    def _run():
        ohlcv_a = md_a.get_ohlcv(period)
        ohlcv_b = md_b.get_ohlcv(period)
        a = ohlcv_a["Close"].pct_change().dropna().values
        b = ohlcv_b["Close"].pct_change().dropna().values
        n = min(len(a), len(b))
        return analyze_copula(a[-n:], b[-n:])

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _run)
        return _nan_safe({"ticker_a": ticker_a, "ticker_b": ticker_b, **result.__dict__})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/quant/multifractal/{ticker}")
async def get_multifractal(ticker: str):
    """Markov-Switching Multifractal (MSM) vol model and DFA Hurst exponent."""
    from quant_engine.multifractal import analyze_multifractal
    import asyncio
    md = _get_market_data(ticker.upper())
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, analyze_multifractal, ticker.upper(), md)
    return _nan_safe({"ticker": ticker.upper(), **result.__dict__})

@app.get("/api/v1/quant/granger/{ticker}")
async def get_granger(ticker: str):
    """Granger causality: which macro/sector series drive this ticker?"""
    from quant_engine.granger import analyze_granger
    import asyncio
    md = _get_market_data(ticker.upper())
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, analyze_granger, ticker.upper(), md)
    tests_serial = [t.__dict__ for t in result.tests]
    return _nan_safe({"ticker": ticker.upper(),
            **{k: v for k, v in result.__dict__.items() if k != "tests"},
            "tests": tests_serial})

@app.get("/api/v1/quant/causal/{ticker}")
async def get_causal(ticker: str):
    """Do-calculus / structural causal model: causal factor analysis."""
    from quant_engine.causal_engine import analyze_causal
    import asyncio
    md = _get_market_data(ticker.upper())
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, analyze_causal, ticker.upper(), md)
    effects_serial = [e.__dict__ for e in result.causal_effects]
    return _nan_safe({"ticker": ticker.upper(),
            **{k: v for k, v in result.__dict__.items() if k != "causal_effects"},
            "causal_effects": effects_serial})

@app.get("/api/v1/quant/lob/{ticker}")
async def get_lob(ticker: str):
    """LOB microstructure: Kyle lambda, Amihud, PIN, VPIN, Roll spread."""
    from quant_engine.lob import analyze_lob
    import asyncio
    md = _get_market_data(ticker.upper())
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, analyze_lob, ticker.upper(), md)
    return _nan_safe({"ticker": ticker.upper(), **result.__dict__})

@app.get("/api/v1/quant/quantum/{ticker}")
async def get_quantum_finance(ticker: str):
    """Quantum finance: QAE option pricing, QAOA portfolio, quantum annealing selection."""
    from quant_engine.quantum_finance import analyze_quantum_finance
    import asyncio
    md = _get_market_data(ticker.upper())
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, analyze_quantum_finance, ticker.upper(), md)
    return _nan_safe({"ticker": ticker.upper(), **result.__dict__})


# ─── Quant Lab: Monte Carlo, GARCH, HMM, Explain ─────────────────────────────

@app.post("/api/v1/quant/monte-carlo")
async def quant_monte_carlo(body: dict):
    """Monte Carlo price simulation with GARCH vol integration. Returns path percentile bands."""
    import yfinance as _yf2
    import numpy as _np2
    ticker = (body.get("ticker") or "AAPL").upper()
    n_paths = min(int(body.get("paths") or 1000), 5000)
    days    = min(int(body.get("days")  or 30),   60)

    def _run():
        from quant_engine.garch import GARCHModel
        hist = _yf2.Ticker(ticker).history(period="1y")["Close"].dropna()
        if len(hist) < 60:
            raise ValueError("Not enough price history")
        rets = hist.pct_change().dropna()
        drift = float(rets.mean())
        current_price = float(hist.iloc[-1])

        # Fit GARCH for dynamic vol forecasts
        garch_vols = []
        try:
            gr = GARCHModel(rets).fit_and_forecast(horizon=days)
            raw = gr.forecast_daily or []
            garch_vols = [(v / 100.0 if v > 1.0 else v) for v in raw[:days]]
        except Exception:
            pass
        if len(garch_vols) < days:
            hist_vol = float(rets.std())
            garch_vols += [hist_vol] * (days - len(garch_vols))

        rng = _np2.random.default_rng(42)
        paths = _np2.zeros((days + 1, n_paths))
        paths[0] = current_price
        for t in range(1, days + 1):
            vol_t = garch_vols[t - 1]
            drift_adj = drift - 0.5 * vol_t ** 2
            paths[t] = paths[t - 1] * _np2.exp(drift_adj + vol_t * rng.standard_normal(n_paths))

        pcts = [5, 10, 25, 50, 75, 90, 95]
        path_bands = []
        for t in range(days + 1):
            row = {"day": t}
            for p in pcts:
                row[f"p{p}"] = round(float(_np2.percentile(paths[t], p)), 4)
            path_bands.append(row)

        final = paths[-1]
        prob_up = float((_np2.sum(final > current_price) / n_paths) * 100)
        return {
            "ticker": ticker,
            "current_price": round(current_price, 4),
            "simulation_days": days,
            "num_paths": n_paths,
            "mean_price":     round(float(_np2.mean(final)), 4),
            "median_price":   round(float(_np2.median(final)), 4),
            "prob_above_current": round(prob_up, 2),
            "expected_return_pct": round((float(_np2.mean(final)) / current_price - 1) * 100, 3),
            "p10_price": round(float(_np2.percentile(final, 10)), 4),
            "p90_price": round(float(_np2.percentile(final, 90)), 4),
            "path_bands": path_bands,
            "garch_vols_used": [round(v * 100, 3) for v in garch_vols[:10]],
            "notes": [
                f"Simulated {n_paths:,} paths over {days} trading days",
                f"P(price > ${current_price:.2f}) = {prob_up:.1f}%",
                f"GARCH day-1 vol: {garch_vols[0]*100:.2f}% (daily)",
            ],
        }

    loop = asyncio.get_running_loop()
    try:
        return _nan_safe(await asyncio.wait_for(loop.run_in_executor(None, _run), timeout=30.0))
    except Exception as _e:
        raise HTTPException(status_code=500, detail=str(_e))


@app.post("/api/v1/quant/garch")
async def quant_garch_standalone(body: dict):
    """Standalone GARCH vol model with multi-step forecast series for charting."""
    import yfinance as _yf2
    ticker = (body.get("ticker") or "AAPL").upper()
    horizon = min(int(body.get("horizon") or 30), 60)

    def _run():
        from quant_engine.garch import GARCHModel
        hist = _yf2.Ticker(ticker).history(period="2y")["Close"].dropna()
        if len(hist) < 100:
            raise ValueError("Not enough price history")
        rets = hist.pct_change().dropna()
        gr = GARCHModel(rets).fit_and_forecast(horizon=horizon)

        vols = gr.forecast_daily or []
        forecast_series = [
            {"day": i + 1, "vol_pct": round(v, 4)}
            for i, v in enumerate(vols[:horizon])
        ]
        regime_colors = {"LOW": "#22c55e", "NORMAL": "#60a5fa", "HIGH": "#f59e0b", "EXTREME": "#ef4444"}
        return {
            "ticker": ticker,
            "vol_1day":    round(gr.vol_1day, 4),
            "vol_5day":    round(gr.vol_5day, 4),
            "vol_10day":   round(gr.vol_10day, 4),
            "current_vol": round(gr.current_vol, 4),
            "vol_percentile": round(gr.vol_percentile, 2),
            "vol_regime":  gr.vol_regime,
            "regime_color": regime_colors.get(gr.vol_regime, "#60a5fa"),
            "omega":       round(gr.omega, 8),
            "alpha":       round(gr.alpha, 6),
            "beta":        round(gr.beta, 6),
            "persistence": round(gr.persistence, 6),
            "long_run_vol": round(gr.long_run_vol, 4),
            "converged":   gr.converged,
            "forecast_series": forecast_series,
            "signal": f"{gr.vol_regime}_VOL",
            "notes": [
                f"GARCH persistence α+β = {gr.persistence:.4f} ({'explosive' if gr.persistence >= 1 else 'mean-reverting'})",
                f"Long-run vol: {gr.long_run_vol:.2f}% (annualized)",
                f"Current vol at {gr.vol_percentile:.0f}th percentile of 2-year history",
            ],
        }

    loop = asyncio.get_running_loop()
    try:
        return _nan_safe(await asyncio.wait_for(loop.run_in_executor(None, _run), timeout=30.0))
    except Exception as _e:
        raise HTTPException(status_code=500, detail=str(_e))


@app.post("/api/v1/quant/hmm")
async def quant_hmm_standalone(body: dict):
    """HMM regime detection with state sequence and transition matrix."""
    import yfinance as _yf2
    import numpy as _np2
    ticker = (body.get("ticker") or "AAPL").upper()

    def _run():
        from quant_engine.hmm import RegimeDetector
        hist = _yf2.Ticker(ticker).history(period="2y")["Close"].dropna()
        if len(hist) < 100:
            raise ValueError("Not enough price history")
        rets = hist.pct_change().dropna()
        rd   = RegimeDetector(n_states=3)
        res  = rd.fit_predict(rets)

        # Build 60-day state sequence for sparkline
        state_seq = []
        try:
            import numpy as np_inner
            from hmmlearn.hmm import GaussianHMM
            X = rets.values.reshape(-1, 1)
            m = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
            m.fit(X)
            states = m.predict(X)
            means = [float(m.means_[i, 0]) for i in range(3)]
            order = sorted(range(3), key=lambda i: means[i])
            label_map = {order[0]: "Bear", order[1]: "Neutral", order[2]: "Bull"}
            tail = states[-60:]
            date_idx = hist.index[-60:] if len(hist) >= 60 else hist.index
            for i, s in enumerate(tail):
                state_seq.append({
                    "t": i,
                    "regime": label_map.get(s, "Neutral"),
                    "state": int(s),
                })
        except Exception:
            pass

        probs = res.probabilities or {}
        return {
            "ticker": ticker,
            "current_regime": res.current_regime,
            "probabilities":  {k: round(v, 4) for k, v in probs.items()},
            "transition_risk": {k: round(v, 4) for k, v in (res.transition_risk or {}).items()},
            "state_means":  [round(float(v), 6) for v in (res.state_means or [])],
            "state_vols":   [round(float(v), 6) for v in (res.state_vols or [])],
            "regime_duration_days": res.regime_duration_days,
            "state_sequence": state_seq,
            "signal": res.current_regime.upper().replace(" ", "_"),
            "notes": [
                f"Current regime: {res.current_regime} ({res.regime_duration_days} days)",
                f"3-state Gaussian HMM, Viterbi decoding",
            ],
        }

    loop = asyncio.get_running_loop()
    try:
        return _nan_safe(await asyncio.wait_for(loop.run_in_executor(None, _run), timeout=30.0))
    except Exception as _e:
        raise HTTPException(status_code=500, detail=str(_e))


@app.post("/api/v1/quant/explain")
async def quant_explain(body: dict):
    """Gemini-powered plain-English explanation of any quant model result."""
    import os as _os2
    model_id  = (body.get("model_id") or "unknown")
    ticker    = (body.get("ticker") or "").upper()
    ticker_b  = (body.get("ticker_b") or "").upper()
    data      = body.get("data") or {}
    import json as _json2
    data_str  = _json2.dumps(data, indent=2)[:3000]

    MODEL_NAMES = {
        "monte_carlo": "Monte Carlo Price Simulation",
        "garch": "GARCH Volatility Model",
        "hmm": "Hidden Markov Model Regime Detection",
        "heston": "Heston Stochastic Volatility Model",
        "sabr": "SABR Volatility Smile Model",
        "rough_vol": "Rough Volatility (rBergomi / Fractional Brownian Motion)",
        "multifractal": "Multifractal Volatility (MSM / MF-DFA)",
        "granger": "Granger Causality Analysis",
        "causal": "Do-Calculus Causal SCM",
        "lob": "Limit Order Book Microstructure",
        "copula": "Copula Tail Dependence",
        "quantum": "Quantum Finance (QAE + QAOA)",
    }
    name = MODEL_NAMES.get(model_id, model_id.replace("_", " ").title())
    subject = f"{ticker} vs {ticker_b}" if ticker_b else ticker

    system = (
        f"You are AlphaAgent Quant AI — an expert quantitative analyst. "
        f"A user just ran the **{name}** model on **{subject}**. "
        f"Explain the results in plain English for a sophisticated investor who is NOT a quant. "
        f"Rules:\n"
        f"- Use plain language, no heavy math notation\n"
        f"- Lead with the most actionable insight (1 sentence)\n"
        f"- Then explain what the numbers mean in market terms\n"
        f"- End with 1-2 practical implications for trading\n"
        f"- Max 200 words. Use **bold** for key terms.\n"
    )
    msgs = [{"role": "user", "content": f"Explain this {name} result for {subject}:\n\n{data_str}"}]

    gemini_key = _os2.environ.get("GEMINI_API_KEY", "")
    if gemini_key:
        try:
            explanation = await _call_gemini(msgs, system=system, max_tokens=400)
            return {"explanation": explanation, "model": "gemini"}
        except Exception as _ge:
            logger.debug(f"quant/explain Gemini error: {_ge}")

    api_key = _os2.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        try:
            import anthropic as _an2
            c = _an2.Anthropic(api_key=api_key)
            m = c.messages.create(model="claude-haiku-4-5-20251001", max_tokens=400,
                                  system=system, messages=msgs)
            return {"explanation": m.content[0].text if m.content else "", "model": "claude"}
        except Exception:
            pass

    raise HTTPException(status_code=503, detail="Set GEMINI_API_KEY or ANTHROPIC_API_KEY to enable AI explanations")


# ─── Markov Regime Chain ─────────────────────────────────────────────────────

@app.get("/api/v1/quant/markov-regime/{ticker}")
async def quant_markov_regime(
    ticker: str,
    window: int = 20,
    bull_thresh: float = 0.05,
    bear_thresh: float = 0.05,
):
    """
    Observable Markov chain regime analysis:
    rolling-return labeling (±thresh over window days) →
    3×3 ML transition matrix → Chapman-Kolmogorov n-step forecast →
    stationary distribution → signed signal (bull_prob − bear_prob) →
    walk-forward backtest (Sharpe + max drawdown).
    """
    from quant_engine.markov_regime import MarkovRegimeDetector
    ticker = ticker.upper()
    try:
        import yfinance as yf
        df = yf.download(ticker, period="2y", interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 80:
            raise ValueError(f"Insufficient price data for {ticker}")

        prices = df["Close"].dropna()
        detector = MarkovRegimeDetector(
            window=int(window),
            bull_thresh=float(bull_thresh),
            bear_thresh=-float(bear_thresh),
        )
        result = await asyncio.get_running_loop().run_in_executor(
            None, lambda: detector.analyze(prices)
        )

        return {
            "ticker":               ticker,
            "current_regime":       result.current_regime,
            "regime_duration_days": result.regime_duration_days,
            "current_probs":        result.current_probs,
            "forecast_steps":       result.forecast_steps,
            "transition_matrix":    result.transition_matrix,
            "state_labels":         result.state_labels,
            "stationary":           result.stationary,
            "signal":               result.signal,
            "conviction":           result.conviction,
            "signed_score":         result.signed_score,
            "regime_history":       result.regime_history,
            "backtest": {
                "sharpe":            result.backtest_sharpe,
                "max_dd_pct":        result.backtest_max_dd,
                "total_return_pct":  result.backtest_total_return,
                "win_rate_pct":      result.backtest_win_rate,
                "n_trades":          result.backtest_n_trades,
                "equity":            result.backtest_equity,
            },
            "notes": result.notes,
        }
    except Exception as e:
        logger.error(f"markov-regime failed for {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Phase 6: RL Portfolio Rebalancer ────────────────────────────────────────

@app.post("/api/v1/portfolio/rl-optimize")
async def rl_optimize_portfolio(body: dict):
    """
    Uses Reinforcement Learning (PPO) or minimum-variance fallback to suggest
    optimal portfolio weight rebalancing.

    Body:
      {
        "current_weights": {"AAPL": 0.4, "MSFT": 0.3, "NVDA": 0.3},
        "vol_regime": "NORMAL",     # optional
        "period": "6mo"             # historical returns window
      }

    Returns target weights, estimated Sharpe, turnover %, and method used.
    """
    try:
        current_weights: dict = body.get("current_weights", {})
        if not current_weights:
            raise HTTPException(status_code=400, detail="current_weights is required.")

        tickers = list(current_weights.keys())
        vol_regime = body.get("vol_regime", "NORMAL")
        period = body.get("period", "6mo")

        import yfinance as yf
        df = yf.download(tickers, period=period, interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            close = df["Close"]
        else:
            close = df[["Close"]] if "Close" in df.columns else df

        returns_df = close.pct_change().dropna()
        if len(returns_df) < 10:
            raise HTTPException(status_code=400, detail="Insufficient historical data for optimization.")

        rebalancer = RLRebalancer(tickers=tickers)
        result = rebalancer.optimize(
            current_weights=current_weights,
            returns_df=returns_df,
            vol_regime=vol_regime,
        )

        return {
            "tickers": tickers,
            "current_weights": current_weights,
            "recommended_weights": result.ticker_weights,
            "method": result.method,
            "expected_sharpe": result.expected_sharpe,
            "turnover_pct": result.turnover_pct,
            "rebalance_needed": result.rebalance_needed,
            "reasoning": result.reasoning,
            "warnings": result.warnings,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RL optimize failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── WebSocket Streaming ─────────────────────────────────────────────────────

@app.websocket("/ws/v1/stream/{ticker}")
async def websocket_endpoint(websocket: WebSocket, ticker: str):
    """
    Streams the agentic reasoning process step-by-step.
    """
    await websocket.accept()
    logger.info(f"WebSocket client connected for ticker: {ticker}")
    
    try:
        await websocket.send_json({"type": "log", "agent": "SYSTEM", "message": f"Fetching market data for {ticker}..."})

        active_agents = registry.get_active_agents()
        for name in active_agents:
            await websocket.send_json({"type": "log", "agent": name, "message": f"Queued: {name}"})

        await websocket.send_json({"type": "log", "agent": "SYSTEM", "message": "Running all agents in parallel..."})

        initial_state = {"ticker": ticker, "market_data": MarketData(ticker), "registry": registry}
        result_state = graph.invoke(initial_state)
        final_info = result_state["final_signal"]
        packet = final_info["packet"]

        # Emit per-agent verdicts from the completed results
        for res in packet.agent_results:
            await websocket.send_json({
                "type": "log",
                "agent": res.agent_name,
                "message": f"Verdict: {res.vote} ({round(res.probability_up * 100, 1)}%)",
            })

        await websocket.send_json({"type": "log", "agent": "SYSTEM", "message": "Fusing results..."})

        serialisable = {
            "ticker": packet.ticker,
            "direction": str(packet.direction),
            "conviction_pct": packet.conviction_pct,
            "probability_up": final_info["probability_up"],
            "multiplier": final_info["multiplier"],
            "entropy": final_info["entropy"],
            "warnings": packet.warnings,
            "override_active": packet.override_active,
            "override_reason": packet.override_reason,
        }
        await websocket.send_json({"type": "result", "data": serialisable})
        await websocket.send_json({"type": "log", "agent": "SYSTEM", "message": "Signal generation complete."})
        
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        await websocket.send_json({"status": "error", "message": str(e)})
    finally:
        await websocket.close()


# ─── Execution ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Use port 8088 to avoid conflicts
    uvicorn.run(app, host="0.0.0.0", port=8088)
