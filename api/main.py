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
import logging
from pathlib import Path
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from data.market import MarketData
from agents.registry import AgentRegistry
from orchestrator.graph import build_alpha_graph
from agents.state import SignalPacket
from api.monitoring import Monitor
from database.manager import DatabaseManager
from database.models import Trade, Portfolio, AgentLog
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

# ─── REST Endpoints ──────────────────────────────────────────────────────────

_FRONTEND = Path(__file__).parent.parent / "frontend"

@app.get("/")
async def root():
    return FileResponse(str(_FRONTEND / "index.html"))

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


def _fetch_quote_fast(sym: str) -> dict | None:
    """Fetch live quote for a single symbol using yfinance fast_info. Returns None on failure."""
    import yfinance as yf
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
        return {"status": "ok", "message": "Settings updated and saved."}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/v1/settings/reset")
async def reset_settings():
    """Reload settings from disk (discards any unsaved in-memory changes)."""
    settings.reload()
    return {"status": "ok", "message": "Settings reloaded from disk."}


@app.get("/api/v1/market/summary")
async def get_market_summary():
    """Returns live prices, day-change for US, Global, Crypto, Commodities and FX."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    sections = {
        "us":     _US_SYMBOLS,
        "global": _GLOBAL_SYMBOLS,
        "assets": _ASSET_SYMBOLS,
        "fx":     _FX_SYMBOLS,
    }

    def fetch_section(label_sym_pairs):
        out = []
        for label, sym in label_sym_pairs:
            q = _fetch_quote_fast(sym)
            if q:
                out.append({"label": label, "symbol": sym, **q})
        return out

    results = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {
            name: ex.submit(fetch_section, list(syms.items()))
            for name, syms in sections.items()
        }
        for name, fut in futures.items():
            try:
                results[name] = fut.result(timeout=15)
            except Exception:
                results[name] = []

    return results


@app.get("/api/v1/market/news")
async def get_market_news(limit: int = 40):
    """
    Returns recent global market news from RSS feeds + yfinance headlines.
    Sources: MarketWatch, Reuters, CNBC, BBC, Yahoo Finance, WSJ, Bloomberg.
    """
    import yfinance as yf
    from concurrent.futures import ThreadPoolExecutor
    import time as _time

    seen, articles = set(), []

    # ── 1. RSS feeds (parallel) ──────────────────────────────────────────
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

    # ── 2. yfinance news as supplement ───────────────────────────────────
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
                ts = item.get("providerPublishTime") or item.get("pubDate") or int(_time.time())
                articles.append({"title": title, "publisher": publisher,
                                  "link": link, "published_at": int(ts), "ticker": sym})
        except Exception:
            continue

    articles.sort(key=lambda x: x.get("published_at", 0), reverse=True)
    return {"news": articles[:limit]}


@app.get("/api/v1/market/earnings")
async def get_upcoming_earnings():
    """Returns upcoming earnings dates for major stocks."""
    import yfinance as yf
    from datetime import date, timedelta
    import pandas as pd

    today = date.today()
    cutoff = today + timedelta(days=settings.get("data.earnings_horizon_days", 30))
    results = []

    for sym in _EARNINGS_WATCHLIST:
        try:
            t = yf.Ticker(sym)
            cal = t.calendar
            if cal is None:
                continue
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
                q  = _fetch_quote_fast(sym)
                eps_est = cal.get("EPS Estimate") or cal.get("Earnings Average")
                rev_est = cal.get("Revenue Estimate") or cal.get("Revenue Average")
                results.append({
                    "ticker":           sym,
                    "company":          getattr(fi, "company_name", sym) if hasattr(fi, "company_name") else sym,
                    "earnings_date":    str(earnings_date),
                    "days_until":       (earnings_date - today).days,
                    "price":            q["price"] if q else None,
                    "change_pct":       q["change_pct"] if q else None,
                    "positive":         q["positive"] if q else True,
                    "eps_estimate":     round(float(eps_est), 2) if eps_est and str(eps_est) not in ("nan","None") else None,
                    "revenue_estimate_b": round(float(rev_est) / 1e9, 2) if rev_est and str(rev_est) not in ("nan","None") else None,
                })
        except Exception:
            continue

    results.sort(key=lambda x: x["days_until"])
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
    Period: 1mo, 3mo, 6mo, 1y, 2y
    """
    import yfinance as yf
    import pandas as pd
    valid_periods = {"1mo", "3mo", "6mo", "1y", "2y"}
    if period not in valid_periods:
        period = "3mo"
    try:
        t = yf.Ticker(ticker.upper())
        hist = t.history(period=period, interval="1d", auto_adjust=True)
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


@app.get("/api/v1/signal/{ticker}")
async def get_signal(ticker: str):
    """
    Executes the full multi-agent orchestrator for a given ticker.
    """
    start_time = time.time()
    logger.info(f"Received signal request for: {ticker}")
    
    try:
        md = MarketData(ticker)
        initial_state = {
            "ticker": ticker,
            "market_data": md,
            "registry": registry
        }
        
        # Invoke the LangGraph pipeline
        result_state = graph.invoke(initial_state)
        
        final_info = result_state["final_signal"]
        packet = final_info["packet"]
        
        # Enrich packet with metadata
        latency_ms = (time.time() - start_time) * 1000
        packet.computation_time_ms = latency_ms
        packet.agents_used = len(packet.agent_results)
        
        # Record metrics
        Monitor.record_request(latency_ms, success=True)
        for res in packet.agent_results:
            Monitor.record_agent_run(res.agent_name, res.computation_time_ms, success=True)
            
        # ── Persistent Storage ──
        signal_data = {
            "direction": packet.direction,
            "probability": final_info["probability_up"],
            "conviction": packet.conviction_pct,
            "multiplier": final_info.get("multiplier", 1.0)
        }
        trade_id = db_manager.record_signal(ticker, signal_data, packet.agent_results)
            
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
            direction   = packet.direction,
            probability = final_info["probability_up"],
            conviction  = packet.conviction_pct,
            entropy     = final_info.get("entropy", 0.0),
            multiplier  = final_info.get("multiplier", 1.0),
            agents      = packet.agent_results,
            warnings    = packet.warnings,
            holding_block = holding_block,
        )

        return {
            "trade_id": trade_id,
            "ticker": ticker,
            "direction": packet.direction,
            "probability": final_info["probability_up"],
            "conviction": packet.conviction_pct,
            "multiplier": final_info.get("multiplier", 1.0),
            "entropy": final_info.get("entropy", 0.0),
            "agents": packet.agent_results,
            "warnings": packet.warnings,
            "holding_period": holding_block,
            "summary": summary,
            "latency_ms": round(latency_ms, 1)
        }
        
    except Exception as e:
        logger.error(f"Signal generation failed for {ticker}: {e}", exc_info=True)
        Monitor.record_request(0, success=False)
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

RULES: Answer only from this context. Do not invent numbers. If entropy > 0.8, note that agents strongly disagree. Use bullet points when listing multiple items. Keep answers ≤200 words unless the user explicitly asks for more detail."""

    # ── Call Claude ────────────────────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {"answer": (
            f"AI assistant is offline — ANTHROPIC_API_KEY not set. "
            f"Signal summary: {ticker} is **{direction}** with {prob*100:.1f}% probability up "
            f"and {conviction:.1f}% conviction. "
            f"Recommended: {summary.get('action', 'N/A')}"
        )}

    try:
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": question}],
        )
        answer = msg.content[0].text if msg.content else "No response generated."
        return {"answer": answer, "ticker": ticker}
    except Exception as e:
        logger.error(f"Chat API error: {e}")
        raise HTTPException(status_code=500, detail=f"AI chat error: {str(e)}")


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
        return {
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
                for t in result.trades[-50:]  # last 50 trades
            ],
        }
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


@app.get("/api/v1/leaderboard")
async def get_leaderboard(days: int = 90, ticker: str = None):
    """
    Returns agent performance rankings over the specified window.
    Shows demo data until real signal history is logged.
    """
    try:
        lb = AgentLeaderboard(db_manager=db_manager)
        result = lb.evaluate_performance(ticker=ticker, window_days=days)
        return {
            "evaluation_window_days": result.evaluation_window_days,
            "total_signals_evaluated": result.total_signals_evaluated,
            "is_demo": result.total_signals_evaluated == 0,
            "rankings": [
                {
                    "rank": s.rank,
                    "agent": s.agent_name,
                    "n_predictions": s.n_predictions,
                    "directional_accuracy_pct": s.directional_accuracy,
                    "brier_score": s.brier_score,
                    "information_ratio": s.information_ratio,
                    "avg_confidence": s.avg_confidence,
                }
                for s in result.scores
            ],
        }
    except Exception as e:
        logger.error(f"Leaderboard failed: {e}", exc_info=True)
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
        md = MarketData(ticker.upper())
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
