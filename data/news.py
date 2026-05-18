"""
AlphaAgent — News Storage (SQLite keyword search)

Fetches news from yfinance and stores in SQLite for fast keyword-based retrieval.
No sentence-transformer or ChromaDB required — zero model loading delay.
"""

import sqlite3
import logging
import time
from pathlib import Path

import yfinance as yf

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DB_PATH = str(_PROJECT_ROOT / "data" / "news.db")
_MAX_ARTICLE_AGE_DAYS = 7


class NewsDatabase:
    """
    SQLite-backed news store with keyword search.
    Replaces the previous ChromaDB + sentence-transformer implementation
    so the server starts instantly with no model loading.
    """

    def __init__(self, db_path: str = _DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS news (
                    id          TEXT PRIMARY KEY,
                    ticker      TEXT NOT NULL,
                    title       TEXT NOT NULL,
                    publisher   TEXT,
                    url         TEXT,
                    timestamp   INTEGER NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_news_ticker ON news(ticker)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_news_ts    ON news(timestamp)")
            conn.commit()

    # no-op warm_up kept for API compatibility
    def warm_up(self) -> None:
        pass

    def _cleanup_old_articles(self) -> None:
        cutoff = int(time.time()) - _MAX_ARTICLE_AGE_DAYS * 86400
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("DELETE FROM news WHERE timestamp < ?", (cutoff,))
                if cur.rowcount:
                    logger.info(f"[News] Pruned {cur.rowcount} stale articles.")
                conn.commit()
        except Exception as e:
            logger.warning(f"[News] TTL cleanup failed: {e}")

    def fetch_and_store_news(self, ticker: str) -> None:
        self._cleanup_old_articles()
        try:
            articles = yf.Ticker(ticker).news or []
        except Exception:
            return

        rows = []
        for a in articles:
            title = a.get("title", "")
            if not title:
                continue
            ts  = int(a.get("providerPublishTime", 0))
            doc_id = f"{ticker}_{ts}_{hash(title) & 0xFFFFFFFF}"
            rows.append((doc_id, ticker.upper(), title,
                          a.get("publisher", ""), a.get("link", ""), ts))

        if rows:
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany(
                    "INSERT OR IGNORE INTO news(id,ticker,title,publisher,url,timestamp) VALUES(?,?,?,?,?,?)",
                    rows,
                )
                conn.commit()
            logger.info(f"[News] Stored {len(rows)} articles for {ticker}.")

    def search_news(self, query: str, ticker: str = None, n_results: int = 5, **_) -> list:
        """
        Keyword search: returns titles containing any word from the query.
        Falls back to most-recent articles if no keyword matches.
        """
        words = [w.strip().lower() for w in query.split() if len(w.strip()) > 3]
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                if ticker:
                    base = "SELECT title FROM news WHERE ticker=? ORDER BY timestamp DESC LIMIT 50"
                    rows = conn.execute(base, (ticker.upper(),)).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT title FROM news ORDER BY timestamp DESC LIMIT 100"
                    ).fetchall()

                titles = [r["title"] for r in rows]

                # Score by keyword overlap
                scored = []
                for t in titles:
                    tl = t.lower()
                    score = sum(1 for w in words if w in tl)
                    scored.append((score, t))
                scored.sort(key=lambda x: -x[0])

                results = [t for _, t in scored if _ > 0]
                if not results:
                    results = titles  # fall back to recency
                return results[:n_results]
        except Exception as e:
            logger.warning(f"[News] search_news failed: {e}")
            return []
