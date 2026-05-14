"""
AlphaAgent — SQLite Caching Layer

Caches API responses to avoid redundant calls and respect rate limits.
Data is stored with TTL (time-to-live) so stale data auto-expires.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional


class DataCache:
    """
    SQLite-based cache for API responses.
    
    Usage:
        cache = DataCache()
        
        # Check cache first
        data = cache.get("yfinance", "NVDA", "ohlcv_1y")
        if data is None:
            data = fetch_from_api(...)
            cache.set("yfinance", "NVDA", "ohlcv_1y", data, ttl_seconds=3600)
    """
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(Path(__file__).parent.parent / "data" / "cache.db")
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Create the cache table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    source TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    PRIMARY KEY (source, ticker, data_type)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_expiry 
                ON cache (expires_at)
            """)
            conn.commit()
    
    def get(self, source: str, ticker: str, data_type: str) -> Optional[Any]:
        """
        Get cached data if it exists and hasn't expired.
        
        Args:
            source: Data source name (e.g., "yfinance", "fred", "newsapi")
            ticker: Stock ticker (e.g., "NVDA") or "global" for non-ticker data
            data_type: Type of data (e.g., "ohlcv_1y", "info", "financials")
            
        Returns:
            Cached data (deserialized from JSON) or None if not found/expired
        """
        now = time.time()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT data FROM cache 
                WHERE source = ? AND ticker = ? AND data_type = ? AND expires_at > ?
                """,
                (source, ticker.upper(), data_type, now)
            )
            row = cursor.fetchone()
        
        if row is None:
            return None
        
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return None
    
    def set(
        self, 
        source: str, 
        ticker: str, 
        data_type: str, 
        data: Any, 
        ttl_seconds: int = 3600
    ):
        """
        Store data in cache with a time-to-live.
        
        Args:
            source: Data source name
            ticker: Stock ticker or "global"
            data_type: Type of data
            data: Data to cache (must be JSON-serializable)
            ttl_seconds: Time-to-live in seconds (default: 1 hour)
        """
        now = time.time()
        expires_at = now + ttl_seconds
        
        serialized = json.dumps(data, default=str)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache 
                (source, ticker, data_type, data, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (source, ticker.upper(), data_type, serialized, now, expires_at)
            )
            conn.commit()
    
    def invalidate(self, source: str, ticker: str, data_type: str):
        """Remove a specific cache entry."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM cache WHERE source = ? AND ticker = ? AND data_type = ?",
                (source, ticker.upper(), data_type)
            )
            conn.commit()
    
    def invalidate_ticker(self, ticker: str):
        """Remove all cache entries for a ticker."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache WHERE ticker = ?", (ticker.upper(),))
            conn.commit()
    
    def cleanup_expired(self):
        """Remove all expired cache entries."""
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM cache WHERE expires_at < ?", (now,))
            conn.commit()
            return cursor.rowcount
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            valid = conn.execute(
                "SELECT COUNT(*) FROM cache WHERE expires_at > ?", (now,)
            ).fetchone()[0]
            expired = total - valid
        
        return {
            "total_entries": total,
            "valid_entries": valid,
            "expired_entries": expired,
            "db_path": self.db_path,
        }
