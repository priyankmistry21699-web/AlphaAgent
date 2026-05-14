"""
AlphaAgent — Market Data Layer

Wraps yfinance with caching, validation, and convenience methods.
This is the primary data source for Phase 1 — no API key required.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from data.cache import DataCache
from data.validation import (
    validate_ohlcv,
    validate_financials,
    compute_returns,
    ensure_minimum_history,
)

logger = logging.getLogger(__name__)


class MarketData:
    """
    Unified market data interface for a single ticker.
    
    Handles:
        - OHLCV price data (daily, with caching)
        - Company info (P/E, sector, market cap, etc.)
        - Financial statements (income, balance sheet, cash flow)
        - Options chain (IV, put/call, open interest)
        - Insider transactions
        - Institutional holders
    
    Usage:
        >>> md = MarketData("NVDA")
        >>> ohlcv = md.get_ohlcv(period="1y")
        >>> info = md.get_info()
        >>> returns = md.get_returns()
    """
    
    @staticmethod
    def _cache_cfg() -> dict:
        from config.settings_manager import settings
        return settings.get_section("cache")

    @property
    def OHLCV_TTL(self):
        return self._cache_cfg().get("price_ttl_seconds", 300)

    @property
    def INFO_TTL(self):
        return self._cache_cfg().get("info_ttl_seconds", 86400)

    @property
    def FINANCIALS_TTL(self):
        return self._cache_cfg().get("financials_ttl_seconds", 604800)

    @property
    def OPTIONS_TTL(self):
        return self._cache_cfg().get("options_ttl_seconds", 300)

    @property
    def INSIDER_TTL(self):
        return self._cache_cfg().get("insider_ttl_seconds", 3600)

    @property
    def HOLDERS_TTL(self):
        return self._cache_cfg().get("holders_ttl_seconds", 86400)
    
    def __init__(self, ticker: str, cache: Optional[DataCache] = None):
        self.ticker = ticker.upper()
        self.cache = cache or DataCache()
        self._yf_ticker = yf.Ticker(self.ticker)
        self._ohlcv_cache: dict[str, pd.DataFrame] = {}
        
        logger.info(f"MarketData initialized for {self.ticker}")
    
    # ─── Price Data ───────────────────────────────────────────────────────
    
    def get_ohlcv(self, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Get OHLCV price data with caching and validation.
        
        Args:
            period: Data period (e.g., "1y", "2y", "5y", "6mo", "3mo")
            interval: Data interval (e.g., "1d", "1h", "15m")
            
        Returns:
            Validated DataFrame with columns [Open, High, Low, Close, Volume]
        """
        cache_key = f"ohlcv_{period}_{interval}"
        
        # Check in-memory cache first (fastest)
        if cache_key in self._ohlcv_cache:
            return self._ohlcv_cache[cache_key]
        
        # Check SQLite cache
        cached = self.cache.get("yfinance", self.ticker, cache_key)
        if cached is not None:
            df = pd.DataFrame(cached)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], utc=True)
                df = df.set_index("Date")
            elif df.index.dtype == object:
                df.index = pd.to_datetime(df.index)
            
            self._ohlcv_cache[cache_key] = df
            logger.debug(f"[{self.ticker}] OHLCV from cache ({len(df)} rows)")
            return df
        
        # Fetch from yfinance
        logger.info(f"[{self.ticker}] Fetching OHLCV: period={period}, interval={interval}")
        try:
            df = self._yf_ticker.history(period=period, interval=interval)
        except Exception as e:
            logger.error(f"[{self.ticker}] yfinance error: {e}")
            return pd.DataFrame()
        
        # Validate and clean
        df = validate_ohlcv(df, self.ticker)
        
        if df.empty:
            logger.warning(f"[{self.ticker}] No valid OHLCV data")
            return df
        
        # Cache it
        cache_df = df.copy()
        cache_df.index = cache_df.index.astype(str)
        self.cache.set(
            "yfinance", self.ticker, cache_key,
            cache_df.to_dict(orient="list") | {"Date": list(cache_df.index)},
            ttl_seconds=self.OHLCV_TTL
        )
        self._ohlcv_cache[cache_key] = df
        
        logger.info(f"[{self.ticker}] OHLCV fetched: {len(df)} rows, "
                     f"{df.index[0].date()} to {df.index[-1].date()}")
        return df
    
    def get_returns(
        self, period: str = "1y", method: str = "log"
    ) -> pd.Series:
        """Get return series from price data."""
        ohlcv = self.get_ohlcv(period=period)
        if ohlcv.empty:
            return pd.Series(dtype=float)
        return compute_returns(ohlcv, method=method)
    
    def get_current_price(self) -> float:
        """Get the latest closing price."""
        ohlcv = self.get_ohlcv(period="5d")
        if ohlcv.empty:
            return 0.0
        return float(ohlcv["Close"].iloc[-1])
    
    # ─── Company Info ─────────────────────────────────────────────────────
    
    def get_info(self) -> dict:
        """
        Get company info (P/E, market cap, sector, etc.).
        
        Returns dict with keys like:
            trailingPE, forwardPE, priceToBook, marketCap, sector,
            industry, fullTimeEmployees, dividendYield, etc.
        """
        cached = self.cache.get("yfinance", self.ticker, "info")
        if cached is not None:
            return cached
        
        logger.info(f"[{self.ticker}] Fetching company info")
        try:
            info = self._yf_ticker.info
        except Exception as e:
            logger.error(f"[{self.ticker}] Info fetch error: {e}")
            return {}
        
        # Validate
        info = validate_financials(info, self.ticker)
        
        # Cache
        self.cache.set("yfinance", self.ticker, "info", info, 
                       ttl_seconds=self.INFO_TTL)
        return info
    
    # ─── Financial Statements ─────────────────────────────────────────────
    
    def get_financials(self) -> dict:
        """
        Get financial statements (income, balance sheet, cash flow).
        
        Returns:
            Dict with keys: "income", "balance", "cashflow"
            Each value is a DataFrame with quarterly data.
        """
        cached = self.cache.get("yfinance", self.ticker, "financials")
        if cached is not None:
            result = {}
            for key in ["income", "balance", "cashflow"]:
                if key in cached:
                    df = pd.DataFrame(cached[key])
                    if "Date" in df.columns:
                        df = df.set_index("Date")
                    result[key] = df
            return result
        
        logger.info(f"[{self.ticker}] Fetching financial statements")
        try:
            result = {
                "income": self._yf_ticker.quarterly_financials,
                "balance": self._yf_ticker.quarterly_balance_sheet,
                "cashflow": self._yf_ticker.quarterly_cashflow,
            }
        except Exception as e:
            logger.error(f"[{self.ticker}] Financials fetch error: {e}")
            return {}
        
        # Cache (convert DataFrames to dicts)
        cache_data = {}
        for key, df in result.items():
            if df is not None and not df.empty:
                cache_df = df.copy()
                cache_df.columns = cache_df.columns.astype(str)
                cache_data[key] = cache_df.to_dict()
        
        self.cache.set("yfinance", self.ticker, "financials", 
                       cache_data, ttl_seconds=self.FINANCIALS_TTL)
        return result
    
    # ─── Options Data ─────────────────────────────────────────────────────
    
    def get_options_chain(self, expiry_index: int = 0) -> Optional[dict]:
        """
        Get options chain for the nearest expiry.
        
        Args:
            expiry_index: 0 = nearest expiry, 1 = next, etc.
            
        Returns:
            Dict with "calls", "puts" DataFrames and "expiry" date string
        """
        try:
            expirations = self._yf_ticker.options
            if not expirations:
                logger.warning(f"[{self.ticker}] No options data available")
                return None
            
            if expiry_index >= len(expirations):
                expiry_index = 0
            
            expiry = expirations[expiry_index]
            chain = self._yf_ticker.option_chain(expiry)
            
            return {
                "calls": chain.calls,
                "puts": chain.puts,
                "expiry": expiry,
                "all_expiries": list(expirations),
            }
        except Exception as e:
            logger.error(f"[{self.ticker}] Options fetch error: {e}")
            return None
    
    # ─── Insider & Institutional Data ─────────────────────────────────────
    
    def get_insider_transactions(self) -> pd.DataFrame:
        """Get recent insider buy/sell transactions (SEC Form 4)."""
        try:
            transactions = self._yf_ticker.insider_transactions
            if transactions is None or transactions.empty:
                return pd.DataFrame()
            return transactions
        except Exception as e:
            logger.error(f"[{self.ticker}] Insider data error: {e}")
            return pd.DataFrame()
    
    def get_institutional_holders(self) -> pd.DataFrame:
        """Get top institutional holders."""
        try:
            holders = self._yf_ticker.institutional_holders
            if holders is None or holders.empty:
                return pd.DataFrame()
            return holders
        except Exception as e:
            logger.error(f"[{self.ticker}] Holders data error: {e}")
            return pd.DataFrame()
    
    def get_major_holders(self) -> pd.DataFrame:
        """Get major holder breakdown (% held by insiders, institutions, etc.)."""
        try:
            holders = self._yf_ticker.major_holders
            if holders is None or holders.empty:
                return pd.DataFrame()
            return holders
        except Exception as e:
            logger.error(f"[{self.ticker}] Major holders error: {e}")
            return pd.DataFrame()
    
    # ─── Utility Methods ──────────────────────────────────────────────────
    
    def has_sufficient_history(self, min_days: int = 200) -> bool:
        """Check if we have enough data for analysis."""
        ohlcv = self.get_ohlcv(period="1y")
        return ensure_minimum_history(ohlcv, min_days, self.ticker)
        
    def get_yfinance_ticker(self) -> yf.Ticker:
        """Get the underlying yfinance Ticker object for direct access."""
        return self._yf_ticker
    
    def summary(self) -> dict:
        """Quick summary of available data for this ticker."""
        ohlcv = self.get_ohlcv(period="5d")
        info = self.get_info()
        
        return {
            "ticker": self.ticker,
            "current_price": self.get_current_price(),
            "company_name": info.get("longName", "Unknown"),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", None),
            "forward_pe": info.get("forwardPE", None),
            "dividend_yield": info.get("dividendYield", None),
            "52w_high": info.get("fiftyTwoWeekHigh", None),
            "52w_low": info.get("fiftyTwoWeekLow", None),
        }
    
    def __repr__(self) -> str:
        return f"MarketData('{self.ticker}')"
