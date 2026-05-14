"""
AlphaAgent — Macro Data Layer

Fetches macroeconomic data directly from the Federal Reserve Economic Data (FRED).
Uses pandas_datareader, which does not require an API key for basic series.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from data.cache import DataCache

logger = logging.getLogger(__name__)


class MacroData:
    """
    Unified interface for fetching macroeconomic indicators.
    
    Supported Indicators (FRED Series IDs):
        - T10Y2Y: 10-Year minus 2-Year Yield (Yield Curve Inversion)
        - FEDFUNDS: Federal Funds Rate (Interest Rates)
        - CPIAUCSL: Consumer Price Index (Inflation)
        - VIXCLS: CBOE Volatility Index (Macro Fear)
        - UNRATE: Unemployment Rate
    """
    
    # Macro data updates slowly, so we can cache it for a long time.
    MACRO_TTL = 86400  # 24 hours
    
    def __init__(self, cache: Optional[DataCache] = None):
        self.cache = cache or DataCache()
        logger.info("MacroData initialized")

    def get_series(self, series_id: str, years_back: int = 5) -> pd.DataFrame:
        """
        Fetch a specific FRED series with SQLite caching.
        """
        cache_key = f"fred_{series_id}_{years_back}y"
        
        # Check Cache
        cached = self.cache.get("macro", "FRED", cache_key)
        if cached is not None:
            df = pd.DataFrame(cached)
            if "DATE" in df.columns:
                df["DATE"] = pd.to_datetime(df["DATE"])
                df = df.set_index("DATE")
            logger.debug(f"[FRED] Loaded {series_id} from cache")
            return df
            
        # Fetch from FRED
        logger.info(f"[FRED] Fetching {series_id} over past {years_back} years...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years_back * 365)
        
        try:
            # FRED provides a direct CSV endpoint that does not require an API key
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            logger.info(f"[FRED] Fetching {series_id} from {url}")
            
            # Read direct to pandas
            df = pd.read_csv(url, parse_dates=["observation_date"], index_col="observation_date", na_values=".")
            df.index.name = "DATE"
            
            # Filter by date
            df = df[df.index >= pd.Timestamp(start_date)]
            
            # Clean missing data (FRED sometimes returns '.' as NaNs on holidays)
            df = df.ffill().dropna()
            
            if not df.empty:
                # Rename the column to match standard format
                df.columns = ["VALUE"]
                
                # Cache it
                cache_df = df.copy()
                cache_df.index = cache_df.index.astype(str)
                self.cache.set(
                    "macro", "FRED", cache_key,
                    cache_df.to_dict(orient="list") | {"DATE": list(cache_df.index)},
                    ttl_seconds=self.MACRO_TTL
                )
            return df
            
        except Exception as e:
            logger.error(f"[FRED] Failed to fetch {series_id}: {e}")
            return pd.DataFrame()

    def get_macro_snapshot(self) -> dict:
        """
        Fetches the most recent value for key macroeconomic indicators.
        """
        snapshot = {}
        
        # Yield Curve (Inverted if < 0)
        t10y2y = self.get_series("T10Y2Y", years_back=1)
        if not t10y2y.empty:
            snapshot["yield_curve"] = float(t10y2y.iloc[-1].iloc[0])
            
        # Fed Funds Rate
        fedfunds = self.get_series("FEDFUNDS", years_back=1)
        if not fedfunds.empty:
            snapshot["fed_funds_rate"] = float(fedfunds.iloc[-1].iloc[0])
            
        # VIX
        vix = self.get_series("VIXCLS", years_back=1)
        if not vix.empty:
            snapshot["vix"] = float(vix.iloc[-1].iloc[0])
            
        # Unemployment
        unrate = self.get_series("UNRATE", years_back=1)
        if not unrate.empty:
            snapshot["unemployment"] = float(unrate.iloc[-1].iloc[0])
            
        return snapshot
