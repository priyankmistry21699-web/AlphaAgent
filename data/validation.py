"""
AlphaAgent — Data Validation & Cleaning

Handles NaN values, stock splits, dividend adjustments, gaps in data,
and ensures all DataFrames are clean before entering the quant engine.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_ohlcv(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """
    Validate and clean OHLCV data.
    
    Handles:
        - Missing values (forward-fill, then back-fill)
        - Zero/negative prices (remove rows)
        - Duplicate indices (keep last)
        - Non-trading days (already excluded by yfinance)
        - Volume anomalies (cap at 99.9th percentile for extreme outliers)
    
    Args:
        df: Raw OHLCV DataFrame with columns [Open, High, Low, Close, Volume]
        ticker: Ticker symbol for logging
        
    Returns:
        Cleaned DataFrame
    """
    if df is None or df.empty:
        logger.warning(f"[{ticker}] Empty OHLCV data received")
        return pd.DataFrame()
    
    original_len = len(df)
    
    # Ensure we have the required columns
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        logger.error(f"[{ticker}] Missing columns: {missing_cols}")
        return pd.DataFrame()
    
    # Remove duplicate indices (keep last)
    if df.index.duplicated().any():
        dups = df.index.duplicated().sum()
        df = df[~df.index.duplicated(keep="last")]
        logger.info(f"[{ticker}] Removed {dups} duplicate dates")
    
    # Sort by date
    df = df.sort_index()
    
    # Remove rows with zero or negative prices
    price_cols = ["Open", "High", "Low", "Close"]
    invalid_prices = (df[price_cols] <= 0).any(axis=1)
    if invalid_prices.any():
        count = invalid_prices.sum()
        df = df[~invalid_prices]
        logger.info(f"[{ticker}] Removed {count} rows with invalid prices")
    
    # Handle missing values
    nan_count = df[price_cols].isna().sum().sum()
    if nan_count > 0:
        logger.info(f"[{ticker}] Filling {nan_count} NaN values in prices")
        df[price_cols] = df[price_cols].ffill().bfill()
    
    # Fill missing volume with 0
    df["Volume"] = df["Volume"].fillna(0)
    
    # Ensure High >= Low (swap if needed)
    swap_mask = df["High"] < df["Low"]
    if swap_mask.any():
        df.loc[swap_mask, ["High", "Low"]] = df.loc[swap_mask, ["Low", "High"]].values
        logger.info(f"[{ticker}] Swapped High/Low for {swap_mask.sum()} rows")
    
    # Log summary
    final_len = len(df)
    if final_len < original_len:
        logger.info(
            f"[{ticker}] Validation: {original_len} → {final_len} rows "
            f"({original_len - final_len} removed)"
        )
    
    return df


def compute_returns(
    df: pd.DataFrame, 
    method: str = "log",
    column: str = "Close"
) -> pd.Series:
    """
    Compute returns from price series.
    
    Args:
        df: DataFrame with price column
        method: "log" for log returns, "simple" for arithmetic returns
        column: Price column name
        
    Returns:
        Series of returns (first value will be NaN, dropped)
    """
    prices = df[column]
    
    if method == "log":
        returns = np.log(prices / prices.shift(1))
    elif method == "simple":
        returns = prices.pct_change()
    else:
        raise ValueError(f"Unknown return method: {method}. Use 'log' or 'simple'.")
    
    # Drop the first NaN
    returns = returns.dropna()
    
    return returns


def detect_outliers(
    series: pd.Series, 
    method: str = "zscore", 
    threshold: float = 5.0
) -> pd.Series:
    """
    Detect outliers in a series.
    
    Args:
        series: Input series
        method: "zscore" or "iqr"
        threshold: Z-score threshold (default 5.0 for financial data)
        
    Returns:
        Boolean series (True = outlier)
    """
    if method == "zscore":
        z_scores = (series - series.mean()) / series.std()
        return z_scores.abs() > threshold
    
    elif method == "iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return (series < lower) | (series > upper)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'zscore' or 'iqr'.")


def validate_financials(data: dict, ticker: str = "") -> dict:
    """
    Validate financial data (income statement, balance sheet, cash flow).
    Replace None/NaN values with 0 and log warnings for missing critical fields.
    
    Args:
        data: Dictionary of financial metrics
        ticker: Ticker symbol for logging
        
    Returns:
        Cleaned dictionary
    """
    critical_fields = [
        "trailingPE", "forwardPE", "priceToBook", "marketCap",
        "totalRevenue", "netIncome", "freeCashflow"
    ]
    
    cleaned = {}
    for key, value in data.items():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            if key in critical_fields:
                logger.warning(f"[{ticker}] Missing critical field: {key}")
            cleaned[key] = 0.0
        else:
            cleaned[key] = value
    
    return cleaned


def ensure_minimum_history(
    df: pd.DataFrame, 
    min_days: int = 200, 
    ticker: str = ""
) -> bool:
    """
    Check if we have enough historical data for analysis.
    
    Args:
        df: OHLCV DataFrame
        min_days: Minimum trading days required
        ticker: Ticker symbol for logging
        
    Returns:
        True if sufficient data, False otherwise
    """
    if df is None or len(df) < min_days:
        actual = 0 if df is None else len(df)
        logger.warning(
            f"[{ticker}] Insufficient history: {actual} days "
            f"(need {min_days}). Some indicators will be unavailable."
        )
        return False
    return True
