"""
AlphaAgent — Options Intelligence Engine

Calculates options market positioning, primarily the Put/Call Ratio
to determine if the options market is hedging heavily (bearish) or 
speculating (bullish).
"""

import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class OptionsResult:
    put_call_ratio: float
    implied_move_pct: float   # ATM straddle / price (decimal), over nearest expiry horizon
    expiry_days: int          # Calendar days to the nearest expiry used
    warnings: list[str] = field(default_factory=list)


def analyze_options(ticker_obj) -> OptionsResult:
    """
    Takes a yfinance Ticker object and calculates options intelligence 
    based on the nearest expiration date.
    """
    warnings = []
    
    try:
        # Get all expiration dates
        expirations = ticker_obj.options
        if not expirations:
            return OptionsResult(1.0, 0.0, 0, ["No options chain available."])

        # Get the nearest expiration (usually the most liquid short-term proxy)
        nearest_exp = expirations[0]
        expiry_dt = datetime.strptime(nearest_exp, "%Y-%m-%d")
        expiry_days = max(1, (expiry_dt - datetime.utcnow()).days)
        chain = ticker_obj.option_chain(nearest_exp)
        
        calls = chain.calls
        puts = chain.puts
        
        if calls.empty or puts.empty:
            return OptionsResult(1.0, 0.0, expiry_days, ["Options chain is empty."])
            
        # 1. Calculate Put/Call Open Interest Ratio
        total_call_oi = calls['openInterest'].sum()
        total_put_oi = puts['openInterest'].sum()
        
        if total_call_oi == 0:
            pc_ratio = 1.0
        else:
            pc_ratio = total_put_oi / total_call_oi
            
        if pc_ratio > 1.5:
            warnings.append(f"High Put/Call Ratio ({pc_ratio:.2f}). Market is heavily hedged (Bearish/Fearful).")
        elif pc_ratio < 0.6:
            warnings.append(f"Low Put/Call Ratio ({pc_ratio:.2f}). Market is highly speculative (Complacent).")
            
        # 2. Estimate Implied Move
        # (ATM Straddle Approximation = ATM Call Price + ATM Put Price)
        # We need current price to find ATM
        current_price = ticker_obj.history(period="1d")['Close'].iloc[-1]
        
        # Find ATM strikes
        atm_call = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:1]]
        atm_put = puts.iloc[(puts['strike'] - current_price).abs().argsort()[:1]]
        
        if not atm_call.empty and not atm_put.empty:
            call_px = float(atm_call['lastPrice'].fillna(0).values[0])
            put_px = float(atm_put['lastPrice'].fillna(0).values[0])
            straddle_price = call_px + put_px
            implied_move = straddle_price / current_price if straddle_price > 0 else 0.0
        else:
            implied_move = 0.0
            
        return OptionsResult(
            put_call_ratio=pc_ratio,
            implied_move_pct=implied_move,
            expiry_days=expiry_days,
            warnings=warnings,
        )

    except Exception as e:
        logger.error(f"Options analysis failed: {e}")
        return OptionsResult(1.0, 0.0, 0, [f"Options API Error: {str(e)}"])
