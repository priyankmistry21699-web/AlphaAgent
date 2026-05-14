"""
AlphaAgent — Momentum & Hurst Exponent Engine

Calculates advanced momentum characteristics like the Hurst Exponent 
to determine if the asset is strongly trending, mean-reverting, or 
experiencing random walk.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class MomentumResult:
    hurst_exponent: float
    regime_type: str        # "TRENDING", "MEAN_REVERTING", "RANDOM_WALK"
    momentum_12m_1m: float  # (12 Month Return) - (1 Month Return)


def calculate_hurst(time_series: pd.Series, max_lag: int = 20) -> float:
    """
    Calculates the Hurst Exponent of a time series.
    H < 0.5: Mean Reverting
    H = 0.5: Random Walk (Brownian Motion)
    H > 0.5: Trending (Persistent)
    """
    lags = range(2, max_lag)
    
    # Calculate the array of the variances of the lagged differences
    tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
    
    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    
    # Return the Hurst exponent (multiply by 2.0 because of the sqrt above)
    return poly[0] * 2.0


class MomentumEngine:
    """Computes deep momentum characteristics."""
    
    def __init__(self, prices: pd.Series):
        self.prices = prices.dropna()
        
    def analyze(self) -> MomentumResult:
        if len(self.prices) < 252:
            return MomentumResult(0.5, "RANDOM_WALK", 0.0)
            
        # 1. Calculate Hurst Exponent
        hurst = calculate_hurst(self.prices.values)
        
        # Determine regime
        if hurst > 0.6:
            regime = "TRENDING"
        elif hurst < 0.4:
            regime = "MEAN_REVERTING"
        else:
            regime = "RANDOM_WALK"
            
        # 2. Calculate 12M-1M Momentum (Academic Standard)
        # We look at the return over the last 12 months, excluding the most recent month
        current_price = self.prices.iloc[-1]
        price_1m_ago = self.prices.iloc[-21] # approx 1 month (21 trading days)
        price_12m_ago = self.prices.iloc[-252] # approx 12 months (252 trading days)
        
        ret_12m = (current_price - price_12m_ago) / price_12m_ago
        ret_1m = (current_price - price_1m_ago) / price_1m_ago
        
        # The 12M-1M momentum factor isolates long-term trend from short-term noise/reversion
        mom_12m_1m = ret_12m - ret_1m
        
        return MomentumResult(
            hurst_exponent=hurst,
            regime_type=regime,
            momentum_12m_1m=mom_12m_1m
        )
