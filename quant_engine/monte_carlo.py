"""
AlphaAgent — Monte Carlo Price Simulator

Simulates 10,000+ parallel future price paths using Geometric Brownian Motion (GBM).
Integrates with the GARCH model to use dynamic, forecasted volatility rather
than static historical volatility.

Key concepts:
  - GBM assumes returns are normally distributed with a drift (average return)
    and a shock (volatility * random standard normal).
  - GARCH integration: Instead of constant volatility, we use the specific
    daily volatility forecasted by GARCH.
  - Confidence Intervals: Gives us exact mathematical probabilities (e.g.,
    "There is a 95% chance the price will land between $130 and $150").
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from quant_engine.garch import GARCHResult

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceInterval:
    """Price confidence interval from Monte Carlo simulation."""
    level: float                           # e.g., 0.68, 0.95, 0.99
    low: float
    high: float


@dataclass
class MonteCarloResult:
    """Output from Monte Carlo price simulation."""
    current_price: float
    simulation_days: int
    num_paths: int
    
    # Statistical results
    mean_price: float
    median_price: float
    expected_return_pct: float             # Mean expected return
    prob_above_current: float              # % of paths ending above current price
    
    # Confidence Intervals
    ci_68: ConfidenceInterval              # 1 standard deviation
    ci_95: ConfidenceInterval              # 2 standard deviations
    ci_99: ConfidenceInterval              # 3 standard deviations
    
    # Risk metrics derived from simulation
    max_simulated_price: float
    min_simulated_price: float
    
    # Optional: the actual final prices for plotting
    final_prices: Optional[np.ndarray] = None


class MonteCarloEngine:
    """
    Monte Carlo simulator using Geometric Brownian Motion.
    
    Usage:
        >>> engine = MonteCarloEngine(current_price=150.0)
        >>> result = engine.simulate(days=5, drift=0.001, vol=0.02)
        >>> print(result.ci_95)
    """
    
    def __init__(self, current_price: float):
        """
        Initialize the simulator.
        
        Args:
            current_price: The starting stock price.
        """
        self.current_price = current_price
        
    def simulate_gbm(
        self, 
        days: int, 
        drift_daily: float, 
        vol_daily: float, 
        num_paths: int = 10000,
        seed: Optional[int] = None
    ) -> MonteCarloResult:
        """
        Standard GBM simulation with constant volatility.
        
        Formula: S(t) = S(t-1) * exp((μ - σ^2/2) + σ * Z)
        where μ is drift, σ is volatility, Z is random normal.
        
        Args:
            days: Days to simulate ahead
            drift_daily: Expected daily return (decimal, e.g., 0.001 for 0.1%)
            vol_daily: Expected daily volatility (decimal, e.g., 0.02 for 2%)
            num_paths: Number of simulation paths
            seed: Random seed for reproducibility
            
        Returns:
            MonteCarloResult
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Pre-compute the deterministic part of the exponent: (μ - σ^2 / 2)
        drift_adj = drift_daily - (0.5 * vol_daily**2)
        
        # Generate all random shocks at once: array of shape (days, num_paths)
        Z = np.random.standard_normal((days, num_paths))
        
        # Calculate daily returns for all paths
        daily_returns = np.exp(drift_adj + vol_daily * Z)
        
        # Calculate cumulative returns: product along the days axis
        cum_returns = np.cumprod(daily_returns, axis=0)
        
        # The final prices are the starting price * the last cumulative return
        final_prices = self.current_price * cum_returns[-1, :]
        
        return self._process_results(final_prices, days, num_paths)
        
    def simulate_garch(
        self, 
        days: int, 
        drift_daily: float, 
        garch_forecasts: list[float], 
        num_paths: int = 10000,
        seed: Optional[int] = None
    ) -> MonteCarloResult:
        """
        Advanced GBM simulation using dynamic GARCH volatility forecasts.
        
        Args:
            days: Days to simulate ahead
            drift_daily: Expected daily return
            garch_forecasts: List of daily volatility forecasts (in decimals!)
                             from GARCHModel. Must be at least `days` long.
            num_paths: Number of simulation paths
            seed: Random seed
            
        Returns:
            MonteCarloResult
        """
        if len(garch_forecasts) < days:
            logger.warning(f"GARCH forecasts ({len(garch_forecasts)}) < simulation days ({days}). Padding with last value.")
            pad_val = garch_forecasts[-1] if garch_forecasts else 0.02
            garch_forecasts = garch_forecasts + [pad_val] * (days - len(garch_forecasts))
            
        if seed is not None:
            np.random.seed(seed)
            
        # Generate random shocks
        Z = np.random.standard_normal((days, num_paths))
        
        # Initialize paths matrix
        paths = np.zeros((days + 1, num_paths))
        paths[0] = self.current_price
        
        # Simulate step-by-step using dynamic daily volatility
        for t in range(1, days + 1):
            # Use the specific GARCH vol forecast for this day (convert % to decimal if needed)
            # Assuming garch_forecasts are passed as decimals. If passed as %, divide by 100.
            vol_t = garch_forecasts[t-1]
            if vol_t > 1.0: # Safeguard: if passed as percentage like 2.5 instead of 0.025
                vol_t = vol_t / 100.0
                
            drift_adj = drift_daily - (0.5 * vol_t**2)
            paths[t] = paths[t-1] * np.exp(drift_adj + vol_t * Z[t-1])
            
        final_prices = paths[-1]
        
        return self._process_results(final_prices, days, num_paths)

    def _process_results(self, final_prices: np.ndarray, days: int, num_paths: int) -> MonteCarloResult:
        """Process the final array of simulated prices into standard metrics."""
        
        mean_p = float(np.mean(final_prices))
        median_p = float(np.median(final_prices))
        
        # Expected return vs current price
        exp_ret = (mean_p / self.current_price) - 1.0
        
        # Probability of making money
        prob_up = float(np.sum(final_prices > self.current_price) / num_paths)
        
        # Confidence Intervals (Percentiles)
        # 68% CI = 16th to 84th percentile
        # 95% CI = 2.5th to 97.5th percentile
        # 99% CI = 0.5th to 99.5th percentile
        ci_68_low, ci_68_high = np.percentile(final_prices, [16, 84])
        ci_95_low, ci_95_high = np.percentile(final_prices, [2.5, 97.5])
        ci_99_low, ci_99_high = np.percentile(final_prices, [0.5, 99.5])
        
        return MonteCarloResult(
            current_price=self.current_price,
            simulation_days=days,
            num_paths=num_paths,
            mean_price=mean_p,
            median_price=median_p,
            expected_return_pct=exp_ret * 100.0,
            prob_above_current=prob_up * 100.0,
            ci_68=ConfidenceInterval(0.68, float(ci_68_low), float(ci_68_high)),
            ci_95=ConfidenceInterval(0.95, float(ci_95_low), float(ci_95_high)),
            ci_99=ConfidenceInterval(0.99, float(ci_99_low), float(ci_99_high)),
            max_simulated_price=float(np.max(final_prices)),
            min_simulated_price=float(np.min(final_prices)),
            final_prices=final_prices  # Optional, can be removed if memory is an issue
        )
