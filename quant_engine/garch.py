"""
AlphaAgent — GARCH Volatility Forecasting

Implements GARCH(1,1) and EGARCH models for predicting future price
volatility. This is the core of the Volatility Agent.

Key concepts:
  - GARCH predicts TOMORROW'S volatility, not today's
  - High vol = bigger expected moves (both up AND down)
  - EGARCH captures asymmetry: crashes increase vol more than rallies
  - Vol regime classification: LOW / NORMAL / HIGH / EXTREME

Why GARCH matters for trading:
  - Position sizing: lower size when vol is high
  - Stop-loss: wider stops when vol is high
  - Signal confidence: signals during high vol are less reliable
  - Option pricing: IV vs GARCH forecast = variance risk premium
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

from config.settings_manager import settings

logger = logging.getLogger(__name__)


@dataclass
class GARCHResult:
    """Output from GARCH volatility model."""
    
    # Forecasts
    vol_1day: float = 0.0              # 1-day ahead vol forecast (annualized %)
    vol_5day: float = 0.0              # 5-day ahead vol forecast (annualized %)
    vol_10day: float = 0.0             # 10-day ahead vol forecast (annualized %)
    vol_1day_daily: float = 0.0        # 1-day vol (daily %, not annualized)
    
    # Historical comparison
    current_vol: float = 0.0           # Most recent realized vol
    vol_percentile: float = 0.0        # Where current vol sits (0-100)
    vol_regime: str = "NORMAL"         # LOW / NORMAL / HIGH / EXTREME
    
    # Model parameters
    omega: float = 0.0                 # Long-run variance constant
    alpha: float = 0.0                 # News impact (yesterday's shock)
    beta: float = 0.0                  # Persistence (yesterday's variance)
    persistence: float = 0.0           # alpha + beta (should be < 1)
    long_run_vol: float = 0.0          # Unconditional volatility
    
    # Multi-step forecast
    forecast_daily: list = None        # Daily vol forecasts for next N days
    
    # Model quality
    converged: bool = False
    aic: float = 0.0
    bic: float = 0.0
    
    def __post_init__(self):
        if self.forecast_daily is None:
            self.forecast_daily = []


class GARCHModel:
    """
    GARCH(1,1) volatility forecasting model.
    
    Usage:
        >>> model = GARCHModel(returns_series)
        >>> result = model.fit_and_forecast(horizon=5)
        >>> print(f"Tomorrow's vol: {result.vol_1day:.2f}%")
        >>> print(f"Regime: {result.vol_regime}")
    """
    
    @property
    def REGIME_THRESHOLDS(self) -> dict:
        v = settings.get_section("volatility")
        return {
            "LOW":     v.get("regime_low_pct",    25),
            "NORMAL":  v.get("regime_normal_pct", 75),
            "HIGH":    v.get("regime_high_pct",   95),
            "EXTREME": 100,
        }
    
    def __init__(self, returns: pd.Series, scale: float = 100.0):
        """
        Initialize GARCH model.
        
        Args:
            returns: Log return series (as decimal, e.g., 0.02 for 2%)
            scale: Scaling factor for numerical stability (default: 100 = percentage)
        """
        self.raw_returns = returns.dropna()
        self.scale = scale
        self.scaled_returns = self.raw_returns * scale  # arch library works better with %
        self.model = None
        self.fit_result = None
    
    def fit_and_forecast(self, horizon: int = 10) -> GARCHResult:
        """
        Fit GARCH(1,1) and forecast volatility.
        
        Args:
            horizon: Number of days to forecast ahead
            
        Returns:
            GARCHResult with forecasts and model details
        """
        result = GARCHResult()
        
        if len(self.scaled_returns) < 100:
            logger.warning("Insufficient data for GARCH (need 100+ returns)")
            result.vol_regime = "UNKNOWN"
            return result
        
        # ── Fit GARCH(1,1) ──────────────────────────────────
        try:
            self.model = arch_model(
                self.scaled_returns,
                vol="Garch",
                p=1, q=1,
                mean="Constant",
                dist="normal"
            )
            self.fit_result = self.model.fit(disp="off", show_warning=False)
            result.converged = True
            
        except Exception as e:
            logger.error(f"GARCH fit failed: {e}")
            # Fallback: use historical vol
            result.vol_1day_daily = float(self.raw_returns.std())
            result.vol_1day = result.vol_1day_daily * np.sqrt(252) * 100
            result.vol_regime = self._classify_regime_simple()
            return result
        
        # ── Extract Parameters ──────────────────────────────
        params = self.fit_result.params
        result.omega = float(params.get("omega", 0))
        result.alpha = float(params.get("alpha[1]", 0))
        result.beta = float(params.get("beta[1]", 0))
        result.persistence = result.alpha + result.beta
        
        # Long-run (unconditional) volatility
        if result.persistence < 1.0 and result.omega > 0:
            long_run_var = result.omega / (1 - result.persistence)
            result.long_run_vol = float(np.sqrt(long_run_var))  # In scaled units
        
        # Model quality
        result.aic = float(self.fit_result.aic)
        result.bic = float(self.fit_result.bic)
        
        # ── Forecast ────────────────────────────────────────
        try:
            forecast = self.fit_result.forecast(horizon=horizon)
            variance_forecast = forecast.variance.iloc[-1]  # Last row = from latest data
            
            # Convert variance to volatility (daily, in %)
            daily_vols = np.sqrt(variance_forecast.values)
            result.forecast_daily = [float(v) for v in daily_vols]
            
            # Key forecasts (convert from scaled % to actual %)
            if len(daily_vols) >= 1:
                result.vol_1day_daily = float(daily_vols[0] / self.scale)
                result.vol_1day = float(daily_vols[0] / self.scale * np.sqrt(252) * 100)
            
            if len(daily_vols) >= 5:
                # 5-day vol: sqrt of sum of daily variances
                var_5day = sum(v**2 for v in daily_vols[:5])
                result.vol_5day = float(np.sqrt(var_5day) / self.scale * np.sqrt(252/5) * 100)
            
            if len(daily_vols) >= 10:
                var_10day = sum(v**2 for v in daily_vols[:10])
                result.vol_10day = float(np.sqrt(var_10day) / self.scale * np.sqrt(252/10) * 100)
            
        except Exception as e:
            logger.error(f"GARCH forecast failed: {e}")
        
        # ── Current vol + regime ────────────────────────────
        conditional_vol = self.fit_result.conditional_volatility
        result.current_vol = float(conditional_vol.iloc[-1] / self.scale * np.sqrt(252) * 100)
        
        # Percentile rank
        result.vol_percentile = float(
            stats.percentileofscore(conditional_vol.values, conditional_vol.iloc[-1])
        )
        
        # Classify regime
        result.vol_regime = self._classify_regime(result.vol_percentile)
        
        logger.info(
            f"GARCH fit: α={result.alpha:.4f}, β={result.beta:.4f}, "
            f"persistence={result.persistence:.4f} | "
            f"1-day vol: {result.vol_1day:.1f}% (ann.) | "
            f"Regime: {result.vol_regime} ({result.vol_percentile:.0f}th pctl)"
        )
        
        return result
    
    def _classify_regime(self, percentile: float) -> str:
        """Classify vol regime based on percentile."""
        if percentile < self.REGIME_THRESHOLDS["LOW"]:
            return "LOW"
        elif percentile < self.REGIME_THRESHOLDS["NORMAL"]:
            return "NORMAL"
        elif percentile < self.REGIME_THRESHOLDS["HIGH"]:
            return "HIGH"
        else:
            return "EXTREME"
    
    def _classify_regime_simple(self) -> str:
        """Fallback regime classification using simple historical vol."""
        recent_vol = self.raw_returns[-20:].std() * np.sqrt(252)
        full_vol = self.raw_returns.std() * np.sqrt(252)
        ratio = recent_vol / full_vol if full_vol > 0 else 1.0
        
        if ratio < 0.7:
            return "LOW"
        elif ratio < 1.3:
            return "NORMAL"
        elif ratio < 2.0:
            return "HIGH"
        else:
            return "EXTREME"
    
    def get_conditional_volatility(self) -> Optional[pd.Series]:
        """Get the full conditional volatility series (for plotting)."""
        if self.fit_result is None:
            return None
        return self.fit_result.conditional_volatility / self.scale * np.sqrt(252) * 100
