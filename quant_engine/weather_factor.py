"""
AlphaAgent — Weather Factor for Energy Commodities

Temperature deviation from seasonal norms drives 40-60% of natural gas
price movement and significant variation in heating oil, power, and
agricultural commodities.

Uses NOAA's free public API (no key required) for US population-weighted
heating degree day (HDD) and cooling degree day (CDD) anomalies.

For natural gas:
  - High HDD anomaly (cold) → bullish nat gas (heating demand)
  - High CDD anomaly (hot) → bullish nat gas (power demand)
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WeatherSignal:
    region: str
    current_temp_f: float
    seasonal_avg_f: float
    anomaly_f: float
    hdd_anomaly: float            # heating degree day vs normal
    cdd_anomaly: float            # cooling degree day vs normal
    signal: str                   # "BULLISH_NATGAS" / "BEARISH_NATGAS" / "NEUTRAL"
    direction: str                # "HOT" / "COLD" / "NORMAL"


# Major US population-weighted cities (proxy for US weather)
US_REPRESENTATIVE_CITIES = {
    "NYC":     "USW00094728",
    "Chicago": "USW00094846",
    "Houston": "USW00012960",
    "LA":      "USW00023174",
    "DC":      "USW00013743",
}

# NOAA Climate Data Online API base
NOAA_BASE = "https://www.ncei.noaa.gov/access/services/data/v1"


def _seasonal_normal_temp(month: int) -> float:
    """
    US population-weighted seasonal average temperature (°F).
    Calibrated from 30-year normals.
    """
    seasonal = {
        1: 33.0, 2: 36.0, 3: 44.0, 4: 54.0, 5: 63.0, 6: 72.0,
        7: 76.0, 8: 75.0, 9: 68.0, 10: 56.0, 11: 45.0, 12: 36.0,
    }
    return seasonal.get(month, 55.0)


def get_us_weather_signal() -> Optional[WeatherSignal]:
    """
    Compute US weather anomaly signal for energy/commodity sensitivity.
    Falls back to seasonal average if NOAA fetch fails.
    """
    try:
        import requests
        from datetime import date

        end_date = date.today()
        start_date = end_date - timedelta(days=14)
        params = {
            "dataset": "global-summary-of-the-day",
            "stations": ",".join(US_REPRESENTATIVE_CITIES.values()),
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "format": "json",
            "dataTypes": "TEMP",
        }
        try:
            r = requests.get(NOAA_BASE, params=params, timeout=8)
            if r.status_code == 200:
                data = r.json()
                if data:
                    temps = [float(d.get("TEMP", 0)) for d in data
                             if d.get("TEMP") not in (None, "", "9999.9")]
                    if temps:
                        current_avg = float(np.mean(temps))
                    else:
                        current_avg = _seasonal_normal_temp(end_date.month)
                else:
                    current_avg = _seasonal_normal_temp(end_date.month)
            else:
                current_avg = _seasonal_normal_temp(end_date.month)
        except Exception:
            current_avg = _seasonal_normal_temp(end_date.month)

        seasonal_avg = _seasonal_normal_temp(end_date.month)
        anomaly = current_avg - seasonal_avg
        # HDD / CDD baseline: 65°F
        hdd_curr = max(0.0, 65 - current_avg)
        hdd_norm = max(0.0, 65 - seasonal_avg)
        cdd_curr = max(0.0, current_avg - 65)
        cdd_norm = max(0.0, seasonal_avg - 65)
        hdd_anom = hdd_curr - hdd_norm
        cdd_anom = cdd_curr - cdd_norm

        # Signal: bullish nat gas if HDD anomaly > 3 (extra cold) or CDD anomaly > 3 (extra hot)
        if hdd_anom > 3.0 or cdd_anom > 3.0:
            signal = "BULLISH_NATGAS"
        elif hdd_anom < -3.0 or cdd_anom < -3.0:
            signal = "BEARISH_NATGAS"
        else:
            signal = "NEUTRAL"

        direction = "HOT" if anomaly > 3 else "COLD" if anomaly < -3 else "NORMAL"

        return WeatherSignal(
            region="US_avg",
            current_temp_f=round(current_avg, 1),
            seasonal_avg_f=round(seasonal_avg, 1),
            anomaly_f=round(anomaly, 1),
            hdd_anomaly=round(hdd_anom, 2),
            cdd_anomaly=round(cdd_anom, 2),
            signal=signal,
            direction=direction,
        )
    except Exception as e:
        logger.warning(f"Weather signal failed: {e}")
        return None


_WEATHER_CACHE: dict = {}
_WEATHER_TTL = 6 * 3600   # 6 hours


def get_weather_signal_cached() -> Optional[WeatherSignal]:
    """Cached weather signal — refreshes every 6 hours."""
    cached = _WEATHER_CACHE.get("us")
    if cached and time.time() < cached[1]:
        return cached[0]
    sig = get_us_weather_signal()
    _WEATHER_CACHE["us"] = (sig, time.time() + _WEATHER_TTL)
    return sig
