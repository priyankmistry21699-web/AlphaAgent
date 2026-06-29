"""
AlphaAgent — FRED Macro Nowcasting

Composite "GDPNow-style" nowcasting indicator combining high-frequency
FRED data into a single real-time growth/inflation gauge.

Atlanta Fed GDPNow does this for GDP using ~13 monthly indicators. Our
version uses a smaller free FRED subset to produce a leading composite.

Inputs (all from FRED):
  - Initial jobless claims (weekly)
  - Industrial production (monthly)
  - Retail sales (monthly)
  - Building permits (monthly)
  - ISM manufacturing PMI (monthly)
  - 10Y-2Y yield spread (daily)

Output: composite z-score where positive = expansion accelerating,
negative = expansion decelerating.
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class NowcastResult:
    composite_z: float                # composite leading indicator z-score
    components: Dict[str, float]      # individual z-scores
    direction: str                    # "ACCELERATING" / "DECELERATING" / "STABLE"
    n_components: int
    interpretation: str


# FRED series IDs for the composite
NOWCAST_SERIES = {
    "ICSA":      ("Initial Jobless Claims",   -1.0),  # inverted (high claims = bad)
    "INDPRO":    ("Industrial Production",     1.0),
    "RSAFS":     ("Retail Sales",              1.0),
    "PERMIT":    ("Building Permits",          1.0),
    "T10Y2Y":    ("Yield Curve 10Y-2Y",        1.0),
    "UMCSENT":   ("Consumer Sentiment",        1.0),
}


def compute_nowcast(macro_data) -> Optional[NowcastResult]:
    """
    Compute composite nowcast from FRED data.

    macro_data : MacroData instance (must have .get_series method)
    """
    try:
        components = {}
        for sid, (name, sign) in NOWCAST_SERIES.items():
            try:
                s = macro_data.get_series(sid, years_back=3)
                if s is None or len(s) < 13:
                    continue
                series = s.iloc[:, 0] if hasattr(s, "shape") and len(s.shape) > 1 else s
                series = series.dropna()
                if len(series) < 13:
                    continue
                # z-score the most recent observation vs 3y distribution
                current = float(series.iloc[-1])
                mu = float(series.mean())
                sigma = float(series.std())
                if sigma > 0:
                    z = (current - mu) / sigma * sign
                    components[sid] = round(z, 3)
            except Exception:
                continue

        if not components:
            return None

        composite_z = float(np.mean(list(components.values())))

        if composite_z > 0.5:
            direction = "ACCELERATING"
            interpretation = "Expansion accelerating — risk-on environment"
        elif composite_z < -0.5:
            direction = "DECELERATING"
            interpretation = "Growth slowing — defensive posture warranted"
        else:
            direction = "STABLE"
            interpretation = "No clear nowcast signal"

        return NowcastResult(
            composite_z=round(composite_z, 3),
            components=components,
            direction=direction,
            n_components=len(components),
            interpretation=interpretation,
        )
    except Exception as e:
        logger.warning(f"Nowcast failed: {e}")
        return None


_NOWCAST_CACHE: dict = {}
_NOWCAST_TTL = 4 * 3600   # 4 hours


def get_nowcast_cached(macro_data) -> Optional[NowcastResult]:
    cached = _NOWCAST_CACHE.get("default")
    if cached and time.time() < cached[1]:
        return cached[0]
    r = compute_nowcast(macro_data)
    _NOWCAST_CACHE["default"] = (r, time.time() + _NOWCAST_TTL)
    return r
