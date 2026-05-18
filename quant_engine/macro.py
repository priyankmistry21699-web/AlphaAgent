"""
AlphaAgent — Macro Math Engine

Calculates recession probability based on the Federal Reserve Economic Data (FRED).
"""

from dataclasses import dataclass

@dataclass
class MacroResult:
    recession_probability: float  # 0.0 to 1.0
    yield_curve: float
    fed_funds_rate: float
    vix: float
    unemployment: float
    regime: str                   # "EXPANSION", "SLOWDOWN", "RECESSION", "RECOVERY"
    warnings: list[str]


def analyze_macro_environment(snapshot: dict) -> MacroResult:
    """
    Takes a snapshot of FRED data and calculates recession risk.
    """
    yc    = snapshot.get("yield_curve", 1.0)
    ffr   = snapshot.get("fed_funds_rate", 0.0)
    vix   = snapshot.get("vix", 15.0)
    unemp = snapshot.get("unemployment", 4.0)

    # Dynamic benchmarks — fetched from FRED via data/macro.py
    # nrou:         CBO natural rate of unemployment (FRED: NROU), fallback 5.0
    # dfii10:       10Y TIPS real yield = market's estimate of real r* (FRED: DFII10), fallback 0.5
    # r_star_nominal: nominal neutral rate = real r* + 2% inflation target
    nrou          = snapshot.get("nrou", 5.0)
    dfii10        = snapshot.get("dfii10", 0.5)
    r_star_nominal = dfii10 + 2.0   # nominal neutral rate

    warnings = []
    recession_prob = 0.0

    # 1. Yield Curve Inversion (The strongest predictor of a recession)
    # If the 10Y-2Y is negative, the curve is inverted.
    if yc < 0:
        recession_prob += 0.4
        warnings.append(f"Yield Curve is INVERTED ({yc:.2f}). Severe recession warning.")
    elif yc < 0.2:
        recession_prob += 0.2
        warnings.append(f"Yield Curve is dangerously flat ({yc:.2f}).")

    # 2. Unemployment Rate — compared against live NAIRU (FRED NROU), not fixed 5.0
    if unemp > nrou:
        recession_prob += 0.2
        warnings.append(f"Unemployment {unemp:.1f}% above NAIRU {nrou:.1f}%.")
    if unemp > nrou + 1.0:
        recession_prob += 0.15

    # 3. Interest Rates — restrictive when FFR > r* + 100bps (Taylor Rule margin)
    if ffr > r_star_nominal + 1.0:
        recession_prob += 0.15
        warnings.append(f"Fed Funds {ffr:.2f}% highly restrictive vs r* {r_star_nominal:.2f}%.")

    # 4. VIX (Market Fear)
    if vix > 25.0:
        recession_prob += 0.1
        warnings.append(f"VIX shows elevated market fear ({vix}).")

    recession_prob = min(1.0, recession_prob)

    # Determine Regime
    if recession_prob > 0.65:
        regime = "RECESSION"
    elif recession_prob > 0.40:
        regime = "SLOWDOWN"
    elif yc > 1.0 and unemp < nrou and ffr < r_star_nominal:
        regime = "RECOVERY"
    else:
        regime = "EXPANSION"
        
    return MacroResult(
        recession_probability=recession_prob,
        yield_curve=yc,
        fed_funds_rate=ffr,
        vix=vix,
        unemployment=unemp,
        regime=regime,
        warnings=warnings
    )
