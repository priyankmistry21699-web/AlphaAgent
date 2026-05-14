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
    yc = snapshot.get("yield_curve", 1.0)
    ffr = snapshot.get("fed_funds_rate", 0.0)
    vix = snapshot.get("vix", 15.0)
    unemp = snapshot.get("unemployment", 4.0)
    
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
        
    # 2. Unemployment Rate
    if unemp > 5.0:
        recession_prob += 0.2
        warnings.append(f"Unemployment is high ({unemp}%).")
    if unemp > 6.0:
        recession_prob += 0.15
        
    # 3. Interest Rates (Fed Funds)
    # Extremely high rates choke the economy
    if ffr > 5.0:
        recession_prob += 0.15
        warnings.append(f"Fed Funds Rate is highly restrictive ({ffr}%).")
        
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
    elif yc > 1.0 and unemp < 5.0 and ffr < 3.0:
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
