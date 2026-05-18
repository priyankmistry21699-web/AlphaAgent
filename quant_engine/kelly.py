"""
AlphaAgent — Kelly Criterion Engine

Calculates optimal position sizing based on the probability of success,
the win/loss ratio, and volatility constraints.
"""

from dataclasses import dataclass

@dataclass
class KellyResult:
    full_kelly: float
    half_kelly: float
    vol_adjusted_kelly: float


class KellyCriterion:
    """
    Computes position sizes that maximize the long-term growth rate of capital.
    """
    
    def __init__(self, prob_win: float, expected_win_pct: float, expected_loss_pct: float):
        """
        Args:
            prob_win: Probability of winning the trade (e.g., 0.65 from our agents)
            expected_win_pct: The average expected gain if right (e.g., 0.05 for 5%)
            expected_loss_pct: The average expected loss if wrong (e.g., 0.02 for 2%). Must be positive.
        """
        self.prob_win = min(0.99, max(0.01, prob_win))
        self.prob_loss = 1.0 - self.prob_win
        
        # Win/Loss Ratio (b)
        # If I risk $1 and win $2, b = 2.
        if expected_loss_pct <= 0:
            self.win_loss_ratio = 1.0 # Safe default
        else:
            self.win_loss_ratio = expected_win_pct / expected_loss_pct
            
    # Caps adapt to GARCH vol regime — tighter in high-vol to prevent ruin
    _CAPS = {
        "LOW":     (0.20, 0.10, 0.10),
        "NORMAL":  (0.15, 0.08, 0.08),
        "HIGH":    (0.10, 0.05, 0.05),
        "EXTREME": (0.05, 0.03, 0.03),
    }

    def calculate(self, current_volatility: float = 0.0,
                  vol_regime: str = "NORMAL") -> KellyResult:
        """
        Calculates the exact percentage of the portfolio to risk.

        Args:
            current_volatility: Daily vol metric (fraction, e.g. 0.02 = 2%).
            vol_regime: GARCH regime string — LOW / NORMAL / HIGH / EXTREME.
        """
        full_k = self.prob_win - (self.prob_loss / self.win_loss_ratio)
        full_k = max(0.0, full_k)

        half_k = full_k / 2.0

        vol_adjusted_k = half_k
        if current_volatility > 0.03:
            vol_adjusted_k *= (0.03 / current_volatility)

        cap_full, cap_half, cap_vol = self._CAPS.get(
            vol_regime.upper(), self._CAPS["NORMAL"]
        )
        full_k         = min(cap_full, full_k)
        half_k         = min(cap_half, half_k)
        vol_adjusted_k = min(cap_vol,  vol_adjusted_k)

        return KellyResult(
            full_kelly=full_k,
            half_kelly=half_k,
            vol_adjusted_kelly=vol_adjusted_k,
        )
