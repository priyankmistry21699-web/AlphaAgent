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
            
    def calculate(self, current_volatility: float = 0.0) -> KellyResult:
        """
        Calculates the exact percentage of the portfolio to risk.
        
        Args:
            current_volatility: Optional volatility metric to further reduce the size.
        """
        # Full Kelly Formula: K = W - ((1 - W) / R)
        # where W is prob_win and R is win_loss_ratio
        
        full_k = self.prob_win - (self.prob_loss / self.win_loss_ratio)
        
        # If Full Kelly is negative, we shouldn't take the trade (expectation is negative)
        full_k = max(0.0, full_k)
        
        # Institutional standard: Half-Kelly (less drawdown, safer compounding)
        half_k = full_k / 2.0
        
        # Volatility adjusted
        # If market is extremely volatile (e.g., VIX > 30), reduce size further
        vol_adjusted_k = half_k
        if current_volatility > 0.03: # >3% daily vol is very high
            vol_adjusted_k *= (0.03 / current_volatility)
            
        # Hard cap at 30% of portfolio to prevent catastrophic ruin on black swans
        full_k = min(0.30, full_k)
        half_k = min(0.15, half_k)
        vol_adjusted_k = min(0.15, vol_adjusted_k)
        
        return KellyResult(
            full_kelly=full_k,
            half_kelly=half_k,
            vol_adjusted_kelly=vol_adjusted_k
        )
