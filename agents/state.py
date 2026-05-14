"""
AlphaAgent — Pydantic State Schemas

Core data models that flow between agents. Every agent reads and writes
to these schemas, ensuring type safety and consistent inter-agent communication.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────

class Direction(str, Enum):
    """Signal direction output."""
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"


class Confidence(str, Enum):
    """Confidence level based on agent agreement (entropy)."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Regime(str, Enum):
    """Market regime detected by HMM."""
    BULL = "BULL"
    BEAR = "BEAR"
    CRISIS = "CRISIS"
    SIDEWAYS = "SIDEWAYS"


class VolRegime(str, Enum):
    """Volatility regime classification."""
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


# ─── Agent Output Schema ─────────────────────────────────────────────────────

class FactorScore(BaseModel):
    """Individual factor score within an agent."""
    name: str
    value: float                           # Raw computed value
    score: float = Field(ge=0, le=100)     # Normalized 0-100 score
    interpretation: str = ""               # Human-readable meaning


class AgentResult(BaseModel):
    """Standard output from every analysis agent."""
    agent_name: str
    vote: Direction = Direction.HOLD                   # Directional vote
    probability_up: float = Field(ge=0.0, le=1.0)    # P(stock goes up)
    confidence: float = Field(ge=0.0, le=1.0)         # How sure the agent is
    reasoning: str                                      # LLM-generated explanation
    factor_scores: dict[str, FactorScore] = {}         # Individual factor details
    warnings: list[str] = []                            # Risk flags
    computation_time_ms: float = 0.0                   # Performance tracking
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ─── Volatility Agent Output (special — doesn't predict direction) ──────────

class VolatilityResult(BaseModel):
    """Output from the Volatility Agent (predicts size of move, not direction)."""
    agent_name: str = "volatility"
    vol_1day: float                        # GARCH 1-day forecast (%)
    vol_5day: float                        # GARCH 5-day forecast (%)
    iv_vs_rv: float                        # Implied / Realized vol ratio
    vol_regime: VolRegime                  # Current vol regime
    beta: float                            # Dynamic beta to SPY
    spy_correlation: float                 # Correlation with market
    reasoning: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ─── Monte Carlo Output ──────────────────────────────────────────────────────

class ConfidenceInterval(BaseModel):
    """Price confidence interval from Monte Carlo simulation."""
    low: float
    high: float
    level: float                           # e.g., 0.68, 0.95, 0.99


class MonteCarloResult(BaseModel):
    """Output from Monte Carlo price simulation."""
    current_price: float
    simulation_days: int
    num_paths: int
    mean_price: float
    median_price: float
    ci_68: ConfidenceInterval
    ci_95: ConfidenceInterval
    ci_99: ConfidenceInterval
    prob_above_current: float              # % of paths ending above current price
    expected_return_pct: float             # Mean expected return


# ─── Regime Detection Output ─────────────────────────────────────────────────

class RegimeResult(BaseModel):
    """Output from HMM regime detection."""
    current_regime: Regime
    regime_probabilities: dict[str, float]  # {"bull": 0.78, "bear": 0.16, ...}
    transition_risk: dict[str, float]       # {"to_bear": 0.05, "to_crisis": 0.01}
    regime_duration_days: int               # How long in current regime


# ─── Risk Metrics ─────────────────────────────────────────────────────────────

class RiskMetrics(BaseModel):
    """Risk metrics computed by EVT and Kelly."""
    var_95: float                          # Value at Risk (95%)
    var_99: float                          # Value at Risk (99%)
    cvar_99: float                         # Conditional VaR / Expected Shortfall
    max_drawdown_estimate: float           # Estimated max drawdown
    kelly_fraction: float                  # Kelly optimal fraction
    half_kelly_fraction: float             # Conservative half-Kelly
    position_size_pct: float               # Recommended position size (% of portfolio)
    stop_loss_pct: float                   # EVT-derived stop loss
    take_profit_pct: float                 # Monte Carlo-derived target


# ─── Signal Decay / Holding Period ────────────────────────────────────────────

class HoldingPeriod(BaseModel):
    """Signal timing analysis."""
    half_life_days: float                  # Signal decays to 50% strength
    optimal_hold_min: int                  # Minimum hold (days)
    optimal_hold_max: int                  # Maximum hold (days)
    signal_strength: float = Field(ge=0, le=1)  # Current signal strength


# ─── The Final Signal Packet ──────────────────────────────────────────────────

class SignalPacket(BaseModel):
    """
    The FINAL output of the entire AlphaAgent pipeline.
    This is what the user sees. Every field is computed, not guessed.
    """
    # Core signal
    ticker: str
    direction: Direction
    conviction_pct: float = Field(ge=0, le=100)    # Bayesian posterior × 100
    confidence: Confidence                          # Agent agreement level
    
    # Agent breakdown
    agent_results: list[AgentResult] = []           # Each agent's individual output
    volatility_result: Optional[VolatilityResult] = None
    
    # Expected move
    monte_carlo: Optional[MonteCarloResult] = None
    
    # Regime
    regime: Optional[RegimeResult] = None
    
    # Risk
    risk_metrics: Optional[RiskMetrics] = None
    
    # Timing
    holding_period: Optional[HoldingPeriod] = None
    
    # Override flags
    override_active: bool = False
    override_reason: str = ""
    
    # Warnings
    warnings: list[str] = []
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    computation_time_ms: float = 0.0
    factors_computed: int = 0
    agents_used: int = 0
