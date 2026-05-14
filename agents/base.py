"""
AlphaAgent — Base Agent Interface

Defines the abstract BaseAgent class that every specific agent must inherit from.
This ensures that all agents, no matter how different their internal logic,
communicate using the exact same standard interface (`AgentResult`).
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from agents.state import AgentResult, Direction, Confidence

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all AlphaAgent analysis agents.
    
    Every agent must implement the `_run_analysis` method, which takes 
    in raw market data and returns an `AgentResult`.
    """
    
    # Define the name of the agent in the subclass (e.g., "technical")
    name: str = "base"
    
    def analyze(self, ticker: str, data: Any, **kwargs) -> AgentResult:
        """
        Public entry point for agent execution.
        Wraps the internal analysis with error handling and timing.
        
        Args:
            ticker: Stock symbol being analyzed
            data: MarketData instance or specific data dictionary
            
        Returns:
            AgentResult (Standardized Pydantic schema)
        """
        logger.info(f"[{ticker}] Starting {self.name.capitalize()} Agent analysis")
        start_time = time.time()
        
        try:
            # ── Execute the specific agent's logic ──
            result = self._run_analysis(ticker, data, **kwargs)
            
            # ── Validation & Formatting ──
            # Ensure the agent name is set correctly
            result.agent_name = self.name
            
            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000
            result.computation_time_ms = execution_time_ms
            
            logger.info(
                f"[{ticker}] {self.name.capitalize()} Agent finished in {execution_time_ms:.1f}ms "
                f"| P(Up): {result.probability_up:.2f} | Confidence: {result.confidence:.2f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"[{ticker}] {self.name.capitalize()} Agent failed: {e}", exc_info=True)
            
            # ── Fail gracefully ──
            # If an agent crashes, it returns a 50% probability (neutral)
            # with 0% confidence, so it doesn't break the whole pipeline.
            return AgentResult(
                agent_name=self.name,
                vote=Direction.HOLD,
                probability_up=0.5,
                confidence=0.0,
                reasoning=f"AGENT FAILURE: {str(e)}",
                warnings=[f"Agent crashed during analysis"],
                computation_time_ms=(time.time() - start_time) * 1000
            )

    @abstractmethod
    def _run_analysis(self, ticker: str, data: Any, **kwargs) -> AgentResult:
        """
        Internal method to be implemented by subclasses.
        This is where the actual math/logic happens.
        """
        pass
    
    def _build_error_result(self, reason: str) -> "AgentResult":
        """
        Return a neutral (50%) result when input data is unavailable.
        Prevents agent failures from crashing the whole pipeline.
        """
        return AgentResult(
            agent_name=self.name,
            probability_up=0.5,
            confidence=0.0,
            reasoning=f"Data unavailable: {reason}",
            warnings=[reason],
        )

    def _map_score_to_probability(self, score: float, min_val: float, max_val: float) -> float:
        """
        Helper method to map a raw score to a 0.0 - 1.0 probability.
        
        Args:
            score: Raw calculated score
            min_val: Minimum possible raw score (maps to 0.0)
            max_val: Maximum possible raw score (maps to 1.0)
            
        Returns:
            Probability between 0.0 and 1.0
        """
        # Clamp score to bounds
        score = max(min_val, min(max_val, score))
        
        # Linear map to 0-1
        prob = (score - min_val) / (max_val - min_val)
        return float(prob)
