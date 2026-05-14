"""
AlphaAgent — Bayesian Fusion Engine

Instead of simply averaging probabilities, this module uses Bayesian 
Updating to fuse probabilities from multiple independent agents, 
penalizing highly correlated agents to avoid confirmation bias.
"""

import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BayesianResult:
    prior: float
    posterior: float
    entropy: float


class BayesianFusion:
    """
    Fuses multiple probability estimates into a single posterior probability.
    """
    
    def __init__(self, prior: float = 0.5):
        """
        Args:
            prior: The initial belief before any agent speaks (0.5 = totally neutral).
        """
        # Constrain prior between 0.01 and 0.99 to prevent log(0)
        self.prior = min(0.99, max(0.01, prior))
        
        # We store beliefs in log-odds form: log(p / (1-p))
        self.log_odds_posterior = np.log(self.prior / (1.0 - self.prior))
        
    def update(self, agent_prob: float, correlation: float = 0.0, confidence: float = 1.0) -> None:
        """
        Updates the internal posterior probability using a new piece of evidence.
        
        Args:
            agent_prob: The probability returned by the agent (0.0 to 1.0)
            correlation: How correlated this agent is to previous evidence (0.0 = independent, 1.0 = same info)
            confidence: How confident the agent is in this specific prediction (0.0 to 1.0)
        """
        if agent_prob is None or np.isnan(agent_prob):
            return
            
        agent_prob = min(0.99, max(0.01, agent_prob))
        
        # Calculate the "Bayes Factor" (the evidence)
        evidence_log_odds = np.log(agent_prob / (1.0 - agent_prob))
        
        # If confidence is low, we shrink the evidence towards 0 (neutral)
        evidence_log_odds *= confidence
        
        # If this agent uses the same data as another agent (high correlation),
        # we penalize the evidence so we don't double-count it.
        # e.g., if correlation = 0.8, we only add 20% of the evidence's weight.
        penalty_factor = 1.0 - correlation
        
        # Add the weighted evidence to our posterior
        self.log_odds_posterior += (evidence_log_odds * penalty_factor)
        
    @property
    def posterior(self) -> float:
        """
        Converts the internal log-odds back into a standard probability (0 to 1).
        """
        # Inverse log-odds (sigmoid function)
        p = 1.0 / (1.0 + np.exp(-self.log_odds_posterior))
        return float(p)
        
    @property
    def entropy(self) -> float:
        """
        Calculates Shannon Entropy (uncertainty). 
        0.0 = Absolute certainty, 1.0 = Total confusion.
        """
        p = self.posterior
        if p <= 0 or p >= 1:
            return 0.0
        
        # Binary Cross Entropy
        ent = - (p * np.log2(p) + (1 - p) * np.log2(1 - p))
        return float(ent)
