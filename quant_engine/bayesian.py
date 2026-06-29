"""
AlphaAgent — Bayesian Fusion Engine

Instead of simply averaging probabilities, this module uses Bayesian
Updating to fuse probabilities from multiple independent agents,
penalizing highly correlated agents to avoid confirmation bias.
Tracks convergence so each agent's contribution is visible.
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Tuple

logger = logging.getLogger(__name__)

@dataclass
class BayesianResult:
    prior: float
    posterior: float
    entropy: float


@dataclass
class CouncilStep:
    agent: str
    prob_before: float
    prob_after: float
    agent_prob: float
    weight: float        # effective weight after corr + confidence penalties


class BayesianFusion:
    """
    Fuses multiple probability estimates into a single posterior probability.
    Tracks each agent's contribution as a "council convergence" path.
    """

    def __init__(self, prior: float = 0.5, sensitivity: float = 1.4):
        self.prior = min(0.99, max(0.01, prior))
        self.log_odds_posterior = np.log(self.prior / (1.0 - self.prior))
        self.council: List[CouncilStep] = []
        # sensitivity > 1 amplifies weak signals so near-neutral agents (P≈0.52)
        # still move the posterior meaningfully. 1.4 = 40% amplification.
        self.sensitivity = sensitivity

    def update(self, agent_prob: float, correlation: float = 0.0,
               confidence: float = 1.0, agent_name: str = "") -> None:
        if agent_prob is None or np.isnan(agent_prob):
            return

        prob_before = self.posterior
        agent_prob = min(0.99, max(0.01, agent_prob))

        # Extract likelihood ratio relative to the prior so that agents sharing
        # the same prior don't double-count evidence already in the posterior.
        prior_log_odds = np.log(self.prior / (1.0 - self.prior))
        evidence_log_odds = np.log(agent_prob / (1.0 - agent_prob)) - prior_log_odds
        evidence_log_odds *= confidence * self.sensitivity
        penalty_factor = 1.0 - correlation
        effective_weight = confidence * penalty_factor

        self.log_odds_posterior += (evidence_log_odds * penalty_factor)

        self.council.append(CouncilStep(
            agent=agent_name,
            prob_before=prob_before,
            prob_after=self.posterior,
            agent_prob=agent_prob,
            weight=effective_weight,
        ))

    @property
    def posterior(self) -> float:
        p = 1.0 / (1.0 + np.exp(-self.log_odds_posterior))
        return float(p)

    @property
    def entropy(self) -> float:
        p = self.posterior
        if p <= 0 or p >= 1:
            return 0.0
        ent = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        return float(ent)

    def council_summary(self) -> List[dict]:
        """Returns council convergence steps for API serialization."""
        return [
            {
                "agent": s.agent,
                "agent_prob": round(s.agent_prob, 4),
                "prob_before": round(s.prob_before, 4),
                "prob_after": round(s.prob_after, 4),
                "delta": round(s.prob_after - s.prob_before, 4),
                "weight": round(s.weight, 3),
            }
            for s in self.council
        ]

    def agreement_score(self) -> float:
        """0 = all agents in perfect agreement, 1 = maximum disagreement."""
        if len(self.council) < 2:
            return 0.0
        probs = [s.agent_prob for s in self.council]
        return float(np.std(probs) * 2)  # normalized to ~0-1 range
