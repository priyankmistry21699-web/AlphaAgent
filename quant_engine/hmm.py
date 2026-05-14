"""
AlphaAgent — Hidden Markov Model (HMM) Regime Detection

Uses unsupervised machine learning to classify the market into hidden 
"regimes" (e.g., Bull, Bear, Crisis) based on price returns and volatility.
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import logging
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass
class RegimeResult:
    current_regime: str
    probabilities: Dict[str, float]
    transition_risk: Dict[str, float]
    state_means: list
    state_vols: list
    regime_duration_days: int = 0


class RegimeDetector:
    """
    Fits a 3-state Gaussian HMM to historical returns to identify the current market environment.
    """
    
    def __init__(self, n_states: int = 3):
        self.n_states = n_states
        # We use full covariance and 100 iterations for convergence
        self.model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
        
    def fit_predict(self, returns: pd.Series) -> RegimeResult:
        """
        Fits the HMM and predicts the regime of the last data point.
        """
        # Drop NaNs and ensure we have enough data
        returns = returns.dropna()
        if len(returns) < 100:
            logger.warning("Not enough data to fit HMM safely. Defaulting to NORMAL regime.")
            return self._default_result()
            
        # HMM requires 2D array of shape (n_samples, n_features)
        X = returns.values.reshape(-1, 1)
        
        try:
            # Fit the model
            self.model.fit(X)
            
            # Predict the hidden states for the entire series
            hidden_states = self.model.predict(X)
            
            # Get the state probabilities for the very last day (today)
            posterior_probs = self.model.predict_proba(X)[-1]
            
            # Identify which hidden state corresponds to Bull, Bear, or Crisis
            # We do this by looking at the variance (volatility) and mean of each state.
            means = self.model.means_.flatten()
            vols = np.sqrt(self.model.covars_.flatten())
            
            # Sort states by volatility (lowest to highest)
            # State 0 (lowest vol) = usually Bull / Normal Expansion
            # State 1 (medium vol) = usually Bear / High Volatility
            # State 2 (highest vol) = usually Crisis / Panic
            sorted_indices = np.argsort(vols)
            bull_idx = sorted_indices[0]
            bear_idx = sorted_indices[1]
            crisis_idx = sorted_indices[2]
            
            # Map probabilities
            probs = {
                "bull": float(posterior_probs[bull_idx]),
                "bear": float(posterior_probs[bear_idx]),
                "crisis": float(posterior_probs[crisis_idx])
            }
            
            # Determine current regime
            current_state_idx = hidden_states[-1]
            if current_state_idx == bull_idx:
                current_regime = "BULL"
            elif current_state_idx == bear_idx:
                current_regime = "BEAR"
            else:
                current_regime = "CRISIS"

            # Count consecutive days in the current regime (duration)
            duration = 1
            for i in range(len(hidden_states) - 2, -1, -1):
                if hidden_states[i] == current_state_idx:
                    duration += 1
                else:
                    break

            # Transition Matrix Risk (probability of moving to crisis tomorrow)
            trans_mat = self.model.transmat_
            transition_risk = {
                "to_bear": float(trans_mat[current_state_idx, bear_idx]),
                "to_crisis": float(trans_mat[current_state_idx, crisis_idx])
            }

            return RegimeResult(
                current_regime=current_regime,
                probabilities=probs,
                transition_risk=transition_risk,
                state_means=means[sorted_indices].tolist(),
                state_vols=vols[sorted_indices].tolist(),
                regime_duration_days=duration,
            )
            
        except Exception as e:
            logger.error(f"HMM fitting failed: {e}")
            return self._default_result()

    def _default_result(self) -> RegimeResult:
        return RegimeResult(
            current_regime="BULL",
            probabilities={"bull": 0.5, "bear": 0.4, "crisis": 0.1},
            transition_risk={"to_bear": 0.0, "to_crisis": 0.0},
            state_means=[0.0, 0.0, 0.0],
            state_vols=[0.01, 0.02, 0.05]
        )
