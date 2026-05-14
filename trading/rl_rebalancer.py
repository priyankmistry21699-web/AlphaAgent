"""
AlphaAgent — Reinforcement Learning Portfolio Rebalancer

Trains a PPO (Proximal Policy Optimization) agent to make continuous
portfolio rebalancing decisions. The RL agent learns to balance:
  - Return maximization (reward signal)
  - Risk control (Sharpe-adjusted reward)
  - Transaction cost minimization (turnover penalty)

Environment specification:
  State space  (Box): [portfolio_weights(n), vol_regime_onehot(4),
                        normalized_factor_scores(5), days_since_rebalance]
  Action space (Box): delta_weights(n) ∈ [-max_shift, +max_shift]
                       (relative reallocation, automatically normalized to sum=0)
  Reward       : Sharpe contribution from the period minus turnover cost
  Episode      : One full historical year of trading days

Uses stable-baselines3 PPO with MlpPolicy. Falls back to a rule-based
Mean-Variance optimizer (Markowitz) when stable-baselines3 is unavailable.

Training:
    rebalancer = RLRebalancer(tickers=["AAPL", "MSFT", "NVDA"])
    rebalancer.train(total_timesteps=50_000)
    rebalancer.save("models/rl_rebalancer.zip")

Inference:
    weights = rebalancer.get_weights(current_state)
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODEL_DIR = _PROJECT_ROOT / "models"

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    _HAS_SB3 = True
except ImportError:
    _HAS_SB3 = False
    logger.info("[RL] stable-baselines3 / gymnasium not installed — using rule-based fallback.")


@dataclass
class RLRebalanceResult:
    ticker_weights: Dict[str, float]     # Recommended portfolio weights
    method: str                          # "PPO" | "MEAN_VARIANCE" | "EQUAL_WEIGHT"
    expected_sharpe: float               # Estimated forward Sharpe from model
    turnover_pct: float                  # % weight change from current allocation
    rebalance_needed: bool               # Whether turnover exceeds threshold
    reasoning: str
    warnings: List[str] = field(default_factory=list)


# ── Gymnasium Environment ─────────────────────────────────────────────────────

if _HAS_SB3:
    class PortfolioEnv(gym.Env):
        """
        Custom gymnasium environment for portfolio weight optimization.

        State: [w_1..w_n, vol_onehot_4, factor_5, days_held] → dim = n+10
        Action: [δw_1..δw_n] continuous weight adjustments in [-0.20, +0.20]
        Reward: Sharpe contribution per step minus turnover penalty
        """

        metadata = {"render_modes": []}

        def __init__(
            self,
            returns: pd.DataFrame,    # shape (T, n_assets) daily returns
            initial_weights: Optional[np.ndarray] = None,
            lookback: int = 20,       # rolling window for Sharpe computation
            max_shift: float = 0.20,  # max per-asset weight change per step
            tc_bps: float = 5.0,      # transaction cost in basis points
        ):
            super().__init__()
            self.returns = returns.fillna(0.0).values  # (T, n)
            self.n_assets = self.returns.shape[1]
            self.lookback = lookback
            self.max_shift = max_shift
            self.tc = tc_bps / 10_000

            # State: weights(n) + vol_regime_onehot(4) + factor_scores(5) + days_held(1)
            obs_dim = self.n_assets + 10
            self.observation_space = spaces.Box(
                low=-2.0, high=2.0, shape=(obs_dim,), dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=-max_shift, high=max_shift,
                shape=(self.n_assets,), dtype=np.float32,
            )

            if initial_weights is not None:
                self._init_weights = initial_weights.copy()
            else:
                self._init_weights = np.ones(self.n_assets) / self.n_assets

            self._weights = self._init_weights.copy()
            self._step = lookback
            self._days_held = 0
            self._recent_returns: List[float] = []

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self._weights = self._init_weights.copy()
            self._step = self.lookback
            self._days_held = 0
            self._recent_returns = []
            return self._get_obs(), {}

        def step(self, action: np.ndarray):
            old_weights = self._weights.copy()

            # Apply weight shift and re-normalize to simplex
            new_weights = old_weights + action
            new_weights = np.clip(new_weights, 0.0, 1.0)
            total = new_weights.sum()
            if total < 1e-6:
                new_weights = old_weights
            else:
                new_weights = new_weights / total
            self._weights = new_weights

            # Portfolio return for this step
            if self._step < len(self.returns):
                daily_ret = float(np.dot(self._weights, self.returns[self._step]))
            else:
                daily_ret = 0.0

            # Transaction cost from turnover
            turnover = float(np.sum(np.abs(new_weights - old_weights)))
            cost = turnover * self.tc

            net_ret = daily_ret - cost
            self._recent_returns.append(net_ret)
            if len(self._recent_returns) > self.lookback:
                self._recent_returns.pop(0)

            # Sharpe-based reward (rolling)
            if len(self._recent_returns) >= 5:
                r = np.array(self._recent_returns)
                reward = float(r.mean() / (r.std() + 1e-8))
            else:
                reward = float(net_ret)

            self._step += 1
            self._days_held += 1
            terminated = self._step >= len(self.returns) - 1
            return self._get_obs(), reward, terminated, False, {}

        def _get_obs(self) -> np.ndarray:
            obs = np.zeros(self.n_assets + 10, dtype=np.float32)
            obs[:self.n_assets] = self._weights.astype(np.float32)
            # Vol regime placeholder (LOW=0, NORMAL=1, HIGH=2, EXTREME=3)
            obs[self.n_assets] = 1.0    # default NORMAL
            # Factor scores placeholder (all neutral)
            obs[self.n_assets + 5] = float(np.clip(self._days_held / 60.0, 0, 1))
            return obs


# ── Main Rebalancer Class ─────────────────────────────────────────────────────

class RLRebalancer:
    """
    Trains and runs a PPO agent for portfolio weight optimization.

    When stable-baselines3 is not installed or no model is loaded,
    falls back to minimum-variance (Markowitz) rebalancing.

    Usage:
        rb = RLRebalancer(tickers=["AAPL", "MSFT", "NVDA"])
        rb.train(returns_df, total_timesteps=50_000)
        rb.save()

        result = rb.optimize(
            current_weights={"AAPL": 0.4, "MSFT": 0.4, "NVDA": 0.2},
            returns_df=recent_returns,
        )
    """

    MIN_TURNOVER_THRESHOLD = 0.05   # Only rebalance if turnover > 5%

    def __init__(self, tickers: List[str], model_path: Optional[str] = None):
        self.tickers = tickers
        self.n = len(tickers)
        self._model = None
        self._env = None

        if model_path and _HAS_SB3:
            try:
                self._model = PPO.load(model_path)
                logger.info(f"[RL] Loaded model from {model_path}")
            except Exception as e:
                logger.warning(f"[RL] Could not load model from {model_path}: {e}")

    def train(
        self,
        returns_df: pd.DataFrame,
        total_timesteps: int = 100_000,
        learning_rate: float = 3e-4,
    ) -> bool:
        """
        Trains the PPO agent on historical return data.

        Args:
            returns_df:       DataFrame of daily returns, columns = tickers.
            total_timesteps:  Number of environment steps for training.
            learning_rate:    PPO learning rate.
        Returns:
            True if training succeeded, False if SB3 unavailable.
        """
        if not _HAS_SB3:
            logger.warning("[RL] stable-baselines3 not installed. Cannot train.")
            return False

        rets = returns_df[self.tickers].fillna(0.0)
        env = PortfolioEnv(rets)

        try:
            check_env(env, warn=False)
        except Exception as e:
            logger.warning(f"[RL] Environment check warning: {e}")

        self._model = PPO(
            "MlpPolicy", env,
            learning_rate=learning_rate,
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            clip_range=0.2,
            verbose=0,
        )

        logger.info(f"[RL] Training PPO for {total_timesteps} timesteps...")
        self._model.learn(total_timesteps=total_timesteps)
        self._env = env
        logger.info("[RL] Training complete.")
        return True

    def save(self, path: Optional[str] = None) -> str:
        """Saves the trained model to disk."""
        if self._model is None:
            raise RuntimeError("No trained model to save.")
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        save_path = path or str(_MODEL_DIR / "rl_portfolio.zip")
        self._model.save(save_path)
        logger.info(f"[RL] Model saved to {save_path}")
        return save_path

    def optimize(
        self,
        current_weights: Dict[str, float],
        returns_df: pd.DataFrame,
        vol_regime: str = "NORMAL",
        factor_scores: Optional[Dict[str, float]] = None,
    ) -> RLRebalanceResult:
        """
        Returns optimal target weights given current portfolio state.

        Uses PPO inference if a model is loaded, otherwise falls back
        to minimum-variance optimization via Markowitz.

        Args:
            current_weights:  Current ticker → weight mapping.
            returns_df:       Recent daily returns DataFrame (tickers as columns).
            vol_regime:       "LOW" | "NORMAL" | "HIGH" | "EXTREME"
            factor_scores:    Optional factor exposure scores (value, mom, etc.)
        Returns:
            RLRebalanceResult with target weights and diagnostics.
        """
        w_current = np.array([current_weights.get(t, 0.0) for t in self.tickers])
        w_current = w_current / max(w_current.sum(), 1e-8)

        if self._model is not None and _HAS_SB3:
            return self._rl_optimize(w_current, returns_df, vol_regime)
        else:
            return self._mv_optimize(w_current, returns_df, vol_regime)

    # ── PPO Inference ─────────────────────────────────────────────────────────

    def _rl_optimize(
        self,
        w_current: np.ndarray,
        returns_df: pd.DataFrame,
        vol_regime: str,
    ) -> RLRebalanceResult:
        regime_map = {"LOW": 0, "NORMAL": 1, "HIGH": 2, "EXTREME": 3}
        regime_onehot = np.zeros(4, dtype=np.float32)
        regime_onehot[regime_map.get(vol_regime, 1)] = 1.0

        obs = np.zeros(self.n + 10, dtype=np.float32)
        obs[:self.n] = w_current.astype(np.float32)
        obs[self.n:self.n + 4] = regime_onehot

        action, _ = self._model.predict(obs, deterministic=True)
        w_new = w_current + action
        w_new = np.clip(w_new, 0.0, 1.0)
        w_new = w_new / max(w_new.sum(), 1e-8)

        turnover = float(np.sum(np.abs(w_new - w_current)))
        sharpe = self._estimate_sharpe(w_new, returns_df)

        return self._build_result(
            w_new, turnover, sharpe, method="PPO",
            reasoning=f"PPO agent action under {vol_regime} vol regime.",
        )

    # ── Markowitz Fallback ────────────────────────────────────────────────────

    def _mv_optimize(
        self,
        w_current: np.ndarray,
        returns_df: pd.DataFrame,
        vol_regime: str,
    ) -> RLRebalanceResult:
        """Minimum-variance portfolio as fallback when RL is unavailable."""
        try:
            rets = returns_df[self.tickers].fillna(0.0)
            if len(rets) < 10:
                raise ValueError("Insufficient returns for covariance estimation.")
            cov = rets.cov().values
            # Regularize: add shrinkage to diagonal
            cov = cov + np.eye(self.n) * 0.001

            # Minimum variance: solve w = Σ^{-1} 1 / (1^T Σ^{-1} 1)
            ones = np.ones(self.n)
            cov_inv = np.linalg.pinv(cov)
            w_mv = cov_inv @ ones
            w_mv = np.clip(w_mv / w_mv.sum(), 0, 0.40)  # cap at 40% per position
            w_mv = w_mv / w_mv.sum()

            # In HIGH/EXTREME regime, blend toward equal weight for safety
            if vol_regime in ("HIGH", "EXTREME"):
                w_eq = np.ones(self.n) / self.n
                w_mv = 0.5 * w_mv + 0.5 * w_eq

            turnover = float(np.sum(np.abs(w_mv - w_current)))
            sharpe = self._estimate_sharpe(w_mv, rets)

            return self._build_result(
                w_mv, turnover, sharpe, method="MEAN_VARIANCE",
                reasoning=f"Markowitz min-variance under {vol_regime} regime.",
            )
        except Exception as e:
            logger.warning(f"[RL] MV fallback failed: {e} — using equal weight.")
            w_eq = np.ones(self.n) / self.n
            return self._build_result(
                w_eq, float(np.sum(np.abs(w_eq - w_current))), 0.0,
                method="EQUAL_WEIGHT",
                reasoning=f"Equal weight fallback ({e}).",
                warnings=[f"Optimization failed: {e}"],
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _estimate_sharpe(self, weights: np.ndarray, returns_df: pd.DataFrame) -> float:
        """Estimates annualized Sharpe from historical returns using target weights."""
        try:
            rets = returns_df[self.tickers].fillna(0.0).values
            port_ret = rets @ weights
            if port_ret.std() < 1e-8:
                return 0.0
            return round(float(port_ret.mean() / port_ret.std() * np.sqrt(252)), 3)
        except Exception:
            return 0.0

    def _build_result(
        self,
        weights: np.ndarray,
        turnover: float,
        sharpe: float,
        method: str,
        reasoning: str,
        warnings: Optional[List[str]] = None,
    ) -> RLRebalanceResult:
        ticker_weights = {t: round(float(w), 4) for t, w in zip(self.tickers, weights)}
        return RLRebalanceResult(
            ticker_weights=ticker_weights,
            method=method,
            expected_sharpe=sharpe,
            turnover_pct=round(turnover * 100, 2),
            rebalance_needed=turnover > self.MIN_TURNOVER_THRESHOLD,
            reasoning=reasoning,
            warnings=warnings or [],
        )
