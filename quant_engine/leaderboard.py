"""
AlphaAgent — Agent Performance Leaderboard

Evaluates each agent's historical predictive accuracy by loading logged
signals from the database and comparing predicted direction against the
actual price move over the subsequent 5 trading days.

Scoring metrics per agent:
  - Directional accuracy   : % of correct direction calls
  - Brier score            : mean squared error of probabilities (lower = better)
  - Information Coefficient: Pearson corr(prob - 0.5, forward_return)
  - IC Information Ratio   : IC mean / IC std  (t-stat of IC stability)
  - Information ratio      : risk-adjusted signal quality (annualised)
  - Composite rank score   : 30% accuracy + 25% calibration + 25% IC_IR + 20% IR

Falls back to a realistic demo leaderboard when no database data is available.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)

_AGENTS_YAML = Path(__file__).resolve().parent.parent / "config" / "agents.yaml"
_FORWARD_DAYS = 5          # evaluation horizon: 5 trading days after signal


@dataclass
class AgentScore:
    agent_name: str
    n_predictions: int
    directional_accuracy: float      # percent, 0-100
    brier_score: float               # lower is better; 0.25 = random
    avg_confidence: float            # mean confidence, 0-1
    information_ratio: float         # annualised signal IR
    ic: float = 0.0                  # mean Information Coefficient (IC)
    ic_ir: float = 0.0               # IC Information Ratio = IC / IC_std
    rolling_ic: List[float] = field(default_factory=list)  # IC in 20-signal windows
    rank: int = 0


@dataclass
class LeaderboardResult:
    scores: List[AgentScore]
    evaluation_window_days: int
    total_signals_evaluated: int
    weight_suggestions: Dict[str, float] = field(default_factory=dict)


class AgentLeaderboard:
    """
    Ranks agents by historical predictive performance using logged data from
    the AlphaAgent database. Accepts an optional DatabaseManager instance;
    returns a realistic demo leaderboard gracefully when none is provided.
    """

    def __init__(self, db_manager=None):
        self.db_manager = db_manager

    def evaluate_performance(
        self,
        ticker: Optional[str] = None,
        window_days: int = 90,
    ) -> LeaderboardResult:
        if self.db_manager is None:
            return self._demo_leaderboard(window_days)

        try:
            agent_data = self._load_agent_logs(ticker, window_days)
            if not agent_data:
                logger.info("[Leaderboard] No logged predictions — returning demo.")
                return self._demo_leaderboard(window_days)
            return self._score_agents(agent_data, window_days)
        except Exception as e:
            logger.error(f"[Leaderboard] Evaluation failed: {e}")
            return self._demo_leaderboard(window_days)

    def tune_weights(self, window_days: int = 90) -> Dict[str, float]:
        """
        Derives recommended agent weights from IC_IR and writes them to
        config/agents.yaml. Returns the new weight dict.

        Weight derivation:
          raw_weight ∝ max(0, IC_IR)   (zero-floor: negative IC_IR → zero weight)
          Normalised so weights across voting agents sum to 1.0.
          Risk agent always stays at 0.0 (circuit breaker, not voter).
        """
        result = self.evaluate_performance(window_days=window_days)
        ic_irs = {
            s.agent_name: max(0.0, s.ic_ir)
            for s in result.scores
            if s.agent_name != "risk"
        }
        total = sum(ic_irs.values())

        if total < 1e-6:
            logger.warning("[Leaderboard] All IC_IRs are zero — cannot tune weights.")
            return {}

        new_weights = {name: round(w / total, 4) for name, w in ic_irs.items()}

        # Write back to agents.yaml
        try:
            if _AGENTS_YAML.exists():
                with open(_AGENTS_YAML, "r") as f:
                    config = yaml.safe_load(f) or {}
                for agent_name, w in new_weights.items():
                    if agent_name in config.get("agents", {}):
                        config["agents"][agent_name]["weight"] = w
                config["agents"]["risk"] = config["agents"].get("risk", {})
                config["agents"]["risk"]["weight"] = 0.0
                with open(_AGENTS_YAML, "w") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                logger.info(f"[Leaderboard] Agent weights updated: {new_weights}")
        except Exception as e:
            logger.error(f"[Leaderboard] Failed to write weights to YAML: {e}")

        return new_weights

    def get_leaderboard_stats(self) -> List[tuple]:
        """Legacy helper: returns (agent_name, metrics_dict) sorted by rank."""
        result = self.evaluate_performance()
        return [
            (s.agent_name, {
                "hit_rate": s.directional_accuracy / 100,
                "brier_score": s.brier_score,
                "information_ratio": s.information_ratio,
                "ic": s.ic,
                "ic_ir": s.ic_ir,
                "rank": s.rank,
            })
            for s in result.scores
        ]

    # ── Database loading ──────────────────────────────────────────────────────

    def _load_agent_logs(
        self,
        ticker: Optional[str],
        window_days: int,
    ) -> Dict[str, List[dict]]:
        from database.models import AgentLog
        from datetime import datetime, timedelta

        db = self.db_manager.SessionLocal()
        cutoff = datetime.utcnow() - timedelta(days=window_days)
        try:
            query = db.query(AgentLog).filter(AgentLog.timestamp >= cutoff)
            if ticker:
                query = query.filter(AgentLog.ticker == ticker)
            logs = query.all()

            agent_data: Dict[str, List[dict]] = {}
            for log in logs:
                name = log.agent_name
                agent_data.setdefault(name, []).append({
                    "ticker": log.ticker,
                    "timestamp": log.timestamp,
                    "probability": float(log.probability or 0.5),
                    "confidence": float(log.confidence or 0.5),
                    "vote": log.vote or "HOLD",
                })
            return agent_data
        finally:
            db.close()

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_agents(
        self,
        agent_data: Dict[str, List[dict]],
        window_days: int,
    ) -> LeaderboardResult:
        import yfinance as yf
        import pandas as pd

        scores: List[AgentScore] = []
        total_evaluated = 0

        for agent_name, predictions in agent_data.items():
            probs, actuals, confidences, fwd_returns = [], [], [], []

            for pred in predictions:
                try:
                    ticker = pred["ticker"]
                    ts = pred["timestamp"]
                    prob = pred["probability"]
                    conf = pred["confidence"]

                    df = yf.download(
                        ticker, start=ts, period=f"{_FORWARD_DAYS + 3}d",
                        interval="1d", auto_adjust=True, progress=False,
                    )
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    if len(df) < 2:
                        continue

                    actual_ret = float(df["Close"].iloc[-1] / df["Close"].iloc[0] - 1)
                    probs.append(prob)
                    actuals.append(1 if actual_ret > 0 else 0)
                    confidences.append(conf)
                    fwd_returns.append(actual_ret)
                except Exception:
                    continue

            if len(probs) < 3:
                continue

            p = np.array(probs)
            y = np.array(actuals, dtype=float)
            c = np.array(confidences)
            r = np.array(fwd_returns)

            predicted_dir = (p > 0.5).astype(int)
            accuracy = float(np.mean(predicted_dir == y)) * 100
            brier = float(np.mean((p - y) ** 2))
            avg_conf = float(np.mean(c))

            # ── IC: Pearson corr between (prob - 0.5) and forward return ──────
            centered_p = p - 0.5
            if centered_p.std() > 0 and r.std() > 0:
                ic_mean = float(np.corrcoef(centered_p, r)[0, 1])
            else:
                ic_mean = 0.0

            # Rolling IC in windows of 20 signals
            window = 20
            rolling_ic = []
            for start in range(0, len(p) - window + 1, window // 2):
                w_p = centered_p[start:start + window]
                w_r = r[start:start + window]
                if w_p.std() > 0 and w_r.std() > 0:
                    rolling_ic.append(round(float(np.corrcoef(w_p, w_r)[0, 1]), 4))

            ic_ir = float(np.mean(rolling_ic) / np.std(rolling_ic)) \
                if len(rolling_ic) >= 2 and np.std(rolling_ic) > 0 else ic_mean

            # ── Annualised Information Ratio (signal return quality) ──────────
            outcomes = np.where(predicted_dir == y, 1.0, -1.0)
            ir_std = float(outcomes.std())
            ir = float(outcomes.mean() / ir_std * np.sqrt(252)) if ir_std > 0 else 0.0

            scores.append(AgentScore(
                agent_name=agent_name,
                n_predictions=len(probs),
                directional_accuracy=round(accuracy, 1),
                brier_score=round(brier, 4),
                avg_confidence=round(avg_conf, 3),
                information_ratio=round(ir, 3),
                ic=round(ic_mean, 4),
                ic_ir=round(ic_ir, 4),
                rolling_ic=rolling_ic,
            ))
            total_evaluated += len(probs)

        scores.sort(key=self._composite_score, reverse=True)
        for i, s in enumerate(scores):
            s.rank = i + 1

        # Derive weight suggestions from IC_IR
        weight_suggestions = self._derive_weights(scores)

        return LeaderboardResult(
            scores=scores,
            evaluation_window_days=window_days,
            total_signals_evaluated=total_evaluated,
            weight_suggestions=weight_suggestions,
        )

    @staticmethod
    def _composite_score(s: AgentScore) -> float:
        cal_quality = max(0.0, (0.25 - s.brier_score) / 0.25 * 100)
        ir_score = max(0.0, s.information_ratio * 10)
        ic_ir_score = max(0.0, s.ic_ir * 20)
        return (
            s.directional_accuracy * 0.30
            + cal_quality * 0.25
            + ic_ir_score * 0.25
            + ir_score * 0.20
        )

    @staticmethod
    def _derive_weights(scores: List[AgentScore]) -> Dict[str, float]:
        ic_irs = {
            s.agent_name: max(0.0, s.ic_ir)
            for s in scores
            if s.agent_name != "risk"
        }
        total = sum(ic_irs.values())
        if total < 1e-6:
            return {}
        return {name: round(w / total, 4) for name, w in ic_irs.items()}

    # ── Demo fallback ─────────────────────────────────────────────────────────

    def _demo_leaderboard(self, window_days: int = 90) -> LeaderboardResult:
        demo = [
            AgentScore("fundamental",  47, 62.0, 0.2271, 0.72, 0.834, ic=0.068, ic_ir=1.21, rank=1),
            AgentScore("macro",        43, 59.5, 0.2408, 0.68, 0.751, ic=0.055, ic_ir=0.98, rank=2),
            AgentScore("technical",    51, 58.9, 0.2461, 0.65, 0.718, ic=0.051, ic_ir=0.91, rank=3),
            AgentScore("insider",      38, 57.9, 0.2537, 0.61, 0.641, ic=0.042, ic_ir=0.75, rank=4),
            AgentScore("geopolitical", 35, 56.2, 0.2614, 0.55, 0.544, ic=0.034, ic_ir=0.61, rank=5),
            AgentScore("sentiment",    48, 55.6, 0.2689, 0.52, 0.491, ic=0.029, ic_ir=0.52, rank=6),
            AgentScore("volatility",   44, 54.4, 0.2741, 0.63, 0.423, ic=0.022, ic_ir=0.39, rank=7),
            AgentScore("currency",     36, 53.8, 0.2799, 0.49, 0.389, ic=0.018, ic_ir=0.32, rank=8),
        ]
        weight_suggestions = self._derive_weights(demo)
        return LeaderboardResult(
            scores=demo,
            evaluation_window_days=window_days,
            total_signals_evaluated=0,
            weight_suggestions=weight_suggestions,
        )
