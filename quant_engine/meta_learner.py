"""
AlphaAgent — LightGBM Meta-Learner (Stacking Layer)

Theory:
  Take the 9 agent probability outputs as features and train a gradient-
  boosting model to predict 5-day forward return direction.  This captures
  non-linear interactions between agents that sequential Bayes misses.

Architecture:
  Input  : [p_technical, p_fundamental, p_sentiment, p_macro, p_risk,
            p_currency, p_geopolitical, p_insider, p_volatility,
            conviction, entropy, agreement_score, vix_regime, spy_regime]
  Target : binary — 1 if 5d_return > 0 else 0
  Model  : LightGBM classifier with purged-k-fold CV
  Output : blended probability (alpha * lgbm + (1-alpha) * bayesian)

Training:
  - Auto-trains when >= MIN_SAMPLES labelled examples exist in the DB
  - Persists model to quant_engine/models/meta_learner.pkl
  - Retrains monthly (or on demand)

Inference:
  - If model not trained yet → returns None (falls back to Bayesian)
  - blend_weight controls how much to trust the meta-learner vs Bayesian
"""

import logging
import pickle
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_DIR  = Path(__file__).resolve().parent / "models"
_MODEL_PATH = _MODEL_DIR / "meta_learner.pkl"
_META_PATH  = _MODEL_DIR / "meta_learner_meta.pkl"   # training metadata

MIN_SAMPLES    = 80      # minimum labelled examples to train
BLEND_WEIGHT   = 0.35    # how much weight LightGBM gets vs Bayesian
FORWARD_DAYS   = 5       # outcome horizon
RETRAIN_DAYS   = 30      # retrain every N days


@dataclass
class MetaLearnerResult:
    lgbm_prob:    float          # LightGBM output
    blended_prob: float          # alpha*lgbm + (1-alpha)*bayesian
    blend_weight: float          # actual weight used
    n_train:      int            # training sample size
    feature_importance: Dict[str, float]


FEATURE_NAMES = [
    "p_technical", "p_fundamental", "p_sentiment", "p_macro",
    "p_currency", "p_geopolitical", "p_insider", "p_volatility",
    "bayesian_prob", "conviction", "entropy", "agreement_score",
]


class MetaLearner:
    """LightGBM stacking layer over 9 Bayesian agent probabilities."""

    def __init__(self):
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._meta  = {}
        self._load()

    def _load(self):
        try:
            if _MODEL_PATH.exists():
                with open(_MODEL_PATH, "rb") as f:
                    self._model = pickle.load(f)
            if _META_PATH.exists():
                with open(_META_PATH, "rb") as f:
                    self._meta = pickle.load(f)
        except Exception as e:
            logger.warning(f"[MetaLearner] Load failed: {e}")

    def _save(self):
        try:
            with open(_MODEL_PATH, "wb") as f:
                pickle.dump(self._model, f)
            with open(_META_PATH, "wb") as f:
                pickle.dump(self._meta, f)
        except Exception as e:
            logger.warning(f"[MetaLearner] Save failed: {e}")

    # ── Feature extraction ────────────────────────────────────────────────

    @staticmethod
    def build_features(
        agent_probs: Dict[str, float],
        bayesian_prob: float,
        conviction: float,
        entropy: float,
        agreement_score: float,
    ) -> np.ndarray:
        row = [
            agent_probs.get("technical",    0.5),
            agent_probs.get("fundamental",  0.5),
            agent_probs.get("sentiment",    0.5),
            agent_probs.get("macro",        0.5),
            agent_probs.get("currency",     0.5),
            agent_probs.get("geopolitical", 0.5),
            agent_probs.get("insider",      0.5),
            agent_probs.get("volatility",   0.5),
            bayesian_prob,
            conviction,
            entropy,
            agreement_score,
        ]
        return np.array(row, dtype=np.float32).reshape(1, -1)

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(
        self,
        agent_probs: Dict[str, float],
        bayesian_prob: float,
        conviction: float  = 0.0,
        entropy: float     = 0.5,
        agreement_score: float = 0.5,
    ) -> Optional[MetaLearnerResult]:
        """
        Returns blended probability or None if model not trained yet.
        """
        if self._model is None:
            return None

        try:
            X = self.build_features(
                agent_probs, bayesian_prob, conviction, entropy, agreement_score
            )
            lgbm_prob = float(self._model.predict_proba(X)[0, 1])

            # Adaptive blend: trust meta-learner more when conviction is high
            adaptive_weight = BLEND_WEIGHT * (0.5 + conviction)
            adaptive_weight = min(0.60, max(0.15, adaptive_weight))

            blended = adaptive_weight * lgbm_prob + (1.0 - adaptive_weight) * bayesian_prob

            importance = {}
            if hasattr(self._model, "feature_importances_"):
                for name, imp in zip(FEATURE_NAMES, self._model.feature_importances_):
                    importance[name] = round(float(imp), 4)

            return MetaLearnerResult(
                lgbm_prob=round(lgbm_prob, 4),
                blended_prob=round(blended, 4),
                blend_weight=round(adaptive_weight, 3),
                n_train=self._meta.get("n_train", 0),
                feature_importance=importance,
            )
        except Exception as e:
            logger.warning(f"[MetaLearner] Predict error: {e}")
            return None

    # ── Training ──────────────────────────────────────────────────────────

    def should_retrain(self) -> bool:
        last = self._meta.get("trained_at", 0)
        return (time.time() - last) > (RETRAIN_DAYS * 86400)

    def train(self, db_manager) -> bool:
        """
        Loads historical signals from the DB, fetches outcomes,
        trains LightGBM with purged-k-fold CV.
        Returns True on success.
        """
        try:
            import lightgbm as lgb
            from sklearn.model_selection import StratifiedKFold
            from sklearn.metrics import roc_auc_score
            import yfinance as yf
        except ImportError as e:
            logger.error(f"[MetaLearner] Missing dependency: {e}")
            return False

        logger.info("[MetaLearner] Building training dataset from DB...")

        rows_X, rows_y = [], []
        try:
            db = db_manager.SessionLocal()
            from database.models import AgentLog, Trade

            # Group agent logs by (ticker, timestamp window)
            from sqlalchemy import func
            trade_rows = (
                db.query(Trade)
                .filter(Trade.status == "CLOSED", Trade.pnl_pct.isnot(None))
                .order_by(Trade.timestamp)
                .all()
            )
            db.close()

            if len(trade_rows) < MIN_SAMPLES:
                logger.info(f"[MetaLearner] Only {len(trade_rows)} labelled trades — need {MIN_SAMPLES}")
                return False

            for trade in trade_rows:
                if trade.agent_audit is None:
                    continue
                agent_probs = {}
                for a in trade.agent_audit:
                    name = a.get("agent", "").lower()
                    prob = a.get("prob", 0.5)
                    try:
                        agent_probs[name] = float(prob)
                    except Exception:
                        pass

                if len(agent_probs) < 5:
                    continue

                # Use direction + pnl to infer bayesian_prob proxy
                direction = trade.direction or "LONG"
                pnl_pct   = trade.pnl_pct or 0.0
                outcome   = 1 if pnl_pct > 0 else 0

                bayesian_prob = trade.probability or 0.5
                conviction    = abs(bayesian_prob - 0.5) * 2

                X = self.build_features(
                    agent_probs, bayesian_prob, conviction, 0.5, 0.5
                )
                rows_X.append(X[0])
                rows_y.append(outcome)

        except Exception as e:
            logger.error(f"[MetaLearner] DB load error: {e}")
            return False

        if len(rows_X) < MIN_SAMPLES:
            logger.info(f"[MetaLearner] {len(rows_X)} valid samples — need {MIN_SAMPLES}")
            return False

        X = np.array(rows_X, dtype=np.float32)
        y = np.array(rows_y, dtype=np.int32)

        logger.info(f"[MetaLearner] Training on {len(X)} samples | "
                    f"positive rate={y.mean():.2%}")

        # Purged k-fold (simple version: use time-ordered split with gap)
        n_splits   = min(5, len(X) // 20)
        kf         = StratifiedKFold(n_splits=max(3, n_splits), shuffle=False)
        oof_probs  = np.zeros(len(X))

        params = {
            "objective":     "binary",
            "metric":        "auc",
            "learning_rate": 0.05,
            "num_leaves":    15,
            "min_child_samples": 10,
            "subsample":     0.8,
            "colsample_bytree": 0.8,
            "reg_alpha":     0.1,
            "reg_lambda":    0.1,
            "n_estimators":  200,
            "verbose":       -1,
        }

        fold_aucs = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            # Purge: skip last 5 train samples (simulate embargo)
            train_idx = train_idx[:-5] if len(train_idx) > 10 else train_idx
            model = lgb.LGBMClassifier(**params)
            model.fit(X[train_idx], y[train_idx],
                      eval_set=[(X[val_idx], y[val_idx])],
                      callbacks=[lgb.early_stopping(20, verbose=False),
                                 lgb.log_evaluation(-1)])
            preds = model.predict_proba(X[val_idx])[:, 1]
            oof_probs[val_idx] = preds
            auc = roc_auc_score(y[val_idx], preds)
            fold_aucs.append(auc)
            logger.info(f"[MetaLearner] Fold {fold+1} AUC={auc:.4f}")

        mean_auc = float(np.mean(fold_aucs))
        logger.info(f"[MetaLearner] CV AUC={mean_auc:.4f} over {len(fold_aucs)} folds")

        # Final model on all data
        final_model = lgb.LGBMClassifier(**params)
        final_model.fit(X, y, callbacks=[lgb.log_evaluation(-1)])
        self._model = final_model
        self._meta  = {
            "trained_at": time.time(),
            "n_train":    len(X),
            "cv_auc":     round(mean_auc, 4),
            "pos_rate":   round(float(y.mean()), 4),
        }
        self._save()
        logger.info(f"[MetaLearner] Model saved. CV AUC={mean_auc:.4f}")
        return True


# ── Module-level singleton ────────────────────────────────────────────────────

_meta_learner: Optional[MetaLearner] = None


def get_meta_learner() -> MetaLearner:
    global _meta_learner
    if _meta_learner is None:
        _meta_learner = MetaLearner()
    return _meta_learner
