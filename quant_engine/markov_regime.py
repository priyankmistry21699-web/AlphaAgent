"""
AlphaAgent — Observable Markov Chain Regime Detector

Labels every day Bull / Bear / Sideways via a rolling-return rule (default 20-day, ±5%),
builds a 3×3 maximum-likelihood transition matrix, forecasts n-steps ahead via
Chapman-Kolmogorov matrix powers, solves for the stationary distribution,
emits a signed signal (bull_prob − bear_prob), and runs a walk-forward backtest.
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import List, Dict

logger = logging.getLogger(__name__)

STATES = ['Bull', 'Bear', 'Sideways']
STATE_IDX = {s: i for i, s in enumerate(STATES)}
STATE_COLORS = {'Bull': '#22c55e', 'Bear': '#ef4444', 'Sideways': '#f59e0b'}


@dataclass
class MarkovRegimeResult:
    current_regime: str
    regime_duration_days: int

    current_probs: Dict[str, float]      # one-hot of current state
    forecast_steps: List[Dict]           # [{step, bull, bear, sideways}] × 20
    transition_matrix: List[List[float]] # 3×3, rows=from, cols=to
    state_labels: List[str]

    stationary: Dict[str, float]         # long-run regime mix

    signal: str                          # LONG | SHORT | NEUTRAL
    conviction: float                    # 0-100
    signed_score: float                  # bull_1step − bear_1step

    regime_history: List[Dict]           # last 60 days [{date, regime, return_pct}]

    backtest_sharpe: float
    backtest_max_dd: float
    backtest_total_return: float
    backtest_win_rate: float
    backtest_n_trades: int
    backtest_equity: List[Dict]          # [{idx, value}]

    notes: str = ""


class MarkovRegimeDetector:
    """
    Observable Markov chain regime detector.

    Rolling-return labeling → 3×3 ML transition matrix →
    Chapman-Kolmogorov n-step forecast → stationary distribution →
    signed signal → walk-forward backtest.
    """

    def __init__(
        self,
        window: int = 20,
        bull_thresh: float = 0.05,
        bear_thresh: float = -0.05,
    ):
        self.window = window
        self.bull_thresh = bull_thresh
        self.bear_thresh = bear_thresh

    # ── Regime labeling ───────────────────────────────────────────────────────

    def label_regimes(self, prices: pd.Series) -> pd.Series:
        """Label each day: rolling window return vs ±thresh → Bull/Bear/Sideways."""
        rolling_ret = prices.pct_change(self.window)

        def _classify(r):
            if pd.isna(r):
                return np.nan
            if r > self.bull_thresh:
                return 'Bull'
            if r < self.bear_thresh:
                return 'Bear'
            return 'Sideways'

        return rolling_ret.apply(_classify)

    # ── Transition matrix ─────────────────────────────────────────────────────

    def build_transition_matrix(self, labels: pd.Series) -> np.ndarray:
        """
        Maximum-likelihood 3×3 transition matrix with Laplace smoothing.
        T[i,j] = P(next=j | current=i).
        """
        T = np.ones((3, 3))   # Laplace prior: add 1 to every count
        clean = labels.dropna().values

        for k in range(len(clean) - 1):
            fi = STATE_IDX.get(clean[k])
            ti = STATE_IDX.get(clean[k + 1])
            if fi is not None and ti is not None:
                T[fi, ti] += 1

        row_sums = T.sum(axis=1, keepdims=True)
        return T / row_sums

    # ── Stationary distribution ───────────────────────────────────────────────

    @staticmethod
    def stationary_distribution(T: np.ndarray) -> np.ndarray:
        """
        Solve πT = π, Σπ = 1 via left eigenvector corresponding to eigenvalue 1.
        Falls back to uniform if numerics are bad.
        """
        try:
            eigenvalues, eigenvectors = np.linalg.eig(T.T)
            idx = int(np.argmin(np.abs(eigenvalues - 1.0)))
            pi = np.real(eigenvectors[:, idx])
            pi = np.maximum(pi, 0.0)
            total = pi.sum()
            if total > 1e-10:
                return pi / total
        except Exception:
            pass
        return np.ones(3) / 3.0

    # ── n-step Chapman-Kolmogorov ─────────────────────────────────────────────

    def forecast_n_steps(
        self, T: np.ndarray, current_state_idx: int, max_steps: int = 20
    ) -> List[Dict]:
        """Forecast regime probabilities at each horizon via T^n matrix powers."""
        state_vec = np.zeros(3)
        state_vec[current_state_idx] = 1.0

        results = []
        for n in range(1, max_steps + 1):
            T_n = np.linalg.matrix_power(T, n)
            probs = state_vec @ T_n
            results.append({
                'step': n,
                'bull':     round(float(probs[0]), 4),
                'bear':     round(float(probs[1]), 4),
                'sideways': round(float(probs[2]), 4),
            })
        return results

    # ── Walk-forward backtest ─────────────────────────────────────────────────

    def walk_forward_backtest(
        self,
        prices: pd.Series,
        labels: pd.Series,
        signal_thresh: float = 0.05,
    ) -> Dict:
        """
        No-lookahead walk-forward:
          At day t build T from history [0:t], generate 1-step signal,
          realise next day's return.
        Signal: LONG if (bull_p − bear_p) > thresh, SHORT if < −thresh.
        """
        daily_ret = prices.pct_change()

        combined = pd.DataFrame({
            'ret':   daily_ret,
            'label': labels,
        }).dropna()

        min_history = max(self.window * 3, 60)
        if len(combined) < min_history + 10:
            return {
                'sharpe': 0.0, 'max_dd': 0.0, 'total_return': 0.0,
                'win_rate': 0.0, 'n_trades': 0,
                'equity': [{'idx': 0, 'value': 100.0}],
            }

        equity = [100.0]
        strategy_returns = []
        trades = 0
        wins = 0
        equity_series = [{'idx': 0, 'value': 100.0}]

        for i in range(min_history, len(combined) - 1):
            hist_labels = combined['label'].iloc[:i]
            T_wf = self.build_transition_matrix(hist_labels)

            cur_state = combined['label'].iloc[i]
            cur_idx = STATE_IDX.get(cur_state, 2)

            sv = np.zeros(3)
            sv[cur_idx] = 1.0
            probs_next = sv @ T_wf

            score = float(probs_next[0]) - float(probs_next[1])  # bull − bear

            if score > signal_thresh:
                sig = 1
            elif score < -signal_thresh:
                sig = -1
            else:
                sig = 0

            next_ret = float(combined['ret'].iloc[i + 1])
            strat_ret = sig * next_ret
            strategy_returns.append(strat_ret)

            if sig != 0:
                trades += 1
                if strat_ret > 0:
                    wins += 1

            equity.append(equity[-1] * (1.0 + strat_ret))
            equity_series.append({'idx': len(equity_series), 'value': round(equity[-1], 2)})

        if len(strategy_returns) < 10:
            return {
                'sharpe': 0.0, 'max_dd': 0.0, 'total_return': 0.0,
                'win_rate': 0.0, 'n_trades': 0,
                'equity': equity_series,
            }

        arr = np.array(strategy_returns)
        sharpe = float(arr.mean() / arr.std() * np.sqrt(252)) if arr.std() > 0 else 0.0

        eq_arr = np.array([e['value'] for e in equity_series])
        peak = np.maximum.accumulate(eq_arr)
        dd = (eq_arr - peak) / peak
        max_dd = float(dd.min()) * 100.0

        total_return = float((equity_series[-1]['value'] / 100.0 - 1.0) * 100.0)
        win_rate = float(wins / trades * 100.0) if trades > 0 else 0.0

        return {
            'sharpe':       round(sharpe, 3),
            'max_dd':       round(max_dd, 2),
            'total_return': round(total_return, 2),
            'win_rate':     round(win_rate, 1),
            'n_trades':     trades,
            'equity':       equity_series,
        }

    # ── Main analysis pipeline ────────────────────────────────────────────────

    def analyze(self, prices: pd.Series) -> MarkovRegimeResult:
        min_pts = self.window * 4
        if len(prices) < min_pts:
            raise ValueError(f"Need at least {min_pts} price points, got {len(prices)}")

        labels = self.label_regimes(prices)
        clean_labels = labels.dropna()

        T = self.build_transition_matrix(clean_labels)

        current_regime = clean_labels.iloc[-1]
        current_idx = STATE_IDX.get(current_regime, 2)

        # Regime duration
        duration = 1
        for k in range(len(clean_labels) - 2, -1, -1):
            if clean_labels.iloc[k] == current_regime:
                duration += 1
            else:
                break

        pi = self.stationary_distribution(T)
        stationary = {s: round(float(pi[i]), 4) for i, s in enumerate(STATES)}

        forecast_steps = self.forecast_n_steps(T, current_idx, max_steps=20)

        one_step = forecast_steps[0]
        bull_p1 = one_step['bull']
        bear_p1 = one_step['bear']
        signed_score = bull_p1 - bear_p1

        if signed_score > 0.08:
            signal = 'LONG'
        elif signed_score < -0.08:
            signal = 'SHORT'
        else:
            signal = 'NEUTRAL'
        conviction = round(abs(signed_score) * 100.0, 1)

        bt = self.walk_forward_backtest(prices, labels)

        # Regime history (last 60 labelled days)
        rolling_ret = prices.pct_change(self.window)
        hist60 = clean_labels.tail(60)
        regime_history = []
        for dt, reg in hist60.items():
            rv = rolling_ret.get(dt, np.nan)
            regime_history.append({
                'date':       str(dt.date()) if hasattr(dt, 'date') else str(dt)[:10],
                'regime':     reg,
                'return_pct': round(float(rv) * 100.0, 2) if pd.notna(rv) else 0.0,
            })

        current_probs = {s: (1.0 if s == current_regime else 0.0) for s in STATES}

        # Build notes: 90-day regime mix
        counts = clean_labels.tail(90).value_counts()
        total_c = len(clean_labels.tail(90))
        mix = ', '.join(f"{s}: {counts.get(s, 0) / total_c * 100:.0f}%" for s in STATES)
        notes = f"Last 90d regime mix — {mix}. Window={self.window}d, thresholds ±{self.bull_thresh*100:.0f}%."

        return MarkovRegimeResult(
            current_regime=current_regime,
            regime_duration_days=duration,
            current_probs=current_probs,
            forecast_steps=forecast_steps,
            transition_matrix=T.tolist(),
            state_labels=STATES,
            stationary=stationary,
            signal=signal,
            conviction=conviction,
            signed_score=round(signed_score, 4),
            regime_history=regime_history,
            backtest_sharpe=bt['sharpe'],
            backtest_max_dd=bt['max_dd'],
            backtest_total_return=bt['total_return'],
            backtest_win_rate=bt['win_rate'],
            backtest_n_trades=bt['n_trades'],
            backtest_equity=bt['equity'],
            notes=notes,
        )
