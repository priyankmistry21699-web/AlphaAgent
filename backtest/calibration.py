"""
AlphaAgent — Backtest Probability Calibration

Post-backtest calibration analysis for the quantitative signal engine.
Measures how well the signal probabilities predict actual outcomes,
and derives isotonic-regression auto-calibration curves.

Workflow:
  1. Run BacktestEngine to get a list of (probability, direction, return_pct) trades.
  2. Convert to (probability, binary_outcome) pairs where outcome=1 if
     direction matched actual return direction.
  3. Feed into CalibrationAnalyzer to:
     - Compute Brier score (lower = better calibrated)
     - Build reliability diagram (predicted prob vs actual frequency per bin)
     - Fit Platt scaling logistic calibrator
     - Fit isotonic regression (non-parametric, non-decreasing)
     - Compute Information Coefficient (rank correlation of prob vs return)

Usage:
    from backtest.engine import BacktestEngine
    from backtest.calibration import CalibrationAnalyzer

    engine = BacktestEngine()
    result = engine.run("AAPL", period="3y")

    analyzer = CalibrationAnalyzer()
    report = analyzer.analyze(result.trades)
    print(f"Brier score: {report.brier_score}")
    print(f"IC: {report.information_coefficient}")
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from quant_engine.calibration import ProbabilityCalibrator

logger = logging.getLogger(__name__)


@dataclass
class CalibrationBin:
    bin_label: str           # e.g. "0.5–0.6"
    mean_predicted: float    # avg predicted probability in bin
    actual_frequency: float  # actual win rate in bin
    n_trades: int
    calibration_gap: float   # actual_frequency - mean_predicted


@dataclass
class CalibrationReport:
    n_trades: int
    brier_score: float           # Mean squared error of probabilities (0-1, lower=better)
    log_loss: float              # Log-loss (lower=better)
    information_coefficient: float  # Rank correlation of prob vs return (higher=better)
    ic_ir: float                 # IC / std(IC) across rolling windows
    platt_slope: float           # Platt scaling slope (near 1.0 = well calibrated)
    platt_intercept: float       # Platt scaling intercept (near 0 = well calibrated)
    platt_converged: bool
    reliability_bins: List[CalibrationBin] = field(default_factory=list)
    isotonic_curve: Dict[str, List[float]] = field(default_factory=dict)  # {raw: [...], calibrated: [...]}
    calibrated_probabilities: List[float] = field(default_factory=list)
    signal_accuracy_pct: float = 0.0
    overconfidence: float = 0.0  # positive = model overconfident


class CalibrationAnalyzer:
    """
    Wraps the ProbabilityCalibrator to provide backtest-specific
    probability calibration analysis.

    Converts backtest trade records → calibration diagnostics.
    """

    ROLLING_WINDOW = 20   # window for rolling IC computation

    def analyze(
        self,
        trades: list,
        return_calibrated: bool = True,
    ) -> CalibrationReport:
        """
        Analyzes calibration quality from a list of BacktestTrade objects
        (or dicts with 'probability', 'direction', 'return_pct' keys).

        Args:
            trades:             List of BacktestTrade or dict objects.
            return_calibrated:  Whether to compute calibrated probabilities.
        Returns:
            CalibrationReport with diagnostics.
        """
        probs, outcomes, returns = self._extract(trades)

        if len(probs) < 10:
            logger.warning("[CalibAnalyzer] Too few trades (%d) for reliable calibration.", len(probs))
            return CalibrationReport(
                n_trades=len(probs),
                brier_score=0.25,
                log_loss=0.693,
                information_coefficient=0.0,
                ic_ir=0.0,
                platt_slope=1.0,
                platt_intercept=0.0,
                platt_converged=False,
                signal_accuracy_pct=50.0,
            )

        p = np.array(probs, dtype=float)
        y = np.array(outcomes, dtype=float)
        r = np.array(returns, dtype=float)

        # ── Platt scaling calibration ─────────────────────────────────────────
        cal = ProbabilityCalibrator()
        cal.fit(probs, outcomes)
        eval_res = cal.evaluate(probs, outcomes)

        # ── Reliability diagram bins ──────────────────────────────────────────
        bins: List[CalibrationBin] = []
        for b in eval_res.reliability_bins:
            bins.append(CalibrationBin(
                bin_label=b["bin"],
                mean_predicted=b["mean_predicted"],
                actual_frequency=b["actual_freq"],
                n_trades=b["n"],
                calibration_gap=round(b["actual_freq"] - b["mean_predicted"], 3),
            ))

        # ── Overconfidence ────────────────────────────────────────────────────
        # Positive = model predicts higher probs than actual outcomes warrant
        overall_actual = float(y.mean())
        overall_predicted = float(p.mean())
        overconfidence = round(overall_predicted - overall_actual, 4)

        # ── Information Coefficient ───────────────────────────────────────────
        # Rank correlation between predicted probability and actual return
        from scipy.stats import spearmanr
        try:
            ic_val, _ = spearmanr(p, r)
            ic_val = float(ic_val) if not np.isnan(ic_val) else 0.0
        except Exception:
            ic_val = 0.0

        # Rolling IC for IC_IR computation
        ic_vals = []
        for start in range(0, len(probs) - self.ROLLING_WINDOW, self.ROLLING_WINDOW // 2):
            window_p = p[start:start + self.ROLLING_WINDOW]
            window_r = r[start:start + self.ROLLING_WINDOW]
            try:
                ic_w, _ = spearmanr(window_p, window_r)
                if not np.isnan(ic_w):
                    ic_vals.append(float(ic_w))
            except Exception:
                pass

        ic_ir = 0.0
        if len(ic_vals) >= 3:
            ic_ir = round(float(np.mean(ic_vals) / (np.std(ic_vals) + 1e-8)), 3)

        # ── Isotonic regression calibration ───────────────────────────────────
        iso_curve: Dict[str, List[float]] = {}
        calibrated_probs: List[float] = list(cal.transform(probs))

        if return_calibrated:
            try:
                from sklearn.isotonic import IsotonicRegression
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(p, y)
                iso_calibrated = [round(float(c), 4) for c in iso.predict(p)]
                # Build a lookup curve for visualization
                x_curve = np.linspace(0.3, 0.8, 50)
                iso_curve = {
                    "raw": [round(float(v), 3) for v in x_curve],
                    "calibrated": [round(float(c), 3) for c in iso.predict(x_curve)],
                }
                calibrated_probs = iso_calibrated
            except ImportError:
                pass  # sklearn optional; fall back to Platt output

        accuracy_pct = float(y.mean() * 100)

        return CalibrationReport(
            n_trades=len(probs),
            brier_score=eval_res.brier_score,
            log_loss=eval_res.log_loss,
            information_coefficient=round(ic_val, 4),
            ic_ir=ic_ir,
            platt_slope=eval_res.slope,
            platt_intercept=eval_res.intercept,
            platt_converged=eval_res.converged,
            reliability_bins=bins,
            isotonic_curve=iso_curve,
            calibrated_probabilities=calibrated_probs,
            signal_accuracy_pct=round(accuracy_pct, 1),
            overconfidence=overconfidence,
        )

    @staticmethod
    def _extract(trades: list):
        """
        Converts trade list to (probs, outcomes, returns) arrays.

        Handles both BacktestTrade dataclass instances and plain dicts.
        outcome = 1 if the trade direction matched the actual price movement.
        """
        probs, outcomes, returns = [], [], []
        for t in trades:
            if hasattr(t, 'probability'):
                prob = float(t.probability)
                direction = str(t.direction)
                return_pct = float(t.return_pct)
            elif isinstance(t, dict):
                prob = float(t.get('probability', 0.5))
                direction = str(t.get('direction', 'HOLD'))
                return_pct = float(t.get('return_pct', 0.0))
            else:
                continue

            # Binary outcome: 1 if model was right, 0 if wrong
            if direction == 'LONG':
                outcome = 1 if return_pct > 0 else 0
                p_used = prob
            elif direction == 'SHORT':
                outcome = 1 if return_pct < 0 else 0
                p_used = 1.0 - prob  # SHORT uses 1-p as the "confidence" in bear direction
            else:
                continue  # skip HOLD

            probs.append(np.clip(p_used, 0.01, 0.99))
            outcomes.append(outcome)
            returns.append(return_pct)

        return probs, outcomes, returns


def run_calibration_report(ticker: str, period: str = "3y") -> CalibrationReport:
    """
    Convenience function: run the backtest and return calibration diagnostics.

    Usage:
        report = run_calibration_report("AAPL", period="3y")
        print(f"Brier: {report.brier_score}, IC: {report.information_coefficient}")
    """
    from backtest.engine import BacktestEngine
    engine = BacktestEngine()
    result = engine.run(ticker, period=period)
    analyzer = CalibrationAnalyzer()
    return analyzer.analyze(result.trades)
