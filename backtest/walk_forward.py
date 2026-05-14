"""
AlphaAgent — Walk-Forward Validator

Time-series cross-validation to prevent look-ahead bias. Downloads data once,
slices it into anchored train+test windows, and runs the BacktestEngine on
each test fold. Aggregates metrics across folds to estimate out-of-sample
strategy performance.

Window structure:
  |---- train (train_years) ----|-- test (test_months) --|
                                |---- train+1 ------------|-- test --|
  (expanding anchor, not rolling)
"""

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine, BacktestResult
from backtest.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    fold: int
    train_period: str
    test_period: str
    metrics: PerformanceMetrics
    signal_accuracy: float
    n_trades: int


@dataclass
class WalkForwardResult:
    ticker: str
    n_folds: int
    folds: List[WalkForwardFold]
    avg_sharpe: float
    avg_max_drawdown_pct: float
    avg_signal_accuracy: float
    avg_cagr_pct: float
    consistency_score: float   # % of folds with positive total return


class WalkForwardValidator:
    """
    Splits historical data into anchored train+test windows and runs the
    BacktestEngine on each test fold.
    """

    def __init__(
        self,
        n_folds: int = 4,
        train_years: float = 1.0,
        test_months: int = 3,
    ):
        self.n_folds = n_folds
        self.train_years = train_years
        self.test_months = test_months

    def validate(self, ticker: str) -> WalkForwardResult:
        import yfinance as yf

        # Total data needed: 1y train + n_folds * test_months
        total_months = int(self.train_years * 12) + self.n_folds * self.test_months
        total_years = max(3, int(np.ceil(total_months / 12)) + 1)
        period = f"{total_years}y"

        logger.info(f"[WalkForward] {ticker}: {self.n_folds} folds, downloading {period}")

        df = yf.download(ticker, period=period, interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        spy_df = yf.download("SPY", period=period, interval="1d",
                              auto_adjust=True, progress=False)
        if isinstance(spy_df.columns, pd.MultiIndex):
            spy_df.columns = spy_df.columns.get_level_values(0)

        prices = df["Close"].dropna()
        opens = df["Open"].reindex(prices.index).ffill()
        spy_returns = spy_df["Close"].pct_change().dropna()

        train_days = int(self.train_years * 252)
        test_days = int(self.test_months * 21)

        engine = BacktestEngine()
        folds: List[WalkForwardFold] = []

        for fold_idx in range(self.n_folds):
            # Expanding anchor: train grows each fold
            test_start = train_days + fold_idx * test_days
            test_end = test_start + test_days

            if test_end > len(prices):
                logger.warning(f"[WalkForward] Fold {fold_idx + 1}: not enough data, skipping.")
                break

            # We give the engine the full history up to test_end so it has its
            # rolling 252-day lookback even at the start of the test window.
            fold_prices = prices.iloc[:test_end]
            fold_opens = opens.iloc[:test_end]

            # Only evaluate the test window portion
            if len(fold_prices) < 252 + test_days:
                continue

            train_start = str(prices.index[0].date())
            train_end = str(prices.index[test_start - 1].date())
            test_start_str = str(prices.index[test_start].date())
            test_end_str = str(prices.index[test_end - 1].date())

            try:
                result = engine._run_on_data(ticker, fold_prices, fold_opens, spy_returns)

                # Filter trades to only those in the test window
                test_trades = [
                    t for t in result.trades
                    if t.date >= test_start_str
                ]

                # Rebuild equity curve for test window only
                test_equity = result.equity_curve[result.equity_curve.index >= pd.Timestamp(test_start_str)]
                if test_equity.empty:
                    test_equity = result.equity_curve.tail(test_days)

                from backtest.metrics import compute_metrics
                test_metrics = compute_metrics(
                    equity_curve=test_equity,
                    trades=[{"return_pct": t.return_pct} for t in test_trades],
                    benchmark_returns=spy_returns.reindex(test_equity.index).dropna(),
                )

                correct = sum(
                    1 for t in test_trades
                    if (t.direction == "LONG" and t.return_pct > 0)
                    or (t.direction == "SHORT" and t.return_pct < 0)
                )
                acc = correct / len(test_trades) * 100 if test_trades else 50.0

                folds.append(WalkForwardFold(
                    fold=fold_idx + 1,
                    train_period=f"{train_start} → {train_end}",
                    test_period=f"{test_start_str} → {test_end_str}",
                    metrics=test_metrics,
                    signal_accuracy=round(acc, 1),
                    n_trades=len(test_trades),
                ))
            except Exception as e:
                logger.error(f"[WalkForward] Fold {fold_idx + 1} failed: {e}")

        if not folds:
            raise ValueError(f"Walk-forward validation yielded no valid folds for {ticker}.")

        avg_sharpe = float(np.mean([f.metrics.sharpe_ratio for f in folds]))
        avg_mdd = float(np.mean([f.metrics.max_drawdown_pct for f in folds]))
        avg_acc = float(np.mean([f.signal_accuracy for f in folds]))
        avg_cagr = float(np.mean([f.metrics.cagr_pct for f in folds]))
        consistency = sum(1 for f in folds if f.metrics.total_return_pct > 0) / len(folds) * 100

        return WalkForwardResult(
            ticker=ticker,
            n_folds=len(folds),
            folds=folds,
            avg_sharpe=round(avg_sharpe, 3),
            avg_max_drawdown_pct=round(avg_mdd, 2),
            avg_signal_accuracy=round(avg_acc, 1),
            avg_cagr_pct=round(avg_cagr, 2),
            consistency_score=round(consistency, 1),
        )
