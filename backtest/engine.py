"""
AlphaAgent — Backtesting Engine

Replays quantitative signals on historical OHLCV data to measure strategy
performance. Uses only the quant components (GARCH, HMM, momentum, mean
reversion) — not LLM-based agents — so historical replay is fast and
reproducible without API costs.

Signal pipeline per rolling window:
  1. Jegadeesh-Titman 12M-1M momentum
  2. RSI(14) mean reversion bias
  3. GARCH regime (volatility-scaled Kelly sizing)
  4. HMM market regime (Bull/Bear/Crisis bias)
  5. Bollinger Band z-score (mean reversion)
  6. Bayesian fusion → final probability
  7. Kelly fraction → position size (capped at max_position_pct)

Execution model:
  - Entry at next-day open
  - Exit after hold_days calendar days at close
  - Round-trip transaction cost: base 5 bps + bid-ask spread (2 bps)
  - Short-side borrow cost: 50 bps annualised (≈ 0.97 bps per 5-day hold)
  - Non-overlapping positions (i advances by hold_days after each trade)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from backtest.metrics import PerformanceMetrics, compute_metrics

logger = logging.getLogger(__name__)

LOOKBACK_DAYS = 252
MIN_HISTORY = LOOKBACK_DAYS + 30

# Short borrow cost: 50 bps / year → per 5-day hold
_BORROW_COST_ANN = 0.0050
_BID_ASK_BPS = 2.0           # additional 2 bps round-trip for market impact


@dataclass
class BacktestTrade:
    date: str
    direction: str
    entry_price: float
    exit_price: float
    return_pct: float
    probability: float
    position_pct: float
    regime: str
    hold_days: int
    gross_return_pct: float = 0.0
    cost_bps: float = 0.0


@dataclass
class BacktestResult:
    ticker: str
    start_date: str
    end_date: str
    equity_curve: pd.Series
    trades: List[BacktestTrade]
    metrics: PerformanceMetrics
    n_signals: int = 0
    signal_accuracy: float = 0.0


class BacktestEngine:
    """Rolling-window backtester for quantitative signals."""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        hold_days: int = 5,
        max_position_pct: float = 0.20,
        transaction_cost_bps: float = 5.0,
        borrow_cost_ann: float = _BORROW_COST_ANN,
        bid_ask_bps: float = _BID_ASK_BPS,
    ):
        self.initial_capital = initial_capital
        self.hold_days = hold_days
        self.max_position_pct = max_position_pct
        self.tc = transaction_cost_bps / 10_000          # base round-trip
        self.bid_ask = bid_ask_bps / 10_000              # market-impact round-trip
        self.borrow_cost_per_hold = borrow_cost_ann * hold_days / 252  # per hold period

    def run(self, ticker: str, period: str = "3y") -> BacktestResult:
        """Downloads data and runs the backtest for the given period."""
        import yfinance as yf

        logger.info(f"[Backtest] {ticker} — downloading {period} of data")

        df = yf.download(ticker, period=period, interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty or len(df) < MIN_HISTORY + self.hold_days:
            raise ValueError(
                f"Insufficient data for {ticker}: got {len(df)} rows, "
                f"need {MIN_HISTORY + self.hold_days}"
            )

        spy_df = yf.download("SPY", period=period, interval="1d",
                              auto_adjust=True, progress=False)
        if isinstance(spy_df.columns, pd.MultiIndex):
            spy_df.columns = spy_df.columns.get_level_values(0)
        spy_returns = spy_df["Close"].pct_change().dropna()

        return self._run_on_data(ticker, df["Close"], df["Open"], spy_returns)

    def _run_on_data(
        self,
        ticker: str,
        prices: pd.Series,
        opens: pd.Series,
        spy_returns: pd.Series,
    ) -> BacktestResult:
        prices = prices.dropna()
        opens = opens.reindex(prices.index).ffill()

        capital = self.initial_capital
        equity_vals: List[float] = []
        equity_dates = []
        trades: List[BacktestTrade] = []

        i = LOOKBACK_DAYS
        while i < len(prices) - self.hold_days:
            window_prices = prices.iloc[i - LOOKBACK_DAYS:i]
            window_returns = window_prices.pct_change().dropna()

            prob, regime, kelly_frac = self._compute_signal(window_prices, window_returns)

            if prob > 0.55:
                direction = "LONG"
            elif prob < 0.45:
                direction = "SHORT"
            else:
                direction = "HOLD"
                equity_vals.append(capital)
                equity_dates.append(prices.index[i])
                i += 1
                continue

            entry_idx = min(i + 1, len(opens) - 1)
            exit_idx = min(i + self.hold_days, len(prices) - 1)
            entry_price = float(opens.iloc[entry_idx])
            exit_price = float(prices.iloc[exit_idx])

            pos_pct = min(kelly_frac, self.max_position_pct)
            position_capital = capital * pos_pct

            gross_ret = (exit_price - entry_price) / entry_price
            if direction == "SHORT":
                gross_ret = -gross_ret

            # Cost model: base TC + bid-ask spread, plus borrow fee for shorts
            round_trip_cost = self.tc + self.bid_ask
            if direction == "SHORT":
                round_trip_cost += self.borrow_cost_per_hold

            net_ret = gross_ret - round_trip_cost
            capital += position_capital * net_ret

            trades.append(BacktestTrade(
                date=str(prices.index[i].date()),
                direction=direction,
                entry_price=round(entry_price, 4),
                exit_price=round(exit_price, 4),
                return_pct=round(net_ret * 100, 4),
                gross_return_pct=round(gross_ret * 100, 4),
                cost_bps=round(round_trip_cost * 10_000, 2),
                probability=round(prob, 4),
                position_pct=round(pos_pct * 100, 2),
                regime=regime,
                hold_days=self.hold_days,
            ))

            for j in range(self.hold_days):
                idx = i + j
                if idx < len(prices):
                    equity_vals.append(capital)
                    equity_dates.append(prices.index[idx])

            i += self.hold_days

        if not equity_vals:
            equity_curve = pd.Series(
                [self.initial_capital],
                index=[prices.index[LOOKBACK_DAYS]],
            )
        else:
            equity_curve = pd.Series(equity_vals, index=equity_dates)
            equity_curve = equity_curve[~equity_curve.index.duplicated(keep="last")]

        bench = spy_returns.reindex(equity_curve.index).dropna()
        metrics = compute_metrics(
            equity_curve=equity_curve,
            trades=[{"return_pct": t.return_pct} for t in trades],
            benchmark_returns=bench,
        )

        correct = sum(
            1 for t in trades
            if (t.direction == "LONG" and t.return_pct > 0)
            or (t.direction == "SHORT" and t.return_pct < 0)
        )
        accuracy = correct / len(trades) * 100 if trades else 50.0

        return BacktestResult(
            ticker=ticker,
            start_date=str(prices.index[LOOKBACK_DAYS].date()),
            end_date=str(prices.index[-1].date()),
            equity_curve=equity_curve,
            trades=trades,
            metrics=metrics,
            n_signals=len(trades),
            signal_accuracy=round(accuracy, 1),
        )

    # ── Signal computation ────────────────────────────────────────────────────

    def _compute_signal(
        self,
        prices: pd.Series,
        returns: pd.Series,
    ) -> tuple:
        """Returns (probability_up, regime_label, kelly_fraction)."""
        prob_components: List[float] = []
        regime = "NORMAL"
        kelly_frac = 0.10

        # ── 1. Jegadeesh-Titman 12M-1M Momentum ──────────────────────────
        try:
            if len(prices) >= 252:
                jt = float(prices.iloc[-1] / prices.iloc[-252] - 1) - \
                     float(prices.iloc[-1] / prices.iloc[-21] - 1)
                if jt > 0.12:   prob_components.append(0.66)
                elif jt > 0.04: prob_components.append(0.58)
                elif jt < -0.12: prob_components.append(0.34)
                elif jt < -0.04: prob_components.append(0.42)
                else:            prob_components.append(0.50)
        except Exception:
            prob_components.append(0.50)

        # ── 2. RSI(14) ────────────────────────────────────────────────────
        try:
            delta = prices.diff().dropna()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain.iloc[-1] / loss.iloc[-1] if float(loss.iloc[-1]) > 0 else 100.0
            rsi = float(100 - 100 / (1 + rs))
            if rsi < 25:         prob_components.append(0.68)
            elif rsi < 35:       prob_components.append(0.60)
            elif rsi > 75:       prob_components.append(0.32)
            elif rsi > 65:       prob_components.append(0.40)
            else:                prob_components.append(0.50 + (50 - rsi) * 0.003)
        except Exception:
            prob_components.append(0.50)

        # ── 3. GARCH Volatility Regime ────────────────────────────────────
        try:
            from quant_engine.garch import GARCHModel
            garch = GARCHModel(returns)
            garch_res = garch.fit_and_forecast(horizon=5)
            regime_prob = {"LOW": 0.57, "NORMAL": 0.50, "HIGH": 0.44, "EXTREME": 0.38}
            kelly_map  = {"LOW": 0.15, "NORMAL": 0.10, "HIGH": 0.06, "EXTREME": 0.03}
            prob_components.append(regime_prob.get(garch_res.vol_regime, 0.50))
            kelly_frac = kelly_map.get(garch_res.vol_regime, 0.10)
            regime = garch_res.vol_regime
        except Exception:
            prob_components.append(0.50)

        # ── 4. HMM Regime ─────────────────────────────────────────────────
        try:
            from quant_engine.hmm import RegimeDetector
            if len(returns) >= 100:
                detector = RegimeDetector(n_states=3)
                hmm_res = detector.fit_predict(returns)
                hmm_prob = {"BULL": 0.61, "BEAR": 0.43, "CRISIS": 0.33}
                prob_components.append(hmm_prob.get(hmm_res.current_regime, 0.50))
                regime = hmm_res.current_regime
        except Exception:
            prob_components.append(0.50)

        # ── 5. Bollinger Band Z-Score ─────────────────────────────────────
        try:
            ma_20 = float(prices.rolling(20).mean().iloc[-1])
            std_20 = float(prices.rolling(20).std().iloc[-1])
            if std_20 > 0:
                z = (float(prices.iloc[-1]) - ma_20) / std_20
                if z < -2.0:   prob_components.append(0.65)
                elif z < -1.0: prob_components.append(0.57)
                elif z > 2.0:  prob_components.append(0.35)
                elif z > 1.0:  prob_components.append(0.43)
                else:          prob_components.append(0.50)
        except Exception:
            prob_components.append(0.50)

        # ── Bayesian Fusion ───────────────────────────────────────────────
        try:
            from quant_engine.bayesian import BayesianFusion
            fusion = BayesianFusion(prior=0.5)
            for p in prob_components:
                fusion.update(agent_prob=p, confidence=0.7, correlation=0.0)
            final_prob = fusion.posterior
        except Exception:
            final_prob = float(np.mean(prob_components)) if prob_components else 0.5

        return final_prob, regime, kelly_frac
