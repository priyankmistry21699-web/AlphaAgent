"""
AlphaAgent — Performance Metrics

Computes standard quantitative finance performance metrics from an
equity curve and trade log.

Metrics:
  Sharpe, Sortino, Calmar, Omega ratio, Max Drawdown (trading days & calendar),
  Alpha, Beta, Information Ratio, VaR/CVaR from equity curve,
  monthly returns matrix (for heatmap rendering in the dashboard).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class PerformanceMetrics:
    total_return_pct: float
    cagr_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration_days: int     # trading days in deepest drawdown
    win_rate_pct: float
    profit_factor: float
    total_trades: int
    avg_trade_return_pct: float
    alpha_vs_spy: float
    beta_vs_spy: float
    information_ratio: float
    volatility_ann_pct: float
    var_95_pct: float = 0.0             # 1-day 95% VaR from equity curve returns
    cvar_95_pct: float = 0.0            # 1-day 95% CVaR (Expected Shortfall)
    monthly_returns: Dict[str, float] = field(default_factory=dict)  # "YYYY-MM" → pct


def compute_metrics(
    equity_curve: pd.Series,
    trades: List[dict],
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.05,
) -> PerformanceMetrics:
    """
    Computes performance metrics from a daily equity curve.

    equity_curve:       pd.Series indexed by date, values are portfolio NAV
    trades:             list of dicts with 'return_pct' key (percent, not decimal)
    benchmark_returns:  daily benchmark return series (e.g. SPY), aligned to equity_curve
    risk_free_rate:     annual risk-free rate
    """
    if len(equity_curve) < 2:
        return _empty_metrics()

    daily_returns = equity_curve.pct_change().dropna()
    n_years = len(equity_curve) / 252.0

    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0
    cagr = float((1 + total_return) ** (1 / max(n_years, 0.01)) - 1)

    vol_daily = float(daily_returns.std())
    vol_ann = vol_daily * np.sqrt(252)

    rf_daily = float((1 + risk_free_rate) ** (1 / 252) - 1)
    excess = daily_returns - rf_daily
    sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0

    downside = daily_returns[daily_returns < rf_daily] - rf_daily
    downside_std = float(downside.std()) if len(downside) > 1 else vol_daily
    sortino = float(excess.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0

    # ── Omega ratio: E[excess above threshold] / E[shortfall below threshold] ──
    above = daily_returns[daily_returns > rf_daily] - rf_daily
    below = rf_daily - daily_returns[daily_returns <= rf_daily]
    omega = float(above.sum() / below.sum()) if below.sum() > 0 else float("inf")
    omega = min(omega, 99.0)

    # ── Drawdown ──────────────────────────────────────────────────────────────
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    max_dd = float(drawdown.min())

    dd_duration = 0
    current_streak = 0
    for in_dd in (drawdown < 0):
        if in_dd:
            current_streak += 1
            dd_duration = max(dd_duration, current_streak)
        else:
            current_streak = 0

    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    # ── Trade stats ───────────────────────────────────────────────────────────
    trade_returns = [t.get("return_pct", 0.0) for t in trades]
    n_trades = len(trade_returns)
    wins = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r <= 0]
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 50.0
    avg_trade_ret = float(np.mean(trade_returns)) if trade_returns else 0.0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # ── VaR / CVaR (historical simulation from daily equity returns) ──────────
    sorted_ret = np.sort(daily_returns.values)
    var_idx = int(np.floor(0.05 * len(sorted_ret)))
    var_95 = float(sorted_ret[var_idx]) * 100 if var_idx > 0 else 0.0
    cvar_95 = float(sorted_ret[:var_idx].mean()) * 100 if var_idx > 0 else 0.0

    # ── Alpha, Beta, Information Ratio ────────────────────────────────────────
    alpha, beta, info_ratio = 0.0, 1.0, 0.0
    if benchmark_returns is not None and len(benchmark_returns) > 10:
        aligned = pd.concat([daily_returns, benchmark_returns], axis=1).dropna()
        if len(aligned) > 10:
            port_r = aligned.iloc[:, 0].values
            bench_r = aligned.iloc[:, 1].values
            bench_var = float(np.var(bench_r))
            beta = float(np.cov(port_r, bench_r)[0, 1] / bench_var) if bench_var > 0 else 1.0
            alpha = float(np.mean(port_r) - beta * np.mean(bench_r)) * 252
            active_ret = port_r - bench_r
            ir_std = float(np.std(active_ret))
            info_ratio = float(np.mean(active_ret) / ir_std * np.sqrt(252)) if ir_std > 0 else 0.0

    # ── Monthly returns heatmap ───────────────────────────────────────────────
    monthly_returns: Dict[str, float] = {}
    try:
        monthly_nav = equity_curve.resample("ME").last()
        monthly_ret = monthly_nav.pct_change().dropna()
        for dt, ret in monthly_ret.items():
            key = dt.strftime("%Y-%m")
            monthly_returns[key] = round(float(ret) * 100, 2)
    except Exception:
        pass

    return PerformanceMetrics(
        total_return_pct=round(total_return * 100, 2),
        cagr_pct=round(cagr * 100, 2),
        sharpe_ratio=round(sharpe, 3),
        sortino_ratio=round(sortino, 3),
        calmar_ratio=round(calmar, 3),
        omega_ratio=round(omega, 3),
        max_drawdown_pct=round(max_dd * 100, 2),
        max_drawdown_duration_days=dd_duration,
        win_rate_pct=round(win_rate, 1),
        profit_factor=round(min(profit_factor, 99.0), 3),
        total_trades=n_trades,
        avg_trade_return_pct=round(avg_trade_ret, 4),
        alpha_vs_spy=round(alpha * 100, 3),
        beta_vs_spy=round(beta, 3),
        information_ratio=round(info_ratio, 3),
        volatility_ann_pct=round(vol_ann * 100, 2),
        var_95_pct=round(var_95, 3),
        cvar_95_pct=round(cvar_95, 3),
        monthly_returns=monthly_returns,
    )


def _empty_metrics() -> PerformanceMetrics:
    return PerformanceMetrics(
        total_return_pct=0.0, cagr_pct=0.0, sharpe_ratio=0.0,
        sortino_ratio=0.0, calmar_ratio=0.0, omega_ratio=1.0,
        max_drawdown_pct=0.0, max_drawdown_duration_days=0,
        win_rate_pct=50.0, profit_factor=1.0, total_trades=0,
        avg_trade_return_pct=0.0, alpha_vs_spy=0.0, beta_vs_spy=1.0,
        information_ratio=0.0, volatility_ann_pct=0.0,
        var_95_pct=0.0, cvar_95_pct=0.0, monthly_returns={},
    )
