"""
AlphaAgent CLI

Rich-powered command-line interface for running stock analysis,
backtesting, universe scanning, portfolio optimization, and the
agent performance leaderboard.

Usage:
    python cli.py AAPL                          # signal analysis
    python cli.py AAPL --backtest               # quant backtest
    python cli.py AAPL --backtest --wf          # + walk-forward validation
    python cli.py --scan mega_cap               # universe scan
    python cli.py --leaderboard                 # agent leaderboard
    python cli.py --optimize AAPL MSFT NVDA     # portfolio optimization
    python cli.py AAPL --agents technical fundamental
    python cli.py NVDA --json
"""

import argparse
import json
import logging
import sys
import time
from typing import Optional, List

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from rich import box
from rich.columns import Columns
from rich.align import Align

console = Console()

logging.basicConfig(level=logging.WARNING)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _direction_badge(direction: str) -> Text:
    color_map = {"LONG": "bold green", "SHORT": "bold red", "HOLD": "bold yellow"}
    return Text(f" {direction} ", style=color_map.get(direction, "white"))


def _confidence_bar(pct: float) -> str:
    filled = int(pct / 5)
    empty = 20 - filled
    return f"[green]{'█' * filled}[/green][dim]{'░' * empty}[/dim] {pct:.0f}%"


def _prob_color(p: float) -> str:
    if p > 0.60:
        return "green"
    if p < 0.40:
        return "red"
    return "yellow"


def _metric_color(val: float, good_positive: bool = True) -> str:
    if good_positive:
        return "green" if val > 0 else "red"
    return "red" if val > 0 else "green"


# ── Signal Analysis (existing) ────────────────────────────────────────────────

def run_analysis(ticker: str, output_json: bool = False, agent_filter: Optional[list] = None):
    """Run full AlphaAgent analysis pipeline and display results."""
    from data.market import MarketData
    from agents.registry import AgentRegistry
    from orchestrator.graph import build_alpha_graph, AlphaGraphState

    ticker = ticker.upper()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"Analyzing [bold cyan]{ticker}[/bold cyan]...", total=None)

        try:
            market_data = MarketData(ticker)
            registry = AgentRegistry()
            graph = build_alpha_graph()

            state = AlphaGraphState(
                ticker=ticker,
                market_data=market_data,
                registry=registry,
            )
            result_state = graph.invoke(state)
            progress.update(task, completed=True)

        except Exception as e:
            progress.stop()
            console.print(f"[bold red]ERROR:[/bold red] {e}")
            sys.exit(1)

    final = result_state.get("final_signal") if isinstance(result_state, dict) else None
    if hasattr(result_state, "final_signal"):
        final = result_state.final_signal

    if not final:
        console.print("[bold red]No result returned from orchestrator.[/bold red]")
        sys.exit(1)

    packet = final["packet"]
    prob_up = final["probability_up"]
    entropy = final.get("entropy", 0.0)
    multiplier = final.get("multiplier", 1.0)

    if output_json:
        out = {
            "ticker": ticker,
            "direction": packet.direction.value,
            "conviction_pct": packet.conviction_pct,
            "probability_up": prob_up,
            "confidence": packet.confidence.value,
            "entropy": entropy,
            "multiplier": multiplier,
            "warnings": packet.warnings,
            "agents": [
                {
                    "name": r.agent_name,
                    "vote": r.vote.value if r.vote else None,
                    "probability_up": r.probability_up,
                    "confidence": r.confidence,
                    "reasoning": r.reasoning,
                }
                for r in packet.agent_results
            ],
        }
        console.print(json.dumps(out, indent=2))
        return

    # ── Header ────────────────────────────────────────────────────────────────
    console.print()
    console.rule(f"[bold cyan]AlphaAgent — {ticker}[/bold cyan]")
    console.print()

    # ── Signal Panel ──────────────────────────────────────────────────────────
    signal_table = Table.grid(expand=True, padding=(0, 2))
    signal_table.add_column(justify="center")
    signal_table.add_column(justify="center")
    signal_table.add_column(justify="center")
    signal_table.add_column(justify="center")

    prob_col = _prob_color(prob_up)
    signal_table.add_row(
        f"[bold]Direction[/bold]\n{_direction_badge(packet.direction.value)}",
        f"[bold]Probability Up[/bold]\n[{prob_col} bold]{prob_up * 100:.1f}%[/{prob_col} bold]",
        f"[bold]Conviction[/bold]\n[cyan bold]{packet.conviction_pct:.1f}%[/cyan bold]",
        f"[bold]Confidence[/bold]\n[magenta bold]{packet.confidence.value}[/magenta bold]",
    )
    console.print(Panel(signal_table, title="[bold]Final Signal[/bold]", border_style="cyan", box=box.ROUNDED))

    # ── Position Sizing ───────────────────────────────────────────────────────
    size_table = Table.grid(expand=True, padding=(0, 2))
    size_table.add_column(justify="center")
    size_table.add_column(justify="center")
    size_table.add_column(justify="center")

    size_table.add_row(
        f"[bold]Position Multiplier[/bold]\n[yellow]{multiplier:.1f}x[/yellow]",
        f"[bold]Signal Entropy[/bold]\n[{'red' if entropy > 0.8 else 'green'}]{entropy:.3f}[/]",
        f"[bold]Regime[/bold]\n[blue]{packet.regime.current_regime.value if packet.regime else 'N/A'}[/blue]",
    )
    console.print(Panel(size_table, title="[bold]Risk & Position[/bold]", border_style="yellow", box=box.ROUNDED))

    # ── Holding Period ────────────────────────────────────────────────────────
    if packet.holding_period:
        hp = packet.holding_period
        hp_table = Table.grid(expand=True, padding=(0, 2))
        hp_table.add_column(justify="center")
        hp_table.add_column(justify="center")
        hp_table.add_column(justify="center")
        hp_table.add_row(
            f"[bold]Signal Half-Life[/bold]\n[white]{hp.half_life_days:.1f} days[/white]",
            f"[bold]Optimal Hold[/bold]\n[white]{hp.optimal_hold_min:.0f} – {hp.optimal_hold_max:.0f} days[/white]",
            f"[bold]Signal Strength[/bold]\n{_confidence_bar(hp.signal_strength * 100)}",
        )
        console.print(Panel(hp_table, title="[bold]Signal Decay[/bold]", border_style="blue", box=box.ROUNDED))

    # ── Agent Breakdown ───────────────────────────────────────────────────────
    agent_table = Table(
        title="Agent Breakdown", box=box.ROUNDED,
        show_header=True, header_style="bold magenta",
        border_style="dim", expand=True,
    )
    agent_table.add_column("Agent", style="bold", no_wrap=True)
    agent_table.add_column("Vote", justify="center")
    agent_table.add_column("P(Up)", justify="right")
    agent_table.add_column("Confidence", justify="right")
    agent_table.add_column("Reasoning", max_width=60)

    for r in packet.agent_results:
        if agent_filter and r.agent_name not in agent_filter:
            continue
        vote_str = r.vote.value if r.vote else "—"
        vote_color = {"LONG": "green", "SHORT": "red", "HOLD": "yellow"}.get(vote_str, "white")
        pc = _prob_color(r.probability_up)
        reasoning_short = (r.reasoning[:80] + "…") if len(r.reasoning) > 80 else r.reasoning
        agent_table.add_row(
            r.agent_name,
            f"[{vote_color}]{vote_str}[/{vote_color}]",
            f"[{pc}]{r.probability_up * 100:.1f}%[/{pc}]",
            f"{r.confidence * 100:.0f}%",
            reasoning_short,
        )
    console.print(agent_table)

    # ── Factor Scores ─────────────────────────────────────────────────────────
    all_factors = [
        (r.agent_name, key, fs)
        for r in packet.agent_results
        for key, fs in r.factor_scores.items()
    ]
    if all_factors:
        factor_table = Table(
            title="Factor Scores", box=box.SIMPLE,
            header_style="bold blue", expand=True,
        )
        factor_table.add_column("Agent", style="dim")
        factor_table.add_column("Factor", style="bold")
        factor_table.add_column("Score", justify="right")
        factor_table.add_column("Interpretation", max_width=60)

        for agent_name, key, fs in all_factors:
            sc = "green" if fs.score >= 60 else "red" if fs.score <= 40 else "yellow"
            factor_table.add_row(
                agent_name, fs.name,
                f"[{sc}]{fs.score:.0f}[/{sc}]",
                fs.interpretation,
            )
        console.print(factor_table)

    # ── Warnings ──────────────────────────────────────────────────────────────
    for w in packet.warnings:
        console.print(f"[bold yellow]  WARNING:[/bold yellow] {w}")

    console.print()
    console.rule("[dim]AlphaAgent — End of Report[/dim]")
    console.print()


# ── Backtest ──────────────────────────────────────────────────────────────────

def run_backtest(ticker: str, period: str = "3y", walk_forward: bool = False):
    """Run the quant-only backtester and display a performance report."""
    from backtest.engine import BacktestEngine

    ticker = ticker.upper()
    console.print()
    console.rule(f"[bold cyan]Backtest — {ticker} ({period})[/bold cyan]")
    console.print()

    with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"),
                  console=console, transient=True) as progress:
        task = progress.add_task(f"Running backtest for {ticker}...", total=None)
        try:
            engine = BacktestEngine()
            result = engine.run(ticker, period=period)
            progress.update(task, completed=True)
        except Exception as e:
            progress.stop()
            console.print(f"[bold red]Backtest failed:[/bold red] {e}")
            return

    m = result.metrics

    # ── Performance Panel ─────────────────────────────────────────────────────
    perf_table = Table.grid(expand=True, padding=(0, 3))
    for _ in range(4):
        perf_table.add_column(justify="center")

    ret_c = _metric_color(m.total_return_pct)
    cagr_c = _metric_color(m.cagr_pct)
    alpha_c = _metric_color(m.alpha_vs_spy)

    perf_table.add_row(
        f"[bold]Total Return[/bold]\n[{ret_c} bold]{m.total_return_pct:+.2f}%[/{ret_c} bold]",
        f"[bold]CAGR[/bold]\n[{cagr_c} bold]{m.cagr_pct:+.2f}%[/{cagr_c} bold]",
        f"[bold]Sharpe[/bold]\n[cyan bold]{m.sharpe_ratio:.3f}[/cyan bold]",
        f"[bold]Sortino[/bold]\n[cyan bold]{m.sortino_ratio:.3f}[/cyan bold]",
    )
    perf_table.add_row(
        f"[bold]Max Drawdown[/bold]\n[red bold]{m.max_drawdown_pct:.2f}%[/red bold]",
        f"[bold]Calmar[/bold]\n[cyan]{m.calmar_ratio:.3f}[/cyan]",
        f"[bold]Win Rate[/bold]\n[yellow]{m.win_rate_pct:.1f}%[/yellow]",
        f"[bold]Profit Factor[/bold]\n[yellow]{m.profit_factor:.2f}x[/yellow]",
    )
    perf_table.add_row(
        f"[bold]Alpha vs SPY[/bold]\n[{alpha_c} bold]{m.alpha_vs_spy:+.2f}%[/{alpha_c} bold]",
        f"[bold]Beta[/bold]\n[white]{m.beta_vs_spy:.3f}[/white]",
        f"[bold]Info Ratio[/bold]\n[white]{m.information_ratio:.3f}[/white]",
        f"[bold]Annualised Vol[/bold]\n[white]{m.volatility_ann_pct:.2f}%[/white]",
    )
    console.print(Panel(perf_table, title="[bold]Performance Metrics[/bold]",
                        border_style="cyan", box=box.ROUNDED))

    # ── Trade Summary ─────────────────────────────────────────────────────────
    summary_table = Table.grid(expand=True, padding=(0, 3))
    for _ in range(3):
        summary_table.add_column(justify="center")
    summary_table.add_row(
        f"[bold]Trades[/bold]\n[white]{result.n_signals}[/white]",
        f"[bold]Signal Accuracy[/bold]\n[yellow]{result.signal_accuracy:.1f}%[/yellow]",
        f"[bold]Period[/bold]\n[dim]{result.start_date} → {result.end_date}[/dim]",
    )
    console.print(Panel(summary_table, title="[bold]Trade Summary[/bold]",
                        border_style="yellow", box=box.ROUNDED))

    # ── Recent Trades Table ───────────────────────────────────────────────────
    if result.trades:
        trade_table = Table(title="Recent Trades (last 10)", box=box.SIMPLE,
                            header_style="bold magenta", expand=True)
        trade_table.add_column("Date")
        trade_table.add_column("Dir", justify="center")
        trade_table.add_column("Entry", justify="right")
        trade_table.add_column("Exit", justify="right")
        trade_table.add_column("Return", justify="right")
        trade_table.add_column("P(Up)", justify="right")
        trade_table.add_column("Regime")

        for t in result.trades[-10:]:
            rc = "green" if t.return_pct > 0 else "red"
            dc = {"LONG": "green", "SHORT": "red"}.get(t.direction, "yellow")
            trade_table.add_row(
                t.date,
                f"[{dc}]{t.direction}[/{dc}]",
                f"${t.entry_price:.2f}",
                f"${t.exit_price:.2f}",
                f"[{rc}]{t.return_pct:+.2f}%[/{rc}]",
                f"{t.probability * 100:.1f}%",
                t.regime,
            )
        console.print(trade_table)

    # ── Walk-Forward ──────────────────────────────────────────────────────────
    if walk_forward:
        _display_walk_forward(ticker)

    console.print()
    console.rule("[dim]AlphaAgent — Backtest Complete[/dim]")
    console.print()


def _display_walk_forward(ticker: str):
    from backtest.walk_forward import WalkForwardValidator

    console.print()
    with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"),
                  console=console, transient=True) as progress:
        task = progress.add_task("Running walk-forward validation...", total=None)
        try:
            validator = WalkForwardValidator(n_folds=4)
            wf = validator.validate(ticker)
            progress.update(task, completed=True)
        except Exception as e:
            progress.stop()
            console.print(f"[bold red]Walk-forward failed:[/bold red] {e}")
            return

    wf_table = Table(title="Walk-Forward Results", box=box.ROUNDED,
                     header_style="bold blue", expand=True)
    wf_table.add_column("Fold", justify="center")
    wf_table.add_column("Test Period")
    wf_table.add_column("Return", justify="right")
    wf_table.add_column("Sharpe", justify="right")
    wf_table.add_column("MaxDD", justify="right")
    wf_table.add_column("Win Rate", justify="right")
    wf_table.add_column("Accuracy", justify="right")
    wf_table.add_column("Trades", justify="center")

    for f in wf.folds:
        rc = "green" if f.metrics.total_return_pct > 0 else "red"
        wf_table.add_row(
            str(f.fold),
            f.test_period,
            f"[{rc}]{f.metrics.total_return_pct:+.2f}%[/{rc}]",
            f"{f.metrics.sharpe_ratio:.3f}",
            f"{f.metrics.max_drawdown_pct:.2f}%",
            f"{f.metrics.win_rate_pct:.1f}%",
            f"{f.signal_accuracy:.1f}%",
            str(f.n_trades),
        )

    console.print(wf_table)

    summary = Table.grid(expand=True, padding=(0, 3))
    for _ in range(5):
        summary.add_column(justify="center")
    sc = "green" if wf.avg_cagr_pct > 0 else "red"
    summary.add_row(
        f"[bold]Avg CAGR[/bold]\n[{sc} bold]{wf.avg_cagr_pct:+.2f}%[/{sc} bold]",
        f"[bold]Avg Sharpe[/bold]\n[cyan]{wf.avg_sharpe:.3f}[/cyan]",
        f"[bold]Avg MaxDD[/bold]\n[red]{wf.avg_max_drawdown_pct:.2f}%[/red]",
        f"[bold]Avg Accuracy[/bold]\n[yellow]{wf.avg_signal_accuracy:.1f}%[/yellow]",
        f"[bold]Consistency[/bold]\n[white]{wf.consistency_score:.0f}% positive folds[/white]",
    )
    console.print(Panel(summary, title="[bold]Walk-Forward Summary[/bold]",
                        border_style="blue", box=box.ROUNDED))


# ── Universe Scanner ──────────────────────────────────────────────────────────

def run_scan(universe: str = "mega_cap", top_n: int = 10, direction: Optional[str] = None):
    """Scan a universe of tickers and display top signals."""
    from data.universe import UniverseManager

    console.print()
    console.rule(f"[bold cyan]Universe Scan — {universe.upper()}[/bold cyan]")
    console.print()

    um = UniverseManager()
    tickers = um.get_universe(universe)
    console.print(f"[dim]Scanning {len(tickers)} tickers...[/dim]")
    console.print()

    with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"),
                  console=console, transient=True) as progress:
        task = progress.add_task(f"Running signals on {universe}...", total=None)
        try:
            results = um.scan(universe=universe, top_n=top_n, direction_filter=direction)
            progress.update(task, completed=True)
        except Exception as e:
            progress.stop()
            console.print(f"[bold red]Scan failed:[/bold red] {e}")
            return

    scan_table = Table(title=f"Top {top_n} Signals — {universe}", box=box.ROUNDED,
                       header_style="bold magenta", expand=True)
    scan_table.add_column("Rank", justify="center", width=5)
    scan_table.add_column("Ticker", style="bold", no_wrap=True)
    scan_table.add_column("Direction", justify="center")
    scan_table.add_column("P(Up)", justify="right")
    scan_table.add_column("Conviction", justify="right")
    scan_table.add_column("Regime")
    scan_table.add_column("Warnings", max_width=50)

    for rank, r in enumerate(results, 1):
        dc = {"LONG": "green", "SHORT": "red"}.get(r.direction, "yellow")
        pc = _prob_color(r.probability)
        conv_bar = _confidence_bar(r.conviction_pct)
        warn_str = " | ".join(r.warnings[:2]) if r.warnings else "—"
        scan_table.add_row(
            str(rank),
            r.ticker,
            f"[{dc} bold]{r.direction}[/{dc} bold]",
            f"[{pc}]{r.probability * 100:.1f}%[/{pc}]",
            conv_bar,
            r.regime,
            f"[dim]{warn_str}[/dim]",
        )

    console.print(scan_table)
    console.print()
    console.rule("[dim]AlphaAgent — Scan Complete[/dim]")
    console.print()


# ── Agent Leaderboard ─────────────────────────────────────────────────────────

def run_leaderboard(window_days: int = 90):
    """Display the agent performance leaderboard."""
    from quant_engine.leaderboard import AgentLeaderboard

    console.print()
    console.rule("[bold cyan]Agent Performance Leaderboard[/bold cyan]")
    console.print()

    lb = AgentLeaderboard()
    result = lb.evaluate_performance(window_days=window_days)

    lb_table = Table(title=f"Agent Rankings ({window_days}-day window)",
                     box=box.ROUNDED, header_style="bold yellow", expand=True)
    lb_table.add_column("Rank", justify="center", width=5)
    lb_table.add_column("Agent", style="bold")
    lb_table.add_column("Predictions", justify="right")
    lb_table.add_column("Accuracy", justify="right")
    lb_table.add_column("Brier Score", justify="right")
    lb_table.add_column("Info Ratio", justify="right")
    lb_table.add_column("Confidence", justify="right")

    medals = {1: "[gold1]#1[/gold1]", 2: "[grey82]#2[/grey82]", 3: "[dark_orange]#3[/dark_orange]"}

    for s in result.scores:
        rank_str = medals.get(s.rank, f"#{s.rank}")
        acc_c = "green" if s.directional_accuracy >= 55 else "red" if s.directional_accuracy < 50 else "yellow"
        ir_c = "green" if s.information_ratio > 0.5 else "red" if s.information_ratio < 0 else "yellow"
        brier_c = "green" if s.brier_score < 0.22 else "red" if s.brier_score > 0.26 else "yellow"

        lb_table.add_row(
            rank_str,
            s.agent_name,
            str(s.n_predictions),
            f"[{acc_c} bold]{s.directional_accuracy:.1f}%[/{acc_c} bold]",
            f"[{brier_c}]{s.brier_score:.4f}[/{brier_c}]",
            f"[{ir_c}]{s.information_ratio:.3f}[/{ir_c}]",
            f"{s.avg_confidence * 100:.0f}%",
        )

    console.print(lb_table)

    if result.total_signals_evaluated == 0:
        console.print("[dim italic]Note: showing demo data — run signals to build real history.[/dim italic]")

    console.print()
    console.rule("[dim]AlphaAgent — Leaderboard[/dim]")
    console.print()


# ── Portfolio Optimizer ───────────────────────────────────────────────────────

def run_optimize(tickers: List[str], method: str = "max_sharpe", period: str = "1y"):
    """Run portfolio optimization on a list of tickers."""
    from quant_engine.portfolio_optimizer import PortfolioOptimizer

    tickers = [t.upper() for t in tickers]
    console.print()
    console.rule(f"[bold cyan]Portfolio Optimizer — {method.upper()}[/bold cyan]")
    console.print()

    with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"),
                  console=console, transient=True) as progress:
        task = progress.add_task(f"Optimizing {len(tickers)} assets...", total=None)
        try:
            optimizer = PortfolioOptimizer()
            result = optimizer.optimize(tickers, method=method, period=period)
            progress.update(task, completed=True)
        except Exception as e:
            progress.stop()
            console.print(f"[bold red]Optimization failed:[/bold red] {e}")
            return

    # ── Weights Table ─────────────────────────────────────────────────────────
    weights_table = Table(title=f"Optimal Weights ({result.method})",
                          box=box.ROUNDED, header_style="bold magenta", expand=True)
    weights_table.add_column("Ticker", style="bold")
    weights_table.add_column("Weight", justify="right")
    weights_table.add_column("Allocation Bar")

    for ticker, w in sorted(result.weights.items(), key=lambda x: -x[1]):
        bar = _confidence_bar(w * 100)
        weights_table.add_row(ticker, f"{w * 100:.2f}%", bar)

    console.print(weights_table)

    # ── Portfolio Metrics ─────────────────────────────────────────────────────
    ret_c = _metric_color(result.expected_return_ann)
    metrics_table = Table.grid(expand=True, padding=(0, 3))
    for _ in range(3):
        metrics_table.add_column(justify="center")
    metrics_table.add_row(
        f"[bold]Expected Return[/bold]\n[{ret_c} bold]{result.expected_return_ann:+.2f}%[/{ret_c} bold]",
        f"[bold]Expected Vol[/bold]\n[yellow]{result.expected_vol_ann:.2f}%[/yellow]",
        f"[bold]Sharpe Ratio[/bold]\n[cyan bold]{result.sharpe_ratio:.3f}[/cyan bold]",
    )
    console.print(Panel(metrics_table, title="[bold]Portfolio Metrics (Annualised)[/bold]",
                        border_style="cyan", box=box.ROUNDED))

    console.print()
    console.rule("[dim]AlphaAgent — Optimization Complete[/dim]")
    console.print()


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AlphaAgent — Quantitative Stock Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py AAPL                              # signal analysis
  python cli.py AAPL --backtest                   # backtest
  python cli.py AAPL --backtest --wf              # backtest + walk-forward
  python cli.py --scan mega_cap                   # universe scan
  python cli.py --leaderboard                     # agent leaderboard
  python cli.py --optimize AAPL MSFT NVDA         # portfolio optimization
  python cli.py NVDA --json                       # machine-readable output
        """,
    )

    # Core signal arguments
    parser.add_argument("ticker", type=str, nargs="?", help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("--json", action="store_true", dest="output_json", help="Output raw JSON")
    parser.add_argument("--agents", nargs="+", metavar="AGENT",
                        help="Filter output to specific agents")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Backtest arguments
    parser.add_argument("--backtest", action="store_true", help="Run quant backtest")
    parser.add_argument("--period", type=str, default="3y",
                        help="Data period for backtest (default: 3y)")
    parser.add_argument("--wf", action="store_true", dest="walk_forward",
                        help="Include walk-forward validation with backtest")

    # Universe scan
    parser.add_argument("--scan", type=str, metavar="UNIVERSE",
                        help="Scan a ticker universe (mega_cap, sp500_core, technology, ...)")
    parser.add_argument("--top", type=int, default=10,
                        help="Number of top results to show for scan (default: 10)")
    parser.add_argument("--direction", type=str, choices=["LONG", "SHORT"],
                        help="Filter scan by direction")

    # Leaderboard
    parser.add_argument("--leaderboard", action="store_true",
                        help="Show agent performance leaderboard")
    parser.add_argument("--days", type=int, default=90,
                        help="Evaluation window for leaderboard in days (default: 90)")

    # Portfolio optimization
    parser.add_argument("--optimize", nargs="+", metavar="TICKER",
                        help="Optimize portfolio weights (e.g., --optimize AAPL MSFT NVDA)")
    parser.add_argument("--method", type=str,
                        choices=["max_sharpe", "min_variance", "risk_parity", "signal_weighted"],
                        default="max_sharpe",
                        help="Optimization method (default: max_sharpe)")

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, force=True)

    # ── Route to the appropriate function ─────────────────────────────────────

    if args.leaderboard:
        run_leaderboard(window_days=args.days)

    elif args.scan:
        run_scan(universe=args.scan, top_n=args.top, direction=args.direction)

    elif args.optimize:
        run_optimize(tickers=args.optimize, method=args.method, period=args.period)

    elif args.backtest:
        if not args.ticker:
            console.print("[bold red]--backtest requires a ticker. E.g.: python cli.py AAPL --backtest[/bold red]")
            sys.exit(1)
        run_backtest(ticker=args.ticker, period=args.period, walk_forward=args.walk_forward)

    elif args.ticker:
        run_analysis(ticker=args.ticker, output_json=args.output_json, agent_filter=args.agents)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
