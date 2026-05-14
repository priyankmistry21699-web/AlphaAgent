"""
AlphaAgent — Portfolio Agent

Tracks the live paper portfolio: holdings, unrealized P&L, portfolio-level
risk (correlation, beta, max drawdown estimate), and rebalancing signals.

Unlike the other analysis agents, this agent does not vote on a single ticker.
It is invoked by the portfolio API endpoints to produce a portfolio-wide view.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class PositionView:
    ticker: str
    shares: float
    avg_price: float
    last_price: float
    market_value: float
    allocation_pct: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    beta: float = 1.0
    contribution_to_portfolio_vol: float = 0.0


@dataclass
class PortfolioView:
    total_value: float
    total_cost: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    positions: List[PositionView] = field(default_factory=list)

    # Portfolio-level risk metrics
    portfolio_beta: float = 1.0
    portfolio_vol_ann_pct: float = 0.0
    max_drawdown_estimate_pct: float = 0.0
    herfindahl_concentration: float = 0.0    # 0 = perfectly diversified, 1 = single stock

    # Correlation matrix summary
    avg_pairwise_correlation: float = 0.0

    # Rebalancing signals
    overweight_tickers: List[str] = field(default_factory=list)
    underweight_tickers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class PortfolioAgent:
    """
    Computes portfolio-level analytics for the current holdings.

    Usage:
        agent = PortfolioAgent()
        view = agent.analyze(holdings_list)
    """

    @property
    def MAX_SINGLE_POSITION_PCT(self):
        from config.settings_manager import settings
        return settings.get("portfolio.max_single_position_pct", 30.0)

    @property
    def MIN_SINGLE_POSITION_PCT(self):
        from config.settings_manager import settings
        return settings.get("portfolio.min_single_position_pct", 2.0)

    @property
    def LOOKBACK_PERIOD(self):
        from config.settings_manager import settings
        return settings.get("data.portfolio_lookback", "1y")

    def analyze(self, holdings: List[Dict]) -> PortfolioView:
        """
        Main entry point.

        Args:
            holdings: List of dicts from DatabaseManager.get_portfolio_performance()["positions"]
                      Each dict must have: ticker, shares, avg_price, last_price, market_value.

        Returns:
            PortfolioView with risk analytics and rebalancing signals.
        """
        if not holdings:
            return PortfolioView(
                total_value=0.0,
                total_cost=0.0,
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0,
                warnings=["Portfolio is empty."],
            )

        tickers = [h["ticker"] for h in holdings]
        total_value = sum(h["market_value"] for h in holdings)
        total_cost = sum(h["shares"] * h["avg_price"] for h in holdings)
        unrealized_pnl = total_value - total_cost
        unrealized_pnl_pct = (unrealized_pnl / total_cost * 100) if total_cost > 0 else 0.0

        # ── 1. Position views ─────────────────────────────────────────────────
        positions = []
        for h in holdings:
            alloc = h["market_value"] / total_value * 100 if total_value > 0 else 0.0
            upnl = (h["last_price"] - h["avg_price"]) * h["shares"]
            upnl_pct = (h["last_price"] - h["avg_price"]) / h["avg_price"] * 100 if h["avg_price"] > 0 else 0.0
            positions.append(PositionView(
                ticker=h["ticker"],
                shares=h["shares"],
                avg_price=h["avg_price"],
                last_price=h["last_price"],
                market_value=h["market_value"],
                allocation_pct=round(alloc, 2),
                unrealized_pnl=round(upnl, 2),
                unrealized_pnl_pct=round(upnl_pct, 2),
            ))

        warnings = []

        # ── 2. Historical returns for portfolio analytics ─────────────────────
        returns_df = self._fetch_returns(tickers)
        weights = np.array([h["market_value"] / total_value for h in holdings])

        beta = 1.0
        port_vol = 0.0
        avg_corr = 0.0
        max_dd_est = 0.0

        if returns_df is not None and not returns_df.empty and len(returns_df.columns) > 0:
            # Align weights to available tickers
            available = [t for t in tickers if t in returns_df.columns]
            avail_idx = [tickers.index(t) for t in available]
            sub_returns = returns_df[available].dropna(how="all")
            sub_weights = weights[avail_idx]
            if sub_weights.sum() > 0:
                sub_weights = sub_weights / sub_weights.sum()

            if len(sub_returns) > 20 and len(available) >= 1:
                cov = sub_returns.cov() * 252
                port_variance = float(sub_weights @ cov.values @ sub_weights)
                port_vol = np.sqrt(max(0.0, port_variance)) * 100

                # Simple drawdown estimate: 2 * annual vol (rough upper bound)
                max_dd_est = min(99.0, port_vol * 2)

                # Beta: weighted average of individual betas vs SPY
                spy_ret = self._fetch_spy_returns(len(sub_returns))
                if spy_ret is not None:
                    betas = []
                    for col in available:
                        col_ret = sub_returns[col].dropna()
                        aligned = col_ret.align(spy_ret, join="inner")[0]
                        spy_aligned = spy_ret.reindex(aligned.index).dropna()
                        aligned = aligned.reindex(spy_aligned.index)
                        if len(aligned) > 20 and spy_aligned.std() > 0:
                            b = float(np.cov(aligned, spy_aligned)[0, 1] / spy_aligned.var())
                            betas.append(b)
                        else:
                            betas.append(1.0)
                    beta = float(np.dot(sub_weights, betas))

                    # Back-fill individual position betas
                    for pos, b in zip([p for p in positions if p.ticker in available], betas):
                        pos.beta = round(b, 2)

                # Average pairwise correlation (diversification measure)
                if len(available) >= 2:
                    corr = sub_returns.corr()
                    upper = corr.where(
                        np.triu(np.ones(corr.shape), k=1).astype(bool)
                    )
                    avg_corr = float(upper.stack().mean())

        # ── 3. Concentration (Herfindahl index) ──────────────────────────────
        allocs = np.array([p.allocation_pct / 100 for p in positions])
        hhi = float(np.sum(allocs ** 2))

        # ── 4. Rebalancing signals ────────────────────────────────────────────
        overweight = [p.ticker for p in positions if p.allocation_pct > self.MAX_SINGLE_POSITION_PCT]
        underweight = [p.ticker for p in positions if p.allocation_pct < self.MIN_SINGLE_POSITION_PCT]

        if overweight:
            warnings.append(f"Overweight positions (>{self.MAX_SINGLE_POSITION_PCT}%): {overweight}")
        if underweight:
            warnings.append(f"Underweight positions (<{self.MIN_SINGLE_POSITION_PCT}%): {underweight}")
        if hhi > 0.25:
            warnings.append(f"High concentration risk (HHI={hhi:.2f}). Consider diversifying.")
        if avg_corr > 0.70:
            warnings.append(f"High avg pairwise correlation ({avg_corr:.2f}). Portfolio not well diversified.")
        if port_vol > 30.0:
            warnings.append(f"Portfolio annualized vol is {port_vol:.1f}% — high risk.")
        if abs(beta - 1.0) > 0.5:
            direction = "high beta (market-amplified)" if beta > 1.0 else "low beta (defensive)"
            warnings.append(f"Portfolio beta {beta:.2f} ({direction}).")

        return PortfolioView(
            total_value=round(total_value, 2),
            total_cost=round(total_cost, 2),
            unrealized_pnl=round(unrealized_pnl, 2),
            unrealized_pnl_pct=round(unrealized_pnl_pct, 2),
            positions=positions,
            portfolio_beta=round(beta, 2),
            portfolio_vol_ann_pct=round(port_vol, 2),
            max_drawdown_estimate_pct=round(max_dd_est, 1),
            herfindahl_concentration=round(hhi, 3),
            avg_pairwise_correlation=round(avg_corr, 3),
            overweight_tickers=overweight,
            underweight_tickers=underweight,
            warnings=warnings,
        )

    def _fetch_returns(self, tickers: List[str]) -> Optional[pd.DataFrame]:
        try:
            raw = yf.download(
                tickers,
                period=self.LOOKBACK_PERIOD,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw.empty:
                return None

            if isinstance(raw.columns, pd.MultiIndex):
                closes = raw["Close"]
            else:
                closes = raw[["Close"]] if "Close" in raw.columns else raw

            if isinstance(closes, pd.Series):
                closes = closes.to_frame(name=tickers[0])

            return closes.pct_change().dropna(how="all")
        except Exception as e:
            logger.warning(f"[Portfolio] Could not fetch returns: {e}")
            return None

    def _fetch_spy_returns(self, n_rows: int) -> Optional[pd.Series]:
        try:
            spy = yf.download(
                "SPY",
                period=self.LOOKBACK_PERIOD,
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if spy.empty:
                return None
            closes = spy["Close"].squeeze()
            return closes.pct_change().dropna().tail(n_rows)
        except Exception as e:
            logger.warning(f"[Portfolio] Could not fetch SPY returns: {e}")
            return None
