"""
AlphaAgent — Transaction Cost Model

Realistic cost estimation for backtest P&L correction:
  - Bid-ask spread (proportional to price volatility + market cap tier)
  - Market impact (square-root model: ~σ × sqrt(volume_fraction))
  - Commission (fixed per trade, ~$0.005/share IBKR)
  - Borrow cost for shorts (annualised)

Usage:
    from quant_engine.transaction_costs import TransactionCostModel
    tcm = TransactionCostModel()
    cost = tcm.estimate(
        ticker="AAPL", side="BUY", notional=100_000,
        adv_dollar=2e10, daily_vol=0.018, price=200.0
    )
    # cost.total_bps, cost.total_dollars, cost.breakdown
"""

import logging
from dataclasses import dataclass, field
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass
class TCMResult:
    total_dollars: float
    total_bps: float                  # in basis points of notional
    breakdown: Dict[str, float] = field(default_factory=dict)


class TransactionCostModel:
    """
    Realistic transaction-cost estimator calibrated to retail electronic execution.

    Tiers (large/mid/small) auto-selected from ADV:
      large : ADV > $1B (AAPL, MSFT) → ~1 bp spread
      mid   : $100M < ADV < $1B      → ~3 bps spread
      small : ADV < $100M             → ~10 bps spread
    """

    def __init__(self,
                 commission_per_share: float = 0.005,
                 short_borrow_annual_bps: float = 100.0,
                 market_impact_const: float = 0.10):
        self.commission_ps         = commission_per_share
        self.short_borrow_ann_bps  = short_borrow_annual_bps
        self.mi_const              = market_impact_const

    def _spread_bps(self, adv_dollar: float) -> float:
        if adv_dollar > 1e9:    return 1.0
        if adv_dollar > 1e8:    return 3.0
        if adv_dollar > 1e7:    return 10.0
        return 30.0

    def _market_impact_bps(self, notional: float, adv_dollar: float,
                           daily_vol: float) -> float:
        """
        Square-root impact model (Almgren-Chriss simplification):
        impact_bps = const × daily_vol_bps × sqrt(participation_rate)
        """
        if adv_dollar <= 0:
            return 0.0
        participation = notional / adv_dollar
        participation = min(participation, 0.30)   # cap at 30% ADV
        return self.mi_const * (daily_vol * 10000) * (participation ** 0.5)

    def estimate(self,
                 ticker: str,
                 side: str,                # BUY / SELL / SHORT / COVER
                 notional: float,
                 adv_dollar: float = 5e8,
                 daily_vol: float = 0.02,
                 price: float = 100.0,
                 holding_days: int = 1) -> TCMResult:
        """
        Estimate one-leg transaction cost for a trade.
        Returns total cost in dollars + bps + per-component breakdown.
        """
        side = side.upper()
        notional = max(0.0, float(notional))

        # 1. Half-spread crossed on entry (paying the spread)
        spread = self._spread_bps(adv_dollar)
        spread_cost_bps = spread / 2.0
        spread_dollars  = notional * spread_cost_bps / 10000

        # 2. Market impact (square-root)
        mi_bps     = self._market_impact_bps(notional, adv_dollar, daily_vol)
        mi_dollars = notional * mi_bps / 10000

        # 3. Commission
        if price > 0:
            shares = notional / price
            commission = shares * self.commission_ps
        else:
            commission = 0.0

        # 4. Short borrow cost (per-day prorate)
        borrow = 0.0
        if side in ("SHORT", "COVER"):
            ann_bps  = self.short_borrow_ann_bps
            borrow   = notional * (ann_bps / 10000) * (holding_days / 252.0)

        total = spread_dollars + mi_dollars + commission + borrow
        total_bps = total / max(notional, 1.0) * 10000

        return TCMResult(
            total_dollars=round(total, 2),
            total_bps=round(total_bps, 3),
            breakdown={
                "spread":     round(spread_dollars, 2),
                "impact":     round(mi_dollars, 2),
                "commission": round(commission, 2),
                "borrow":     round(borrow, 2),
            },
        )

    def round_trip(self, **kwargs) -> TCMResult:
        """Round-trip (open + close) cost — both legs."""
        leg = self.estimate(**kwargs)
        return TCMResult(
            total_dollars=round(leg.total_dollars * 2, 2),
            total_bps=round(leg.total_bps * 2, 3),
            breakdown={k: round(v * 2, 2) for k, v in leg.breakdown.items()},
        )
