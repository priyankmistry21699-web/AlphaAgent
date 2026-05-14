"""
AlphaAgent — Crisis Scenario Stress Tester

Applies historical crisis shocks to a portfolio and reports estimated P&L impact.

Scenarios:
  1. 2008 GFC Peak-to-Trough (-56.8% SPY)
  2. 2020 COVID Crash (-33.9% SPY over 33 days)
  3. Rate Shock (+200 bps overnight — 2022-style)
  4. Sector Collapse (-30% in the ticker's sector)
  5. Black Monday (-20% single day — 1987 style)
  6. Custom (user-defined shock %)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)


# ── Pre-defined crisis scenarios ──────────────────────────────────────────────
# Each entry defines the SPY shock and a set of sector-specific multipliers
# (ratio of the sector's drawdown to SPY's drawdown during that crisis).

SCENARIOS: Dict[str, Dict] = {
    "2008_gfc": {
        "label": "2008 Global Financial Crisis",
        "spy_shock_pct": -56.8,
        "duration_days": 517,
        "sector_multipliers": {
            "Financial Services": 2.1,
            "Real Estate": 1.9,
            "Consumer Cyclical": 1.4,
            "Basic Materials": 1.3,
            "Technology": 1.0,
            "Healthcare": 0.7,
            "Consumer Defensive": 0.6,
            "Utilities": 0.7,
            "Energy": 1.0,
            "Industrials": 1.2,
            "Communication Services": 0.9,
        },
        "description": "Lehman collapse + housing bubble. Financials down >70%, SPY -56.8% peak-to-trough.",
    },
    "2020_covid": {
        "label": "2020 COVID Crash",
        "spy_shock_pct": -33.9,
        "duration_days": 33,
        "sector_multipliers": {
            "Energy": 2.0,
            "Financial Services": 1.4,
            "Real Estate": 1.3,
            "Consumer Cyclical": 1.2,
            "Industrials": 1.2,
            "Technology": 0.8,
            "Healthcare": 0.9,
            "Consumer Defensive": 0.7,
            "Utilities": 0.85,
            "Basic Materials": 1.1,
            "Communication Services": 0.9,
        },
        "description": "Fastest bear market in history. Energy & financials hit hardest; tech recovered quickly.",
    },
    "rate_shock_200bps": {
        "label": "Rate Shock +200 bps (2022-style)",
        "spy_shock_pct": -19.4,
        "duration_days": 180,
        "sector_multipliers": {
            "Real Estate": 2.2,
            "Utilities": 1.8,
            "Technology": 1.6,
            "Consumer Cyclical": 1.4,
            "Communication Services": 1.3,
            "Financial Services": 0.7,
            "Energy": 0.5,
            "Consumer Defensive": 0.9,
            "Healthcare": 0.9,
            "Basic Materials": 1.0,
            "Industrials": 1.0,
        },
        "description": "Fed hikes 200bps. Duration-sensitive assets (REITs, long-growth tech) hit hardest.",
    },
    "sector_collapse": {
        "label": "Sector Collapse (-30%)",
        "spy_shock_pct": -8.0,
        "duration_days": 60,
        "sector_multipliers": {},  # Applied by caller via sector override
        "description": "An isolated sector collapses -30% (e.g., banking crisis, pharma recall, energy embargo).",
        "_is_sector_specific": True,
    },
    "black_monday": {
        "label": "Black Monday (-20% single day)",
        "spy_shock_pct": -20.0,
        "duration_days": 1,
        "sector_multipliers": {
            "Financial Services": 1.3,
            "Technology": 1.2,
            "Consumer Cyclical": 1.1,
            "Healthcare": 0.9,
            "Consumer Defensive": 0.85,
            "Utilities": 0.80,
            "Energy": 1.0,
            "Real Estate": 1.1,
            "Basic Materials": 1.1,
            "Industrials": 1.1,
            "Communication Services": 1.0,
        },
        "description": "Single-day panic liquidation. High-beta stocks down >20%; defensive sectors cushion.",
    },
}


@dataclass
class PositionImpact:
    ticker: str
    sector: str
    current_value: float
    shocked_value: float
    pnl: float
    pnl_pct: float


@dataclass
class StressResult:
    scenario_id: str
    scenario_label: str
    description: str
    spy_shock_pct: float
    duration_days: int
    portfolio_value_before: float
    portfolio_value_after: float
    portfolio_pnl: float
    portfolio_pnl_pct: float
    position_impacts: List[PositionImpact] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class StressTester:
    """
    Applies historical crisis shocks to a portfolio.

    Usage:
        tester = StressTester()
        results = tester.run_all(holdings)
        result  = tester.run_scenario("2008_gfc", holdings)
    """

    def _get_sector(self, ticker: str) -> str:
        try:
            info = yf.Ticker(ticker).info
            return info.get("sector", "Unknown")
        except Exception:
            return "Unknown"

    def _ticker_beta(self, ticker: str) -> float:
        """Fetch beta from yfinance; default 1.0 if unavailable."""
        try:
            info = yf.Ticker(ticker).info
            beta = info.get("beta")
            if beta and isinstance(beta, (int, float)) and 0.1 < beta < 5.0:
                return float(beta)
        except Exception:
            pass
        return 1.0

    def run_scenario(
        self,
        scenario_id: str,
        holdings: List[Dict],
        custom_shock_pct: Optional[float] = None,
        target_sector: Optional[str] = None,
    ) -> StressResult:
        """
        Applies one crisis scenario to the portfolio.

        Args:
            scenario_id:      Key from SCENARIOS dict (or "custom").
            holdings:         List of position dicts (ticker, shares, last_price, market_value, sector?).
            custom_shock_pct: For scenario_id="custom", the % shock to apply to SPY.
            target_sector:    For sector_collapse, which sector collapses.
        """
        if scenario_id == "custom":
            scenario = {
                "label": f"Custom Shock ({custom_shock_pct:+.1f}%)",
                "spy_shock_pct": custom_shock_pct or -10.0,
                "duration_days": 30,
                "sector_multipliers": {},
                "description": "User-defined custom stress scenario.",
            }
        elif scenario_id not in SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario_id}'. Available: {list(SCENARIOS.keys())}")
        else:
            scenario = SCENARIOS[scenario_id]

        spy_shock = scenario["spy_shock_pct"] / 100.0
        sector_mults = dict(scenario.get("sector_multipliers", {}))

        # For sector_collapse, the target sector gets -30%, rest follow SPY mildly
        if scenario.get("_is_sector_specific") and target_sector:
            sector_mults = {target_sector: (-0.30 / spy_shock) if spy_shock != 0 else 3.75}

        position_impacts = []
        total_before = sum(h.get("market_value", 0.0) for h in holdings)
        total_after = 0.0
        warnings = []

        for h in holdings:
            ticker = h["ticker"]
            current_value = h.get("market_value", 0.0)
            sector = h.get("sector") or self._get_sector(ticker)
            beta = h.get("beta") or self._ticker_beta(ticker)

            # Apply sector multiplier; fall back to beta-adjusted SPY shock
            if sector in sector_mults:
                shock = spy_shock * sector_mults[sector]
            else:
                shock = spy_shock * max(0.5, min(beta, 2.5))

            shocked_value = current_value * (1 + shock)
            pnl = shocked_value - current_value
            pnl_pct = shock * 100

            position_impacts.append(PositionImpact(
                ticker=ticker,
                sector=sector,
                current_value=round(current_value, 2),
                shocked_value=round(shocked_value, 2),
                pnl=round(pnl, 2),
                pnl_pct=round(pnl_pct, 2),
            ))
            total_after += shocked_value

        total_pnl = total_after - total_before
        total_pnl_pct = (total_pnl / total_before * 100) if total_before > 0 else 0.0

        if total_pnl_pct < -25:
            warnings.append(
                f"Severe impact: portfolio loses {total_pnl_pct:.1f}% in this scenario."
            )

        worst = min(position_impacts, key=lambda x: x.pnl_pct, default=None)
        if worst and worst.pnl_pct < -40:
            warnings.append(
                f"{worst.ticker} could lose {worst.pnl_pct:.1f}% — consider hedging this position."
            )

        return StressResult(
            scenario_id=scenario_id,
            scenario_label=scenario["label"],
            description=scenario["description"],
            spy_shock_pct=scenario["spy_shock_pct"],
            duration_days=scenario["duration_days"],
            portfolio_value_before=round(total_before, 2),
            portfolio_value_after=round(total_after, 2),
            portfolio_pnl=round(total_pnl, 2),
            portfolio_pnl_pct=round(total_pnl_pct, 2),
            position_impacts=sorted(position_impacts, key=lambda x: x.pnl_pct),
            warnings=warnings,
        )

    def run_all(
        self,
        holdings: List[Dict],
        include_custom: Optional[float] = None,
    ) -> List[StressResult]:
        """Runs all standard scenarios and returns sorted by severity."""
        results = []
        for sid in SCENARIOS:
            try:
                results.append(self.run_scenario(sid, holdings))
            except Exception as e:
                logger.error(f"[Stress] Scenario '{sid}' failed: {e}")

        if include_custom is not None:
            try:
                results.append(
                    self.run_scenario("custom", holdings, custom_shock_pct=include_custom)
                )
            except Exception as e:
                logger.error(f"[Stress] Custom scenario failed: {e}")

        return sorted(results, key=lambda r: r.portfolio_pnl_pct)
