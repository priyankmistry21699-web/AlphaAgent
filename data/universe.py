"""
AlphaAgent — Universe Manager

Manages ticker universes (watchlists) for multi-ticker scanning.
Provides pre-built universes and supports custom lists.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# ── Pre-built universes ───────────────────────────────────────────────────────

MEGA_CAP: List[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "BRK-B", "LLY", "JPM",
]

SP500_CORE: List[str] = MEGA_CAP + [
    "V", "UNH", "XOM", "JNJ", "MA", "PG", "HD", "COST", "ABBV", "MRK",
    "CVX", "AVGO", "PEP", "BAC", "TMO", "ORCL", "ACN", "CRM", "NFLX", "AMD",
]

TECHNOLOGY: List[str] = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    "AVGO", "ORCL", "CRM", "AMD", "INTC",
]

FINANCIALS: List[str] = [
    "JPM", "BAC", "WFC", "GS", "MS",
    "BLK", "C", "AXP", "SCHW", "USB",
]

HEALTHCARE: List[str] = [
    "UNH", "JNJ", "LLY", "ABBV", "MRK",
    "TMO", "ABT", "BMY", "AMGN", "GILD",
]

ENERGY: List[str] = [
    "XOM", "CVX", "COP", "EOG", "SLB",
    "MPC", "PSX", "VLO", "OXY", "HAL",
]

CONSUMER: List[str] = [
    "AMZN", "TSLA", "HD", "MCD", "NKE",
    "SBUX", "LOW", "TGT", "DG", "ROST",
]

UNIVERSES: Dict[str, List[str]] = {
    "mega_cap":   MEGA_CAP,
    "sp500_core": SP500_CORE,
    "technology": TECHNOLOGY,
    "financials": FINANCIALS,
    "healthcare": HEALTHCARE,
    "energy":     ENERGY,
    "consumer":   CONSUMER,
}


@dataclass
class ScanResult:
    ticker: str
    direction: str
    probability: float
    conviction_pct: float
    regime: str
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


class UniverseManager:
    """
    Manages ticker universes and provides bulk signal scanning.

    Usage:
        um = UniverseManager()
        signals = um.scan("mega_cap", top_n=5)
    """

    def __init__(self, custom_tickers: Optional[List[str]] = None):
        self.custom_tickers = custom_tickers or []

    def list_universes(self) -> List[str]:
        names = list(UNIVERSES.keys())
        if self.custom_tickers:
            names.append("custom")
        return names

    def get_universe(self, name: str = "mega_cap") -> List[str]:
        if name == "custom":
            return self.custom_tickers
        tickers = UNIVERSES.get(name)
        if tickers is None:
            logger.warning(f"[Universe] Unknown universe '{name}' — defaulting to mega_cap.")
            return MEGA_CAP
        return tickers

    def scan(
        self,
        universe: str = "mega_cap",
        top_n: int = 5,
        registry=None,
        direction_filter: Optional[str] = None,
    ) -> List[ScanResult]:
        """
        Runs the full agent pipeline on every ticker in the universe.
        Returns up to top_n results, sorted by conviction (highest first).

        direction_filter: 'LONG' | 'SHORT' | None (all)
        """
        from orchestrator.graph import build_alpha_graph
        from data.market import MarketData
        from agents.registry import AgentRegistry

        tickers = self.get_universe(universe)
        graph = build_alpha_graph()
        reg = registry or AgentRegistry()

        results: List[ScanResult] = []

        for ticker in tickers:
            try:
                logger.info(f"[Scan] {ticker}...")
                md = MarketData(ticker)
                state = graph.invoke({
                    "ticker": ticker,
                    "market_data": md,
                    "registry": reg,
                })
                info = state["final_signal"]
                packet = info["packet"]
                regime_label = (
                    packet.regime.current_regime.value
                    if packet.regime and hasattr(packet.regime.current_regime, "value")
                    else str(packet.regime.current_regime) if packet.regime else "UNKNOWN"
                )
                results.append(ScanResult(
                    ticker=ticker,
                    direction=str(packet.direction.value if hasattr(packet.direction, "value") else packet.direction),
                    probability=round(float(info["probability_up"]), 3),
                    conviction_pct=round(float(packet.conviction_pct), 1),
                    regime=regime_label,
                    warnings=list(packet.warnings[:3]),
                ))
            except Exception as e:
                logger.error(f"[Scan] {ticker} failed: {e}")
                results.append(ScanResult(
                    ticker=ticker,
                    direction="HOLD",
                    probability=0.5,
                    conviction_pct=0.0,
                    regime="UNKNOWN",
                    error=str(e)[:120],
                ))

        # Filter and sort
        filtered = [
            r for r in results
            if r.error is None
            and (direction_filter is None or r.direction == direction_filter)
        ]
        filtered.sort(key=lambda x: x.conviction_pct, reverse=True)
        return filtered[:top_n]
