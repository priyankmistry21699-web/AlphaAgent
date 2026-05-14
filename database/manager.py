"""
AlphaAgent — Database Manager

Handles connection pooling, session management, and CRUD operations
for the AlphaAgent persistence layer.
"""

from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base, Trade, Portfolio, AgentLog

# Absolute path so the DB is always in the project root regardless of CWD
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DB_PATH = _PROJECT_ROOT / "alpha_agent.db"
DB_URL = f"sqlite:///{_DB_PATH}"


class DatabaseManager:
    """Manages SQLite database lifecycle."""

    def __init__(self, db_url: str = DB_URL):
        self.engine = create_engine(db_url, connect_args={"check_same_thread": False})
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        Base.metadata.create_all(bind=self.engine)

    def get_db(self):
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    # ── Signal recording ──────────────────────────────────────────────────────

    def record_signal(self, ticker: str, signal_data: dict, agents: list) -> int:
        """Saves a generated signal and its agent audit trail to the DB."""
        db = self.SessionLocal()
        try:
            new_trade = Trade(
                ticker=ticker,
                direction=signal_data["direction"],
                probability=signal_data["probability"],
                conviction=signal_data["conviction"],
                multiplier=signal_data["multiplier"],
                agent_audit=[{
                    "agent": a.agent_name,
                    "vote": str(a.vote),
                    "prob": a.probability_up,
                    "reason": a.reasoning,
                } for a in agents],
            )
            db.add(new_trade)

            for a in agents:
                log = AgentLog(
                    agent_name=a.agent_name,
                    ticker=ticker,
                    vote=str(a.vote),
                    probability=a.probability_up,
                    confidence=a.confidence,
                    latency_ms=a.computation_time_ms,
                )
                db.add(log)

            db.commit()
            return new_trade.id
        finally:
            db.close()

    # ── Portfolio CRUD ────────────────────────────────────────────────────────

    def add_position(
        self,
        ticker: str,
        shares: float,
        avg_price: float,
        current_price: Optional[float] = None,
    ) -> Portfolio:
        """
        Upserts a position. If the ticker already exists, shares and avg_price
        are blended (dollar-weighted average entry).
        """
        db = self.SessionLocal()
        try:
            existing = db.query(Portfolio).filter(Portfolio.ticker == ticker).first()
            price = current_price or avg_price

            if existing:
                total_cost = existing.shares * existing.avg_price + shares * avg_price
                total_shares = existing.shares + shares
                existing.avg_price = total_cost / total_shares if total_shares > 0 else avg_price
                existing.shares = total_shares
                existing.last_price = price
                existing.market_value = total_shares * price
                existing.allocation_pct = 0.0  # recalculated by get_portfolio_performance
                existing.last_updated = datetime.now(timezone.utc)
                db.commit()
                db.refresh(existing)
                return existing
            else:
                position = Portfolio(
                    ticker=ticker,
                    shares=shares,
                    avg_price=avg_price,
                    last_price=price,
                    market_value=shares * price,
                    allocation_pct=0.0,
                )
                db.add(position)
                db.commit()
                db.refresh(position)
                return position
        finally:
            db.close()

    def close_position(self, ticker: str, exit_price: float) -> Optional[Dict[str, Any]]:
        """
        Fully closes a position. Records the realized P&L on the most recent
        open Trade for that ticker, then removes it from the Portfolio table.
        """
        db = self.SessionLocal()
        try:
            position = db.query(Portfolio).filter(Portfolio.ticker == ticker).first()
            if not position:
                return None

            pnl_pct = (exit_price - position.avg_price) / position.avg_price * 100

            # Mark the latest open trade for this ticker as closed
            trade = (
                db.query(Trade)
                .filter(Trade.ticker == ticker, Trade.status == "OPEN")
                .order_by(Trade.timestamp.desc())
                .first()
            )
            if trade:
                trade.exit_price = exit_price
                trade.pnl_pct = pnl_pct
                trade.status = "CLOSED"

            result = {
                "ticker": ticker,
                "shares": position.shares,
                "avg_entry": position.avg_price,
                "exit_price": exit_price,
                "pnl_pct": round(pnl_pct, 2),
                "realized_pnl": round(position.shares * (exit_price - position.avg_price), 2),
            }

            db.delete(position)
            db.commit()
            return result
        finally:
            db.close()

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Refreshes last_price and market_value for every holding in the portfolio."""
        db = self.SessionLocal()
        try:
            holdings = db.query(Portfolio).all()
            total_value = sum(h.shares * prices.get(h.ticker, h.last_price) for h in holdings)
            for h in holdings:
                price = prices.get(h.ticker, h.last_price)
                h.last_price = price
                h.market_value = h.shares * price
                h.allocation_pct = (h.market_value / total_value * 100) if total_value > 0 else 0.0
                h.last_updated = datetime.now(timezone.utc)
            db.commit()
        finally:
            db.close()

    def get_portfolio_performance(self) -> Dict[str, Any]:
        """
        Returns a snapshot of portfolio performance:
        total value, unrealized P&L, allocation breakdown,
        and win-rate from closed trades.
        """
        db = self.SessionLocal()
        try:
            holdings = db.query(Portfolio).all()
            closed_trades = db.query(Trade).filter(Trade.status == "CLOSED").all()

            total_value = sum(h.market_value for h in holdings)
            total_cost = sum(h.shares * h.avg_price for h in holdings)
            unrealized_pnl = total_value - total_cost
            unrealized_pnl_pct = (unrealized_pnl / total_cost * 100) if total_cost > 0 else 0.0

            # Recalculate allocation %
            positions = []
            for h in holdings:
                alloc = (h.market_value / total_value * 100) if total_value > 0 else 0.0
                positions.append({
                    "ticker": h.ticker,
                    "shares": h.shares,
                    "avg_price": h.avg_price,
                    "last_price": h.last_price,
                    "market_value": round(h.market_value, 2),
                    "allocation_pct": round(alloc, 2),
                    "unrealized_pnl_pct": round(
                        (h.last_price - h.avg_price) / h.avg_price * 100
                        if h.avg_price > 0 else 0.0, 2
                    ),
                })

            # Win-rate from closed trades
            pnl_list = [t.pnl_pct for t in closed_trades if t.pnl_pct is not None]
            wins = [p for p in pnl_list if p > 0]
            win_rate = len(wins) / len(pnl_list) * 100 if pnl_list else 0.0
            avg_win = sum(wins) / len(wins) if wins else 0.0
            losses = [p for p in pnl_list if p <= 0]
            avg_loss = sum(losses) / len(losses) if losses else 0.0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

            return {
                "total_market_value": round(total_value, 2),
                "total_cost_basis": round(total_cost, 2),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "unrealized_pnl_pct": round(unrealized_pnl_pct, 2),
                "n_positions": len(holdings),
                "n_closed_trades": len(closed_trades),
                "win_rate_pct": round(win_rate, 1),
                "avg_win_pct": round(avg_win, 2),
                "avg_loss_pct": round(avg_loss, 2),
                "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else None,
                "positions": sorted(positions, key=lambda x: x["market_value"], reverse=True),
            }
        finally:
            db.close()
