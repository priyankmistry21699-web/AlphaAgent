"""
AlphaAgent — Database Manager
"""

from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import (
    Base, Trade, Portfolio, AgentLog,
    AIPortfolioState, SignalHistory, BacktestResult, Settings, WarmupRegistry,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DB_PATH = _PROJECT_ROOT / "alpha_agent.db"
DB_URL = f"sqlite:///{_DB_PATH}"


class DatabaseManager:

    def __init__(self, db_url: str = DB_URL):
        self.engine = create_engine(db_url, connect_args={"check_same_thread": False})
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        Base.metadata.create_all(bind=self.engine)
        self._migrate()

    def _migrate(self) -> None:
        from sqlalchemy import text
        migrations = [
            "ALTER TABLE trades ADD COLUMN horizon VARCHAR(10)",
        ]
        with self.engine.connect() as conn:
            for stmt in migrations:
                try:
                    conn.execute(text(stmt))
                    conn.commit()
                except Exception:
                    pass

    def get_db(self):
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    # ── Signal recording ──────────────────────────────────────────────────────

    def record_signal(self, ticker: str, signal_data: dict, agents: list, horizon: str = "1m") -> int:
        db = self.SessionLocal()
        try:
            new_trade = Trade(
                ticker=ticker,
                direction=signal_data["direction"],
                probability=signal_data["probability"],
                conviction=signal_data["conviction"],
                multiplier=signal_data["multiplier"],
                horizon=horizon,
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

    def add_position(self, ticker: str, shares: float, avg_price: float,
                     current_price: Optional[float] = None) -> Portfolio:
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
                existing.allocation_pct = 0.0
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
        db = self.SessionLocal()
        try:
            position = db.query(Portfolio).filter(Portfolio.ticker == ticker).first()
            if not position:
                return None

            pnl_pct = (exit_price - position.avg_price) / position.avg_price * 100

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
        db = self.SessionLocal()
        try:
            holdings = db.query(Portfolio).all()
            closed_trades = db.query(Trade).filter(Trade.status == "CLOSED").all()

            total_value = sum(h.market_value for h in holdings)
            total_cost = sum(h.shares * h.avg_price for h in holdings)
            unrealized_pnl = total_value - total_cost
            unrealized_pnl_pct = (unrealized_pnl / total_cost * 100) if total_cost > 0 else 0.0

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

    # ── AI Portfolio persistence ──────────────────────────────────────────────

    _DEFAULT_AI_PORTFOLIO = {
        "positions": [],
        "capital": 100_000.0,
        "created_at": None,
        "daily_returns": [],
    }

    def load_ai_portfolio(self) -> dict:
        """Returns the persisted AI portfolio, or a fresh default if none exists."""
        db = self.SessionLocal()
        try:
            row = db.query(AIPortfolioState).filter(AIPortfolioState.id == 1).first()
            if not row:
                return dict(self._DEFAULT_AI_PORTFOLIO)
            return {
                "positions":     row.positions or [],
                "capital":       row.capital or 100_000.0,
                "created_at":    row.created_at,
                "daily_returns": row.daily_returns or [],
            }
        finally:
            db.close()

    def save_ai_portfolio(self, portfolio: dict) -> None:
        """Upserts the AI portfolio state (always id=1)."""
        db = self.SessionLocal()
        try:
            row = db.query(AIPortfolioState).filter(AIPortfolioState.id == 1).first()
            if row:
                row.positions = portfolio.get("positions", [])
                row.capital = portfolio.get("capital", 100_000.0)
                row.created_at = portfolio.get("created_at")
                row.daily_returns = portfolio.get("daily_returns", [])
            else:
                row = AIPortfolioState(
                    id=1,
                    positions=portfolio.get("positions", []),
                    capital=portfolio.get("capital", 100_000.0),
                    created_at=portfolio.get("created_at"),
                    daily_returns=portfolio.get("daily_returns", []),
                )
                db.add(row)
            db.commit()
        finally:
            db.close()

    def get_recent_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        db = self.SessionLocal()
        try:
            rows = (
                db.query(Trade)
                .order_by(Trade.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id":          r.id,
                    "ticker":      r.ticker,
                    "timestamp":   r.timestamp.isoformat() if r.timestamp else None,
                    "direction":   r.direction,
                    "probability": r.probability,
                    "conviction":  r.conviction,
                    "horizon":     r.horizon or "1m",
                    "status":      r.status,
                    "pnl_pct":     r.pnl_pct,
                }
                for r in rows
            ]
        finally:
            db.close()

    # ── Signal history ────────────────────────────────────────────────────────

    def record_scan_result(self, region: str, asset_type: str, budget: float,
                           picks: list) -> int:
        """Saves a portfolio scan result to history."""
        db = self.SessionLocal()
        try:
            longs  = [p for p in picks if p.get("direction", "LONG") == "LONG"]
            shorts = [p for p in picks if p.get("direction") == "SHORT"]
            row = SignalHistory(
                region=region,
                asset_type=asset_type,
                budget=budget,
                pick_count=len(picks),
                long_count=len(longs),
                short_count=len(shorts),
                picks=picks,
            )
            db.add(row)
            db.commit()
            return row.id
        finally:
            db.close()

    def get_signal_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        db = self.SessionLocal()
        try:
            rows = (
                db.query(SignalHistory)
                .order_by(SignalHistory.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id":          r.id,
                    "timestamp":   r.timestamp.isoformat() if r.timestamp else None,
                    "region":      r.region,
                    "asset_type":  r.asset_type,
                    "budget":      r.budget,
                    "pick_count":  r.pick_count,
                    "long_count":  r.long_count,
                    "short_count": r.short_count,
                    "picks":       r.picks or [],
                }
                for r in rows
            ]
        finally:
            db.close()

    # ── Backtest results ──────────────────────────────────────────────────────

    def save_backtest(self, ticker: str, strategy: str, params: dict,
                      metrics: dict, equity_curve: list = None) -> int:
        db = self.SessionLocal()
        try:
            row = BacktestResult(
                ticker=ticker,
                strategy=strategy,
                params=params,
                total_return_pct=metrics.get("total_return_pct"),
                sharpe=metrics.get("sharpe"),
                max_drawdown_pct=metrics.get("max_drawdown_pct"),
                win_rate_pct=metrics.get("win_rate_pct"),
                n_trades=metrics.get("n_trades"),
                equity_curve=equity_curve or [],
                extra={k: v for k, v in metrics.items()
                       if k not in ("total_return_pct", "sharpe", "max_drawdown_pct",
                                    "win_rate_pct", "n_trades")},
            )
            db.add(row)
            db.commit()
            return row.id
        finally:
            db.close()

    def get_backtest_history(self, ticker: str = None, limit: int = 20) -> List[Dict[str, Any]]:
        db = self.SessionLocal()
        try:
            q = db.query(BacktestResult).order_by(BacktestResult.timestamp.desc())
            if ticker:
                q = q.filter(BacktestResult.ticker == ticker.upper())
            rows = q.limit(limit).all()
            return [
                {
                    "id":                r.id,
                    "timestamp":         r.timestamp.isoformat() if r.timestamp else None,
                    "ticker":            r.ticker,
                    "strategy":          r.strategy,
                    "params":            r.params,
                    "total_return_pct":  r.total_return_pct,
                    "sharpe":            r.sharpe,
                    "max_drawdown_pct":  r.max_drawdown_pct,
                    "win_rate_pct":      r.win_rate_pct,
                    "n_trades":          r.n_trades,
                }
                for r in rows
            ]
        finally:
            db.close()

    # ── Settings ──────────────────────────────────────────────────────────────

    def get_setting(self, key: str, default=None):
        db = self.SessionLocal()
        try:
            row = db.query(Settings).filter(Settings.key == key).first()
            return row.value if row else default
        finally:
            db.close()

    def set_setting(self, key: str, value) -> None:
        db = self.SessionLocal()
        try:
            row = db.query(Settings).filter(Settings.key == key).first()
            if row:
                row.value = value
            else:
                db.add(Settings(key=key, value=value))
            db.commit()
        finally:
            db.close()

    def get_all_settings(self) -> Dict[str, Any]:
        db = self.SessionLocal()
        try:
            rows = db.query(Settings).all()
            return {r.key: r.value for r in rows}
        finally:
            db.close()

    def bulk_save_settings(self, settings: Dict[str, Any]) -> None:
        db = self.SessionLocal()
        try:
            for key, value in settings.items():
                row = db.query(Settings).filter(Settings.key == key).first()
                if row:
                    row.value = value
                else:
                    db.add(Settings(key=key, value=value))
            db.commit()
        finally:
            db.close()

    # ── Warmup registry ───────────────────────────────────────────────────────

    def register_warmup(self, ticker: str, status: str = "pending",
                        duration_ms: float = None, error_msg: str = None) -> None:
        db = self.SessionLocal()
        try:
            row = db.query(WarmupRegistry).filter(WarmupRegistry.ticker == ticker).first()
            if row:
                row.status = status
                if duration_ms is not None:
                    row.duration_ms = duration_ms
                if error_msg is not None:
                    row.error_msg = error_msg
                if status == "done":
                    row.last_warmed_at = datetime.now(timezone.utc)
            else:
                row = WarmupRegistry(
                    ticker=ticker,
                    status=status,
                    duration_ms=duration_ms,
                    error_msg=error_msg,
                    last_warmed_at=datetime.now(timezone.utc) if status == "done" else None,
                )
                db.add(row)
            db.commit()
        finally:
            db.close()

    def increment_warmup_cache(self, ticker: str, hit: bool) -> None:
        db = self.SessionLocal()
        try:
            row = db.query(WarmupRegistry).filter(WarmupRegistry.ticker == ticker).first()
            if row:
                if hit:
                    row.cache_hits = (row.cache_hits or 0) + 1
                else:
                    row.cache_misses = (row.cache_misses or 0) + 1
                db.commit()
        finally:
            db.close()

    def get_warmup_registry(self) -> List[Dict[str, Any]]:
        db = self.SessionLocal()
        try:
            rows = db.query(WarmupRegistry).order_by(WarmupRegistry.ticker).all()
            return [
                {
                    "ticker":        r.ticker,
                    "status":        r.status,
                    "cache_hits":    r.cache_hits or 0,
                    "cache_misses":  r.cache_misses or 0,
                    "hit_rate":      round(
                        (r.cache_hits or 0) /
                        max((r.cache_hits or 0) + (r.cache_misses or 0), 1) * 100, 1
                    ),
                    "duration_ms":   r.duration_ms,
                    "last_warmed_at": r.last_warmed_at.isoformat() if r.last_warmed_at else None,
                    "error_msg":     r.error_msg,
                }
                for r in rows
            ]
        finally:
            db.close()
