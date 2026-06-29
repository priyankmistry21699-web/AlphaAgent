"""
AlphaAgent — Database Models
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean, Index, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    direction = Column(String(10))
    probability = Column(Float)
    conviction = Column(Float)
    multiplier = Column(Float)

    agent_audit = Column(JSON)
    horizon = Column(String(10), nullable=True)

    entry_price = Column(Float)
    exit_price = Column(Float)
    pnl_pct = Column(Float)
    status = Column(String(20), default="OPEN")

    __table_args__ = (
        Index("ix_trades_ticker_ts", "ticker", "timestamp"),
        Index("ix_trades_status", "status"),
    )


class Portfolio(Base):
    __tablename__ = "portfolio"

    ticker = Column(String(10), primary_key=True)
    shares = Column(Float, default=0.0)
    avg_price = Column(Float, default=0.0)
    last_price = Column(Float, default=0.0)
    market_value = Column(Float, default=0.0)
    allocation_pct = Column(Float, default=0.0)
    last_updated = Column(DateTime(timezone=True), onupdate=func.now())


class AgentLog(Base):
    __tablename__ = "agent_logs"

    id = Column(Integer, primary_key=True)
    agent_name = Column(String(50), nullable=False, index=True)
    ticker = Column(String(10), nullable=False, index=True)
    vote = Column(String(10))
    probability = Column(Float)
    confidence = Column(Float)
    latency_ms = Column(Float)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    __table_args__ = (
        Index("ix_agent_logs_agent_ticker_ts", "agent_name", "ticker", "timestamp"),
    )


class AIPortfolioState(Base):
    """Single-row store for the AI-managed portfolio. id is always 1."""
    __tablename__ = "ai_portfolio_state"

    id = Column(Integer, primary_key=True, default=1)
    positions = Column(JSON, default=list)
    capital = Column(Float, default=100_000.0)
    created_at = Column(Float, nullable=True)   # unix timestamp
    daily_returns = Column(JSON, default=list)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class SignalHistory(Base):
    """Records every portfolio scan result with its top picks."""
    __tablename__ = "signal_history"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    region = Column(String(20))
    asset_type = Column(String(20))
    budget = Column(Float)
    pick_count = Column(Integer, default=0)
    long_count = Column(Integer, default=0)
    short_count = Column(Integer, default=0)
    picks = Column(JSON)        # full list of pick dicts

    __table_args__ = (
        Index("ix_signal_history_ts", "timestamp"),
    )


class BacktestResult(Base):
    """Stores every backtest run with params and performance metrics."""
    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    ticker = Column(String(10), index=True)
    strategy = Column(String(50))
    params = Column(JSON)               # {horizon, start, end, ...}
    total_return_pct = Column(Float)
    sharpe = Column(Float)
    max_drawdown_pct = Column(Float)
    win_rate_pct = Column(Float)
    n_trades = Column(Integer)
    equity_curve = Column(JSON)         # [{date, value}, ...]
    extra = Column(JSON)                # catch-all for extra metrics

    __table_args__ = (
        Index("ix_backtest_ticker_ts", "ticker", "timestamp"),
    )


class Settings(Base):
    """Key-value settings store. Mirrors config/settings.yaml but survives restarts."""
    __tablename__ = "settings"

    key = Column(String(100), primary_key=True)
    value = Column(JSON)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class WarmupRegistry(Base):
    """Tracks per-ticker warmup state and cache statistics."""
    __tablename__ = "warmup_registry"

    ticker = Column(String(10), primary_key=True)
    status = Column(String(20), default="pending")  # pending | running | done | failed
    cache_hits = Column(Integer, default=0)
    cache_misses = Column(Integer, default=0)
    duration_ms = Column(Float, nullable=True)
    last_warmed_at = Column(DateTime(timezone=True), nullable=True)
    error_msg = Column(Text, nullable=True)
