"""
AlphaAgent — Database Models

Defines the schema for trade history, portfolio state, and
agent audit logs using SQLAlchemy.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class Trade(Base):
    """Stores every signal generated and its subsequent outcome."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Signal data
    direction = Column(String(10))
    probability = Column(Float)
    conviction = Column(Float)
    multiplier = Column(Float)

    # JSON blob of all agent results
    agent_audit = Column(JSON)

    # Outcome (populated by close_position)
    entry_price = Column(Float)
    exit_price = Column(Float)
    pnl_pct = Column(Float)
    status = Column(String(20), default="OPEN")  # OPEN | CLOSED | CANCELLED

    __table_args__ = (
        Index("ix_trades_ticker_ts", "ticker", "timestamp"),
        Index("ix_trades_status", "status"),
    )


class Portfolio(Base):
    """Current real-time holdings — one row per ticker."""
    __tablename__ = "portfolio"

    ticker = Column(String(10), primary_key=True)
    shares = Column(Float, default=0.0)
    avg_price = Column(Float, default=0.0)
    last_price = Column(Float, default=0.0)
    market_value = Column(Float, default=0.0)
    allocation_pct = Column(Float, default=0.0)
    last_updated = Column(DateTime(timezone=True), onupdate=func.now())


class AgentLog(Base):
    """Detailed performance log for each agent run."""
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
