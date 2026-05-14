"""
AlphaAgent — Backtesting Engine

Simulates historical trading using the quantitative models to 
calculate Profit & Loss (PnL), Win Rate, and Max Drawdown.
"""

import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Dict

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    ticker: str
    entry_date: str
    entry_price: float
    direction: str  # "LONG" or "SHORT"
    position_size: float
    exit_date: str = None
    exit_price: float = None
    pnl_pct: float = None

@dataclass
class BacktestResult:
    ticker: str
    start_date: str
    end_date: str
    total_trades: int
    win_rate: float
    total_return_pct: float
    max_drawdown_pct: float
    trades: List[Trade]


class BacktestEngine:
    """
    Runs historical simulations of trading strategies over a given dataset.
    Note: For speed, this engine runs purely on the math layer (Technical/Fundamental) 
    rather than spinning up LangGraph for every single historical day.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        
    def run_fast_backtest(self, ticker: str, ohlcv_df: pd.DataFrame, signal_series: pd.Series) -> BacktestResult:
        """
        Runs a vectorized/fast backtest using pre-computed signals.
        
        Args:
            ticker: Stock symbol
            ohlcv_df: Historical price data
            signal_series: Pandas series of 1 (Buy), -1 (Sell), 0 (Hold) matching the index of ohlcv_df
        """
        logger.info(f"Starting backtest for {ticker} over {len(ohlcv_df)} days.")
        
        active_trade: Trade = None
        self.equity_curve.append(self.initial_capital)
        
        for i in range(1, len(ohlcv_df)):
            date = str(ohlcv_df.index[i])[:10]
            current_price = float(ohlcv_df['Close'].iloc[i])
            signal = signal_series.iloc[i-1] # Use yesterday's signal for today's action
            
            # 1. Check if we should exit an active trade
            if active_trade:
                # Simple logic: If we are LONG and signal becomes SELL (-1), exit
                if active_trade.direction == "LONG" and signal == -1:
                    active_trade.exit_date = date
                    active_trade.exit_price = current_price
                    active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
                    
                    # Update capital
                    trade_profit = active_trade.position_size * active_trade.pnl_pct
                    self.current_capital += trade_profit
                    
                    self.trades.append(active_trade)
                    active_trade = None
                    
            # 2. Check if we should enter a new trade
            if not active_trade:
                if signal == 1:
                    # Enter LONG
                    position_size = self.current_capital * 0.10 # Risk 10% per trade
                    active_trade = Trade(
                        ticker=ticker,
                        entry_date=date,
                        entry_price=current_price,
                        direction="LONG",
                        position_size=position_size
                    )
                    
            # Track daily equity
            if active_trade:
                unrealized_pnl = (current_price - active_trade.entry_price) / active_trade.entry_price
                daily_equity = self.current_capital + (active_trade.position_size * unrealized_pnl)
            else:
                daily_equity = self.current_capital
                
            self.equity_curve.append(daily_equity)
            
        # Close any open trades at the end of the backtest
        if active_trade:
            active_trade.exit_date = str(ohlcv_df.index[-1])[:10]
            active_trade.exit_price = float(ohlcv_df['Close'].iloc[-1])
            active_trade.pnl_pct = (active_trade.exit_price - active_trade.entry_price) / active_trade.entry_price
            self.trades.append(active_trade)
            
        return self._compile_results(ticker, str(ohlcv_df.index[0])[:10], str(ohlcv_df.index[-1])[:10])
        
    def _compile_results(self, ticker: str, start_date: str, end_date: str) -> BacktestResult:
        if not self.trades:
            return BacktestResult(ticker, start_date, end_date, 0, 0.0, 0.0, 0.0, [])
            
        winning_trades = sum(1 for t in self.trades if t.pnl_pct > 0)
        win_rate = winning_trades / len(self.trades)
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        
        # Calculate Max Drawdown
        peak = self.initial_capital
        max_dd = 0.0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd
                
        return BacktestResult(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            total_trades=len(self.trades),
            win_rate=win_rate,
            total_return_pct=total_return,
            max_drawdown_pct=max_dd,
            trades=self.trades
        )
