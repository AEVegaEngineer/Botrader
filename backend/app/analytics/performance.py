"""
Performance analytics module for trading strategies.
Calculates Sharpe, Sortino, turnover, fees, and other metrics.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for a trading strategy"""
    # Returns
    total_return: float
    annualized_return: float
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Drawdown
    max_drawdown: float
    current_drawdown: float
    
    # Trading activity
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float  # Total wins / Total losses
    
    # Turnover and costs
    turnover: float  # Portfolio turnover rate
    total_fees: float
    fees_pct_of_pnl: float
    
    # Time period
    start_date: datetime
    end_date: datetime
    trading_days: int

@dataclass
class TradeRecord:
    """Single trade record"""
    timestamp: datetime
    symbol: str
    side: str  # BUY or SELL
    quantity: float
    price: float
    fee: float
    pnl: Optional[float] = None  # For closed positions

class PerformanceAnalyzer:
    """Analyzes trading performance and calculates metrics"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation (default: 2%)
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_metrics(
        self,
        equity_curve: List[float],
        trades: List[TradeRecord],
        timestamps: List[datetime],
        initial_balance: float
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            equity_curve: Time series of portfolio values
            trades: List of all trades executed
            timestamps: Timestamps corresponding to equity curve
            initial_balance: Starting portfolio value
            
        Returns:
            PerformanceMetrics object
        """
        if len(equity_curve) == 0 or len(timestamps) == 0:
            return self._empty_metrics()
        
        equity_array = np.array(equity_curve)
        
        # Returns calculation
        returns = np.diff(equity_array) / equity_array[:-1]
        total_return = (equity_array[-1] - initial_balance) / initial_balance
        
        # Time period
        start_date = timestamps[0]
        end_date = timestamps[-1]
        trading_days = len(equity_curve)
        years = trading_days / 252  # Assuming 252 trading days per year
        
        # Annualized return
        if years > 0:
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = 0.0
        
        # Sharpe ratio
        if len(returns) > 1:
            excess_returns = returns - (self.risk_free_rate / 252)
            sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0.0
        
        # Sortino ratio (using downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1:
            downside_dev = np.std(downside_returns)
            sortino_ratio = (annualized_return - self.risk_free_rate) / downside_dev / np.sqrt(252) if downside_dev > 0 else 0
        else:
            sortino_ratio = 0.0
        
        # Drawdown
        peak = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - peak) / peak
        max_drawdown = np.min(drawdown)
        current_drawdown = drawdown[-1]
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade statistics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl and t.pnl < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([abs(t.pnl) for t in losing_trades]) if losing_trades else 0
        
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Turnover calculation (total traded value / average portfolio value)
        total_traded_value = sum(t.quantity * t.price for t in trades)
        avg_portfolio_value = np.mean(equity_array)
        turnover = total_traded_value / avg_portfolio_value if avg_portfolio_value > 0 else 0
        
        # Fees
        total_fees = sum(t.fee for t in trades)
        total_pnl = equity_array[-1] - initial_balance
        fees_pct_of_pnl = total_fees / abs(total_pnl) if total_pnl != 0 else 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            total_trades=total_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            turnover=turnover,
            total_fees=total_fees,
            fees_pct_of_pnl=fees_pct_of_pnl,
            start_date=start_date,
            end_date=end_date,
            trading_days=trading_days
        )
    
    def calculate_rolling_sharpe(
        self,
        returns: np.ndarray,
        window: int = 30
    ) -> np.ndarray:
        """
        Calculate rolling Sharpe ratio.
        
        Args:
            returns: Array of returns
            window: Rolling window size (default: 30 days)
            
        Returns:
            Array of rolling Sharpe ratios
        """
        if len(returns) < window:
            return np.array([])
        
        rolling_sharpe = []
        for i in range(window, len(returns) + 1):
            window_returns = returns[i-window:i]
            excess_returns = window_returns - (self.risk_free_rate / 252)
            if np.std(window_returns) > 0:
                sharpe = np.mean(excess_returns) / np.std(window_returns) * np.sqrt(252)
            else:
                sharpe = 0
            rolling_sharpe.append(sharpe)
        
        return np.array(rolling_sharpe)
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics when no data available"""
        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            current_drawdown=0.0,
            total_trades=0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            turnover=0.0,
            total_fees=0.0,
            fees_pct_of_pnl=0.0,
            start_date=datetime.now(),
            end_date=datetime.now(),
            trading_days=0
        )
    
    def compare_strategies(
        self,
        strategy_metrics: Dict[str, PerformanceMetrics]
    ) -> pd.DataFrame:
        """
        Create comparison table of multiple strategies.
        
        Args:
            strategy_metrics: Dict of strategy_name -> PerformanceMetrics
            
        Returns:
            DataFrame with comparative metrics
        """
        data = []
        for name, metrics in strategy_metrics.items():
            data.append({
                'Strategy': name,
                'Return': metrics.total_return,
                'Ann. Return': metrics.annualized_return,
                'Sharpe': metrics.sharpe_ratio,
                'Sortino': metrics.sortino_ratio,
                'Max DD': metrics.max_drawdown,
                'Win Rate': metrics.win_rate,
                'Profit Factor': metrics.profit_factor,
                'Trades': metrics.total_trades
            })
        
        return pd.DataFrame(data)
    
    def get_monthly_returns(
        self,
        equity_curve: List[float],
        timestamps: List[datetime]
    ) -> pd.DataFrame:
        """
        Calculate monthly returns for calendar visualization.
        
        Args:
            equity_curve: Portfolio values over time
            timestamps: Corresponding timestamps
            
        Returns:
            DataFrame with monthly returns
        """
        df = pd.DataFrame({
            'timestamp': timestamps,
            'equity': equity_curve
        })
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Resample to monthly
        monthly = df.resample('M').last()
        monthly['return'] = monthly['equity'].pct_change()
        monthly['year'] = monthly.index.year
        monthly['month'] = monthly.index.month
        
        # Pivot for heatmap
        pivot = monthly.pivot(index='year', columns='month', values='return')
        
        return pivot
