"""
Portfolio management for multi-asset trading.
Tracks positions, calculates portfolio-level risk metrics.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Represents a position in a single asset"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    
    @property
    def market_value(self) -> float:
        """Current market value of position"""
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss"""
        return (self.current_price - self.entry_price) * self.quantity
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized PnL as percentage"""
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price

@dataclass
class Portfolio:
    """Manages a multi-asset portfolio"""
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    
    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + positions)"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
    
    @property
    def total_unrealized_pnl(self) -> float:
        """Total unrealized PnL across all positions"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    def add_position(self, symbol: str, quantity: float, entry_price: float, current_price: float):
        """Add or update a position"""
        if symbol in self.positions:
            # Update existing position (average entry price)
            existing = self.positions[symbol]
            total_quantity = existing.quantity + quantity
            if total_quantity != 0:
                avg_entry = (existing.entry_price * existing.quantity + entry_price * quantity) / total_quantity
                self.positions[symbol] = Position(symbol, total_quantity, avg_entry, current_price)
            else:
                # Position closed
                del self.positions[symbol]
        else:
            self.positions[symbol] = Position(symbol, quantity, entry_price, current_price)
    
    def update_price(self, symbol: str, new_price: float):
        """Update current price for a position"""
        if symbol in self.positions:
            pos = self.positions[symbol]
            self.positions[symbol] = Position(pos.symbol, pos.quantity, pos.entry_price, new_price)
    
    def close_position(self, symbol: str) -> Optional[float]:
        """Close a position and return realized PnL"""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        realized_pnl = pos.unrealized_pnl
        self.cash += pos.market_value
        del self.positions[symbol]
        
        logger.info(f"Closed {symbol} position - Realized PnL: ${realized_pnl:.2f}")
        return realized_pnl
    
    def close_all_positions(self) -> float:
        """Close all positions and return total realized PnL"""
        total_pnl = 0.0
        symbols = list(self.positions.keys())
        
        for symbol in symbols:
            pnl = self.close_position(symbol)
            if pnl is not None:
                total_pnl += pnl
        
        logger.warning(f"🚨 All positions closed - Total PnL: ${total_pnl:.2f}")
        return total_pnl
    
    def get_weights(self) -> Dict[str, float]:
        """Calculate current portfolio weights"""
        total = self.total_value
        if total == 0:
            return {}
        
        weights = {}
        for symbol, pos in self.positions.items():
            weights[symbol] = pos.market_value / total
        
        # Add cash weight
        weights['CASH'] = self.cash / total
        
        return weights
    
    def calculate_portfolio_volatility(self, returns_data: Dict[str, List[float]]) -> float:
        """
        Calculate portfolio volatility given historical returns and current weights.
        
        Args:
            returns_data: Dict of symbol -> list of historical returns
            
        Returns:
            Portfolio volatility (annualized)
        """
        weights = self.get_weights()
        
        # Filter to symbols with positions
        active_symbols = [s for s in weights.keys() if s != 'CASH' and s in returns_data]
        
        if not active_symbols:
            return 0.0
        
        # Get weights as array
        w = np.array([weights[s] for s in active_symbols])
        
        # Build returns matrix
        returns_matrix = np.array([returns_data[s] for s in active_symbols])
        
        # Calculate covariance matrix
        cov_matrix = np.cov(returns_matrix)
        
        # Portfolio variance
        portfolio_var = np.dot(w, np.dot(cov_matrix, w))
        portfolio_vol = np.sqrt(portfolio_var)
        
        # Annualize (assuming daily returns)
        annualized_vol = portfolio_vol * np.sqrt(252)
        
        return annualized_vol
    
    def calculate_var(self, returns_data: Dict[str, List[float]], confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) for the portfolio.
        
        Args:
            returns_data: Dict of symbol -> list of historical returns
            confidence: Confidence level (e.g., 0.95 for 95% VaR)
            
        Returns:
            VaR in dollar terms
        """
        weights = self.get_weights()
        active_symbols = [s for s in weights.keys() if s != 'CASH' and s in returns_data]
        
        if not active_symbols:
            return 0.0
        
        # Get weights as array
        w = np.array([weights[s] for s in active_symbols])
        
        # Build returns matrix (assets x time)
        returns_matrix = np.array([returns_data[s] for s in active_symbols])
        
        # Portfolio returns over time
        portfolio_returns = np.dot(w, returns_matrix)
        
        # Calculate VaR at confidence level
        var_percentile = np.percentile(portfolio_returns, (1 - confidence) * 100)
        
        # Convert to dollar amount
        var_dollars = abs(var_percentile * self.total_value)
        
        return var_dollars
    
    def get_summary(self) -> dict:
        """Get portfolio summary for monitoring"""
        return {
            "total_value": self.total_value,
            "cash": self.cash,
            "positions_value": sum(pos.market_value for pos in self.positions.values()),
            "unrealized_pnl": self.total_unrealized_pnl,
            "num_positions": len(self.positions),
            "weights": self.get_weights(),
            "positions": {
                symbol: {
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "market_value": pos.market_value,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "unrealized_pnl_pct": pos.unrealized_pnl_pct
                }
                for symbol, pos in self.positions.items()
            }
        }
