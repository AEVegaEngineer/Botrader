from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, List

@dataclass
class DailyStats:
    date: date
    starting_balance: float
    current_balance: float
    realized_pnl: float = 0.0
    trades_count: int = 0

    @property
    def daily_return(self) -> float:
        if self.starting_balance == 0:
            return 0.0
        return (self.current_balance - self.starting_balance) / self.starting_balance

@dataclass
class TradeRisk:
    symbol: str
    side: str
    quantity: float
    price: float
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    @property
    def notional(self) -> float:
        """Calculate trade notional value"""
        return self.quantity * self.price
    
    @property
    def risk_amount(self) -> float:
        """Calculate dollar risk based on stop loss"""
        if self.stop_loss == 0.0:
            return 0.0
        return abs(self.price - self.stop_loss) * self.quantity

@dataclass
class PositionSizingConfig:
    """Configuration for volatility-targeted position sizing"""
    target_volatility: float
    lookback_days: int
    min_position_size: float
    max_leverage: float

@dataclass
class CircuitBreakerState:
    """Track circuit breaker state and reasons"""
    is_active: bool = False
    triggered_at: Optional[datetime] = None
    reason: str = ""
    drawdown_pct: float = 0.0
    daily_loss_pct: float = 0.0
    latency_ms: float = 0.0
    error_rate: float = 0.0
    
@dataclass
class LatencyTracker:
    """Track operation latencies for circuit breaker"""
    recent_latencies: List[float] = field(default_factory=list)
    max_samples: int = 100
    
    def add_latency(self, latency_ms: float):
        self.recent_latencies.append(latency_ms)
        if len(self.recent_latencies) > self.max_samples:
            self.recent_latencies.pop(0)
    
    @property
    def avg_latency(self) -> float:
        if not self.recent_latencies:
            return 0.0
        return sum(self.recent_latencies) / len(self.recent_latencies)
    
    @property
    def max_latency(self) -> float:
        if not self.recent_latencies:
            return 0.0
        return max(self.recent_latencies)

@dataclass
class ErrorTracker:
    """Track error rates for circuit breaker"""
    recent_operations: List[bool] = field(default_factory=list)  # True = success, False = error
    max_samples: int = 100
    
    def add_operation(self, success: bool):
        self.recent_operations.append(success)
        if len(self.recent_operations) > self.max_samples:
            self.recent_operations.pop(0)
    
    @property
    def error_rate(self) -> float:
        if not self.recent_operations:
            return 0.0
        errors = sum(1 for op in self.recent_operations if not op)
        return errors / len(self.recent_operations)
