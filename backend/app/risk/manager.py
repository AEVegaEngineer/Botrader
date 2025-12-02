import logging
from datetime import date, datetime
from typing import Optional
import numpy as np
from app.core.config import settings
from app.risk.limits import (
    DailyStats, 
    TradeRisk, 
    CircuitBreakerState, 
    LatencyTracker, 
    ErrorTracker,
    PositionSizingConfig
)

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, initial_balance: float):
        self.balance = initial_balance
        self.daily_stats = DailyStats(
            date=date.today(),
            starting_balance=initial_balance,
            current_balance=initial_balance
        )
        self.peak_balance = initial_balance
        self.is_halted = False
        
        # Circuit Breaker Components
        self.circuit_breaker = CircuitBreakerState()
        self.latency_tracker = LatencyTracker(max_samples=settings.ERROR_RATE_WINDOW)
        self.error_tracker = ErrorTracker(max_samples=settings.ERROR_RATE_WINDOW)
        
        # Position Sizing Config
        self.position_config = PositionSizingConfig(
            target_volatility=settings.TARGET_DAILY_VOL,
            lookback_days=settings.VOL_LOOKBACK_DAYS,
            min_position_size=settings.MIN_POSITION_SIZE,
            max_leverage=settings.MAX_LEVERAGE
        )
        
        # Historical returns for volatility calculation
        self.historical_returns: list[float] = []

    def update_balance(self, new_balance: float):
        """Update balance and trigger circuit breaker checks"""
        # Calculate return for volatility tracking
        if self.balance > 0:
            period_return = (new_balance - self.balance) / self.balance
            self.historical_returns.append(period_return)
            # Keep only lookback window
            max_samples = self.position_config.lookback_days * 24  # Assuming hourly updates
            if len(self.historical_returns) > max_samples:
                self.historical_returns = self.historical_returns[-max_samples:]
        
        self.balance = new_balance
        self.daily_stats.current_balance = new_balance
        
        # Update Peak for Drawdown
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance
            
        # Check Global Kill Switch
        self._check_global_limits()

    def add_latency_sample(self, latency_ms: float):
        """Track operation latency for circuit breaker monitoring"""
        self.latency_tracker.add_latency(latency_ms)
        self._check_global_limits()
    
    def add_operation_result(self, success: bool):
        """Track operation success/failure for circuit breaker monitoring"""
        self.error_tracker.add_operation(success)
        self._check_global_limits()

    def _check_global_limits(self):
        """Check all circuit breaker conditions"""
        reasons = []
        
        # 1. Max Drawdown
        drawdown = (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0
        if drawdown > settings.MAX_DRAWDOWN_PCT:
            reasons.append(f"Max Drawdown: {drawdown:.2%} > {settings.MAX_DRAWDOWN_PCT:.2%}")
            self.circuit_breaker.drawdown_pct = drawdown
            
        # 2. Daily Loss
        daily_loss = -self.daily_stats.daily_return
        if daily_loss > settings.MAX_DAILY_LOSS_PCT:
            reasons.append(f"Daily Loss: {daily_loss:.2%} > {settings.MAX_DAILY_LOSS_PCT:.2%}")
            self.circuit_breaker.daily_loss_pct = daily_loss
        
        # 3. Latency Spike
        max_latency = self.latency_tracker.max_latency
        if max_latency > settings.MAX_LATENCY_MS:
            reasons.append(f"High Latency: {max_latency:.0f}ms > {settings.MAX_LATENCY_MS:.0f}ms")
            self.circuit_breaker.latency_ms = max_latency
        
        # 4. Error Rate
        error_rate = self.error_tracker.error_rate
        if error_rate > settings.MAX_ERROR_RATE:
            reasons.append(f"High Error Rate: {error_rate:.2%} > {settings.MAX_ERROR_RATE:.2%}")
            self.circuit_breaker.error_rate = error_rate
        
        # Trigger circuit breaker if any condition is met
        if reasons and not self.is_halted:
            self.is_halted = True
            self.circuit_breaker.is_active = True
            self.circuit_breaker.triggered_at = datetime.now()
            self.circuit_breaker.reason = "; ".join(reasons)
            logger.critical(f"🚨 CIRCUIT BREAKER TRIGGERED: {self.circuit_breaker.reason}")

    def check_trade_risk(self, trade: TradeRisk) -> bool:
        """Validate trade against risk limits"""
        if self.is_halted:
            logger.warning("Trade rejected: Risk Manager is HALTED.")
            return False

        # 1. Check notional cap
        if trade.notional > settings.MAX_TRADE_NOTIONAL:
            logger.warning(f"Trade rejected: Notional ${trade.notional:.2f} exceeds max ${settings.MAX_TRADE_NOTIONAL:.2f}")
            return False
        
        # 2. Check leverage limit
        max_position_size = self.balance * settings.MAX_LEVERAGE
        if trade.notional > max_position_size:
            logger.warning(f"Trade rejected: Notional ${trade.notional:.2f} exceeds max position ${max_position_size:.2f}")
            return False
        
        # 3. Check stop loss risk if provided
        if trade.stop_loss > 0:
            max_risk = self.balance * settings.RISK_PER_TRADE_PCT
            if trade.risk_amount > max_risk:
                logger.warning(f"Trade rejected: Risk ${trade.risk_amount:.2f} exceeds max ${max_risk:.2f}")
                return False

        return True

    def get_current_volatility(self) -> float:
        """Calculate current portfolio volatility from historical returns"""
        if len(self.historical_returns) < 2:
            # Default to target volatility if insufficient data
            return self.position_config.target_volatility
        
        # Calculate rolling volatility (annualized)
        returns_array = np.array(self.historical_returns)
        volatility = np.std(returns_array) * np.sqrt(365 * 24)  # Annualized from hourly
        
        return volatility if volatility > 0 else self.position_config.target_volatility

    def get_target_position_size(self, price: float, asset_volatility: Optional[float] = None) -> float:
        """
        Calculate volatility-targeted position size.
        
        Formula: position_size = (target_vol / asset_vol) * (balance / price) * leverage
        
        Args:
            price: Current asset price
            asset_volatility: Asset's volatility (if None, uses portfolio volatility as proxy)
        
        Returns:
            Target position size in asset units
        """
        if price <= 0:
            logger.warning("Invalid price for position sizing")
            return 0.0
        
        # Use provided volatility or calculate from portfolio
        current_vol = asset_volatility if asset_volatility else self.get_current_volatility()
        
        # Avoid division by zero
        if current_vol < 0.0001:
            current_vol = self.position_config.target_volatility
        
        # Volatility-targeted sizing: scale inversely with volatility
        vol_scalar = self.position_config.target_volatility / current_vol
        
        # Calculate base position size
        base_notional = self.balance * self.position_config.max_leverage * vol_scalar
        position_size = base_notional / price
        
        # Apply minimum position size
        if position_size < self.position_config.min_position_size:
            position_size = self.position_config.min_position_size
        
        # Cap at max notional
        if position_size * price > settings.MAX_TRADE_NOTIONAL:
            position_size = settings.MAX_TRADE_NOTIONAL / price
        
        logger.info(f"Position Sizing - Vol: {current_vol:.4f}, Target: {self.position_config.target_volatility:.4f}, "
                   f"Scalar: {vol_scalar:.2f}, Size: {position_size:.6f}")
        
        return position_size
    
    def reset_circuit_breaker(self):
        """Manually reset circuit breaker after review"""
        logger.warning("🔓 Circuit Breaker manually reset")
        self.is_halted = False
        self.circuit_breaker = CircuitBreakerState()
        # Clear trackers
        self.latency_tracker = LatencyTracker(max_samples=settings.ERROR_RATE_WINDOW)
        self.error_tracker = ErrorTracker(max_samples=settings.ERROR_RATE_WINDOW)
    
    def get_risk_status(self) -> dict:
        """Get current risk metrics for monitoring"""
        drawdown = (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0
        
        return {
            "is_halted": self.is_halted,
            "circuit_breaker": {
                "active": self.circuit_breaker.is_active,
                "reason": self.circuit_breaker.reason,
                "triggered_at": self.circuit_breaker.triggered_at.isoformat() if self.circuit_breaker.triggered_at else None
            },
            "metrics": {
                "balance": self.balance,
                "peak_balance": self.peak_balance,
                "drawdown_pct": drawdown,
                "daily_return_pct": self.daily_stats.daily_return,
                "current_volatility": self.get_current_volatility(),
                "avg_latency_ms": self.latency_tracker.avg_latency,
                "max_latency_ms": self.latency_tracker.max_latency,
                "error_rate": self.error_tracker.error_rate
            },
            "limits": {
                "max_drawdown_pct": settings.MAX_DRAWDOWN_PCT,
                "max_daily_loss_pct": settings.MAX_DAILY_LOSS_PCT,
                "max_latency_ms": settings.MAX_LATENCY_MS,
                "max_error_rate": settings.MAX_ERROR_RATE,
                "max_trade_notional": settings.MAX_TRADE_NOTIONAL
            }
        }
