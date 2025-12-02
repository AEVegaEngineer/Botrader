"""
Tests for Risk Management module.
"""
import pytest
import numpy as np
from datetime import date, datetime
from app.risk.manager import RiskManager
from app.risk.limits import TradeRisk, CircuitBreakerState
from app.core.config import settings

class TestRiskManager:
    """Test suite for RiskManager"""
    
    def test_initialization(self):
        """Test risk manager initializes correctly"""
        initial_balance = 10000.0
        rm = RiskManager(initial_balance)
        
        assert rm.balance == initial_balance
        assert rm.peak_balance == initial_balance
        assert not rm.is_halted
        assert not rm.circuit_breaker.is_active
    
    def test_drawdown_circuit_breaker(self):
        """Test circuit breaker triggers on max drawdown"""
        initial_balance = 10000.0
        rm = RiskManager(initial_balance)
        
        # Simulate a large loss
        new_balance = initial_balance * (1 - settings.MAX_DRAWDOWN_PCT - 0.01)
        rm.update_balance(new_balance)
        
        assert rm.is_halted
        assert rm.circuit_breaker.is_active
        assert "Drawdown" in rm.circuit_breaker.reason
    
    def test_daily_loss_circuit_breaker(self):
        """Test circuit breaker triggers on max daily loss"""
        initial_balance = 10000.0
        rm = RiskManager(initial_balance)
        
        # Simulate daily loss exceeding limit
        new_balance = initial_balance * (1 - settings.MAX_DAILY_LOSS_PCT - 0.01)
        rm.update_balance(new_balance)
        
        assert rm.is_halted
        assert rm.circuit_breaker.is_active
    
    def test_latency_circuit_breaker(self):
        """Test circuit breaker triggers on high latency"""
        rm = RiskManager(10000.0)
        
        # Add high latency samples
        for _ in range(10):
            rm.add_latency_sample(settings.MAX_LATENCY_MS + 1000)
        
        assert rm.is_halted
        assert rm.circuit_breaker.is_active
        assert "Latency" in rm.circuit_breaker.reason
    
    def test_error_rate_circuit_breaker(self):
        """Test circuit breaker triggers on high error rate"""
        rm = RiskManager(10000.0)
        
        # Add mostly failed operations
        for i in range(100):
            rm.add_operation_result(i >= 90)  # 90% failure rate
        
        assert rm.is_halted
        assert rm.circuit_breaker.is_active
        assert "Error Rate" in rm.circuit_breaker.reason
    
    def test_trade_risk_validation_notional(self):
        """Test trade risk validation rejects oversized trades"""
        rm = RiskManager(10000.0)
        
        # Create trade exceeding max notional
        trade = TradeRisk(
            symbol="BTCUSDT",
            side="BUY",
            quantity=1.0,
            price=settings.MAX_TRADE_NOTIONAL + 1000,
            stop_loss=0.0
        )
        
        assert not rm.check_trade_risk(trade)
    
    def test_trade_risk_validation_leverage(self):
        """Test trade risk validation rejects over-leveraged trades"""
        rm = RiskManager(10000.0)
        
        # Create trade exceeding leverage limit
        trade = TradeRisk(
            symbol="BTCUSDT",
            side="BUY",
            quantity=2.0,  # 2 BTC
            price=50000.0,  # = $100k notional, 10x leverage on $10k balance
            stop_loss=0.0
        )
        
        assert not rm.check_trade_risk(trade)
    
    def test_trade_risk_validation_stop_loss(self):
        """Test trade risk validation with stop loss"""
        rm = RiskManager(10000.0)
        
        # Create trade with acceptable stop loss risk
        trade = TradeRisk(
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.1,
            price=50000.0,
            stop_loss=49000.0  # $100 risk on $10k balance = 1%
        )
        
        assert rm.check_trade_risk(trade)
        
        # Create trade with too much stop loss risk
        trade_risky = TradeRisk(
            symbol="BTCUSDT",
            side="BUY",
            quantity=1.0,
            price=50000.0,
            stop_loss=45000.0  # $5000 risk > 1% of $10k
        )
        
        assert not rm.check_trade_risk(trade_risky)
    
    def test_volatility_position_sizing(self):
        """Test volatility-targeted position sizing"""
        rm = RiskManager(10000.0)
        
        # Add some historical returns to calculate volatility
        for _ in range(100):
            rm.historical_returns.append(np.random.normal(0, 0.01))
        
        # Get position size
        price = 50000.0
        position_size = rm.get_target_position_size(price)
        
        assert position_size > 0
        assert position_size * price <= settings.MAX_TRADE_NOTIONAL
    
    def test_volatility_scaling(self):
        """Test position size scales inversely with volatility"""
        rm = RiskManager(10000.0)
        
        # Low volatility scenario
        for _ in range(100):
            rm.historical_returns.append(np.random.normal(0, 0.005))  # Low vol
        
        low_vol_size = rm.get_target_position_size(50000.0)
        
        # High volatility scenario
        rm.historical_returns = []
        for _ in range(100):
            rm.historical_returns.append(np.random.normal(0, 0.05))  # High vol
        
        high_vol_size = rm.get_target_position_size(50000.0)
        
        # Higher vol should result in smaller position
        assert high_vol_size < low_vol_size
    
    def test_reset_circuit_breaker(self):
        """Test circuit breaker can be reset"""
        rm = RiskManager(10000.0)
        
        # Trigger circuit breaker
        rm.update_balance(5000.0)  # Large drawdown
        assert rm.is_halted
        
        # Reset
        rm.reset_circuit_breaker()
        assert not rm.is_halted
        assert not rm.circuit_breaker.is_active
    
    def test_get_risk_status(self):
        """Test risk status dictionary"""
        rm = RiskManager(10000.0)
        rm.update_balance(9500.0)
        
        status = rm.get_risk_status()
        
        assert "is_halted" in status
        assert "circuit_breaker" in status
        assert "metrics" in status
        assert "limits" in status
        
        assert status["metrics"]["balance"] == 9500.0
        assert status["metrics"]["peak_balance"] == 10000.0
        assert status["metrics"]["drawdown_pct"] == 0.05

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
