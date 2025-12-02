"""
Tests for Execution Engine (TWAP/VWAP).
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from app.execution.engine import ExecutionEngine
from app.execution.vwap import VWAPCalculator, VolumeBar

class TestExecutionEngine:
    """Test suite for execution engine"""
    
    @pytest.mark.asyncio
    async def test_aggressive_execution(self):
        """Test aggressive execution mode"""
        engine = ExecutionEngine()
        
        result = await engine.execute_order(
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.1,
            price=50000.0,
            style="AGGRESSIVE"
        )
        
        assert result["status"] == "FILLED"
        assert result["quantity"] == 0.1
        assert result["avg_price"] > 0
        assert "slippage_bps" in result
    
    @pytest.mark.asyncio
    async def test_passive_execution(self):
        """Test passive execution mode"""
        engine = ExecutionEngine()
        
        result = await engine.execute_order(
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.1,
            price=50000.0,
            style="PASSIVE"
        )
        
        assert result["status"] == "FILLED"
        # Passive should have better pricing (negative slippage)
        assert result["slippage_bps"] <= 0
    
    @pytest.mark.asyncio
    async def test_twap_execution(self):
        """Test TWAP execution"""
        engine = ExecutionEngine()
        
        result = await engine.execute_order(
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.5,
            price=50000.0,
            style="TWAP"
        )
        
        assert result["status"] == "FILLED"
        assert result["quantity"] == 0.5
        # TWAP should have executed multiple slices
        assert result["execution_time_ms"] > 0
    
    @pytest.mark.asyncio
    async def test_vwap_execution(self):
        """Test VWAP execution with volume data"""
        engine = ExecutionEngine()
        
        # Create mock volume bars
        base_time = datetime.now()
        volume_bars = [
            VolumeBar(base_time + timedelta(minutes=i), 50000 + i*10, 100 + i*10)
            for i in range(10)
        ]
        
        result = await engine.execute_order(
            symbol="BTCUSDT",
            side="BUY",
            quantity=1.0,
            price=50000.0,
            style="VWAP",
            volume_bars=volume_bars
        )
        
        assert result["status"] == "FILLED"
        assert "vwap_benchmark" in result
        assert result["vwap_benchmark"] > 0
    
    @pytest.mark.asyncio
    async def test_execution_metrics(self):
        """Test execution metrics tracking"""
        engine = ExecutionEngine()
        
        # Execute multiple orders
        for _ in range(5):
            await engine.execute_order(
                symbol="BTCUSDT",
                side="BUY",
                quantity=0.1,
                price=50000.0,
                style="AGGRESSIVE"
            )
        
        metrics = engine.get_execution_metrics()
        
        assert metrics["total_orders"] == 5
        assert "avg_slippage_bps" in metrics
        assert "total_slippage" in metrics

class TestVWAPCalculator:
    """Test suite for VWAP calculator"""
    
    def create_volume_bars(self):
        """Create sample volume bars"""
        base_time = datetime.now()
        return [
            VolumeBar(base_time + timedelta(minutes=i), 50000 + i*100, 100 + i*5)
            for i in range(20)
        ]
    
    def test_vwap_calculation(self):
        """Test VWAP calculation"""
        bars = self.create_volume_bars()
        calculator = VWAPCalculator(bars)
        
        vwap = calculator.calculate_vwap()
        
        assert vwap > 0
        # VWAP should be within range of prices
        assert vwap >= min(b.price for b in bars)
        assert vwap <= max(b.price for b in bars)
    
    def test_volume_profile(self):
        """Test volume profile creation"""
        bars = self.create_volume_bars()
        calculator = VWAPCalculator(bars)
        
        profile = calculator.create_volume_profile()
        
        assert len(profile) == len(bars)
        # Should sum to 1
        assert abs(sum(profile) - 1.0) < 0.01
        # All values should be non-negative
        assert all(v >= 0 for v in profile)
    
    def test_execution_schedule(self):
        """Test VWAP execution schedule creation"""
        bars = self.create_volume_bars()
        calculator = VWAPCalculator(bars)
        
        schedule = calculator.create_execution_schedule(
            total_quantity=1.0,
            start_time=datetime.now(),
            num_slices=5
        )
        
        assert len(schedule) == 5
        # Total quantity should sum to target
        total_qty = sum(s.quantity for s in schedule)
        assert abs(total_qty - 1.0) < 0.01
    
    def test_volume_weighted_price_range(self):
        """Test volume-weighted price range calculation"""
        bars = self.create_volume_bars()
        calculator = VWAPCalculator(bars)
        
        low, high = calculator.get_volume_weighted_price_range(percentile=0.8)
        
        assert low > 0
        assert high > low
        # Should be within overall price range
        assert low >= min(b.price for b in bars)
        assert high <= max(b.price for b in bars)
    
    def test_empty_volume_bars(self):
        """Test VWAP calculator handles empty data"""
        calculator = VWAPCalculator([])
        
        vwap = calculator.calculate_vwap()
        assert vwap == 0.0
        
        profile = calculator.create_volume_profile()
        assert len(profile) == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
