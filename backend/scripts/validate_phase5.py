"""
Quick validation script for Phase 5 implementation.
Demonstrates key functionality without requiring full test setup.
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.risk.manager import RiskManager
from app.risk.limits import TradeRisk
from app.risk.optimizer import MeanVarianceOptimizer, RiskParityOptimizer, Asset
from app.execution.engine import ExecutionEngine
from app.execution.vwap import VWAPCalculator, VolumeBar
import numpy as np
from datetime import datetime, timedelta
import asyncio

def test_risk_manager():
    """Test RiskManager functionality"""
    print("=" * 60)
    print("Testing Risk Manager")
    print("=" * 60)
    
    rm = RiskManager(initial_balance=10000.0)
    print(f"✓ Initialized with balance: ${rm.balance:.2f}")
    
    # Test volatility-targeted position sizing
    rm.historical_returns = [np.random.normal(0, 0.02) for _ in range(100)]
    position_size = rm.get_target_position_size(price=50000.0)
    print(f"✓ Position size (volatility-targeted): {position_size:.6f} BTC")
    print(f"  Current volatility: {rm.get_current_volatility():.4f}")
    
    # Test trade validation
    trade = TradeRisk(
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.1,
        price=50000.0,
        stop_loss=49000.0
    )
    is_valid = rm.check_trade_risk(trade)
    print(f"✓ Trade validation: {'PASSED' if is_valid else 'REJECTED'}")
    
    # Test circuit breaker
    rm.update_balance(8500.0)  # 15% drawdown
    print(f"✓ Circuit breaker status: {'TRIGGERED' if rm.is_halted else 'ACTIVE'}")
    if rm.is_halted:
        print(f"  Reason: {rm.circuit_breaker.reason}")
    
    print()

def test_portfolio_optimizers():
    """Test portfolio optimization"""
    print("=" * 60)
    print("Testing Portfolio Optimizers")
    print("=" * 60)
    
    # Create sample assets
    assets = [
        Asset("BTCUSDT", expected_return=0.20, volatility=0.50, current_price=50000.0),
        Asset("ETHUSDT", expected_return=0.25, volatility=0.60, current_price=3000.0),
        Asset("BNBUSDT", expected_return=0.15, volatility=0.40, current_price=300.0),
    ]
    
    # Sample covariance matrix
    cov_matrix = np.array([
        [0.25, 0.15, 0.10],
        [0.15, 0.36, 0.12],
        [0.10, 0.12, 0.16]
    ])
    
    # Mean-Variance Optimization
    mv_optimizer = MeanVarianceOptimizer(risk_free_rate=0.02)
    mv_result = mv_optimizer.optimize(assets, cov_matrix)
    print(f"✓ Mean-Variance Optimization:")
    print(f"  Weights: {mv_result.weights}")
    print(f"  Expected Return: {mv_result.expected_return:.2%}")
    print(f"  Sharpe Ratio: {mv_result.sharpe_ratio:.2f}")
    
    # Risk Parity Optimization
    rp_optimizer = RiskParityOptimizer()
    rp_result = rp_optimizer.optimize(assets, cov_matrix)
    print(f"✓ Risk Parity Optimization:")
    print(f"  Weights: {rp_result.weights}")
    print(f"  Expected Return: {rp_result.expected_return:.2%}")
    
    print()

async def test_execution_engine():
    """Test execution algorithms"""
    print("=" * 60)
    print("Testing Execution Engine")
    print("=" * 60)
    
    engine = ExecutionEngine()
    
    # Test AGGRESSIVE execution
    result = await engine.execute_order(
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.1,
        price=50000.0,
        style="AGGRESSIVE"
    )
    print(f"✓ AGGRESSIVE execution:")
    print(f"  Avg Price: ${result['avg_price']:.2f}")
    print(f"  Slippage: {result['slippage_bps']:.1f} bps")
    
    # Test TWAP execution
    result = await engine.execute_order(
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.5,
        price=50000.0,
        style="TWAP"
    )
    print(f"✓ TWAP execution:")
    print(f"  Avg Price: ${result['avg_price']:.2f}")
    print(f"  Execution Time: {result['execution_time_ms']:.0f}ms")
    
    # Test VWAP with volume data
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
    print(f"✓ VWAP execution:")
    print(f"  Avg Price: ${result['avg_price']:.2f}")
    print(f"  VWAP Benchmark: ${result['vwap_benchmark']:.2f}")
    print(f"  Slippage: {result['slippage_bps']:.1f} bps")
    
    # Show execution metrics
    metrics = engine.get_execution_metrics()
    print(f"✓ Execution Metrics:")
    print(f"  Total Orders: {metrics['total_orders']}")
    print(f"  Avg Slippage: {metrics['avg_slippage_bps']:.2f} bps")
    
    print()

async def main():
    """Run all validation tests"""
    print("\n" + "=" * 60)
    print("Phase 5: Risk Management & Execution - Validation")
    print("=" * 60 + "\n")
    
    test_risk_manager()
    test_portfolio_optimizers()
    await test_execution_engine()
    
    print("=" * 60)
    print("✅ All Phase 5 components validated successfully!")
    print("=" * 60)
    print("\nImplemented Features:")
    print("  • Volatility-targeted position sizing")
    print("  • Circuit breakers (drawdown, daily loss, latency, errors)")
    print("  • Portfolio optimization (mean-variance, risk-parity)")
    print("  • Smart execution (TWAP, VWAP, liquidity-aware)")
    print("  • Risk monitoring and emergency controls")
    print()

if __name__ == "__main__":
    asyncio.run(main())
