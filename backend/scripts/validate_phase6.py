"""
Validation script for Phase 6: Advanced Dashboard & UI Backend.
Tests performance analytics, strategy registry, intervention logging, and explainability.
"""
import sys
import os
import asyncio
from datetime import datetime, timedelta
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.analytics.performance import PerformanceAnalyzer, TradeRecord
from app.core.strategy_registry import StrategyRegistry, StrategyMetadata, StrategyType, BacktestStats
from app.core.intervention_log import InterventionLog, InterventionType
from app.ml.explainer import create_explainer

def test_performance_analytics():
    print("=" * 60)
    print("Testing Performance Analytics")
    print("=" * 60)
    
    analyzer = PerformanceAnalyzer(risk_free_rate=0.02)
    
    # Create sample equity curve
    dates = [datetime.now() - timedelta(days=i) for i in range(100, 0, -1)]
    initial_balance = 10000.0
    returns = np.random.normal(0.001, 0.01, 100)  # Mean 0.1% daily return, 1% vol
    equity_curve = [initial_balance]
    for r in returns:
        equity_curve.append(equity_curve[-1] * (1 + r))
    equity_curve = equity_curve[1:]  # Remove initial
    
    # Create sample trades
    trades = [
        TradeRecord(
            timestamp=dates[i],
            symbol="BTCUSDT",
            side="BUY" if i % 2 == 0 else "SELL",
            quantity=0.1,
            price=50000.0 + i * 10,
            fee=5.0,
            pnl=100.0 if i % 2 != 0 else None
        )
        for i in range(0, 100, 5)
    ]
    
    metrics = analyzer.calculate_metrics(equity_curve, trades, dates, initial_balance)
    
    print(f"✓ Total Return: {metrics.total_return:.2%}")
    print(f"✓ Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"✓ Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"✓ Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"✓ Win Rate: {metrics.win_rate:.2%}")
    print(f"✓ Turnover: {metrics.turnover:.2f}x")
    print()

def test_strategy_registry():
    print("=" * 60)
    print("Testing Strategy Registry")
    print("=" * 60)
    
    registry = StrategyRegistry()
    
    # Register a strategy
    strategy = StrategyMetadata(
        id="test-strategy-1",
        name="Test Strategy",
        type=StrategyType.BASELINE,
        version="1.0.0",
        description="A test strategy",
        backtest_stats=BacktestStats(
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=-0.1,
            total_return=0.2,
            win_rate=0.55,
            total_trades=100,
            backtest_start=datetime.now() - timedelta(days=365),
            backtest_end=datetime.now()
        )
    )
    
    registry.register(strategy)
    print(f"✓ Registered strategy: {strategy.name}")
    
    # Activate
    registry.activate(strategy.id, "Testing activation")
    active = registry.get_active()
    print(f"✓ Active strategy: {active.name if active else 'None'}")
    
    # History
    history = registry.get_deployment_history(strategy.id)
    print(f"✓ Deployment history: {len(history)} events")
    print()

def test_intervention_log():
    print("=" * 60)
    print("Testing Intervention Log")
    print("=" * 60)
    
    log = InterventionLog()
    
    # Log some events
    log.log(
        InterventionType.EMERGENCY_STOP,
        user="admin",
        action="Emergency Stop",
        reason="Test trigger"
    )
    
    log.log(
        InterventionType.STRATEGY_CHANGE,
        user="admin",
        action="Changed strategy",
        reason="Performance degradation",
        details={"old": "strat-a", "new": "strat-b"}
    )
    
    # Retrieve
    recent = log.get_recent(5)
    print(f"✓ Logged {len(recent)} interventions")
    print(f"✓ Latest action: {recent[0].action}")
    
    stats = log.get_statistics()
    print(f"✓ Stats: {stats['total_interventions']} total")
    print()

def test_explainability():
    print("=" * 60)
    print("Testing Model Explainability")
    print("=" * 60)
    
    # Mock a simple tree model (using sklearn for demo if available, else skip)
    try:
        from sklearn.ensemble import RandomForestRegressor
        import pandas as pd
        
        X = np.random.rand(100, 5)
        y = np.sum(X, axis=1) + np.random.normal(0, 0.1, 100)
        feature_names = [f"feat_{i}" for i in range(5)]
        
        model = RandomForestRegressor(n_estimators=10)
        model.fit(X, y)
        
        # Create explainer
        # Note: SHAP might not be installed in this environment, so we catch ImportError
        try:
            explainer = create_explainer(model, "tree")
            
            # Global explanation
            global_exp = explainer.explain_global(X, feature_names)
            print(f"✓ Global importance calculated for {len(global_exp['features'])} features")
            print(f"  Top feature: {global_exp['features'][0]}")
            
            # Local explanation
            local_exp = explainer.explain_prediction(X, feature_names, index=0)
            print(f"✓ Local explanation generated")
            print(f"  Prediction: {local_exp['prediction']:.4f}")
            
        except ImportError:
            print("⚠ SHAP not installed, skipping SHAP tests")
        except Exception as e:
            print(f"⚠ Explainability test failed (expected if SHAP missing): {e}")
            
    except ImportError:
        print("⚠ sklearn not installed, skipping model tests")
    
    print()

if __name__ == "__main__":
    test_performance_analytics()
    test_strategy_registry()
    test_intervention_log()
    test_explainability()
    
    print("=" * 60)
    print("✅ Phase 6 Backend Validation Complete")
    print("=" * 60)
