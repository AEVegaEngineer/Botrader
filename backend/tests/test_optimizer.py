"""
Tests for Portfolio Optimization module.
"""
import pytest
import numpy as np
from app.risk.optimizer import (
    MeanVarianceOptimizer,
    RiskParityOptimizer,
    Asset,
    calculate_covariance_matrix
)

class TestPortfolioOptimizers:
    """Test suite for portfolio optimizers"""
    
    def create_test_assets(self):
        """Create sample assets for testing"""
        return [
            Asset(symbol="BTCUSDT", expected_return=0.20, volatility=0.50, current_price=50000.0),
            Asset(symbol="ETHUSDT", expected_return=0.25, volatility=0.60, current_price=3000.0),
            Asset(symbol="BNBUSDT", expected_return=0.15, volatility=0.40, current_price=300.0),
        ]
    
    def create_test_covariance_matrix(self):
        """Create sample covariance matrix"""
        # Simplified covariance matrix
        cov = np.array([
            [0.25, 0.15, 0.10],   # BTC variances and covariances
            [0.15, 0.36, 0.12],   # ETH
            [0.10, 0.12, 0.16]    # BNB
        ])
        return cov
    
    def test_mean_variance_optimizer(self):
        """Test mean-variance optimization"""
        assets = self.create_test_assets()
        cov_matrix = self.create_test_covariance_matrix()
        
        optimizer = MeanVarianceOptimizer(risk_free_rate=0.02)
        result = optimizer.optimize(assets, cov_matrix)
        
        # Check weights sum to 1
        total_weight = sum(result.weights.values())
        assert abs(total_weight - 1.0) < 0.01
        
        # Check all weights are non-negative
        for weight in result.weights.values():
            assert weight >= 0
        
        # Check we have positive expected return and sharpe
        assert result.expected_return > 0
        assert result.sharpe_ratio > 0
    
    def test_risk_parity_optimizer(self):
        """Test risk parity optimization"""
        assets = self.create_test_assets()
        cov_matrix = self.create_test_covariance_matrix()
        
        optimizer = RiskParityOptimizer()
        result = optimizer.optimize(assets, cov_matrix)
        
        # Check weights sum to 1
        total_weight = sum(result.weights.values())
        assert abs(total_weight - 1.0) < 0.01
        
        # Check all weights are non-negative
        for weight in result.weights.values():
            assert weight >= 0
        
        # Lower volatility assets should have higher weights in risk parity
        assert result.weights["BNBUSDT"] > result.weights["ETHUSDT"]
    
    def test_covariance_calculation(self):
        """Test covariance matrix calculation from returns"""
        returns_data = {
            "BTCUSDT": [0.01, -0.02, 0.03, -0.01, 0.02],
            "ETHUSDT": [0.02, -0.03, 0.04, -0.02, 0.03],
            "BNBUSDT": [0.01, -0.01, 0.02, -0.01, 0.01]
        }
        
        symbols, cov_matrix = calculate_covariance_matrix(returns_data)
        
        # Check dimensions
        assert len(symbols) == 3
        assert cov_matrix.shape == (3, 3)
        
        # Check symmetry
        assert np.allclose(cov_matrix, cov_matrix.T)
        
        # Check diagonal is positive (variances)
        for i in range(3):
            assert cov_matrix[i, i] > 0
    
    def test_optimizer_with_single_asset(self):
        """Test optimizers handle single asset correctly"""
        assets = [Asset(symbol="BTCUSDT", expected_return=0.20, volatility=0.50, current_price=50000.0)]
        cov_matrix = np.array([[0.25]])
        
        mv_optimizer = MeanVarianceOptimizer()
        result = mv_optimizer.optimize(assets, cov_matrix)
        
        # Should allocate 100% to the only asset
        assert abs(result.weights["BTCUSDT"] - 1.0) < 0.01
    
    def test_optimizer_with_zeros(self):
        """Test optimizers handle edge cases"""
        assets = self.create_test_assets()
        
        # All zeros expected return
        for asset in assets:
            asset.expected_return = 0.0
        
        cov_matrix = self.create_test_covariance_matrix()
        
        optimizer = MeanVarianceOptimizer()
        result = optimizer.optimize(assets, cov_matrix)
        
        # Should still produce valid weights
        assert abs(sum(result.weights.values()) - 1.0) < 0.01

class TestPortfolioMetrics:
    """Test portfolio risk metrics"""
    
    def test_portfolio_metrics(self):
        """Test basic portfolio metric calculations"""
        # This would test the Portfolio class from portfolio.py
        # which we'll add if needed
        pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
