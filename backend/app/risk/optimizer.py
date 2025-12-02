"""
Portfolio optimization module for multi-asset allocation.
Implements mean-variance and risk-parity optimizers.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)

@dataclass
class Asset:
    """Represents a single tradeable asset"""
    symbol: str
    expected_return: float  # Annualized expected return
    volatility: float  # Annualized volatility
    current_price: float
    
@dataclass
class OptimizationResult:
    """Result from portfolio optimization"""
    weights: Dict[str, float]  # Asset symbol -> weight
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    
class PortfolioOptimizer(ABC):
    """Abstract base class for portfolio optimizers"""
    
    @abstractmethod
    def optimize(self, assets: List[Asset], covariance_matrix: np.ndarray) -> OptimizationResult:
        """
        Optimize portfolio allocation.
        
        Args:
            assets: List of assets to allocate across
            covariance_matrix: Covariance matrix of asset returns (NxN)
            
        Returns:
            OptimizationResult with optimal weights
        """
        pass
    
    def _validate_inputs(self, assets: List[Asset], covariance_matrix: np.ndarray):
        """Validate optimizer inputs"""
        n = len(assets)
        if covariance_matrix.shape != (n, n):
            raise ValueError(f"Covariance matrix shape {covariance_matrix.shape} doesn't match {n} assets")
        
        if not np.allclose(covariance_matrix, covariance_matrix.T):
            raise ValueError("Covariance matrix must be symmetric")

class MeanVarianceOptimizer(PortfolioOptimizer):
    """
    Mean-Variance optimization using Markowitz portfolio theory.
    Maximizes Sharpe ratio (return/volatility).
    """
    
    def __init__(self, risk_free_rate: float = 0.0):
        self.risk_free_rate = risk_free_rate
    
    def optimize(self, assets: List[Asset], covariance_matrix: np.ndarray) -> OptimizationResult:
        """Find portfolio with maximum Sharpe ratio"""
        self._validate_inputs(assets, covariance_matrix)
        
        n = len(assets)
        expected_returns = np.array([asset.expected_return for asset in assets])
        
        # Objective: Minimize negative Sharpe ratio
        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            if portfolio_vol == 0:
                return 0
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            return -sharpe
        
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        
        # Bounds: 0 <= weight <= 1 (long-only)
        bounds = tuple((0.0, 1.0) for _ in range(n))
        
        # Initial guess: equal weights
        initial_weights = np.array([1.0 / n] * n)
        
        # Optimize
        result = minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            # Fall back to equal weights
            optimal_weights = initial_weights
        else:
            optimal_weights = result.x
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(optimal_weights, expected_returns)
        portfolio_vol = np.sqrt(np.dot(optimal_weights, np.dot(covariance_matrix, optimal_weights)))
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        # Convert to dict
        weights_dict = {asset.symbol: float(w) for asset, w in zip(assets, optimal_weights)}
        
        logger.info(f"Mean-Variance Optimization - Sharpe: {sharpe:.2f}, Return: {portfolio_return:.2%}, Vol: {portfolio_vol:.2%}")
        logger.info(f"Weights: {weights_dict}")
        
        return OptimizationResult(
            weights=weights_dict,
            expected_return=portfolio_return,
            expected_volatility=portfolio_vol,
            sharpe_ratio=sharpe
        )

class RiskParityOptimizer(PortfolioOptimizer):
    """
    Risk Parity optimization - equalizes risk contribution across assets.
    Each asset contributes equally to portfolio volatility.
    """
    
    def optimize(self, assets: List[Asset], covariance_matrix: np.ndarray) -> OptimizationResult:
        """Find portfolio with equal risk contributions"""
        self._validate_inputs(assets, covariance_matrix)
        
        n = len(assets)
        expected_returns = np.array([asset.expected_return for asset in assets])
        
        # Objective: Minimize sum of squared differences in risk contributions
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            if portfolio_vol == 0:
                return 1e10  # Large penalty
            
            # Marginal contribution to risk
            marginal_contrib = np.dot(covariance_matrix, weights) / portfolio_vol
            
            # Risk contribution of each asset
            risk_contrib = weights * marginal_contrib
            
            # Target: equal risk contribution
            target_contrib = portfolio_vol / n
            
            # Minimize squared deviations from target
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        
        # Bounds: 0 <= weight <= 1 (long-only)
        bounds = tuple((0.0, 1.0) for _ in range(n))
        
        # Initial guess: inverse volatility weights
        vols = np.array([asset.volatility for asset in assets])
        inv_vols = 1.0 / vols
        initial_weights = inv_vols / np.sum(inv_vols)
        
        # Optimize
        result = minimize(
            risk_parity_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            # Fall back to inverse volatility weights
            optimal_weights = initial_weights
        else:
            optimal_weights = result.x
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(optimal_weights, expected_returns)
        portfolio_vol = np.sqrt(np.dot(optimal_weights, np.dot(covariance_matrix, optimal_weights)))
        sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        # Convert to dict
        weights_dict = {asset.symbol: float(w) for asset, w in zip(assets, optimal_weights)}
        
        logger.info(f"Risk Parity Optimization - Return: {portfolio_return:.2%}, Vol: {portfolio_vol:.2%}")
        logger.info(f"Weights: {weights_dict}")
        
        return OptimizationResult(
            weights=weights_dict,
            expected_return=portfolio_return,
            expected_volatility=portfolio_vol,
            sharpe_ratio=sharpe
        )

def calculate_covariance_matrix(returns_data: Dict[str, List[float]]) -> tuple[List[str], np.ndarray]:
    """
    Calculate covariance matrix from historical returns.
    
    Args:
        returns_data: Dict of symbol -> list of returns
        
    Returns:
        Tuple of (symbols, covariance_matrix)
    """
    symbols = sorted(returns_data.keys())
    
    # Convert to numpy array
    returns_matrix = np.array([returns_data[symbol] for symbol in symbols])
    
    # Calculate covariance matrix
    cov_matrix = np.cov(returns_matrix)
    
    return symbols, cov_matrix
