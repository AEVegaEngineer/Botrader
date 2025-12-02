"""
Strategy registry for managing multiple trading strategies.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)

class StrategyType(Enum):
    """Types of trading strategies"""
    BASELINE = "baseline"
    TREE_MODEL = "tree_model"
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    CNN = "cnn"
    LOB_MODEL = "lob_model"
    RL_AGENT = "rl_agent"
    ENSEMBLE = "ensemble"
    RULE_BASED = "rule_based"

@dataclass
class BacktestStats:
    """Backtest statistics for a strategy"""
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    total_return: float
    win_rate: float
    total_trades: int
    backtest_start: datetime
    backtest_end: datetime
    
    def to_dict(self) -> dict:
        return {
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'total_return': self.total_return,
            'win_rate': self.win_rate,
            'total_trades': self.total_trades,
            'backtest_start': self.backtest_start.isoformat(),
            'backtest_end': self.backtest_end.isoformat()
        }

@dataclass
class StrategyMetadata:
    """Metadata for a trading strategy"""
    id: str
    name: str
    type: StrategyType
    version: str
    description: str
    backtest_stats: Optional[BacktestStats] = None
    config: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    is_active: bool = False
    deployment_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type.value,
            'version': self.version,
            'description': self.description,
            'backtest_stats': self.backtest_stats.to_dict() if self.backtest_stats else None,
            'config': self.config,
            'created_at': self.created_at.isoformat(),
            'last_modified': self.last_modified.isoformat(),
            'is_active': self.is_active,
            'deployment_history': self.deployment_history
        }

class StrategyRegistry:
    """
    Singleton registry for managing trading strategies.
    Stores metadata, tracks active strategy, and manages deployment history.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.strategies: Dict[str, StrategyMetadata] = {}
        self.active_strategy_id: Optional[str] = None
        self._initialized = True
        
        logger.info("Strategy Registry initialized")
    
    def register(self, strategy: StrategyMetadata) -> bool:
        """
        Register a new strategy.
        
        Args:
            strategy: StrategyMetadata object
            
        Returns:
            True if registered successfully
        """
        if strategy.id in self.strategies:
            logger.warning(f"Strategy {strategy.id} already registered, updating...")
        
        strategy.last_modified = datetime.now()
        self.strategies[strategy.id] = strategy
        
        logger.info(f"Registered strategy: {strategy.name} ({strategy.id})")
        return True
    
    def unregister(self, strategy_id: str) -> bool:
        """
        Unregister a strategy.
        
        Args:
            strategy_id: Strategy ID to remove
            
        Returns:
            True if unregistered successfully
        """
        if strategy_id not in self.strategies:
            logger.warning(f"Strategy {strategy_id} not found")
            return False
        
        if self.active_strategy_id == strategy_id:
            logger.error(f"Cannot unregister active strategy {strategy_id}")
            return False
        
        del self.strategies[strategy_id]
        logger.info(f"Unregistered strategy: {strategy_id}")
        return True
    
    def activate(self, strategy_id: str, reason: str = "Manual activation") -> bool:
        """
        Activate a strategy (deactivates current active strategy).
        
        Args:
            strategy_id: Strategy ID to activate
            reason: Reason for activation
            
        Returns:
            True if activated successfully
        """
        if strategy_id not in self.strategies:
            logger.error(f"Strategy {strategy_id} not found")
            return False
        
        # Deactivate current strategy
        if self.active_strategy_id:
            old_strategy = self.strategies[self.active_strategy_id]
            old_strategy.is_active = False
            old_strategy.deployment_history.append({
                'action': 'deactivated',
                'timestamp': datetime.now().isoformat(),
                'reason': f"Replaced by {strategy_id}"
            })
        
        # Activate new strategy
        new_strategy = self.strategies[strategy_id]
        new_strategy.is_active = True
        new_strategy.deployment_history.append({
            'action': 'activated',
            'timestamp': datetime.now().isoformat(),
            'reason': reason
        })
        
        self.active_strategy_id = strategy_id
        
        logger.info(f"Activated strategy: {new_strategy.name} ({strategy_id})")
        return True
    
    def deactivate(self, strategy_id: str, reason: str = "Manual deactivation") -> bool:
        """
        Deactivate a strategy.
        
        Args:
            strategy_id: Strategy ID to deactivate
            reason: Reason for deactivation
            
        Returns:
            True if deactivated successfully
        """
        if strategy_id not in self.strategies:
            logger.error(f"Strategy {strategy_id} not found")
            return False
        
        strategy = self.strategies[strategy_id]
        strategy.is_active = False
        strategy.deployment_history.append({
            'action': 'deactivated',
            'timestamp': datetime.now().isoformat(),
            'reason': reason
        })
        
        if self.active_strategy_id == strategy_id:
            self.active_strategy_id = None
        
        logger.info(f"Deactivated strategy: {strategy.name} ({strategy_id})")
        return True
    
    def get(self, strategy_id: str) -> Optional[StrategyMetadata]:
        """Get strategy by ID"""
        return self.strategies.get(strategy_id)
    
    def get_active(self) -> Optional[StrategyMetadata]:
        """Get currently active strategy"""
        if self.active_strategy_id:
            return self.strategies.get(self.active_strategy_id)
        return None
    
    def list_all(self) -> List[StrategyMetadata]:
        """List all registered strategies"""
        return list(self.strategies.values())
    
    def list_by_type(self, strategy_type: StrategyType) -> List[StrategyMetadata]:
        """List strategies by type"""
        return [s for s in self.strategies.values() if s.type == strategy_type]
    
    def get_deployment_history(self, strategy_id: str) -> List[Dict[str, Any]]:
        """Get deployment history for a strategy"""
        strategy = self.get(strategy_id)
        if strategy:
            return strategy.deployment_history
        return []
    
    def update_config(self, strategy_id: str, config: Dict[str, Any]) -> bool:
        """Update strategy configuration"""
        strategy = self.get(strategy_id)
        if strategy:
            strategy.config.update(config)
            strategy.last_modified = datetime.now()
            logger.info(f"Updated config for strategy: {strategy_id}")
            return True
        return False
    
    def to_dict(self) -> dict:
        """Export registry to dictionary"""
        return {
            'strategies': {sid: s.to_dict() for sid, s in self.strategies.items()},
            'active_strategy_id': self.active_strategy_id
        }

# Global registry instance
registry = StrategyRegistry()

def get_registry() -> StrategyRegistry:
    """Get the global strategy registry instance"""
    return registry
