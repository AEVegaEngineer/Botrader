"""
Intervention logging for manual trading actions.
Provides audit trail for emergency stops, strategy changes, and manual trades.
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)

class InterventionType(Enum):
    """Types of manual interventions"""
    EMERGENCY_STOP = "emergency_stop"
    RESUME_TRADING = "resume_trading"
    STRATEGY_CHANGE = "strategy_change"
    MANUAL_TRADE = "manual_trade"
    POSITION_CLOSE = "position_close"
    RISK_LIMIT_OVERRIDE = "risk_limit_override"
    CONFIGURATION_CHANGE = "configuration_change"

@dataclass
class Intervention:
    """Record of a manual intervention"""
    id: str
    timestamp: datetime
    type: InterventionType
    user: str  # User ID or username
    action: str  # Human-readable description
    reason: str
    details: Dict[str, Any]  # Additional context
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'type': self.type.value,
            'user': self.user,
            'action': self.action,
            'reason': self.reason,
            'details': self.details
        }

class InterventionLog:
    """
    Logger for manual interventions.
    Maintains audit trail of all manual actions.
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
        
        self.interventions: List[Intervention] = []
        self._next_id = 1
        self._initialized = True
        
        logger.info("Intervention Log initialized")
    
    def log(
        self,
        intervention_type: InterventionType,
        user: str,
        action: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a manual intervention.
        
        Args:
            intervention_type: Type of intervention
            user: User performing the action
            action: Human-readable description
            reason: Reason for the intervention
            details: Additional context
            
        Returns:
            Intervention ID
        """
        intervention_id = f"INT-{self._next_id:06d}"
        self._next_id += 1
        
        intervention = Intervention(
            id=intervention_id,
            timestamp=datetime.now(),
            type=intervention_type,
            user=user,
            action=action,
            reason=reason,
            details=details or {}
        )
        
        self.interventions.append(intervention)
        
        logger.critical(
            f"🔔 INTERVENTION: [{intervention_type.value}] {action} by {user} - {reason}"
        )
        
        return intervention_id
    
    def get_all(
        self,
        limit: Optional[int] = None,
        intervention_type: Optional[InterventionType] = None,
        user: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Intervention]:
        """
        Get interventions with optional filtering.
        
        Args:
            limit: Max number of interventions to return (most recent first)
            intervention_type: Filter by type
            user: Filter by user
            start_date: Filter by start date
            end_date: Filter by end date
            
        Returns:
            List of interventions
        """
        filtered = self.interventions
        
        # Apply filters
        if intervention_type:
            filtered = [i for i in filtered if i.type == intervention_type]
        
        if user:
            filtered = [i for i in filtered if i.user == user]
        
        if start_date:
            filtered = [i for i in filtered if i.timestamp >= start_date]
        
        if end_date:
            filtered = [i for i in filtered if i.timestamp <= end_date]
        
        # Sort by timestamp (most recent first)
        filtered = sorted(filtered, key=lambda x: x.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            filtered = filtered[:limit]
        
        return filtered
    
    def get_by_id(self, intervention_id: str) -> Optional[Intervention]:
        """Get intervention by ID"""
        for intervention in self.interventions:
            if intervention.id == intervention_id:
                return intervention
        return None
    
    def get_recent(self, count: int = 10) -> List[Intervention]:
        """Get most recent interventions"""
        return self.get_all(limit=count)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get intervention statistics"""
        total = len(self.interventions)
        
        by_type = {}
        for intervention in self.interventions:
            type_name = intervention.type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1
        
        by_user = {}
        for intervention in self.interventions:
            by_user[intervention.user] = by_user.get(intervention.user, 0) + 1
        
        return {
            'total_interventions': total,
            'by_type': by_type,
            'by_user': by_user,
            'latest': self.get_recent(1)[0].to_dict() if self.interventions else None
        }
    
    def clear_old_entries(self, days: int = 90):
        """
        Clear intervention logs older than specified days.
        
        Args:
            days: Keep only interventions from last N days
        """
        cutoff = datetime.now() - timedelta(days=days)
        original_count = len(self.interventions)
        
        self.interventions = [i for i in self.interventions if i.timestamp >= cutoff]
        
        removed = original_count - len(self.interventions)
        if removed > 0:
            logger.info(f"Cleared {removed} old intervention logs (older than {days} days)")
    
    def export_to_dict(self) -> List[dict]:
        """Export all interventions to dict format"""
        return [i.to_dict() for i in self.interventions]
    
    def export_to_json(self, filepath: str):
        """Export interventions to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.export_to_dict(), f, indent=2)
        logger.info(f"Exported {len(self.interventions)} interventions to {filepath}")

# Global intervention log instance
intervention_log = InterventionLog()

def get_intervention_log() -> InterventionLog:
    """Get the global intervention log instance"""
    return intervention_log

# Helper functions for common interventions
def log_emergency_stop(user: str, reason: str, details: Optional[Dict] = None) -> str:
    """Log emergency stop intervention"""
    return intervention_log.log(
        InterventionType.EMERGENCY_STOP,
        user=user,
        action="Emergency stop triggered",
        reason=reason,
        details=details
    )

def log_strategy_change(user: str, old_strategy: str, new_strategy: str, reason: str) -> str:
    """Log strategy change intervention"""
    return intervention_log.log(
        InterventionType.STRATEGY_CHANGE,
        user=user,
        action=f"Changed strategy from {old_strategy} to {new_strategy}",
        reason=reason,
        details={'old_strategy': old_strategy, 'new_strategy': new_strategy}
    )

def log_manual_trade(user: str, symbol: str, side: str, quantity: float, reason: str) -> str:
    """Log manual trade intervention"""
    return intervention_log.log(
        InterventionType.MANUAL_TRADE,
        user=user,
        action=f"Manual {side} {quantity} {symbol}",
        reason=reason,
        details={'symbol': symbol, 'side': side, 'quantity': quantity}
    )
