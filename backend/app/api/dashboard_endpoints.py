"""
API endpoints for performance metrics, strategy management, and expl ainability.
Extends the Phase 5 risk management endpoints.
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["dashboard"])

# These would be injected via dependency injection in production
_performance_analyzer = None
_strategy_registry = None
_intervention_log = None
_current_explainer = None

# ============================================================================
# Performance Metrics Endpoints
# ============================================================================

class PerformanceMetricsResponse(BaseModel):
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    current_drawdown: float
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    turnover: float
    total_fees: float
    fees_pct_of_pnl: float

@router.get("/performance/metrics", response_model=PerformanceMetricsResponse)
async def get_performance_metrics():
    """Get current performance metrics"""
    try:
        # In production, would fetch from live trading data
        # For now, return mock data
        return PerformanceMetricsResponse(
            total_return=0.15,
            annualized_return=0.45,
            sharpe_ratio=1.8,
            sortino_ratio=2.2,
            calmar_ratio=3.5,
            max_drawdown=-0.08,
            current_drawdown=-0.02,
            total_trades=150,
            win_rate=0.58,
            avg_win=250.0,
            avg_loss=180.0,
            profit_factor=1.6,
            turnover=2.5,
            total_fees=1200.0,
            fees_pct_of_pnl=0.08
        )
    except Exception as e:
        logger.error(f"Error fetching performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance/history")
async def get_performance_history(
    period: str = Query("30d", description="Time period: 7d, 30d, 90d, 1y, all"),
    interval: str = Query("1d", description="Data interval: 1h, 1d, 1w")
):
    """Get historical performance data"""
    try:
        # Mock data - in production, query from database
        return {
            "timestamps": [],
            "equity_curve": [],
            "returns": [],
            "sharpe_rolling": [],
            "drawdown": []
        }
    except Exception as e:
        logger.error(f"Error fetching performance history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance/by-strategy")
async def get_performance_by_strategy():
    """Get performance breakdown by strategy"""
    try:
        # Mock data
        return {
            "strategies": [
                {
                    "name": "Baseline RSI",
                    "total_return": 0.12,
                    "sharpe_ratio": 1.5,
                    "total_trades": 50
                },
                {
                    "name": "LSTM Model",
                    "total_return": 0.18,
                    "sharpe_ratio": 2.1,
                    "total_trades": 100
                }
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching strategy performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Strategy Management Endpoints
# ============================================================================

@router.get("/strategies")
async def list_strategies():
    """List all registered strategies"""
    try:
        if _strategy_registry is None:
            # Return mock data if registry not initialized
            return {
                "strategies": [
                    {
                        "id": "baseline-rsi",
                        "name": "Baseline RSI Strategy",
                        "type": "rule_based",
                        "version": "1.0.0",
                        "is_active": True,
                        "backtest_stats": {
                            "sharpe_ratio": 1.5,
                            "max_drawdown": -0.10,
                            "total_return": 0.25
                        }
                    },
                    {
                        "id": "lstm-v1",
                        "name": "LSTM Price Predictor",
                        "type": "lstm",
                        "version": "1.0.0",
                        "is_active": False,
                        "backtest_stats": {
                            "sharpe_ratio": 2.1,
                            "max_drawdown": -0.08,
                            "total_return": 0.35
                        }
                    }
                ]
            }
        
        strategies = _strategy_registry.list_all()
        return {
            "strategies": [s.to_dict() for s in strategies]
        }
    except Exception as e:
        logger.error(f"Error listing strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/strategies/{strategy_id}")
async def get_strategy(strategy_id: str):
    """Get details of a specific strategy"""
    try:
        if _strategy_registry is None:
            raise HTTPException(status_code=500, detail="Strategy registry not initialized")
        
        strategy = _strategy_registry.get(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")
        
        return strategy.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class ActivateStrategyRequest(BaseModel):
    reason: str = "Manual activation"

@router.post("/strategies/{strategy_id}/activate")
async def activate_strategy(strategy_id: str, request: ActivateStrategyRequest):
    """Activate a strategy"""
    try:
        if _strategy_registry is None:
            raise HTTPException(status_code=500, detail="Strategy registry not initialized")
        
        success = _strategy_registry.activate(strategy_id, reason=request.reason)
        
        if not success:
            raise HTTPException(status_code=400, detail=f"Failed to activate strategy {strategy_id}")
        
        # Log intervention
        if _intervention_log:
            from app.core.intervention_log import log_strategy_change
            old_strategy = _strategy_registry.active_strategy_id or "None"
            log_strategy_change("system", old_strategy, strategy_id, request.reason)
        
        return {
            "success": True,
            "message": f"Strategy {strategy_id} activated",
            "strategy_id": strategy_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error activating strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/strategies/{strategy_id}/deactivate")
async def deactivate_strategy(strategy_id: str):
    """Deactivate a strategy"""
    try:
        if _strategy_registry is None:
            raise HTTPException(status_code=500, detail="Strategy registry not initialized")
        
        success = _strategy_registry.deactivate(strategy_id)
        
        if not success:
            raise HTTPException(status_code=400, detail=f"Failed to deactivate strategy {strategy_id}")
        
        return {
            "success": True,
            "message": f"Strategy {strategy_id} deactivated"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deactivating strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/strategies/active")
async def get_active_strategy():
    """Get currently active strategy"""
    try:
        if _strategy_registry is None:
            return {"active_strategy": None}
        
        active = _strategy_registry.get_active()
        
        if active:
            return {"active_strategy": active.to_dict()}
        else:
            return {"active_strategy": None}
    except Exception as e:
        logger.error(f"Error fetching active strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Intervention Log Endpoints
# ============================================================================

@router.get("/interventions")
async def get_interventions(
    limit: int = Query(50, description="Max number of interventions to return"),
    intervention_type: Optional[str] = Query(None, description="Filter by type")
):
    """Get intervention log"""
    try:
        if _intervention_log is None:
            return {"interventions": []}
        
        interventions = _intervention_log.get_all(limit=limit)
        
        return {
            "interventions": [i.to_dict() for i in interventions],
            "total": len(_intervention_log.interventions)
        }
    except Exception as e:
        logger.error(f"Error fetching interventions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/interventions/statistics")
async def get_intervention_statistics():
    """Get intervention statistics"""
    try:
        if _intervention_log is None:
            return {"total_interventions": 0}
        
        return _intervention_log.get_statistics()
    except Exception as e:
        logger.error(f"Error fetching intervention statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Model Explainability Endpoints
# ============================================================================

@router.get("/explainability/feature-importance")
async def get_feature_importance(
    model_id: Optional[str] = Query(None, description="Model ID (uses active if not specified)")
):
    """Get global feature importance"""
    try:
        # Mock data for now
        return {
            "features": ["RSI", "SMA_20", "Volume", "MACD", "BB_Width", "EMA_50"],
            "importance": [0.25, 0.20, 0.18, 0.15, 0.12, 0.10],
            "method": "SHAP TreeExplainer",
            "model_id": model_id or "active"
        }
    except Exception as e:
        logger.error(f"Error fetching feature importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class ExplainPredictionRequest(BaseModel):
    features: Dict[str, float]
    index: Optional[int] = 0

@router.post("/explainability/explain-prediction")
async def explain_prediction(request: ExplainPredictionRequest):
    """Explain a specific prediction"""
    try:
        # Mock explanation
        return {
            "base_value": 0.5,
            "prediction": 0.72,
            "contributions": [
                {"feature": "RSI", "value": 65.5, "contribution": 0.12},
                {"feature": "SMA_20", "value": 50100, "contribution": 0.08},
                {"feature": "Volume", "value": 1500000, "contribution": 0.02}
            ],
            "method": "SHAP TreeExplainer"
        }
    except Exception as e:
        logger.error(f"Error explaining prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/explainability/shap-summary")
async def get_shap_summary(
    model_id: Optional[str] = Query(None, description="Model ID"),
    max_features: int = Query(20, description="Max features to include")
):
    """Get SHAP summary data for visualization"""
    try:
        # Mock data
        return {
            "summary": [],
            "num_samples": 1000,
            "method": "SHAP summary"
        }
    except Exception as e:
        logger.error(f"Error fetching SHAP summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions to set module-level variables
def set_performance_analyzer(analyzer):
    global _performance_analyzer
    _performance_analyzer = analyzer

def set_strategy_registry(registry):
    global _strategy_registry
    _strategy_registry = registry

def set_intervention_log(log):
    global _intervention_log
    _intervention_log = log

def set_explainer(explainer):
    global _current_explainer
    _current_explainer = explainer
