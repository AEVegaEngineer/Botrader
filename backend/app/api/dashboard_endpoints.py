"""
API endpoints for performance metrics, strategy management, and expl ainability.
Extends the Phase 5 risk management endpoints.
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging
from sqlalchemy import text
from app.core.database import engine
from app.analytics.performance import TradeRecord, PerformanceMetrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["dashboard"])

# These would be injected via dependency injection in production
_performance_analyzer = None
_strategy_registry = None
_intervention_log = None
_current_explainer = None

async def _get_trades_from_db(limit: int = 10000) -> List[Dict]:
    """Get trades from database"""
    try:
        query = text("""
            SELECT time, symbol, side, quantity, price, pnl, capital_after, strategy
            FROM paper_trades
            ORDER BY time ASC
            LIMIT :limit
        """)
        async with engine.connect() as conn:
            result = await conn.execute(query, {"limit": limit})
            rows = result.fetchall()
            
        trades = []
        for row in rows:
            trades.append({
                "time": row.time,
                "symbol": row.symbol,
                "side": row.side,
                "quantity": float(row.quantity),
                "price": float(row.price),
                "pnl": float(row.pnl) if row.pnl is not None else None,
                "capital_after": float(row.capital_after) if row.capital_after else None,
                "strategy": row.strategy
            })
        return trades
    except Exception as e:
        logger.error(f"Error fetching trades from database: {e}")
        return []

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
        if _performance_analyzer is None:
            # Return empty metrics if analyzer not initialized
            return PerformanceMetricsResponse(
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                max_drawdown=0.0,
                current_drawdown=0.0,
                total_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                turnover=0.0,
                total_fees=0.0,
                fees_pct_of_pnl=0.0
            )
        
        # Get trades from database
        trades_data = await _get_trades_from_db(limit=10000)
        
        if not trades_data:
            # No trades yet
            return PerformanceMetricsResponse(
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                max_drawdown=0.0,
                current_drawdown=0.0,
                total_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                turnover=0.0,
                total_fees=0.0,
                fees_pct_of_pnl=0.0
            )
        
        # Build equity curve from capital_after values or calculate from trades
        equity_curve = []
        timestamps = []
        initial_balance = 10000.0  # Default initial capital
        
        # Check if we have capital_after values
        has_capital_after = any(t.get('capital_after') is not None for t in trades_data)
        
        if has_capital_after:
            # Use capital_after values when available
            for trade in trades_data:
                if trade.get('capital_after') is not None:
                    equity_curve.append(trade['capital_after'])
                    timestamps.append(trade['time'])
            # Calculate initial balance from first trade
            if trades_data and trades_data[0].get('capital_after') is not None:
                first_trade = trades_data[0]
                if first_trade['side'] == 'BUY':
                    # Capital before = capital_after + cost
                    initial_balance = first_trade['capital_after'] + (first_trade['quantity'] * first_trade['price'])
                elif first_trade['side'] == 'SELL':
                    # Capital before = capital_after - revenue
                    initial_balance = first_trade['capital_after'] - (first_trade['quantity'] * first_trade['price'])
        else:
            # Calculate from trades
            current_balance = initial_balance
            for trade in trades_data:
                if trade['side'] == 'BUY':
                    current_balance -= trade['quantity'] * trade['price']
                elif trade['side'] == 'SELL':
                    current_balance += trade['quantity'] * trade['price']
                equity_curve.append(current_balance)
                timestamps.append(trade['time'])
        
        # Convert trades to TradeRecord format
        trade_records = []
        for trade in trades_data:
            trade_records.append(TradeRecord(
                timestamp=trade['time'],
                symbol=trade['symbol'],
                side=trade['side'],
                quantity=trade['quantity'],
                price=trade['price'],
                fee=0.0,  # Paper trading has no fees
                pnl=trade.get('pnl')
            ))
        
        # Calculate metrics
        if equity_curve and timestamps:
            metrics = _performance_analyzer.calculate_metrics(
                equity_curve=equity_curve,
                trades=trade_records,
                timestamps=timestamps,
                initial_balance=initial_balance
            )
            
            return PerformanceMetricsResponse(
                total_return=metrics.total_return,
                annualized_return=metrics.annualized_return,
                sharpe_ratio=metrics.sharpe_ratio,
                sortino_ratio=metrics.sortino_ratio,
                calmar_ratio=metrics.calmar_ratio,
                max_drawdown=metrics.max_drawdown,
                current_drawdown=metrics.current_drawdown,
                total_trades=metrics.total_trades,
                win_rate=metrics.win_rate,
                avg_win=metrics.avg_win,
                avg_loss=metrics.avg_loss,
                profit_factor=metrics.profit_factor,
                turnover=metrics.turnover,
                total_fees=metrics.total_fees,
                fees_pct_of_pnl=metrics.fees_pct_of_pnl
            )
        else:
            # No data
            return PerformanceMetricsResponse(
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                max_drawdown=0.0,
                current_drawdown=0.0,
                total_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                turnover=0.0,
                total_fees=0.0,
                fees_pct_of_pnl=0.0
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
        # Return empty data until real trading history exists
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
        # Return empty until strategies have performance data
        return {
            "strategies": []
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
            # Return empty array if registry not initialized
            return {
                "strategies": []
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
        # Return empty until model is active
        return {
            "features": [],
            "importance": [],
            "method": None,
            "model_id": model_id or "none"
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
        # Return null until model is active
        return {
            "base_value": None,
            "prediction": None,
            "contributions": [],
            "method": None
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
        # Return empty until model is active
        return {
            "summary": [],
            "num_samples": 0,
            "method": None
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
