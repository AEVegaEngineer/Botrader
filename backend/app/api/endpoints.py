"""
FastAPI endpoints for risk management and trading control.
Provides emergency stop, risk monitoring, and trading control.
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["risk", "trading"])

# Global risk manager instance (should be injected via dependency in production)
# For now, we'll use a module-level variable that's set by the main app
_risk_manager = None
_portfolio = None

def get_risk_manager():
    """Dependency to get risk manager instance"""
    if _risk_manager is None:
        raise HTTPException(status_code=500, detail="Risk manager not initialized")
    return _risk_manager

def get_portfolio():
    """Dependency to get portfolio instance"""
    if _portfolio is None:
        raise HTTPException(status_code=500, detail="Portfolio not initialized")
    return _portfolio

def set_risk_manager(rm):
    """Set the global risk manager (called from main app)"""
    global _risk_manager
    _risk_manager = rm

def set_portfolio(p):
    """Set the global portfolio (called from main app)"""
    global _portfolio
    _portfolio = p

# Request/Response Models
class EmergencyStopRequest(BaseModel):
    close_positions: bool = True
    reason: Optional[str] = "Manual emergency stop"

class EmergencyStopResponse(BaseModel):
    success: bool
    message: str
    positions_closed: int
    total_pnl: float
    timestamp: str

class ResumeTradingRequest(BaseModel):
    confirmed: bool = False

class ResumeTradingResponse(BaseModel):
    success: bool
    message: str
    timestamp: str

class RiskStatusResponse(BaseModel):
    is_halted: bool
    circuit_breaker: dict
    metrics: dict
    limits: dict
    portfolio_summary: Optional[dict] = None

# Endpoints

@router.post("/emergency-stop", response_model=EmergencyStopResponse)
async def emergency_stop(
    request: EmergencyStopRequest,
    risk_manager = Depends(get_risk_manager),
    portfolio = Depends(get_portfolio)
):
    """
    Emergency stop - immediately halt trading and optionally close positions.
    
    This triggers the circuit breaker and prevents any new trades from being executed.
    """
    try:
        logger.critical(f"🚨 EMERGENCY STOP TRIGGERED: {request.reason}")
        
        # Trigger circuit breaker manually
        risk_manager.is_halted = True
        risk_manager.circuit_breaker.is_active = True
        risk_manager.circuit_breaker.triggered_at = datetime.now()
        risk_manager.circuit_breaker.reason = request.reason
        
        positions_closed = 0
        total_pnl = 0.0
        
        # Close all positions if requested
        if request.close_positions:
            total_pnl = portfolio.close_all_positions()
            positions_closed = len(portfolio.positions)
        
        return EmergencyStopResponse(
            success=True,
            message=f"Emergency stop activated. Trading halted. {positions_closed} positions closed.",
            positions_closed=positions_closed,
            total_pnl=total_pnl,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Error during emergency stop: {e}")
        raise HTTPException(status_code=500, detail=f"Emergency stop failed: {str(e)}")

@router.post("/resume-trading", response_model=ResumeTradingResponse)
async def resume_trading(
    request: ResumeTradingRequest,
    risk_manager = Depends(get_risk_manager)
):
    """
    Resume trading after emergency stop.
    
    Requires confirmation to prevent accidental resumption.
    """
    if not request.confirmed:
        raise HTTPException(
            status_code=400, 
            detail="Must confirm resumption by setting 'confirmed=true'"
        )
    
    try:
        logger.warning("🔓 Trading resumption requested")
        
        # Reset circuit breaker
        risk_manager.reset_circuit_breaker()
        
        return ResumeTradingResponse(
            success=True,
            message="Circuit breaker reset. Trading resumed.",
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Error resuming trading: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to resume trading: {str(e)}")

@router.get("/risk-status", response_model=RiskStatusResponse)
async def get_risk_status(
    risk_manager = Depends(get_risk_manager),
    portfolio = Depends(get_portfolio)
):
    """
    Get current risk status and metrics.
    
    Returns circuit breaker state, risk metrics, and portfolio summary.
    """
    try:
        risk_status = risk_manager.get_risk_status()
        portfolio_summary = portfolio.get_summary()
        
        return RiskStatusResponse(
            is_halted=risk_status["is_halted"],
            circuit_breaker=risk_status["circuit_breaker"],
            metrics=risk_status["metrics"],
            limits=risk_status["limits"],
            portfolio_summary=portfolio_summary
        )
    
    except Exception as e:
        logger.error(f"Error getting risk status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get risk status: {str(e)}")

@router.get("/portfolio", response_model=dict)
async def get_portfolio(
    portfolio = Depends(get_portfolio)
):
    """
    Get detailed portfolio information.
    
    Returns positions, weights, and risk metrics.
    """
    try:
        return portfolio.get_summary()
    
    except Exception as e:
        logger.error(f"Error getting portfolio: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get portfolio: {str(e)}")

@router.post("/positions/close/{symbol}")
async def close_position(
    symbol: str,
    risk_manager = Depends(get_risk_manager),
    portfolio = Depends(get_portfolio)
):
    """
    Close a specific position.
    
    Args:
        symbol: Symbol to close (e.g., BTCUSDT)
    """
    try:
        if symbol not in portfolio.positions:
            raise HTTPException(status_code=404, detail=f"No position found for {symbol}")
        
        pnl = portfolio.close_position(symbol)
        
        return {
            "success": True,
            "symbol": symbol,
            "realized_pnl": pnl,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error closing position {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to close position: {str(e)}")

@router.post("/positions/close-all")
async def close_all_positions(
    portfolio = Depends(get_portfolio)
):
    """
    Close all open positions.
    """
    try:
        total_pnl = portfolio.close_all_positions()
        
        return {
            "success": True,
            "total_pnl": total_pnl,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error closing all positions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to close positions: {str(e)}")
