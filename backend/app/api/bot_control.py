"""
Bot control and status endpoints.
Provides start/stop bot functionality and current status.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
import os
import json
from datetime import datetime
from sqlalchemy import text
from app.core.database import engine
from app.models.market_data import PaperTrade

logger = logging.getLogger(__name__)

router = APIRouter(tags=["bot_control"])

# Global bot state
_bot_status = {
    "is_running": False,
    "status": "Stopped",
    "started_at": None,
    "trades_count": 0,
    "active_strategy": "rsi_strategy"
}

STRATEGY_FILE = "active_strategy.txt"
BOT_STATUS_FILE = "bot_status.json"

def _save_bot_status():
    """Save bot status to file for paper trader to read"""
    try:
        with open(BOT_STATUS_FILE, 'w') as f:
            json.dump(_bot_status, f)
    except Exception as e:
        logger.error(f"Error saving bot status: {e}")

async def _get_trade_history_from_db(limit: int = 100) -> List[dict]:
    """Get trade history from database"""
    try:
        query = text("""
            SELECT time, symbol, side, quantity, price, pnl, capital_after, strategy
            FROM paper_trades
            ORDER BY time DESC
            LIMIT :limit
        """)
        async with engine.connect() as conn:
            result = await conn.execute(query, {"limit": limit})
            rows = result.fetchall()
            
        trades = []
        for row in rows:
            trades.append({
                "time": row.time.isoformat() if row.time else None,
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
        logger.error(f"Error fetching trade history from database: {e}")
        return []

class BotStatusResponse(BaseModel):
    is_running: bool
    status: str
    started_at: Optional[str] = None
    trades_count: int
    active_strategy: str

class TradeHistoryResponse(BaseModel):
    trades: List[dict]

class PriceResponse(BaseModel):
    price: float
    symbol: str
    timestamp: str

class PerformanceResponse(BaseModel):
    wins: int
    losses: int
    total_pnl: float
    win_rate: float
    equity_curve: List[dict]

class StrategyChangeRequest(BaseModel):
    strategy: str

@router.post("/start")
async def start_bot():
    """Start the trading bot"""
    global _bot_status
    try:
        logger.info("📈 Starting bot via API")
        _bot_status["is_running"] = True
        _bot_status["status"] = "Running"
        _bot_status["started_at"] = datetime.now().isoformat()
        _save_bot_status()
        
        return {
            "success": True,
            "message": "Bot started successfully",
            **_bot_status
        }
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop")
async def stop_bot():
    """Stop the trading bot"""
    global _bot_status
    try:
        logger.info("📉 Stopping bot via API")
        _bot_status["is_running"] = False
        _bot_status["status"] = "Stopped"
        _save_bot_status()
        
        return {
            "success": True,
            "message": "Bot stopped successfully",
            **_bot_status
        }
    except Exception as e:
        logger.error(f"Error stopping bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status", response_model=BotStatusResponse)
async def get_status():
    """Get current bot status"""
    # Check file for strategy
    if os.path.exists(STRATEGY_FILE):
        with open(STRATEGY_FILE, 'r') as f:
            _bot_status["active_strategy"] = f.read().strip()
    
    # Update trades count from database
    trades = await _get_trade_history_from_db(limit=1)
    _bot_status["trades_count"] = len(await _get_trade_history_from_db(limit=10000))
            
    return BotStatusResponse(**_bot_status)

@router.get("/strategies")
async def get_strategies():
    """Get available strategies"""
    return {
        "strategies": [
            {"id": "rsi_strategy", "name": "RSI Strategy", "description": "Simple RSI Mean Reversion"},
            {"id": "ml_action_transformer", "name": "ML Action Transformer", "description": "Transformer-based Action Classifier"}
        ],
        "active": _bot_status["active_strategy"]
    }

@router.post("/strategy")
async def set_strategy(req: StrategyChangeRequest):
    """Set active strategy"""
    global _bot_status
    if req.strategy not in ["rsi_strategy", "ml_action_transformer"]:
        raise HTTPException(status_code=400, detail="Invalid strategy")
        
    _bot_status["active_strategy"] = req.strategy
    
    # Write to file for persistence/sharing
    with open(STRATEGY_FILE, 'w') as f:
        f.write(req.strategy)
        
    return {"success": True, "active_strategy": req.strategy}

@router.get("/history", response_model=TradeHistoryResponse)
async def get_history():
    """Get trade history"""
    trades = await _get_trade_history_from_db(limit=100)
    return TradeHistoryResponse(trades=trades)

@router.get("/price", response_model=PriceResponse)
async def get_price():
    """Get current BTC price"""
    try:
        from app.core.database import engine
        from sqlalchemy import text
        
        # Get latest candle from database
        query = text("SELECT close FROM candles WHERE symbol = 'BTCUSDT' ORDER BY time DESC LIMIT 1")
        async with engine.connect() as conn:
            result = await conn.execute(query)
            row = result.fetchone()
        
        if row:
            return PriceResponse(
                price=float(row.close),
                symbol="BTCUSDT",
                timestamp=datetime.now().isoformat()
            )
        else:
            # Return a default if no data
            return PriceResponse(
                price=0.0,
                symbol="BTCUSDT", 
                timestamp=datetime.now().isoformat()
            )
    except Exception as e:
        logger.error(f"Error fetching price: {e}")
        return PriceResponse(
            price=0.0,
            symbol="BTCUSDT",
            timestamp=datetime.now().isoformat()
        )

@router.get("/performance", response_model=PerformanceResponse)
async def get_performance():
    """Get performance metrics"""
    trades = await _get_trade_history_from_db(limit=10000)
    # Filter trades with PnL (closed positions)
    closed_trades = [t for t in trades if t.get('pnl') is not None]
    wins = len([t for t in closed_trades if t.get('pnl', 0) > 0])
    losses = len([t for t in closed_trades if t.get('pnl', 0) < 0])
    total_pnl = sum(t.get('pnl', 0) for t in closed_trades)
    win_rate = wins / max(len(closed_trades), 1)
    
    # Build equity curve from capital_after values
    equity_curve = []
    for trade in trades:
        if trade.get('capital_after') is not None:
            equity_curve.append({
                "time": trade.get('time'),
                "equity": trade.get('capital_after')
            })
    
    return PerformanceResponse(
        wins=wins,
        losses=losses,
        total_pnl=total_pnl,
        win_rate=win_rate,
        equity_curve=equity_curve
    )

# Helper functions (kept for backward compatibility, but trades are now in database)
def add_trade(trade: dict):
    """Add a trade to history (deprecated - trades are now saved to database)"""
    logger.warning("add_trade() called but trades are now saved to database directly")
    pass

def get_bot_status():
    """Get bot status dict"""
    return _bot_status
