import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.core.database import engine, Base
from app.services.collector import BinanceCollector

# Import Routers
from app.api import endpoints as risk_endpoints
from app.api import dashboard_endpoints
from app.api import indicator_endpoints
from app.api import bot_control

# Import Dependencies
from app.risk.manager import RiskManager
from app.risk.portfolio import Portfolio
from app.core.strategy_registry import StrategyRegistry, StrategyMetadata, StrategyType
from app.core.intervention_log import get_intervention_log
from app.analytics.performance import PerformanceAnalyzer

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global Instances
collector = BinanceCollector()
risk_manager = RiskManager(initial_balance=10000.0)
portfolio = Portfolio(cash=10000.0)
strategy_registry = StrategyRegistry()
intervention_log = get_intervention_log()
performance_analyzer = PerformanceAnalyzer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up Botrader Backend...")
    
    # Initialize Database
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Initialize bot status file for paper trader
    import json
    import os
    bot_status_file = "bot_status.json"
    if not os.path.exists(bot_status_file):
        with open(bot_status_file, 'w') as f:
            json.dump({
                "is_running": False,
                "status": "Stopped",
                "started_at": None,
                "trades_count": 0,
                "active_strategy": "rsi_strategy"
            }, f)
        logger.info("Created bot_status.json file")
    
    # Start Collector
    asyncio.create_task(collector.start())
    
    # Register default strategies
    rsi_strategy = StrategyMetadata(
        id="rsi_strategy",
        name="RSI Strategy",
        type=StrategyType.RULE_BASED,
        version="1.0.0",
        description="Simple RSI Mean Reversion (Buy RSI<30, Sell RSI>70)",
        config={"rsi_period": 14, "rsi_lower": 30, "rsi_upper": 70}
    )
    ml_strategy = StrategyMetadata(
        id="ml_action_transformer",
        name="ML Action Transformer",
        type=StrategyType.TRANSFORMER,
        version="1.0.0",
        description="Transformer-based Action Classifier",
        config={"seq_len": 64}
    )
    strategy_registry.register(rsi_strategy)
    strategy_registry.register(ml_strategy)
    logger.info("Registered default strategies")
    
    # Inject Dependencies into Routers
    risk_endpoints.set_risk_manager(risk_manager)
    risk_endpoints.set_portfolio(portfolio)
    
    dashboard_endpoints.set_strategy_registry(strategy_registry)
    dashboard_endpoints.set_performance_analyzer(performance_analyzer)
    dashboard_endpoints.set_intervention_log(intervention_log)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Botrader Backend...")
    collector.stop()

app = FastAPI(lifespan=lifespan, title="Botrader API", version="1.0.0")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:3000"], # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(risk_endpoints.router)
app.include_router(dashboard_endpoints.router)
app.include_router(indicator_endpoints.router)
app.include_router(bot_control.router)

# Root Endpoints (Legacy/Frontend Compatibility)

@app.get("/")
async def root():
    return {"message": "Botrader Backend Running", "status": "active"}

@app.get("/status")
async def get_status():
    """System status"""
    return {
        "status": "running",
        "collector": "active" if collector else "inactive",
        "risk_manager": "active" if not risk_manager.is_halted else "halted",
        "timestamp": asyncio.get_event_loop().time()
    }

@app.get("/history")
async def get_history():
    """Mock history for frontend"""
    return {
        "trades": [],
        "performance": []
    }

@app.get("/price")
async def get_price():
    """Get current price (mock or from collector)"""
    price = collector.get_latest_price()
    return {"symbol": "BTCUSDT", "price": price if price > 0 else 50000.0}

@app.post("/start")
async def start_bot():
    """Start the bot"""
    logger.info("Bot start requested")
    return {"status": "started", "message": "Bot started successfully"}

@app.post("/stop")
async def stop_bot():
    """Stop the bot"""
    logger.info("Bot stop requested")
    return {"status": "stopped", "message": "Bot stopped successfully"}

@app.get("/performance")
async def get_performance_summary():
    """Get performance summary"""
    return {
        "total_pnl": 0.0,
        "win_rate": 0.0,
        "sharpe_ratio": 0.0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
