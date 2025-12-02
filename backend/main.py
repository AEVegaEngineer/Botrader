import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.database import engine, Base
from app.services.collector import BinanceCollector

collector = BinanceCollector()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    asyncio.create_task(collector.start())
    
    yield
    
    # Shutdown
    print("Shutting down...")
    collector.stop()

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Botrader Backend Running"}
