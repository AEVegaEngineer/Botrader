from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

DATABASE_URL = settings.DATABASE_URL

engine = create_async_engine(DATABASE_URL, echo=True)

AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
        # Convert to hypertables
        # We use execute directly on the connection for raw SQL
        await conn.execute(text("SELECT create_hypertable('candles', 'time', if_not_exists => TRUE, migrate_data => TRUE);"))
        await conn.execute(text("SELECT create_hypertable('trades', 'time', if_not_exists => TRUE, migrate_data => TRUE);"))
        await conn.execute(text("SELECT create_hypertable('lob_snapshots', 'time', if_not_exists => TRUE, migrate_data => TRUE);"))
        await conn.execute(text("SELECT create_hypertable('candle_indicators', 'time', if_not_exists => TRUE, migrate_data => TRUE);"))
