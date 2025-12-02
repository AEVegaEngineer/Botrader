import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Botrader"
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "password")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "botrader")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "timescaledb")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Risk Management
    MAX_DRAWDOWN_PCT: float = 0.10 # 10%
    MAX_DAILY_LOSS_PCT: float = 0.05 # 5%
    MAX_LEVERAGE: float = 1.0
    RISK_PER_TRADE_PCT: float = 0.01 # 1%
    MAX_TRADE_NOTIONAL: float = 10000.0 # Max $ per trade
    
    # Volatility-Targeted Position Sizing
    TARGET_DAILY_VOL: float = 0.02 # Target 2% daily volatility
    VOL_LOOKBACK_DAYS: int = 30 # Rolling window for volatility calculation
    MIN_POSITION_SIZE: float = 0.0001 # Minimum BTC position size
    
    # Circuit Breaker Thresholds
    MAX_LATENCY_MS: float = 5000.0 # 5 seconds max latency
    MAX_ERROR_RATE: float = 0.10 # 10% error rate threshold
    ERROR_RATE_WINDOW: int = 100 # Last N operations to track
    
    # Execution Settings
    TWAP_SLICES: int = 5 # Number of slices for TWAP
    TWAP_INTERVAL_SEC: int = 10 # Seconds between TWAP slices
    VWAP_LOOKBACK_HOURS: int = 24 # Hours of volume data for VWAP
    LIQUIDITY_THRESHOLD: float = 0.001 # Min spread for passive execution

settings = Settings()
