import asyncio
import logging
from app.core.database import init_db
from app.services.collector import BinanceCollector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    logger.info("Starting Collector Service...")
    
    # Initialize Database (Create Tables & Hypertables)
    try:
        await init_db()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return

    # Start Collector
    collector = BinanceCollector()
    try:
        await collector.start()
    except KeyboardInterrupt:
        logger.info("Collector stopped by user.")
    except Exception as e:
        logger.error(f"Collector crashed: {e}")
    finally:
        collector.stop()

if __name__ == "__main__":
    asyncio.run(main())
