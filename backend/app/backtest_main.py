import asyncio
import logging
import argparse
from app.backtest.engine import BacktestEngine
from app.backtest.strategies.simple_rsi import SimpleRSI
from app.backtest.strategies.rsi_sma import RsiSmaStrategy
from app.backtest.strategies.ml_action_transformer import MLActionTransformerStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    parser = argparse.ArgumentParser(description='Run Backtest')
    parser.add_argument('--strategy', type=str, default='simple_rsi', help='Strategy to run')
    args = parser.parse_args()

    engine = BacktestEngine(symbol="BTCUSDT")
    
    # Load Data
    success = await engine.load_data()
    if not success:
        return

    # Add Strategy
    if args.strategy == 'simple_rsi':
        engine.add_strategy(SimpleRSI)
    elif args.strategy == 'rsi_sma':
        engine.add_strategy(RsiSmaStrategy)
    elif args.strategy == 'ml_action_transformer':
        engine.add_strategy(MLActionTransformerStrategy)
    else:
        logger.error(f"Unknown strategy: {args.strategy}")
        return

    # Add Analyzers
    engine.add_analyzers()

    # Run
    engine.run()

if __name__ == "__main__":
    asyncio.run(main())
