import logging
import asyncio
from typing import Optional, List
from datetime import datetime, timedelta
from app.core.config import settings
from app.execution.vwap import VWAPCalculator, VolumeBar

logger = logging.getLogger(__name__)

class ExecutionMetrics:
    """Track execution quality metrics"""
    def __init__(self):
        self.orders = []
    
    def add_execution(self, benchmark_price: float, execution_price: float, quantity: float):
        """Record an execution"""
        slippage = execution_price - benchmark_price
        slippage_bps = (slippage / benchmark_price) * 10000 if benchmark_price > 0 else 0
        
        self.orders.append({
            'timestamp': datetime.now(),
            'benchmark_price': benchmark_price,
            'execution_price': execution_price,
            'quantity': quantity,
            'slippage': slippage,
            'slippage_bps': slippage_bps
        })
    
    def get_summary(self) -> dict:
        """Get execution metrics summary"""
        if not self.orders:
            return {
                'total_orders': 0,
                'avg_slippage_bps': 0,
                'total_slippage': 0
            }
        
        total_slippage = sum(o['slippage'] * o['quantity'] for o in self.orders)
        avg_slippage_bps = sum(o['slippage_bps'] for o in self.orders) / len(self.orders)
        
        return {
            'total_orders': len(self.orders),
            'avg_slippage_bps': avg_slippage_bps,
            'total_slippage': total_slippage,
            'orders': self.orders
        }

class ExecutionEngine:
    def __init__(self, client=None):
        self.client = client  # Binance client or similar
        self.metrics = ExecutionMetrics()

    async def execute_order(
        self, 
        symbol: str, 
        side: str, 
        quantity: float, 
        price: Optional[float] = None, 
        style: str = "AGGRESSIVE",
        volume_bars: Optional[List[VolumeBar]] = None
    ) -> dict:
        """
        Executes an order with the specified style.
        
        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Order quantity
            price: Reference price (used as benchmark)
            style: Execution style - AGGRESSIVE, PASSIVE, TWAP, VWAP
            volume_bars: Historical volume data for VWAP (required if style=VWAP)
            
        Returns:
            Execution result dict
        """
        logger.info(f"Executing {side} {quantity} {symbol} @ {price} ({style})")
        
        # Select execution strategy
        if style == "AGGRESSIVE":
            return await self._execute_aggressive(symbol, side, quantity, price)
        
        elif style == "PASSIVE":
            return await self._execute_passive(symbol, side, quantity, price)
        
        elif style == "TWAP":
            return await self._execute_twap(symbol, side, quantity, price)
        
        elif style == "VWAP":
            if not volume_bars:
                logger.warning("VWAP requested but no volume bars provided, falling back to TWAP")
                return await self._execute_twap(symbol, side, quantity, price)
            return await self._execute_vwap(symbol, side, quantity, price, volume_bars)
        
        else:
            logger.warning(f"Unknown style {style}, using AGGRESSIVE")
            return await self._execute_aggressive(symbol, side, quantity, price)

    async def _execute_aggressive(self, symbol: str, side: str, quantity: float, price: float) -> dict:
        """
        Aggressive execution - market order or aggressive limit.
        Prioritizes fill speed over price.
        """
        # In real trading: client.order_market(symbol, side, quantity)
        # For paper/sim: just return filled at current price
        
        # Simulate small slippage for aggressive execution
        slippage_bps = 5  # 5 basis points
        execution_price = price * (1 + slippage_bps / 10000) if side == "BUY" else price * (1 - slippage_bps / 10000)
        
        self.metrics.add_execution(price, execution_price, quantity)
        
        logger.info(f"AGGRESSIVE: Filled {quantity} {symbol} @ {execution_price:.2f} (benchmark: {price:.2f})")
        
        return {
            "status": "FILLED",
            "avg_price": execution_price,
            "quantity": quantity,
            "slippage_bps": slippage_bps,
            "execution_time_ms": 50  # Fast execution
        }

    async def _execute_passive(self, symbol: str, side: str, quantity: float, price: float) -> dict:
        """
        Passive execution - limit order at best bid/ask.
        Prioritizes price over fill speed.
        """
        # In real trading: place limit order at bid (sell) or ask (buy)
        # Potentially wait for fill with timeout
        
        # Simulate better pricing but with some uncertainty
        slippage_bps = -2  # Negative slippage = price improvement
        execution_price = price * (1 + slippage_bps / 10000) if side == "BUY" else price * (1 - slippage_bps / 10000)
        
        # Simulate execution delay
        await asyncio.sleep(0.5)
        
        self.metrics.add_execution(price, execution_price, quantity)
        
        logger.info(f"PASSIVE: Filled {quantity} {symbol} @ {execution_price:.2f} (benchmark: {price:.2f})")
        
        return {
            "status": "FILLED",
            "avg_price": execution_price,
            "quantity": quantity,
            "slippage_bps": slippage_bps,
            "execution_time_ms": 500  # Slower execution
        }

    async def _execute_twap(self, symbol: str, side: str, quantity: float, price: float) -> dict:
        """
        Time-Weighted Average Price execution.
        Splits order into equal slices over time.
        """
        num_slices = settings.TWAP_SLICES
        interval_sec = settings.TWAP_INTERVAL_SEC
        
        chunk_size = quantity / num_slices
        total_filled = 0
        weighted_price = 0
        
        logger.info(f"TWAP: Splitting {quantity} into {num_slices} slices of {chunk_size:.6f}")
        
        for i in range(num_slices):
            # Simulate price movement
            price_drift = (i - num_slices / 2) * 0.0001  # Small random walk
            slice_price = price * (1 + price_drift)
            
            # Execute slice
            logger.info(f"TWAP Slice {i+1}/{num_slices}: {chunk_size:.6f} @ {slice_price:.2f}")
            
            total_filled += chunk_size
            weighted_price += slice_price * chunk_size
            
            # Wait before next slice (except last one)
            if i < num_slices - 1:
                await asyncio.sleep(interval_sec)
        
        avg_price = weighted_price / total_filled if total_filled > 0 else price
        slippage_bps = ((avg_price - price) / price) * 10000 if price > 0 else 0
        
        self.metrics.add_execution(price, avg_price, total_filled)
        
        logger.info(f"TWAP Complete: Filled {total_filled:.6f} @ avg {avg_price:.2f} (benchmark: {price:.2f})")
        
        return {
            "status": "FILLED",
            "avg_price": avg_price,
            "quantity": total_filled,
            "slippage_bps": slippage_bps,
            "execution_time_ms": num_slices * interval_sec * 1000
        }

    async def _execute_vwap(
        self, 
        symbol: str, 
        side: str, 
        quantity: float, 
        price: float,
        volume_bars: List[VolumeBar]
    ) -> dict:
        """
        Volume-Weighted Average Price execution.
        Distributes order according to historical volume profile.
        """
        # Calculate VWAP and create execution schedule
        vwap_calc = VWAPCalculator(volume_bars)
        vwap_benchmark = vwap_calc.calculate_vwap()
        
        # Create execution schedule
        schedule = vwap_calc.create_execution_schedule(
            total_quantity=quantity,
            start_time=datetime.now(),
            num_slices=settings.TWAP_SLICES  # Reuse TWAP slices config
        )
        
        total_filled = 0
        weighted_price = 0
        
        logger.info(f"VWAP: Executing {quantity} across {len(schedule)} slices, benchmark: {vwap_benchmark:.2f}")
        
        for i, slice_info in enumerate(schedule):
            # Execute this slice
            slice_quantity = slice_info.quantity
            slice_target = slice_info.target_price
            
            # Simulate execution around target price
            execution_price = slice_target * (1 + (0.0001 if side == "BUY" else -0.0001))
            
            logger.info(f"VWAP Slice {i+1}/{len(schedule)}: {slice_quantity:.6f} @ {execution_price:.2f}")
            
            total_filled += slice_quantity
            weighted_price += execution_price * slice_quantity
            
            # Wait before next slice (except last one)
            if i < len(schedule) - 1:
                # Calculate time to next slice
                if i + 1 < len(schedule):
                    wait_time = (schedule[i+1].timestamp - schedule[i].timestamp).total_seconds()
                    await asyncio.sleep(min(wait_time, settings.TWAP_INTERVAL_SEC))
        
        avg_price = weighted_price / total_filled if total_filled > 0 else price
        slippage_bps = ((avg_price - vwap_benchmark) / vwap_benchmark) * 10000 if vwap_benchmark > 0 else 0
        
        self.metrics.add_execution(vwap_benchmark, avg_price, total_filled)
        
        logger.info(f"VWAP Complete: Filled {total_filled:.6f} @ avg {avg_price:.2f} (VWAP: {vwap_benchmark:.2f})")
        
        return {
            "status": "FILLED",
            "avg_price": avg_price,
            "quantity": total_filled,
            "vwap_benchmark": vwap_benchmark,
            "slippage_bps": slippage_bps,
            "execution_time_ms": len(schedule) * settings.TWAP_INTERVAL_SEC * 1000
        }

    async def select_execution_style(
        self, 
        symbol: str, 
        quantity: float, 
        spread: float, 
        volume: float
    ) -> str:
        """
        Automatically select execution style based on market conditions.
        
        Args:
            symbol: Trading symbol
            quantity: Order size
            spread: Current bid-ask spread
            volume: Recent trading volume
            
        Returns:
            Recommended execution style
        """
        # Calculate relative spread
        # Note: Would need current price, assuming it's implicit in spread
        
        # Simple heuristic:
        # - Tight spread + high volume = PASSIVE
        # - Wide spread or low volume = AGGRESSIVE or TWAP/VWAP
        
        if spread < settings.LIQUIDITY_THRESHOLD:
            # Tight spread, good liquidity
            logger.info(f"Tight spread ({spread:.5f}), using PASSIVE execution")
            return "PASSIVE"
        else:
            # Wider spread, use time-sliced execution
            logger.info(f"Wide spread ({spread:.5f}), using TWAP execution")
            return "TWAP"
    
    def get_execution_metrics(self) -> dict:
        """Get execution quality metrics"""
        return self.metrics.get_summary()
