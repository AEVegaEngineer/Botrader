"""
VWAP (Volume-Weighted Average Price) calculation and execution scheduling.
"""
import logging
from typing import List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class VolumeBar:
    """Represents a volume bar (price + volume at a timestamp)"""
    timestamp: datetime
    price: float
    volume: float

@dataclass
class VWAPSlice:
    """Represents a slice in VWAP execution"""
    timestamp: datetime
    quantity: float
    target_price: float

class VWAPCalculator:
    """Calculate VWAP and create execution schedules"""
    
    def __init__(self, volume_bars: List[VolumeBar]):
        """
        Initialize with historical volume data.
        
        Args:
            volume_bars: List of historical volume bars
        """
        self.volume_bars = sorted(volume_bars, key=lambda x: x.timestamp)
    
    def calculate_vwap(self) -> float:
        """
        Calculate VWAP from historical volume bars.
        
        Returns:
            VWAP price
        """
        if not self.volume_bars:
            logger.warning("No volume bars provided for VWAP calculation")
            return 0.0
        
        total_pv = sum(bar.price * bar.volume for bar in self.volume_bars)
        total_volume = sum(bar.volume for bar in self.volume_bars)
        
        if total_volume == 0:
            logger.warning("Total volume is zero, returning average price")
            return sum(bar.price for bar in self.volume_bars) / len(self.volume_bars)
        
        vwap = total_pv / total_volume
        logger.info(f"Calculated VWAP: {vwap:.2f} from {len(self.volume_bars)} bars")
        
        return vwap
    
    def create_volume_profile(self) -> List[float]:
        """
        Create normalized volume profile (distribution of volume over time).
        
        Returns:
            List of volume percentages (sums to 1.0)
        """
        if not self.volume_bars:
            return []
        
        volumes = np.array([bar.volume for bar in self.volume_bars])
        total_volume = np.sum(volumes)
        
        if total_volume == 0:
            # Equal distribution if no volume data
            return [1.0 / len(self.volume_bars)] * len(self.volume_bars)
        
        volume_profile = volumes / total_volume
        
        return volume_profile.tolist()
    
    def create_execution_schedule(
        self, 
        total_quantity: float, 
        start_time: datetime,
        num_slices: int = None
    ) -> List[VWAPSlice]:
        """
        Create VWAP-based execution schedule.
        Distributes order quantity according to historical volume profile.
        
        Args:
            total_quantity: Total quantity to execute
            start_time: When to start execution
            num_slices: Number of slices (default: same as volume bars)
            
        Returns:
            List of VWAPSlice objects with timing and size
        """
        if not self.volume_bars:
            logger.error("No volume bars available for scheduling")
            return []
        
        volume_profile = self.create_volume_profile()
        
        # Use num_slices or default to number of bars
        if num_slices is None:
            num_slices = len(volume_profile)
        
        # If num_slices < len(volume_profile), aggregate
        if num_slices < len(volume_profile):
            # Group volume profile into num_slices buckets
            bucket_size = len(volume_profile) / num_slices
            aggregated_profile = []
            
            for i in range(num_slices):
                start_idx = int(i * bucket_size)
                end_idx = int((i + 1) * bucket_size)
                bucket_volume = sum(volume_profile[start_idx:end_idx])
                aggregated_profile.append(bucket_volume)
            
            # Normalize to sum to 1
            total = sum(aggregated_profile)
            volume_profile = [v / total for v in aggregated_profile]
        
        # Calculate time interval between slices
        if len(self.volume_bars) > 1:
            total_duration = self.volume_bars[-1].timestamp - self.volume_bars[0].timestamp
            slice_interval = total_duration / num_slices
        else:
            slice_interval = timedelta(minutes=1)  # Default 1 minute
        
        # Create slices
        slices = []
        for i in range(num_slices):
            slice_time = start_time + (i * slice_interval)
            slice_quantity = total_quantity * volume_profile[min(i, len(volume_profile) - 1)]
            
            # Use corresponding bar price or average
            bar_idx = min(i, len(self.volume_bars) - 1)
            target_price = self.volume_bars[bar_idx].price
            
            slices.append(VWAPSlice(
                timestamp=slice_time,
                quantity=slice_quantity,
                target_price=target_price
            ))
        
        logger.info(f"Created VWAP schedule with {len(slices)} slices over {total_duration if len(self.volume_bars) > 1 else 'default interval'}")
        
        return slices
    
    def get_volume_weighted_price_range(self, percentile: float = 0.8) -> Tuple[float, float]:
        """
        Get price range that contains X% of volume.
        
        Args:
            percentile: Percentage of volume to include (e.g., 0.8 for 80%)
            
        Returns:
            Tuple of (low_price, high_price)
        """
        if not self.volume_bars:
            return (0.0, 0.0)
        
        # Sort by price
        sorted_bars = sorted(self.volume_bars, key=lambda x: x.price)
        
        total_volume = sum(bar.volume for bar in sorted_bars)
        target_volume = total_volume * percentile
        
        cumulative_volume = 0
        low_price = sorted_bars[0].price
        high_price = sorted_bars[-1].price
        
        # Find price range containing target volume
        for i, bar in enumerate(sorted_bars):
            cumulative_volume += bar.volume
            if cumulative_volume >= target_volume:
                high_price = bar.price
                break
        
        logger.info(f"{percentile:.0%} of volume is between ${low_price:.2f} and ${high_price:.2f}")
        
        return (low_price, high_price)
