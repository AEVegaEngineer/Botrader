import pandas as pd
import numpy as np
from enum import IntEnum

class Action(IntEnum):
    HOLD = 0
    SELL = 1
    BUY = 2

def compute_labels(
    df: pd.DataFrame, 
    horizon_candles: int = 15, 
    th_up: float = 0.001, 
    th_down: float = -0.001,
    transaction_cost: float = 0.0
) -> pd.DataFrame:
    """
    Computes action labels based on future returns.
    
    Args:
        df: DataFrame with 'close' column.
        horizon_candles: Number of candles to look ahead.
        th_up: Positive threshold for BUY (e.g. 0.001 for 0.1%).
        th_down: Negative threshold for SELL (e.g. -0.001 for -0.1%).
        transaction_cost: Cost to subtract from return (approximate).
        
    Returns:
        DataFrame with 'label' column and 'future_ret' column.
    """
    df = df.copy()
    
    # Compute future return: (price_{t+H} - price_t) / price_t
    df['future_ret'] = df['close'].shift(-horizon_candles) / df['close'] - 1.0
    
    # Adjust for transaction cost if needed (simple approximation)
    # If we buy, we pay cost. If we sell, we pay cost. 
    # Here we just want to see if the move is significant enough.
    # We can subtract cost from the absolute return or just adjust thresholds.
    # User said: "Optionally subtract a fixed transaction cost / spread to be realistic."
    # Let's subtract it from the return for long, and add it for short? 
    # Or just subtract from the magnitude?
    # Let's do: effective_return = future_ret - transaction_cost (for long logic)
    # For short logic, we want future_ret to be very negative.
    
    # Let's keep it simple:
    # BUY if future_ret > th_up + cost
    # SELL if future_ret < th_down - cost
    
    # Or as user said: "r_future = ...; Optionally subtract...; If r_future >= th_up => BUY"
    # I will implement: r_net = r_future
    # If transaction_cost > 0:
    #   For BUY check: r_future must be > th_up + transaction_cost
    #   For SELL check: r_future must be < th_down - transaction_cost
    
    # Actually, let's just use the thresholds passed in. The caller can incorporate cost into thresholds 
    # or we can do it here. The user prompt implies the thresholds are separate.
    
    # Let's define the logic strictly as requested:
    # r_future = ...
    # if r_future >= th_up: BUY
    # elif r_future <= th_down: SELL
    # else: HOLD
    
    # If transaction_cost is provided, we can penalize the return.
    # But usually cost depends on the direction.
    # Let's assume the user wants us to simply use the thresholds for now, 
    # and maybe `transaction_cost` is just a parameter to adjust `th_up`/`th_down` if we wanted to auto-calc them.
    # I will stick to the explicit thresholds for clarity, but allow `transaction_cost` to shift them if non-zero.
    
    effective_th_up = th_up + transaction_cost
    effective_th_down = th_down - transaction_cost
    
    conditions = [
        (df['future_ret'] >= effective_th_up),
        (df['future_ret'] <= effective_th_down)
    ]
    choices = [Action.BUY.value, Action.SELL.value]
    
    df['label'] = np.select(conditions, choices, default=Action.HOLD.value)
    
    # Remove rows where label cannot be computed (end of series)
    df = df.dropna(subset=['future_ret'])
    
    return df

def get_label_distribution(df: pd.DataFrame) -> dict:
    """Returns the distribution of labels."""
    if 'label' not in df.columns:
        return {}
    counts = df['label'].value_counts(normalize=True).to_dict()
    # Map back to names
    return {Action(k).name: v for k, v in counts.items()}
