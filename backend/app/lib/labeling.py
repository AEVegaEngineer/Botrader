import pandas as pd
import numpy as np

def add_labels(df: pd.DataFrame, horizons: list = [1, 5, 15]) -> pd.DataFrame:
    """
    Adds target labels to the dataframe.
    """
    for h in horizons:
        # Next-k returns
        df[f'ret_{h}m'] = df['close'].shift(-h) / df['close'] - 1
        
        # Directional move (1 for up, 0 for down)
        df[f'dir_{h}m'] = (df[f'ret_{h}m'] > 0).astype(int)
        
    return df
