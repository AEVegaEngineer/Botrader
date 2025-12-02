import pandas as pd
import ta

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators for a given DataFrame.
    Input DataFrame must have 'open', 'high', 'low', 'close', 'volume' columns.
    Returns a new DataFrame with added indicator columns.
    """
    # Ensure we work on a copy to avoid side effects
    df = df.copy()
    
    # Trend
    df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['EMA_20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['EMA_50'] = ta.trend.ema_indicator(df['close'], window=50)
    
    # Momentum
    df['RSI_14'] = ta.momentum.rsi(df['close'], window=14)
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_SIGNAL'] = macd.macd_signal()
    df['MACD_DIFF'] = macd.macd_diff()
    
    # Volatility
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['BBL_20_2.0'] = bb.bollinger_lband()
    df['BBM_20_2.0'] = bb.bollinger_mavg()
    df['BBU_20_2.0'] = bb.bollinger_hband()
        
    df['ATR_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    
    return df

def add_labels(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Add target labels for supervised learning.
    horizon: number of periods to look ahead.
    """
    df = df.copy()
    
    # Next-k returns
    df[f'ret_{horizon}m'] = df['close'].shift(-horizon) / df['close'] - 1
    
    # Direction (1 for up, 0 for down/flat)
    df[f'dir_{horizon}m'] = (df[f'ret_{horizon}m'] > 0).astype(int)
    
    return df
