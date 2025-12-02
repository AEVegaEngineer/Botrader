import pandas as pd
from app.lib.indicators import compute_indicators

class FeatureRegistry:
    """
    Central registry for feature definitions.
    Ensures consistency between offline training (build_dataset.py) and online inference (collector.py).
    """
    
    @staticmethod
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all registered features for the given DataFrame.
        Expects df to have columns: ['open', 'high', 'low', 'close', 'volume']
        """
        # For now, we simply wrap the existing compute_indicators function.
        # In the future, we can add more complex feature engineering here.
        return compute_indicators(df)

    @staticmethod
    def get_feature_names():
        """
        Returns the list of feature columns that are expected to be generated.
        """
        # This list must match what compute_indicators returns
        return [
            'SMA_20', 'EMA_20', 'SMA_50', 'EMA_50',
            'RSI_14', 'MACD', 'MACD_SIGNAL', 'MACD_DIFF',
            'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'ATR_14'
        ]
