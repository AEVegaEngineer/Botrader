import unittest
import pandas as pd
import numpy as np
from app.lib.indicators import compute_indicators, add_labels

class TestIndicators(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1min')
        self.df = pd.DataFrame({
            'open': np.random.rand(100) * 100,
            'high': np.random.rand(100) * 100,
            'low': np.random.rand(100) * 100,
            'close': np.random.rand(100) * 100,
            'volume': np.random.rand(100) * 1000
        }, index=dates)

    def test_compute_indicators(self):
        df_res = compute_indicators(self.df)
        
        # Check if columns are added
        self.assertIn('SMA_20', df_res.columns)
        self.assertIn('RSI_14', df_res.columns)
        self.assertIn('ATR_14', df_res.columns)
        
        # Check for NaNs in the beginning (should be present due to window)
        self.assertTrue(pd.isna(df_res['SMA_20'].iloc[0]))
        
        # Check for non-NaNs later
        self.assertFalse(pd.isna(df_res['SMA_20'].iloc[50]))

    def test_add_labels(self):
        df_res = add_labels(self.df, horizon=5)
        
        self.assertIn('ret_5m', df_res.columns)
        self.assertIn('dir_5m', df_res.columns)
        
        # Check logic
        # ret_5m at t should be close[t+5] / close[t] - 1
        expected_ret = self.df['close'].iloc[5] / self.df['close'].iloc[0] - 1
        self.assertAlmostEqual(df_res['ret_5m'].iloc[0], expected_ret)

if __name__ == '__main__':
    unittest.main()
