import unittest
from unittest.mock import patch
import datetime
import pandas as pd
import numpy as np
from ml_investing_wne.cumsum_filter import cumsum_filter
import pandas.testing as pd_testing


datetime_col = [datetime.datetime(2021, 1, 1, 12, 0, 0), datetime.datetime(2021, 1, 2, 0, 0, 0),
                    datetime.datetime(2021, 1, 2, 12, 0, 0), datetime.datetime(2021, 1, 3, 0, 0, 0),
                    datetime.datetime(2021, 1, 3, 12, 0, 0), datetime.datetime(2021, 1, 4, 0, 0, 0),
                    datetime.datetime(2021, 1, 4, 12, 0, 0), datetime.datetime(2021, 1, 5, 0, 0, 0),
                    datetime.datetime(2021, 1, 5, 12, 0, 0), datetime.datetime(2021, 1, 6, 0, 0, 0),
                    datetime.datetime(2021, 1, 6, 12, 0, 0), datetime.datetime(2021, 1, 7, 0, 0, 0),
                    datetime.datetime(2021, 1, 7, 12, 0, 0), datetime.datetime(2021, 1, 8, 0, 0, 0),
                    datetime.datetime(2021, 1, 8, 12, 0, 0)
                    ]

close = [100.0, 101.0, 103.0, 103.0, 102.0, 100, 106.0, 107.0, 108.0, 109.0, 111.0, 110, 111.0, 110.0, 112.0]
_open = [99] + close[:-1]
volume = [10.0, 20.0, 30.0, 40.0, 10.0, 20.0, 30.0, 40.0, 10.0, 20.0, 10, 10, 10, 10, 10]


test_df = pd.DataFrame({"datetime": datetime_col, 
                        "close": close,
                         "open": _open,
                         "volume": volume 
                         })
test_df['low'] = test_df[['open', 'close']].min(axis=1) * 0.99
test_df['low'] = np.floor(pd.to_numeric(test_df['low'] , errors='coerce')).astype('int64')
test_df['high'] = test_df[['open', 'close']].max(axis=1) * 1.01
test_df['high'] = np.ceil(pd.to_numeric(test_df['high'] , errors='coerce')).astype('int64')

class TestBinance(unittest.TestCase):

    def test_cumsum_filter(self):
        df_processed = cumsum_filter(test_df, 0.02)
        df_processed = df_processed.astype({'open':'int64','high':'int64','low':'int64','close':'int64', 'volume': 'int64'})
        df_expected = pd.DataFrame({"datetime": [datetime.datetime(2021, 1, 2, 12, 0, 0), datetime.datetime(2021, 1, 4, 0, 0, 0),
                                                datetime.datetime(2021, 1, 4, 12, 0, 0), datetime.datetime(2021, 1, 6, 0, 0, 0),
                                                datetime.datetime(2021, 1, 8, 12, 0, 0) ],
                         "open": [99, 103, 100, 106, 109],                         
                         "high": [105, 105, 108, 111, 114],
                         "low": [98, 99, 99, 104, 107],
                         "close": [103, 100,106, 109, 112],
                         "volume": [60,70,30, 70, 50],
                        "group_id": [0, 1, 2, 3, 4]
                         })
        df_expected.set_index('group_id', inplace=True)
        pd_testing.assert_frame_equal(df_processed, df_expected)


if __name__ == '__main__':
    unittest.main()