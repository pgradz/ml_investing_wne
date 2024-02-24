import datetime
import unittest
from unittest import mock
from unittest.mock import patch

import numpy as np
import pandas as pd
import pandas.testing as pd_testing

from ml_investing_wne.data_engineering.crypto_factory import CryptoFactory


class TestCryptoFactory(unittest.TestCase):

    def setUp(self) -> None:
        # first two records will be discarded when running crypto_factory.get_daily_volatility. Last three rows will be discarded because of t_final=3
        self.datetime_col = [
                            datetime.datetime(2020, 12, 31 , 0, 0, 0), datetime.datetime(2020, 12, 31, 12, 0, 0),
                            datetime.datetime(2021, 1, 1, 12, 0, 0), datetime.datetime(2021, 1, 2, 0, 0, 0),
                            datetime.datetime(2021, 1, 2, 12, 0, 0), datetime.datetime(2021, 1, 3, 0, 0, 0),
                            datetime.datetime(2021, 1, 3, 12, 0, 0), datetime.datetime(2021, 1, 4, 0, 0, 0),
                            datetime.datetime(2021, 1, 4, 12, 0, 0), datetime.datetime(2021, 1, 5, 0, 0, 0),
                            datetime.datetime(2021, 1, 5, 12, 0, 0), datetime.datetime(2021, 1, 6, 0, 0, 0),
                            datetime.datetime(2021, 1, 6, 12, 0, 0), datetime.datetime(2021, 1, 7, 0, 0, 0),
                            datetime.datetime(2021, 1, 7, 12, 0, 0)
                            ]
        self._open = [100.0, 100.0, 100.0 ,100.0, 100.5, 102.0, 101.0, 99.5, 104.0, 104.0, 104.5, 104.5,109, 109, 109 ]
        self.close = [100.0, 100.0, 100.0, 100.5, 102, 101, 99.5, 104, 104, 104.5, 104.5, 109, 109, 109, 109]
        self.high = [100.0, 100.0, 100, 101.0, 102.5, 102, 101, 104, 104.5, 104.5, 104.5, 110, 109, 109, 109]
        self.low = [100.0, 100.0, 100, 100, 100.5, 101, 98, 99.5, 103.5, 104, 104, 104.5, 109, 109, 109]
        self.volume = [1000, 2000, 1500, 1800, 1200, 2500, 3000, 2200, 1800, 1900, 2100, 1300, 100, 2600, 2700]
        self.y_pred = [1, 0, 0, 0, 1, 1, 1, 1, 1]
        self.test_df = pd.DataFrame({"datetime": self.datetime_col, "open": self._open, 'close': self.close, 'high': self.high, 'low': self.low, 'volume': self.volume})
        self.test_df.set_index('datetime', inplace=True)
        self.mock_args = mock.MagicMock()
        self.mock_args.exchange = 'Binance'
        self.mock_args.currency = 'BTCUSDT'
        
                
    @patch("ml_investing_wne.data_engineering.crypto_factory.CryptoFactory.load_data")
    def test_run_3_barriers(self, m):

        m.return_value=self.test_df
        result_df_3_barriers_additional_info = pd.DataFrame({
                                                            "datetime": self.datetime_col[2:11],
                                                            "prc_change": [0.02, -0.02, -0.02, -0.02, 0.02, 0.5/104.0 ,0.02, 0.02, 0.02],
                                                            "barrier_touched": ['top','bottom','bottom','bottom','top','vertical','top','top','top'] ,
                                                            "barrier_touched_date": [datetime.datetime(2021, 1, 2, 12, 0, 0), datetime.datetime(2021, 1, 3, 12, 0, 0),
                                                                                    datetime.datetime(2021, 1, 3, 12, 0, 0), datetime.datetime(2021, 1, 3, 12, 0, 0),
                                                                                    datetime.datetime(2021, 1, 4, 0, 0, 0), datetime.datetime(2021, 1, 5, 12, 0, 0), 
                                                                                    datetime.datetime(2021, 1, 6, 0, 0, 0), datetime.datetime(2021, 1, 6, 0, 0, 0),
                                                                                    datetime.datetime(2021, 1, 6, 0, 0, 0)],                                                
                                                            "top_barrier": [1.02 * value for value in self.close[2:11]],
                                                            "bottom_barrier": [0.98 * value for value in self.close[2:11]],
                                                            "time_step":[2, 3, 2, 1, 1, 3, 3, 2, 1]
                                                            })
        

        crypto_factory = CryptoFactory(self.mock_args)
        crypto_factory.df_time_aggregated = crypto_factory.df
        crypto_factory.run_3_barriers(t_final=3,upper_lower_multipliers = [1, 1], fixed_barrier = 0.02)
        pd_testing.assert_frame_equal(crypto_factory.df_3_barriers_additional_info, result_df_3_barriers_additional_info)



    @patch("ml_investing_wne.data_engineering.crypto_factory.CryptoFactory.load_data")
    def test_generate_dollarbars(self, m):
        # Create a test DataFrame
        datetime_col = [
            datetime.datetime(2020, 12, 31 , 0, 0, 0), datetime.datetime(2020, 12, 31, 12, 0, 0),
            datetime.datetime(2021, 1, 1, 12, 0, 0), datetime.datetime(2021, 1, 2, 0, 0, 0),
            datetime.datetime(2021, 1, 2, 12, 0, 0), datetime.datetime(2021, 1, 3, 0, 0, 0),
            datetime.datetime(2021, 1, 3, 12, 0, 0), datetime.datetime(2021, 1, 4, 0, 0, 0),
            datetime.datetime(2021, 1, 4, 12, 0, 0), datetime.datetime(2021, 1, 5, 0, 0, 0),
            datetime.datetime(2021, 1, 5, 12, 0, 0), datetime.datetime(2021, 1, 6, 0, 0, 0),
            datetime.datetime(2021, 1, 6, 12, 0, 0), datetime.datetime(2021, 1, 7, 0, 0, 0),
            datetime.datetime(2021, 1, 7, 12, 0, 0)
        ]
        prices = [100.0, 100.0, 100.0 ,100.0, 100.5, 102.0, 101.0, 99.5, 104.0, 104.0, 104.5, 104.5,109, 109, 109]
        volumes = [1000, 2000, 1500, 1800, 1200, 2500, 3000, 2200, 1800, 1900, 2100, 1300, 100, 2600, 2700]
        test_df = pd.DataFrame({"datetime": datetime_col, "price": prices, "volume": volumes})

        # Set the expected result
        expected_result = pd.DataFrame({
            "datetime": [
                datetime.datetime(2020, 12, 31, 12, 0, 0),
                datetime.datetime(2021, 1, 2, 0, 0, 0),
                datetime.datetime(2021, 1, 3, 0, 0, 0),
                datetime.datetime(2021, 1, 3, 12, 0, 0),
                datetime.datetime(2021, 1, 4, 0, 0, 0),
                datetime.datetime(2021, 1, 5, 0, 0, 0),
                datetime.datetime(2021, 1, 5, 12, 0, 0),
                datetime.datetime(2021, 1, 7, 0, 0, 0),
                datetime.datetime(2021, 1, 7, 12, 0, 0)
            ],
            "open": [100, 100, 100.5, 101, 99.5, 104, 104.5, 104.5,109],
            "high": [ 100, 100, 102, 101, 99.5, 104, 104.5, 109, 109],
            "low": [100, 100, 100.5, 101, 99.5, 104,104.5,104.5,109],
            "close": [100,100,102,101,99.5,104,104.5,109,109],
            "volume": [3000.0,3300,3700,3000,2200,3700,2100,4000,2700]
        })
        expected_result.set_index('datetime', inplace=True)
        

        # Mock the load_data method
        m.return_value = test_df

        # Create an instance of CryptoFactory
        crypto_factory = CryptoFactory(self.mock_args)

        # Call the generate_dollarbars method
        crypto_factory.generate_dollarbars(frequency=200000)
        # change crypto_factory.df_volume_bars dtypes to float
        crypto_factory.df_volume_bars = crypto_factory.df_volume_bars.astype(float)

        # Assert that the generated DataFrame matches the expected result
        pd_testing.assert_frame_equal(crypto_factory.df_volume_bars, expected_result)



if __name__ == '__main__':
    unittest.main()
