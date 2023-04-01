import unittest
from unittest.mock import patch
import datetime
import pandas as pd
from ml_investing_wne.data_engineering.crypto_factory import CryptoFactory
import pandas.testing as pd_testing


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
        self.y_pred = [1, 0, 0, 0, 1, 1, 1, 1, 1]
        self.test_df = pd.DataFrame({"datetime": self.datetime_col, "open": self._open, 'close': self.close, 'high': self.high, 'low': self.low})
        self.test_df.set_index('datetime', inplace=True)
                
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
        crypto_factory = CryptoFactory(provider='Binance', currency='BTCUSDT')
        crypto_factory.df_time_aggregated = crypto_factory.df
        crypto_factory.run_3_barriers(t_final=3,upper_lower_multipliers = [1, 1], fixed_barrier = 0.02)
        pd_testing.assert_frame_equal(crypto_factory.df_3_barriers_additional_info, result_df_3_barriers_additional_info)


if __name__ == '__main__':
    unittest.main()