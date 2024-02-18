import datetime
import unittest
from unittest.mock import patch

import pandas as pd
import pandas.testing as pd_testing

from ml_investing_wne.binance_data_processor import BinanceDataProcessor

datetime_col = [datetime.datetime(2021, 1, 1, 12, 0, 0), datetime.datetime(2021, 1, 2, 0, 0, 0),
                    datetime.datetime(2021, 1, 2, 12, 0, 0), datetime.datetime(2021, 1, 3, 0, 0, 0),
                    datetime.datetime(2021, 1, 3, 12, 0, 0), datetime.datetime(2021, 1, 4, 0, 0, 0),
                    datetime.datetime(2021, 1, 4, 12, 0, 0), datetime.datetime(2021, 1, 5, 0, 0, 0),
                    datetime.datetime(2021, 1, 5, 12, 0, 0), datetime.datetime(2021, 1, 6, 0, 0, 0)]

test_df = pd.DataFrame({"time": datetime_col, "price": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
                                    "volume": [10.0, 20.0, 30.0, 40.0, 10.0, 20.0, 30.0, 40.0, 10.0, 20.0]})

test_df_2 = pd.DataFrame({"time": [datetime.datetime(2021, 1, 6, 12, 0, 0), datetime.datetime(2021, 1, 7, 0, 0, 0), datetime.datetime(2021, 1, 7, 12, 0, 0)] , 
                          "price": [100.0, 101.0, 102.0],
                            "volume": [10.0, 20.0, 30.0]})



class TestBinance(unittest.TestCase):

    @patch("ml_investing_wne.binance_data_processor.pd.read_csv", return_value=[test_df])
    def test_load_one_chunk(self):

        binance_processor = BinanceDataProcessor(file='file_path', volume_frequency=50)
        binance_processor.load_chunks(chunksize=3)
        binance_processor.processed_df = binance_processor.processed_df.astype({'open':'float','high':'float','low':'float','close':'float', 'volume': 'float', 'seconds':'float'})
        binance_processor.processed_df ['datetime'] = pd.to_datetime(binance_processor.processed_df ['datetime'])
        print(binance_processor.processed_df)
        print(binance_processor.remaining_df)
        df_processed = pd.DataFrame({"datetime": [datetime.datetime(2021, 1, 2, 12, 0, 0), datetime.datetime(2021, 1, 3, 12, 0, 0), 
                                          datetime.datetime(2021, 1, 4, 12, 0, 0), datetime.datetime(2021, 1, 5, 12, 0, 0)],
                            "open": [100.0, 103.0, 105.0, 107.0],
                            "high": [102.0, 104.0, 106.0, 108.0],
                            "low": [100.0, 103.0, 105.0, 107.0],
                            "close": [102.0, 104.0, 106.0, 108.0],
                            "volume": [60.0, 50.0, 50.0, 50.0],
                            "seconds": [86400.0, 43200.0, 43200.0, 43200.0]
                           })
        pd_testing.assert_frame_equal(binance_processor.processed_df, df_processed)
        self.assertEqual(binance_processor.processed_df['volume'].sum() + binance_processor.remaining_df[:,2].sum(), test_df['volume'].sum())



    @patch("ml_investing_wne.binance_data_processor.pd.read_csv", return_value=[test_df, test_df_2])
    def test_load_two_chunks(self):

        binance_processor = BinanceDataProcessor(file='file_path', volume_frequency=50)
        binance_processor.load_chunks(chunksize=3)
        binance_processor.processed_df = binance_processor.processed_df.astype({'open':'float','high':'float','low':'float','close':'float', 'volume': 'float', 'seconds':'float'})
        binance_processor.processed_df ['datetime'] = pd.to_datetime(binance_processor.processed_df ['datetime'])
        print(binance_processor.processed_df)
        print(binance_processor.remaining_df)
        df_processed = pd.DataFrame({"datetime": [datetime.datetime(2021, 1, 2, 12, 0, 0), datetime.datetime(2021, 1, 3, 12, 0, 0), 
                                          datetime.datetime(2021, 1, 4, 12, 0, 0), datetime.datetime(2021, 1, 5, 12, 0, 0), datetime.datetime(2021, 1, 7, 0, 0, 0)],
                            "open": [100.0, 103.0, 105.0, 107.0, 109.0],
                            "high": [102.0, 104.0, 106.0, 108.0, 109.0],
                            "low": [100.0, 103.0, 105.0, 107.0, 100.0],
                            "close": [102.0, 104.0, 106.0, 108.0, 101.0],
                            "volume": [60.0, 50.0, 50.0, 50.0, 50.0],
                            "seconds": [86400.0, 43200.0, 43200.0, 43200.0, 86400.0]
                           })
        pd_testing.assert_frame_equal(binance_processor.processed_df, df_processed)
        self.assertEqual(binance_processor.processed_df['volume'].sum() + binance_processor.remaining_df[:,2].sum(), test_df['volume'].sum() + test_df_2['volume'].sum())
    

if __name__ == '__main__':
    unittest.main()









