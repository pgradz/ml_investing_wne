import os
import pandas as pd
import numpy as np
import logging
from ml_investing_wne.utils import get_logger


# logger = logging.getLogger(__name__)
logger = get_logger()

class BinanceDataProcessor():

    FIRST_COLUMNS = ['datetime', 'price', 'volume']

    def __init__(self, file=None, volume_frequency=500, freq='60min', strategy='volume_bars', files_path=None, output_path=None) -> None:
        self.file = file
        self.volume_frequency = volume_frequency
        self.freq = freq
        self.strategy = strategy
        if files_path is None:
            self.files_path = [file]
        else:
            self.files_path  = [os.path.join(files_path, file) for file in os.listdir(files_path)]
            self.files_path.sort()
        if output_path is not None:
            if self.strategy == 'volume_bars':
                self.output_path = os.path.join(output_path, f'volume_bars_{self.volume_frequency}.csv')
            elif self.strategy == 'time_aggregated':
                self.output_path = os.path.join(output_path, f'time_aggregated_{self.freq}.csv')
            else:
                pass
        else:
            self.output_path = output_path
        self.remaining_df = None
        self.processed_df = None
        self.running_volume_sum = 0

    def clean_binance(self, df):
        df['datetime'] = pd.to_datetime(df['time'],unit='ms')
        df = df[BinanceDataProcessor.FIRST_COLUMNS]
        return df
    
    def generate_volumebars(self, df):
        
        df_np = df.to_numpy()
        self.running_volume_sum += df_np[:,2].sum()
        if isinstance(self.remaining_df, np.ndarray):
            df_np = np.concatenate((self.remaining_df, df_np), axis=0)
        times = df_np[:,0]
        prices = df_np[:,1]
        volumes = df_np[:,2]
        ans =  np.empty([len(prices), 7], dtype=object)
        candle_counter = 0
        vol = 0
        lasti = 0
        for i, _ in enumerate(prices):
            if i % 100000 == 0:
                pass
                # logger.info(f'processed {i/len(prices) * 100} %')
            vol += volumes[i]
            if vol >= self.volume_frequency:
                ans[candle_counter][0] = times[i]                                       # time
                ans[candle_counter][1] = prices[lasti]                                  # open
                ans[candle_counter][2] = np.max(prices[lasti:i+1])                      # high
                ans[candle_counter][3] = np.min(prices[lasti:i+1])                      # low
                ans[candle_counter][4] = prices[i]                                      # close
                ans[candle_counter][5] = np.sum(volumes[lasti:i+1])                     # volume
                ans[candle_counter][6] = (times[i] - times[lasti]).total_seconds()      # time
                candle_counter += 1
                lasti = i+1
                vol = 0
        self.remaining_df = df_np[lasti:,:]
        df = pd.DataFrame(ans[:candle_counter], 
                            columns = ['datetime','open','high', 'low','close', 'volume', 'seconds'])
        return df 
    
    def time_aggregation(self, df):

        if isinstance(self.remaining_df, pd.DataFrame):
            df = pd.concat([self.remaining_df, df], ignore_index=True)

        # move part of the last period to the next batch
        df['datetime_rounded'] = df['datetime'].dt.floor(self.freq)
        remaining_df = df.loc[df['datetime_rounded']== df['datetime_rounded'].max()]
        self.remaining_df = remaining_df.drop(columns=['datetime_rounded'])
        df = df.loc[df['datetime_rounded']< df['datetime_rounded'].max()].copy()
        df.drop(columns=['datetime_rounded'], inplace=True)
        
        df.set_index('datetime', inplace=True)
        if self.freq == '1440min':
            price = df['price'].resample('D').agg({'open': 'first',
                                           'high': 'max',
                                           'low': 'min',
                                           'close': 'last'
                                           })
            volume = df['volume'].resample('D').agg({'volume':'sum'})

        else:
            price = df['price'].resample(self.freq).agg({'open': 'first',
                                           'high': 'max',
                                           'low': 'min',
                                           'close': 'last'
                                           })
            volume = df['volume'].resample(self.freq).agg({'volume':'sum'})

        df = price.join(volume)
        # logging.info(f"Forward filling the following nulls")
        # logging.info(f"{df.isnull().sum()}")
        df.fillna(method="ffill", inplace=True)
        df.reset_index(inplace=True)

        return df 


    def load_chunks(self, chunksize=1000000):

        for file in self.files_path:
            logger.info(f'processing {file}')
            for chunk in pd.read_csv(file, names=['trade_id', 'price', 'volume','quoteQty', 'time', 'is_buyer_maker','is_best_match'], chunksize=chunksize):
                df_chunk = self.clean_binance(chunk)
                if self.strategy == 'volume_bars':
                    df_chunk = self.generate_volumebars(df_chunk)
                elif self.strategy == 'time_aggregated':
                    df_chunk = self.time_aggregation(df_chunk)
                else:
                    logging.info("flow not implemented")
                if isinstance(self.processed_df, pd.DataFrame):
                    self.processed_df = pd.concat([self.processed_df, df_chunk], ignore_index=True)
                else:
                    self.processed_df = df_chunk
            # save work in progress        
            if self.output_path is not None:
                self.processed_df.to_csv(self.output_path, index=False)

        if self.output_path is not None:
            self.processed_df.to_csv(self.output_path, index=False)

        try:
            volume_check_ratio = (self.processed_df['volume'].sum()+self.remaining_df[:,2].sum())/self.running_volume_sum * 100
        except:
            volume_check_ratio = (self.processed_df['volume'].sum()+self.remaining_df['volume'].sum())/self.running_volume_sum * 100

        logger.info(f'volume check, processed to raw files ratio: {volume_check_ratio} %')

#binance_processor = BinanceDataProcessor(file='/Users/i0495036/Downloads/BTCUSDT-trades-2023-01.csv')
# binance_processor = BinanceDataProcessor(volume_frequency=50000, 
#                                          files_path='/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/raw/crypto/binance_ETHUSDT',
#                                          output_path='/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/processed/binance_ETHUSDT',
#                                          strategy='volume_bars'
#                                          )
# binance_processor = BinanceDataProcessor(freq='15min', 
#                                          files_path='/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/raw/crypto/binance_ETHUSDT',
#                                          output_path='/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/processed/binance_ETHUSDT',
#                                          strategy='time_aggregated'
#                                          )
# binance_processor = BinanceDataProcessor(freq='10min', 
#                                          files_path='/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/raw/crypto/binance_BTCUSDT',
#                                          output_path='/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/processed/binance_BTCUSDT',
#                                          strategy='time_aggregated'
#                                          )


# binance_processor = BinanceDataProcessor(freq='10min', 
#                                          files_path='/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/raw/crypto/binance_SOLUSDT',
#                                          output_path='/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/processed/binance_SOLUSDT',
#                                          strategy='time_aggregated'

binance_processor = BinanceDataProcessor(freq='60min', 
                                         files_path='/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/raw/crypto/binance_ETHBTC',
                                         output_path='/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/processed/binance_ETHBTC',
                                         strategy='time_aggregated'
                                         )
                                         
binance_processor.load_chunks()
