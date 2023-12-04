import os
import pandas as pd
import numpy as np
import logging
import boto3
import io
from ml_investing_wne.utils import get_logger


# logger = logging.getLogger(__name__)
logger = get_logger()

class BinanceDataProcessor():
    """ 
    Class for processing binance data
    """

    FIRST_COLUMNS = ['datetime', 'price', 'volume']
    EXCHANGE = 'binance'

    def __init__(self, crypto :str=None, file: str=None, volume_frequency: int=500, value_frequency: int=10000000, freq: str='60min', strategy:str ='volume_bars', 
    files_path: str=None, output_path: str=None, s3_path: str=None) -> None:
        '''
        it allows to process data in two ways:
        1. volume_bars - it generates volume bars from raw data
        2. time_aggregated - it aggregates data based on time
        data can be processed from one file or from multiple files in a directory
        Args:
            file: path to file
            volume_frequency: frequency of volume bars
            freq: frequency of time aggregation
            strategy: strategy to use, either volume_bars or time_aggregated
            files_path: path to directory with files
            output_path: path to save processed data
            s3_path: path to s3 bucket
        '''
        self.file = file
        self.volume_frequency = volume_frequency
        self.value_frequency = value_frequency
        self.freq = freq
        self.strategy = strategy
        if file is not None:
            self.files_path = [file]
        if files_path is not None:
            self.files_path  = [os.path.join(files_path, file) for file in os.listdir(files_path)]
            self.files_path.sort()
        if s3_path is not None:
            self.s3_path = s3_path
            s3 = boto3.resource('s3')
            bucket = s3.Bucket(s3_path)
            self.files_path = []
            for obj in bucket.objects.all():
                if obj.key.endswith('.csv') and obj.key.startswith((BinanceDataProcessor.EXCHANGE + '_' + crypto)):
                    self.files_path.append(obj.key)
            self.files_path.sort()
            print(self.files_path)

        if output_path is not None:
            output_path = os.path.join(output_path, (BinanceDataProcessor.EXCHANGE + '_' + crypto))
            if self.strategy == 'volume_bars':
                self.output_path = os.path.join(output_path, f'volume_bars_{self.volume_frequency}.csv')
            elif self.strategy == 'time_aggregated':
                self.output_path = os.path.join(output_path, f'time_aggregated_{self.freq}.csv')
            elif self.strategy == 'dollar_bars':
                self.output_path = os.path.join(output_path, f'dollar_bars_{self.value_frequency}.csv')
            else:
                logger.info('Output_path is missing')
            logger.info(f'output path is {self.output_path}')
        else:
            logging.error(msg='output_path is required')

        self.remaining_df = None
        self.processed_df = None
        self.running_volume_sum = 0
        self.running_value_sum = 0

    def clean_binance(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Cleans binance data. Converts time to datetime and drops unnecessary columns
        '''
        df['datetime'] = pd.to_datetime(df['time'],unit='ms')
        df = df[BinanceDataProcessor.FIRST_COLUMNS]
        return df
    
    def generate_volumebars(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Generates volume bars from binance data
        '''
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

    def generate_dollar_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Generates dollar bars from binance data
        '''
        df['value'] = df['price'] * df['volume']
        df_np = df.to_numpy()
        self.running_value_sum += df_np[:,3].sum()
        if isinstance(self.remaining_df, np.ndarray):
            df_np = np.concatenate((self.remaining_df, df_np), axis=0)
        times = df_np[:,0]
        prices = df_np[:,1]
        volumes = df_np[:,2]
        values = df_np[:,3]
        ans =  np.empty([len(prices), 7], dtype=object)
        candle_counter = 0
        val = 0
        lasti = 0
        for i, _ in enumerate(prices):
            if i % 100000 == 0:
                pass
                # logger.info(f'processed {i/len(prices) * 100} %')
            val += values[i]
            if val >= self.value_frequency:
                ans[candle_counter][0] = times[i]                                       # time
                ans[candle_counter][1] = prices[lasti]                                  # open
                ans[candle_counter][2] = np.max(prices[lasti:i+1])                      # high
                ans[candle_counter][3] = np.min(prices[lasti:i+1])                      # low
                ans[candle_counter][4] = prices[i]                                      # close
                ans[candle_counter][5] = np.sum(volumes[lasti:i+1])                     # volume
                ans[candle_counter][6] = (times[i] - times[lasti]).total_seconds()      # time
                candle_counter += 1
                lasti = i+1
                val = 0
        self.remaining_df = df_np[lasti:,:]
        df = pd.DataFrame(ans[:candle_counter], 
                            columns = ['datetime','open','high', 'low','close', 'volume', 'seconds'])
        return df

    
    def time_aggregation(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Aggregates data based on time
        '''
        self.running_volume_sum += df['volume'].sum()
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
        '''
        Loads data in chunks and processes it
        '''

        for file in self.files_path:
            logger.info(f'processing {file}')
            for chunk in pd.read_csv(file, names=['trade_id', 'price', 'volume','quoteQty', 'time', 'is_buyer_maker','is_best_match'], chunksize=chunksize):
                df_chunk = self.clean_binance(chunk)
                if self.strategy == 'volume_bars':
                    df_chunk = self.generate_volumebars(df_chunk)
                elif self.strategy == 'time_aggregated':
                    df_chunk = self.time_aggregation(df_chunk)
                elif self.strategy == 'dollar_bars':
                    df_chunk = self.generate_dollar_bars(df_chunk)
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
    
    def load_from_s3(self, chunksize=1000000):

        s3 = boto3.resource('s3')
        for file in self.files_path:

            logger.info(f'processing {file}')
            obj = s3.Object(self.s3_path, file)
            for df in pd.read_csv(f"s3://{self.s3_path}/{file}", names=['trade_id', 'price', 'volume','quoteQty', 'time', 'is_buyer_maker','is_best_match'], chunksize=chunksize):
 
                # df = pd.read_csv( f"s3://{self.s3_path}/{file}", names=['trade_id', 'price', 'volume','quoteQty', 'time', 'is_buyer_maker','is_best_match'])
                df = self.clean_binance(df)
                if self.strategy == 'volume_bars':
                    df = self.generate_volumebars(df)
                elif self.strategy == 'time_aggregated':
                    df = self.time_aggregation(df)
                else:
                    logging.info("flow not implemented")
                if isinstance(self.processed_df, pd.DataFrame):
                    self.processed_df = pd.concat([self.processed_df, df], ignore_index=True)
                else:
                    self.processed_df = df
            # save work in progress        
            self.processed_df.to_csv(self.output_path, index=False)

#binance_processor = BinanceDataProcessor(file='/Users/i0495036/Downloads/BTCUSDT-trades-2023-01.csv')
# binance_processor = BinanceDataProcessor(volume_frequency=100000, 
#                                          files_path='/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/raw/crypto/binance_ETHUSDT',
#                                          output_path='/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/processed',
#                                          strategy='volume_bars'
#                                          )
# binance_processor.load_chunks()
# binance_processor = BinanceDataProcessor(value_frequency=50000000, crypto = 'ETHUSDT',
#                                          files_path='/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/raw/crypto/binance_ETHUSDT',
#                                          output_path='/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/processed',
#                                          strategy='dollar_bars'
#                                          )
# binance_processor.load_chunks()
# binance_processor = BinanceDataProcessor(freq='1440min', 
#                                          crypto = 'ETHUSDT',
#                                          files_path='/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/raw/crypto/binance_ETHUSDT',
#                                          output_path='/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/processed',
#                                          strategy='time_aggregated'
#                                          )
# binance_processor.load_chunks()
# binance_processor = BinanceDataProcessor(freq='10min', 
#                                          files_path='/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/raw/crypto/binance_BTCUSDT',
#                                          output_path='/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/processed/binance_BTCUSDT',
#                                          strategy='time_aggregated'
#                                          )

# binance_processor = BinanceDataProcessor(freq='1min', 
#                                          crypto = 'LTCUSDT',
#                                          files_path='/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/raw/crypto/binance_LTCUSDT',
#                                          output_path='/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/processed',
#                                          strategy='time_aggregated'
#                                          )
# binance_processor.load_chunks()


# binance_processor = BinanceDataProcessor(freq='1440min', 
#                                          crypto = 'SOLUSDT',
#                                          files_path='/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/raw/crypto/binance_SOLUSDT',
#                                          output_path='/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/processed',
#                                          strategy='time_aggregated'
#                                          )
# binance_processor.load_chunks()

binance_processor = BinanceDataProcessor(volume_frequency=30000, 
                                         crypto = 'ETHUSDT',
                                         files_path='/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/raw/crypto/binance_ETHUSDT',
                                         output_path='/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/processed',
                                         strategy='volume_bars'
                                         )
binance_processor.load_chunks()

# binance_processor = BinanceDataProcessor(volume_frequency='60min', 
#                                          files_path='/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/raw/crypto/binance_ETHBTC',
#                                          output_path='/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/processed/binance_ETHBTC',
#                                          strategy='time_aggregated'
#                                          )
# binance_processor = BinanceDataProcessor(freq='60min', 
#                                          crypto = 'BTCUSDT',
#                                          s3_path='crypto-data-uw',
#                                          output_path='/home/ec2-user/SageMaker/ml_investing_wne/src/ml_investing_wne/data/processed',
#                                          strategy='time_aggregated'
#                                          )
                                         
# binance_processor.load_from_s3()
# binance_processor = BinanceDataProcessor(freq='1440min', 
#                                          crypto = 'MATICUSDT',
#                                          files_path='/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/raw/crypto/binance_MATICUSDT',
#                                          output_path='/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/processed',
#                                          strategy='time_aggregated'
#                                          )
# binance_processor.load_chunks()
