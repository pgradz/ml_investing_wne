import os
import datetime
import pandas as pd
import numpy as np
import logging
import plotly.graph_objects as go
from ml_investing_wne import config

logger = logging.getLogger(__name__)


class CryptoFactory():

    FIRST_COLUMNS = ['datetime', 'price', 'volume']

    def __init__(self, provider, currency, df=None) -> None:
        self.provider = provider
        self.currency = currency
        self.df_time_aggregated = None
        self.df_volume_bars = None
        if isinstance(df, pd.DataFrame):
            self.df_time_aggregated = df
        else:
            self.df = self.load_data(self.provider, self.currency)


    def load_data(self, exchange, currency):
        """ Loads csv files for corresponding exchange and  currency, and returns dataframe in 
        standardized format: date, price, volume, others

        Args:
            exchange (str): name of the exchange
            currency (str): name of the crypocurrency

        Returns:
            pd.DataFrame: dataframe in standardized format: date, price, volume, others
        """
        if self.provider == 'Bitstamp':
            df = self.load_bitstamp(self.currency)
        elif self.provider == 'Binance':
            df = self.load_binance(self.currency)
        else:
            pass

        return df
    
    def load_binance(self, currency):
        '''
        '''
        if config.RUN_SUBTYPE == 'volume_bars':   
            file_path = os.path.join(config.processed_data_path, f'binance_{currency}', f'volume_bars_{config.volume}.csv')
            # volume bars are almost exactly the same, so don't bring any information
        elif config.RUN_SUBTYPE in ['time_aggregated', 'triple_barrier_time_aggregated']:  
            file_path = os.path.join(config.processed_data_path, f'binance_{currency}', f'time_aggregated_{config.freq}.csv')
        df = pd.read_csv(file_path, parse_dates=['datetime'])
        df.set_index('datetime', inplace=True)

        if config.RUN_SUBTYPE in ['time_aggregated', 'triple_barrier_time_aggregated']:  
            self.df_time_aggregated = df
        if config.RUN_SUBTYPE == 'volume_bars':  
            df.drop(columns=['volume'], inplace=True)
            df = self.deal_with_duplicates(df)
            self.df_volume_bars = df

        return df

    def load_bitstamp(self, currency):
        """ Loads data from Bitstamp
        Args:
            currency (str): name of the crypocurrency

        Returns:
            pd.DataFrame: dataframe in standardized format: date, price, volume, others
        """
        file_path = os.path.join(config.raw_data_path, f'Bitstamp_{currency}.csv')
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['t'],unit='ms')
        df.sort_values(by='datetime', inplace=True)
        logger.info(f'head of raw dataset: {df.head()}')
        df.rename(columns={'p':'price', 'q': 'volume'}, inplace=True)
        df.drop(columns=['t'], inplace=True)
        rest_of_columns = [x for x in df.columns if (x not in CryptoFactory.FIRST_COLUMNS)]
        column_order = CryptoFactory.FIRST_COLUMNS + rest_of_columns
        df = df[column_order]

        return df
        
        
    def generate_volumebars(self, frequency=10):
        
        df_np = self.df.to_numpy()
        times = df_np[:,0]
        prices = df_np[:,1]
        volumes = df_np[:,2]
        ans =  np.empty([len(prices), 6], dtype=object)
        candle_counter = 0
        vol = 0
        lasti = 0
        for i, _ in enumerate(prices):
            vol += volumes[i]
            if vol >= frequency:
                ans[candle_counter][0] = times[i]                          # time
                ans[candle_counter][1] = prices[lasti]                     # open
                ans[candle_counter][2] = np.max(prices[lasti:i+1])         # high
                ans[candle_counter][3] = np.min(prices[lasti:i+1])         # low
                ans[candle_counter][4] = prices[i]                         # close
                ans[candle_counter][5] = np.sum(volumes[lasti:i+1])        # volume
                candle_counter += 1
                lasti = i+1
                vol = 0

        df = pd.DataFrame(ans[:candle_counter], 
                            columns = ['datetime','open','high', 'low','close', 'volume'])
        df = self.deal_with_duplicates(df)
        # df = df.set_index('datetime')
        self.df_volume_bars = df


    def generate_dollarbars(self, frequency=1000):
        df_np = self.df.to_numpy()
        times = df_np[:,0]
        prices = df_np[:,1]
        volumes = df_np[:,2]
        ans =  np.empty([len(prices), 6], dtype=object)
        candle_counter = 0
        dollars = 0
        lasti = 0
        for i, _ in enumerate(prices):
            dollars += volumes[i]*prices[i]
            if dollars >= frequency:
                ans[candle_counter][0] = times[i]                          # time
                ans[candle_counter][1] = prices[lasti]                     # open
                ans[candle_counter][2] = np.max(prices[lasti:i+1])         # high
                ans[candle_counter][3] = np.min(prices[lasti:i+1])         # low
                ans[candle_counter][4] = prices[i]                         # close
                ans[candle_counter][5] = np.sum(volumes[lasti:i+1])        # volume
                candle_counter += 1
                lasti = i+1
                dollars = 0
        df = pd.DataFrame(ans[:candle_counter], 
                            columns = ['datetime','open','high', 'low','close', 'volume'])
        df = self.deal_with_duplicates(df)
        # df = df.set_index('datetime')
        self.df_volume_bars = df

    def deal_with_duplicates(self, df):

        df = df.groupby('datetime').agg({'open': 'first',
                                    'high': 'max',
                                    'low': 'min',
                                    'close': 'last'})
        return df



    def time_aggregation(self, freq):
        
        df = self.df.set_index('datetime')
        if freq == '1440min':
            price = df['price'].resample('D').agg({'open': 'first',
                                           'high': 'max',
                                           'low': 'min',
                                           'close': 'last'
                                           })
            volume = df['volume'].resample('D').agg({'volume':'sum'})

        else:
            price = df['price'].resample(freq).agg({'open': 'first',
                                           'high': 'max',
                                           'low': 'min',
                                           'close': 'last'
                                           })
            volume = df['volume'].resample(freq).agg({'volume':'sum'})

        df = price.join(volume)
        logging.info(f"Forward filling the following nulls")
        logging.info(f"{df.isnull().sum()}")
        df.fillna(method="ffill", inplace=True)
        # option to drop missing rows
        # n_rows_before = df.shape[0]
        # df.dropna(inplace=True)
        # n_rows_after = df.shape[0]
        # logging.info(f"""number of missing rows with this time aggregation is 
        #             {n_rows_before-n_rows_after} out of {n_rows_before} all rows""")
        self.df_time_aggregated = df
        
    def plot_volumebars(self):
        self._plot_ohlc(self.df_volume_bars)

    def _plot_ohlc(self, df):

        fig = go.Figure(data=go.Ohlc(x=df.index,
                             open=df['open'],
                             high=df['high'],
                             low=df['low'],
                             close=df['close']))
        fig.write_image(os.path.join(config.processed_data_path, 
                                    f"{config.currency}_{config.provider}.png"))


    def get_daily_volatility(self,close,span0=20):
        # simple percentage returns
        df0=close.pct_change()
        # 20 days, a month EWM's std as boundary
        df0=df0.ewm(span=span0).std()
        df0.dropna(inplace=True)
        return df0


    def get_3_barriers(self, _open, close, high, low, daily_volatility, t_final, 
                        upper_lower_multipliers):
    #create a container
        barriers = pd.DataFrame(columns=['periods_passed', 'open',
                'close', 'high', 'low', 'vert_barrier', \
                'top_barrier', 'bottom_barrier'], \
                index = daily_volatility.index)
        for day, vol in daily_volatility.items():
            periods_passed = len(daily_volatility.loc \
                        [daily_volatility.index[0] : day])

            #set the vertical barrier, without - 1, vertical barrier was t_final + 1
            if (periods_passed + t_final < len(daily_volatility.index) \
                and t_final != 0):
                vert_barrier = daily_volatility.index[
                                    periods_passed + t_final - 1]
            else:
                vert_barrier = np.nan
            #set the top barrier
            if upper_lower_multipliers[0] > 0:
                top_barrier = close.loc[day] + close.loc[day] * \
                            upper_lower_multipliers[0] * vol
            else:
                #set it to NaNs
                top_barrier = pd.Series(index=close.index)
            #set the bottom barrier
            if upper_lower_multipliers[1] > 0:
                bottom_barrier = close.loc[day] - close.loc[day] * \
                            upper_lower_multipliers[1] * vol
            else: 
                #set it to NaNs
                bottom_barrier = pd.Series(index=close.index)
            barriers.loc[day, ['periods_passed', 'open', 'close', 'high', 'low',\
            'vert_barrier','top_barrier', 'bottom_barrier']] = \
            periods_passed, _open.loc[day], close.loc[day], high.loc[day], low.loc[day], \
            vert_barrier, top_barrier, bottom_barrier

        # placeholder values
        barriers['barrier_touched'] = None
        barriers['y_pred'] = None
        barriers['prc_change'] = None
        barriers['barrier_touched_date'] = None
        
        return barriers


    def get_3_barriers_labels(self, barriers):

        ties = 0

        for i in range(len(barriers.index)-1):

            start = barriers.index[i]
            start_looking = barriers.index[i+1]
            end = barriers.vert_barrier[i]
            if pd.notna(end):
                # assign the initial and final price
                price_initial = barriers.close[start]
                price_final = barriers.close[end]
        # assign the top and bottom barriers
                top_barrier = barriers.top_barrier[i]
                bottom_barrier = barriers.bottom_barrier[i]
        #set the profit taking and stop loss conditons
                current_snapshot = barriers[start_looking: end]
                
                if current_snapshot.loc[current_snapshot['low'] <= bottom_barrier].shape[0] == 0:
                    bottom_barrier_date = datetime.datetime(9999,1,1)
                else: 
                    bottom_barrier_date = current_snapshot.loc[current_snapshot['low'] <= bottom_barrier].index.min()
                if current_snapshot.loc[current_snapshot['high'] >= top_barrier].shape[0] == 0:
                    top_barrier_date = datetime.datetime(9999,1,1)
                else: 
                    top_barrier_date = current_snapshot.loc[current_snapshot['high']  >= top_barrier].index.min()
                    
                if bottom_barrier_date == top_barrier_date == datetime.datetime(9999,1,1):
                    barriers['barrier_touched'][i] = 'vertical'
                    barriers['prc_change'][i] = (price_final - price_initial)/price_initial
                    if barriers['prc_change'][i] > 0:
                        barriers['y_pred'][i] = 1
                    else:
                        barriers['y_pred'][i] = 0
                    barriers['barrier_touched_date'][i] = barriers['vert_barrier'][i]
                elif bottom_barrier_date < top_barrier_date:
                    barriers['barrier_touched'][i] = 'bottom'
                    barriers['y_pred'][i] = 0
                    barriers['prc_change'][i] = (bottom_barrier - price_initial)/price_initial
                    barriers['barrier_touched_date'][i] = bottom_barrier_date
                elif bottom_barrier_date >= top_barrier_date:
                    if bottom_barrier_date == top_barrier_date:
                        ties+=1
                    barriers['barrier_touched'][i] = 'top'
                    barriers['y_pred'][i] = 1
                    barriers['prc_change'][i] = (top_barrier - price_initial)/price_initial
                    barriers['barrier_touched_date'][i] = top_barrier_date
                
        share_of_ties = ties/barriers.shape[0]
        logging.info(f'number of ties: {ties}, share: {share_of_ties:.2}f')

        return barriers

    def run_3_barriers(self, t_final=10, upper_lower_multipliers = [2, 2], fixed_barrier = None):
        # t_final how many days we hold the stock which set the vertical barrier
        # upper_lower_multipliers the up and low boundary multipliers
        try:
            close = self.df_time_aggregated['close']
            low = self.df_time_aggregated['low']
            high = self.df_time_aggregated['high']
            _open = self.df_time_aggregated['open']
        except:
            logging.error('Run time aggreagtion first')

        daily_volatility = self.get_daily_volatility(close)
        # overwrite dynamic (volatility based) barrier with fixed percentage
        if fixed_barrier:
            daily_volatility = pd.Series(data=np.repeat(fixed_barrier, len(daily_volatility)),
                                index=daily_volatility.index, name='close')
            upper_lower_multipliers = [1, 1]

        close = close[daily_volatility.index]
        low = low[daily_volatility.index]
        high = high[daily_volatility.index]
        _open = _open[daily_volatility.index]

        barriers = self.get_3_barriers(_open= _open, close=close, high=high, low=low,
                                        daily_volatility=daily_volatility, t_final=t_final,
                                        upper_lower_multipliers=upper_lower_multipliers)
        barriers = self.get_3_barriers_labels(barriers)
        self.df_3_barriers_additional_info = barriers[['prc_change', 'barrier_touched',
                                                        'barrier_touched_date', 'top_barrier', 
                                                        'bottom_barrier']]
        barrier_touched_freq = self.df_3_barriers_additional_info['barrier_touched'].value_counts(normalize=True)
        logging.info(f"frequency of barrier touched: {barrier_touched_freq}")                                                
        self.df_3_barriers_additional_info.reset_index(inplace=True)
        self.df_3_barriers_additional_info['time_step'] = None

        for i in self.df_3_barriers_additional_info.index:    
            barrier_touched_date = self.df_3_barriers_additional_info['barrier_touched_date'][i]
            if barrier_touched_date is None:
                continue
            next_index = self.df_3_barriers_additional_info.loc[self.df_3_barriers_additional_info['datetime']==barrier_touched_date].index[0]
            self.df_3_barriers_additional_info['time_step'][i] = next_index - i

        time_step_freq = self.df_3_barriers_additional_info['time_step'].value_counts(normalize=True)
        logging.info(f"frequency of tripple barrier length: {time_step_freq}") 
        rows_df_3_barriers_additional_info= self.df_3_barriers_additional_info.shape[1]
        self.df_3_barriers_additional_info.dropna(inplace=True)
        logging.info(f"Rows with Na dropped from df_3_barriers_additional_info: {self.df_3_barriers_additional_info.shape[1] - rows_df_3_barriers_additional_info}") 
        self.df_3_barriers_additional_info = self.df_3_barriers_additional_info.astype({'prc_change':'float', 'barrier_touched_date':'datetime64[ns]', 
                                                                                        'top_barrier': 'float', 'bottom_barrier': 'float', 'time_step': 'int64'})

        self.df_3_barriers = barriers[['open', 'close', 'high', 'low', 'y_pred']]
        self.df_3_barriers.dropna(inplace=True)
