import logging
import os

import pandas as pd

import ml_investing_wne.config as config

# logger is a child of a logger from main module that is calling this module
logger = logging.getLogger(__name__)

def import_forex_csv(raw_data_path, currency, names=['currency', 'datetime', 'bid', 'ask'], **kwargs):
    '''
    read all csv files for currency pairs defined in config
    :param raw_data_path: str path to raw data folder
    :param currencies: currency
    :param names: list of headers
    :param kwargs:
    :return: generator of pandas dataframes
    '''
    path = os.path.join(raw_data_path, currency)
    files = [os.path.join(path, f) for f in os.listdir(path) if '.csv' in f]
    for file in files:
        yield pd.read_csv(file, **kwargs, parse_dates=['datetime'], names=names)


def aggregate_time_window(df, freq):
    '''

    '''
    # for development - to be removed
    # df = pd.concat(import_truefx_csv(config.raw_data_path, config.currencies, nrows=10000))
    df['spread'] = df['ask'] - df['bid']
    currency = df['currency'].unique()[0]
    df_temp = df.loc[df['currency'] == currency]
    df_agg = df_temp.set_index('datetime')
    df_agg_bid = df_agg['bid'].resample(freq).agg({'bid_open': 'first',
                                                   'bid_high': 'max',
                                                   'bid_min': 'min',
                                                   'bid_close': 'last'
                                                   # ,'no_of_ticks': 'size'
                                                   })
    df_agg_ask = df_agg['ask'].resample(freq).agg({'ask_open': 'first',
                                                   'ask_high': 'max',
                                                   'ask_min': 'min',
                                                   'ask_close': 'last'})
    # df_agg_spread = df_agg['spread'].resample(freq).agg({'spread_open': 'first',
    #                                                      'spread_high': 'max',
    #                                                      'spread_min': 'min',
    #                                                      'spread_close': 'last'})

    df_all = pd.merge(df_agg_bid, df_agg_ask, how='inner', left_index=True, right_index=True)
    # df_all = pd.merge(df_all, df_agg_spread, how='inner', left_index=True, right_index=True)
    df_all['currency'] = currency

    return df_all


def check_time_delta(df):
    '''
    checks that time delta between observations is always the same
    :param df: pandas dataframe with datetime as index
    '''
    df['datetime'] = df.index
    df['time_delta'] = (df['datetime'] - df['datetime'].shift())
    check = df['time_delta'].value_counts()
    if len(check) == 1:
        logger.info('there is only one interval in data : {}'.format(check.index[0]))
    else:
        logger.error('there are multiple time intervals in data')
    df.drop(['datetime', 'time_delta'], axis=1, inplace=True)


def import_hist_data_csv(currency, raw_data_path = config.raw_data_path, **kwargs):
    '''
    read all csv files for currency pairs defined in config
    :param raw_data_path: str path to raw data folder
    :param currencies: currency
    :param names: list of headers
    :param kwargs:
    :return: generator of pandas dataframes
    '''
    path = os.path.join(raw_data_path, currency)
    files = [os.path.join(path, f) for f in os.listdir(path) if '.csv' in f]
    for file in files:
        yield pd.read_csv(file, sep=';', names=['datetime_text', 'open', 'high', 'low',
                                                'close', 'volume'], header=None)


def get_hist_data(currency=config.currency):

    df = pd.concat(import_hist_data_csv(currency))

    df['year'] = df['datetime_text'].str[:4].astype(int)
    df['month'] = df['datetime_text'].str[4:6].astype(int)
    df['day'] = df['datetime_text'].str[6:8].astype(int)
    df['hour'] = df['datetime_text'].str[9:11].astype(int)
    df['minute'] = df['datetime_text'].str[11:13].astype(int)

    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
    # covert to warsaw time
    # df['datetime_2'] = df['datetime'].dt.tz_localize('Etc/GMT+5').dt.tz_convert('Europe/Warsaw')
    # df['datetime'] = df['datetime'].dt.tz_localize('US/Eastern').dt.tz_convert('Europe/Warsaw')
    # df['datetime'] = df['datetime'].dt.tz_localize('Etc/GMT+5').dt.tz_convert('Europe/Warsaw')
    # df.loc[df['datetime_3']!=df['datetime_2']]
    # strip time zone so later can be compared with datetime
    df['datetime'] = df['datetime'].dt.tz_localize(None)
    df = df.sort_values(by=['datetime'], ascending=True)
    df = df.drop_duplicates()
    df['datetime'].nunique()
    # this can be a proxy for assessing if simple strategy can be profitable
    # df['datetime'] = df['datetime'] - pd.Timedelta(minutes=15)
    df = df.set_index('datetime')
    df.drop(columns=['year', 'month', 'day', 'hour', 'minute', 'volume', 'datetime_text'], inplace=True)
    if config.freq == '1440min':
        df = df.resample('D').agg({'open': 'first',
                                           'high': 'max',
                                           'low': 'min',
                                           'close': 'last'
                                           })
    else:
        df = df.resample(config.freq).agg({'open': 'first',
                                           'high': 'max',
                                           'low': 'min',
                                           'close': 'last'
                                           })
    return df