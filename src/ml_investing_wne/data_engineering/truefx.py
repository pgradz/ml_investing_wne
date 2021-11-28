import os
import pandas as pd
import logging

import ml_investing_wne.config as config

# logger is a child of a logger from main module that is calling this module
logger = logging.getLogger(__name__)

def import_truefx_csv(raw_data_path, currency, names=['currency', 'datetime', 'bid', 'ask'], **kwargs):
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
                                                   'bid_close': 'last',
                                                   'no_of_ticks': 'size'})
    df_agg_ask = df_agg['ask'].resample(freq).agg({'ask_open': 'first',
                                                   'ask_high': 'max',
                                                   'ask_min': 'min',
                                                   'ask_close': 'last'})
    df_agg_spread = df_agg['spread'].resample(freq).agg({'spread_open': 'first',
                                                         'spread_high': 'max',
                                                         'spread_min': 'min',
                                                         'spread_close': 'last'})

    df_all = pd.merge(df_agg_bid, df_agg_ask, how='inner', left_index=True, right_index=True)
    df_all = pd.merge(df_all, df_agg_spread, how='inner', left_index=True, right_index=True)
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
