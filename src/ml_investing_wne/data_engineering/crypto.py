import logging
import os

import pandas as pd

import ml_investing_wne.config as config

# logger is a child of a logger from main module that is calling this module
logger = logging.getLogger(__name__)

df = pd.read_csv(os.path.join(config.raw_data_path, 'crypto', config.currency + '.csv'))
df.index = pd.to_datetime(df['t'], unit='ms')
freq = config.freq
import datetime

temp = df.loc[df.index<datetime.datetime(2019,8,1,0,0,0,0)]
temp = df.loc[df.index>datetime.datetime(2021,1,1,0,0,0,0)]
df.d.unique()
df_agg_close = df['p'].resample(freq).agg({'open': 'first',
                                               'high': 'max',
                                               'low': 'min',
                                               'close': 'last'})

df_agg_volume = df['q'].resample(freq).agg({'volume': 'sum',
                                                     'volume_avg': 'mean'})

df_all = pd.merge(df_agg_close, df_agg_volume, how='inner', left_index=True, right_index=True)
df_all['currency'] = currency

check = df_agg_close.copy()
check['close'] = check['close'].fillna(value=0)
ax = check['close'].plot()
ax.set_ylabel('close')
ax.set_title('Binance ETHUSDT')


def import_binance_csv(raw_data_path, currency, **kwargs):
    '''
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

