import os
import pandas as pd
import numpy as np
import logging
import datetime
import pandas_ta as ta
import ml_investing_wne.config as config
from ml_investing_wne.data_engineering.load_data import import_forex_csv, aggregate_time_window, \
    check_time_delta

logger = logging.getLogger(__name__)

def prepare_processed_dataset(plot=False, df=None, allow_null=False, features=True, add_target=True):
    '''

    :param plot:  flag whether to plot price evolution
    :param df: if no df passed, data will be loaded from folder passed in config
    :param allow_null: drop rows with nulls, effect of computing TA indictors (first obs)
    :param features: flag whether to add TA features
    :return: pandas dataframe
    '''

    if not isinstance(df, pd.DataFrame):
        df = pd.concat(import_forex_csv(config.raw_data_path, config.currency))
        df = aggregate_time_window(df, config.freq)
        check_time_delta(df)
        # pick bid for technical indicators
        df.rename(columns={'bid_open': 'open', 'bid_close': 'close', 'bid_high': 'high', 'bid_min': 'low'}, inplace=True)

    if plot:
        check = df.copy()
        check['close'] = check['close'].fillna(value=0)
        ax = check['close'].plot()
        ax.set_ylabel('close')
        ax.set_title(config.currency)

    # eliminate weekends
    df = df.loc[df['close'].notna()].copy()
    if add_target:
        df['y_pred'] = df['close'].shift(-config.steps_ahead) / df['close']

    # lag of target - depending on the setup, it can leak information from the future
    # df['y_pred_lag'] = df['y_pred'].shift(1)
    #df.ta.indicators()
    if features:
        MA = [5, 10, 15, 20, 50]
        for ma in MA:
            # df.ta.sma(length=ma, append=True)
            df.ta.ema(length=ma, append=True)
            df.ta.variance(length=ma, append=True)

        df.ta.macd(append=True) # adds 3 columns
        df.ta.rsi(append=True) # adds 1 column
        df.ta.rsi(append=True, length=10)
        df.ta.rsi(append=True, length=6)
        #df.ta.rsi(length=10, append=True)
        #df.ta.rsi(length=20, append=True)
        df.ta.stoch(append=True) # adds 2 columns
        df.ta.willr(append=True) # adds 1 column
        df.ta.bbands(append=True) # adds 5 columns

        df['roc_1'] = df['close'].shift(-1) / df['close'] - 1
        df['roc_1'] = df['roc_1'].shift(1)
        # df['roc_2'] = df['close'].shift(-2) / df['close'] - 1
        # df['roc_2'] = df['roc_2'].shift(2)
        # df['roc_3'] = df['close'].shift(-3) / df['close'] - 1
        # df['roc_3'] = df['roc_3'].shift(3)
        # df['roc_4'] = df['close'].shift(-4) / df['close'] - 1
        # df['roc_4'] = df['roc_4'].shift(4)
        # df['roc_5'] = df['close'].shift(-5) / df['close'] - 1
        # df['roc_5'] = df['roc_5'].shift(5)
        # df['roc_10'] = df['close'].shift(-10) / df['close'] - 1
        # df['roc_10'] = df['roc_10'].shift(10)
        # volume indicators
        # df.ta.atr(append=True) # adds 1 column
        # df.ta.obv(append=True) # adds 1 column
        df.ta.cmf(append=True)
        df.ta.mfi(append=True) 

        df['datetime'] = df.index
        if df['datetime'].dtype == 'object':
            df['datetime'] = pd.to_datetime(df['datetime'])
        df['hour'] = df.datetime.dt.hour
        df['weekday'] = df.datetime.dt.weekday
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / df['hour'].max())
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / df['hour'].max())
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / df['weekday'].max())
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / df['weekday'].max())
    # check of encoding
    # df.plot.scatter('hour_sin', 'hour_cos')
    # df.plot.scatter('weekday_sin', 'weekday_cos')
        df.drop(columns=['datetime', 'hour', 'weekday'], axis=1, inplace=True)
        # this will be true for 24h time aggregated method
        if df['hour_sin'].isnull().all():
            df.drop(columns=['hour_sin', 'hour_cos'], axis=1, inplace=True)

    if not allow_null:
        df.dropna(inplace=True)

    # Convert all object columns to float
    for column in df.select_dtypes(include='object'):
        df[column] = df[column].astype('float')

    output_directory = os.path.join(config.processed_data_path, config.currency)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    path = os.path.join(output_directory, '{}_processed_{}.csv'.format(config.currency, config.freq))
    df.to_csv(path)
    logger.info('exported to {}'.format(path))

    return df
