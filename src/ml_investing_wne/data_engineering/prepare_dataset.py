import os
import pandas as pd
import numpy as np
import logging
import datetime
import pandas_ta as ta
import ml_investing_wne.config as config
from ml_investing_wne.data_engineering.truefx import import_truefx_csv, aggregate_time_window, check_time_delta

logger = logging.getLogger(__name__)

def prepare_processed_dataset(plot=False, df=None, allow_null=False):

    if not isinstance(df, pd.DataFrame):
        df = pd.concat(import_truefx_csv(config.raw_data_path, config.currency))
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
    #df.ta.indicators()

    MA = [3, 5, 10, 13, 20]
    for ma in MA:
        df.ta.sma(length=ma, append=True)
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
    df.ta.xsignals(append=True) # does nothing
    # ta.roc doesn't work as expected
    # df.ta.roc(append=True)
    # df.ta.roc(length=12, append=True)
    # df.ta.roc(length=1, append=True)
    df['y_pred'] = df['close'].shift(-1) / df['close']
    df['roc_1'] = df['y_pred'].shift(1)
    # df['y_roc_12'] = df['close'].shift(-12) / df['close']
    # df['y_roc_12'] = df['y_roc_12'].shift(12)
    df['datetime'] = df.index
    df['hour'] = df.datetime.dt.hour
    df['weekday'] = df.datetime.dt.weekday
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / df['hour'].max())
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / df['hour'].max())
    df['weekday_sin'] = np.sin(2 * np.pi * df['hour'] / df['weekday'].max())
    df['weekday_cos'] = np.cos(2 * np.pi * df['hour'] / df['weekday'].max())
    # check of encoding
    # df.plot.scatter('hour_sin', 'hour_cos')
    # df.plot.scatter('weekday_sin', 'weekday_cos')
    df.drop(columns=['datetime'], axis=1, inplace=True)
    if not allow_null:
        df.dropna(inplace=True)
    output_directory = os.path.join(config.processed_data_path, config.currency)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    path = os.path.join(output_directory, '{}_processed_{}.csv'.format(config.currency, config.freq))
    df.to_csv(path)
    logger.info('exported to {}'.format(path))

    return df

