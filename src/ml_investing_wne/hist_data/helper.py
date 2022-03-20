import os
import pandas as pd
import ml_investing_wne.config as config

data_path = '/Users/i0495036/Documents/sandbox/ml_investing_wne/ml_investing_wne/src/ml_investing_wne/data/raw/hist_data'


def import_hist_data_csv(currency, raw_data_path = data_path, names=['currency', 'datetime', 'bid', 'ask'], **kwargs):
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
        yield pd.read_csv(file, sep=';', names=['datetime_text', 'open', 'high', 'low', 'close', 'volume'], header=None)


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
    # this is proxy for assessing if simple strategy can be profitable
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
