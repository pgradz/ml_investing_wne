import os
import pandas as pd
import ml_investing_wne.config as config


def import_truefx_csv(raw_data_path, currencies, names=['currency', 'datetime', 'bid', 'ask'], **kwargs):
    '''
    read all csv files for currency pairs defined in config
    :param raw_data_path: str path to raw data folder
    :param currencies: list of currencies
    :param names: list of headers
    :param kwargs:
    :return: generator of pandas dataframes
    '''
    for currency in currencies:
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

    currencies = df['currency'].unique()
    for currency in currencies:
        output_directory = os.path.join(config.processed_data_path, currency.replace('/',''))
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        df_temp = df.loc[df['currency'] == currency]
        df_agg = df.set_index('datetime')
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
        df_all.to_csv(os.path.join(output_directory, '{}_processed_{}.csv'.format(currency.replace('/', ''), freq)))
