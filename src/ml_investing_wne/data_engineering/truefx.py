import os
import pandas as pd


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
