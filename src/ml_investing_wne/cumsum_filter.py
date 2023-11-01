import random
import os
import pandas as pd
import numpy as np
import tensorflow as tf

from ml_investing_wne import config
from ml_investing_wne.utils import get_logger
from ml_investing_wne.experiment_factory import create_asset, experiment_factory

random.seed(config.seed)
np.random.seed(config.seed)
tf.random.set_seed(config.seed)

logger = get_logger()
EXCHANGE = 'binance'
threshold = 0.02
output_path='/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/processed'
output_path = os.path.join(output_path, f'{EXCHANGE}_{config.currency}')
range_bar = True


def cumsum_filter(df,h,range_bar=False):
    
    # make index numeric
    df.reset_index(inplace=True)
    # measure progress
    df_size = df.shape[0]
    prc_10, prc25, prc50, prc75 =  int(0.1*df_size), int(0.25*df_size), int(0.5*df_size), int(0.75*df_size)


    group_id,s_pos,s_neg = 0,0,0
    df['prc_change'] =  df['close']/ df['close'].shift(1) - 1
    # get column index of prc_change
    prc_change_index = df.columns.get_loc('prc_change')
    df['group_id'] = 0
    logger.info('Starting cumsum filter')
    for i in df.index[1:]:
        if range_bar:
            s_pos,s_neg=s_pos+df.iloc[i, prc_change_index], s_neg+df.iloc[i, prc_change_index]
        else:
            s_pos,s_neg=max(0,s_pos+df.iloc[i, prc_change_index]),min(0,s_neg+df.iloc[i, prc_change_index])
        df.loc[i, 'group_id'] = group_id
        if s_neg<-h or s_pos>h:
            group_id += 1
            s_pos,s_neg = 0,0
        if i in [prc_10, prc25, prc50, prc75]:
            logger.info(f'Progress: {i/df_size}')

    df_agg = df.groupby('group_id').agg({ 'datetime': 'last',
                                    'open': 'first',
                                    'high': 'max',
                                    'low': 'min',
                                    'close': 'last',
                                    'volume': 'sum'})

    if range_bar:
        file_name = f'range_{threshold}.csv'
    else:
        file_name = f'cumsum_{threshold}.csv'

    return df_agg, file_name

if __name__ == "__main__":

    # set the config for time bars and 1 min bars
    asset = create_asset()
    experiment = experiment_factory(asset).get_experiment()
    df = experiment.df[['open', 'high', 'low', 'close', 'volume']]
    df2, file_name = cumsum_filter(df, threshold, range_bar=range_bar)
    output_path = os.path.join(output_path, file_name)
    df2.to_csv(output_path, index=False)