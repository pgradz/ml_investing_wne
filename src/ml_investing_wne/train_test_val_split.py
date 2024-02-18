import datetime
import logging
import os
import re

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

import ml_investing_wne.config as config

logger = logging.getLogger(__name__)

# prepare sequences
def split_sequences(sequences_x, sequences_y, n_steps, datetime_series, steps_ahead, name, time_step=None):
    X, y = list(), list()
    i = 0
    jump = 0
    while  i  <  len(sequences_x)+1:
        
        # find the end of this pattern
        end_ix = i + n_steps
        # print('i ', i, datetime_series[i])
        # print('end_ix  ', end_ix,  datetime_series[end_ix])
        if i == 0:
            logger.info('first sequence begins: {}'.format(datetime_series[i]))
            logger.info('first sequence ends: {}'.format(datetime_series[end_ix-1]))
            joblib.dump(datetime_series[end_ix-1], os.path.join(config.package_directory, 'models',
                                           'first_sequence_ends_{}_{}_{}.save'.format(name, config.currency, config.freq)))
        # check if we are beyond the dataset
        if end_ix + steps_ahead > len(sequences_x) + 1:
            logger.info('last sequence begins: {}'.format(datetime_series[i-jump]))
            logger.info('last sequence ends: {}'.format(datetime_series[end_ix-jump-1]))
            joblib.dump(datetime_series[end_ix-jump-1], os.path.join(config.package_directory, 'models',
                                           'last_sequence_ends_{}_{}_{}.save'.format(name, config.currency, config.freq)))
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences_x[i:end_ix,:], sequences_y[end_ix - 1]
        X.append(seq_x)
        y.append(seq_y)
        if isinstance(time_step, pd.Series):
            jump = time_step[i]
        else:
            jump = 1
        i+=jump
    return np.array(X), np.array(y)


def _train_test_val_split(df: pd.DataFrame,train_end, val_end,
                         test_end, seq_len) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    ''' Objective of this function is to split the data into train, test and validation sets in a manner that
    there is no overlap between the sets. It will remove some of the data at the end of train and val sets so
    test set is exactly what is expected and is comparable between different configurations'.cs/
    :param df: dataframe to split
    :return: train, test and validation sets, train date index, val date index, test date index
    '''
    train = df.loc[df.datetime < train_end] 
    # remove last seq_len rows from train
    train = train.iloc[:-seq_len]
    # update train_end
    train_end = train_date_index.iloc[-1]['datetime']
    # validation
    val = df.loc[(df.datetime > train_end) & (df.datetime < val_end)]
    val = val.iloc[:-seq_len]
    # update val_end
    val_end = val.iloc[-1]['datetime']
    # test
    test = df.loc[(df.datetime > val_end) & (df.datetime < test_end)]

    # xxx_date_index is needed to later get back datetime index
    train_date_index = train.reset_index()
    train_date_index = date_index[['index', 'datetime']]

    val_date_index = val.reset_index()
    val_date_index = date_index[['index', 'datetime']]

    test_date_index = test.reset_index()
    test_date_index = date_index[['index', 'datetime']]

    return train, val, test, train_date_index, val_date_index, test_date_index



def train_test_val_split(df, sc_x=None, nb_classes=config.nb_classes, freq=config.freq,
                         seq_len=config.seq_len, seq_stride=config.seq_stride,
                         train_end=config.train_end, val_end=config.val_end,
                         test_end=config.test_end, binarize_target=True, batch_size=config.batch) -> (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    '''
    args:
        df: dataframe to split
        sc_x: scaler to use, if None, StandardScaler will be used. Option of passing scaler is needed for making ad-hoc predictions
        nb_classes: number of classes to predict
        freq: frequency of the data
        seq_len: length of the sequence
        seq_stride: stride of the sequence
        steps_ahead: how many steps ahead to predict
        train_end: end of the train set
        val_end: end of the validation set  
        test_end: end of the test set
        binarize_target: if True, target will be binarized
    '''
    # columns to be dropped from training. y_pred is the target and datetime was carried for technical purposes. index and level_0 are just in case.
    COLUMNS_TO_DROP = ['y_pred', 'datetime', 'index', 'level_0']
     # take care if more than two classes
    if binarize_target:
        if nb_classes == 2:
            df['y_pred'] = [1 if y > 1 else 0 for y in df['y_pred']]
        else:
            df['y_pred'] = pd.qcut(df['y_pred'], nb_classes, labels=range(nb_classes))

    # split train val test
    # move datetime from index to column
    df.reset_index(inplace=True)
    train, val, test, train_date_index, val_date_index, test_date_index = _train_test_val_split(df, train_end=train_end, val_end=val_end, test_end=test_end, seq_len=seq_len)
    train_y = train['y_pred']
    val_y = val['y_pred']
    test_y = test['y_pred']
    # drop columns
    for col in colunns_to_drop:
        try:
            train.drop(columns=[col], inplace=True)
            val.drop(columns=[col], inplace=True)
            test.drop(columns=[col], inplace=True)
        except:
            pass
    # scaler, if not passed in the function, has to be fit on train set, it's easier to do it here
    if not sc_x:
        sc_x = StandardScaler()
        train = sc_x.fit_transform(train)
    else:
        train = sc_x.transform(train)
    val = sc_x.transform(val)
    test = sc_x.transform(test)
    joblib.dump(sc_x, os.path.join(config.package_directory, 'models',
                                   'sc_x_{}_{}.save'.format(config.currency, freq)))
    # create tensorflow datasets
    train_dataset = tf.keras.utils.timeseries_dataset_from_array(data=train.values, targets=train_y.values[sequence_length-1:],
                                                            sequence_length=seq_len, sequence_stride=seq_stride, batch_size=batch_size)                                                            
    train_date_index_dataset = tf.keras.utils.timeseries_dataset_from_array(data=train.values, targets=train_date_index.values[sequence_length-1:, 0].astype(int),
                                                            sequence_length=seq_len, sequence_stride=seq_stride, batch_size=batch_size)
    val_dataset = tf.keras.utils.timeseries_dataset_from_array(data=val.values, targets=val_y.values[sequence_length-1:], 
                                                            sequence_length=seq_len, sequence_stride=seq_stride, batch_size=batch_size)
    val_date_index_dataset = tf.keras.utils.timeseries_dataset_from_array(data=val.values, targets=val_date_index.values[sequence_length-1:, 0].astype(int),
                                                            sequence_length=seq_len, sequence_stride=seq_stride, batch_size=batch_size)
    test_dataset = tf.keras.utils.timeseries_dataset_from_array(data=test.values, targets=test_y.values[sequence_length-1:],
                                                            sequence_length=seq_len, sequence_stride=seq_stride, batch_size=batch_size)
    test_date_index_dataset = tf.keras.utils.timeseries_dataset_from_array(data=test.values, targets=test_date_index.values[sequence_length-1:, 0].astype(int),
                                                            sequence_length=seq_len, sequence_stride=seq_stride, batch_size=batch_size)

    # get back date indices
    train_date_index = get_datetime_indices(train_date_index_dataset, train_date_index)
    val_date_index = get_datetime_indices(val_date_index_dataset, val_date_index)
    test_date_index = get_datetime_indices(test_date_index_dataset, test_date_index)

    return train_dataset, val_dataset, test_dataset, train_date_index, val_date_index, test_date_index

    
def get_datetime_indices(date_index_dataset: tf.data.Dataset, date_index: pd.DataFrame) -> pd.DataFrame:
    '''
    This function is used to get back datetime index from the dataset that has the same shape as original dataframes and 
    can be concatenated with them

    args:
        date_index_dataset: dataset with date indices
        date_index: original date index
    return:
        date_index: date index with the same shape as original (train, val, test) dataframe
    '''
    for i, batch in enumerate(date_index_dataset):
        if i == 0:
            indices = batch[1].numpy()
        else:
            indices =  np.concatenate((indices, batch[1].numpy()), axis=0)

    date_index = date_index.iloc[indices].reset_index(drop=True)

    return date_index


def train_test_val_split_deprecated(df, sc_x=None, nb_classes=config.nb_classes, freq=config.freq,
                         seq_len=config.seq_len, steps_ahead=config.steps_ahead,
                         train_end=config.train_end, val_end=config.val_end,
                         test_end=config.test_end, binarize_target=True, time_step=None):

    colunns_to_drop = ['y_pred', 'datetime', 'index', 'level_0']
    minutes_offset = seq_len * int(re.findall("\d+", freq)[0])
    # take care if more than two classes
    if binarize_target:
        if nb_classes == 2:
            df['y_pred'] = [1 if y > 1 else 0 for y in df['y_pred']]
        else:
            df['y_pred'] = pd.qcut(df['y_pred'], nb_classes, labels=range(nb_classes))

    # split train val test
    df.reset_index(inplace=True)
    train = df.loc[df.datetime < train_end]
    train_datetime = train['datetime']
    train_x = train.copy()
    for col in colunns_to_drop:
        try:
            train_x.drop(columns=[col], inplace=True)
        except:
            pass
    train_y = train['y_pred']
    if time_step:
        train_time_step = train['time_step']
    else:
        train_time_step = None
    # validation
    val = df.loc[(df.datetime > (train_end - datetime.timedelta(minutes=minutes_offset)))
                 & (df.datetime < val_end)]
    val.reset_index(inplace=True)
    val_datetime = val['datetime']
    val_x = val.copy()
    for col in colunns_to_drop:
        try:
            val_x.drop(columns=[col], inplace=True)
        except:
            pass
    val_y = val['y_pred']
    if time_step:
        val_time_step = val['time_step']
    else:
        val_time_step = None
    # test
    test = df.loc[(df.datetime > (val_end - datetime.timedelta(minutes=minutes_offset))) &
                   (df.datetime < test_end)]
    test.reset_index(inplace=True)
    test_datetime = test['datetime']
    test_x = test.copy()
    for col in colunns_to_drop:
        try:
            test_x.drop(columns=[col], inplace=True)
        except:
            pass
    test_y = test['y_pred']
    if time_step:
        test_time_step = test['time_step']
    else:
        test_time_step = None
    # scaler has to be fit on train set, it's easier to do it here
    try:
        train_x = train_x.drop(columns=['time_step'])
        val_x = val_x.drop(columns=['time_step'])
        test_x = test_x.drop(columns=['time_step'])
        logger.info('time step succesfully droped')
    except:
        logger.info('didnt find time step in the dataset')
    if not sc_x:
        sc_x = StandardScaler()
        train_x = sc_x.fit_transform(train_x)
    else:
        train_x = sc_x.transform(train_x)
    val_x = sc_x.transform(val_x)
    test_x = sc_x.transform(test_x)
    joblib.dump(sc_x, os.path.join(config.package_directory, 'models',
                                   'sc_x_{}_{}.save'.format(config.currency, freq)))

    X, y = split_sequences(train_x, train_y, seq_len, train_datetime,
                           steps_ahead=steps_ahead, name='train', time_step=train_time_step)
    X_val, y_val = split_sequences(val_x, val_y, seq_len, val_datetime,
                                   steps_ahead=steps_ahead, name='val', time_step=val_time_step)
    X_test, y_test = split_sequences(test_x, test_y, seq_len, test_datetime,
                                     steps_ahead=steps_ahead, name='test', time_step=test_time_step)

    # You always have to give a 4D array as input to the cnn when using conv2d
    # So input data has a shape of (batch_size, height, width, depth)
    # if using conv2d instead of conv1d then:
    if config.input_dim == '2d':
        X = X.reshape(X.shape + (1,))
        X_val = X_val.reshape(X_val.shape + (1,))
        X_test = X_test.reshape(X_test.shape + (1,))

    y_cat = to_categorical(y)
    y_val_cat = to_categorical(y_val)
    y_test_cat = to_categorical(y_test)

    return X, y, X_val, y_val, X_test, y_test, y_cat, y_val_cat, y_test_cat, train

def test_split(df, seq_len, sc_x):
    '''
    This is split used in xtb module
    :param df:
    :param seq_len:
    :param sc_x:
    :return:
    '''
    # classification ?
    if config.nb_classes == 2:
        df['y_pred'] = [1 if y > 1 else 0 for y in df['y_pred']]
    else:
        df['y_pred'] = pd.qcut(df['y_pred'], config.nb_classes, labels=range(config.nb_classes))

    # split train val test
    test = df.copy()
    test.reset_index(inplace=True)
    test_datetime = test['datetime']
    test_x = test.drop(columns=['y_pred', 'datetime'])
    test_y = test['y_pred']
    test_x = sc_x.transform(test_x)
    #test_x = test_x.to_numpy() - it was to check that last row is kept
    X_test, y_test = split_sequences(test_x, test_y, seq_len, test_datetime,
                                     steps_ahead=config.steps_ahead, name='test_xtb')

    # You always have to give a 4D array as input to the cnn when using conv2d
    # So input data has a shape of (batch_size, height, width, depth)
    # if using conv2d instead of conv1d then:
    if config.input_dim == '2d':
        X_test = X_test.reshape(X_test.shape + (1,))

    y_test_cat = to_categorical(y_test)

    return X_test, y_test, y_test_cat