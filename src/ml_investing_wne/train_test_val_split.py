import os
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
from tensorflow.keras.utils import to_categorical
import ml_investing_wne.config as config


logger = logging.getLogger(__name__)

# prepare sequences
def split_sequences(sequences_x, sequences_y, n_steps, datetime_series, steps_ahead, name):
    X, y = list(), list()
    for i in range(len(sequences_x)):
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
        if end_ix  +  steps_ahead> len(sequences_x):
            logger.info('last sequence begins: {}'.format(datetime_series[i-2]))
            logger.info('last sequence ends: {}'.format(datetime_series[end_ix-2]))
            joblib.dump(datetime_series[end_ix-2], os.path.join(config.package_directory, 'models',
                                           'last_sequence_ends_{}_{}_{}.save'.format(name, config.currency, config.freq)))
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences_x[i:end_ix,:], sequences_y[end_ix - 1 + steps_ahead]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def train_test_val_split(df, seq_len, sc_x=None):
    # classification ?
    if config.nb_classes == 2:
        df['y_pred'] = [1 if y > 1 else 0 for y in df['y_pred']]
    else:
        df['y_pred'] = pd.qcut(df['y_pred'], config.nb_classes, labels=range(config.nb_classes))

    # split train val test
    df.reset_index(inplace=True)
    train = df.loc[df.datetime < config.train_end]
    train_datetime = train['datetime']
    train_x = train.drop(columns=['y_pred', 'datetime'])
    train_y = train['y_pred']
    val = df.loc[(df.datetime >= config.train_end) & (df.datetime < config.val_end)]
    val.reset_index(inplace=True)
    val_datetime = val['datetime']
    val_x = val.drop(columns=['y_pred', 'datetime', 'index'])
    val_y = val['y_pred']
    test = df.loc[(df.datetime > config.val_end) & (df.datetime < config.test_end)]
    test.reset_index(inplace=True)
    test_datetime = test['datetime']
    test_x = test.drop(columns=['y_pred', 'datetime', 'index'])
    test_y = test['y_pred']
    if not sc_x:
        sc_x = StandardScaler()
        train_x = sc_x.fit_transform(train_x)
    else:
        train_x = sc_x.transform(train_x)
    val_x = sc_x.transform(val_x)
    test_x = sc_x.transform(test_x)
    joblib.dump(sc_x, os.path.join(config.package_directory, 'models',
                                   'sc_x_{}_{}.save'.format(config.currency, config.freq)))

    X, y = split_sequences(train_x, train_y, seq_len, train_datetime, steps_ahead=config.steps_ahead, name='train')
    X_val, y_val = split_sequences(val_x, val_y, seq_len, val_datetime, steps_ahead=config.steps_ahead, name='val')
    X_test, y_test = split_sequences(test_x, test_y, seq_len, test_datetime, steps_ahead=config.steps_ahead, name='test')

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
    X_test, y_test = split_sequences(test_x, test_y, seq_len, test_datetime, steps_ahead=config.steps_ahead, name='test_xtb')

    # You always have to give a 4D array as input to the cnn when using conv2d
    # So input data has a shape of (batch_size, height, width, depth)
    # if using conv2d instead of conv1d then:
    if config.input_dim == '2d':
        X_test = X_test.reshape(X_test.shape + (1,))

    y_test_cat = to_categorical(y_test)

    return X_test, y_test, y_test_cat