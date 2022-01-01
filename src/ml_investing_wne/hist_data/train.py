import os
import pandas as pd
import logging
import datetime
import joblib
import mlflow.keras
from tensorflow.keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from ml_investing_wne.xtb.xAPIConnector import APIClient, APIStreamClient, loginCommand
from ml_investing_wne.data_engineering.prepare_dataset import prepare_processed_dataset
import ml_investing_wne.config as config
from ml_investing_wne.train_test_val_split import train_test_val_split
from ml_investing_wne.helper import confusion_matrix_plot, compute_profitability_classes
import importlib

build_model = getattr(importlib.import_module('ml_investing_wne.cnn.{}'.format(config.model)), 'build_model')

# logger settings
logger = logging.getLogger()
# You can set a different logging level for each logging handler but it seems you will have to set the
# logger's level to the "lowest".
logger.setLevel(logging.INFO)
stream_h = logging.StreamHandler()
file_h = logging.FileHandler(os.path.join(config.package_directory, 'logs', 'app.log'))
stream_h.setLevel(logging.INFO)
file_h.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
stream_h.setFormatter(formatter)
file_h.setFormatter(formatter)
logger.addHandler(stream_h)
logger.addHandler(file_h)

mlflow.tensorflow.autolog()
# sc_x = joblib.load(os.path.join(config.package_directory, 'models',
#                                    'sc_x_{}_{}.save'.format(config.currency, config.freq)))
# model = load_model(os.path.join(config.package_directory, 'models',
#                                     '{}_{}_{}.hdf5'.format(config.model, config.currency, config.freq)))


def import_hist_data_csv(raw_data_path, currency, names=['currency', 'datetime', 'bid', 'ask'], **kwargs):
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

data_path = '/Users/i0495036/Documents/sandbox/ml_investing_wne/ml_investing_wne/src/ml_investing_wne/data/raw/hist_data'
df = pd.concat(import_hist_data_csv(data_path, config.currency))

df['year']=df['datetime_text'].str[:4].astype(int)
df['month']=df['datetime_text'].str[4:6].astype(int)
df['day']=df['datetime_text'].str[6:8].astype(int)
df['hour']=df['datetime_text'].str[9:11].astype(int)
df['minute']=df['datetime_text'].str[11:13].astype(int)

df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
# covert to warsaw time
df['datetime'] = df['datetime'].dt.tz_localize('Etc/GMT+5').dt.tz_convert('Europe/Warsaw')
# strip time zone so later can be compared with datetime
df['datetime'] = df['datetime'].dt.tz_localize(None)
df = df.sort_values(by=['datetime'], ascending=True)
df = df.drop_duplicates()
df['datetime'].nunique()
df = df.set_index('datetime')
df.drop(columns=['year', 'month', 'day', 'hour', 'minute', 'volume', 'datetime_text' ], inplace=True)

df = df.resample(config.freq).agg({'open': 'first',
                                               'high': 'max',
                                               'low': 'min',
                                               'close': 'last'
                                               })

df = prepare_processed_dataset(df=df)

X, y, X_val, y_val, X_test, y_test, y_cat, y_val_cat, y_test_cat, train = train_test_val_split(df, config.seq_len)

mlflow.set_experiment(experiment_name='hist_data' + '_' + config.model + '_' + str(config.nb_classes))
early_stop = EarlyStopping(monitor='val_accuracy', patience=5)
model_path_final = os.path.join(config.package_directory, 'models',
                               '{}_{}_{}_{}.h5'.format(config.model, 'hist_data', config.currency, config.freq))
model_checkpoint = ModelCheckpoint(filepath=model_path_final, monitor='val_accuracy', verbose=1, save_best_only=True)
csv_logger = CSVLogger(os.path.join(config.package_directory, 'logs', 'keras_log.csv'), append=True, separator=';')
callbacks = [early_stop, model_checkpoint, csv_logger]

# continue training or start new model
if len(config.load_model) > 1:
    model = load_model(os.path.join(config.package_directory, 'models',
                                    '{}_{}_{}_{}.h5'.format(config.model, 'hist_data', config.load_model, config.freq)))
else:
    model = build_model(input_shape=(X.shape[1], X.shape[2]), nb_classes=config.nb_classes)

history = model.fit(X, y_cat, batch_size=64, epochs=config.epochs, verbose=2,
                    validation_data=(X_val, y_val_cat), callbacks=callbacks)

test_loss, test_acc = model.evaluate(X_test, y_test_cat)
logger.info('Test accuracy : {}'.format(test_acc))
logger.info('Test loss : {}'.format(test_loss))
mlflow.log_metric("test_acc", test_acc)
mlflow.log_metric("test_loss", test_loss)
mlflow.log_metric("test_loss", test_loss)
mlflow.set_tag('currency', config.currency)
mlflow.set_tag('frequency', config.freq)
mlflow.set_tag('steps_ahead', config.steps_ahead)
mlflow.log_metric('y_distribution', y.mean())
mlflow.log_metric('y_val_distribution', y_val.mean())
mlflow.log_metric('y_test_distribution', y_test.mean())
mlflow.log_metric('cost', config.pips)

y_pred = model.predict(X_test)

df['cost'] = (config.pips/10000)/df['close']
start_date = joblib.load(os.path.join(config.package_directory, 'models',
                                           'first_sequence_ends_{}_{}_{}.save'.format('test', config.currency, config.freq)))
end_date = joblib.load(os.path.join(config.package_directory, 'models',
                                           'last_sequence_ends_{}_{}_{}.save'.format('test', config.currency, config.freq)))
lower_bounds =[0.1,0.15,0.2,0.25, 0.3,0.35, 0.4, 0.45, 0.5]
upper_bounds = [1 - lower for lower in lower_bounds]

for lower_bound, upper_bound in zip(lower_bounds, upper_bounds):
    portfolio_result, hit_ratio, time_active = compute_profitability_classes(df, y_pred, start_date, end_date, lower_bound, upper_bound)
    mlflow.log_metric("portfolio_result_{}_{}".format(lower_bound, upper_bound), portfolio_result)
    mlflow.log_metric("hit_ratio_{}_{}".format(lower_bound, upper_bound), hit_ratio)
    mlflow.log_metric("time_active_{}_{}".format(lower_bound, upper_bound), time_active)
    mlflow.log_artifact(os.path.join(config.package_directory, 'models',
                                     'portfolio_evolution_{}_{}_{}_{}_{}.png'.
                                     format(config.model, config.currency, config.nb_classes,
                                            lower_bound, upper_bound)))

mlflow.log_artifact(os.path.join(config.package_directory, 'models', 'cut_off_analysis_{}_{}_{}.csv'.
                                 format(config.model, config.currency, config.nb_classes)))


for lower_bound, upper_bound in zip(lower_bounds, upper_bounds):
    portfolio_result, hit_ratio, time_active = compute_profitability_classes(df, y_pred[-1318:], datetime.datetime(2021,10,8,10,0,0), end_date, lower_bound, upper_bound)
