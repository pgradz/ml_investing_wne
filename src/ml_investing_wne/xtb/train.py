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

logger = logging.getLogger()
sc_x = joblib.load(os.path.join(config.package_directory, 'models',
                                   'sc_x_{}_{}.save'.format(config.currency, config.freq)))
model = load_model(os.path.join(config.package_directory, 'models',
                                    '{}_{}_{}.hdf5'.format(config.model, config.currency, config.freq)))

start = datetime.datetime(2021, 1, 1, 1, 0, 0, 0)
userId = 12896600
password = "xoh10026"
symbol = 'EURPLN'

client = APIClient()

# connect to RR socket, login
loginResponse = client.execute(loginCommand(userId=userId, password=password))
logger.info(str(loginResponse))

# check if user logged in correctly
if (loginResponse['status'] == False):
    print('Login failed. Error code: {0}'.format(loginResponse['errorCode']))

# get ssId from login response
ssid = loginResponse['streamSessionId']

resp = client.commandExecute('getChartLastRequest', {'info': {"period": 60, "start": int(start.timestamp() * 1000),
                                                              "symbol": symbol}})

df = pd.DataFrame(resp['returnData']['rateInfos'])
df['datetime'] = pd.to_datetime(df['ctm'], unit='ms')
df['close'] = (df['open'] + df['close'])/100000
df['high'] = (df['open'] + df['high'])/100000
df['low'] = (df['open'] + df['low'])/100000
df['open'] = df['open']/100000
df = df.set_index('datetime')
df.drop(columns=['ctm', 'ctmString', 'vol'], inplace=True)
df = prepare_processed_dataset(df=df)
X, y, X_val, y_val, X_test, y_test, y_cat, y_val_cat, y_test_cat, train = train_test_val_split(df, config.seq_len)


mlflow.set_experiment(experiment_name=symbol + '_' + str(config.nb_classes))
early_stop = EarlyStopping(monitor='val_accuracy', patience=10)
model_path_final = os.path.join(config.package_directory, 'models',
                               '{}_{}_{}.h5'.format(model, symbol, config.freq))
model_checkpoint = ModelCheckpoint(filepath=model_path_final, monitor='val_accuracy', verbose=1, save_best_only=True)
csv_logger = CSVLogger(os.path.join(config.package_directory, 'logs', 'keras_log.csv'), append=True, separator=';')
callbacks = [early_stop, model_checkpoint, csv_logger]

# continue training or start new model
model = build_model(input_shape=(X.shape[1], X.shape[2]), nb_classes=config.nb_classes)

history = model.fit(X, y_cat, batch_size=64, epochs=config.epochs, verbose=2,
                    validation_data=(X_val, y_val_cat), callbacks=callbacks)