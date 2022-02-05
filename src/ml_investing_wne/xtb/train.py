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
from ml_investing_wne.helper import confusion_matrix_plot, compute_profitability_classes, check_hours
import importlib

build_model = getattr(importlib.import_module('ml_investing_wne.cnn.{}'.format(config.model)), 'build_model')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

start = datetime.datetime(2019, 7, 1, 1, 0, 0, 0)

symbol = 'W20'
# config.train_end = datetime.datetime(2021, 9, 1, 0, 0, 0)
# config.val_end = datetime.datetime(2021, 11, 1, 0, 0, 0)
# config.test_end = datetime.datetime(2021, 12, 31, 15, 0, 0)
#
# sc_x = joblib.load(os.path.join(config.package_directory, 'models',
#                                    'sc_x_{}_{}.save'.format(config.currency, config.freq)))
client = APIClient()

# connect to RR socket, login
loginResponse = client.execute(loginCommand(userId=config.userId, password=config.password))
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

#EURPLN
# df['datetime'] = pd.to_datetime(df['ctm'], unit='ms')
# df['close'] = (df['open'] + df['close'])/10000
# df['high'] = (df['open'] + df['high'])/10000
# df['low'] = (df['open'] + df['low'])/10000
# df['open'] = df['open']/10000

df['datetime'] = df['datetime'].dt.tz_localize('GMT').dt.tz_convert('US/Eastern').dt.tz_localize(None)
df = df.set_index('datetime')
df.drop(columns=['ctm', 'ctmString', 'vol'], inplace=True)
df = df[['open', 'high', 'low', 'close']]

df = df.resample(config.freq).agg({'open': 'first',
                                               'high': 'max',
                                               'low': 'min',
                                               'close': 'last'
                                               })

df = prepare_processed_dataset(df=df)
#X, y, X_val, y_val, X_test, y_test, y_cat, y_val_cat, y_test_cat, train = train_test_val_split(df, config.seq_len, sc_x)
X, y, X_val, y_val, X_test, y_test, y_cat, y_val_cat, y_test_cat, train = train_test_val_split(df, config.seq_len)


mlflow.set_experiment(experiment_name=symbol + '_xtb_retrain_' + config.model + '_' + str(config.nb_classes))
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
model_path_final = os.path.join(config.package_directory, 'models',
                               '{}_{}_xtb_retrain_{}.h5'.format(config.model, symbol, config.freq))
model_checkpoint = ModelCheckpoint(filepath=model_path_final, monitor='val_accuracy', verbose=1, save_best_only=True)
csv_logger = CSVLogger(os.path.join(config.package_directory, 'logs', 'keras_log.csv'), append=True, separator=';')
callbacks = [early_stop, model_checkpoint, csv_logger]

if len(config.currency) > 1:
    model = load_model(os.path.join(config.package_directory, 'models',
                                    '{}_{}_{}_{}'.format(config.model, 'hist_data', symbol, config.freq)))
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


check_hours(df, y_pred, start_date, end_date, lower_bound=0.35, upper_bound=0.65)


for lower_bound, upper_bound in zip(lower_bounds, upper_bounds):
    portfolio_result, hit_ratio, time_active = compute_profitability_classes(df, y_pred, start_date, end_date, lower_bound, upper_bound,
                                                                             time_waw_list=[
                                                                                 datetime.time(20, 0, 0),
                                                                                            datetime.time(22,0,0)])


model.save(os.path.join(config.package_directory, 'models',
                                '{}_{}_{}_{}'.format(config.model, 'hist_data', config.currency, config.freq)))
