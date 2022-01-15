import os
import pandas as pd
import logging
import datetime
import joblib
import mlflow.keras
from tensorflow.keras.models import load_model
from ml_investing_wne.xtb.xAPIConnector import APIClient, APIStreamClient, loginCommand
from ml_investing_wne.data_engineering.prepare_dataset import prepare_processed_dataset
import ml_investing_wne.config as config
from ml_investing_wne.train_test_val_split import test_split
from ml_investing_wne.helper import confusion_matrix_plot, compute_profitability_classes, check_hours


logger = logging.getLogger()
logger.setLevel(logging.INFO)
sc_x = joblib.load(os.path.join(config.package_directory, 'models',
                                   'sc_x_{}_{}.save'.format(config.currency, config.freq)))
# hist model
model = load_model(os.path.join(config.package_directory, 'models',
                                    '{}_hist_data_{}_{}'.format(config.model, config.currency, config.freq)))
# old model
# model = load_model(os.path.join(config.package_directory, 'models',
#                                     '{}_{}_{}.h5'.format(config.model, config.currency, config.freq)))


start = datetime.datetime(2021, 6, 27, 1, 0, 0, 0)
userId = 12896600
password = "xoh10026"
symbol = 'EURGBP'

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

# For JPY
# df = pd.DataFrame(resp['returnData']['rateInfos'])
# df['datetime'] = pd.to_datetime(df['ctm'], unit='ms')
# df['close'] = (df['open'] + df['close'])/1000
# df['high'] = (df['open'] + df['high'])/1000
# df['low'] = (df['open'] + df['low'])/1000
# df['open'] = df['open']/1000

# if hist model
df['datetime'] = df['datetime'].dt.tz_localize('GMT').dt.tz_convert('US/Eastern').dt.tz_localize(None)
df = df.sort_values(by=['datetime'], ascending=True)
df = df.set_index('datetime')

df.drop(columns=['ctm', 'ctmString', 'vol'], inplace=True)
# order in hist data - open, high, low, close
df = df[['open', 'high', 'low', 'close']]

df = df.resample(config.freq).agg({'open': 'first',
                                               'high': 'max',
                                               'low': 'min',
                                               'close': 'last'
                                               })

df = prepare_processed_dataset(df=df)
X_test, y_test, y_test_cat = test_split(df, config.seq_len, sc_x)

test_loss, test_acc = model.evaluate(X_test, y_test_cat)
logger.info('Test accuracy : {}'.format(test_acc))
logger.info('Test loss : {}'.format(test_loss))

prediction = df.copy()
prediction.reset_index(inplace=True)
df['y_pred'] = df['close'].shift(-1) / df['close'] - 1
# new_start = config.val_end + config.seq_len * datetime.timedelta(minutes=int(''.join(filter(str.isdigit, config.freq))))
prediction = df.iloc[95:]
y_pred = model.predict(X_test)
prediction['trade'] = y_pred.argmax(axis=1)
prediction.reset_index(inplace=True)
prediction['y_prob'] = y_pred[:, 1]

correct_predictions = prediction.loc[(prediction['y_pred'] > 0) & (prediction['y_prob'] > 0.5) ].shape[0]
correct_predictions =  correct_predictions + prediction.loc[(prediction['y_pred'] < 0) & (prediction['y_prob'] < 0.5)].shape[0]
correct_predictions / prediction.shape[0]

df['cost'] = (config.pips/10000)/df['close']
df.reset_index(inplace=True)
# prediction[['datetime','y_prob','close']].to_csv('xtb_test_3.csv', decimal=',', sep=';')
start_date = joblib.load(os.path.join(config.package_directory, 'models',
                                           'first_sequence_ends_{}_{}_{}.save'.format('test_xtb', config.currency, config.freq)))
end_date = joblib.load(os.path.join(config.package_directory, 'models',
                                           'last_sequence_ends_{}_{}_{}.save'.format('test_xtb', config.currency, config.freq)))

mlflow.set_experiment(experiment_name=symbol + '_xtb_last_period_' + config.model + '_' + str(config.nb_classes))
lower_bounds =[0.1,0.15,0.2,0.25, 0.3,0.35, 0.4, 0.45, 0.5]
upper_bounds = [1 - lower for lower in lower_bounds]
mlflow.set_tag('frequency', config.freq)

for lower_bound, upper_bound in zip(lower_bounds, upper_bounds):
    portfolio_result, hit_ratio, time_active = compute_profitability_classes(df, y_pred, start_date, end_date, lower_bound, upper_bound)
    mlflow.log_metric("portfolio_result_{}_{}".format(lower_bound, upper_bound), portfolio_result)
    mlflow.log_metric("hit_ratio_{}_{}".format(lower_bound, upper_bound), hit_ratio)
    mlflow.log_metric("time_active_{}_{}".format(lower_bound, upper_bound), time_active)
    mlflow.log_artifact(os.path.join(config.package_directory, 'models',
                                     'portfolio_evolution_{}_{}_{}_{}_{}.png'.
                                     format(config.model, config.currency, config.nb_classes,
                                            lower_bound, upper_bound)))

check_hours(df, y_pred, start_date, end_date, lower_bound=0.45, upper_bound=0.55)

for lower_bound, upper_bound in zip(lower_bounds, upper_bounds):
    portfolio_result, hit_ratio, time_active = compute_profitability_classes(df, y_pred, start_date, end_date, lower_bound, upper_bound,
                                                                             time_waw_list=[
                                                                                 datetime.time(20, 0, 0),
                                                                                            datetime.time(22,0,0)])

for lower_bound, upper_bound in zip(lower_bounds, upper_bounds):
    portfolio_result, hit_ratio, time_active = compute_profitability_classes(df, y_pred, start_date, end_date, lower_bound, upper_bound,
                                                                             time_waw_list=[datetime.time(20,0,0)])




#### CHECK HOURS
import numpy as np
lower_bound = 0.45
upper_bound = 0.55
df['y_pred'] = df['close'].shift(-1) / df['close'] - 1
# new_start = config.val_end + config.seq_len * datetime.timedelta(minutes=int(''.join(filter(str.isdigit, config.freq))))
prediction = df.loc[(df.datetime >= start_date) & (df.datetime <= end_date)]
prediction['datetime_waw'] = prediction['datetime'].dt.tz_localize('US/Eastern').dt.tz_convert(
    'Europe/Warsaw').dt.tz_localize(None)
prediction['hour_waw'] = prediction['datetime_waw'].dt.time
# prediction['trade'] = y_pred.argmax(axis=1)
prediction['prediction'] = y_pred[:, 1]
conditions = [
    (prediction['prediction'] <= lower_bound),
    (prediction['prediction'] > lower_bound) & (prediction['prediction'] <= upper_bound),
    (prediction['prediction'] > upper_bound)
]
values = [0, 0.5, 1]
prediction['trade'] = np.select(conditions, values)
time_waw_list = [datetime.time(20,0,0), datetime.time(22,0,0)]
if time_waw_list:
    prediction.loc[~prediction['hour_waw'].isin(time_waw_list), 'trade'] = 0.5

prediction['y_pred_abs'] = np.abs(prediction['y_pred'])
prediction['difference'] = prediction['y_pred_abs'] - prediction['cost']
prediction.loc[prediction['trade']!=0.5, 'difference'].describe(percentiles=[0.1, 0.2,0.25,0.33,0.4,0.5,0.6,0.66,0.75, 0.8, 0.9])