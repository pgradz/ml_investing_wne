import os
import pandas as pd
import logging
import datetime
import joblib
from tensorflow.keras.models import load_model
from ml_investing_wne.xtb.xAPIConnector import APIClient, APIStreamClient, loginCommand
from ml_investing_wne.data_engineering.prepare_dataset import prepare_processed_dataset
import ml_investing_wne.config as config
from ml_investing_wne.train_test_val_split import test_split
from ml_investing_wne.helper import confusion_matrix_plot, compute_profitability_classes


logger = logging.getLogger()
sc_x = joblib.load(os.path.join(config.package_directory, 'models',
                                   'sc_x_{}_{}.save'.format(config.currency, config.freq)))
model = load_model(os.path.join(config.package_directory, 'models',
                                    '{}_{}_{}.hdf5'.format(config.model, config.currency, config.freq)))

start = datetime.datetime(2021, 10, 1, 1, 0, 0, 0)
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
pips = 25
df['cost'] = (pips/10000)/df['close']
df.reset_index(inplace=True)
portfolio_result = compute_profitability_classes(df.iloc[95:], y_pred, datetime.datetime(2021, 10, 4, 9, 0, 0),
                                                 datetime.datetime(2021, 12, 28, 16, 0, 0))








