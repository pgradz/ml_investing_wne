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

start = datetime.datetime(2021, 11, 15, 1, 0, 0, 0)
userId = 12896600
password = "xoh10026"
symbol = 'EURGBP'

client = APIClient()

# connect to RR socket, login
loginResponse = client.execute(loginCommand(userId=userId, password=password))


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
df['hour'] = df['datetime'].dt.time
df['close'] = df['close'].round(4)
df['high'] = df['high'].round(4)
df['low'] = df['low'].round(4)
df['open'] = df['open'].round(4)
temp = df.loc[df['hour'].isin([datetime.time(20,0,0),datetime.time(22,0,0), datetime.time(0,0,0)])]