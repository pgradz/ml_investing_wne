import os
import pandas as pd
import logging
import datetime
import time
import joblib
from tensorflow.keras.models import load_model
from ml_investing_wne.xtb.xAPIConnector import APIClient, APIStreamClient, loginCommand
from ml_investing_wne.data_engineering.prepare_dataset import prepare_processed_dataset
import ml_investing_wne.config as config
from ml_investing_wne.train_test_val_split import test_split
from ml_investing_wne.helper import confusion_matrix_plot, compute_profitability_classes
from ml_investing_wne.xtb.Trader import Trader


# logger settings
logger = logging.getLogger()
# You can set a different logging level for each logging handler but it seems you will have to set the
# logger's level to the "lowest".
logger.setLevel(logging.INFO)
stream_h = logging.StreamHandler()
file_h = logging.FileHandler(os.path.join(config.package_directory, 'logs', 'trading.log'))
stream_h.setLevel(logging.INFO)
file_h.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
stream_h.setFormatter(formatter)
file_h.setFormatter(formatter)
logger.addHandler(stream_h)
logger.addHandler(file_h)

start = datetime.datetime(2021, 10, 1, 1, 0, 0, 0)
userId = 12896600
password = "xoh10026"
symbol = 'USDCHF'
freq = '60min'
input_dim = '1d'
model = 'resnet'

sc_x = joblib.load(os.path.join(config.package_directory, 'models','production',
                                'sc_x_{}_{}.save'.format(symbol, freq)))
model = load_model(os.path.join(config.package_directory, 'models', 'production',
                                    '{}_hist_data_{}_{}'.format(model, symbol, freq)))

client = APIClient()

loginResponse = client.execute(loginCommand(userId=userId, password=password))
logger.info(str(loginResponse))

# check if user logged in correctly
if (loginResponse['status'] == False):
    print('Login failed. Error code: {0}'.format(loginResponse['errorCode']))

# get ssId from login response
ssid = loginResponse['streamSessionId']
balance = client.commandExecute('getMarginLevel')

#USDCHF
trader = Trader(client, symbol, volume=0.1, upper_bound=0.65, lower_bound=0.35, max_spread=2.1, start=start, model=model,
                sc_x=sc_x, time_interval_in_min=60)
trader.trade()