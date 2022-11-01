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
from ml_investing_wne.helper import get_scaler, compute_profitability_classes
from ml_investing_wne.xtb.Trader import Trader
from ml_investing_wne import config
from ml_investing_wne.models import model_factory

# logger settings
logger = logging.getLogger()
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

sc_x = get_scaler()
# config currency and load_model should be the same!
model = model_factory()
client = APIClient()

loginResponse = client.execute(loginCommand(userId=config.USER_ID, password=config.PASSWORD))
logger.info(str(loginResponse))

# check if user logged in correctly
if (loginResponse['status'] == False):
    print('Login failed. Error code: {0}'.format(loginResponse['errorCode']))

# get ssId from login response
ssid = loginResponse['streamSessionId']
balance = client.commandExecute('getMarginLevel')

if __name__ == "__main__":
    trader = Trader(client, symbol=config.currency, volume=0.1, upper_bound=0.7, 
                    lower_bound=0.3, max_spread=2.1)
    trader.trade()