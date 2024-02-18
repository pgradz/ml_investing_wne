import datetime
import logging

import mlflow.keras

from ml_investing_wne import config
from ml_investing_wne.data_engineering.prepare_dataset import \
    prepare_processed_dataset
from ml_investing_wne.helper import (evaluate_model, get_callbacks,
                                     get_final_model_path,
                                     get_ml_flow_experiment_name)
from ml_investing_wne.models import model_factory
from ml_investing_wne.train_test_val_split import train_test_val_split
from ml_investing_wne.xtb.xAPIConnector import APIClient, loginCommand
from ml_investing_wne.xtb.xtb_utils import prepare_xtb_data

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# need to define starting period for api request
start = datetime.datetime(2019, 7, 1, 1, 0, 0, 0)

client = APIClient()
# connect to RR socket, login
loginResponse = client.execute(loginCommand(userId=config.USER_ID, password=config.PASSWORD))
logger.info(str(loginResponse))
# check if user logged in correctly
if (loginResponse['status'] == False):
    logger.error('Login failed. Error code: {0}'.format(loginResponse['errorCode']))

# get ssId from login response
ssid = loginResponse['streamSessionId']

resp = client.commandExecute('getChartLastRequest', {'info': {"period": 60, 
                                                     "start": int(start.timestamp() * 1000),
                                                     "symbol": config.currency}})

df = prepare_xtb_data(resp)                                                   
df = prepare_processed_dataset(df=df)
#X, y, X_val, y_val, X_test, y_test, y_cat, y_val_cat, y_test_cat, train = train_test_val_split(df, config.seq_len, sc_x)
X, y, X_val, y_val, X_test, y_test, y_cat, y_val_cat, y_test_cat, train = train_test_val_split(df)

mlflow.set_experiment(experiment_name=get_ml_flow_experiment_name('xtb'))
callbacks = get_callbacks('xtb')
model = model_factory(X)
history = model.fit(X, y_cat, batch_size=config.batch, epochs=config.epochs, verbose=2,
                        validation_data=(X_val, y_val_cat), callbacks=callbacks)
model.save(get_final_model_path('xtb'))
evaluate_model(model, df, X_test, y_test_cat, y, y_val, y_test)