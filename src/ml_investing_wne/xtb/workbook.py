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

# sc_x = joblib.load(os.path.join(config.package_directory, 'models',
#                                 'sc_x_{}_{}.save'.format(config.currency, config.freq)))
# model = load_model(os.path.join(config.package_directory, 'models',
#                                 '{}_{}_{}.hdf5'.format(config.model, config.currency, config.freq)))

start = datetime.datetime(2021, 10, 1, 1, 0, 0, 0)
userId = 12896600
password = "xoh10026"
symbol = 'EURPLN'

client = APIClient()

loginResponse = client.execute(loginCommand(userId=userId, password=password))

resp = client.commandExecute('getChartLastRequest', {'info': {"period": 60, "start": int(start.timestamp() * 1000),
                                                              "symbol": symbol}}, verbose=False)
df = pd.DataFrame(resp['returnData']['rateInfos'])
last_timestamp = df.loc[df.index[-1], 'ctmString']
last_price = df.loc[df.index[-1], 'close']
df['datetime'] = pd.to_datetime(df['ctm'], unit='ms')
# df['datetime'].tz_localize(tz='Europe/Berlin')
# df['datetime'].tz_convert(tz='Europe/Berlin')
df['close'] = (df['open'] + df['close']) / 100000
df['high'] = (df['open'] + df['high']) / 100000
df['low'] = (df['open'] + df['low']) / 100000
df['open'] = df['open'] / 100000
df = df.set_index('datetime')
df.drop(columns=['ctm', 'ctmString', 'vol'], inplace=True)
df = prepare_processed_dataset(df=df, allow_null=True)
X_test, y_test, y_test_cat = test_split(df, config.seq_len, sc_x)
y_pred = model.predict(X_test[-1:])

order = 0
# if y_pred > 0.7:



tick = client.commandExecute('getTickPrices', {"level": 0,
                                               "symbols": [symbol],
                                               "timestamp": int(datetime.datetime.now().timestamp() * 1000 - 100000)
                                               }
                             )

trade = client.commandExecute('tradeTransaction', {
    "tradeTransInfo":
        {"cmd": 0,
         "customComment": "Decision to buy at {}".format(last_price),
         "expiration": int(datetime.datetime.now().timestamp() * 1000 + 60000),
         "offset": 0,
         "order": order,
         "price": tick['returnData']['quotations'][0]['ask'],
         "sl": 0.0,
         "symbol": symbol,
         "tp": 0.0,
         "type": 0,
         "volume": 0.1}
})


open_trades = client.commandExecute("getTrades", {"openedOnly": True})
len(open_trades['returnData'])
open_trades['returnData'][0]['order']


trade = client.commandExecute('tradeTransaction', {
    "tradeTransInfo":
        {"cmd": 1,
         "customComment": "Decision to buy at {}".format(last_price),
         "expiration": int(datetime.datetime.now().timestamp() * 1000 + 60000),
         "offset": 0,
         "order": order,
         "price": tick['returnData']['quotations'][0]['bid'],
         "sl": 0.0,
         "symbol": symbol,
         "tp": 0.0,
         "type": 0,
         "volume": 0.1}
})

close = client.commandExecute('tradeTransaction', {
    "tradeTransInfo":
        {"cmd": 1,
         "customComment": "Decision to buy at {}".format(last_price),
         "expiration": int(datetime.datetime.now().timestamp() * 1000 + 60000),
         "offset": 0,
         "order": open_trades['returnData'][0]['order'],
         "price": tick['returnData']['quotations'][0]['bid'],
         "sl": 0.0,
         "symbol": symbol,
         "tp": 0.0,
         "type": 2,
         "volume": 0.1}
})
