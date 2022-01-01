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

sc_x = joblib.load(os.path.join(config.package_directory, 'models',
                                'sc_x_{}_{}.save'.format(symbol, config.freq)))
model = load_model(os.path.join(config.package_directory, 'models',
                                '{}_{}_{}.hdf5'.format(config.model, symbol, config.freq)))

client = APIClient()

loginResponse = client.execute(loginCommand(userId=userId, password=password))
logger.info(str(loginResponse))

# check if user logged in correctly
if (loginResponse['status'] == False):
    print('Login failed. Error code: {0}'.format(loginResponse['errorCode']))

# get ssId from login response
ssid = loginResponse['streamSessionId']
balance = client.commandExecute('getMarginLevel')


class Trader():
    def __init__(self, client, symbol, volume, upper_bound, lower_bound):
        self.client = client
        self.symbol = symbol
        self.volume = volume
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.order = 0
        self.retries = 3
        self.last_timestamp = None
        self.last_price = None
        self.y_pred = None
        self.position = None
        self.trade_status()

    def make_predictions(self):
        resp = self.client.commandExecute('getChartLastRequest',
                                          {'info': {"period": 60,
                                                    "start": int(start.timestamp() * 1000),
                                                    "symbol": self.symbol
                                                    }
                                           }, verbose=False
                                          )
        df = pd.DataFrame(resp['returnData']['rateInfos'])
        last_timestamp = df.loc[df.index[-1], 'ctmString']
        self.last_price = df.loc[df.index[-1], 'open'] + df.loc[df.index[-1], 'close']
        df['datetime'] = pd.to_datetime(df['ctm'], unit='ms')
        df['close'] = (df['open'] + df['close']) / 100000
        df['high'] = (df['open'] + df['high']) / 100000
        df['low'] = (df['open'] + df['low']) / 100000
        df['open'] = df['open'] / 100000

        df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('Europe/Warsaw')
        df['datetime'] = df['datetime'].dt.tz_localize(None)
        df = df.sort_values(by=['datetime'], ascending=True)
        df = df.set_index('datetime')
        # order in hist data - open, high, low, close
        df = df[['open', 'high', 'low', 'close']]

        df = prepare_processed_dataset(df=df, allow_null=True)
        X_test, y_test, y_test_cat = test_split(df, config.seq_len, sc_x)
        y_pred = model.predict(X_test[-1:])
        self.y_pred = y_pred[0][1]
        logger.info('Prediction for next hour as of {} is {} and last close price {}'.format(last_timestamp,
                                                                                             self.y_pred,
                                                                                             self.last_price))
        return last_timestamp

    def get_tick_prices(self):
        self.tick = self.client.commandExecute('getTickPrices', {"level": 0,
                                                                 "symbols": [self.symbol],
                                                                 "timestamp": int(
                                                                     datetime.datetime.now().timestamp() * 1000 - 100000)
                                                                 }
                                               )

    def order_status(self, order):
        order_status = client.commandExecute("tradeTransactionStatus", {"order": order})
        try:
            if order_status['returnData']['requestStatus'] == 3:
                return True
            else:
                return False
        except:
            logger.info('Order not found!')
            return False

    def order_status_wait(self, order):
        trade_successful = False
        retries = 0
        while not trade_successful and retries <= self.retries:
            trade_successful = self.order_status(order)
            if trade_successful:
                logger.info('Transaction processed successfully')
                return True
            else:
                retries += 1
                time.sleep(3)
        logger.info('Transaction NOT processed successfully')
        return False

    def trade_status(self):
        open_trades = self.client.commandExecute("getTrades", {"openedOnly": True})
        open_trades_symbol = []
        for i in open_trades['returnData']:
            if i['symbol'] == self.symbol:
                open_trades_symbol.append(i)
        if len(open_trades_symbol) == 0:
            self.position = None
        elif len(open_trades_symbol) == 1:
            self.order = open_trades_symbol[0]['order']
            if open_trades_symbol[0]['cmd'] == 0:
                self.position = 'long'
            elif open_trades_symbol[0]['cmd'] == 1:
                self.position = 'short'
            else:
                logger.info('unknown transaction status!')
        else:
            logger.info('There is more than one transaction open!')

    def go_long(self):
        trade = self.client.commandExecute('tradeTransaction',
                                           {
                                               "tradeTransInfo":
                                                   {"cmd": 2,
                                                    "customComment": "Decision to buy at {}".format(self.last_price),
                                                    "expiration": int(
                                                        datetime.datetime.now().timestamp() * 1000 + 300000),
                                                    "offset": 0,
                                                    "order": self.order,
                                                    "price": self.tick['returnData']['quotations'][0]['ask'],
                                                    "sl": 0.0,
                                                    "symbol": self.symbol,
                                                    "tp": 0.0,
                                                    "type": 0,
                                                    "volume": self.volume
                                                    }
                                           })
        self.order_status_wait(trade['returnData']['order'])
        self.trade_status()
        if self.position != 'long':
            logger.info('long transaction unsuccessful')

    def close_long(self):
        trade = self.client.commandExecute('tradeTransaction', {
            "tradeTransInfo":
                {"cmd": 1,
                 "customComment": "Decision to close long at {}".format(self.last_price),
                 "expiration": int(datetime.datetime.now().timestamp() * 1000 + 300000),
                 "offset": 0,
                 "order": self.order,
                 "price": self.tick['returnData']['quotations'][0]['bid'],
                 "sl": 0.0,
                 "symbol": self.symbol,
                 "tp": 0.0,
                 "type": 2,
                 "volume": self.volume}
        })
        self.order_status_wait(trade['returnData']['order'])
        self.trade_status()
        if self.position is not None:
            logger.info('closing long position unsuccessful')

    def go_short(self):
        trade = self.client.commandExecute('tradeTransaction',
                                           {
                                               "tradeTransInfo":
                                                   {"cmd": 1,
                                                    "customComment": "Decision to sell at {}".format(self.last_price),
                                                    "expiration": int(
                                                        datetime.datetime.now().timestamp() * 1000 + 300000),
                                                    "offset": 0,
                                                    "order": self.order,
                                                    "price": self.tick['returnData']['quotations'][0]['bid'],
                                                    "sl": 0.0,
                                                    "symbol": self.symbol,
                                                    "tp": 0.0,
                                                    "type": 0,
                                                    "volume": self.volume
                                                    }
                                           })

        self.order_status_wait(trade['returnData']['order'])
        self.trade_status()
        if self.position != 'short':
            logger.info('short transaction unsuccessful')

    def close_short(self):
        trade = self.client.commandExecute('tradeTransaction',
                                           {
                                               "tradeTransInfo":
                                                   {"cmd": 0,
                                                    "customComment": "Decision to close short at {}".format(
                                                        self.last_price),
                                                    "expiration": int(
                                                        datetime.datetime.now().timestamp() * 1000 + 300000),
                                                    "offset": 0,
                                                    "order": self.order,
                                                    "price": self.tick['returnData']['quotations'][0]['ask'],
                                                    "sl": 0.0,
                                                    "symbol": self.symbol,
                                                    "tp": 0.0,
                                                    "type": 2,
                                                    "volume": self.volume
                                                    }
                                           })

        self.order_status_wait(trade['returnData']['order'])
        self.trade_status()
        if self.position is not None:
            logger.info('closing short position unsuccessful')

    def trade(self):

        while (True):
            last_timestamp = self.last_timestamp
            # proceed further only if new data has arrived
            while (last_timestamp == self.last_timestamp):
                last_timestamp = self.make_predictions()
                if last_timestamp != self.last_timestamp:
                    self.last_timestamp = last_timestamp
                    break
                else:
                    time.sleep(30)
            # update status as sometimes it's too quick to be updated
            self.trade_status()
            self.get_tick_prices()
            if self.y_pred > self.upper_bound:
                if self.position == 'long':
                    pass
                if self.position is None:
                    self.go_long()
                if self.position == 'short':
                    self.close_short()
                    # verify that short was closed
                    if self.position is None:
                        self.go_long()
            elif self.y_pred > self.lower_bound:
                if self.position == 'long':
                    self.close_long()
                elif self.position == 'short':
                    self.close_long()
                else:
                    pass
            else:
                if self.position == 'long':
                    self.close_long()
                    self.go_short()
                elif self.position is None:
                    self.go_short()
                else:
                    pass

            now = datetime.datetime.now()
            next_hour = now + datetime.timedelta(hours=1)
            next_hour = next_hour.replace(second=0, microsecond=0, minute=0)
            difference = (next_hour - now).total_seconds()
            logger.info('going to wait {} seconds'.format(difference))
            time.sleep(difference)


trader = Trader(client, symbol, volume=0.1, upper_bound=0.6, lower_bound=0.4)
trader.trade()

trader = Trader(client, symbol, volume=0.1, upper_bound=0.65, lower_bound=0.35)
trader.trade()

open_trades = client.commandExecute("getTrades", {"openedOnly": True})
open_trades_symbol = []
for i in open_trades['returnData']:
    if i['symbol'] == 'EURPLN':
        open_trades_symbol.append(i)

{
    "command": "tradeTransactionStatus",
    "arguments": {
        "order": 43
    }
}

trade['returnData']

open_trades = client.commandExecute("getTrades", {"openedOnly": True})
len(open_trades['returnData'])
open_trades['returnData'][0]['order']

trade = client.commandExecute('tradeTransaction', {
    "tradeTransInfo":
        {"cmd": 1,
         "customComment": "Decision to buy at",
         "expiration": int(datetime.datetime.now().timestamp() * 1000 + 60000),
         "offset": 0,
         "order": 333046520,
         "price": tick['returnData']['quotations'][0]['bid'],
         "sl": 0.0,
         "symbol": symbol,
         "tp": 0.0,
         "type": 2,
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

trade_status = client.commandExecute("tradeTransactionStatus", {"order": trade['returnData']['order']})
trade_status['returnData']['requestStatus']
trade_status = client.commandExecute("tradeTransactionStatus", {"order": 123})

open_trades = client.commandExecute("getTrades", {"openedOnly": True})
open_trades_symbol = []
for i in open_trades['returnData']:
    if i['symbol'] == symbol:
        open_trades_symbol.append(i)


resp = client.commandExecute('getChartLastRequest',
                                          {'info': {"period": 60,
                                                    "start": int(start.timestamp() * 1000),
                                                    "symbol": symbol
                                                    }
                                           }, verbose=False
                                          )