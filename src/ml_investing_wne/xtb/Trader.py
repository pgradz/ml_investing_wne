import datetime
import logging
import time

import pandas as pd

import ml_investing_wne.config as config
from ml_investing_wne.data_engineering.prepare_dataset import \
    prepare_processed_dataset
from ml_investing_wne.train_test_val_split import test_split
from ml_investing_wne.xtb.xtb_utils import prepare_xtb_data

logger = logging.getLogger(__name__)

class Trader():
    def __init__(self, client, symbol, volume, upper_bound, lower_bound, max_spread, start, model, sc_x, time_interval_in_min,
                 freq, hours_to_trade, hours_to_exclude, take_profit_pips):
        self.client = client
        self.symbol = symbol
        self.volume = volume
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.max_spread = max_spread
        self.start = start
        self.model = model
        self.sc_x = sc_x
        self.time_interval_in_min = time_interval_in_min
        self.freq = freq
        self.hours_to_trade = hours_to_trade
        self.hours_to_exclude = hours_to_exclude
        self.take_profit_pips = take_profit_pips
        self.order = 0
        self.retries = 8
        self.last_timestamp = None
        self.last_price = None
        self.y_pred = None
        self.position = None
        self.trade_status()
        self.spread = 100

    def make_predictions(self):
        resp = self.client.commandExecute('getChartLastRequest',
                                          {'info': {"period": self.time_interval_in_min,
                                                    "start": int(self.start.timestamp() * 1000),
                                                    "symbol": self.symbol
                                                    }
                                           }, verbose=False
                                          )
        df = pd.DataFrame(resp['returnData']['rateInfos'])
        last_timestamp = df.loc[df.index[-1], 'ctmString']
        self.last_price = df.loc[df.index[-1], 'open'] + df.loc[df.index[-1], 'close']
        df = prepare_xtb_data(resp) 
        df = prepare_processed_dataset(df=df, allow_null=True)
        X_test, y_test, y_test_cat = test_split(df, config.seq_len, self.sc_x)
        y_pred = self.model.predict(X_test[-1:])
        self.y_pred = y_pred[0][1]
        logger.info('Prediction for next hour as of {} is {} and last close price {}'.format(
            last_timestamp,self.y_pred, self.last_price))
        return last_timestamp

    def get_tick_prices(self):
        self.tick = self.client.commandExecute('getTickPrices', 
                                                {"level": 0,
                                                 "symbols": [self.symbol],
                                                "timestamp": int(
                                                datetime.datetime.now().timestamp() * 1000 - 100000)
                                                }
                                               )
        try:
            self.spread = self.tick['returnData']['quotations'][0]['spreadTable']
            if self.spread < self.max_spread:
                logger.info('Spread is lower than max spread, we can trade!')
            else:
                logger.info('Spread is higher than max spread, we can not trade!')
        except:
            logger.info('Spread not available!')
            self.spread = None

    def order_status(self, order):
        order_status = self.client.commandExecute("tradeTransactionStatus", {"order": order})
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
                                                    "tp": self.tick['returnData']['quotations'][0]['bid'] + self.take_profit_pips/10000,
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
                 "expiration": int(datetime.datetime.now().timestamp() * 1000 + 600000),
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
                                                    "tp": self.tick['returnData']['quotations'][0]['ask'] - self.take_profit_pips/10000,
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
                                                        datetime.datetime.now().timestamp() * 1000 + 600000),
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

    @staticmethod
    def ceil_dt(dt, delta):
        return dt + (datetime.datetime.min - dt) % delta

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
            # update status as sometimes it's too quick to be updated in other methods
            self.trade_status()
            self.get_tick_prices()
            # sometimes spread is not available, retry then
            if self.spread is None:
                retry = 0
                while (self.spread is None or self.spread > self.max_spread) and retry < self.retries:
                    self.get_tick_prices()
                    if self.spread <= self.max_spread:
                        break
                    retry += 1
                    time.sleep(15)
            # trading logic
            if datetime.datetime.now().hour in self.hours_to_trade:

                if self.y_pred > self.upper_bound:
                    if self.position == 'long':
                        pass
                    if self.position is None and self.spread < self.max_spread:
                        self.go_long()
                    if self.position == 'short':
                        self.close_short()
                        # verify that short was closed
                        if self.position is None and self.spread < self.max_spread:
                            self.go_long()
                elif self.y_pred > self.lower_bound:
                    if self.position == 'long':
                        self.close_long()
                    elif self.position == 'short':
                        self.close_short()
                    else:
                        pass
                else:
                    if self.position == 'long':
                        self.close_long()
                        # verify that long was closed
                        if self.position is None and self.spread < self.max_spread:
                            self.go_short()
                    elif self.position is None and self.spread < self.max_spread:
                        self.go_short()
                    else:
                        pass
            elif datetime.datetime.now().hour not in self.hours_to_exclude:
                # before closing transaction check again spread
                if self.position in ['long', 'short'] and self.spread > self.max_spread:
                    retry = 0
                    while self.spread > self.max_spread and retry < self.retries:
                        self.get_tick_prices()
                        if self.spread <= self.max_spread:
                            break
                        retry += 1
                        time.sleep(15)
                if self.position == 'long':
                    self.close_long()
                elif self.position == 'short':
                    self.close_short()
                else:
                    logger.info('currently no position - nothing to do')

            now = datetime.datetime.now()
            next_time = self.ceil_dt(now, datetime.timedelta(minutes=self.time_interval_in_min))
            #next_hour = now + datetime.timedelta(hours=1)
            #next_hour = next_hour.replace(second=0, microsecond=0, minute=0)
            difference = (next_time - now).total_seconds()
            logger.info('going to wait {} seconds'.format(difference))
            time.sleep(difference)

