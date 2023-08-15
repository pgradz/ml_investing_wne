import pandas as pd
import numpy as np
import datetime
from ml_investing_wne import config


backtest = pd.read_csv('/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/models/backtest_ETHUSDT.csv', parse_dates=['datetime'])
trades = backtest[backtest['transaction'].notna()]
trades.reset_index(inplace=True, drop=True)
trades.sort_values(by=['datetime'], inplace=True)

for i in trades.index:
    trades.loc[i, 'price_entry'] = trades.loc[i, 'close']
    # prices
    if trades.loc[i, 'transaction'] in ['sell', 'buy']:
        if trades.loc[i, 'barrier_touched'] == 'top':
            trades.loc[i, 'exit_price'] = trades.loc[i, 'top_barrier']
        elif trades.loc[i, 'barrier_touched'] == 'bottom':
            trades.loc[i, 'exit_price'] = trades.loc[i, 'bottom_barrier']
        elif trades.loc[i, 'barrier_touched'] == 'vertical':
            trades.loc[i, 'exit_price'] = trades.loc[i+1, 'close']
    elif trades.loc[i, 'transaction'] == 'No trade':
        trades.loc[i, 'exit_price'] = trades.loc[i+1, 'close']


df_daily = pd.read_csv('/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/processed/binance_ETHUSDT/time_aggregated_1440min.csv', parse_dates=['datetime'])

df_daily['datetime'] = pd.to_datetime(df_daily['datetime']) 
df_daily['datetime'] = df_daily['datetime'] +  pd.DateOffset(hours=23) + pd.DateOffset(minutes=59) +  pd.DateOffset(seconds=59)
start_date = backtest['datetime'].min().date()
start_datetime = datetime.datetime.combine(start_date, datetime.time(23, 0, 0))
df_daily = df_daily.loc[df_daily['datetime']>=start_datetime]
cols = ['datetime', 'open','close', 'barrier_touched','barrier_touched_date', 'top_barrier','bottom_barrier', 'transaction', 'budget', 'exit_price']
df_daily['barrier_touched'] = None
df_daily['barrier_touched_date'] = None
df_daily['top_barrier'] = None
df_daily['bottom_barrier'] = None
df_daily['transaction'] = None
df_daily['budget'] = None
df_daily['exit_price'] = None

trades_and_close = pd.concat([trades[cols], df_daily[cols]])
trades_and_close.sort_values(by='datetime', inplace=True)
trades_and_close['transaction'] = trades_and_close['transaction'].ffill()
trades_and_close.reset_index(inplace=True, drop=True)


results = []
transaction = 'No trade'

for i in trades.index:
    if i == 366:
        print('here')
    # open position
    if transaction == 'No trade':
        if trades_and_close.loc[i, 'transaction'] == 'No trade':
            continue
        else:
            _datetime = trades_and_close.loc[i, 'datetime']
            transaction = trades_and_close.loc[i, 'transaction']
            entry_price = trades_and_close.loc[i, 'close']
            exit_position_price = trades_and_close.loc[i, 'exit_price']
            barrier_touched = trades_and_close.loc[i, 'barrier_touched']

            hold_period = 0
            continue
            # results.append([transaction, entry_price, exit_price, barrier_touched, _return, entry_cost, exit_cost, hold_period])
    
    # no change in open position
    if transaction == trades_and_close.loc[i, 'transaction'] and transaction != 'No trade':
        # this means that this is end of the day record
        if trades_and_close.loc[i, 'barrier_touched'] is None:
            if hold_period == 0:
                _return = trades_and_close.loc[i, 'close'] / entry_price -1
                entry_cost = config.cost
            else:
                _datetime = trades_and_close.loc[i, 'datetime']
                entry_price = trades_and_close.loc[i, 'open']
                _return = trades_and_close.loc[i, 'close'] / entry_price -1
                entry_cost = 0
            exit_cost = 0
            exit_price = trades_and_close.loc[i, 'close']
            hold_period += 1
            results.append([_datetime, transaction, entry_price, exit_price, _return, entry_cost, exit_cost])
            continue
            
        # if current row is intraday record we have to close it as this would be automatic take profit or stop loss
        else:
            exit_price = exit_position_price
            _return = exit_price/entry_price -1
            exit_cost = config.cost
            results.append([_datetime, transaction, entry_price, exit_price, _return, entry_cost, exit_cost])
            # open new position in the same direction
            _datetime = trades_and_close.loc[i, 'datetime']
            entry_price = trades_and_close.loc[i, 'close']
            exit_position_price = trades_and_close.loc[i, 'exit_price']
            barrier_touched = trades_and_close.loc[i, 'barrier_touched']
            continue

        
    # change in open position
    if transaction != trades_and_close.loc[i, 'transaction']:
        _datetime = trades_and_close.loc[i, 'datetime']
        # close the position
        
        exit_price = exit_position_price
        entry_price = trades_and_close.loc[i-1, 'close']
        _return = exit_price / entry_price -1
        if hold_period == 0:
            entry_cost = config.cost
        else:
            entry_cost = 0
        exit_cost = config.cost
        results.append([_datetime, transaction, entry_price, exit_price, _return, entry_cost, exit_cost])

        transaction = trades_and_close.loc[i, 'transaction']
        if transaction in ['buy','sell']:
            entry_price = trades_and_close.loc[i, 'close']
            exit_position_price = trades_and_close.loc[i, 'exit_price']
            barrier_touched = trades_and_close.loc[i, 'barrier_touched']
            _return = 0
            entry_cost = config.cost
            exit_cost = 0
            hold_period = 0
            continue



daily_returns = pd.DataFrame(results, columns = ['datetime', 'transaction', 'entry_price', 'exit_price','return', 'entry_cost', 'exit_cost'])
daily_returns['return'] = np.where(daily_returns['transaction']== 'sell', -daily_returns['return'], daily_returns['return'])

daily_returns['return'] = daily_returns['return'] - daily_returns['entry_cost'] - daily_returns['exit_cost']
            
 

