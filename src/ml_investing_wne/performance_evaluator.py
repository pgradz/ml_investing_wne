'''
This script calculates daily returns of the strategy. It is based on the backtest results.
'''
import datetime
import logging
import os

import numpy as np
import pandas as pd
from scipy.stats import binom_test

from ml_investing_wne.utils import get_logger

logger = logging.getLogger(__name__)

backtest_folder = '/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/models/BTCUSDT_96_time_aggregated_keras_tuner_transformer_learnable_encoding_1440min'
# backtest_folder = '/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/models/ensemble_full_run_MATICUSDT_cumsum_triple_barrier_with_volume_no_SMA'
# backtest_folder = '/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/models/ensemble_full_run_SOLUSDT_cumsum_triple_barrier'
# backtest_folder = '/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/models/ensemble_full_run_BTCUSDT_cumsum_triple_barrier_with_volume_no_SMA'
# backtest_folder = '/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/models/ensemble_full_run_BTCUSDT_cumsum'
# backtest_folder = '/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/models/ensemble_full_run_ETHUSDT_cumsum'
# backtest_folder = '/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/models/ensemble_full_run_MATICUSDT_cumsum'
daily_records = '/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/processed/binance_BTCUSDT/time_aggregated_1440min.csv'
# daily_records = '/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/processed/binance_SOLUSDT/time_aggregated_1440min.csv'
# daily_records = '/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/processed/binance_MATICUSDT/time_aggregated_1440min.csv'
# daily_records = '/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/data/processed/binance_ETHUSDT/time_aggregated_1440min.csv'
# output folder
output_folder = '/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/results'
# output_folder = '/root/ml_investing_wne/src/ml_investing_wne/models/results'

class PerformanceEvaluator():
    '''
    This class calculates daily returns of the strategy. It is based on the backtest results.
    '''

    def __init__(self, backtest_folder, daily_records_file=None, daily_records_dataframe=None, risk_free_rate=0.0314, seeds = ['12345', '123456', '1234567'],name=None, 
                 start_date=datetime.date(2022, 3, 31), end_date=datetime.date(2023, 7, 1),
                 cost=0.001):
        # risk free rate is taken as average overnight rate for corresponding period from https://treasurydirect.gov/GA-FI/FedInvest/selectOvernightRateDate
        self.backtest_folder = backtest_folder
        self.triple_barrier = False
        if 'triple_barrier' in self.backtest_folder:
            self.triple_barrier = True
        if not name:
            self.name = os.path.splitext(os.path.basename(backtest_folder))[0]
        else:
            self.name = name
        self.start_date = start_date
        self.end_date = end_date
        self.cost = cost
        if daily_records_file is not None:
            self.df_daily = pd.read_csv(daily_records, parse_dates=['datetime'])
            self.df_daily['datetime'] = pd.to_datetime(self.df_daily['datetime']) 
        elif daily_records_dataframe is not None:
            self.df_daily = daily_records_dataframe
        # shift daily records to represent end of the day    
        self.df_daily['datetime'] = self.df_daily['datetime'] +  pd.DateOffset(hours=23) + pd.DateOffset(minutes=59) +  pd.DateOffset(seconds=59)
        self.risk_free_rate = risk_free_rate
        self.seeds = seeds
        self.metrics = {}
        self.metrics['3. accuracy'] = []
        self.metrics['2. correct_transactions'] = []
        self.metrics['1. gain_loss'] = []
        self.metrics['4. sharpe_ratio'] = []
        self.metrics['5. sortino_ratio'] = []
        self.metrics['6. max_drawdown'] = []
        self.metrics['7. share_of_time_active'] = []
        self.metrics['8. no. of trades'] = []


    def load_backtest_data(self, seed):
        backtest_files  = [os.path.join(self.backtest_folder, file) for file in os.listdir(self.backtest_folder) if file.endswith( seed + '.csv')]
        for file in backtest_files:
            yield pd.read_csv(file, parse_dates=['datetime'])


    def format_backtest_results(self):
        '''
        This function formats backtest results so they can be used to calculate daily returns
        
        '''
        # removes periods of waiting with open position
        trades = self.backtest[self.backtest['transaction'].notna()]
        trades.reset_index(inplace=True, drop=True)
        trades.sort_values(by=['datetime'], inplace=True)

        if self.triple_barrier:
            # add entry and exit price to each trade
            for i in trades.index:
                trades.loc[i, 'price_entry'] = trades.loc[i, 'close']
                # prices
                if trades.loc[i, 'transaction'] in ['sell', 'buy']:
                    if trades.loc[i, 'barrier_touched'] == 'top':
                        trades.loc[i, 'exit_price'] = trades.loc[i, 'top_barrier']
                    elif trades.loc[i, 'barrier_touched'] == 'bottom':
                        trades.loc[i, 'exit_price'] = trades.loc[i, 'bottom_barrier']
                    elif trades.loc[i, 'barrier_touched'] == 'vertical':
                        if i+1 <= trades.index.max():
                            trades.loc[i, 'exit_price'] = trades.loc[i+1, 'close']
                        else:
                            trades.loc[i, 'exit_price'] = trades.loc[i, 'close'] * (1+trades.loc[i, 'prc_change'])
                elif trades.loc[i, 'transaction'] == 'No trade':
                    if i+1 <= trades.index.max():
                        trades.loc[i, 'exit_price'] = trades.loc[i+1, 'close']

        self.trades = trades


    def accuracy_by_threshold(self, df, lower_bound, upper_bound, type='3. accuracy'):
        '''calculate accuracy for a given upper and lowe threshold
        lower_bound: lower bound of threshold
        upper_bound: upper bound of threshold'''

        y_pred_class = [1 if y > 0.5 else 0 for y in df['prediction']]
        actual = [1 if y > 0 else 0 for y in df['prc_change']]
        df_accuracy = pd.DataFrame({'prediction': df['prediction'], 'y_true': actual, 'y_pred': y_pred_class})
        predictions_above_threshold = df_accuracy.loc[(df_accuracy['prediction'] < lower_bound) | (df_accuracy['prediction'] > upper_bound)]
        accuracy = (predictions_above_threshold['y_true'] == predictions_above_threshold['y_pred']).mean()
        # change to percentages
        accuracy = accuracy * 100
        logger.info('accuracy for threshold between %.2f and %.2f : %.4f, based on %.1f observations', lower_bound, upper_bound, accuracy, predictions_above_threshold.shape[0]/df_accuracy.shape[0])

        self.metrics[type].append(accuracy)

        return None

    def combine_trades_and_daily_close(self):
        '''
        This function combines trades and daily close so daily returns can be calculated
        ''' 
        start_date = self.backtest['datetime'].min().date()
        start_datetime = datetime.datetime.combine(start_date, datetime.time(23, 0, 0))
        df_daily = self.df_daily.loc[self.df_daily['datetime']>=start_datetime].copy()
        if self.triple_barrier:
            cols = ['datetime', 'open','close', 'high', 'low', 'barrier_touched','barrier_touched_date', 'top_barrier','bottom_barrier', 'transaction', 'budget', 'exit_price']
            df_daily['barrier_touched'] = None
            df_daily['barrier_touched_date'] = None
            df_daily['top_barrier'] = None
            df_daily['bottom_barrier'] = None
            df_daily['exit_price'] = None
        else: 
            cols = ['datetime', 'open','close', 'high', 'low', 'transaction', 'budget']

        df_daily['transaction'] = None
        df_daily['budget'] = None

        # concatenate trades and end of the day records so performance at the end of the day can be calculated
        trades_and_close = pd.concat([self.trades[cols], df_daily[cols]])
        trades_and_close.sort_values(by='datetime', inplace=True)
        trades_and_close['transaction'] = trades_and_close['transaction'].ffill()
        trades_and_close.reset_index(inplace=True, drop=True)
        if self.triple_barrier:
            max_date = pd.to_datetime(trades_and_close.loc[trades_and_close['transaction'].isin(['buy','sell'])]['barrier_touched_date']).max()
            self.trades_and_close = trades_and_close.loc[trades_and_close['datetime']<=max_date]
        else:
            self.trades_and_close = trades_and_close
        

    def get_daily_returns_one_step(self):

        results = []
        transaction = 'No trade'
        i = 0

        while i <= self.trades_and_close.index.max():
            # open position
            # if i == 437:
            #     print('debug')

            if transaction == 'No trade':
                if self.trades_and_close.loc[i, 'transaction'] == 'No trade':
                    i += 1
                    continue
                else:
                    datetime_start = self.trades_and_close.loc[i, 'datetime']
                    transaction = self.trades_and_close.loc[i, 'transaction']
                    entry_price = self.trades_and_close.loc[i, 'close']
                    entry_position_price = entry_price 
                    hold_period = 0
                    i += 1
                    continue

            if transaction == self.trades_and_close.loc[i, 'transaction'] and transaction != 'No trade':
                # this means that this is end of the day record
                # if self.trades_and_close.loc[i, 'budget'] is None:
                datetime_end = self.trades_and_close.loc[i, 'datetime']
                if hold_period == 0:
                    entry_cost = self.cost
                else:
                    entry_cost = 0
                    
                exit_price = self.trades_and_close.loc[i, 'close']
                exit_cost = 0
                hold_period += 1
                daily_return = exit_price/entry_price -1
                trade_return = exit_price/entry_position_price -1
                results.append([datetime_start, datetime_end, transaction, entry_position_price, entry_price, exit_price, daily_return, trade_return, entry_cost, exit_cost])
                # entry price for the next row becomes close of the current one
                datetime_start = datetime_end
                entry_price = self.trades_and_close.loc[i, 'close']
                entry_cost = 0
                i += 1
                continue

            # change in open position
            if transaction != self.trades_and_close.loc[i, 'transaction']:
                datetime_end = self.trades_and_close.loc[i, 'datetime']
                # close the position
                
                exit_price = self.trades_and_close.loc[i, 'close']
                entry_price = self.trades_and_close.loc[i-1, 'close']
                daily_return = exit_price/entry_price -1
                trade_return = exit_price/entry_position_price -1
                if hold_period == 0:
                    entry_cost = self.cost
                else:
                    entry_cost = 0
                exit_cost = self.cost
                results.append([datetime_start, datetime_end, transaction, entry_position_price, entry_price, exit_price, daily_return, trade_return, entry_cost, exit_cost])

                transaction = self.trades_and_close.loc[i, 'transaction']
                if transaction in ['buy','sell']:
                    datetime_start = self.trades_and_close.loc[i, 'datetime']
                    entry_price = self.trades_and_close.loc[i, 'close']
                    entry_position_price = entry_price 
                    daily_return = 0
                    entry_cost = self.cost
                    exit_cost = 0
                    hold_period = 0
                i += 1
                continue


        daily_returns = pd.DataFrame(results, columns = ['datetime_start', 'datetime_end', 'transaction', 'entry_position_price', 'entry_price', 'exit_price','daily_return', 'trade_return', 'entry_cost', 'exit_cost'])
        daily_returns['trade_return'] = np.where(daily_returns['transaction']== 'sell', -daily_returns['trade_return'], daily_returns['trade_return'])
                    
        self.daily_returns = daily_returns
    
    def get_daily_returns(self):

        if self.triple_barrier:
            self.get_daily_returns_triple_barrier()
        else:
            self.get_daily_returns_one_step()

    def get_daily_returns_triple_barrier(self):
        '''
        This function calculates daily returns of the strategy based on transaction and close prices
        '''

        results = []
        transaction = 'No trade'
        i = 0

        while i <= self.trades_and_close.index.max():

            if transaction == 'No trade':
                if self.trades_and_close.loc[i, 'transaction'] == 'No trade':
                    i += 1
                    continue
                else:
                    datetime_start = self.trades_and_close.loc[i, 'datetime']
                    transaction = self.trades_and_close.loc[i, 'transaction']
                    entry_price = self.trades_and_close.loc[i, 'close']
                    entry_position_price = entry_price 
                    exit_position_price = self.trades_and_close.loc[i, 'exit_price']
                    barrier_touched = self.trades_and_close.loc[i, 'barrier_touched']

                    hold_period = 0
                    i += 1
                    continue
                    # results.append([transaction, entry_price, exit_price, barrier_touched, _return, entry_cost, exit_cost, hold_period])
            
            # no change in open position
            if transaction == self.trades_and_close.loc[i, 'transaction'] and transaction != 'No trade':
                # this means that this is end of the day record
                if self.trades_and_close.loc[i, 'barrier_touched'] is None:
                    datetime_end = self.trades_and_close.loc[i, 'datetime']
                    if hold_period == 0:
                        entry_cost = self.cost
                    else:
                        entry_cost = 0
                    # finding a day in which take profit or stop loss was triggered but it can't be the same day 
                    if (((barrier_touched == 'top' and self.trades_and_close.loc[i, 'high'] > exit_position_price) or 
                    (barrier_touched == 'bottom' and self.trades_and_close.loc[i, 'low'] < exit_position_price)) and 
                    (datetime_end.date() > datetime_start.date())):
                        exit_price = exit_position_price
                        exit_cost = self.cost
                        daily_return = exit_price/entry_price -1
                        trade_return = exit_price/entry_position_price -1
                        results.append([datetime_start, datetime_end, transaction, entry_position_price, entry_price, exit_price, daily_return, trade_return, entry_cost, exit_cost])
                        transaction = 'No trade'
                        # if that situation occured, transaction is closed and we have to wait for next bar - jump to it
                        try:
                            i = min(self.trades_and_close[(self.trades_and_close['datetime']>datetime_end) & (self.trades_and_close['barrier_touched'].notna())].index)
                        # there might no more rows
                        except ValueError:
                            # make sure no more transactions will be made
                            self.trades_and_close.loc[self.trades_and_close['datetime']>datetime_end, 'transaction'] = 'No trade'
                            i += 1
                            continue
                    else:
                        exit_price = self.trades_and_close.loc[i, 'close']
                        exit_cost = 0
                        hold_period += 1
                        daily_return = exit_price/entry_price -1
                        trade_return = exit_price/entry_position_price -1
                        results.append([datetime_start, datetime_end, transaction, entry_position_price, entry_price, exit_price, daily_return, trade_return, entry_cost, exit_cost])
                        # entry price for the next row becomes close of the current one
                        datetime_start = datetime_end
                        entry_price = self.trades_and_close.loc[i, 'close']
                        entry_cost = 0
                        i += 1
                    continue
                    
                # if current row is intraday record we have to close it as this would be automatic take profit or stop loss
                else:
                    datetime_end = self.trades_and_close.loc[i, 'datetime']
                    if hold_period == 0:
                        entry_cost = self.cost
                    else:
                        entry_cost = 0
                    exit_price = exit_position_price
                    daily_return = exit_price/entry_price -1
                    trade_return = exit_price/entry_position_price -1
                    exit_cost = self.cost
                    results.append([datetime_start, datetime_end, transaction, entry_position_price, entry_price, exit_price, daily_return, trade_return, entry_cost, exit_cost])
                    # open new position in the same direction
                    datetime_start = datetime_end
                    entry_price = self.trades_and_close.loc[i, 'close']
                    entry_position_price = entry_price 
                    exit_position_price = self.trades_and_close.loc[i, 'exit_price']
                    barrier_touched = self.trades_and_close.loc[i, 'barrier_touched']
                    hold_period = 0
                    i += 1
                    continue

                
            # change in open position
            if transaction != self.trades_and_close.loc[i, 'transaction']:
                datetime_end = self.trades_and_close.loc[i, 'datetime']
                # close the position
                
                exit_price = exit_position_price
                entry_price = self.trades_and_close.loc[i-1, 'close']
                daily_return = exit_price/entry_price -1
                trade_return = exit_price/entry_position_price -1
                if hold_period == 0:
                    entry_cost = self.cost
                else:
                    entry_cost = 0
                exit_cost = self.cost
                results.append([datetime_start, datetime_end, transaction, entry_position_price, entry_price, exit_price, daily_return, trade_return, entry_cost, exit_cost])


                transaction = self.trades_and_close.loc[i, 'transaction']
                if transaction in ['buy','sell']:
                    datetime_start = self.trades_and_close.loc[i, 'datetime']
                    entry_price = self.trades_and_close.loc[i, 'close']
                    entry_position_price = entry_price 
                    exit_position_price = self.trades_and_close.loc[i, 'exit_price']
                    barrier_touched = self.trades_and_close.loc[i, 'barrier_touched']
                    daily_return = 0
                    entry_cost = self.cost
                    exit_cost = 0
                    hold_period = 0
                    i += 1
                    continue

        # close last transaction
        if transaction != 'No trade':
            datetime_end = pd.to_datetime(self.trades_and_close['barrier_touched_date']).max()
            if hold_period == 0:
                entry_cost = self.cost
            else:
                entry_cost = 0
            exit_price = exit_position_price
            daily_return = exit_price/entry_price -1
            trade_return = exit_price/entry_position_price -1
            exit_cost = self.cost
            results.append([datetime_start, datetime_end, transaction, entry_position_price, entry_price, exit_price, daily_return, trade_return, entry_cost, exit_cost])
            i += 1


        daily_returns = pd.DataFrame(results, columns = ['datetime_start', 'datetime_end', 'transaction', 'entry_position_price', 'entry_price', 'exit_price','daily_return', 'trade_return', 'entry_cost', 'exit_cost'])
        daily_returns['trade_return'] = np.where(daily_returns['transaction']== 'sell', -daily_returns['trade_return'], daily_returns['trade_return'])
                    
        self.daily_returns = daily_returns

    def format_daily_returns(self):

        for i in self.daily_returns.index:
            if self.daily_returns.loc[i, 'entry_cost'] > 0:
                if i == 0:
                    portfolio = 100 * (1-self.daily_returns.loc[i, 'entry_cost'])
                else:
                    portfolio = self.daily_returns.loc[i-1, 'result'] * (1-self.daily_returns.loc[i, 'entry_cost'])
                self.daily_returns.loc[i, 'starting_portfolio'] = portfolio

            self.daily_returns.loc[i, 'result'] = portfolio * (1+self.daily_returns.loc[i, 'trade_return']) * (1-self.daily_returns.loc[i, 'exit_cost'])
            if i > 0:
                self.daily_returns['daily_portfolio_return'] = self.daily_returns['result'].pct_change()

        self.daily_returns.loc[0, 'daily_portfolio_return'] = self.daily_returns.loc[0, 'result'] / 100 -1

        logger.info('End result in transformed df {}'.format(self.daily_returns['result'].iloc[-1]))
        logger.info('End result in original df {}'.format(self.trades['budget'].iloc[-1]))

        # subtract 100 to make gain/loss
        self.metrics['1. gain_loss'].append(self.daily_returns['result'].iloc[-1] - 100)

    def compute_financial_performance(self):
        '''
        This function calculates financial performance of the strategy
        '''
        risk_free_rate = self.risk_free_rate/365 # crypto works whole year
        # Max drawdown
        max_drawdown = self.max_drawdown_on_portfolio(self.daily_returns['result'])
        self.metrics['6. max_drawdown'].append(max_drawdown)
        # to compute sharpe ratio add 0 returns for days without trades
        # get value of portfolio at the end of each day
        last = self.daily_returns.groupby(self.daily_returns['datetime_end'].dt.date).agg({'result': 'last'}).reset_index()
        # convert datetime_end to date
        last['datetime_end'] = pd.to_datetime(last['datetime_end'])
        # add starting date if not present - it shouldn't be - because it is one day before first trade
        if self.start_date < self.daily_returns['datetime_end'].min().date():
            last = pd.concat([pd.DataFrame({'datetime_end': [self.start_date], 'result': [100]}), last], ignore_index=True)
        # add last date if not present - it shouldn't be - because it is one day after last trade
        if self.end_date > self.daily_returns['datetime_end'].max().date():
            last = pd.concat([last, pd.DataFrame({'datetime_end': [self.end_date], 'result': np.nan })], ignore_index=True)
        last['datetime_end'] = pd.to_datetime(last['datetime_end'])
        # #set datetime_end as index
        last.set_index('datetime_end', inplace=True)
        # # fill missing days
        last = last.resample('D').ffill()
        # # calculate daily returns
        last['daily_return'] = last['result'].pct_change()
        # # fill first observation assuming starting capital of 100
        # last.loc[last.index[0], 'daily_return'] = last.loc[last.index[0], 'result'] / 100 -1
        # # calculate sharpe ratio
        # remove artifically added first and last row
        last = last.iloc[1:-1]
        sharpe_ratio = (last['daily_return'].mean() - risk_free_rate) / last['daily_return'].std()
        annualized_sharpe_ratio = (365**0.5) * sharpe_ratio
        self.metrics['4. sharpe_ratio'].append(annualized_sharpe_ratio)
        # sortino ratio
        downside_returns = [r for r in last['daily_return'] if r < 0]
        expected_return = np.mean(last['daily_return'])
        downside_std = np.std(downside_returns)
        sortino_ratio = expected_return / downside_std
        annualized_sortino_ratio = (365**0.5) * sortino_ratio
        self.metrics['5. sortino_ratio'].append(annualized_sortino_ratio)
        # trade duration in minutes
        self.daily_returns['duration'] = (self.daily_returns['datetime_end'] - self.daily_returns['datetime_start']).dt.total_seconds()/60
        # share of time active
        self.metrics['7. share_of_time_active'].append(self.daily_returns['duration'].sum() / ((self.end_date - self.start_date).total_seconds()/60) * 100)
        self.metrics['8. no. of trades'].append(self.daily_returns.loc[self.daily_returns['entry_cost'] > 0].shape[0])
  

    def max_drawdown_on_portfolio(self,portfolio_values):
        """
        Calculate the maximum drawdown of a portfolio's value evolution.

        Args:
            portfolio_values (list or numpy array): List of portfolio values over time.

        Returns:
            float: Maximum drawdown as a percentage.
        """
        if len(portfolio_values) == 0:
            return 0.0
        # make sure portfolio values start with 100
        portfolio_start = pd.Series([100])
        portfolio_values = pd.concat([portfolio_start, portfolio_values]).reset_index(drop=True)
        peak = portfolio_values[0]
        max_drawdown_percentage = 0.0
        current_drawdown = 0.0

        for value in portfolio_values:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                if drawdown > current_drawdown:
                    current_drawdown = drawdown
            if current_drawdown > max_drawdown_percentage:
                max_drawdown_percentage = current_drawdown

        return max_drawdown_percentage * 100.0
    
    def average_metrics(self):
        self.metrics_avg = {}
        for key in self.metrics.keys():
            self.metrics_avg[key] = round(np.nanmean(self.metrics[key]),2)
        logger.info('Average metrics: {}'.format(self.metrics_avg))
        metrics_df = pd.DataFrame.from_dict(self.metrics_avg,  orient='index', columns=['value']).reset_index()
        metrics_df.sort_values(by='index', inplace=True)
        metrics_df.loc[metrics_df['index']=='8. no. of trades', 'value'] = round(metrics_df.loc[metrics_df['index']=='8. no. of trades', 'value'], 0)
        postive_trades =  round(metrics_df.loc[metrics_df['index']=='2. correct_transactions', 'value'].values[0]/100 * metrics_df.loc[metrics_df['index']=='8. no. of trades', 'value'].values[0],0)
        no_of_trades = metrics_df.loc[metrics_df['index']=='8. no. of trades', 'value'].values[0]
        p_value = self.get_p_value(postive_trades, no_of_trades)
        p_value_row = pd.DataFrame({'index': ['9. p_value'], 'value': [p_value]})
        metrics_df = pd.concat([metrics_df, p_value_row], ignore_index=True)
        metrics_df.to_csv(output_folder+ '/'+ self.name + '.csv', index=False)
        # round without decimals for no. of trades
    
    @staticmethod
    def get_p_value(postive_trades, no_of_trades):
        '''
        This function calculates p-value for binomial test
        '''
        p_value = binom_test(postive_trades, no_of_trades, 0.5, alternative='greater')
        return p_value


    def run(self):
        for seed in self.seeds:
            self.backtest = pd.concat(self.load_backtest_data(seed))
            self.backtest.sort_values(by=['datetime'], inplace=True)
            #check if any trade was made
            if self.backtest[self.backtest['transaction']!='No trade'].empty:
                logger.info('No trades made')
                self.metrics['1. gain_loss'].append(0)
                self.metrics['2. correct_transactions'].append(np.nan)
                self.accuracy_by_threshold(self.backtest,0.5, 0.5, type='3. accuracy')
                self.metrics['4. sharpe_ratio'].append(np.nan)
                self.metrics['5. sortino_ratio'].append(np.nan)
                self.metrics['6. max_drawdown'].append(0)
                self.metrics['7. share_of_time_active'].append(0)
                self.metrics['8. no. of trades'].append(0)
                continue
            self.format_backtest_results()
            self.accuracy_by_threshold(self.backtest,0.5, 0.5, type='3. accuracy')
            self.accuracy_by_threshold(self.trades,0.4, 0.6, type='2. correct_transactions')
            self.combine_trades_and_daily_close()
            self.get_daily_returns()
            self.format_daily_returns()
            self.compute_financial_performance()
        self.average_metrics()
    

if __name__ == "__main__":

    logger = get_logger()
    performance_evaluator = PerformanceEvaluator(backtest_folder, daily_records)
    performance_evaluator.run()


