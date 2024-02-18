import os
import warnings
from datetime import datetime

from pandas.errors import SettingWithCopyWarning

import ml_investing_wne.config as config
from ml_investing_wne.performance_evaluator import PerformanceEvaluator
from ml_investing_wne.utils import get_logger

logger = get_logger()
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def get_modified_directories(root_dir):
    modified_dirs = []
    for root, dirs, files in os.walk(root_dir):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            modification_time = os.path.getmtime(dir_path)
            if modification_time > datetime(2024, 2, 9, 0, 0, 0).timestamp() and \
            (dir.startswith('ETH') or dir.startswith('BTC') or dir.startswith('ADA') or dir.startswith('LTC')):
                modified_dirs.append(dir)
    return modified_dirs


if __name__ == "__main__":
    root_dir = "/root/ml_investing_wne/src/ml_investing_wne/models"
    #root_dir = "/root/FEDformer/results"
    daily_records_eth = os.path.join(config.processed_data_path, f'binance_ETHUSDT', 'time_aggregated_1440min.csv')
    daily_records_btc = os.path.join(config.processed_data_path, f'binance_BTCUSDT', 'time_aggregated_1440min.csv')
    daily_records_ada = os.path.join(config.processed_data_path, f'binance_ADAUSDT', 'time_aggregated_1440min.csv')
    daily_records_ltc = os.path.join(config.processed_data_path, f'binance_LTCUSDT', 'time_aggregated_1440min.csv')
    daily_records_link = os.path.join(config.processed_data_path, f'binance_LINKUSDT', 'time_aggregated_1440min.csv')
    
    modified_dirs = get_modified_directories(root_dir)
    print("Modified directories:")
    for dir in modified_dirs:
        print(dir)
        if dir.startswith('ETH'):
            daily_records = daily_records_eth
        elif dir.startswith('BTC'):
            daily_records = daily_records_btc
        elif dir.startswith('ADA'):
            daily_records = daily_records_ada
        elif dir.startswith('LTC'):
            daily_records = daily_records_ltc
        elif dir.startswith('LINK'):
            daily_records = daily_records_link
        else:  
            continue
        
        performance_evaluator = PerformanceEvaluator(os.path.join(root_dir, dir), daily_records, cost=config.cost)
        performance_evaluator.run()