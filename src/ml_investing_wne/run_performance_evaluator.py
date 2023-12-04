import os
from datetime import datetime
import warnings
import ml_investing_wne.config as config
from ml_investing_wne.performance_evaluator import PerformanceEvaluator
from ml_investing_wne.utils import get_logger
from pandas.errors import SettingWithCopyWarning

logger = get_logger()
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def get_modified_directories(root_dir):
    modified_dirs = []
    for root, dirs, files in os.walk(root_dir):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            modification_time = os.path.getmtime(dir_path)
            if modification_time > datetime(2023, 10, 27, 0, 0, 0).timestamp() and (dir.startswith('ETH') or dir.startswith('BTC')):
                modified_dirs.append(dir)
    return modified_dirs


if __name__ == "__main__":
    root_dir = "/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/models"
    daily_records_eth = os.path.join(config.processed_data_path, f'binance_ETHUSDT', 'time_aggregated_1440min.csv')
    daily_records_btc = os.path.join(config.processed_data_path, f'binance_BTCUSDT', 'time_aggregated_1440min.csv')
    modified_dirs = get_modified_directories(root_dir)
    print("Modified directories:")
    for dir in modified_dirs:
        print(dir)
        if dir.startswith('ETH'):
            daily_records = daily_records_eth
        else:  
            daily_records = daily_records_btc
        
        performance_evaluator = PerformanceEvaluator(os.path.join('/Users/i0495036/Documents/sandbox/ml_investing_wne/src/ml_investing_wne/models',dir), daily_records)
        performance_evaluator.run()