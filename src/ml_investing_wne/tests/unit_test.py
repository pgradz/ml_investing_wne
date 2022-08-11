import numpy as np
import pytest
import pandas as pd
import numpy as np
import datetime
from ml_investing_wne.helper import compute_profitability_classes
import ml_investing_wne.config as config

@pytest.fixture
def df_file():
    df_file = pd.read_csv('EURCHF_1h.csv', sep=';', decimal=',', parse_dates=['datetime'])
    #df_file = pd.read_csv('/Users/i0495036/Documents/sandbox/ml_investing_wne/ml_investing_wne/src/ml_investing_wne/tests/EURCHF_1h.csv', sep=';', decimal=',', parse_dates=['datetime'])
    return df_file

@pytest.fixture
def y_pred_file():
    y_pred_file = pd.read_csv('EURCHF_y_pred_1h.csv', sep=';', decimal=',')
    #y_pred_file = pd.read_csv('/Users/i0495036/Documents/sandbox/ml_investing_wne/ml_investing_wne/src/ml_investing_wne/tests/EURCHF_y_pred_1h.csv', sep=';', decimal=',')
    y_pred_file = np.array(y_pred_file)
    return y_pred_file

@pytest.fixture
def seq_lenstart_date():
    return datetime.datetime(2021,7,1,15,0,0)

@pytest.fixture
def end_date():
    return datetime.datetime(2021,12,24,14,0,0)

def test_compute_profitability_classes(df_file, y_pred_file, start_date, end_date):
    lower_bound = 0.4
    upper_bound = 0.6
    config.steps_ahead = 1
    budget, hits_ratio, share_of_time_active = \
    compute_profitability_classes(df_file, y_pred_file, start_date, end_date, lower_bound, upper_bound, time_waw_list=None)
    assert budget > 100
    assert round(budget,1) == 100.8

# excel computation would require more complicated formulas to be correct
# def test_compute_profitability_classes_5_steps_ahead(df_file, y_pred_file, start_date, end_date):
#     lower_bound = 0.4
#     upper_bound = 0.6
#     config.steps_ahead = 5
#     budget, hits_ratio, share_of_time_active = \
#     compute_profitability_classes(df_file, y_pred_file, start_date, end_date, lower_bound, upper_bound, time_waw_list=None)
#     assert budget < 100
#     assert round(budget,1) == 91
#
#
# df = pd.read_csv('/Users/i0495036/Documents/sandbox/ml_investing_wne/ml_investing_wne/src/ml_investing_wne/tests/EURCHF_1h.csv', sep=';', decimal=',', parse_dates=['datetime'])
# y_pred = np.array(pd.read_csv('/Users/i0495036/Documents/sandbox/ml_investing_wne/ml_investing_wne/src/ml_investing_wne/tests/EURCHF_y_pred_1h.csv', sep=';', decimal=','))
# date_start = datetime.datetime(2021,7,1,15,0,0)
# date_end = datetime.datetime(2021,12,24,14,0,0)
# config.steps_ahead = 5