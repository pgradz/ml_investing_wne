import pandas as pd
import datetime
from ml_investing_wne.data_engineering.prepare_dataset import prepare_processed_dataset
import ml_investing_wne.config as config
from ml_investing_wne.train_test_val_split import train_test_val_split
from ml_investing_wne.helper import confusion_matrix_plot, compute_profitability_classes, check_hours
from ml_investing_wne.hist_data.helper import import_hist_data_csv
import matplotlib.pyplot as plt


currency = 'EURCHF'
df = import_hist_data_csv(currency=config.currency)
df = prepare_processed_dataset(df=df)