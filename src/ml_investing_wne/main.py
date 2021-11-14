import pandas as pd

from ml_investing_wne.data_engineering.truefx import import_truefx_csv, aggregate_time_window
import ml_investing_wne.config as config

df = pd.concat(import_truefx_csv(config.raw_data_path, config.currencies))
aggregate_time_window(df, '5min')
# TODO: aggregate_time_window will create intervals even if there was no data, add checks