import pandas as pd

from ml_investing_wne.data_engineering.truefx import import_truefx_csv
import ml_investing_wne.config as config

df = pd.concat(import_truefx_csv(config.raw_data_path, config.currencies, nrows=1))
