import pandas as pd
import pandas.api.types as ptypes

from ml_investing_wne.data_engineering.truefx import import_truefx_csv
import ml_investing_wne.config as config


def test_import_truefx_csv():
    df = pd.concat(import_truefx_csv(config.raw_data_path_test, config.currencies, nrows=1))
    assert df.shape[0] > 0
    assert all(ptypes.is_numeric_dtype(df[col]) for col in ['bid', 'ask'])
    assert ptypes.is_string_dtype(df['currency'])
    assert ptypes.is_datetime64_any_dtype(df['datetime'])
