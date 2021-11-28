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



'''
code to test RSI. It's getting slighlty different results than pandas_ta

delta = df['close'].diff()
up = delta.clip(lower=0)
down = -1*delta.clip(upper=0)
ema_up = up.ewm(com=14, adjust=False).mean()
ema_down = down.ewm(com=14, adjust=False).mean()
rs = ema_up/ema_down
df['RSI'] = 100 - (100/(1 + rs))

'''