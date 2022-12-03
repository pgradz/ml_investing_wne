import pandas as pd

df = pd.read_csv('/Users/i0495036/Downloads/xtb/EURCHF_60.csv')

df['datetime'] = pd.to_datetime(df['ctm'], unit='ms')
df['close'] = (df['open'] + df['close'])
df['high'] = (df['open'] + df['high'])
df['low'] = (df['open'] + df['low'])
df['y_pred'] = df['close'].shift(-1) / df['close']
df['y_pred'] = [1 if y > 1 else 0 for y in df['y_pred']]
# df['datetime_london'] = df['datetime'].dt.tz_convert('Europe/London').dt.tz_localize(None)
df['hour_london'] = df['datetime'].dt.time
naive_forecast_group = df.groupby('hour_london')['y_pred'].mean()

naive_predictor = round(naive_forecast_group).to_dict()
