import datetime

import matplotlib.pyplot as plt
import pandas as pd
import plotly.figure_factory as ff

import ml_investing_wne.config as config
from ml_investing_wne.data_engineering.prepare_dataset import \
    prepare_processed_dataset
from ml_investing_wne.hist_data.helper import get_hist_data

currency = 'USDCHF'
df = get_hist_data(currency=currency)
df = prepare_processed_dataset(df=df)

hist_data = []
inside_cost = []

# 15 minutes data
for i in [1,2]:
    df['close_diff'] = (df['close'].shift(-i) - df['close']) * 10000
    print('Frequency {} '.format(i))
    print(df.close_diff.describe(percentiles=[0.25, 0.33, 0.5, 0.66, 0.75]))

#30 minutes data
for i in [1,2,4,8,16,24]:
    df['close_diff'] = (df['close'].shift(-i) - df['close']) * 10000
    hist_data.append(df.loc[(df['close_diff'].notna()) & (abs(df['close_diff'])<50)]['close_diff'].to_list())
    inside_cost.append(str(round(df.loc[(df['close_diff'].notna()) & (abs(df['close_diff'])<=2)].shape[0]/df.loc[df['close_diff'].notna()].shape[0],3)*100)+'%')
    print('Frequency {} H'.format(i))
    print(df.close_diff.describe(percentiles=[0.25, 0.33, 0.5, 0.66, 0.75]))
print(inside_cost)

#60 minutes data
for i in [1,2,4,8,12]:
    df['close_diff'] = (df['close'].shift(-i) - df['close']) * 10000
    hist_data.append(df.loc[(df['close_diff'].notna()) & (abs(df['close_diff'])<50)]['close_diff'].to_list())
    inside_cost.append(str(round(df.loc[(df['close_diff'].notna()) & (abs(df['close_diff'])<=2)].shape[0]/df.loc[df['close_diff'].notna()].shape[0],338)*100)+'%')
    print('Frequency {} H'.format(i))
    print(df.close_diff.describe(percentiles=[0.25, 0.33, 0.5, 0.66, 0.75]))

print(inside_cost)
df.reset_index(inplace=True)
df['close_diff'] = (df['close'].shift(-1) - df['close']) * 10000
df['datetime_waw'] = df['datetime'].dt.tz_localize('US/Eastern').dt.tz_convert(
    'Europe/Warsaw').dt.tz_localize(None)
df['hour_waw'] = df['datetime_waw'].dt.time
df.loc[abs(df['close_diff'])<50].boxplot(column = 'close_diff', by = 'hour_waw', figsize=(30,10))
# df.iloc[1+config.steps_ahead,3] / df.iloc[1,3]
df['change'] = [1 if y > 0 else 0 for y in df['close_diff']]
print(df.groupby('hour_waw')['change'].mean())

group_labels_new = []
group_labels = ['1h', '2h','4h','12h','24h']
for a, b in zip(group_labels, inside_cost):
    group_labels_new.append(a +' - ' + b)
fig = ff.create_distplot(hist_data, group_labels_new, show_hist=False, show_rug=False,)
fig.show()
