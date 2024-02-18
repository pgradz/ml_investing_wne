# %%
import random

import mlflow.keras
import numpy as np
import tensorflow as tf

from ml_investing_wne import config
from ml_investing_wne.experiment_factory import (create_asset,
                                                 experiment_factory)
from ml_investing_wne.utils import get_logger

random.seed(config.seed)
np.random.seed(config.seed)
tf.random.set_seed(config.seed)

# %%
import pandas as pd

pd.set_option('display.max_columns', None)
import numpy as np

# %%
config.currency = 'MATICUSDT'

# %%
asset = create_asset()
experiment = experiment_factory(asset).get_experiment()

# %%
experiment.df.head(2)

# %%
config.currency = 'BTCUSDT'
btc = create_asset()
experiment_btc = experiment_factory(btc).get_experiment()
experiment_btc.df.rename(columns={'close':'close_btc', 'volume':'volume_btc', 'roc_1': 'roc_1_btc'}, inplace=True)
# experiment.df =  experiment.df[['close', 'volume', 'roc_1','y_pred']].merge(experiment_btc.df[['close_btc', 'volume_btc','roc_1_btc']], left_index=True, right_index=True)
experiment.df =  experiment.df.merge(experiment_btc.df[['close_btc', 'volume_btc','roc_1_btc']], left_index=True, right_index=True)
experiment.df.head(1)

# %%
config.currency = 'ETHUSDT'
eth = create_asset()
experiment_eth = experiment_factory(eth).get_experiment()
experiment_eth.df.rename(columns={'close':'close_eth', 'volume':'volume_eth', 'roc_1': 'roc_1_eth'}, inplace=True)
# experiment.df = experiment.df[['close', 'volume', 'roc_1','y_pred', 'close_btc', 'volume_btc','roc_1_btc']].merge(experiment_eth.df[['close_eth', 'volume_eth','roc_1_eth']], left_index=True, right_index=True)
experiment.df = experiment.df.merge(experiment_eth.df[['close_eth', 'volume_eth','roc_1_eth']], left_index=True, right_index=True)
experiment.df.head(1)

# %%
config.currency = 'SOLUSDT'
sol = create_asset()
experiment_sol = experiment_factory(sol).get_experiment()
experiment_sol.df.head(1)


# %%
# experiment_sol.df =  experiment_sol.df[['close', 'volume', 'roc_1','y_pred']].merge(experiment_btc.df[['close_btc', 'volume_btc','roc_1_btc']], left_index=True, right_index=True)
experiment_sol.df =  experiment_sol.df.merge(experiment_btc.df[['close_btc', 'volume_btc','roc_1_btc']], left_index=True, right_index=True)
# experiment_sol.df = experiment_sol.df[['close', 'volume', 'roc_1','y_pred', 'close_btc', 'volume_btc','roc_1_btc']].merge(experiment_eth.df[['close_eth', 'volume_eth','roc_1_eth']], left_index=True, right_index=True)
experiment_sol.df = experiment_sol.df.merge(experiment_eth.df[['close_eth', 'volume_eth','roc_1_eth']], left_index=True, right_index=True)
experiment_sol.df.head(1)

# %%
config.currency = 'LTCUSDT'
ltc = create_asset()
experiment_ltc = experiment_factory(ltc).get_experiment()
experiment_ltc.df.head(1)

# %%
# experiment_ltc.df =  experiment_ltc.df[['close', 'volume', 'roc_1','y_pred']].merge(experiment_btc.df[['close_btc', 'volume_btc','roc_1_btc']], left_index=True, right_index=True)
# experiment_ltc.df = experiment_ltc.df[['close', 'volume', 'roc_1','y_pred','close_btc', 'volume_btc','roc_1_btc']].merge(experiment_eth.df[['close_eth', 'volume_eth','roc_1_eth']], left_index=True, right_index=True)
experiment_ltc.df =  experiment_ltc.df.merge(experiment_btc.df[['close_btc', 'volume_btc','roc_1_btc']], left_index=True, right_index=True)
experiment_ltc.df = experiment_ltc.df.merge(experiment_eth.df[['close_eth', 'volume_eth','roc_1_eth']], left_index=True, right_index=True)
experiment_ltc.df.head(1)

# %%
experiment.train_test_val_split()

# %%
experiment_sol.train_test_val_split()

# %%
experiment_ltc.train_test_val_split()

# %%
experiment.X = np.concatenate([experiment.X, experiment_sol.X, experiment_ltc.X])
experiment.X_val = np.concatenate([experiment.X_val, experiment_sol.X_val, experiment_ltc.X_val])
experiment.X_test = np.concatenate([experiment.X_test, experiment_sol.X_test, experiment_ltc.X_test])

experiment.y_cat = np.concatenate([experiment.y_cat, experiment_sol.y_cat, experiment_ltc.y_cat])
experiment.y_val_cat = np.concatenate([experiment.y_val_cat, experiment_sol.y_val_cat, experiment_ltc.y_val_cat])
experiment.y_test_cat = np.concatenate([experiment.y_test_cat, experiment_sol.y_test_cat, experiment_ltc.y_test_cat])

experiment.y_test = np.concatenate([experiment.y_test, experiment_sol.y_test, experiment_ltc.y_test])


# %% [markdown]
# 

# %%
print(experiment.X.shape)

# %%


# %%
experiment.train_model()

# %%
experiment.evaluate_model()

# %%
from sklearn.metrics import precision_recall_curve

# %%
y_pred = experiment.model.predict(experiment.X_test)
y_pred_class = y_pred.argmax(axis=-1)
y_pred_class

# %%
y_pred[:,1]

# %%
precision, recall, thresholds = precision_recall_curve(experiment.y_test, y_pred[:,1])

# %%
precision[9500]

# %%
thresholds[9500]

# %%
import matplotlib.pyplot as plt

plt.fill_between(recall, precision)
plt.ylabel("Precision")
plt.xlabel("Recall")
plt.title("Train Precision-Recall curve")

# %%



