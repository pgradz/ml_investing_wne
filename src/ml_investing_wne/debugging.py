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

# asset = create_asset()
# experiment = experiment_factory(asset).get_experiment()

config.currency = 'BTCUSDT'
btc = create_asset()
experiment_btc = experiment_factory(btc).get_experiment()
experiment_btc.train_test_val_split()
experiment_btc.train_model()
