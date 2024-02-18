import datetime
import importlib
import os
import random

import joblib
import matplotlib.pyplot as plt
import mlflow.keras
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
from plotly.subplots import make_subplots
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

import ml_investing_wne.config as config
from ml_investing_wne.data_engineering.load_data import get_hist_data
from ml_investing_wne.data_engineering.prepare_dataset import \
    prepare_processed_dataset
from ml_investing_wne.helper import get_final_model_path, load_test_dates
from ml_investing_wne.train_test_val_split import train_test_val_split
from ml_investing_wne.utils import get_logger

seed = 12345
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

logger = get_logger()

df = get_hist_data(currency=config.currency)
df = prepare_processed_dataset(df=df)
X, y, X_val, y_val, X_test, y_test, y_cat, y_val_cat, y_test_cat, train = train_test_val_split(df)

model_name = get_final_model_path()
model = load_model(model_name)

plot_model(model, to_file=os.path.join(config.package_directory, 'models', 'model_plot_transformer.png'), show_shapes=True,
            show_layer_names=True)

obs = 418
model.layers[0]

layer_0_output = model.layers[0](X_test[obs:obs+1,:,:])
layer_1_output = model.layers[1](X_test[obs:obs+1,:,:])

obs_x = X_test[obs:obs+1,:,:]