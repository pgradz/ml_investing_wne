from keras_tuner.tuners import hyperband
import datetime
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
import mlflow.keras
import importlib
import joblib
from sklearn.metrics import roc_auc_score, f1_score

import ml_investing_wne.config as config
from ml_investing_wne.data_engineering.load_data import get_hist_data
from ml_investing_wne.data_engineering.prepare_dataset import prepare_processed_dataset
from ml_investing_wne.train_test_val_split import train_test_val_split
from ml_investing_wne.helper import confusion_matrix_plot, compute_profitability_classes
from ml_investing_wne.utils import get_logger

logger = get_logger()

df = get_hist_data(currency=config.currency)
df = prepare_processed_dataset(df=df, features=False)
X, y, X_val, y_val, X_test, y_test, y_cat, y_val_cat, y_test_cat, train = train_test_val_split(df)


import os
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import ml_investing_wne.config as config
import keras_tuner as kt

# https://keras.io/examples/timeseries/timeseries_transformer_classification/

def transformer_encoder(inputs, head_size, num_heads, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    # x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

class PositionEmbeddingLayer(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super(PositionEmbeddingLayer, self).__init__(**kwargs)
        self.position_embedding_layer = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
 
    def call(self, inputs):        
        position_indices = tf.range(tf.shape(inputs)[1])
        embedded_indices = self.position_embedding_layer(position_indices)
        return inputs + embedded_indices
    
    def get_config(self):
        config = super().get_config().copy()
        return config

def build_model(hp,
    input_shape=(96,4),
    nb_classes=2
):
    inputs = keras.Input(shape=input_shape)
    x = PositionEmbeddingLayer(input_shape[0], input_shape[1])(inputs)
    dropout = hp.Choice('dropout', values=[0.1, 0.15, 0.2, 0.25])
    head_size = hp.Int('head_size', min_value=32, max_value=256, step=32)
    num_heads = hp.Int('num_heads', min_value=1, max_value=8, step=1)
    num_transformer_blocks = hp.Int('num_transformer_blocks', min_value=1, max_value=8, step=1)
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, dropout)

        
    mlp_units = [hp.Int('mlp_units', min_value=32, max_value=256, step=32)]
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(nb_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  metrics=['accuracy'])

    return model


tuner = hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(X, y_cat, epochs=50, validation_data=(X_val, y_val_cat), callbacks=[stop_early])

print(f"""
The hyperparameter search is complete. 
The optimal dropout is {best_hps.get('dropout')} 
The optimal head_size is {best_hps.get('head_size')}
The optimal num_heads is {best_hps.get('num_heads')}
The optimal num_transformer_blocks is {best_hps.get('num_transformer_blocks')}
The optimal mlp_units is {best_hps.get('mlp_units')}
and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")