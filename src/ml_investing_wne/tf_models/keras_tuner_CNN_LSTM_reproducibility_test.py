import os
import random
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from keras_tuner.tuners import hyperband
import keras_tuner as kt
import logging

from ml_investing_wne import config
from ml_investing_wne.utils import get_logger
from ml_investing_wne.experiment_factory import create_asset, experiment_factory

random.seed(config.seed)
np.random.seed(config.seed)
tf.random.set_seed(config.seed)

tf.keras.backend.set_image_data_format("channels_last")
logger = logging.getLogger(__name__)

asset = create_asset()
experiment= experiment_factory(asset).get_experiment()
experiment.train_test_val_split()

def my_custom_metric(y_true, y_pred):
    """This function calculates the average of the validation and training loss."""
    # Get the validation loss
    validation_loss = tf.keras.metrics.Mean(name='val_loss')
    validation_loss.update_state(y_true, y_pred)

    # Get the training loss
    training_loss = tf.keras.metrics.Mean(name='train_loss')
    training_loss.update_state(y_true, y_pred)

    # Calculate the average of the validation and training loss
    average_loss = (validation_loss + training_loss) / 2

    return average_loss

# def build_model(input_shape, n_feature_maps, kernel_size_1, kernel_size_2, kernel_size_3, dropout,lstm_neurons, learning_rate, nb_classes=2):

#         input_layer = keras.layers.Input(input_shape)

#         # BLOCK 1
#         conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=kernel_size_1, padding='same')(input_layer)
#         conv_x = keras.layers.BatchNormalization()(conv_x)
#         conv_x = keras.layers.Activation('relu')(conv_x)

#         conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=kernel_size_2, padding='same')(conv_x)
#         conv_y = keras.layers.BatchNormalization()(conv_y)
#         conv_y = keras.layers.Activation('relu')(conv_y)

#         conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=kernel_size_3, padding='same')(conv_y)
#         conv_z = keras.layers.BatchNormalization()(conv_z)

#         # expand channels for the sum
#         shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
#         shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

#         output_block_1 = keras.layers.add([shortcut_y, conv_z])
#         output_block_1 = keras.layers.Activation('relu')(output_block_1)
#         output_block_1 = keras.layers.Dropout(dropout)(output_block_1, training=True)

#         # FINAL
#         lstm_layer = keras.layers.LSTM(lstm_neurons)(output_block_1)

#         output_layer = keras.layers.Dense(nb_classes, activation='softmax')(lstm_layer)
#         model = keras.models.Model(inputs=input_layer, outputs=output_layer)
#         model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=learning_rate ) ,
#                     metrics=['accuracy'])
#         return model

# model = build_model(input_shape=(96,41),n_feature_maps= 24, kernel_size_1= 4,kernel_size_2= 4, kernel_size_3= 1,dropout= 0.2, lstm_neurons= 20, learning_rate= 0.0001)
# experiment.set_model(model)
# experiment.model.summary()
# experiment.train_model()
experiment.hyperparameter_tunning()

# print(config.currency)
# print(config.RUN_SUBTYPE)
# print(config.seq_stride)
class MyHyperModel():

    def __init__(self, input_shape, train_dataset, val_dataset, 
                 currency=config.currency, seq_len=config.seq_len, 
                 RUN_SUBTYPE=config.RUN_SUBTYPE, model=config.model, 
                 seed=config.seed):
        self.input_shape = input_shape
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.currency = currency
        self.seq_len = seq_len
        self.RUN_SUBTYPE = RUN_SUBTYPE
        self.model = model
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def build_model_for_tunning(self, hp, nb_classes=2):

        n_feature_maps = hp.Int('n_feature_maps', min_value=8, max_value=64, step=4)
        input_layer = keras.layers.Input(self.input_shape)

        # BLOCK 1
        kernel_size_1 = hp.Int('kernel_size_1', min_value=1, max_value=12, step=1)
        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=kernel_size_1, padding='same')(input_layer)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        kernel_size_2 = hp.Int('kernel_size_2', min_value=1, max_value=12, step=1)
        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=kernel_size_2, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        kernel_size_3 = hp.Int('kernel_size_3', min_value=1, max_value=12, step=1)
        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=kernel_size_3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)
        dropout = hp.Choice('dropout', values=[0.1, 0.15, 0.2, 0.25, 0.3])
        output_block_1 = keras.layers.Dropout(dropout)(output_block_1, training=True)

        # FINAL
        lstm_neurons = hp.Int('lstm_neurons', min_value=8, max_value=64, step=4)
        lstm_layer = keras.layers.LSTM(lstm_neurons)(output_block_1)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(lstm_layer)
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate ) ,
                    metrics=[ my_custom_metric])
        return model


    def run_tuner(self):

        self.tuner = kt.Hyperband(self.build_model_for_tunning,
                            objective=my_custom_metric,
                            max_epochs=10,
                            hyperband_iterations=1, 
                            factor=3,
                            directory='keras_tuner',
                            project_name=f'{self.currency}_{self.seq_len}_{self.RUN_SUBTYPE}_{self.model}')


        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        self.tuner.search(self.train_dataset, epochs=10,validation_data=self.val_dataset, callbacks=[stop_early])

        self.best_hps=self.tuner.get_best_hyperparameters(num_trials=1)[0]

        logger.info(f"""
        The hyperparameter search is complete. 
        The optimal dropout is {self.best_hps.get('dropout')} 
        The optimal n_feature_maps is {self.best_hps.get('n_feature_maps')}
        The optimal kernel_size_1 is {self.best_hps.get('kernel_size_1')}
        The optimal kernel_size_2 is {self.best_hps.get('kernel_size_2')}
        The optimal kernel_size_3 is {self.best_hps.get('kernel_size_3')}
        The optimal lstm_neurons is {self.best_hps.get('lstm_neurons')}
        and the optimal learning rate for the optimizer
        is {self.best_hps.get('learning_rate')}.
        """)


    def get_best_model(self):

        model = self.tuner.hypermodel.build(self.best_hps)
        # model = self.reconstruct_model(self.best_hps.get('n_feature_maps'), self.best_hps.get('kernel_size_1'), self.best_hps.get('kernel_size_2'), 
        #                                self.best_hps.get('kernel_size_3'), self.best_hps.get('dropout'),self.best_hps.get('lstm_neurons'), 
        #                                self.best_hps.get('learning_rate'))

        return model



  

