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
from ml_investing_wne.tf_models.keras_tuner_base import MyHyperModelBase


tf.keras.backend.set_image_data_format("channels_last")
logger = logging.getLogger(__name__)


class MyHyperModel(MyHyperModelBase):


    def build_model_for_tuning(self, hp, nb_classes=2):

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
                    metrics=['accuracy'])
        return model


    def get_best_model(self, model_index=0):

        self.best_hps=self.tuner.get_best_hyperparameters(num_trials=3)[model_index]

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


        model = self.tuner.hypermodel.build(self.best_hps)

        return model

  
    def return_top_n_models(self, n=3):
        '''
        Keras tuner get_best_hyperparameters contains a bug and doesn't always return models in the same order, hence this workaround
        '''

        models = []
        model_representation = []
        best_hps_all = self.tuner.get_best_hyperparameters(num_trials=30)
        for i in range(30):
            best_hps = best_hps_all[i]
            model_candidate = dict((k, best_hps[k]) for k in ['n_feature_maps', 'kernel_size_1', 'kernel_size_2','kernel_size_3','lstm_neurons','dropout','learning_rate'] if k in best_hps)
            if model_candidate not in model_representation:
                model_representation.append(model_candidate)
                models.append(self.tuner.hypermodel.build(best_hps))
                logger.info(f"""
                The optimal dropout is {best_hps.get('dropout')} 
                The optimal n_feature_maps is {best_hps.get('n_feature_maps')}
                The optimal kernel_size_1 is {best_hps.get('kernel_size_1')}
                The optimal kernel_size_2 is {best_hps.get('kernel_size_2')}
                The optimal kernel_size_3 is {best_hps.get('kernel_size_3')}
                The optimal lstm_neurons is {best_hps.get('lstm_neurons')}
                and the optimal learning rate for the optimizer
                is {best_hps.get('learning_rate')}.
                """)
            else:
                continue
            if len(model_representation) == n:
                break

        return models



    def get_best_unique_model(self, model_index=0):
        '''
        Sometimes top models are the same. With this function we can get unique models'''

        model_representation = []
        for i in range(model_index+1):

            best_hps = self.tuner.get_best_hyperparameters(num_trials=i+1)[i]
            model_representation.append(dict((k, best_hps[k]) for k in ['n_feature_maps', 'kernel_size_1', 'kernel_size_2','kernel_size_3','lstm_neurons','dropout','learning_rate'] if k in best_hps))

            if i == model_index:
                self.best_hps = best_hps
                model = self.tuner.hypermodel.build(self.best_hps)
                

        for j in range(len(model_representation)-1):
            if model_representation[j] != model_representation[-1]:
                continue
            else:
                logger.info(f""" found duplicated model, increasing model_index by 1""")
                model = self.get_best_unique_model(model_index=model_index+1)
                break

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

        self.model = model
        return  model



