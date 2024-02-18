import logging
import os
import random

import keras_tuner as kt
import numpy as np
import tensorflow as tf
from keras_tuner.tuners import hyperband
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

from ml_investing_wne import config
from ml_investing_wne.utils import get_logger

# from ml_investing_wne.experiment_factory import create_asset, experiment_factory

# random.seed(config.seed)
# np.random.seed(config.seed)
# tf.random.set_seed(config.seed)

tf.keras.backend.set_image_data_format("channels_last")
logger = logging.getLogger(__name__)

# asset = create_asset()
# experiment= experiment_factory(asset).get_experiment()
# experiment.train_test_val_split()

# print(config.currency)
# print(config.run_subtype)
# print(config.seq_stride)
class MyHyperModel():

    def __init__(self, input_shape, train_dataset, val_dataset, 
                 currency=config.currency, seq_len=config.seq_len, 
                 run_subtype=config.run_subtype, model=config.model, 
                 seed=config.seed):
        self.input_shape = input_shape
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.currency = currency
        self.seq_len = seq_len
        self.run_subtype = run_subtype
        self.model = model
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def build_model_for_tuning(self, hp, nb_classes=2):

        input_layer = keras.layers.Input(self.input_shape)

        lstm_neurons_1 = hp.Int('lstm_neurons_1', min_value=16, max_value=256, step=16)
        lstm_neurons_2 = hp.Int('lstm_neurons_2', min_value=16, max_value=256, step=16)
        dropout = hp.Choice('dropout', values=[0.1, 0.15, 0.2, 0.25, 0.3])

        lstm_layer_1 = keras.layers.LSTM(lstm_neurons_1, return_sequences=True)(input_layer)
        output_block_1 = keras.layers.Dropout(dropout)(lstm_layer_1, training=True)
        lstm_layer_2 = keras.layers.LSTM(lstm_neurons_2)(output_block_1)
        output_block_2 = keras.layers.Dropout(dropout)(lstm_layer_2, training=True)
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(output_block_2)
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate ) ,
                    metrics=['accuracy'])
        
        return model


    def run_tuner(self):

        self.tuner = kt.Hyperband(self.build_model_for_tuning,
                            objective='val_accuracy',
                            max_epochs=20,
                            hyperband_iterations=1, 
                            factor=3,
                            directory='keras_tuner',
                            project_name=f'{self.currency}_{self.seq_len}_{self.run_subtype}_{self.model}')


        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        self.tuner.search(self.train_dataset, epochs=20,validation_data=self.val_dataset, callbacks=[stop_early])


    def get_best_model(self, model_index=0):

        self.best_hps=self.tuner.get_best_hyperparameters(num_trials=3)[model_index]

        logger.info(f"""
        The hyperparameter search is complete. 
        The optimal dropout is {self.best_hps.get('dropout')} 
        The optimal lstm_neurons_1 is {self.best_hps.get('lstm_neurons_1')}
        The optimal lstm_neurons_2 is {self.best_hps.get('lstm_neurons_2')}
        and the optimal learning rate for the optimizer
        is {self.best_hps.get('learning_rate')}.
        """)


        model = self.tuner.hypermodel.build(self.best_hps)
        # model = self.reconstruct_model(self.best_hps.get('n_feature_maps'), self.best_hps.get('kernel_size_1'), self.best_hps.get('kernel_size_2'), 
        #                                self.best_hps.get('kernel_size_3'), self.best_hps.get('dropout'),self.best_hps.get('lstm_neurons'), 
        #                                self.best_hps.get('learning_rate'))

        return model

  

    def get_best_unique_model(self, model_index=0):
        '''
        Sometimes top models are the same. With this function we can get unique models'''

        model_representation = []
        for i in range(model_index+1):

            best_hps = self.tuner.get_best_hyperparameters(num_trials=i+1)[i]
            model_representation.append(dict((k, best_hps[k]) for k in ['lstm_neurons_1', 'lstm_neurons_2', 'kernel_size_2','dropout','learning_rate'] if k in best_hps))

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
        The optimal lstm_neurons_1 is {self.best_hps.get('lstm_neurons_1')}
        The optimal lstm_neurons_2 is {self.best_hps.get('lstm_neurons_2')}
        and the optimal learning rate for the optimizer
        is {self.best_hps.get('learning_rate')}.
        """)
        self.model = model
        return  model



