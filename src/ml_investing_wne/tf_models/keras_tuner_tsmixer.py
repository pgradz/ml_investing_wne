# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of TSMixer."""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import ml_investing_wne.config as config
from tensorflow import keras
from keras_tuner.tuners import hyperband
import keras_tuner as kt
import logging

from ml_investing_wne import config
from ml_investing_wne.utils import get_logger

tf.keras.backend.set_image_data_format("channels_last")
logger = logging.getLogger(__name__)


def res_block(inputs, norm_type, activation, dropout, ff_dim):
    """Residual block of TSMixer."""

    norm = (
        layers.LayerNormalization
        if norm_type == 'L'
        else layers.BatchNormalization
    )   
    # Temporal Linear
    x = norm(axis=[-2, -1])(inputs)
    x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
    x = layers.Dense(x.shape[-1], activation=activation)(x)
    x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Input Length, Channel]
    x = layers.Dropout(dropout)(x)
    res = x + inputs    
    # Feature Linear
    x = norm(axis=[-2, -1])(res)
    x = layers.Dense(ff_dim, activation=activation)(
        x
    )  # [Batch, Input Length, FF_Dim]
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1])(x)  # [Batch, Input Length, Channel]
    x = layers.Dropout(dropout)(x)
    return x + res  


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


    def build_model_for_tuning(self, hp, nb_classes=2):
        """Build TSMixer model."""

        inputs = keras.layers.Input(self.input_shape)
        x = inputs  # [Batch, Input Length, Channel]
        
        n_block = hp.Int('n_block', min_value=1, max_value=3, step=1)
        ff_dim = hp.Int('ff_dim', min_value=16, max_value=128, step=16)
        dropout = hp.Choice('dropout', values=[0.1, 0.15, 0.2, 0.25, 0.3])
        norm_type = hp.Choice('norm_type', values=['B','L'])
        activation = hp.Choice('activation', values=['relu','gelu'])
        
        for _ in range(n_block):
            x = res_block(x, norm_type, activation, dropout, ff_dim)        

        gap_layer = layers.GlobalAveragePooling1D()(x)       
        output_layer = layers.Dense(nb_classes, activation='softmax')(gap_layer)
        model = tf.keras.models.Model(inputs=inputs, outputs=output_layer)
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                        metrics=['accuracy'])       
        return model
    
    def run_tuner(self):

        self.tuner = kt.Hyperband(self.build_model_for_tuning,
                            objective='val_accuracy',
                            max_epochs=20,
                            hyperband_iterations=1, 
                            factor=3,
                            directory='keras_tuner',
                            project_name=f'{self.currency}_{self.seq_len}_{self.RUN_SUBTYPE}_{self.model}')


        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        self.tuner.search(self.train_dataset, epochs=20,validation_data=self.val_dataset, callbacks=[stop_early])


    def get_best_model(self, model_index=0):

        self.best_hps=self.tuner.get_best_hyperparameters(num_trials=3)[model_index]

        logger.info(f"""
        The hyperparameter search is complete. 
        The optimal dropout is {self.best_hps.get('dropout')} 
        The optimal n_block is {self.best_hps.get('n_block')}
        The optimal norm_type is {self.best_hps.get('norm_type')}
        The optimal ff_dim is {self.best_hps.get('ff_dim')}
        The optimal activation is {self.best_hps.get('activation')}
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
            model_representation.append(dict((k, best_hps[k]) for k in ['dropout', 'n_block', 'norm_type','activation', 'ff_dim','learning_rate'] if k in best_hps))

            if i == model_index:
                self.best_hps = best_hps
                model = self.tuner.hypermodel.build(self.best_hps)
                

        for j in range(len(model_representation)-1):
            if model_representation[j] != model_representation[-1]:
                continue
            else:
                logger.info(f""" found duplicated model, increasing model_index by 1""")
                self.get_best_unique_model(model_index=model_index+1)
                break

        logger.info(f"""
        The hyperparameter search is complete. 
        The optimal dropout is {self.best_hps.get('dropout')} 
        The optimal n_block is {self.best_hps.get('n_block')}
        The optimal norm_type is {self.best_hps.get('norm_type')}
        The optimal ff_dim is {self.best_hps.get('ff_dim')}
        The optimal activation is {self.best_hps.get('activation')}
        and the optimal learning rate for the optimizer
        is {self.best_hps.get('learning_rate')}.
        """)

        return  model




# model = build_model(input_shape=(96, 40), nb_classes=2, norm_type='B', activation='relu', n_block=2, dropout=0.15, ff_dim=64,  target_slice=None)
# model.summary()
# plot_model(model, to_file=os.path.join(config.package_directory, 'models', 'model_plot_tsmixer.png'),
#            show_shapes=True, show_layer_names=True)
