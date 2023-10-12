from abc import ABC, abstractmethod
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

class MyHyperModelBase(ABC):

    def __init__(self, input_shape, train_dataset, val_dataset, 
                 project_name,
                 currency=config.currency, seq_len=config.seq_len, 
                 RUN_SUBTYPE=config.RUN_SUBTYPE, model=config.model, 
                 seed=config.seed):
        self.input_shape = input_shape
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.project_name = project_name
        self.currency = currency
        self.seq_len = seq_len
        self.RUN_SUBTYPE = RUN_SUBTYPE
        self.model = model
        self.seed = seed
    
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)


    def run_tuner(self):

        self.tuner = kt.Hyperband(self.build_model_for_tuning,
                            objective='val_accuracy',
                            max_epochs=20,
                            hyperband_iterations=1, 
                            factor=3,
                            directory='keras_tuner',
                            project_name=self.project_name)


        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        self.tuner.search(self.train_dataset, epochs=20,validation_data=self.val_dataset, callbacks=[stop_early])

    @abstractmethod
    def build_model_for_tuning(self, hp, nb_classes=2):
        pass
    
    @abstractmethod
    def get_best_model(self, model_index=0):
        pass

    @abstractmethod
    def get_best_unique_model(self, model_index=0):
        pass

    
        
