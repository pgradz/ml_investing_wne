import os
import random
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import ml_investing_wne.config as config
import keras_tuner as kt
from ml_investing_wne.utils import get_logger
import logging


logger = logging.getLogger(__name__)

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


    def build_model_for_tuning(self, 
                               hp,
                               nb_classes=2
                               ):
        inputs = keras.Input(shape=self.input_shape)
        x = PositionEmbeddingLayer(self.input_shape[0], self.input_shape[1])(inputs)
        
        num_transformer_blocks = hp.Int('num_transformer_blocks', min_value=1, max_value=8, step=1)
        head_size = hp.Int('head_size', min_value=8, max_value=64, step=8)
        num_heads = hp.Int('num_heads', min_value=1, max_value=6, step=1)
        dropout = hp.Choice('dropout', values=[0.1, 0.15, 0.2, 0.25, 0.3])
        mlp_dim = hp.Int('mlp_dim', min_value=8, max_value=64, step=8)
        for _ in range(num_transformer_blocks):

            x = transformer_encoder(x, head_size, num_heads, dropout)

        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        x = layers.Dense(mlp_dim, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        outputs = layers.Dense(nb_classes, activation="softmax")(x)

        model = keras.Model(inputs, outputs)
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
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
        The optimal num_transformer_blocks is {self.best_hps.get('num_transformer_blocks')}
        The optimal head_size is {self.best_hps.get('head_size')}
        The optimal num_heads is {self.best_hps.get('num_heads')}
        The optimal mlp_dim is {self.best_hps.get('mlp_dim')}
        and the optimal learning rate for the optimizer
        is {self.best_hps.get('learning_rate')}.
        """)


        model = self.tuner.hypermodel.build(self.best_hps)

        return model

  

    def get_best_unique_model(self, model_index=0):
        '''
        Sometimes top models are the same. With this function we can get unique models'''

        model_representation = []
        for i in range(model_index+1):

            best_hps = self.tuner.get_best_hyperparameters(num_trials=i+1)[i]
            model_representation.append(dict((k, best_hps[k]) for k in ['num_transformer_blocks', 'head_size', 'num_heads','mlp_dim','dropout','learning_rate'] if k in best_hps))

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
        The optimal num_transformer_blocks is {self.best_hps.get('num_transformer_blocks')}
        The optimal head_size is {self.best_hps.get('head_size')}
        The optimal num_heads is {self.best_hps.get('num_heads')}
        The optimal mlp_dim is {self.best_hps.get('mlp_dim')}
        and the optimal learning rate for the optimizer
        is {self.best_hps.get('learning_rate')}.
        """)

        return  model







#
# model = build_model(input_shape=(96, 40), head_size=256, num_heads=, ff_dim=32,
#                     num_transformer_blocks=2, mlp_units=[128], mlp_dropout=0.4, dropout=0.25)
# model.summary()
#
# plot_model(model, to_file=os.path.join(config.package_directory, 'models', 'model_plot_transformer.png'), show_shapes=True,
#            show_layer_names=True)
