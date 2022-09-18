import tensorflow.keras as keras
import tensorflow as tf
import numpy as no
import os

import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import ml_investing_wne.config as config


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
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
    # x = layers.Dropout(dropout)(x)
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

def build_model(input_shape, head_size, num_heads, num_transformer_blocks,ff_dim, mlp_units, mlp_dropout, dropout=0, nb_classes=2):
    n_feature_maps = 32

    input_layer = keras.layers.Input(input_shape)

    # BLOCK 1

    conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = keras.layers.Activation('relu')(output_block_1)
    output_block_1 = keras.layers.Dropout(0.25)(output_block_1, training=True)

    # BLOCK 2

    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_2 = keras.layers.add([shortcut_y, conv_z])
    output_block_2 = keras.layers.Activation('relu')(output_block_2)
    output_block_2 = keras.layers.Dropout(0.25)(output_block_2, training=True)

    # BLOCK 3

    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # no need to expand channels because they are equal
    shortcut_y = keras.layers.BatchNormalization()(output_block_2)

    output_block_3 = keras.layers.add([shortcut_y, conv_z])
    output_block_3 = keras.layers.Activation('relu')(output_block_3)
    output_block_3 = keras.layers.Dropout(0.25)(output_block_3, training=True)

    # FINAL
    lstm = keras.layers.LSTM(64)(output_block_3)
    # gap_layer_resnet = keras.layers.GlobalAveragePooling1D()(output_block_3)
    
    # TRANSFORMER
    x = PositionEmbeddingLayer(input_shape[0], input_shape[1])(input_layer)
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout=dropout)
    gap_layer_transformer = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    
    # FINAL
    transformer_resnet = keras.layers.concatenate([lstm, gap_layer_transformer])
    for dim in mlp_units:
        mlp = layers.Dense(dim, activation="relu")(transformer_resnet)
        mlp = layers.Dropout(mlp_dropout)(mlp)
    outputs = layers.Dense(nb_classes, activation="softmax")(mlp)

    model = keras.models.Model(inputs=input_layer, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return model


# model = build_model(input_shape=(96, 40), head_size=64, num_heads=4, ff_dim=64,
#                      num_transformer_blocks=4, mlp_units=[128], mlp_dropout=0.25, dropout=0.25)
#
# model.summary()
# plot_model(model, to_file=os.path.join(config.package_directory, 'models', 'model_plot_transformer_resnet.png'), show_shapes=True,
#            show_layer_names=True)
