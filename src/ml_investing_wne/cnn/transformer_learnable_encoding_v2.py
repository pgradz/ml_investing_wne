import os
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import ml_investing_wne.config as config


# https://keras.io/examples/timeseries/timeseries_transformer_classification/

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
        return embedded_indices

    def get_config(self):
        config = super().get_config().copy()
        return config


def build_model(
        input_shape,
        head_size,
        num_heads,
        ff_dim,
        embedding_size,
        num_transformer_blocks,
        mlp_units,
        dropout=0,
        mlp_dropout=0,
        nb_classes=2
):
    inputs = keras.Input(shape=input_shape)

    conv_x = keras.layers.Conv1D(filters=embedding_size, kernel_size=8, padding='same')(inputs)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=embedding_size, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=embedding_size, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=embedding_size, kernel_size=1, padding='same')(inputs)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = keras.layers.Activation('relu')(output_block_1)

    position_embeeding = PositionEmbeddingLayer(input_shape[0], embedding_size)(inputs)

    x = output_block_1 + position_embeeding
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(nb_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return model
