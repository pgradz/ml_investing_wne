
import os
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import ml_investing_wne.config as config
import numpy as np

# https://keras.io/examples/timeseries/timeseries_transformer_classification/

class PositionEmbeddingFixedWeights(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super(PositionEmbeddingFixedWeights, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        position_embedding_matrix = self.get_position_encoding(self.sequence_length, self.output_dim)
        self.position_embedding_layer = layers.Embedding(
            input_dim=self.sequence_length, output_dim=self.output_dim,
            weights=[position_embedding_matrix],
            trainable=False
        )

    def get_position_encoding(self, seq_len, d, n=10000):
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d / 2)):
                denominator = np.power(n, 2 * i / d)
                P[k, 2 * i] = np.sin(k / denominator)
                P[k, 2 * i + 1] = np.cos(k / denominator)
        return P

    def call(self, inputs):
        position_indices = tf.range(tf.shape(inputs)[1])
        embedded_indices = self.position_embedding_layer(position_indices)
        return inputs + embedded_indices

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'sequence_length': self.sequence_length,
            'output_dim': self.output_dim
        })
        return config

# my_embedding_layer = PositionEmbeddingFixedWeights(sequence_length=96, output_dim=40)
#
# image = X[np.random.choice(range(X.shape[0]))]
# tf.range(tf.shape(image)[0])
# my_embedding_layer = PositionEmbeddingFixedWeights(sequence_length=96,
#                                                    output_dim=40)(image)
#
# seq_len = 96
# d = 40
# n = 10000
# P = np.zeros((seq_len, d))
# for k in range(seq_len):
#     for i in np.arange(int(d / 2)):
#         denominator = np.power(n, 2 * i / d)
#         P[k, 2 * i] = np.sin(k / denominator)
#         P[k, 2 * i + 1] = np.cos(k / denominator)



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
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
    nb_classes=2
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    x = PositionEmbeddingFixedWeights(sequence_length=inputs.shape[1], output_dim=inputs.shape[2])(x)
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


# model = build_model(input_shape=(96, 40), head_size=256, num_heads=4, ff_dim=32,
#                     num_transformer_blocks=2, mlp_units=[128], mlp_dropout=0.4, dropout=0.25)
# model.summary()
#
# plot_model(model, to_file=os.path.join(config.package_directory, 'models', 'model_plot_transformer_positional_encoding.png'), show_shapes=True,
#            show_layer_names=True)
