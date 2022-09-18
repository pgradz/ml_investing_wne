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
        return inputs + embedded_indices
    
    def get_config(self):
        config = super().get_config().copy()
        return config

def build_model(hp,
    input_shape,
    num_transformer_blocks,
    dropout=0,
    mlp_dropout=0,
    nb_classes=2
):
    inputs = keras.Input(shape=input_shape)
    x = PositionEmbeddingLayer(input_shape[0], input_shape[1])(inputs)
    
    head_size = hp.Int('units', min_value=32, max_value=256, step=32)
    num_heads = hp.Int('units', min_value=1, max_value=8, step=1)
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        
    mlp_units = hp.Int('units', min_value=32, max_value=256, step=32)
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(nb_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  metrics=['accuracy'])

    return model

#
# model = build_model(input_shape=(96, 40), head_size=256, num_heads=, ff_dim=32,
#                     num_transformer_blocks=2, mlp_units=[128], mlp_dropout=0.4, dropout=0.25)
# model.summary()
#
# plot_model(model, to_file=os.path.join(config.package_directory, 'models', 'model_plot_transformer.png'), show_shapes=True,
#            show_layer_names=True)
