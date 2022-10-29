import os
from tensorflow import keras
from tensorflow.keras import layers
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
    x = layers.Dense(inputs.shape[-1], activation="relu")(x)
    x = layers.Dropout(dropout)(x)
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
<<<<<<< HEAD
 
    return model

#
# model = build_model(input_shape=(96, 40), head_size=64, num_heads=4, ff_dim=32,
#                     num_transformer_blocks=4, mlp_units=[128], mlp_dropout=0.4, dropout=0.25)
# model.summary()
=======

    return model

#
model = build_model(input_shape=(96, 40), head_size=64, num_heads=4, ff_dim=32,
                    num_transformer_blocks=4, mlp_units=[128], mlp_dropout=0.4, dropout=0.25)
model.summary()
>>>>>>> aebd5e9 (local old changes)
#
# plot_model(model, to_file=os.path.join(config.package_directory, 'models', 'model_plot_transformer.png'), show_shapes=True,
#            show_layer_names=True)
