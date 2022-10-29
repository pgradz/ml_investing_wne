import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers


<<<<<<< HEAD
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
=======
def transformer_encoder(inputs, head_size, num_heads, dropout=0):
>>>>>>> aebd5e9 (local old changes)
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
<<<<<<< HEAD
    # x = layers.Dropout(dropout)(x)
    # x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_model(input_shape, head_size, num_heads, num_transformer_blocks,ff_dim, mlp_units, mlp_dropout, dropout=0, nb_classes=2):
    n_feature_maps = input_shape[1]
=======
    x = layers.Dropout(dropout)(x)
    return x + res

def build_model(input_shape, nb_classes,head_size, num_heads, num_transformer_blocks, dropout=0):
    n_feature_maps = 40
>>>>>>> aebd5e9 (local old changes)

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
    output_block_1 = keras.layers.Dropout(dropout)(output_block_1, training=True)

    # BLOCK 2

    conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(output_block_1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_2 = keras.layers.add([shortcut_y, conv_z])
    output_block_2 = keras.layers.Activation('relu')(output_block_2)
    output_block_2 = keras.layers.Dropout(dropout)(output_block_2, training=True)
    
    # TRANSFORMER
    x = input_layer
    for _ in range(num_transformer_blocks):
<<<<<<< HEAD
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout=dropout)
=======
        x = transformer_encoder(x, head_size, num_heads, dropout=dropout)
>>>>>>> aebd5e9 (local old changes)

    # FINAL
    transformer_resnet = keras.layers.add([output_block_2, x])
    output_block_3 = keras.layers.LSTM(64)(transformer_resnet)
    output_block_3 = layers.Dropout(dropout)(output_block_3)
    output_layer = keras.layers.Dense(nb_classes, activation='softmax')(output_block_3)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

<<<<<<< HEAD
    return model
=======
    return model


# model = build_model(input_shape=(96, 40), nb_classes=2, head_size=64, num_heads=4,
#                     num_transformer_blocks=4,  dropout=0.25)
# model.summary()
>>>>>>> aebd5e9 (local old changes)
