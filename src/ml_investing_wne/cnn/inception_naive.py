import tensorflow.keras as keras
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import numpy as np
import os
import ml_investing_wne.config as config


def build_model(input_shape, nb_classes):

    input_layer = keras.layers.Input(input_shape)
    #
    input_inception = keras.layers.Conv1D(filters=32, kernel_size=1,
                                              padding='same', activation='relu', use_bias=False)(input_layer)


    # 1st inception
    kernel_size_s = [1, 3, 5]
    #kernel_size_s = [40 // (2 ** i) for i in range(3)]

    conv_list = []

    for i in range(len(kernel_size_s)):
        conv_list.append(keras.layers.Conv1D(filters=32, kernel_size=kernel_size_s[i],
                                             strides=1, padding='same', activation='relu', use_bias=False)(
            input_inception))

    max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=1, padding='same')(input_inception)

    conv_list.append(max_pool_1)

    x_1 = keras.layers.Concatenate(axis=2)(conv_list)
    x_1 = keras.layers.BatchNormalization()(x_1)
    x_1 = keras.layers.Activation(activation='relu')(x_1)

    # 2nd inception
    conv_list = []

    for i in range(len(kernel_size_s)):
        conv_list.append(keras.layers.Conv1D(filters=32, kernel_size=kernel_size_s[i],
                                             strides=1, padding='same', activation='relu', use_bias=False)(
            x_1))

    max_pool_2 = keras.layers.MaxPool1D(pool_size=3, strides=1, padding='same')(x_1)

    conv_list.append(max_pool_2)

    x_2 = keras.layers.Concatenate(axis=2)(conv_list)
    x_2 = keras.layers.BatchNormalization()(x_2)
    x_2 = keras.layers.Activation(activation='relu')(x_2)

    #3nd inception
    conv_list = []

    for i in range(len(kernel_size_s)):
        conv_list.append(
            keras.layers.Conv1D(filters=32, kernel_size=kernel_size_s[i],
                                strides=1, padding='same', activation='relu',
                                use_bias=False)(
                x_2))

    max_pool_3 = keras.layers.MaxPool1D(pool_size=3, strides=1, padding='same')(
        x_2)

    conv_list.append(max_pool_3)

    x_3 = keras.layers.Concatenate(axis=2)(conv_list)
    x_3 = keras.layers.BatchNormalization()(x_3)
    x_3 = keras.layers.Activation(activation='relu')(x_3)

    gap_layer = keras.layers.GlobalAveragePooling1D()(x_3)

    output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return model


model = build_model(input_shape=(96, 40), nb_classes=2)
model.summary()

plot_model(model, to_file=os.path.join(config.package_directory, 'models', 'model_plot_inception.png'), show_shapes=True,
           show_layer_names=True)

