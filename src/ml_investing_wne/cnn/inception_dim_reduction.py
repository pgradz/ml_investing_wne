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
    conv_list = []
    x_1_1 = keras.layers.Conv1D(filters=32, kernel_size=1,
                                             strides=1, padding='same', activation='relu', use_bias=False)(
            input_inception)
    conv_list.append(x_1_1)
    x_1_3_a = keras.layers.Conv1D(filters=32, kernel_size=1,
                                             strides=1, padding='same', activation='relu', use_bias=False)(
            input_inception)
    x_1_3_b = keras.layers.Conv1D(filters=32, kernel_size=3,
                                             strides=1, padding='same', activation='relu', use_bias=False)(
            x_1_3_a)
    conv_list.append(x_1_3_b)
    x_1_5_a = keras.layers.Conv1D(filters=32, kernel_size=1,
                                             strides=1, padding='same', activation='relu', use_bias=False)(
            input_inception)
    x_1_5_b = keras.layers.Conv1D(filters=32, kernel_size=5,
                                             strides=1, padding='same', activation='relu', use_bias=False)(
            x_1_5_a)
    conv_list.append(x_1_5_b)
    max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=1, padding='same')(input_inception)
    max_pool_1_1 = keras.layers.Conv1D(filters=32, kernel_size=1,
                                             strides=1, padding='same', activation='relu', use_bias=False)(
            max_pool_1)
    conv_list.append(max_pool_1_1)

    x_1 = keras.layers.Concatenate(axis=2)(conv_list)
    x_1 = keras.layers.BatchNormalization()(x_1)
    x_1 = keras.layers.Activation(activation='relu')(x_1)

    # 2nd inception
    conv_list = []
    x_2_1 = keras.layers.Conv1D(filters=32, kernel_size=1,
                                             strides=1, padding='same', activation='relu', use_bias=False)(
            x_1)
    conv_list.append(x_2_1)
    x_2_3_a = keras.layers.Conv1D(filters=32, kernel_size=1,
                                             strides=1, padding='same', activation='relu', use_bias=False)(
            x_1)
    x_2_3_b = keras.layers.Conv1D(filters=32, kernel_size=3,
                                             strides=1, padding='same', activation='relu', use_bias=False)(
            x_2_3_a)
    conv_list.append(x_2_3_b)
    x_2_5_a = keras.layers.Conv1D(filters=32, kernel_size=1,
                                             strides=1, padding='same', activation='relu', use_bias=False)(
            x_1)
    x_2_5_b = keras.layers.Conv1D(filters=32, kernel_size=5,
                                             strides=1, padding='same', activation='relu', use_bias=False)(
            x_2_5_a)
    conv_list.append(x_2_5_b)
    max_pool_2 = keras.layers.MaxPool1D(pool_size=3, strides=1, padding='same')(x_1)
    max_pool_2_1 = keras.layers.Conv1D(filters=32, kernel_size=1,
                                             strides=1, padding='same', activation='relu', use_bias=False)(
            max_pool_2)
    conv_list.append(max_pool_2_1)

    x_2 = keras.layers.Concatenate(axis=2)(conv_list)
    x_2 = keras.layers.BatchNormalization()(x_2)
    x_2 = keras.layers.Activation(activation='relu')(x_2)

    gap_layer = keras.layers.GlobalAveragePooling1D()(x_2)

    output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return model


model = build_model(input_shape=(96, 40), nb_classes=2)
model.summary()

plot_model(model, to_file=os.path.join(config.package_directory, 'models', 'model_plot_inception.png'), show_shapes=True,
           show_layer_names=True)

