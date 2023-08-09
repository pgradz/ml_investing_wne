import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import plot_model
import ml_investing_wne.config as config

# tf.keras.backend.set_image_data_format("channels_first")
tf.keras.backend.set_image_data_format("channels_last")

def build_model(input_shape, nb_classes):
    

    input_layer = keras.layers.Input(input_shape)

    # BLOCK 1

    conv_x = keras.layers.Conv1D(filters=40, kernel_size=3, padding='same')(input_layer)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=40, kernel_size=9, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=40, kernel_size=2, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=40, kernel_size=1, padding='same')(input_layer)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = keras.layers.Activation('relu')(output_block_1)
    output_block_1 = keras.layers.Dropout(0.3)(output_block_1, training=True)

    # BLOCK 2

    conv_x = keras.layers.Conv1D(filters=120, kernel_size=3, padding='same')(output_block_1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=120, kernel_size=9, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=120, kernel_size=2, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=120, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_2 = keras.layers.add([shortcut_y, conv_z])
    output_block_2 = keras.layers.Activation('relu')(output_block_2)
    output_block_2 = keras.layers.Dropout(0.3)(output_block_2, training=True)

    # FINAL
    
    lstm_layer = keras.layers.LSTM(72)(output_block_2)
    # output_layer = keras.layers.Dense(1, activation='softmax')(lstm_layer)
    # model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    # model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(),
    #               metrics=['accuracy'])
    
    output_layer = keras.layers.Dense(nb_classes, activation='softmax')(lstm_layer)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  metrics=['accuracy'])

    return model

# model = build_model(input_shape=(96, 40), nb_classes=2)
# model.summary()
# plot_model(model, to_file=os.path.join(config.package_directory, 'models', 'model_plot_resnet_regularized.png'),
#            show_shapes=True, show_layer_names=True)
