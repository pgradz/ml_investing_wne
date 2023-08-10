import os
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from keras_tuner.tuners import hyperband
import keras_tuner as kt

from ml_investing_wne import config
from ml_investing_wne.utils import get_logger
from ml_investing_wne.experiment_factory import create_asset, experiment_factory

tf.keras.backend.set_image_data_format("channels_last")
logger = get_logger()

asset = create_asset()
experiment= experiment_factory(asset).get_experiment()
experiment.train_test_val_split()

print(config.currency)
print(config.RUN_SUBTYPE)
print(config.seq_stride)


def build_model(hp, input_shape=(96,41), nb_classes=2):

    n_feature_maps = hp.Int('n_feature_maps', min_value=8, max_value=128, step=8)
    input_layer = keras.layers.Input(input_shape)

    # BLOCK 1
    kernel_size_1 = hp.Int('kernel_size_1', min_value=1, max_value=12, step=1)
    conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=kernel_size_1, padding='same')(input_layer)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    kernel_size_2 = hp.Int('kernel_size_2', min_value=1, max_value=12, step=1)
    conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=kernel_size_2, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    kernel_size_3 = hp.Int('kernel_size_3', min_value=1, max_value=12, step=1)
    conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=kernel_size_3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = keras.layers.Activation('relu')(output_block_1)
    dropout = hp.Choice('dropout', values=[0.1, 0.15, 0.2, 0.25, 0.3])
    output_block_1 = keras.layers.Dropout(dropout)(output_block_1, training=True)

    # BLOCK 2
    # n_feature_maps_2 = hp.Int('n_feature_maps_2', min_value=8, max_value=128, step=8)
    # conv_x = keras.layers.Conv1D(filters=n_feature_maps_2, kernel_size=kernel_size_1, padding='same')(output_block_1)
    # conv_x = keras.layers.BatchNormalization()(conv_x)
    # conv_x = keras.layers.Activation('relu')(conv_x)

    # conv_y = keras.layers.Conv1D(filters=n_feature_maps_2, kernel_size=kernel_size_2, padding='same')(conv_x)
    # conv_y = keras.layers.BatchNormalization()(conv_y)
    # conv_y = keras.layers.Activation('relu')(conv_y)

    # conv_z = keras.layers.Conv1D(filters=n_feature_maps_2, kernel_size=kernel_size_3, padding='same')(conv_y)
    # conv_z = keras.layers.BatchNormalization()(conv_z)

    # # expand channels for the sum
    # shortcut_y = keras.layers.Conv1D(filters=n_feature_maps_2, kernel_size=1, padding='same')(output_block_1)
    # shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    # output_block_2 = keras.layers.add([shortcut_y, conv_z])
    # output_block_2 = keras.layers.Activation('relu')(output_block_2)
    # output_block_2 = keras.layers.Dropout(dropout)(output_block_2, training=True)

    # BLOCK 3

    # conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
    # conv_x = keras.layers.BatchNormalization()(conv_x)
    # conv_x = keras.layers.Activation('relu')(conv_x)

    # conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    # conv_y = keras.layers.BatchNormalization()(conv_y)
    # conv_y = keras.layers.Activation('relu')(conv_y)

    # conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    # conv_z = keras.layers.BatchNormalization()(conv_z)

    # # no need to expand channels because they are equal
    # shortcut_y = keras.layers.BatchNormalization()(output_block_2)

    # output_block_3 = keras.layers.add([shortcut_y, conv_z])
    # output_block_3 = keras.layers.Activation('relu')(output_block_2)
    # output_block_3 = keras.layers.Dropout(0.25)(output_block_3, training=True)
    # FINAL
    lstm_neurons = hp.Int('lstm_neurons', min_value=8, max_value=128, step=8)
    lstm_layer = keras.layers.LSTM(lstm_neurons)(output_block_1)
    # output_layer = keras.layers.Dense(1, activation='softmax')(lstm_layer)
    # model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    # model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(),
    #               metrics=['accuracy'])
    
    output_layer = keras.layers.Dense(nb_classes, activation='softmax')(lstm_layer)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate ) ,
                  metrics=['accuracy'])
    return model


tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name=f'{config.currency}_{config.seq_len}_{config.RUN_SUBTYPE}_{config.seq_stride}_1cnns_filtered')

# tuner = kt.RandomSearch(
#     hypermodel=build_model,
#     objective="val_accuracy",
#     max_trials=3,
#     executions_per_trial=2,
#     overwrite=True,
#     directory="my_dir",
#     project_name="helloworld",
# )

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(experiment.train_dataset, epochs=10, validation_data=experiment.val_dataset, callbacks=[stop_early])

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

# print(f"""
# The hyperparameter search is complete. 
# The optimal dropout is {best_hps.get('dropout')} 
# The optimal n_feature_maps is {best_hps.get('n_feature_maps')}
# The optimal n_feature_maps_2 is {best_hps.get('n_feature_maps_2')}

# The optimal kernel_size_1 is {best_hps.get('kernel_size_1')}
# The optimal kernel_size_2 is {best_hps.get('kernel_size_2')}
# The optimal kernel_size_3 is {best_hps.get('kernel_size_3')}

# The optimal lstm_neurons is {best_hps.get('lstm_neurons')}
# and the optimal learning rate for the optimizer
# is {best_hps.get('learning_rate')}.
# """)

# print(f"""
# The hyperparameter search is complete. 
# The optimal dropout is {best_hps.get('dropout')} 
# The optimal n_feature_maps is {best_hps.get('n_feature_maps')}
# The optimal kernel_size_1 is {best_hps.get('kernel_size_1')}
# The optimal kernel_size_2 is {best_hps.get('kernel_size_2')}
# The optimal kernel_size_3 is {best_hps.get('kernel_size_3')}
# The optimal lstm_neurons is {best_hps.get('lstm_neurons')}
# and the optimal learning rate for the optimizer
# is {best_hps.get('learning_rate')}.
# """)

print(f"""
The hyperparameter search is complete. 
The optimal dropout is {best_hps.get('dropout')} 
The optimal n_feature_maps is {best_hps.get('n_feature_maps')}
The optimal kernel_size_1 is {best_hps.get('kernel_size_1')}
The optimal kernel_size_2 is {best_hps.get('kernel_size_2')}
The optimal kernel_size_3 is {best_hps.get('kernel_size_3')}
The optimal lstm_neurons is {best_hps.get('lstm_neurons')}
and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")