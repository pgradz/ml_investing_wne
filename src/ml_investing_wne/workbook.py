


model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(144, 35)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# fit model
model.fit(X, y, epochs=50, validation_data=(X_val, y_val))

model.predict(X_test)

import keras
from keras import backend as K
from keras.models import load_model, Model
from keras.layers import Flatten, Dense, Dropout, Activation, Input, LSTM, Reshape, Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt


def create_deeplob(T, NF, number_of_lstm):
    input_lmd = Input(shape=(T, NF, 1))

    # build the convolutional block
    conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(input_lmd)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    # conv_first1 = Conv2D(32, (1, 10))(conv_first1)
    # conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    # conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    # conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    # conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    # conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    #
    # # build the inception module
    # convsecond_1 = Conv2D(64, (1, 1), padding='same')(conv_first1)
    # convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)
    # convsecond_1 = Conv2D(64, (3, 1), padding='same')(convsecond_1)
    # convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)
    #
    # convsecond_2 = Conv2D(64, (1, 1), padding='same')(conv_first1)
    # convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)
    # convsecond_2 = Conv2D(64, (5, 1), padding='same')(convsecond_2)
    # convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)
    #
    # convsecond_3 = MaxPooling2D((3, 1), strides=(1, 1), padding='same')(conv_first1)
    # convsecond_3 = Conv2D(64, (1, 1), padding='same')(convsecond_3)
    # convsecond_3 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_3)
    #
    # convsecond_output = keras.layers.concatenate([convsecond_1, convsecond_2, convsecond_3], axis=3)
    # conv_reshape = Reshape((int(convsecond_output.shape[1]), int(convsecond_output.shape[3])))(convsecond_output)
    # conv_reshape = keras.layers.Dropout(0.2, noise_shape=(None, 1, int(conv_reshape.shape[2])))(conv_reshape,
    #                                                                                             training=True)
    #
    # # build the last LSTM layer
    # conv_lstm = LSTM(number_of_lstm)(conv_reshape)

    # build the output layer
    i = Flatten()(conv_first1)
    h = Dense(100, activation='relu')(i)
    out = Dense(1, activation='sigmoid')(h)
    model = Model(inputs=input_lmd, outputs=out)
    # adam = keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

X = X.reshape(X.shape + (1,))
X_val = X_val.reshape(X_val.shape + (1,))

deeplob = create_deeplob(X.shape[1], X.shape[2], 64)

deeplob.fit(X, y, validation_data=(X_val, y_val),
            epochs=50, batch_size=128, verbose=2 )

deeplob.predict(X_test.reshape(X_test.shape + (1,)))

'''
about cnn (images)

You always have to give a 4D array as input to the cnn. So input data has a shape of (batch_size, height, width, depth)
'''