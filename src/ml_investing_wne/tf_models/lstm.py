import os
import tensorflow.keras as keras
from tensorflow.keras.utils import plot_model
import ml_investing_wne.config as config
from keras.layers import LSTM, Dense
from keras.models import Sequential


def build_model(input_shape, nb_classes):


    model = Sequential()

    model.add(LSTM(512, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    
    return model


# model = build_model(input_shape=(96, 40), nb_classes=2)
# model.summary()