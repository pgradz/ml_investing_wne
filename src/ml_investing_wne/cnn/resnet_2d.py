import os
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import plot_model
import ml_investing_wne.config as config


def res_identity(x, filters):
  #renet block where dimension doesnot change.
  #The skip connection is just simple identity conncection
  #we will have 3 blocks and then input will be added

  x_skip = x # this will be used for addition with the residual block
  f1, f2 = filters

  #first block
  x = keras.layers.Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=keras.regularizers.l2(0.001))(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)

  #second block # bottleneck (but size kept same with padding)
  x = keras.layers.Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.001))(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)

  # third block activation used after adding the input
  x = keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=keras.regularizers.l2(0.001))(x)
  x = keras.layers.BatchNormalization()(x)
  # x = Activation(activations.relu)(x)

  # add the input
  x = keras.layers.Add()([x, x_skip])
  x = keras.layers.Activation('relu')(x)

  return x

def res_conv(x, s, filters):
  '''
  here the input size changes'''
  x_skip = x
  f1, f2 = filters

  # first block
  x = keras.layers.Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=keras.regularizers.l2(0.001))(x)
  # when s = 2 then it is like downsizing the feature map
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)

  # second block
  x = keras.layers.Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.001))(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)

  #third block
  x = keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=keras.regularizers.l2(0.001))(x)
  x = keras.layers.BatchNormalization()(x)

  # shortcut
  x_skip = keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=keras.regularizers.l2(0.001))(x_skip)
  x_skip = keras.layers.BatchNormalization()(x_skip)

  # add
  x = keras.layers.Add()([x, x_skip])
  x = keras.layers.Activation('relu')(x)

  return x


def build_model(input_shape, nb_classes):

  input_im = keras.layers.Input(input_shape) # cifar 10 images size
  x = keras.layers.ZeroPadding2D(padding=(3, 3))(input_im)

  # 1st stage
  # here we perform maxpooling, see the figure above

  x = keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation('relu')(x)
  x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

  #2nd stage
  # frm here on only conv block and identity block, no pooling

  x = res_conv(x, s=1, filters=(64, 256))
  x = res_identity(x, filters=(64, 256))
  # x = res_identity(x, filters=(64, 256))

  # 3rd stage

  x = res_conv(x, s=2, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))
  # x = res_identity(x, filters=(128, 512))

  # # 4th stage
  #
  # x = res_conv(x, s=2, filters=(256, 1024))
  # x = res_identity(x, filters=(256, 1024))
  # x = res_identity(x, filters=(256, 1024))
  # x = res_identity(x, filters=(256, 1024))
  # x = res_identity(x, filters=(256, 1024))
  # x = res_identity(x, filters=(256, 1024))

  x = keras.layers.AveragePooling2D((2, 2), padding='same')(x)

  x = keras.layers.Flatten()(x)
  x = keras.layers.Dense(nb_classes, activation='softmax',
            kernel_initializer='he_normal')(x)  # multi-class

  # define the model

  model = keras.models.Model(inputs=input_im, outputs=x, name='Resnet50')

  model.compile(loss='categorical_crossentropy',
                optimizer=keras.optimizers.Adam(),
                metrics=['accuracy'])

  return model

#
# model = build_model(input_shape=(96, 40, 1), nb_classes=2)
# model.summary()
# 
# plot_model(model, to_file=os.path.join(config.package_directory, 'models', 'model_plot_resnet_2d.png'), show_shapes=True,
#            show_layer_names=True)
