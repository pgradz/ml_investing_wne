# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of TSMixer."""

import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import ml_investing_wne.config as config


def res_block(inputs, norm_type, activation, dropout, ff_dim):
    """Residual block of TSMixer."""

    norm = (
        layers.LayerNormalization
        if norm_type == 'L'
        else layers.BatchNormalization
    )   
    # Temporal Linear
    x = norm(axis=[-2, -1])(inputs)
    x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
    x = layers.Dense(x.shape[-1], activation=activation)(x)
    x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Input Length, Channel]
    x = layers.Dropout(dropout)(x)
    res = x + inputs    
    # Feature Linear
    x = norm(axis=[-2, -1])(res)
    x = layers.Dense(ff_dim, activation=activation)(
        x
    )  # [Batch, Input Length, FF_Dim]
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1])(x)  # [Batch, Input Length, Channel]
    x = layers.Dropout(dropout)(x)
    return x + res  


def build_model(
    input_shape,
    nb_classes,
    norm_type,
    activation,
    n_block,
    dropout,
    ff_dim,
    target_slice):
    """Build TSMixer model."""

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs  # [Batch, Input Length, Channel]
    for _ in range(n_block):
        x = res_block(x, norm_type, activation, dropout, ff_dim)        
    if target_slice:
        x = x[:, :, target_slice]     

    # x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
    # x = layers.Dense(pred_len)(x)  # [Batch, Channel, Output Length]
    # outputs = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Output Length, Channel])
           
    output_layer = layers.Dense(nb_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(),
                    metrics=['accuracy'])       
    return model        


model = build_model(input_shape=(96, 40), nb_classes=2, norm_type='B', activation='relu', n_block=2, dropout=0.15, ff_dim=64,  target_slice=None)
model.summary()
plot_model(model, to_file=os.path.join(config.package_directory, 'models', 'model_plot_tsmixer.png'),
           show_shapes=True, show_layer_names=True)
