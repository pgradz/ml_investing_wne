import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import ml_investing_wne.config as config
from keras import backend as K
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
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1, activation="relu",  padding='same')(x)
    x = layers.Dropout(dropout)(x)
    return x + res


class T2V(keras.layers.Layer):

    def __init__(self, output_dim=None, **kwargs):
        self.output_dim = output_dim
        super(T2V, self).__init__(**kwargs)

    def build(self, input_shape):
        # to do: why -1 and 1
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[-1], self.output_dim),
                                 initializer='uniform',
                                 trainable=True)
        self.P = self.add_weight(name='P',
                                 shape=(input_shape[1], self.output_dim),
                                 initializer='uniform',
                                 trainable=True)
        self.w = self.add_weight(name='w',
                                 shape=(input_shape[1], 1),
                                 initializer='uniform',
                                 trainable=True)
        self.p = self.add_weight(name='p',
                                 shape=(input_shape[1], 1),
                                 initializer='uniform',
                                 trainable=True)
        super(T2V, self).build(input_shape)

    def call(self, x):
        original = self.w * x + self.p
        sin_trans = K.sin(K.dot(x, self.W) + self.P)

        return K.concatenate([sin_trans, original], -1)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim
        })
        return config


class Time2Vec(keras.layers.Layer):
    def __init__(self, kernel_size=1):
        super(Time2Vec, self).__init__(trainable=True, name='Time2VecLayer')
        self.k = kernel_size

    def build(self, input_shape):
        # trend
        self.wb = self.add_weight(name='wb', shape=(input_shape[1],), initializer='uniform',
                                  trainable=True)
        self.bb = self.add_weight(name='bb', shape=(input_shape[1],), initializer='uniform',
                                  trainable=True)
        # periodic
        self.wa = self.add_weight(name='wa', shape=(1, input_shape[1], self.k),
                                  initializer='uniform', trainable=True)
        self.ba = self.add_weight(name='ba', shape=(1, input_shape[1], self.k),
                                  initializer='uniform', trainable=True)
        super(Time2Vec, self).build(input_shape)

    def call(self, inputs, **kwargs):
        bias = self.wb * inputs + self.bb
        dp = keras.dot(inputs, self.wa) + self.ba
        wgts = K.sin(dp)  # or K.cos(.)

        ret = K.concatenate([K.expand_dims(bias, -1), wgts], -1)
        ret = K.reshape(ret, (-1, inputs.shape[1] * (self.k + 1)))
        return ret

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * (self.k + 1))



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
    x = T2V(40)(inputs)
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

    return model
# 
# model = build_model(input_shape=(96, 40), head_size=64, num_heads=4, ff_dim=32,
#                     num_transformer_blocks=4, mlp_units=[128], mlp_dropout=0.4, dropout=0.25)
# model.summary()
# 
# plot_model(model, to_file=os.path.join(config.package_directory, 'models', 'model_plot_transformer.png'), show_shapes=True,
#            show_layer_names=True)