import tensorflow as tf


def Conv_1D_block(inputs, model_width, kernel, strides):
    # 1D Convolutional Block with BatchNormalization
    x = tf.keras.layers.Conv1D(model_width, kernel, strides=strides, padding="same", kernel_initializer="he_normal")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


def Conv_1D_block_2(inputs, model_width, kernel, strides, nl):
    # This function defines a 1D convolution operation with BN and activation.
    x = tf.keras.layers.Conv1D(model_width, kernel, strides=strides, padding="same", kernel_initializer="he_normal")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    if nl == 'HS':
        x = x * tf.keras.activations.relu(x + 3.0, max_value=6.0) / 6.0
    elif nl == 'RE':
        x = tf.keras.activations.relu(x, max_value=6.0)

    return x


def Conv_1D_DW(inputs, model_width, kernel, strides, alpha):
    # 1D Depthwise Separable Convolutional Block with BatchNormalization
    model_width = int(model_width * alpha)
    x = tf.keras.layers.SeparableConv1D(model_width, kernel, strides=strides, depth_multiplier=1, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv1D(model_width, 1, strides=1, padding="same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


def bottleneck_block(inputs, filters, kernel, t, alpha, s, r=False):
    tchannel = tf.keras.backend.int_shape(inputs)[-1] * t
    cchannel = int(filters * alpha)

    x = Conv_1D_block(inputs, tchannel, 1, 1)
    x = tf.keras.layers.SeparableConv1D(filters, kernel, strides=s, depth_multiplier=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv1D(cchannel, 1, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('linear')(x)

    if r:
        x = tf.keras.layers.concatenate([x, inputs], axis=-1)

    return x


def bottleneck_block_2(inputs, filters, kernel, e, s, squeeze, nl, alpha):
    # This function defines a basic bottleneck structure.

    input_shape = tf.keras.backend.int_shape(inputs)

    tchannel = int(e)
    cchannel = int(alpha * filters)

    r = s == 1 and input_shape[2] == filters

    x = Conv_1D_block_2(inputs, tchannel, 1, 1, nl)

    x = tf.keras.layers.SeparableConv1D(filters, kernel, strides=s, depth_multiplier=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if nl == 'HS':
        x = x * tf.keras.activations.relu(x + 3.0, max_value=6.0) / 6.0
    if nl == 'RE':
        x = tf.keras.activations.relu(x, max_value=6.0)

    if squeeze:
        x = _squeeze(x)

    x = tf.keras.layers.Conv1D(cchannel, 1, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if r:
        x = tf.keras.layers.Add()([x, inputs])

    return x


def inverted_residual_block(inputs, filters, kernel, t, alpha, strides, n):
    if strides == 1:
        x = bottleneck_block(inputs, filters, kernel, t, alpha, strides, True)
    else:
        x = bottleneck_block(inputs, filters, kernel, t, alpha, strides)

    for i in range(1, n):
        x = bottleneck_block(x, filters, kernel, t, alpha, 1, True)

    return x


def _squeeze(inputs):
    # This function defines a squeeze structure.

    input_channels = int(inputs.shape[-1])

    x = tf.keras.layers.GlobalAveragePooling1D()(inputs)
    x = tf.keras.layers.Dense(input_channels, activation='relu')(x)
    x = tf.keras.layers.Dense(input_channels, activation='hard_sigmoid')(x)
    x = tf.keras.layers.Reshape((1, input_channels))(x)
    x = tf.keras.layers.Multiply()([inputs, x])

    return x

def MLP(x, pooling='avg', dropout_rate=False, problem_type='Classification', output_nums=2):
    if pooling == 'avg':
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    elif pooling == 'max':
        x = tf.keras.layers.GlobalMaxPool1D()(x)
    # Final Dense Outputting Layer for the outputs
    x = tf.keras.layers.Flatten()(x)
    if dropout_rate:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(output_nums, activation='linear')(x)
    if problem_type == 'Classification':
        outputs = tf.keras.layers.Dense(output_nums, activation='softmax')(x)

    return outputs

def build_model(input_shape, alpha=1, nb_classes=2):
    # inputs = tf.keras.Input((length, num_channel))
    inputs = tf.keras.layers.Input(input_shape)

    x = Conv_1D_block_2(inputs, 16, 3, strides=2, nl='HS')
    x = bottleneck_block_2(x, 16, 3, e=16, s=2, squeeze=True, nl='RE',
                           alpha=alpha)
    x = bottleneck_block_2(x, 24, 3, e=72, s=2, squeeze=False, nl='RE',
                           alpha=alpha)
    x = bottleneck_block_2(x, 24, 3, e=88, s=1, squeeze=False, nl='RE',
                           alpha=alpha)
    x = bottleneck_block_2(x, 40, 5, e=96, s=2, squeeze=True, nl='HS',
                           alpha=alpha)
    x = bottleneck_block_2(x, 40, 5, e=240, s=1, squeeze=True, nl='HS',
                           alpha=alpha)
    x = bottleneck_block_2(x, 40, 5, e=240, s=1, squeeze=True, nl='HS',
                           alpha=alpha)
    x = bottleneck_block_2(x, 48, 5, e=120, s=1, squeeze=True, nl='HS',
                           alpha=alpha)
    x = bottleneck_block_2(x, 48, 5, e=144, s=1, squeeze=True, nl='HS',
                           alpha=alpha)
    x = bottleneck_block_2(x, 96, 5, e=288, s=2, squeeze=True, nl='HS',
                           alpha=alpha)
    x = bottleneck_block_2(x, 96, 5, e=576, s=1, squeeze=True, nl='HS',
                           alpha=alpha)
    x = bottleneck_block_2(x, 96, 5, e=576, s=1, squeeze=True, nl='HS',
                           alpha=alpha)
    x = Conv_1D_block_2(x, 576, 1, strides=1, nl='HS')
    x = x * tf.keras.activations.relu(x + 3.0, max_value=6.0) / 6.0
    x = tf.keras.layers.Conv1D(1280, 1, padding='same')(x)

    outputs = MLP(x, output_nums=nb_classes)
    model = tf.keras.Model(inputs, outputs)

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return model

# model = build_model(input_shape=(96, 40), 2)
# model.summary()
#
# plot_model(model, to_file=os.path.join(config.package_directory, 'models', 'model_plot_resnet.png'), show_shapes=True,
#            show_layer_names=True)