# https://keras.io/examples/vision/image_classification_with_vision_transformer/

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size_dim_1, patch_size_dim_2):
        super(Patches, self).__init__()
        self.patch_size_dim_1 = patch_size_dim_1
        self.patch_size_dim_2 = patch_size_dim_2

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size_dim_1, self.patch_size_dim_2, 1],
            # strides=[1, self.patch_size_dim_1, self.patch_size_dim_2, 1],
            strides=[1, 1, self.patch_size_dim_2, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def build_model(input_shape, projection_dim, num_heads, transformer_layers, mlp_head_units,
                patch_size_dim_1, patch_size_dim_2, num_classes):

    # num_patches = int(input_shape[0]/patch_size_dim_1
    num_patches = 91
    transformer_units = [projection_dim * 2, projection_dim]
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    # Create patches.
    patches = Patches(patch_size_dim_1, patch_size_dim_2)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model


model = build_model(input_shape=(96, 40, 1), projection_dim=64, num_heads=4, transformer_layers=4,
                    mlp_head_units = [256, 128], patch_size_dim_1=6,patch_size_dim_2=40,
                    num_classes = 2)

model.summary()
X_2d = X.reshape(X.shape + (1,))
X_val_2d = X_val.reshape(X_val.shape + (1,))
history = model.fit(X_2d, y_cat, batch_size=64, epochs=10, verbose=2,
                    validation_data=(X_val_2d, y_val_cat))



image = X_2d[np.random.choice(range(X_2d.shape[0]))]
resized_image = tf.image.resize(
    tf.convert_to_tensor([image]), size=(96, 40)
)

plt.figure(figsize=(4, 4))
plt.imshow(image.astype("uint8"))
plt.axis("off")

patches = Patches(6, 40)(resized_image)
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")


n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))

for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (6, 40, 1))
    plt.imshow(patch_img.numpy().astype("uint8"))
    plt.axis("off")




patch_img = tf.reshape(image, (96, 40, 1))

patches[0].shape


n = 10
# images is a 1 x 10 x 10 x 1 array that contains the numbers 1 through 100
images = [[[[x * n + y + 1] for y in range(n)] for x in range(n)]]
images = np.array(images)
# We generate two outputs as follows:
# 1. 3x3 patches with stride length 5
# 2. Same as above, but the rate is increased to 2
tf.image.extract_patches(images=images,
                         sizes=[1, 3, 3, 1],
                         strides=[1, 5, 5, 1],
                         rates=[1, 1, 1, 1],
                         padding='VALID')

# Yields:
[[[[1  2  3 11 12 13 21 22 23]
   [6  7  8 16 17 18 26 27 28]]
 [[51 52 53 61 62 63 71 72 73]
    [56 57 58 66 67 68 76 77 78]]]]