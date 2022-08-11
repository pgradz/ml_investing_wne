# https://keras.io/examples/vision/image_classification_with_vision_transformer/

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

num_classes = 2
input_shape = (96, 40, 1)

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 100
image_size = 72  # We'll resize input images to this size
patch_size = 6  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024]

#(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
image = x_train[np.random.choice(range(x_train.shape[0]))]
resized_image = tf.image.resize(
    tf.convert_to_tensor([image]), size=(image_size, image_size)
)

import matplotlib.pyplot as plt
plt.figure(figsize=(4, 4))
plt.imshow(image.astype("uint8"))
plt.axis("off")

patches = Patches(6,6)(resized_image)
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))

for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img.numpy().astype("uint8"))
    plt.axis("off")

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
            strides=[1, self.patch_size_dim_1, self.patch_size_dim_2, 1],
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


X_2d = X.reshape(X.shape + (1,))

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