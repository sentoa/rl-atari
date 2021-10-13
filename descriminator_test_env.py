import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(21*21*128, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((21, 21, 128)))
    assert model.output_shape == (None, 21, 21, 128)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 42, 42, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(4, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 84, 84, 4)

    return model

model = make_generator_model()
print(model.summary())


noise = tf.random.normal([1, 100])
generated_image = model(noise, training=False)
generated_image = tf.squeeze(generated_image)
state = np.array(generated_image)
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
data = np.random.random(size=(3, 3, 3))
z, x, y = state.nonzero()
ax.scatter(x, y, z, c=z, alpha=1)
plt.show()