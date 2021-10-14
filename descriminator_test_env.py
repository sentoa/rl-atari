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
import time as t

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

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[84, 84, 4]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


def test_image():
    noise = tf.random.normal([1, 100])
    qwe = model(noise, training=False)
    generated_image = tf.squeeze(qwe)
    state = np.array(generated_image)
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    data = np.random.random(size=(3, 3, 3))
    z, x, y = state.nonzero()
    ax.scatter(x, y, z, c=z, alpha=1)
    plt.show()
    t1 = t.time()
    decision = discriminator(qwe)
    print(decision)
    t2 = t.time()
    print(t2-t1)


model = make_generator_model()
print(model.summary())

discriminator = make_discriminator_model()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


def discriminator_loss(real_output, fake_output):
    # IMAGE LOSS
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

checkpoint_dir = 'checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(discriminator_optimizer=discriminator_optimizer,
                                 discriminator=discriminator)

@tf.function
def train_step(real_image):
    noise = tf.random.normal([1, 100])

    with tf.GradientTape() as disc_tape:
        generated_images = model(noise, training=True)

        real_output = discriminator(real_image, training=True)
        fake_output = discriminator(generated_images, training=True)

        disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


## MAIN START HERE IN FUCKED UP CODE ATM
real_image = np.load('test.dat.npy')
real_image = tf.convert_to_tensor(real_image, dtype=tf.float32)
real_image = tf.expand_dims(real_image, axis=0)

train_step(real_image)
checkpoint.save(file_prefix = checkpoint_prefix)