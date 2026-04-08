!pip install tensorflow matplotlib

import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
import numpy as np
import os

!rm -rf horse2zebra
!rm -f horse2zebra.zip horse2zebra.zip.*
!wget -O horse2zebra.zip https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/horse2zebra.zip
!unzip -oq horse2zebra.zip

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 1
AUTOTUNE = tf.data.AUTOTUNE

trainA = tf.data.Dataset.list_files("horse2zebra/trainA/*.jpg")
trainB = tf.data.Dataset.list_files("horse2zebra/trainB/*.jpg")
testA = tf.data.Dataset.list_files("horse2zebra/testA/*.jpg")
testB = tf.data.Dataset.list_files("horse2zebra/testB/*.jpg")

def load_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = (tf.cast(image, tf.float32) / 127.5) - 1
    return image

trainA = trainA.map(load_image, num_parallel_calls=AUTOTUNE).shuffle(1000).batch(BATCH_SIZE)
trainB = trainB.map(load_image, num_parallel_calls=AUTOTUNE).shuffle(1000).batch(BATCH_SIZE)
testA = testA.map(load_image, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)
testB = testB.map(load_image, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)

def build_generator():
    inputs = layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3])
    x = layers.Conv2D(64, 7, padding='same')(inputs)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, 3, strides=2, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)
    x = layers.ReLU()(x)
    outputs = layers.Conv2D(3, 7, padding='same', activation='tanh')(x)
    return Model(inputs, outputs)

def build_discriminator():
    inputs = layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3])
    x = layers.Conv2D(64, 4, strides=2, padding='same')(inputs)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(256, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    outputs = layers.Conv2D(1, 4, padding='same')(x)
    return Model(inputs, outputs)

G_AB = build_generator()
G_BA = build_generator()
D_A = build_discriminator()
D_B = build_discriminator()

loss_obj = tf.keras.losses.MeanSquaredError()

def generator_loss(fake):
    return loss_obj(tf.ones_like(fake), fake)

def discriminator_loss(real, fake):
    return (loss_obj(tf.ones_like(real), real) + loss_obj(tf.zeros_like(fake), fake)) * 0.5

def cycle_loss(real, cycled):
    return tf.reduce_mean(tf.abs(real - cycled))

G_AB_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
G_BA_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
D_A_opt  = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
D_B_opt  = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

@tf.function
def train_step(real_A, real_B):
    with tf.GradientTape(persistent=True) as tape:
        fake_B = G_AB(real_A, training=True)
        cycled_A = G_BA(fake_B, training=True)

        fake_A = G_BA(real_B, training=True)
        cycled_B = G_AB(fake_A, training=True)

        disc_real_A = D_A(real_A, training=True)
        disc_fake_A = D_A(fake_A, training=True)
        disc_real_B = D_B(real_B, training=True)
        disc_fake_B = D_B(fake_B, training=True)

        gen_AB_loss = generator_loss(disc_fake_B)
        gen_BA_loss = generator_loss(disc_fake_A)

        cycle_l = cycle_loss(real_A, cycled_A) + cycle_loss(real_B, cycled_B)

        total_gen_AB = gen_AB_loss + 10 * cycle_l
        total_gen_BA = gen_BA_loss + 10 * cycle_l

        disc_A_loss = discriminator_loss(disc_real_A, disc_fake_A)
        disc_B_loss = discriminator_loss(disc_real_B, disc_fake_B)

    G_AB_opt.apply_gradients(zip(tape.gradient(total_gen_AB, G_AB.trainable_variables), G_AB.trainable_variables))
    G_BA_opt.apply_gradients(zip(tape.gradient(total_gen_BA, G_BA.trainable_variables), G_BA.trainable_variables))
    D_A_opt.apply_gradients(zip(tape.gradient(disc_A_loss, D_A.trainable_variables), D_A.trainable_variables))
    D_B_opt.apply_gradients(zip(tape.gradient(disc_B_loss, D_B.trainable_variables), D_B.trainable_variables))

    return total_gen_AB, total_gen_BA, disc_A_loss, disc_B_loss

EPOCHS = 3

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    for i, (real_A, real_B) in enumerate(tf.data.Dataset.zip((trainA, trainB))):
        g1, g2, d1, d2 = train_step(real_A, real_B)
        if i % 100 == 0:
            print(f"Step {i}: G_AB={g1:.4f}, G_BA={g2:.4f}, D_A={d1:.4f}, D_B={d2:.4f}")

def display_images(orig, trans):
    orig = (orig[0] + 1) / 2.0
    trans = (trans[0] + 1) / 2.0
    plt.subplot(1,2,1); plt.imshow(orig); plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(trans); plt.axis("off")
    plt.show()

for s in testA.take(1):
    display_images(s, G_AB(s, training=False))

for s in testB.take(1):
    display_images(s, G_BA(s, training=False))
