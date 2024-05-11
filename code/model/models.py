"""
Homework 5 - CNNs
CS1430 - Computer Vision
Brown University
"""

import tensorflow as tf
import keras.optimizers
from keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense

import hyperparameters as hp

class VGGModel(tf.keras.Model):
    def __init__(self):
        super(VGGModel, self).__init__()

        self.optimizer = keras.optimizers.Adam(
              learning_rate=hp.learning_rate,
        )

        # Don't change the below:

        self.vgg16 = [
              # Block 1
              Conv2D(32, 3, 1, padding="same", activation="relu", name="block1_conv1"),
              Conv2D(32, 3, 1, padding="same", activation="relu", name="block1_conv2"),
              MaxPool2D(2, name="block1_pool"),
              # Block 2
              Conv2D(64, 3, 1, padding="same", activation="relu", name="block2_conv1"),
              Conv2D(64, 3, 1, padding="same", activation="relu", name="block2_conv2"),
              MaxPool2D(2, name="block2_pool"),
              # Block 3
              Conv2D(128, 3, 1, padding="same", activation="relu", name="block3_conv1"),
              Conv2D(128, 3, 1, padding="same", activation="relu", name="block3_conv2"),
              Conv2D(128, 3, 1, padding="same", activation="relu", name="block3_conv3"),
              MaxPool2D(2, name="block3_pool"),
              # Block 4
              Conv2D(256, 3, 1, padding="same", activation="relu", name="block4_conv1"),
              Conv2D(256, 3, 1, padding="same", activation="relu", name="block4_conv2"),
              Conv2D(256, 3, 1, padding="same", activation="relu", name="block4_conv3"),
              MaxPool2D(2, name="block4_pool"),
              # Block 5
              Conv2D(512, 3, 1, padding="same", activation="relu", name="block5_conv1"),
              Conv2D(512, 3, 1, padding="same", activation="relu", name="block5_conv2"),
              Conv2D(512, 3, 1, padding="same", activation="relu", name="block5_conv3"),
              MaxPool2D(2, name="block5_pool"),
       ]



    
        for layer in self.vgg16:
            layer.trainable = False

        # TODO: Write a classification head for our 15-scene classification task.

        self.head = [
                     tf.keras.layers.Flatten(),
                     tf.keras.layers.Dense(512, activation='relu'),
                     tf.keras.layers.Dropout(0.5),
                     tf.keras.layers.Dense(3, activation='softmax')  # Assuming 15 output
                     ]

        # Don't change the below:
        self.vgg16 = tf.keras.Sequential(self.vgg16, name="vgg_base")
        self.head = tf.keras.Sequential(self.head, name="vgg_head")

    def call(self, x):
        """ Passes the image through the network. """

        x = self.vgg16(x)
        x = self.head(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for model. """
        loss = keras.losses.SparseCategoricalCrossentropy()(labels, predictions)
        return loss
