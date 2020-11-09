import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

print('Using:')
print('\t\u2022 TensorFlow version:', tf.__version__)

# Set the random seed so things are reproducible
tf.random.set_seed(7)

# Create 5 random input features
features = tf.random.normal((1, 5))

# Create random weights for our neural network
weights = tf.random.normal((1, 5))

# Create a random bias term for our neural network
bias = tf.random.normal((1, 1))

print('Features:\n', features)
print('\nWeights:\n', weights)
print('\nBias:\n', bias)


def sigmoid_activation(x):

    return 1 / (1 + tf.exp(-x))


print('Features Shape:', features.shape)
print('Weights Shape:', weights.shape)
print('Bias Shape:', bias.shape)

# Solution
y = sigmoid_activation(tf.matmul(features, weights, transpose_b = True) + bias)

print('label:\n', y)

print('Weights Shape:', weights.shape)

