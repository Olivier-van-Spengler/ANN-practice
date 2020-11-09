import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Set the random seed so things are reproducible
tf.random.set_seed(7)

# Create 3 random input features
features = tf.random.normal((1,3))

# Define the size of each layer in our network
n_input = features.shape[1]     # Number of input units, must match number of input features
n_hidden = 2                    # Number of hidden units
n_output = 1                    # Number of output units

# Create random weights connecting the inputs to the hidden layer
W1 = tf.random.normal((n_input, n_hidden))

# Create random weights connecting the hidden layer to the output layer
W2 = tf.random.normal((n_hidden, n_output))

# Create random bias terms for the hidden and output layers
B1 = tf.random.normal((1, n_hidden))
B2 = tf.random.normal((1, n_output))

def sigmoid_activation(x):

    return 1 / (1 + tf.exp(-x))

## Solution
hidden = sigmoid_activation(tf.matmul(features, W1) + B1)

output = sigmoid_activation(tf.matmul(hidden, W2) + B2)

print(output)