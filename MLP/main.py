import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from load_data import load_mat

dataset = load_mat('dataset.mat')

# Check that dataset is a tuple
print('dataset has type:', type(dataset))

# Print the number of elements in dataset
print('dataset has {:,} elements '.format(len(dataset)))
# Does not give correct output

# TODO: need to split data
# TODO: Create pipeline??

# Model
model = tf.keras.Sequential([
             tf.keras.Input(shape=8,),
             tf.keras.layers.Dense(128,  activation = 'relu'),
             tf.keras.layers.Dense(64, activation = 'relu'),
             tf.keras.layers.Dense(10, activation = 'softmax')
])

model.summary()

model_weights_biases = model.get_weights()

print('\nThere are {:,} NumPy ndarrays in our list\n'.format(len(model_weights_biases)))

print(model_weights_biases)

