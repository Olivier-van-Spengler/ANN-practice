import tensorflow as tf
import pandas as pd
import tensorflow_datasets as tfds
import numpy as np
from load_data import load_mat

dataset = (load_mat('dataset.mat'))
labelset = (load_mat('label.mat'))

print(labelset['label'].shape)


labels = pd.get_dummies(pd.Series(labelset['label']))
# Make dummy variables for rank
#data = pd.concat([labelset, pd.get_dummies(labelset['label'], prefix='label')], axis=1)
#data = data.drop('label', axis=1)
print(labels)

# Check that dataset is a tuple
print('dataset has type:', type(dataset))
print('dataset has type:', type(labelset))

# Print the number of elements in dataset
print('dataset has {:,} elements '.format(dataset['data']))
# Does not give correct output

# TODO: Need to split data
# TODO: Create pipeline??

N = 3
#length

# Model
model = tf.keras.Sequential([
             tf.keras.Input(shape=10,),
             tf.keras.layers.Dense(64, activation = 'relu'),
             tf.keras.layers.Dense(32, activation = 'relu'),
             tf.keras.layers.Dense(N, activation = 'softmax')
])

model.summary()

model_weights_biases = model.get_weights()

print('\nThere are {:,} NumPy ndarrays in our list\n'.format(len(model_weights_biases)))

print(model_weights_biases)

for i, layer in enumerate(model.layers):

    if len(layer.get_weights()) > 0:
        w = layer.get_weights()[0]
        b = layer.get_weights()[1]

        print('\nLayer {}: {}\n'.format(i, layer.name))
        print('\u2022 Weights:\n', w)
        print('\n\u2022 Biases:\n', b)
        print('\nThis layer has a total of {:,} weights and {:,} biases'.format(w.size, b.size))
        print('\n------------------------')

    else:
        print('\nLayer {}: {}\n'.format(i, layer.name))
        print('This layer has no weights or biases.')
        print('\n------------------------')

