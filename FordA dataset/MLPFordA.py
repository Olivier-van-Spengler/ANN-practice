import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import data
train_data = np.loadtxt('FordA_TRAIN.txt')
test_data = np.loadtxt('FordA_TEST.txt')
print(len(train_data))
print(len(test_data))

# Separate labels
train_labels = train_data[0:len(train_data), 0]
test_labels = test_data[0:len(test_data), 0]
print(test_labels)
np.delete(train_data, 0:len(train_data), 0])
np.delete(test_data, 0:len(test_data), 0])
print(test_data)

# One hot encode
