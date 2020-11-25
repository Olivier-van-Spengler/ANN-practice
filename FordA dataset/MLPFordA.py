import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#<<<<<<< HEAD:main.py


train_data = np.loadtxt('FordA_TRAIN.txt')
test_data = np.loadtxt('FordA_TEST.txt')

i=0
for x in test_data:
    i += 1
    print(test_data[i,0])


print(type(train_data))
print(len(train_data))
#print(train_data)

# One hot encode
