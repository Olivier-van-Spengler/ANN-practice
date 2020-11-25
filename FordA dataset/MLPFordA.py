import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#<<<<<<< HEAD:main.py


train_data = np.loadtxt('FordA_TRAIN.txt')
test_data = np.loadtxt('FordA_TEST.txt')
print(len(test_data))

i=0
for x in train_data:
    i += 1
    train_label = train_data[i, 0]
    if i == len(train_data)-1:
        break

i=0
for x in test_data:
    i += 1
    test_label = test_data[i, 0]
    if i == len(test_data)-1:
        break



print(test_label[5])

#i=0
#for x in test_data:
#    i += 1
#    print(f'Trajectory {i} | System {test_data[i,0]}: {test_data[i,0:]}')
#    if i == (len(test_data)-1):
#        break


print(test_label)
print(type(train_data))
print(len(train_data))
#print(train_data)

# One hot encode
