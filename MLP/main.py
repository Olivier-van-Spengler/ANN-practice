import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
#<<<<<<< HEAD:main.py
from MLP.load_data import load_label,load_data

labels = load_label('labels.mat')
dataset = load_data('dataset.mat')

tf_dataset = tf.data.Dataset.from_tensor_slices((dataset,labels))

i = 0
for element,label in tf_dataset:
    i += 1
    print(f'Trajectory {i} | System {label}: {element}')

#=======

#>>>>>>> d2dfe78a590c8d7a315309f96561ef94940c85fa:MLP/main.py

# TODO: One hot encode
#One Hot Encode

ohe_labels = np.zeros((labels.size, labels.max()))
ohe_labels[np.arange(labels.size), labels-1] = 1
#print(len(ohe_labels))
#print(type(ohe_labels))

# TODO: Need to split data

#split = tf.split(tf_dataset, 90)
#print(split)

element_train, element_test, label_train, label_test = train_test_split(dataset, ohe_labels, test_size=0.2)

# TODO: Create pipeline

N = 3
length = len(tf_dataset)

# Model
model = tf.keras.Sequential([
             tf.keras.Input(shape=10,),
             tf.keras.layers.Dense(64, activation = 'relu'),
             tf.keras.layers.Dense(32, activation = 'relu'),
             tf.keras.layers.Dense(N, activation = 'softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#loss, accuracy = model.evaluate(testing_batches)

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

result = model.fit(dataset,ohe_labels, 10)
print(result)
