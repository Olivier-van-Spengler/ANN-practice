import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#<<<<<<< HEAD:main.py
from MLP.load_data import load_label,load_data

labels = load_label('labels.mat')
dataset = load_data('dataset.mat')

tf_dataset = tf.data.Dataset.from_tensor_slices((dataset,labels))
print(type(tf_dataset))

i = 0
for element,label in tf_dataset:
    i += 1
    print(f'Trajectory {i} | System {label}: {element}')

#=======

#>>>>>>> d2dfe78a590c8d7a315309f96561ef94940c85fa:MLP/main.py

# One hot encode
ohe_labels = np.zeros((labels.size, labels.max()))
ohe_labels[np.arange(labels.size), labels-1] = 1

# Split data
element_train, element_test, label_train, label_test = train_test_split(dataset, ohe_labels, test_size=0.2)

# Create pipeline
N = 3
length = len(tf_dataset)

# Build model
model = tf.keras.Sequential([
             tf.keras.Input(shape=10,),
             tf.keras.layers.Dense(64, activation = 'relu'),
             tf.keras.layers.Dropout(.1),
             tf.keras.layers.Dense(32, activation = 'relu'),
             tf.keras.layers.Dropout(.1),
             tf.keras.layers.Dense(N, activation = 'softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Setting weights and biases
model_weights_biases = model.get_weights()

print('\nThere are {:,} NumPy ndarrays in our list\n'.format(len(model_weights_biases)))

#print(model_weights_biases)

#for i, layer in enumerate(model.layers):

 #   if len(layer.get_weights()) > 0:
  #      w = layer.get_weights()[0]
  #      b = layer.get_weights()[1]

 #       print('\nLayer {}: {}\n'.format(i, layer.name))
  #      print('\u2022 Weights:\n', w)
 #       print('\n\u2022 Biases:\n', b)
  #      print('\nThis layer has a total of {:,} weights and {:,} biases'.format(w.size, b.size))
  #      print('\n------------------------')

 #   else:
 #       print('\nLayer {}: {}\n'.format(i, layer.name))
  #      print('This layer has no weights or biases.')
  #      print('\n------------------------')

# Training
epochs = 5
batch_size = 5

history = model.fit(element_train,label_train,batch_size=batch_size,epochs=epochs,validation_data=(element_test, label_test),)
print(history)

training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

training_loss = history.history['loss']
validation_loss = history.history['val_loss']

epochs_range = range(epochs)

# Plotting results
#plt.figure(figsize=(8, 8))
#plt.subplot(1, 2, 1)
#plt.plot(epochs_range, training_accuracy, label='Training Accuracy')
#plt.plot(epochs_range, validation_accuracy, label='Validation Accuracy')
#plt.legend(loc='lower right')
#plt.title('Training and Validation Accuracy')

#plt.subplot(1, 2, 2)
#plt.plot(epochs_range, training_loss, label='Training Loss')
#plt.plot(epochs_range, validation_loss, label='Validation Loss')
#plt.legend(loc='upper right')
#plt.title('Training and Validation Loss')
#plt.show()


# Saving results
print(history.history)