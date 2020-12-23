import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

from MLP.load_data import load_label,load_data

# Import data
labels = load_label('../matlab/Random linear and stable systems/l_10_1000.mat')
dataset = load_data('../matlab/Random linear and stable systems/d_10_1000.mat')
print(type(labels))
print(max(labels))

tf_dataset = tf.data.Dataset.from_tensor_slices((dataset,labels))
print(type(tf_dataset))

#i = 0
#or element,label in tf_dataset:
#    i += 1
#    print(f'Trajectory {i} | System {label}: {element}')

# One hot encode
ohe_labels = np.zeros((labels.size, labels.max()))
ohe_labels[np.arange(labels.size), labels-1] = 1

# Split data
element_train, element_test, label_train, label_test = train_test_split(dataset, ohe_labels, test_size=0.2)

# Convert to Tensor
element_train = tf.convert_to_tensor(element_train, np.float32)
label_train = tf.convert_to_tensor(label_train, np.float32)
element_test = tf.convert_to_tensor(element_test, np.float32)
label_test = tf.convert_to_tensor(label_test, np.float32)
print(type(element_test))
print(element_train.shape)

# Create pipeline
N = max(labels)
length = len(tf_dataset)

# Build model
model = tf.keras.Sequential([
             tf.keras.Input(shape=10,),
             #tf.keras.layers.Dropout(.1),
             tf.keras.layers.Dense(400, activation = 'relu'),
             #tf.keras.layers.Dropout(.2),
             tf.keras.layers.Dense(400, activation = 'relu'),
             #tf.keras.layers.Dropout(.2),
             tf.keras.layers.Dense(400, activation = 'relu'),
             #tf.keras.layers.Dropout(.3),
             tf.keras.layers.Dense(N, activation = 'softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Setting weights and biases
model_weights_biases = model.get_weights()

print('\nThere are {:,} NumPy ndarrays in our list\n'.format(len(model_weights_biases)))

# Training
epochs = 150
batch_size = 10

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                                  verbose=1, patience=3,
                                                  restore_best_weights='true')

history = model.fit(element_train,
                    label_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(element_test, label_test),
                    callbacks=early_stopping)
print(history)




training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

training_loss = history.history['loss']
validation_loss = history.history['val_loss']

epochs_range = range(epochs)

# Plotting results
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, training_accuracy, label='Training Accuracy')
plt.plot(epochs_range, validation_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, training_loss, label='Training Loss')
plt.plot(epochs_range, validation_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# Saving results
hist_df = pd.DataFrame(history.history)
hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

