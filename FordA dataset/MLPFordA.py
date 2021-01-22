import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sktime
from sktime.utils.load_data import load_from_arff_to_dataframe, load_from_tsfile_to_dataframe


def adapt_data(df):
    data = list()
    for index, row in df.iterrows():
        try:
            data.append(row.values.tolist()[0])
        except Exception as e:
            print(e)
    return data


train_x, train_y = load_from_tsfile_to_dataframe(
    os.path.join('.', "FordA_TRAIN.ts")
)
test_x, test_y = load_from_tsfile_to_dataframe(
    os.path.join('.', "FordA_TEST.ts")
)

print(type(train_x))
train_x = adapt_data(train_x)
print(type(train_x))
#train_x = np.reshape(3601, 500)
test_x = adapt_data(test_x)

tf_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))

# i = 0
# for element, label in tf_dataset:
#     i += 1
#     print(f'Example {i} | Label {label}: {element}')

# CHECKED UNTIL HERE

#>>>>>>> 5f5ad796965ac542e90ec54bbe821dd1c1c2edba
# Convert to numpy.ndarray
# train_x = np.ndarray(train_x)

# train_x = tf.convert_to_tensor(train_x)
#train_x = pd.DataFrame(train_x).to_numpy()
#train_y = pd.DataFrame(train_y).to_numpy()
#test_x = pd.DataFrame(test_x).to_numpy()
#test_y = pd.DataFrame(test_y).to_numpy()

# train_x = np.asarray(train_x).astype(np.float32)
# train_y = np.asarray(train_y).astype(np.float32)
# test_x = np.asarray(test_x).astype(np.float32)
# test_y = np.asarray(test_y).astype(np.float32)

print(type(train_x))
print((train_x[0:2]))
# One hot encode

# Create pipeline
N = 2
length = len(train_x)
InputShape = len(train_x[0])
print(InputShape)

# Build model
model = tf.keras.Sequential([
    tf.keras.Input(shape=500, ),
    tf.keras.layers.Dropout(.1),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dropout(.3),
    tf.keras.layers.Dense(N, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Setting weights and biases
model_weights_biases = model.get_weights()

# Training algorithm
epochs = 10
batch_size = 5

history = model.fit(train_x,
                    train_y,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(test_x, test_y), )
print(history)

training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

training_loss = history.history['loss']
validation_loss = history.history['val_loss']

epochs_range = range(epochs)
