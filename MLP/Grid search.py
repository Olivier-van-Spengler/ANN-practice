import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
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
print(type(element_train))
print(element_train.shape)

# Convert to Tensor
#element_train = tf.convert_to_tensor(element_train, np.float32)
#label_train = tf.convert_to_tensor(label_train, np.float32)
#element_test = tf.convert_to_tensor(element_test, np.float32)
#label_test = tf.convert_to_tensor(label_test, np.float32)
#print(type(element_test))
#print(element_test.shape)

# Create pipeline
N = max(labels)
length = len(tf_dataset)

def create_model(hidden_layers, nodes, dropout_rate, activation_function):
    # loss_function, learn_rate, batch_size
    # Build model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(nodes, input_dim=10,
                                    kernel_initializer='normal', activation=activation_function))
    model.add(tf.keras.layers.Dropout(dropout_rate))

    for i in range(hidden_layers):
        # Add one hidden layer
        model.add(tf.keras.layers.Dense(nodes,
                                        kernel_initializer='normal', activation=activation_function))
        model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # Compile the model
    adam = tf.keras.optimizers.Adam(lr=learn_rate)
    model.compile(loss=loss_function, optimizer=adam, metrics=['accuracy'])
    return model


# Create the model
model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, verbose=1)

#model.summary()

# Define the parameters that you wish to use in your Grid Search along
# with the list of values that you wish to try out
learn_rate = 1 #, 0.1, 1]
dropout_rate = [0.0, 0.2, 0.4]
batch_size = [10, 20]
epochs = [25, 50]
nodes = [50, 100, 200, 400, 800]
hidden_layers = [1, 2, 3]
loss_function = 'categorical_crossentropy' # 'poisson', 'sparse_categorical_crossentropy']
activation_function = ['relu', 'tanh', 'sigmoid']

seed = 42

# Make a dictionary of the grid search parameters
param_grid = dict(hidden_layers=hidden_layers, nodes=nodes, dropout_rate=dropout_rate,
                  activation_function=activation_function, batch_size=batch_size, epochs=epochs
                  )
# loss_function=loss_function, learn_rate=learn_rate,

# Build and fit the GridSearchCV
# grid = sklearn.model_selection.GridSearchCV(estimator=model, param_grid=param_grid,
   #                 cv=sklearn.model_selection.KFold(random_state=seed), verbose=10)

# Build and fit the RandomizedSearchCV
grid = sklearn.model_selection.RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                                                  n_iter= 10,
                    cv=sklearn.model_selection.KFold(random_state=seed), verbose=10)

grid_results = grid.fit(element_train, label_train)

# Summarize the results in a readable format
print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))

means = grid_results.cv_results_['mean_test_score']
stds = grid_results.cv_results_['std_test_score']
params = grid_results.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print('{0} ({1}) with: {2}'.format(mean, stdev, param))



