import pandas as pd
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

# Load MNIST
train_data = pd.read_csv('Kaggle_MNIST/train.csv', low_memory=False)
train_data = train_data.values
X_train, y_train = train_data[:, 1:], train_data[:, 0]

test_data = pd.read_csv('Kaggle_MNIST/test.csv', low_memory=False)
X_test = test_data.values
test_ID = [(x + 1) for x in range(X_test.shape[0])]

# Normalization
mean_vals = np.mean(X_train, axis=0)  # along each column
std_vals = np.std(X_train)
X_train_normal = (X_train - mean_vals) / std_vals
X_test_normal = (X_test - mean_vals) / std_vals  # test set is normalized with train mean and std

tf.set_random_seed(1)  # reproducability

# Encode target (Categorical => encoded)
y_train_encoded = to_categorical(y_train)

# Three layers NN: 2 hidden layers with 50 units and last layers with 10 class labels
# 2 hidden layers undego 'tanh' output layer 'softmax'
model = Sequential()

# First layer
model.add(
    Dense(units=50,  # output shape batchx50
          input_dim=X_train_normal.shape[1],  # input layer shape batchx784
          kernel_initializer='glorot_uniform',  # Xavier uniform for initial weights
          bias_initializer='zeros',
          activation='tanh'))

# Second layer
model.add(
    Dense(units=50,
          input_dim=50,
          kernel_initializer='glorot_uniform',
          bias_initializer='zeros',
          activation='tanh'))

# Output layer
model.add(
    Dense(units=y_train_encoded.shape[1],  # shape batchx10 (Class labels)
          input_dim=50,
          kernel_initializer='glorot_uniform',
          bias_initializer='zeros',
          activation='softmax')
)

# Instantiate Optimizer (SGD)
optimizer = optimizers.SGD(lr=0.001, decay=1e-7,  # fast lr decay
                           momentum=0.9, nesterov=False)  # accelerate SGD

# Configure learning process
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy')

# Train (mini-batches SGD)
model.fit(X_train_normal, y_train_encoded,
          batch_size=64, epochs=50,
          verbose=0, validation_split=0.1)  # reserve 10% for validation

# Predict
y_pred = model.predict_classes(X_test_normal,
                               verbose=0)
