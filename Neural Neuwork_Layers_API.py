import tensorflow as tf
import numpy as np



# Load MNIST

# Train data
data = open('mnist_train.csv', 'r')
train_list = data.readlines()
data.close()

# Test data
#data = open('mnist_test.csv', 'r')
#y_test = data.readlines()
#data.close()

# PROCESS
def process(data):
    data = np.asarray(data)  # convert to numpy for vectorization purpose
    data = np.char.split(data, sep=',')  # Elim commas
    f = lambda x: [np.asfarray(y) for y in x]  # Convert inner images to numpy arr
    data = np.asarray(f(data))  # normilize obtained list
    return data

def inputs_targets(data):
    data = process(data)
    y_train = data[:, 0]  # get labels
    X_train = data[:, 1:]  # get pixels
    X_train = ((X_train / 255.0) - 0.5) * 2  # rescale from (-1,1)
    return X_train, y_train

X_train, y_train = inputs_targets(train_list)
#print(y_train.shape)
#print(X_train.shape)

n_features = X_train.shape[1]
n_classes = 10  # numbers between 0-9
