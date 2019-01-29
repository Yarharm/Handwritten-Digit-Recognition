import tensorflow as tf
import numpy as np
import pandas as pd

# Load MNIST

# Train data
train_data = pd.read_csv('Kaggle_MNIST/train.csv', low_memory=False)
train_data = train_data.values

X_train = train_data[:, 1:]
y_train = train_data[:, 0]

# Test data
test_list = pd.read_csv('Kaggle_MNIST/test.csv', low_memory=False)
X_test = test_list.values
test_ID = [(x + 1) for x in range(X_test.shape[0])]

# Normalization
mean_vals = np.mean(X_train, axis=0)  # along each column
std_vals = np.std(X_train)
X_train_normal = (X_train - mean_vals) / std_vals
X_test_normal = (X_test - mean_vals) / std_vals  # test set is normalized with train mean and std
#print(X_train_normal.shape, y_train.shape)
#print(X_test_normal.shape, y_test.shape)


# Model
# 4-layers: 1 input, 2 hidden, 1 output
# Activation function: hidden layers => tanh, output layer => softmax
n_features = X_train_normal.shape[1]
n_classes = 10  # numbers between 0-9

# Define a Dataflow graph
g = tf.Graph()

# define placeholders[edges] (input variables)
with g.as_default():
    tf.set_random_seed(1)
    tf_x = tf.placeholder(dtype=tf.float32,  # 60000x10 tensor
                          shape=(None, n_features),
                          name='tf_x')
    tf_y = tf.placeholder(dtype=tf.int64,  # 60000x1 tensor
                          shape=None,
                          name='tf_y')
    # encode target
    y_onehot = tf.one_hot(indices=tf_y, depth=n_classes)  # 10000x10

    # first hidden layer
    h1 = tf.layers.dense(inputs=tf_x, units=50,
                         activation=tf.tanh, name='hid_layer1')  # 60000x50 tensor
    # second hidden layer
    h2 = tf.layers.dense(inputs=h1, units=50,
                         activation=tf.tanh, name='hid_layer2')

    # Non-normalized predictions [logits] output layer
    logits = tf.layers.dense(inputs=h2, units=10,
                             activation=None, name='out_layer')  # softmax is not an acivation func

    predict = {
        'classes': tf.argmax(logits, axis=1,
                             name='predicted_classes'),
        'probabilities': tf.nn.softmax(logits,
                                        name='softmax_tensor')
    }

# Cost function and optimizer
with g.as_default():
    cost = tf.losses.softmax_cross_entropy(onehot_labels=y_onehot,
                                           logits=logits)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=cost)  # operation to comput and apply gradients with LR
    init_op = tf.global_variables_initializer()  # initialize global variables on the graph

# Get mini-batches generator with shuffling(disabled)
def batches_gen(X, y, batch_size=128, shuffle=False):
    X_copy = np.array(X)
    y_copy = np.array(y)

    if shuffle:
        data = np.column_stack((X_copy, y_copy))
        np.random.shuffle(data)
        X_copy = data[:, :-1]
        y_copy = data[:, -1].astype(int)

    for i in range(0, X.shape[0], batch_size):
        yield(X_copy[i:i+batch_size, :], y_copy[i:i+batch_size])


# Tensorflow session
session = tf.Session(graph=g)
# Initialize global variables
session.run(init_op)

# 50 epochs for training
for epoch in range(50):
    costs = []
    batch_generator = batches_gen(X_train_normal,
                                  y_train,
                                  batch_size=64)
    for batch_X, batch_y in batch_generator:
        # dictionary to feed to NN
        feed = {tf_x: batch_X, tf_y: batch_y}
        _, batch_cost = session.run([train_op, cost], feed_dict=feed)
        costs.append(batch_cost)

# Predict
feed = {tf_x: X_test_normal}
y_pred = session.run(predict['classes'],
                      feed_dict=feed)
print(y_pred)
