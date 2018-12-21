import numpy as np
import scipy.special as spec
import matplotlib.pyplot as plt

# 3 layer neura network with stochastic gradient descend
class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # number of nodes for the input, hidden and output layers
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        # learning rate
        self.lr = learning_rate

        # matrices wih(input => hidden) and who(hidden => output)
        # weights in range of normal distribution with standard div 1/sqrt(n)
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # activation function (Sigmoid = expit)
        self.activation_func = lambda x: spec.expit(x)


    # adjust the weights
    def train(self, input_list, target_list):
        # targets => training data
        targets = np.array(target_list, ndmin=2).T
        inputs = np.array(input_list, ndmin=2).T

        # propagate through network
        hidden_in = np.dot(self.wih, inputs)
        hidden_out = self.activation_func(hidden_in)
        final_in = np.dot(self.who, hidden_out)
        final_out = self.activation_func(final_in)

        # error (target - actual)
        # output_errors => Refine weights between hidden and final layers
        # hidden_errors => Refine weight between input and hidden layers
        output_errors = targets - final_out
        hidden_errors = np.dot(self.who.T, output_errors)

        # update weights between hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_out * (1.0 - final_out)), np.transpose(hidden_out))

        # uodate weights between input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_out * (1.0 - hidden_out)), np.transpose(inputs))


    # get output from the neural network
    def query(self, input_list):
        # convert inputs into 2-d array for the dot product
        # |.T| => transpose
        inputs = np.array(input_list, ndmin=2).T

        # calculate hidden layer (Applying Sigmoid on the dot product)
        hidden_in = np.dot(self.wih, inputs)
        hidden_out = self.activation_func(hidden_in)

        # calculate final layer
        final_in = np.dot(self.who, hidden_out)
        return self.activation_func(final_in)

# Main

# input, hidden and output nodes
input_nodes = 784 # 28x28 image
hidden_nodes = 100 # any reasonable number between 10 and 784, to avoid overfitting and underfitting
output_nodes = 10 # number of possible numbers between [0..9]

# Experimentally derived, through brute force
learning_rate = 0.2

# passes over the training dataset
epochs = 2

# create NN
nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# TRAINING PHASE (For 2 epochs
# load training data from the subset of MNIST.csv of 100 elements
training_data = open("MNIST/mnist_train_10k.csv", "r")
training_list = training_data.readlines()
training_data.close()

# train NN
for e in range(epochs):
    for label in training_list:
        # rescale inputs from the training_list in range (0.00, 1.00]
        inputs = np.asfarray(label.split(',')[1:]) / 255 * 0.99 + 0.01
        # create target array full of 0.01 and label of 0.99
        targets = np.zeros(output_nodes) + 0.01
        targets[int(label.split(',')[0])] = 0.99
        nn.train(inputs, targets)

# TESTING PHASE
# load testing data from the subset of MNIST.csv of 10 elements
testing_data = open("MNIST/mnist_test_1k.csv", "r")
testing_list = testing_data.readlines()
testing_data.close()

# score for each record in the test set
score = []

for record in testing_list:
    correct_answer = int(record.split(',')[0])
#    print(correct_answer, "correct answer")
    #get inputs
    inputs = np.asfarray(record.split(',')[1:]) / 255.0 * 0.99 + 0.01
    outputs = nn.query(inputs)
    # get label spitted by the NN
    answer = np.argmax(outputs)
#    print(answer, "network output")
    # add 1 => correct answer/ 0 => incorrect
    if(answer == correct_answer):
        score.append(1)
    else:
        score.append(0)

score_array = np.asarray(score)
print("Performance in %: ", score_array.sum() / score_array.size)