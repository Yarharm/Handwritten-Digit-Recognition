import numpy as np
import scipy.special as spec


# 3 layer neura network
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
n = NeuralNetwork(3, 3, 3, 0.3)
print(n.query([1.0, 0.5, -1.5]))