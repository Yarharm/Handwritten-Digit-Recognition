import numpy as np



# neural network class
class neural_network:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # number of nodes for the input, hidden and output layers
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        # learning rate
        self.lr = learning_rate

        # matrices wih(input => hidden) and who(hidden => output)
        # weights in range of normal distribution with standart div 1/sqrt(n)
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

    # adjust the weights
    def train(self):
        pass

    # get output from the neural network
    def query(self):
        pass

