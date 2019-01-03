import numpy as np
import scipy.special as spec
import matplotlib.pyplot as plt


class NeuralNetwork:
    """ Three layers Neural Network with the Stochastic Gradient Descent

        Parameters
        ------------
        inodes : int
            Number of nodes in the first layer of the network.
        hnodes : int
            Number of nodes in the second layer of the network.
        onodes : int
            Number of nodes in the third layer of the network.
        eta : float
            Learning rate (between 0.0 and 1.0).

        Attributes
        ------------
        wih : {array-like}, shape = [n_hnodes, n_inodes]
            Weights between 1 and 2 layers of the network, where
            n_hnodes is a number of nodes in the 2nd layer and
            n_inodes is a number of nodes in the 1st layer

        who : {array-like}, shape = [n_onodes, n_hnodes]
            Weights between 2 and 3 layers of the network, where
            n_onodes is a number of nodes in 3rd layer and
            n_hnodes is a number of nodes in the 2nd layer

        activation_func : float
            Sigmoid function (between 0.0 and 1.0)
    """
    def __init__(self, inodes, hnodes, onodes, eta=0.1):
        self.inodes = inodes
        self.hnodes = hnodes
        self.onodes = onodes
        self.eta = eta

        # Weights in range of normal distribution with standard div 1/sqrt(n)
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        self.activation_func = lambda x: spec.expit(x)

    def train(self, input_list, target_list):
        """ Train the network.

        :param input_list : {array-like}, shape = [n_records, n_features]
            Data for the training features, where n_records is the number of samples
            and n_features is the number of features.
        :param target_list : array-like, shape = [n_records]
            Data for the target feature.

        :return : None
        """
        # Targets => training data
        targets = np.array(target_list, ndmin=2).T
        inputs = np.array(input_list, ndmin=2).T

        # Propagate through network
        hidden_in = np.dot(self.wih, inputs)
        hidden_out = self.activation_func(hidden_in)
        final_in = np.dot(self.who, hidden_out)
        final_out = self.activation_func(final_in)

        # output_errors => Refine weights between hidden and final layers
        # hidden_errors => Refine weight between input and hidden layers
        output_errors = targets - final_out
        hidden_errors = np.dot(self.who.T, output_errors)

        # Update weights between hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_out * (1.0 - final_out)), np.transpose(hidden_out))

        # Uodate weights between input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_out * (1.0 - hidden_out)), np.transpose(inputs))

    def query(self, input_list):
        """ Return predicted number (between 0 and 9)

        :param input_list : {array-like}, shape = [n_records, n_features]
            Data for the training features, where n_records is the number of samples
            and n_features is the number of features.

        :return : array of n_onodes floats
            highest float in the array corresponds to the predicted index,
            where n_onodes is a number of nodes in the 3rd layer.
        """
        # Convert inputs into 2-d array for the dot product
        inputs = np.array(input_list, ndmin=2).T

        # Calculate hidden layer (Applying Sigmoid on the dot product)
        hidden_in = np.dot(self.wih, inputs)
        hidden_out = self.activation_func(hidden_in)

        # Calculate final layer
        final_in = np.dot(self.who, hidden_out)
        return self.activation_func(final_in)

# Input, hidden and output nodes
input_nodes = 784  # 28x28 image
hidden_nodes = 100  # any reasonable number between 10 and 784, to avoid overfitting and underfitting
output_nodes = 10  # number of possible numbers between [0..9]

# Experimentally derived, through brute force( GridSearch for the hyperparameter tuning )
eta = 0.2

# Passes over the training dataset
epochs = 2

# Create NN
nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, eta)

# TRAINING PHASE (For 2 epochs
# Load training data from the subset of MNIST.csv of 100 elements
training_data = open("MNIST/mnist_train_10k.csv", "r")
training_list = training_data.readlines()
training_data.close()

# Train NN
for e in range(epochs):
    for label in training_list:
        # Rescale inputs from the training_list in range (0.00, 1.00]
        inputs = np.asfarray(label.split(',')[1:]) / 255 * 0.99 + 0.01
        # Create target array full of 0.01 and label of 0.99
        targets = np.zeros(output_nodes) + 0.01
        targets[int(label.split(',')[0])] = 0.99
        nn.train(inputs, targets)

# TESTING PHASE
# Load testing data from the subset of MNIST.csv of 10 elements
testing_data = open("MNIST/mnist_test_1k.csv", "r")
testing_list = testing_data.readlines()
testing_data.close()

# Score for each record in the test set
score = []

for record in testing_list:
    correct_answer = int(record.split(',')[0])
    # Print(correct_answer, "correct answer")
    # Get inputs
    inputs = np.asfarray(record.split(',')[1:]) / 255.0 * 0.99 + 0.01
    outputs = nn.query(inputs)
    # Get label spitted by the NN
    answer = np.argmax(outputs)
    # Print(answer, "network output")
    # Add 1 => correct answer/ 0 => incorrect
    if(answer == correct_answer):
        score.append(1)
    else:
        score.append(0)
# Evaluate performance
score_array = np.asarray(score)
print("Performance in %: ", score_array.sum() / score_array.size)