import numpy as np


class Neuron:
    def __init__(self):
        self.bias = 0

    # sigmoid activation function
    def activate(self, inputs, weights):
        z = np.dot(inputs, weights) + self.bias
        return 1 / (1 + np.exp(-z))


def derivative(x):
    return x * (1 - x)
