# To activate this environment, use
#
#     $ conda activate my-env
#
# To deactivate an active environment, use
#
#     $ conda deactivate

import nnfs
import numpy as np
import matplotlib as mpl
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, num_inputs, num_neurons):
        self.weights = 0.01 * np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense1.forward(X)
activation1.forward(dense1.output)

print(activation1.output)