import numpy as np
#import matplotlib as mpl

# a network of 4 inputs, 3 neurons (also the outputs)
inputs = [1.0, 2.0, 3.0]
weights = [[0.2, 0.8, -0.5, 1.0],   #each list within the list represents weights for a single neuron
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

inputs = np.array([inputs])
weights = np.array([weights])

bias = [2, 3, 0.5]

print(f"Inputs: {inputs}") 
print(f"weights: {weights}")
print(f"bias: {bias}")
