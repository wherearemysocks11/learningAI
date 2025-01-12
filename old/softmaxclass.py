import math
import numpy as np

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

#E = math.e

#exp_values = []
#for output in layer_output:
#    exp_values.append(E**output)
exp_values = np.exp(layer_outputs)

#print (exp_values)

#norm_values = []
#for value in exp_values:
#    norm_values.append( value / (sum(exp_values)) )
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)