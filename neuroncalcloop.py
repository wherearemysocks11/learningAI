inputs = [1.0, 2.0, 3.0, 2.5]

neurons_weights = [[0.2, 0.8, -0.5, 1.0],
                   [0.5, -0.91, 0.26, -0.5],
                   [-0.26, -0.27, 0.17, 0.87]]
# a list of sets (lists) of weights for each neuron

biases = [2, 3, 0.5]
# a bias for every neuron

layer_outputs = []

for neuron_weights, neuron_bias in zip(neurons_weights, biases):
    # extracts the weights and relevant biases for this neuron/iteration

    neuron_output = 0
    # this resets the value of the neuron output every time a new one is calculated

    for input, weight in zip(inputs, neuron_weights):
        # extracts a input and its relevant weight for this input into the neuron

        neuron_output += (input * weight)
        # adding the sum of the input and its weight to the final neuron value

    neuron_output += neuron_bias
    # adding the bias to the neurons final value

    layer_outputs.append(neuron_output)
    # adding the neuron output to the list of neuron outputs (a.k.a a layer output)

print(f"Output: {layer_outputs}")