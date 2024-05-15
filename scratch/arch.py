import numpy as np


class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.zeros(output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        e2x = np.exp(2 * x)
        return (e2x - 1) / (e2x + 1)

    def relu(self, x):
        return np.maximum(0, x)

    def activation(self, x):
        return self.sigmoid(x)

    def forward(self, inputs):
        result = []
        for weights_row, bias in zip(self.weights, self.biases):
            sum_result = 0
            for weight, inpt in zip(weights_row, inputs):
                sum_result += weight * inpt
            sum_result += bias
            result.append(self.activation(sum_result))
        return result

    def node_cost(self, output_activation, expected_output):
        return (output_activation - expected_output) ** 2


class NN:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def predict(self, inputs):
        outputs = self.forward(inputs)
        max_index = 0
        max_value = outputs[0]
        for i in range(1, len(outputs)):
            if outputs[i] > max_value:
                max_value = outputs[i]
                max_index = i
        return max_index

    def cost(self, data_point):
        outputs = self.forward(data_point["inputs"])


def total_cost(data, network):
    total_cost = 0
    for data_point in data:
        total_cost += cost(data_point, network)
    return total_cost / len(data)
