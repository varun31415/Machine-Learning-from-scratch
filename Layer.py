import numpy as np

class Layer:
    def __init__(self, neurons, next_layer_nuerons):
        self.weights = np.random.rand(neurons, next_layer_nuerons) - 0.5
        self.biases = np.random.rand(1, next_layer_nuerons) - 0.5

    def forward(self, input_data):
        self.input_before = input_data
        self.input_after = np.dot(self.input_before,self.weights) + self.biases
        self.output = self.tanh(self.input_after)
        return self.output

    def sigmoid_prime(self, x):
        return np.exp(-x)/(np.exp(-x) + 1)**2

    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_prime(self, x):
        return 1-np.tanh(x)**2

    def backward(self, error, learning_rate):
        output_error = self.tanh_prime(self.input_after) * error

        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input_before.T, output_error)

        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * output_error

        return input_error
