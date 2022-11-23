import numpy as np

class Layer:
    def __init__(self, neurons, next_layer_nuerons):
        self.weights = np.random.rand(neurons, next_layer_nuerons) - 0.5
        self.biases = np.random.rand(1, next_layer_nuerons) - 0.5

    def forward(self, input_data):
        self.input_before = input_data
        self.input_after = np.dot(self.input_before,self.weights)
        self.output = 1 - 1/np.exp(self.input_after + self.biases)
        return self.output

    def sigmoid_prime(x):
        print(np.exp(-x)/(np.exp(-x) + 1)**2)
        return np.exp(-x)/(np.exp(-x) + 1)**2

    def backward(self, output_error, learning_rate):
        # fc layer back prop
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input_before.T, output_error)

        self.sigmoid_prime(self.input_after) * input_error

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error

        #activation back prop
        return self.sigmoid_prime(self.input_after) * input_error
