import numpy as np

class Network:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, input_layer):
        output = input_layer
        #forward prop
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def loss(self, y_true, y_pred):
        # loss
        return np.mean(np.power(y_true-y_pred, 2))

    def loss_prime(self, y_true, y_pred):
        # loss prime
        return 2*(y_pred-y_true)/y_true.size
    
    def train(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)

        for i in range(epochs):
            for j in range(samples):
                output = x_train[j]
                # calculate output
                for layer in self.layers:
                    output = layer.forward(output)
                
                # back prop
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)
            print("Epoch " + str(i) + "/" + str(epochs))
        print("Done training. ")