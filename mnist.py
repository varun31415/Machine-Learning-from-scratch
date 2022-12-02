from keras.datasets import mnist
from Network import Network
from Layer import Layer
import numpy as np
import pickle

(train_x, y_train), (test_x, y_test) = mnist.load_data()

net = Network()
net.add(Layer(784, 16))
net.add(Layer(16, 16))
net.add(Layer(16, 10))

x_train = []
x_test = []

for x in train_x:
    x_reshaped = x.reshape(1, 784)
    x_train.append(x_reshaped)

net.train(x_train, y_train, 10, 0.1)

network_file = open("network_file1.pickle", "wb")
pickle.dump(net, network_file)
network_file.close()