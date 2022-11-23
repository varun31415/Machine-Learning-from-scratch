from Layer import Layer
from Network import Network
import numpy as np

#XOR test
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

network = Network()
network.add(Layer(2,3))
network.add(Layer(3,1))

network.train(x_train, y_train, 1000, 0.1)