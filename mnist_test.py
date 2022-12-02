import pickle
from keras.datasets import mnist
from Network import Network

(x_train, y_train), (test_x, y_test) = mnist.load_data()

x_test = []
for x in test_x:
    x_reshaped = x.reshape(1, 784)
    x_test.append(x_reshaped)
    
net = pickle.load(open("network_file1.pickle", "rb"))
print(net.layers[2].weights)
print(net.predict(x_test[1]))
print(y_test[1])
