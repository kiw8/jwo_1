import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = (np.random.uniform(0,1,(n_inputs,n_neurons)))
        self.biases = np.random.uniform(0,1,(1,n_neurons))


    def forward(self, inputs):
        return np.dot(inputs, np.array(self.weights)) + self.biases


    def forward_with_relu(self, inputs):
        output = self.forward(inputs)
        return np.maximum(0, output)


inputs, y = spiral_data(samples=100, classes=3)
plt.scatter(inputs[:, 0], inputs[:, 1], c=y, cmap='brg')
plt.show()


DNN = Layer_Dense(2, 5)
outputs = DNN.forward_with_relu(inputs)
print(outputs)