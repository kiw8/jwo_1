import numpy as np

inputs = [[1.0,2.0,3.0,2.5],
          [2.0,5.0,-1.0,2.0],
          [-1.5,2.7,3.3,-0.8],
          [2.0, 1.8, -0.5, -1.3],
          [1.3, 2.25, 1.8, 1.5]]

weights = [[0.2,0.8,-0.5,1.0],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]

biases = [2.0,3.0,0.5]

layers_outputs = np.dot(inputs,np.array(weights).T)+biases
print(layers_outputs)

weights_2 = [[0.8,0.1,-0.3],
           [0.4,-0.51,0.13],
           [-0.13,-0.25,0.11]]

biases_2 = [1.0,2.0,0.8]

layers_outputs2 = np.dot(layers_outputs, np.array(weights_2).T)+biases_2
print(layers_outputs2)







