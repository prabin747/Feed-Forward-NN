# This is a program that performs image classification on digits using a forward propagation neural network.


import numpy as np
from sigmoid import sigmoid
from cost import cost
from predict import predict

# Suppose X has dimension m x n where m is the number of examples and n is the number of features.
# So the neural network will have n + 1 input layers including the bias (+1). 
# Suppose there are h hidden layers and p output layers.

# W1 is the weight matrix connecting input and hidden layers.
# W2 is the weight matrix connecting hidden and output layers.

# X is m by n and Y is m by p.
def train(Xin, Yin, h, num_iter, gamma, lambda_reg):

    m = np.shape(Xin)[0] # Same as np.shape(Yin)[0]
    n = np.shape(Xin)[1]
    p = np.shape(Yin)[1]

    # Initial weights.
    W1 = np.random.uniform(-0.12, 0.12, h * (n + 1)).reshape(h, n + 1) # h by n + 1 
    W2 = np.random.uniform(-0.12, 0.12, p * (h + 1)).reshape(p, h + 1) # p by h + 1

    counter = 1
    while counter <= num_iter:

        v = predict(Xin, Yin, h, W1, W2)
        A1, A2, A3 = v[0], v[1], v[2] # m by n + 1, m by h + 1, m by p

        J = cost(Yin, A3, lambda_reg, W1, W2)
        print(counter, J, sep = ': ')
    
        # For backpropagation
        delta3 = (A3 - Yin) # m by p
        W2bar = W2[:, 1:] # This is W2 with first column removed with shape p by h.
        A2prime = A2[:, 1:] # m by h
        delta2 = A2prime * np.dot(delta3, W2bar) # m by h

        # Change in weights
        # The first column of W1 and W2 are not regularized.
       
        dW2 = -gamma/m * np.dot(delta3.transpose(), A2) 
        dW2[:, 1:] = dW2[:, 1:] + lambda_reg/m * W2[:, 1:]
    
        dW1 = -gamma/m * np.dot(delta2.transpose(), A1) 
        dW1[:, 1:] = dW1[:, 1:] + lambda_reg/m * W1[:, 1:]

        W1 = W1 + dW1
        W2 = W2 + dW2
        
        counter = counter + 1

    return [W1, W2]


