import numpy as np
from sigmoid import sigmoid
from cost import cost

def predict(X, Y, h, W1, W2):

    # print(W1.shape) h by n + 1
    # print(W2.shape) p by h + 1

    m = np.shape(X)[0] # Same as np.shape(y)[0]
    A1 = np.insert(X, 0, np.ones(m), axis=1) # m by n + 1
    
    Z2 = np.dot(A1, W1.transpose()) # m by h
    A2prime = sigmoid(Z2) # m by h
    A2 = np.insert(A2prime, 0, np.ones(m), axis=1) # m by h + 1

    Z3 = np.dot(A2, W2.transpose()) # m by p
    A3 = sigmoid(Z3) # m by p. Same as Yhat.

    return [A1, A2, A3]