import numpy as np

def mean_squared_error(x, y):
    return np.mean(np.dot(x - y, x - y))

def accuracy(x, y):
    v = np.zeros(len(x))
    v[x != y] = 1
    return sum(v)/len(v) * 100 # In percent.