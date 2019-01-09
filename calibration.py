from predict import predict
from cost import cost
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

W = sio.loadmat('weights_ref.mat')
W1 = np.array(W['Theta1'])
W2 = np.array(W['Theta2'])

data = sio.loadmat('data.mat')
Xin = np.array(data['X'])
yin = np.array(data['y'])  # This data was for octave. So zero is labeled as 10.

# Convert each output digit from 0 through 9 to a vector in 10 dimensions. 
# 1 is represented as [1, 0, 0, 0, 0, 0, 0, 0, 0, 0].
# 2 is represented as [0, 1, 0, 0, 0, 0, 0, 0, 0, 0].
# 3 is represented as [0, 0, 1, 0, 0, 0, 0, 0, 0, 0].
# ...
# 0 is represented as [0, 0, 0, 0, 0, 0, 0, 0, 0, 1].

Yin = np.zeros(len(yin) * 10).reshape(len(yin), 10)
for j in np.arange(0, yin.shape[0]):
    if yin[j][0] == 10:
        Yin[j, 9] = 1
    else:
        Yin[j, yin[j][0]-1] = 1

pred = predict(Xin, Yin, 25, W1, W2)
Yhat = pred[2]

cost0 = cost(Yin, Yhat, 0, W1, W2)
cost1 = cost(Yin, Yhat, 1, W1, W2)
print(cost0, cost1)