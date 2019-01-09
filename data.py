import scipy.io as sio
import numpy as np

data = sio.loadmat('data.mat')

Xin = np.array(data['X'])
yin = np.array(data['y']) # This data was for octave. So zero is labeled as 10.

# Convert each output digit from 0 through 9 to a vector in 10 dimensions. 
# 1 is represented as [1, 0, 0, 0, 0, 0, 0, 0, 0, 0].
# 2 is represented as [0, 1, 0, 0, 0, 0, 0, 0, 0, 0].
# 3 is represented as [0, 0, 1, 0, 0, 0, 0, 0, 0, 0].
# ...
# 0 is represented as [0, 0, 0, 0, 0, 0, 0, 0, 0, 1].

# The other digits are represented by a 1 in the corresponding position.
Yin = np.zeros(len(yin) * 10).reshape(len(yin), 10)
for j in np.arange(0, yin.shape[0]):
    if yin[j][0] == 10:
        Yin[j, 9] = 1
    else:
        Yin[j, yin[j][0]-1] = 1
