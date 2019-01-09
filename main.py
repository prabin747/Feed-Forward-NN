from train import train
from data import Xin, Yin, yin
from predict import predict
from error import mean_squared_error, accuracy

import numpy as np

W = train(Xin, Yin, 25, 400, 0.05, 4)
W1, W2 = W[0], W[1]
pred = predict(Xin, Yin, 25, W1, W2)
yhat = np.argmax(pred[2], axis=1)
yhat[yhat == 0] = 10
y = [yin[x][0] for x in np.arange(0, len(yin))]

print(accuracy(y, yhat))

