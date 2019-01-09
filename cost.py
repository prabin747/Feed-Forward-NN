import numpy as np

# lambda_reg is the regularization parameter.
def cost(Y, Yhat, lambda_reg, W1, W2):

    m = np.shape(Yhat)[0]
    p = np.shape(Yhat)[1]
    ID = np.ones(m * p).reshape(m, p)

    #print(W1.shape, W2.shape)
    M = -(Y * np.log(Yhat) + (ID - Y) * np.log(ID - Yhat))
    val1 = 1/m * M.sum()
    val2 = lambda_reg/(2 * m) * ((W1[:,1:] * W1[:,1:]).sum() + (W2[:,1:] * W2[:,1:]).sum())
    J = val1 + val2
    
    '''
    # Alternate method to derive total cost.
    J = 0
    for i in np.arange(0, m):
        y = Y[i, :]
        yhat = Yhat[i, :]
        ID = np.ones(len(y))
        cost = -1/m * (np.dot(y, np.log(yhat)) + np.dot((ID - y), np.log(ID - yhat)))
        J = J + cost
    '''

    return J
