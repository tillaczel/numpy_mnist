import numpy as np


class MSE:
    def __init__(self):
        pass

    def forward(self, y, y_hat):
        return np.mean(np.square(y-y_hat), axis=0)/2

    def backward(self, y, y_hat):
        return (y_hat-y)/y.shape[0]


class CrossEntropy:
    def __init__(self):
        pass

    def forward(self, y, y_hat):
        return -np.mean(np.sum(y*np.log(y_hat+1e-8), axis=1), axis=0)

    def backward(self, y, y_hat):
        return -y/(y_hat+1e-8)/y.shape[0]

