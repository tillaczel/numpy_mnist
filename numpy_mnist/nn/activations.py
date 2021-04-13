import numpy as np

from numpy_mnist.nn import Layer


class LeakyReLu(Layer):
    def __init__(self, slope=0.01):
        super().__init__()

        self.slope = slope

    def forward(self, x):
        return np.where(x > 0, x, x * self.slope)

    def backward(self, x, dy):
        return np.where(x > 0, 1, self.slope) * dy, []


class ReLu(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x, dy):
        return (0<x).astype(np.int32)*dy, []


class SoftMax(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        logit = np.exp(x-np.max(x))
        logit = logit/(np.sum(logit, axis=1, keepdims=True))
        return logit

    def backward(self, x, dy):
        prob = self.forward(x)
        grad = -np.matmul(np.expand_dims(prob, axis=2), np.expand_dims(prob, axis=1))
        grad += np.expand_dims(prob, axis=1)*np.eye(prob.shape[1])
        return np.matmul(grad, np.expand_dims(dy, axis=2))[:, :, 0], []
