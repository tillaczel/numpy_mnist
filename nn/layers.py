import numpy as np


class Linear:
    @property
    def params(self):
        return [self.w, self.b]

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.w = np.random.normal(0, (2/input_dim)**0.5, (input_dim, output_dim))
        self.b = np.random.normal(0, 1, (1, output_dim))

    def forward(self, x):
        return x@self.w+self.b

    def backward(self, x, dy):
        return dy@self.w.T, [x.T@dy, np.ones((x.shape[0], 1)).T@dy]