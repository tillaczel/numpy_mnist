import numpy as np

from nn import Layer


class Linear(Layer):
    @property
    def params(self):
        return [self.w, self.b]

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.w = np.random.normal(0, (2/input_dim)**0.5, (input_dim, output_dim))
        self.b = np.random.normal(0, 1, (1, output_dim))

    def forward(self, x):
        return x@self.w+self.b

    def backward(self, x, dy):
        return dy@self.w.T, [x.T@dy, np.ones((x.shape[0], 1)).T@dy]


class DropOut(Layer):
    def __init__(self, p):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, x):
        self.mask = np.random.choice([0, 1], size=x.shape, p=[self.p, 1-self.p])
        return x*self.mask

    def backward(self, x, dy):
        return dy*self.mask, []
