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
        self.b = np.zeros((1, output_dim))

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


class BatchNorm(Layer):
    def __init__(self, input_dim, gamma=0.9, eps=1e-5):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        self.run_mu = np.zeros((1, input_dim))
        self.run_var = np.zeros((1, input_dim))

    def update_run_var(self, mu, var):
        self.run_mu = self.gamma * self.run_mu + (1.0 - self.gamma) * mu
        self.run_var = self.gamma * self.run_var + (1.0 - self.gamma) * var

    def forward(self, x):
        if self._training_mode:
            mu = np.mean(x, axis=0, keepdims=True)
            var = np.mean((x - mu) ** 2, axis=0, keepdims=True)
            self.update_run_var(mu, var)
        else:
            mu = self.run_mu
            var = self.run_var
        var = var+self.eps
        x = (x - mu) / np.sqrt(var)
        return x

    def backward(self, x, dy):
        return dy, []
