import numpy as np


class Optimizer:
    def __init__(self, model, loss, lr=0.1):
        self.layers = model.layers
        self.loss = loss
        self.lr = lr
        
    def step(self, x, y):
        activations = [x.copy()]
        # Forward
        for layer in self.layers:
            activations.append(layer.forward(activations[-1]))
        # Backward
        dx = self.loss.backward(y, activations[-1])
        for i, layer in enumerate(self.layers[::-1]):
            dx, dws = layer.backward(activations[-2-i], dx)
            for j, (param, dw) in enumerate(zip(layer.params, dws)):
                self.update_param(param, dw, i, j)
        return self.loss.forward(y, activations[-1])

    def update_param(self, param, dw, i, j):
        raise NotImplementedError

    def decay_learning_rate(self, i, decay_fraction=1/2, decay_frequency=10):
        if (i+1) % decay_frequency == 0:
            self.lr = self.lr * decay_fraction



class SGD(Optimizer):
    def __init__(self, model, loss, lr):
        super().__init__(model, loss, lr)

    def update_param(self, param, dw, i, j):
        param += -self.lr*dw


class Momentum(Optimizer):
    def __init__(self, model, loss, lr, beta):
        super().__init__(model, loss, lr)
        self.v = [[np.zeros(p.shape) for p in layer.params] for layer in model.layers][::-1]
        self.beta = beta

    def update_param(self, param, dw, i, j):
        self.v[i][j] = self.beta*self.v[i][j]+(1-self.beta)*dw
        param += -self.lr*self.v[i][j]

