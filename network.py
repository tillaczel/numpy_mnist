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


class LeakyReLu:
    @property
    def params(self):
        return []

    def __init__(self, slope=0.01):
        self.slope = slope

    def forward(self, x):
        return np.where(x > 0, x, x * self.slope)

    def backward(self, x, dy):
        return np.where(x > 0, 1, self.slope) * dy, []

class ReLu:
    @property
    def params(self):
        return []
    
    def __init__(self):        
        pass
    
    def forward(self, x):
        return np.maximum(0, x)
    
    def backward(self, x, dy):
        return (0<x).astype(np.int32)*dy, []
    
    
class SoftMax:
    @property
    def params(self):
        return []
    
    def __init__(self):        
        pass
    
    def forward(self, x):
        logit = np.exp(x-np.max(x))
        return logit/(np.sum(logit, axis=1, keepdims=True))
    
    def backward(self, x, dy):
        prob = self.forward(x)
        grad = -np.matmul(np.expand_dims(prob, axis=2), np.expand_dims(prob, axis=1))
        grad += np.expand_dims(prob, axis=1)*np.eye(prob.shape[1])
        return np.matmul(grad, np.expand_dims(dy, axis=2))[:, :, 0], []
    

class Model:
    def __init__(self, layers):
        self.layers = layers
        
    def forward(self, x):
        h = x.copy()
        for layer in self.layers:
            h = layer.forward(h)
        return h


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
        return -np.mean(np.sum(y*np.log(y_hat), axis=1), axis=0)
    
    def backward(self, y, y_hat):
        return -y/(y_hat)/y.shape[0]
    
    
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

