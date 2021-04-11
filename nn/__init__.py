class Model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        h = x.copy()
        for layer in self.layers:
            h = layer.forward(h)
        return h

    def train(self):
        for layer in self.layers:
            layer.train()

    def eval(self):
        for layer in self.layers:
            layer.eval()


class Layer:
    @property
    def params(self):
        return []

    def __init__(self):
        self._training_mode = True

    def forward(self, x):
        raise NotImplementedError

    def backward(self, x, dy):
        raise NotImplementedError

    def train(self):
        self._training_mode = True

    def eval(self):
        self._training_mode = False
