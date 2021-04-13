import numpy as np
import os


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

    def save(self, path):
        params_dict = dict()
        for i, layer in enumerate(self.layers):
            params_dict[i] = dict()
            for j, param in enumerate(layer.params):
                params_dict[i][j] = param
        dir_path = os.path.dirname(os.path.abspath(path))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        np.save(path, params_dict)

    def load(self, path):
        params_dict = np.load(path, allow_pickle=True)
        for i, layer in enumerate(self.layers):
            for j, param in enumerate(layer.params):
                param *= 0
                param += params_dict.item()[i][j]



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
