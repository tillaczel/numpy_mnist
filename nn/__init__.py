class Model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        h = x.copy()
        for layer in self.layers:
            h = layer.forward(h)
        return h