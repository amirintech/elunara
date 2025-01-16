import numpy as np
from engine import Value


class Unit:
    def __init__(self, n_input):
        self.w = [
            Value(np.random.uniform(-1, 1), label="weight") for _ in range(n_input)
        ]
        self.b = Value(np.random.uniform(-1, 1), label="bias")

    def __call__(self, x):
        activation = (np.array(x) @ np.array(self.w)) + self.b
        out = activation.relu()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_input, n_units):
        self.units = [Unit(n_input) for _ in range(n_units)]

    def __call__(self, x):
        outs = [u(x) for u in self.units]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for u in self.units for p in u.parameters()]


class MLP:
    def __init__(self, n_input, layers):
        sizes = [n_input] + layers
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(layers))]

    def __call__(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]
