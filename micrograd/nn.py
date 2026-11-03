from micrograd.engine import Value
import random


class Neuron:
    def __init__(self, nin):
        self.nin = nin
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        out = sum([wi * xi for (wi, xi) in zip(self.w, x)], self.b)
        return out


class Layer:
    def __init__(self, nin, nout):
        self.nout = nout
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, layer):
        return [n(layer) for n in self.neurons]


class MLP:
    def __init__(self, nin, nouts):
        self.nin = nin
        sizes = [nin] + nouts + [1]
        self.layers = [Layer(sizes[i - 1], sizes[i]) for i in range(1, len(sizes) - 1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        # print(x)
        return x[0].tanh()
