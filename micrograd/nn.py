from micrograd.engine import Value
import random


class Neuron:
    def __init__(self, nin):
        self.nin = nin
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        out = sum([wi * xi for (wi, xi) in zip(self.w, x, strict=True)], self.b)
        return out.tanh()


class Layer:
    def __init__(self, nin, nout):
        self.nout = nout
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]


class MLP:
    def __init__(self, nin, nouts):
        assert nouts and nouts[-1] == 1, "MLP must end in a single output neuron"
        self.nin = nin
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i - 1], sizes[i]) for i in range(1, len(sizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x[0]
