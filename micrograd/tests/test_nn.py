from micrograd.engine import Value
from micrograd.nn import Neuron
from micrograd.nn import Layer
from micrograd.nn import MLP


def test_neuron():
    n = Neuron(3)
    w = [wi.data for wi in n.w]
    b = n.b.data
    x = [Value(2.0), Value(3.0), Value(-1.0)]
    y = n(x)
    assert y.data == w[0] * 2.0 + w[1] * 3.0 + w[2] * -1.0 + b


def test_neuron_backward():
    n = Neuron(3)
    x = [Value(2.0), Value(3.0), Value(-1.0)]
    y = n(x)
    y.gradient = 1.0
    y.backward()
    for i in range(3):
        assert n.w[i].gradient == x[i].data * y.gradient
    assert n.b.gradient == 1.0 * y.gradient


def test_layer():

    l = Layer(3, 2)
    x = [Value(2.0), Value(3.0)]
    y = l(x)
    assert len(y) == 2
    for i in range(2):
        assert (
            y[i].data
            == sum([l.neurons[i].w[j].data * x[j].data for j in range(2)])
            + l.neurons[i].b.data
        )


def test_mlp():

    mlp = MLP(3, [2, 2])
    x = [Value(2.0), Value(3.0), Value(-1.0)]
    y = mlp(x)
    assert isinstance(y, Value)


def test_mlp_backward():
    mlp = MLP(3, [2, 2])
    x = [Value(2.0), Value(3.0), Value(-1.0)]
    y = mlp(x)
    y.gradient = 1.0
    y.backward()
    for layer in mlp.layers:
        for neuron in layer.neurons:
            for w in neuron.w:
                assert w.gradient is not None
            assert neuron.b.gradient is not None
