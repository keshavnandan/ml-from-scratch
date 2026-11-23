import math
import random
import pytest
from micrograd.engine import Value
from micrograd.nn import Neuron, Layer, MLP


def test_neuron():
    n = Neuron(3)
    w = [wi.data for wi in n.w]
    b = n.b.data
    x = [Value(2.0), Value(3.0), Value(-1.0)]
    y = n(x)
    assert y.data == pytest.approx(math.tanh(w[0] * 2.0 + w[1] * 3.0 + w[2] * -1.0 + b))


def test_neuron_backward():
    n = Neuron(3)
    x = [Value(2.0), Value(3.0), Value(-1.0)]
    y = n(x)
    y.gradient = 1.0
    y.backward()
    local = 1.0 - y.data**2  # d tanh(z)/dz
    for i in range(3):
        assert n.w[i].gradient == pytest.approx(x[i].data * local)
    assert n.b.gradient == pytest.approx(local)


def test_layer():

    l = Layer(2, 2)
    x = [Value(2.0), Value(3.0)]
    y = l(x)
    assert len(y) == 2
    for i in range(2):
        z = (
            sum([l.neurons[i].w[j].data * x[j].data for j in range(2)])
            + l.neurons[i].b.data
        )
        assert y[i].data == pytest.approx(math.tanh(z))


def test_mlp():

    mlp = MLP(3, [2, 1])
    x = [Value(2.0), Value(3.0), Value(-1.0)]
    y = mlp(x)
    assert isinstance(y, Value)


def test_network_is_not_affine():
    """An affine map satisfies f(a)-f(0) + f(b)-f(0) == f(a+b)-f(0).
    A real MLP must violate this."""
    random.seed(0)
    m = MLP(2, [4, 1])

    def f(xs):
        return m([Value(v) for v in xs]).data

    zero = f([0.0, 0.0])
    a, b = [1.0, 0.5], [-0.3, 2.0]
    lhs = (f(a) - zero) + (f(b) - zero)
    rhs = f([a[0] + b[0], a[1] + b[1]]) - zero
    assert abs(lhs - rhs) > 1e-6, "network behaves affinely — nonlinearity missing"


def test_mlp_layer_shapes():
    m = MLP(3, [4, 4, 1])
    assert len(m.layers) == 3  # 3->4, 4->4, 4->1
    assert [len(l.neurons) for l in m.layers] == [4, 4, 1]
    assert [len(l.neurons[0].w) for l in m.layers] == [3, 4, 4]


def test_mlp_parameter_count():
    m = MLP(3, [4, 4, 1])
    total = sum(len(n.w) + 1 for l in m.layers for n in l.neurons)
    assert total == (3 * 4 + 4) + (4 * 4 + 4) + (4 * 1 + 1)  # 41


def test_mlp_backward_gradients_are_live():
    random.seed(0)
    m = MLP(3, [4, 4, 1])
    y = m([Value(2.0), Value(3.0), Value(-1.0)])
    y.backward()
    grads = [w.gradient for l in m.layers for n in l.neurons for w in n.w]
    assert any(g != 0.0 for g in grads), "no gradient reached the weights"
    assert all(g == g for g in grads), "NaN in gradients"


def test_neuron_rejects_wrong_input_length():
    n = Neuron(3)
    with pytest.raises(ValueError):
        n([Value(1.0), Value(2.0)])  # needs zip(strict=True)


def test_mlp_requires_single_output():
    with pytest.raises(AssertionError):
        MLP(3, [4, 2])


def test_tanh_large_input_does_not_overflow():
    assert Value(1000.0).tanh().data == 1.0
    assert Value(-1000.0).tanh().data == -1.0


def test_neuron_parameters():
    n = Neuron(3)
    params = n.parameters()
    assert len(params) == 4  # 3 weights + 1 bias
    assert all(isinstance(p, Value) for p in params)


def test_layer_parameters():
    l = Layer(2, 3)
    params = l.parameters()
    assert len(params) == 9  # 3 neurons * (2 weights + 1 bias)
    assert all(isinstance(p, Value) for p in params)


def test_mlp_parameters():
    m = MLP(2, [3, 1])
    params = m.parameters()
    assert len(params) == 13  # (3*(2+1)) + (1*(3+1)) = 9 + 4 = 13
    assert all(isinstance(p, Value) for p in params)


def test_mlp_zero_grad():
    m = MLP(2, [3, 1])
    # Set some gradients
    for p in m.parameters():
        p.gradient = random.uniform(-1, 1)
    m.zero_grad()
    assert all(p.gradient == 0.0 for p in m.parameters())
