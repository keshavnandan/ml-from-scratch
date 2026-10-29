import pytest
from micrograd.engine import Value


def test_add():
    a = Value(2.0)
    b = Value(3.0)
    out = a + b
    assert out.data == 5.0
    assert a in out._children
    assert b in out._children
    assert out._op == "+"


def test_mul():
    a = Value(2.0)
    b = Value(3.0)
    out = a * b
    assert out.data == 6.0
    assert a in out._children
    assert b in out._children
    assert out._op == "*"


def test_op_scalar():
    assert (Value(4.0) * 2).data == 8.0
    assert (Value(4.0) + 2).data == 6.0


def test_op_scalar_reverse():
    assert (2 * Value(4.0)).data == 8.0
    assert (2 + Value(4.0)).data == 6.0


def test_chaining():
    a, b, c = Value(2.0), Value(3.0), Value(4.0)
    d = a + b
    out = d * c
    assert out.data == 20.0
    assert b not in out._children


def test_backward_chain():
    a, b = Value(2.0), Value(3.0)
    c = a * b
    c.gradient = 1.0
    c._backward()
    assert a.gradient == 3.0
    assert b.gradient == 2.0

    d = a + b
    d.gradient = 1.0
    d._backward()
    assert a.gradient == 4.0  # 3.0 from previous +
    assert b.gradient == 3.0  # 2.0 from previous +

def test_backward_topo():
    a, b = Value(2.0), Value(3.0)
    c = a * b
    d = a + b
    e = c + d
    e.gradient = 1.0
    e.backward()
    assert a.gradient == 4.0  # 3.0 from c + 1.0 from d
    assert b.gradient == 3.0  # 2.0 from c + 1.0 from d

def test_duplicate_backward():
    a = Value(2.0)
    b = a + a
    b.gradient = 1.0
    b._backward()
    assert a.gradient == 2.0  # 1.0 from each addition