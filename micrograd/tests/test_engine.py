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
    e.backward()
    assert a.gradient == 4.0  # 3.0 from c + 1.0 from d
    assert b.gradient == 3.0  # 2.0 from c + 1.0 from d

def test_duplicate_backward():
    a = Value(2.0)
    b = a + a
    b.backward()
    assert a.gradient == 2.0  # 1.0 from each addition

def test_pow():
    a = Value(2.0)
    b = a ** 3
    assert b._children == {a}
    assert b._op == "**3"
    assert b.data == 8.0
    b.backward()
    assert a.gradient == 12.0  # derivative of x^3 at x=2 is 3*2^2 = 12

def test_true_div():
    a = Value(4.0)
    b = Value(2.0)
    c = a / b
    assert c.data == 2.0
    c.backward()
    assert a.gradient == 0.5  # derivative of x/y w.r.t x is 1/y
    assert b.gradient == -1.0  # derivative of x/y w.r.t y is -x/y^2

def test_neg():
    a = Value(2.0)
    b = -a
    assert b.data == -2.0
    b.backward()
    assert a.gradient == -1.0  # derivative of -x w.r.t x is -1

def test_sub():
    a = Value(5.0)
    b = Value(3.0)
    c = a - b
    assert c.data == 2.0
    c.backward()
    assert a.gradient == 1.0  # derivative of x-y w.r.t x is 1
    assert b.gradient == -1.0  # derivative of x-y w.r.t y is -1