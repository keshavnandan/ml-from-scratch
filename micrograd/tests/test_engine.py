import pytest
from micrograd.engine import Value

def test_add_returns_value():
    out = Value(2.0) + Value(3.0)
    assert isinstance(out, Value), "operators must stay inside the graph"
    assert out.data == 5.0

def test_mul():
    assert (Value(4.0) * Value(2.5)).data == 10.0

def test_op_scalar():
    assert (Value(4.0) * 2).data == 8.0
    assert (Value(4.0) + 2).data == 6.0

def test_op_scalar_reverse():
    assert (2 * Value(4.0)).data == 8.0
    assert (2 + Value(4.0)).data == 6.0

def test_chaining():
    a, b, c = Value(2.0), Value(3.0), Value(4.0)
    assert ((a + b) * c).data == 20.0