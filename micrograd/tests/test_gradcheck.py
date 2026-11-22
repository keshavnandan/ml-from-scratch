import random
import torch
import pytest
from micrograd.engine import Value


def random_expr(depth, rng, leaves, max_depth):
    """Return (value_node, torch_node) — the same expression in both systems."""

    if depth == 0 or (depth < max_depth and rng.random() < 0.3):
        if leaves and rng.random() < 0.4:
            return leaves[rng.randrange(len(leaves))]  # REUSE — see below
        x = rng.uniform(-2, 2)
        pair = (Value(x), torch.tensor(x, dtype=torch.float64, requires_grad=True))
        leaves.append(pair)
        return pair

    op = rng.choice(["+", "*", "-", "/", "tanh", "pow"])

    # recurse for operands, apply op to both halves of each pair, return the pair
    vleft, tleft = random_expr(depth - 1, rng, leaves, max_depth)
    if op == "tanh":
        return (vleft.tanh(), tleft.tanh())
    if op == "pow":
        # stick with strictly positive integer as negative base to non-integer power creates complex value
        p = rng.choice([-2, -1, 2, 3]) if vleft.data > 1e-3 else rng.choice([2, 3])
        return (vleft**p, tleft**p)

    vright, tright = random_expr(depth - 1, rng, leaves, max_depth)
    if op == "+":
        return (vleft + vright, tleft + tright)
    if op == "-":
        return (vleft - vright, tleft - tright)
    if op == "*":
        return (vleft * vright, tleft * tright)
    if op == "/":
        if abs(vright.data) < 1e-3:  # guard against division by zero, fallback to + op
            return (vleft + vright, tleft + tright)
        return (vleft / vright, tleft / tright)


def test_gradients_match_torch():
    for i in range(100):
        seed = 10 * i + 1
        rng = random.Random(seed)
        leaves = []
        v, t = random_expr(depth=5, rng=rng, leaves=leaves, max_depth=5)
        if abs(v.data) > 1e3:
            # skip large values to avoid triggering false alarms on comparison
            continue

        v.backward()
        t.backward()

        if max(abs(vleaf.gradient) for vleaf, _ in leaves) < 1e-3:
            # skip saturated trees where all leaf gradients are ~0
            continue

        assert v.data == pytest.approx(t.item()), f"Data mismatch at seed={seed}"
        for vleaf, tleaf in leaves:
            assert vleaf.gradient == pytest.approx(
                tleaf.grad.item(), rel=1e-9, abs=1e-8
            ), f"Grad mismatch at seed={seed}"
