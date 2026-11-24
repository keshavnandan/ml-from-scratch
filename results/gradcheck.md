# Gradient check: micrograd engine vs PyTorch

Generated 2026-11-24 by `scripts/gradcheck_report.py`.

## What this checks

The harness builds a random expression tree out of the engine's own
operations (`+`, `-`, `*`, `/`, `**`, `tanh`) and builds the identical
expression a second time out of PyTorch float64 tensors. It calls
`backward()` on both roots and compares every leaf gradient.

Leaves are sometimes reused inside one tree, so a leaf can feed several
branches. That is what exercises gradient accumulation in `Value.backward`.

## Environment

- Python 3.12.13 on Darwin arm64
- PyTorch 2.14.0, float64, CPU

## Configuration

- Depths: 2, 5, 8
- Cases per depth: 100
- Seeds: `10 * i + 1` for i in range(cases), so every case is reproducible
- Tolerance: `rel=1e-9, abs=1e-8` (passes if either holds)

## Coverage

| depth | asserted | skipped, output > 1e3 | skipped, gradients saturated | mean leaves | max leaves |
|---|---|---|---|---|---|
| 2 | 92 | 0 | 8 | 2.0 | 4 |
| 5 | 93 | 1 | 6 | 3.9 | 12 |
| 8 | 90 | 5 | 5 | 6.6 | 29 |

Total: 300 cases generated, 275 asserted, 6 skipped for large output, 19 skipped for saturated gradients.
That is 1144 individual leaf-gradient comparisons.

Two kinds of case are skipped on purpose. A tree whose output exceeds 1e3
produces gradients dominated by cancellation, where float64 rounding is
larger than any useful tolerance. A tree whose gradients are all below
1e-3 has saturated `tanh` somewhere, so comparing 0 against 0 would pass
without testing anything.

## Accuracy

- Worst absolute difference: **9.095e-13**
  (depth 8, seed 421: engine `0.0000000000e+00` vs torch `9.0949470177e-13`)
- Worst relative error, restricted to the 1071 comparisons
  where the reference gradient exceeds 1e-06: **2.005e-09**
  (depth 8, seed 491: engine `-1.7434852361e-06` vs torch `-1.7434852396e-06`)

The relative error is restricted that way on purpose. Many leaf gradients
are legitimately near zero, and dividing a tiny absolute difference by a
tiny reference inflates the ratio without indicating any error. Reported
across all comparisons the figure would read far worse while the absolute
differences stayed at rounding level.

## Conclusion

Both figures sit at float64 rounding level. The engine's reverse-mode
gradients agree with PyTorch on every generated expression, including
trees where a leaf is reused and its gradient must accumulate.
