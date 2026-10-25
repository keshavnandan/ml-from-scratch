# Environment

Generated 2026-10-25 17:54 by `check_env.py`. Re-run after any PyTorch or OS upgrade.

## Versions

- Python 3.12.13
- PyTorch 2.14.0
- Platform macOS-26.6.2-arm64-arm-64bit
- Machine arm64

## Checks

| Check | Result | Detail |
|---|---|---|
| MPS available | **PASS** | is_available()=True, is_built()=True |
| float32 CPU->MPS->CPU round trip | **PASS** | bit-identical=True, max |diff|=0 over 10000 elements |
| float64 on MPS refused | **PASS** | refused as expected — TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead. |
| Seeded RNG reproducible | **PASS** | identical=True over 1000 samples |

## What this means for Phase 1

- **Gradient checks run on CPU in float64.** MPS has no float64, so the 1e-6 tolerance in exit test 1 is only meaningful on CPU.
- **CI runs on Linux without MPS.** Keep device-specific code out of the test path, or skip it when MPS is unavailable.
- **Seeding recipe:** call `torch.manual_seed(seed)` plus `random.seed(seed)` and `numpy.random.seed(seed)` at the top of every entry point. Record the seed in any results file you commit.
