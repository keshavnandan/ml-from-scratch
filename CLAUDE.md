# ml-from-scratch

Companion repo to a 12-month AI research transition plan.
Full plan: ROADMAP.local.md (gitignored, read it at session start).

## Current position
Phase 1, Weeks 1–4 — micrograd. Gate: engine rebuilt from blank,
gradient check vs torch to 1e-6, MLP trains on toy data.
Next: Karpathy L2 (makemore).

## Conventions
- Python 3.12, pytest, ruff
- M4 Max / MPS only, no CUDA — never suggest CUDA-dependent code
- No autograd libraries inside micrograd/; torch only in tests, as reference