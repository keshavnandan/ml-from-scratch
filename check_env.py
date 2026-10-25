#!/usr/bin/env python3
"""
Write results/environment.md for ml-from-scratch.

Records the facts Phase 1 depends on: versions, whether MPS is usable,
whether float32 survives a CPU -> MPS -> CPU round trip unchanged, and
whether float64 works on MPS (it does not, which is why every gradient
check in Phase 1 runs on CPU).

usage:  python3 check_env.py [output_path]
        default output: results/environment.md
"""

import datetime
import platform
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    sys.exit("torch not installed — activate the venv first")


def mps_available() -> tuple[bool, str]:
    try:
        ok = torch.backends.mps.is_available()
        built = torch.backends.mps.is_built()
        return ok, f"is_available()={ok}, is_built()={built}"
    except Exception as exc:                       # noqa: BLE001
        return False, f"raised {type(exc).__name__}: {exc}"


def roundtrip_float32(n: int = 10_000) -> tuple[bool, str]:
    """float32 should survive CPU -> MPS -> CPU bit-identically."""
    try:
        torch.manual_seed(0)
        x = torch.randn(n, dtype=torch.float32)
        y = x.to("mps").to("cpu")
        same = torch.equal(x, y)
        worst = (x - y).abs().max().item()
        return same, f"bit-identical={same}, max |diff|={worst:g} over {n} elements"
    except Exception as exc:                       # noqa: BLE001
        return False, f"raised {type(exc).__name__}: {exc}"


def float64_on_mps() -> tuple[bool, str]:
    """Expected to FAIL. If it ever succeeds, the Phase 1 CPU rule can relax."""
    try:
        torch.ones(3, dtype=torch.float64, device="mps")
        return True, "float64 tensor allocated on MPS — unexpected, re-check the plan"
    except Exception as exc:                       # noqa: BLE001
        return False, f"refused as expected — {type(exc).__name__}: {exc}"


def determinism(n: int = 1000) -> tuple[bool, str]:
    """Same seed, same numbers, twice."""
    try:
        torch.manual_seed(1234)
        a = torch.randn(n)
        torch.manual_seed(1234)
        b = torch.randn(n)
        return torch.equal(a, b), f"identical={torch.equal(a, b)} over {n} samples"
    except Exception as exc:                       # noqa: BLE001
        return False, f"raised {type(exc).__name__}: {exc}"


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "results/environment.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    mps_ok, mps_detail = mps_available()
    rt_ok, rt_detail = roundtrip_float32()
    f64_ok, f64_detail = float64_on_mps()
    det_ok, det_detail = determinism()

    def mark(ok: bool, expected: bool = True) -> str:
        return "PASS" if ok == expected else "CHECK"

    rows = [
        ("MPS available", mark(mps_ok), mps_detail),
        ("float32 CPU->MPS->CPU round trip", mark(rt_ok), rt_detail),
        ("float64 on MPS refused", mark(f64_ok, expected=False), f64_detail),
        ("Seeded RNG reproducible", mark(det_ok), det_detail),
    ]

    lines = [
        "# Environment",
        "",
        f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} by `check_env.py`. "
        "Re-run after any PyTorch or OS upgrade.",
        "",
        "## Versions",
        "",
        f"- Python {sys.version.split()[0]}",
        f"- PyTorch {torch.__version__}",
        f"- Platform {platform.platform()}",
        f"- Machine {platform.machine()}",
        "",
        "## Checks",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    lines += [f"| {name} | **{res}** | {detail} |" for name, res, detail in rows]
    lines += [
        "",
        "## What this means for Phase 1",
        "",
        "- **Gradient checks run on CPU in float64.** MPS has no float64, so the "
        "1e-6 tolerance in exit test 1 is only meaningful on CPU.",
        "- **CI runs on Linux without MPS.** Keep device-specific code out of the "
        "test path, or skip it when MPS is unavailable.",
        "- **Seeding recipe:** call `torch.manual_seed(seed)` plus `random.seed(seed)` "
        "and `numpy.random.seed(seed)` at the top of every entry point. Record the "
        "seed in any results file you commit.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {out}")
    for name, res, detail in rows:
        print(f"  {res:<5} {name} — {detail}")


if __name__ == "__main__":
    main()
