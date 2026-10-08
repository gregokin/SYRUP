"""Thin entry for the matched CPU/GPU, bisection/Newton storm comparison of task gpu_newton.

    python benchmarks/gpu_newton/compare_cases.py --case rfid  --case-dir outputs/rfid/case  --output-dir <NEW> \\
        [--cuda-mode auto|fused|split] [--rounds 3] [--order balanced] [--allow-maple-source-change]
    python benchmarks/gpu_newton/compare_cases.py --case plot1 --case-dir outputs/plot1       --output-dir <NEW> ...

It runs `benchmarks/newton_cpu/compare_cases.py` (the single harness: same verified cases, RunGuard, first call / complete
warm-up / balanced rounds, capture of final fields and the full outlet hydrograph outside every timer) with the default
contender list `bisection_numba,newton_numba,bisection_cuda,newton_cuda`. Any argument, including `--contenders`, overrides
the default. Select the device with CUDA_VISIBLE_DEVICES before starting. No physics, timing or guard code lives here.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "newton_cpu"))

import compare_cases as harness

DEFAULT_CONTENDERS = "bisection_numba,newton_numba,bisection_cuda,newton_cuda"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not any(a == "--contenders" or a.startswith("--contenders=") for a in argv):
        argv += ["--contenders", DEFAULT_CONTENDERS]
    return harness.main(argv)


if __name__ == "__main__":
    sys.exit(main())
