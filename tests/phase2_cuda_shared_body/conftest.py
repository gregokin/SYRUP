"""S2-A4.1 shared CUDA body checks (author-written, CPU only, NO GPU/CuPy/NVRTC: nothing here is a hardware claim). The reference is the pre-A4.1 common CUDA module, kept as task evidence
(`s2_a4_1_pre_sources/subsurface_cuda.py`, byte-identical to the frozen A3 file) and loaded as a separate module."""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

TASK = Path(__file__).resolve().parents[3]
OLD_PATH = TASK / "s2_a4_1_pre_sources" / "subsurface_cuda.py"
NEW_PATH = Path(__file__).resolve().parents[2] / "src" / "maple_syrup" / "subsurface_cuda.py"

STUB = r"""
#include <cmath>
#include <cstdint>
#define __device__
#define __global__
#define __forceinline__ inline
#define __restrict__
struct dim3_ { unsigned x; };
static dim3_ threadIdx, blockIdx, blockDim;
using std::isfinite;
"""


@pytest.fixture(scope="session")
def old():
    if not OLD_PATH.is_file():
        pytest.skip("the pre-A4.1 source copy is task evidence and is not present in this tree")
    spec = importlib.util.spec_from_file_location("old_subsurface_cuda_pre_a41", OLD_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def new():
    from maple_syrup import subsurface_cuda

    return subsurface_cuda


def have_gxx() -> bool:
    return shutil.which("g++") is not None


def gxx(args, src: str, suffix: str = ".cpp"):
    d = Path(tempfile.mkdtemp())
    f = d / ("k" + suffix)
    f.write_text(src)
    return d, f


def preprocess(source: str) -> str:
    """The macro-expanded text (comments dropped) as `g++ -E -P` sees it."""
    d, f = gxx(None, source)
    out = subprocess.run(["g++", "-E", "-P", "-x", "c++", str(f)], check=True, capture_output=True, text=True).stdout
    return out
