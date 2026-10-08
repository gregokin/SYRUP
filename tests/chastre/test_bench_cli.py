"""CPU-only refusal tests of the Chastre timing entry (no case output, no CuPy, no solver needed). Nothing here was run by its
author (file-only tools); Codex records results."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "chastre"))


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0", "-1"])
def test_min_free_gpu_memory_must_be_finite_and_positive_before_any_work(value, tmp_path):
    import run_chastre_timing

    out = tmp_path / "out"
    with pytest.raises(SystemExit) as info:
        run_chastre_timing.main(["--case-dir", str(tmp_path / "no_such_case"), "--output-dir", str(out),
                                 f"--min-free-gpu-gib={value}"])
    assert info.value.code == 2
    assert not out.exists()  # refused before the case was opened or any output created
