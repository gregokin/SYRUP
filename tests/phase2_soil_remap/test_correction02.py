"""Correction02: the CUDA device algorithm memory (outputs + 10-word diagnostic + block partials, explicit static shared memory, no library reduction) and the raw-integer packet.
Everything here runs on the CPU: a FAKE `cupy` records every allocation and refuses any other call, and fake kernels stand in for the two launches. Author-written; Root runs the real device tests."""
from __future__ import annotations

import numpy as np
import pytest

from maple_syrup import soil_water_remap as sr
from maple_syrup import soil_water_remap_cuda as rc


class FakeArray(np.ndarray):
    def get(self):
        return np.array(self)


class FakeCupy:
    """Only `zeros`, `float64` and `uint64` exist: any library reduction/stack/where/argmax/sum call raises AttributeError, and every allocation is recorded."""

    float64, uint64 = np.float64, np.uint64

    def __init__(self):
        self.allocations: list[tuple[tuple, str, int]] = []

    def zeros(self, shape, dtype):
        a = np.zeros(shape, dtype=dtype).view(FakeArray)
        self.allocations.append((tuple(np.atleast_1d(shape)), np.dtype(dtype).name, a.nbytes))
        return a


@pytest.fixture
def fake(monkeypatch):
    launches = []
    state = {"words": None}

    def column_kernel(grid, block, args):
        launches.append(("column", grid, block, args[11].shape))

    def finalize_kernel(grid, block, args):
        launches.append(("finalize", grid, block, args[0].shape))
        if state["words"] is not None:
            args[2][:] = np.asarray(state["words"], dtype=np.uint64)

    monkeypatch.setattr(rc, "_load_kernel", lambda cp, dev: (column_kernel, finalize_kernel))
    monkeypatch.setattr(rc, "current_device_id", lambda cp: 0)
    return FakeCupy(), launches, state


def host_inputs(L, C):
    return (np.zeros((L + 1, C)), np.zeros((L + 1, C)), np.zeros((L, C)), np.zeros(C), np.zeros(C))


@pytest.mark.parametrize("L,C", [(1, 1), (3, 127), (3, 128), (3, 129), (5, 1000), (2, 100_001)])
def test_run_allocates_exactly_the_modelled_device_arrays(fake, L, C):
    cp, launches, _ = fake
    rc.run(cp, *host_inputs(L, C), 1e-15, 1e-15)
    est = sr.estimate_remap_memory(L, C, backend="cuda")
    allocated = sum(n for _, _, n in cp.allocations)
    assert allocated == est.device_required_bytes - est.reserve_bytes == est.output_bytes + sr.DEVICE_DIAG_BYTES + 5 * 8 * sr.cuda_block_count(C)
    names = sorted((shape, dt) for shape, dt, _ in cp.allocations)
    nb = sr.cuda_block_count(C)
    assert ((5, nb), "float64") in [(s_, d) for s_, d, _ in cp.allocations] and ((sr.DIAG_WORDS,), "uint64") in [(s_, d) for s_, d, _ in cp.allocations]
    assert [x[0] for x in launches] == ["column", "finalize"]  # exactly two launches
    assert launches[0][1] == (nb,) and launches[0][2] == (sr.CUDA_BLOCK,) and launches[1][1] == (1,) and launches[1][2] == (sr.CUDA_FINAL_BLOCK,)


def test_run_uses_no_library_reduction_or_helper_array(fake):
    cp, _, _ = fake
    rc.run(cp, *host_inputs(2, 300), 1e-15, 1e-15)  # a FakeCupy attribute outside zeros/float64/uint64 would have raised AttributeError
    for name in ("sum", "stack", "argmax", "where", "abs", "bitwise_or", "asarray", "max"):
        assert not hasattr(cp, name)


def test_the_packet_is_raw_integers_not_floats(fake):
    cp, _, state = fake
    first_col = 2**36 + 12345  # far beyond float32 and still exact; the gate keeps columns below MAX_CUDA_COLUMNS
    flags = sr.FLAG_WATER | sr.FLAG_THETA
    bits = lambda x: int(np.array([x], dtype=np.float64).view(np.uint64)[0])  # noqa: E731
    state["words"] = [flags | sr.FLAG_OUTPUT, 7, (first_col << 8) | flags, bits(0.25), bits(0.125), bits(1.5), bits(2.5), bits(3.5), bits(4.5), bits(5.5)]
    out = rc.run(cp, *host_inputs(2, 10), 1e-15, 1e-15)
    info, sums = out[5], out[6]
    assert info == (flags | sr.FLAG_OUTPUT, 7, first_col, flags)  # exact integers
    assert sums == {"old": 1.5, "new": 2.5, "added": 3.5, "removed": 4.5, "scale": 5.5, "max_abs_residual": 0.25, "max_ratio": 0.125}


def test_no_flagged_columns_gives_the_empty_info(fake):
    cp, _, state = fake
    state["words"] = [0, 0, 0xFFFFFFFFFFFFFFFF] + [0] * 7
    assert rc.run(cp, *host_inputs(1, 5), 1e-15, 1e-15)[5] == (0, 0, None, 0)


def test_empty_selection_allocates_no_partials_and_launches_nothing(fake):
    cp, launches, _ = fake
    out = rc.run(cp, *host_inputs(2, 0), 1e-15, 1e-15)
    assert launches == [] and out[5] == (0, 0, None, 0)
    assert all(shape != (5, 0) for shape, _, _ in cp.allocations)


@pytest.mark.parametrize("C", [0, 1, 128, 129, 10**6])
def test_estimate_terms_for_blocks_partials_and_shared_memory(C):
    e = sr.estimate_remap_memory(3, C, backend="cuda", reserve_bytes=5, host_reserve_bytes=9)
    nb = sr.cuda_block_count(C)
    assert nb == -(-C // sr.CUDA_BLOCK) and e.terms["cuda_blocks"] == nb
    assert e.terms["cuda_block_partials_5xBlocks"] == 5 * 8 * nb
    assert e.terms["cuda_static_shared_memory_bytes_per_block"] == {"column_kernel": 5 * 8 * sr.CUDA_BLOCK, "finalize_kernel": 5 * 8 * sr.CUDA_FINAL_BLOCK}
    assert e.device_required_bytes == 8 * 3 * C + 32 * C + sr.DEVICE_DIAG_BYTES + 5 * 8 * nb + 5
    assert e.host_required_bytes == sr.PACKET_BYTES + sr.HOST_OBJECT_ALLOWANCE_BYTES + 9 and sr.PACKET_BYTES == 80
    assert e.terms["stats_block_5xC"] == 0 and e.terms["flags"] == 0  # no per-column stats/flag array on the device


def test_block_sizes_are_powers_of_two_for_the_reduction_trees():
    for n in (sr.CUDA_BLOCK, sr.CUDA_FINAL_BLOCK):
        assert n & (n - 1) == 0
    assert sr.CUDA_REDUCED_ROWS == 5


def test_explicit_column_range_gate_before_any_allocation():
    ok = sr.MAX_CUDA_COLUMNS
    sr.estimate_remap_memory(1, ok, backend="cuda")  # metadata only
    with pytest.raises(sr.SoilRemapError):
        sr.estimate_remap_memory(1, ok + 1, backend="cuda")
    assert (ok << 8) < 2**64 and ok == (2**31 - 1) * sr.CUDA_BLOCK
    sr.estimate_remap_memory(1, ok + 1, backend="numpy")  # the gate is a CUDA contract only


def test_cuda_source_text_has_no_cub_or_library_reduction_markers():
    src = rc._source()
    assert "__shared__ double sh[NROWS][BLOCK]" in src and "__shared__ double sh[NROWS][FINAL_BLOCK]" in src
    for banned in ("cub::", "thrust::", "cooperative_groups"):
        assert banned not in src
