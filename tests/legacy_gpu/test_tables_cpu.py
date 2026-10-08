"""CPU-only: the static walk tables and the GPU gather algorithm (a Python oracle) against the accepted CPU walk, bit for bit."""
from __future__ import annotations

import numpy as np
import pytest

from maple_syrup import legacy_native as N
from maple_syrup.legacy_native_cuda import (
    CudaLegacyError,
    build_walk_tables,
    estimate_bytes,
    validate_limits,
    validate_walk_tables,
)
from maple_syrup.legacy_native_numba import get_kernels

from .helpers import DX, GRAPHS, emulate, random_walk_inputs, tally

LIMITS = np.array([1, 1, 3, 3, 6, 6, 12], dtype=np.int64)  # short limits exercise the limit stop on these tiny networks


def follow(net, i):
    """Independent restatement of the original walk's credited cells for source `i` (-1 = ring), up to a generous cap."""
    out = []
    pos = int(net.walk_first[i])
    for _ in range(64):
        out.append(pos)
        if pos < 0:
            break
        nxt = int(net.walk_next[pos])
        if nxt in (N.STOP_TERMINAL, N.STOP_INACTIVE):
            break
        pos = nxt
    return out


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_tables_reproduce_every_source_path_and_are_ordered(name):
    net = N.native_network(GRAPHS[name]())
    big = np.array([64] * 7, dtype=np.int64)  # a cap larger than every path here: the table holds the full natural paths
    tb = build_walk_tables(net, big)
    target = np.full(tb.n_records, -2, dtype=np.int64)
    for c in range(net.active.size):
        for p in range(int(tb.tgt_ptr[c]), int(tb.tgt_ptr[c + 1])):
            target[tb.tgt_rec[p]] = c
    target[tb.ring_rec] = -1
    assert not np.any(target == -2)  # every record is exactly one cell or the ring
    for s, i in enumerate(tb.src_cells.tolist()):
        recs = target[tb.rec_off[s]:tb.rec_off[s + 1]]
        assert recs[0] == i  # the local credit comes first
        assert recs[1:].tolist() == follow(net, i)
        expected_kind = 1 if recs[-1] == -1 else (2 if net.terminal[recs[-1]] else (3 if net.inactive[recs[-1]] else 0))
        assert tb.end_kind[s] == expected_kind
    for c in range(net.active.size):  # ascending record ids per target = the CPU source order
        ids = tb.tgt_rec[int(tb.tgt_ptr[c]):int(tb.tgt_ptr[c + 1])]
        assert np.all(np.diff(ids) > 0)
    assert tb.tgt_ptr[-1] + tb.ring_rec.size == tb.n_records


@pytest.mark.parametrize("name", sorted(GRAPHS))
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_gpu_gather_algorithm_equals_the_cpu_walk_bitwise_with_equal_tallies(name, seed):
    net = N.native_network(GRAPHS[name]())
    rng = np.random.default_rng(seed)
    nc = 3
    det, par, law, regime = random_walk_inputs(net, rng, nc)
    tb = build_walk_tables(net, LIMITS)
    _, code, depos_gpu, ring_gpu = emulate(net, tb, det, par, law, regime, LIMITS, DX, 1.0)
    n = net.active.size
    depos, ring, inactive_dep, erased = np.zeros((n, nc)), np.zeros(nc), np.zeros(nc), np.zeros(nc)
    counts = np.zeros(N.N_COUNTS, dtype=np.int64)
    get_kernels(False).walk(net.source_order_index, det, par, law, regime, LIMITS, net.slope_zero, net.walk_first, net.walk_next,
                            net.aspect0, net.inactive, np.zeros(n, dtype=bool), False, DX, 1.0, depos, ring, inactive_dep, erased,
                            counts)
    act = net.active
    assert np.array_equal(depos_gpu[act], depos[act])  # the same additions in the same order: identical bits
    assert np.array_equal(ring_gpu, ring)
    np.testing.assert_allclose(depos_gpu[net.inactive].sum(axis=0), inactive_dep, rtol=1e-14, atol=0.0)
    tallied = tally(net, tb, code)
    assert np.array_equal(tallied[:N.C_ERASE_CELLS], counts[:N.C_ERASE_CELLS])  # walks, ring, terminal, inactive, limit, vge, aspect0, local, zero slope


def test_zero_slope_diffuse_forced_detachment_stays_mobile_and_other_no_law_cases_deposit_locally():
    net = N.native_network(GRAPHS["terminal_pit"]())
    tb = build_walk_tables(net, LIMITS)
    nc = 2
    n = net.active.size
    det = np.zeros((n, nc))
    det[2] = 0.5  # the pit: slope zero
    par = np.full((n, nc), 1.0)
    law = np.zeros((n, nc), dtype=bool)
    regime = np.full((n, nc), 2, dtype=np.int8)  # diffuse
    _, code, depos, _ = emulate(net, tb, det, par, law, regime, LIMITS, DX, 1.0)
    assert not depos.any() and np.all(code[2] == 0x40)  # no walk, no deposition: the detachment stays in the pool
    regime[:] = 5  # concentrated without excess stream power: deposit locally
    _, code, depos, _ = emulate(net, tb, det, par, law, regime, LIMITS, DX, 1.0)
    assert np.all(depos[2] == 0.5) and np.all(code[2] == 0x20)


def _replace(tb, **changes):
    import dataclasses

    return dataclasses.replace(tb, **changes)


def test_valid_tables_pass_and_every_malformed_table_is_refused_before_device_work():
    net = N.native_network(GRAPHS["converging"]())
    tb = build_walk_tables(net, LIMITS)
    validate_walk_tables(net, tb, LIMITS)  # the builder's own tables are valid
    assert not tb.rec_off.flags.writeable and not tb.tgt_rec.flags.writeable  # static by contract
    with pytest.raises(CudaLegacyError, match="WalkTables"):
        validate_walk_tables(net, object(), LIMITS)
    bad_cases = {
        "dtype": _replace(tb, rec_off=tb.rec_off.astype(np.int32)),
        "shape": _replace(tb, src_cells=tb.src_cells[:-1]),
        "non_monotone": _replace(tb, rec_off=np.concatenate(([0], np.full(tb.rec_off.size - 1, tb.n_records))).astype(np.int64)),
        "index_out_of_range": _replace(tb, tgt_rec=np.full_like(tb.tgt_rec, tb.n_records)),
        "bad_inverse": _replace(tb, tgt_rec=tb.tgt_rec[::-1].copy()),
        "bad_end_kind": _replace(tb, end_kind=np.zeros_like(tb.end_kind) + 3),
        "wrong_order": _replace(tb, src_cells=tb.src_cells[::-1].copy()),
        "bad_count": _replace(tb, n_records=tb.n_records + 1),
        "bool_count": _replace(tb, cap=True),
        "inactive_target": _replace(tb, inactive_targets=np.array([net.active.size], dtype=np.int32)),
    }
    for bad in bad_cases.values():
        with pytest.raises(CudaLegacyError):
            validate_walk_tables(net, bad, LIMITS)
    with pytest.raises(CudaLegacyError, match="frozen network"):
        validate_walk_tables(net, tb, np.array([2, 2, 2, 2, 2, 2, 2]))  # tables of other limits


@pytest.mark.parametrize("limits", [np.array([1, 2]), np.array([0] * 7), np.array([-1, 1, 1, 1, 1, 1, 1]),
                                    np.array([1.0] * 7), np.zeros((7, 1), dtype=np.int64), np.array([2**31] * 7)])
def test_invalid_limits_fail_before_any_work(limits):
    with pytest.raises(CudaLegacyError):
        validate_limits(limits)
    net = N.native_network(GRAPHS["converging"]())
    with pytest.raises(CudaLegacyError):
        build_walk_tables(net, limits)


def test_budgets_fail_before_allocation_and_unsupported_modes_are_rejected_before_any_mutation():
    net = N.native_network(GRAPHS["converging"]())
    with pytest.raises(CudaLegacyError, match="exceed the budget"):
        build_walk_tables(net, LIMITS, record_budget=1)
    from maple_syrup.legacy_native_cuda import CudaLegacyContext

    for kwargs in ({"source_order": "legacy"}, {"erase_on": True}):
        with pytest.raises(CudaLegacyError, match="index"):
            CudaLegacyContext(net, None, None, limits=LIMITS, dt=1.0, n_steps=1, **kwargs)
    tb = build_walk_tables(net, LIMITS)

    class Physics:
        shape = tuple(net.shape)

    est = estimate_bytes(net, Physics(), tb.n_records, 100, 6, tb.nbytes())
    assert est["total"] == sum(est[k] for k in ("static", "dynamic", "walk_values", "walk_codes", "partials", "ledger"))
    assert est["walk_values"] == tb.n_records * 6 * 8
