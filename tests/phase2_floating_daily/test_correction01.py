"""G3 S1/S2 correction01: the five independently reproduced gaps, as regressions (CPU/host; NumPy and Numba where relevant). Real owners/beds/lineages; labelled fault injection only where stated."""
from __future__ import annotations

import tracemalloc

import builders as b
import numpy as np
import pytest
import texture_fixtures as fx
from dbuild import AVAIL, PART, RUN, close, day, full_day, lineage, make, start

from maple_syrup import floating_daily as fd
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import native_floating_soil as nfs
from maple_syrup import soil_evaporation as se

BACKENDS = ["numpy"] + (["numba"] if __import__("importlib").util.find_spec("numba") is not None else [])


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param


# ---- 1. the final close audits the CURRENT native water against the window/lifetime ledger ------------------------------------------------------------
def test_close_refuses_water_changed_outside_the_accepted_transactions(backend):
    st, *_ = lineage(backend=backend, saturation=0.5)
    inter = day(st).state
    w = inter.water.water_m
    before = float(np.asarray(w).sum())
    saved = np.array(w, copy=True)
    w.flags.writeable = True
    w[0, 1] += 1e-6  # FAULT INJECTION (labelled): a finite, under-saturation change to the genuine interevent water outside every transaction
    try:
        with pytest.raises(fd.DailyRefusal, match="final close"):
            close(inter)
        assert inter.role == "interevent" and inter.cursor.ordinal == 0  # no closed boundary was minted
    finally:
        w[...] = saved
    assert float(np.asarray(w).sum()) == before
    assert close(inter).state.cursor.ordinal == 1  # the untouched genuine interevent closes


def test_close_audit_covers_lifetime_rain_withdrawal_and_clock(backend):
    st, *_ = lineage(backend=backend, saturation=0.6)
    for d in range(4):
        r = day(st, p=1.0 if d % 2 else 0.0, i=2.0, t=5.0)
        assert r.outcome == "interevent"
        inter = r.state
        if d == 2:
            av = np.asarray(fsl.floating_storage_m3)  # noqa: F841 - keep the import used
            req = np.zeros((st.ctx.n_layers, st.ctx.n_cells))
            req[0] = 1e-5
            inter = fd.request_floating_withdrawal(inter, expected_parent=fd.expect_parent(inter), request_m=req, expected_sequence=0, available_bytes_fn=AVAIL).state
        st = close(inter).state
    led = st.ledger
    assert led.days_applied == 4 and led.transpiration_m3 > 0.0 and led.rain_m3 > 0.0 and led.bound_m3 > 0.0 and led.expected_t_s == 4 * 86400.0
    # the step budget's authoritative rain is the ledger rain: depth * area * active cells, no extra tolerance
    assert led.rain_m3 == pytest.approx(2 * 1.0 / 1000.0 * st.ctx.area * int(st.ctx.active.sum()), rel=1e-12)
    # a mutated clock (outside the transactions) is refused at the next close
    inter = day(st).state
    object.__setattr__(inter.water, "t_s", inter.water.t_s + 1.0)
    with pytest.raises(fd.DailyRefusal, match="clock"):
        close(inter)


def test_off_and_not_due_output_add_no_extra_audit_work_relative_to_the_actual_close(backend, tmp_path, monkeypatch):
    from maple_syrup import floating_daily_output as fo
    from maple_syrup import soil_moisture_output as g0

    calls = []
    real = fsl.floating_storage_m3
    monkeypatch.setattr(fsl, "floating_storage_m3", lambda *a, **k: calls.append(1) or real(*a, **k))
    st, *_ = lineage(backend=backend)
    calls.clear()
    close(day(st).state)
    base = len(calls)
    assert base == 1  # the audited close performs ONE native storage query
    lim = g0.MoistureOutputLimits(10, 10**7, 8192, 4096, 10**7, 0, 0)
    (tmp_path / "o").mkdir()
    out = fo.FloatingMoistureOutput.create(tmp_path / "o", st, expected_parent=fd.expect_parent(st), every_n_days=3, limits=lim, host_available_bytes_fn=AVAIL, disk_available_bytes_fn=AVAIL)
    calls.clear()
    close(day(st).state, output=out, host_available_bytes_fn=AVAIL, disk_available_bytes_fn=AVAIL)  # not due
    assert len(calls) == base  # no extra storage/array work from the output on a not-due day


# ---- 2. the constructor is admitted as ONE composed, dimension-aware plan before any allocation ---------------------------------------------------------
@pytest.mark.parametrize("layers", [(0.16,), (0.01, 0.05, 0.10), tuple([0.16 / 65] * 65)])
def test_composed_construction_admission_exact_need_minus_one_and_zero(backend, layers):
    owner, bed = make(backend=backend, layers=layers)
    need = fd._construction_need(owner, bed, nfs.DEFAULT_BATCH_CELLS, 0)
    L, N = owner.column.n_layers, owner.column.n_cells
    assert need >= (L * N + 3 * N + L + 1) * 8 + N  # includes every retained array
    reads = []
    for avail in (need - 1, 0):
        with pytest.raises(fd.DailyRefusal) as ex:
            fd.start_floating_daily_lineage(owner, bed, partition=PART, run_digest=RUN, available_bytes_fn=lambda a=avail: reads.append(1) or a)
        assert ex.value.code == "memory" and owner.is_current
    st = fd.start_floating_daily_lineage(owner, bed, partition=PART, run_digest=RUN, available_bytes_fn=lambda: need)
    assert st.role == "boundary" and not owner.is_current


def test_real_l65_n4096_owner_is_admitted_for_all_retained_arrays_and_the_peak_stays_within_the_plan(backend):
    layers = tuple([0.16 / 65] * 65)
    bed = b.bed_from(fx.stratified_dense(shallow_cells=(), ny=64, nx=64))
    owner, _ = make(backend=backend, layers=layers, bed=bed)
    L, N = owner.column.n_layers, owner.column.n_cells
    assert (L, N) == (65, 4096)
    need = fd._construction_need(owner, bed, nfs.DEFAULT_BATCH_CELLS, 0)
    assert need > 557408 and need > 8 * L * N  # the reproduced reader (557 kB) could not hold the (L, N) zero block: now refused
    with pytest.raises(fd.DailyRefusal):
        fd.start_floating_daily_lineage(owner, bed, partition=PART, run_digest=RUN, available_bytes_fn=lambda: 557408)
    assert owner.is_current
    tracemalloc.start()
    tracemalloc.reset_peak()
    t0 = tracemalloc.get_traced_memory()[0]
    st = fd.start_floating_daily_lineage(owner, bed, partition=PART, run_digest=RUN, available_bytes_fn=lambda: need)
    peak = tracemalloc.get_traced_memory()[1] - t0
    tracemalloc.stop()
    assert peak <= need, (peak, need)
    assert peak >= 8 * L * N  # non-vacuous: the zero block alone
    assert not owner.is_current and st.ctx.n_cells == 4096


def test_cold_source_hash_is_in_the_plan_and_admitted_before_the_hash(backend, monkeypatch):
    from maple_syrup import pond_checkpoint as pc
    from maple_syrup import provenance
    from maple_syrup import sediment_bed as sb

    owner, bed = make(backend=backend)
    base = fd._construction_need(owner, bed, nfs.DEFAULT_BATCH_CELLS, 0) - fd._cold_source_bytes()
    monkeypatch.setattr(pc, "_SOURCE_CACHE", {})
    monkeypatch.setattr(sb, "_MAPLE_DIGEST_CACHE", {})
    cold = int(provenance._READ_CHUNK_BYTES) + fd.COLD_METADATA_ALLOWANCE_BYTES
    need = fd._construction_need(owner, bed, nfs.DEFAULT_BATCH_CELLS, 0)
    assert need == base + cold
    with pytest.raises(fd.DailyRefusal):
        fd.start_floating_daily_lineage(owner, bed, partition=PART, run_digest=RUN, available_bytes_fn=lambda: need - 1)
    assert "pair" not in pc._SOURCE_CACHE and "digest" not in sb._MAPLE_DIGEST_CACHE and owner.is_current  # refused BEFORE any source hash
    tracemalloc.start()
    tracemalloc.reset_peak()
    t0 = tracemalloc.get_traced_memory()[0]
    fd.start_floating_daily_lineage(owner, bed, partition=PART, run_digest=RUN, available_bytes_fn=lambda: need)
    peak = tracemalloc.get_traced_memory()[1] - t0
    tracemalloc.stop()
    assert "pair" in pc._SOURCE_CACHE and peak <= need and peak > int(provenance._READ_CHUNK_BYTES)


# ---- 3. the sparse helper is admitted ---------------------------------------------------------------------------------------------------------------
def test_sparse_aggregation_is_admitted_before_any_cast_validation_or_dense_allocation(monkeypatch):
    L, N = 65, 4096
    z = np.zeros(0, dtype=np.int64)
    need = 8 * L * N + fd.CONSTRUCTION_ALLOWANCE_BYTES
    with pytest.raises(fd.DailyRefusal) as ex:
        fd.aggregate_sparse_demand(L, N, z, z, np.zeros(0), max_links=1, available_bytes_fn=lambda: need - 1)  # ZERO links still needs the dense aggregate: refused
    assert ex.value.code == "memory"
    reads = []
    for bad in ([1], np.zeros(2), np.zeros((2, 2), dtype=np.int64)):  # metadata is refused BEFORE the reader is consulted
        with pytest.raises(fd.DailyRefusal):
            fd.aggregate_sparse_demand(L, N, bad, z, np.zeros(0), max_links=1, available_bytes_fn=lambda: reads.append(1) or need)
    assert reads == []
    built = []
    real = np.zeros
    empty = np.zeros(0)  # created BEFORE the allocation spy is installed
    monkeypatch.setattr(fd.np, "zeros", lambda *a, **k: built.append(a) or real(*a, **k))
    with pytest.raises(fd.DailyRefusal):
        fd.aggregate_sparse_demand(L, N, z, z, empty, max_links=1, available_bytes_fn=lambda: 0)
    assert built == []  # nothing dense was allocated on a refusal
    monkeypatch.undo()
    out = fd.aggregate_sparse_demand(L, N, z, z, np.zeros(0), max_links=1, available_bytes_fn=lambda: need)
    assert out.shape == (L, N) and not out.any()
    for bad in (-1, True, 1.5):
        with pytest.raises(fd.DailyRefusal):
            fd.aggregate_sparse_demand(L, N, z, z, np.zeros(0), max_links=1, available_bytes_fn=lambda: need, reserve_bytes=bad)
    k = 1000
    cells, layers = np.arange(k) % N, np.arange(k) % L
    amounts = np.ones(k)
    need_k = 8 * L * N + k * 26 + fd.CONSTRUCTION_ALLOWANCE_BYTES
    tracemalloc.start()
    tracemalloc.reset_peak()
    t0 = tracemalloc.get_traced_memory()[0]
    out = fd.aggregate_sparse_demand(L, N, cells, layers, amounts, max_links=k, available_bytes_fn=lambda: need_k)
    peak = tracemalloc.get_traced_memory()[1] - t0
    tracemalloc.stop()
    assert out.sum() == k and peak <= need_k


# ---- 4/5. the model-source pair and the COMPLETE physics configuration are bound separately -----------------------------------------------------
def test_source_pair_is_sealed_in_the_context_and_physics_config_is_a_separate_hash(backend):
    from maple_syrup.pond_checkpoint import pond_checkpoint_source_pair

    st, *_ = lineage(backend=backend)
    pair = dict(st.ctx.source_pair)
    assert pair == pond_checkpoint_source_pair()  # the ACTUAL imported source pair, cold once, sealed (read-only mapping)
    with pytest.raises(TypeError):
        st.ctx.source_pair["maple_syrup_tree_sha256"] = "0" * 64
    assert st.ctx.physics_config_sha256 not in pair.values() and len(st.ctx.physics_config_sha256) == 64
    ident = st.frame_identity()
    assert ident["source_sha256"] != st.ctx.physics_config_sha256


def test_changed_evaporation_partition_changes_the_physics_config_and_the_frame_source(backend, tmp_path):
    from maple_syrup import floating_daily_output as fo
    from maple_syrup import soil_moisture_output as g0

    owner_a, bed = make(backend=backend)
    owner_b, bed_b = make(backend=backend)
    a = start(owner_a, bed)
    b_ = fd.start_floating_daily_lineage(owner_b, bed_b, partition=se.EvaporationPartition(1.25, 0.5, 1.0, 0.125, 0.5), run_digest=RUN, available_bytes_fn=AVAIL)  # soil_factor .75 -> .5
    assert a.ctx.config_sha256 == b_.ctx.config_sha256  # the native input config is the SAME ...
    assert a.ctx.physics_config_sha256 != b_.ctx.physics_config_sha256  # ... the complete daily physics config is not
    assert a.frame_identity()["source_sha256"] != b_.frame_identity()["source_sha256"] and dict(a.ctx.source_pair) == dict(b_.ctx.source_pair)
    lim = g0.MoistureOutputLimits(10, 10**7, 8192, 4096, 10**7, 0, 0)
    (tmp_path / "o").mkdir()
    out = fo.FloatingMoistureOutput.create(tmp_path / "o", a, expected_parent=fd.expect_parent(a), every_n_days=1, limits=lim, host_available_bytes_fn=AVAIL, disk_available_bytes_fn=AVAIL)
    with pytest.raises(g0.MoistureOutputError, match="binding"):
        fo.FloatingMoistureOutput.resume(tmp_path / "o", b_, expected_parent=fd.expect_parent(b_), limits=lim, host_available_bytes_fn=AVAIL)
    inter = day(b_).state
    closed = fd.close_floating_day(inter, expected_parent=fd.expect_parent(inter)).state
    with pytest.raises(g0.MoistureOutputError, match="run/source/config"):
        out.notify_day_closed(closed, expected_parent=fd.expect_parent(closed))
    # and the physics really differs (the independent evidence behind the binding): next-day evaporation
    ra, rb = full_day(a), full_day(b_)
    assert ra.ledger.evaporation_m3 != rb.ledger.evaporation_m3
    assert out.rec["physics_config_sha256"] == a.ctx.physics_config_sha256 and out.rec["code_source_pair"] == dict(a.ctx.source_pair)


def test_native_config_and_context_references_are_held_water_free(backend):
    st, *_ = lineage(backend=backend)
    c = st.ctx
    assert c.config.sha256 == c.config_sha256 and c.matrix_op.column is c.column and c.matrix_op.sha256  # held template references, not just hashes
    assert not hasattr(c, "water") and st.water is not None
