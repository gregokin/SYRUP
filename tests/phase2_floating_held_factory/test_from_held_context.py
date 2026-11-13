# ruff: noqa: RUF059
"""H1: `FloatingPondVertical.from_held_context` from a genuine H0 `HeldContext` -- no NativeOwner, no initial W/J/I, no bed read. Original case -> real wet/splash Pond -> R2a /5-/6 file + H0 export -> DROP the original -> context
factory vertical + a new PondEvent (static case/topology/forcing only) -> file load (prepared resume, provisional scope, full validation, adoption LAST) -> bit-exact continuation. Dense/published Windowed, NumPy/Numba."""
from __future__ import annotations

import fpond_builders as fb
import numpy as np
import pytest
from case import AVAIL, IDENT, build, run, snapshot

from maple_syrup import floating_checkpoint as fc
from maple_syrup import floating_held_context as fh
from maple_syrup import floating_pond as fp
from maple_syrup import native_floating_soil as nfs
from maple_syrup.pond_event import PondEvent
from maple_syrup.rainfall import constant_rainfall
from maple_syrup.sediment_physics import plot1_sediment_parameters
from maple_syrup.splash import splash_parameters

REFUSE = (fc.PondCheckpointError, fp.FloatingPondError, fh.HeldContextError)


@pytest.fixture(params=["numpy", "numba"])
def backend(request):
    if request.param == "numba":
        pytest.importorskip("numba")
    return request.param


def spec_of(vert):
    n = vert.n_cells
    return fh.PreferentialSpec(np.full(n, 2.0e-3), 0.5, 1e-5, np.array([0.0, 1.0, 1.0]), (), None)


def context_of(vert):
    rec = fh.export_held_context(vert, spec_of(vert), available_bytes_fn=AVAIL)
    meta, arrays = rec.meta(), {k: np.array(v, copy=True) for k, v in rec.arrays().items()}
    parsed = fh.record_from_parts(meta, arrays, available_bytes_fn=AVAIL)
    t = vert.held_template()
    exp = fh.ExpectedHeldIdentities(vert.config.sha256, t.column.sha256, t.matrix_op.sha256, None if t.preferential is None else t.preferential.sha256)
    return fh.reconstruct_held_context(parsed, exp, available_bytes_fn=AVAIL)


def held_event(tmp, ctx, backend, windowed, monkeypatch):
    """A new PondEvent with ONLY static case pieces; the soil comes from the context. The original-owner factory is disabled on this path."""
    monkeypatch.setattr(nfs, "initialize_native_floating_soil", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no native owner on the restore path")))
    wh = fb._windowed_helpers()
    z = fb.ph.plane(base=0.12, drop=0.004)
    dense, c = wh.dense_deep_bed(z, depth=0.004)
    topo = fb.ph.make_topology(dense, fb.ph.east_outlets(*z.shape), drop=0.02)
    bed = dense
    if windowed:
        bed, c = wh.windowed_counterpart(dense, c, tmp / "store")
    vert = fp.FloatingPondVertical.from_held_context(ctx, available_bytes_fn=AVAIL)
    be = type("B", (), {"hydraulics": backend, "xp": np})()
    ctl = wh.control_for(be, windowed=windowed, max_dt_s=0.2)
    ev = PondEvent(c, None, plot1_sediment_parameters(xp=np), constant_rainfall(0.0, 1000.0, 300.0), topo, control=ctl, splash=splash_parameters(np.linspace(0.4, 3.0, fb.NC), xp=np), floating_vertical=vert)
    return ev, vert, bed


@pytest.mark.parametrize("windowed", [False, True])
def test_restore_into_a_held_context_vertical_continues_bit_exactly(tmp_path, backend, windowed, monkeypatch):
    (tmp_path / "a").mkdir()
    ev, st, vert = build(tmp_path / "a", backend=backend, windowed=windowed)
    st = run(ev, st, 8)
    ck = tmp_path / "ck"
    fc.save_floating_pond_checkpoint(ck, st, ev, IDENT, available_bytes_fn=AVAIL)
    ref = run(ev, st, 6)
    ctx = context_of(vert)
    (tmp_path / "b").mkdir()
    ev2, v2, bed2 = held_event(tmp_path / "b", ctx, backend, windowed, monkeypatch)
    assert v2._owner is None and v2._initial is None and v2.identity == vert.identity
    for call in (lambda: v2.initial_soil_water(), lambda: v2.check_initial(None, None, 0.0), lambda: ev2.initial_state(bed2, np.zeros(ev2.shape))):
        with pytest.raises(fp.FloatingPondError, match="no initial payload"):
            call()
    t0, t1 = v2.held_template(), vert.held_template()
    assert t0.column is ctx.column and t0.column.sha256 == t1.column.sha256 and t0.source_witness == t1.source_witness and t0.generation == t1.generation and np.array_equal(t0.depth0_m, t1.depth0_m)
    got = fc.load_floating_pond_checkpoint(ck, ev2, IDENT, available_bytes_fn=AVAIL)
    a, b = snapshot(ev2, got, v2), snapshot(ev, st, vert)
    assert set(a) == set(b) and all(a[k].tobytes() == b[k].tobytes() for k in a)
    cont = run(ev2, got, 6)
    a, b = snapshot(ev2, cont, v2), snapshot(ev, ref, vert)
    assert set(a) == set(b) and all(a[k].tobytes() == b[k].tobytes() for k in a)


def test_wrong_context_refuses_and_late_failure_leaves_the_target_retryable(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    ev, st, vert = build(tmp_path / "a")
    st = run(ev, st, 6)
    ck = tmp_path / "ck"
    fc.save_floating_pond_checkpoint(ck, st, ev, IDENT, available_bytes_fn=AVAIL)
    # a genuine but DIFFERENT context (other config: a different matrix conductivity) belongs to another vertical identity
    other, _o, _b = fb.make(surface=fb.ph.plane(base=0.12, drop=0.004), bottom="closed", matrix_conductivity_m_s=3e-6)[0:6:1][4], None, None

    wrong = fh.reconstruct_held_context(fh.record_from_parts(*(lambda r: (r.meta(), {k: np.array(v, copy=True) for k, v in r.arrays().items()}))(fh.export_held_context(other, None, available_bytes_fn=AVAIL)), available_bytes_fn=AVAIL),  # noqa: PLC3002
                                        fh.ExpectedHeldIdentities(other.config.sha256, other.held_template().column.sha256, other.held_template().matrix_op.sha256, None), available_bytes_fn=AVAIL)
    (tmp_path / "w").mkdir()
    evw, vw, _bw = held_event(tmp_path / "w", wrong, "numpy", False, monkeypatch)
    reg0 = len(vw._registry)
    with pytest.raises(REFUSE):
        fc.load_floating_pond_checkpoint(ck, evw, IDENT, available_bytes_fn=AVAIL)
    assert len(vw._registry) == reg0 and vw._provisional is None
    # the right context: a failed (tampered-seal-free) first attempt with a zero reader leaves the target retryable
    (tmp_path / "b").mkdir()
    ev2, v2, _b2 = held_event(tmp_path / "b", context_of(vert), "numpy", False, monkeypatch)
    with pytest.raises(Exception, match="estimated"):
        fc.load_floating_pond_checkpoint(ck, ev2, IDENT, available_bytes_fn=lambda: 0)
    assert len(v2._registry) == 0 and v2._provisional is None
    assert fc.load_floating_pond_checkpoint(ck, ev2, IDENT, available_bytes_fn=AVAIL) is not None


def test_factory_refusals_and_admission_order(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    ev, st, vert = build(tmp_path / "a")
    ctx = context_of(vert)
    for bad in (object(), vert.held_template(), type("Fake", (), {})()):
        with pytest.raises(fh.HeldContextError):
            fp.FloatingPondVertical.from_held_context(bad, available_bytes_fn=AVAIL)
    calls = []
    with pytest.raises(nfs.NativeFloatingMemoryError):
        fp.FloatingPondVertical.from_held_context(ctx, available_bytes_fn=lambda: (calls.append(1), 0)[1])
    assert calls == [1]
    with pytest.raises(Exception):  # noqa: B017 - strict non-bool reserve
        fp.FloatingPondVertical.from_held_context(ctx, available_bytes_fn=AVAIL, reserve_bytes=True)
    v = fp.FloatingPondVertical.from_held_context(ctx, available_bytes_fn=AVAIL)
    need = v.construction_bytes
    assert fp.FloatingPondVertical.from_held_context(ctx, available_bytes_fn=lambda: need).construction_bytes == need
    with pytest.raises(nfs.NativeFloatingMemoryError):
        fp.FloatingPondVertical.from_held_context(ctx, available_bytes_fn=lambda: need - 1)
    assert not ctx.column._maps["theta_saturation"].flags.writeable and v._held is ctx  # references, not copies
