"""CW-C1: the host checkpoint bundle written from and resumed into either namespace.

CPU tests (always run): the manifest records `backend: numpy`, a pre-CW-C1 manifest without the key still
loads, a bundle tagged for another backend or an unsupported one is refused, an extra manifest key is refused,
the graph digest of a NumPy graph equals the pre-CW-C1 algorithm (host digest unchanged), and the namespace
reconstruction is the identity on NumPy.

GPU tests (skip honestly without a device): a wet pause with a pending ledger and a non-trivial phase /
mobile pool saved from the device and resumed on the device reproduces the uninterrupted event BITWISE on every
science array; the completed handoff round-trips and cannot replay; a device bundle loaded with host
parameters (and a host bundle loaded with device parameters) is refused as a cross-backend migration; a
corrupted payload / hash / backend tag is refused; the device graph digest equals the host digest of the same
terrain.

Nothing here was run by its author (file-only tools); the root records the executed results.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np
import pytest

pytest.importorskip("maple")

from maple.core.backend import array_namespace, is_array
from maple.core.types.voxel import VoxelColumnState, VoxelSurfaceCache
from test_complete_backend import fixture_xp
from test_coupled_backends import H

from maple_syrup.checkpoint import (
    _graph_digest,
    load_checkpoint,
    save_checkpoint,
    sha256_file,
    validate_checkpoint_state,
)
from maple_syrup.complete_event import DryHandoff, complete_event
from maple_syrup.event_backend import item_to_namespace, state_to_namespace
from maple_syrup.sediment_event import SedimentEventError

IDENTITY = {"case": "controlled actual MAPLE bed", "forcing": "fixed", "source": "test", "options": {"dt": 1}}


def arrays_any(value, path=""):
    """Every PHYSICAL array reachable through dataclass fields and dicts, as host copies (NumPy or CuPy leaves).

    The one explicit exclusion is MAPLE's `VoxelColumnState.surface_cache` (`VoxelSurfaceCache`): a derived,
    identity-bound, non-persisted cache of the voxel totals (`init=False, compare=False`), deliberately absent from
    MAPLE's snapshot and therefore from a restored column. It is NOT physical state and is checked separately by
    `assert_surface_cache_consistent` (never carried into the restart comparison, never blindly accepted). Every
    other field, including every `init=False` field of any other type, is walked as before."""
    if is_array(value):
        yield path, H(value)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            if isinstance(value, VoxelColumnState) and f.name == "surface_cache":
                continue
            yield from arrays_any(getattr(value, f.name), path + "." + f.name)
    elif isinstance(value, dict):
        for key, v in value.items():
            yield from arrays_any(v, path + "." + key)


def assert_surface_cache_consistent(column, *, expect_none=False):
    """The derived voxel-totals cache of a column is either absent, or a `VoxelSurfaceCache` whose binding is
    the column's exact `mass_kg` object and whose totals equal, bitwise, an INDEPENDENT recomputation
    `sum(axis=-1)` of the physical mass in the array's own namespace (MAPLE's documented invariant). A stale
    cache bound to another array, or one disagreeing with the physical mass, fails; a restored column after a
    MAPLE snapshot decode is expected to carry none."""
    assert isinstance(column, VoxelColumnState)
    cache = column.surface_cache
    if expect_none:
        assert cache is None, "a restored column must not carry a persisted / stale surface cache"
        return
    if cache is None:
        return
    assert isinstance(cache, VoxelSurfaceCache)
    assert cache.describes(column.mass_kg), "surface cache is bound to a different mass array (stale)"
    xp = array_namespace(column.mass_kg)
    recomputed = xp.sum(column.mass_kg, axis=-1)
    assert_bitwise(cache.voxel_totals_kg, recomputed, "surface_cache.voxel_totals_kg vs mass_kg.sum(axis=-1)")


def assert_bitwise(a, b, name=""):
    """True bitwise equality of two host arrays: same dtype, same shape, same raw bytes (not value equality:
    -0.0 / 0.0 or differently typed equal values do not pass)."""
    a, b = np.ascontiguousarray(H(a)), np.ascontiguousarray(H(b))
    assert a.dtype == b.dtype, f"{name}: dtype {a.dtype} != {b.dtype}"
    assert a.shape == b.shape, f"{name}: shape {a.shape} != {b.shape}"
    assert a.tobytes() == b.tobytes(), f"{name}: raw bytes differ"


def assert_science_equal_any(a, b):
    """`tests/phase6/test_checkpoint.assert_science_equal` through host conversions: BITWISE (dtype, shape, raw
    bytes) on every array of the dry state, the listed result arrays and the per-class totals; equal accounting,
    quiet monitor and counters."""
    first, second = dict(arrays_any(a.dry_state)), dict(arrays_any(b.dry_state))
    assert first.keys() == second.keys()
    for name, value in first.items():
        assert_bitwise(value, second[name], name)
    for handoff in (a, b):  # derived caches are not science arrays; each must still be absent or consistent
        assert_surface_cache_consistent(handoff.dry_state.bed.voxel_column)
        assert_surface_cache_consistent(handoff.progress.result.state.bed.voxel_column)
    for key in ("hydrograph", "sediment_hydrograph", "cumulative_pickup_kg", "cumulative_deposition_kg",
                "cumulative_export_request_kg", "initial_bed_by_cell_class_kg", "ledger_process_totals_reset_kg",
                "peak_depth_m", "peak_velocity_m_s", "last_velocity_m_s", "cumulative_rain_m", "cumulative_drainage_m"):
        assert_bitwise(getattr(a.progress.result, key), getattr(b.progress.result, key), key)
    for key in a.progress.result.by_class:
        assert_bitwise(a.progress.result.by_class[key], b.progress.result.by_class[key], key)
    assert a.accounting == b.accounting
    assert a.dry_state.t_s == b.dry_state.t_s
    assert a.progress.quiet == b.progress.quiet
    assert a.progress.result.n_accepted_steps == b.progress.result.n_accepted_steps
    assert a.progress.result.n_commits == b.progress.result.n_commits
    assert a.dry_state.bed.committed_topography.commit_count == b.dry_state.bed.committed_topography.commit_count
    assert a.dry_state.graph.input_sha256 == b.dry_state.graph.input_sha256


def legacy_graph_digest(graph):
    """The pre-CW-C1 host digest algorithm (NumPy arrays by `isinstance`, everything else by `repr`)."""
    h = hashlib.sha256()
    for f in dataclasses.fields(graph):
        if f.name in ("xp", "input_sha256"):
            continue
        value = getattr(graph, f.name)
        if (f.name == "policy" and value == "strict") or (f.name == "pit_storage" and (value is None or not np.any(value))):
            continue
        h.update(f.name.encode())
        if isinstance(value, np.ndarray):
            h.update(str((value.shape, value.dtype.str)).encode())
            h.update(value.tobytes())
        else:
            h.update(repr(value).encode())
    return h.hexdigest()


def rewrite_manifest(path, mutate):
    metadata = path / "checkpoint.json"
    m = json.loads(metadata.read_text())
    mutate(m)
    metadata.write_text(json.dumps(m))


# =====================================================================================================================
# CPU
# =====================================================================================================================
def test_host_manifest_records_numpy_and_legacy_manifests_without_the_tag_still_load(tmp_path):
    args, kw = fixture_xp(np, wet=True)
    paused = complete_event(*args, **kw, checkpoint_callback=lambda p: True)
    path = tmp_path / "checkpoint"
    save_checkpoint(path, paused, args[1], args[2], args[6], IDENTITY)
    m = json.loads((path / "checkpoint.json").read_text())
    assert m["backend"] == "numpy" and m["schema"] == "maple-syrup-checkpoint/2"
    restored = load_checkpoint(path, args[1], args[2], args[6], IDENTITY)
    assert restored.result.state.graph.xp is np and restored.result.state.phase.xp is np
    assert item_to_namespace(paused, np) is paused and state_to_namespace(paused.result.state, np) is paused.result.state
    # pre-CW-C1 bundle: no backend key -> NumPy
    rewrite_manifest(path, lambda m: m.pop("backend"))
    legacy = load_checkpoint(path, args[1], args[2], args[6], IDENTITY)
    np.testing.assert_array_equal(legacy.result.state.bed.water.mobile_mass_by_cell_class_kg,
                                  restored.result.state.bed.water.mobile_mass_by_cell_class_kg)
    # another backend's bundle is a refused migration; an unknown tag or an extra key is refused
    rewrite_manifest(path, lambda m: m.__setitem__("backend", "cupy"))
    with pytest.raises(SedimentEventError, match="cross-backend"):
        load_checkpoint(path, args[1], args[2], args[6], IDENTITY)
    rewrite_manifest(path, lambda m: m.__setitem__("backend", "jax"))
    with pytest.raises(SedimentEventError, match="unsupported backend"):
        load_checkpoint(path, args[1], args[2], args[6], IDENTITY)
    rewrite_manifest(path, lambda m: (m.__setitem__("backend", "numpy"), m.__setitem__("extra", 1)))
    with pytest.raises(SedimentEventError, match="identity or payload set"):
        load_checkpoint(path, args[1], args[2], args[6], IDENTITY)
    # the host graph digest is the pre-CW-C1 digest
    g = paused.result.state.graph
    assert _graph_digest(g) == legacy_graph_digest(g)
    # mismatched column / sediment namespaces are refused before any file is read
    with pytest.raises(SedimentEventError, match="namespace"):
        load_checkpoint(path, args[1], args[2], object(), IDENTITY)


def test_same_namespace_conversion_is_the_identity_and_refusals_precede_it():
    """`item_to_namespace` / `result_to_namespace` / `state_to_namespace` return the very object for a
    supported item already in the target namespace (wet progress and completed handoff alike); an unsupported
    namespace or item type is refused before any conversion."""
    from maple_syrup.event_backend import result_to_namespace

    args, kw = fixture_xp(np, wet=True)
    paused = complete_event(*args, **kw, checkpoint_callback=lambda p: True)
    assert item_to_namespace(paused, np) is paused
    assert result_to_namespace(paused.result, np) is paused.result
    assert state_to_namespace(paused.result.state, np) is paused.result.state
    quiet_args, quiet_kw = fixture_xp(np)
    handoff = complete_event(*quiet_args, **quiet_kw)
    assert isinstance(handoff, DryHandoff) and item_to_namespace(handoff, np) is handoff
    assert state_to_namespace(handoff.dry_state, np) is handoff.dry_state
    with pytest.raises(SedimentEventError, match="unsupported array namespace"):
        item_to_namespace(paused, object())
    with pytest.raises(SedimentEventError, match="EventProgress or a DryHandoff"):
        item_to_namespace(paused.result, np)
    with pytest.raises(SedimentEventError, match="SedimentEventState"):
        state_to_namespace(paused, np)


def test_zero_dimensional_reductions_accept_host_scalars_and_refuse_everything_else():
    """The bundle encodes NumPy scalars through `.item()` (Python int / float) and NumPy reductions as 0-D
    arrays; the loop produces NumPy scalars and 0-D arrays. All of those are valid for the 0-D reduction
    fields with the declared dtype; bools, sequences, broadcastable arrays, wrong dtypes, non-finite or negative
    values are refused. Physical array fields keep their strict array rule (no scalar substitutes)."""
    args, kw = fixture_xp(np, wet=True)
    ctx, column, sediment = args[1], args[2], args[6]
    paused = complete_event(*args, **kw, checkpoint_callback=lambda p: True)
    r = paused.result

    def with_result(**updates):
        return dataclasses.replace(paused, result=dataclasses.replace(r, **updates))

    def regime(value):
        return {**r.regime_cell_steps, "dry": value}

    def ok(progress):
        validate_checkpoint_state(progress, ctx, column, sediment)

    def refused(progress):
        with pytest.raises(SedimentEventError, match="invalid checkpoint accumulator"):
            validate_checkpoint_state(progress, ctx, column, sediment)

    count = int(r.regime_cell_steps["dry"])
    # integer 0-D reductions (regime counts, branch counts): Python int, NumPy int64 scalar, 0-D int64 array
    for value in (count, np.int64(count), np.asarray(count, dtype=np.int64)):
        ok(with_result(regime_cell_steps=regime(value)))
        ok(with_result(cell_steps_no_runon=value))
    # float 0-D reductions: Python float, NumPy float64 scalar, 0-D float64 array
    peak = float(r.peak_outlet_discharge_m3_s)
    for value in (peak, np.float64(peak), np.asarray(peak)):
        ok(with_result(peak_outlet_discharge_m3_s=value))
    # refused: bool, wrong dtype (float for an int field, int for a float field), negative, non-finite,
    # a 1-element array (broadcastable, not 0-D), a list, a float32 scalar
    for value in (True, float(count), -1, np.int32(count), np.asarray([count], dtype=np.int64), [count]):
        refused(with_result(regime_cell_steps=regime(value)))
        refused(with_result(cell_steps_partial_runon=value))
    for value in (False, int(peak), -1.0, float("nan"), float("inf"), np.float32(peak), np.asarray([peak]), [peak]):
        refused(with_result(peak_outlet_discharge_m3_s=value))
    # physical arrays never accept a scalar substitute
    refused(with_result(cumulative_rain_m=0.0))
    refused(with_result(initial_mobile_kg=np.float64(0.0)))
    # the real path: the host scalars the bundle restores load and round-trip bitwise
    assert isinstance(r.regime_cell_steps["dry"], (np.generic, np.ndarray))


# =====================================================================================================================
# GPU
# =====================================================================================================================
def test_gpu_wet_pause_and_same_backend_resume_is_bitwise_with_the_uninterrupted_event(gpu, tmp_path):
    args, kw = fixture_xp(gpu, wet=True, implementation="cuda")
    ctx, column, sediment = args[1], args[2], args[6]
    paused = complete_event(*args, **kw, checkpoint_callback=lambda p: True)
    state = paused.result.state
    assert state.t_s == 2 and state.graph.xp is gpu
    assert np.any(H(state.bed.water.mobile_mass_by_cell_class_kg) > 0)
    assert np.any(H(state.bed.ledger.pending_bed_mass_change_kg) != 0)
    assert np.any(H(state.phase.position_m) > 0) or np.any(H(state.phase.fraction)[..., 1:] > 0)  # non-trivial phase
    count = state.bed.committed_topography.commit_count
    path = tmp_path / "checkpoint"
    save_checkpoint(path, paused, ctx, column, sediment, IDENTITY)
    m = json.loads((path / "checkpoint.json").read_text())
    assert m["backend"] == "cupy"
    with np.load(path / "continuation.npz") as archive:  # the bundle is a host format
        assert all(isinstance(archive[k], np.ndarray) for k in archive.files)
    restored = load_checkpoint(path, ctx, column, sediment, IDENTITY)
    rs = restored.result.state
    assert rs.graph.xp is gpu and rs.phase.xp is gpu and rs.network.xp is gpu and rs.grid.xp is gpu
    assert type(rs.bed.voxel_column.mass_kg) is gpu.ndarray and type(rs.storm.depth_m) is gpu.ndarray
    assert type(restored.result.hydrograph) is gpu.ndarray and isinstance(restored.result.boundaries, np.ndarray)
    assert isinstance(restored.planned_boundaries, np.ndarray) and isinstance(rs.terrain.active_mask, np.ndarray)
    assert rs.bed.committed_topography.commit_count == count and rs.graph.input_sha256 == state.graph.input_sha256
    for name, value in arrays_any(state.bed.ledger):
        assert_bitwise(value, dict(arrays_any(rs.bed.ledger))[name], name)
    paused_arrays, restored_arrays = dict(arrays_any(state)), dict(arrays_any(rs))
    assert paused_arrays.keys() == restored_arrays.keys() and ".bed.voxel_column.mass_kg" in paused_arrays
    for name, value in paused_arrays.items():  # reconstruction alone reproduces every PHYSICAL array of the paused state
        assert_bitwise(value, restored_arrays[name], name)
    # the derived voxel-totals cache is not persisted: the paused column may carry a consistent one (bound to its
    # own mass array, equal to an independent recomputation), the restored column carries none
    assert_surface_cache_consistent(state.bed.voxel_column)
    assert_surface_cache_consistent(rs.bed.voxel_column, expect_none=True)
    full = complete_event(*args, **kw)
    resumed = complete_event(rs, *args[1:], **kw, continuation=restored)
    assert isinstance(full, DryHandoff) and isinstance(resumed, DryHandoff)
    assert_science_equal_any(full, resumed)
    assert float(H(full.progress.result.by_class["actual_pickup"]).sum()) > 0


def test_gpu_complete_handoff_round_trips_and_cannot_replay(gpu, tmp_path):
    args, kw = fixture_xp(gpu, implementation="cuda")
    out = complete_event(*args, **kw)
    path = tmp_path / "handoff"
    save_checkpoint(path, out, args[1], args[2], args[6], IDENTITY)
    with pytest.raises(SedimentEventError, match="completed"):
        load_checkpoint(path, args[1], args[2], args[6], IDENTITY)
    loaded = load_checkpoint(path, args[1], args[2], args[6], IDENTITY, allow_complete=True)
    assert loaded.dry_state.graph.xp is gpu
    assert_science_equal_any(out, loaded)
    with pytest.raises(SedimentEventError, match="wet checkpoint"):
        complete_event(loaded.dry_state, *args[1:], **kw, continuation=loaded)


def test_gpu_cross_backend_loads_are_refused_in_both_directions(gpu, tmp_path):
    dev_args, dev_kw = fixture_xp(gpu, wet=True, implementation="cuda")
    dev_paused = complete_event(*dev_args, **dev_kw, checkpoint_callback=lambda p: True)
    dev_path = tmp_path / "device"
    save_checkpoint(dev_path, dev_paused, dev_args[1], dev_args[2], dev_args[6], IDENTITY)
    host_args, host_kw = fixture_xp(np, wet=True)
    with pytest.raises(SedimentEventError, match="cross-backend"):
        load_checkpoint(dev_path, host_args[1], host_args[2], host_args[6], IDENTITY)
    host_paused = complete_event(*host_args, **host_kw, checkpoint_callback=lambda p: True)
    host_path = tmp_path / "host"
    save_checkpoint(host_path, host_paused, host_args[1], host_args[2], host_args[6], IDENTITY)
    with pytest.raises(SedimentEventError, match="cross-backend"):
        load_checkpoint(host_path, dev_args[1], dev_args[2], dev_args[6], IDENTITY)
    # the two graphs describe the same terrain: the byte-based digest agrees across namespaces
    assert _graph_digest(dev_args[0].graph) == _graph_digest(host_args[0].graph)
    # mixed column / sediment namespaces are refused
    with pytest.raises(SedimentEventError, match="namespace"):
        load_checkpoint(dev_path, dev_args[1], dev_args[2], host_args[6], IDENTITY)


@pytest.mark.parametrize("kind", ["hash", "backend", "negative", "missing", "identity"])
def test_gpu_corrupt_or_incompatible_device_bundle_refused(gpu, tmp_path, kind):
    args, kw = fixture_xp(gpu, wet=True, implementation="cuda")
    paused = complete_event(*args, **kw, checkpoint_callback=lambda p: True)
    path = tmp_path / "checkpoint"
    save_checkpoint(path, paused, args[1], args[2], args[6], IDENTITY)
    expected = json.loads(json.dumps(IDENTITY))
    if kind == "hash":
        with (path / "continuation.npz").open("ab") as f:
            f.write(b"corrupt")
    elif kind == "backend":
        rewrite_manifest(path, lambda m: m.__setitem__("backend", "numpy"))
    elif kind == "missing":
        (path / "maple_state.npz").unlink()
    elif kind == "identity":
        expected["source"] = "changed"
    else:
        m = json.loads((path / "checkpoint.json").read_text())
        node = m["payload"]["fields"]["result"]["fields"]["state"]["state"]["sediment_velocity_m_s"]
        with np.load(path / "continuation.npz") as a:
            data = {k: a[k].copy() for k in a.files}
        data[node["array"]].flat[0] = -1
        np.savez(path / "continuation.npz", **data)
        m["files"]["continuation.npz"] = sha256_file(path / "continuation.npz")
        (path / "checkpoint.json").write_text(json.dumps(m))
    with pytest.raises((SedimentEventError, OSError, ValueError)):
        load_checkpoint(path, args[1], args[2], args[6], expected)
