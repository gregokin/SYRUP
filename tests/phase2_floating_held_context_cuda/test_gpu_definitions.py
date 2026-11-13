"""Actual-GPU test DEFINITIONS for the CUDA held-law record /2 (WRITTEN, NOT RUN by their author: no GPU/CuPy/NVRTC/nvidia-smi was touched). Opt-in (SYRUP_RELEASE_GPU=1); Root runs them on GPU0 with `.venv-cupy`. They use real
native CUDA owners (`initialize_native_floating_soil(backend="cuda")` on a device bed) and genuine `FloatingPondVertical`s; the ORIGINAL device objects define the expected identities. No GPU construction of a Pond or a file
is defined here; no CPU numerical fallback exists; cross-device/architecture independence of the hashes is a code fact, not tested (one UUID)."""
# ruff: noqa: RUF059
from __future__ import annotations

import gc
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_T = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("e3_builders", _T / "phase2_native_floating_soil" / "builders.py")
b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b)
_hspec = importlib.util.spec_from_file_location("hbuild_gpu", _T / "phase2_floating_held_context" / "hbuild.py")

from maple_syrup import floating_held_context as fh
from maple_syrup import floating_held_context_cuda as fhc
from maple_syrup import floating_pond as fp
from maple_syrup import floating_preferential_infiltration as fpi
from maple_syrup import floating_soil_layers_cuda as fcu
from maple_syrup import native_floating_soil as nfs
from maple_syrup import native_floating_soil_cuda as ncuda
from maple_syrup import soil_texture_properties as stp

BIG = 1 << 40
AVAIL = lambda: BIG
TEXTURE_KS = {"matrix_conductivity_m_s": None, "matrix_conductivity_mode": "texture_ks", "latent_saturated_conductivity_m_s": 3e-6}


@pytest.fixture(scope="module")
def cp():
    if os.environ.get("SYRUP_RELEASE_GPU") != "1":
        pytest.skip("GPU tests are opt-in (SYRUP_RELEASE_GPU=1); not selected, no GPU claim")
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the CUDA held record is not exercised (no claim)")
    return backend.cupy_module()


def device_bed(cp, bed):
    from maple.core.types.active_layer import ActiveLayerState
    from maple.core.types.voxel import VoxelColumnState

    return nfs.NativeBed(VoxelColumnState(mass_kg=cp.asarray(bed.voxel_column.mass_kg)), ActiveLayerState(mass_kg=cp.asarray(bed.active_layer.mass_kg)), bed.geometry, bed.grain_classes,
                         bed.mass_resolution_kg, bed.active_mask)


LAYERS = {1: (0.16,), 3: b.LAYERS, 65: tuple([0.16 / 65] * 65)}


def spec_for(cp, L, N, kind="cell", species=0, rate="scalar", weights="enabled"):
    a3 = np.full(N, 2.0e-3) if kind == "cell" else 2.0e-3
    mr = np.full(N, 1e-5) if rate == "cell" else (float("inf") if rate == "inf" else 1e-5)
    bg = np.ones(L) if L == 1 else np.array([0.0] + [1.0] * (L - 1))
    sw = tuple(np.eye(L)[i % L].copy() for i in range(species))
    if weights == "zero":  # every species disabled (all-zero receiving weights); the BACKGROUND stays separate
        sw = tuple(np.zeros(L) for _ in range(species))
    elif weights == "part":  # the first species disabled, the others enabled
        sw = tuple(np.zeros(L) if i == 0 else w for i, w in enumerate(sw))
    sc = tuple(np.full(N, 0.5 + 0.25 * i) for i in range(species)) if species else None
    return fh.PreferentialSpec(a3, 0.5, mr, bg, sw, sc)


def cuda_vertical(cp, *, L=3, mode="explicit", overrides=None, bottom="open", spec=None, bed=None):
    bed = b.stratified() if bed is None else bed
    dbed = device_bed(cp, bed)
    kw = TEXTURE_KS if mode == "texture_ks" else {}
    owner = nfs.initialize_native_floating_soil(dbed, b.config(backend="cuda", layers=LAYERS[L], bottom=bottom, overrides=overrides, **kw), initial_relative_saturation=0.5)
    pref = None
    if spec is not None:  # the ORIGINAL device preferential operator, prepared from the device copies of the user's raw inputs
        a3 = cp.asarray(spec.a3) if isinstance(spec.a3, np.ndarray) else spec.a3
        mr = cp.asarray(spec.max_intake_rate_m_s) if isinstance(spec.max_intake_rate_m_s, np.ndarray) else spec.max_intake_rate_m_s
        contrib = None if spec.species_contributions is None else tuple(cp.asarray(c) for c in spec.species_contributions)
        pref = fpi.prepare_floating_preferential(owner.column, a3=a3, max_fraction_per_hour=spec.max_fraction_per_hour, max_intake_rate_m_s=mr, background_receiver_weights=spec.background_receiver_weights,
                                                 species_receiver_weights=spec.species_receiver_weights, species_contributions=contrib)
    return fp.FloatingPondVertical(owner, available_bytes_fn=AVAIL, preferential=pref), owner


def expected_of(v):
    t = v.held_template()
    return fh.ExpectedHeldIdentities(v.config.sha256, t.column.sha256, t.matrix_op.sha256, None if t.preferential is None else t.preferential.sha256)


@pytest.mark.parametrize("L", [1, 3, 65])
@pytest.mark.parametrize("mode", ["explicit", "texture_ks"])
@pytest.mark.parametrize("overrides", [None, {}])
def test_export_parse_reconstruct_reproduces_the_original_cuda_identities_bit_exact(cp, L, mode, overrides):
    v, owner = cuda_vertical(cp, L=L, mode=mode, overrides=overrides)
    rec = fhc.export_held_context_cuda(v, None, verify=False)
    assert rec.meta()["schema"] == fh.SCHEMA2 and rec.meta()["backend"] == "cuda" and rec.meta()["kernel"] == fhc.kernel_descriptor()
    meta = rec.meta()
    arrays = {k: np.array(a) for k, a in rec.arrays().items()}
    parsed = fh.record_from_parts(meta, arrays, available_bytes_fn=AVAIL)  # through plain host parts (a file codec's job)
    ctx = fh.reconstruct_held_context(parsed, expected_of(v), available_bytes_fn=AVAIL)
    t = v.held_template()
    assert ctx.backend == "cuda" and ctx.column.sha256 == t.column.sha256 and ctx.matrix_op.sha256 == t.matrix_op.sha256 and ctx.config.sha256 == owner.config.sha256
    assert ctx.config.backend == "cuda" and (ctx.config.overrides is None) == (overrides is None)
    assert isinstance(ctx.edges_m, np.ndarray) and ctx.host_depth0_m.tobytes() == cp.asnumpy(t.depth0_m).tobytes() and (ctx.ks_m_s is None) == (mode == "explicit")
    for name in ("water", "state", "history"):
        assert not hasattr(ctx, name)  # no W/J/I/state/owner


@pytest.mark.parametrize("kind,rate,species", [("cell", "scalar", 0), ("scalar", "scalar", 2), ("cell", "cell", 2), ("scalar", "inf", 0)])
@pytest.mark.parametrize("L", [1, 3])
def test_preferential_raw_spec_reprepared_on_the_gpu_matches_the_original_gpu_sha(cp, kind, rate, species, L):
    spec = spec_for(cp, L, 3, kind, species, rate)
    v, owner = cuda_vertical(cp, L=L, spec=spec)
    rec = fhc.export_held_context_cuda(v, spec)  # verify=True: a full device reconstruction inside
    ctx = fh.reconstruct_held_context(rec, expected_of(v), available_bytes_fn=AVAIL)
    assert ctx.preferential.sha256 == v.held_template().preferential.sha256 and ctx.preferential.backend == "cuda"
    # never claimed equal to the CPU-derived preferential sha (the headers contain the backend)


def test_partial_rock_d0_and_no_provider_bed_or_owner_factory_during_restore(cp, monkeypatch):
    v, owner = cuda_vertical(cp, bed=b.stratified(shallow=((0, 2),)))
    rec = fhc.export_held_context_cuda(v, None)

    def boom(*a, **k):
        raise AssertionError("must not run during reconstruct")

    for obj, name in ((stp, "estimate_soil_texture_properties"), (nfs, "_read_texture"), (nfs, "initialize_native_floating_soil"), (nfs, "native_source_witness"), (ncuda, "initialize_cuda"),
                      (nfs, "NativeBed")):
        monkeypatch.setattr(obj, name, boom)
    ctx = fh.reconstruct_held_context(rec, expected_of(v), available_bytes_fn=AVAIL)
    assert ctx.column.sha256 == v.held_template().column.sha256


def test_wrong_inputs_refuse_before_any_owned_large_array_or_transfer_and_a_failed_trial_is_retryable(cp, monkeypatch):
    from maple.core import backend as mb

    v, owner = cuda_vertical(cp, mode="texture_ks", spec=spec_for(cp, 3, 3, "cell", 2))
    rec = fhc.export_held_context_cuda(v, spec_for(cp, 3, 3, "cell", 2))
    exp = expected_of(v)
    dev_need, host_need, _ = fhc.plan(rec.meta())
    up, alloc = [], []
    monkeypatch.setattr(mb, "to_device", lambda *a, _r=mb.to_device, **k: up.append(1) or _r(*a, **k))
    monkeypatch.setattr(cp, "empty", lambda *a, _r=cp.empty, **k: alloc.append(1) or _r(*a, **k))
    for dev_a, host_a, kw in ((0, BIG, {}), (dev_need - 1, BIG, {}), (BIG, 0, {}), (BIG, host_need + fcu.PY_ALLOWANCE_BYTES - 1, {})):
        with pytest.raises(fh.HeldContextError):
            fh.reconstruct_held_context(rec, exp, available_bytes_fn=lambda a=dev_a: a, host_available_bytes_fn=lambda a=host_a: a, **kw)
    assert up == [] and alloc == []  # refused BEFORE any upload or device array
    fh.reconstruct_held_context(rec, exp, available_bytes_fn=lambda: dev_need, host_available_bytes_fn=lambda: host_need + fcu.PY_ALLOWANCE_BYTES)  # exact needs admit
    with pytest.raises(fh.HeldContextError):
        fh.reconstruct_held_context(rec, exp, available_bytes_fn=lambda: dev_need, host_available_bytes_fn=lambda: host_need + fcu.PY_ALLOWANCE_BYTES, reserve_bytes=1)  # reserve counted once per pool
    with pytest.raises(fh.HeldContextError):
        fh.reconstruct_held_context(rec, exp, available_bytes_fn=lambda: dev_need + 5, host_available_bytes_fn=lambda: host_need + fcu.PY_ALLOWANCE_BYTES, reserve_bytes=6)
    fh.reconstruct_held_context(rec, exp, available_bytes_fn=lambda: dev_need + 5, host_available_bytes_fn=lambda: host_need + fcu.PY_ALLOWANCE_BYTES + 7, reserve_bytes=5, host_reserve_bytes=7)
    bad = fh.ExpectedHeldIdentities(exp.config_sha256, "0" * 64, exp.matrix_op_sha256, exp.pref_sha256)
    with pytest.raises(fh.HeldContextError):  # misstated identity: the source record/vertical are unchanged and the retry with the right ones succeeds
        fh.reconstruct_held_context(rec, bad, available_bytes_fn=AVAIL)
    assert fh.reconstruct_held_context(rec, exp, available_bytes_fn=AVAIL).column.sha256 == exp.root_column_sha256


def test_cold_transfers_are_chunked_counted_exact_and_carry_no_water(cp, monkeypatch):
    """Counted traffic, attributed. The instrumentation records each helper call ONCE (the wrapper calls the real helper a single time) and MAPLE's own `read_transfer_counters` delta is the independent control.
    EXPORT D2H = exactly the held device maps. RECONSTRUCT H2D = (a) the H0 record uploads: the held maps + the (L-1) interface diffusivity, plus (b) the constants the ACTUAL public column preparation uploads through
    the counted helper (`fsl._prepare_cuda`): the HOST nominal edges `8(L+1)`, the nominal separations `8*max(L-1, 1)` and the shared scalar `theta_residual` as a one-element array `8`. The matrix scalars (suction,
    vegetation factor) are `cp.array(np.array([v]))` constructions inside `fmi._law_input_cuda`: NOT routed through the counted helper, so they are neither counted by the hook nor by the counters (stated, not assumed)."""
    from maple.core import backend as mb

    L, N = 3, 50003
    v, owner = cuda_vertical(cp, mode="texture_ks", bed=b.bed_from(b.fx.stratified_dense(shallow_cells=(), ny=1, nx=N)))
    down, upl = [], []

    def host_wrapper(x, *a, _r=mb.to_host, **k):
        out = _r(x, *a, **k)  # ONE real call
        down.append(int(np.asarray(out).nbytes))
        return out

    def device_wrapper(x, *a, _r=mb.to_device, **k):
        upl.append(int(np.asarray(x).nbytes))
        return _r(x, *a, **k)

    monkeypatch.setattr(mb, "to_host", host_wrapper)
    monkeypatch.setattr(mb, "to_device", device_wrapper)
    mb.reset_transfer_counters()
    rec = fhc.export_held_context_cuda(v, None, verify=False)
    c_export = mb.read_transfer_counters()
    held = sum(int(a.nbytes) for k, a in rec.arrays().items() if k in ("theta_field_capacity", "theta_saturation", "depth0_m", "active", "ks_m_s"))
    assert max(down) <= fcu.HASH_CHUNK_BYTES and sum(down) == held and upl == []  # D2H: exactly the held maps, chunk-bounded, no H2D, no W/J/I
    assert c_export.device_to_host_bytes == held and c_export.host_to_device_bytes == 0  # independent control (MAPLE counters)
    down.clear()
    mb.reset_transfer_counters()
    fh.reconstruct_held_context(rec, expected_of(v), available_bytes_fn=AVAIL)
    c_rec = mb.read_transfer_counters()
    record_uploads = held + 8 * (L - 1)  # H0: held maps + the interface diffusivity vector
    public_constants = 8 * (L + 1) + 8 * max(L - 1, 1) + 8  # fsl._prepare_cuda: edges + nominal separations + theta_residual scalar (counted helper)
    assert max(upl) <= fcu.HASH_CHUNK_BYTES
    assert sum(upl) == record_uploads + public_constants, (sum(upl), record_uploads, public_constants)  # 3250195 + 16 + 56 = 3250267 on the reproduced run
    assert c_rec.host_to_device_bytes == sum(upl)  # the hook and MAPLE's counters agree: the instrumentation does not double the traffic
    # cold hash/staging reads of the public preparations: each one bounded chunk
    assert all(x <= fcu.HASH_CHUNK_BYTES for x in down) and c_rec.device_to_host_bytes == sum(down)


def test_moderate_n50003_warm_true_allocator_peak_and_plateau(cp):
    class Peak:
        def __init__(self):
            self.pool, self.rounding, self.peak = cp.get_default_memory_pool(), 0, 0
            self.base = int(self.pool.used_bytes())

        def __call__(self, size):
            ptr = self.pool.malloc(size)
            self.peak = max(self.peak, int(self.pool.used_bytes()) - self.base)
            self.rounding += int(ptr.mem.size) - int(size)
            return ptr

    N = 50003
    v, owner = cuda_vertical(cp, mode="texture_ks", bed=b.bed_from(b.fx.stratified_dense(shallow_cells=(), ny=1, nx=N)))
    rec = fhc.export_held_context_cuda(v, None, verify=False)
    exp = expected_of(v)
    dev_need, host_need, _ = fhc.plan(rec.meta())
    fh.reconstruct_held_context(rec, exp, available_bytes_fn=AVAIL)  # warm (kernels/compile)
    gc.collect()
    cp.cuda.Device().synchronize()
    pool = cp.get_default_memory_pool()
    used = []
    for _ in range(5):
        hook = Peak()
        with cp.cuda.using_allocator(hook):
            ctx = fh.reconstruct_held_context(rec, exp, available_bytes_fn=AVAIL)
        assert hook.peak <= dev_need + hook.rounding, (hook.peak, dev_need, hook.rounding)  # TRUE transient peak vs the declared model + the ACTUAL rounding
        del ctx
        gc.collect()
        cp.cuda.Device().synchronize()
        used.append(int(pool.used_bytes()))
    assert max(used) - min(used) <= hook.rounding + 8 * N  # released-context plateau: no record/history growth


_WORKER = r"""
import json, sys
sys.path[:0] = sys.argv[3:]
import numpy as np
from maple_syrup import floating_held_context as fh
big = lambda: 1 << 40
meta = json.loads(open(sys.argv[1] + "/meta.json").read())
names = json.loads(open(sys.argv[1] + "/arrays.json").read())
arrays = {k: np.frombuffer(bytearray(open(sys.argv[1] + f"/{k}.bin", "rb").read()), dtype=np.dtype(dt)).reshape(sh).copy() for k, (dt, sh) in names.items()}
ex = json.loads(open(sys.argv[1] + "/expected.json").read())
rec = fh.record_from_parts(meta, arrays, available_bytes_fn=big)
ctx = fh.reconstruct_held_context(rec, fh.ExpectedHeldIdentities(**ex), available_bytes_fn=big)
print(json.dumps({"column": ctx.column.sha256, "matrix": ctx.matrix_op.sha256, "config": ctx.config.sha256}))
"""


def test_fresh_process_gpu_reconstruct_without_the_original_bed_reproduces_the_cuda_ids(cp, tmp_path):
    """The expected identities come from the ORIGINAL device vertical of THIS process; the fresh process has no bed, owner or case (record parts + static identities only)."""
    v, owner = cuda_vertical(cp, mode="texture_ks")
    rec = fhc.export_held_context_cuda(v, None)
    (tmp_path / "meta.json").write_text(json.dumps(rec.meta()))
    names = {}
    for k, a in rec.arrays().items():
        (tmp_path / f"{k}.bin").write_bytes(a.tobytes())
        names[k] = [str(a.dtype), list(a.shape)]
    (tmp_path / "arrays.json").write_text(json.dumps(names))
    e = expected_of(v)
    (tmp_path / "expected.json").write_text(json.dumps({"config_sha256": e.config_sha256, "root_column_sha256": e.root_column_sha256, "matrix_op_sha256": e.matrix_op_sha256, "pref_sha256": e.pref_sha256}))
    proc = subprocess.run([sys.executable, "-c", _WORKER, str(tmp_path), "", *[p for p in sys.path if p]], capture_output=True, text=True, timeout=600, check=False)
    assert proc.returncode == 0, proc.stderr[-3000:]
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert (got["column"], got["matrix"], got["config"]) == (e.root_column_sha256, e.matrix_op_sha256, e.config_sha256)


@pytest.mark.parametrize("weights", ["zero", "part"])
def test_all_zero_and_partly_disabled_species_receiving_weights_restore_the_exact_original_sha(cp, weights):
    """Explicit raw spec with disabled species (background separate and enabled): original GPU operator -> export -> parse -> reconstruct, exact preferential sha; the older S0/2 definitions had every species enabled."""
    spec = spec_for(cp, 3, 3, "cell", 3, "scalar", weights)
    v, owner = cuda_vertical(cp, L=3, spec=spec)
    rec = fhc.export_held_context_cuda(v, spec)
    meta = rec.meta()
    parsed = fh.record_from_parts(meta, {k: np.array(a) for k, a in rec.arrays().items()}, available_bytes_fn=AVAIL)
    ctx = fh.reconstruct_held_context(parsed, expected_of(v), available_bytes_fn=AVAIL)
    assert ctx.preferential.sha256 == v.held_template().preferential.sha256 and meta["pref"]["has_background"] is True
    assert ctx.preferential.n_species == 3 and ctx.preferential.n_species_enabled == v.held_template().preferential.n_species_enabled


def windowed_cuda_vertical(cp, tmp_path, *, mode="explicit"):
    """A GENUINE native resident-window bed on the device (`dense_deep_bed` + `windowed_counterpart(xp=cp)` of the qualified E4c fixtures), the true six-grain Pond config/native bed, an actual NativeOwner and an
    actual FloatingPondVertical. No numpy shadow, no fake Windowed/Owner class."""
    spec = importlib.util.spec_from_file_location("fpond_builders_h0", _T / "phase2_floating_pond" / "fpond_builders.py")
    fb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fb)
    wh = fb._windowed_helpers()
    z = fb.ph.plane(base=0.12, drop=0.004)
    dense, ctx = wh.dense_deep_bed(z, depth=0.003)
    wbed, wctx = wh.windowed_counterpart(dense, ctx, tmp_path / "store", xp=cp)
    kw = TEXTURE_KS if mode == "texture_ks" else {}
    cfg = fb.config(backend="cuda", **kw)
    owner = nfs.initialize_native_floating_soil(fb.native_bed(wbed, wctx), cfg, initial_relative_saturation=0.5)
    return fp.FloatingPondVertical(owner, available_bytes_fn=AVAIL), owner, wbed


@pytest.mark.parametrize("mode", ["explicit", "texture_ks"])
def test_genuine_windowed_cuda_context_export_parse_reconstruct_bit_exact(cp, tmp_path, mode):
    v, owner, wbed = windowed_cuda_vertical(cp, tmp_path, mode=mode)
    assert type(wbed.voxel_column).__name__ == "WindowedVoxelColumnState"  # the actual native resident-window class, not a dense stand-in
    rec = fhc.export_held_context_cuda(v, None, verify=False)  # captures held maps/config only: no W/J/I
    meta = rec.meta()
    assert meta["n_classes"] == 6 and meta["backend"] == "cuda" and "fl_water" not in rec.arrays()
    parsed = fh.record_from_parts(meta, {k: np.array(a) for k, a in rec.arrays().items()}, available_bytes_fn=AVAIL)
    ctx = fh.reconstruct_held_context(parsed, expected_of(v), available_bytes_fn=AVAIL)
    t = v.held_template()
    assert ctx.column.sha256 == t.column.sha256 and ctx.matrix_op.sha256 == t.matrix_op.sha256 and ctx.config.sha256 == owner.config.sha256
    assert ctx.host_depth0_m.tobytes() == cp.asnumpy(t.depth0_m).tobytes()  # the source D0 and true cfg maps bit-exact
    assert (ctx.ks_m_s is None) == (mode == "explicit") and (mode == "explicit" or ctx.host_ks_m_s.tobytes() == cp.asnumpy(owner.ks_m_s).tobytes())
