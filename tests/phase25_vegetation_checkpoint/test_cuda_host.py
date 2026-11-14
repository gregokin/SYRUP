"""CPU-HOST controls of the explicit CUDA twin (`floating_lifecycle_checkpoint_vegetation_cuda`, AUTHORED, NOT RUN on a device): importing never imports CuPy, the profile / schema / names are distinct from the CPU vegetation and the old
profiles, and without a device every entry refuses typed BEFORE any reader, path check or callback. No GPU / CuPy / NVRTC / nvidia-smi is touched here."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest
from vck_builders import AVAIL, BIG, IDENT, boundary_vns, load, save

from maple_syrup import floating_checkpoint as fc
from maple_syrup import floating_context_checkpoint as fcc
from maple_syrup import floating_held_context_vegetation as fhv
from maple_syrup import floating_lifecycle_checkpoint as lcp
from maple_syrup import floating_lifecycle_checkpoint_cuda as s2
from maple_syrup import floating_lifecycle_checkpoint_vegetation as cpuv
from maple_syrup import floating_lifecycle_checkpoint_vegetation_cuda as cv
from maple_syrup import native_floating_soil as nfs
from maple_syrup import pond_checkpoint as pc

REFUSE = (cpuv.VegetationLifecycleCheckpointError, fc.FloatingCheckpointError, pc.PondCheckpointError, nfs.NativeFloatingMemoryError)


def test_importing_the_twin_never_imports_cupy_or_leaf():
    code = "import sys, maple_syrup.floating_lifecycle_checkpoint_vegetation_cuda as m; assert 'cupy' not in sys.modules, 'cupy'; print('ok')"
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, env=dict(os.environ), check=False)
    assert out.stdout.strip() == "ok", out.stderr


def test_the_profiles_schemas_and_names_are_distinct_from_every_other_family():
    assert cv.PROFILE.endswith("/2-cuda") and cv.PROFILE != cpuv.PROFILE != lcp.PROFILE != s2.PROFILE
    assert cv.CONTEXT_SCHEMA == fhv.SCHEMA4 and cpuv.CONTEXT_SCHEMA == fhv.SCHEMA3 and cv.MANIFEST_NAME == cpuv.MANIFEST_NAME != lcp.MANIFEST_NAME
    assert cv.MANIFEST_KEYS == cpuv.MANIFEST_KEYS == lcp.MANIFEST_KEYS | {"vegetation"}
    assert issubclass(cv.VegetationLifecycleCheckpointCudaError, cpuv.VegetationLifecycleCheckpointError) and issubclass(cpuv.VegetationLifecycleCheckpointError, lcp.LifecycleCheckpointError)


def test_a_cpu_file_is_refused_by_the_cuda_loader_and_a_cuda_profile_by_the_cpu_loader_before_any_payload(tmp_path, leaf):
    ns = boundary_vns(tmp_path / "a", leaf, "numpy", False)
    saved = save(ns, tmp_path / "ck")
    d = tmp_path / "as_cuda"
    shutil.copytree(saved.path, d)
    m = json.loads((d / cpuv.MANIFEST_NAME).read_text())
    m["profile"] = cv.PROFILE
    sealed = fcc._seal(m)
    (d / cpuv.MANIFEST_NAME).write_text(pc._json_dumps(sealed))
    with pytest.raises(REFUSE, match="unknown vegetation lifecycle profile"):
        load(ns, d, type(saved)(d, sealed["manifest_sha256"], saved.kind, 1, "x"))  # the CPU loader reads only the CPU vegetation profile
    # the CUDA loader on a host without a device refuses at the DEVICE GATE, before any reader / path check / factory
    reads, made = [], []

    def reader():
        reads.append(1)
        return BIG

    with pytest.raises(REFUSE):
        cv.load_floating_lifecycle_vegetation_checkpoint_cuda(saved.path, lambda *a, **k: made.append(1), IDENT, saved.manifest_sha256, leaf_package_sha256="0" * 64, external_helpers=[], original_topology=ns.original,
                                                              expected_device_id=0, available_bytes_fn=reader, host_available_bytes_fn=reader, memory_budget_bytes=BIG, host_memory_budget_bytes=BIG)
    assert reads == [] and made == []


def test_the_cuda_save_refuses_a_cpu_head_before_any_io(tmp_path, leaf):
    ns = boundary_vns(tmp_path / "a", leaf, "numpy", False)
    with pytest.raises(REFUSE):
        cv.save_floating_lifecycle_vegetation_checkpoint_cuda(tmp_path / "x", ns.lin, ns.factory, IDENT, original_topology=ns.original, available_bytes_fn=AVAIL, host_available_bytes_fn=AVAIL,
                                                              memory_budget_bytes=BIG, host_memory_budget_bytes=BIG)
    assert not (tmp_path / "x").exists() and not list(tmp_path.glob(".x.partial-*"))
    with pytest.raises(REFUSE, match="BOTH explicit pools|pool"):
        cv.save_floating_lifecycle_vegetation_checkpoint_cuda(tmp_path / "y", ns.lin, ns.factory, IDENT, original_topology=ns.original, available_bytes_fn=AVAIL, host_available_bytes_fn=AVAIL)


def test_the_cpu_restore_factory_refuses_a_meaningless_cuda_quota(tmp_path, leaf):
    from maple_syrup import floating_event_lifecycle as lc
    from maple_syrup import vegetation_lifecycle as vl

    ns = boundary_vns(tmp_path / "a", leaf, "numpy", False)
    saved = save(ns, tmp_path / "ck")
    V = json.loads((saved.path / cpuv.MANIFEST_NAME).read_text())["vegetation"]
    orig = lc.stage_restored_head
    seen = []

    def spy(token, kind, head_rec, hold_rec=None, **kw):  # the token is still unconsumed: the CPU factory refuses the meaningless CUDA argument
        with pytest.raises(vl.VegetationLifecycleError, match="cuda_quota applies only to a CUDA restore"):
            vl.restore_vegetation_record(token, generation=V["generation"], recipe_id=V["recipe_id"], parent_recipe_id=V["parent_recipe_id"], applied=dict(V["applied"]), max_bytes=BIG, cuda_quota={})
        seen.append(1)
        return orig(token, kind, head_rec, hold_rec, **kw)

    lc.stage_restored_head = spy
    try:
        load(ns, saved.path, saved)
    finally:
        lc.stage_restored_head = orig
    assert seen == [1]
