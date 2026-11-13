"""Common CUDA Python surface unchanged (AST equality of every function the slice did not touch), cold import without CuPy/GPU, compile-flag/cache/provenance behaviour through a PLUMBING DOUBLE of CuPy (no GPU,
no NVRTC: not a hardware claim), identical kernel launch arguments and ledger for the common runner, and the no-source-duplication structure of the generated source. Author-written."""
from __future__ import annotations

import ast
import hashlib
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import NEW_PATH, OLD_PATH

from maple_syrup import subsurface as ss

UNTOUCHED = ("require_cuda", "current_device_id", "effective_free_device_bytes", "default_device_available_bytes_fn", "_launch", "prepare_runner", "load_kernels", "kernel_source")


def functions(path: Path) -> dict:
    tree = ast.parse(path.read_text())
    return {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)}


def test_untouched_python_functions_are_ast_identical_to_the_pre_a41_module(old):
    a, b = functions(OLD_PATH), functions(NEW_PATH)
    for name in UNTOUCHED:
        assert a[name] == b[name], name
    assert set(b) - set(a) == {"render_body_source", "render_source"} and set(a) - set(b) == set()  # the only new functions are the two pure renderers (S2-A.4.2 replaced the dormant registry by `render_body_source`)
    assert a["kernel_provenance"] != b["kernel_provenance"]  # openly changed (additive keys + new source hash)


def test_module_constants_and_public_names_are_unchanged(old, new):
    for name in ("KERNEL_NAME", "FACTOR_KERNEL_NAME", "THREADS", "LAUNCHES_PER_STEP", "COMPILE_OPTIONS"):
        assert getattr(old, name) == getattr(new, name), name
    assert set(old.__all__) <= set(new.__all__) and set(new.__all__) - set(old.__all__) == {"BODY_MACROS", "render_body_source", "render_source"}
    assert new.kernel_source() == new._SOURCE == new.render_source("common") and new._SOURCE != old._SOURCE


def test_provenance_keeps_every_old_value_except_the_source_hash_and_adds_the_part_hashes(old, new):
    a, b = old.kernel_provenance(), new.kernel_provenance()
    for k, v in a.items():
        if k != "source_sha256":
            assert b[k] == v, k
    assert b["source_sha256"] == hashlib.sha256(new._SOURCE.encode()).hexdigest() != a["source_sha256"] == hashlib.sha256(old._SOURCE.encode()).hexdigest()
    assert b["source_variant"] == "common" and b["shared_body_sha256"] == hashlib.sha256(new._BODY.encode()).hexdigest()
    assert set(b) - set(a) == {"source_variant", "shared_body_sha256", "common_readers_sha256", "source_structure"}


def test_cold_import_touches_no_cupy_no_device_and_compiles_nothing():
    code = ("import sys; import maple_syrup.subsurface_cuda as c; "
            "assert 'cupy' not in sys.modules, 'cupy imported'; assert c._MODULES == {} and c._FUNCTIONS == {}; "
            "assert c.kernel_source() and c.render_source('common'); print('ok')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""})
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


class FakeFn:
    def __init__(self, name):
        self.name = name
        self.attributes = {"max_threads_per_block": 1024}


class FakeModule:
    made: list = []

    def __init__(self, code, options, backend):
        self.code, self.options, self.backend = code, options, backend
        FakeModule.made.append(self)

    def get_function(self, name):
        return FakeFn(name)


class FakeCupy:
    RawModule = FakeModule


def test_load_kernels_compiles_the_rendered_source_once_per_device_with_the_unchanged_flags(new):
    with new._LOCK:
        saved = (dict(new._MODULES), dict(new._FUNCTIONS))
        new._MODULES.clear()
        new._FUNCTIONS.clear()
    FakeModule.made.clear()
    try:
        f1 = new.load_kernels(FakeCupy, 0)
        f2 = new.load_kernels(FakeCupy, 0)
        assert all(a is b for a, b in zip(f1, f2)) and len(FakeModule.made) == 1  # cached per device (the same function objects)
        m = FakeModule.made[0]
        assert m.code == new.render_source("common") and m.options == new.COMPILE_OPTIONS == ("--fmad=false", "--prec-div=true", "--prec-sqrt=true", "--ftz=false") and m.backend == "nvrtc"
        assert (f1[0].name, f1[1].name) == (new.FACTOR_KERNEL_NAME, new.KERNEL_NAME)
        new.load_kernels(FakeCupy, 1)
        assert len(FakeModule.made) == 2 and set(new._FUNCTIONS) == {0, 1}  # one entry per device, functions only (no array, owner or state)
    finally:
        new._MODULES.clear()
        new._FUNCTIONS.clear()
        new._MODULES.update(saved[0])
        new._FUNCTIONS.update(saved[1])


class FakeKernel:
    def __init__(self):
        self.calls = []

    def __call__(self, grid, block, args):
        self.calls.append((grid, block, [(type(a).__name__, a.dtype.str if hasattr(a, "dtype") else None, getattr(a, "shape", None), id(a)) for a in args]))


class FakeCp:
    """Plumbing double: numpy-backed `empty` / `asarray` / `ascontiguousarray` (no GPU, no CuPy)."""

    def __init__(self):
        self.allocated = []

    def empty(self, shape, dtype=None):
        a = np.empty(shape, dtype=dtype)
        self.allocated.append(tuple(np.atleast_1d(shape)) if not isinstance(shape, int) else (shape,))
        return a

    def asarray(self, a):
        return np.asarray(a)

    def ascontiguousarray(self, a):
        return np.ascontiguousarray(a)


def layers(mode):
    edges = np.array([0.0, 0.1, 0.25, 0.5])
    kw = dict(diffusivity_m2_s=np.array([2e-7, 4e-7])) if mode == "constant" else dict(interface_conductivity_m_s=np.array([1e-6, 2e-6]), diffusion_coefficient=0.25)
    return ss.soil_layers(edges, theta_residual=0.02, theta_field_capacity=0.2, theta_saturation=0.4, theta_evaporation_stress=0.5, diffusion_mode=mode, **kw)


@pytest.mark.parametrize("mode", ["constant", "tietjen_donor_depth"])
@pytest.mark.parametrize("pulse", [None, 1e-3])
def test_the_common_runner_launches_the_same_arguments_and_allocations_and_reports_the_same_ledger(old, new, monkeypatch, mode, pulse):
    lay = layers(mode)
    results = []
    for mod in (old, new):
        monkeypatch.setattr(mod, "current_device_id", lambda cp: 0)
        monkeypatch.setattr(mod, "load_kernels", lambda cp, dev: (FakeKernel(), FakeKernel()))
        cp = FakeCp()
        run, info, preflight = mod.prepare_runner(lay, cp, 0)
        factor_fn, step_fn = None, None
        # capture the kernels the runner closes over through a second runner call with recording kernels
        fk, sk = FakeKernel(), FakeKernel()
        monkeypatch.setattr(mod, "load_kernels", lambda cp_, dev, fk=fk, sk=sk: (fk, sk))
        run, info, preflight = mod.prepare_runner(lay, cp, 0)
        n = 7
        w = np.zeros((3, n)); add = np.zeros((3, n)); pet = np.zeros(n); act = np.ones(n, np.uint8)

        class Col:
            active_u8 = act

        cp.allocated.clear()
        arrays = run(Col, w, None if pulse else add, pet, 3600.0, None, pulse)
        results.append((fk.calls, sk.calls, list(cp.allocated), {k: v for k, v in info.items() if k != "provenance"}, arrays.scratch is not None, len(arrays.scratch)))
    (fa, sa, aa, ia, ha, na), (fb, sb, ab, ib, hb, nb) = results

    def shape_only(calls):  # ids differ between two runs; compare grid/block/types/dtypes/shapes
        return [(g, b, [x[:3] for x in args]) for g, b, args in calls]

    assert shape_only(fa) == shape_only(fb) and shape_only(sa) == shape_only(sb) and aa == ab and ia == ib and (ha, na) == (hb, nb)
    assert len(fb) == 1 and len(sb) == 1  # factor kernel + column kernel: two launches for three layers


# ---- no-source-duplication structure --------------------------------------------------------------------------------------------------------
STAGE_MARKERS = ("// addition with saturation return", "// gravity cascade, top to bottom", "// evaporation from layer 0", "// sequential pair diffusion from the top", "// output checks",
                 "// fit test right after the ONE movement", "closed bottom: deeper pore capacity")


def test_each_physical_stage_and_the_exchange_expression_occur_exactly_once_in_the_rendered_source(new):
    src = new._SOURCE
    for marker in STAGE_MARKERS:
        assert src.count(marker) == 1, marker
    assert src.count("return -expm1(-x);") == 1 and src.count("relax_factor(") == 3  # one expression (comments aside); one definition + the downward and the upward factor call
    assert src.count("maple_syrup_subsurface_step(") == 1 and src.count("maple_syrup_subsurface_factors(") == 1
    assert src.count(new._BODY) == 1 and new._SOURCE.endswith(new._BODY)


def test_the_shared_body_reads_every_parameter_through_the_named_macros(new):
    body = new._BODY
    for raw in (r"\bsat_m\[", r"\bcap_tol\[", r"\bres_m\[", r"\bfc_m\[", r"\bdz\[", r"\bpair_inv\[", r"\bfactor\[", r"\bfactor_up\[", r"\bres_theta0\b", r"\brows\["):
        assert re.search(raw, body) is None, raw  # no direct reader/epilogue left in the body
    readers = new._COMMON_READERS
    for macro in new.BODY_MACROS:
        assert f"#define {macro}" in readers and re.search(rf"\b{macro}\b", body), macro
    assert body.lstrip().startswith("GUARD_INVALID_LANE") and "return;" not in body  # the invalid-lane guard is a hook; the common definition is the early return
    assert "#define GUARD_INVALID_LANE if (i >= n) return;" in readers


def test_rendering_is_pure_stateless_and_never_copies_the_body(new):
    """S2-A.4.2: the dormant `register_variant` registry is gone; `render_body_source` is a pure function of its arguments and the trusted body."""
    assert not hasattr(new, "register_variant") and not hasattr(new, "_VARIANTS")
    with pytest.raises(ss.SubsurfaceError, match="no CUDA source variant 'spatial'"):
        new.render_source("spatial")
    with pytest.raises(ss.SubsurfaceError, match="do not define the body macros"):
        new.render_body_source("#define DZ(l) dz[l]\n", "head")
    with pytest.raises(ss.SubsurfaceError, match="must be strings"):
        new.render_body_source(None, "head")
    probe = new.render_body_source("".join(f"#define {m}\n" for m in new.BODY_MACROS), "HEAD\n", "TAIL\n")
    assert probe.endswith(new._BODY + "TAIL\n") and probe.count(new._BODY) == 1 and "HEAD" in probe
    assert new.render_body_source(new._COMMON_READERS, new._COMMON_STEP_HEAD) == new._SOURCE == new.render_source("common")  # pure and repeatable; nothing was stored by the probe
