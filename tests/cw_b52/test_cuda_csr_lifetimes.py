"""CW-B52: CUDA CSR step temporary lifetimes, against the frozen B46 operator.

The numerical oracle is the ORIGINAL B46 `characteristic_cuda_csr.py`, read from the frozen B46 runtime tree (hash
pinned to `cw_b52_baseline_manifest.json`) and executed under a private module name. It runs in the CURRENT accepted
combined dependency context: its `maple_syrup` imports resolve to this tree, whose every other package module is
byte-identical to the accepted immutable B55 parent (`cw_b55_joint_runtime_frozen`, hard-pinned
`cw_b55_joint_frozen_manifest.json`; asserted against both the manifest and the parent's actual bytes). B55 differs
from B46 outside the CUDA file (E7 + B51 + the dependency helper), so an oracle call is the B46 CUDA operator in the
B55 context, not a full B46 step (CW-B59 correction 1; previously this asserted B46 identity, which no longer holds).

* CPU tests (run by the author): oracle pin, private-tree identity, the change confined to `cuda_sparse_step` with
  unchanged kernels/compile options, and a static last-use structure check.
* DEVICE tests (actual CuPy on a CUDA device; skipped here, root runs them on GPU0): raw-byte differential of every
  output (transport, phase, remaining/arrival/crossing/before, scalars, integer controls, full device report) over
  dry / signed-zero / wet / stored-bin / terminal / masked / multi-substep cases, identical refusals and their
  ordering, caller immutability, capacity-before-write, and a CuPy MemoryHook live-byte timeline at every kernel
  launch with the exact expected releases. A skip claims nothing.

NNZ after the first substep cannot increase (correction 2; verified in BOTH kernels, not assumed). The pickup packet
is injected only in substep 0 (`packet = Pc if k == 0 else zero packet`). In `_source_row` / `_row_candidates`
(Numba) and the CUDA count/write candidate code, a packet with `m == 0` is skipped; a packet settled at its source
neither moves nor arrives; a non-crossing packet becomes at most ONE candidate in its own row only; a crossing packet
at most ONE candidate in its single receiver row only (none when exported or settled on arrival); `mnew <= 0` adds
nothing. The count pass merges candidates per (row, bin). Hence nnz(substep k) <= number of nonzero packets of
substep k, which for k >= 1 is nnz(substep k-1). The peak is substep 0's requirement, and a static cap that fits
substep 0 fits every later substep: a later-substep capacity refusal is unreachable in this scheme, so no test
asserts one.
"""
from __future__ import annotations

import ast
import dataclasses
import difflib
import hashlib
import inspect
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("maple")

from maple.core.backend import (
    freeze,
    gpu_execution_available,
    to_device,
    to_host,
)

from maple_syrup import characteristic_cuda_csr as cuda_csr
from maple_syrup.characteristic_sparse import (
    SparseCapacityError,
    empty_sparse_phase_state,
)
from maple_syrup.routing import PIT_STORAGE, build_routing_graph
from maple_syrup.sediment_transport import (
    TransportError,
    TransportNetwork,
    transport_network,
)

PRIVATE_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_PKG = PRIVATE_ROOT / "src" / "maple_syrup"
B46_PKG = Path("/home/okin/SYRUP/outputs/conservative_water/cw_b46_runtime_frozen/src/maple_syrup")
B46_CUDA_CSR_SHA256 = "c32b3ba19b31e22cf8a225a9af0d27ea4fe0b590dd4cebc9c2afcb7c9404f468"  # cw_b52_baseline_manifest
B55_ROOT = Path("/home/okin/SYRUP/outputs/conservative_water/cw_b55_joint_runtime_frozen")
B55_MANIFEST = Path("/home/okin/SYRUP/agent_handoffs/tasks/conservative_water/cw_b55_joint_frozen_manifest.json")
B55_MANIFEST_SHA256 = "8028b6184e1d2fe797f00885bd99f06e3fc175163feca333edc2983d0392761a"
M44_MAPLE_DIRS = (  # the accepted M44 copy and its CW-E6 materialized cache (same package digest 44e4f4c6...)
    Path("/home/okin/SYRUP/outputs/dependencies/maple_cw_b44_lifetime_frozen/source/src/maple"),
    Path("/home/okin/SYRUP/outputs/dependencies/content_cache/"
         "43a15ee91d799a893398e08269b1bebf76517f8590b19290a253911488f89673/source/src/maple"),
)
DX = 0.5
HAVE_GPU = gpu_execution_available()
needs_gpu = pytest.mark.skipif(not HAVE_GPU, reason="CuPy with a CUDA device is not available here; the CUDA CSR "
                                                     "differential is NOT exercised and no device claim is made")
cp = None
if HAVE_GPU:
    import cupy as cp


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_oracle():
    path = B46_PKG / "characteristic_cuda_csr.py"
    if not path.is_file():
        pytest.skip("frozen B46 oracle unavailable; no before/after claim")
    source = path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == B46_CUDA_CSR_SHA256, "B46 oracle differs from the baseline manifest"
    name = "_cw_b52_b46_characteristic_cuda_csr"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102 -- the hash-pinned B46 oracle text
    return module


@pytest.fixture(scope="module")
def oracle():
    return _load_oracle()


# =============================================================================================================
# CPU: provenance, scope and static lifetime structure
# =============================================================================================================
def test_modules_resolve_to_private_tree_and_oracle_is_pinned_b46(oracle):
    assert Path(cuda_csr.__file__).resolve() == PRIVATE_PKG / "characteristic_cuda_csr.py"
    assert Path(oracle.__file__) == B46_PKG / "characteristic_cuda_csr.py"
    assert oracle.cuda_sparse_step is not cuda_csr.cuda_sparse_step


def test_every_other_package_module_is_byte_identical_to_the_accepted_b55_parent():
    """CW-B59 correction 1: the combined context's parent is the accepted immutable B55 tree, not B46. Every package
    file name and hash except `characteristic_cuda_csr.py` must equal the hard-pinned B55 manifest AND the parent's
    actual bytes; the candidate CUDA file, where it differs from B55, may differ only inside `cuda_sparse_step`
    (the root tree equals B55 and is not required to differ)."""
    import json

    raw = B55_MANIFEST.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == B55_MANIFEST_SHA256, "B55 manifest differs from its pin"
    manifest = json.loads(raw)
    prefix = "src/maple_syrup/"
    pinned = {k[len(prefix):]: v for k, v in manifest["files"].items() if k.startswith(prefix)}
    assert len(pinned) == 55 and all(k.endswith(".py") for k in pinned)
    parent = {str(p.relative_to(B55_ROOT / "src" / "maple_syrup")): _sha(p)
              for p in (B55_ROOT / "src" / "maple_syrup").rglob("*.py") if "__pycache__" not in p.parts}
    assert parent == pinned  # the frozen parent's actual bytes are the pinned ones
    private = {str(p.relative_to(PRIVATE_PKG)): _sha(p)
               for p in PRIVATE_PKG.rglob("*.py") if "__pycache__" not in p.parts}
    assert set(private) == set(pinned)
    cuda = "characteristic_cuda_csr.py"
    assert sorted(k for k in private if k != cuda and private[k] != pinned[k]) == []
    assert pinned[cuda] == manifest["cuda_sha256"]
    assert private[cuda] != B46_CUDA_CSR_SHA256  # the B46 numerical oracle is a genuine comparison
    if private[cuda] != pinned[cuda]:  # a candidate layered on B55: changes confined to cuda_sparse_step
        old = (B55_ROOT / "src" / "maple_syrup" / cuda).read_text()
        new = (PRIVATE_PKG / cuda).read_text()
        func = next(n for n in ast.walk(ast.parse(old)) if isinstance(n, ast.FunctionDef) and n.name == "cuda_sparse_step")
        changed = 0
        for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(a=old.splitlines(), b=new.splitlines(),
                                                             autojunk=False).get_opcodes():
            if tag != "equal":
                changed += 1
                assert func.lineno <= i1 + 1 and i2 <= func.end_lineno, (tag, i1, i2)
        assert changed > 0


def test_change_is_confined_to_cuda_sparse_step_with_unchanged_kernels(oracle):
    old = (B46_PKG / "characteristic_cuda_csr.py").read_text().splitlines()
    new = (PRIVATE_PKG / "characteristic_cuda_csr.py").read_text().splitlines()
    lines, first = inspect.getsourcelines(oracle.cuda_sparse_step)
    lo, hi = first, first + len(lines) - 1  # 1-based inclusive range of the B46 function
    for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes():
        if tag != "equal":
            assert lo <= i1 + 1 and i2 <= hi, (tag, i1, i2, lo, hi)
    assert cuda_csr.COMPILE_OPTIONS == oracle.COMPILE_OPTIONS
    assert cuda_csr.KERNEL_NAMES == oracle.KERNEL_NAMES
    for nb in (1, 8, 32, 100, 128):
        assert cuda_csr.kernel_source(nb) == oracle.kernel_source(nb)
        assert cuda_csr.estimate_local_scratch_bytes(nb) == oracle.estimate_local_scratch_bytes(nb)


def _body_lines():
    src = inspect.getsource(cuda_csr.cuda_sparse_step)
    return src, [ln.strip() for ln in src.splitlines()]


def _index(lines, text, start=0):
    for i in range(start, len(lines)):
        if lines[i].startswith(text):
            return i
    raise AssertionError(f"{text!r} not found after line {start}")


def test_static_last_use_structure():
    """The released names are deleted at their last use, before the substep allocations; no zero accumulators or
    one-substep zero packet are allocated; counts dies before the write arrays; arithmetic lines are the in-place
    forms of the original sums. (Structure only: the device tests prove values and live bytes.)"""
    src, lines = _body_lines()
    ast.parse(textwrap.dedent(src))  # still a valid function
    resolve = _index(lines, "checks.resolve()")
    release = _index(lines, "del a, exponent, inactive3", resolve)
    first_alloc = _index(lines, "Pc = cp.ascontiguousarray(P[idx])", release)
    assert release < first_alloc
    masses = _index(lines, "masses = Mc.reshape(-1)", first_alloc)
    assert lines[masses + 1].startswith("del Mc")
    assert "acc = {name: cp.zeros" not in src and "acc = None" in src
    loop = _index(lines, "for k in range(n_sub):")
    zero_packet = _index(lines, "no_pickup = cp.zeros", loop)
    assert lines[zero_packet - 1].startswith("del Pc") and lines[zero_packet - 2].startswith("packet = None")
    assert any(lines[i].startswith("if k == 1:") for i in range(zero_packet - 5, zero_packet))
    assert src.count("no_pickup = cp.zeros") == 1
    cumsum = _index(lines, "cp.cumsum(counts, out=row_ptr_new[1:])", loop)
    assert lines[cumsum + 1].startswith("del counts")
    assert cumsum < _index(lines, "bins_new = cp.empty(nnz_new", cumsum)
    for text in ("cp.add(d_decay, d_arr_decay, out=d_decay)", "cp.add(d_settled, d_arr_settled, out=d_settled)",
                 "cp.add(d_arr_decay, d_arr_settled, out=d_arr_decay)", "value += 0.0", "acc[name] += value"):
        assert text in src, text
    # d_arr_decay is overwritten only after its two other reads
    assert src.index("cp.add(d_decay, d_arr_decay") < src.index("cp.add(d_arr_decay, d_arr_settled")
    assert "fastmath" not in src


@dataclasses.dataclass(frozen=True, eq=False)
class _Probe:
    data: object
    xp: object
    nested: object
    scalar: float
    count: object


class _ArrayLike:
    """Neither a module nor an ndarray, but exposing dtype/shape: must be refused, never guessed."""

    dtype = np.dtype(np.float64)
    shape = (1,)


def test_comparator_classifies_modules_before_arrays_with_real_numpy():
    """Correction 1 regression (CPU, real NumPy). The numpy MODULE has `dtype` and `shape` exactly like the cupy
    module that broke root's GPU run; it must be recorded as a module while actual arrays keep raw bytes, dtype,
    shape, strides and contiguity. A CuPy-like module stand-in (a real ModuleType with dtype/shape) is classified
    the same way. Signed zeros, NumPy scalars and Python floats stay exact."""
    assert hasattr(np, "dtype") and hasattr(np, "shape")  # the trap
    fake_cupy = types.ModuleType("cupy_like")
    fake_cupy.dtype, fake_cupy.shape = np.dtype, np.shape
    base = np.arange(24, dtype=np.float64).reshape(4, 6)
    view = base[:, ::2]  # non-contiguous
    signed = np.array([0.0, -0.0])
    probe = _Probe(data=view, xp=np, nested={"a": [signed, (fake_cupy, 1)], "b": np.float64(-0.0)}, scalar=-0.0,
                   count=np.int64(7))
    rec = value_rec(probe)
    assert rec[0] == "_Probe"
    fields = dict(rec[1])
    assert fields["xp"] == ("module", "numpy")
    assert fields["data"] == ("numpy", view.tobytes(), "float64", (4, 3), view.strides, False)
    nested = dict(fields["nested"][1])
    assert nested["a"] == ("list", (arr_rec(signed), ("tuple", (("module", "cupy_like"), 1))))
    assert arr_rec(signed)[1] == signed.tobytes() != np.zeros(2).tobytes()  # -0.0 is not +0.0
    assert nested["b"] == ("numpy_scalar", "float64", np.float64(-0.0).tobytes())
    assert fields["scalar"] == ("float", (-0.0).hex()) != ("float", (0.0).hex())
    assert fields["count"] == ("numpy_scalar", "int64", np.int64(7).tobytes())
    # differences are detected, not normalised away
    assert value_rec(dataclasses.replace(probe, data=view.copy())) != rec  # same values, other strides/contiguity
    assert value_rec(dataclasses.replace(probe, scalar=0.0)) != rec
    assert value_rec(dataclasses.replace(probe, nested={"a": [np.zeros(2), (fake_cupy, 1)], "b": np.float64(-0.0)})) != rec
    with pytest.raises(TypeError, match="unclassified"):
        value_rec(_ArrayLike())
    with pytest.raises(TypeError, match="ndarray"):
        arr_rec(np)
    with pytest.raises(TypeError, match="ndarray"):
        arr_rec(fake_cupy)


# =============================================================================================================
# Device helpers
# =============================================================================================================
def chain_full(n, dz=0.015625, walls=1.0):
    z = np.repeat(np.arange(n + 2, dtype=np.float64)[:, None] * dz, 3, axis=1)
    z[:, 0] += walls
    z[:, 2] += walls
    return z


def random_full(rng, ny, nx):
    rows = np.indices((ny + 2, nx + 2))[0].astype(np.float64)
    z = 0.025 * rows + rng.uniform(0.0, 0.02, rows.shape)  # every interior cell drains to the row below
    z[:, 0] += 10.0
    z[:, -1] += 10.0
    return z


def south_export(z):
    export = np.zeros(z.shape, dtype=bool)
    export[0, :] = True
    return export


def device_network(z, *, active=None):
    ny, nx = z.shape[0] - 2, z.shape[1] - 2
    return transport_network(build_routing_graph(z, south_export(z), np.full((ny, nx), 1.0), DX, xp=cp,
                                                 active_mask=active))


@dataclasses.dataclass(frozen=True, eq=False)
class TerminalNetwork(TransportNetwork):
    terminal_host: object = None
    terminal_flat: object = None


def terminal_network(base, pit_cells):
    """`base` with `pit_cells` as declared terminal storage (the tests/conservative_csr C2 fixture, valid form)."""
    xp = base.xp
    pits = np.asarray(pit_cells, dtype=np.int64)
    receiver_host = np.array(base.receiver_host, copy=True)
    receiver_host.reshape(-1)[pits] = PIT_STORAGE
    receiver_index = np.array(to_host(base.receiver_index), copy=True).reshape(-1)
    receiver_index[pits] = pits
    terminal_host = np.zeros(base.shape, dtype=np.bool_)
    terminal_host.reshape(-1)[pits] = True
    fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(TransportNetwork)}
    for cells_name, companions in (("cells_x", ("rows_x", "faces_x", "sign_x")),
                                   ("cells_y", ("faces_y", "cols_y", "sign_y"))):
        cells = to_host(getattr(base, cells_name))
        keep = ~np.isin(cells, pits)
        fields[cells_name] = to_device(np.ascontiguousarray(cells[keep]), xp)
        for name in companions:
            fields[name] = to_device(np.ascontiguousarray(to_host(getattr(base, name))[keep]), xp)
    fields["receiver_index"] = to_device(np.ascontiguousarray(receiver_index), xp)
    fields["receiver_host"] = freeze(receiver_host)
    fields["graph_input_sha256"] = base.graph_input_sha256 + ":terminal"
    fields.update({"terminal_host": freeze(terminal_host), "terminal_flat": to_device(terminal_host.reshape(-1).copy(), xp),
                   "n_terminal": int(pits.size)})
    return TerminalNetwork(**fields)


def is_ndarray(v) -> bool:
    """Structural proof of an array: an actual `numpy.ndarray` or `cupy.ndarray` instance. Duck typing on
    `dtype`/`shape` is NOT used: the `numpy` and `cupy` MODULES have both attributes (correction 1)."""
    if isinstance(v, np.ndarray):
        return True
    cupy = sys.modules.get("cupy")
    return cupy is not None and isinstance(v, cupy.ndarray)


def arr_rec(a):
    if not is_ndarray(a):
        raise TypeError(f"arr_rec needs a numpy/cupy ndarray, got {type(a).__module__}.{type(a).__name__}")
    h = to_host(a)
    return (type(a).__module__.split(".")[0], h.tobytes(), str(h.dtype), h.shape, tuple(a.strides),
            bool(a.flags.c_contiguous))


def value_rec(v):
    """Comparable record of a result tree. Modules are classified FIRST (by type); arrays only by actual ndarray type
    (raw bytes, dtype, shape, strides, contiguity, array module); containers recursively; floats exactly (hex). Any
    other object exposing `dtype`/`shape` is refused rather than guessed."""
    if isinstance(v, types.ModuleType):
        return ("module", v.__name__)
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return (type(v).__name__, tuple((f.name, value_rec(getattr(v, f.name))) for f in dataclasses.fields(v)))
    if is_ndarray(v):
        return arr_rec(v)
    if isinstance(v, np.generic):  # NumPy scalars also expose dtype/shape: exact dtype and raw bytes
        return ("numpy_scalar", str(v.dtype), v.tobytes())
    if hasattr(v, "dtype") and hasattr(v, "shape"):
        raise TypeError(f"unclassified array-like {type(v).__module__}.{type(v).__name__}; refusing to guess")
    if isinstance(v, dict):
        return ("dict", tuple(sorted((k, value_rec(x)) for k, x in v.items())))
    if isinstance(v, (list, tuple)):
        return (type(v).__name__, tuple(value_rec(x) for x in v))
    if isinstance(v, float):
        return ("float", v.hex())
    return v


def outcome(module, net, mobile, phase, pickup, v, rate, settle, dt, **kwargs):
    try:
        step = module.cuda_sparse_step(net, mobile, phase, pickup, v, rate, settle, dt, **kwargs)
    except (TransportError, ValueError) as exc:
        extra = tuple(getattr(exc, k, None) for k in ("required_nnz", "max_nnz", "substep", "n_substeps"))
        return ("refused", type(exc).__name__, str(exc), extra), None
    return ("ok", value_rec(step)), step


def record_substep_nnz(module, net, mobile, phase, pickup, v, rate, settle, dt, **kw):
    """Per-substep stored-bin counts of ONE CUDA step, observed as the host-side `.shape` of each substep's uint8 bin
    write allocation (`cp.empty(nnz_new, uint8)`), i.e. metadata already on the host. No kernel, device read,
    transfer counter or object lifetime of the operator is changed. Restores `cp.empty` on every exit."""
    sizes = []
    real_empty = cp.empty

    def recording(shape, dtype=float, *args, **kwargs):
        if np.dtype(dtype) == np.uint8:
            sizes.append(int(np.prod(np.atleast_1d(shape))))
        return real_empty(shape, dtype, *args, **kwargs)

    cp.empty = recording
    try:
        module.cuda_sparse_step(net, mobile, phase, pickup, v, rate, settle, dt, **kw)
    finally:
        cp.empty = real_empty
    return sizes


def caller_rec(*arrays, phase=None):
    out = [arr_rec(a) for a in arrays]
    if phase is not None:
        out += [arr_rec(getattr(phase, n)) for n in ("row_ptr", "bin", "fraction", "position_m")]
    return out


CASES = ("dry_empty", "signed_zero", "wet_fresh", "wet_stored_bins", "terminal", "masked")


def build_case(case, nc, nb, seed=11):
    rng = np.random.default_rng(seed)
    active = None
    if case == "terminal":
        net = terminal_network(device_network(chain_full(6)), [3])
    elif case == "masked":
        active = np.ones((10, 5), dtype=bool)
        active[7:, :] = False  # top rows: nothing drains into them
        net = device_network(random_full(rng, 10, 5), active=active)
    else:
        net = device_network(random_full(rng, 8, 5))
    ny, nx = net.shape
    shape = (ny, nx, nc)
    pickup = rng.uniform(0.0, 1.0, shape) * (rng.uniform(size=shape) < 0.6)
    v = rng.uniform(0.0, 0.2, shape)
    rate = rng.uniform(0.0, 4.0, shape)
    settle = rng.uniform(size=shape) < 0.15
    mobile = np.zeros(shape)
    if case == "dry_empty":
        pickup[:] = 0.0
    if case == "signed_zero":
        pickup = np.where(rng.uniform(size=shape) < 0.5, -0.0, pickup)
        v = np.where(rng.uniform(size=shape) < 0.3, -0.0, v)
        rate = np.where(rng.uniform(size=shape) < 0.3, -0.0, rate)
        mobile = np.full(shape, -0.0)
    if case == "terminal":
        settle[3, 0, :] = True  # every class settles at the terminal (C2 rule)
    if active is not None:
        inactive = ~active
        for a in (pickup, v, mobile):
            a[inactive] = 0.0
    dev = [cp.asarray(a) for a in (mobile, pickup, v, rate, settle)]
    phase = empty_sparse_phase_state(net, nc, nb)
    return net, dev, phase


def assert_same_outcome(oracle, net, mobile, phase, pickup, v, rate, settle, dt, **kw):
    before = caller_rec(mobile, pickup, v, rate, settle, phase=phase)
    new, new_step = outcome(cuda_csr, net, mobile, phase, pickup, v, rate, settle, dt, **kw)
    old, old_step = outcome(oracle, net, mobile, phase, pickup, v, rate, settle, dt, **kw)
    assert new == old
    assert caller_rec(mobile, pickup, v, rate, settle, phase=phase) == before  # caller bytes untouched by both
    return new, new_step, old_step


# =============================================================================================================
# Device: raw-byte differential against B46
# =============================================================================================================
@needs_gpu
def test_device_and_maple_resolution():
    import maple

    assert Path(maple.__file__).resolve().parent in tuple(p.resolve() for p in M44_MAPLE_DIRS), maple.__file__
    net, dev, phase = build_case("wet_fresh", 3, 8)
    step = cuda_csr.cuda_sparse_step(net, *dev[:1], phase, *dev[1:], 1.0)
    device = cp.cuda.Device().id
    for array in (step.transport.mobile_after_transfer_kg, step.phase_after.fraction, step.mobile_remaining_kg,
                  step.transport.x_face_net_kg):
        assert type(array) is cp.ndarray and array.device.id == device
    assert step.implementation == "cuda" and step.device_report["device_id"] == device


@needs_gpu
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("n_substeps", [1, 2, 3])
@pytest.mark.parametrize("nb", [1, 8, 32])
@pytest.mark.parametrize("nc", [1, 6])
def test_raw_byte_differential_against_b46_two_chained_steps(oracle, case, n_substeps, nb, nc):
    net, (mobile, pickup, v, rate, settle), phase = build_case(case, nc, nb)
    rec, step, ref = assert_same_outcome(oracle, net, mobile, phase, pickup, v, rate, settle, 1.0,
                                         n_substeps=n_substeps, max_nnz=1 << 20)
    assert rec[0] == "ok", rec
    if case == "wet_stored_bins" or case in ("terminal", "masked", "wet_fresh"):
        # continue from EACH module's own published state (stored bins, nonzero pool), fresh pickup
        rec2, step2, _ = assert_same_outcome(oracle, net, step.mobile_remaining_kg, step.phase_after, pickup * 0.5,
                                             v, rate, settle, 1.0, n_substeps=n_substeps, max_nnz=1 << 20)
        assert rec2[0] == "ok"
        _rec3, step3, _ = assert_same_outcome(oracle, net, ref.mobile_remaining_kg, ref.phase_after, pickup * 0.5,
                                              v, rate, settle, 1.0, n_substeps=n_substeps, max_nnz=1 << 20)
        assert value_rec(step3) == value_rec(step2)
        if case == "wet_stored_bins":
            assert step.nnz_after > 0
    if case == "dry_empty":
        assert step.nnz_after == 0


@needs_gpu
def test_identical_refusals_their_ordering_and_caller_immutability(oracle):
    nc, nb = 2, 8
    net, (mobile, pickup, v, rate, settle), phase = build_case("wet_fresh", nc, nb)
    _ok, step, _ = assert_same_outcome(oracle, net, mobile, phase, pickup, v, rate, settle, 1.0)
    base_mobile, base_phase = step.mobile_remaining_kg, step.phase_after
    nan = cp.where(cp.arange(nc) == 0, cp.nan, pickup)
    fast = v * 0.0 + 0.3  # Courant 0.6 > 0.5
    cases = {
        "negative pickup": {"pickup": -pickup - 1.0},
        "nonfinite rate": {"rate": rate * cp.inf},
        "courant": {"v": fast},
        "nan pickup AND courant (ordering)": {"pickup": nan, "v": fast},
        "nonfinite velocity": {"v": v * cp.nan},
        "zero pool with stored bins": {"mobile": base_mobile * 0.0},
        "dt": {"dt": 0.0},
        "substeps": {"n_substeps": 0},
        "capacity": {"max_nnz": 1},
    }
    # Correction 2: a refusal in a LATER substep cannot exist in the current scheme (module docstring "NNZ after
    # the first substep cannot increase"; verified in both kernels' candidate generation). The capacity contract
    # tested here is the reachable one: a multi-substep step whose cap is below substep 0's requirement is refused at
    # substep index 0 with n_substeps reported; a cap EQUAL to that requirement passes every later substep
    # (test_cap_equal_to_first_substep_requirement_passes_multi_substep_on_device).
    first_required = record_substep_nnz(cuda_csr, net, base_mobile, base_phase, pickup, v, rate, settle, 1.0,
                                        n_substeps=2)[0]
    assert first_required >= 2, "fixture must store at least two bins after substep 0"
    cases["capacity in substep 0 of 2"] = {"n_substeps": 2, "max_nnz": first_required - 1}
    for label, change in cases.items():
        args = {"mobile": base_mobile, "phase": base_phase, "pickup": pickup, "v": v, "rate": rate, "settle": settle, "dt": 1.0}
        kw = {}
        for key, value in change.items():
            (args if key in args else kw)[key] = value
        rec, _new, _old = assert_same_outcome(oracle, net, args["mobile"], args["phase"], args["pickup"], args["v"],
                                              args["rate"], args["settle"], args["dt"], **kw)
        assert rec[0] == "refused", (label, rec)
        required, cap, substep, n_sub = rec[3]
        if label == "capacity":
            assert rec[1] == "SparseCapacityError" and substep == 0 and n_sub == 1 and cap == 1 < required, rec
        if label == "capacity in substep 0 of 2":  # refused at the first substep, which carries the peak
            assert rec[1] == "SparseCapacityError" and substep == 0 and n_sub == 2, rec
            assert cap == first_required - 1 and required == first_required, (rec, first_required)
        if label == "courant":
            assert rec[1] == "TransportStepRejected"


@needs_gpu
def test_capacity_refusal_precedes_write_allocation_in_both(oracle, monkeypatch):
    net, (mobile, pickup, v, rate, settle), phase = build_case("wet_fresh", 1, 8)
    requests = {"new": [], "old": []}
    real_empty = cp.empty
    for label, module in (("new", cuda_csr), ("old", oracle)):
        def recording(shape, dtype=float, *args, label=label, **kwargs):
            if np.dtype(dtype) == np.uint8:
                requests[label].append(int(np.prod(np.atleast_1d(shape))))
            return real_empty(shape, dtype, *args, **kwargs)

        monkeypatch.setattr(cp, "empty", recording)
        with pytest.raises(SparseCapacityError):
            module.cuda_sparse_step(net, mobile, phase, pickup, v, rate, settle, 1.0, max_nnz=1)
        monkeypatch.setattr(cp, "empty", real_empty)
    assert requests == {"new": [], "old": []}


# =============================================================================================================
# Device: live-byte timeline (CuPy MemoryHook) with the exact expected releases
# =============================================================================================================
def _r512(nbytes: int) -> int:
    return 0 if nbytes <= 0 else -(-int(nbytes) // 512) * 512


class _LiveBytes:
    """Same-thread CuPy pool allocations live inside the hook (rounded `mem_size`, keyed by pool pointer)."""

    def __init__(self):
        self.live, self.current = {}, 0

    def hook(self):
        tracker = self

        class Hook(cp.cuda.MemoryHook):
            name = "CWB52LiveBytes"

            def malloc_postprocess(self, **kw):
                if kw.get("mem_ptr"):
                    key = (int(kw["device_id"]), int(kw["pmem_id"]))
                    tracker.current += int(kw["mem_size"]) - tracker.live.get(key, 0)
                    tracker.live[key] = int(kw["mem_size"])

            def free_postprocess(self, **kw):
                key = (int(kw["device_id"]), int(kw["pmem_id"]))
                tracker.current -= tracker.live.pop(key, 0)

        return Hook()


def _timeline(module, monkeypatch, *args, **kwargs):
    """Live tracked bytes at every kernel launch of ONE step (launch order), relative to the call's entry."""
    tracker = _LiveBytes()
    events = []
    real = module._function

    def tracked(name, nb):
        kernel = real(name, nb)

        def launch(*a, **k):
            events.append((name, tracker.current))
            return kernel(*a, **k)

        return launch

    real_result = module.TransportStep

    def result(*a, **k):  # the step's result construction: every output and validation reduction is computed
        events.append(("result", tracker.current))
        return real_result(*a, **k)

    module.cuda_sparse_step(*args, **kwargs)  # warm: binding, kernels, row-sum self-check (cached per module)
    cp.cuda.Device().synchronize()
    monkeypatch.setattr(module, "_function", tracked)
    monkeypatch.setattr(module, "TransportStep", result)
    with tracker.hook():
        step = module.cuda_sparse_step(*args, **kwargs)
        cp.cuda.Device().synchronize()
    monkeypatch.setattr(module, "_function", real)
    monkeypatch.setattr(module, "TransportStep", real_result)
    return events, step


@needs_gpu
@pytest.mark.parametrize("n_substeps", [1, 2])
@pytest.mark.parametrize("stored", [False, True])
def test_live_bytes_at_every_kernel_launch_show_the_exact_releases(oracle, monkeypatch, n_substeps, stored):
    """B46 minus CW-B52 live bytes at each launch, with R = r512(n_active nc 8), D = r512(n nc 8), I = r512(n):
      substep 0 count: 6R accumulators + R Mc + R zero packet + 2D (a, exponent) + I (inactive mask)
      substep 0 write: the above + R (counts)
      substep 1 count: R Pc + R Mc + 2D + I            (both now hold 6 accumulators, 7 parts, counts, zero packet)
      substep 1 write: the above + R (counts)
      row-sum normalize: I                             (B46 deleted Mc/Pc/a/exponent/zero packet before it)
      result construction (CW-B54 tree; was a CW-B52 band): EXACTLY 3D (cell scale, tolerance, balance) + 2R (row
        counts, compact remaining) + I + the two gathered face arrays, plus, when nnz > 0, R (fraction sums) +
        2 r512(8 nnz) (row ids, remaining per-bin masses) + r512(nnz - 1) (row-order mask, nnz > 1)
    Exact equality at every kernel launch and at the result; values are compared raw-byte too."""
    nc, nb = 6, 8
    net, (mobile, pickup, v, rate, settle), _phase = build_case("masked", nc, nb)

    def own_state(module):
        # Correction 1: each module gets a phase bound to ITS OWN binding's slot-map arrays. A phase from another
        # module's binding makes `_enqueue_state_values` add two slot-map comparison flags (identity check), which
        # would be a harness-induced live-byte difference, not a lifetime change.
        phase = module.empty_cuda_sparse_phase_state(net, nc, nb)
        mob = mobile
        if stored:
            warm = module.cuda_sparse_step(net, mobile, phase, pickup, v, rate, settle, 1.0)
            mob, phase = warm.mobile_remaining_kg, warm.phase_after
            assert phase.nnz > 0
        assert phase.cell_of_slot is module.bind_network(net).cell_of_slot
        return mob, phase

    new_mobile, new_phase = own_state(cuda_csr)
    old_mobile, old_phase = own_state(oracle)
    assert value_rec(new_phase) == value_rec(old_phase)  # byte-identical start states
    assert arr_rec(new_mobile) == arr_rec(old_mobile)
    binding = cuda_csr.bind_network(net)
    n, n_active = binding.n_cells, binding.n_active
    R, D, Ib = _r512(n_active * nc * 8), _r512(n * nc * 8), _r512(n)
    new_events, new_step = _timeline(cuda_csr, monkeypatch, net, new_mobile, new_phase, pickup, v, rate, settle, 1.0,
                                     n_substeps=n_substeps)
    old_events, old_step = _timeline(oracle, monkeypatch, net, old_mobile, old_phase, pickup, v, rate, settle, 1.0,
                                     n_substeps=n_substeps)
    assert value_rec(new_step) == value_rec(old_step)
    assert [e[0] for e in new_events] == [e[0] for e in old_events]
    expected = [6 * R + 2 * R + 2 * D + Ib, 6 * R + 3 * R + 2 * D + Ib]
    if n_substeps == 2:
        expected += [2 * R + 2 * D + Ib, 3 * R + 2 * D + Ib]
    expected += [Ib]
    saved = [o[1] - w[1] for o, w in zip(old_events, new_events, strict=True)]
    print(f"live-byte savings per launch (R={R}, D={D}, I={Ib}):", list(zip([e[0] for e in new_events], saved)))
    assert [e[0] for e in new_events][-1] == "result"
    assert saved[:-1] == expected, (saved, expected)
    nnz = int(new_step.nnz_after)
    faces = _r512(int(net.cells_x.size) * nc * 8) + _r512(int(net.cells_y.size) * nc * 8)
    # CW-B54: the result-time expectation is exact (this tree also releases the per-cell balance and the owned
    # remaining per-bin masses before the faces); the original CW-B52 band is archived with the CW-B54 task.
    exact = 3 * D + 2 * R + Ib + faces + ((R + 2 * _r512(8 * nnz) + _r512(nnz - 1 if nnz > 1 else 0)) if nnz else 0)
    assert saved[-1] == exact, (saved[-1], exact)


# =============================================================================================================
# Correction 2: the capacity contract and NNZ monotonicity after the first substep
# =============================================================================================================
@needs_gpu
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("n_substeps", [2, 3, 4])
@pytest.mark.parametrize("nb", [1, 8, 32])
def test_nnz_never_increases_after_the_first_substep_on_device(oracle, case, n_substeps, nb):
    nc = 3
    net, (mobile, pickup, v, rate, settle), phase = build_case(case, nc, nb)
    for state_mobile, state_phase, fresh in ((mobile, phase, pickup), (None, None, pickup * 0.5)):
        if state_mobile is None:  # continue from a stored-bin state (nonzero pool), fresh pickup
            first = cuda_csr.cuda_sparse_step(net, mobile, phase, pickup, v, rate, settle, 1.0)
            state_mobile, state_phase = first.mobile_remaining_kg, first.phase_after
        new = record_substep_nnz(cuda_csr, net, state_mobile, state_phase, fresh, v, rate, settle, 1.0,
                                 n_substeps=n_substeps)
        old = record_substep_nnz(oracle, net, state_mobile, state_phase, fresh, v, rate, settle, 1.0,
                                 n_substeps=n_substeps)
        assert new == old and len(new) == n_substeps
        assert all(b <= a for a, b in zip(new, new[1:])), new  # never grows after substep 0
        compact = to_host(fresh).reshape(-1, nc)[to_host(state_phase.cell_of_slot)]
        assert new[0] <= int(state_phase.nnz) + int(np.count_nonzero(compact)), new  # one candidate per packet
        _rec, step, _ = assert_same_outcome(oracle, net, state_mobile, state_phase, fresh, v, rate, settle, 1.0,
                                            n_substeps=n_substeps)
        assert step.peak_nnz == new[0] and step.nnz_after == new[-1]


@needs_gpu
@pytest.mark.parametrize("n_substeps", [2, 3])
def test_cap_equal_to_first_substep_requirement_passes_multi_substep_on_device(oracle, n_substeps):
    net, (mobile, pickup, v, rate, settle), phase = build_case("wet_fresh", 2, 8)
    base = cuda_csr.cuda_sparse_step(net, mobile, phase, pickup, v, rate, settle, 1.0)
    sizes = record_substep_nnz(cuda_csr, net, base.mobile_remaining_kg, base.phase_after, pickup, v, rate, settle, 1.0,
                               n_substeps=n_substeps)
    first = sizes[0]
    assert first >= 2 and all(b <= a for a, b in zip(sizes, sizes[1:]))
    rec, step, _ = assert_same_outcome(oracle, net, base.mobile_remaining_kg, base.phase_after, pickup, v, rate,
                                       settle, 1.0, n_substeps=n_substeps, max_nnz=first)
    assert rec[0] == "ok" and step.peak_nnz == first
    rec, _s, _o = assert_same_outcome(oracle, net, base.mobile_remaining_kg, base.phase_after, pickup, v, rate, settle,
                                      1.0, n_substeps=n_substeps, max_nnz=first - 1)
    assert rec[0] == "refused" and rec[1] == "SparseCapacityError"
    required, cap, substep, n_sub = rec[3]
    assert (required, cap, substep, n_sub) == (first, first - 1, 0, n_substeps)


# --- CPU proof checks on the host (Numba) operator, same scheme and the same candidate rules ---------------------------
def _host_case(nc=3, nb=8, seed=5):
    from maple_syrup.characteristic_numba_csr import numba_available

    if not numba_available():
        pytest.skip("Numba unavailable; the host CSR operator cannot run")
    rng = np.random.default_rng(seed)
    z = random_full(rng, 8, 5)
    net = transport_network(build_routing_graph(z, south_export(z), np.ones((8, 5)), DX))
    shape = (8, 5, nc)
    pickup = rng.uniform(0.0, 1.0, shape) * (rng.uniform(size=shape) < 0.6)
    v = rng.uniform(0.0, 0.2, shape)
    rate = rng.uniform(0.0, 4.0, shape)
    settle = rng.uniform(size=shape) < 0.15
    return net, np.zeros(shape), empty_sparse_phase_state(net, nc, nb), pickup, v, rate, settle


def _host_record(monkeypatch, step_args, **kw):
    """Per-substep write sizes and packets of ONE host step, observed at `run_write_pass` (its preallocated bin
    array's `.size` and the packet argument), passing every argument through unchanged."""
    from maple_syrup import characteristic_numba_csr as kernels
    from maple_syrup.characteristic_sparse import sparse_characteristic_step

    seen = []
    real = kernels.run_write_pass

    def recording(*args, **kwargs):
        seen.append((int(args[17].size), np.array(args[3], copy=True)))
        return real(*args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(kernels, "run_write_pass", recording)
        step = sparse_characteristic_step(*step_args, **kw)
    return step, seen


@pytest.mark.parametrize("n_substeps", [2, 3, 5])
@pytest.mark.parametrize("nb", [1, 8, 32])
def test_host_operator_injects_pickup_once_and_nnz_never_grows(monkeypatch, n_substeps, nb):
    net, mobile, phase, pickup, v, rate, settle = _host_case(nb=nb)
    step, seen = _host_record(monkeypatch, (net, mobile, phase, pickup, v, rate, settle, 1.0), n_substeps=n_substeps)
    sizes = [s for s, _ in seen]
    assert len(seen) == n_substeps
    assert np.count_nonzero(seen[0][1]) > 0 and all(not np.any(p) for _, p in seen[1:])  # pickup only in substep 0
    assert all(b <= a for a, b in zip(sizes, sizes[1:])), sizes
    compact_pickup = int(np.count_nonzero(seen[0][1]))
    assert sizes[0] <= phase.nnz + compact_pickup
    assert step.peak_nnz == sizes[0] and step.nnz_after == sizes[-1]
    # continue from the stored-bin state with a fresh pickup: the same rule holds from a nonempty phase
    step2, seen2 = _host_record(monkeypatch, (net, step.mobile_remaining_kg, step.phase_after, pickup * 0.5, v, rate,
                                              settle, 1.0), n_substeps=n_substeps)
    sizes2 = [s for s, _ in seen2]
    assert all(b <= a for a, b in zip(sizes2, sizes2[1:])), sizes2
    assert sizes2[0] <= step.phase_after.nnz + int(np.count_nonzero(seen2[0][1]))


@pytest.mark.parametrize("n_substeps", [2, 3])
def test_host_capacity_contract_first_substep_refusal_and_equal_cap(monkeypatch, n_substeps):
    from maple_syrup.characteristic_sparse import sparse_characteristic_step

    net, mobile, phase, pickup, v, rate, settle = _host_case()
    args = (net, mobile, phase, pickup, v, rate, settle, 1.0)
    _step, seen = _host_record(monkeypatch, args, n_substeps=n_substeps)
    first = seen[0][0]
    assert first >= 2
    with pytest.raises(SparseCapacityError) as info:
        sparse_characteristic_step(*args, n_substeps=n_substeps, max_nnz=first - 1)
    err = info.value
    assert (err.required_nnz, err.max_nnz, err.substep, err.n_substeps) == (first, first - 1, 0, n_substeps)
    ok = sparse_characteristic_step(*args, n_substeps=n_substeps, max_nnz=first)
    assert ok.peak_nnz == first
