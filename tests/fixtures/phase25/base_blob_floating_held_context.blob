"""H0: the WATER-FREE HELD LAW CONTEXT of a floating Pond vertical -- export and reconstruction without the original bed or owner (CPU/host; NumPy and Numba; CUDA is refused by a typed error before any coercion).

WHAT IT IS. A pure LAW context: the true `NativeFloatingConfig` inputs, the ROOT prepared floating column (raw field-capacity / saturation maps WITH their layouts, the depth `D0` it was prepared with, the active mask and
area), the held matrix operator (law inputs incl. the texture-Ks vector) and, when one exists, the preferential operator re-prepared from a caller-supplied EXPLICIT `PreferentialSpec`. It holds NO water, NO J/I, NO state,
NO `NativeFloatingSoil`, and it never reads a bed, a texture or the Saxton-Rawls provider: every held property is restored from the record, never recomputed from the current bed. It is NOT a physical acceptance, NOT a
source-bed proof, NOT a checkpoint and NOT file provenance: a later verified file/profile owns those; the identities it carries are the content identities the preparation reproduces (`config`, root column, matrix
operator, preferential `sha256`), compared with the caller's independently supplied `ExpectedHeldIdentities`.

EXPORT: `export_held_context(vertical, pref_spec=None, available_bytes_fn=..., reserve_bytes=0)` captures from the genuine `FloatingPondVertical.held_template()` + `vertical.config`; it re-prepares the preferential operator
from the spec on the template's own column and REQUIRES the exact operator `sha256` (the raw `a3`/contribution inputs are never reverse-engineered), snapshots the config inputs (the ORIGINAL layer thicknesses from
`config.layers.thickness_m`, never `diff(edges)` because the sequential fold can change them), and then performs a full dry reconstruction so that every exported record is known to rebuild bit-exact identities.
RECORD: `HeldContextRecord.meta()` (strict JSON-compatible dict; floats as `float.hex`) + `.arrays()` (named sealed C-contiguous arrays) are the serialization parts; `record_from_parts(meta, arrays, max_bytes=...)` is the
strict inverse (exact keys, schema, dtypes, shapes, C-contiguity, all checked BEFORE any copy; admission before the copies) and mints a record whose integrity DIGEST (all metadata and array bytes) is kept in a private weak
table and re-verified before every use: a replaced field, mapping entry or array byte is refused. RECONSTRUCT: `reconstruct_held_context(record, expected, available_bytes_fn=..., reserve_bytes=0)` calls only the public
factories (`nfs.native_floating_config`, `fsl.prepare_floating_column`, `fmi.prepare_floating_matrix`, `fpi.prepare_floating_preferential`), verifies every sha, and mints a `HeldContext` (tight window, plain precomputed
arguments, no callbacks, uncopyable). The minted context's guarantees are exactly "these public factories reproduced these identities from these record values"; it makes no claim about who produced the record.

MEMORY (declared modelled host bytes, not a whole-process bound): `estimate_held_context_memory` = the record arrays (`O(L N + (S+1) N + L K + overrides)`) + the retained prepared objects (the existing column/matrix/
preferential estimates) + the largest preparation scratch + a 64 KiB allowance; reconstruct admits it, then the components admit their own; export adds the sealed copies and the verification reconstruction. Real default host reader when none is given, strict non-Boolean integer reserve, no hot hash (the integrity digest is a one-off cold memoryview pass counted in the allowance).

FUTURE USE (not here): `FloatingPondVertical.from_held_context(ctx, ...)` and the general file codec take a `HeldContext` (config/column/matrix/pref/D0/Ks/active/generation/witness facts) instead of a `NativeFloatingSoil`, and restore the current payload (W/J/I/D) from their own verified file; they own source/file verification and the full native/Pond audits.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from maple_syrup import floating_matrix_infiltration as fmi
from maple_syrup import floating_pond as fp
from maple_syrup import floating_preferential_infiltration as fpi
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import native_floating_soil as nfs
from maple_syrup import soil_texture_properties as stp
from maple_syrup import spatial_soil_layers as sp
from maple_syrup import subsurface as ss

SCHEMA = "maple_syrup.floating_held_context/1"
SCHEMA2 = "maple_syrup.floating_held_context/2"  # the CUDA held-law record (host parts; device re-preparation in floating_held_context_cuda); /1 stays CPU-only and byte-unchanged
ALLOWANCE_BYTES = 1 << 16
SPEC_ARRAY_OVERHEAD_BYTES = 512  # declared per-ndarray Python/NumPy metadata allowance of PreferentialSpec.admitted (calibrated: measured ~130 B/array at S=1000; see the correction02 report)
DEFAULT_MAX_RECORD_BYTES = 1 << 30
_OVERRIDE_PREFIX = "override_"
_CONTRIB_PREFIX = "pref_contribution_"

__all__ = [
    "SCHEMA",
    "ExpectedHeldIdentities",
    "HeldContext",
    "HeldContextError",
    "HeldContextMemory",
    "HeldContextRecord",
    "PreferentialSpec",
    "estimate_held_context_memory",
    "export_held_context",
    "reconstruct_held_context",
    "record_from_parts",
    "require_held_context",
]


class HeldContextError(RuntimeError):
    """Typed refusal of an export / parse / reconstruction (nothing is mutated; nothing is downgraded)."""


_MINT = threading.local()
_RECORDS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # record -> integrity digest (private; never a field)
_CONTEXTS: weakref.WeakSet = weakref.WeakSet()


@contextlib.contextmanager
def _minting():
    _MINT.ok = True
    try:
        yield
    finally:
        _MINT.ok = False


def _hex(x: float) -> str:
    return float(x).hex()


def _unhex(text: Any, name: str, *, allow_inf: bool = False) -> float:
    if type(text) is not str or not 1 <= len(text) <= 64:
        raise HeldContextError(f"{name} must be a float.hex string of at most 64 characters")
    try:
        v = float.fromhex(text)
    except (ValueError, OverflowError):
        raise HeldContextError(f"{name} is not a float.hex string") from None
    if math.isnan(v):
        raise HeldContextError(f"{name} is NaN")
    if v.hex() != text:
        raise HeldContextError(f"{name} is not the canonical float.hex form (exact round trip required)")
    if math.isinf(v) and not allow_inf:
        raise HeldContextError(f"{name} must be finite")
    return v


def _reserve(value: Any) -> int:
    try:
        return fsl._count(value, "reserve_bytes")
    except fsl.FloatingSoilError as exc:
        raise HeldContextError(str(exc)) from None


def _admit(fn: Callable[[], int] | None, need: int, what: str) -> None:
    reader = sp.default_host_available_bytes if fn is None else fn
    if not callable(reader):
        raise HeldContextError("available_bytes_fn must be callable or None (the real host reader)")
    avail = reader()
    if isinstance(avail, (bool, np.bool_)) or not isinstance(avail, (int, np.integer)) or int(avail) < 0:
        raise HeldContextError(f"the available-bytes reader must return a non-Boolean integer >= 0, got {avail!r}")
    if need > int(avail):
        raise HeldContextError(f"{what} needs an estimated {need} B but only {int(avail)} B are reported available; nothing was allocated")


def _seal(a: Any) -> np.ndarray:
    out = np.array(a, copy=True, order="C")
    out.flags.writeable = False
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------
# the preferential specification (the raw inputs the operator cannot give back)
# ---------------------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class PreferentialSpec:
    """EXPLICIT raw inputs of `fpi.prepare_floating_preferential` (a3 and the maximum intake rate: a real scalar or an `(N,)` float64 array; `inf` allowed only for the rate; the background and species receiving weights are
    length-`L` vectors; each species contribution is an `(N,)` array). The prepared operator keeps only derived maps and weights, so these inputs must be supplied by the caller; they are never inferred.

    CONSTRUCTION WORKING SET: the constructor is PARAMETER construction, not an admitted stage: it retains its OWN sealed copies of the caller-supplied arrays (so the working set is exactly those caller-owned inputs,
    once more). `PreferentialSpec.admitted(...)` is the same construction behind the real host reader (metadata-only byte count + allowance + strict reserve) for callers that want it bounded; neither forks any physics
    (the values are only validated for type here; the physical bounds stay with the public preparation)."""

    a3: Any
    max_fraction_per_hour: float
    max_intake_rate_m_s: Any
    background_receiver_weights: Any = None
    species_receiver_weights: tuple = ()
    species_contributions: tuple | None = None

    @classmethod
    def admitted(cls, *, a3: Any, max_fraction_per_hour: Any, max_intake_rate_m_s: Any, background_receiver_weights: Any = None, species_receiver_weights: Any = (), species_contributions: Any = None,
                 available_bytes_fn: Callable[[], int] | None = None, reserve_bytes: int = 0) -> PreferentialSpec:
        reserve = _reserve(reserve_bytes)
        for name, seq in (("species_receiver_weights", species_receiver_weights), ("species_contributions", species_contributions)):
            if seq is not None and type(seq) not in (tuple, list):  # a bounded Sequence only: an unbounded iterator is never consumed
                raise HeldContextError(f"{name} must be a tuple or list (a bounded sequence)")
        need = cls.construction_need(a3, max_intake_rate_m_s, background_receiver_weights, species_receiver_weights, species_contributions) + reserve
        _admit(available_bytes_fn, need, "constructing the preferential spec (sealed copies and per-array metadata)")
        return cls(a3, max_fraction_per_hour, max_intake_rate_m_s, background_receiver_weights, tuple(species_receiver_weights), None if species_contributions is None else tuple(species_contributions))

    @staticmethod
    def construction_need(a3: Any, max_rate: Any, background: Any, species_weights: Any, species_contributions: Any) -> int:
        """Declared host bytes of `admitted(...)` (reserve excluded), from existing array metadata only (no copy, no list/tuple built): the raw payload of every ndarray, PLUS `SPEC_ARRAY_OVERHEAD_BYTES` for each ndarray
        (its owned copy's object header, GC header, allocator rounding, and its slot in the transient/final tuples and the sealing generator), PLUS the allowance. The per-array figure is a DECLARED, calibrated bound
        (see the correction02 measurements), not a tuned cap."""
        raw = n = 0
        for x in (a3, max_rate, background):
            if type(x) is np.ndarray:
                raw += int(x.nbytes)
                n += 1
        for seq in (species_weights, species_contributions or ()):
            for x in seq:
                if type(x) is np.ndarray:
                    raw += int(x.nbytes)
                    n += 1
        return raw + n * SPEC_ARRAY_OVERHEAD_BYTES + ALLOWANCE_BYTES

    def __post_init__(self) -> None:
        def scal_or_arr(v: Any, name: str) -> Any:
            if isinstance(v, (bool, np.bool_)):
                raise HeldContextError(f"{name} must be a real scalar or an (N,) float64 array")
            if isinstance(v, (int, float, np.integer, np.floating)):
                f = float(v)
                if math.isnan(f):
                    raise HeldContextError(f"{name} is NaN")
                return f
            if type(v) is np.ndarray and v.ndim == 1 and v.dtype == np.float64:
                return _seal(v)
            raise HeldContextError(f"{name} must be a real scalar or a 1-D float64 numpy array")

        object.__setattr__(self, "a3", scal_or_arr(self.a3, "a3"))
        object.__setattr__(self, "max_intake_rate_m_s", scal_or_arr(self.max_intake_rate_m_s, "max_intake_rate_m_s"))
        mf = self.max_fraction_per_hour
        if isinstance(mf, (bool, np.bool_)) or not isinstance(mf, (int, float, np.integer, np.floating)) or not math.isfinite(float(mf)):
            raise HeldContextError("max_fraction_per_hour must be a finite real scalar")
        object.__setattr__(self, "max_fraction_per_hour", float(mf))
        bg = self.background_receiver_weights
        if bg is not None:
            if type(bg) is not np.ndarray or bg.ndim != 1 or bg.dtype != np.float64:
                raise HeldContextError("background_receiver_weights must be None or a 1-D float64 numpy array of length L")
            bg = _seal(bg)
        object.__setattr__(self, "background_receiver_weights", bg)
        sw = tuple(self.species_receiver_weights)
        for w in sw:
            if type(w) is not np.ndarray or w.ndim != 1 or w.dtype != np.float64:
                raise HeldContextError("each species receiving-weight vector must be a 1-D float64 numpy array of length L")
        object.__setattr__(self, "species_receiver_weights", tuple(_seal(w) for w in sw))
        sc = self.species_contributions
        if sc is None:
            if sw:
                raise HeldContextError("species receiving weights were given without species_contributions")
        else:
            sc = tuple(sc)
            if len(sc) != len(sw):
                raise HeldContextError("species_contributions must have one (N,) array per species")
            for c in sc:
                if type(c) is not np.ndarray or c.ndim != 1 or c.dtype != np.float64:
                    raise HeldContextError("each species contribution must be a 1-D float64 numpy array")
            sc = tuple(_seal(c) for c in sc)
        object.__setattr__(self, "species_contributions", sc)

    def kwargs(self) -> dict:
        return {"a3": self.a3, "max_fraction_per_hour": self.max_fraction_per_hour, "max_intake_rate_m_s": self.max_intake_rate_m_s, "background_receiver_weights": self.background_receiver_weights,
                "species_receiver_weights": tuple(self.species_receiver_weights), "species_contributions": None if self.species_contributions is None else tuple(self.species_contributions)}


@dataclass(frozen=True)
class ExpectedHeldIdentities:
    """The caller's INDEPENDENT content identities (e.g. from a verified manifest); reconstruction must reproduce exactly these."""

    config_sha256: str
    root_column_sha256: str
    matrix_op_sha256: str
    pref_sha256: str | None = None


@dataclass(frozen=True)
class HeldContextMemory:
    record_bytes: int
    retained_bytes: int
    scratch_bytes: int
    allowance_bytes: int
    reconstruct_bytes: int  # record + retained + scratch + allowance (reserve excluded)
    export_bytes: int  # sealed copies + the verification reconstruction


def estimate_held_context_memory(n_layers: int, n_cells: int, n_classes: int, n_species: int, n_overrides: int, *, backend: str, layouts: dict, has_pref: bool, texture_ks: bool,
                                 pref_array_cells: int = 0) -> HeldContextMemory:
    """Pure arithmetic over the existing component estimates. `pref_array_cells` = how many of the preferential raw inputs are per-cell arrays (a3, max rate, species contributions)."""
    L, N, K, S = (fsl._count(v, n, minimum=m) for v, n, m in ((n_layers, "n_layers", 1), (n_cells, "n_cells", 1), (n_classes, "n_classes", 1), (n_species, "n_species", 0)))
    nov = fsl._count(n_overrides, "n_overrides")
    f8 = 8
    record = (L * 5 + 3 * K + L * N * 2 + N * (2 if texture_ks else 1) + N + nov * L * N) * f8 + N + ((L * (S + 1) + (pref_array_cells) * N) * f8 if has_pref else 0)
    col = fsl.estimate_floating_memory(L, N, layouts=dict(layouts), backend=backend)
    mat = fmi.estimate_floating_matrix_memory(L, N, backend=backend, operation="prepare")
    pref = fpi.estimate_floating_preferential_memory(L, N, backend=backend, operation="prepare", n_species=S) if has_pref else None
    retained = int(col.retained_bytes) + int(mat.retained_bytes if hasattr(mat, "retained_bytes") else 3 * N * f8) + (0 if pref is None else int(pref.retained_bytes if hasattr(pref, "retained_bytes") else (L * N + 2 * N) * f8))
    scratch = max(int(getattr(col, "step_bytes", 0)) - int(col.retained_bytes), 0)
    if pref is not None:
        scratch = max(scratch, int(pref.scratch_bytes))
    recon = record + retained + scratch + ALLOWANCE_BYTES
    return HeldContextMemory(int(record), int(retained), int(scratch), ALLOWANCE_BYTES, int(recon), int(record + recon))


# ---------------------------------------------------------------------------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------------------------------------------------------------------------
_META_KEYS = frozenset({"schema", "backend", "n_layers", "n_cells", "n_classes", "mode", "bottom_drainage", "scalars", "overrides", "layouts", "identities", "facts", "pref"})
_META_KEYS_V2 = _META_KEYS | {"kernel"}
_KERNEL_KEYS = frozenset({"common_abi_sha256", "fsl_resident_source_sha256", "fmc_source_sha256", "fpc_source_sha256"})
_SCALAR_KEYS = ("bulk_density_kg_m3", "theta_residual", "theta_evaporation_stress", "evaporation_factor", "matrix_conductivity_m_s", "matrix_suction_m", "matrix_vegetation_factor", "initial_wetting_m",
                "initial_cumulative_intake_m", "latent_saturated_conductivity_m_s", "cell_area_m2")
_PREF_KEYS = frozenset({"n_species", "a3_kind", "a3_value", "max_fraction_per_hour", "max_rate_kind", "max_rate_value", "has_background"})
_IDENT_KEYS = frozenset({"config_sha256", "root_column_sha256", "matrix_op_sha256", "pref_sha256"})
_FACT_KEYS = frozenset({"generation", "source_witness"})
_HEX64 = frozenset("0123456789abcdef")


def _is_hex64(v: Any) -> bool:
    return type(v) is str and len(v) == 64 and set(v) <= _HEX64


def _expected_arrays(meta: dict) -> dict[str, tuple[tuple, Any]]:
    """The exact array names, shapes and dtypes a META record requires (strict: nothing else is accepted)."""
    L, N, K = meta["n_layers"], meta["n_cells"], meta["n_classes"]
    f8, b1 = np.dtype(np.float64), np.dtype(np.bool_)
    fcshape = {"scalar": (1,), "shared": (L,), "cell": (L, N)}
    spec = {"thickness_m": ((L,), f8), "grain_texture": ((K, 3), f8), "organic_matter_percent": ((L,), f8), "latent_theta_field_capacity": ((L,), f8), "latent_theta_saturation": ((L,), f8),
            "diffusivity_m2_s": ((max(L - 1, 0),), f8), "theta_field_capacity": (fcshape[meta["layouts"]["theta_field_capacity"]], f8), "theta_saturation": (fcshape[meta["layouts"]["theta_saturation"]], f8),
            "depth0_m": ((N,), f8), "active": ((N,), b1)}
    if meta["mode"] == "texture_ks":
        spec["ks_m_s"] = ((N,), f8)
    for k in meta["overrides"] or ():
        spec[_OVERRIDE_PREFIX + k] = ((L, N), f8)
    p = meta["pref"]
    if p is not None:
        if p["a3_kind"] == "cell":
            spec["pref_a3"] = ((N,), f8)
        if p["max_rate_kind"] == "cell":
            spec["pref_max_rate"] = ((N,), f8)
        if p["has_background"]:
            spec["pref_background"] = ((L,), f8)
        if p["n_species"]:
            spec["pref_species_weights"] = ((p["n_species"], L), f8)
            for i in range(p["n_species"]):
                spec[f"{_CONTRIB_PREFIX}{i}"] = ((N,), f8)
    return spec


_LAYOUT_NAMES = ("theta_residual", "theta_field_capacity", "theta_saturation", "interface")  # the genuine FSL column layout keys
_REQUIRED_SCALARS = ("bulk_density_kg_m3", "theta_residual", "theta_evaporation_stress", "evaporation_factor", "matrix_suction_m", "matrix_vegetation_factor", "initial_wetting_m", "initial_cumulative_intake_m",
                     "cell_area_m2")
_MAX_DIM = 1 << 40  # an integer sanity bound on a declared dimension (the analytic byte/count caps below reject any unsupported resource)
_MAX_NAME = 128


def _exact_int(v: Any, name: str, lo: int, hi: int = _MAX_DIM) -> int:
    if type(v) is not int or not lo <= v <= hi:
        raise HeldContextError(f"{name} must be an integer in [{lo}, {hi}]")
    return v


def _closed(d: Any, keys: Any, name: str) -> dict:
    if type(d) is not dict or set(d) != set(keys):
        raise HeldContextError(f"{name} keys differ from the schema (closed key set)")
    return d


def _check_meta(meta: Any, n_arrays: int) -> dict:
    """STRICT, BOUNDED metadata validation with NO serialization, deep copy or allocation proportional to the input: closed key sets at every level, exact types, bounded strings, canonical finite `float.hex`
    scalars (only the two mode-specific conductivities may be None; `inf` only where the preferential physics allows: the scalar maximum intake rate), the genuine FSL layout names and kinds, override names from the
    provider's own set, and every declared count bounded by the number of arrays actually supplied (`n_arrays`) so that a huge declared species/override count can never build its names or lists first."""
    v2 = type(meta) is dict and meta.get("schema") == SCHEMA2  # explicit versioned dispatch: a /2 record is NEVER read as /1 and vice versa
    _closed(meta, _META_KEYS_V2 if v2 else _META_KEYS, "held-context metadata")
    if meta["schema"] != (SCHEMA2 if v2 else SCHEMA):
        raise HeldContextError("unknown held-context schema")
    if v2:
        kern = _closed(meta["kernel"], _KERNEL_KEYS, "kernel")
        for k in _KERNEL_KEYS:
            if not _is_hex64(kern[k]):
                raise HeldContextError(f"kernel.{k} must be a 64-character lowercase hex digest")
    L = _exact_int(meta["n_layers"], "n_layers", 1)
    _exact_int(meta["n_cells"], "n_cells", 1)
    _exact_int(meta["n_classes"], "n_classes", 1)
    if meta["backend"] not in (("cuda",) if v2 else ("numpy", "numba")):
        raise HeldContextError(f"backend {meta['backend']!r} is not supported by this record version (/1: numpy|numba, /2: cuda only; there is no fallback)")
    if meta["mode"] not in ("explicit", "texture_ks") or meta["bottom_drainage"] not in ("open", "closed"):
        raise HeldContextError("mode / bottom_drainage are not recognised")
    sc = _closed(meta["scalars"], _SCALAR_KEYS, "scalars")
    for k in _REQUIRED_SCALARS:
        _unhex(sc[k], f"scalars.{k}")  # a required scalar is a canonical finite float.hex: None / inf / NaN / non-canonical are typed refusals here, before any copy
    explicit = meta["mode"] == "explicit"
    for k, present in (("matrix_conductivity_m_s", explicit), ("latent_saturated_conductivity_m_s", not explicit)):
        if present:
            _unhex(sc[k], f"scalars.{k}")
        elif sc[k] is not None:
            raise HeldContextError(f"scalars.{k} must be None in {meta['mode']} mode")
    ov = meta["overrides"]
    if ov is not None:
        if type(ov) is not list or len(ov) > min(len(stp.OVERRIDES), n_arrays):
            raise HeldContextError("overrides must be None or a short list of provider override names")
        if any(type(k) is not str or k not in stp.OVERRIDES for k in ov) or ov != sorted(set(ov)):
            raise HeldContextError("override names must be distinct, sorted names from the provider's own override set")
    lay = _closed(meta["layouts"], ("theta_field_capacity", "theta_saturation", "all"), "layouts")
    allk = _closed(lay["all"], _LAYOUT_NAMES, "layouts.all")
    if any(allk[k] not in ("scalar", "shared", "cell") for k in _LAYOUT_NAMES) or lay["theta_field_capacity"] != allk["theta_field_capacity"] or lay["theta_saturation"] != allk["theta_saturation"]:
        raise HeldContextError("layout kinds must be scalar/shared/cell and consistent with the column's own layout record")
    ident = _closed(meta["identities"], _IDENT_KEYS, "identities")
    for k in ("config_sha256", "root_column_sha256", "matrix_op_sha256"):
        if not _is_hex64(ident[k]):
            raise HeldContextError(f"identities.{k} must be a 64-character lowercase hex digest")
    if not (ident["pref_sha256"] is None or _is_hex64(ident["pref_sha256"])):
        raise HeldContextError("identities.pref_sha256 must be None or a 64-character lowercase hex digest")
    facts = _closed(meta["facts"], _FACT_KEYS, "facts")
    _exact_int(facts["generation"], "facts.generation", 0)
    if not _is_hex64(facts["source_witness"]):
        raise HeldContextError("facts.source_witness must be a 64-character lowercase hex digest")
    p = meta["pref"]
    if (p is None) != (ident["pref_sha256"] is None):
        raise HeldContextError("the preferential record and its identity must both be present or both absent")
    if p is not None:
        _closed(p, _PREF_KEYS, "pref")
        S = _exact_int(p["n_species"], "pref.n_species", 0)
        if S > n_arrays or type(p["has_background"]) is not bool:  # a declared species count can never exceed the arrays actually supplied
            raise HeldContextError("preferential metadata malformed (species count exceeds the supplied arrays or has_background is not a bool)")
        for kind, val, inf_ok in (("a3_kind", "a3_value", False), ("max_rate_kind", "max_rate_value", True)):
            if p[kind] not in ("scalar", "cell") or (p[kind] == "scalar") != (p[val] is not None):
                raise HeldContextError("preferential kind/value pair malformed")
            if p[val] is not None:
                _unhex(p[val], f"pref.{val}", allow_inf=inf_ok)  # `inf` only for the scalar maximum intake rate (the physics allows an unbounded rate)
        _unhex(p["max_fraction_per_hour"], "pref.max_fraction_per_hour")
    del L
    return meta


def _analytic(meta: dict) -> tuple[int, int]:
    """`(array count, total array bytes)` from the DECLARED inputs by arithmetic alone (no names, no shapes built): used to refuse an unsupported resource before anything proportional is created."""
    L, N, K = meta["n_layers"], meta["n_cells"], meta["n_classes"]
    kinds = {"scalar": 1, "shared": L, "cell": L * N}
    count = 10 + (1 if meta["mode"] == "texture_ks" else 0) + len(meta["overrides"] or ())
    elems = L * 4 + K * 3 + max(L - 1, 0) + kinds[meta["layouts"]["theta_field_capacity"]] + kinds[meta["layouts"]["theta_saturation"]] + N  # thickness, OM, FC/SAT latent, texture, diffusivity, FC/SAT raw, D0
    nbytes = 8 * (elems + (N if meta["mode"] == "texture_ks" else 0) + len(meta["overrides"] or ()) * L * N) + N  # + the bool mask
    p = meta["pref"]
    if p is not None:
        S = p["n_species"]
        count += int(p["a3_kind"] == "cell") + int(p["max_rate_kind"] == "cell") + int(p["has_background"]) + (1 + S if S else 0)
        nbytes += 8 * ((N if p["a3_kind"] == "cell" else 0) + (N if p["max_rate_kind"] == "cell" else 0) + (L if p["has_background"] else 0) + S * L + S * N)
    return count, nbytes


def _check_layout(meta: dict, arrays: Any) -> None:
    """Exact array metadata BEFORE any byte is read, hashed or copied: names, `type is np.ndarray`, dtype, shape, C-contiguity (no conversion, no `__array__` callback)."""
    want = _expected_arrays(meta)
    if not hasattr(arrays, "keys") or set(arrays.keys()) != set(want):
        raise HeldContextError("array names differ from the schema (extra or missing arrays)")
    for name, (shape, dtype) in want.items():
        a = arrays[name]
        if type(a) is not np.ndarray or a.dtype != dtype or tuple(a.shape) != tuple(shape) or not a.flags.c_contiguous:
            raise HeldContextError(f"array {name} is not an exact C-contiguous {dtype} ndarray of shape {tuple(shape)}")


def _digest(meta: dict, arrays: Any) -> str:
    h = hashlib.sha256(json.dumps(meta, sort_keys=True).encode())
    for name in sorted(arrays.keys()):
        a = arrays[name]
        h.update(f"|{name}|{a.dtype}|{a.shape}".encode())
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()


@dataclass(frozen=True, eq=False)
class HeldContextRecord:
    """Opaque owned record of the held law context (module docstring): minted ONLY by `export_held_context` / `record_from_parts`."""

    _meta: Any
    _arrays: Any

    def __post_init__(self) -> None:
        if not getattr(_MINT, "ok", False):
            raise HeldContextError("a HeldContextRecord is minted only by export_held_context / record_from_parts (no direct construction or replace())")

    def __copy__(self):
        raise HeldContextError("a HeldContextRecord is an owned record and cannot be copied")

    def __deepcopy__(self, memo):
        raise HeldContextError("a HeldContextRecord is an owned record and cannot be copied")

    def __reduce_ex__(self, protocol):
        raise HeldContextError("a HeldContextRecord is not pickled; serialise `meta()` and `arrays()` explicitly")

    def meta(self) -> dict:
        _verify(self)
        return json.loads(json.dumps(self._meta))  # the metadata was validated bounded (closed keys, bounded strings): the copy is bounded

    def arrays(self) -> Any:
        _verify(self)
        return self._arrays


def _verify(record: Any) -> None:
    if type(record) is not HeldContextRecord or record not in _RECORDS:
        raise HeldContextError("not a genuinely minted HeldContextRecord")
    meta, arrays = record._meta, record._arrays
    if not isinstance(arrays, MappingProxyType) or type(meta) is not dict:
        raise HeldContextError("the record's containers were replaced (integrity)")
    _check_meta(meta, len(arrays))  # bounded validation in place: no serialization/deep copy before the closed-key checks
    _check_layout(meta, arrays)
    if _digest(meta, arrays) != _RECORDS[record]:
        raise HeldContextError("the record's contents no longer match the integrity digest bound at minting (altered)")


def _mint_record(meta: dict, arrays: dict) -> HeldContextRecord:
    ro = MappingProxyType(dict(arrays))
    digest = _digest(meta, ro)
    with _minting():
        rec = HeldContextRecord(meta, ro)
    _RECORDS[rec] = digest
    return rec


def record_from_parts(meta: Any, arrays: Any, *, max_bytes: int = DEFAULT_MAX_RECORD_BYTES, available_bytes_fn: Callable[[], int] | None = None, reserve_bytes: int = 0) -> HeldContextRecord:
    """STRICT parse of externally stored parts into a minted record (module docstring). ORDER: reserve/max_bytes metadata -> the bounded closed-key metadata validation (no JSON round trip, no deep copy: nothing is
    proportional to the input before it is validated) -> the analytic array count and byte cap from the DECLARED inputs (an unsupported resource is rejected before any name, shape or copy is built) -> exact array
    metadata -> the real admission -> the sealed copies -> the metadata copy (bounded by the validation) and the mint."""
    reserve = _reserve(reserve_bytes)
    if isinstance(max_bytes, (bool, np.bool_)) or not isinstance(max_bytes, (int, np.integer)) or int(max_bytes) < 1:
        raise HeldContextError("max_bytes must be a positive integer")
    if not hasattr(arrays, "keys") or not hasattr(arrays, "__getitem__") or not hasattr(arrays, "__len__"):
        raise HeldContextError("arrays must be a mapping of names to arrays")
    meta = _check_meta(meta, len(arrays))
    count, nbytes = _analytic(meta)
    if count != len(arrays):
        raise HeldContextError(f"the declared inputs require {count} arrays but {len(arrays)} were supplied")
    if nbytes > int(max_bytes):
        raise HeldContextError(f"the declared inputs require {nbytes} array bytes, above max_bytes ({int(max_bytes)} B)")
    _check_layout(meta, arrays)
    total = sum(int(arrays[n].nbytes) for n in arrays)
    if total != nbytes or total > int(max_bytes):
        raise HeldContextError(f"the arrays hold {total} B but the declared inputs imply {nbytes} B (max_bytes {int(max_bytes)} B)")
    _admit(available_bytes_fn, total + ALLOWANCE_BYTES + reserve, "parsing the held-context record (sealed copies)")
    sealed = {n: _seal(arrays[n]) for n in arrays}
    return _mint_record(json.loads(json.dumps(meta)), sealed)  # the metadata copy is bounded: closed keys, bounded strings, counts bounded by the supplied arrays


# ---------------------------------------------------------------------------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _snapshot(vertical: Any, pref_spec: Any) -> tuple[dict, dict]:
    t = vertical.held_template()
    cfg = vertical.config
    col = t.column
    if type(cfg) is not nfs.NativeFloatingConfig or cfg.sha256 != t.config_sha256:
        raise HeldContextError("the vertical's native config does not match its held template identity")
    L, N, K = int(col.n_layers), int(col.n_cells), int(cfg.grain_texture.shape[0])
    texture_ks = cfg.matrix_conductivity_mode == "texture_ks"
    lay = {k: str(v) for k, v in dict(col.layouts).items()}
    meta = {"schema": SCHEMA, "backend": cfg.backend, "n_layers": L, "n_cells": N, "n_classes": K, "mode": cfg.matrix_conductivity_mode, "bottom_drainage": cfg.bottom_drainage,
            "scalars": {"bulk_density_kg_m3": _hex(cfg.bulk_density_kg_m3), "theta_residual": _hex(cfg.theta_residual), "theta_evaporation_stress": _hex(cfg.theta_evaporation_stress),
                        "evaporation_factor": _hex(cfg.evaporation_factor), "matrix_conductivity_m_s": None if cfg.matrix_conductivity_m_s is None else _hex(cfg.matrix_conductivity_m_s),
                        "matrix_suction_m": _hex(cfg.matrix_suction_m), "matrix_vegetation_factor": _hex(cfg.matrix_vegetation_factor), "initial_wetting_m": _hex(cfg.initial_wetting_m),
                        "initial_cumulative_intake_m": _hex(cfg.initial_cumulative_intake_m),
                        "latent_saturated_conductivity_m_s": None if cfg.latent_saturated_conductivity_m_s is None else _hex(cfg.latent_saturated_conductivity_m_s), "cell_area_m2": _hex(col.cell_area_m2)},
            "overrides": None if cfg.overrides is None else sorted(cfg.overrides), "layouts": {"theta_field_capacity": lay["theta_field_capacity"], "theta_saturation": lay["theta_saturation"], "all": lay},
            "identities": {"config_sha256": cfg.sha256, "root_column_sha256": col.sha256, "matrix_op_sha256": t.matrix_op.sha256, "pref_sha256": None if t.preferential is None else t.preferential.sha256},
            "facts": {"generation": int(t.generation), "source_witness": t.source_witness}, "pref": None}
    arrays = {"thickness_m": cfg.layers.thickness_m, "grain_texture": cfg.grain_texture, "organic_matter_percent": cfg.organic_matter_percent, "latent_theta_field_capacity": cfg.latent_theta_field_capacity,
              "latent_theta_saturation": cfg.latent_theta_saturation, "diffusivity_m2_s": cfg.diffusivity_m2_s, "theta_field_capacity": col._maps["theta_field_capacity"],
              "theta_saturation": col._maps["theta_saturation"], "depth0_m": t.depth0_m, "active": col._maps["active"]}
    if texture_ks:
        if t.ks_m_s is None:
            raise HeldContextError("a texture_ks owner must carry its layer-0 conductivity record")
        arrays["ks_m_s"] = t.ks_m_s
    for k in meta["overrides"] or ():
        arrays[_OVERRIDE_PREFIX + k] = cfg.overrides[k]
    if t.preferential is None:
        if pref_spec is not None:
            raise HeldContextError("a preferential spec was given but the vertical has no preferential operator")
    else:
        if type(pref_spec) is not PreferentialSpec:
            raise HeldContextError("the vertical has a preferential operator: its raw inputs are lost, so an explicit PreferentialSpec is required (never inferred)")
        S = len(pref_spec.species_receiver_weights)
        if S != t.preferential.n_species:
            raise HeldContextError("the spec's species count differs from the operator's")
        p = {"n_species": S, "a3_kind": "cell" if isinstance(pref_spec.a3, np.ndarray) else "scalar", "a3_value": None if isinstance(pref_spec.a3, np.ndarray) else _hex(pref_spec.a3),
             "max_fraction_per_hour": _hex(pref_spec.max_fraction_per_hour), "max_rate_kind": "cell" if isinstance(pref_spec.max_intake_rate_m_s, np.ndarray) else "scalar",
             "max_rate_value": None if isinstance(pref_spec.max_intake_rate_m_s, np.ndarray) else _hex(pref_spec.max_intake_rate_m_s), "has_background": pref_spec.background_receiver_weights is not None}
        meta["pref"] = p
        if isinstance(pref_spec.a3, np.ndarray):
            arrays["pref_a3"] = pref_spec.a3
        if isinstance(pref_spec.max_intake_rate_m_s, np.ndarray):
            arrays["pref_max_rate"] = pref_spec.max_intake_rate_m_s
        if p["has_background"]:
            arrays["pref_background"] = pref_spec.background_receiver_weights
        if S:  # NO stack here: the per-species weight vectors stay REFERENCES until the composed admission; the owned `(S, L)` block is built under the quota
            for i, c in enumerate(pref_spec.species_contributions):
                arrays[f"{_CONTRIB_PREFIX}{i}"] = c
    return meta, arrays


def _model(meta: dict) -> HeldContextMemory:
    p = meta["pref"]
    cells = 0 if p is None else (p["a3_kind"] == "cell") + (p["max_rate_kind"] == "cell") + p["n_species"]
    return estimate_held_context_memory(meta["n_layers"], meta["n_cells"], meta["n_classes"], 0 if p is None else p["n_species"], len(meta["overrides"] or ()), backend=meta["backend"],
                                        layouts=meta["layouts"]["all"], has_pref=p is not None, texture_ks=meta["mode"] == "texture_ks", pref_array_cells=int(cells))


def adopt_parts(meta: Any, arrays: Any) -> HeldContextRecord:
    """Mint a record from parts the CALLER's CUDA exporter has just built and OWNS (fresh host arrays created under its admitted quota): the same strict metadata / array-metadata / analytic-count checks as
    `record_from_parts`, then the arrays are sealed IN PLACE (no second copy) and the record is minted by this module. Plain parts only; no callbacks."""
    meta = _check_meta(meta, len(arrays))
    count, nbytes = _analytic(meta)
    if count != len(arrays):
        raise HeldContextError(f"the declared inputs require {count} arrays but {len(arrays)} were supplied")
    _check_layout(meta, arrays)
    if sum(int(arrays[n].nbytes) for n in arrays) != nbytes:
        raise HeldContextError("the arrays do not hold the declared bytes")
    for n in arrays:
        arrays[n].flags.writeable = False
    return _mint_record(json.loads(json.dumps(meta)), dict(arrays))


def export_held_context(vertical: Any, pref_spec: Any = None, *, available_bytes_fn: Callable[[], int] | None = None, reserve_bytes: int = 0, verify: bool = True) -> HeldContextRecord:
    """Capture the held law context of a genuine `FloatingPondVertical` (module docstring). `verify=True` (default) performs a full dry reconstruction so the exported record is known to rebuild the exact identities."""
    reserve = _reserve(reserve_bytes)
    if type(vertical) is not fp.FloatingPondVertical:
        raise HeldContextError("vertical must be a genuine FloatingPondVertical")
    if vertical.backend == "cuda":  # explicit versioned dispatch to the CUDA record (/2): a different module, a different schema, no fallback
        from maple_syrup import floating_held_context_cuda as fhc

        return fhc.export_held_context_cuda(vertical, pref_spec, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve, verify=verify)
    if vertical.backend not in ("numpy", "numba"):
        raise HeldContextError(f"backend {vertical.backend!r} is not supported by the CPU held context")
    meta, arrays = _snapshot(vertical, pref_spec)  # metadata and REFERENCES only: nothing proportional to the arrays is built before the admission below
    m = _model(meta)
    _admit(available_bytes_fn, (m.export_bytes if verify else m.record_bytes + ALLOWANCE_BYTES) + reserve, "exporting the held law context")
    if pref_spec is not None and pref_spec.species_receiver_weights:  # materialised UNDER the admitted quota (one `(S, L)` block)
        arrays["pref_species_weights"] = np.stack(pref_spec.species_receiver_weights)
    pref_check = None
    if pref_spec is not None:  # the spec must reproduce the EXACT operator identity on the template's own column (never reverse-engineered)
        try:
            pref_check = fpi.prepare_floating_preferential(vertical.held_template().column, **pref_spec.kwargs(), available_bytes_fn=available_bytes_fn)
        except (fsl.FloatingSoilError, ss.SubsurfaceError) as exc:
            raise HeldContextError(f"the preferential spec was refused by the public preparation: {exc}") from None
        if pref_check.sha256 != meta["identities"]["pref_sha256"]:
            raise HeldContextError("the preferential spec does not reproduce the operator's exact sha256 (a misstated a3/receiving-weight/contribution input)")
        del pref_check
    record = _mint_record(meta, {k: _seal(v) for k, v in arrays.items()})
    if verify:
        reconstruct_held_context(record, ExpectedHeldIdentities(meta["identities"]["config_sha256"], meta["identities"]["root_column_sha256"], meta["identities"]["matrix_op_sha256"],
                                                                 meta["identities"]["pref_sha256"]), available_bytes_fn=available_bytes_fn, reserve_bytes=reserve)
    return record


# ---------------------------------------------------------------------------------------------------------------------------------------------
# reconstruction
# ---------------------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class HeldContext:
    """A WATER-FREE held law context minted ONLY by `reconstruct_held_context`: no water, no J/I, no state, no owner. Its guarantee is exactly that the public factories reproduced the expected identities from the record."""

    config: Any
    column: Any
    matrix_op: Any
    preferential: Any
    depth0_m: np.ndarray
    ks_m_s: Any
    active: np.ndarray
    area_m2: float
    generation: int
    source_witness: str
    identities: Any
    layouts: Any
    backend: str = "numpy"  # "cuda": config/column/matrix/pref/depth0_m/ks_m_s/active are the DEVICE objects of the actual qualified preparation
    edges_m: Any = None  # CUDA only: the O(L) HOST nominal edges
    host_depth0_m: Any = None  # CUDA only: sealed HOST D0 / mask / Ks references (the record's own arrays: no new mirror)
    host_active: Any = None
    host_ks_m_s: Any = None

    def __post_init__(self) -> None:
        if not getattr(_MINT, "ok", False):
            raise HeldContextError("a HeldContext is minted only by reconstruct_held_context")
        _CONTEXTS.add(self)

    def __copy__(self):
        raise HeldContextError("a HeldContext cannot be copied")

    def __deepcopy__(self, memo):
        raise HeldContextError("a HeldContext cannot be copied")

    def __reduce_ex__(self, protocol):
        raise HeldContextError("a HeldContext is not serialisable (export a record instead)")


def require_held_context(context: Any) -> None:
    """Public check: `context` is a genuine `HeldContext` minted by `reconstruct_held_context` (exact class and private registry membership; no duck types, no foreign classes). Not a file or bed proof."""
    if type(context) is not HeldContext or context not in _CONTEXTS:
        raise HeldContextError("not a genuine HeldContext (minted only by reconstruct_held_context)")


def reconstruct_held_context(record: Any, expected: Any, *, available_bytes_fn: Callable[[], int] | None = None, reserve_bytes: int = 0, host_available_bytes_fn: Callable[[], int] | None = None,
                             host_reserve_bytes: int = 0) -> HeldContext:
    """Rebuild the water-free held law context from a minted, intact record through the public factories only (module docstring). No bed, texture reader, provider or owner is touched."""
    reserve = _reserve(reserve_bytes)
    _verify(record)  # minted AND intact, before any use
    if type(expected) is not ExpectedHeldIdentities:
        raise HeldContextError("expected must be ExpectedHeldIdentities")
    meta, arr = record._meta, record._arrays
    ident = meta["identities"]
    if meta["schema"] == SCHEMA2:  # CUDA record: gate first, two pools, device re-preparation (floating_held_context_cuda); the context is minted HERE after the plain parts were validated
        if (expected.config_sha256, expected.root_column_sha256, expected.matrix_op_sha256, expected.pref_sha256) != (ident["config_sha256"], ident["root_column_sha256"], ident["matrix_op_sha256"], ident["pref_sha256"]):
            raise HeldContextError("the record's identities differ from the expected identities")
        from maple_syrup import floating_held_context_cuda as fhc

        parts = fhc.prepare_parts(record, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve, host_available_bytes_fn=host_available_bytes_fn, host_reserve_bytes=host_reserve_bytes)
        return _mint_cuda_context(parts, meta)
    if host_available_bytes_fn is not None or host_reserve_bytes:
        raise HeldContextError("host_available_bytes_fn / host_reserve_bytes apply only to a CUDA (/2) record (a CPU record has one host pool)")
    if (expected.config_sha256, expected.root_column_sha256, expected.matrix_op_sha256, expected.pref_sha256) != (ident["config_sha256"], ident["root_column_sha256"], ident["matrix_op_sha256"], ident["pref_sha256"]):
        raise HeldContextError("the record's identities differ from the expected identities")
    m = _model(meta)
    _admit(available_bytes_fn, m.reconstruct_bytes + reserve, "reconstructing the held law context")
    sc = {k: (None if v is None else _unhex(v, k)) for k, v in meta["scalars"].items()}
    overrides = None if meta["overrides"] is None else {k: arr[_OVERRIDE_PREFIX + k] for k in meta["overrides"]}
    L = meta["n_layers"]
    try:
        cfg = nfs.native_floating_config(layer_thickness_m=arr["thickness_m"], grain_texture=arr["grain_texture"], organic_matter_percent=arr["organic_matter_percent"], bulk_density_kg_m3=sc["bulk_density_kg_m3"],
                                         latent_theta_field_capacity=arr["latent_theta_field_capacity"], latent_theta_saturation=arr["latent_theta_saturation"], theta_residual=sc["theta_residual"],
                                         theta_evaporation_stress=sc["theta_evaporation_stress"], evaporation_factor=sc["evaporation_factor"], diffusivity_m2_s=arr["diffusivity_m2_s"],
                                         bottom_drainage=meta["bottom_drainage"], matrix_conductivity_m_s=sc["matrix_conductivity_m_s"], matrix_suction_m=sc["matrix_suction_m"],
                                         matrix_vegetation_factor=sc["matrix_vegetation_factor"], initial_wetting_m=sc["initial_wetting_m"], initial_cumulative_intake_m=sc["initial_cumulative_intake_m"],
                                         overrides=overrides, backend=meta["backend"], matrix_conductivity_mode=meta["mode"], latent_saturated_conductivity_m_s=sc["latent_saturated_conductivity_m_s"])
        if cfg.sha256 != ident["config_sha256"]:
            raise HeldContextError("the reconstructed native config does not reproduce the recorded config sha256")

        def raw(name: str, kind: str) -> Any:
            a = arr[name]
            return float(a[0]) if kind == "scalar" else a

        fc, sat = raw("theta_field_capacity", meta["layouts"]["theta_field_capacity"]), raw("theta_saturation", meta["layouts"]["theta_saturation"])
        col = fsl.prepare_floating_column(cfg.nominal_edges_m, theta_residual=cfg.theta_residual, theta_field_capacity=fc, theta_saturation=sat, diffusivity_m2_s=cfg.diffusivity_m2_s if L > 1 else 0.0,
                                          theta_evaporation_stress=cfg.theta_evaporation_stress, evaporation_factor=cfg.evaporation_factor, diffusion_mode="constant", bottom_drainage=cfg.bottom_drainage,
                                          soil_depth_m=arr["depth0_m"], active_mask=arr["active"], cell_area_m2=sc["cell_area_m2"], backend=cfg.backend, available_bytes_fn=available_bytes_fn)
        if col.sha256 != ident["root_column_sha256"]:
            raise HeldContextError("the re-prepared root column does not reproduce the recorded root column sha256")
        ks = arr["ks_m_s"] if meta["mode"] == "texture_ks" else None
        op = fmi.prepare_floating_matrix(col, conductivity_m_s=ks if ks is not None else cfg.matrix_conductivity_m_s, suction_m=cfg.matrix_suction_m, vegetation_factor=cfg.matrix_vegetation_factor,
                                         available_bytes_fn=available_bytes_fn)
        if op.sha256 != ident["matrix_op_sha256"]:
            raise HeldContextError("the re-prepared matrix operator does not reproduce the recorded sha256")
        pref = None
        p = meta["pref"]
        if p is not None:
            sw = tuple(arr["pref_species_weights"][i] for i in range(p["n_species"])) if p["n_species"] else ()
            contrib = tuple(arr[f"{_CONTRIB_PREFIX}{i}"] for i in range(p["n_species"])) if p["n_species"] else None
            pref = fpi.prepare_floating_preferential(col, a3=arr["pref_a3"] if p["a3_kind"] == "cell" else _unhex(p["a3_value"], "a3"), max_fraction_per_hour=_unhex(p["max_fraction_per_hour"], "max_fraction"),
                                                     max_intake_rate_m_s=arr["pref_max_rate"] if p["max_rate_kind"] == "cell" else _unhex(p["max_rate_value"], "max_rate", allow_inf=True),
                                                     background_receiver_weights=arr["pref_background"] if p["has_background"] else None, species_receiver_weights=sw, species_contributions=contrib,
                                                     available_bytes_fn=available_bytes_fn)
            if pref.sha256 != ident["pref_sha256"]:
                raise HeldContextError("the re-prepared preferential operator does not reproduce the recorded sha256")
    except HeldContextError:
        raise
    except (nfs.NativeFloatingError, fsl.FloatingSoilError, ss.SubsurfaceError, ValueError) as exc:
        raise HeldContextError(f"a public factory refused the record: {exc}") from None
    depth0, active = _seal(arr["depth0_m"]), _seal(arr["active"])
    ks_sealed = None if ks is None else _seal(ks)
    layouts = MappingProxyType(dict(meta["layouts"]["all"]))
    identities = ExpectedHeldIdentities(cfg.sha256, col.sha256, op.sha256, None if pref is None else pref.sha256)
    generation, witness, area = int(meta["facts"]["generation"]), meta["facts"]["source_witness"], float(sc["cell_area_m2"])
    with _minting():  # tight window: every argument precomputed, no callbacks
        return HeldContext(cfg, col, op, pref, depth0, ks_sealed, active, area, generation, witness, identities, layouts)


def _mint_cuda_context(parts: Any, meta: dict) -> HeldContext:
    """Checks the plain parts returned by the CUDA preparation (exact types, CUDA backend, identities equal to the record's) and mints the context in a tight window with precomputed arguments."""
    if type(parts) is not dict or set(parts) != {"config", "column", "matrix_op", "preferential", "depth0", "ks", "active", "edges", "host_depth0", "host_active", "host_ks", "area"}:
        raise HeldContextError("internal: the CUDA parts are not the closed plain record")
    cfg, col, op, pref = parts["config"], parts["column"], parts["matrix_op"], parts["preferential"]
    ident = meta["identities"]
    if (type(cfg) is not nfs.NativeFloatingConfig or type(col) is not fsl.FloatingSoilColumn or col.backend != "cuda" or cfg.backend != "cuda"
            or (cfg.sha256, col.sha256, op.sha256, None if pref is None else pref.sha256) != (ident["config_sha256"], ident["root_column_sha256"], ident["matrix_op_sha256"], ident["pref_sha256"])):
        raise HeldContextError("the CUDA parts do not reproduce the recorded CUDA identities")
    identities = ExpectedHeldIdentities(cfg.sha256, col.sha256, op.sha256, None if pref is None else pref.sha256)
    layouts = MappingProxyType(dict(meta["layouts"]["all"]))
    generation, witness, area = int(meta["facts"]["generation"]), meta["facts"]["source_witness"], float(parts["area"])
    depth0, ks, active, edges, hd, ha, hk = parts["depth0"], parts["ks"], parts["active"], parts["edges"], parts["host_depth0"], parts["host_active"], parts["host_ks"]
    with _minting():  # tight window: every argument precomputed, no callbacks
        return HeldContext(cfg, col, op, pref, depth0, ks, active, area, generation, witness, identities, layouts, "cuda", edges, hd, ha, hk)
