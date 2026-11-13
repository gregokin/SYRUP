"""Configuration-driven native layered-water setup (CPU NumPy / Numba; Dense native 9f preparation only).

A strict, versioned, bounded user parameter file (`maple-syrup-layered-case/1`, YAML) plus a public loader and a cold factory that composes the EXISTING
production constructors on a GENUINE `native_dense_profile.NativeDensePreparation`:

    config = load_layered_config("layered.yaml")
    plan   = plan_layered_setup(prep, config, caller_reserve_bytes=..., lifecycle_extra_reserve_bytes=...)
    setup  = prepare_layered_case(prep, config, caller_reserve_bytes=..., lifecycle_extra_reserve_bytes=...)
    setup.state            # the genuine initial BOUNDARY lifecycle state (one water authority: the native bed's depth H + the soil column)
    setup.lifecycle_operator, setup.texture, setup.identities, setup.plan

MAPLE initializes terrain/grains/voxels/active layer/native water and the latitude (`prep.latitude_binding()`); LEAF initializes plants separately. This module only
adds the water-specific setup: the configured soil layers, their retained actual texture, the explicit matrix/preferential/evaporation parameters, the prescribed
cover, initial moisture and persistent history, and the explicit hydraulic domain. It is not a scheduler, not a native/plant initializer, not a CLI, and adds no water
physics. GPU is a typed unsupported choice here; nothing imports CuPy. All sizes are explicit; there is no silent fallback and no default reserve.

Memory (see `plan_layered_setup`): ONE staged metadata admission (no N-sized array before it) is read BEFORE the cold source recheck, any domain staging, the
texture, the routing surface, the pond topology/graph or any owner; its topology/graph term is a code-derived bound of the real builders, and their actual retained
bytes only verify it afterwards. The parameter-file loader admits its YAML stages and every map body separately (fresh availability reading + explicit reserve) before
reading it. The config is a sealed, deeply immutable, authenticated record (re-validated at every entry). The setup never writes to the caller's native arrays. The
returned state aliases the native bed's water depth (one authority) and so keeps the preparation's `BedState` alive; the preparation object / compiled case is not
retained by the result.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import stat
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

import numpy as np
import yaml

from maple_syrup import layered_checkpoint as lcp
from maple_syrup import layered_storm as ls
from maple_syrup import matrix_infiltration as mi
from maple_syrup import native_dense_profile as nd
from maple_syrup import pond_network as pn
from maple_syrup import preferential_infiltration as pi
from maple_syrup import sediment_bed as sb
from maple_syrup import soil_evaporation as se
from maple_syrup import soil_profile_texture as st
from maple_syrup import subsurface as ss
from maple_syrup import water_lifecycle as wl
from maple_syrup.experimental_hydrology import HydraulicControl
from maple_syrup.pond_hydraulics import metadata_routing_graph
from maple_syrup.surface_routing import build_surface_router

__all__ = [
    "BACKENDS", "CUDA_BACKEND", "SCHEMA", "LayeredCaseConfig", "LayeredCaseConfigError", "LayeredCaseError", "LayeredCaseLimits", "LayeredCaseMemoryError",
    "LayeredCaseSetup", "LayeredCaseUnsupportedError", "MapRef", "load_layered_config", "load_layered_cuda_config", "parse_layered_config",
    "parse_layered_cuda_config", "plan_layered_setup", "prepare_layered_case",
]

SCHEMA = "maple-syrup-layered-case/1"
BACKENDS = ("numpy", "numba")
_UNSUPPORTED_BACKENDS = ("cuda", "cupy", "gpu")
CUDA_BACKEND = "cuda"  # the ONLY value the explicit GPU parse profile accepts (see `parse_layered_cuda_config`)
_F8 = 8


class LayeredCaseError(RuntimeError):
    """The setup was refused or failed; the caller's native preparation, bed, depth, material and files are unchanged."""


class LayeredCaseConfigError(LayeredCaseError):
    """The parameter file / its referenced maps are malformed, out of bounds, incomplete or not representable."""


class LayeredCaseUnsupportedError(LayeredCaseError):
    """A requested capability (GPU backend, stage/base-pressure/overflow boundary, optional native state) is not supported by this setup slice."""


class LayeredCaseMemoryError(LayeredCaseError, MemoryError):
    """The staged memory plan exceeds the reported availability; nothing protected was done."""


_SEAL = object()
NODE_BYTES = 256  # per parsed YAML node (the convention of layered_checkpoint.NODE_BYTES: a CPython dict entry/str/int record)
_MAP_RECORD_BYTES = 4096  # the MapRef record, its path/name strings and one bounded header read
_HEADER_READ_BYTES = 4096 + 16  # numpy header cap + magic/version prefix


@dataclass(frozen=True)
class LayeredCaseLimits:
    """Declared bounds of the user parameter file and its map references. Every field is validated as a typed positive integer and honored
    exactly (nothing is silently raised); `max_map_bytes` caps each map file AND the total of all map files."""

    max_yaml_bytes: int = 64 * 1024
    max_yaml_nodes: int = 5000
    max_yaml_depth: int = 8
    max_layers: int = 64
    max_species: int = 16
    max_outlets: int = 4096
    max_map_files: int = 32
    max_map_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if type(v) is not int or v < 1:
                raise LayeredCaseConfigError(f"limit {f.name} must be a positive integer (bool, float and non-positive values refuse), got {v!r}")


def _limits(limits: Any) -> LayeredCaseLimits:
    if limits is None:
        return LayeredCaseLimits()
    if type(limits) is not LayeredCaseLimits:
        raise LayeredCaseConfigError("limits must be a LayeredCaseLimits")
    return limits


class _Admit:
    """Loader-side memory admission: a FRESH availability reading per stage, the explicit reserve, a typed refusal before the guarded allocation. An outer
    admission is not a reservation: it checks the reading at that instant only."""

    def __init__(self, available_bytes_fn: Callable[[], int] | None, reserve_bytes: Any):
        if isinstance(reserve_bytes, bool) or not isinstance(reserve_bytes, (int, np.integer)) or int(reserve_bytes) < 0:
            raise LayeredCaseConfigError("reserve_bytes must be an explicit non-negative integer")
        if available_bytes_fn is not None and not callable(available_bytes_fn):
            raise LayeredCaseConfigError("available_bytes_fn must be callable")
        self.fn = ss.default_host_available_bytes if available_bytes_fn is None else available_bytes_fn
        self.reserve = int(reserve_bytes)
        self.log: list[tuple[str, int]] = []

    def require(self, nbytes: int, what: str) -> None:
        avail = self.fn()
        if isinstance(avail, bool) or not isinstance(avail, (int, np.integer)) or int(avail) < 0:
            raise LayeredCaseConfigError(f"available_bytes_fn must return a non-negative integer, got {avail!r}")
        self.log.append((what, int(nbytes)))
        if int(nbytes) + self.reserve > int(avail):
            raise LayeredCaseMemoryError(f"{what} needs an estimated {int(nbytes)} B (+ reserve {self.reserve} B) but {int(avail)} B are reported available; "
                                         "nothing guarded was read or allocated")


def yaml_stage_bytes(raw_len: int, nodes: int | None = None) -> int:
    """Named working bytes of the YAML stage: raw bytes + decoded text (before the parse); with the event-audited node count, + `nodes * NODE_BYTES` parsed records."""
    return 2 * int(raw_len) + (0 if nodes is None else int(nodes) * NODE_BYTES)


def map_load_working_bytes(file_bytes: int, n_elements: int, kind: str) -> int:
    """Named working bytes of loading ONE map, admitted before its body is read: the raw file bytes (retained as the immutable backing of the array),
    the finite/range validation scratch (float maps: two `n_elements`-byte bool temporaries; masks: none), one bounded header read and the record."""
    scratch = 2 * int(n_elements) if kind == "float" else 0
    return int(file_bytes) + scratch + _HEADER_READ_BYTES + _MAP_RECORD_BYTES


class MapRef:
    """A bounded `.npy` per-cell map (float64 or bool, 2-D `(ny, nx)`). The array is a read-only view of an IMMUTABLE `bytes` object (the file content), so
    `setflags(write=True)` raises; the record is created only by the loader (sealed) and cannot be rebuilt or `dataclasses.replace`d."""

    __slots__ = ("name", "path", "sha256", "array", "kind", "n_true", "__weakref__")

    def __init__(self, seal: object, name: str, path: str, sha256: str, array: np.ndarray, kind: str, n_true: int | None):
        if seal is not _SEAL:
            raise LayeredCaseConfigError("a MapRef is created only by the parameter-file loader")
        for k, v in (("name", name), ("path", path), ("sha256", sha256), ("array", array), ("kind", kind), ("n_true", n_true)):
            object.__setattr__(self, k, v)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("MapRef is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("MapRef is immutable")

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.array.shape)

    def __repr__(self) -> str:
        return f"MapRef({self.name!r}, kind={self.kind}, shape={self.shape}, sha256={self.sha256[:12]}...)"


def _bytes_backed(arr: Any) -> bool:
    """True when the array's base chain ends at an immutable `bytes` object and it is not writeable."""
    if not isinstance(arr, np.ndarray) or arr.flags.writeable:
        return False
    base = arr
    for _ in range(8):
        base = base.base
        if base is None:
            return False
        if isinstance(base, bytes):
            return True
    return False


# --- strict YAML ---------------------------------------------------------------------------------------------------------------------------
class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that refuses duplicate mapping keys and non-string keys."""

    def construct_mapping(self, node, deep=False):  # noqa: D401
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=True)
            if not isinstance(key, str):
                raise yaml.YAMLError(f"mapping keys must be strings, got {key!r}")
            if key in seen:
                raise yaml.YAMLError(f"duplicate key {key!r}")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _bounded_yaml(data: bytes, limits: LayeredCaseLimits, what: str, admit: _Admit | None = None) -> Any:
    if len(data) > limits.max_yaml_bytes:
        raise LayeredCaseConfigError(f"{what} is {len(data)} bytes, over the {limits.max_yaml_bytes}-byte cap")
    if admit is not None:
        admit.require(yaml_stage_bytes(len(data)), "YAML decode")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LayeredCaseConfigError(f"{what} is not UTF-8: {exc}") from None
    nodes = depth = 0
    try:
        for ev in yaml.parse(text, Loader=_StrictLoader):
            if isinstance(ev, yaml.AliasEvent) or getattr(ev, "anchor", None) or getattr(ev, "tag", None):
                raise LayeredCaseConfigError(f"{what}: YAML anchors, aliases and tags are refused")
            if isinstance(ev, (yaml.ScalarEvent, yaml.SequenceStartEvent, yaml.MappingStartEvent)):
                nodes += 1
                if nodes > limits.max_yaml_nodes:
                    raise LayeredCaseConfigError(f"{what}: more than {limits.max_yaml_nodes} YAML nodes")
            if isinstance(ev, (yaml.SequenceStartEvent, yaml.MappingStartEvent)):
                depth += 1
                if depth > limits.max_yaml_depth:
                    raise LayeredCaseConfigError(f"{what}: YAML deeper than {limits.max_yaml_depth}")
            elif isinstance(ev, (yaml.SequenceEndEvent, yaml.MappingEndEvent)):
                depth -= 1
        if admit is not None:
            admit.require(yaml_stage_bytes(len(data), nodes), "YAML parse")
        doc = yaml.load(text, Loader=_StrictLoader)
    except yaml.YAMLError as exc:
        raise LayeredCaseConfigError(f"{what}: invalid YAML: {exc}") from None
    if not isinstance(doc, dict):
        raise LayeredCaseConfigError(f"{what} must parse to a mapping, got {type(doc).__name__}")
    return doc


def _read_regular(path: Path, cap: int, what: str, admit: Callable[[int], None] | None = None) -> bytes:
    """Regular non-symlink FINAL component, size from `fstat` checked against the cap (and the optional admission) BEFORE any byte is read."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise LayeredCaseConfigError(f"{what} {path} cannot be opened as a regular non-symlink file: {exc}") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise LayeredCaseConfigError(f"{what} {path} is not a regular file")
        if info.st_size > cap:
            raise LayeredCaseConfigError(f"{what} {path} is {info.st_size} bytes, over its {cap}-byte cap")
        if admit is not None:
            admit(int(info.st_size))
        data = b""
        while len(data) <= info.st_size:
            more = os.read(fd, info.st_size + 1 - len(data))
            if not more:
                break
            data += more
    finally:
        os.close(fd)
    if len(data) != info.st_size:
        raise LayeredCaseConfigError(f"{what} {path} changed size while being read")
    return data


# --- strict value helpers ------------------------------------------------------------------------------------------------------------------
def _keys(d: Any, required: tuple, optional: tuple, where: str) -> dict:
    if not isinstance(d, dict):
        raise LayeredCaseConfigError(f"{where} must be a mapping, got {type(d).__name__}")
    unknown = sorted(set(d) - set(required) - set(optional))
    missing = sorted(set(required) - set(d))
    if unknown:
        raise LayeredCaseConfigError(f"{where}: unknown key(s) {unknown} (unsupported options are refused, never ignored or approximated)")
    if missing:
        raise LayeredCaseConfigError(f"{where}: missing required key(s) {missing}")
    return d


def _num(v: Any, where: str, *, lo: float | None = None, hi: float | None = None, lo_open: bool = False, hi_open: bool = False) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        hint = " (YAML needs a decimal point for 1.0e-5; a bare 1e-5 is a string)" if isinstance(v, str) else ""
        raise LayeredCaseConfigError(f"{where} must be a number, got {type(v).__name__}{hint}")
    x = float(v)
    if not math.isfinite(x):
        raise LayeredCaseConfigError(f"{where} must be finite")
    if lo is not None and (x < lo or (lo_open and x == lo)):
        raise LayeredCaseConfigError(f"{where} must be {'>' if lo_open else '>='} {lo}, got {x}")
    if hi is not None and (x > hi or (hi_open and x == hi)):
        raise LayeredCaseConfigError(f"{where} must be {'<' if hi_open else '<='} {hi}, got {x}")
    return x


def _vec(v: Any, where: str, n: int | None = None, *, lo: float | None = None, hi: float | None = None, limit: int = 4096) -> tuple[float, ...]:
    if not isinstance(v, list) or not v:
        raise LayeredCaseConfigError(f"{where} must be a non-empty list of numbers")
    if len(v) > limit or (n is not None and len(v) != n):
        raise LayeredCaseConfigError(f"{where} must have {n if n is not None else f'<= {limit}'} entries, got {len(v)}")
    return tuple(_num(x, f"{where}[{i}]", lo=lo, hi=hi) for i, x in enumerate(v))


def _int(v: Any, where: str, lo: int = 0) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise LayeredCaseConfigError(f"{where} must be an integer >= {lo}, got {v!r}")
    return int(v)


def _bool(v: Any, where: str) -> bool:
    if not isinstance(v, bool):
        raise LayeredCaseConfigError(f"{where} must be true or false, got {v!r}")
    return v


def _str_in(v: Any, where: str, allowed: tuple) -> str:
    if not isinstance(v, str) or v not in allowed:
        raise LayeredCaseConfigError(f"{where} must be one of {allowed}, got {v!r}")
    return v


def _npy_header(head: bytes, where: str, kind: str) -> tuple[tuple[int, int], np.dtype, int]:
    """Header preflight on the first bytes only (magic, version, bounded header, dtype, order, rank). Returns (shape, dtype, data offset)."""
    import io

    from numpy.lib import format as npf

    f = io.BytesIO(head)
    try:
        version = npf.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = npf.read_array_header_1_0(f, max_header_size=4096)
        elif version == (2, 0):
            shape, fortran, dtype = npf.read_array_header_2_0(f, max_header_size=4096)
        else:
            raise ValueError(f"unsupported .npy version {version}")
    except Exception as exc:  # noqa: BLE001
        raise LayeredCaseConfigError(f"{where}: invalid .npy header: {exc}") from None
    want = np.dtype("<f8") if kind == "float" else np.dtype("|b1")
    if dtype != want or dtype.hasobject:
        raise LayeredCaseConfigError(f"{where}: map dtype must be {want}, got {dtype}")
    if fortran or len(shape) != 2 or any(int(s) < 1 for s in shape):
        raise LayeredCaseConfigError(f"{where}: map must be a C-order 2-D (ny, nx) array, got shape {shape}, fortran_order={fortran}")
    return (int(shape[0]), int(shape[1])), dtype, int(f.tell())


class _Maps:
    """Bounded `.npy` map loader. Name: ONE path component (no separator, `..`, absolute, `~`, backslash, NUL) opened relative to the parameter directory's
    descriptor with `O_NOFOLLOW`, so no intermediate component is ever traversed and the final component may not be a symlink. Per map, BEFORE its body is read:
    header preflight on the first bytes, exact-size agreement, file-count and remaining-total caps against the `fstat` size, then the working-memory admission."""

    def __init__(self, base_dir: Path | None, limits: LayeredCaseLimits, admit: _Admit | None):
        self.limits, self.admit, self.count, self.total, self.cache = limits, admit, 0, 0, {}
        self.dirfd: int | None = None
        self.base = base_dir
        if base_dir is not None:
            try:
                self.dirfd = os.open(base_dir, os.O_RDONLY | os.O_DIRECTORY)  # the user's explicit parameter directory (may itself be a link target)
            except OSError as exc:
                raise LayeredCaseConfigError(f"parameter directory {base_dir} cannot be opened: {exc}") from None

    def close(self) -> None:
        if self.dirfd is not None:
            os.close(self.dirfd)
            self.dirfd = None

    def load(self, spec: Any, where: str, kind: str) -> MapRef:
        spec = _keys(spec, ("file",), ("sha256",), where)
        name = spec["file"]
        if (not isinstance(name, str) or not name or name in (".", "..") or "/" in name or "\\" in name or name.startswith("~") or "\x00" in name
                or len(name) > 255):
            raise LayeredCaseConfigError(f"{where}.file must be ONE plain file name (no directory component, `..`, absolute path, `~` or backslash) "
                                         "inside the parameter file's directory")
        if self.dirfd is None:
            raise LayeredCaseConfigError(f"{where}: map references need the parameter file's directory (load_layered_config)")
        key = (name, kind)
        if key in self.cache:
            ref = self.cache[key]
        else:
            if self.count >= self.limits.max_map_files:
                raise LayeredCaseConfigError(f"more than {self.limits.max_map_files} map files")
            ref = self._read(name, where, kind)
            self.cache[key] = ref
        want = spec.get("sha256")
        if want is not None and want != ref.sha256:
            raise LayeredCaseConfigError(f"{where}.sha256 does not match the file content ({ref.sha256})")
        return ref

    def _read(self, name: str, where: str, kind: str) -> MapRef:
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.dirfd)
        except OSError as exc:
            raise LayeredCaseConfigError(f"{where} map {name!r} cannot be opened as a regular non-symlink file in the parameter directory: {exc}") from None
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise LayeredCaseConfigError(f"{where} map {name!r} is not a regular file")
            size = int(info.st_size)
            if size > self.limits.max_map_bytes:
                raise LayeredCaseConfigError(f"{where} map {name!r} is {size} bytes, over its {self.limits.max_map_bytes}-byte cap")
            head = os.read(fd, min(size, _HEADER_READ_BYTES))
            shape, dtype, offset = _npy_header(head, where, kind)
            n = shape[0] * shape[1]
            if offset + n * dtype.itemsize != size:
                raise LayeredCaseConfigError(f"{where}: .npy data size {size - offset} B differs from its header ({n * dtype.itemsize} B)")
            if self.total + size > self.limits.max_map_bytes:  # remaining-total cap BEFORE the body is read
                raise LayeredCaseConfigError(f"map files would exceed the {self.limits.max_map_bytes}-byte total cap ({self.total} B already loaded + {size} B)")
            if self.admit is not None:
                self.admit.require(map_load_working_bytes(size, n, kind), f"map {name!r} load")
            os.lseek(fd, 0, os.SEEK_SET)
            raw = _read_body(fd, size)
        finally:
            os.close(fd)
        self.count += 1
        self.total += size
        arr = np.frombuffer(raw, dtype=dtype, count=n, offset=offset).reshape(shape)  # a view of the immutable bytes: not re-enableable for writing
        n_true = None
        if kind == "float":
            if not np.all(np.isfinite(arr)):
                raise LayeredCaseConfigError(f"{where}: map holds non-finite values")
        else:
            n_true = int(np.count_nonzero(arr))
        return MapRef(_SEAL, name, f"<parameter directory>/{name}", hashlib.sha256(raw).hexdigest(), arr, kind, n_true)


def _read_body(fd: int, size: int) -> bytes:
    """ONE bounded read of the whole (already admitted, size-checked) file; growth beyond `size` is refused."""
    with os.fdopen(os.dup(fd), "rb", buffering=0) as fh:
        raw = fh.read(size + 1)
    if len(raw) != size:
        raise LayeredCaseConfigError("a map file changed size while being read")
    return raw


def _scalar_or_map(v: Any, where: str, maps: _Maps, **bounds) -> float | MapRef:
    if isinstance(v, dict):
        ref = maps.load(v, where, "float")
        arr = ref.array
        lo, hi = bounds.get("lo"), bounds.get("hi")
        if (lo is not None and np.any(arr < lo)) or (hi is not None and np.any(arr > hi)) or (bounds.get("lo_open") and lo is not None and np.any(arr <= lo)):
            raise LayeredCaseConfigError(f"{where}: map values outside the allowed range")
        return ref
    return _num(v, where, **bounds)


# --- config -----------------------------------------------------------------------------------------------------------------------------------
_MINTED: "weakref.WeakSet" = weakref.WeakSet()
_CFG_FIELDS = ("backend", "layers", "spec", "matrix", "preferential", "partition", "cover", "moisture_units", "moisture_by_layer", "wetting_history_m",
               "cumulative_intake_m", "friction_factor", "active_mask", "closed_ring_rise_m", "outlets", "identity", "source_path")


@dataclass(frozen=True, eq=False, init=False)
class LayeredCaseConfig:
    """Validated, DEEPLY immutable parameters of one layered-water setup. Built only by `parse_layered_config` (sealed constructor: direct construction and
    `dataclasses.replace` raise); nested records are `MappingProxyType`/tuples; map arrays are views of immutable `bytes`. Every public entry re-validates the
    record (`_verify_config`: minted, identity re-derived from the CURRENT fields, layers re-derived from the echoed soil parameters, map backing)."""

    backend: str
    layers: ss.SoilLayers
    spec: Any  # frozen normalised soil echo (the layers are re-derived from it)
    matrix: tuple  # (conductivity_m_s, suction_m, vegetation_factor): float | MapRef
    preferential: Any  # frozen mapping of normalised preferential parameters
    partition: se.EvaporationPartition
    cover: float | MapRef
    moisture_units: str
    moisture_by_layer: tuple
    wetting_history_m: float | MapRef
    cumulative_intake_m: float | MapRef
    friction_factor: float | MapRef
    active_mask: MapRef | None  # None = every cell active
    closed_ring_rise_m: float
    outlets: tuple  # ((row, col, d_row, d_col, drop_m), ...)
    identity: Any  # frozen {"config_sha256", "yaml_sha256", "map_files"}
    source_path: str | None

    def __init__(self, seal: object = None, **kw: Any) -> None:
        if seal is not _SEAL:
            raise LayeredCaseConfigError("a LayeredCaseConfig is created only by parse_layered_config / load_layered_config (dataclasses.replace and direct "
                                         "construction are refused: they would keep a stale identity)")
        if set(kw) != set(_CFG_FIELDS):
            raise LayeredCaseConfigError("internal: config fields mismatch")
        for k, v in kw.items():
            object.__setattr__(self, k, v)
        _MINTED.add(self)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    @property
    def n_layers(self) -> int:
        return int(self.layers.n_layers)

    @property
    def n_species(self) -> int:
        return len(self.preferential["species"])

    def n_active(self, n_cells: int) -> int:
        """Certified active-cell count of an `n_cells` grid from metadata only (`all` = every cell; a mask map's count was taken at load)."""
        return int(n_cells) if self.active_mask is None else int(self.active_mask.n_true)

    def maps(self) -> list[MapRef]:
        out: list[MapRef] = []

        def walk(o):
            if isinstance(o, MapRef):
                out.append(o)
            elif isinstance(o, Mapping):
                for x in o.values():
                    walk(x)
            elif isinstance(o, (tuple, list)):
                for x in o:
                    walk(x)
        for o in (self.matrix, self.preferential, self.cover, self.wetting_history_m, self.cumulative_intake_m, self.friction_factor, self.active_mask):
            walk(o)
        return out


def _freeze(o: Any) -> Any:
    if isinstance(o, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in o.items()})
    if isinstance(o, (list, tuple)):
        return tuple(_freeze(x) for x in o)
    return o


def _canon(o: Any) -> Any:
    if isinstance(o, MapRef):
        return {"map_file": o.name, "sha256": o.sha256, "shape": list(o.shape), "kind": o.kind}
    if isinstance(o, Mapping):
        return {k: _canon(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_canon(x) for x in o]
    if isinstance(o, float):
        return {"f": o.hex()}
    return o


def _config_sha(backend, soil_echo, matrix, pref, partition, cover, units, by_layer, wet, cum, friction, mask, rise, outlets) -> str:
    cfg = {"schema": SCHEMA, "backend": backend, "soil": soil_echo, "matrix": matrix, "preferential": pref, "evaporation": dataclasses.asdict(partition),
           "cover": cover, "initial": {"units": units, "by_layer": by_layer, "wetting_history_m": wet, "cumulative_intake_m": cum},
           "domain": {"active_mask": mask, "friction_factor": friction, "closed_ring_rise_m": rise, "outlets": [list(o) for o in outlets]}}
    return hashlib.sha256(json.dumps(_canon(cfg), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _make_layers(echo: Any) -> ss.SoilLayers:
    d = echo["diffusion"]
    return ss.soil_layers(list(echo["edges_m"]), theta_residual=echo["theta_residual"], theta_field_capacity=echo["theta_field_capacity"],
                          theta_saturation=echo["theta_saturation"], theta_evaporation_stress=echo["theta_evaporation_stress"], evaporation_factor=1.0,
                          gravity_drainage=echo["gravity_drainage"], bottom_drainage=echo["bottom"], **dict(d))


def _same_layers(a: Any, b: Any) -> bool:
    if type(a) is not type(b) or not dataclasses.is_dataclass(a):
        return False
    for f in dataclasses.fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        try:
            if not np.array_equal(np.asarray(x), np.asarray(y)):
                return False
        except Exception:  # noqa: BLE001
            return False
    return True


def _verify_config(cfg: Any) -> None:
    """Public-entry authentication BEFORE any protected work: minted record, immutable map backing, canonical identity re-derived from the CURRENT fields and
    the layers re-derived through the production validator from the echoed soil parameters. A forged, replaced or stale config refuses."""
    if type(cfg) is not LayeredCaseConfig or cfg not in _MINTED:
        raise LayeredCaseConfigError("config must be a LayeredCaseConfig produced by the parameter-file loader (a forged or rebuilt record is refused)")
    try:
        maps = cfg.maps()
        for m in maps:
            if type(m) is not MapRef or not _bytes_backed(m.array) or m.array.dtype not in (np.dtype("<f8"), np.dtype("|b1")) or m.array.ndim != 2:
                raise LayeredCaseConfigError(f"map {getattr(m, 'name', '?')!r} is not an immutable bytes-backed array")
        if cfg.identity["map_files"] != {m.name: m.sha256 for m in maps}:
            raise LayeredCaseConfigError("the map files of the config differ from its identity")
        if _config_sha(cfg.backend, cfg.spec, cfg.matrix, cfg.preferential, cfg.partition, cfg.cover, cfg.moisture_units, cfg.moisture_by_layer,
                       cfg.wetting_history_m, cfg.cumulative_intake_m, cfg.friction_factor, cfg.active_mask, cfg.closed_ring_rise_m, cfg.outlets) \
                != cfg.identity["config_sha256"]:
            raise LayeredCaseConfigError("the config content no longer matches its identity (stale identity); re-parse the parameter file")
        if not _same_layers(cfg.layers, _make_layers(cfg.spec)):
            raise LayeredCaseConfigError("the config's soil layers differ from its echoed soil parameters")
    except LayeredCaseConfigError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise LayeredCaseConfigError(f"config is not a valid loader product: {type(exc).__name__}: {exc}") from None


def _require_cpu_config(cfg: Any) -> None:
    """EARLY typed refusal (before the preparation, any array, hash or device work) of a GPU-parsed config reaching the CPU plan/factory: the CPU setup is NumPy /
    Numba only and never relabels or reinterprets a `cuda` record (`layered_case_cuda.prepare_layered_cuda_case` is its consumer)."""
    if cfg.backend not in BACKENDS:
        raise LayeredCaseUnsupportedError(f"this CPU setup does not accept a {cfg.backend!r} config; use layered_case_cuda.prepare_layered_cuda_case with a "
                                          "NativeDenseCudaPreparation (the config's backend is never relabelled)")


def load_layered_config(path: str | os.PathLike, *, limits: LayeredCaseLimits | None = None, available_bytes_fn: Callable[[], int] | None = None,
                        reserve_bytes: int = 0) -> LayeredCaseConfig:  # CPU parse profile: `compute.backend: cuda` is refused (see `load_layered_cuda_config`)
    """Read (regular non-symlink final component, `fstat` size checked before the read, admitted) and parse a parameter file; map references are single
    file names resolved relative to its directory (the user's explicit choice). `available_bytes_fn` (default: the native available-host helper) and
    `reserve_bytes` admit the loader's YAML and per-map working memory (fresh reading per stage); the caller's own already-held inputs are not counted."""
    return _load(path, limits, available_bytes_fn, reserve_bytes, "cpu")


def _load(path: Any, limits: Any, available_bytes_fn: Any, reserve_bytes: Any, profile: str) -> LayeredCaseConfig:
    limits = _limits(limits)
    admit = _Admit(available_bytes_fn, reserve_bytes)
    p = Path(path)
    data = _read_regular(p, limits.max_yaml_bytes, "layered case file", admit=lambda size: admit.require(size, "YAML read"))
    return _parse(data, base_dir=p.resolve().parent, limits=limits, source_path=str(p), admit=admit, profile=profile)


def parse_layered_config(data: bytes | str, *, base_dir: str | os.PathLike | None = None, limits: LayeredCaseLimits | None = None,
                         source_path: str | None = None, available_bytes_fn: Callable[[], int] | None = None, reserve_bytes: int = 0) -> LayeredCaseConfig:
    """Parse already-provided YAML text/bytes (its own bytes are the caller's and are not counted) with the loader's admission of the YAML stages and every map."""
    return _parse(data, base_dir=base_dir, limits=_limits(limits), source_path=source_path, admit=_Admit(available_bytes_fn, reserve_bytes), profile="cpu")


def load_layered_cuda_config(path: str | os.PathLike, *, limits: LayeredCaseLimits | None = None, available_bytes_fn: Callable[[], int] | None = None,
                             reserve_bytes: int = 0) -> LayeredCaseConfig:
    """The EXPLICIT GPU parse profile of `load_layered_config`: same schema, validators, bounds and admission, but `compute.backend` must be exactly `cuda`
    (a `numpy`/`numba` file is refused here and a `cuda` file is refused by the CPU loader: the file's backend is never rewritten, relabelled or forced). The
    sealed record keeps `backend == "cuda"` in its identity; only `layered_case_cuda` consumes it. Nothing imports CuPy and no device is touched."""
    return _load(path, limits, available_bytes_fn, reserve_bytes, "cuda")


def parse_layered_cuda_config(data: bytes | str, *, base_dir: str | os.PathLike | None = None, limits: LayeredCaseLimits | None = None,
                              source_path: str | None = None, available_bytes_fn: Callable[[], int] | None = None, reserve_bytes: int = 0) -> LayeredCaseConfig:
    """GPU counterpart of `parse_layered_config` (see `load_layered_cuda_config`)."""
    return _parse(data, base_dir=base_dir, limits=_limits(limits), source_path=source_path, admit=_Admit(available_bytes_fn, reserve_bytes), profile="cuda")


def _parse(data: bytes | str, *, base_dir: Any, limits: LayeredCaseLimits, source_path: str | None, admit: _Admit, profile: str = "cpu") -> LayeredCaseConfig:
    raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    maps = _Maps(None if base_dir is None else Path(base_dir), limits, admit)
    try:
        return _parse_body(raw, maps, limits, source_path, admit, profile)
    finally:
        maps.close()


def _parse_body(raw: bytes, maps: _Maps, limits: LayeredCaseLimits, source_path: str | None, admit: _Admit, profile: str = "cpu") -> LayeredCaseConfig:
    doc = _bounded_yaml(raw, limits, "layered case", admit)
    top = _keys(doc, ("schema", "compute", "soil", "matrix", "preferential", "evaporation", "cover", "initial", "domain"), (), "layered case")
    if top["schema"] != SCHEMA:
        raise LayeredCaseConfigError(f"schema must be {SCHEMA!r}, got {top['schema']!r}")
    comp = _keys(top["compute"], ("backend",), (), "compute")
    backend = comp["backend"]
    if profile == "cuda":
        if backend != CUDA_BACKEND or type(backend) is not str:
            raise LayeredCaseConfigError(f"the GPU parse profile requires compute.backend to be exactly {CUDA_BACKEND!r}, got {backend!r} (a numpy/numba file "
                                         "is parsed by load_layered_config; the file's backend is never rewritten or forced)")
    elif profile != "cpu":
        raise LayeredCaseConfigError("internal: unknown parse profile")
    else:
        if isinstance(backend, str) and backend.lower() in _UNSUPPORTED_BACKENDS:
            raise LayeredCaseUnsupportedError("a GPU backend is not supported by this setup slice (no CuPy is imported; the schema is backend-neutral and a "
                                              "qualified GPU owner/case profile will add a value, not new data)")
        backend = _str_in(backend, "compute.backend", BACKENDS)

    # soil ---------------------------------------------------------------------------------------------------------------------------------
    soil = _keys(top["soil"], ("layers", "theta_residual", "theta_field_capacity", "theta_saturation", "theta_evaporation_stress",
                               "gravity_drainage", "bottom", "diffusion"), (), "soil")
    lay = _keys(soil["layers"], (), ("edges_m", "thickness_m"), "soil.layers")
    if ("edges_m" in lay) == ("thickness_m" in lay):
        raise LayeredCaseConfigError("soil.layers needs exactly one of edges_m (starting at 0) or thickness_m")
    if "edges_m" in lay:
        edges = _vec(lay["edges_m"], "soil.layers.edges_m", limit=limits.max_layers + 1, lo=0.0)
    else:
        thick = _vec(lay["thickness_m"], "soil.layers.thickness_m", limit=limits.max_layers, lo=0.0)
        edges = (0.0, *np.cumsum(np.array(thick)).tolist())  # sequential cumulative sum (declared); validated strictly increasing below
    if len(edges) < 2 or len(edges) - 1 > limits.max_layers:
        raise LayeredCaseConfigError(f"soil needs 1..{limits.max_layers} layers")
    L = len(edges) - 1

    def per_layer(key):
        v = soil[key]
        return _num(v, f"soil.{key}", lo=0.0, hi=1.0) if not isinstance(v, list) else _vec(v, f"soil.{key}", L, lo=0.0, hi=1.0)
    res, fc, sat = per_layer("theta_residual"), per_layer("theta_field_capacity"), per_layer("theta_saturation")
    stress = _num(soil["theta_evaporation_stress"], "soil.theta_evaporation_stress", lo=0.0, hi=1.0)
    gravity = _bool(soil["gravity_drainage"], "soil.gravity_drainage")
    bottom = _str_in(soil["bottom"], "soil.bottom", ss.BOTTOM_DRAINAGE_MODES)
    dif = soil["diffusion"]
    if not isinstance(dif, dict) or dif.get("mode") not in ss.DIFFUSION_MODES:
        raise LayeredCaseConfigError(f"soil.diffusion.mode must be one of {ss.DIFFUSION_MODES}")
    if dif["mode"] == "constant":
        _keys(dif, ("mode", "diffusivity_m2_s"), (), "soil.diffusion")
        dv = dif["diffusivity_m2_s"]
        dkw = dict(diffusion_mode="constant", diffusivity_m2_s=_num(dv, "soil.diffusion.diffusivity_m2_s", lo=0.0) if not isinstance(dv, list)
                   else _vec(dv, "soil.diffusion.diffusivity_m2_s", L - 1, lo=0.0))
    else:
        _keys(dif, ("mode", "interface_conductivity_m_s", "coefficient"), (), "soil.diffusion")
        kv = dif["interface_conductivity_m_s"]
        dkw = dict(diffusion_mode="tietjen_donor_depth",
                   interface_conductivity_m_s=_num(kv, "soil.diffusion.interface_conductivity_m_s", lo=0.0) if not isinstance(kv, list)
                   else _vec(kv, "soil.diffusion.interface_conductivity_m_s", L - 1, lo=0.0),
                   diffusion_coefficient=_num(dif["coefficient"], "soil.diffusion.coefficient", lo=0.0))
    soil_echo = _freeze({"edges_m": list(edges), "theta_residual": res, "theta_field_capacity": fc, "theta_saturation": sat,
                         "theta_evaporation_stress": stress, "gravity_drainage": gravity, "bottom": bottom, "diffusion": dict(dkw)})
    try:
        layers = _make_layers(soil_echo)
    except ss.SubsurfaceError as exc:
        raise LayeredCaseConfigError(f"soil layers refused by the production validator: {exc}") from None

    # matrix -------------------------------------------------------------------------------------------------------------------------------
    m = _keys(top["matrix"], ("conductivity_m_s", "suction_m", "vegetation_factor"), (), "matrix")
    matrix = (_scalar_or_map(m["conductivity_m_s"], "matrix.conductivity_m_s", maps, lo=0.0),
              _scalar_or_map(m["suction_m"], "matrix.suction_m", maps, lo=0.0),
              _scalar_or_map(m["vegetation_factor"], "matrix.vegetation_factor", maps, lo=0.0, hi=1.0))

    # preferential -------------------------------------------------------------------------------------------------------------------------
    p = _keys(top["preferential"], ("a3", "max_fraction_per_hour", "max_intake_rate_m_s", "background_receiver_weights", "species"), (), "preferential")
    a3 = _scalar_or_map(p["a3"], "preferential.a3", maps, lo=0.0, hi=1.0)
    if isinstance(a3, float) and a3 not in pi.A3_VALUES:
        raise LayeredCaseConfigError(f"preferential.a3 must be one of the supported values {pi.A3_VALUES}, got {a3}")
    rate = p["max_intake_rate_m_s"]
    rate = "unlimited" if rate == "unlimited" else _scalar_or_map(rate, "preferential.max_intake_rate_m_s", maps, lo=0.0, lo_open=True)
    bg = _vec(p["background_receiver_weights"], "preferential.background_receiver_weights", L, lo=0.0)
    sp = p["species"]
    if not isinstance(sp, list) or len(sp) > limits.max_species:
        raise LayeredCaseConfigError(f"preferential.species must be a list of at most {limits.max_species} species (may be empty)")
    species = []
    for i, s in enumerate(sp):
        s = _keys(s, ("name", "receiver_weights", "contribution"), (), f"preferential.species[{i}]")
        if not isinstance(s["name"], str) or not s["name"] or len(s["name"]) > 64:
            raise LayeredCaseConfigError(f"preferential.species[{i}].name must be a non-empty string of at most 64 characters")
        species.append({"name": s["name"], "receiver_weights": _vec(s["receiver_weights"], f"preferential.species[{i}].receiver_weights", L, lo=0.0),
                        "contribution": _scalar_or_map(s["contribution"], f"preferential.species[{i}].contribution", maps, lo=0.0)})
    pref = _freeze({"a3": a3, "max_fraction_per_hour": _num(p["max_fraction_per_hour"], "preferential.max_fraction_per_hour", lo=0.0, hi=1.0e6),
                    "max_intake_rate_m_s": rate, "background_receiver_weights": bg, "species": species})

    # evaporation / cover ------------------------------------------------------------------------------------------------------------------
    e = _keys(top["evaporation"], ("canopy_intercept", "canopy_slope", "transpiration_slope", "transpiration_offset", "soil_factor"), (), "evaporation")
    try:
        partition = se.EvaporationPartition(**{k: _num(e[k], f"evaporation.{k}") for k in e})
    except (ss.SubsurfaceError, ValueError) as exc:
        raise LayeredCaseConfigError(f"evaporation partition refused: {exc}") from None
    cover = _scalar_or_map(top["cover"], "cover", maps, lo=0.0, hi=1.0)

    # initial ------------------------------------------------------------------------------------------------------------------------------
    ini = _keys(top["initial"], ("soil_moisture", "wetting_history_m", "cumulative_intake_m"), (), "initial")
    sm = _keys(ini["soil_moisture"], ("units", "by_layer"), (), "initial.soil_moisture")
    units = _str_in(sm["units"], "initial.soil_moisture.units", ("theta", "inventory_depth_m"))
    by_layer = _vec(sm["by_layer"], "initial.soil_moisture.by_layer", L, lo=0.0)
    wet = _scalar_or_map(ini["wetting_history_m"], "initial.wetting_history_m", maps, lo=0.0)
    cum = _scalar_or_map(ini["cumulative_intake_m"], "initial.cumulative_intake_m", maps, lo=0.0)

    # domain -------------------------------------------------------------------------------------------------------------------------------
    d = _keys(top["domain"], ("active_mask", "friction_factor", "closed_ring_rise_m", "outlets"), (), "domain")
    am = d["active_mask"]
    mask = None if am == "all" else maps.load(am, "domain.active_mask", "bool") if isinstance(am, dict) else None
    if am != "all" and not isinstance(am, dict):
        raise LayeredCaseConfigError("domain.active_mask must be 'all' or a bool map reference")
    friction = _scalar_or_map(d["friction_factor"], "domain.friction_factor", maps, lo=0.0, lo_open=True)
    rise = _num(d["closed_ring_rise_m"], "domain.closed_ring_rise_m", lo=0.0, lo_open=True)
    outs = d["outlets"]
    if not isinstance(outs, list) or len(outs) > limits.max_outlets:
        raise LayeredCaseConfigError(f"domain.outlets must be a list of at most {limits.max_outlets} outlet faces (empty = closed basin)")
    outlets, seen = [], set()
    for i, o in enumerate(outs):
        o = _keys(o, ("row", "col", "d_row", "d_col", "drop_m"), (), f"domain.outlets[{i}]")
        row, col = _int(o["row"], f"domain.outlets[{i}].row"), _int(o["col"], f"domain.outlets[{i}].col")
        dr, dc = o["d_row"], o["d_col"]
        if isinstance(dr, bool) or isinstance(dc, bool) or (dr, dc) not in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            raise LayeredCaseConfigError(f"domain.outlets[{i}]: (d_row, d_col) must be one of (1,0), (-1,0), (0,1), (0,-1)")
        drop = _num(o["drop_m"], f"domain.outlets[{i}].drop_m", lo=0.0, lo_open=True)
        if (row, col, dr, dc) in seen:
            raise LayeredCaseConfigError(f"domain.outlets[{i}] duplicates an earlier outlet face")
        seen.add((row, col, dr, dc))
        outlets.append((row, col, int(dr), int(dc), drop))

    outlets_t = tuple(outlets)
    by_layer_t = tuple(by_layer)
    sha = _config_sha(backend, soil_echo, matrix, pref, partition, cover, units, by_layer_t, wet, cum, friction, mask, rise, outlets_t)
    files = {r.name: r.sha256 for r in maps.cache.values()}
    identity = MappingProxyType({"config_sha256": sha, "yaml_sha256": hashlib.sha256(raw).hexdigest(), "map_files": MappingProxyType(dict(sorted(files.items())))})
    cfg = LayeredCaseConfig(_SEAL, backend=backend, layers=layers, spec=soil_echo, matrix=matrix, preferential=pref, partition=partition, cover=cover,
                            moisture_units=units, moisture_by_layer=by_layer_t, wetting_history_m=wet, cumulative_intake_m=cum, friction_factor=friction,
                            active_mask=mask, closed_ring_rise_m=rise, outlets=outlets_t, identity=identity, source_path=source_path)
    _verify_config(cfg)  # the freshly built record must pass its own authentication
    return cfg


# --- shape resolution against the actual native grid (METADATA ONLY: no array is built here) --------------------------------------------------------
def _tree_nbytes(obj: Any) -> int:
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return sum(_tree_nbytes(getattr(obj, f.name)) for f in dataclasses.fields(obj))
    return 0


def _check_prep(prep: Any) -> tuple[Any, Any, int, int, int, int]:
    if type(prep) is not nd.NativeDensePreparation:
        raise LayeredCaseError("prep must be a genuine NativeDensePreparation (native9f ordinary Dense CPU)")
    try:
        prep._p()
        bed, ctx = prep.bed, prep.context
        ny, nx = (int(v) for v in prep.shape)
        return bed, ctx, ny, nx, int(prep.n_voxels), int(prep.n_classes)
    except nd.NativeDenseProfileError as exc:
        raise LayeredCaseError(f"unauthenticated native preparation: {exc}") from None


def _checked_count(v: Any, what: str) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or int(v) < 0:
        raise LayeredCaseError(f"{what} must be an explicit non-negative integer (there is no default)")
    return int(v)


def _avail_reader(fn: Callable[[], int] | None) -> Callable[[], int]:
    if fn is None:
        return ss.default_host_available_bytes
    if not callable(fn):
        raise LayeredCaseError("available_bytes_fn must be callable")
    return fn


def _read_avail(fn: Callable[[], int]) -> int:
    v = fn()
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or int(v) < 0:
        raise LayeredCaseError(f"available_bytes_fn must return a non-negative integer, got {v!r}")
    return int(v)


def _is_map(v: Any) -> bool:
    return isinstance(v, MapRef)


def _check_domain_metadata(cfg: LayeredCaseConfig, ny: int, nx: int) -> None:
    """Cheap metadata refusals (shapes, outlet cells, moisture ranges on L-sized immutable vectors); no N-sized array is created."""
    for m in cfg.maps():
        if m.shape != (ny, nx):
            raise LayeredCaseConfigError(f"map {m.name!r} has shape {m.shape}, the native grid is {(ny, nx)} (no halo, rotation or resampling is applied)")
    if cfg.n_active(ny * nx) < 1:
        raise LayeredCaseConfigError("domain.active_mask has no active cell")
    mask = None if cfg.active_mask is None else cfg.active_mask.array
    for (row, col, dr, dc, _drop) in cfg.outlets:
        tr, tc = row + dr, col + dc
        if not (0 <= row < ny and 0 <= col < nx) or (mask is not None and not bool(mask[row, col])):
            raise LayeredCaseConfigError(f"outlet cell ({row}, {col}) must be an active cell of the {ny}x{nx} native grid")
        if not (tr in (-1, ny) or tc in (-1, nx)):
            raise LayeredCaseUnsupportedError(f"outlet ({row}, {col}) -> ({tr}, {tc}) is not a domain-edge face; only edge normal-flow outlets are supported")
    by_layer = np.array(cfg.moisture_by_layer)  # L doubles
    if cfg.moisture_units == "theta":
        if np.any(by_layer > np.array(cfg.layers.theta_saturation)):
            raise LayeredCaseConfigError("initial.soil_moisture.by_layer (theta) must not exceed theta_saturation in any layer")
    elif np.any(by_layer > np.array(cfg.layers.saturation_m)):
        raise LayeredCaseConfigError("initial.soil_moisture.by_layer (inventory depth, m) must not exceed the saturation depth of its layer")


# --- topology + graph construction bound (derived from the ACTUAL helpers; see correction01_design.md) -----------------------------------------------------
def topology_graph_terms(ny: int, nx: int, n_outlets: int) -> dict[str, int]:
    """Named upper bound (bytes) of ONE `pond_network.build_pond_topology` + `metadata_routing_graph` call and of what they retain. Every term is the sum of the
    allocations of the named helper in the source order (loop temporaries counted once: each iteration's arrays are released on rebind); `escape_python_records`
    covers the host priority queue of `pond_network._escape` (a cell is pushed only when its level decreases, only when a D4 neighbour is popped, each cell is
    popped once => at most 4 N pushes of a 72-B tuple + 32-B int + 8-B pointer) and its six N-lists. Not fitted: tests require the measured allocation peak of the
    real builders to stay below `construction_peak_bytes`. N = ny nx, R = (ny+2)(nx+2), F = ny (nx+1) + (ny+1) nx."""
    N, R = ny * nx, (ny + 2) * (nx + 2)
    F = ny * (nx + 1) + (ny + 1) * nx
    t = {
        "topology_input_copies": 2 * (8 * R + N + 8 * N),  # _check_bed/_host copies, then the _assemble copies handed to the geometry
        "geometry_work_and_retained": 2 * (8 * N + 8 * N + N + 24 * F + 4 * F) + 2 * N + 6 * 8 * N,  # z, f, active, 6 f8 + 4 int8 face arrays (built then frozen), masks, where temps
        "touch_masks": F,
        "topology_digests": 8 * R + 17 * N,
        "escape_python_records": (48 + 24 + 32 + 40) * N + 4 * N * 112 + 33 * N + n_outlets * 512,
        "topology_retained_arrays": 8 * R + 34 * N,
        "metadata_graph_inputs": 8 * R + R + 8 * N + 8 * N + N,
        "graph_builder_inputs": 8 * R + R + 8 * N + N + R + 8 * R,
        "graph_aspect_loop": 8 * N * 3 + 3 * N,
        "graph_slope_chain": 4 * 8 * N,
        "graph_index_arrays": 6 * 16 * N,
        "graph_masks": 16 * N,
        "graph_receiver": 24 * N,
        "graph_conveyance": 40 * N,
        "graph_donors": 32 * N + 60 * N,
        "graph_levels_sort": 4 * N + 52 * N + 16 * N,
        "graph_donor_tables": 100 * N,
        "graph_digest_copies": 8 * R + R + 8 * N + N,
        "graph_retained_outputs": 112 * N,
    }
    t["construction_peak_bytes"] = sum(t.values())
    t["retained_bytes"] = t["topology_retained_arrays"] + (8 * N + 8 * N + N + 24 * F + 4 * F) + t["graph_retained_outputs"]
    return t


# --- memory plan ------------------------------------------------------------------------------------------------------------------------------
def plan_layered_setup(prep: Any, config: LayeredCaseConfig, *, caller_reserve_bytes: int, lifecycle_extra_reserve_bytes: int,
                       available_bytes_fn: Callable[[], int] | None = None) -> dict:
    """Named staged requirement of `prepare_layered_case` from METADATA only: no N-sized array, source audit, material read, texture allocation or owner is created.
    Active counts and shapes come from the certified config (a map's count was taken at load; `all` = N).

    Stages (sequential stages take the max; retained items add; native input arrays are the caller's and the already-loaded config maps are already allocated:
    both are listed, not counted): (A) cold recheck = `plan_preparation`; (D) domain staging = mask N + scalar-expanded friction/cover/contributions + routing
    surface 8N + ring 8R; (B) D + texture (`estimate_texture_memory`; outputs retained); (T) D + retained texture + the native routing-surface scan
    (`layered_checkpoint` voxel/class scan constants x N V K) + the topology/graph construction bound (`topology_graph_terms`, from the real helpers);
    (O) retained texture + topology/graph + retained staging + owners (`layered_storm.estimate_layered_memory` required + initialization + build, native reserve = the
    preparation's `exchange_original_plus_candidate_bytes`, a DIAGNOSTIC convention, not a guarantee of any later exchange) + evaporation + initial-state staging.
    `required_bytes` includes everything before the single outer admission; the admission is a reading, not a reservation. Excluded: import/JIT/compile, allocator
    pools, OS cache, the process. Raises `LayeredCaseMemoryError` when `available_bytes_fn` reports less than required + `caller_reserve_bytes`."""
    _verify_config(config)
    _require_cpu_config(config)
    bed, ctx, ny, nx, V, K = _check_prep(prep)
    reserve = _checked_count(caller_reserve_bytes, "caller_reserve_bytes")
    extra = _checked_count(lifecycle_extra_reserve_bytes, "lifecycle_extra_reserve_bytes")
    _check_domain_metadata(config, ny, nx)
    N, L, S = ny * nx, config.n_layers, config.n_species
    n_active = config.n_active(N)
    try:
        native = nd.plan_preparation(prep.admission)
    except nd.NativeDenseProfileError as exc:
        raise LayeredCaseError(f"native preparation plan refused: {exc}") from None
    exchange = int(dict(prep.memory)["exchange_original_plus_candidate_bytes"])
    batch = min(st.DEFAULT_BATCH_CELLS, N)
    excluded = int(bed.voxel_column.mass_kg.nbytes) + int(bed.active_layer.mass_kg.nbytes)
    texture = st.estimate_texture_memory(representation="dense", backend=config.backend, output_namespace="numpy", n_cells=N, n_active=n_active, n_layers=L,
                                         n_classes=K, n_voxels=V, batch_cells=batch, reserve_bytes=0, input_bytes_excluded=excluded)
    control = HydraulicControl()
    owners = ls.estimate_layered_memory(L, (ny, nx), backend=config.backend, limiter=control.limiter, reserve_bytes=0, extra_reserve_bytes=extra, n_species=S,
                                        diffusion_mode=config.layers.diffusion_mode, native_bed=True, n_classes=K, native_reserve_bytes=exchange)
    evap = se.estimate_evaporation_memory(N, backend=config.backend, reserve_bytes=0, extra_reserve_bytes=0)
    R = (ny + 2) * (nx + 2)
    scalar_contrib = sum(1 for sp in config.preferential["species"] if not _is_map(sp["contribution"]))
    staging_retained = (N  # the water mask handed to the texture builder
                        + (8 * N if not _is_map(config.friction_factor) else 0)  # scalar friction expanded to (ny, nx)
                        + (8 * N if not _is_map(config.cover) else 0)  # scalar cover expanded to (N,)
                        + scalar_contrib * 8 * N)  # scalar species contributions expanded to (N,)
    transient_domain = 8 * N + 8 * R  # routing-surface array and the ring bed `full`
    scan = (lcp.VOXEL_SCAN_MASKS * N * V * K + lcp.VOXEL_SCAN_NV_TEMPORARIES * 8 * N * V + lcp.CLASS_SCAN_NK_TEMPORARIES * 8 * N * K + 6 * 8 * N)
    tg = topology_graph_terms(ny, nx, len(config.outlets))
    state_staging = L * N * _F8
    owners_stage = owners.required_bytes + owners.initialization_bytes + owners.build_bytes + evap.required_bytes + state_staging
    stage_a = int(native["required_bytes"])
    stage_d = staging_retained + transient_domain
    stage_b = stage_d + int(texture.required_bytes)
    stage_t = stage_d + int(texture.output_bytes) + scan + tg["construction_peak_bytes"]
    stage_o = int(texture.output_bytes) + tg["retained_bytes"] + staging_retained + owners_stage
    required = max(stage_a, stage_b, stage_t, stage_o)
    plan = {
        "schema": SCHEMA, "backend": config.backend, "shape": (ny, nx), "n_cells": N, "n_active": n_active, "n_layers": L, "n_classes": K, "n_voxels": V,
        "n_species": S,
        "terms": {"A_cold_recheck_required_bytes": stage_a, "D_staging_retained_bytes": staging_retained, "D_transient_domain_bytes": transient_domain,
                  "B_texture_required_bytes": int(texture.required_bytes), "B_texture_retained_output_bytes": int(texture.output_bytes),
                  "T_routing_surface_scan_bytes": scan, "T_topology_graph_construction_peak_bytes": tg["construction_peak_bytes"],
                  "T_topology_graph_retained_bytes": tg["retained_bytes"], "O_owners_required_bytes": int(owners.required_bytes),
                  "O_owners_initialization_bytes": int(owners.initialization_bytes), "O_owners_build_bytes": int(owners.build_bytes),
                  "O_evaporation_required_bytes": int(evap.required_bytes), "O_initial_state_staging_bytes": state_staging,
                  "native_reserve_exchange_bytes_diagnostic": exchange, "lifecycle_extra_reserve_bytes": extra, "texture_batch_cells": batch,
                  "caller_retained_native_inputs_bytes_listed_not_counted": excluded,
                  "loaded_config_map_bytes_already_allocated_not_counted": sum(int(m.array.nbytes) for m in config.maps())},
        "topology_graph_terms": {k: v for k, v in tg.items()},
        "stage_bytes": {"A": stage_a, "D": stage_d, "B": stage_b, "T": stage_t, "O": stage_o},
        "required_bytes": required, "caller_reserve_bytes": reserve, "required_with_reserve": required + reserve,
        "topology_graph_bytes_actual": None,
        "scope": "metadata-only staged plan (no N-sized array); the single outer admission is a reading, not a reservation; sequential stages take the max, retained items "
                 "add; excludes import/JIT/compile, allocator pools, OS cache and the process; caller-retained native inputs and loaded config maps are listed, not counted",
    }
    if available_bytes_fn is not None:
        avail = _read_avail(_avail_reader(available_bytes_fn))
        if plan["required_with_reserve"] > avail:
            raise LayeredCaseMemoryError(f"layered setup needs an estimated {plan['required_with_reserve']} B (stage plan {required} B + caller reserve "
                                         f"{reserve} B) but {avail} B are reported available; nothing was audited, allocated or built")
    return plan


# --- the result ----------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class LayeredCaseSetup:
    """The genuine initial boundary lifecycle plus every retained setup product. `state` aliases the native bed's water depth (ONE surface authority) and holds the
    preparation's BedState; the NativeDensePreparation/compiled case is not retained here."""

    lifecycle_operator: Any
    state: Any
    texture: Any
    column: Any
    preferential: Any
    matrix: Any
    evaporation: Any
    router: Any
    graph: Any
    topology: Any
    cover: np.ndarray
    config: LayeredCaseConfig
    identities: Any
    plan: dict


def _wrap(stage: str, exc: BaseException) -> LayeredCaseError:
    cls = LayeredCaseMemoryError if isinstance(exc, MemoryError) else LayeredCaseError
    return cls(f"{stage} refused: {type(exc).__name__}: {exc}")


def prepare_layered_case(prep: Any, config: LayeredCaseConfig, *, caller_reserve_bytes: int, lifecycle_extra_reserve_bytes: int,
                         available_bytes_fn: Callable[[], int] | None = None) -> LayeredCaseSetup:
    """Cold, initial-only setup. Order: config authentication + cheap metadata checks (no N-sized array) -> the ONE staged admission (`plan_layered_setup`, which
    includes the topology/graph construction bound) -> `prep.recheck` (fresh source/case audit; never on a later evolved state) -> domain staging -> retained actual
    texture (the production builder refuses shallow material, a cap thicker than the top layer, a mask mismatch) -> routing surface, pond topology + graph (the actual
    retained bytes verify the plan afterwards) -> column, preferential, matrix, evaporation, router, `prepare_lifecycle` -> initial soil state/history ->
    `start_lifecycle`. The caller's native arrays are never written; any failure leaves them, the preparation and all files unchanged."""
    _verify_config(config)
    _require_cpu_config(config)
    bed, ctx, ny, nx, V, K = _check_prep(prep)
    reserve = _checked_count(caller_reserve_bytes, "caller_reserve_bytes")
    extra = _checked_count(lifecycle_extra_reserve_bytes, "lifecycle_extra_reserve_bytes")
    raw_avail = _avail_reader(available_bytes_fn)
    N, L = ny * nx, config.n_layers
    _check_domain_metadata(config, ny, nx)
    thickness = np.array(config.layers.thickness_m)  # L doubles
    by_layer = np.array(config.moisture_by_layer)
    depth0 = by_layer * thickness if config.moisture_units == "theta" else by_layer.copy()
    try:
        plan = plan_layered_setup(prep, config, caller_reserve_bytes=reserve, lifecycle_extra_reserve_bytes=extra, available_bytes_fn=raw_avail)
    except LayeredCaseError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _wrap("memory plan", exc) from None

    def net_avail() -> int:
        return max(_read_avail(raw_avail) - reserve, 0)

    try:
        prep.recheck(caller_reserve_bytes=reserve, available_bytes_fn=raw_avail)
    except nd.NativeDenseProfileError as exc:
        raise _wrap("cold native source/case recheck", exc) from None
    # --- domain staging (admitted above as term D); views of the immutable maps, expansions only for scalars -----------------------------------------------
    def flat(v):
        return np.ascontiguousarray(v.array.reshape(-1)) if _is_map(v) else float(v)
    mask = np.ones((ny, nx), dtype=bool) if config.active_mask is None else np.array(config.active_mask.array, dtype=bool)
    cond, psi, veg = (flat(v) for v in config.matrix)
    pref = config.preferential
    a3 = flat(pref["a3"])
    rate = np.inf if pref["max_intake_rate_m_s"] == "unlimited" else flat(pref["max_intake_rate_m_s"])
    species = pref["species"]
    contributions = [(flat(sp["contribution"]) if _is_map(sp["contribution"]) else np.full(N, float(sp["contribution"]))) for sp in species]
    cover = flat(config.cover) if _is_map(config.cover) else np.full(N, float(config.cover))
    wet, cum = flat(config.wetting_history_m), flat(config.cumulative_intake_m)
    friction2d = config.friction_factor.array if _is_map(config.friction_factor) else np.full((ny, nx), float(config.friction_factor))
    try:
        texture = st.soil_profile_texture_from_bed(bed, ctx, tuple(float(t) for t in thickness), active_mask=mask,
                                                   memory_budget_bytes=int(plan["terms"]["B_texture_required_bytes"]), backend=config.backend,
                                                   batch_cells=int(plan["terms"]["texture_batch_cells"]))
    except (st.SoilProfileTextureError, st.SoilProfileMemoryError, sb.BedIntegrationError) as exc:
        raise _wrap("retained actual soil-profile texture", exc) from None
    try:
        z = np.array(sb.current_routing_surface_m(bed, ctx), dtype=np.float64)
        full = np.full((ny + 2, nx + 2), float(z.max()) + config.closed_ring_rise_m)
        full[1:-1, 1:-1] = z
        faces = []
        for (row, col, dr, dc, drop) in config.outlets:
            full[row + dr + 1, col + dc + 1] = z[row, col] - drop
            faces.append(pn.OutletFace(row, col, dr, dc))
        topo = pn.build_pond_topology(full, mask, friction2d, float(ctx.geometry.dx_m), faces, allow_closed_components=not faces)
        graph = metadata_routing_graph(topo, np)
    except (pn.PondNetworkError, sb.BedIntegrationError, ValueError) as exc:
        raise _wrap("pond topology / routing graph", exc) from None
    # Construction-only inputs are released NOW (before the stage-O reading and every owner): the topology copied what it needs (its own source_z_full/active/friction)
    # and the owners use `topo.active`, so these references would otherwise stay live until return although the plan models them as stage-T transients.
    del z, full, faces, mask, friction2d
    topo_bytes = _tree_nbytes(topo) + _tree_nbytes(graph)
    planned = int(plan["terms"]["T_topology_graph_retained_bytes"])
    if topo_bytes > planned:  # the actual retained arrays VERIFY the up-front bound (they cannot retroactively protect the construction)
        raise LayeredCaseMemoryError(f"internal: the topology/graph retained {topo_bytes} B exceed the planned {planned} B (plan underestimated); no owner was built")
    avail2 = _read_avail(raw_avail)
    need2 = int(plan["stage_bytes"]["O"]) - planned + topo_bytes + reserve
    if need2 > avail2:
        raise LayeredCaseMemoryError(f"layered owners need {need2} B (actual topology/graph {topo_bytes} B) but {avail2} B are reported available; no owner was built")
    plan = {**plan, "topology_graph_bytes_actual": topo_bytes}
    try:
        flat_mask = np.ascontiguousarray(np.asarray(topo.active, dtype=bool).reshape(-1))
        column = ss.prepare_column(config.layers, flat_mask, topo.dx_m * topo.dx_m, backend=config.backend, reserve_bytes=0, available_bytes_fn=net_avail)
        preferential = pi.prepare_preferential(column, a3=a3, max_fraction_per_hour=pref["max_fraction_per_hour"], max_intake_rate_m_s=rate,
                                               background_receiver_weights=list(pref["background_receiver_weights"]),
                                               species_receiver_weights=[list(sp["receiver_weights"]) for sp in species],
                                               species_contributions=contributions if species else None)
        matrix = mi.prepare_matrix(column, conductivity_m_s=cond, suction_m=psi, vegetation_factor=veg)
        evaporation = se.prepare_evaporation(column, config.partition)
        router = build_surface_router("local_inertial", graph, backend=config.backend, geometry=topo.geometry, control=HydraulicControl(), reserve_bytes=0,
                                      available_bytes_fn=net_avail)
        exchange = int(dict(prep.memory)["exchange_original_plus_candidate_bytes"])
        lop = wl.prepare_lifecycle(column, preferential, matrix, router, graph, topo, evaporation, cover, bed=bed, context=ctx, native_reserve_bytes=exchange,
                                   extra_reserve_bytes=extra)
        if config.moisture_units == "theta":
            soil = ss.uniform_state(column, list(by_layer), 0.0)
        else:
            water = np.ascontiguousarray(np.broadcast_to(depth0[:, None], (L, N)))
            soil = ss.column_state(column, water, 0.0)
        history = mi.wetting_history(matrix, wetting_m=wet, cumulative_intake_m=cum)
        state = wl.start_lifecycle(lop, bed=bed, soil=soil, history=history, t_s=0.0)
    except (ss.SubsurfaceError, ss.SubsurfaceMemoryError, ls.LayeredStormError, wl.LifecycleError, MemoryError, ValueError) as exc:
        raise _wrap("water owner construction", exc) from None
    tex_src = json.dumps(texture.source, sort_keys=True, default=str, separators=(",", ":"))
    identities = MappingProxyType({
        "config": config.identity, "layered_case_schema": SCHEMA, "native_profile": prep.profile_name, "native_bundle_id": prep.bundle_id,
        "native_package_digest": prep.package_digest, "syrup_package_digest": prep.syrup_package_digest, "case_identity_sha256": prep.case_identity_sha256,
        "latitude_deg": prep.latitude_deg, "latitude_binding": prep.latitude_binding(), "route": prep.route, "lifecycle_operator_sha256": lop.sha256,
        "texture_source_sha256": hashlib.sha256(tex_src.encode()).hexdigest(), "backend": config.backend})
    return LayeredCaseSetup(lop, state, texture, column, preferential, matrix, evaporation, router, graph, topo, cover, config, identities, plan)
