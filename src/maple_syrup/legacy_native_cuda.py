"""Resident CUDA execution of the native-walk MAHLERAN LEGACY sediment replay (task gpu_sediment, stage B1).

Benchmark-only, exactly like the CPU replay (`legacy_native`, `legacy_native_numba`, `legacy_driver`): frozen terrain and routing, fixed
composition, UNLIMITED supply, an explicit artificial clipping source, ring / inactive-cell deposition as diagnostics, terminal pits
that keep their mobile mass, no MAPLE bed is read or written. It is not a conservative event, restart or wind-handoff model.

One step on the device (every array stays resident; the only per-step host traffic is the hydrology's own 144-byte packet read):

  1. `sg_laws`     thread per cell, classes in a loop: the wet MAHLERAN laws of `legacy_physics_numba` ported term for term
                   (same expression order, no fast math, `--fmad=false`), detachment RATE `requested / dt`, the legacy virtual
                   velocity, `1/L`, the law mask, the regime, plus the CPU validation flags (law bits 0..43, glue bits 44..48).
  2. `sg_values`   thread per (source, class): the original `flow_distrib` walk, evaluated ONCE per source/class: local credit
                   `fract0 d`, then the credits `d (exp(-lo/L) - exp(-u/L))` along the STATIC downstream path, with the exact
                   `step < limit and fract > vge` loop, the ring/terminal/inactive stop kinds, the aspect-0 WEST start, the
                   zero-slope diffuse "no walk, no deposit" and the local no-law deposit. Each credit is written to its OWN
                   record value (no atomics, no duplicate writers); unused records are zeroed.
  3. `sg_gather`   thread per (target cell, class): sums the record values targeting the cell in CPU SOURCE ORDER (source index
                   ascending, then position along the path; the local credit first), starting from 0.0: the same floating point
                   additions in the same order as the CPU `depos[pos, k] += ...`, hence the same deposition rate.
  4. `sg_cn_*`     the ACCEPTED Crank-Nicolson pool, ordered S, W, N, E donor sums, per dependency level (or one fused block on
                   narrow networks), with the actual negative trial and clip source written explicitly.
  5. reduce/ledger deterministic block partial sums in index order plus a fixed final order (no floating point atomics), the
                   cumulative per-cell maps, the validation flags of the CPU `_check_py` and the integer walk/regime tallies.

The walk tables (`WalkTables`) are STATIC bounded geometry built once on the host: one record per (source, credited cell) plus one
local record per source, a target-sorted CSR with the record ids, the ring records and the inactive targets. Nothing grows with time;
no per-step sort or pair history exists. Memory: `records x classes x 8` bytes of values plus the tables; `estimate_bytes` is
computed BEFORE any allocation and the build fails early when it exceeds the budget.

Unsupported on purpose (rejected before any mutation): `source_order='legacy'` and the order-dependent erasure. Zero-mass classes are
NOT skipped in the physics: laws, regimes, virtual velocities, the CN pool (including any seeded mobile mass), the ledgers and the
cumulative maps of ALL declared classes are evaluated. Stage B2 compacts only the walk RECORD values (`record_strategy='compact'`,
the default; `'all'` is the B1 reference): the record array holds `n_records x ne` values for the `ne` classes with a positive
composition fraction in at least one active cell (static, `eligible_record_classes`); every other class has exactly zero
detachment (see its docstring), hence zero records, zero deposition and zero walk code, none of which is written. With `ne == 0`
no record launch is made at all.

Error contract: flags are accumulated on the device in a per-step word (sticky per step, OR) and read in small slices at a declared
cadence (`check_flags`); a nonzero word POISONS the context (unpublished state; `reset()` required) and the driver abandons the run.
Nothing is published by this module. Missing CuPy/device/compile failure raises `CudaUnavailableError` (no CPU fallback).

Nothing here was run by its author (file-only tools); Codex records results.
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from maple_syrup import legacy_native as N
from maple_syrup.legacy_physics_numba import _FLAG_MESSAGES, LegacyPhysicsContext
from maple_syrup.routing_cuda import (
    COMPILE_OPTIONS,
    CudaUnavailableError,
    _cupy,
    _current_device_id,
)
from maple_syrup.sediment_physics import (
    CONCENTRATED_DISTANCE_CAP_M,
    GRAVITY_M_S2,
    LEGACY_MEDIAN_FACTOR,
    REGIME_CODES,
    REYNOLDS_CONCENTRATED,
    REYNOLDS_TRANSITIONAL,
    SUSPENDED_EXPONENT_CAP,
    recession_velocity,
)

__all__ = [
    "LEDGER_ROWS",
    "RECORD_STRATEGIES",
    "CudaLegacyContext",
    "CudaLegacyError",
    "WalkTables",
    "build_walk_tables",
    "eligible_record_classes",
    "estimate_bytes",
    "kernel_provenance",
    "kernel_source",
    "resolve_record_classes",
    "validate_limits",
    "validate_walk_tables",
]

INT32_MAX = 2**31 - 1
MAX_CLASSES = 32  # every actual launch supports it: ring/outlet 32 threads, final reduce 7 * 32 = 224 <= 256 threads
#: threads per block of every kernel launch (the resource check uses exactly these)
LAUNCH_BLOCKS = {"sg_laws": 128, "sg_post_infiltration": 128, "sg_values": 128, "sg_gather": 128, "sg_ring_inactive": 32,
                 "sg_cn_level": 128, "sg_cn_block": 128, "sg_reduce_partial": 256, "sg_reduce_final": 256, "sg_outlet": 32,
                 "sg_tally": 128}

LEDGER_ROWS = 13  # the A1 `LEDGER_COLUMNS`
N_REG = len(REGIME_CODES)
N_COUNT_COLUMNS = N.N_COUNTS + N_REG  # walk tallies then the seven wet regimes
THREADS = 128  # laws / values / gather / cn
REDUCE_THREADS = 256
MAX_REDUCE_BLOCKS = 256
PACK_BIT0, TABLE_BIT, CHECK_BIT0 = 44, 49, 50
PACK_MESSAGES = ("detachment must be finite and >= 0", "sediment velocity must be finite and >= 0",
                 "an inactive cell has detachment", "a law applies with detachment but 1/L is not positive and finite",
                 "a terminal pit has a nonzero sediment velocity (it has no receiver)")
TABLE_MESSAGE = "a walk ran past its static path table (internal consistency)"


class CudaLegacyError(RuntimeError):
    """Configuration, validation or device failure of the CUDA legacy replay (never a silent CPU fallback)."""


def decode_flags(word: int) -> list[str]:
    out = [m for b, m in enumerate(_FLAG_MESSAGES) if word >> b & 1]
    out += [m for b, m in enumerate(PACK_MESSAGES) if word >> (PACK_BIT0 + b) & 1]
    if word >> TABLE_BIT & 1:
        out.append(TABLE_MESSAGE)
    out += [m for b, m in enumerate(N.CHECK_MESSAGES) if word >> (CHECK_BIT0 + b) & 1]
    return out


# --- static walk tables (pure NumPy: importable and testable without CuPy) -------------------------------------------------
@dataclass(frozen=True)
class WalkTables:
    """Static source/target record tables of the downstream walk over a frozen graph. Record `rec_off[s]` is the LOCAL credit
    of source `s`; records `rec_off[s] + 1 + j` are the path credits `j = 0 ..` (the cell credited at walk iteration `j`).
    `end_kind[s]`: 0 path cut by the static cap (the loop always ends on the limit first), 1 the walk ends in the ring, 2 in a
    terminal pit, 3 in an inactive cell (the last record is that cell/ring). Target CSR: `tgt_rec[tgt_ptr[c]:tgt_ptr[c + 1]]` are
    the record ids crediting cell `c` ascending (= CPU source order, then path position)."""

    src_cells: np.ndarray  # (m,) int32 sources in CPU order (index order)
    rec_off: np.ndarray  # (m + 1,) int64
    end_kind: np.ndarray  # (m,) uint8
    tgt_ptr: np.ndarray  # (n + 1,) int64
    tgt_rec: np.ndarray  # (R_cell,) int32 (records with a cell target, grouped by cell)
    ring_rec: np.ndarray  # (R_ring,) int32 record ids crediting the ring, ascending
    inactive_targets: np.ndarray  # (k,) int32 inactive cells that have at least one record
    n_records: int
    max_path: int
    cap: int

    def nbytes(self) -> int:
        return int(sum(a.nbytes for a in (self.src_cells, self.rec_off, self.end_kind, self.tgt_ptr, self.tgt_rec,
                                           self.ring_rec, self.inactive_targets)))


def count_records(network: N.NativeNetwork, cap: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-source path lengths (credited cells, ring included) and end kinds, by vectorised pointer chasing over all sources."""
    src = network.source_order_index.astype(np.int64)
    m = src.size
    plen = np.zeros(m, dtype=np.int64)
    kind = np.zeros(m, dtype=np.uint8)
    pos = network.walk_first[src].astype(np.int64)
    alive = np.arange(m)
    for _ in range(int(cap)):
        if alive.size == 0:
            break
        p = pos[alive]
        plen[alive] += 1
        ring = p < 0
        kind[alive[ring]] = 1
        nxt = np.where(ring, 0, network.walk_next[np.where(ring, 0, p)])
        term = ~ring & (nxt == N.STOP_TERMINAL)
        inact = ~ring & (nxt == N.STOP_INACTIVE)
        kind[alive[term]] = 2
        kind[alive[inact]] = 3
        cont = ~ring & ~term & ~inact
        pos[alive[cont]] = nxt[cont]
        alive = alive[cont]
    return plen, kind


def build_walk_tables(network: N.NativeNetwork, limits: np.ndarray, *, record_budget: int | None = None) -> WalkTables:
    """Build the static tables. `limits` is the per-regime walk limit (cells); the static cap is its maximum. Raises
    `CudaLegacyError` before allocating the record arrays when `n_records` exceeds `record_budget`."""
    limits = validate_limits(limits)
    cap = int(np.max(limits))
    if network.active.size > INT32_MAX:
        raise CudaLegacyError("more cells than a signed 32-bit index can address")
    plen, kind = count_records(network, cap)
    m = plen.size
    rec_off = np.zeros(m + 1, dtype=np.int64)
    np.cumsum(plen + 1, out=rec_off[1:])
    n_records = int(rec_off[-1])
    if record_budget is not None and n_records > record_budget:
        raise CudaLegacyError(f"{n_records} walk records exceed the budget of {record_budget}")
    if n_records > INT32_MAX:
        raise CudaLegacyError(f"{n_records} walk records exceed the signed 32-bit record index (tgt_rec is int32)")
    n = network.active.size
    target = np.empty(n_records, dtype=np.int32)
    src = network.source_order_index.astype(np.int64)
    target[rec_off[:-1]] = src  # the local credit goes to the source cell itself
    pos = network.walk_first[src].astype(np.int64)
    alive = np.arange(m)
    for step in range(cap):
        if alive.size == 0:
            break
        p = pos[alive]
        target[rec_off[alive] + 1 + step] = p  # the cell (or the ring, -1) credited at iteration `step`
        ring = p < 0
        nxt = np.where(ring, 0, network.walk_next[np.where(ring, 0, p)])
        cont = ~ring & (nxt != N.STOP_TERMINAL) & (nxt != N.STOP_INACTIVE)
        # sources whose path is exhausted by the static cap stay alive only while step + 1 < plen
        cont &= (step + 1) < plen[alive]
        pos[alive[cont]] = nxt[cont]
        alive = alive[cont]
    is_ring = target < 0
    ring_rec = np.flatnonzero(is_ring).astype(np.int32)
    cells = np.flatnonzero(~is_ring)
    order = np.argsort(target[cells], kind="stable")  # ties keep ascending record id = CPU source order
    tgt_rec = cells[order].astype(np.int32)
    counts = np.bincount(target[cells], minlength=n).astype(np.int64)
    tgt_ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=tgt_ptr[1:])
    inactive_targets = np.flatnonzero((counts > 0) & network.inactive).astype(np.int32)
    arrays = [src.astype(np.int32), rec_off, kind, tgt_ptr, tgt_rec, ring_rec, inactive_targets]
    for a in arrays:
        a.setflags(write=False)  # static by contract; optional flag so accidental host edits fail loudly
    return WalkTables(src_cells=arrays[0], rec_off=arrays[1], end_kind=arrays[2], tgt_ptr=arrays[3], tgt_rec=arrays[4],
                      ring_rec=arrays[5], inactive_targets=arrays[6], n_records=n_records,
                      max_path=int(plen.max()) if plen.size else 0, cap=cap)


def validate_limits(limits) -> np.ndarray:
    """The per-regime walk limits: a 1-D integer array with one entry per regime code, all >= 0 and the maximum >= 1."""
    arr = np.asarray(limits)
    if arr.ndim != 1 or arr.size != N_REG or arr.dtype.kind not in "iu":
        raise CudaLegacyError(f"limits must be a 1-D integer array of {N_REG} regime entries")
    if np.any(arr < 0) or int(arr.max()) < 1 or int(arr.max()) > INT32_MAX:
        raise CudaLegacyError("the walk limits must be non-negative with a positive maximum (below 2**31)")
    return arr.astype(np.int64)


def validate_walk_tables(network: N.NativeNetwork, tables: WalkTables, limits) -> None:
    """One-time full validation of caller-supplied tables BEFORE any device allocation: exact types, dtypes, shapes, index ranges,
    monotone offsets, and then equality of every array with the tables rebuilt from the frozen network and limits (so the source
    order, the target inverse, the stop kinds, the aspect-0 west start and the cap are all proven, not trusted)."""
    if not isinstance(tables, WalkTables):
        raise CudaLegacyError("tables must be a WalkTables built by build_walk_tables")
    limits = validate_limits(limits)
    n, m = int(network.active.size), int(network.active_idx.size)
    spec = {"src_cells": (np.int32, (m,)), "rec_off": (np.int64, (m + 1,)), "end_kind": (np.uint8, (m,)),
            "tgt_ptr": (np.int64, (n + 1,)), "tgt_rec": (np.int32, None), "ring_rec": (np.int32, None),
            "inactive_targets": (np.int32, None)}
    for name, (dtype, shape) in spec.items():
        a = getattr(tables, name)
        if type(a) is not np.ndarray or a.dtype != dtype or a.ndim != 1 or not a.flags.c_contiguous:
            raise CudaLegacyError(f"tables.{name} must be a C-contiguous 1-D {np.dtype(dtype)} ndarray")
        if shape is not None and a.shape != shape:
            raise CudaLegacyError(f"tables.{name} has shape {a.shape}, expected {shape}")
    for name in ("n_records", "max_path", "cap"):
        if isinstance(getattr(tables, name), bool) or not isinstance(getattr(tables, name), (int, np.integer)):
            raise CudaLegacyError(f"tables.{name} must be an int")
    R = int(tables.n_records)
    if not 0 <= R <= INT32_MAX or int(tables.rec_off[0]) != 0 or int(tables.rec_off[-1]) != R or np.any(np.diff(tables.rec_off) < 1):
        raise CudaLegacyError("tables.rec_off is not a monotone offset table ending at n_records")
    if tables.tgt_rec.size + tables.ring_rec.size != R or int(tables.tgt_ptr[0]) != 0 or int(tables.tgt_ptr[-1]) != tables.tgt_rec.size \
            or np.any(np.diff(tables.tgt_ptr) < 0):
        raise CudaLegacyError("tables.tgt_ptr / tgt_rec / ring_rec are inconsistent")
    for name in ("tgt_rec", "ring_rec"):
        a = getattr(tables, name)
        if a.size and (int(a.min()) < 0 or int(a.max()) >= R):
            raise CudaLegacyError(f"tables.{name} holds a record index outside [0, n_records)")
    if m and (int(tables.src_cells.min()) < 0 or int(tables.src_cells.max()) >= n):
        raise CudaLegacyError("tables.src_cells holds a cell index outside the grid")
    if tables.inactive_targets.size and (int(tables.inactive_targets.min()) < 0 or int(tables.inactive_targets.max()) >= n):
        raise CudaLegacyError("tables.inactive_targets holds a cell index outside the grid")
    fresh = build_walk_tables(network, limits)
    for name in spec:
        if not np.array_equal(getattr(tables, name), getattr(fresh, name)):
            raise CudaLegacyError(f"tables.{name} differs from the tables of the frozen network and limits")
    if (tables.n_records, tables.max_path, tables.cap) != (fresh.n_records, fresh.max_path, fresh.cap):
        raise CudaLegacyError("tables scalars differ from the tables of the frozen network and limits")


def level_bounds(graph, order: np.ndarray) -> np.ndarray:
    """Boundaries of the dependency levels inside `order` (active cells, donors before receivers)."""
    level = np.asarray(graph.level).reshape(-1)[order]
    if level.size and np.any(np.diff(level) < 0):
        raise CudaLegacyError("the network order is not sorted by dependency level")
    cuts = np.flatnonzero(np.diff(level)) + 1 if level.size else np.zeros(0, dtype=np.int64)
    return np.concatenate(([0], cuts, [level.size])).astype(np.int64)


RECORD_STRATEGIES = ("compact", "all")


def eligible_record_classes(network: N.NativeNetwork, fractions) -> np.ndarray:
    """STATIC record-class eligibility from the immutable composition: class `k` is eligible iff `fractions[i, k] > 0` for at least
    one ACTIVE cell `i`. PROOF that the other classes carry no record value: in `sg_laws` both detachment terms of a class are forced
    to exactly 0 where `!(f > 0)` (rain `rd` and flow `fd`), inactive cells request 0, and `det = (0 * mass) / dt` is +0 for a finite
    mass; `sg_values` then takes no branch (`d > 0` false), so every record value, the local credit, the walk code and the
    deposition of the class are +0 in every step and the CPU walk adds +0 only. Returns the eligible class indices (int32, strictly
    increasing; possibly empty). Pure host work on the static fractions: no per-step scan, no device read."""
    fr = np.asarray(fractions)
    n = int(network.active.size)
    if fr.dtype.kind != "f" or fr.ndim != 2 or fr.shape[0] != n or not 1 <= fr.shape[1] <= MAX_CLASSES:
        raise CudaLegacyError(f"fractions must be a float (cells={n}, classes) array with 1..{MAX_CLASSES} classes")
    return np.flatnonzero(((fr > 0.0) & network.active[:, None]).any(axis=0)).astype(np.int32)


def resolve_record_classes(network: N.NativeNetwork, fractions, strategy: str, record_classes=None) -> np.ndarray:
    """The record classes of a strategy: `all` = every class (the B1 reference, identity rank); `compact` = the static eligible
    classes, or an explicit `record_classes` that must be strictly increasing, in range and a SUPERSET of the eligible classes (a
    subset would silently drop deposition). Validated before any allocation."""
    if strategy not in RECORD_STRATEGIES:
        raise CudaLegacyError(f"record strategy must be one of {RECORD_STRATEGIES}, got {strategy!r}")
    fr = np.asarray(fractions)
    eligible = eligible_record_classes(network, fr)
    nc = int(fr.shape[1])
    if strategy == "all":
        if record_classes is not None:
            raise CudaLegacyError("record_classes may only be supplied with the compact strategy")
        return np.arange(nc, dtype=np.int32)
    if record_classes is None:
        return eligible
    arr = np.asarray(record_classes)
    if arr.ndim != 1 or arr.dtype.kind not in "iu" or arr.dtype == np.bool_ or (arr.size and (int(arr.min()) < 0 or int(arr.max()) >= nc)) \
            or np.any(np.diff(arr.astype(np.int64)) <= 0):
        raise CudaLegacyError("record_classes must be a strictly increasing 1-D integer array of class indices in range")
    missing = np.setdiff1d(eligible, arr)
    if missing.size:
        raise CudaLegacyError(f"record_classes omit composition-eligible classes {missing.tolist()}: deposition would be dropped")
    return arr.astype(np.int32)


def estimate_bytes(network: N.NativeNetwork, physics: LegacyPhysicsContext, n_records: int, n_steps: int, nc: int,
                   tables_bytes: int, n_record_classes: int | None = None) -> dict[str, int]:
    """Device bytes of the context (before allocation). Persistent static, dynamic per-cell state, record values, ledgers. The record
    values hold `n_record_classes` (default `nc`) classes per record; the non-record physical classes keep every other array."""
    ne = nc if n_record_classes is None else int(n_record_classes)
    if not 0 <= ne <= nc:
        raise CudaLegacyError("n_record_classes must be within [0, nc]")
    n = int(network.active.size)
    m = int(network.active_idx.size)
    f8 = 8
    static = (n * nc * f8 * 3 + n * f8 * 4 + n * 3 + n * 4 * 4 + int(network.order.size) * 4 + tables_bytes + 9 * nc * f8
              + n * 4 * 2 + n * 2 + max(1, ne) * 4)  # + the record-class index array
    dynamic = n * nc * f8 * 19 + n * nc * 2  # det, v_used, v_prev, rate, depos, clip, M1/2, Q1/2, Qin1/2, cum x3 (+ spare)
    values = int(n_records) * ne * f8 if ne else f8  # zero record classes: a single placeholder element (no zero-size device array)
    codes = m * nc
    partial = MAX_REDUCE_BLOCKS * 7 * nc * f8
    ledger = n_steps * (LEDGER_ROWS * nc * f8 + 8 + N_COUNT_COLUMNS * 8)
    return {"static": static, "dynamic": dynamic, "walk_values": values, "walk_codes": codes, "partials": partial,
            "ledger": ledger, "total": static + dynamic + values + codes + partial + ledger}


# --- kernel source --------------------------------------------------------------------------------------------------------
def _c(v: float) -> str:
    return repr(float(v))


_TEMPLATE = r"""
#define NC __NC__
#define NE __NEX__ /* record classes, at least 1 so that `t / NE` is never a constant division by zero; NE_RECORD is the true count */
#define NE_RECORD __NE__ /* 0 = no record kernel is ever launched */
#define BLK __BLK__
typedef unsigned long long ull;
typedef long long ll;
#define SG_FINITE(x) (((x) - (x)) == 0.0)
#define RE_CONC __RE_CONC__
#define RE_TRANS __RE_TRANS__
#define CONC_CAP __CONC_CAP__
#define SUSP_CAP __SUSP_CAP__
#define GRAV __GRAV__
#define MEDIAN __MEDIAN__

__device__ __forceinline__ double sg_nan() { return __longlong_as_double(0x7ff8000000000000LL); }
__device__ __forceinline__ double sg_max(double a, double b) { if (a != a || b != b) return sg_nan(); return a > b ? a : b; }
__device__ __forceinline__ double sg_min(double a, double b) { if (a != a || b != b) return sg_nan(); return a < b ? a : b; }
__device__ __forceinline__ double sg_pw(double x, double y) {
    if (x < 0.0) return sg_nan();
    if (x == 0.0) return 0.0;
    return pow(x, y);
}

/* sc: 0 rho, 1 ref, 2 mass, 3 nu, 4 hz, 5 -, 6 bag_scale, 7 decay, 8 p_par, 9 dt ; cfg: ke_model, veg_literal, fraction_scaling,
   dist_literal, depth_mm ; cls: (9, NC) row-major */
extern "C" __global__ void sg_laws(
    const int n, const double* depth, const double* vel, const double* rain, const double* prev,
    const unsigned char* active, const unsigned char* terminal, const double* slope, const double* veg_factor,
    const double* d50, const double* bag_c1, const double* slope_pow, const double* fractions, const double* cap_rate,
    const double* cls, const double* sc, const int* cfg,
    double* det, double* v_used, double* rate, unsigned char* law, signed char* regime, ull* flagword)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ull flags = 0ull;
    const double rho = sc[0], ref = sc[1], mass = sc[2], nu = sc[3], hz = sc[4], bag_scale = sc[6], decay = sc[7],
                 p_par = sc[8], dt = sc[9];
    const int ke_model = cfg[0], veg_literal = cfg[1], fraction_scaling = cfg[2], dist_literal = cfg[3], depth_mm = cfg[4];
    const double d = depth[i], v = vel[i], r = rain[i];
    const bool a = active[i] != 0;
    const double s = slope[i];
    if (!SG_FINITE(d)) flags |= 1ull << 0;
    if (d < 0.0) flags |= 1ull << 1;
    if (!SG_FINITE(v)) flags |= 1ull << 2;
    if (v < 0.0) flags |= 1ull << 3;
    if (!SG_FINITE(r)) flags |= 1ull << 4;
    if (r < 0.0) flags |= 1ull << 5;
    if ((!a) && v != 0.0) flags |= 1ull << 8;

    const double ustar = sqrt(GRAV * d * s);
    const double re = v * d / nu;
    const double spw = 1000.0 * GRAV * d * v * s;
    const double rm = r * 1000.0;
    const double inten = r * 3.6e6;
    const bool wet = a && d > 0.0;
    const bool raining = a && r > 0.0;
    const bool conc = wet && re >= RE_CONC;
    const bool trans = wet && (!conc) && re > RE_TRANS;
    const bool rain_b = wet && (!conc) && raining;
    const bool dry_b = wet && (!conc) && (!raining);
    const bool flow_cell = conc || trans;
    int rcell;
    if (conc) rcell = 5;
    else if (rain_b && trans) rcell = 3;
    else if (rain_b) rcell = 2;
    else if (dry_b && trans) rcell = 4;
    else if (wet) rcell = 1;
    else rcell = 0;

    double ke, kf;
    if (raining) {
        const double lg = log10(inten);
        if (ke_model == 0) ke = sg_max(11.9 + 8.73 * lg, 0.0) * veg_factor[i];
        else if (veg_literal == 1) ke = 29.0 - 20.88 * exp(-180.0 * rm) * veg_factor[i];
        else ke = (29.0 - 20.88 * exp(-180.0 * rm)) * veg_factor[i];
        kf = sg_max((11.9 + 8.73 * lg) * rm, 0.0);
    } else { ke = 0.0; kf = 0.0; }
    const double energy = sg_max(ke * rm * 1.2e3, 0.0);
    if (!SG_FINITE(inten)) flags |= 1ull << 9;
    if (!SG_FINITE(energy)) flags |= 1ull << 10;
    if (!SG_FINITE(kf)) { flags |= 1ull << 12; flags |= 1ull << 30; }
    if (kf < 0.0) flags |= 1ull << 41;
    if (!SG_FINITE(ke)) flags |= 1ull << 29;
    if (ke < 0.0) flags |= 1ull << 40;
    if (!SG_FINITE(ustar)) flags |= 1ull << 26;
    if (ustar < 0.0) flags |= 1ull << 37;
    if (!SG_FINITE(re)) flags |= 1ull << 27;
    if (re < 0.0) flags |= 1ull << 38;
    if (!SG_FINITE(spw)) flags |= 1ull << 28;
    if (spw < 0.0) flags |= 1ull << 39;

    const double vd_b = (0.525 * sg_pw(kf, 2.35)) * sg_pw(spw, 0.981);
    const double ld_b = (5.0e-2 * sg_pw(kf, 1.85)) * sg_pw(spw, 0.481);
    const double d50i = d50[i];
    const double dfl = depth_mm == 1 ? d * 1000.0 : d * 1.0;
    const double log_arg = 12.0 * dfl / d50i;
    const double la = log_arg > 0.0 ? log_arg : 1.0;
    const double bag = sg_max(bag_c1[i] * log10(la), 0.0);
    const double xs = spw - bag;
    const bool capacity = xs > 0.0;
    const double xsp = capacity ? xs : 0.0;
    const double lc_a = 2.85e-3 * sg_pw(xsp, 1.31);
    const double vc = sg_min(((1.92e-2 * sg_pw(xsp, 1.01)) * (1000.0 / 3600.0)) * 1.0e-3, v);
    const double spf = sg_min(7.331976e-3 * spw, SUSP_CAP);
    const double es = 727.51805244 * exp(spf);
    if (!SG_FINITE(bag)) flags |= 1ull << 15;
    if (!SG_FINITE(vc)) flags |= 1ull << 17;
    const bool term = terminal[i] != 0;

    for (int k = 0; k < NC; ++k) {
        const ll idx = (ll)i * NC + k;
        const double f = fractions[idx], cap = cap_rate[idx];
        const double pwe = energy == 0.0 ? 0.0 : sg_pw(energy, cls[1 * NC + k]);
        const double x = (cls[0 * NC + k] * pwe) * slope_pow[idx];
        if (!SG_FINITE(x)) flags |= 1ull << 11;
        double rd = 0.0;
        if (rain_b) {
            rd = 2.0 * x / rho / ref;
            rd = rd * exp(-cls[2 * NC + k] * (d * 100.0));
            rd = sg_max(rd, 0.0);
            if (fraction_scaling == 1) rd = rd * f;
            if (k == 1 && rd > cap) rd = cap;
            if (!(f > 0.0)) rd = 0.0;
        }
        const double theta = (ustar * ustar) / cls[3 * NC + k];
        if (!SG_FINITE(theta)) flags |= 1ull << 19;
        const bool pos = theta > 0.0;
        const double arg = pos ? theta : 1.0;
        const double pc = log(0.049 / (arg * 0.25));
        const double t = pc / 0.702;
        const double inner = 1.0 - exp(p_par * (t * t));
        double sgn;
        if (pc > 0.0) sgn = 1.0; else if (pc < 0.0) sgn = -1.0; else if (pc == 0.0) sgn = 0.0; else sgn = sg_nan();
        double p = 0.5 - (0.5 * sgn) * sqrt(sg_max(inner, 0.0));
        if (!pos) p = 0.0;
        if (!SG_FINITE(p)) flags |= 1ull << 31;
        if (p < 0.0) flags |= 1ull << 42;
        if (p > 1.0) flags |= 1ull << 43;
        double fd = ((p * hz) * f) / ref;
        if (flow_cell && fd > cap) fd = cap;
        if (!(f > 0.0)) fd = 0.0;
        if (!flow_cell) fd = 0.0;
        const double rp = rd * mass, fp = fd * mass, req = rp + fp;
        if (!SG_FINITE(rp)) flags |= 1ull << 22;
        if (rp < 0.0) flags |= 1ull << 33;
        if (!SG_FINITE(fp)) flags |= 1ull << 23;
        if (fp < 0.0) flags |= 1ull << 34;
        double requested = 0.0;
        if (a) {
            requested = req;
            if (!SG_FINITE(req)) flags |= 1ull << 21;
            if (req < 0.0) flags |= 1ull << 32;
        }
        const double vd = sg_min((vd_b / cls[6 * NC + k]) * (1.0e-2 / 60.0), v);
        const double ld = ld_b * cls[7 * NC + k];
        double lc = lc_a * cls[4 * NC + k];
        double ls = es * cls[5 * NC + k];
        if (dist_literal == 1) { lc = lc * MEDIAN; ls = ls * MEDIAN; }
        lc = sg_min(lc, CONC_CAP);
        ls = ls * bag_scale;
        if (!SG_FINITE(vd)) flags |= 1ull << 13;
        if (!SG_FINITE(ld)) flags |= 1ull << 14;
        if (!SG_FINITE(lc)) flags |= 1ull << 16;
        if (!SG_FINITE(ls)) flags |= 1ull << 18;
        const double pv = prev[idx];
        const double dec = pv * decay;
        if (!SG_FINITE(dec)) flags |= 1ull << 20;
        if (!SG_FINITE(pv)) flags |= 1ull << 6;
        if (pv < 0.0) flags |= 1ull << 7;
        const bool susp = conc && (ustar >= cls[8 * NC + k]);
        const bool conc_class = (conc && !susp) || (dry_b && trans);
        const bool diff_class = rain_b;
        const bool lawk = susp || conc_class || diff_class;
        double vl, dl;  /* the law velocity and the law's travel distance */
        if (susp) { vl = v; dl = ls; }
        else if (conc_class) { vl = vc; dl = lc; }
        else { vl = vd; dl = ld; }
        const bool no_cap = (conc_class && !capacity) || (diff_class && !(kf > 0.0));
        const bool st = (!wet) || no_cap;
        const bool ldef = lawk && (!no_cap) && (dl > 0.0);
        double rt, vel_k;
        if (ldef) { rt = 1.0 / dl; vel_k = vl; } else { rt = 0.0; vel_k = dec; }
        if (!a) vel_k = 0.0;
        if (st) vel_k = 0.0;
        if (!SG_FINITE(rt)) flags |= 1ull << 25;
        if (rt < 0.0) flags |= 1ull << 36;
        if (!SG_FINITE(vel_k)) flags |= 1ull << 24;
        if (vel_k < 0.0) flags |= 1ull << 35;
        /* glue (A1 `_pack_py`) */
        const double dr = requested / dt;
        double vu = ldef ? vl : dec;
        if (!a) vu = 0.0;
        if (!(SG_FINITE(dr) && dr >= 0.0)) flags |= 1ull << 44;
        if (!(SG_FINITE(vu) && vu >= 0.0)) flags |= 1ull << 45;
        if ((!a) && dr != 0.0) flags |= 1ull << 46;
        if (ldef && dr > 0.0 && !(SG_FINITE(rt) && rt > 0.0)) flags |= 1ull << 47;
        if (term && vu != 0.0) flags |= 1ull << 48;
        det[idx] = dr;
        v_used[idx] = vu;
        rate[idx] = rt;
        law[idx] = ldef ? 1 : 0;
        regime[idx] = (signed char)(susp ? 6 : rcell);
    }
    if (flags) atomicOr(flagword, flags);
}

/* the ORIGINAL infilt.for d(1): hpre = max(h_old - max(intake - rain, 0), 0); the complete branch (intake >= h_old + rain, active
   cells, tested first) is exactly 0 */
extern "C" __global__ void sg_post_infiltration(const int n, const double* old_depth, const double* rain_m, const double* intake_m,
                                               const unsigned char* active, double* out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double h = old_depth[i], r = rain_m[i], j = intake_m[i];
    const double sum = h + r;
    const bool complete = (active[i] != 0) && (j >= sum);
    double ex = j - r;
    ex = sg_max(ex, 0.0);
    double o = h - ex;
    o = sg_max(o, 0.0);
    out[i] = complete ? 0.0 : o;
}

/* thread per (source s, RECORD class e): the flow_distrib walk of one source and the physical class k = elig[e]; every record value of
   that class is written (zero when unused). The record array holds only the NE record classes: values[record * NE + e]. A class that
   is not a record class has zero detachment by construction (zero composition in every active cell), so its record values, its
   deposition and its walk code are exactly zero and are never written (the code and deposition arrays start at, and keep, zero). */
extern "C" __global__ void sg_values(
    const int m, const int* src_cells, const ll* rec_off, const unsigned char* end_kind, const int* elig, const double* det,
    const double* rate, const unsigned char* law, const signed char* regime, const long long* limits, const unsigned char* slope_zero,
    const double dx, const double dt, double* values, unsigned char* code, ull* flagword)
{
    const ll t = (ll)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (ll)m * NE) return;
    const int s = (int)(t / NE), e = (int)(t % NE), k = elig[e];
    const int i = src_cells[s];
    const ll idx = (ll)i * NC + k;
    const ll base = rec_off[s];
    const ll plen = rec_off[s + 1] - base - 1;
    const double d = det[idx];
    unsigned char c = 0;
    ll written = 0;
    double rec0 = 0.0;
    if (d > 0.0) {
        if (!law[idx]) {
            const int r = regime[idx];
            if (slope_zero[i] && (r == 2 || r == 3)) { c = 0x40; }  /* diffuse_flow_transport 16-21: no walk, no deposition */
            else { rec0 = d; c = 0x20; }                           /* conc_flow_transport 49-54: deposit locally */
        } else {
            const double par = rate[idx];
            const double vge = 1.0e-19 * dt;
            double fract = 1.0 - exp(-par * dx);
            rec0 = fract * d;
            c = 0x10;
            ll step = 0;
            const ll limit = limits[regime[idx]];
            bool stopped = false;
            while (step < limit && fract > vge) {
                if (step >= plen) { atomicOr(flagword, 1ull << 49); break; }
                const double u = ((double)step + 2.0) * dx;
                const double lo = u - dx;
                fract = exp(-lo * par) - exp(-u * par);
                values[(base + 1 + step) * NE + e] = d * fract;
                written = step + 1;
                if (step == plen - 1 && end_kind[s] != 0) { c |= end_kind[s]; stopped = true; break; }
                step += 1;
            }
            if (!stopped) c |= (step >= limit) ? 4 : 5;
        }
    }
    values[base * NE + e] = rec0;
    for (ll j = written; j < plen; ++j) values[(base + 1 + j) * NE + e] = 0.0;
    code[(ll)s * NC + k] = c;
}

/* thread per (target cell, record class e): CPU-order sequential sum of the records crediting the cell; deposition of a non-record
   class is never written (it is exactly zero) */
extern "C" __global__ void sg_gather(const int n, const ll* tgt_ptr, const int* tgt_rec, const int* elig, const double* values,
                                    double* depos)
{
    const ll t = (ll)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (ll)n * NE) return;
    const int c = (int)(t / NE), e = (int)(t % NE);
    double acc = 0.0;
    for (ll p = tgt_ptr[c]; p < tgt_ptr[c + 1]; ++p) acc += values[(ll)tgt_rec[p] * NE + e];
    depos[(ll)c * NC + elig[e]] = acc;
}

/* one block of NE threads (one per record class): ring (CPU record order) and inactive-cell deposition rates -> ledger rows 3 and 4
   (kg per step); the rows of a non-record class stay at their initial zero */
extern "C" __global__ void sg_ring_inactive(
    const int n_ring, const int* ring_rec, const int n_in, const int* in_cells, const int* elig, const double* values,
    const double* depos, const double dt, double* ledger_row)
{
    const int e = threadIdx.x;
    if (e >= NE) return;
    const int k = elig[e];
    double ring = 0.0;
    for (int p = 0; p < n_ring; ++p) ring += values[(ll)ring_rec[p] * NE + e];
    double inact = 0.0;
    for (int p = 0; p < n_in; ++p) inact += depos[(ll)in_cells[p] * NC + k];
    ledger_row[3 * NC + k] = ring * dt;
    ledger_row[4 * NC + k] = inact * dt;
}

__device__ __forceinline__ void sg_cn_cell(
    const int i, const int k, const int* donors, const double* det, const double* depos, const double* v_used,
    const double* M1, const double* Q1, const double* Qin1, double* M2, double* Q2, double* Qin2, double* clip,
    const double dt, const double dx)
{
    const ll idx = (ll)i * NC + k;
    double qin = 0.0;
    for (int sl = 0; sl < 4; ++sl) {
        const int dn = donors[(ll)i * 4 + sl];
        if (dn >= 0) {
            const double q = Q2[(ll)dn * NC + k];
            if (q >= 0.0) qin += q;
        }
    }
    Qin2[idx] = qin;
    const double v = v_used[idx];
    const double rhs = M1[idx] / dt + 0.5 * (qin - Q1[idx] + Qin1[idx]) + (det[idx] - depos[idx]);
    double m = rhs / (1.0 / dt + 0.5 * v / dx);
    double cs = 0.0;
    if (m < 0.0) {
        cs = -m * (1.0 + 0.5 * dt * v / dx);
        m = 0.0;
    }
    clip[idx] = cs;
    M2[idx] = m;
    Q2[idx] = m * v / dx;
}

extern "C" __global__ void sg_cn_level(
    const int lo, const int hi, const int* order, const int* donors, const double* det, const double* depos,
    const double* v_used, const double* M1, const double* Q1, const double* Qin1, double* M2, double* Q2, double* Qin2,
    double* clip, const double dt, const double dx)
{
    const ll t = (ll)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (ll)(hi - lo) * NC) return;
    const int i = order[lo + (int)(t / NC)];
    sg_cn_cell(i, (int)(t % NC), donors, det, depos, v_used, M1, Q1, Qin1, M2, Q2, Qin2, clip, dt, dx);
}

/* one block, all levels, an unconditional barrier between levels (narrow networks) */
extern "C" __global__ void sg_cn_block(
    const int n_levels, const ll* bounds, const int* order, const int* donors, const double* det, const double* depos,
    const double* v_used, const double* M1, const double* Q1, const double* Qin1, double* M2, double* Q2, double* Qin2,
    double* clip, const double dt, const double dx)
{
    for (int lv = 0; lv < n_levels; ++lv) {
        const ll lo = bounds[lv], hi = bounds[lv + 1];
        for (ll t = threadIdx.x; t < (hi - lo) * NC; t += blockDim.x) {
            const int i = order[lo + t / NC];
            sg_cn_cell(i, (int)(t % NC), donors, det, depos, v_used, M1, Q1, Qin1, M2, Q2, Qin2, clip, dt, dx);
        }
        __syncthreads();
    }
}

/* partials: 7 quantities per class per block (det, dep, clip, M1, M2, terminal M2, terminal dep), the cumulative per-cell maps and the
   CPU `_check_py` validation flags (bits 50..56) */
extern "C" __global__ void sg_reduce_partial(
    const int m, const int* active_idx, const unsigned char* terminal, const double* det, const double* depos, const double* clip,
    const double* M1, const double* M2, const double* Q2, const double* Qin2, double* cum_det, double* cum_dep, double* cum_clip,
    const double dt, double* partial, ull* flagword)
{
    double acc[7][NC];
    for (int q = 0; q < 7; ++q) for (int k = 0; k < NC; ++k) acc[q][k] = 0.0;
    ull flags = 0ull;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < m; t += gridDim.x * blockDim.x) {
        const int i = active_idx[t];
        const bool term = terminal[i] != 0;
        for (int k = 0; k < NC; ++k) {
            const ll idx = (ll)i * NC + k;
            const double d = det[idx], p = depos[idx], c = clip[idx], m1 = M1[idx], m2 = M2[idx], q = Q2[idx], qi = Qin2[idx];
            if (!(SG_FINITE(d) && d >= 0.0)) flags |= 1ull << 50;
            if (!(SG_FINITE(p) && p >= 0.0)) flags |= 1ull << 51;
            if (!(SG_FINITE(c) && c >= 0.0)) flags |= 1ull << 52;
            if (!(SG_FINITE(m2) && m2 >= 0.0 && SG_FINITE(q) && q >= 0.0 && SG_FINITE(qi) && qi >= 0.0)) flags |= 1ull << 53;
            if (!SG_FINITE(m1)) flags |= 1ull << 54;
            const double nd = cum_det[idx] + d * dt, np_ = cum_dep[idx] + p * dt, nc_ = cum_clip[idx] + c;
            if (!(SG_FINITE(nd) && SG_FINITE(np_) && SG_FINITE(nc_))) flags |= 1ull << 55;
            cum_det[idx] = nd; cum_dep[idx] = np_; cum_clip[idx] = nc_;
            acc[0][k] += d; acc[1][k] += p; acc[2][k] += c; acc[3][k] += m1; acc[4][k] += m2;
            if (term) { acc[5][k] += m2; acc[6][k] += p; }
        }
    }
    __shared__ double sh[BLK];
    for (int q = 0; q < 7; ++q) {
        for (int k = 0; k < NC; ++k) {
            sh[threadIdx.x] = acc[q][k];
            __syncthreads();
            for (int off = BLK / 2; off > 0; off >>= 1) {
                if (threadIdx.x < off) sh[threadIdx.x] += sh[threadIdx.x + off];
                __syncthreads();
            }
            if (threadIdx.x == 0) partial[((ll)blockIdx.x * 7 + q) * NC + k] = sh[0];
            __syncthreads();
        }
    }
    if (flags) atomicOr(flagword, flags);
}

/* fixed-order final sum over the partial blocks; also the per-class finiteness of the sums (bit 56) */
extern "C" __global__ void sg_reduce_final(const int nblocks, const double* partial, const double dt, double* ledger_row, ull* flagword)
{
    const int t = threadIdx.x;
    if (t >= 7 * NC) return;
    const int q = t / NC, k = t % NC;
    double sum = 0.0;
    for (int b = 0; b < nblocks; ++b) sum += partial[((ll)b * 7 + q) * NC + k];
    if (!SG_FINITE(sum)) atomicOr(flagword, 1ull << 56);
    if (q == 0) ledger_row[0 * NC + k] = sum * dt;      /* pickup */
    else if (q == 1) ledger_row[1 * NC + k] = sum * dt; /* deposition_active */
    else if (q == 2) ledger_row[5 * NC + k] = sum;      /* clip */
    else if (q == 3) ledger_row[6 * NC + k] = sum;      /* old mobile */
    else if (q == 4) ledger_row[7 * NC + k] = sum;      /* new mobile */
    else if (q == 5) ledger_row[8 * NC + k] = sum;      /* terminal mobile */
    else ledger_row[2 * NC + k] = sum * dt;             /* deposition in terminal pits */
}

/* outlets in index order: CN export, endpoint export, instantaneous outlet flux (rows 9, 10, 11) */
extern "C" __global__ void sg_outlet(const int n_out, const int* out_cells, const double* Q1, const double* Q2, const double dt,
                                    double* ledger_row, ull* flagword)
{
    const int k = threadIdx.x;
    if (k >= NC) return;
    double cn = 0.0, ep = 0.0, fl = 0.0;
    for (int p = 0; p < n_out; ++p) {
        const ll idx = (ll)out_cells[p] * NC + k;
        cn += 0.5 * (Q1[idx] + Q2[idx]) * dt;
        ep += Q2[idx] * dt;
        fl += Q2[idx];
    }
    if (!(SG_FINITE(cn) && SG_FINITE(ep) && SG_FINITE(fl))) atomicOr(flagword, 1ull << 56);
    ledger_row[9 * NC + k] = cn;
    ledger_row[10 * NC + k] = ep;
    ledger_row[11 * NC + k] = fl;
    ledger_row[12 * NC + k] = 0.0;  /* erased deposition: not supported (erasure is rejected) */
}

/* integer tallies over (source, class): the walk counters (indices of legacy_native.C_*) and the wet regimes; `counts` row of 17 */
extern "C" __global__ void sg_tally(
    const int m, const int* src_cells, const signed char* regime, const unsigned char* code, const unsigned char* aspect0,
    const double* rain, ll* counts)
{
    __shared__ int sh[17];
    if (threadIdx.x < 17) sh[threadIdx.x] = 0;
    __syncthreads();
    for (ll t = (ll)blockIdx.x * blockDim.x + threadIdx.x; t < (ll)m * NC; t += (ll)gridDim.x * blockDim.x) {
        const int s = (int)(t / NC), k = (int)(t % NC);
        const int i = src_cells[s];
        const ll idx = (ll)i * NC + k;
        atomicAdd(&sh[10 + regime[idx]], 1);
        const unsigned char c = code[t];
        if (c & 0x10) {
            atomicAdd(&sh[0], 1);
            if (aspect0[i]) atomicAdd(&sh[6], 1);
            const int stop = c & 0x0f;
            if (stop == 1) atomicAdd(&sh[1], 1);
            else if (stop == 2) atomicAdd(&sh[2], 1);
            else if (stop == 3) atomicAdd(&sh[3], 1);
            else if (stop == 4) atomicAdd(&sh[4], 1);
            else if (stop == 5) atomicAdd(&sh[5], 1);
        }
        if (c & 0x20) atomicAdd(&sh[7], 1);
        if (c & 0x40) atomicAdd(&sh[8], 1);
        if (k == 0) {  /* the CPU walk counts erase cells (wet no-rain, or dry raining) once per source cell even with erasure off */
            const int r0 = regime[(ll)i * NC];
            if (r0 == 1 || (r0 == 0 && rain[i] > 0.0)) atomicAdd(&sh[9], 1);
        }
    }
    __syncthreads();
    if (threadIdx.x < 17) atomicAdd((unsigned long long*)&counts[threadIdx.x], (unsigned long long)sh[threadIdx.x]);
}
"""


def kernel_source(nc: int, ne: int | None = None) -> str:
    """CUDA source for `nc` physical classes and `ne` record classes (default `nc`: the all-class reference strategy)."""
    ne = nc if ne is None else ne
    if not (isinstance(nc, (int, np.integer)) and isinstance(ne, (int, np.integer)) and not isinstance(nc, bool)
            and not isinstance(ne, bool) and 1 <= nc <= MAX_CLASSES and 0 <= ne <= nc):
        raise CudaLegacyError(f"need 1 <= nc <= {MAX_CLASSES} and 0 <= ne <= nc; got nc={nc!r}, ne={ne!r}")
    text = (_TEMPLATE.replace("__NC__", str(int(nc))).replace("__NEX__", str(max(1, int(ne)))).replace("__NE__", str(int(ne)))
            .replace("__BLK__", str(REDUCE_THREADS))
            .replace("__RE_CONC__", _c(REYNOLDS_CONCENTRATED)).replace("__RE_TRANS__", _c(REYNOLDS_TRANSITIONAL))
            .replace("__CONC_CAP__", _c(CONCENTRATED_DISTANCE_CAP_M)).replace("__SUSP_CAP__", _c(SUSPENDED_EXPONENT_CAP))
            .replace("__GRAV__", _c(GRAVITY_M_S2)).replace("__MEDIAN__", _c(LEGACY_MEDIAN_FACTOR)))
    return text


_KERNEL_NAMES = ("sg_laws", "sg_post_infiltration", "sg_values", "sg_gather", "sg_ring_inactive", "sg_cn_level", "sg_cn_block",
                 "sg_reduce_partial", "sg_reduce_final", "sg_outlet", "sg_tally")
_MODULES: dict[tuple[int, int, int], Any] = {}


def kernel_provenance(nc: int = 6, ne: int | None = None) -> dict[str, Any]:
    ne = nc if ne is None else ne
    text = kernel_source(nc, ne)
    return {"module": "maple_syrup.legacy_native_cuda", "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "compile_options": list(COMPILE_OPTIONS), "fastmath": False, "kernels": list(_KERNEL_NAMES),
            "threads": THREADS, "reduce_threads": REDUCE_THREADS, "n_classes_compiled": nc, "n_record_classes_compiled": ne,
            "determinism": "no floating point atomics; fixed-order block partials and CPU-order target gather; integer atomics only"}


def _functions(cp, nc: int, ne: int) -> dict[str, Any]:
    device = _current_device_id(cp)
    key = (device, nc, ne)  # the record-class count is compiled in: it is part of the cache key
    if key not in _MODULES:
        try:
            module = cp.RawModule(code=kernel_source(nc, ne), options=COMPILE_OPTIONS, backend="nvrtc")
            fns = {name: module.get_function(name) for name in _KERNEL_NAMES}
            for name, fn in fns.items():
                limit = int(dict(fn.attributes).get("max_threads_per_block", 0))
                need = LAUNCH_BLOCKS[name]  # exactly the block size this kernel is launched with
                if limit < need:
                    raise CudaUnavailableError(f"kernel {name} supports only {limit} threads per block, {need} needed; no fallback")
        except CudaUnavailableError:
            raise
        except Exception as exc:
            raise CudaUnavailableError(f"legacy CUDA kernels failed to compile/load ({type(exc).__name__}: {exc}); no fallback") from exc
        _MODULES[key] = fns
    return _MODULES[key]


# --- the context ----------------------------------------------------------------------------------------------------------
class CudaLegacyContext:
    """Resident device state of one legacy storm. Static arrays are uploaded once and never modified; dynamic arrays ping-pong.

    `step(row, depth, velocity, rain)` enqueues one legacy step on the current stream and returns nothing (no host read). Failures are
    detected by `check_flags` (small device-to-host slices). A poisoned context refuses further steps until `reset()`."""

    def __init__(self, network: N.NativeNetwork, physics: LegacyPhysicsContext, graph: Any, *, limits: np.ndarray, dt: float,
                 n_steps: int, tables: WalkTables | None = None, memory_budget_bytes: int | None = None,
                 cn_mode: str = "auto", source_order: str = "index", erase_on: bool = False,
                 record_strategy: str = "compact", record_classes=None):
        if source_order != "index" or erase_on:
            raise CudaLegacyError("the CUDA replay supports source_order='index' without erasure only (rejected before any mutation)")
        if cn_mode not in ("auto", "level", "block"):
            raise CudaLegacyError("cn_mode must be auto, level or block")
        if not (isinstance(dt, float) and dt > 0.0 and math.isfinite(dt)):
            raise CudaLegacyError("dt must be a positive finite float")
        if not isinstance(n_steps, int) or isinstance(n_steps, bool) or not 1 <= n_steps <= INT32_MAX // 64:
            raise CudaLegacyError("n_steps must be a positive int below 2**31 / 64")
        if memory_budget_bytes is not None and (isinstance(memory_budget_bytes, bool) or not isinstance(memory_budget_bytes, int)
                                                or memory_budget_bytes < 1):
            raise CudaLegacyError("memory_budget_bytes must be a positive int")
        if not isinstance(network, N.NativeNetwork) or not isinstance(physics, LegacyPhysicsContext):
            raise CudaLegacyError("network must be a NativeNetwork and physics a LegacyPhysicsContext")
        limits = validate_limits(limits)
        if tuple(physics.shape) != tuple(network.shape) or not np.array_equal(np.asarray(physics.active), network.active):
            raise CudaLegacyError("the wet-law context and the network differ")
        nc = int(physics.n_classes)
        if not 1 <= nc <= MAX_CLASSES:
            raise CudaLegacyError(f"1 to {MAX_CLASSES} grain classes are supported (every launch is sized for them)")
        if int(network.active.size) * nc > INT32_MAX:
            raise CudaLegacyError("cells x classes exceeds the signed 32-bit launch arithmetic")
        if tables is not None:
            validate_walk_tables(network, tables, limits)  # one-time, before any device work
        # record classes from the immutable composition (static, host only); supplied parameters are validated before any allocation
        elig = resolve_record_classes(network, np.asarray(physics.fractions, dtype=np.float64).reshape(int(network.active.size), nc),
                                      record_strategy, record_classes)
        t0 = time.perf_counter()
        self.cp = cp = _cupy()
        self._device_id = _current_device_id(cp)
        self.record_strategy = record_strategy
        self.elig_host = elig
        self.elig_host.setflags(write=False)
        self.ne = ne = int(elig.size)
        self.nc = nc
        self.n = n = int(network.active.size)
        self.shape = tuple(network.shape)
        self.dt, self.dx, self.n_steps = dt, float(network.dx_m), n_steps
        self.m = int(network.active_idx.size)
        self.tables = tables if tables is not None else build_walk_tables(network, limits)
        self.estimate = estimate_bytes(network, physics, self.tables.n_records, n_steps, nc, self.tables.nbytes(), ne)
        self.record_info = {"strategy": record_strategy, "n_physical_classes": nc, "n_record_classes": ne,
                            "record_classes": [int(k) for k in elig], "excluded_classes": [k for k in range(nc) if k not in set(elig.tolist())],
                            "eligibility": "static: a class is a record class iff some active cell has a positive fraction",
                            "record_values_bytes_estimated": int(self.estimate["walk_values"]),
                            "all_class_reference_values_bytes": int(self.tables.n_records) * nc * 8,
                            "n_records": int(self.tables.n_records)}
        free, total = cp.cuda.runtime.memGetInfo()
        self.memory = {"free_before_bytes": int(free), "total_bytes": int(total)}
        budget = memory_budget_bytes if memory_budget_bytes is not None else int(free * 0.9)
        if self.estimate["total"] > budget or self.estimate["total"] > free:
            raise CudaLegacyError(f"estimated device memory {self.estimate['total'] / 2**30:.2f} GiB exceeds the budget "
                                  f"{budget / 2**30:.2f} GiB / free {free / 2**30:.2f} GiB; refusing before any allocation")
        self.fns = _functions(cp, nc, ne)
        self.compile_s = time.perf_counter() - t0
        self.stats = {"h2d_static_bytes": 0, "d2h_flag_bytes": 0, "d2h_flag_reads": 0, "d2h_row_bytes": 0, "launches": 0,
                      "steps": 0}
        up = self._upload
        # static network and physics (read-only by contract)
        self.d_active = up(network.active.astype(np.uint8))
        self.d_terminal = up(network.terminal.astype(np.uint8))
        self.d_slope_zero = up(network.slope_zero.astype(np.uint8))
        self.d_aspect0 = up(network.aspect0.astype(np.uint8))
        self.d_donors = up(network.donors.astype(np.int32))
        self.d_order = up(network.order.astype(np.int32))
        self.d_active_idx = up(network.active_idx.astype(np.int32))
        self.d_outlet_idx = up(network.outlet_idx.astype(np.int32))
        bounds = level_bounds(graph, network.order)
        self.d_bounds = up(bounds)
        self.n_levels = int(bounds.size - 1)
        self.max_level_width = int(np.max(np.diff(bounds))) if self.n_levels else 0
        self.cn_mode = ("block" if self.max_level_width <= THREADS else "level") if cn_mode == "auto" else cn_mode
        self.bounds_host = bounds
        self.d_slope, self.d_veg_factor, self.d_d50, self.d_bag_c1 = (up(np.asarray(a, dtype=np.float64)) for a in (
            physics.slope, physics.veg_factor, physics.d50, physics.bagnold_prefactor))
        self.d_slope_pow, self.d_fractions, self.d_cap_rate = (up(np.asarray(a, dtype=np.float64)) for a in (
            physics.slope_power, physics.fractions, physics.cap_rate))
        self.d_cls = up(np.ascontiguousarray(physics.class_constants, dtype=np.float64))
        self.d_cfg = up(np.asarray(physics.config, dtype=np.int32))
        sc = np.zeros(10)
        sc[0], sc[1] = physics.particle_density_kg_m3, physics.reference_interval_s
        sc[2] = physics.cell_area_m2 * physics.particle_density_kg_m3 * dt
        sc[3], sc[4] = physics.kinematic_viscosity_m2_s, physics.flow_detachment_depth_scale_m
        sc[6] = physics.bagnold_density_scale
        sc[7] = recession_velocity(1.0, dt, factor_per_reference_s=physics.recession_factor_per_reference_s,
                                   reference_interval_s=physics.reference_interval_s)
        sc[8], sc[9] = physics.p_par, dt
        self.d_sc = up(sc)
        self.d_limits = up(limits)
        tb = self.tables
        self.d_src_cells, self.d_end_kind = up(tb.src_cells), up(tb.end_kind)
        self.d_rec_off, self.d_tgt_ptr = up(tb.rec_off), up(tb.tgt_ptr)
        self.d_tgt_rec, self.d_ring_rec, self.d_in_cells = up(tb.tgt_rec), up(tb.ring_rec), up(tb.inactive_targets)
        self.d_elig = up(elig if ne else np.zeros(1, dtype=np.int32))  # never read when ne == 0 (the record launches are skipped)
        # dynamic state
        f8 = np.float64
        z = cp.zeros
        nn = n * nc
        self.det, self.v_used, self.rate = z(nn, f8), z(nn, f8), z(nn, f8)
        self.v_prev = z(nn, f8)
        self.depos, self.clip = z(nn, f8), z(nn, f8)
        self.M1, self.M2, self.Q1, self.Q2, self.Qin1, self.Qin2 = (z(nn, f8) for _ in range(6))
        self.cum_det, self.cum_dep, self.cum_clip = z(nn, f8), z(nn, f8), z(nn, f8)
        self.law = z(nn, np.uint8)
        self.regime = z(nn, np.int8)
        self.values = z(tb.n_records * ne if ne else 1, f8)  # records x RECORD classes (compact) or x all classes (reference)
        self.code = z(self.m * nc, np.uint8)
        self.nblocks = max(1, min(MAX_REDUCE_BLOCKS, -(-self.m // REDUCE_THREADS)))
        self.partial = z(MAX_REDUCE_BLOCKS * 7 * nc, f8)
        self.ledger = z(n_steps * LEDGER_ROWS * nc, f8)
        self.flags = z(n_steps, np.uint64)
        self.counts = z(n_steps * N_COUNT_COLUMNS, np.int64)
        self.row = 0
        self._checked = 0
        self.poisoned: str | None = None
        self.host_flags = np.zeros(n_steps, dtype=np.uint64)
        self.host_ledger = np.zeros((n_steps, LEDGER_ROWS, nc))
        self.host_counts = np.zeros((n_steps, N_COUNT_COLUMNS), dtype=np.int64)
        free_after, _ = cp.cuda.runtime.memGetInfo()
        self.memory["free_after_allocation_bytes"] = int(free_after)
        self.memory["allocated_by_context_bytes"] = int(free - free_after)
        self.bounds_host.setflags(write=False)
        self._seal()

    # -- sealed structure (creation device, scalars, array metadata; pointer/shape/dtype/stride, never content) --
    _STATIC = ("d_active", "d_terminal", "d_slope_zero", "d_aspect0", "d_donors", "d_order", "d_active_idx", "d_outlet_idx",
               "d_bounds", "d_slope", "d_veg_factor", "d_d50", "d_bag_c1", "d_slope_pow", "d_fractions", "d_cap_rate", "d_cls",
               "d_cfg", "d_sc", "d_limits", "d_src_cells", "d_end_kind", "d_rec_off", "d_tgt_ptr", "d_tgt_rec", "d_ring_rec",
               "d_in_cells", "d_elig")
    _DYNAMIC = ("det", "rate", "depos", "clip", "cum_det", "cum_dep", "cum_clip", "law", "regime", "values", "code", "partial",
                "ledger", "flags", "counts")
    _PAIRS = (("M1", "M2"), ("Q1", "Q2"), ("Qin1", "Qin2"), ("v_prev", "v_used"))  # swapped by step(): the PAIR is the identity
    _HOST = ("host_flags", "host_ledger", "host_counts")

    @staticmethod
    def _meta(a: Any):
        if hasattr(a, "data") and hasattr(a.data, "ptr"):  # device array
            return ("dev", int(a.data.ptr), tuple(a.shape), a.dtype.str, tuple(a.strides), int(a.device.id))
        return ("host", int(a.ctypes.data), tuple(a.shape), a.dtype.str, tuple(a.strides))

    def _scalar_signature(self) -> tuple:
        tb = self.tables
        return (self._device_id, self.n, self.nc, self.m, self.n_steps, self.dt, self.dx, self.cn_mode, self.n_levels,
                self.max_level_width, tuple(self.shape), self.bounds_host.tobytes(), self.nblocks, id(tb), tb.n_records,
                int(tb.ring_rec.size), int(tb.inactive_targets.size), int(tb.src_cells.size), int(self.d_outlet_idx.size),
                self.record_strategy, self.ne, self.elig_host.tobytes())

    def _seal(self) -> None:
        self._sealed_scalars = self._scalar_signature()
        self._sealed_static = {k: self._meta(getattr(self, k)) for k in self._STATIC}
        self._sealed_dynamic = {k: self._meta(getattr(self, k)) for k in self._DYNAMIC}
        self._sealed_pairs = {p: frozenset(self._meta(getattr(self, k)) for k in p) for p in self._PAIRS}
        self._sealed_host = {k: self._meta(getattr(self, k)) for k in self._HOST}

    def _guard(self, op: str) -> None:
        """Before ANY launch, read or mutation: the current device is the creation device and every sealed scalar and array
        metadata item is intact (a replaced or resized array, a changed `n`/`nc`/`dt`/`n_steps`/bounds/tables is refused)."""
        cp = self.cp
        if int(cp.cuda.Device().id) != self._device_id:
            raise CudaLegacyError(f"{op}: the current CUDA device differs from the context's creation device {self._device_id}")
        try:
            if self._scalar_signature() != self._sealed_scalars:
                raise CudaLegacyError(f"{op}: a sealed context scalar or table changed")
            for group, sealed in (("static", self._sealed_static), ("dynamic", self._sealed_dynamic), ("host", self._sealed_host)):
                for name, meta in sealed.items():
                    if self._meta(getattr(self, name)) != meta:
                        raise CudaLegacyError(f"{op}: {group} array {name} was replaced or its metadata changed")
            for pair, meta in self._sealed_pairs.items():
                a, b = (getattr(self, k) for k in pair)
                if a is b or frozenset((self._meta(a), self._meta(b))) != meta:
                    raise CudaLegacyError(f"{op}: the ping-pong pair {pair} is not the sealed pair")
        except (AttributeError, TypeError, ValueError) as exc:
            raise CudaLegacyError(f"{op}: the sealed context structure is damaged ({type(exc).__name__}: {exc})") from exc

    @staticmethod
    def _strict_int(value: Any, name: str) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise CudaLegacyError(f"{name} must be an integer, got {type(value).__name__}")
        return int(value)

    # -- helpers --
    def _upload(self, a: np.ndarray):
        a = np.ascontiguousarray(a)
        self.stats["h2d_static_bytes"] += int(a.nbytes)
        return self.cp.asarray(a)

    def _launch(self, name: str, grid: int, block: int, args: tuple) -> None:
        self.fns[name]((max(1, int(grid)),), (int(block),), args)
        self.stats["launches"] += 1

    def _check_dev(self, a: Any, name: str):
        cp = self.cp
        if type(a) is not cp.ndarray or a.dtype != np.float64 or tuple(a.shape) != self.shape or not a.flags.c_contiguous:
            raise CudaLegacyError(f"{name} must be a C-contiguous float64 cupy array of shape {self.shape} on the current device")
        if int(a.device.id) != int(cp.cuda.Device().id):
            raise CudaLegacyError(f"{name} lives on another CUDA device")

    @property
    def nbytes_persistent(self) -> int:
        return int(self.memory.get("allocated_by_context_bytes", 0))

    # -- kernels --
    def post_infiltration_depth(self, old_depth, rain_m, intake_m, out):
        """The ORIGINAL d(1) on the device (same formula as `legacy_driver.post_infiltration_depth`); `out` must not alias an input."""
        self._guard("post_infiltration_depth")
        for a, nm in ((old_depth, "old_depth"), (rain_m, "rain_m"), (intake_m, "intake_m"), (out, "out")):
            self._check_dev(a, nm)
        o0 = int(out.data.ptr)
        o1 = o0 + int(out.nbytes)
        for a, nm in ((old_depth, "old_depth"), (rain_m, "rain_m"), (intake_m, "intake_m")):
            a0 = int(a.data.ptr)
            if a0 < o1 and o0 < a0 + int(a.nbytes):  # exact byte-span overlap: shifted contiguous views are caught
                raise CudaLegacyError(f"out overlaps the memory of {nm}")
        self._launch("sg_post_infiltration", -(-self.n // THREADS), THREADS,
                     (np.int32(self.n), old_depth, rain_m, intake_m, self.d_active, out))

    def _walk_launches(self, flag_slot, ledger_row) -> None:
        """Record values, CPU-order gather and the ring/inactive ledger rows for the RECORD classes only. With no record class nothing
        carries detachment: `depos` and the ring/inactive rows stay at their zero state and no launch is made (no empty grid)."""
        if self.ne == 0:
            return
        m, n, dt, dx = self.m, self.n, self.dt, self.dx
        self._launch("sg_values", -(-(m * self.ne) // THREADS), THREADS, (
            np.int32(m), self.d_src_cells, self.d_rec_off, self.d_end_kind, self.d_elig, self.det, self.rate, self.law, self.regime,
            self.d_limits, self.d_slope_zero, np.float64(dx), np.float64(dt), self.values, self.code, flag_slot))
        self._launch("sg_gather", -(-(n * self.ne) // THREADS), THREADS, (
            np.int32(n), self.d_tgt_ptr, self.d_tgt_rec, self.d_elig, self.values, self.depos))
        self._launch("sg_ring_inactive", 1, 32, (
            np.int32(self.tables.ring_rec.size), self.d_ring_rec, np.int32(self.tables.inactive_targets.size), self.d_in_cells,
            self.d_elig, self.values, self.depos, np.float64(dt), ledger_row))

    def step(self, row: int, depth, velocity, rain) -> None:
        """Enqueue legacy step `row` (0-based, consecutive). `depth` is the wet-law depth of the declared time level."""
        self._guard("step")
        if self.poisoned is not None:
            raise CudaLegacyError(f"the context is poisoned ({self.poisoned}); reset() or build a new one")
        row = self._strict_int(row, "row")
        if row != self.row or not 0 <= row < self.n_steps:
            raise CudaLegacyError(f"step rows must be consecutive from 0; got {row}, expected {self.row}")
        for a, nm in ((depth, "depth"), (velocity, "velocity"), (rain, "rain")):
            self._check_dev(a, nm)
        nc, n, m, dt, dx = self.nc, self.n, self.m, self.dt, self.dx
        flag_slot = self.flags[row:row + 1]
        ledger_row = self.ledger[row * LEDGER_ROWS * nc:(row + 1) * LEDGER_ROWS * nc]
        counts_row = self.counts[row * N_COUNT_COLUMNS:(row + 1) * N_COUNT_COLUMNS]
        self._launch("sg_laws", -(-n // THREADS), THREADS, (
            np.int32(n), depth, velocity, rain, self.v_prev, self.d_active, self.d_terminal, self.d_slope, self.d_veg_factor,
            self.d_d50, self.d_bag_c1, self.d_slope_pow, self.d_fractions, self.d_cap_rate, self.d_cls, self.d_sc, self.d_cfg,
            self.det, self.v_used, self.rate, self.law, self.regime, flag_slot))
        self._walk_launches(flag_slot, ledger_row)
        if self.cn_mode == "block":
            self._launch("sg_cn_block", 1, THREADS, (
                np.int32(self.n_levels), self.d_bounds, self.d_order, self.d_donors, self.det, self.depos, self.v_used, self.M1,
                self.Q1, self.Qin1, self.M2, self.Q2, self.Qin2, self.clip, np.float64(dt), np.float64(dx)))
        else:
            for lv in range(self.n_levels):
                lo, hi = int(self.bounds_host[lv]), int(self.bounds_host[lv + 1])
                self._launch("sg_cn_level", -(-((hi - lo) * nc) // THREADS), THREADS, (
                    np.int32(lo), np.int32(hi), self.d_order, self.d_donors, self.det, self.depos, self.v_used, self.M1, self.Q1,
                    self.Qin1, self.M2, self.Q2, self.Qin2, self.clip, np.float64(dt), np.float64(dx)))
        self._launch("sg_reduce_partial", self.nblocks, REDUCE_THREADS, (
            np.int32(m), self.d_active_idx, self.d_terminal, self.det, self.depos, self.clip, self.M1, self.M2, self.Q2, self.Qin2,
            self.cum_det, self.cum_dep, self.cum_clip, np.float64(dt), self.partial, flag_slot))
        self._launch("sg_reduce_final", 1, 256, (np.int32(self.nblocks), self.partial, np.float64(dt), ledger_row, flag_slot))
        self._launch("sg_outlet", 1, 32, (
            np.int32(int(self.d_outlet_idx.size)), self.d_outlet_idx, self.Q1, self.Q2,
            np.float64(dt), ledger_row, flag_slot))
        self._launch("sg_tally", min(256, -(-(m * nc) // THREADS)), THREADS, (
            np.int32(m), self.d_src_cells, self.regime, self.code, self.d_aspect0, rain, counts_row))
        # time-level swap (host pointer swap: no data dependence)
        self.M1, self.M2 = self.M2, self.M1
        self.Q1, self.Q2 = self.Q2, self.Q1
        self.Qin1, self.Qin2 = self.Qin2, self.Qin1
        self.v_prev, self.v_used = self.v_used, self.v_prev
        self.row += 1
        self.stats["steps"] += 1

    # -- reads (small, counted) --
    def check_flags(self, upto: int | None = None) -> int:
        """Read the new per-step flag words and, with them, the ledger and tally rows (one counted slice read each), store them on the
        host and POISON the context if any word is nonzero. Returns the number of rows now on the host."""
        self._guard("check_flags")
        cp = self.cp
        upto = self.row if upto is None else self._strict_int(upto, "upto")
        if not self._checked <= upto <= self.row <= self.n_steps:  # nothing is transferred for an invalid request
            raise CudaLegacyError(f"upto must satisfy {self._checked} <= upto <= {self.row} (rows executed); got {upto}")
        a, b = self._checked, upto
        if b == a:
            return a
        nc = self.nc
        flags = cp.asnumpy(self.flags[a:b])
        led = cp.asnumpy(self.ledger[a * LEDGER_ROWS * nc:b * LEDGER_ROWS * nc]).reshape(b - a, LEDGER_ROWS, nc)
        cnt = cp.asnumpy(self.counts[a * N_COUNT_COLUMNS:b * N_COUNT_COLUMNS]).reshape(b - a, N_COUNT_COLUMNS)
        self.stats["d2h_flag_reads"] += 1
        self.stats["d2h_flag_bytes"] += int(flags.nbytes)
        self.stats["d2h_row_bytes"] += int(led.nbytes + cnt.nbytes)
        self.host_flags[a:b], self.host_ledger[a:b], self.host_counts[a:b] = flags, led, cnt
        self._checked = b
        bad = np.flatnonzero(flags)
        if bad.size:
            row = a + int(bad[0])
            self.poisoned = f"step {row}: " + "; ".join(decode_flags(int(flags[bad[0]])))
            raise CudaLegacyError(f"legacy CUDA step failed validation ({self.poisoned}); the context is poisoned, nothing is published")
        if not np.all(np.isfinite(led)):
            self.poisoned = "non-finite ledger value"
            raise CudaLegacyError("a non-finite ledger value was produced; the context is poisoned")
        return b

    def download_maps(self) -> dict[str, np.ndarray]:
        """Final per-cell maps `(n, nc)` on the host (counted transfers; call once, after the loop)."""
        self._guard("download_maps")
        cp = self.cp
        out = {name: cp.asnumpy(getattr(self, name)).reshape(self.n, self.nc) for name in (
            "cum_det", "cum_dep", "cum_clip")}
        out["mobile"] = cp.asnumpy(self.M1).reshape(self.n, self.nc)
        out["v_prev"] = cp.asnumpy(self.v_prev).reshape(self.n, self.nc)
        self.stats["d2h_map_bytes"] = int(sum(a.nbytes for a in out.values()))
        return out

    def regime_tallies(self) -> np.ndarray:
        self._guard("regime_tallies")
        return self.host_counts[:self._checked, N.N_COUNTS:]

    def walk_tallies(self) -> np.ndarray:
        self._guard("walk_tallies")
        return self.host_counts[:self._checked, :N.N_COUNTS]

    def walk_only(self, det, inv_l, law, regime) -> dict[str, np.ndarray]:
        """TEST HOOK: run only the walk (`sg_values`, `sg_gather`, ring/inactive, tally) on caller-supplied host `(n, nc)` detachment
        rates, 1/L, law mask and regime codes, bypassing the wet laws, so the walk rules (aspect-0 west start, credit before stop,
        zero-slope no-law, local deposits) are exercised directly. Allowed only on a fresh/reset context (row 0); it overwrites the
        scratch arrays and `reset()` restores a clean state. Returns host deposition rates, ring/inactive ledger rows (kg per step),
        record codes and the 17 tallies. Uploads are counted as `h2d_test_bytes`."""
        self._guard("walk_only")
        if self.row != 0 or self.poisoned is not None:
            raise CudaLegacyError("walk_only needs a fresh (row 0, not poisoned) context")
        cp, n, nc = self.cp, self.n, self.nc
        spec = (("det", det, np.float64), ("inv_l", inv_l, np.float64), ("law", law, np.bool_), ("regime", regime, np.int8))
        for name, a, dtype in spec:
            if type(a) is not np.ndarray or a.dtype != dtype or a.shape != (n, nc):
                raise CudaLegacyError(f"{name} must be a host {np.dtype(dtype)} ndarray of shape {(n, nc)}")
        if np.any(~np.isfinite(det)) or np.any(det < 0) or np.any(regime < 0) or np.any(regime >= N_REG):
            raise CudaLegacyError("det must be finite and >= 0 and regime codes within [0, 7)")
        excluded = np.setdiff1d(np.arange(nc), self.elig_host)
        if excluded.size and np.any(det[:, excluded] > 0.0):  # refused BEFORE any mutation: never silently dropped
            raise CudaLegacyError(f"forced detachment in non-record classes {excluded.tolist()[:8]}: use record_strategy='all' "
                                  "(the compact strategy has no record values for them)")
        self.det[:] = cp.asarray(np.ascontiguousarray(det).reshape(-1))
        self.rate[:] = cp.asarray(np.ascontiguousarray(inv_l).reshape(-1))
        self.law[:] = cp.asarray(np.ascontiguousarray(law).reshape(-1).astype(np.uint8))
        self.regime[:] = cp.asarray(np.ascontiguousarray(regime).reshape(-1))
        self.stats["h2d_test_bytes"] = self.stats.get("h2d_test_bytes", 0) + int(det.nbytes + inv_l.nbytes + law.size + regime.nbytes)
        flag = cp.zeros(1, dtype=np.uint64)
        row = cp.zeros(LEDGER_ROWS * nc, dtype=np.float64)
        counts = cp.zeros(N_COUNT_COLUMNS, dtype=np.int64)
        zero_rain = cp.zeros(n, dtype=np.float64)
        m = self.m
        self._walk_launches(flag, row)
        self._launch("sg_tally", min(256, -(-(m * nc) // THREADS)), THREADS, (
            np.int32(m), self.d_src_cells, self.regime, self.code, self.d_aspect0, zero_rain, counts))
        led = cp.asnumpy(row).reshape(LEDGER_ROWS, nc)
        return {"depos": cp.asnumpy(self.depos).reshape(n, nc), "ring_kg_per_step": led[3], "inactive_kg_per_step": led[4],
                "code": cp.asnumpy(self.code).reshape(self.m, nc), "counts": cp.asnumpy(counts), "flag": int(cp.asnumpy(flag)[0])}

    def reset(self) -> None:
        """Zero every dynamic buffer and clear the poison flag; static tables are kept. The counters of the run so far are kept as
        `stats_before_reset` (e.g. the warm-up) and the measured counters restart, so later per-step figures are not contaminated."""
        self._guard("reset")
        self.stats_before_reset = dict(self.stats)
        for key in ("d2h_flag_bytes", "d2h_flag_reads", "d2h_row_bytes", "launches", "steps", "d2h_map_bytes", "h2d_test_bytes"):
            if key in self.stats:
                self.stats[key] = 0
        for name in ("det", "v_used", "v_prev", "rate", "depos", "clip", "M1", "M2", "Q1", "Q2", "Qin1", "Qin2", "cum_det",
                     "cum_dep", "cum_clip", "law", "regime", "values", "code", "partial", "ledger", "flags", "counts"):
            getattr(self, name).fill(0)
        self.row = 0
        self._checked = 0
        self.poisoned = None
        self.host_flags[:] = 0
        self.host_ledger[:] = 0
        self.host_counts[:] = 0
