"""Compiled wrappers and preallocated step engine for the native-walk legacy benchmark (task gpu_sediment, A1).

* `get_kernels(compiled)`: the `legacy_native` plain-loop kernels compiled with Numba (`fastmath` off), or the pure-Python
  originals (`compiled=False`, tests only). The Crank-Nicolson kernel is the ACCEPTED `legacy_transport._cn` / `_cn_py`,
  unchanged. Missing Numba with `compiled=True` is an explicit error, never a silent fallback.
* `WetLawRunner`: calls the accepted compiled wet-law kernel (`legacy_physics_numba.compiled_kernel`, shared code unchanged)
  on a prepared `LegacyPhysicsContext`, but writes into buffers allocated ONCE instead of 17 fresh full-grid arrays per step.
  Validation is the kernel's own flag word, raised as `SedimentPhysicsError` (as `legacy_physics_step` does).
* `StepEngine`: one legacy step on reused buffers: wet laws, glue (`det`, `v_used`), source walk, Crank-Nicolson pool, a
  compiled validation pass, per-class reduction and cumulative per-cell maps. Ping-pong time levels (level 1 = the previous
  step's level 2).

Failure contract of `StepEngine.step`:

1. Inputs must be host `numpy.ndarray` of dtype float64 and the grid shape (no conversion, no other namespace). A refusal here,
   and a refusal by the wet laws or the glue validation, happen before any PERSISTENT state (pools, fluxes, velocity memory,
   cumulative maps, tallies) or the walk scratch is touched; the engine stays usable. Wet-law output buffers are scratch and may
   hold the failed step's values.
2. From the walk onward the engine-owned scratch (deposition and clipping buffers) is dirty. The compiled validation pass
   (`_check_py`) refuses any non-finite or negative new pool/flux/inflow/deposition/detachment/clip, an overflowing
   cumulative map and a non-finite per-class sum BEFORE the reduction, the cumulative maps and the swap. The accepted
   Crank-Nicolson kernel only clips negative pools, so NaN or Infinity (including a clip source of Infinity) is refused here, not
   laundered. Any failure after the walk POISONS the engine: `step` then raises until `reset()` or a new engine. The persistent
   time levels are still unchanged at that point, but continuation is declared unusable and the driver abandons the run
   (nothing is published).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from maple_syrup import legacy_native as N
from maple_syrup import legacy_transport as L
from maple_syrup.legacy_physics_numba import (
    _FLAG_MESSAGES,
    _N_SC,
    _S_BAG_SCALE,
    _S_DECAY,
    _S_HZ,
    _S_MASS,
    _S_NU,
    _S_P_PAR,
    _S_REF,
    _S_RHO,
    LegacyPhysicsContext,
    LegacyPhysicsNumbaUnavailableError,
    compiled_kernel,
)
from maple_syrup.sediment_physics import (
    REGIME_CODES,
    SedimentPhysicsError,
    recession_velocity,
)

__all__ = ["LEDGER_COLUMNS", "LEDGER_UNITS", "LegacyNativeError", "StepEngine", "StepResult", "WetLawRunner", "get_kernels"]

_PACK_MESSAGES = ("detachment must be finite and >= 0", "sediment velocity must be finite and >= 0",
                  "an inactive cell has detachment", "a law applies with detachment but 1/L is not positive and finite",
                  "a terminal pit has a nonzero sediment velocity (it has no receiver)")
_KERNELS: dict[bool, SimpleNamespace] = {}


class LegacyNativeError(ValueError):
    pass


def get_kernels(compiled: bool = True) -> SimpleNamespace:
    if compiled in _KERNELS:
        return _KERNELS[compiled]
    if compiled:
        try:
            import numba
        except ImportError as exc:
            raise LegacyPhysicsNumbaUnavailableError(
                "compiled native legacy kernels were requested but Numba is not installed; there is no fallback") from exc
        if L.KERNEL_IMPLEMENTATION != "numba":
            raise LegacyPhysicsNumbaUnavailableError("legacy_transport kernels are running as pure Python (Numba import failed)")
        jit = numba.njit(cache=False, fastmath=False, nogil=True)
        ns = SimpleNamespace(pack=jit(N._pack_py), walk=jit(N._walk_py), check=jit(N._check_py),
                             reduce=jit(N._reduce_py), cn=L._cn, compiled=True)
    else:
        ns = SimpleNamespace(pack=N._pack_py, walk=N._walk_py, check=N._check_py, reduce=N._reduce_py, cn=L._cn_py,
                             compiled=False)
    _KERNELS[compiled] = ns
    return ns


def _host_f64(name: str, a, shape: tuple[int, ...]) -> np.ndarray:
    """Strict host input: exactly a NumPy ndarray, float64, the declared shape. No conversion is attempted."""
    if type(a) is not np.ndarray:
        raise LegacyNativeError(f"{name} must be a host numpy.ndarray, got {type(a).__module__}.{type(a).__name__}")
    if a.dtype != np.float64:
        raise LegacyNativeError(f"{name} must be float64, got {a.dtype}")
    if tuple(a.shape) != tuple(shape):
        raise LegacyNativeError(f"{name} must have shape {tuple(shape)}, got {tuple(a.shape)}")
    return a


class WetLawRunner:
    """Accepted compiled wet-law kernel with reusable output buffers (flat C-contiguous, class-minor)."""

    def __init__(self, ctx: LegacyPhysicsContext):
        if not isinstance(ctx, LegacyPhysicsContext):
            raise LegacyNativeError("ctx must be a LegacyPhysicsContext from prepare_legacy_physics")
        self.ctx = ctx
        ny, nx = ctx.shape
        self.n, self.nc = ny * nx, ctx.n_classes
        n, nc = self.n, self.nc
        z = np.zeros
        self.requested, self.raindrop, self.flow = z(n * nc), z(n * nc), z(n * nc)
        self.svel, self.rate, self.prob = z(n * nc), z(n * nc), z(n * nc)
        self.law, self.settle, self.cap_applied = (np.zeros(n * nc, dtype=np.bool_) for _ in range(3))
        self.regime = np.zeros(n * nc, dtype=np.int8)
        self.ustar, self.reynolds, self.power, self.ke, self.ke_flux, self.d50 = (z(n) for _ in range(6))
        self.counts = np.zeros(len(REGIME_CODES), dtype=np.int64)
        self._dt = None
        self._scalars = np.zeros(_N_SC)

    def nbytes(self) -> int:
        return int(sum(getattr(self, name).nbytes for name in (
            "requested", "raindrop", "flow", "svel", "rate", "prob", "law", "settle", "cap_applied", "regime", "ustar",
            "reynolds", "power", "ke", "ke_flux", "d50")))

    def _set_scalars(self, dt: float) -> np.ndarray:
        if self._dt != dt:
            c = self.ctx
            s = self._scalars
            s[:] = 0.0
            s[_S_RHO] = c.particle_density_kg_m3
            s[_S_REF] = c.reference_interval_s
            s[_S_MASS] = c.cell_area_m2 * c.particle_density_kg_m3 * dt
            s[_S_NU] = c.kinematic_viscosity_m2_s
            s[_S_HZ] = c.flow_detachment_depth_scale_m
            s[_S_BAG_SCALE] = c.bagnold_density_scale
            s[_S_DECAY] = recession_velocity(1.0, dt, factor_per_reference_s=c.recession_factor_per_reference_s,
                                             reference_interval_s=c.reference_interval_s)
            s[_S_P_PAR] = c.p_par
            self._dt = dt
        return self._scalars

    @property
    def decay(self) -> float:
        return float(self._scalars[_S_DECAY])

    def run(self, depth_flat: np.ndarray, velocity_flat: np.ndarray, rain_flat: np.ndarray, prev_flat: np.ndarray,
            dt: float) -> None:
        for name, a, size in (("depth", depth_flat, self.n), ("velocity", velocity_flat, self.n),
                              ("rain", rain_flat, self.n), ("previous sediment velocity", prev_flat, self.n * self.nc)):
            if not (type(a) is np.ndarray and a.dtype == np.float64 and a.ndim == 1 and a.size == size
                    and a.flags["C_CONTIGUOUS"]):
                raise LegacyNativeError(f"{name} must be a flat C-contiguous float64 host array of size {size}")
        if not (isinstance(dt, float) and dt > 0.0 and np.isfinite(dt)):
            raise LegacyNativeError("dt must be a positive finite float")
        sc = self._set_scalars(dt)
        self.counts[:] = 0
        c = self.ctx
        flags = int(compiled_kernel()(
            depth_flat, velocity_flat, rain_flat, prev_flat, c.active, c.slope, c.veg_factor, c.d50, c.bagnold_prefactor,
            c.slope_power, c.fractions, c.cap_rate, c.class_constants, sc, c.config, self.requested, self.raindrop,
            self.flow, self.svel, self.rate, self.law, self.settle, self.regime, self.ustar, self.reynolds, self.power,
            self.ke, self.ke_flux, self.prob, self.cap_applied, self.d50, self.counts))
        if flags:
            raise SedimentPhysicsError("; ".join(m for b, m in enumerate(_FLAG_MESSAGES) if flags >> b & 1))


@dataclass
class StepResult:
    """Views into engine buffers, valid until the next `step` call (copy what you keep)."""

    ledger_row: np.ndarray  # (n_columns, nc) per-step totals, `LEDGER_COLUMNS`
    walk_counts: np.ndarray  # (N_COUNTS,)
    regime_counts: np.ndarray  # (7,)


LEDGER_COLUMNS = ("pickup_kg", "deposition_active_kg", "deposition_pit_kg", "deposition_ring_kg",
                  "deposition_inactive_kg", "effective_clip_source_kg", "old_mobile_kg", "new_mobile_kg",
                  "mobile_terminal_kg", "cn_export_kg", "endpoint_export_kg", "outlet_flux_kg_s", "erased_deposition_kg")
#: `kg` columns are per-step masses (summable over steps); `kg/s` is an instantaneous rate (never summed into a total);
#: `kg state` columns are inventories at one instant (never summed over steps).
LEDGER_UNITS = {"pickup_kg": "kg per step", "deposition_active_kg": "kg per step", "deposition_pit_kg": "kg per step",
                "deposition_ring_kg": "kg per step", "deposition_inactive_kg": "kg per step",
                "effective_clip_source_kg": "kg per step", "old_mobile_kg": "kg state (start of step)",
                "new_mobile_kg": "kg state (end of step)", "mobile_terminal_kg": "kg state (end of step)",
                "cn_export_kg": "kg per step", "endpoint_export_kg": "kg per step", "outlet_flux_kg_s": "kg/s (end of step)",
                "erased_deposition_kg": "kg per step"}


class StepEngine:
    """Preallocated legacy step. `source_order` is 'index' (the accepted replay's order, default) or 'legacy' (north row
    first); `erase_on` needs the legacy order. `erase_dry_rain` marks dry raining cells as erase cells too (the no-splash
    benchmark patch zeroes `depos_soil` there)."""

    def __init__(self, network: N.NativeNetwork, runner: WetLawRunner, *, dt: float = 1.0, compiled: bool = True,
                 source_order: str = "index", erase_on: bool = False, erase_dry_rain: bool = True):
        if source_order not in ("index", "legacy"):
            raise LegacyNativeError("source_order must be 'index' or 'legacy'")
        if erase_on and source_order != "legacy":
            raise LegacyNativeError("the order-dependent legacy erasure needs source_order='legacy'")
        if tuple(runner.ctx.shape) != tuple(network.shape) or network.active.size != runner.n:
            raise LegacyNativeError("the wet-law context and the network have different shapes")
        if not np.array_equal(np.asarray(runner.ctx.active), network.active):
            raise LegacyNativeError("the wet-law context and the network have different active masks")
        if not (isinstance(dt, float) and dt > 0.0 and np.isfinite(dt)):
            raise LegacyNativeError("dt must be a positive finite float")
        self.net, self.runner, self.dt = network, runner, dt
        self.k = get_kernels(compiled)
        self.n, self.nc = runner.n, runner.nc
        n, nc = self.n, self.nc
        self.src = network.source_order_legacy if source_order == "legacy" else network.source_order_index
        self.erase_on, self.erase_dry_rain = bool(erase_on), bool(erase_dry_rain)
        self.limits = N.walk_limits(network.dx_m)
        z = np.zeros
        self.det, self.v_used, self.v_prev = z((n, nc)), z((n, nc)), z((n, nc))
        self.depos, self.clip = z((n, nc)), z((n, nc))
        self.M1, self.Q1, self.Qin1 = z((n, nc)), z((n, nc)), z((n, nc))
        self.M2, self.Q2, self.Qin2 = z((n, nc)), z((n, nc)), z((n, nc))
        self.cum_det, self.cum_dep, self.cum_clip = z((n, nc)), z((n, nc)), z((n, nc))
        self.erase = np.zeros(n, dtype=np.bool_)
        self.ring, self.inactive_dep, self.erased = z(nc), z(nc), z(nc)
        self.cn_export, self.endpoint_export = z(nc), z(nc)
        self.reduced = z((len(N.REDUCE_ROWS), nc))
        self.probe = z((5, nc))
        self.ledger_row = z((len(LEDGER_COLUMNS), nc))
        self.counts = np.zeros(N.N_COUNTS, dtype=np.int64)
        self.total_counts = np.zeros(N.N_COUNTS, dtype=np.int64)
        self.total_regime_counts = np.zeros(len(REGIME_CODES), dtype=np.int64)
        self._law2d, self._regime2d = runner.law.reshape(n, nc), runner.regime.reshape(n, nc)
        self.timers = {"physics_s": 0.0, "glue_walk_cn_s": 0.0}
        self.poisoned: str | None = None

    def nbytes(self) -> int:
        names = ("det", "v_used", "v_prev", "depos", "clip", "M1", "Q1", "Qin1", "M2", "Q2", "Qin2", "cum_det", "cum_dep",
                 "cum_clip")
        return int(sum(getattr(self, a).nbytes for a in names) + self.runner.nbytes())

    def reset(self) -> None:
        """Zero every persistent and scratch buffer and clear the poison flag (a fresh engine state)."""
        for name in ("det", "v_used", "v_prev", "depos", "clip", "M1", "Q1", "Qin1", "M2", "Q2", "Qin2", "cum_det",
                     "cum_dep", "cum_clip", "ring", "inactive_dep", "erased", "cn_export", "endpoint_export", "reduced",
                     "probe", "ledger_row", "counts", "total_counts", "total_regime_counts"):
            getattr(self, name).fill(0)
        self.erase.fill(False)
        self.poisoned = None

    def step(self, depth: np.ndarray, velocity: np.ndarray, rain: np.ndarray) -> StepResult:
        """One legacy step with law depth `depth`, the NEW velocity and this step's rain rate (m/s); see the module
        docstring for the failure contract."""
        if self.poisoned is not None:
            raise LegacyNativeError(f"the engine was poisoned by an earlier failure ({self.poisoned}); reset() or build a new one")
        dt, n, nc, k = self.dt, self.n, self.nc, self.k
        grid = tuple(self.net.shape)
        d_flat = _host_f64("depth", depth, grid).reshape(-1)
        v_flat = _host_f64("velocity", velocity, grid).reshape(-1)
        rain_flat = _host_f64("rain", rain, grid).reshape(-1)
        d_flat, v_flat, rain_flat = (np.ascontiguousarray(a) for a in (d_flat, v_flat, rain_flat))
        r = self.runner
        t0 = time.perf_counter()
        r.run(d_flat, v_flat, rain_flat, self.v_prev.reshape(-1), dt)
        t1 = time.perf_counter()
        flags = int(k.pack(self.net.active, self.net.terminal, rain_flat, r.requested.reshape(n, nc), self._law2d,
                           r.svel.reshape(n, nc), r.rate.reshape(n, nc), self._regime2d, self.v_prev, r.decay, dt,
                           self.erase_dry_rain, self.det, self.v_used, self.erase))
        if flags:
            raise SedimentPhysicsError("; ".join(m for b, m in enumerate(_PACK_MESSAGES) if flags >> b & 1))
        net = self.net
        try:  # from here the scratch is dirty: any failure poisons the engine
            self.counts[:] = 0
            self.ring[:] = 0.0
            self.inactive_dep[:] = 0.0
            self.erased[:] = 0.0
            self.cn_export[:] = 0.0
            self.endpoint_export[:] = 0.0
            k.walk(self.src, self.det, r.rate.reshape(n, nc), self._law2d, self._regime2d, self.limits, net.slope_zero,
                   net.walk_first, net.walk_next, net.aspect0, net.inactive, self.erase, self.erase_on, net.dx_m, dt,
                   self.depos, self.ring, self.inactive_dep, self.erased, self.counts)
            k.cn(net.order, net.donors, net.cn_receiver, net.outlet, self.M1, self.Q1, self.Qin1, self.det, self.depos,
                 self.v_used, dt, net.dx_m, self.M2, self.Q2, self.Qin2, self.clip, self.cn_export, self.endpoint_export)
            bad = int(k.check(net.active_idx, net.outlet_idx, dt, self.det, self.depos, self.clip, self.M1, self.M2, self.Q2,
                              self.Qin2, self.cum_det, self.cum_dep, self.cum_clip, self.probe))
            if bad:
                raise SedimentPhysicsError("; ".join(m for b, m in enumerate(N.CHECK_MESSAGES) if bad >> b & 1))
            for name in ("cn_export", "endpoint_export", "ring", "inactive_dep", "erased"):
                if not np.isfinite(getattr(self, name)).all():
                    raise SedimentPhysicsError(f"the per-class {name} tally is not finite")
            k.reduce(net.active_idx, net.terminal, net.outlet_idx, dt, self.det, self.depos, self.clip, self.M1, self.M2,
                     self.Q2, self.cum_det, self.cum_dep, self.cum_clip, self.reduced)
        except BaseException as exc:
            self.poisoned = f"{type(exc).__name__}: {exc}"
            raise
        red = self.reduced
        row = self.ledger_row
        row[0], row[1], row[2] = red[0] * dt, red[1] * dt, red[6] * dt
        row[3], row[4], row[5] = self.ring * dt, self.inactive_dep * dt, red[2]
        row[6], row[7], row[8] = red[3], red[4], red[5]
        row[9], row[10], row[11], row[12] = self.cn_export, self.endpoint_export, red[7], self.erased * dt
        self.M1, self.M2 = self.M2, self.M1
        self.Q1, self.Q2 = self.Q2, self.Q1
        self.Qin1, self.Qin2 = self.Qin2, self.Qin1
        self.v_prev, self.v_used = self.v_used, self.v_prev
        self.total_counts += self.counts
        self.total_regime_counts += r.counts
        t2 = time.perf_counter()
        self.timers["physics_s"] += t1 - t0
        self.timers["glue_walk_cn_s"] += t2 - t1
        return StepResult(row, self.counts, r.counts)

    @property
    def mobile(self) -> np.ndarray:
        """Current pool (level 1 after the swap), `(n, nc)` kg. A view; copy to keep."""
        return self.M1
