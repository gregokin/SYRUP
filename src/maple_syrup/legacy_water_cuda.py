"""Per-step water accounting of the CUDA legacy replay (task gpu_sediment, stage B3): the three water series and the four cumulative maps.

Benchmark-only, exactly like `legacy_gpu_driver`. After each hydrology step the driver records, DEVICE-ONLY,

  * the series row `[outlet_discharge_m3_s, export_m3, budget_residual_m3]` (scalars that are 0-d views of the hydrology step's packet), and
  * `cum_X += X_step` for X in rain, intake, saturation_return, drainage (the step's column arrays).

Two selectable implementations of the SAME arithmetic (`--water-accounting`):

  `separate`  the B1/B2 reference: three device-to-device scalar copies into the series row and four `cupy.add(cum, x, out=cum)` (7 device
              operations per step, one tiny kernel each);
  `fused`     ONE launch of `sg_water_account` per step: global thread 0 copies the three packet doubles (words `_P_OUTLET`, `_P_EXPORT`,
              `_P_RESIDUAL` of the 144-byte hydrology packet, the same memory the scalar views alias) into the series row, and every thread adds
              one cell of the four sources into the four maps, `cum[i] = cum[i] + x[i]`, a single IEEE double addition per cell and map exactly
              like `cupy.add` (`--fmad=false`, no fast math, no floating point atomics, no reduction).

No host read, no host scalar, no host-to-device copy and no all-domain allocation happens per step in either mode; the driver's single
counted 144-byte packet read is unchanged. The packet itself is not modified. These launches are driver water work: `CudaLegacyContext.stats`
counts SEDIMENT launches only and is not changed by this module (`WaterAccounting.stats` counts these).

Guards run BEFORE any launch or write: creation device = current device, sealed scalars and array metadata (pointer, shape, dtype, strides)
unchanged, strict integer consecutive rows, a packet of the right type/size/device whose words alias the step's scalar views, float64
C-contiguous grids of the declared shape on the current device, and no byte-span overlap between any source and any destination. A failure
while writing (not a validation failure) poisons the accountant until `reset()`.

Nothing here was run by its author (file-only tools); Codex records results.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np

from maple_syrup import hydrology_cuda as hc
from maple_syrup.routing_cuda import (
    COMPILE_OPTIONS,
    CudaUnavailableError,
    _cupy,
    _current_device_id,
)

__all__ = ["MODES", "WaterAccounting", "WaterAccountingError", "kernel_provenance", "kernel_source"]

MODES = ("separate", "fused")
CUM_KEYS = ("rain", "intake", "saturation_return", "drainage")
THREADS = 128
INT32_MAX = 2**31 - 1
OPS_PER_STEP = {"separate": 7, "fused": 1}  # device operations (separate: 3 scalar copies + 4 adds)

_TEMPLATE = r"""
#define W_OUTLET __W_OUTLET__
#define W_EXPORT __W_EXPORT__
#define W_RESIDUAL __W_RESIDUAL__
extern "C" __global__ void sg_water_account(
    const long long n, const double* packet, double* wrow, const double* rain, const double* intake, const double* satret,
    const double* drain, double* c_rain, double* c_intake, double* c_sat, double* c_drain)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {  /* the series row: outlet discharge, export, budget residual (the packet words the step's scalar views alias) */
        wrow[0] = packet[W_OUTLET];
        wrow[1] = packet[W_EXPORT];
        wrow[2] = packet[W_RESIDUAL];
    }
    if (i >= n) return;
    c_rain[i] = c_rain[i] + rain[i];
    c_intake[i] = c_intake[i] + intake[i];
    c_sat[i] = c_sat[i] + satret[i];
    c_drain[i] = c_drain[i] + drain[i];
}
"""
_FUNCTIONS: dict[int, Any] = {}


class WaterAccountingError(RuntimeError):
    """Invalid use or device failure of the water accountant (no silent fallback)."""


def kernel_source() -> str:
    return (_TEMPLATE.replace("__W_OUTLET__", str(hc._P_OUTLET)).replace("__W_EXPORT__", str(hc._P_EXPORT))
            .replace("__W_RESIDUAL__", str(hc._P_RESIDUAL)))


def kernel_provenance() -> dict[str, Any]:
    return {"module": "maple_syrup.legacy_water_cuda", "source_sha256": hashlib.sha256(kernel_source().encode()).hexdigest(),
            "compile_options": list(COMPILE_OPTIONS), "fastmath": False, "kernels": ["sg_water_account"], "threads": THREADS,
            "packet_words": {"outlet": hc._P_OUTLET, "export": hc._P_EXPORT, "residual": hc._P_RESIDUAL, "total": hc.PACKET_WORDS},
            "arithmetic": "one IEEE double addition per cell and map (cum = cum + x), identical to cupy.add; no atomics, no reduction"}


def _function(cp):
    """The compiled kernel of the current device and whether it came from the process cache (`(fn, cached, compile_seconds)`)."""
    device = _current_device_id(cp)
    t0 = time.perf_counter()
    cached = device in _FUNCTIONS
    if not cached:
        try:
            fn = cp.RawKernel(kernel_source(), "sg_water_account", options=COMPILE_OPTIONS, backend="nvrtc")
            limit = int(dict(fn.attributes).get("max_threads_per_block", 0))
        except Exception as exc:
            raise CudaUnavailableError(f"water accounting kernel failed to compile/load ({type(exc).__name__}: {exc}); no fallback") from exc
        if limit < THREADS:
            raise CudaUnavailableError(f"sg_water_account supports only {limit} threads per block, {THREADS} needed; no fallback")
        _FUNCTIONS[device] = fn
    return _FUNCTIONS[device], cached, time.perf_counter() - t0


class WaterAccounting:
    """Device-resident series `(steps, 3)` and four cumulative `(ny, nx)` maps with the selected implementation. `account(row, step, packet)`
    enqueues the step's accounting on the current stream and returns nothing."""

    def __init__(self, cp, shape: tuple[int, int], steps: int, mode: str):
        if mode not in MODES:
            raise WaterAccountingError(f"water accounting mode must be one of {MODES}, got {mode!r}")
        if (not isinstance(shape, tuple) or len(shape) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1 for v in shape)
                or int(shape[0]) * int(shape[1]) > INT32_MAX):
            raise WaterAccountingError(f"shape must be a (ny, nx) tuple of positive ints with fewer than 2**31 cells, got {shape!r}")
        if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or not 1 <= steps <= INT32_MAX // 8:
            raise WaterAccountingError("steps must be a positive int")
        t_setup = time.perf_counter()
        self.cp = cp
        self._device_id = int(cp.cuda.Device().id)
        self.mode, self.shape, self.steps = mode, (int(shape[0]), int(shape[1])), int(steps)
        self.n = self.shape[0] * self.shape[1]
        # compiled/loaded here: never inside the timed loop. `compile_cached` True = taken from the process cache (a first use in the
        # process is the uncached, NVRTC-compiling case); `compile_s` is the time spent obtaining the kernel either way.
        self.fn, self.compile_cached, self.compile_s = _function(cp) if mode == "fused" else (None, None, 0.0)
        self.water = cp.zeros((self.steps, 3), dtype=np.float64)
        self.cum = {k: cp.zeros(self.shape, dtype=np.float64) for k in CUM_KEYS}
        self.row = 0
        self.poisoned: str | None = None
        self.stats = {"launches": 0, "steps": 0}
        self._seal()
        self.setup_s = time.perf_counter() - t_setup  # constructor wall time (compile/load + allocation), outside any step loop

    # -- sealed structure --
    def _arrays(self) -> dict[str, Any]:
        return {"water": self.water, **{f"cum_{k}": v for k, v in self.cum.items()}}

    @staticmethod
    def _meta(a: Any):
        return (int(a.data.ptr), tuple(a.shape), a.dtype.str, tuple(a.strides), int(a.device.id))

    def _scalars(self) -> tuple:
        return (self._device_id, self.mode, self.shape, self.steps, self.n, tuple(self.cum))

    def _seal(self) -> None:
        self._sealed_scalars = self._scalars()
        self._sealed_arrays = {k: self._meta(a) for k, a in self._arrays().items()}

    def _guard(self, op: str) -> None:
        cp = self.cp
        if int(cp.cuda.Device().id) != self._device_id:
            raise WaterAccountingError(f"{op}: the current CUDA device differs from the creation device {self._device_id}")
        try:
            if self._scalars() != self._sealed_scalars:
                raise WaterAccountingError(f"{op}: a sealed scalar changed")
            for name, a in self._arrays().items():
                if self._meta(a) != self._sealed_arrays[name]:
                    raise WaterAccountingError(f"{op}: array {name} was replaced or its metadata changed")
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise WaterAccountingError(f"{op}: the sealed structure is damaged ({type(exc).__name__}: {exc})") from exc

    def _grid(self, a: Any, name: str) -> None:
        cp = self.cp
        if type(a) is not cp.ndarray or a.dtype != np.float64 or tuple(a.shape) != self.shape or not a.flags.c_contiguous:
            raise WaterAccountingError(f"{name} must be a C-contiguous float64 cupy array of shape {self.shape}")
        if int(a.device.id) != self._device_id:
            raise WaterAccountingError(f"{name} lives on another CUDA device")

    @staticmethod
    def _span(a: Any) -> tuple[int, int]:
        p = int(a.data.ptr)
        return p, p + int(a.nbytes)

    @staticmethod
    def _fields(obj: Any, names: tuple[str, ...], what: str) -> tuple[Any, ...]:
        """Required attributes gathered safely: a missing one is a typed refusal, never a bare AttributeError."""
        missing = [n for n in names if not hasattr(obj, n)]
        if missing:
            raise WaterAccountingError(f"{what} lacks the required field(s) {missing}")
        return tuple(getattr(obj, n) for n in names)

    def _scalar(self, a: Any, word: int, base: int, name: str) -> None:
        """A step scalar must be EXACTLY a 0-d float64 cupy view of the creation device at the packet word it is accounted from
        (a uint64/float32 view or a vector view sharing the address, a host scalar or an object that merely carries `.data.ptr` is
        refused: the fused kernel would read the word as a double while the separate path would copy a wrongly cast value)."""
        cp = self.cp
        if type(a) is not cp.ndarray or a.dtype != np.float64 or a.shape != () or not a.flags.c_contiguous:
            raise WaterAccountingError(f"the step's {name} scalar must be a 0-d float64 cupy array, got "
                                       f"{type(a).__name__} {getattr(a, 'dtype', None)} {getattr(a, 'shape', None)}")
        if int(a.device.id) != self._device_id:
            raise WaterAccountingError(f"the step's {name} scalar lives on another CUDA device")
        if int(a.data.ptr) != base + 8 * word:
            raise WaterAccountingError(f"the step's {name} scalar is not the packet word it is accounted from (a mismatched packet)")

    def _validate(self, row: Any, step: Any, packet: Any) -> tuple[Any, Any, Any, Any, Any]:
        """Everything is checked here, before the first write; returns `(rain, intake, saturation_return, drainage, packet)`."""
        if isinstance(row, (bool, np.bool_)) or not isinstance(row, (int, np.integer)):
            raise WaterAccountingError(f"row must be an integer, got {type(row).__name__}")
        if int(row) != self.row or not 0 <= int(row) < self.steps:
            raise WaterAccountingError(f"rows must be consecutive from 0; got {row}, expected {self.row}")
        cp = self.cp
        col, route = self._fields(step, ("column", "route"), "step")
        sources = self._fields(col, ("rain_m", "intake_m", "saturation_return_m", "drainage_m"), "step.column")
        scalars = self._fields(route, ("outlet_discharge_m3_s", "export_m3", "budget_residual_m3"), "step.route")
        for a, name in zip(sources, ("rain_m", "intake_m", "saturation_return_m", "drainage_m"), strict=True):
            self._grid(a, name)
        if type(packet) is not cp.ndarray or packet.dtype != np.uint64 or packet.shape != (hc.PACKET_WORDS,) or not packet.flags.c_contiguous \
                or int(packet.device.id) != self._device_id:
            raise WaterAccountingError(f"packet must be a contiguous uint64[{hc.PACKET_WORDS}] cupy array on the current device")
        base = int(packet.data.ptr)
        for scalar, word, name in zip(scalars, (hc._P_OUTLET, hc._P_EXPORT, hc._P_RESIDUAL), ("outlet", "export", "residual"), strict=True):
            self._scalar(scalar, word, base, name)
        spans = [self._span(a) for a in (self.water, *self.cum.values())]
        for src in (*sources, packet):
            s0, s1 = self._span(src)
            if any(s0 < d1 and d0 < s1 for d0, d1 in spans):
                raise WaterAccountingError("a hydrology source overlaps an accounting destination")
        return sources + (packet,)

    def account(self, row: int, step: Any, packet: Any) -> None:
        """Record the series row and update the four cumulative maps for hydrology step `row`."""
        self._guard("account")
        if self.poisoned is not None:
            raise WaterAccountingError(f"the accountant is poisoned ({self.poisoned}); reset() first")
        rain, intake, satret, drain, packet = self._validate(row, step, packet)  # everything is checked BEFORE the first write
        cp, r = self.cp, int(row)
        try:
            if self.mode == "fused":
                self.fn((-(-self.n // THREADS),), (THREADS,), (
                    np.int64(self.n), packet, self.water[r], rain, intake, satret, drain, self.cum["rain"], self.cum["intake"],
                    self.cum["saturation_return"], self.cum["drainage"]))
                self.stats["launches"] += 1
            else:
                route = step.route
                self.water[r, 0] = route.outlet_discharge_m3_s
                self.water[r, 1] = route.export_m3
                self.water[r, 2] = route.budget_residual_m3
                cp.add(self.cum["rain"], rain, out=self.cum["rain"])
                cp.add(self.cum["intake"], intake, out=self.cum["intake"])
                cp.add(self.cum["saturation_return"], satret, out=self.cum["saturation_return"])
                cp.add(self.cum["drainage"], drain, out=self.cum["drainage"])
                self.stats["launches"] += OPS_PER_STEP["separate"]
        except BaseException as exc:  # a partial write is possible only here
            self.poisoned = f"{type(exc).__name__}: {exc}"
            raise
        self.row += 1
        self.stats["steps"] += 1

    def reset(self) -> None:
        """Zero the series and the maps and restart the row counter; the warm-up counters are kept as `stats_before_reset`."""
        self._guard("reset")
        self.stats_before_reset = dict(self.stats)
        self.stats = {"launches": 0, "steps": 0}
        self.water.fill(0.0)
        for a in self.cum.values():
            a.fill(0.0)
        self.row = 0
        self.poisoned = None

    def nbytes(self) -> int:
        return int(sum(a.nbytes for a in self._arrays().values()))

    def summary(self) -> dict[str, Any]:
        return {"mode": self.mode, "device_operations_per_step": OPS_PER_STEP[self.mode], "launches": self.stats["launches"],
                "steps": self.stats["steps"], "bytes": self.nbytes(), "kernel": kernel_provenance() if self.mode == "fused" else None,
                "setup_s": self.setup_s, "compile_s": self.compile_s, "compile_cached": self.compile_cached,
                "setup_note": "constructor wall time (kernel compile/load for fused + allocation), outside every step loop and excluded from "
                              "loop_wall_s_excluding_progress; compile_cached False = first use in the process (NVRTC compile), True = process "
                              "cache hit (a warm-up loop's accountant makes the measured loop's a cache hit); separate has no kernel",
                "host_reads_per_step": 0, "host_to_device_copies_per_step": 0,
                "scope": "driver water accounting only (the series row and four cumulative maps); `gpu.transfers.launches` and "
                         "`launches_per_step_sediment` count SEDIMENT launches and exclude these"}


def make(shape: tuple[int, int], steps: int, mode: str) -> WaterAccounting:
    """Convenience constructor on the current CuPy device (explicit CudaUnavailableError without CuPy)."""
    return WaterAccounting(_cupy(), shape, steps, mode)
