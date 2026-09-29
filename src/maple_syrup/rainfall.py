"""MAHLERAN-compatible rainfall forcing (Phase 3a).

Reference: MAHLERAN 1.2.3 `Subroutines_In_out/Set_rain_xml.f90`
(`rain_type` 1 constant, 2 read from file) and `time_conv`
(`Subroutines_In_out/Functions.for` 190-204). See docs/phase3/rainfall.md.

Two parts, deliberately separate:

* `RainfallSchedule` -- an immutable, validated HOST description of
  piecewise-constant rainfall intensity in time. Evaluation is stateless:
  `depth_m(t, dt)` is the exact integral over `[t, t + dt]`, and
  `pieces(t, dt)` / `next_edge_s(t)` expose every rate discontinuity so a
  nonlinear consumer (infiltration) never sees a rate averaged across a
  jump. There is no clock or cursor to persist or restore.
* `RainfallField` -- a validated, backend-resident `(ny, nx)` multiplier
  (nonnegative scale times an explicit Boolean active mask). Applying a
  host scalar from the schedule is one elementwise multiply in the field's
  own array namespace; no field ever crosses host/device per step.

Units: time in seconds from the event origin, intensity input in mm/h,
rates in m/s, depths in m. Legacy files are parsed once, before the model
loop; wall clocks never enter the evaluation path.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

__all__ = [
    "MM_PER_H_TO_M_PER_S",
    "RAINFALL_TIMING_CONVENTION",
    "RainfallError",
    "RainfallField",
    "RainfallPiece",
    "RainfallProvenance",
    "RainfallSchedule",
    "constant_rainfall",
    "parse_legacy_rainfall_file",
    "parse_legacy_rainfall_text",
    "rainfall_field",
]

MM_PER_H_TO_M_PER_S = 1.0 / 3.6e6
SECONDS_PER_DAY = 86400
# A decreasing clock is read as one midnight crossing only when that is the
# shorter reading of the clock difference (unwrapped gap < 12 h).
_MAX_ROLLOVER_GAP_S = SECONDS_PER_DAY // 2

RAINFALL_TIMING_CONVENTION = (
    "interval-ending: header clock is t=0; record k (clock T_k, intensity I_k mm/h) "
    "applies on (T_{k-1}, T_k]; zero before the first edge and after the final record; "
    "exact interval integration (legacy iter*dt > T_k switching lag not reproduced)"
)


class RainfallError(ValueError):
    """Invalid rainfall input or evaluation request. Raised before any
    result is produced; nothing is partially constructed."""


# --- schedule -------------------------------------------------------------------
@dataclass(frozen=True)
class RainfallProvenance:
    """Where a schedule came from. `sha256` is of the exact file bytes."""

    kind: str  # "legacy_file" or "constant"
    convention: str = RAINFALL_TIMING_CONVENTION
    path: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    start_clock: str | None = None  # header text, e.g. "18:00:00.00"
    start_clock_s: float | None = None  # seconds after local midnight
    rollover_record: int | None = None  # 1-based record index after midnight, if any
    n_records: int | None = None
    blank_lines_skipped: int = 0


@dataclass(frozen=True)
class RainfallPiece:
    """A sub-interval `[start_s, end_s]` over which the rate is constant."""

    start_s: float
    end_s: float
    rate_m_per_s: float

    @property
    def depth_m(self) -> float:
        return self.rate_m_per_s * (self.end_s - self.start_s)


def _check_time(t_s: Any, dt_s: Any) -> tuple[float, float]:
    for name, value in (("t_s", t_s), ("dt_s", dt_s)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise RainfallError(f"{name} must be a real number, got {type(value).__name__}")
    t0, dt = float(t_s), float(dt_s)
    if not (math.isfinite(t0) and math.isfinite(dt)):
        raise RainfallError(f"t_s and dt_s must be finite, got {t0!r}, {dt!r}")
    if t0 < 0.0:
        raise RainfallError(f"t_s must be >= 0 (seconds from the event origin), got {t0!r}")
    if dt < 0.0:
        raise RainfallError(f"dt_s must be >= 0, got {dt!r}")
    t1 = t0 + dt
    if not math.isfinite(t1):
        raise RainfallError("t_s + dt_s overflows")
    return t0, t1


@dataclass(frozen=True, eq=False)
class RainfallSchedule:
    """Piecewise-constant rainfall: `rate_m_per_s[k]` on
    `[edges_s[k], edges_s[k+1]]`, zero outside `[edges_s[0], edges_s[-1]]`.

    Construct through `parse_legacy_rainfall_file`, `parse_legacy_rainfall_text`
    or `constant_rainfall`, or directly from edges and mm/h intensities. Arrays
    are private read-only float64 copies.
    """

    edges_s: np.ndarray  # (n + 1,) strictly increasing, finite, edges_s[0] >= 0
    intensity_mm_per_h: np.ndarray  # (n,) finite, >= 0, as given
    provenance: RainfallProvenance
    rate_m_per_s: np.ndarray = field(init=False)  # (n,) derived from intensity_mm_per_h

    def __post_init__(self) -> None:
        try:
            edges = np.array(self.edges_s, dtype=np.float64)
            intensity = np.array(self.intensity_mm_per_h, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise RainfallError(f"edges/intensities must be numeric: {exc}") from None
        if edges.ndim != 1 or intensity.ndim != 1 or edges.size != intensity.size + 1 or intensity.size < 1:
            raise RainfallError(
                f"need 1-D edges of length n+1 and intensities of length n >= 1, "
                f"got shapes {edges.shape} and {intensity.shape}"
            )
        if not (np.all(np.isfinite(edges)) and np.all(np.isfinite(intensity))):
            raise RainfallError("edges and intensities must be finite")
        if edges[0] < 0.0:
            raise RainfallError(f"first edge must be >= 0 s from the origin, got {edges[0]!r}")
        if np.any(np.diff(edges) <= 0.0):
            raise RainfallError("edges must be strictly increasing (no duplicate or reversed times)")
        if np.any(intensity < 0.0):
            raise RainfallError("intensities must be >= 0 mm/h")
        if not isinstance(self.provenance, RainfallProvenance):
            raise RainfallError("provenance must be a RainfallProvenance")
        rate = intensity * MM_PER_H_TO_M_PER_S
        for array in (edges, intensity, rate):
            array.flags.writeable = False
        object.__setattr__(self, "edges_s", edges)
        object.__setattr__(self, "intensity_mm_per_h", intensity)
        object.__setattr__(self, "rate_m_per_s", rate)

    @property
    def start_s(self) -> float:
        return float(self.edges_s[0])

    @property
    def end_s(self) -> float:
        return float(self.edges_s[-1])

    def total_depth_m(self) -> float:
        return math.fsum((self.rate_m_per_s * np.diff(self.edges_s)).tolist())

    def rate_after_m_per_s(self, t_s: float) -> float:
        """Rate on the interval that starts at `t_s` (right-continuous), i.e.
        the rate a forward step from `t_s` sees first. Zero outside."""
        t0, _ = _check_time(t_s, 0.0)
        k = int(np.searchsorted(self.edges_s, t0, side="right")) - 1
        return float(self.rate_m_per_s[k]) if 0 <= k < self.rate_m_per_s.size else 0.0

    def next_edge_s(self, t_s: float) -> float:
        """The first rate discontinuity strictly after `t_s`; `math.inf` once
        `t_s >= end_s` (the rate is zero forever after)."""
        t0, _ = _check_time(t_s, 0.0)
        k = int(np.searchsorted(self.edges_s, t0, side="right"))
        return float(self.edges_s[k]) if k < self.edges_s.size else math.inf

    def pieces(self, t_s: float, dt_s: float) -> tuple[RainfallPiece, ...]:
        """`[t, t + dt]` split at every interior edge into constant-rate
        pieces, contiguous and in order (zero-rate pieces included). Empty
        for `dt = 0`."""
        t0, t1 = _check_time(t_s, dt_s)
        if t1 <= t0:
            return ()
        edges = self.edges_s
        lo = int(np.searchsorted(edges, t0, side="right"))
        hi = int(np.searchsorted(edges, t1, side="left"))
        bounds = [t0, *edges[lo:hi].tolist(), t1]
        n = self.rate_m_per_s.size
        result = []
        for j in range(len(bounds) - 1):
            k = lo - 1 + j  # segment index of [bounds[j], bounds[j+1]]
            rate = float(self.rate_m_per_s[k]) if 0 <= k < n else 0.0
            result.append(RainfallPiece(bounds[j], bounds[j + 1], rate))
        return tuple(result)

    def depth_m(self, t_s: float, dt_s: float) -> float:
        """Exact rainfall depth (m) over `[t, t + dt]`."""
        return math.fsum(piece.depth_m for piece in self.pieces(t_s, dt_s))


def constant_rainfall(start_s: float, end_s: float, intensity_mm_per_h: float) -> RainfallSchedule:
    """Constant intensity on `[start_s, end_s]` (legacy `rain_type = 1`,
    `rval = rf_mean / 3600` while rain is on), zero elsewhere."""
    for name, value in (("start_s", start_s), ("end_s", end_s), ("intensity_mm_per_h", intensity_mm_per_h)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise RainfallError(f"{name} must be a real number, got {type(value).__name__}")
    return RainfallSchedule(
        edges_s=[float(start_s), float(end_s)],
        intensity_mm_per_h=[float(intensity_mm_per_h)],
        provenance=RainfallProvenance(kind="constant"),
    )


# --- legacy file parser ---------------------------------------------------------
_CLOCK = r"(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)"
_HEADER_RE = re.compile(rf"{_CLOCK}")
_RECORD_RE = re.compile(rf"{_CLOCK}\s+(\S+)")
_NUMBER_RE = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


def _clock_seconds(match: re.Match, line_no: int) -> Decimal:
    hours, minutes, seconds = int(match.group(1)), int(match.group(2)), Decimal(match.group(3))
    if hours >= 24 or minutes >= 60 or seconds >= 60:
        raise RainfallError(f"line {line_no}: clock {_clock_text(match)!r} out of range (hh<24, mm<60, ss<60)")
    return Decimal(hours * 3600 + minutes * 60) + seconds


def _clock_text(match: re.Match) -> str:
    return f"{match.group(1)}:{match.group(2)}:{match.group(3)}"


def parse_legacy_rainfall_text(
    text: str, *, path: str | None = None, sha256: str | None = None, size_bytes: int | None = None
) -> RainfallSchedule:
    """Parse the MAHLERAN `rain_type = 2` format: one header line holding the
    start clock `hh:mm:ss[.f]`, then records `hh:mm:ss[.f] <intensity mm/h>`
    where each clock is the END of its interval.

    Tolerated deliberately: leading/trailing blanks, any run of blanks/tabs
    between the two fields, CR/LF line ends, blank lines. Rejected: anything
    else on a line, out-of-range clocks, negative/non-finite/non-numeric
    intensity, non-increasing clocks other than one midnight crossing whose
    unwrapped gap is < 12 h, a total span >= 24 h, and a file with no records.
    """
    header: tuple[str, Decimal] | None = None
    clocks: list[Decimal] = []
    intensities: list[float] = []
    day_offset = 0
    rollover_record: int | None = None
    previous: Decimal | None = None
    blanks = 0
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            blanks += 1
            continue
        if header is None:
            match = _HEADER_RE.fullmatch(line)
            if match is None:
                raise RainfallError(f"line {line_no}: header must be a single clock hh:mm:ss[.f], got {raw!r}")
            header = (line, _clock_seconds(match, line_no))
            previous = header[1]
            continue
        match = _RECORD_RE.fullmatch(line)
        if match is None:
            raise RainfallError(f"line {line_no}: record must be 'hh:mm:ss[.f] intensity', got {raw!r}")
        clock = _clock_seconds(match, line_no)
        token = match.group(4)
        if _NUMBER_RE.fullmatch(token) is None:
            raise RainfallError(f"line {line_no}: intensity {token!r} is not a decimal number")
        value = float(token)
        if not math.isfinite(value):
            raise RainfallError(f"line {line_no}: intensity {token!r} is not finite")
        if value < 0.0:
            raise RainfallError(f"line {line_no}: negative intensity {token!r}")
        if clock == previous:
            raise RainfallError(f"line {line_no}: duplicate clock {_clock_text(match)!r}")
        if clock < previous:
            unwrapped_gap = clock + SECONDS_PER_DAY - previous
            if rollover_record is not None:
                raise RainfallError(f"line {line_no}: clock goes backwards after the midnight crossing")
            if unwrapped_gap >= _MAX_ROLLOVER_GAP_S:
                raise RainfallError(
                    f"line {line_no}: clock {_clock_text(match)!r} is out of order (a midnight "
                    f"crossing would imply a {unwrapped_gap} s interval)"
                )
            day_offset = SECONDS_PER_DAY
            rollover_record = len(clocks) + 1
        absolute = clock + day_offset
        if absolute - header[1] >= SECONDS_PER_DAY:
            raise RainfallError(f"line {line_no}: schedule would span 24 h or more")
        clocks.append(absolute)
        intensities.append(value)
        previous = clock
    if header is None or not clocks:
        raise RainfallError("rainfall file needs a header clock and at least one record")
    origin = header[1]
    edges = [0.0] + [float(c - origin) for c in clocks]
    return RainfallSchedule(
        edges_s=edges,
        intensity_mm_per_h=intensities,
        provenance=RainfallProvenance(
            kind="legacy_file",
            path=path,
            sha256=sha256,
            size_bytes=size_bytes,
            start_clock=header[0],
            start_clock_s=float(origin),
            rollover_record=rollover_record,
            n_records=len(clocks),
            blank_lines_skipped=blanks,
        ),
    )


def parse_legacy_rainfall_file(path: str | Path) -> RainfallSchedule:
    """Read and parse a legacy rainfall file once, recording its SHA-256."""
    data = Path(path).read_bytes()
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise RainfallError(f"{path}: rainfall file is not ASCII ({exc})") from None
    return parse_legacy_rainfall_text(
        text, path=str(path), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data)
    )


# --- spatial application ----------------------------------------------------------
@dataclass(frozen=True, eq=False)
class RainfallField:
    """Validated `(ny, nx)` float64 multiplier in one array namespace: the
    scale where `active_mask` is True, 0 elsewhere. Build with
    `rainfall_field`. The multiplier is private (read-only on NumPy)."""

    multiplier: Any
    xp: ModuleType

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.multiplier.shape)

    def apply(self, value: float, *, out: Any = None) -> Any:
        """`multiplier * value` for a host scalar (a depth in m or a rate in
        m/s), resident in the field's namespace. `out`, if given, must be a
        float64 `(ny, nx)` array of that namespace and is reused."""
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise RainfallError(f"value must be a real host scalar, got {type(value).__name__}")
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise RainfallError(f"value must be finite and >= 0, got {value!r}")
        if out is None:
            return self.multiplier * value
        from maple.core.backend import array_namespace, is_array

        if not is_array(out) or array_namespace(self.multiplier, out) is not self.xp:
            raise RainfallError("out must be an array of the field's namespace")
        if out.shape != self.multiplier.shape or out.dtype != np.float64:
            raise RainfallError(f"out must be float64 {self.shape}, got {out.dtype} {out.shape}")
        return self.xp.multiply(self.multiplier, value, out=out)

    def depth_m(self, schedule: RainfallSchedule, t_s: float, dt_s: float, *, out: Any = None) -> Any:
        """Per-cell rainfall depth (m) over `[t, t + dt]`."""
        return self.apply(schedule.depth_m(t_s, dt_s), out=out)


def rainfall_field(
    ny: int,
    nx: int,
    *,
    scale: Any = None,
    active_mask: Any = None,
    xp: ModuleType | None = None,
) -> RainfallField:
    """Build the spatial rainfall multiplier for an authoritative `(ny, nx)`
    grid. `scale` (float64, finite, >= 0; default 1) and `active_mask` (bool;
    default all True) must be arrays of exactly that shape in one namespace,
    which is also `xp` if given (default NumPy when no array is given).

    Masked cells receive zero rainfall; the mask is not a negative rate and
    this module never reads or alters stored water. Inputs are not mutated
    or retained. Validation costs one device read, once.
    """
    from maple.core.backend import (
        array_namespace,
        finite_flag,
        freeze,
        is_array,
        negative_flag,
        read_flags,
    )

    for name, value in (("ny", ny), ("nx", nx)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise RainfallError(f"{name} must be a positive integer, got {value!r}")
    shape = (int(ny), int(nx))
    for name, array in (("scale", scale), ("active_mask", active_mask)):
        if array is not None and not is_array(array):
            raise RainfallError(f"{name} must be a NumPy/CuPy array, got {type(array).__name__}")
        if array is not None and tuple(array.shape) != shape:
            raise RainfallError(f"{name} shape {tuple(array.shape)} != (ny, nx) {shape}")
    namespace = array_namespace(scale, active_mask)  # raises on mixed namespaces
    if xp is not None and (scale is not None or active_mask is not None) and xp is not namespace:
        raise RainfallError(f"xp {xp.__name__!r} does not match the arrays' namespace {namespace.__name__!r}")
    namespace = xp if xp is not None else namespace
    if namespace.__name__ not in ("numpy", "cupy"):
        raise RainfallError(f"xp must be numpy or cupy, got {namespace.__name__!r}")
    if scale is not None and scale.dtype != np.float64:
        raise RainfallError(f"scale must be float64, got {scale.dtype}")
    if active_mask is not None and active_mask.dtype != np.bool_:
        raise RainfallError(f"active_mask must be bool, got {active_mask.dtype}")
    if scale is not None:
        finite, negative = read_flags([finite_flag(scale), negative_flag(scale)])
        if not finite or negative:
            raise RainfallError("scale must be finite and >= 0 everywhere")
    base = scale if scale is not None else namespace.ones(shape, dtype=np.float64)
    if active_mask is None:
        multiplier = namespace.array(base, dtype=np.float64, copy=True)
    else:
        multiplier = namespace.where(active_mask, base, 0.0)
    return RainfallField(multiplier=freeze(multiplier), xp=namespace)
