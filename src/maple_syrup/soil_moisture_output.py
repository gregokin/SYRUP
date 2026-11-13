"""Phase 2 G0: native every-N-completed-days SOIL-MOISTURE DIAGNOSTIC output (a writer/cadence component; NOT a restart checkpoint, NOT yet a CLI option or wired into any scheduler).

What it is
----------
A caller that drives the qualified native DAILY lifecycle (`daily_water`, `water_lifecycle`) explicitly constructs a `MoistureOutput` with a positive-integer cadence `N` (no default: it is off unless
constructed) and calls `notify_day_closed` once per accepted FINAL daily close, in order. The N-th, 2N-th, ... COMPLETED nominal day since the declared origin is saved as a sealed, self-describing snapshot
of the exact per-layer per-cell WATER INVENTORY (`water_m`, metres of water-equivalent depth, shape `(L, N_cells)` C order, float64) plus the static geometry needed to recover theta and volumes. Nothing in
the physical state (water, bed, history, ledgers, cursor, clock) is read for any purpose but the snapshot and nothing is mutated or advanced.

What counts as a completed day (derived from the native cursor, never from physical time)
------------------------------------------------------------------------------------------
Every accepted daily occurrence (`no_rain`, `recharge`, `storm`, `storm_fallback`) consumes exactly one ordinal of `DailyCursor` (`recharge.DailyCursor.ordinal` = the NEXT occurrence to consume), but it leaves an
`interevent` state: the day is FINAL only after any accepted LEAF withdrawals (`request_withdrawal`) and the genuine `close_interevent`, which yields a closed `boundary` carrying the SAME cursor (`ordinal` unchanged).
`notify_day_closed` therefore accepts ONLY a genuine, minted, digest-intact `boundary` state of a daily lineage with `pending is None` whose last phase receipt is the `interevent` close, bound to the
sealed lifecycle operator, with its exact-instance `ExpectedParent`. Days completed since the origin are `d = cursor.ordinal - origin_ordinal`; a storm day is ONE day however long the physical storm lasted
(`t_s` is recorded but never divides into days), and the state returned by `daily_water_action`/`finish_daily_storm` (before T and close) is refused because its role is `interevent`.

Sequential notification (no silent skips)
-----------------------------------------
`create` binds the lineage origin (the closed boundary supplied, `d = 0`, nothing is saved at 0) and `notify_day_closed` must then be called for d = 1, 2, 3, ... including days that are not due (they only advance a
bounded scalar). A skipped day, a rewound/other branch, or a day that regresses is refused. Repeating the call for the SAME accepted day with the SAME state instance is idempotent (no transfer, no write); a different
state for an already-notified day is a conflict. A caller that has no T on some day simply closes the interevent; closure establishes finality.

Snapshot format (bounded, no whole-array archive)
--------------------------------------------------
`<root>/output.json` (sealed policy + lineage + static record), `<root>/active.npy` (`(N_cells,)` bool, the column's mask, inactive entries preserved) and `<root>/day_<d:012d>/{water_m.npy, manifest.json}`.
`water_m.npy` is a genuine NumPy v1.0 `.npy` written in bounded row/column pieces of `chunk_bytes` (no host mirror); `manifest.json` is strict JSON (no NaN/Infinity), canonical, with a body SHA-256, and carries: day,
ordinal, cadence, nominal elapsed seconds (`d * 86400`, separate from the physical elapsed time `t_s`), optional ISO date, the water array's name/dtype/shape/order/units/size/SHA-256, the static record's SHA-256, and
the identities (run digest, lifecycle operator sha, state digest and revision, episode, reset count, request sequence, hydro steps, cursor scalars, storm-operator sha, last-phase receipt digest, this module's
sha). The profile is recorded as `fixed_full_porous_common_column`: dz/saturation are the common column's per-layer arrays and there is NO floating porous-depth array (a later floating owner must add the
actual porous-depth array under a new schema; no rock or porous depth is invented here). `theta = water_m / dz` and `volume_m3 = water_m * cell_area_m2` are recovered by `read_snapshot` without executing any physics.
The snapshot reconstructs the SAVED moisture only: not an unsaved time, not a restart.

Publication, crash safety, restart
----------------------------------
Atomic NO-CLOBBER: exclusive `mkdir` reservation of `day_<d>` (relative to an `O_NOFOLLOW|O_DIRECTORY` root descriptor), exclusive `O_CREAT|O_EXCL|O_NOFOLLOW` files, `fsync`, and the manifest-LAST `os.link` seal (the
pattern of `layered_checkpoint`; never a rename into an occupied target). Readers refuse an unsealed day directory. A crash or any failure leaves the physical state, the cursor and every sealed snapshot untouched;
a failed attempt removes only its own unsealed reservation (regular files of the two expected names, verified, no symlinks). The cadence is a bounded scalar in memory; on disk it is DERIVED from the sealed
snapshots (no mutable cursor file, no unbounded history). `MoistureOutput.resume` re-binds an existing output to a (possibly LAGGING) physical state: every due day <= the state's day must already be sealed (otherwise
the output missed a due day: refused), snapshots AHEAD of the state are kept and, when the replay reaches them, VERIFIED against the live state (array SHA-256 and identities; a mismatch is a conflict) and never overwritten. Contract
for the caller: notify first, then persist the physical checkpoint of that day. The old physical checkpoint schemas are not changed; this companion output state is separate and is to be wired at final Phase 2 integration.

Memory, bytes, quotas, devices
------------------------------
`MoistureOutputLimits` is an explicit record (no defaults): `max_snapshots`, `max_total_bytes` (all sealed bytes incl. static + manifests), `max_manifest_bytes`, `chunk_bytes` (host staging piece), `host_reserve_bytes`,
`disk_reserve_bytes`. Before ANY transfer or file write a due day checks the count, total-byte and disk quotas and the host staging requirement
`chunk_bytes + max_manifest_bytes + 65536` (+ reserve) against a truthful readable availability. Peak staging is ONE chunk (a device slice is downloaded with MAPLE's counted `to_host`, one transfer per piece,
zero transfers on a not-due day; a CPU array is written from views with no copy); on-disk bytes per snapshot = `len(npy header) + 8 L N + len(manifest)`. On CUDA the CURRENT device is gated by the lifecycle's own
preflight (`_check_lop`) BEFORE any quota reader or copy; device data stays resident on unsaved days. The module imports no CuPy and probes no device.

Not claimed: CPU/GPU bitwise identity of the physics; only that the same array bytes give the same snapshot bytes. No windowed/rock/floating profile; no CLI; no scheduler wiring.

Correction01 contracts (2026-11-13, after Root's independent probes)
---------------------------------------------------------------------
* A saved array is trusted only after its stored bytes are streamed against the manifest (header, size, regular file, SHA-256): `resume` does this for EVERY sealed snapshot (a declared COLD read of the sealed bytes, bounded by one `chunk_bytes`
  buffer) and `notify_day_closed` re-proves the saved file before reporting `verified_existing`, then compares it with the live water.
* After the manifest seal nothing sealed is ever deleted. A failure before the seal removes only the writer's own unsealed files; a failure at/after the seal raises `MoistureOutputIndeterminate` (published, durability uncertain) with payload and
  manifest intact and the cadence NOT advanced; the retry verifies the sealed identity and advances exactly once without rewriting. A leftover manifest temporary that is a hard link of the sealed manifest is tidied, nothing else.
* The static mask is streamed in `chunk_bytes` pieces like the water. Every quota (count, total bytes, host staging budget/reserve/reading, disk) is admitted BEFORE any download or write; the streaming loop never holds two host pieces.
* Reading is bounded: strict JSON (duplicate keys, non-finite numbers, huge integers and deep nesting refused), exact key/type/range validation of `output.json` and every manifest (typed errors, no raw `KeyError`/`TypeError`), and `read_snapshot(max_bytes)` caps ALL returned
  arrays (water + mask + dz) BEFORE any array is read; arrays are read straight into their final buffers (no temporary copies).
* `resume` followed by the SAME closed state is idempotent (a due day is re-proved against the sealed bytes; a non-due day is a repeat of the instance `resume` accepted); any other branch conflicts. No per-day directory enumeration: existence is a descriptor-relative no-follow `stat`;
  the resume audit is a streaming `scandir` with scalar counters (O(snapshot count) time, O(1) memory); due-day coverage is a count of validated unique multiples.
* Source identity: every snapshot records the closing phase receipt's cached `phase_source_digest` (a PER-PHASE digest that changes every day: informational, not a lineage-constant binding; no re-hash) and the writer module sha; `resume` refuses a changed writer module (no silent rebind, no migration of old G0 evidence) and a changed explicit caller identity.
  The full runtime source PAIR is not hashed here: the later adapter supplies it as the sealed `caller_identity`.

Correction02 contracts (2026-11-13)
-----------------------------------
* Quota inventory is exact across indeterminate publication: a snapshot that is sealed but whose notification could not complete is credited to `saved`/`used` immediately (a bounded scalar `_pending_day` receipt) while the cadence `last_day` stays
  pending; the retry (`verified_existing`) and a fresh `resume` scan never credit it twice, so the next day is refused by `max_snapshots`/`max_total_bytes` BEFORE any download or write.
* Metadata memory is modelled, admitted BEFORE it is built, and calibrated: `MoistureOutputLimits.staging_bytes(record_json_bytes) = chunk_bytes + SCRATCH_FACTOR * (max_manifest_bytes + record_json_bytes) + 64 KiB + host_reserve_bytes`
  with `SCRATCH_FACTOR = 6`. `create` bounds the record analytically from the known layer count (`JSON_FIXED_BYTES + len(caller_identity) + JSON_PER_LAYER_BYTES * L`, 3072 + 50 L) and admits that BEFORE the lists/JSON are built; `resume`/`verify_all`
  fstat `output.json` (<= 1 MiB) before reading it. Warm tracemalloc controls at L = 2, 128, 1024 are inside the plan (test_correction02.py). `read_snapshot(max_bytes)` caps only the RETURNED arrays; its additional metadata scratch is bounded by
  `SCRATCH_FACTOR * record size + 64 KiB` (at most about 6 MiB for the 1 MiB record ceiling), declared here and not part of `max_bytes`. The writer object keeps the parsed record resident (O(L), about 64 B per layer).
* Finiteness of loaded water is checked in `FINITE_BLOCK`-element blocks (no `(L, N)` boolean). Calendar overflow of `origin_date + day` and a day beyond `MAX_DAY = 10**12 - 1` (the 12-digit directory-name contract) are typed refusals before any I/O; resealed invalid
  day/date fields raise only `MoistureOutputError`.
"""


from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import stat
import sys
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from maple_syrup import recharge as rc
from maple_syrup import water_lifecycle as wl

__all__ = [
    "MoistureNotice",
    "MoistureOutput",
    "MoistureOutputConflict",
    "MoistureOutputError",
    "MoistureOutputIndeterminate",
    "MoistureOutputLimits",
    "MoistureOutputQuotaError",
    "MoistureSnapshot",
    "SCHEMA_OUTPUT",
    "SCHEMA_SNAPSHOT",
    "read_output_record",
    "read_snapshot",
    "snapshot_name",
]

SCHEMA_OUTPUT = "maple_syrup.soil_moisture_output/2"
SCHEMA_SNAPSHOT = "maple_syrup.soil_moisture_snapshot/2"
PROFILE_KIND = "fixed_full_porous_common_column"
DAY_S = float(rc.DAY_S)
OUTPUT_NAME, ACTIVE_NAME, WATER_NAME, MANIFEST_NAME = "output.json", "active.npy", "water_m.npy", "manifest.json"
TMP_SUFFIX = ".tmp"
PY_ALLOWANCE_BYTES = 65536
MAX_SMALL_JSON_BYTES = 1 << 20
MAX_LAYERS, MAX_CELLS, MAX_COUNT = 1 << 16, 1 << 40, (1 << 62) - 1
MAX_CALLER_IDENTITY_BYTES = 4096
MAX_DAY = 10**12 - 1  # the representational cap of the 12-digit `day_<d>` directory name (not a physics limit)
SCRATCH_FACTOR = 6  # declared conservative multiple of a JSON record's bytes for its transient Python/bytes/str/parsed copies (calibrated by the tracemalloc controls; see the module docstring)
JSON_FIXED_BYTES = 3072  # fixed keys/digests of an output record
JSON_PER_LAYER_BYTES = 50  # two float lists, <= 25 characters per entry including the comma
FINITE_BLOCK = 1 << 16  # elements per block of the bounded finiteness check
_UNITS = {"water_m": "m water-equivalent depth", "dz_m": "m", "cell_area_m2": "m2"}
_SHA = set("0123456789abcdef")
_FLAGS_RO = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
_FLAGS_DIR = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
_FLAGS_NEW = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
_MODULE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class MoistureOutputError(ValueError):
    """Invalid input, state or on-disk record. Nothing was published and no cadence was advanced."""


class MoistureOutputConflict(MoistureOutputError):
    """A different state/branch/record for an already notified or already saved day. Nothing was overwritten."""


class MoistureOutputQuotaError(MoistureOutputError, MemoryError):
    """A declared quota or the reported availability is insufficient. Nothing was transferred or written."""


class MoistureOutputIndeterminate(MoistureOutputError):
    """The snapshot was PUBLISHED (manifest sealed) but a later durability step failed: payload and seal are intact, the cadence did not advance, and a retry verifies the sealed identity and advances once. `path` names it."""

    def __init__(self, message: str, path: str) -> None:
        super().__init__(message)
        self.path = path


def _to_host(array: Any) -> np.ndarray:  # indirection so tests can count the (MAPLE-counted) device transfers
    from maple.core.backend import to_host

    return to_host(array)


def _count(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < minimum or int(value) > MAX_COUNT:
        raise MoistureOutputError(f"{name} must be an integer in [{minimum}, 2**62) (not a bool, float or non-finite), got {value!r}")
    return int(value)


@dataclass(frozen=True)
class MoistureOutputLimits:
    """Explicit quota record; every field is required (nothing is raised or defaulted silently). `host_memory_budget_bytes` bounds the modelled host staging of ONE call (`chunk_bytes + max_manifest_bytes + 64 KiB + host_reserve_bytes`)."""

    max_snapshots: int
    max_total_bytes: int
    max_manifest_bytes: int
    chunk_bytes: int
    host_memory_budget_bytes: int
    host_reserve_bytes: int
    disk_reserve_bytes: int

    def __post_init__(self) -> None:
        for name, minimum in (("max_snapshots", 1), ("max_total_bytes", 1), ("max_manifest_bytes", 256), ("chunk_bytes", 8), ("host_memory_budget_bytes", 1), ("host_reserve_bytes", 0), ("disk_reserve_bytes", 0)):
            object.__setattr__(self, name, _count(getattr(self, name), name, minimum=minimum))
        if self.max_manifest_bytes > MAX_SMALL_JSON_BYTES:
            raise MoistureOutputError(f"max_manifest_bytes may not exceed {MAX_SMALL_JSON_BYTES}")

    def staging_bytes(self, record_json_bytes: int = 0) -> int:
        """Modelled host peak of ONE call: one data piece + the transient copies of the JSON it parses/builds (`SCRATCH_FACTOR` x (manifest bound + the output record's size, known or bounded from L)) + 64 KiB + reserve."""
        return self.chunk_bytes + SCRATCH_FACTOR * (self.max_manifest_bytes + int(record_json_bytes)) + PY_ALLOWANCE_BYTES + self.host_reserve_bytes


@dataclass(frozen=True)
class MoistureNotice:
    day: int  # completed nominal days since the origin
    ordinal: int
    due: bool
    action: str  # not_due | saved | verified_existing | repeat_same_state
    path: str | None
    bytes_written: int
    device_chunks: int  # counted device-to-host pieces of THIS call (0 on a CPU lifecycle and on every not-due day)


@dataclass(frozen=True)
class MoistureSnapshot:
    manifest: dict
    water_m: np.ndarray  # (L, N) float64, read-only
    active: np.ndarray  # (N,) bool, read-only
    static: dict
    dz_m: np.ndarray
    cell_area_m2: float

    def theta(self) -> np.ndarray:
        """Volumetric water content `water_m / dz` per layer and cell (no physics)."""
        return self.water_m / self.dz_m[:, None]

    def volume_m3(self) -> np.ndarray:
        """Water volume `water_m * cell_area_m2` per layer and cell (inactive entries are preserved as saved)."""
        return self.water_m * self.cell_area_m2


# ---- strict JSON and exact validation --------------------------------------------------------------------------------------------------------------------------------------------------------
def snapshot_name(day: int) -> str:
    return f"day_{int(day):012d}"


def _canon(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True).encode("ascii")


def _strict_loads(raw: bytes) -> Any:
    """Strict JSON: ASCII only; duplicate keys, NaN/Infinity constants, non-finite numeric results (`1e999`), integers of more than 40 digits and pathological nesting are refused."""

    def bad_const(token: str) -> Any:
        raise MoistureOutputError(f"non-finite JSON constant {token!r} is refused")

    def no_dup(pairs: list) -> dict:
        out: dict = {}
        for k, v in pairs:
            if k in out:
                raise MoistureOutputError(f"duplicate JSON key {k!r} is refused")
            out[k] = v
        return out

    def num(text: str) -> float:
        v = float(text)
        if not math.isfinite(v):
            raise MoistureOutputError(f"JSON number {text[:24]!r} is not finite")
        return v

    def integer(text: str) -> int:
        if len(text) > 40:
            raise MoistureOutputError("a JSON integer of more than 40 digits is refused")
        return int(text)

    try:
        return json.loads(raw.decode("ascii"), parse_constant=bad_const, object_pairs_hook=no_dup, parse_float=num, parse_int=integer)
    except MoistureOutputError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise MoistureOutputError(f"invalid strict JSON: {type(exc).__name__}") from None


def _body_sha(obj: dict) -> str:
    return hashlib.sha256(_canon({k: v for k, v in obj.items() if k != "body_sha256"})).hexdigest()


def _is_int(v: Any, lo: int, hi: int = MAX_COUNT) -> bool:
    return type(v) is int and lo <= v <= hi


def _is_num(v: Any) -> bool:
    return type(v) in (int, float) and math.isfinite(v)


def _is_hex(v: Any) -> bool:
    return type(v) is str and len(v) == 64 and set(v) <= _SHA


def _need(obj: Any, keys: set, what: str) -> dict:
    if type(obj) is not dict or set(obj) != keys:
        extra = [] if type(obj) is not dict else sorted(set(obj) ^ keys)
        raise MoistureOutputError(f"{what} must be an object with exactly the expected keys (difference: {extra})")
    return obj


def _chk(cond: bool, what: str) -> None:
    if not cond:
        raise MoistureOutputError(f"invalid {what}")


_REC_KEYS = {"schema", "kind", "every_n_days", "origin_ordinal", "origin_date", "run_digest", "lop_sha256", "clock_contract", "profile", "profile_note", "n_layers", "n_cells", "shape_yx", "dz_m",
             "saturation_m", "cell_area_m2", "n_active", "active_sha256", "units", "max_manifest_bytes", "module_sha256", "caller_identity", "not_a_restart", "static_sha256", "body_sha256"}


def _validate_record(rec: Any) -> dict:
    rec = _need(rec, _REC_KEYS, "output.json")
    _chk(rec["schema"] == SCHEMA_OUTPUT and rec["kind"] == "soil_moisture_output" and rec["profile"] == PROFILE_KIND and rec["not_a_restart"] is True and rec["units"] == _UNITS, "output.json header")
    _chk(_is_int(rec["every_n_days"], 1, 1 << 31) and _is_int(rec["origin_ordinal"], 0) and _is_int(rec["n_layers"], 1, MAX_LAYERS) and _is_int(rec["n_cells"], 1, MAX_CELLS), "output.json counts")
    _chk(_is_hex(rec["run_digest"]) and _is_hex(rec["lop_sha256"]) and _is_hex(rec["active_sha256"]) and _is_hex(rec["module_sha256"]) and _is_hex(rec["static_sha256"]) and _is_hex(rec["body_sha256"]), "output.json digests")
    _chk(type(rec["clock_contract"]) is str and type(rec["profile_note"]) is str and len(rec["clock_contract"]) < 200 and len(rec["profile_note"]) < 400, "output.json strings")
    _chk(rec["origin_date"] is None or (type(rec["origin_date"]) is str and _date_ok(rec["origin_date"])), "origin_date")
    L, N = rec["n_layers"], rec["n_cells"]
    sy = rec["shape_yx"]
    _chk(type(sy) is list and len(sy) == 2 and all(_is_int(v, 1, MAX_CELLS) for v in sy) and sy[0] * sy[1] == N, "shape_yx")
    for key in ("dz_m", "saturation_m"):
        v = rec[key]
        _chk(type(v) is list and len(v) == L and all(_is_num(x) for x in v), key)
    _chk(all(x > 0.0 for x in rec["dz_m"]) and all(x > 0.0 for x in rec["saturation_m"]), "layer geometry (positive)")
    _chk(_is_num(rec["cell_area_m2"]) and rec["cell_area_m2"] > 0.0 and _is_int(rec["n_active"], 0, N), "cell area / n_active")
    _chk(_is_int(rec["max_manifest_bytes"], 256, MAX_SMALL_JSON_BYTES), "max_manifest_bytes")
    ci = rec["caller_identity"]
    _chk(ci is None or (type(ci) is dict and len(_canon(ci)) <= MAX_CALLER_IDENTITY_BYTES and all(type(k) is str and type(v) is str for k, v in ci.items())), "caller_identity")
    body = {k: v for k, v in rec.items() if k not in ("static_sha256", "body_sha256")}
    _chk(rec["static_sha256"] == hashlib.sha256(_canon(body)).hexdigest() and rec["body_sha256"] == _body_sha(rec), "output.json checksums")
    return rec


def _date_for(rec: dict, day: int, *, strict: bool) -> str | None:
    """ISO date `origin_date + day` (None without an origin date). A date beyond year 9999 is a TYPED refusal (strict) or an invalid-record error, never an `OverflowError`."""
    if rec["origin_date"] is None:
        return None
    try:
        return (datetime.date.fromisoformat(rec["origin_date"]) + datetime.timedelta(days=int(day))).isoformat()
    except (OverflowError, ValueError):
        raise MoistureOutputError(f"day {day} from origin date {rec['origin_date']} is beyond the representable calendar (year 9999); refused before any I/O") from None


def _date_ok(text: str) -> bool:
    try:
        datetime.date.fromisoformat(text)
        return len(text) == 10
    except ValueError:
        return False


_MAN_KEYS = {"schema", "kind", "not_a_restart", "day", "ordinal", "every_n_days", "origin_ordinal", "nominal_day_s", "nominal_elapsed_s", "physical_t_s", "date", "profile", "static_sha256", "run_digest", "water",
             "identity", "body_sha256"}
_WATER_KEYS = {"file", "dtype", "shape", "order", "units", "bytes", "file_sha256"}
_ID_KEYS = {"lop_sha256", "state_digest", "state_revision", "episode", "reset_count", "request_sequence", "hydro_steps", "storm_operator_sha256", "last_plan_digest", "last_branch", "last_parent_digest",
            "cursor_digest", "last_phase_receipt_digest", "phase_source_digest", "module_sha256"}


def _check_manifest(man: Any, rec: dict, name: str) -> dict:
    man = _need(man, _MAN_KEYS, "manifest.json")
    _chk(man["schema"] == SCHEMA_SNAPSHOT and man["kind"] == "daily_soil_moisture_diagnostic" and man["not_a_restart"] is True and man["profile"] == PROFILE_KIND, "manifest header")
    _chk(_is_int(man["day"], 1) and _is_int(man["ordinal"], 0) and man["ordinal"] == rec["origin_ordinal"] + man["day"] and man["origin_ordinal"] == rec["origin_ordinal"], "manifest day/ordinal")
    _chk(snapshot_name(man["day"]) == name and man["every_n_days"] == rec["every_n_days"] and man["day"] % rec["every_n_days"] == 0, "manifest day/cadence vs directory")
    _chk(man["nominal_day_s"] == DAY_S and _is_num(man["nominal_elapsed_s"]) and man["nominal_elapsed_s"] == man["day"] * DAY_S and _is_num(man["physical_t_s"]) and man["physical_t_s"] >= 0.0, "manifest times")
    _chk(man["day"] <= MAX_DAY, "manifest day (beyond the representable directory name)")
    _chk(man["date"] == _date_for(rec, man["day"], strict=False), "manifest date")
    _chk(man["static_sha256"] == rec["static_sha256"] and man["run_digest"] == rec["run_digest"] and _is_hex(man["body_sha256"]) and man["body_sha256"] == _body_sha(man), "manifest binding/checksum")
    w = _need(man["water"], _WATER_KEYS, "manifest water record")
    L, N = rec["n_layers"], rec["n_cells"]
    _chk(w["file"] == WATER_NAME and w["dtype"] == "<f8" and w["order"] == "C" and w["units"] == _UNITS["water_m"] and w["shape"] == [L, N] and w["bytes"] == len(_npy_header((L, N), "<f8")) + 8 * L * N
         and _is_hex(w["file_sha256"]), "manifest water record")
    i = _need(man["identity"], _ID_KEYS, "manifest identity")
    for k in ("lop_sha256", "state_digest", "storm_operator_sha256", "cursor_digest", "last_phase_receipt_digest", "module_sha256"):
        _chk(i[k] is None and k == "storm_operator_sha256" or _is_hex(i[k]), f"identity.{k}")
    for k in ("last_plan_digest", "last_parent_digest", "phase_source_digest"):
        _chk(i[k] is None or _is_hex(i[k]), f"identity.{k}")
    _chk(all(_is_int(i[k], 0) for k in ("state_revision", "episode", "reset_count", "request_sequence", "hydro_steps")) and (i["last_branch"] is None or type(i["last_branch"]) is str), "identity counters")
    _chk(i["lop_sha256"] == rec["lop_sha256"], "identity lop binding")
    return man


# ---- descriptor-relative file helpers ---------------------------------------------------------------------------------------------------------------------------------------------------------
def _open_dir(name: str | Path, *, dir_fd: int | None = None) -> int:
    try:
        return os.open(name, _FLAGS_DIR, dir_fd=dir_fd)
    except OSError as exc:
        raise MoistureOutputError(f"cannot open directory {str(name)!r} (missing, not a directory or a symlink): {exc.strerror}") from None


def _lstat(dirfd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=dirfd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise MoistureOutputError(f"cannot stat {name!r}: {exc.strerror}") from None


def _open_regular(dirfd: int, name: str, size: int | None = None, max_size: int | None = None) -> int:
    try:
        fd = os.open(name, _FLAGS_RO, dir_fd=dirfd)
    except OSError as exc:
        raise MoistureOutputError(f"cannot open {name!r} (missing or a symlink): {exc.strerror}") from None
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or (size is not None and info.st_size != size) or (max_size is not None and info.st_size > max_size):
        os.close(fd)
        raise MoistureOutputError(f"{name!r} is not a regular file of the expected size")
    return fd


def _read_small(dirfd: int, name: str, max_bytes: int) -> bytes:
    fd = _open_regular(dirfd, name, max_size=max_bytes)
    try:
        raw = b""
        while True:
            piece = os.read(fd, 1 << 16)
            if not piece:
                break
            raw += piece
            if len(raw) > max_bytes:
                raise MoistureOutputError(f"{name!r} exceeds {max_bytes} bytes")
        return raw
    finally:
        os.close(fd)


def _write_new(dirfd: int, name: str, chunks: Any) -> int:
    fd = os.open(name, _FLAGS_NEW, 0o600, dir_fd=dirfd)
    total = 0
    try:
        for piece in chunks:
            view = memoryview(piece)
            while len(view):
                n = os.write(fd, view)
                view = view[n:]
                total += n
        os.fsync(fd)
    finally:
        os.close(fd)
    return total


def _npy_header(shape: tuple[int, ...], descr: str) -> bytes:
    dims = "(" + ", ".join(str(int(n)) for n in shape) + ("," if len(shape) == 1 else "") + ")"
    text = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {dims}, }}"
    pad = (64 - (10 + len(text) + 1) % 64) % 64
    return b"\x93NUMPY\x01\x00" + (len(text) + pad + 1).to_bytes(2, "little") + (text + " " * pad + "\n").encode("latin1")


def _disk_available(dirfd: int) -> int:
    info = os.fstatvfs(dirfd)
    return int(info.f_bavail) * int(info.f_frsize)


def _hash_stored(dirfd: int, name: str, head: bytes, size: int, chunk_bytes: int) -> str:
    """SHA-256 of a STORED file, streamed through one preallocated buffer of at most `chunk_bytes`; refuses a non-regular/symlinked/wrong-sized file or a wrong `.npy` header."""
    fd = _open_regular(dirfd, name, size=size)
    try:
        buf = bytearray(max(8, min(chunk_bytes, size)))
        mv = memoryview(buf)
        h = hashlib.sha256()
        got = 0
        while got < size:
            n = os.readv(fd, [mv[:min(len(buf), size - got)]])
            if n <= 0:
                raise MoistureOutputError(f"{name!r} is shorter than recorded")
            if got < len(head):  # the header region may span several pieces: compare exactly the overlap
                hi = min(got + n, len(head))
                if bytes(mv[:hi - got]) != head[got:hi]:
                    raise MoistureOutputError(f"{name!r} header mismatch")
            h.update(mv[:n])
            got += n
        if os.read(fd, 1):
            raise MoistureOutputError(f"{name!r} has trailing bytes")
        return h.hexdigest()
    finally:
        os.close(fd)


def _check_saved_water(dayfd: int, man: dict, chunk_bytes: int) -> None:
    w = man["water"]
    head = _npy_header(tuple(w["shape"]), "<f8")
    if _hash_stored(dayfd, WATER_NAME, head, w["bytes"], chunk_bytes) != w["file_sha256"]:
        raise MoistureOutputError("the saved water array does not match its manifest checksum (corrupt snapshot)")


def _members(dayfd: int) -> tuple[set[str], bool]:
    """Day-directory members (at most a few entries; a streaming scandir) and whether a manifest temporary is a hard link of the sealed manifest."""
    names: set[str] = set()
    with os.scandir(dayfd) as it:
        for entry in it:
            names.add(entry.name)
            if len(names) > 4:
                raise MoistureOutputError("a snapshot directory has too many entries")
    tmp_ok = False
    if MANIFEST_NAME + TMP_SUFFIX in names and MANIFEST_NAME in names:
        a, b = _lstat(dayfd, MANIFEST_NAME), _lstat(dayfd, MANIFEST_NAME + TMP_SUFFIX)
        tmp_ok = a is not None and b is not None and (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)
    return names, tmp_ok


def _tidy_sealed(dayfd: int, names: set[str], tmp_ok: bool) -> None:
    """Remove ONLY a manifest temporary that is a hard link of the sealed manifest (a seal interrupted before its unlink); the sealed data is untouched."""
    if tmp_ok:
        os.unlink(MANIFEST_NAME + TMP_SUFFIX, dir_fd=dayfd)
        names.discard(MANIFEST_NAME + TMP_SUFFIX)
        try:
            os.fsync(dayfd)
        except OSError:
            pass


# ---- reading ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def read_output_record(root: str | Path) -> dict:
    """Validated static output record (strict JSON, exact keys/types/ranges, checksums)."""
    fd = _open_dir(root)
    try:
        return _read_output_record(fd)
    finally:
        os.close(fd)


def _record_size(rootfd: int) -> int:
    """Size of `output.json` (a descriptor-relative no-follow `fstat`), bounded by `MAX_SMALL_JSON_BYTES`; known BEFORE the record is read so its transient copies can be admitted."""
    info = _lstat(rootfd, OUTPUT_NAME)
    if info is None or not stat.S_ISREG(info.st_mode) or info.st_size > MAX_SMALL_JSON_BYTES:
        raise MoistureOutputError("output.json is missing, not a regular file, or above the record ceiling")
    return int(info.st_size)


def _read_output_record(rootfd: int) -> dict:
    return _validate_record(_strict_loads(_read_small(rootfd, OUTPUT_NAME, MAX_SMALL_JSON_BYTES)))


def _read_into(fd: int, flat_bytes: memoryview, h: Any) -> None:
    got = 0
    total = len(flat_bytes)
    while got < total:
        n = os.readv(fd, [flat_bytes[got:got + (1 << 20)]])
        if n <= 0:
            raise MoistureOutputError("a file is shorter than recorded")
        h.update(flat_bytes[got:got + n])
        got += n
    if os.read(fd, 1):
        raise MoistureOutputError("a file has trailing bytes")


def read_snapshot(path: str | Path, *, max_bytes: int) -> MoistureSnapshot:
    """Reconstruct ONE sealed snapshot (water, mask, dz, theta/volume helpers) without executing any physics. `path` is a `day_*` directory under its output root. `max_bytes` caps the TOTAL of the arrays this call
    returns (`8 L N` water + `N` mask + `8 L` dz) and is checked, from the validated record, BEFORE any array is read; arrays are read straight into their final buffers (no temporary copy). Metadata is bounded by
    `MAX_SMALL_JSON_BYTES` per record. Descriptor-relative `O_NOFOLLOW` opens, strict JSON, exact validation and checksums."""
    max_bytes = _count(max_bytes, "max_bytes", minimum=1)
    path = Path(path)
    parent = _open_dir(path.parent)
    try:
        rec = _read_output_record(parent)
        L, N = rec["n_layers"], rec["n_cells"]
        need = 8 * L * N + N + 8 * L
        if need > max_bytes:
            raise MoistureOutputError(f"the snapshot's arrays need {need} bytes but max_bytes is {max_bytes}; nothing was read")
        dayfd = _open_dir(path.name, dir_fd=parent)
        try:
            man = _check_manifest(_strict_loads(_read_small(dayfd, MANIFEST_NAME, rec["max_manifest_bytes"])), rec, path.name)
            names, tmp_ok = _members(dayfd)
            if names - ({MANIFEST_NAME + TMP_SUFFIX} if tmp_ok else set()) != {MANIFEST_NAME, WATER_NAME}:
                raise MoistureOutputError(f"snapshot directory has unexpected entries {sorted(names)}")
            active = _load_active(parent, rec)
            dz = np.asarray(rec["dz_m"], dtype=np.float64)
            water = _load_water(dayfd, man, L, N)
        finally:
            os.close(dayfd)
    finally:
        os.close(parent)
    water.flags.writeable = False
    active.flags.writeable = False
    return MoistureSnapshot(man, water, active, rec, dz, float(rec["cell_area_m2"]))


def _load_active(rootfd: int, rec: dict) -> np.ndarray:
    N = rec["n_cells"]
    head = _npy_header((N,), "|b1")
    fd = _open_regular(rootfd, ACTIVE_NAME, size=len(head) + N)
    try:
        h = hashlib.sha256()
        got = os.read(fd, len(head))
        if got != head:
            raise MoistureOutputError("active.npy header mismatch")
        h.update(head)
        raw = np.empty(N, dtype=np.uint8)
        _read_into(fd, memoryview(raw), h)
    finally:
        os.close(fd)
    if h.hexdigest() != rec["active_sha256"] or int(raw.max()) > 1 or int(np.count_nonzero(raw)) != rec["n_active"]:
        raise MoistureOutputError("active.npy does not match the sealed record")
    return raw.view(np.bool_)


def _load_water(dayfd: int, man: dict, L: int, N: int) -> np.ndarray:
    w = man["water"]
    head = _npy_header((L, N), "<f8")
    fd = _open_regular(dayfd, WATER_NAME, size=w["bytes"])
    try:
        h = hashlib.sha256()
        if os.read(fd, len(head)) != head:
            raise MoistureOutputError("water_m.npy header mismatch")
        h.update(head)
        data = np.empty((L, N), dtype=np.float64)
        _read_into(fd, memoryview(data.reshape(-1)).cast("B"), h)
    finally:
        os.close(fd)
    if h.hexdigest() != w["file_sha256"]:
        raise MoistureOutputError("water_m.npy checksum mismatch")
    flat = data.reshape(-1)
    for a in range(0, flat.size, FINITE_BLOCK):  # bounded temporaries: one FINITE_BLOCK-element boolean at a time, never a (L, N) bitmap
        if not np.isfinite(flat[a:a + FINITE_BLOCK]).all():
            raise MoistureOutputError("water_m.npy has non-finite entries")
    return data


# ---- the writer/cadence owner ---------------------------------------------------------------------------------------------------------------------------------------------------------------
class MoistureOutput:
    """Bounded cadence receipt + writer bound to ONE output directory, ONE lifecycle operator and ONE daily lineage. Construct with `create` or `resume`; there is no default cadence. A normal-API caller owns this object
    (its Python attributes are not defended against deliberate tampering)."""

    def __init__(self, token: object, **fields: Any) -> None:
        if token is not _TOKEN:
            raise MoistureOutputError("MoistureOutput is created only by MoistureOutput.create() or .resume()")
        self.__dict__.update(fields)

    @classmethod
    def create(cls, directory: str | Path, lop: Any, origin_state: Any, *, expected_parent: Any, every_n_days: Any, limits: MoistureOutputLimits, origin_date: str | None = None,
               caller_identity: dict | None = None, host_available_bytes_fn: Callable[[], int] | None = None, disk_available_bytes_fn: Callable[[], int] | None = None) -> "MoistureOutput":
        """Declare the lineage origin (a genuine closed daily boundary; `d = 0`) and create the sealed static record in an EMPTY directory. `every_n_days` must be a positive integer (no default). Every quota
        is admitted BEFORE the mask is downloaded or any file is written; the mask is then streamed in `chunk_bytes` pieces (a cold, separately counted transfer on CUDA)."""
        n = _count(every_n_days, "every_n_days", minimum=1)
        if n > 1 << 31:
            raise MoistureOutputError("every_n_days is unreasonably large")
        if not isinstance(limits, MoistureOutputLimits):
            raise MoistureOutputError("limits must be a MoistureOutputLimits")
        date0 = None if origin_date is None else _parse_date(origin_date)
        ci = _check_caller_identity(caller_identity)
        _check_boundary(lop, origin_state, expected_parent, require_close=False)
        column = lop.column
        L, N = int(column.n_layers), int(column.n_cells)
        if not (1 <= L <= MAX_LAYERS and 1 <= N <= MAX_CELLS):
            raise MoistureOutputError(f"the column's {L} layers x {N} cells are outside the supported record range")
        # O(L) metadata is ADMITTED before any list/JSON of that size is built: the record's size is bounded analytically from the known L (2 float lists) and the caller identity
        json_bound = JSON_FIXED_BYTES + (0 if ci is None else len(_canon(ci))) + JSON_PER_LAYER_BYTES * L
        if json_bound > MAX_SMALL_JSON_BYTES:
            raise MoistureOutputQuotaError(f"a {L}-layer output record needs up to {json_bound} bytes, above the {MAX_SMALL_JSON_BYTES}-byte record ceiling")
        head = _npy_header((N,), "|b1")
        static_bytes = json_bound + len(head) + N
        if static_bytes > limits.max_total_bytes:
            raise MoistureOutputQuotaError(f"the static record needs up to {static_bytes} bytes but max_total_bytes is {limits.max_total_bytes}")
        _admit_host(limits, limits.staging_bytes(json_bound), host_available_bytes_fn)
        rec = {"schema": SCHEMA_OUTPUT, "kind": "soil_moisture_output", "every_n_days": n, "origin_ordinal": int(origin_state.daily.ordinal), "origin_date": None if date0 is None else date0.isoformat(),
               "run_digest": origin_state.daily.run_digest, "lop_sha256": lop.sha256, "clock_contract": origin_state.daily.clock_contract, "profile": PROFILE_KIND,
               "profile_note": "common column dz/active only; no floating porous-depth array exists in this schema (a floating owner needs a new schema)", "n_layers": L, "n_cells": N,
               "shape_yx": [int(s) for s in lop.shape], "dz_m": [float(v) for v in np.asarray(column.layers.thickness_m, dtype=np.float64)],
               "saturation_m": [float(v) for v in np.asarray(column.layers.saturation_m, dtype=np.float64)], "cell_area_m2": float(column.cell_area_m2), "n_active": N,
               "active_sha256": "0" * 64, "units": dict(_UNITS), "max_manifest_bytes": limits.max_manifest_bytes, "module_sha256": _MODULE_SHA, "caller_identity": ci, "not_a_restart": True}
        _validate_inputs_finite(rec)
        bound = len(_canon(dict(rec, static_sha256="0" * 64, body_sha256="0" * 64)))
        if bound > json_bound:  # the analytic bound must hold (it is what was admitted)
            raise MoistureOutputError("internal: the output record exceeded its analytic size bound")
        rootfd = _open_dir(directory)
        try:
            with os.scandir(rootfd) as it:
                if any(True for _ in it):
                    raise MoistureOutputError("the output directory must be empty at creation (use MoistureOutput.resume for an existing output)")
            _check_disk(rootfd, static_bytes, limits, disk_available_bytes_fn)
            made: list[str] = []
            try:
                active_sha, n_active = _stream_mask(rootfd, column.active, N, head, limits.chunk_bytes, made)
                rec["n_active"], rec["active_sha256"] = n_active, active_sha
                rec["static_sha256"] = hashlib.sha256(_canon(rec)).hexdigest()
                rec["body_sha256"] = _body_sha(rec)
                blob = _canon(rec)
                _validate_record(json.loads(blob))
                made.append(OUTPUT_NAME + TMP_SUFFIX)
                _write_new(rootfd, OUTPUT_NAME + TMP_SUFFIX, [blob])
                _seal_link(rootfd, OUTPUT_NAME + TMP_SUFFIX, OUTPUT_NAME)
            except BaseException as exc:
                if _lstat(rootfd, OUTPUT_NAME) is None:  # never sealed: remove ONLY our own files
                    for nm in made:
                        try:
                            os.unlink(nm, dir_fd=rootfd)
                        except OSError:
                            pass
                    raise
                if isinstance(exc, MoistureOutputIndeterminate):
                    raise
                raise MoistureOutputIndeterminate(f"output.json is sealed but a durability step failed ({type(exc).__name__}); nothing was removed", str(directory)) from exc
            ident = os.fstat(rootfd)
        finally:
            os.close(rootfd)
        return cls._make(directory, lop, rec, limits, last_day=0, saved=0, used=len(blob) + len(head) + N, ident=(ident.st_dev, ident.st_ino))

    @classmethod
    def resume(cls, directory: str | Path, lop: Any, state: Any, *, expected_parent: Any, limits: MoistureOutputLimits, caller_identity: dict | None = None,
               host_available_bytes_fn: Callable[[], int] | None = None) -> "MoistureOutput":
        """Re-bind an existing output to a closed daily boundary (the physical restart point, possibly LAGGING the output). Verifies the static record and mask, every sealed manifest, STREAMS every sealed array against its manifest
        (a declared cold read of the sealed bytes through one `chunk_bytes` buffer) and that every due day up to the state's day is sealed; tidies crashed unsealed reservations of this writer; refuses a changed writer module or caller identity.
        Snapshots ahead of the state are kept and verified when the replay reaches them."""
        if not isinstance(limits, MoistureOutputLimits):
            raise MoistureOutputError("limits must be a MoistureOutputLimits")
        ci = _check_caller_identity(caller_identity)
        _check_boundary(lop, state, expected_parent, require_close=False)
        rootfd = _open_dir(directory)
        try:
            _admit_host(limits, limits.staging_bytes(_record_size(rootfd)), host_available_bytes_fn)  # the record's size is known (fstat) before it is read or parsed
            rec = _read_output_record(rootfd)
            _bind_check(rec, lop, state)
            if rec["module_sha256"] != _MODULE_SHA:
                raise MoistureOutputError("the output was written by a different writer module: no silent rebind or migration (old evidence is preserved and must be read with its own module)")
            if rec["caller_identity"] != ci:
                raise MoistureOutputError("the explicit caller identity differs from the one sealed in the output record")
            ident = os.fstat(rootfd)
            d = int(state.daily.ordinal) - rec["origin_ordinal"]
            if d < 0:
                raise MoistureOutputError("the state is older than the output's origin")
            if d >= 1 and (state.last_phase_receipt is None or state.last_phase_receipt.kind != "interevent"):
                raise MoistureOutputError("the boundary is not the close of a daily interevent (no completed day)")
            _verify_mask(rootfd, rec, limits.chunk_bytes)
            count, used, covered = _scan(rootfd, rec, limits, d)
            if covered != d // rec["every_n_days"]:
                raise MoistureOutputError(f"the output is missing a due snapshot at or before day {d}; it cannot be recreated: refused")
        finally:
            os.close(rootfd)
        out = cls._make(directory, lop, rec, limits, last_day=d, saved=count, used=used, ident=(ident.st_dev, ident.st_ino))
        out.__dict__["_state_ref"] = weakref.ref(state)
        return out

    @classmethod
    def _make(cls, directory, lop, rec, limits, *, last_day, saved, used, ident) -> "MoistureOutput":
        return cls(_TOKEN, directory=str(directory), lop_sha256=lop.sha256, rec=rec, limits=limits, last_day=last_day, saved=saved, used=used, ident=ident, _state_ref=None, _last_notice=None, _pending_day=None,
                   every_n_days=rec["every_n_days"], origin_ordinal=rec["origin_ordinal"], run_digest=rec["run_digest"])

    def notify_day_closed(self, lop: Any, state: Any, *, expected_parent: Any, host_available_bytes_fn: Callable[[], int] | None = None,
                          disk_available_bytes_fn: Callable[[], int] | None = None) -> MoistureNotice:
        """Call once per accepted FINAL daily close, in order (module docstring). Raises before any transfer/write on invalid input; a failed save leaves the cadence unchanged."""
        _check_boundary(lop, state, expected_parent, require_close=True)  # the device gate (CUDA preflight) runs FIRST, before any reader
        _bind_check(self.rec, lop, state)
        d = int(state.daily.ordinal) - self.origin_ordinal
        if d < 0:
            raise MoistureOutputConflict("the state precedes the lineage origin")
        if d == self.last_day and d >= 1:
            ref = self._state_ref
            same = ref is not None and ref() is state
            if same and self._last_notice is not None:
                return MoistureNotice(d, int(state.daily.ordinal), self._last_notice.due, "repeat_same_state", self._last_notice.path, 0, 0)
            if d % self.every_n_days == 0:  # a due day: re-prove against the SEALED bytes (this also serves the first call after resume)
                return self._due(lop, state, d, host_available_bytes_fn, disk_available_bytes_fn, replay=True)
            if same:  # a non-due day: the instance `resume` accepted is its own proof
                return MoistureNotice(d, int(state.daily.ordinal), False, "repeat_same_state", None, 0, 0)
            raise MoistureOutputConflict(f"day {d} was already notified for a different state instance/branch; refused (nothing overwritten)")
        if d > MAX_DAY:
            raise MoistureOutputError(f"day {d} exceeds the representable day-directory range {MAX_DAY}; refused before any I/O")
        _date_for(self.rec, d, strict=True)  # a calendar overflow is a typed refusal before any state change
        if d != self.last_day + 1:
            raise MoistureOutputError(f"days must be notified sequentially: expected day {self.last_day + 1}, got {d} (a skipped, rewound or foreign branch is refused)")
        due = d % self.every_n_days == 0
        if not due:
            return self._advance(d, state, MoistureNotice(d, int(state.daily.ordinal), False, "not_due", None, 0, 0), saved=0, used=0)
        return self._due(lop, state, d, host_available_bytes_fn, disk_available_bytes_fn, replay=False)

    def _advance(self, d: int, state: Any, notice: MoistureNotice, *, saved: int, used: int) -> MoistureNotice:
        self.__dict__.update(last_day=d, saved=self.saved + saved, used=self.used + used, _state_ref=weakref.ref(state), _last_notice=notice, _pending_day=None)
        return notice

    def _due(self, lop: Any, state: Any, d: int, host_fn: Any, disk_fn: Any, *, replay: bool) -> MoistureNotice:
        lim, rec = self.limits, self.rec
        L, N = rec["n_layers"], rec["n_cells"]
        water = state.payload.soil.water_m
        if tuple(water.shape) != (L, N):
            raise MoistureOutputError("the state's water array disagrees with the output geometry")
        head = _npy_header((L, N), "<f8")
        manifest = self._manifest(state, d, "0" * 64, len(head) + 8 * L * N)
        mblob = _canon(manifest)
        if len(mblob) > lim.max_manifest_bytes:
            raise MoistureOutputQuotaError(f"the manifest needs {len(mblob)} bytes but max_manifest_bytes is {lim.max_manifest_bytes}")
        need_disk = len(head) + 8 * L * N + len(mblob)
        rootfd = _open_dir(self.directory)
        try:
            info = os.fstat(rootfd)
            if (info.st_dev, info.st_ino) != self.ident:
                raise MoistureOutputConflict("the output directory is not the one this output was created for")
            name = snapshot_name(d)
            exists = _lstat(rootfd, name) is not None  # a descriptor-relative no-follow stat: no directory enumeration
            _admit_host(lim, lim.staging_bytes(), host_fn)  # truthful host reading BEFORE any transfer; the lifecycle already gated the device
            if exists:
                return self._verify_existing(rootfd, name, state, d, water, head, L, N, lop)
            if replay:
                raise MoistureOutputConflict(f"day {d} was notified before but its sealed snapshot is missing; refused")
            if self.saved + 1 > lim.max_snapshots:
                raise MoistureOutputQuotaError(f"max_snapshots {lim.max_snapshots} reached")
            if self.used + need_disk > lim.max_total_bytes:
                raise MoistureOutputQuotaError(f"a snapshot needs {need_disk} bytes but only {lim.max_total_bytes - self.used} of max_total_bytes remain")
            _check_disk(rootfd, need_disk, lim, disk_fn)
            return self._write(rootfd, name, state, d, water, head, L, N, manifest, need_disk, lop)
        finally:
            os.close(rootfd)

    def _manifest(self, state: Any, d: int, file_sha: str, file_bytes: int) -> dict:
        rec, cur, rcpt = self.rec, state.daily, state.last_phase_receipt
        date = _date_for(rec, d, strict=True)
        m = {"schema": SCHEMA_SNAPSHOT, "kind": "daily_soil_moisture_diagnostic", "not_a_restart": True, "day": d, "ordinal": int(cur.ordinal), "every_n_days": self.every_n_days, "origin_ordinal": self.origin_ordinal,
             "nominal_day_s": DAY_S, "nominal_elapsed_s": d * DAY_S, "physical_t_s": float(state.t_s), "date": date, "profile": PROFILE_KIND, "static_sha256": rec["static_sha256"], "run_digest": rec["run_digest"],
             "water": {"file": WATER_NAME, "dtype": "<f8", "shape": [rec["n_layers"], rec["n_cells"]], "order": "C", "units": _UNITS["water_m"], "bytes": file_bytes, "file_sha256": file_sha},
             "identity": {"lop_sha256": state.lop_sha256, "state_digest": state.digest, "state_revision": int(state.revision), "episode": int(state.episode), "reset_count": int(state.reset_count),
                          "request_sequence": int(state.request_sequence), "hydro_steps": int(state.hydro_count), "storm_operator_sha256": state.storm_operator_sha256,
                          "last_plan_digest": cur.last_plan_digest, "last_branch": cur.last_branch, "last_parent_digest": cur.last_parent_digest, "cursor_digest": cur.digest(),
                          "last_phase_receipt_digest": rcpt.digest(), "phase_source_digest": rcpt.source_digest, "module_sha256": _MODULE_SHA}}
        m["body_sha256"] = _body_sha(m)
        return m

    def _stream(self, water: Any, head: bytes, L: int, N: int, sink: Any) -> tuple[str, int]:
        """Hash (and optionally write) the `.npy` bytes of the live `(L, N)` water in contiguous pieces of at most `chunk_bytes`. EXACTLY one host piece is alive at a time (released before the next download)."""
        h = hashlib.sha256(head)
        if sink is not None:
            sink.write(head)
        per, pieces = max(1, self.limits.chunk_bytes // 8), 0
        for l in range(L):
            for a in range(0, N, per):
                host = _to_host(water[l, a:a + per])
                if host.dtype != np.float64 or not host.flags.c_contiguous:
                    raise MoistureOutputError("the water slice is not a contiguous float64 piece")
                mv = memoryview(host).cast("B")
                h.update(mv)
                if sink is not None:
                    sink.write(mv)
                pieces += 1
                del mv, host
        return h.hexdigest(), pieces

    def _write(self, rootfd: int, name: str, state: Any, d: int, water: Any, head: bytes, L: int, N: int, manifest: dict, need_disk: int, lop: Any) -> MoistureNotice:
        try:
            os.mkdir(name, 0o700, dir_fd=rootfd)  # exclusive reservation
        except FileExistsError:
            raise MoistureOutputConflict(f"{name} appeared concurrently; refused (nothing overwritten)") from None
        dayfd = _open_dir(name, dir_fd=rootfd)
        path = str(Path(self.directory) / name)
        made: list[str] = []
        try:
            try:  # ---- before the seal: failures remove only this writer's own unsealed files
                made.append(WATER_NAME)
                fd = os.open(WATER_NAME, _FLAGS_NEW, 0o600, dir_fd=dayfd)
                try:
                    with os.fdopen(fd, "wb", closefd=False) as fh:
                        sha, pieces = self._stream(water, head, L, N, fh)
                        fh.flush()
                    os.fsync(fd)
                finally:
                    os.close(fd)
                man = dict(manifest)
                man["water"] = dict(man["water"], file_sha256=sha)
                man["body_sha256"] = _body_sha(man)
                blob = _canon(man)
                made.append(MANIFEST_NAME + TMP_SUFFIX)
                _write_new(dayfd, MANIFEST_NAME + TMP_SUFFIX, [blob])
                _seal_link(dayfd, MANIFEST_NAME + TMP_SUFFIX, MANIFEST_NAME, unlink=False)
            except BaseException as exc:
                if _lstat(dayfd, MANIFEST_NAME) is None:  # not published: remove only own files
                    _discard(rootfd, name, dayfd, made)
                    raise
                self._credit_published(d, len(blob) + len(head) + 8 * L * N)
                raise MoistureOutputIndeterminate(f"{name} was published (manifest sealed) but the seal step failed afterwards ({type(exc).__name__}); payload and seal are intact and the cadence did not advance: retry to verify and advance", path) from exc
            try:  # ---- at/after the seal: NOTHING is ever removed
                os.unlink(MANIFEST_NAME + TMP_SUFFIX, dir_fd=dayfd)
                os.fsync(dayfd)
                os.fsync(rootfd)
            except BaseException as exc:
                self._credit_published(d, len(blob) + len(head) + 8 * L * N)
                raise MoistureOutputIndeterminate(f"{name} was published but a durability step failed ({type(exc).__name__}); payload and seal are intact and the cadence did not advance: retry to verify and advance", path) from exc
        finally:
            try:
                os.close(dayfd)
            except OSError:
                pass
        notice = MoistureNotice(d, int(state.daily.ordinal), True, "saved", path, need_disk, pieces if lop._cuda is not None else 0)
        return self._advance(d, state, notice, saved=1, used=len(blob) + len(head) + 8 * L * N)

    def _credit_published(self, d: int, nbytes: int) -> None:
        """A snapshot is on disk (sealed) but its notification could not complete: count it in the quota inventory NOW (exactly once, a bounded scalar receipt) while the cadence day stays pending. The retry
        (`verified_existing`) and a fresh `resume` scan do not credit it again."""
        if self._pending_day != d:
            self.__dict__.update(saved=self.saved + 1, used=self.used + nbytes, _pending_day=d)

    def _verify_existing(self, rootfd: int, name: str, state: Any, d: int, water: Any, head: bytes, L: int, N: int, lop: Any) -> MoistureNotice:
        dayfd = _open_dir(name, dir_fd=rootfd)
        try:
            names, tmp_ok = _members(dayfd)
            if MANIFEST_NAME not in names:
                raise MoistureOutputError(f"{name} is an unsealed reservation (a crashed write); resume() tidies it, it is never adopted or overwritten")
            if names - ({MANIFEST_NAME + TMP_SUFFIX} if tmp_ok else set()) != {MANIFEST_NAME, WATER_NAME}:
                raise MoistureOutputError(f"{name} has unexpected members {sorted(names)}")
            man = _check_manifest(_strict_loads(_read_small(dayfd, MANIFEST_NAME, self.limits.max_manifest_bytes)), self.rec, name)
            ident = man["identity"]
            if (man["ordinal"] != int(state.daily.ordinal) or ident["state_digest"] != state.digest or ident["cursor_digest"] != state.daily.digest() or float(man["physical_t_s"]) != float(state.t_s)
                    or ident["lop_sha256"] != state.lop_sha256):
                raise MoistureOutputConflict(f"the sealed snapshot of day {d} was written for a different state/branch; nothing was overwritten")
            _check_saved_water(dayfd, man, self.limits.chunk_bytes)  # the STORED bytes must match their manifest before anything relies on them
            sha, pieces = self._stream(water, head, L, N, None)  # the replayed day must hold exactly the saved bytes
            if sha != man["water"]["file_sha256"]:
                raise MoistureOutputConflict(f"the sealed snapshot of day {d} differs from the live water inventory; nothing was overwritten")
            _tidy_sealed(dayfd, names, tmp_ok)
        finally:
            os.close(dayfd)
        notice = MoistureNotice(d, int(state.daily.ordinal), True, "verified_existing", str(Path(self.directory) / name), 0, pieces if lop._cuda is not None else 0)
        return self._advance(d, state, notice, saved=0, used=0)

    def verify_all(self) -> int:
        """Re-check EVERY sealed snapshot's stored bytes (streamed through one `chunk_bytes` buffer) and the mask; returns the sealed count. A declared cold read."""
        rootfd = _open_dir(self.directory)
        try:
            _admit_host(self.limits, self.limits.staging_bytes(_record_size(rootfd)), None)
            rec = _read_output_record(rootfd)
            if rec != self.rec:
                raise MoistureOutputConflict("output.json changed since this output was bound")
            _verify_mask(rootfd, rec, self.limits.chunk_bytes)
            count, _, _ = _scan(rootfd, rec, self.limits, 0)
        finally:
            os.close(rootfd)
        return count


_TOKEN = object()


# ---- shared checks and bounded helpers --------------------------------------------------------------------------------------------------------------------------------------------------------
def _parse_date(text: Any) -> datetime.date:
    if type(text) is not str or not _date_ok(text):
        raise MoistureOutputError("origin_date must be an ISO 'YYYY-MM-DD' string")
    return datetime.date.fromisoformat(text)


def _check_caller_identity(ci: Any) -> dict | None:
    if ci is None:
        return None
    if type(ci) is not dict or not all(type(k) is str and type(v) is str for k, v in ci.items()):
        raise MoistureOutputError("caller_identity must be a dict of str to str")
    if len(_canon(ci)) > MAX_CALLER_IDENTITY_BYTES:
        raise MoistureOutputError(f"caller_identity exceeds {MAX_CALLER_IDENTITY_BYTES} bytes")
    return dict(ci)


def _validate_inputs_finite(rec: dict) -> None:
    for key in ("dz_m", "saturation_m"):
        if not all(math.isfinite(x) for x in rec[key]):
            raise MoistureOutputError(f"{key} of the lifecycle column is not finite")
    if not (math.isfinite(rec["cell_area_m2"]) and rec["cell_area_m2"] > 0.0):
        raise MoistureOutputError("the lifecycle column's cell area is not finite and positive")


def _check_boundary(lop: Any, state: Any, expected_parent: Any, *, require_close: bool) -> None:
    """Genuine lop/state/instance and the CLOSED daily-boundary finality (module docstring). On CUDA `_check_lop` queries the CURRENT device first (before any reader or copy)."""
    try:
        wl._check_lop(lop)  # genuine operator; on CUDA the preflight gates the current device
        wl._require_parent(state, expected_parent)
        wl._check_state(lop, state)
    except wl.LifecycleError as exc:
        raise MoistureOutputError(f"not a genuine lifecycle operator/state/expected parent: {exc}") from None
    if state.role != "boundary":
        raise MoistureOutputError(f"only a closed 'boundary' is a final daily close, got role {state.role!r} (the state before T/close is not final)")
    cur = state.daily
    if cur is None or type(cur) is not rc.DailyCursor:
        raise MoistureOutputError("the state does not belong to a daily lineage")
    if cur.pending is not None:
        raise MoistureOutputError("a storm occurrence is pending: the day is not complete")
    if require_close:
        r = state.last_phase_receipt
        if r is None or r.kind != "interevent" or cur.ordinal < 1:
            raise MoistureOutputError("the boundary is not the close of a daily interevent (no completed day)")


def _bind_check(rec: dict, lop: Any, state: Any) -> None:
    if rec["lop_sha256"] != lop.sha256 or state.lop_sha256 != lop.sha256:
        raise MoistureOutputError("the output is bound to a different lifecycle operator")
    if state.daily.run_digest != rec["run_digest"] or state.daily.clock_contract != rec["clock_contract"]:
        raise MoistureOutputError("the state belongs to a different daily lineage/clock contract than the output")
    if list(rec["shape_yx"]) != [int(s) for s in lop.shape] or rec["n_cells"] != int(lop.column.n_cells) or rec["n_layers"] != int(lop.column.n_layers):
        raise MoistureOutputError("the output geometry disagrees with the lifecycle column")


def _read_count(value: Any, what: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise MoistureOutputError(f"the {what} available-bytes reader must return an integer >= 0, got {value!r}")
    return int(value)


def _admit_host(lim: MoistureOutputLimits, staging: int, host_fn: Callable[[], int] | None) -> None:
    """Host staging admission (modelled `staging` vs the explicit budget and a truthful reading) BEFORE any download or file write."""
    if staging > lim.host_memory_budget_bytes:
        raise MoistureOutputQuotaError(f"snapshot staging needs {staging} host bytes but host_memory_budget_bytes is {lim.host_memory_budget_bytes}; nothing was transferred")
    if host_fn is None:
        from maple_syrup import spatial_soil_layers as sp

        host_fn = sp.default_host_available_bytes
    avail = _read_count(host_fn(), "host")
    if staging > avail:
        raise MoistureOutputQuotaError(f"snapshot staging needs {staging} host bytes but only {avail} are reported available; nothing was transferred")


def _check_disk(rootfd: int, need: int, lim: MoistureOutputLimits, disk_fn: Callable[[], int] | None) -> None:
    avail = _read_count(_disk_available(rootfd) if disk_fn is None else disk_fn(), "disk")
    if need + lim.disk_reserve_bytes > avail:
        raise MoistureOutputQuotaError(f"the output needs {need} disk bytes (+ reserve {lim.disk_reserve_bytes}) but only {avail} are available; nothing was written")


def _seal_link(dirfd: int, tmp_name: str, final_name: str, *, unlink: bool = True) -> None:
    """Atomic no-clobber publication: hard-link the finished temporary to its final name (fails if it exists). `unlink=True` also drops the temporary and fsyncs (used for the static record)."""
    os.link(tmp_name, final_name, src_dir_fd=dirfd, dst_dir_fd=dirfd, follow_symlinks=False)
    if unlink:
        os.unlink(tmp_name, dir_fd=dirfd)
        os.fsync(dirfd)


def _discard(rootfd: int, name: str, dayfd: int, made: list[str]) -> None:
    """Remove ONLY this writer's own unsealed regular files (named in `made`) and the then-empty reservation; never anything sealed, unknown or symlinked."""
    try:
        for entry in made:
            info = _lstat(dayfd, entry)
            if info is not None and stat.S_ISREG(info.st_mode):
                os.unlink(entry, dir_fd=dayfd)
        os.rmdir(name, dir_fd=rootfd)
    except OSError:
        pass


def _stream_mask(rootfd: int, active: Any, N: int, head: bytes, chunk_bytes: int, made: list[str]) -> tuple[str, int]:
    """Stream the column's `(N,)` bool mask to `active.npy` in pieces of `chunk_bytes` (a cold, separately counted transfer on CUDA), returning (SHA-256 of the file, active count)."""
    made.append(ACTIVE_NAME)
    fd = os.open(ACTIVE_NAME, _FLAGS_NEW, 0o600, dir_fd=rootfd)
    h = hashlib.sha256(head)
    n_active = 0
    try:
        with os.fdopen(fd, "wb", closefd=False) as fh:
            fh.write(head)
            for a in range(0, N, chunk_bytes):
                host = _to_host(active[a:a + chunk_bytes])
                if host.dtype != np.bool_ or not host.flags.c_contiguous:
                    raise MoistureOutputError("the column mask is not a contiguous bool array")
                mv = memoryview(host.view(np.uint8)).cast("B")
                h.update(mv)
                fh.write(mv)
                n_active += int(np.count_nonzero(host))
                del mv, host
            fh.flush()
        os.fsync(fd)
    finally:
        os.close(fd)
    return h.hexdigest(), n_active


def _verify_mask(rootfd: int, rec: dict, chunk_bytes: int) -> None:
    N = rec["n_cells"]
    head = _npy_header((N,), "|b1")
    if _hash_stored(rootfd, ACTIVE_NAME, head, len(head) + N, chunk_bytes) != rec["active_sha256"]:
        raise MoistureOutputError("active.npy does not match the sealed record")


def _tidy_static_tmp(rootfd: int) -> None:
    """`output.json.tmp` is tidied ONLY as the hard link of the VALIDATED sealed `output.json` (a creation interrupted between the link and the unlink): the same regular file (device + inode). Any other
    regular file, symlink or special file under that name is not ours: refuse and preserve it."""
    a, b = _lstat(rootfd, OUTPUT_NAME), _lstat(rootfd, OUTPUT_NAME + TMP_SUFFIX)
    if a is None or b is None or not stat.S_ISREG(a.st_mode) or not stat.S_ISREG(b.st_mode) or (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
        raise MoistureOutputError(f"{OUTPUT_NAME + TMP_SUFFIX} is not a hard link of the sealed {OUTPUT_NAME}; it is not this writer's temporary, nothing was removed")
    os.unlink(OUTPUT_NAME + TMP_SUFFIX, dir_fd=rootfd)
    try:
        os.fsync(rootfd)
    except OSError:
        pass


def _scan(rootfd: int, rec: dict, lim: MoistureOutputLimits, d: int) -> tuple[int, int, int]:
    """Streaming audit with scalar counters (O(1) memory, O(snapshot count) time): returns (sealed count, bytes used, count of sealed days <= d). Every sealed manifest is validated, its stored water
    streamed against the manifest checksum, symlinked/non-regular members refused, an unsealed reservation of this writer tidied, any unknown entry refused; the counts/bytes respect the quotas."""
    used = 0
    for nm in (OUTPUT_NAME, ACTIVE_NAME):
        info = _lstat(rootfd, nm)
        if info is None or not stat.S_ISREG(info.st_mode):
            raise MoistureOutputError(f"{nm} is missing or not a regular file")
        used += info.st_size
    count = covered = seen = 0
    static_tmp = False
    with os.scandir(rootfd) as it:
        for entry in it:
            seen += 1
            if seen > lim.max_snapshots + 4:
                raise MoistureOutputQuotaError("the output directory holds more entries than max_snapshots allows")
            name = entry.name
            if name in (OUTPUT_NAME, ACTIVE_NAME):
                continue
            if name == OUTPUT_NAME + TMP_SUFFIX:
                static_tmp = True  # judged after the loop (its hard-link proof needs the validated output.json); never touched here
                continue
            if not (name.startswith("day_") and len(name) == 16 and name[4:].isdigit()):
                raise MoistureOutputError(f"unexpected entry {name!r} in the output directory")
            dayfd = _open_dir(name, dir_fd=rootfd)
            try:
                names, tmp_ok = _members(dayfd)
                if MANIFEST_NAME not in names:
                    # Only a reservation made by THIS writer (its two expected REGULAR names and nothing else) is cleaned. Any unknown or non-regular member proves the directory is not ours:
                    # refuse and preserve EVERYTHING in it (no known file is deleted either).
                    own = {WATER_NAME, MANIFEST_NAME + TMP_SUFFIX}
                    for member in sorted(names):
                        info = _lstat(dayfd, member)
                        if member not in own or info is None or not stat.S_ISREG(info.st_mode):
                            raise MoistureOutputError(f"{name} is an unsealed directory with unrecognised or non-regular member {member!r}; it is not this writer's reservation, nothing was removed")
                    _discard(rootfd, name, dayfd, sorted(names))  # a crashed, never-sealed reservation
                    continue
                if names - ({MANIFEST_NAME + TMP_SUFFIX} if tmp_ok else set()) != {MANIFEST_NAME, WATER_NAME}:
                    raise MoistureOutputError(f"{name} has unexpected members {sorted(names)}")
                man = _check_manifest(_strict_loads(_read_small(dayfd, MANIFEST_NAME, lim.max_manifest_bytes)), rec, name)
                _check_saved_water(dayfd, man, lim.chunk_bytes)
                _tidy_sealed(dayfd, names, tmp_ok)
                used += man["water"]["bytes"] + os.stat(MANIFEST_NAME, dir_fd=dayfd, follow_symlinks=False).st_size
            finally:
                os.close(dayfd)
            count += 1
            if man["day"] <= d:
                covered += 1
    if static_tmp:
        _tidy_static_tmp(rootfd)
    if count > lim.max_snapshots or used > lim.max_total_bytes:
        raise MoistureOutputQuotaError("the existing output already exceeds the declared quotas")
    return count, used, covered


if sys.byteorder != "little":  # the .npy files are written '<f8' straight from memory
    raise ImportError("soil_moisture_output requires a little-endian host")
