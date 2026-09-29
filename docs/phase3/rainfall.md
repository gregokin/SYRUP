# Phase 3a — rainfall forcing

Module: `src/maple_syrup/rainfall.py`. Tests: `tests/phase3/test_rainfall.py`.
Reference: MAHLERAN 1.2.3 `src/Subroutines_In_out/Set_rain_xml.f90` (`rain_type` 1 and 2) and `time_conv` (`Functions.for` 190–204). Stochastic `rain_type = 3` is not ported.

This document covers the forcing alone. Infiltration, routing and the event driver are not implemented. The `inf_type = 3` `ksat` update inside `set_rain_xml` belongs to infiltration and is left for that task.

## Units and time origin

- Time: seconds from the event origin, `t >= 0`. For a legacy file, the origin (`t = 0`) is the header clock.
- Intensity input: mm/h, the legacy unit. Stored rates are m/s (`mm/h / 3.6e6`). Depths are in m.
- Wall clocks are parsed once, when the file is read, and never enter evaluation.

## Legacy file format and the retained timing physics

The first non-blank line holds the start clock (`hh:mm:ss[.fraction]`). Each later line is `hh:mm:ss[.fraction] intensity`, where the clock marks the **end** of its interval. This matches the Fortran code: on the first call it reads record 1 and applies its intensity from `time_last = 0` to `time_next = T_1`. Each later record `k` applies on `(T_{k-1}, T_k]`.

Retained: the interval-ending assignment and the zero rate after the final record. At end-of-file the legacy code sets the intensity to `'0.0'` and keeps it there.

**Not reproduced:** the step-lagged switch. The legacy code changes to the next record only on the first step with `iter*dt > time_next`, so each switch can lag by up to one `dt`, depending on the step size. Here each record is integrated over its exact interval, so a step that spans a record end receives each part at its own rate. The Plot 1 test checks that `[59, 60] s` receives record 1 and `[60, 61] s` receives record 2.

Parser rules:

- **Tolerated:** leading and trailing blanks, any run of blanks or tabs between the two fields, CR/LF line endings, and blank lines. Skipped blank lines are counted in the provenance.
- **Rejected:** non-ASCII bytes, extra fields, out-of-range clocks (hh ≥ 24, mm ≥ 60, ss ≥ 60, or not two digits), and non-numeric, negative or non-finite intensities, including overflow such as `1e999`.
- **Rejected:** duplicate clocks (a zero-length interval) and a file with no records.
- **Stricter than legacy:** the whole intensity token is read, not columns 13–22 only.
- **Midnight rollover:** a decreasing clock counts as one midnight crossing only when the unwrapped gap is under 12 h, i.e. when crossing midnight is the shorter reading. Only one crossing is allowed, and the total span must stay under 24 h. Anything else is out of order and is rejected.

  The legacy code instead subtracts 86400 s from `start_sec` on every record whose clock string sorts below the start clock. After midnight that is every later record, so this is not reproduced.

- **Provenance:** `RainfallProvenance` records the path, SHA-256 of the exact bytes, size, header clock, rollover record, record count, skipped blank lines and the timing-convention string.

Plot 1 (`Input/input_p1/p1_01_08_06.dat`) has 27 one-minute records from 18:00:00.00 to 18:27:00.00, a total of 9.652 mm.

## Schedule API (host, immutable, stateless)

- `RainfallSchedule(edges_s, intensity_mm_per_h, provenance)`:
  - `edges_s` is strictly increasing, finite and ≥ 0, with `n + 1` entries.
  - The `n` intensities are finite and ≥ 0.
  - Arrays are read-only copies, and `rate_m_per_s` is derived from the intensities.
- `constant_rainfall(start_s, end_s, intensity_mm_per_h)` is the legacy `rain_type = 1` (`rf_mean`) over an explicit window.
- `depth_m(t, dt)` is the exact integral over `[t, t + dt]`, and zero outside the schedule.
  - `t` and `dt` must be finite real numbers with `t ≥ 0` and `dt ≥ 0`. `dt = 0` gives 0.
  - One step may span any number of records.
- `pieces(t, dt)` splits `[t, t + dt]` at every interior edge into contiguous constant-rate `RainfallPiece`s, including zero-rate pieces.
- `next_edge_s(t)` returns the first discontinuity strictly after `t`, or `inf` from the end onwards.
- `rate_after_m_per_s(t)` returns the rate immediately after `t`.

There is no cursor or clock state, so restart needs only the model time. A nonlinear consumer such as infiltration must substep at `pieces` / `next_edge_s` boundaries and must not use `depth_m / dt` as a rate across a jump.

The edge search, and the Python loop over the pieces of one step, run on the host once per time interval. They never run per cell.

## Spatial application

`rainfall_field(ny, nx, *, scale=None, active_mask=None, xp=None)` validates the inputs once and returns a `RainfallField` whose float64 `(ny, nx)` multiplier is `scale` where `active_mask` is True and 0 elsewhere.

- `(ny, nx)` is authoritative. `scale` must be float64, finite and ≥ 0 everywhere, including masked cells. `active_mask` must be bool.
- Both arrays must share one namespace, resolved by MAPLE's `array_namespace`, which raises on a NumPy/CuPy mix. An explicit `xp` must match that namespace.
- Validation costs one batched read (`read_flags`). The inputs are not mutated or retained.
- On NumPy the multiplier is read-only. On CuPy, MAPLE's `freeze` cannot enforce this.

`apply(value, out=None)` and `depth_m(schedule, t, dt, out=None)` multiply the host scalar into the multiplier inside the field's namespace. The optional `out` buffer is reused. No field is copied between host and device at any step.

Mask semantics: a masked cell receives zero rainfall. The mask is not a negative rate, and this module never reads or removes stored water.

The legacy code folds both roles into `rmask`:

| `rmask` value | Legacy effect |
|---|---|
| `< -9000` (`-9900` on the first `rain_type = 2` call) | cell gets 0 |
| `≥ 0` | multiplier |
| between those | `r2` left at its previous value (stale) |

The separate scale and Boolean mask here remove that ambiguity.

## Status

This is forcing only; it is ready to be consumed by a kernel. No infiltration, routing, event driver, GPU runtime or performance result is claimed. The CuPy test skips when no device is present.
