"""CUDA (CuPy) tests of the array paths of maple_syrup.weather_pet. Every test uses the `gpu` fixture and skips
honestly without CuPy/a device; none of them was run by the author (GPU launches are Codex's call after checking
the allocation). The scalar astronomy is host-side and identical on both backends; what is checked here is that the
elementwise terrain/apply/depth/integration paths produce genuine CuPy arrays equal to the NumPy reference within
an explicit ULP bound, that public accessors hand out detached copies on the device (CuPy has no read-only mode),
and that the owned payload is never reached by a public write. Naive datetimes are the module's declared calendar
contract (DTZ001 silenced).

Equivalence bound: the factor is table[code] * cos(slope). CUDA's and the host libm's `cos` are each correctly
rounded to within 1 ULP but need not round identically, and the product adds one more rounding, so device and
host may differ by up to 2 ULP of the result (Codex's GPU 0 run observed exactly 1 ULP on 1 of 91 cells). The bound
is therefore 2 ULP of the strictly positive reference value (`np.spacing`), not a relative physical tolerance; no
MAPLE water-budget tolerance is involved. SHA identity equality is asserted only when the bytes are identical."""
# ruff: noqa: DTZ001
from __future__ import annotations

import dataclasses
import datetime as dt
import math
from fractions import Fraction

import numpy as np
import pytest

from maple_syrup import weather_pet as wp

STEEP = 0.2  # rad, above the 5 deg source flat threshold
ULPS = 2  # cos (1 ULP) followed by one multiply (1 ULP)


def host(a):
    return np.asarray(a.get()) if hasattr(a, "get") else np.asarray(a)


def assert_within_ulps(got, ref, ulps: int = ULPS, scale: float = 1.0) -> None:
    """|got - ref| <= ulps * spacing(|ref|) elementwise, on strictly positive references (scaled products inherit
    the same ULP count because scaling by a finite scalar is one correctly rounded multiply applied to both)."""
    got, ref = np.asarray(got, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    assert got.shape == ref.shape
    assert np.all(ref > 0.0) and np.all(np.isfinite(got))
    tol = ulps * np.spacing(np.abs(ref)) * (1.0 if scale == 1.0 else 2.0)
    diff = np.abs(got - ref)
    assert np.all(diff <= tol), f"max {np.max(diff / np.spacing(np.abs(ref))):.3f} ULP exceeds {ulps}"


@pytest.mark.parametrize("lat", [37.0, -37.0])
@pytest.mark.parametrize("threshold", [5.0, None])
def test_cuda_terrain_factor_matches_numpy_within_2_ulp_and_shares_identity(gpu, lat, threshold):
    rng = np.random.default_rng(11)
    codes = rng.integers(0, 9, size=(7, 13)).astype(np.int8)
    slope = rng.uniform(0.0, 1.4, size=(7, 13))  # includes gentle (< 5 deg) and steep slopes
    ref = wp.terrain_factor(codes, slope, lat, backend="numpy", flat_threshold_deg=threshold)
    d_codes, d_slope = gpu.asarray(codes), gpu.asarray(slope)
    dev = wp.terrain_factor(d_codes, d_slope, lat, backend="cuda", flat_threshold_deg=threshold)
    assert dev.backend == "cuda" and dev.shape == ref.shape and dev.cell_count == ref.cell_count
    assert dev.hemisphere_table == ref.hemisphere_table and dev.flat_policy == ref.flat_policy
    assert dev.flat_cell_count == ref.flat_cell_count
    values = dev.values
    assert type(values) is gpu.ndarray and values.dtype == gpu.float64
    assert_within_ulps(host(values), ref.values)
    assert math.isfinite(dev.max_value) and abs(dev.max_value - ref.max_value) <= ULPS * np.spacing(ref.max_value)
    if host(values).tobytes() == ref.values.tobytes():
        assert dev.sha256 == ref.sha256  # identical bytes -> identical identity across backends
    else:
        assert dev.sha256 != ref.sha256  # a 1-ULP difference is a different payload and must say so
    assert dev.verify_identity()
    # caller arrays untouched on device
    assert host(d_codes).tobytes() == codes.tobytes() and host(d_slope).tobytes() == slope.tobytes()


def test_cuda_public_accessors_are_detached_copies(gpu):
    codes = gpu.asarray(np.array([0, 1, 5], dtype=np.int8))
    slope = gpu.asarray(np.array([0.0, STEEP, 0.3]))
    tf = wp.terrain_factor(codes, slope, 45.0, backend="cuda")
    expected = host(tf.values)
    a, b = tf.values, tf.values
    assert a.data.ptr != b.data.ptr and a.data.ptr != tf._payload._internal().data.ptr  # copies, not the owned storage
    a[:] = -10.0  # writing the copy (CuPy has no read-only arrays) ...
    a *= 0.5
    assert host(tf.values).tobytes() == expected.tobytes() and tf.verify_identity()  # ... never reaches the payload
    c = tf.detached_values()
    c[:] = float("nan")
    assert host(tf.values).tobytes() == expected.tobytes()
    d = wp.potential_demand(wp.DailyWeather(dt.date(2023, 7, 15), 14.8, 20.7, 26.6), 45.0)
    t = wp.apply_terrain(d, tf)
    cells = t.pet_mm_day_cells
    assert type(cells) is gpu.ndarray and cells.data.ptr != t._cells._internal().data.ptr
    cells[:] = -10.0
    assert host(t.pet_mm_day_cells).tobytes() == host(tf._payload._internal() * d.pet_mm_day).tobytes()
    r = wp.integrate_interevent_pet_chunks([(t.date, 43200)], {t.date: t}, "interevent")
    assert host(r.depth_mm).tobytes() == host((tf._payload._internal() * d.pet_mm_day) * 0.5).tobytes()
    assert bool(gpu.all(r.depth_mm >= 0.0))


def test_cuda_refuses_numpy_arrays_and_vice_versa(gpu):
    codes = np.zeros(3, dtype=np.int8)
    slope = np.zeros(3)
    with pytest.raises(wp.WeatherValidationError):
        wp.terrain_factor(codes, slope, 10.0, backend="cuda")
    with pytest.raises(wp.WeatherValidationError):
        wp.terrain_factor(gpu.asarray(codes), gpu.asarray(slope), 10.0, backend="numpy")


def test_cuda_slope_and_code_validation(gpu):
    codes = gpu.zeros(3, dtype=gpu.int8)
    with pytest.raises(wp.WeatherValidationError):
        wp.terrain_factor(codes, gpu.asarray([0.0, -0.1, 0.0]), 10.0, backend="cuda")
    with pytest.raises(wp.WeatherValidationError):
        wp.terrain_factor(gpu.asarray([0, 9, 0], dtype=gpu.int8), gpu.zeros(3), 10.0, backend="cuda")
    with pytest.raises(wp.WeatherValidationError):
        wp.terrain_factor(codes, gpu.zeros(3, dtype=gpu.float32), 10.0, backend="cuda")


def test_cuda_source_flat_classification(gpu):
    one_deg = math.radians(1.0)
    codes = gpu.asarray(wp.aspect_codes_from_labels(["N", "S", "S"]))
    slope = gpu.asarray(np.array([one_deg, 0.0, STEEP]))
    default = host(wp.terrain_factor(codes, slope, 37.0, backend="cuda").values)
    assert_within_ulps(default, np.array([math.cos(one_deg), 1.0, 1.10 * math.cos(STEEP)]))
    assert default[1] == 1.0  # table value times cos(0) is exact on any backend
    caller = host(wp.terrain_factor(codes, slope, 37.0, backend="cuda", flat_threshold_deg=None).values)
    assert_within_ulps(caller, np.array([0.90 * math.cos(one_deg), 1.10, 1.10 * math.cos(STEEP)]))


def test_cuda_apply_terrain_and_depth_stay_on_device(gpu):
    rec = wp.DailyWeather(dt.date(2023, 7, 15), 14.8, 20.7, 26.6)
    d = wp.potential_demand(rec, 45.0)
    codes = gpu.asarray(np.array([0, 1, 5], dtype=np.int8))
    slope = gpu.asarray(np.array([0.0, STEEP, 0.3]))
    tf = wp.terrain_factor(codes, slope, 45.0, backend="cuda")
    t = wp.apply_terrain(d, tf)
    assert t.terrain is tf and t.terrain_applied
    ref = d.pet_mm_day * wp.terrain_factor(host(codes), host(slope), 45.0).values
    assert_within_ulps(host(t.pet_mm_day_cells), ref, scale=d.pet_mm_day)
    depth = wp.pet_depth_mm(t, 43200)
    assert type(depth) is gpu.ndarray
    assert_within_ulps(host(depth), 0.5 * ref, scale=d.pet_mm_day)
    with pytest.raises(wp.WeatherValidationError):
        wp.apply_terrain(t, tf)  # no double adjustment on device either
    with pytest.raises(wp.WeatherValidationError):
        wp.apply_terrain(d, tf.values)  # raw device array refused
    with pytest.raises(TypeError):
        dataclasses.replace(t, _cells=tf.values)  # not a dataclass: no foreign payload, no flipped flag
    with pytest.raises(AttributeError):
        t.pet_mm_day = -1.0  # type: ignore[misc]
    with pytest.raises(wp.WeatherValidationError):
        wp.TerrainFactor(object(), tf.values, 45.0, "north_literal", tf.sha256, "x", 5.0, 0, 1.0)  # type: ignore[arg-type]
    assert t.provenance()["terrain"]["backend"] == "cuda" and t.provenance()["terrain"]["sha256"] == tf.sha256
    # overflow guards hold on device too: O(1) product check, one end-of-integration reduction
    too_big = wp.potential_demand(wp.DailyWeather.explicit_pet(dt.date(2023, 7, 15), 20.0, 1.72e308), None, "explicit")
    with pytest.raises(wp.WeatherValidationError, match="would overflow"):
        wp.apply_terrain(too_big, tf)
    # scalar reference sum stays finite (2 x 0.86e308) while the south cell's sum overflows
    mid = wp.apply_terrain(wp.potential_demand(wp.DailyWeather.explicit_pet(dt.date(2023, 7, 15), 20.0, 0.86e308), None, "explicit"), tf)
    with pytest.raises(wp.WeatherValidationError, match="per-cell PET overflowed"):
        wp.integrate_interevent_pet_chunks([(mid.date, 86400), (mid.date, 86400)], {mid.date: mid}, "interevent")


def test_cuda_interevent_integration_sums_device_arrays_bounded(gpu):
    tf = wp.terrain_factor(gpu.asarray(np.array([0, 5], dtype=np.int8)), gpu.asarray(np.array([0.0, STEEP])), 30.0, backend="cuda")
    south = 1.10 * math.cos(STEEP)

    def demand(day, pet):
        rec = wp.DailyWeather(dt.date(2023, 7, day), 10.0, 15.0, 20.0, pet_mm_day=pet)
        return wp.apply_terrain(wp.potential_demand(rec, 30.0, method="explicit"), tf)

    records = {dt.date(2023, 7, 1): demand(1, 4.8), dt.date(2023, 7, 2): demand(2, 2.4)}
    r = wp.integrate_interevent_pet(dt.datetime(2023, 7, 1, 12), dt.datetime(2023, 7, 2, 12), records, "interevent")
    assert type(r.depth_mm) is gpu.ndarray and r.terrain_sha256_first == tf.sha256 and r.terrain_identity_changes == 0
    assert r.pieces == () and r.chunk_count == 2 and r.terrain_applied
    np.testing.assert_allclose(host(r.depth_mm), np.array([3.6, south * 3.6]), rtol=1e-14)
    # long stream: constant resident device arrays, no per-chunk history
    pool = gpu.get_default_memory_pool()
    n_cells, n_chunks = 50_000, 200
    big = wp.terrain_factor(gpu.zeros(n_cells, dtype=gpu.int8), gpu.zeros(n_cells), 30.0, backend="cuda")

    def stream(date):
        return wp.apply_terrain(wp.potential_demand(wp.DailyWeather.explicit_pet(date, 15.0, 2.4), None, "explicit"), big)

    def chunks():
        day = dt.date(2023, 1, 1)
        for _ in range(n_chunks):
            yield day, Fraction(43200)
            day += dt.timedelta(days=1)

    gpu.cuda.Stream.null.synchronize()
    used_before = pool.used_bytes()
    r2 = wp.integrate_interevent_pet_chunks(chunks(), stream, "interevent")
    gpu.cuda.Stream.null.synchronize()
    used_after = pool.used_bytes()
    assert used_after - used_before < 6 * n_cells * 8  # result + transient, not n_chunks arrays
    assert r2.chunk_count == n_chunks and r2.pieces == ()
    np.testing.assert_allclose(host(r2.depth_mm), np.full(n_cells, 1.2 * n_chunks), rtol=1e-12)
    # numpy terrain mixed into a cuda integration is refused (backend mismatch), not broadcast
    tf_np = wp.terrain_factor(np.array([0, 5], dtype=np.int8), np.array([0.0, STEEP]), 30.0)
    mixed = {dt.date(2023, 7, 1): demand(1, 4.8),
             dt.date(2023, 7, 2): wp.apply_terrain(wp.potential_demand(wp.DailyWeather(dt.date(2023, 7, 2), 10.0, 15.0, 20.0, pet_mm_day=2.4), 30.0, "explicit"), tf_np)}
    with pytest.raises(wp.WeatherValidationError, match="backend"):
        wp.integrate_interevent_pet(dt.datetime(2023, 7, 1, 12), dt.datetime(2023, 7, 2, 12), mixed, "interevent")
