"""Phase 3a rainfall forcing checks.

Expected values are hand-computed rectangle integrals (intensity in mm/h
times duration, / 3.6e6 for metres) or come from a test-only parse of the
actual Plot 1 file, never from the module's own arithmetic.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import os
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from maple_syrup.rainfall import (
    RainfallError,
    RainfallSchedule,
    constant_rainfall,
    parse_legacy_rainfall_file,
    parse_legacy_rainfall_text,
    rainfall_field,
)

MAHLERAN_ROOT = Path(os.environ.get("MAPLE_SYRUP_MAHLERAN_ROOT", "/home/okin/MAHLERAN"))
PLOT1_RAIN = MAHLERAN_ROOT / "Input" / "input_p1" / "p1_01_08_06.dat"


def mm(depth_mm: float) -> float:
    return depth_mm / 1000.0


@pytest.fixture(scope="module")
def plot1_path() -> Path:
    if not PLOT1_RAIN.is_file():
        pytest.skip(f"MAHLERAN Plot 1 rainfall file not available at {PLOT1_RAIN}")
    return PLOT1_RAIN


# Edges 0, 10, 25, 26, 100 s; 3.6, 0, 7.2, 36 mm/h = 1e-6, 0, 2e-6, 1e-5 m/s.
IRREGULAR = "00:00:00\n00:00:10 3.6\n00:00:25 0\n00:00:26 7.2\n00:01:40 36\n"
IRREGULAR_TOTAL_M = 10 * 1e-6 + 0 + 1 * 2e-6 + 74 * 1e-5


@pytest.fixture
def irregular() -> RainfallSchedule:
    return parse_legacy_rainfall_text(IRREGULAR)


# --- constant factory -------------------------------------------------------------
def test_constant_rectangle_integrals():
    s = constant_rainfall(100.0, 700.0, 36.0)  # 36 mm/h = 1e-5 m/s
    assert s.depth_m(0.0, 100.0) == 0.0
    assert s.depth_m(50.0, 100.0) == pytest.approx(50 * 1e-5, rel=1e-14)
    assert s.depth_m(200.0, 30.0) == pytest.approx(30 * 1e-5, rel=1e-14)
    assert s.depth_m(650.0, 100.0) == pytest.approx(50 * 1e-5, rel=1e-14)
    assert s.depth_m(700.0, 1000.0) == 0.0
    assert s.depth_m(0.0, 1e4) == pytest.approx(mm(36.0 * 600 / 3600), rel=1e-14)
    assert s.depth_m(300.0, 0.0) == 0.0
    assert s.rate_after_m_per_s(99.9) == 0.0
    assert s.rate_after_m_per_s(100.0) == pytest.approx(1e-5, rel=1e-15)
    assert s.rate_after_m_per_s(700.0) == 0.0
    assert s.provenance.kind == "constant"


@pytest.mark.parametrize(
    "args",
    [(10.0, 10.0, 1.0), (10.0, 5.0, 1.0), (-1.0, 5.0, 1.0), (0.0, 5.0, -1.0), (0.0, math.inf, 1.0),
     (0.0, 5.0, math.nan), (True, 5.0, 1.0), ("0", 5.0, 1.0)],
)
def test_constant_rejects_invalid(args):
    with pytest.raises(RainfallError):
        constant_rainfall(*args)


# --- legacy parser ----------------------------------------------------------------
def test_midnight_rollover_and_rectangles():
    s = parse_legacy_rainfall_text("23:58:00.00\n23:59:00.00 12.0\n00:00:30.00 36.0\n00:02:00.00 0.0\n")
    np.testing.assert_array_equal(s.edges_s, [0.0, 60.0, 150.0, 240.0])
    assert s.provenance.rollover_record == 2
    assert s.total_depth_m() == pytest.approx(mm(12 * 60 / 3600 + 36 * 90 / 3600), rel=1e-14)
    assert s.depth_m(30.0, 150.0) == pytest.approx(mm(12 * 30 / 3600 + 36 * 90 / 3600), rel=1e-14)
    assert s.depth_m(240.0, 1e5) == 0.0


def test_first_record_after_midnight_and_fractional_seconds():
    s = parse_legacy_rainfall_text("23:59:00\n00:01:00 60\n")
    np.testing.assert_array_equal(s.edges_s, [0.0, 120.0])
    assert s.provenance.rollover_record == 1
    f = parse_legacy_rainfall_text("18:00:00.25\n18:00:01.75 36\n")
    np.testing.assert_array_equal(f.edges_s, [0.0, 1.5])
    assert f.total_depth_m() == pytest.approx(1.5 * 1e-5, rel=1e-14)


def test_blanks_crlf_and_tabs_are_tolerated():
    s = parse_legacy_rainfall_text("\n  18:00:00.00  \r\n\r\n18:01:00.00\t 15.24 \r\n\n")
    np.testing.assert_array_equal(s.edges_s, [0.0, 60.0])
    np.testing.assert_array_equal(s.intensity_mm_per_h, [15.24])
    assert s.provenance.blank_lines_skipped == 3
    assert s.provenance.start_clock == "18:00:00.00"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "\n\n",
        "18:00:00.00\n",  # header only
        "18:00\n18:01:00 1.0\n",  # malformed header
        "18:00:00 5.0\n18:01:00 1.0\n",  # header carries an intensity
        "18:00:00\n18:01:00\n",  # missing intensity
        "18:00:00\n18:01:00 1.0 2.0\n",  # extra field
        "18:00:00\n18:01:00 abc\n",
        "18:00:00\n18:01:00 1,5\n",
        "18:00:00\n18:01:00 -1.0\n",
        "18:00:00\n18:01:00 nan\n",
        "18:00:00\n18:01:00 inf\n",
        "18:00:00\n18:01:00 1e999\n",  # overflows to inf
        "18:00:00\n24:00:00 1.0\n",
        "18:00:00\n18:60:00 1.0\n",
        "18:00:00\n18:01:60 1.0\n",
        "18:00:00\n8:01:00 1.0\n",  # one-digit hour
        "18:00:00\n18:00:00 1.0\n",  # zero-length first interval
        "18:00:00\n18:01:00 1\n18:01:00 2\n",  # duplicate
        "18:00:00\n18:05:00 1\n18:03:00 1\n",  # out of order, not a midnight crossing
        "08:00:00\n09:00:00 1\n07:00:00 1\n",  # backwards 2 h: 22 h unwrapped gap
        "23:00:00\n23:30:00 1\n00:30:00 1\n00:10:00 1\n",  # backwards after the crossing
        "12:00:00\n23:00:00 1\n02:00:00 1\n12:00:00 1\n",  # exactly 24 h span
        "12:00:00\n23:00:00 1\n02:00:00 1\n11:00:00 1\n13:00:00 1\n",  # > 24 h span
    ],
)
def test_malformed_text_is_rejected(text):
    with pytest.raises(RainfallError):
        parse_legacy_rainfall_text(text)


def test_non_ascii_file_is_rejected(tmp_path):
    path = tmp_path / "rain.dat"
    path.write_bytes("18:00:00\n18:01:00 1.0 é\n".encode())
    with pytest.raises(RainfallError):
        parse_legacy_rainfall_file(path)


@pytest.mark.parametrize(
    "edges, intensity",
    [([0.0, 10.0, 10.0], [1.0, 1.0]), ([0.0, 10.0, 5.0], [1.0, 1.0]), ([-1.0, 10.0], [1.0]),
     ([0.0, 10.0], [1.0, 2.0]), ([0.0], []), ([0.0, 10.0], [-1.0]), ([0.0, math.nan], [1.0]),
     ([0.0, 10.0], [math.inf]), ([[0.0, 10.0]], [1.0])],
)
def test_direct_construction_is_validated(edges, intensity):
    with pytest.raises(RainfallError):
        RainfallSchedule(edges_s=edges, intensity_mm_per_h=intensity,
                         provenance=constant_rainfall(0.0, 1.0, 0.0).provenance)


# --- evaluation -------------------------------------------------------------------
def test_irregular_steps_spanning_knots(irregular):
    s = irregular
    assert s.depth_m(5.0, 95.0) == pytest.approx(5 * 1e-6 + 0 + 1 * 2e-6 + 74 * 1e-5, rel=1e-14)
    assert s.depth_m(9.5, 17.0) == pytest.approx(0.5 * 1e-6 + 0 + 1 * 2e-6 + 0.5 * 1e-5, rel=1e-14)
    assert s.depth_m(12.0, 10.0) == 0.0  # inside the zero-intensity interval
    assert s.depth_m(25.25, 0.5) == pytest.approx(0.5 * 2e-6, rel=1e-14)
    t, total = 0.0, 0.0
    for dt in (0.3, 7.0, 0.0, 12.5, 5.2, 40.0, 50.0, 3.0):
        total += s.depth_m(t, dt)
        t += dt
    assert t > s.end_s
    assert total == pytest.approx(IRREGULAR_TOTAL_M, rel=1e-13)


def test_pieces_split_at_every_discontinuity(irregular):
    got = [(p.start_s, p.end_s, p.rate_m_per_s) for p in irregular.pieces(5.0, 95.0)]
    expected = [(5.0, 10.0, 1e-6), (10.0, 25.0, 0.0), (25.0, 26.0, 2e-6), (26.0, 100.0, 1e-5)]
    assert len(got) == len(expected)
    for (a0, a1, ar), (b0, b1, br) in zip(got, expected):
        assert (a0, a1) == (b0, b1)
        assert ar == pytest.approx(br, rel=1e-15)
    tail = irregular.pieces(95.0, 10.0)
    assert [(p.start_s, p.end_s) for p in tail] == [(95.0, 100.0), (100.0, 105.0)]
    assert tail[1].rate_m_per_s == 0.0
    before = constant_rainfall(50.0, 60.0, 3.6).pieces(0.0, 55.0)
    assert [(p.start_s, p.end_s) for p in before] == [(0.0, 50.0), (50.0, 55.0)]
    assert before[0].rate_m_per_s == 0.0
    assert irregular.pieces(3.0, 0.0) == ()
    assert irregular.depth_m(3.0, 0.0) == 0.0


def test_next_edge_progression(irregular):
    visited, t = [], 0.0
    while math.isfinite(t := irregular.next_edge_s(t)):
        visited.append(t)
    assert visited == [10.0, 25.0, 26.0, 100.0]
    assert irregular.next_edge_s(25.5) == 26.0
    assert irregular.next_edge_s(100.0) == math.inf
    late = constant_rainfall(30.0, 90.0, 1.0)
    assert late.next_edge_s(0.0) == 30.0
    assert late.next_edge_s(30.0) == 90.0


def test_additivity_under_random_partitions(irregular):
    rng = np.random.default_rng(20260929)
    whole = irregular.depth_m(3.0, 110.0)
    for _ in range(20):
        cuts = np.sort(rng.uniform(3.0, 113.0, size=rng.integers(1, 40)))
        points = [3.0, *cuts.tolist(), 113.0]
        parts = sum(irregular.depth_m(a, b - a) for a, b in pairwise(points))
        assert parts == pytest.approx(whole, rel=1e-12)


def test_evaluation_is_stateless(irregular):
    windows = [(0.0, 12.0), (24.0, 3.0), (90.0, 20.0), (1.0, 1.0)]
    forward = [irregular.depth_m(t, dt) for t, dt in windows]
    backward = [irregular.depth_m(t, dt) for t, dt in reversed(windows)][::-1]
    assert forward == backward


@pytest.mark.parametrize(
    "t, dt",
    [(math.nan, 1.0), (0.0, math.nan), (math.inf, 1.0), (0.0, math.inf), (0.0, -1.0), (-1.0, 1.0),
     (True, 1.0), ("0", 1.0), (None, 1.0), (1e308, 1e308)],
)
def test_invalid_time_arguments_are_rejected(irregular, t, dt):
    with pytest.raises(RainfallError):
        irregular.depth_m(t, dt)
    with pytest.raises(RainfallError):
        irregular.pieces(t, dt)


def test_schedule_is_immutable():
    edges, intensity = [0.0, 10.0], [3.6]
    s = RainfallSchedule(edges_s=edges, intensity_mm_per_h=intensity,
                         provenance=constant_rainfall(0.0, 1.0, 0.0).provenance)
    edges[1], intensity[0] = 1e6, 1e6
    assert s.total_depth_m() == pytest.approx(10 * 1e-6, rel=1e-14)
    for array in (s.edges_s, s.intensity_mm_per_h, s.rate_m_per_s):
        with pytest.raises(ValueError):
            array[0] = 5.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.edges_s = np.array([0.0, 1.0])


# --- actual Plot 1 file -------------------------------------------------------------
def _independent_plot1(path: Path) -> tuple[list[float], list[float]]:
    rows = [line.split() for line in path.read_text(encoding="ascii").splitlines() if line.strip()]

    def seconds(clock: str) -> float:
        h, m, s = clock.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    origin = seconds(rows[0][0])
    return [seconds(r[0]) - origin for r in rows[1:]], [float(r[1]) for r in rows[1:]]


def test_plot1_file_total_and_provenance(plot1_path):
    s = parse_legacy_rainfall_file(plot1_path)
    ends, intensities = _independent_plot1(plot1_path)
    starts = [0.0, *ends[:-1]]
    independent_total = sum(i * (b - a) / 3.6e6 for a, b, i in zip(starts, ends, intensities))
    assert s.total_depth_m() == pytest.approx(independent_total, rel=1e-13)
    # 27 one-minute records totalling 38 x 15.24 mm/h-minutes = 9.652 mm.
    assert s.total_depth_m() == pytest.approx(mm(9.652), rel=1e-12)
    assert s.depth_m(0.0, 1e5) == pytest.approx(mm(9.652), rel=1e-12)
    np.testing.assert_array_equal(s.edges_s, [0.0, *ends])
    assert s.end_s == 1620.0
    p = s.provenance
    assert p.kind == "legacy_file" and p.n_records == 27 and p.rollover_record is None
    assert p.start_clock == "18:00:00.00" and p.start_clock_s == 18 * 3600
    assert p.sha256 == hashlib.sha256(plot1_path.read_bytes()).hexdigest()
    assert p.size_bytes == plot1_path.stat().st_size


def test_plot1_switches_exactly_at_record_end_times(plot1_path):
    s = parse_legacy_rainfall_file(plot1_path)
    one_second_1 = mm(15.24 / 3600)  # first record: 15.24 mm/h until 60 s
    assert s.depth_m(59.0, 1.0) == pytest.approx(one_second_1, rel=1e-13)
    assert s.depth_m(60.0, 1.0) == 0.0  # second record: 0 mm/h on (60, 120]
    assert s.depth_m(120.0, 1.0) == pytest.approx(one_second_1, rel=1e-13)
    assert s.depth_m(56.0, 7.0) == pytest.approx(4 * one_second_1, rel=1e-13)
    assert s.depth_m(1620.0, 600.0) == 0.0  # nothing after the final record
    edges, t = [], 0.0
    while math.isfinite(t := s.next_edge_s(t)):
        edges.append(t)
    assert edges == [60.0 * k for k in range(1, 28)]


# --- spatial application -------------------------------------------------------------
SCALE = np.array([[1.0, 0.5, 2.0], [0.0, 1.5, 1.0]])
MASK = np.array([[True, False, True], [True, True, False]])


def test_field_applies_scale_and_mask():
    field = rainfall_field(2, 3, scale=SCALE, active_mask=MASK)
    got = field.apply(4.0)
    np.testing.assert_array_equal(got, [[4.0, 0.0, 8.0], [0.0, 6.0, 0.0]])
    assert field.shape == (2, 3) and got.dtype == np.float64
    uniform = rainfall_field(2, 3).apply(3.0)
    np.testing.assert_array_equal(uniform, np.full((2, 3), 3.0))
    masked_only = rainfall_field(2, 3, active_mask=MASK).apply(1.0)
    np.testing.assert_array_equal(masked_only, MASK.astype(np.float64))


def test_field_depth_from_schedule_and_out_reuse():
    field = rainfall_field(2, 3, scale=SCALE, active_mask=MASK)
    s = constant_rainfall(0.0, 600.0, 36.0)
    out = np.full((2, 3), -1.0)
    result = field.depth_m(s, 10.0, 60.0, out=out)
    assert result is out
    np.testing.assert_allclose(out, [[6e-4, 0.0, 1.2e-3], [0.0, 9e-4, 0.0]], rtol=1e-14, atol=0)
    np.testing.assert_array_equal(field.depth_m(s, 600.0, 60.0), np.zeros((2, 3)))


def test_field_inputs_are_not_mutated_or_retained():
    scale, mask = SCALE.copy(), MASK.copy()
    field = rainfall_field(2, 3, scale=scale, active_mask=mask)
    np.testing.assert_array_equal(scale, SCALE)
    np.testing.assert_array_equal(mask, MASK)
    scale[:] = 99.0
    mask[:] = True
    np.testing.assert_array_equal(field.apply(1.0), np.where(MASK, SCALE, 0.0))
    with pytest.raises(ValueError):
        field.multiplier[0, 0] = 5.0


@pytest.mark.parametrize(
    "ny, nx, kwargs",
    [
        (3, 2, {"scale": SCALE}),  # transposed shape
        (2, 3, {"scale": SCALE[0]}),  # 1-D
        (2, 3, {"scale": SCALE[..., None]}),  # 3-D
        (2, 3, {"scale": SCALE.astype(np.float32)}),
        (2, 3, {"scale": SCALE.tolist()}),  # not an array
        (2, 3, {"scale": np.where(MASK, SCALE, -1.0)}),  # negative, even in a masked cell
        (2, 3, {"scale": np.where(MASK, SCALE, np.nan)}),
        (2, 3, {"scale": np.where(MASK, SCALE, np.inf)}),
        (2, 3, {"active_mask": MASK.astype(np.int8)}),  # mask must be Boolean
        (2, 3, {"active_mask": MASK.T}),
        (0, 3, {}),
        (True, 3, {}),
        (2.0, 3, {}),
        (2, 3, {"scale": SCALE, "xp": math}),  # xp disagrees with the arrays
        (2, 3, {"xp": math}),
    ],
)
def test_field_misuse_is_rejected(ny, nx, kwargs):
    with pytest.raises(RainfallError):
        rainfall_field(ny, nx, **kwargs)


@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf, True, "1", None])
def test_field_apply_rejects_invalid_values(value):
    with pytest.raises(RainfallError):
        rainfall_field(2, 3).apply(value)


@pytest.mark.parametrize("out", [np.empty((3, 2)), np.empty((2, 3), dtype=np.float32), [[0.0] * 3] * 2])
def test_field_apply_rejects_bad_out(out):
    with pytest.raises(RainfallError):
        rainfall_field(2, 3).apply(1.0, out=out)


def test_cupy_field_stays_device_resident():
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; GPU path not exercised (no GPU claim)")
    cp = backend.cupy_module()
    scale, mask = backend.to_device(SCALE, cp), backend.to_device(MASK, cp)
    field = rainfall_field(2, 3, scale=scale, active_mask=mask)
    s = constant_rainfall(0.0, 600.0, 36.0)
    out = cp.empty((2, 3), dtype=cp.float64)
    before = backend.read_transfer_counters()
    for k in range(10):
        result = field.depth_m(s, 30.0 * k, 30.0, out=out)
    delta = backend.read_transfer_counters().delta(before)
    assert backend.is_device_array(result)
    assert (delta.host_to_device, delta.device_to_host, delta.scalar_reads) == (0, 0, 0)
    host = rainfall_field(2, 3, scale=SCALE, active_mask=MASK).depth_m(s, 270.0, 30.0)
    np.testing.assert_array_equal(backend.to_host(result), host)
