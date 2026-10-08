"""B3: the fused versus separate water accounting of the CUDA legacy replay. CPU-only source/validation contracts, real-GPU bitwise equivalence
and guard tests (queued for Codex on a verified-idle device), and the actual CLI in both modes. Written without being run."""
from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from maple_syrup import legacy_gpu_driver as G
from maple_syrup import legacy_water_cuda as W

from .helpers import gpu_available
from .test_cli_gpu import APPLIED, PLOT1, needs_plot1, run_cli

needs_gpu = pytest.mark.skipif(not gpu_available(), reason="no CuPy / CUDA device")
ROOT = Path(__file__).resolve().parents[2]
SHAPE, STEPS = (3, 4), 6


# ---- CPU-only contracts -----------------------------------------------------------------------------------------------------
def test_modes_defaults_and_module_provenance():
    assert W.MODES == ("separate", "fused") and W.OPS_PER_STEP == {"separate": 7, "fused": 1}
    args = G.build_parser().parse_args(["--case-kind", "rfid", "--case", "c", "--output", "o"])
    assert args.water_accounting == "fused"  # provisional default until the root's real-GPU gates
    assert G.build_parser().parse_args(["--case-kind", "rfid", "--case", "c", "--output", "o", "--water-accounting", "separate"]
                                       ).water_accounting == "separate"
    with pytest.raises(SystemExit):
        G.build_parser().parse_args(["--case-kind", "rfid", "--case", "c", "--output", "o", "--water-accounting", "bogus"])
    assert "maple_syrup.legacy_water_cuda" in G._GPU_MODULES and "maple_syrup.legacy_water_cuda" in G.module_digests()


def test_kernel_source_reads_the_packet_words_of_the_hydrology_and_does_no_fast_math():
    from maple_syrup import hydrology_cuda as hc

    src = W.kernel_source()
    assert f"#define W_OUTLET {hc._P_OUTLET}" in src and f"#define W_EXPORT {hc._P_EXPORT}" in src
    assert f"#define W_RESIDUAL {hc._P_RESIDUAL}" in src and "atomic" not in src and "fma" not in src.lower()
    prov = W.kernel_provenance()
    assert prov["fastmath"] is False and "--fmad=false" in prov["compile_options"] and len(prov["source_sha256"]) == 64
    assert prov["packet_words"]["total"] == hc.PACKET_WORDS


@pytest.mark.parametrize("args", [("bogus", SHAPE, STEPS), ("fused", (3,), STEPS), ("fused", (3, 0), STEPS), ("fused", (3, True), STEPS),
                                  ("fused", [3, 4], STEPS), ("fused", SHAPE, 0), ("fused", SHAPE, True), ("fused", SHAPE, 2.0),
                                  ("fused", (2**16, 2**16), STEPS)])
def test_constructor_arguments_are_validated_before_the_device_is_touched(args):
    mode, shape, steps = args
    with pytest.raises(W.WaterAccountingError):
        W.WaterAccounting(object(), shape, steps, mode)  # a bare object: any use of CuPy would raise AttributeError instead


def test_the_driver_loop_uses_the_accountant_and_no_inline_water_operations():
    source = inspect.getsource(G._loop)
    assert "acct.account(row, step, packet)" in source and "cp.add(" not in source and "water[row" not in source
    assert "WaterAccounting(cp, (ny, nx), steps, args.water_accounting)" in source
    assert source.index("WaterAccounting(cp") < source.index("t_loop0 = time.perf_counter()")  # constructed (compiled) before the timer
    execute = inspect.getsource(G._execute)
    assert "water_accounting_setup_s" in execute and "water_accounting_compile_cached" in execute
    assert "cuda_step_with_packet" in source and "[0]  #" not in source  # the packet is kept for the fused kernel
    execute = inspect.getsource(G._execute)
    assert '"water_accounting": r["water_accounting"]' in execute and "launches_scope" in execute
    # the sediment launch counter is not redefined: it still comes from the sediment context only
    assert 'ctx.stats["launches"]' in execute and "acct" not in execute


# ---- real-GPU tests ----------------------------------------------------------------------------------------------------------
def fake_step(cp, rng, shape=SHAPE, packet=None):
    """A synthetic hydrology step: a uint64[18] packet whose doubles at the real words are the scalars (0-d views, like the hydrology's
    own), and four source grids."""
    from maple_syrup import hydrology_cuda as hc

    packet = cp.zeros(hc.PACKET_WORDS, dtype=np.uint64) if packet is None else packet
    f = packet.view(np.float64)
    f[hc._P_OUTLET], f[hc._P_EXPORT], f[hc._P_RESIDUAL] = (float(v) for v in rng.uniform(-1, 1, 3) * np.array([1e-3, 1e-6, 1e-12]))
    route = SimpleNamespace(outlet_discharge_m3_s=f[hc._P_OUTLET], export_m3=f[hc._P_EXPORT], budget_residual_m3=f[hc._P_RESIDUAL])
    grids = [cp.asarray(rng.uniform(0.0, 1e-4, shape) * (1.0 + 1e-13 * rng.standard_normal(shape))) for _ in range(4)]
    col = SimpleNamespace(rain_m=grids[0], intake_m=grids[1], saturation_return_m=grids[2], drainage_m=grids[3])
    return SimpleNamespace(route=route, column=col), packet


def make(cp, mode, steps=STEPS):
    acct = W.WaterAccounting(cp, SHAPE, steps, mode)
    return acct


def seed_state(acct, rng, cp):
    """Nonzero initial cumulative maps and a sentinel in every series row (written in place: the sealed metadata is unchanged)."""
    for a in acct.cum.values():
        a[:] = cp.asarray(rng.uniform(0.0, 1.0, a.shape))
    acct.water.fill(7.25)


def snapshot(acct):
    return {k: a.copy() for k, a in acct._arrays().items()}


def same(a, b):
    return all(a[k].tobytes() == b[k].tobytes() and a[k].shape == b[k].shape for k in a)


@needs_gpu
def test_fused_equals_separate_bitwise_over_several_steps_with_nonzero_initial_state_and_sentinels():
    import cupy as cp

    sep, fus = make(cp, "separate"), make(cp, "fused")
    for acct in (sep, fus):
        seed_state(acct, np.random.default_rng(5), cp)
    host_cum = {k: cp.asnumpy(a).copy() for k, a in sep.cum.items()}
    rng = np.random.default_rng(6)
    for row in range(4):  # only 4 of the 6 rows are executed: rows 4 and 5 must keep their sentinel
        step, packet = fake_step(cp, rng)
        before = packet.copy()
        sep.account(row, step, packet)
        fus.account(row, step, packet)
        assert bytes(cp.asnumpy(packet)) == bytes(cp.asnumpy(before))  # the hydrology packet is never modified
        for key, src in zip(W.CUM_KEYS, (step.column.rain_m, step.column.intake_m, step.column.saturation_return_m,
                                          step.column.drainage_m), strict=True):
            host_cum[key] = host_cum[key] + cp.asnumpy(src)  # the independent host addition, one per step and cell
        assert same(snapshot(sep), snapshot(fus)), row
    for key in W.CUM_KEYS:
        assert np.array_equal(cp.asnumpy(fus.cum[key]), host_cum[key]), key
    water = cp.asnumpy(fus.water)
    assert (water[4:] == 7.25).all() and not (water[:4] == 7.25).any()  # unexecuted rows keep their sentinel
    assert fus.stats == {"launches": 4, "steps": 4} and sep.stats == {"launches": 28, "steps": 4}
    s = fus.summary()
    assert s["device_operations_per_step"] == 1 and s["host_reads_per_step"] == 0 and s["host_to_device_copies_per_step"] == 0
    assert s["kernel"]["fastmath"] is False and sep.summary()["kernel"] is None and "SEDIMENT" in s["scope"]


@needs_gpu
def test_series_columns_are_outlet_export_residual_in_that_order_and_dtype_is_float64():
    import cupy as cp

    from maple_syrup import hydrology_cuda as hc

    for mode in W.MODES:
        acct = make(cp, mode)
        step, packet = fake_step(cp, np.random.default_rng(2))
        acct.account(0, step, packet)
        f = cp.asnumpy(packet.view(np.float64))
        row = cp.asnumpy(acct.water)[0]
        assert row.tolist() == [f[hc._P_OUTLET], f[hc._P_EXPORT], f[hc._P_RESIDUAL]] and acct.water.dtype == np.float64, mode
        assert acct.water.shape == (STEPS, 3) and all(a.dtype == np.float64 and a.shape == SHAPE for a in acct.cum.values())


@needs_gpu
@pytest.mark.parametrize("mode", W.MODES)
def test_reset_restores_a_clean_state_and_the_repeat_is_bitwise_identical(mode):
    import cupy as cp

    acct = make(cp, mode)

    def run():
        rng = np.random.default_rng(8)
        for row in range(STEPS):
            step, packet = fake_step(cp, rng)
            acct.account(row, step, packet)
        return snapshot(acct)

    first = run()
    warm = dict(acct.stats)
    acct.reset()
    assert acct.stats == {"launches": 0, "steps": 0} and acct.stats_before_reset == warm and acct.row == 0
    assert not cp.asnumpy(acct.water).any() and not any(cp.asnumpy(a).any() for a in acct.cum.values())
    assert same(first, run())


@needs_gpu
@pytest.mark.parametrize("mode", W.MODES)
def test_invalid_use_is_refused_before_any_write_launch_or_row_change(mode):
    import cupy as cp

    acct = make(cp, mode)
    rng = np.random.default_rng(3)
    seed_state(acct, rng, cp)
    step, packet = fake_step(cp, rng)
    _other_step, other_packet = fake_step(cp, rng)
    before, stats = snapshot(acct), dict(acct.stats)
    cases = []
    for bad_row in (1, -1, STEPS, True, 0.0, "0", None):
        cases.append((bad_row, step, packet))
    cases.append((0, step, other_packet))  # the packet is not the one the step's scalars alias
    cases.append((0, step, packet.astype(np.int64)))
    cases.append((0, step, cp.zeros(17, dtype=np.uint64)))
    cases.append((0, step, np.zeros(18, dtype=np.uint64)))  # a host packet
    host_scalars = SimpleNamespace(route=SimpleNamespace(outlet_discharge_m3_s=1.0, export_m3=2.0, budget_residual_m3=3.0), column=step.column)
    cases.append((0, host_scalars, packet))  # host scalars would imply a materialisation
    for field in ("rain_m", "intake_m", "saturation_return_m", "drainage_m"):
        for bad in (step.column.__dict__[field].astype(np.float32), cp.zeros((4, 3)), cp.zeros((3, 8))[:, ::2],
                    np.zeros(SHAPE), acct.cum["rain"], acct.cum["drainage"]):  # dtype, shape, non-contiguous, host, overlaps a map
            col = SimpleNamespace(**{**step.column.__dict__, field: bad})
            cases.append((0, SimpleNamespace(route=step.route, column=col), packet))
    cases.append((0, SimpleNamespace(route=step.route), packet))
    for row, s, p in cases:
        with pytest.raises(W.WaterAccountingError):
            acct.account(row, s, p)
        assert acct.row == 0 and acct.poisoned is None and acct.stats == stats and same(snapshot(acct), before)
    acct.account(0, step, packet)  # the same context still works after every refusal
    assert acct.row == 1


def malformed_scalars(cp, packet, word):
    """Impostors of the packet word `word` that share (or fake) the CORRECT address but are not a 0-d float64 cupy view."""
    base = int(packet.data.ptr) + 8 * word
    return {
        "uint64_view": packet[word],  # a 0-d uint64 view at the right address
        "float32_view": packet.view(np.float32)[2 * word],  # a 0-d float32 view at the right address
        "vector_view": packet.view(np.float64)[word:word + 1],  # shape (1,)
        "two_word_view": packet.view(np.float64)[word:word + 2],
        "host_numpy_0d": np.array(1.0),
        "host_float": 1.0,
        "pointer_impostor": SimpleNamespace(data=SimpleNamespace(ptr=base), dtype=np.float64, shape=()),
        "copy_not_a_view": packet.view(np.float64)[word].copy(),  # a float64 0-d device array at the WRONG address
    }


@needs_gpu
@pytest.mark.parametrize("mode", W.MODES)
def test_route_scalars_must_be_exact_0d_float64_views_of_the_packet_word_before_any_write(mode):
    import cupy as cp

    from maple_syrup import hydrology_cuda as hc

    acct = make(cp, mode)
    rng = np.random.default_rng(12)
    seed_state(acct, rng, cp)
    step, packet = fake_step(cp, rng)
    before, stats = snapshot(acct), dict(acct.stats)
    packet_before = bytes(cp.asnumpy(packet))
    for field, word in (("outlet_discharge_m3_s", hc._P_OUTLET), ("export_m3", hc._P_EXPORT), ("budget_residual_m3", hc._P_RESIDUAL)):
        for kind, bad in malformed_scalars(cp, packet, word).items():
            route = SimpleNamespace(**{**step.route.__dict__, field: bad})
            with pytest.raises(W.WaterAccountingError):
                acct.account(0, SimpleNamespace(route=route, column=step.column), packet)
            assert acct.row == 0 and acct.poisoned is None and acct.stats == stats and same(snapshot(acct), before), (field, kind)
    assert bytes(cp.asnumpy(packet)) == packet_before
    acct.account(0, step, packet)  # the genuine 0-d float64 views still pass
    assert acct.row == 1


@needs_gpu
@pytest.mark.parametrize("mode", W.MODES)
def test_a_missing_step_column_or_route_field_is_a_typed_refusal_and_the_state_stays_usable(mode):
    import cupy as cp

    acct = make(cp, mode)
    rng = np.random.default_rng(13)
    seed_state(acct, rng, cp)
    step, packet = fake_step(cp, rng)
    before, stats = snapshot(acct), dict(acct.stats)
    broken = [SimpleNamespace(route=step.route), SimpleNamespace(column=step.column), SimpleNamespace(), object()]
    for field in ("rain_m", "intake_m", "saturation_return_m", "drainage_m"):
        broken.append(SimpleNamespace(route=step.route, column=SimpleNamespace(**{k: v for k, v in step.column.__dict__.items() if k != field})))
    for field in ("outlet_discharge_m3_s", "export_m3", "budget_residual_m3"):
        route = SimpleNamespace(**{k: v for k, v in step.route.__dict__.items() if k != field})
        broken.append(SimpleNamespace(route=route, column=step.column))
    for bad in broken:
        with pytest.raises(W.WaterAccountingError, match="lacks|required"):
            acct.account(0, bad, packet)
        assert acct.row == 0 and acct.poisoned is None and acct.stats == stats and same(snapshot(acct), before)
    acct.account(0, step, packet)
    assert acct.row == 1


@needs_gpu
def test_setup_and_compile_costs_are_recorded_outside_the_loop_and_the_second_context_hits_the_cache():
    import cupy as cp

    first, second = make(cp, "fused"), make(cp, "fused")
    sep = make(cp, "separate")
    for acct in (first, second, sep):
        s = acct.summary()
        assert s["setup_s"] >= 0.0 and s["compile_s"] >= 0.0 and "setup_note" in s and acct.stats == {"launches": 0, "steps": 0}
    assert second.compile_cached is True and sep.compile_cached is None and sep.compile_s == 0.0 and sep.fn is None
    assert second.setup_s >= second.compile_s


@needs_gpu
def test_a_source_overlapping_the_series_is_refused():
    import cupy as cp

    acct = make(cp, "fused")
    step, packet = fake_step(cp, np.random.default_rng(1))
    big = acct.water.reshape(-1)[:12].reshape(SHAPE)  # a view of the series row memory used as a source
    col = SimpleNamespace(**{**step.column.__dict__, "rain_m": big})
    with pytest.raises(W.WaterAccountingError, match="overlap"):
        acct.account(0, SimpleNamespace(route=step.route, column=col), packet)


@needs_gpu
def test_the_sealed_structure_and_the_poison_contract():
    import cupy as cp

    acct = make(cp, "fused")
    step, packet = fake_step(cp, np.random.default_rng(4))
    for attr, value in (("steps", 99), ("mode", "separate"), ("n", 5)):
        old = getattr(acct, attr)
        setattr(acct, attr, value)
        with pytest.raises(W.WaterAccountingError, match="sealed"):
            acct.account(0, step, packet)
        setattr(acct, attr, old)
    original = acct.cum["rain"]
    acct.cum["rain"] = cp.zeros_like(original)
    with pytest.raises(W.WaterAccountingError, match="cum_rain"):
        acct.account(0, step, packet)
    with pytest.raises(W.WaterAccountingError, match="cum_rain"):
        acct.reset()
    acct.cum["rain"] = original
    assert acct.row == 0 and acct.stats["launches"] == 0
    real = acct.fn

    def broken(*a, **k):
        raise RuntimeError("simulated launch failure")

    acct.fn = broken
    with pytest.raises(RuntimeError):
        acct.account(0, step, packet)
    assert acct.poisoned and acct.row == 0
    with pytest.raises(W.WaterAccountingError, match="poisoned"):
        acct.account(0, step, packet)
    acct.fn = real
    acct.reset()
    assert acct.poisoned is None
    acct.account(0, step, packet)


@needs_gpu
def test_a_context_is_refused_on_another_device_before_any_write():
    import cupy as cp

    acct = make(cp, "fused")
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("a second CUDA device is not visible")
    step, packet = fake_step(cp, np.random.default_rng(1))
    before = snapshot(acct)
    with cp.cuda.Device(1), pytest.raises(W.WaterAccountingError, match="device"):
        acct.account(0, step, packet)
    assert same(snapshot(acct), before) and acct.stats["launches"] == 0


# ---- the actual CLI, both modes ------------------------------------------------------------------------------------------------
def load_harness():
    spec = importlib.util.spec_from_file_location("compare_cpu_gpu_b3", ROOT / "benchmarks" / "legacy_gpu" / "compare_cpu_gpu.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["compare_cpu_gpu_b3"] = module
    spec.loader.exec_module(module)
    return module


@needs_gpu
@needs_plot1
@pytest.mark.parametrize("depth_mode, solver", [("previous", "bisection"), ("post_infiltration", "bisection"), ("previous", "newton")])
def test_actual_cli_fused_and_separate_agree_bitwise_and_both_match_the_cpu_within_the_declared_bounds(tmp_path, depth_mode, solver):
    extra = ["--depth-time-level", depth_mode, "--root-solver", solver]
    cpu = run_cli("maple_syrup.legacy_driver", tmp_path, "cpu", *extra)
    sep = run_cli("maple_syrup.legacy_gpu_driver", tmp_path, "sep", *extra, "--water-accounting", "separate")
    fus = run_cli("maple_syrup.legacy_gpu_driver", tmp_path, "fus", *extra, "--water-accounting", "fused")
    for proc in (cpu, sep, fus):
        assert proc.returncode == 0, proc.stderr[-2000:]
    s, f = np.load(tmp_path / "sep" / "legacy_ledger.npz"), np.load(tmp_path / "fus" / "legacy_ledger.npz")
    assert set(s.files) == set(f.files)
    for key in s.files:  # the accounting mode must not change a single bit of any saved array (water series, maps, sediment)
        assert s[key].tobytes() == f[key].tobytes() and s[key].shape == f[key].shape, key
    ss = json.loads((tmp_path / "sep" / "legacy_summary.json").read_text())["gpu"]
    fs = json.loads((tmp_path / "fus" / "legacy_summary.json").read_text())["gpu"]
    assert ss["water_accounting"]["mode"] == "separate" and ss["water_accounting"]["device_operations_per_step"] == 7
    assert fs["water_accounting"]["mode"] == "fused" and fs["water_accounting"]["device_operations_per_step"] == 1
    assert fs["water_accounting"]["kernel"]["source_sha256"] and fs["water_accounting"]["launches"] == 40
    assert ss["launches_per_step_sediment"] == fs["launches_per_step_sediment"] and "SEDIMENT" in fs["launches_scope"]
    assert fs["transfers"]["hydrology_packet_bytes_per_step"] == 144
    harness = load_harness()
    for name in ("sep", "fus"):
        report = harness.compare(tmp_path / "cpu", tmp_path / name)
        assert report["accepted_by_declared_bounds"], (name, report["flags"])
    assert APPLIED.is_file() and PLOT1.is_dir()
