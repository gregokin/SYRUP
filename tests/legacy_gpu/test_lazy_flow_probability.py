"""P2: the LAZY flow-detachment probability of `sg_laws` in the PRODUCTION `maple_syrup.legacy_native_cuda` against the frozen B3C1 module.

The production module (`ROOT/src/maple_syrup/legacy_native_cuda.py`, imported as `maple_syrup.legacy_native_cuda`, the module every driver and
test uses) and an independently loaded copy of the frozen B3C1 snapshot (`outputs/dependencies/syrup_gpu_sediment_gpu_b3c1`, each with its own
kernel cache `_MODULES`) build contexts on the same unchanged package helpers (`legacy_native`, `legacy_physics_numba`, ...). The production
module computes the Gaussian probability `p` only when it can reach an output or an error word; every result (laws, regimes, velocities, flags,
ledger rows, maps, tallies) must equal the B3C1 baseline BITWISE, including the exact error words. The tests inject values into the CONTENT
of device arrays of test contexts only (`d_cls`, `d_sc`, `d_slope`, `d_fractions`): the context seals (pointers, shapes, dtypes) are not
bypassed, and a replacement is still refused. The GPU tests need the frozen B3C1 snapshot (git-ignored; absent on a clean checkout, hence the
skip) and a device; the pin and range tests run anywhere and always exercise the production file. The archived P2 candidate package
(`outputs/dependencies/syrup_gpu_sediment_flowprob_candidate`) is used ONLY to qualify its own archive receipt; a test on that archive
alone says nothing about production, so every production claim below reads `CAND_FILE`. Written without being run."""
from __future__ import annotations

import difflib
import functools
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from maple_syrup import legacy_native as N

from .helpers import GRAPHS, fraction_pattern, gpu_available, physics_for

ROOT = Path(__file__).resolve().parents[2]
DEPS = ROOT / "outputs" / "dependencies"
BASE_DIR, ARCHIVE_DIR = DEPS / "syrup_gpu_sediment_gpu_b3c1", DEPS / "syrup_gpu_sediment_flowprob_candidate"
REL = "src/maple_syrup/legacy_native_cuda.py"
BASE_FILE, ARCHIVE_FILE = BASE_DIR / REL, ARCHIVE_DIR / REL
CAND_FILE = ROOT / "src" / "maple_syrup" / "legacy_native_cuda.py"  # the PRODUCTION module, the adopted P2 source
BASE_PINNED_SHA256 = "11f219a98b680a5ef32fd3a57089f578d33c797063d9399c182eab217b75701c"  # the B3C1 snapshot receipt value (pre-adoption root)
CAND_PINNED_SHA256 = "584db1b7e4c8cf10b808433dc793a37dc7e3083d4e4ae06d2bc011bc7a12bbe9"  # the accepted P2 candidate bytes (root-verified pin)
FRACTIONS6 = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.2])
NC = 6
needs_baseline = pytest.mark.skipif(not BASE_FILE.is_file(), reason="the frozen B3C1 snapshot (git-ignored) is absent")
needs_archive = pytest.mark.skipif(not (BASE_FILE.is_file() and ARCHIVE_FILE.is_file()),
                                   reason="the frozen B3C1 snapshot or the archived P2 candidate package (git-ignored) is absent")
needs_gpu = pytest.mark.skipif(not gpu_available() or importlib.util.find_spec("numba") is None, reason="no CuPy / CUDA device or no Numba")
BIT_THETA, BIT_P_NONFINITE = 1 << 19, 1 << 31  # the error-word bits of a non-finite theta / non-finite p
LAZY_GUARD = "if (flow_cell || !SG_FINITE(theta) || !SG_FINITE(p_par) || !(p_par < 0.0)) {"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # the dataclass machinery looks the module up
    spec.loader.exec_module(module)
    return module


@functools.lru_cache(maxsize=1)
def modules():
    """(baseline, production): the independently loaded frozen B3C1 file and the IMPORTED production module, with the source pins asserted."""
    assert sha256(BASE_FILE) == BASE_PINNED_SHA256, "the baseline file is not the pinned B3C1 legacy_native_cuda.py"
    import maple_syrup.legacy_native_cuda as production

    assert Path(production.__file__).resolve() == CAND_FILE.resolve(), f"the imported module is not the root file: {production.__file__}"
    assert sha256(CAND_FILE) == CAND_PINNED_SHA256, "the production legacy_native_cuda.py is not the pinned accepted P2 candidate"
    return _load("legacy_native_cuda_b3c1_baseline", BASE_FILE), production


# ---- CPU: the production file, no ignored dependency ------------------------------------------------------------------------------------
def test_the_production_module_is_the_root_file_and_matches_the_pinned_accepted_p2_candidate():
    import maple_syrup.legacy_native_cuda as production

    assert Path(production.__file__).resolve() == CAND_FILE.resolve()
    assert CAND_FILE.is_file() and sha256(CAND_FILE) == CAND_PINNED_SHA256
    assert sha256(CAND_FILE) != BASE_PINNED_SHA256  # it is no longer the B3C1 root
    text = CAND_FILE.read_text()
    assert text.count(LAZY_GUARD) == 1 and "double p = 0.0;" in text  # the lazy block is present exactly once
    assert production.kernel_source(6).count(LAZY_GUARD) == 1  # and reaches the generated kernel source
    assert production.kernel_provenance(6)["fastmath"] is False and production.kernel_provenance(6)["compile_options"]  # unchanged flags path


# ---- CPU: source qualification (frozen B3C1 baseline and the archived candidate package) -------------------------------------------------
@needs_archive
def test_the_baseline_is_the_pinned_b3c1_file_and_the_archived_candidate_package_differs_only_in_the_cuda_file():
    """Archive receipt qualification of the frozen packages (git-ignored). This does NOT exercise production by itself; the last assertion ties
    the archived candidate bytes to the root file so the archive's full-storm evidence applies to the adopted source."""
    base_receipt = json.loads((BASE_DIR / "snapshot_receipt.json").read_text())["files"]
    cand_receipt = json.loads((ARCHIVE_DIR / "snapshot_receipt.json").read_text())["files"]
    assert base_receipt[REL] == BASE_PINNED_SHA256 == sha256(BASE_FILE)
    assert cand_receipt[REL] == BASE_PINNED_SHA256  # the receipt records the pre-edit state of the candidate copy
    assert len(cand_receipt) == 46
    for rel, digest in cand_receipt.items():
        if rel != REL:  # every other file of the candidate package is byte-identical to its recorded digest
            assert sha256(ARCHIVE_DIR / rel) == digest, rel
    assert sha256(ARCHIVE_FILE) == CAND_PINNED_SHA256 != BASE_PINNED_SHA256  # the one intended edit exists in the archive
    assert sha256(CAND_FILE) == sha256(ARCHIVE_FILE) == CAND_PINNED_SHA256  # the ROOT file is byte-identical to the accepted archive


@needs_baseline
def test_the_production_diff_from_b3c1_is_confined_to_the_probability_block_of_sg_laws():
    base, cand = BASE_FILE.read_text().splitlines(), CAND_FILE.read_text().splitlines()
    first = next(i for i, line in enumerate(base) if "const bool pos = theta > 0.0;" in line)
    last = next(i for i, line in enumerate(base) if "double fd = ((p * hz) * f) / ref;" in line)
    assert first < last
    ops = [op for op in difflib.SequenceMatcher(a=base, b=cand, autojunk=False).get_opcodes() if op[0] != "equal"]
    assert ops, "no difference between the production file and the baseline"
    for tag, i1, i2, j1, j2 in ops:
        assert first <= i1 and i2 <= last + 1, f"a {tag} edit outside the probability block: baseline lines {i1 + 1}-{i2}"
    def norm(line: str) -> str:
        return line.strip().removeprefix("double ")  # `double p = ...` becomes the assignment `p = ...` inside the guard

    removed = [norm(line) for tag, i1, i2, _, _ in ops for line in base[i1:i2] if tag in ("delete", "replace")]
    kept = {norm(line) for line in cand}
    assert removed and all(line in kept for line in removed)  # every original line of the block survives (re-indented inside the guard)


def p_block(theta: float, p_par: float) -> float:
    """The probability block in float64 NumPy (same expression order as the kernel); an arithmetic sanity model, NOT the CUDA math library."""
    f = np.float64
    with np.errstate(all="ignore"):
        pos = theta > 0.0
        arg = f(theta) if pos else f(1.0)
        pc = np.log(f(0.049) / (arg * f(0.25)))
        t = pc / f(0.702)
        inner = f(1.0) - np.exp(f(p_par) * (t * t))
        sgn = 1.0 if pc > 0 else (-1.0 if pc < 0 else (0.0 if pc == 0 else np.nan))
        clamped = np.nan if np.isnan(inner) else max(inner, 0.0)
        p = f(0.5) - (f(0.5) * f(sgn)) * np.sqrt(f(clamped))
    return 0.0 if not pos else float(p)


THETAS = [-3.0, -0.0, 0.0, 5e-324, 1e-320, 1e-310, 2.2250738585072014e-308, 1e-300, 1e-100, 1e-10, 1e-3, 0.196, 4 * 0.049, 1.0, 1e3, 1e10,
          1e100, 1e300, 1.7976931348623157e308]
NEGATIVE_FINITE_P_PAR = [-2.0 / np.pi, -1e-300, -1e-10, -1.0, -1e10, -1e300, -1.7976931348623157e308]


def test_for_finite_theta_and_finite_strictly_negative_p_par_the_probability_is_finite_and_within_unit_range():
    for theta in THETAS:
        for p_par in NEGATIVE_FINITE_P_PAR:
            p = p_block(theta, p_par)
            assert np.isfinite(p) and 0.0 <= p <= 1.0, (theta, p_par, p)  # none of the flags 31 / 42 / 43 can be raised


def test_the_fallback_conditions_are_needed_other_p_par_values_do_produce_invalid_probabilities():
    assert np.isnan(p_block(1e-320, 0.0))  # p_par == 0 with t*t == +inf: 0 * inf
    assert np.isnan(p_block(4 * 0.049, -np.inf))  # t == 0 exactly: (-inf) * 0
    assert np.isnan(p_block(1.0, np.nan)) and np.isnan(p_block(1e-320, np.nan))
    assert np.isfinite(p_block(1.0, -np.inf)) and 0.0 <= p_block(1.0, -np.inf) <= 1.0  # finite here, but the fallback keeps -inf on the original path
    assert p_block(0.0, 0.0) == 0.0 and p_block(-1.0, np.nan) == 0.0  # theta <= 0 forces p = 0 whatever p_par is


# ---- GPU: frozen B3C1 baseline vs the PRODUCTION module ---------------------------------------------------------------------------------
MODERATE = [(0.0, 0.0, 0.0), (0.0, 0.0, 2e-5), (1e-3, 0.1, 2e-5), (1e-3, 0.1, 0.0), (5e-3, 0.2, 2e-5), (5e-3, 0.2, 0.0), (1e-2, 0.5, 2e-5)]
WITH_SUSPENSION = [*MODERATE, (0.5, 3.0, 0.0)]
NON_FLOW_ONLY = [(1e-3, 0.1, 2e-5), (1e-3, 0.1, 0.0), (1e-12, 0.1, 2e-5), (0.0, 0.0, 0.0)]
THETA_STRESS = [(1e-12, 0.1, 2e-5), (1e-12, 0.1, 0.0), (1e-3, 0.1, 2e-5), (5e-3, 0.2, 2e-5), (1e-2, 0.5, 2e-5), (0.0, 0.0, 0.0)]


def input_states(net, shape, n_steps, categories, seed=1):
    """Per-step device inputs whose cells rotate through `categories` (depth m, velocity m/s, rain m/s) so that every cell meets every regime;
    inactive cells are dry and still, cells without a receiver (zero slope, pits) have zero velocity (a validated invariant)."""
    rng = np.random.default_rng(seed)
    table = np.array(categories, dtype=np.float64)
    n = net.active.size
    for step in range(n_steps):
        picks = table[(np.arange(n) + step) % len(table)] * (1.0 + 0.1 * rng.random((n, 1)))
        depth, vel, rain = picks[:, 0].copy(), picks[:, 1].copy(), picks[:, 2].copy()
        depth[~net.active], rain[~net.active] = 0.0, 0.0
        vel[~net.active | net.slope_zero] = 0.0
        yield depth.reshape(shape), vel.reshape(shape), rain.reshape(shape)


def build_pair(graph_name, fractions=None, n_steps=8):
    import cupy as cp

    graph = GRAPHS[graph_name]()
    net = N.native_network(graph)
    physics = physics_for(graph, FRACTIONS6 if fractions is None else fractions)
    contexts = [mod.CudaLegacyContext(net, physics, graph, limits=N.walk_limits(net.dx_m), dt=1.0, n_steps=n_steps) for mod in modules()]
    return cp, net, tuple(graph.shape), contexts


def snapshot(cp, ctx, rows):
    """Everything the step wrote, as host bytes (device flags are read directly so error words can be compared without poisoning)."""
    nc = ctx.nc
    out = {name: cp.asnumpy(getattr(ctx, name)).copy() for name in ("det", "rate", "law", "regime", "depos", "clip", "v_prev", "M1", "Q1", "Qin1",
                                                                    "cum_det", "cum_dep", "cum_clip", "values", "code")}
    out["ledger"] = cp.asnumpy(ctx.ledger[:rows * 13 * nc]).copy()
    out["flags"] = cp.asnumpy(ctx.flags[:rows]).copy()
    out["counts"] = cp.asnumpy(ctx.counts[:rows * ctx.host_counts.shape[1]]).copy()
    return out


def differing(a, b):
    return [k for k in sorted(a) if a[k].shape != b[k].shape or a[k].dtype != b[k].dtype or a[k].tobytes() != b[k].tobytes()]


def set_cls3(value, k=2):
    def apply(ctx, cp):
        ctx.d_cls.reshape(-1)[3 * ctx.nc + k] = value  # row 3 of the class constants: theta = ustar^2 / row3
    return apply


def set_p_par(value):
    def apply(ctx, cp):
        ctx.d_sc[8] = value
    return apply


def seed_absent(classes):
    """Seeded mobile pools (active cells) and recession velocities (active, non-terminal cells: a terminal pit may not carry a velocity) of the
    given classes, written in place."""
    def apply(ctx, cp):
        n = ctx.n
        active = ctx.d_active != 0
        moving = active & (ctx.d_terminal == 0)
        for j, k in enumerate(classes):
            ctx.M1.reshape(n, ctx.nc)[:, k] = cp.where(active, 0.25 + 0.05 * j, 0.0)
            ctx.v_prev.reshape(n, ctx.nc)[:, k] = cp.where(moving, 0.02 + 0.005 * j, 0.0)
    return apply


def compare(graph_name, categories, *, n_steps=6, fractions=None, mutations=(), seed=1):
    """Run both contexts on identical inputs and return (equal-differences, baseline snapshot, production snapshot)."""
    cp, net, shape, contexts = build_pair(graph_name, fractions, n_steps)
    snaps = []
    for ctx in contexts:
        for mutate in mutations:
            mutate(ctx, cp)
        ctx._guard("test content injection")  # the seals still hold: only array CONTENT was changed
        for row, (depth, vel, rain) in enumerate(input_states(net, shape, n_steps, categories, seed)):
            ctx.step(row, cp.asarray(depth), cp.asarray(vel), cp.asarray(rain))
        cp.cuda.Stream.null.synchronize()
        snaps.append(snapshot(cp, ctx, n_steps))
    return differing(*snaps), snaps[0], snaps[1]


@needs_baseline
@needs_gpu
def test_baseline_and_production_load_independently_with_separate_kernel_caches_and_different_sources():
    base, cand = modules()
    assert base is not cand and base._MODULES is not cand._MODULES
    assert cand.__name__ == "maple_syrup.legacy_native_cuda" and Path(cand.__file__).resolve() == CAND_FILE.resolve()
    assert base.kernel_source(6) != cand.kernel_source(6) and base.kernel_provenance(6)["source_sha256"] != cand.kernel_provenance(6)["source_sha256"]
    cp, _, _, contexts = build_pair("converging", n_steps=1)
    assert all(any(key[1:] == (6, 6) or key[1:] == (6, 2) for key in mod._MODULES) for mod in (base, cand))  # each compiled its own variant
    with pytest.raises(base.CudaLegacyError, match="d_sc"):  # the seal of the baseline AND the production module still refuses a replaced array
        contexts[0].d_sc = cp.zeros_like(contexts[0].d_sc)
        contexts[0]._guard("replaced")
    with pytest.raises(cand.CudaLegacyError, match="d_sc"):
        contexts[1].d_sc = cp.zeros_like(contexts[1].d_sc)
        contexts[1]._guard("replaced")


@needs_baseline
@needs_gpu
def test_every_wet_dry_rain_and_flow_regime_gives_bitwise_equal_results_and_no_error_word():
    seen = set()
    for name in sorted(GRAPHS):
        diff, base, cand = compare(name, MODERATE, n_steps=len(MODERATE))
        assert diff == [], (name, diff)
        assert not base["flags"].any() and not cand["flags"].any(), name  # valid inputs: no error word in either module
        seen |= set(np.unique(base["regime"]).tolist())
    # regimes accumulated over the final steps only; run a union over all steps via a second pass with a shifted rotation
    for name in ("converging", "channel_to_ring"):
        for shift in range(1, 3):
            diff, base, _ = compare(name, MODERATE, n_steps=len(MODERATE), seed=shift)
            assert diff == []
            seen |= set(np.unique(base["regime"]).tolist())
    assert {0, 1, 2, 3, 4, 5} <= seen, f"not every wet regime was exercised: {sorted(seen)}"  # dry, wet no law, rain/diffuse, rain+trans, trans, concentrated


@needs_baseline
@needs_gpu
def test_suspension_cells_are_covered_and_equal():
    seen = set()
    for name in sorted(GRAPHS):
        diff, base, _ = compare(name, WITH_SUSPENSION, n_steps=len(WITH_SUSPENSION))
        assert diff == [], (name, diff)
        seen |= set(np.unique(base["regime"]).tolist())
    assert 6 in seen, "no suspended cell was produced; the category (0.5 m, 3 m/s) did not exceed the class settling limits"


@needs_baseline
@needs_gpu
@pytest.mark.parametrize("name", ["converging", "terminal_pit", "inactive_neighbour"])
def test_zero_fraction_classes_and_seeded_absent_grain_pools_and_velocities_are_unchanged(name):
    shape = tuple(GRAPHS[name]().shape)
    frac = fraction_pattern("two", shape)  # classes 0, 1, 2, 5 have no composition anywhere
    diff, base, cand = compare(name, MODERATE, n_steps=len(MODERATE), fractions=frac, mutations=(seed_absent([0, 1, 2, 5]),))
    assert diff == [], (name, diff)
    assert not base["flags"].any() and not cand["flags"].any()
    n = base["det"].size // NC
    assert not base["det"].reshape(n, NC)[:, [0, 1, 2, 5]].any()  # zero composition: no detachment, but the seeded pools evolve (M1 compared above)


THETA_CASES = {"theta_zero_by_infinite_divisor": float("inf"), "theta_subnormal": 1e308, "theta_huge": 1e-300, "theta_default": None,
               "theta_infinite_by_zero_divisor": 0.0, "theta_nan": float("nan")}


@needs_baseline
@needs_gpu
@pytest.mark.parametrize("case", sorted(THETA_CASES))
def test_theta_zero_subnormal_finite_huge_and_invalid_give_bitwise_equal_results_and_identical_error_words(case):
    value = THETA_CASES[case]
    mutations = () if value is None else (set_cls3(value),)
    for name in ("converging", "terminal_pit"):
        diff, base, cand = compare(name, THETA_STRESS, n_steps=len(THETA_STRESS), mutations=mutations)
        assert diff == [], (case, name, diff)
        invalid = case in ("theta_infinite_by_zero_divisor", "theta_nan")
        assert bool((base["flags"] & BIT_THETA).any()) == invalid and bool((cand["flags"] & BIT_THETA).any()) == invalid, (case, name)


P_PAR_CASES = {"zero": 0.0, "positive": 1.0, "nan": float("nan"), "plus_inf": float("inf"), "minus_inf": float("-inf"), "tiny_negative": -1e-300,
               "huge_negative": -1e300, "default": None}


@needs_baseline
@needs_gpu
@pytest.mark.parametrize("case", sorted(P_PAR_CASES))
def test_p_par_zero_positive_nan_infinite_and_extreme_values_keep_every_error_word_for_non_flow_cells(case):
    value = P_PAR_CASES[case]
    mutations = () if value is None else (set_p_par(value),)
    diff, base, cand = compare("converging", NON_FLOW_ONLY, n_steps=len(NON_FLOW_ONLY), mutations=mutations)
    assert diff == [], (case, diff)
    if case == "nan":  # only wet NON-FLOW cells here: the original raises flag 31 for them, so the production module must take the original block
        assert (base["flags"] & BIT_P_NONFINITE).any() and (cand["flags"] & BIT_P_NONFINITE).any()


@needs_baseline
@needs_gpu
@pytest.mark.parametrize("name", ["converging", "terminal_pit"])
def test_p_par_zero_with_a_subnormal_theta_still_raises_the_non_finite_probability_word_on_non_flow_cells(name):
    """0 * inf: with theta subnormal, t*t overflows to +inf and p_par == 0 gives NaN. Only the fallback path reproduces the baseline error word."""
    diff, base, cand = compare(name, NON_FLOW_ONLY, n_steps=len(NON_FLOW_ONLY), mutations=(set_cls3(1e308), set_p_par(0.0)))
    assert diff == [], diff
    assert (base["flags"] & BIT_P_NONFINITE).any() and (cand["flags"] & BIT_P_NONFINITE).any()


@needs_baseline
@needs_gpu
def test_the_production_module_is_bitwise_repeatable_after_reset_and_equal_to_the_baseline_across_the_reset():
    cp, net, shape, contexts = build_pair("converging", n_steps=len(MODERATE))
    results = []
    for ctx in contexts:
        runs = []
        for _ in range(2):
            ctx.reset()
            for row, (depth, vel, rain) in enumerate(input_states(net, shape, len(MODERATE), MODERATE, seed=7)):
                ctx.step(row, cp.asarray(depth), cp.asarray(vel), cp.asarray(rain))
            ctx.check_flags()
            runs.append((ctx.host_ledger.copy(), ctx.host_counts.copy(), ctx.download_maps()))
        assert runs[0][0].tobytes() == runs[1][0].tobytes() and runs[0][1].tobytes() == runs[1][1].tobytes()
        assert all(runs[0][2][k].tobytes() == runs[1][2][k].tobytes() for k in runs[0][2])
        results.append(runs[1])
    assert results[0][0].tobytes() == results[1][0].tobytes() and results[0][1].tobytes() == results[1][1].tobytes()
    assert all(results[0][2][k].tobytes() == results[1][2][k].tobytes() for k in results[0][2])


@needs_baseline
@needs_gpu
def test_slope_and_fraction_content_injection_extremes_stay_equal():
    """Declared test injection into static content: a vanishing and a huge slope (theta -> 0 / large) and an absent grain in every cell."""
    def slopes(scale):
        def apply(ctx, cp):
            ctx.d_slope[:] = ctx.d_slope * scale
        return apply

    def no_fraction(ctx, cp):
        ctx.d_fractions.reshape(ctx.n, ctx.nc)[:, 4] = 0.0

    for mutate in (slopes(1e-300), slopes(1e300), no_fraction):
        diff, _, _ = compare("converging", THETA_STRESS, n_steps=len(THETA_STRESS), mutations=(mutate,))
        assert diff == []
