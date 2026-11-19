"""Bounded two-parameter (d, stress) search: global grid, per-d stress profile, boundary-edge optimisations and several local refinements, then the minimum over EVERY evaluated pair. The objective is any callable
f(d, stress) -> finite float (the calibration passes the pooled native-column score); nothing here knows the physics. Local refinements use the transformed coordinate z = log1p(d / D_SCALE), which resolves the
region near d = 0 (z = 0 is exactly d = 0), and scaled unit-box coordinates. SciPy's Nelder-Mead is used when importable and ONLY when it reports success; otherwise (or on failure) a deterministic bounded
cyclic coordinate descent is used and labelled. No dependency is added. A dense grid does not prove a global optimum: only bounded evidence (grid, multi-start, edge checks) is reported.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from benchmarks.subsurface_stress_calibration import search as SE

D_MIN, D_MAX = 0.0, 1.0
D_SCALE = 0.001                       # z = log1p(d / D_SCALE)
Z_MAX = math.log1p(D_MAX / D_SCALE)
STRESS_MIN, STRESS_MAX = SE.STRESS_MIN, SE.STRESS_MAX
BOUND_TOL = 1e-9
N_REFINE = 6                          # separated grid/edge candidates refined in 2D (old incumbents are additional)
SEPARATION = 0.08                     # min distance of refinement starts in the unit (z, stress) square


class JointSearchError(ValueError):
    pass


def z_of_d(d: float) -> float:
    return math.log1p(float(d) / D_SCALE)


def d_of_z(z: float) -> float:
    """Inverse of `z_of_d`, saturating exactly at the declared bounds (z <= 0 is d = 0; z >= Z_MAX is d = D_MAX), so a rounding overshoot of the exponential can never produce an out-of-range d."""
    z = float(z)
    if z <= 0.0:
        return 0.0
    if z >= Z_MAX:
        return D_MAX
    return min(D_SCALE * math.expm1(z), D_MAX)


def check_point(d: Any, s: Any, *, d_lo: float = D_MIN, d_hi: float = D_MAX, s_lo: float = STRESS_MIN, s_hi: float = STRESS_MAX) -> tuple:
    """Validate one (d, stress) BEFORE any model call; returns plain floats. Refuses bool/non-finite/out-of-range values (never clips)."""
    for name, v in (("d", d), ("stress", s)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)) or not math.isfinite(float(v)):
            raise JointSearchError(f"{name} must be a finite number, got {v!r}")
    d, s = float(d), float(s)
    if not (d_lo <= d <= d_hi):
        raise JointSearchError(f"d={d!r} outside the declared range [{d_lo}, {d_hi}]")
    if not (s_lo <= s <= s_hi):
        raise JointSearchError(f"stress={s!r} outside the declared range [{s_lo}, {s_hi}]")
    return d, s


def d_grid(n_log: int = 41, extra: Sequence[float] = (0.05,)) -> np.ndarray:
    """0 + log-spaced positive values 1e-6..1 (n_log points, exact endpoints) + the extra exact values; sorted, unique."""
    if isinstance(n_log, bool) or not isinstance(n_log, (int, np.integer)) or n_log < 3:
        raise JointSearchError(f"n_log must be an integer >= 3, got {n_log!r}")
    log = np.logspace(-6.0, 0.0, n_log)
    log[0], log[-1] = 1e-6, 1.0
    g = np.unique(np.concatenate([[0.0], log, np.asarray(extra, dtype=float)]))
    if g[0] != 0.0 or g[-1] != D_MAX or not np.all(np.isfinite(g)):
        raise JointSearchError("d grid must contain exactly 0 and 1")
    return g


def stress_grid(n: int = 81, extra: Sequence[float] = ()) -> np.ndarray:
    g = SE.dense_grid(n, STRESS_MIN, STRESS_MAX)
    g = np.unique(np.concatenate([g, np.asarray(extra, dtype=float)]))
    if g[0] != STRESS_MIN or g[-1] != STRESS_MAX:
        raise JointSearchError("extra stresses must lie inside the supported range")
    return g


class CachedObjective2D:
    """Exact-pair cache (a repeated pair is never re-run) and the ordered evaluation trace; every pair is validated BEFORE the objective runs and a non-finite value is refused."""

    def __init__(self, fn: Callable[[float, float], float]):
        self._fn, self.cache, self.order = fn, {}, []

    def __call__(self, d: float, s: float) -> float:
        key = check_point(d, s)
        if key not in self.cache:
            v = self._fn(*key)
            if isinstance(v, bool) or not math.isfinite(float(v)):
                raise JointSearchError(f"objective at {key!r} is not finite: {v!r}")
            self.cache[key] = float(v)
            self.order.append(key)
        return self.cache[key]


def best_of(cache: dict, pairs: Any = None) -> tuple:
    """((d, s), value): the minimum over the given pairs (default EVERY cached pair), exact ties to the smaller d then the smaller stress."""
    items = cache.items() if pairs is None else ((p, cache[p]) for p in pairs)
    return min(items, key=lambda kv: (kv[1], kv[0][0], kv[0][1]))


def grid_local_minima(y: np.ndarray) -> list:
    """Indices (i, j) of grid points with no strictly lower 8-neighbour (plateau members all qualify; selection thins them later)."""
    n, m = y.shape
    out = []
    for i in range(n):
        for j in range(m):
            lo = y[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2]
            if y[i, j] <= lo.min():
                out.append((i, j))
    return out


def _coordinate_descent(fn: Callable[[float, float], float], u0: tuple, sweeps: int = 12, tol: float = 1e-12) -> tuple:
    """Deterministic cyclic 1-D bounded refinement of fn(u, v) on [0, 1]^2 (golden section along each axis)."""
    u = [float(u0[0]), float(u0[1])]
    best = fn(*u)
    for _ in range(sweeps):
        before = best
        for ax in (0, 1):
            def line(t, ax=ax):
                w = list(u)
                w[ax] = t
                return fn(*w)
            t = SE.golden(line, 0.0, 1.0, 1e-10)
            if line(t) < best:
                u[ax], best = t, line(t)
        if before - best <= tol * max(abs(before), 1e-300):
            break
    return u[0], u[1], best


def refine_box(f: CachedObjective2D, z_box: tuple, s_box: tuple, start: tuple) -> dict:
    """Bounded 2-D minimisation of f(d_of_z(z), s) in the box z_box x s_box from `start` = (z, s). Unit-square coordinates (u, v); every trial point is built inside the box. The returned point is verified
    (finite value, inside the box) and never clipped."""
    (z0, z1), (s0, s1) = z_box, s_box
    if not (z0 <= start[0] <= z1 and s0 <= start[1] <= s1) or z1 < z0 or s1 < s0:
        raise JointSearchError("refinement start outside its box")

    def lerp(a, b, t):
        return a if t <= 0.0 else b if t >= 1.0 else min(max(a + t * (b - a), a), b)

    def fn(u, v):
        return f(d_of_z(lerp(z0, z1, u)), lerp(s0, s1, v))
    u0 = (0.0 if z1 == z0 else (start[0] - z0) / (z1 - z0), 0.0 if s1 == s0 else (start[1] - s0) / (s1 - s0))
    how, ok, ures = "coordinate_descent", False, None
    try:
        from scipy.optimize import minimize
    except ImportError:
        minimize = None
    if minimize is not None:
        r = minimize(lambda x: fn(float(x[0]), float(x[1])), np.array(u0, dtype=float), method="Nelder-Mead", bounds=[(0.0, 1.0), (0.0, 1.0)], options={"xatol": 1e-10, "fatol": 1e-18, "maxiter": 600})
        x = np.asarray(r.x, dtype=float)
        if bool(getattr(r, "success", False)) and np.all(np.isfinite(x)) and math.isfinite(float(r.fun)) and np.all((x >= 0.0) & (x <= 1.0)):
            how, ok, ures = "scipy_nelder_mead", True, (float(x[0]), float(x[1]))
        else:
            how = "coordinate_descent_after_scipy_failure"
    if ures is None:
        u, v, _ = _coordinate_descent(fn, u0)
        ures = (u, v)
    if not all(0.0 <= t <= 1.0 and math.isfinite(t) for t in ures):
        raise JointSearchError("refinement left its box")
    d, s = d_of_z(lerp(z0, z1, ures[0])), lerp(s0, s1, ures[1])
    return {"method": how, "scipy_success": ok, "d": d, "stress": s, "objective": f(d, s)}


def _refine_1d(g: Callable[[float], float], a: float, b: float) -> tuple:
    return SE.refine_bracket(g, a, b, 1e-10) if b > a else (a, "degenerate_bracket")


def near_optimal_region(d: Any, s: Any, mse: Any, fractions: Any = (0.01, 0.05), best: float | None = None) -> dict:
    """DESCRIPTIVE region of GRID points with MSE <= (1 + fraction) x `best` (the actual best evaluated value, e.g. the refined selected MSE; default the grid minimum): count, d and stress extents, number of
    4-connected components. A coarse grid may hold no point within the band: the region is then reported EMPTY (zero points, null extents), never a crash or an invented extent. Grid sampling only; NOT a confidence region."""
    y = np.asarray(mse, dtype=float)
    d, s = np.asarray(d, dtype=float), np.asarray(s, dtype=float)
    if y.shape != (d.size, s.size) or not (np.all(np.isfinite(d)) and np.all(np.isfinite(s)) and np.all(np.isfinite(y))) or np.any(np.diff(d) <= 0) or np.any(np.diff(s) <= 0):
        raise JointSearchError("need finite strictly increasing axes and a finite objective of shape (n_d, n_stress)")
    grid_min = float(y.min())
    best = grid_min if best is None else float(best)
    if not math.isfinite(best):
        raise JointSearchError("the reference best value must be finite")
    out = {"descriptive_only": "not a confidence or credible region; grid-sampled points only", "min_mse": best, "grid_min_mse": grid_min, "reference": "best evaluated value" if best != grid_min else "grid minimum", "fractions": {}}
    for fr in fractions:
        ok = y <= best * (1.0 + fr)
        seen, comps = np.zeros_like(ok), 0
        for i, j in zip(*np.nonzero(ok)):
            if seen[i, j]:
                continue
            comps += 1
            stack = [(i, j)]
            seen[i, j] = True
            while stack:
                a, b = stack.pop()
                for p, q in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                    if 0 <= p < ok.shape[0] and 0 <= q < ok.shape[1] and ok[p, q] and not seen[p, q]:
                        seen[p, q] = True
                        stack.append((p, q))
        ii, jj = np.nonzero(ok)
        if ii.size == 0:
            out["fractions"][f"{fr:g}"] = {"n_points": 0, "d_extent": None, "stress_extent": None, "n_components": 0, "empty": "no grid point lies within the band of the best evaluated value"}
            continue
        out["fractions"][f"{fr:g}"] = {"n_points": int(ok.sum()), "d_extent": [float(d[ii].min()), float(d[ii].max())], "stress_extent": [float(s[jj].min()), float(s[jj].max())], "n_components": comps, "empty": None}
    return out


def search_joint(objective: Callable[[float, float], float], d_axis: Any, s_axis: Any, incumbents: Sequence[tuple] = (), *, n_refine: int = N_REFINE) -> dict:
    """Grid -> per-d stress profile (bounded 1-D refinement at every grid d, including d = 0 and d = 1) -> stress-edge d refinements -> several separated 2-D refinements (+ every incumbent) -> the minimum over
    EVERY evaluated pair (exact ties: smaller d, then smaller stress). Incumbents must lie on the grids (their exact values are evaluated and admitted). Never claims an unbounded optimum."""
    f = objective if isinstance(objective, CachedObjective2D) else CachedObjective2D(objective)
    if isinstance(n_refine, bool) or not isinstance(n_refine, (int, np.integer)) or n_refine < 0:
        raise JointSearchError(f"n_refine must be a non-negative integer, got {n_refine!r}")
    xs, ss = np.asarray(d_axis, dtype=float), np.asarray(s_axis, dtype=float)
    if xs.ndim != 1 or ss.ndim != 1 or not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ss))):
        raise JointSearchError("axes must be one-dimensional and entirely finite")
    if xs.size < 3 or ss.size < 3 or np.any(np.diff(xs) <= 0) or np.any(np.diff(ss) <= 0):
        raise JointSearchError("axes need >= 3 strictly increasing points")
    if xs[0] != D_MIN or xs[-1] != D_MAX or ss[0] != STRESS_MIN or ss[-1] != STRESS_MAX:
        raise JointSearchError("axes must span the declared bounds exactly")
    for inc in incumbents:
        check_point(*inc)
    zs = np.array([z_of_d(x) for x in xs])
    zs[0], zs[-1] = 0.0, Z_MAX
    y = np.array([[f(float(d), float(s)) for s in ss] for d in xs])                       # 2-D grid (n_d, n_stress)
    n_grid_eval = len(f.cache)

    def ds_neighbours(i, lo, hi):
        return max(i - 1, 0), min(i + 1, hi - 1)

    profile = []                                                                           # best stress at every grid d, refined in a bracket of the neighbouring stress grid points
    for i, d in enumerate(xs):
        j = int(min(range(ss.size), key=lambda k: (y[i, k], ss[k])))
        a, b = ds_neighbours(j, 0, ss.size)
        x_ref, how = _refine_1d(lambda s, d=float(d): f(d, s), float(ss[a]), float(ss[b]))
        (_, s_best), v = min((((float(d), c), f(float(d), c)) for c in {float(ss[j]), float(x_ref)}), key=lambda kv: (kv[1], kv[0][1]))
        profile.append({"d": float(d), "stress": s_best, "objective": v, "grid_best_stress": float(ss[j]), "refine_method": how, "stress_at_lower_bound": s_best <= STRESS_MIN + BOUND_TOL, "stress_at_upper_bound": s_best >= STRESS_MAX - BOUND_TOL})
    edges = {}
    for name, s_edge in (("stress_lo", float(ss[0])), ("stress_hi", float(ss[-1]))):         # d optimised along each stress edge (z coordinate)
        i = int(min(range(xs.size), key=lambda k: (y[k, 0 if name == "stress_lo" else -1], xs[k])))
        a, b = ds_neighbours(i, 0, xs.size)
        z_ref, how = _refine_1d(lambda z, s=s_edge: f(d_of_z(z), s), float(zs[a]), float(zs[b]))
        cand = {float(d_of_z(z_ref)), float(xs[i])}
        d_best, v = min(((c, f(c, s_edge)) for c in cand), key=lambda kv: (kv[1], kv[0]))
        edges[name] = {"stress": s_edge, "d": d_best, "objective": v, "refine_method": how}
    # candidate starts: grid local minima, per-d profile minimisers, edge optima, old incumbents
    gl = [(float(xs[i]), float(ss[j])) for i, j in grid_local_minima(y)]
    cands = [(p, f.cache[p]) for p in gl]
    cands += [((p["d"], p["stress"]), p["objective"]) for p in profile]
    cands += [((e["d"], e["stress"]), e["objective"]) for e in edges.values()]
    cands = sorted({c[0]: c[1] for c in cands}.items(), key=lambda kv: (kv[1], kv[0][0], kv[0][1]))
    chosen = []
    for (d, s), v in cands:
        if len(chosen) >= n_refine:
            break
        u = (z_of_d(d) / Z_MAX, (s - STRESS_MIN) / (STRESS_MAX - STRESS_MIN))
        if all(math.hypot(u[0] - c[0], u[1] - c[1]) >= SEPARATION for c, _ in chosen):
            chosen.append((u, (d, s)))
    starts = [(p, "separated_candidate") for _, p in chosen] + [(tuple(map(float, inc)), "old_incumbent") for inc in incumbents]
    minima = []
    for (d0, s0), role in starts:
        if (d0, s0) not in f.cache:
            f(d0, s0)
        i = int(np.argmin(np.abs(xs - d0)))
        j = int(np.argmin(np.abs(ss - s0)))
        a, b = ds_neighbours(i, 0, xs.size)
        c, e = ds_neighbours(j, 0, ss.size)
        z0 = z_of_d(d0)                                                                      # the box is the neighbouring grid cells and only widens so that it always contains its own start
        z_box, s_box = (min(float(zs[a]), z0), max(float(zs[b]), z0)), (min(float(ss[c]), s0), max(float(ss[e]), s0))
        r = refine_box(f, z_box, s_box, (z0, s0))
        minima.append({"role": role, "start": [d0, s0], "z_box": list(z_box), "stress_box": list(s_box), **r, "d_at_lower_bound": r["d"] <= D_MIN + BOUND_TOL, "d_at_upper_bound": r["d"] >= D_MAX - BOUND_TOL,
                       "stress_at_lower_bound": r["stress"] <= STRESS_MIN + BOUND_TOL, "stress_at_upper_bound": r["stress"] >= STRESS_MAX - BOUND_TOL})
    for p in profile:                                                                       # the lowest of ALL cached stresses at this fixed d (a sampled point may beat the returned minimiser)
        cands_d = [(v, s) for (d, s), v in f.cache.items() if d == p["d"]]
        v, s = min(cands_d)
        p.update({"objective": v, "stress": s, "n_cached_stresses": len(cands_d), "stress_at_lower_bound": s <= STRESS_MIN + BOUND_TOL, "stress_at_upper_bound": s >= STRESS_MAX - BOUND_TOL})
    for name, e in edges.items():                                                          # likewise along each stress edge: the lowest of ALL cached d at that stress
        v, d = min((v, d) for (d, s), v in f.cache.items() if s == e["stress"])
        e.update({"d": d, "objective": v})
    (bd, bs), bv = best_of(f.cache)
    flags = {"d_at_lower_bound": bd <= D_MIN + BOUND_TOL, "d_at_upper_bound": bd >= D_MAX - BOUND_TOL, "stress_at_lower_bound": bs <= STRESS_MIN + BOUND_TOL, "stress_at_upper_bound": bs >= STRESS_MAX - BOUND_TOL}
    boundary = any(flags.values())
    sel = {"d": bd, "stress": bs, "objective": bv, **flags, "on_grid": bool(bd in set(xs.tolist()) and bs in set(ss.tolist())), "z": z_of_d(bd)}
    return {"d_axis": [float(x) for x in xs], "stress_axis": [float(x) for x in ss], "grid_objective": y.tolist(), "n_grid_evaluations": n_grid_eval, "n_evaluations": len(f.cache), "profile_over_d": profile, "stress_edges": edges,
            "local_minima": minima, "selected": sel, "boundary_solution": boundary, "requires_root_bounded_extension_decision": bool(flags["d_at_upper_bound"]),
            "near_optimal": near_optimal_region(xs, ss, y, best=bv), "trace": [[d, s, f.cache[(d, s)]] for d, s in f.order],
            "limits_note": f"bounded search over d in [{D_MIN}, {D_MAX}], stress in [{STRESS_MIN}, {STRESS_MAX}]; a dense grid plus multi-start refinement and edge checks is not a proof of the global mathematical optimum; "
                           "a solution at a bound means the optimum was not located inside it, no unbounded optimum is claimed and the bounds are not silently expanded; near-optimal regions are descriptive, not confidence regions"}
