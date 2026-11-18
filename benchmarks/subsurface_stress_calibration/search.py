"""One-dimensional bounded search of the evaporation-stress threshold: dense global grid, every local minimum refined, global comparison, descriptive near-optimum intervals. The objective is any callable
(the calibration passes the pooled native-column score); nothing here knows the physics. scipy's bounded scalar minimiser is used when importable, otherwise a pure golden-section search; no dependency is added.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

STRESS_MIN = 0.010001          # native column: stress must EXCEED the layer-0 residual 0.01
STRESS_MAX = 1.0               # the driver's accepted range
N_GRID = 199                   # linspace spacing ~0.005 including both endpoints
BOUND_TOL = 1e-9
GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


class SearchError(ValueError):
    pass


class CachedObjective:
    """Exact-float cache (a repeated stress is never re-run) with the ordered evaluation record."""

    def __init__(self, fn: Callable[[float], float]):
        self._fn, self.cache, self.order = fn, {}, []

    def __call__(self, x: float) -> float:
        x = float(x)
        if x not in self.cache:
            v = float(self._fn(x))
            if not math.isfinite(v):
                raise SearchError(f"objective at {x!r} is not finite: {v!r}")
            self.cache[x] = v
            self.order.append(x)
        return self.cache[x]


def dense_grid(n: int = N_GRID, lo: float = STRESS_MIN, hi: float = STRESS_MAX) -> np.ndarray:
    if n < 3 or not (lo < hi):
        raise SearchError("grid needs n >= 3 and lo < hi")
    g = np.linspace(lo, hi, n)
    g[0], g[-1] = lo, hi
    return g


def local_minimum_runs(y: Any) -> list:
    """Index runs (first, last) of equal-valued plateaus that are not higher than either neighbour and are strictly lower than at least one existing neighbour, or span the whole grid. An end plateau counts when it is lower than its only neighbour."""
    y = np.asarray(y, dtype=float)
    runs, i, n = [], 0, y.size
    while i < n:
        j = i
        while j + 1 < n and y[j + 1] == y[i]:
            j += 1
        left = y[i - 1] if i > 0 else None
        right = y[j + 1] if j + 1 < n else None
        lower_l, lower_r = left is None or y[i] < left, right is None or y[i] < right
        if lower_l and lower_r:
            runs.append((i, j))
        i = j + 1
    return runs


def golden(fn: Callable[[float], float], a: float, b: float, xatol: float = 1e-10, max_iter: int = 200) -> float:
    c, d = b - GOLDEN * (b - a), a + GOLDEN * (b - a)
    fc, fd = fn(c), fn(d)
    for _ in range(max_iter):
        if abs(b - a) <= xatol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - GOLDEN * (b - a)
            fc = fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + GOLDEN * (b - a)
            fd = fn(d)
    return 0.5 * (a + b)


def refine_bracket(fn: Callable[[float], float], a: float, b: float, xatol: float = 1e-10) -> tuple:
    """(x, method) minimising fn on [a, b]; scipy.optimize.minimize_scalar(method="bounded") when available and ONLY if it reports success with a finite x and value, otherwise golden section
    (the method label then says so; a failed SciPy refinement is never reported as a successful one)."""
    try:
        from scipy.optimize import minimize_scalar
    except ImportError:
        return golden(fn, a, b, xatol), "golden_section"
    r = minimize_scalar(fn, bounds=(a, b), method="bounded", options={"xatol": xatol})
    if bool(getattr(r, "success", False)) and math.isfinite(float(r.x)) and math.isfinite(float(r.fun)) and a <= float(r.x) <= b:
        return float(r.x), "scipy_bounded"
    return golden(fn, a, b, xatol), "golden_section_after_scipy_failure"


def search(objective: Callable[[float], float], *, n_grid: int = N_GRID, lo: float = STRESS_MIN, hi: float = STRESS_MAX, xatol: float = 1e-10) -> dict:
    """Grid -> local-minimum runs -> bounded refinement of each inside its neighbouring grid points -> global comparison over EVERY evaluated stress (grid and refined). Ties go to the smaller stress.
    Returns the whole evaluated profile, all minima with bound/plateau flags, and the global selection; it never claims an unbounded optimum."""
    f = objective if isinstance(objective, CachedObjective) else CachedObjective(objective)
    xs = dense_grid(n_grid, lo, hi)
    ys = np.array([f(x) for x in xs])
    minima = []
    for a, b in local_minimum_runs(ys):
        left, right = xs[max(a - 1, 0)], xs[min(b + 1, xs.size - 1)]
        x_ref, how = refine_bracket(f, float(left), float(right), xatol)
        x_ref = float(min(max(x_ref, lo), hi))
        grid_best = min(range(a, b + 1), key=lambda k: (ys[k], xs[k]))
        val, x = min((f(x_ref), x_ref), (float(ys[grid_best]), float(xs[grid_best])))
        minima.append({"grid_run": [float(xs[a]), float(xs[b])], "n_grid_points_in_plateau": b - a + 1, "refine_bracket": [float(left), float(right)], "refine_method": how, "stress": x, "objective": val,
                       "refined_stress": x_ref, "at_lower_bound": x <= lo + BOUND_TOL, "at_upper_bound": x >= hi - BOUND_TOL, "flat_plateau": b > a})
    if not minima:
        raise SearchError("no local minimum found on the grid")
    best_x, best_v = min(f.cache.items(), key=lambda kv: (kv[1], kv[0]))                      # the actual minimum over EVERY evaluated stress; ties to the smaller stress
    owner = next((m for m in minima if m["refine_bracket"][0] <= best_x <= m["refine_bracket"][1]), None)
    best = {"stress": best_x, "objective": best_v, "at_lower_bound": best_x <= lo + BOUND_TOL, "at_upper_bound": best_x >= hi - BOUND_TOL, "on_grid": best_x in set(map(float, xs)),
            "flat_plateau": bool(owner and owner["flat_plateau"]), "n_grid_points_in_plateau": owner["n_grid_points_in_plateau"] if owner else None, "belongs_to_minimum": owner}
    prof = sorted(f.cache.items())
    return {"lo": lo, "hi": hi, "n_grid": n_grid, "grid": [float(x) for x in xs], "grid_objective": [float(y) for y in ys], "n_evaluations": len(f.cache),
            "profile": {"stress": [p[0] for p in prof], "objective": [p[1] for p in prof]}, "minima": minima, "n_local_minima": len(minima), "selected": best,
            "multiple_minima": len(minima) > 1, "boundary_solution": bool(best["at_lower_bound"] or best["at_upper_bound"]),
            "limits_note": f"bounded search over the supported range [{lo}, {hi}] only; a minimum at a bound means the optimum was not located inside it and no unbounded optimum is claimed"}


def near_optimal_intervals(stress: Any, mse: Any, fractions: Any = (0.01, 0.05)) -> dict:
    """DESCRIPTIVE intervals: for each fraction the maximal contiguous runs of evaluated stresses (sorted profile, grid points only meaningful) whose MSE <= (1 + fraction) x the minimum. NOT a confidence interval."""
    x, y = np.asarray(stress, dtype=float), np.asarray(mse, dtype=float)
    if x.shape != y.shape or x.size == 0 or np.any(np.diff(x) <= 0):
        raise SearchError("need strictly increasing stress with matching objective values")
    best = float(y.min())
    out = {"descriptive_only": "not a confidence or credible interval", "min_mse": best, "intervals": {}}
    for fr in fractions:
        ok = y <= best * (1.0 + fr)
        runs, i = [], 0
        while i < ok.size:
            if ok[i]:
                j = i
                while j + 1 < ok.size and ok[j + 1]:
                    j += 1
                runs.append([float(x[i]), float(x[j])])
                i = j + 1
            else:
                i += 1
        out["intervals"][f"{fr:g}"] = runs
    return out


def retained_bound_report(stress: float, lo: float, hi: float) -> dict:
    return {"at_lower_bound": stress <= lo + BOUND_TOL, "at_upper_bound": stress >= hi - BOUND_TOL}
