"""Safeguarded Newton root solver for the method-5 cell equation (CPU only; NumPy and Numba forms).

It solves the SAME corrected scalar equation as the bisection default of `routing.py`,

    h + c k h^{3/2} = R,          R >= 0, k >= 0, c = dt / (2 dx) > 0,

for the flow depth `h_flow`; everything else of the step (D4 donor order, coherent old flux, storage identity
`h_new = R - c q`, budgets, constitutive and balance checks, tolerances) is shared and unchanged. It is a
selectable ROOT SOLVER, not the explicit or local-inertia physics.

Per cell (`newton_root_scalar` is the single specification; `newton_root_level` is its NumPy vectorized
form and `compiled_sweep_newton` its Numba form, same operations in the same order, no fastmath):

1. `R > 0` is false (zero, negative, NaN): `h_flow = 0`, as the bisection leaves `lo = 0`.
2. `trial(R) = ((sqrt(R) R) k) c + R` is not above `R` (zero conveyance k = 0, a pit; a flux term lost in
   rounding; tiny/subnormal R): the root is `h_flow = R` analytically, no iteration.
3. Otherwise `g(h) = trial(h) - R` is increasing and CONVEX, so Newton started to the right of the root
   converges monotonically. The root obeys `h = R / (1 + a sqrt(h))`, `a = c k`, hence `h >= x_lo = R / (1 +
   a sqrt(R))` and `h <= x0 = R / (1 + a sqrt(x_lo))`: the start `x0` (two square roots, two divisions, exact
   IEEE operations, so bitwise equal in every form) is a valid upper bound. Its ratio to the exact root is NOT bounded: it grows without limit
   (asymptotically ~ (a sqrt R)^(1/6)) as a sqrt(R) grows; it is close in the observed physical cases, and the
   iteration cap plus the bracketed fallback protect any range. Any positive start is safe (a start left of the root only costs an overshooting first step). Each pass
   evaluates `trial(x)` in exactly the bisection operation sequence, tightens a bracket `[lo, hi]`
   (`trial < R` on `lo`, `>= R` on `hi`; `[0, R]` initially), and takes the Newton step
   `x - g/(1 + 1.5 c k sqrt(x))`. A step that leaves the open bracket (or is NaN) is replaced by the bracket
   midpoint (a SAFEGUARD step). Newton stops when the step is <= 16 eps x (or g == 0).
4. Finalization keeps the bisection invariant `trial(h_flow) < R` (so `h_new = R - c q >= h_flow >= 0`
   without clipping): the converged point and then points `x (1 - m eps)`, m = 1, 2, 4, ..., 64 are tried until
   one satisfies it (ACCEPTED: `lo` is then within ~64 eps x of the root). If Newton did not converge within
   `max_iter` passes, or no such point was found, the bracket is bisected (<= 1200 halvings: the FALLBACK;
   always terminates, guaranteed bracket) until it is 4 eps wide.

The result is `h_flow = lo`. Nothing is clipped; the shared constitutive check `|h_new - h_flow| <=
root_tolerance_m` still decides acceptance.

Numerical iteration control: `newton_max_iterations` (default 50, expected use 4-8). The scientific
tolerances are untouched. Statistics per sweep: `(max Newton passes of a cell, total passes, safeguard +
fallback bisection steps, cells that entered the fallback, cells solved iteratively)`.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_NEWTON_MAX_ITERATIONS",
    "MAX_NEWTON_ITERATIONS",
    "ROOT_SOLVERS",
    "SMALL_LEVEL_CELLS",
    "STAT_NAMES",
    "compiled_root",
    "compiled_sweep_newton",
    "newton_root_level",
    "newton_root_scalar",
    "reset_compiled",
    "run_sweep",
    "stats_dict",
]

ROOT_SOLVERS = ("bisection", "newton")
DEFAULT_NEWTON_MAX_ITERATIONS = 50
MAX_NEWTON_ITERATIONS = 1000
STAT_NAMES = ("max_newton_iterations", "total_newton_iterations", "bisection_safeguard_steps",
              "fallback_cells", "iterated_cells")
_EPS = float(np.finfo(np.float64).eps)
_CONVERGED_STEP = 16.0 * _EPS
_FALLBACK_STEP = 4.0 * _EPS
_FALLBACK_LIMIT = 1200  # halvings from [0, R] down to the subnormal range plus the 52 mantissa bits
SMALL_LEVEL_CELLS = 16  # suggested `small_level` for the OPTIONAL cell-by-cell helper of `newton_root_level` (default is 0: off)
_POLISH_TRIES = 8  # m = 0 (the point itself), 1, 2, 4, ..., 64


def _build_root(decorate):
    """The scalar root function with `decorate` applied to it and its helper (identity: pure Python reference;
    numba.njit: the compiled form)."""

    @decorate
    def trial(x, k, c):
        # exactly the bisection operation sequence: mid + (((sqrt(mid) mid) k) c)
        t = np.sqrt(x)
        t = t * x
        t = t * k
        t = t * c
        return t + x

    @decorate
    def root(rhs, k, c, max_iter):
        if not rhs > 0.0:
            return 0.0, 0, 0, 0
        if not trial(rhs, k, c) > rhs:
            return rhs, 0, 0, 0
        a = c * k
        h3 = 1.5 * a
        lo = 0.0
        hi = rhs
        x = rhs
        if a > 0.0:  # a == 0 only if c k underflows although the trial term did not
            x = rhs / (1.0 + a * np.sqrt(rhs / (1.0 + a * np.sqrt(rhs))))
            if not x < rhs:  # NaN / inf from an overflowing a
                x = rhs
        iters = 0
        safe = 0
        fb = 0
        converged = False
        while iters < max_iter:
            iters += 1
            t = trial(x, k, c)
            if t < rhs:
                lo = max(lo, x)
            elif x < hi:
                hi = x
            f = t - rhs
            if f == 0.0:
                converged = True
                break
            xn = x - f / (1.0 + h3 * np.sqrt(x))
            tiny = abs(xn - x) <= _CONVERGED_STEP * xn
            if not tiny and not (xn > lo and xn < hi):
                xn = lo + 0.5 * (hi - lo)
                safe += 1
                tiny = abs(xn - x) <= _CONVERGED_STEP * xn
            x = xn
            if tiny:
                converged = True
                break
        accepted = False
        if converged:
            for j in range(_POLISH_TRIES):
                m = 0.0 if j == 0 else float(1 << (j - 1))
                y = x - m * _EPS * x
                if not y > lo:  # a known point with trial < R already lies at or above y
                    accepted = lo > 0.0
                    break
                if trial(y, k, c) < rhs:
                    lo = y
                    accepted = True
                    break
                hi = min(hi, y)
        if not accepted:
            fb = 1
            for _ in range(_FALLBACK_LIMIT):
                mid = lo + 0.5 * (hi - lo)
                if not (mid > lo and mid < hi):
                    break
                if trial(mid, k, c) < rhs:
                    lo = mid
                else:
                    hi = mid
                safe += 1
                if hi - lo <= _FALLBACK_STEP * hi:
                    break
        return lo, iters, safe, fb

    return root


newton_root_scalar = _build_root(lambda f: f)
newton_root_scalar.__doc__ = "Pure-Python specification of the per-cell root: `(h_flow, passes, bisection_steps, fallback)`."

_ROOT: Any = None
_SWEEP: Any = None


def _require_numba():
    from maple_syrup.routing_numba import _require_numba as require

    return require()


def _options():
    return {"cache": os.environ.get("MAPLE_SYRUP_NUMBA_CACHE", "0") == "1", "fastmath": False, "nogil": True,
            "boundscheck": False}


def compiled_root():
    """The nopython-compiled scalar root (lazily, once per process; missing Numba raises NumbaUnavailableError)."""
    global _ROOT
    if _ROOT is None:
        numba = _require_numba()
        opts = _options()
        # the helper must be a dispatcher visible to the root, so build both with numba.njit
        _ROOT = _build_root(numba.njit(**opts))
    return _ROOT


def _make_sweep(root):
    def sweep(bounds, k_lo, donor_position, donor_mask, base_lo, c, max_iter,
              qin_new_lo, q_new_lo, flow_lo, rhs_lo, stats):
        """Level-ordered sweep, the donor sum and storage identity of `routing_numba._sweep`, with the Newton root.
        `stats` (int64[5]) receives STAT_NAMES. No fastmath, no prange (cells depend on their donors)."""
        top = 0
        total = 0
        safeguards = 0
        fallbacks = 0
        iterated = 0
        for lev in range(bounds.shape[0] - 1):
            for p in range(bounds[lev], bounds[lev + 1]):
                qin = 0.0
                for s in range(4):
                    if donor_mask[s, p]:
                        qin = qin + q_new_lo[donor_position[s, p]]
                    else:
                        qin = qin + 0.0
                rhs = base_lo[p] + qin * c
                k = k_lo[p]
                flow, it, sf, fb = root(rhs, k, c, max_iter)
                q = np.sqrt(flow)
                q = q * flow
                q = q * k
                qin_new_lo[p] = qin
                q_new_lo[p] = q
                flow_lo[p] = flow
                rhs_lo[p] = rhs
                top = max(top, it)
                total += it
                safeguards += sf
                fallbacks += fb
                if it > 0:
                    iterated += 1
        stats[0] = top
        stats[1] = total
        stats[2] = safeguards
        stats[3] = fallbacks
        stats[4] = iterated

    return sweep


def compiled_sweep_newton():
    """The nopython-compiled Newton sweep (CPU NumPy only, lazily, once per process). Its signature is
    `routing_numba._sweep`'s with `max_iter` for `iterations` and a trailing int64[5] `stats` output."""
    global _SWEEP
    if _SWEEP is None:
        numba = _require_numba()
        _SWEEP = numba.njit(**_options())(_make_sweep(compiled_root()))
    return _SWEEP


def reset_compiled() -> None:
    global _ROOT, _SWEEP
    _ROOT = None
    _SWEEP = None


def stats_dict(stats) -> dict[str, int]:
    return {name: int(v) for name, v in zip(STAT_NAMES, stats, strict=True)}


def run_sweep(graph, base_lo: np.ndarray, c: float, max_iter: int):
    """Compiled Newton sweep on host arrays: `(qin_new_lo, q_new_lo, flow_lo, rhs_lo, stats)` in level order."""
    from maple_syrup.routing import RoutingError

    if graph.xp is not np:
        raise RoutingError(
            f"the Newton root solver runs on host NumPy arrays only; the graph lives in {graph.xp.__name__!r} "
            "and no host/device transfer is performed")
    sweep = compiled_sweep_newton()
    n = graph.n_active
    outputs = tuple(np.zeros(n, dtype=np.float64) for _ in range(4))
    stats = np.zeros(len(STAT_NAMES), dtype=np.int64)
    bounds = np.asarray(graph.level_bounds, dtype=np.int64)
    sweep(bounds, graph.conveyance_lo, graph.donor_position, graph.donor_mask,
          np.ascontiguousarray(base_lo, dtype=np.float64), float(c), int(max_iter), *outputs, stats)
    return (*outputs, stats)


# --- NumPy vectorized form -------------------------------------------------------------------------------
def _trial(x, k, c):
    t = np.sqrt(x)
    t = t * x
    t = t * k
    t = t * c
    return t + x


def newton_root_level(rhs: np.ndarray, k: np.ndarray, c: float, max_iter: int, small_level: int = 0):
    """`newton_root_scalar` over a whole dependency level (1-D float64 arrays): the same operations, with
    index sets that shrink as cells converge. Returns `(flow, passes, steps, fallback)` per cell (int64/bool
    arrays); the NumPy namespace only. The DEFAULT (`small_level=0`) is fully vectorized, with no Python loop
    over cells. OPTIONAL helper, never used by `route_step` or any benchmark: with `small_level > 0` a level of at most
    that many cells is solved cell by cell with the pure-Python specification (the identical IEEE operations, hence
    bitwise the same result); `SMALL_LEVEL_CELLS` is only a suggested value for experiments."""
    n = rhs.size
    flow = np.zeros(n, dtype=np.float64)
    passes = np.zeros(n, dtype=np.int64)
    steps = np.zeros(n, dtype=np.int64)
    fallback = np.zeros(n, dtype=np.bool_)
    if n <= small_level:
        with np.errstate(all="ignore"):
            for i, (r_i, k_i) in enumerate(zip(rhs.tolist(), k.tolist(), strict=True)):
                flow[i], passes[i], steps[i], fallback[i] = newton_root_scalar(r_i, k_i, c, max_iter)
        return flow, passes, steps, fallback
    with np.errstate(all="ignore"):
        pos = np.flatnonzero(rhs > 0.0)
        if pos.size == 0:
            return flow, passes, steps, fallback
        t_full = _trial(rhs[pos], k[pos], c)
        analytic = ~(t_full > rhs[pos])
        flow[pos[analytic]] = rhs[pos][analytic]
        idx = pos[~analytic]
        if idx.size == 0:
            return flow, passes, steps, fallback
        r, kk = rhs[idx], k[idx]
        m = idx.size
        a = c * kk
        h3 = 1.5 * a
        x = r / (1.0 + a * np.sqrt(r / (1.0 + a * np.sqrt(r))))
        x = np.where((a > 0.0) & (x < r), x, r)  # a == 0, inf and NaN give x = r, as the scalar form
        lo = np.zeros(m, dtype=np.float64)
        hi = r.copy()
        it = np.zeros(m, dtype=np.int64)
        sf = np.zeros(m, dtype=np.int64)
        conv = np.zeros(m, dtype=np.bool_)
        live = np.arange(m)
        for _ in range(max_iter):
            if live.size == 0:
                break
            xl, rl, kl = x[live], r[live], kk[live]
            lol, hil = lo[live], hi[live]
            t = _trial(xl, kl, c)
            below = t < rl
            lol = np.where(below & (xl > lol), xl, lol)
            hil = np.where(~below & (xl < hil), xl, hil)
            lo[live], hi[live] = lol, hil
            it[live] += 1
            f = t - rl
            xn = xl - f / (1.0 + h3[live] * np.sqrt(xl))
            tiny = np.abs(xn - xl) <= _CONVERGED_STEP * xn
            bad = ~tiny & ~((xn > lol) & (xn < hil))
            xn = np.where(bad, lol + 0.5 * (hil - lol), xn)
            sf[live] += bad
            done = tiny | (bad & (np.abs(xn - xl) <= _CONVERGED_STEP * xn))
            x[live] = xn
            conv[live] = done
            live = live[~done]
        # finalization: first point of x, x(1 - m eps) that satisfies trial < R
        accepted = np.zeros(m, dtype=np.bool_)
        trying = np.flatnonzero(conv)
        for j in range(_POLISH_TRIES):
            if trying.size == 0:
                break
            mj = 0.0 if j == 0 else float(1 << (j - 1))
            xt = x[trying]
            y = xt - mj * _EPS * xt
            lot = lo[trying]
            proceed = y > lot
            ok = proceed & (_trial(y, kk[trying], c) < r[trying])
            lo[trying] = np.where(ok, y, lot)
            accepted[trying[ok | (~proceed & (lot > 0.0))]] = True
            lower = proceed & ~ok & (y < hi[trying])
            hi[trying] = np.where(lower, y, hi[trying])
            trying = trying[proceed & ~ok]
        # bracket guarantee: bisect whatever is not tight
        need = np.flatnonzero(~accepted)
        fallback_local = np.zeros(m, dtype=np.bool_)
        fallback_local[need] = True
        for _ in range(_FALLBACK_LIMIT):
            if need.size == 0:
                break
            lol, hil = lo[need], hi[need]
            mid = lol + 0.5 * (hil - lol)
            valid = (mid > lol) & (mid < hil)
            need, mid = need[valid], mid[valid]
            if need.size == 0:
                break
            below = _trial(mid, kk[need], c) < r[need]
            lo[need] = np.where(below, mid, lo[need])
            hi[need] = np.where(below, hi[need], mid)
            sf[need] += 1
            need = need[~(hi[need] - lo[need] <= _FALLBACK_STEP * hi[need])]
        flow[idx] = lo
        passes[idx] = it
        steps[idx] = sf
        fallback[idx] = fallback_local
    return flow, passes, steps, fallback
