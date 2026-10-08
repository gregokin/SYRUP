"""Builders shared by the GPU Newton tests (graphs with exact level widths, pairs of host/device graphs, bit checks)."""
from __future__ import annotations

import functools

import numpy as np
from test_routing import chain_full, make_graph, random_full, valley_full

C_DEFAULT = 0.5 / 3.0
SPECIAL = [0.0, -0.0, -1e-3, np.nan, -np.nan, np.inf, -np.inf, 5e-324, 2.2250738585072014e-308, 1e-300, 1e-12, 3e-3,
           1e200, 1e308, -5e-324]
NAMES = ("chain", "valley_branching", "valley_wide", "random_w127", "random_w128", "random_w129", "random_w257",
         "plane_129")


@functools.cache
def graph_pair(name):
    """(host graph, CuPy graph) built from identical inputs."""
    import cupy as cp

    rng = np.random.default_rng(11)
    if name == "chain":
        z, ff = chain_full(9), 5.0
    elif name == "valley_branching":
        z, ff = valley_full(6, 5), 5.0
    elif name == "valley_wide":
        z, ff = valley_full(3, 41), 5.0
    elif name.startswith("plane_"):
        nx = int(name.split("_")[1])
        z = np.repeat((np.arange(5, dtype=np.float64) * 0.015625)[:, None], nx + 2, axis=1)
        ff = 5.0
    else:
        width = int(name.split("_w")[1])
        z = random_full(rng, 6, width)
        ff = rng.uniform(5.0, 30.0, (6, width))
    return make_graph(z, ff=ff), make_graph(z, ff=ff, xp=cp)


def bases(n, seed=3):
    rng = np.random.default_rng(seed)
    wet = rng.uniform(1e-4, 3e-3, n)
    return {
        "wet": wet,
        "dry": np.zeros(n),
        "mixed": wet * (rng.random(n) < 0.45),
        "deep": rng.uniform(0.05, 2.0, n),
        "negative": -wet,
        "subnormal": rng.uniform(1.0, 2.0, n) * 5e-320,
        "special": np.resize(np.array(SPECIAL), n),
    }


def bit_mismatches(ref, got) -> int:
    """Number of float64 positions whose bits differ (two NaNs agree when their sign bits agree)."""
    ref, got = np.asarray(ref, dtype=np.float64), np.asarray(got, dtype=np.float64)
    assert ref.shape == got.shape
    rb, gb = ref.view(np.uint64), got.view(np.uint64)
    both_nan = np.isnan(ref) & np.isnan(got)
    bad = (rb != gb) & ~both_nan
    bad |= both_nan & ((rb >> np.uint64(63)) != (gb >> np.uint64(63)))
    return int(bad.sum())
