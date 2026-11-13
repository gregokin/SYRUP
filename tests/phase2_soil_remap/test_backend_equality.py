"""NumPy reference vs compiled Numba: bitwise per-column equality on valid inputs and identical refusal classification. Author-written; runs only when Numba is importable."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import arrays, numba_ok, random_columns, run

from maple_syrup import soil_water_remap as sr

pytestmark = pytest.mark.skipif(not numba_ok(), reason="Numba is not importable (no claim made for the compiled backend)")


@pytest.mark.parametrize("L", [1, 2, 3, 5])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_numba_equals_numpy_bitwise(L, seed):
    a = arrays(*random_columns(np.random.default_rng(1000 * L + seed), L, 257))
    r, n = run(*a, "numpy", area=3.0), run(*a, "numba", area=3.0)
    np.testing.assert_array_equal(n.water_m.view(np.uint64), r.water_m.view(np.uint64))
    for name in ("removed_above_m", "removed_below_m", "added_above_m", "added_below_m"):
        np.testing.assert_array_equal(getattr(n, name).view(np.uint64), getattr(r, name).view(np.uint64))
    assert n.closure["max_abs_column_residual_m"] == r.closure["max_abs_column_residual_m"]
    np.testing.assert_allclose(n.closure["global_residual_m3"], r.closure["global_residual_m3"], rtol=0, atol=r.closure["global_bound_m3"])


def test_identical_refusal_classification():
    a = list(arrays(*random_columns(np.random.default_rng(1), 3, 20)))
    a[1][2, 7] = a[1][1, 7]
    a[2][0, 4] = -1.0
    seen = []
    for backend in ("numpy", "numba"):
        with pytest.raises(sr.SoilRemapValidationError) as info:
            run(*a, backend)
        seen.append((info.value.bit, info.value.n_flagged, info.value.first_column))
    assert seen[0] == seen[1] == (sr.FLAG_WATER, 2, 4)


def test_empty_selection_numba():
    z = np.zeros
    res = run(z((3, 0)), z((3, 0)), z((2, 0)), z(0), z(0), "numba")
    assert res.n_columns == 0 and res.water_m.shape == (2, 0)
