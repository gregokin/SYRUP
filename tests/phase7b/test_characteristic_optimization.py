"""Exercise compact-candidate routing/merging against the unchanged array path."""
from types import SimpleNamespace

import numpy as np
import pytest

from maple_syrup.characteristic_transport import NARROW_SPREAD, _substep_array


@pytest.fixture(scope='module')
def compiled_reference():
    numba = pytest.importorskip('numba')
    from .characteristic_reference import _substep
    return numba.njit(fastmath=False, boundscheck=True)(_substep)


@pytest.mark.parametrize('nb', [1, 7, 32, 128])
@pytest.mark.parametrize('density', [0.0, 0.03, 1.0])
def test_compact_candidates_against_array(nb, density, compiled_reference):
    pytest.importorskip('numba')
    from maple_syrup.characteristic_numba import compiled_substep

    rng = np.random.default_rng(714 + nb)
    n, nc, dx, dt = 12, 3, .5, 1.
    shape = (n, nc, nb)
    w = np.where(rng.random(shape) < density, 10. ** rng.uniform(-25, 1, shape), 0.)
    x = (np.arange(nb) + rng.random(shape)) * dx / nb
    x[w == 0] = 0.
    p = np.where(rng.random((n, nc)) < density, rng.random((n, nc)), 0.)
    v = rng.uniform(0, .25, (n, nc))
    r = 10. ** rng.uniform(-10, 4, (n, nc))
    r[:, 0] = 0.
    settle = rng.random((n, nc)) < .2
    receiver = np.minimum(np.arange(n) + 1, n - 1).astype(np.int64)
    receiver[:4] = 5  # multiple upstream arrivals into the same bins
    outlet = np.arange(n) == n - 1
    network = SimpleNamespace(receiver_index=receiver, outlet_flat=outlet)
    inputs = [w, x, p, v, r, settle, receiver, outlet]
    copies = [a.copy() for a in inputs]
    actual = compiled_substep()(*inputs, dx, dt, nb, NARROW_SPREAD)
    reference = compiled_reference(*inputs, dx, dt, nb, NARROW_SPREAD)
    for a, b in zip(actual, reference, strict=True):
        np.testing.assert_array_equal(a, b)
    if density == 0.0:
        # Explicitly exercises the exact-zero fast return, not a mass cutoff.
        assert all(np.count_nonzero(a) == 0 for a in actual)
    with np.errstate(all='ignore'):
        expected = _substep_array(np, network, w, x, p, v, r, settle, dx, dt, nb)
    for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
        # Equivalent reference reductions can differ by floating-point order.
        np.testing.assert_allclose(a, b, rtol=2e-13, atol=2e-15, err_msg=f'output {index}')
    for a, b in zip(inputs, copies, strict=True):
        np.testing.assert_array_equal(a, b)
    # Independent cell/class mass identity, including converging transfers.
    remaining, _, dep, settled, exported, outflow, inflow, arrival_dep, arrival_settle = actual
    np.testing.assert_allclose(remaining.sum(-1) + dep + settled + exported + arrival_dep + arrival_settle,
                               w.sum(-1) + p + inflow - outflow, rtol=2e-13, atol=2e-15)
