from dataclasses import replace

import numpy as np
from maple.surface.voxels._numerics import summation_error_bound_kg

from maple_syrup.complete_event import complete_event
from maple_syrup.conservation import reservoir_bound_kg, volume_roundoff_bound_m3

from .test_complete_event import fixture


def test_same_actual_maple_bound_without_mass_resolution_floor():
    for calls in (0, 1, 1000):
        operands = (np.array([2e-12, 3e-12]), np.array([1e-12, 4e-12]))
        assert reservoir_bound_kg(calls, *operands) == summation_error_bound_kg(max(calls, 1), 4e-12)
    assert reservoir_bound_kg(1, np.zeros(6)) == 0
    assert volume_roundoff_bound_m3(10, 2.5) == summation_error_bound_kg(10, 1.) * 2.5


def test_closure_uses_genuine_maple_operands_and_detects_missing_mass():
    args, kw = fixture()
    result = complete_event(*args, **kw).progress.result
    closure = result.closure()
    scale = max(max(closure[key]) for key in ('initial_bed_kg', 'initial_mobile_kg',
                                             'final_bed_kg', 'final_mobile_kg', 'export_actual_kg'))
    expected = summation_error_bound_kg(result.n_maple_water_calls, scale)
    assert closure['tolerance_kg'] == [expected] * len(closure['initial_bed_kg'])
    corrupted_initial = result.initial_bed_inventory_kg.copy()
    corrupted_initial[0] += 100 * expected
    broken = replace(result, initial_bed_inventory_kg=corrupted_initial,
                     numerical_residual_scalar_abs_kg=np.asarray(1.))
    assert not broken.closure()['closed']  # diagnostic residual cannot excuse lost mass
