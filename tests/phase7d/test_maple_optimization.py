"""Opt-in regression suite for the recorded isolated MAPLE candidate.

Run with the candidate environment; an accidental baseline import is an error.
Original routines are loaded from the accepted dependency, never copied here.
"""
import dataclasses
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from maple.core.backend import resolve_backend, to_host
from maple.core.boundaries import AxisBoundary, BoundaryKind
from maple.core.parameters.geometry import GeometrySpec
from maple.coupling.sediment_ledger import accumulate
from maple.surface.voxels import transfer
from maple.surface.voxels.capacity import max_voxel_mass_kg

from maple_syrup.dependency import resolve_maple_dependency
from maple_syrup.provenance import source_tree_digest

BASELINE = Path(__file__).resolve().parents[2] / 'outputs/dependencies/maple_d3d007024/source/src/maple'
EXPECTED = '72310c49ae3b2db99f8ab919669303e98b14ee4479eb17522e5f6ad7ff474f65'


@pytest.fixture(scope='module', autouse=True)
def candidate_identity():
    assert source_tree_digest(resolve_maple_dependency().package_dir).digest_sha256 == EXPECTED


def reference(relative, name):
    spec = importlib.util.spec_from_file_location(name, BASELINE / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def old_transfer():
    return reference('surface/voxels/transfer.py', 'phase7d_original_transfer')


@pytest.fixture(scope='module')
def old_ledger():
    return reference('coupling/sediment_ledger/accumulate.py', 'phase7d_original_accumulate')


@pytest.fixture(params=['numpy', 'cupy'])
def xp(request):
    try:
        return resolve_backend(request.param).xp
    except Exception as exc:
        if request.param == 'cupy':
            pytest.skip(f'GPU unavailable: {exc}')
        raise


@pytest.mark.parametrize('nc', [1, 2, 6, 9, 17])
@pytest.mark.parametrize('nv', [1, 2, 20, 64])
def test_extraction_exact_against_original(xp, nc, nv, old_transfer):
    g = GeometrySpec(nx=3, ny=4, dx_m=1., dy_m=1., voxel_dz_m=1.,
                     bulk_density_kg_m3=10., active_layer_thickness_m=.2,
                     boundary_x=AxisBoundary(kind=BoundaryKind.PERIODIC),
                     boundary_y=AxisBoundary(kind=BoundaryKind.PERIODIC))
    rng = np.random.default_rng(812 + nc + nv)
    mass = rng.uniform(.01, 1, (4, 3, nv, nc))
    mass *= max_voxel_mass_kg(g) / mass.sum(-1)[..., None]
    for y in range(4):
        for x in range(3):
            full = rng.integers(0, nv+1)
            mass[y, x, full:] = 0
            if full:
                mass[y, x, full-1] *= rng.uniform(.01, .99)
    total = mass.sum((-1, -2))
    interior = mass.sum(-1)[..., ::-1].cumsum(-1)[..., nv // 2]
    for req in (interior, np.nextafter(interior, 0), np.nextafter(interior, np.inf),
                np.zeros_like(total), np.full_like(total, 1e-14), total,
                total * 1.1, total * .5, np.nextafter(total, 0)):
        a, b, request = xp.asarray(mass.copy()), xp.asarray(mass.copy()), xp.asarray(req)
        old = old_transfer._extract_surface_mixture_batched_inplace(a, request, g, 1e-10)
        new = transfer._extract_surface_mixture_batched_inplace(b, request, g, 1e-10)
        np.testing.assert_array_equal(to_host(a), to_host(b))
        for field in dataclasses.fields(old):
            np.testing.assert_array_equal(to_host(getattr(old, field.name)),
                                          to_host(getattr(new, field.name)), err_msg=field.name)


def test_kahan_exact_and_inputs_unmodified(xp, old_ledger):
    rng = np.random.default_rng(19)
    running = xp.asarray(rng.uniform(-1e8, 1e8, (6, 4, 3, 6)))
    compensation = xp.asarray(rng.uniform(-1e-8, 1e-8, running.shape))
    saw_compensation = False
    for scale in [1., 1e-9, 1e7, 1e-15] * 16:
        increment = xp.asarray(rng.uniform(-scale, scale, running.shape))
        snapshots = [to_host(x).copy() for x in (running, compensation, increment)]
        old = old_ledger._kahan_add(running, compensation, increment)
        new = accumulate._kahan_add(running, compensation, increment)
        for a, b in zip(old, new, strict=True):
            np.testing.assert_array_equal(to_host(a), to_host(b))
        for a, b in zip((running, compensation, increment), snapshots, strict=True):
            np.testing.assert_array_equal(to_host(a), b)
        running, compensation = new
        saw_compensation |= bool(np.any(to_host(compensation) != 0))
    assert saw_compensation


def test_zero_request_preserves_original_boundary_reconciliation(xp, old_transfer):
    """Do not replace a zero demand by a no-op: legacy boundary reconciliation
    can drain a tiny top voxel. This regression preserves, not endorses, that
    preexisting shared behavior for a later independent scientific review.
    """
    g = GeometrySpec(nx=1, ny=1, dx_m=1., dy_m=1., voxel_dz_m=1.,
                     bulk_density_kg_m3=10., active_layer_thickness_m=.2,
                     boundary_x=AxisBoundary(kind=BoundaryKind.PERIODIC),
                     boundary_y=AxisBoundary(kind=BoundaryKind.PERIODIC))
    mass = np.array([[[[5., 5.], [5e-16, 5e-16]]]])
    a, b = xp.asarray(mass.copy()), xp.asarray(mass.copy())
    request = xp.zeros((1, 1))
    old = old_transfer._extract_surface_mixture_batched_inplace(a, request, g, 1e-10)
    new = transfer._extract_surface_mixture_batched_inplace(b, request, g, 1e-10)
    np.testing.assert_array_equal(to_host(a), to_host(b))
    np.testing.assert_array_equal(to_host(old.actual_mass_by_class_kg), to_host(new.actual_mass_by_class_kg))
    assert float(to_host(new.actual_total_mass_kg)[0, 0]) > 0
