"""Transport operator on topologies with NO face of one orientation.

A pure north/south chain or plane has no x faces; a pure east/west one has
no y faces. The face-diagnostic scatter must be a no-op there (zero arrays of
the canonical shapes) while every crossing, export and per-cell balance is
unchanged. The CPU tests always run; the CuPy tests run only when an actual
CUDA device is available and compare the device result with the host result
field by field. These are KERNEL checks, not a coupled GPU event."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("maple")

from maple_syrup.routing import build_routing_graph
from maple_syrup.sediment_transport import transport_network, transport_step

NC = 3


def south_plane_full(ny, nx, *, dz=0.015, wall=10.0):
    """Elevation rising northward (row 0 = south); high east/west walls; the
    south ring row exports. Every active cell drains south: y faces only."""
    z = np.repeat(np.arange(ny + 2, dtype=np.float64)[:, None] * dz, nx + 2, axis=1)
    z[:, [0, -1]] += wall
    exports = np.zeros(z.shape, dtype=bool)
    exports[0, :] = True
    return z, exports


def west_plane_full(ny, nx, *, dz=0.015, wall=10.0):
    """The transpose: elevation rising eastward, high north/south walls, the
    west ring column exports. Every active cell drains west: x faces only."""
    z = np.repeat(np.arange(nx + 2, dtype=np.float64)[None, :] * dz, ny + 2, axis=0)
    z[[0, -1], :] += wall
    exports = np.zeros(z.shape, dtype=bool)
    exports[:, 0] = True
    return z, exports


def graph_for(full, xp=None):
    z, exports = full
    ny, nx = z.shape[0] - 2, z.shape[1] - 2
    return build_routing_graph(z, exports, np.full((ny, nx), 21.45), 0.5, xp=xp)


def inputs(rng, ny, nx, xp=None):
    pool = rng.uniform(0.0, 2.0, (ny, nx, NC))
    v = rng.uniform(0.05, 0.4, (ny, nx, NC))
    rate = rng.uniform(0.0, 2.0, (ny, nx, NC))
    settle = np.zeros((ny, nx, NC), dtype=bool)
    settle[ny // 2, nx // 2, 0] = True
    if xp is None:
        return pool, v, rate, settle
    return tuple(xp.asarray(a) for a in (pool, v, rate, settle))


def check_single_orientation(step, network, *, missing: str, ny: int, nx: int):
    """Absent-orientation diagnostics are exactly zero with canonical shapes;
    the present orientation carries every crossing and the export; per-cell
    and per-class balances hold within the operator's own declared bounds."""
    from maple.core.backend import to_host

    export = to_host(step.export_request_kg)
    out_internal = to_host(step.internal_transfer_out_kg)
    x_gross, y_gross = to_host(step.x_face_gross_kg), to_host(step.y_face_gross_kg)
    x_net, y_net = to_host(step.x_face_net_kg), to_host(step.y_face_net_kg)
    assert x_gross.shape == (ny, nx + 1, NC) and y_gross.shape == (ny + 1, nx, NC)
    if missing == "x":
        assert network.cells_x.size == 0 and network.cells_y.size == ny * nx
        assert not np.any(x_gross) and not np.any(x_net)
        present_gross = y_gross
        np.testing.assert_array_equal(y_gross[0], export[0])  # outlet row crosses the south face row 0
        assert np.all(y_net <= 0.0)  # +y is north; everything moves south
    else:
        assert network.cells_y.size == 0 and network.cells_x.size == ny * nx
        assert not np.any(y_gross) and not np.any(y_net)
        present_gross = x_gross
        np.testing.assert_array_equal(x_gross[:, 0], export[:, 0])  # outlet column crosses the west face column 0
        assert np.all(x_net <= 0.0)  # +x is east; everything moves west
    # every crossing (internal + export) appears exactly once in the present orientation
    np.testing.assert_allclose(present_gross.sum(axis=(0, 1)), (out_internal + export).sum(axis=(0, 1)),
                               rtol=1e-12, atol=1e-15)
    assert float(export.sum()) > 0.0 and float(out_internal.sum()) > 0.0
    residual = np.abs(to_host(step.budget_residual_by_class_kg))
    assert np.all(residual <= to_host(step.budget_tolerance_by_class_kg))
    after, divergence = to_host(step.mobile_after_transfer_kg), to_host(step.divergence_kg)
    assert np.all(np.isfinite(after)) and np.all(np.isfinite(divergence)) and np.all(after >= 0.0)


@pytest.mark.parametrize("missing, builder", [("x", south_plane_full), ("y", west_plane_full)])
def test_cpu_single_orientation_topologies_keep_conservation_and_zero_absent_faces(missing, builder):
    ny, nx = 6, 4
    rng = np.random.default_rng(3)
    graph = graph_for(builder(ny, nx))
    network = transport_network(graph)
    pool, v, rate, settle = inputs(rng, ny, nx)
    step = transport_step(network, pool, v, rate, settle, 0.5, n_substeps=2)
    check_single_orientation(step, network, missing=missing, ny=ny, nx=nx)
    # per-cell identity from the returned pieces
    np.testing.assert_allclose(step.mobile_after_transfer_kg - pool, step.divergence_kg, rtol=0, atol=1e-13)


@pytest.mark.parametrize("missing, builder", [("x", south_plane_full), ("y", west_plane_full)])
def test_cupy_single_orientation_topologies_match_cpu_when_a_gpu_is_available(missing, builder):
    from maple.core.backend import cupy_module, gpu_execution_available, to_host

    if not gpu_execution_available():
        pytest.skip("CuPy with a CUDA device is not available; GPU path not exercised, no claim made")
    cp = cupy_module()
    ny, nx = 6, 4
    rng = np.random.default_rng(3)
    full = builder(ny, nx)
    graph_np, graph_cp = graph_for(full), graph_for(full, xp=cp)
    network_np, network_cp = transport_network(graph_np), transport_network(graph_cp)
    pool, v, rate, settle = inputs(rng, ny, nx)
    host = transport_step(network_np, pool, v, rate, settle, 0.5, n_substeps=2)
    device = transport_step(network_cp, cp.asarray(pool), cp.asarray(v), cp.asarray(rate), cp.asarray(settle), 0.5,
                            n_substeps=2)
    check_single_orientation(host, network_np, missing=missing, ny=ny, nx=nx)
    check_single_orientation(device, network_cp, missing=missing, ny=ny, nx=nx)
    for name in ("mobile_after_transfer_kg", "deposition_request_kg", "export_request_kg", "decay_deposition_kg",
                 "settled_kg", "internal_transfer_in_kg", "internal_transfer_out_kg", "divergence_kg",
                 "x_face_gross_kg", "y_face_gross_kg", "x_face_net_kg", "y_face_net_kg",
                 "mobile_before_by_class_kg", "mobile_after_by_class_kg", "deposition_request_by_class_kg",
                 "export_request_by_class_kg"):
        np.testing.assert_allclose(to_host(getattr(device, name)), getattr(host, name), rtol=1e-12, atol=1e-15,
                                   err_msg=name)
    # Near-zero subtraction residuals need not match between reduction backends.
    # Both independently satisfy the unchanged MAPLE bound above; physical
    # arrays and transferred totals retain the tighter parity checks.
    assert device.n_substeps == host.n_substeps == 2
