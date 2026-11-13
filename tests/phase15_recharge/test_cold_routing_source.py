"""Phase 1.5 S3 correction02: the COLD initial routing source of the CUDA layered setup comes from the ACTUAL native bed the lifecycle will own (device bed,
`sediment_bed.current_routing_surface_m(bed, ctx)`), not from a host re-computation on the compiled bed (the two can differ in the last bit; the native strict held-source
equality in `layered_storm.prepare_layered_storm` correctly refuses that). These are CPU controls of the actual construction code (genuine CPU NCP/file-case factory,
NumPy namespace, `_host_stage(..., bed=, ctx=)`); the stand-ins below are explicitly labeled injection/arithmetic controls and NOT CUDA acceptance. The real device controls
are in `test_daily_cuda.py` (written, unrun). Needs SYRUP_NCP_TARGET like the file-case lane (strict fail for a configured unusable target; skip only if unset)."""
from __future__ import annotations

from types import SimpleNamespace

import ncpcases as nc
import numpy as np
import pytest

from maple_syrup import layered_case as lc
from maple_syrup import layered_case_cuda as lcc
from maple_syrup import sediment_bed as sb
from maple_syrup import pond_network as pn

pytestmark = [pytest.mark.skipif(nc.skip_reason() is not None, reason=nc.skip_reason() or "")]


@pytest.fixture(autouse=True)
def _configured_bundle_or_fail():
    if nc.skip_reason() is None:
        nc.require_configured_bundle()


def min_plan(c, n_cells):
    return {"shape": c.shape, "host_terms": {"B_texture_required": 1 << 30, "texture_batch_cells": n_cells}}


def test_topology_source_is_exactly_the_current_routing_surface_of_the_given_bed(tmp_path, monkeypatch):
    c = nc.build_file_case(tmp_path, nc.yaml_for(2))
    calls = []
    real = sb.current_routing_surface_m
    monkeypatch.setattr(sb, "current_routing_surface_m", lambda bed, ctx: (calls.append((bed, ctx)), real(bed, ctx))[1])
    texture, topo, graph = lcc._host_stage(c.admission.compiled, c.cfg, min_plan(c, c.n), np, c.prep.bed, c.prep.context)
    assert len(calls) == 1 and calls[0][0] is c.prep.bed and calls[0][1] is c.prep.context  # the ACTUAL bed/context, evaluated once
    expected = np.asarray(real(c.prep.bed, c.prep.context))
    act = np.asarray(topo.active)
    assert np.array_equal(np.asarray(topo.source_bed_m)[act], expected[act])  # exact, bit for bit, on every active cell
    assert texture is not None and graph is not None


def test_injected_last_bit_difference_is_followed_so_the_native_strict_equality_holds(tmp_path, monkeypatch):
    """INJECTION control (stand-in for the observed 1-ULP CPU/device difference): when the actual bed's routing surface differs by one ULP from the host re-computation,
    the topology must carry the ACTUAL bed's values (the legacy host path would not)."""
    c = nc.build_file_case(tmp_path, nc.yaml_for(2))
    real = sb.current_routing_surface_m
    base = np.asarray(real(c.prep.bed, c.prep.context)).copy()
    bumped = base.copy()
    bumped.flat[3] = np.nextafter(bumped.flat[3], np.inf)
    monkeypatch.setattr(sb, "current_routing_surface_m", lambda bed, ctx: bumped.copy() if bed is c.prep.bed else real(bed, ctx))
    _, topo_dev, _ = lcc._host_stage(c.admission.compiled, c.cfg, min_plan(c, c.n), np, c.prep.bed, c.prep.context)
    _, topo_host, _ = lcc._host_stage(c.admission.compiled, c.cfg, min_plan(c, c.n), np)  # legacy host-only path (compiled bed shell)
    act = np.asarray(topo_dev.active)
    assert np.array_equal(np.asarray(topo_dev.source_bed_m)[act], bumped[act])
    assert not np.array_equal(np.asarray(topo_host.source_bed_m)[act], bumped[act])  # exactly the defect shape: a 1-ULP disagreement the strict guard would refuse


@pytest.mark.parametrize("kind", ["nan", "shape", "dtype"])
def test_invalid_native_surface_is_refused_before_any_topology_is_built(kind, tmp_path, monkeypatch):
    c = nc.build_file_case(tmp_path, nc.yaml_for(2))
    real = sb.current_routing_surface_m
    base = np.asarray(real(c.prep.bed, c.prep.context)).copy()

    def bad(bed, ctx):
        if bed is not c.prep.bed:
            return real(bed, ctx)
        if kind == "nan":
            out = base.copy()
            out.flat[0] = np.nan
            return out
        return base[:-1] if kind == "shape" else base.astype(np.float32)

    built = []
    realt = pn.build_pond_topology
    monkeypatch.setattr(sb, "current_routing_surface_m", bad)
    monkeypatch.setattr(pn, "build_pond_topology", lambda *a, **k: (built.append(1), realt(*a, **k))[1])
    with pytest.raises(lc.LayeredCaseError, match="cold routing surface of the actual native bed"):
        lcc._host_stage(c.admission.compiled, c.cfg, min_plan(c, c.n), np, c.prep.bed, c.prep.context)
    assert built == []  # shape/dtype/finite are validated BEFORE the topology/graph build


def test_plan_counts_the_cold_surface_stage_and_its_single_transfer(tmp_path, monkeypatch):
    """ARITHMETIC control with a labeled stand-in for the A3b device-planner result (no device exists here): the new named stages S (device) and Z (host) are counted."""
    c = nc.build_file_case(tmp_path, nc.yaml_for(2))
    ny, nx = c.shape
    N = ny * nx
    monkeypatch.setattr(lcc.ndc, "plan_native_dense_cuda", lambda *a, **k: SimpleNamespace(host_required_bytes=1234, device_terms={"validation_scratch": 4096}, transfer_bytes=0))
    vc = c.admission.compiled.voxel_column
    V, K = int(vc.mass_kg.shape[2]), int(vc.mass_kg.shape[3])
    g = dict(ny=ny, nx=nx, V=V, K=K, hr=1 << 20, dr=1 << 20, extra=1 << 20, native=1 << 22, dev=0)
    plan = lcc._plan(c.admission, c.cfg, g, None, None)
    S, Z = plan["stage_bytes"]["device"]["S"], plan["stage_bytes"]["host"]["Z"]
    assert plan["device_terms"]["S_surface_d2h_transfer_bytes"] == 8 * N and plan["host_terms"]["Z_downloaded_surface"] == 8 * N  # ONE N-surface transfer, no deep bed
    assert S == plan["device_terms"]["S_cold_routing_surface_peak"] and S >= 3 * 8 * N + lcc.ALLOCATOR_ROUNDING_BYTES
    assert plan["device_required_bytes"] >= S and plan["host_required_bytes"] >= Z
    assert Z < S  # the dense diagnostic temporaries live on the DEVICE stage, not the host surface stage
    assert plan["actual"]["cold_surface_source"] is None  # filled by the real setup
