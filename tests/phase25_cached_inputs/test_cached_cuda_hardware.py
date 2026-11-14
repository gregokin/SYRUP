"""OPTIONAL minimal extension of the S0 REAL-CUDA lane to the cached-derived binding: a restored recipe feeds the CUDA preparation on the CURRENT device column and must equal the live-binding law.
AUTHORED, NEVER RUN by its author; skipped unless Root sets SYRUP_VEG_BINDING_RELEASE_GPU=1 (see test_cuda_hardware.py for the hardware prerequisites). Imports no CuPy at collection."""
from __future__ import annotations

import os

import numpy as np
import pytest
from cached_support import loadable, stage, wh, wvb
from test_cuda_hardware import (  # noqa: F401  (the S0 hardware fixtures and helpers)
    FLAG,
    columns,
    cp,
    gprep,
    traffic,
)
from veg_binding_support import BIG, full_source, helper_pin, leaf_pin, two_species

pytestmark = pytest.mark.skipif(not os.environ.get(FLAG), reason=f"DEFINITION ONLY: Root sets {FLAG}=1 on real hardware (never run by the author)")


def _live(leaf):
    _cat, _st, p, surf = two_species(leaf, 3, grass_w=[0, 1, 1], shrub_w=[0, 1, 1])
    m = np.ones((3, 3), dtype=bool)
    ctl = wvb.site_controls(a1=0.013, a2=0.067, a3=0.4, max_fraction_per_hour=0.7, max_intake_rate_m_s=1e-5, background_receiver_weights=[0.0, 1.0, 1.0], n_layers=3, n_cells=9, max_bytes=BIG)
    b = wvb.bind_projection(p, ctl, expected_leaf_package_sha256=leaf_pin(), expected_projection_sha256=p.identity_record()["content_sha256"], expected_source=full_source(p),
                            expected_external_helpers=helper_pin(), active_mask=m, current_surface=surf, max_bytes=BIG)
    return p, b, m


def test_cached_binding_cuda_law_equals_the_live_binding_law_with_expected_current_ids(leaf, cp, traffic):  # noqa: F811
    p, b, m = _live(leaf)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    meta, arrays = loadable(rec)
    rec2 = stage(rec, p, b, meta, arrays, m)
    cb = wh.cached_binding(rec2, expected_binding_sha256=b.sha256, max_bytes=BIG)
    cpu, gpu = columns(cp, m)
    live_law = gprep(gpu, b, m)
    law = gprep(gpu, cb, m, expected_matrix_op_sha256=live_law.matrix_sha256, expected_preferential_op_sha256=live_law.preferential_sha256)
    assert law.binding_origin == "cached_derived" and law.law_sha256 == live_law.law_sha256 and law.g_sha256 == live_law.g_sha256
    assert np.array_equal(cp.asnumpy(law.matrix_op._law[2]), cp.asnumpy(live_law.matrix_op._law[2]))
    cpu_law = wvb.prepare_vegetation_law(cpu, cb, conductivity_m_s=2e-6, suction_m=0.3, memory_budget_bytes=BIG)
    assert np.allclose(cp.asnumpy(law.preferential_op._law[2]), cpu_law.preferential_op._law[2], rtol=1e-12, atol=1e-15)
    assert traffic["h2d_bytes"] > 0 and law.step_taken is False
