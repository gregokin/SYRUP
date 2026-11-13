"""Builders of the D2 floating intake/withdrawal suite (author-written; loads the D1 builder by path)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from maple_syrup import floating_soil_layers as fl
from maple_syrup import subsurface as ss

_p = Path(__file__).resolve().parents[1] / "phase2_floating_column" / "conftest.py"
_spec = importlib.util.spec_from_file_location("d1_floating_builders", _p)
_d1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_d1)
BACKENDS, column = _d1.BACKENDS, _d1.column


def state(col, w, *, initial=False):
    w = np.ascontiguousarray(np.asarray(w, dtype=np.float64).reshape(col.n_layers, col.n_cells))
    return (fl.floating_initial_state if initial else fl.floating_continued_state)(col, w, 0.0)


def pair(cp, edges, depth, *, tr=0.1, tf=0.3, ts=0.5, active=None, bottom="open", stress=0.2, area=1.0):
    """The same inputs as a NumPy column and as a (fake) CUDA column (host-compiled plumbing double)."""
    from hostd2 import fake

    edges = np.asarray(edges, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    N, L = depth.size, len(edges) - 1
    act = np.ones(N, bool) if active is None else active
    dev = lambda x: fake(x) if isinstance(x, np.ndarray) else x
    common = {"theta_evaporation_stress": stress, "diffusion_mode": "constant", "bottom_drainage": bottom, "cell_area_m2": area}
    cpu = fl.prepare_floating_column(edges, theta_residual=tr, theta_field_capacity=tf, theta_saturation=ts, soil_depth_m=depth, active_mask=act, diffusivity_m2_s=np.zeros(L - 1), **common)
    gpu = fl.prepare_floating_column(edges, theta_residual=dev(tr), theta_field_capacity=dev(tf), theta_saturation=dev(ts), soil_depth_m=dev(depth), active_mask=dev(act), backend="cuda",
                                     diffusivity_m2_s=dev(np.zeros(L - 1)), **common)
    return cpu, gpu


REFUSALS = (fl.FloatingSoilError, ss.SubsurfaceError, ss.SubsurfaceMemoryError)  # every typed refusal of the floating components (FloatingMemoryError, MatrixError, PreferentialError ... derive from these)
