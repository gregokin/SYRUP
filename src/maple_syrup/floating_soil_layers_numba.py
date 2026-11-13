"""Argument/geometry binding of the FLOATING porous column onto the SHARED compiled Numba body (`maple_syrup.subsurface_numba`, floating variant). Component only; NOT a step engine.

There is no physical source here: the step is `subsurface_numba._build(numba, False, floating=True)` (the very statements of the common/spatial kernels, with the floating geometry/boundary/zero-interface/inactive hooks) and the staged
reconciliation is `subsurface_numba._build_floating_reconcile` (which jits the SAME `_gravity_stage_py` cascade the step kernel uses). This module only binds the owner's raw read-only maps, the per-cell bottom-open flags and the geometry arrays
to that private signature and returns fresh outputs. No fastmath, no clipping; `math.expm1` may differ from NumPy by an ulp (declared bound rtol 2e-12 / atol 1e-14). Numba is imported lazily; there is no fallback to the reference.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from maple_syrup import subsurface as ss
from maple_syrup import subsurface_numba as sn
from maple_syrup.conservation import volume_roundoff_bound_m3

__all__ = ["numba_available", "reconcile", "require_numba", "reset_compiled", "step"]

KIND = {"scalar": 0, "shared": 1, "cell": 2}


def numba_available() -> bool:
    return sn.numba_available()


def require_numba():
    try:
        sn.require_numba()
    except ss.SubsurfaceBackendUnavailableError as exc:
        from maple_syrup.floating_soil_layers import FloatingBackendUnavailableError

        raise FloatingBackendUnavailableError(str(exc)) from None


def reset_compiled() -> None:
    sn.reset_compiled()


def _flat_maps(column: Any) -> tuple[dict, dict]:
    m, kinds = column._maps, column.layouts
    names = ("theta_residual", "theta_field_capacity", "theta_saturation")
    return {n: m[n].reshape(-1) for n in names}, {n: KIND[kinds[n]] for n in names}


def step(column: Any, water: np.ndarray, added: Any, pet: np.ndarray, dt: float, share: Any, recharge: float | None) -> ss.StepArrays:
    """One compiled floating step through the shared body; fresh `StepArrays` (host). Never writes an input; inactive placeholders are restored bit for bit."""
    m = column._maps
    maps, kinds = _flat_maps(column)
    bo = (m["bottom_open"] & m["active"]).astype(np.float64)  # (N,) 0/1 flags (O(N), per call)
    out = sn.floating_launch(water, added, pet, m["active"].view(np.uint8), dt, edges=m["edges"], depth=m["soil_depth"], sep_nominal=m["sep_nominal"], maps=maps, kinds=kinds,
                             interface=m["interface"].reshape(-1), interface_kind=KIND[column.layouts["interface"]], bottom_open_f64=bo,
                             tolerance_coefficient=volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0), donor_coefficient=0.0 if column.diffusion_coefficient is None else float(column.diffusion_coefficient),
                             stress=float(column.theta_evaporation_stress), factor=float(column.evaporation_factor), directional=0 if column.diffusion_mode == "constant" else 1, share=share, recharge=recharge)
    inactive = ~m["active"]
    if inactive.any():  # the shared body copies inactive entries through; the placeholders are restored exactly (no capacity law applies)
        out.water_m[:, inactive] = water[:, inactive]
    return out


def reconcile(column: Any, water: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kernel = sn.compiled_floating_reconcile()
    L, N = water.shape
    m = column._maps
    w_out, perc, surface, export = np.empty((L, N)), np.zeros((max(L - 1, 0), N)), np.empty(N), np.empty(N)
    kernel(water, m["active"].view(np.uint8), (m["bottom_open"] & m["active"]).view(np.uint8), m["soil_depth"], m["edges"], m["theta_field_capacity"].reshape(-1), KIND[column.layouts["theta_field_capacity"]],
           m["theta_saturation"].reshape(-1), KIND[column.layouts["theta_saturation"]], w_out, perc if L > 1 else np.zeros((1, N)), surface, export)
    return w_out, perc, surface, export
