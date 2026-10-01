"""MAPLE's numerical conservation policy, without a separate SYRUP epsilon.

The block policy follows maple.aeolian.scheduler.block: the shared MAPLE
summation bound, the actual number of accounting steps, and the largest
real reservoir endpoint/boundary operand. Physical mass resolution and
reported numerical residuals are neither a roundoff floor nor missing mass.
"""
from __future__ import annotations

import numpy as np


def reservoir_bound_kg(n_accounting_steps: int, *operands) -> float:
    from maple.surface.voxels._numerics import summation_error_bound_kg

    if isinstance(n_accounting_steps, bool) or not isinstance(n_accounting_steps, int) or n_accounting_steps < 0:
        raise ValueError('n_accounting_steps must be a nonnegative integer')
    arrays = [np.asarray(v, dtype=np.float64) for v in operands]
    if not arrays or any(not np.isfinite(v).all() for v in arrays):
        raise ValueError('finite genuine reservoir operands required')
    scale = max(float(np.max(np.abs(v), initial=0.0)) for v in arrays)
    return float(summation_error_bound_kg(max(n_accounting_steps, 1), scale))


def volume_roundoff_bound_m3(n_terms: int, scale_m3: float) -> float:
    """Same MAPLE dimensionless FP64 summation coefficient, in volume units.

    Normalizing the kg helper at one kg extracts its dimensionless bound;
    multiplying by the genuine m3 operand supplies volume units. This does
    not reinterpret water volume as sediment mass or use mass resolution.
    """
    from maple.surface.voxels._numerics import summation_error_bound_kg

    if isinstance(n_terms, bool) or not isinstance(n_terms, int) or n_terms < 1:
        raise ValueError('n_terms must be a positive integer')
    if not np.isfinite(scale_m3) or scale_m3 < 0:
        raise ValueError('volume scale must be finite and nonnegative')
    return float(summation_error_bound_kg(n_terms, 1.0) * scale_m3)
