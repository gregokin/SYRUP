"""The caller's STATIC recipe for the CUDA context container (primitive parameters only: terrain, grid, forcing, sediment/splash parameters, control through the H2 `static_case` primitives, with CuPy parameter objects). It never
touches an original bed, owner, probe or texture provider: `factory(held_context)` builds the event from the verified H0 `/2` context with `FloatingPondVertical.from_held_context`. Used by the GPU definitions and the fresh worker."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import static_case as sc


def factory(windowed: bool, *, available_bytes_fn=None, batch_cells=None):
    def make(ctx):
        from maple.core.backend import cupy_module, to_device

        from maple_syrup import floating_pond as fp
        from maple_syrup.experimental_storm import ExperimentalControl
        from maple_syrup.pond_event import PondEvent, PondEventControl
        from maple_syrup.rainfall import constant_rainfall
        from maple_syrup.sediment_physics import plot1_sediment_parameters
        from maple_syrup.splash import splash_parameters

        cp = cupy_module()
        kw = {} if batch_cells is None else {"batch_cells": batch_cells}
        from maple_syrup import floating_soil_layers_cuda as fcu

        reader = fcu.device_reader(cp) if available_bytes_fn is None else available_bytes_fn
        vert = fp.FloatingPondVertical.from_held_context(ctx, available_bytes_fn=reader, **kw)
        bed_ctx = sc.bed_context()
        dctx = replace(bed_ctx, initial_available_fraction=to_device(bed_ctx.initial_available_fraction, cp))
        ctl = PondEventControl(backend="cuda", time=ExperimentalControl(max_dt_s=0.2, min_dt_s=1.0 / 1024.0, max_retries=12, max_steps=50_000), bed_storage="windowed" if windowed else "dense", deep_publication="step")
        return PondEvent(dctx, None, plot1_sediment_parameters(xp=cp), constant_rainfall(0.0, 1000.0, 300.0), sc.topology(), control=ctl, splash=splash_parameters(np.linspace(0.4, 3.0, sc.NC), xp=cp), floating_vertical=vert)

    return make
