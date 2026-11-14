"""Support of the CUDA HARDWARE definitions (`test_cuda_hardware.py`): everything is imported LAZILY (no CuPy / GPU import at collection, none during the author's CPU runs). The fixtures are the qualified Phase 2 ones, loaded by
PATH under unique module names: the CUDA event<->daily bridge definitions (`Side`, `matched_pair`, pit / dry kwargs: a genuinely MATCHED CPU twin of every GPU boundary) and the CUDA lifecycle definitions (`opened`, `complete`: the thin
closed lake with a real terminal deposit, Dense and published Windowed). No fake class, head, generation, quiet state or solver."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

T = Path(__file__).resolve().parents[1]
_PATHS = ("phase2_floating_event_lifecycle", "phase2_floating_event_refresh", "phase2_floating_context_checkpoint", "phase2_floating_checkpoint", "phase2_floating_pond", "phase2_floating_checkpoint_cuda",
          "phase2_native_floating_soil", "phase2_compact_soil_profile", "pond_event", "pond_windowed", "phase2_floating_event_refresh_cuda", "phase2_floating_event_lifecycle_cuda",
          "phase2_floating_event_daily_bridge_cuda")


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, T / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def modules() -> SimpleNamespace:
    """The qualified GPU definition modules (appended to the END of sys.path so no other suite's helper is shadowed)."""
    for p in _PATHS:
        q = str(T / p)
        if q not in sys.path:
            sys.path.append(q)
    bd = _load("veg_lc_bridge_gpu_defs", "phase2_floating_event_daily_bridge_cuda/test_gpu_definitions.py")
    return SimpleNamespace(bd=bd, ld=bd.ld, rd=bd.ld.rd, e5=bd.e5)


def apply_estimate(boundary, binding):
    """The composed DEVICE/HOST model exactly as `vegetation_lifecycle_cuda.finish_cuda` forms it (mirrored here from the public pieces; the test then refuses at need - 1 and accepts the exact need)."""
    from maple_syrup import vegetation_lifecycle as vl
    from maple_syrup import vegetation_lifecycle_cuda as vlc

    root = boundary.root
    S = len(binding.enabled_species)
    held, free, wstage = vl.stream_numbers(binding)
    return vlc.estimate_apply_bytes_cuda(
        int(root.column.n_layers), int(root.column.n_cells), S, binding_resident_bytes=binding.resident_bytes, borrowed_bytes=binding.borrowed_bytes,
        old_retained_device_bytes=vl._retained(root.matrix_op) + vl._retained(root.pref_op), stream_per_next_held_bytes=held, stream_per_next_free_bytes=free, stream_weights_stage_bytes=wstage,
        k_suction_cell_arrays=int(root.config.matrix_conductivity_mode == "texture_ks"),
        a3_cell_map=binding.controls.a3_layout == "cell", max_rate_cell_map=binding.controls.max_rate_layout == "cell")
