"""Phase 2.5 vegetation lifecycle, CUDA: HOST-checkable static contract (runs on the CPU, not hardware acceptance). The real GPU definitions are in `test_cuda_hardware.py` (authored, NOT RUN, gated by SYRUP_VEG_LIFECYCLE_RELEASE_GPU)."""
from __future__ import annotations

import inspect

from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_event_lifecycle_cuda as flc
from maple_syrup import floating_event_refresh_cuda as fercu
from maple_syrup import native_floating_soil_cuda as ncuda
from maple_syrup import vegetation_lifecycle as vl
from maple_syrup import vegetation_lifecycle_cuda as vlc


def test_static_the_cuda_carry_hooks_exist_and_the_default_branches_are_unchanged():
    assert "vegetation=fm.held_cell_vegetation_factor(payload.matrix_op)" in inspect.getsource(fercu)
    dev = inspect.getsource(ncuda._device_op)
    assert "vegetation=None" in dev and "if vegetation is None else vegetation" in dev
    assert "vegetation=occurrence.vegetation" in inspect.getsource(flc.close_cuda)  # the GPU close constructor does not drop the record
    assert "arrays_equal_cuda" in inspect.getsource(lc._check_bound_cover) and "splash_cover_device" in inspect.getsource(lc._check_bound_cover)


def test_static_the_cuda_transition_order_gate_then_checks_then_admission_then_native_pieces_then_head_last():
    src = inspect.getsource(vl.apply_vegetation_generation)
    assert src.index("flc._gate(root.column)") < src.index("_require_live_leaf(binding)") < src.index("verify_binding") < src.index("hash_device_surface") < src.index("return expected") < src.index("finish_cuda")
    fin = inspect.getsource(vlc.finish_cuda)
    order = [fin.index(t) for t in ("fcu.admit", "prepare_vegetation_law", "verify_prepared_law", "rebind_floating_matrix_history", "ncuda._up", "mint_and_install")]
    assert order == sorted(order)  # admission BEFORE every allocation; the shared mint replaces the head LAST
    assert "lineage._head = head" in inspect.getsource(vl.mint_and_install) and "lineage._head" not in fin
    assert "host_active_mask=host_mask" in fin and "stream_hash" not in fin and "column.mask_sha256" not in fin  # the device mask identity is S0's business (sealed host mask), never compared with LEAF's byte hash here


def test_static_the_surface_hash_gates_the_host_pool_before_the_first_transfer_and_never_builds_a_whole_host_surface():
    src = inspect.getsource(vlc.hash_device_surface)
    assert src.index("fcu.admit") < src.index("to_host(") and "surface_hash_chunk_bytes" in src and "asnumpy" not in src and ".get()" not in src


def test_static_the_daily_and_event_consumers_use_the_resident_device_covers():
    rec_src = inspect.getsource(vl.VegetationRecord)
    assert "shade_cover_device" in rec_src and "def bound_shade" in rec_src


def test_static_each_hashed_chunk_is_released_before_the_next_transfer_allocates():
    src = inspect.getsource(vlc.hash_device_surface)
    loop = src[src.index("for lo in range"):]
    assert loop.index("h.update(") < loop.index("del chunk") and loop.count("to_host(") == 1  # one chunk bound at a time: released before the next iteration's copy


def test_the_public_error_policy_wraps_the_native_memory_refusal_family_only():
    from maple_syrup import floating_soil_layers as fsl

    assert not issubclass(fsl.FloatingMemoryError, fsl.FloatingSoilError) and issubclass(fsl.FloatingMemoryError, MemoryError)  # why the wrapper must name it (the confirmed hardware finding)
    for fn in (vlc.finish_cuda, vlc.hash_device_surface, vl.apply_vegetation_generation):
        src = inspect.getsource(fn)
        assert "FloatingMemoryError" in src and "except Exception" not in src and "except MemoryError" not in src and "except BaseException" not in src


def test_the_apply_model_has_an_explicit_hash_stage_of_the_actual_chunk_buffer():
    assert vlc.surface_hash_chunk_bytes(12) == 96 and vlc.surface_hash_chunk_bytes(300000) == vlc.SURFACE_HASH_CHUNK_BYTES and vlc.surface_hash_chunk_bytes(1) == 8
    est = vlc.estimate_apply_bytes_cuda(3, 12, 1, binding_resident_bytes=1000, borrowed_bytes=500, old_retained_device_bytes=2000, stream_per_next_held_bytes=400, stream_per_next_free_bytes=300)
    assert est["host_stages"]["hash"] == 1000 + 500 + 96 and est["surface_hash_host_buffer_bytes"] == 96  # N = 12 needs 96 B, not a fixed 1 MiB
    assert est["peak_host_bytes"] == max(est["host_stages"].values()) and set(est["host_stages"]) == {"hash", "law", "rebind", "seal"}
    big = vlc.estimate_apply_bytes_cuda(3, 300000, 1, binding_resident_bytes=1000, borrowed_bytes=500, old_retained_device_bytes=2000, stream_per_next_held_bytes=400, stream_per_next_free_bytes=300)
    assert big["host_stages"]["hash"] == 1500 + vlc.SURFACE_HASH_CHUNK_BYTES  # a large surface stays bounded by the declared 1 MiB chunk


def test_contract_the_public_surface_hash_returns_the_counted_bytes_after_the_loop():
    import ast
    import textwrap

    fn = ast.parse(textwrap.dedent(inspect.getsource(vlc.hash_device_surface))).body[0]
    last = fn.body[-1]
    assert isinstance(last, ast.Return) and isinstance(last.value, ast.Name) and last.value.id == "moved"  # the final statement is `return moved`, AFTER the loop (a regression removed it once)
    loop = next(n for n in fn.body if isinstance(n, ast.For))
    assert not any(isinstance(n, ast.Return) for n in ast.walk(loop))  # not returned from inside the loop
    ann = fn.returns
    assert isinstance(ann, ast.Name) and ann.id == "int"
