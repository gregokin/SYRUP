"""CC2: explicit CUDA CONTINUED-ORIGIN physical child profile (Dense and native Windowed device beds). AUTHOR-WRITTEN, NOT RUN ON A DEVICE (Root qualifies it on hardware).

The CUDA analogue of `floating_continued_checkpoint` (CC1, CPU): a Pond state continued from a lifecycle boundary carries the LAST actual routing topology (possibly generation > 0), the held source and the GLOBAL original datum, with
truthful fresh event-local counters (`n_refreshes == 0`). The historical CUDA child (`maple-syrup-floating-pond-checkpoint/7|8`) and CPU child (`/1-/6`), and the CPU continued profile (`maple-syrup-floating-continued-pond-checkpoint/1|2`),
are unchanged and refuse this family (and it refuses them) by schema/profile/keys.

New closed profile family: schemas `maple-syrup-floating-continued-cuda-pond-checkpoint/1` (Dense device bed) and `/2` (native Windowed), profile `floating-vertical-continued-origin/2-cuda`; manifest = the CUDA `/7|/8` closed key set +
`continued_origin`; companion = the CUDA `/7|/8` companion + the host array `co_routing_source` (carried routing origin, O(N padded) host bytes, modelled in the declared table bytes of both pools' composed admission). The native MAPLE snapshot
carries the new schema tag in its provenance (a historical loader's provenance check therefore refuses it). Saved/loaded through the EXISTING `floating_checkpoint_cuda` save/load with the optional explicit `original_topology=` (default
`None` = the historical behaviour; the CPU-neutral validator copy `floating_continued_checkpoint.validate_continued_state` supplies the two declared rule differences: datum attested against the caller's generation-0 original, and the local
counter rule `n_refreshes == generation - carried generation`). Everything else is the historical code: device gate FIRST, two-pool admission before any D2H/H2D/read/copy (reserves once), header/table preflight before decode,
SourcePair and kernel descriptor, Windowed published resident window + EXTERNAL deep-store reference with the mapping-cache bound, `VerifiedFloatingInputsCuda` -> `prepare_from_verified_inputs` -> `resume_scope` -> `adopt_resumed` LAST.

`read_continued_origin_cuda` is a HOST metadata reader (no device is touched): it runs the CPU shared core `read_origin_core` (strict manifest, seal, schema/profile/keys, identity, SourcePair, the kernel/ABI descriptor, structure, admission, companion hash,
zip + table + EVERY `.npy` header audit, THEN one small array) with the CUDA schemas. Not claimed: outer container, lifecycle/cursor, quiet/completion proof, a GPU qualification.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from maple_syrup import floating_checkpoint as fc
from maple_syrup import floating_checkpoint_cuda as fcc
from maple_syrup import floating_continued_checkpoint as cc
from maple_syrup.checkpoint import DEFAULT_MAX_CHECKPOINT_BYTES
from maple_syrup.floating_continued_checkpoint import ContinuedOrigin, origin_topology
from maple_syrup.pond_event import PondEvent, PondEventState
from maple_syrup.pond_network import PondTopology

__all__ = ["CONT_GPU_PROFILE", "CONT_GPU_SCHEMA", "CONT_GPU_WINDOWED_SCHEMA", "ContinuedOrigin", "load_floating_continued_checkpoint_cuda", "origin_topology", "read_continued_origin_cuda",
           "save_floating_continued_checkpoint_cuda"]

CONT_GPU_SCHEMA = fcc.CONT_GPU_SCHEMA
CONT_GPU_WINDOWED_SCHEMA = fcc.CONT_GPU_WINDOWED_SCHEMA
CONT_GPU_PROFILE = fcc.CONT_GPU_PROFILE


def _kernel_check(data: dict) -> None:
    if data.get("kernel") != fcc._kernel():
        raise fc.FloatingCheckpointError("kernel/ABI descriptor mismatch: the CUDA kernels differ from the ones that wrote the checkpoint (refused before any payload is read)")


def read_continued_origin_cuda(path: Any, expected_identity: dict, *, available_bytes_fn: Callable[[], int], reserve_bytes: int = 0, max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES) -> ContinuedOrigin:
    """Host metadata read of a CUDA continued-origin file (module docstring); never touches a device. Returns the pure `ContinuedOrigin` value (it authorises nothing: the loader re-verifies against the event built from it)."""
    return cc.read_origin_core(path, expected_identity, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve_bytes, max_bytes=max_bytes, schemas=fcc.CONT_READ_SCHEMAS, dense_keys=fcc.CONT_MANIFEST_KEYS,
                               windowed_keys=fcc.CONT_WINDOWED_MANIFEST_KEYS, profile=fcc.CONT_GPU_PROFILE, windowed_schemas=(fcc.CONT_GPU_WINDOWED_SCHEMA,), extra_check=_kernel_check)


def save_floating_continued_checkpoint_cuda(path: Any, state: Any, event: PondEvent, identity: dict, *, original_topology: PondTopology, **quota: Any) -> Path:
    """`floating_checkpoint_cuda.save_floating_pond_checkpoint_cuda(..., original_topology=...)`; `quota` are its keywords (two pools, reserves, budgets, `max_bytes`)."""
    if original_topology is None:
        raise fc.FloatingCheckpointError("original_topology is required by the continued-origin profile")
    return fcc.save_floating_pond_checkpoint_cuda(path, state, event, identity, original_topology=original_topology, **quota)


def load_floating_continued_checkpoint_cuda(path: Any, event: PondEvent, expected_identity: dict, *, original_topology: PondTopology, **quota: Any) -> PondEventState:
    """`floating_checkpoint_cuda.load_floating_pond_checkpoint_cuda(..., original_topology=...)`."""
    if original_topology is None:
        raise fc.FloatingCheckpointError("original_topology is required by the continued-origin profile")
    return fcc.load_floating_pond_checkpoint_cuda(path, event, expected_identity, original_topology=original_topology, **quota)
