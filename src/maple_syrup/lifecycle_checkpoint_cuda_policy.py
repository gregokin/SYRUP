"""Closed private codec policy of the native9f CUDA lifecycle physical checkpoint (A6 G5a; UNQUALIFIED).

A sealed, immutable, module-private description of ONE codec version (schema, profile, manifest name, payload files, allowed (role, source kind) pairs, scalar-part table, manifest key set, derived allowed directory names).
Exactly three registered module literals exist (the third, `V3`, is the Phase 1.5 DAILY-lineage codec: schema `/3`, profile `…_v3`, manifest `lifecycle_checkpoint_cuda_v3.json`, the closed `daily` scalar part for every role, PARTIAL-storm support and its OWN reviewed helper-identity set; V1/V2 keep their C1 helper pins through the module constant of `lifecycle_checkpoint_cuda`, never overwritten): `V1` (identical to the constants of `lifecycle_checkpoint_cuda`; used by the V1 public entry points) and `V2` (schema `/2`, profile `…_v2`, manifest `lifecycle_checkpoint_cuda_v2.json`, the extra completed-`storm` pair; `implemented=True` since the V2 writer/loader branches exist in the engines and `lifecycle_checkpoint_cuda_v2`; an engine refuses any policy with `implemented=False`).
`policy_of` selects by OBJECT IDENTITY: a copy, a field-matching clone, a subclass instance or any other object is refused; nothing public accepts a policy.
This module is pure: no CuPy, no I/O, no device, no mutable global switch.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

from maple_syrup import lifecycle_checkpoint as lk

__all__ = ["CodecPolicyError", "V1", "V2", "V3", "V3_HELPER_PINS", "policy_of"]

_SEAL = object()


class CodecPolicyError(ValueError):
    """The selector received something other than a registered literal policy (or an unimplemented one was used)."""


class _CodecPolicy:
    __slots__ = ("name", "schema", "profile", "manifest_name", "files", "pairs", "roles", "scalar_parts", "manifest_keys", "allowed_names", "storm_allowed", "implemented", "daily", "partial_storm", "helper_pins")

    def __init__(self, seal: object, **kw: Any) -> None:
        if seal is not _SEAL:
            raise CodecPolicyError("a codec policy is created only by the policy module")
        for key in self.__slots__:
            value = kw.pop(key)
            if key == "scalar_parts":
                value = MappingProxyType({k: frozenset(v) for k, v in value.items()})
            elif key in ("manifest_keys", "allowed_names"):
                value = frozenset(value)
            elif key in ("files", "pairs", "roles"):
                value = tuple(value)
            elif key == "helper_pins":
                value = None if value is None else MappingProxyType({str(k): str(v) for k, v in value.items()})
            object.__setattr__(self, key, value)
        if kw:
            raise CodecPolicyError(f"unexpected policy fields {sorted(kw)}")

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("a codec policy is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("a codec policy is immutable")

    def __copy__(self) -> Any:
        raise CodecPolicyError("a codec policy cannot be copied")

    def __deepcopy__(self, memo: Any) -> Any:
        raise CodecPolicyError("a codec policy cannot be copied")

    def __reduce_ex__(self, protocol: Any) -> Any:
        raise CodecPolicyError("a codec policy cannot be serialized")

    def __repr__(self) -> str:
        return f"<CudaCodecPolicy {self.name}>"


_PAYLOAD_NATIVE = "maple_state.npz"
_PAYLOAD_COMPANION = "lifecycle_state.npz"
_V1_PAIRS = (("boundary", None), ("interevent", "storm_end"), ("interevent", "boundary_origin"))
_V1_PARTS = {("boundary", None): {"lifecycle"}, ("interevent", "storm_end"): {"lifecycle", "t_state", "source"}, ("interevent", "boundary_origin"): {"lifecycle", "t_state", "source"}}
_KEYS = frozenset(lk.MANIFEST_KEYS | {"source_kind"})

V1 = _CodecPolicy(_SEAL, name="v1", schema="maple-syrup-water-lifecycle-checkpoint-cuda/1", profile="native9f_dense_lifecycle_cuda_v1", manifest_name="lifecycle_checkpoint_cuda.json",
                  files=(_PAYLOAD_NATIVE, _PAYLOAD_COMPANION), pairs=_V1_PAIRS, roles=("boundary", "interevent"), scalar_parts=_V1_PARTS, manifest_keys=_KEYS,
                  allowed_names={"lifecycle_checkpoint_cuda.json", _PAYLOAD_NATIVE, _PAYLOAD_COMPANION}, storm_allowed=False, implemented=True, daily=False, partial_storm=False, helper_pins=None)
V2 = _CodecPolicy(_SEAL, name="v2", schema="maple-syrup-water-lifecycle-checkpoint-cuda/2", profile="native9f_dense_lifecycle_cuda_v2", manifest_name="lifecycle_checkpoint_cuda_v2.json",
                  files=(_PAYLOAD_NATIVE, _PAYLOAD_COMPANION), pairs=_V1_PAIRS + (("storm", None),), roles=("boundary", "interevent", "storm"),
                  scalar_parts={**_V1_PARTS, ("storm", None): {"lifecycle", "storm"}}, manifest_keys=_KEYS,
                  allowed_names={"lifecycle_checkpoint_cuda_v2.json", _PAYLOAD_NATIVE, _PAYLOAD_COMPANION}, storm_allowed=True, implemented=True, daily=False, partial_storm=False,
                  helper_pins=None)
# --- V3 (Phase 1.5 DAILY lineages) ---------------------------------------------------------------------------------------------------------------------------------------
#: The REVIEWED helper-identity set of the V3 codec: sha256 of each reused/changed helper FILE of the package, checked against the freshly STREAMED package inventory on every
#: save/load (see `lifecycle_checkpoint_cuda._fresh_source_pair`). It names the CURRENT Phase 1.5 `lifecycle_checkpoint.py` (carrying the daily policy/cursor engine), `recharge.py` and
#: `water_lifecycle.py` explicitly; it never overwrites or bypasses the C1 `HELPER_MODULE_PINS` of V1/V2, and there is no caller-supplied digest. A later edit of any pinned file
#: makes V3 refuse until this literal is re-reviewed (a test asserts equality with the files on disk).
V3_HELPER_PINS = {
    "layered_checkpoint.py": "4f4336d98f40febb56e15bcdb5a4c06e2e1dc1187c330d08b79849c4c0d72d51",
    "pond_checkpoint.py": "94cd2339566646ba44f2015dad13a67194c26a8dd3cc85b56916f62d2b4a704f",
    "lifecycle_checkpoint.py": "14a673937a4685f8b5c1dc4b42d5b1dca23a7c37a70c2f578119e48dd8883bc3",
    "checkpoint_capability.py": "36dc5d6a07245fca4badbadd0d754f2a250e688c2cc8b0b30d3a8c37c55ccbd6",
    "layered_storm.py": "bd125b743e58c588619a86ee952934deccea360ef3af215ae2b230af60d73c78",
    "recharge.py": "4b2606b0da58b5b12991012d7a9965ecbeadc33298224e0df1592d115c763de0",
    "water_lifecycle.py": "33ae320b623bb0404979655bfa0073d992988ca8309842ae4a8e7c081d8dc162",
}
_V3_PARTS = {key: set(parts) | {"daily"} for key, parts in {**_V1_PARTS, ("storm", None): {"lifecycle", "storm"}}.items()}
V3 = _CodecPolicy(_SEAL, name="v3", schema="maple-syrup-water-lifecycle-checkpoint-cuda/3", profile="native9f_dense_lifecycle_cuda_v3", manifest_name="lifecycle_checkpoint_cuda_v3.json",
                  files=(_PAYLOAD_NATIVE, _PAYLOAD_COMPANION), pairs=_V1_PAIRS + (("storm", None),), roles=("boundary", "interevent", "storm"), scalar_parts=_V3_PARTS, manifest_keys=_KEYS,
                  allowed_names={"lifecycle_checkpoint_cuda_v3.json", _PAYLOAD_NATIVE, _PAYLOAD_COMPANION}, storm_allowed=True, implemented=True, daily=True, partial_storm=True,
                  helper_pins=V3_HELPER_PINS)
_REGISTERED = (V1, V2, V3)


def policy_of(policy: Any) -> _CodecPolicy:
    """`None` -> V1 (the default of every V1 entry point); otherwise the argument must BE one of the three registered literal objects. No hashing/iteration/formatting of the argument."""
    if policy is None:
        return V1
    if policy is V1 or policy is V2 or policy is V3:
        return policy
    raise CodecPolicyError("unknown codec policy (only the registered literal policies are accepted)")
