"""Closed private codec policy of the native9f CUDA lifecycle physical checkpoint (A6 G5a; UNQUALIFIED).

A sealed, immutable, module-private description of ONE codec version (schema, profile, manifest name, payload files, allowed (role, source kind) pairs, scalar-part table, manifest key set, derived allowed directory names).
Exactly two registered module literals exist: `V1` (identical to the constants of `lifecycle_checkpoint_cuda`; used by the V1 public entry points) and `V2` (schema `/2`, profile `…_v2`, manifest `lifecycle_checkpoint_cuda_v2.json`, the extra completed-`storm` pair; `implemented=True` since the V2 writer/loader branches exist in the engines and `lifecycle_checkpoint_cuda_v2`; an engine refuses any policy with `implemented=False`).
`policy_of` selects by OBJECT IDENTITY: a copy, a field-matching clone, a subclass instance or any other object is refused; nothing public accepts a policy.
This module is pure: no CuPy, no I/O, no device, no mutable global switch.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

from maple_syrup import lifecycle_checkpoint as lk

__all__ = ["CodecPolicyError", "V1", "V2", "policy_of"]

_SEAL = object()


class CodecPolicyError(ValueError):
    """The selector received something other than a registered literal policy (or an unimplemented one was used)."""


class _CodecPolicy:
    __slots__ = ("name", "schema", "profile", "manifest_name", "files", "pairs", "roles", "scalar_parts", "manifest_keys", "allowed_names", "storm_allowed", "implemented")

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
                  allowed_names={"lifecycle_checkpoint_cuda.json", _PAYLOAD_NATIVE, _PAYLOAD_COMPANION}, storm_allowed=False, implemented=True)
V2 = _CodecPolicy(_SEAL, name="v2", schema="maple-syrup-water-lifecycle-checkpoint-cuda/2", profile="native9f_dense_lifecycle_cuda_v2", manifest_name="lifecycle_checkpoint_cuda_v2.json",
                  files=(_PAYLOAD_NATIVE, _PAYLOAD_COMPANION), pairs=_V1_PAIRS + (("storm", None),), roles=("boundary", "interevent", "storm"),
                  scalar_parts={**_V1_PARTS, ("storm", None): {"lifecycle", "storm"}}, manifest_keys=_KEYS,
                  allowed_names={"lifecycle_checkpoint_cuda_v2.json", _PAYLOAD_NATIVE, _PAYLOAD_COMPANION}, storm_allowed=True, implemented=True)
_REGISTERED = (V1, V2)


def policy_of(policy: Any) -> _CodecPolicy:
    """`None` -> V1 (the default of every V1 entry point); otherwise the argument must BE one of the two registered literal objects. No hashing/iteration/formatting of the argument."""
    if policy is None:
        return V1
    if policy is V1 or policy is V2:
        return policy
    raise CodecPolicyError("unknown codec policy (only the registered literal policies are accepted)")
