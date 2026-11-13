"""Call-local verified-checkpoint capability and the exact tagged-float tree codec shared by the lifecycle checkpoint owners.

The lifecycle checkpoint codec (`lifecycle_checkpoint`) mints ONE `VerifiedCheckpoint` after its preflight, archive audits,
native load and identity checks, and passes it to the owner restore functions of `interevent_water`, `interevent_transpiration`
and `water_lifecycle`. Each owner validates its own sub-record and mints its OWN genuine object with its OWN private seal; this
module never holds or exports an owner seal and never touches an owner's registry.

Trust boundary (also in the docs): the capability binds the owner calls to one decoded checkpoint (record digests and the exact
decoded objects by identity) and expires when the codec call ends. It is NOT a signature, not a sandbox against code that edits
private module attributes, and not proof that a historical physical quiet audit ever ran: a checksum shows source consistency
and integrity only.
"""

from __future__ import annotations

import hashlib
import json
import math
import weakref
from types import MappingProxyType
from typing import Any

__all__ = ["CheckpointCapabilityError", "VerifiedCheckpoint", "close", "decode_tree", "encode_tree", "record_sha256", "require"]

_SEAL = object()
_MINTED = weakref.WeakSet()
MAX_TREE_DEPTH = 8
MAX_TREE_NODES = 512
MAX_STRING = 4096


class CheckpointCapabilityError(ValueError):
    """The capability or a tagged record is missing, forged, expired, malformed or not bound to the offered objects."""


# --- exact tagged-float trees ------------------------------------------------------------------------------------------------
def encode_tree(value: Any, _depth: int = 0, _budget: list | None = None) -> Any:
    """Plain Python tree -> JSON-safe tree. A float is `{"f": float.hex()}` (finite only); bool/int/str/None pass; `dict` (str
    keys), `list`, `tuple` recurse. Bounded depth and node count (refused, never truncated)."""
    budget = [0] if _budget is None else _budget
    budget[0] += 1
    if _depth > MAX_TREE_DEPTH or budget[0] > MAX_TREE_NODES:
        raise CheckpointCapabilityError("record tree exceeds its declared depth/node bound")
    if type(value).__module__ == "numpy" and hasattr(value, "item") and getattr(value, "shape", None) == ():
        value = value.item()  # a NumPy scalar becomes its exact Python value; arrays are never accepted
    if value is None or type(value) in (bool, int, str):
        if type(value) is str and len(value) > MAX_STRING:
            raise CheckpointCapabilityError("record string exceeds its declared bound")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise CheckpointCapabilityError("non-finite float in a record")
        return {"f": value.hex()}
    if type(value) is dict:
        out = {}
        for k, v in value.items():
            if type(k) is not str:
                raise CheckpointCapabilityError("record keys must be exact strings")
            out[k] = encode_tree(v, _depth + 1, budget)
        return out
    if type(value) in (list, tuple):
        return [encode_tree(v, _depth + 1, budget) for v in value]
    raise CheckpointCapabilityError(f"unsupported record value {type(value).__name__}")


def decode_tree(tree: Any, _depth: int = 0, _budget: list | None = None) -> Any:
    """Inverse of `encode_tree`. A raw JSON float, a non-canonical/non-finite hex, an unexpected `{"f": ...}` shape, or an
    over-bound tree is refused."""
    budget = [0] if _budget is None else _budget
    budget[0] += 1
    if _depth > MAX_TREE_DEPTH or budget[0] > MAX_TREE_NODES:
        raise CheckpointCapabilityError("record tree exceeds its declared depth/node bound")
    if tree is None or type(tree) in (bool, int):
        return tree
    if type(tree) is str:
        if len(tree) > MAX_STRING:
            raise CheckpointCapabilityError("record string exceeds its declared bound")
        return tree
    if type(tree) is float:
        raise CheckpointCapabilityError("raw JSON floats are refused: every float must be a canonical hex tag")
    if type(tree) is dict:
        if set(tree) == {"f"}:
            text = tree["f"]
            if type(text) is not str:
                raise CheckpointCapabilityError("float tag must hold a hex string")
            try:
                value = float.fromhex(text)
            except (ValueError, OverflowError):
                raise CheckpointCapabilityError("float tag is not a valid hex float") from None
            if not math.isfinite(value) or value.hex() != text:
                raise CheckpointCapabilityError("float tag is not a canonical finite hex float")
            return value
        return {k: decode_tree(v, _depth + 1, budget) for k, v in tree.items()}
    if type(tree) is list:
        return [decode_tree(v, _depth + 1, budget) for v in tree]
    raise CheckpointCapabilityError(f"unsupported record node {type(tree).__name__}")


def record_sha256(tree: Any) -> str:
    """Canonical SHA-256 of an ENCODED tree (so the digest is over the exact saved text, not a re-rendered float)."""
    try:
        text = json.dumps(tree, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise CheckpointCapabilityError(f"record is not strict JSON: {exc}") from None
    return hashlib.sha256(text.encode("ascii")).hexdigest()


# --- the capability ----------------------------------------------------------------------------------------------------------
class VerifiedCheckpoint:
    """Factory-only, immutable apart from its one `live` flag. Binds the lifecycle operator identity, the role, the manifest
    digest, the digest of each named encoded sub-record and the exact decoded objects for the duration of one codec call."""

    __slots__ = ("__weakref__", "_live", "_objects", "lop_sha256", "manifest_sha256", "parts", "role")

    def __init__(self, seal: object, *, role: str, lop_sha256: str, manifest_sha256: str, parts: dict, objects: dict) -> None:
        if seal is not _SEAL:
            raise CheckpointCapabilityError("VerifiedCheckpoint is minted only by the checkpoint codec's verified path")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "lop_sha256", lop_sha256)
        object.__setattr__(self, "manifest_sha256", manifest_sha256)
        object.__setattr__(self, "parts", MappingProxyType(dict(parts)))  # canonical owned, read-only map of scalar digests
        object.__setattr__(self, "_objects", MappingProxyType({k: tuple(v) for k, v in objects.items()}))
        object.__setattr__(self, "_live", [True])
        _MINTED.add(self)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("VerifiedCheckpoint is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("VerifiedCheckpoint is immutable")

    @property
    def live(self) -> bool:
        return bool(self._live[0])


def _issue(*, role: str, lop_sha256: str, manifest_sha256: str, records: dict, objects: dict) -> VerifiedCheckpoint:
    """INTERNAL issuer, called only by `lifecycle_checkpoint` after its own preflight/audit/decode/identity checks. It is not in
    `__all__` and is not a public API: it accepts whatever records it is given, so it is a trusted-Python-caller convention, not a
    proof that the codec ran and not a defence against code that edits private module state (there are no signatures and no
    sandbox). The owners still run their own physical audits and bind the exact decoded objects and records."""
    parts = {name: record_sha256(tree) for name, tree in records.items()}
    return VerifiedCheckpoint(_SEAL, role=role, lop_sha256=lop_sha256, manifest_sha256=manifest_sha256, parts=parts,
                              objects={k: tuple(v) for k, v in objects.items()})


def close(cap: VerifiedCheckpoint) -> None:
    """Expire the capability (the codec calls this in a `finally`)."""
    if type(cap) is VerifiedCheckpoint and cap in _MINTED:
        cap._live[0] = False


def require(cap: Any, part: str, record: Any, objects: tuple, *, lop_sha256: str | None = None) -> None:
    """Owner-side check: a genuine, live capability whose named part digest equals the digest of the offered ENCODED `record`
    and whose bound objects are exactly (`is`) the offered ones."""
    if type(cap) is not VerifiedCheckpoint or cap not in _MINTED:
        raise CheckpointCapabilityError("a verified-checkpoint capability minted by the checkpoint codec is required")
    if not cap.live:
        raise CheckpointCapabilityError("the verified-checkpoint capability has expired (it is call-local)")
    if lop_sha256 is not None and cap.lop_sha256 != lop_sha256:
        raise CheckpointCapabilityError("the capability is bound to a different lifecycle operator")
    if part not in cap.parts or cap.parts[part] != record_sha256(record):
        raise CheckpointCapabilityError(f"record {part!r} is not the one the capability verified")
    bound = cap._objects.get(part)
    if bound is None or len(bound) != len(objects) or any(a is not b for a, b in zip(bound, objects)):
        raise CheckpointCapabilityError(f"objects offered for {part!r} are not the exact decoded objects the capability bound")
