"""Test-only helpers of tests/checkpoint_compat: installers for a SEPARATE synthetic `maple.core.types.sediment_ledger`
module object or a broken import of it (pure registry / refusal tests only; the real module is never edited), checkpoint
manifest rewriting, and small codec helpers. Nothing here is a production fallback or a physical stand-in class."""

from __future__ import annotations

import importlib.abc
import importlib.util
import json
import sys
import types
from pathlib import Path

LEDGER_MODULE = "maple.core.types.sediment_ledger"
MISSING = object()


class SyntheticLedgerImportFailure(RuntimeError):
    pass


class _BrokenLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise SyntheticLedgerImportFailure("synthetic failure while executing maple.core.types.sediment_ledger")


class _BrokenLedgerFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == LEDGER_MODULE:
            return importlib.util.spec_from_loader(fullname, _BrokenLoader())
        return None


class LedgerBinding:
    """Every change goes through the fixture's one MonkeyPatch, undone (and the restoration proven) at teardown."""

    def __init__(self, patch, real):
        self._patch, self.real = patch, real

    def module(self, **bindings):
        """Install a synthetic module object carrying every public binding of the real module, then `bindings`
        (`MISSING` deletes the name). Returns it. The real module object is not touched."""
        module = types.ModuleType(LEDGER_MODULE)
        module.__dict__.update({k: v for k, v in vars(self.real).items() if not k.startswith("__")})
        for name, value in bindings.items():
            if value is MISSING:
                module.__dict__.pop(name, None)
            else:
                module.__dict__[name] = value
        self._patch.setitem(sys.modules, LEDGER_MODULE, module)
        return module

    def none_entry(self):
        """`sys.modules[name] = None`: the import machinery raises ModuleNotFoundError."""
        self._patch.setitem(sys.modules, LEDGER_MODULE, None)

    def broken_import(self):
        """A present module whose execution raises `SyntheticLedgerImportFailure`."""
        self._patch.delitem(sys.modules, LEDGER_MODULE)
        self._patch.setattr(sys, "meta_path", [_BrokenLedgerFinder(), *sys.meta_path])


def compact_present() -> bool:
    import importlib

    return "CompactSedimentLedgerState" in vars(importlib.import_module(LEDGER_MODULE))


def rewrite_pond(path: Path, manifest) -> None:
    """Edit a pond checkpoint manifest and re-seal its own hash (payload hashes unchanged), so the refusal under test
    comes from the deeper validator and not from the manifest hash."""
    from maple_syrup import pond_checkpoint as pc

    meta = json.loads((path / pc.MANIFEST_NAME).read_text())
    manifest(meta)
    (path / pc.MANIFEST_NAME).write_text(json.dumps(pc._seal_manifest(meta), indent=1, sort_keys=True))


def rewrite_legacy(path: Path, manifest) -> None:
    """Edit a `checkpoint.json` manifest (its payload tree carries no self-hash; payload file hashes are unchanged)."""
    meta = json.loads((path / "checkpoint.json").read_text())
    manifest(meta)
    (path / "checkpoint.json").write_text(json.dumps(meta))


def tree_bytes(path: Path) -> dict:
    return {p.name: p.read_bytes() for p in sorted(Path(path).iterdir()) if p.is_file()}


class ArrayStore:
    """`add_array` / `get_array` callbacks of the pond native codec over an in-memory dict."""

    def __init__(self):
        self.arrays = {}

    def add(self, array):
        name = f"native_{len(self.arrays):04d}"
        self.arrays[name] = array
        return name

    def get(self, name):
        return self.arrays[name]
