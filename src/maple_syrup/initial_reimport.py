"""CW-B21: INITIAL-only reimport of a persisted resident-window assembly under a reviewed MAPLE source change.

`restore_initial_assembly` refuses an initial assembly whose recorded MAPLE source differs from the imported
MAPLE, and stays that strict. This module is the one separate, explicit path for reusing such an assembly's
immutable generation-zero deep version under a NEW MAPLE source without rebuilding its row tiles or copying the
deep store:

* The source change must be described by an exact, reviewed transition artifact
  (`maple-syrup-initial-source-transition/1`): full producer and consumer package file manifests and package
  digests, every changed file (added / removed / modified, with hashes), the pyproject change, this tool's
  digest, the exact original initial operands it applies to, and a review record. Everything except the review
  is recomputed here from the ACTUAL producer tree, the ACTUAL imported consumer MAPLE and the ACTUAL original
  files, and must be equal; a missing or extra change, or a manifest that does not describe the actual sources,
  is refused. `propose_source_transition` writes the unreviewed form (review null), which this module refuses
  to import; the root adds the review after independent review.
* Only a genuinely initial state is accepted: deep version 0 with no parent, snapshot time 0 and step 0, zero
  commits, a genuinely zero compiled ledger, zero mobile sediment, and exactly dry water for a case that
  authored no initial depth. Nothing is cleared or rewritten to pass these checks.
* The current MAPLE snapshot loader (full streamed column validation with prefix re-derivation) and the genuine
  compiled first tile (actual configuration, classes, datum, availability fractions and `BedContext`) are used.
* The original initial operands are hashed before and after; they, the original receipt and binding, and the
  deep version are never written. The new metadata directory must be absent. It receives byte-exact copies of
  the two operand files, a NEW receipt whose `source_maple` names the current consumer and whose
  `initial_reimport` record separately keeps the original producer, the consumer, the transition digest and the
  original initial-binding identity, and then (last) a new binding. The snapshot keeps its external reference to
  the original immutable deep version. On failure the binding is absent and a `REIMPORT_FAILED.json` record is
  left, which `restore_initial_assembly` refuses.

The new binding gives a different initial identity, so a wet checkpoint of a run started from the original
metadata does not resume under the new metadata; wet restart identities are unchanged and strict.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from maple_syrup.case_import import _json_safe
from maple_syrup.chastre_case import hash_tree
from maple_syrup.provenance import (
    SOURCE_DIGEST_ALGORITHM,
    capture_maple_provenance,
    environment_record,
    source_tree_digest,
)
from maple_syrup.sediment_bed import BedContext, BedIntegrationError, BedState
from maple_syrup.windowed_case import (
    INITIAL_BINDING_NAME,
    INITIAL_BINDING_SCHEMA,
    INITIAL_FILES,
    INITIAL_REIMPORT_FAILED_NAME,
    INITIAL_REIMPORT_SCHEMA,
    WindowedCaseAssembly,
    _check_rows,
    _check_tile,
    _sha256_bytes,
    initial_binding_identity,
    initial_identity_record,
    restore_initial_assembly,
)

TRANSITION_SCHEMA = 'maple-syrup-initial-source-transition/1'
#: The SYRUP files whose exact content is bound by the transition's tool digest.
TOOL_FILES = ('initial_reimport.py', 'windowed_case.py')
REVIEW_KEYS = ('reviewer', 'reviewed_at', 'evidence')
_COPIED = ('initial_resident.npz', 'initial_baseline.npz')
_BASELINE_FIELDS = ('reference_elevation_m', 'elevation_offset_m', 'committed_active_layer_mass_by_class_kg',
                    'committed_voxel_mass_by_class_kg')
_READ_ONLY = 0o444


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def tool_digest() -> dict[str, str]:
    here = Path(__file__).resolve().parent
    return {name: _sha256_file(here / name) for name in TOOL_FILES}


def _canonical(value):
    return json.loads(json.dumps(_json_safe(value), sort_keys=True, allow_nan=False))


def _files_digest(files: dict[str, str]) -> str:
    """The `source_tree_digest` algorithm (v1) over an explicit `{relpath: sha256}` manifest."""
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(rel.encode('utf-8'))
        h.update(b'\0')
        h.update(files[rel].encode('ascii'))
        h.update(b'\n')
    return h.hexdigest()


def _side(record: dict[str, Any], files: dict[str, str], metadata: dict[str, Any]) -> dict[str, Any]:
    return {'package_source_digest': {k: record[k] for k in
                                      ('root', 'algorithm', 'digest_sha256', 'file_count', 'total_bytes')},
            'files': dict(sorted(files.items())), 'metadata_file_sha256': dict(metadata)}


def _read_original(store_dir: Path) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    """Hash the original binding and every bound operand, then read the receipt from bytes that hash as bound."""
    identity = initial_binding_identity(store_dir)
    receipt_bytes = (store_dir / 'assembly_receipt.json').read_bytes()
    if _sha256_bytes(receipt_bytes) != identity['initial_files_sha256']['assembly_receipt.json']:
        raise BedIntegrationError('original initial assembly receipt changed')
    receipt = json.loads(receipt_bytes)
    if receipt.get('status') != 'ready':
        raise BedIntegrationError('original initial assembly is not ready')
    for marker in ('ASSEMBLY_FAILED.json', INITIAL_REIMPORT_FAILED_NAME):
        if (store_dir / marker).exists():
            raise BedIntegrationError('original initial assembly directory carries a failure record')
    return identity, receipt, receipt_bytes


def _producer_side(receipt: dict[str, Any], producer_package_dir) -> dict[str, Any]:
    """The ACTUAL original producer tree, required to be the one the original receipt names and digests."""
    old = receipt['source_maple']
    recorded = old['package_source_digest']
    package_dir = Path(producer_package_dir).resolve()
    if str(package_dir) != recorded['root'] or str(package_dir) != old['dependency']['package_dir']:
        raise BedIntegrationError('producer package directory is not the one recorded by the original receipt')
    tree = source_tree_digest(package_dir)
    actual = tree.as_record()
    if {k: actual[k] for k in ('algorithm', 'digest_sha256', 'file_count', 'total_bytes')} != \
            {k: recorded[k] for k in ('algorithm', 'digest_sha256', 'file_count', 'total_bytes')}:
        raise BedIntegrationError('actual producer MAPLE source does not match the original receipt')
    if recorded['algorithm'] != SOURCE_DIGEST_ALGORITHM:
        raise BedIntegrationError('original receipt uses an unsupported source digest algorithm')
    root = old['dependency']['source_root']
    metadata = {}
    if root is not None:
        pyproject = Path(root) / 'pyproject.toml'
        metadata['pyproject.toml'] = _sha256_file(pyproject) if pyproject.is_file() else None
    if metadata != old['metadata_file_sha256']:
        raise BedIntegrationError('actual producer pyproject does not match the original receipt')
    return _side(actual, dict(tree.files), metadata)


def _consumer_side(expected_maple_root) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """The ACTUAL imported MAPLE: (transition side, provenance without the file list, runtime record)."""
    import maple

    from maple_syrup.dependency import resolve_maple_dependency

    provenance = capture_maple_provenance(resolve_maple_dependency(expected_maple_root), include_files=True)
    digest = provenance['package_source_digest']
    files = {rel: sha for rel, sha in digest['files']}
    side = _side(digest, files, provenance['metadata_file_sha256'])
    record = dict(provenance)
    record['package_source_digest'] = {k: v for k, v in digest.items() if k != 'files'}
    runtime = {'maple_module_file': str(Path(maple.__file__).resolve()), 'sys_path': list(sys.path),
               'environment': environment_record()}
    return side, record, runtime


def _changes(producer: dict[str, Any], consumer: dict[str, Any]) -> dict[str, Any]:
    a, b = producer['files'], consumer['files']
    metadata = {name: {'producer': producer['metadata_file_sha256'].get(name),
                       'consumer': consumer['metadata_file_sha256'].get(name)}
                for name in sorted(set(producer['metadata_file_sha256']) | set(consumer['metadata_file_sha256']))
                if producer['metadata_file_sha256'].get(name) != consumer['metadata_file_sha256'].get(name)}
    return {'added': {p: b[p] for p in sorted(set(b) - set(a))},
            'removed': {p: a[p] for p in sorted(set(a) - set(b))},
            'modified': {p: {'producer': a[p], 'consumer': b[p]} for p in sorted(set(a) & set(b)) if a[p] != b[p]},
            'metadata': metadata}


def _expected_transition(store_dir: Path, producer_package_dir, expected_maple_root):
    identity, receipt, receipt_bytes = _read_original(store_dir)
    producer = _producer_side(receipt, producer_package_dir)
    consumer, consumer_record, runtime = _consumer_side(expected_maple_root)
    for side in (producer, consumer):
        if _files_digest(side['files']) != side['package_source_digest']['digest_sha256']:
            raise BedIntegrationError('package manifest does not reproduce its package digest')
    changes = _changes(producer, consumer)
    if not any(changes.values()):
        raise BedIntegrationError('MAPLE source is unchanged; use restore_initial_assembly, not a reimport')
    expected = {'schema': TRANSITION_SCHEMA,
                'subject': {'original_initial_binding_sha256': identity['initial_binding_sha256'],
                            'original_initial_files_sha256': identity['initial_files_sha256'],
                            'deep_manifest_digest': receipt['deep_manifest_digest'],
                            'deep_generation': receipt['deep_generation']},
                'producer': producer, 'consumer': consumer, 'changes': changes, 'tool_sha256': tool_digest()}
    return _canonical(expected), identity, receipt, receipt_bytes, consumer_record, runtime


def propose_source_transition(store_dir, producer_package_dir, out_path, *, expected_maple_root=None):
    """Write the UNREVIEWED transition artifact (review null) for the root to review; it is not importable as is."""
    out = Path(out_path)
    expected = _expected_transition(Path(store_dir), producer_package_dir, expected_maple_root)[0]
    payload = (json.dumps({**expected, 'review': None}, sort_keys=True, indent=1, allow_nan=False) + '\n').encode()
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o644)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(payload)
    return expected


def _mismatch(name, expected, actual):
    if name == 'changes':
        missing = [f'{k}:{p}' for k in expected for p in expected[k] if p not in actual.get(k, {})]
        extra = [f'{k}:{p}' for k in actual for p in actual[k] if p not in expected.get(k, {})]
        if missing:
            return f'transition is missing actual change(s) {missing[:5]}'
        if extra:
            return f'transition approves change(s) that are not actual {extra[:5]}'
        return 'transition change hashes differ from the actual sources'
    return {'producer': 'transition producer manifest does not match the actual original producer',
            'consumer': 'transition consumer manifest does not match the actual imported MAPLE',
            'tool_sha256': 'transition tool digest does not match this reimport tool',
            'subject': 'transition does not name these original initial operands',
            'schema': 'transition schema mismatch'}[name]


def _final_source_guard(expected, consumer_record, expected_maple_root) -> None:
    """CW-B25: re-capture the ACTUAL imported MAPLE (package manifest, digest, pyproject/metadata and the full
    provenance record written as `source_maple`) and this tool's digest after the acceptance restore, and refuse
    unless they equal what was validated and recorded. Runtime/environment details are not compared."""
    side, record, _ = _consumer_side(expected_maple_root)
    if _canonical(side) != expected['consumer'] or _canonical(record) != _canonical(consumer_record):
        raise BedIntegrationError('imported MAPLE source changed before the reimport completed')
    if _canonical(tool_digest()) != expected['tool_sha256']:
        raise BedIntegrationError('reimport tool changed before the reimport completed')


def _validated_transition(path, expected) -> tuple[dict[str, Any], str]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise BedIntegrationError('source-transition artifact is missing (or a symlink)')
    data = path.read_bytes()
    try:
        artifact = json.loads(data)
    except ValueError as exc:
        raise BedIntegrationError('source-transition artifact is not JSON') from exc
    if not isinstance(artifact, dict) or set(artifact) != {*expected, 'review'}:
        raise BedIntegrationError('source-transition artifact keys differ from the schema')
    for name in ('schema', 'subject', 'producer', 'consumer', 'changes', 'tool_sha256'):
        if artifact[name] != expected[name]:
            raise BedIntegrationError(_mismatch(name, expected[name], artifact[name]))
    review = artifact['review']
    if (not isinstance(review, dict) or set(review) != set(REVIEW_KEYS)
            or not all(isinstance(review[k], str) and review[k].strip() for k in REVIEW_KEYS)):
        raise BedIntegrationError('source-transition artifact is not reviewed (review record required)')
    return artifact, _sha256_bytes(data)


def _array_sha256(a) -> str:
    """SHA-256 of a host array's C-order bytes, read through its buffer (no full bytes copy when contiguous)."""
    return hashlib.sha256(np.ascontiguousarray(a).data).hexdigest()


def _leaf_digests(state) -> list[tuple[str, tuple[int, ...], str]]:
    from maple.core.backend import array_leaves

    return [(str(a.dtype), tuple(a.shape), _array_sha256(a)) for a in array_leaves(state)]


def _validate_initial(tiles, geometry, grain_classes, store_dir, receipt, binding, max_host_bytes):
    """Genuine current-MAPLE validation of the original operands; returns (assembly, report). Clears nothing."""
    from maple.case_tools.compilers.case_compiler import load_compiled_case
    from maple.core.parameters import config_to_dict
    from maple.core.parameters.case_initialization import (
        resolve_initial_available_fraction_array,
    )
    from maple.core.types.sediment_ledger import (
        process_index,
        zeros_sediment_ledger_state,
    )
    from maple.core.types.topographic_commit import CommittedTopographyState
    from maple.surface.topographic_commit.windowed_commit import (
        is_windowed_already_committed,
    )
    from maple.surface.voxels.windowed_snapshot import load_windowed_state_snapshot

    if len(tiles) != len(receipt['source_tiles']):
        raise BedIntegrationError('initial reimport tile coverage changed')
    for tile, old in zip(tiles, receipt['source_tiles'], strict=True):
        if (tile.row_start, tile.row_stop, str(tile.case_dir.resolve()), tile.files) != (
                old['row_start'], old['row_stop'], old['case_dir'], old['files']) or hash_tree(tile.case_dir) != tile.files:
            raise BedIntegrationError('initial reimport original source tile changed')
    first = load_compiled_case(tiles[0].case_dir)
    if hash_tree(tiles[0].case_dir) != tiles[0].files:
        raise BedIntegrationError('initial reimport first tile changed during load')
    resolution = first.config.numerics.mass_resolution_kg
    snap = load_windowed_state_snapshot(store_dir / 'initial_resident.npz', geometry=geometry,
                                        grain_classes=grain_classes, mass_resolution_kg=resolution,
                                        max_bytes=max_host_bytes)
    column = snap['voxel_column']
    signature = _check_tile(first, tiles[0], geometry, grain_classes, column.n_voxels,
                            datum_offset_m=receipt['datum_offset_m'])
    if _canonical(config_to_dict(dataclasses.replace(first.config, geometry=geometry))) != receipt['maple_configuration']:
        raise BedIntegrationError('initial reimport configuration changed')
    version = column.deep_store
    if (receipt['deep_generation'] != 0 or column.generation != 0 or version.version_id != 0
            or version.parent_version_id is not None or version.parent_manifest_digest is not None):
        raise BedIntegrationError('initial reimport requires the generation-zero deep version (evolved version refused)')
    if column.manifest_digest != receipt['deep_manifest_digest'] or version.manifest_digest != receipt['deep_manifest_digest']:
        raise BedIntegrationError('initial reimport deep version differs from the original receipt')
    if snap['time_s'] != 0.0 or snap['step'] != 0:
        raise BedIntegrationError('initial reimport requires an initial snapshot (time 0, step 0)')
    if (binding['commit_count'] != 0 or binding['last_commit_time_s'] != 0.0
            or signature['commit_count'] != 0 or signature['last_commit_time_s'] != 0.0):
        raise BedIntegrationError('initial reimport requires zero commits')
    water = snap['water']
    if np.any(water.mobile_mass_by_cell_class_kg != 0):
        raise BedIntegrationError('initial reimport refuses mobile sediment (not an initial state)')
    if first.water_initial_depth_spec.depth_m is not None:
        raise BedIntegrationError('authored initial water depth cannot be proven initial from one tile; refused')
    if np.any(water.depth_m != 0):
        raise BedIntegrationError('initial reimport refuses wet water (the case authored a dry initial state)')
    rows = slice(tiles[0].row_start, tiles[0].row_stop)
    tile_fields = ((snap['active_layer'].mass_kg, first.active_layer.mass_kg),
                   (snap['sediment_availability'].available_mass_kg, first.sediment_availability.available_mass_kg),
                   (snap['sediment_availability'].bound_mass_kg, first.sediment_availability.bound_mass_kg),
                   (water.depth_m, first.water.depth_m),
                   (water.mobile_mass_by_cell_class_kg, first.water.mobile_mass_by_cell_class_kg))
    for persisted, compiled in tile_fields:
        if np.ascontiguousarray(persisted[rows]).tobytes() != np.ascontiguousarray(compiled).tobytes():
            raise BedIntegrationError('initial reimport resident state differs from the genuine compiled first tile')
    import zipfile

    baseline_path = store_dir / 'initial_baseline.npz'
    with zipfile.ZipFile(baseline_path) as z:
        if sum(i.file_size for i in z.infolist()) > max_host_bytes:
            raise BedIntegrationError('initial baseline exceeds host budget')
    with np.load(baseline_path, allow_pickle=False) as z:
        if set(z.files) != set(_BASELINE_FIELDS) | {'initial_available_fraction'}:
            raise BedIntegrationError('initial reimport baseline fields differ')
        arrays = {n: z[n].copy() for n in z.files}
    class_ids = [g.class_id for g in grain_classes.classes]
    fractions = np.asarray(resolve_initial_available_fraction_array(first.sediment_availability_spec, geometry, class_ids))
    stored = arrays['initial_available_fraction']
    if fractions.dtype != stored.dtype or fractions.shape != stored.shape or fractions.tobytes() != stored.tobytes():
        raise BedIntegrationError('current MAPLE resolves different initial availability fractions')
    committed = CommittedTopographyState(*(arrays[n] for n in _BASELINE_FIELDS), binding['last_commit_time_s'],
                                         binding['commit_count'])
    ledger = zeros_sediment_ledger_state(geometry.ny, geometry.nx, len(grain_classes.classes),
                                         present_processes=(process_index('water_erosion_deposition'),))
    state = BedState(column, snap['active_layer'], water, ledger, committed, snap['sediment_availability'])
    if not is_windowed_already_committed(ledger, committed, column, state.active_layer, geometry, resolution,
                                        binding['last_commit_time_s']):
        raise BedIntegrationError('initial saved baseline is not the actual committed bed')
    context = BedContext(geometry, grain_classes, resolution, first.config.topographic_commit,
                         first.config.avalanching, stored)
    report = {'deep_version': {'store_dir': str(version.store_dir.resolve()), 'version_id': version.version_id,
                               'manifest_digest': version.manifest_digest,
                               'manifest_file_sha256': _sha256_file(version.manifest_path)},
              'first_tile_rows': [tiles[0].row_start, tiles[0].row_stop],
              'checks': ['current MAPLE snapshot loader: streamed full column validation and prefix re-derivation',
                         'genuine compiled first tile: geometry, classes, datum, configuration, zero ledger/history',
                         'every original tile file manifest re-hashed',
                         'deep version 0, no parent, digest equal to the original receipt',
                         'snapshot time 0 and step 0; zero commits',
                         'zero mobile sediment; exactly dry water for a case that authored no depth',
                         'first-tile rows of active, available, bound, depth and mobile byte-equal to the compiled tile',
                         'availability fractions re-resolved by current MAPLE byte-equal to the stored fractions',
                         'stored baseline is the actual committed bed']}
    return WindowedCaseAssembly(state, context, receipt), report


def _write_new(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _copy_exact(source: Path, target: Path, expected_sha256: str) -> None:
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    h = hashlib.sha256()
    with source.open('rb') as src, os.fdopen(fd, 'wb') as dst:
        for block in iter(lambda: src.read(1 << 20), b''):
            h.update(block)
            dst.write(block)
        dst.flush()
        os.fsync(dst.fileno())
    if h.hexdigest() != expected_sha256:
        raise BedIntegrationError(f'original {source.name} changed while being copied')


def reimport_initial_assembly(tiles, geometry, grain_classes, *, store_dir, metadata_dir, transition_path,
                              producer_package_dir, datum_offset_m, max_host_bytes,
                              expected_maple_root=None) -> dict[str, Any]:
    """Bind the original generation-zero initial assembly to the current MAPLE under a reviewed transition.

    Returns the new initial identity (`initial_identity_record` form) with the original identity, transition
    digest and new metadata directory. The new directory then restores through the ordinary strict
    `restore_initial_assembly`. Refuses before writing anything unless every check passes."""
    tiles = tuple(tiles)
    _check_rows(tiles, geometry)
    if isinstance(datum_offset_m, bool) or not isinstance(datum_offset_m, (int, float)) or not np.isfinite(datum_offset_m):
        raise BedIntegrationError('datum_offset_m must be the finite verified case datum')
    if isinstance(max_host_bytes, bool) or not isinstance(max_host_bytes, int) or max_host_bytes < 1:
        raise BedIntegrationError('max_host_bytes must be a positive integer')
    store = Path(store_dir).resolve()
    target = Path(metadata_dir)
    if target.exists() or target.is_symlink():
        raise BedIntegrationError('initial reimport metadata directory already exists')
    target_resolved = target.resolve()
    if target_resolved == store or target_resolved.is_relative_to(store):
        raise BedIntegrationError('initial reimport metadata directory must lie outside the original store')
    expected, before, receipt, receipt_bytes, consumer_record, runtime = _expected_transition(
        store, producer_package_dir, expected_maple_root)
    if receipt['plan']['estimated_host_peak_bytes'] > max_host_bytes:
        raise BedIntegrationError('initial assembly plan is above the supplied host budget')
    if receipt.get('datum_offset_m') != float(datum_offset_m):
        raise BedIntegrationError('initial assembly datum differs from the verified case datum; refused before use')
    artifact, transition_sha256 = _validated_transition(transition_path, expected)
    assembly, report = _validate_initial(tiles, geometry, grain_classes, store, receipt, before['binding'],
                                         max_host_bytes)
    leaves = _leaf_digests(assembly.state)
    fractions = _array_sha256(assembly.context.initial_available_fraction)
    del assembly
    after = initial_binding_identity(store)
    if initial_identity_record(after) != initial_identity_record(before):
        raise BedIntegrationError('original initial operands changed during the reimport validation')
    _, consumer_again, _ = _consumer_side(expected_maple_root)
    if consumer_again['package_source_digest'] != consumer_record['package_source_digest']:
        raise BedIntegrationError('imported MAPLE source changed during the reimport')

    record = {'schema': INITIAL_REIMPORT_SCHEMA,
              'scope': 'INITIAL-only: generation-zero deep version, time 0, step 0, zero commits/ledger/mobile, '
                       'dry initial water; wet snapshots and evolved versions are refused',
              'producer': receipt['source_maple'], 'consumer': consumer_record, 'consumer_runtime': runtime,
              'transition': {'path': str(Path(transition_path).resolve()), 'sha256': transition_sha256,
                             'review': artifact['review'], 'changes': artifact['changes']},
              'original_initial_binding': initial_identity_record(before),
              'original_store_dir': str(store),
              'original_assembly_receipt_sha256': _sha256_bytes(receipt_bytes),
              'copied_operands': {name: before['initial_files_sha256'][name] for name in _COPIED},
              'tool_sha256': expected['tool_sha256'], 'validation': report,
              'physical_leaf_sha256': [list(x) for x in leaves], 'initial_available_fraction_sha256': fractions}
    new_receipt = _canonical({**receipt, 'source_maple': consumer_record, 'initial_reimport': record})
    target.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(target)  # exclusive: refuses a directory created since the check
    written = []
    try:
        for name in _COPIED:
            _copy_exact(store / name, target / name, before['initial_files_sha256'][name])
            written.append(target / name)
        receipt_path = target / 'assembly_receipt.json'
        _write_new(receipt_path, (json.dumps(new_receipt, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())
        written.append(receipt_path)
        binding = {'schema': INITIAL_BINDING_SCHEMA,
                   'files': {name: _sha256_file(target / name) for name in INITIAL_FILES},
                   'last_commit_time_s': before['binding']['last_commit_time_s'],
                   'commit_count': before['binding']['commit_count']}
        for path in written:
            path.chmod(_READ_ONLY)
        binding_path = target / INITIAL_BINDING_NAME
        _write_new(binding_path, (json.dumps(binding, sort_keys=True, allow_nan=False) + '\n').encode())
        written.append(binding_path)
        binding_path.chmod(_READ_ONLY)
        # Acceptance through the ordinary strict path, under the current MAPLE.
        restored = restore_initial_assembly(tiles, geometry, grain_classes, store_dir=target,
                                            max_host_bytes=max_host_bytes, datum_offset_m=datum_offset_m)
        if (_leaf_digests(restored.state) != leaves
                or _array_sha256(restored.context.initial_available_fraction) != fractions):
            raise BedIntegrationError('reimported initial state is not byte-identical to the original operands')
        new_identity = restored.initial_binding
        del restored
        final = initial_binding_identity(store)
        if initial_identity_record(final) != initial_identity_record(before):
            raise BedIntegrationError('original initial operands changed during the reimport')
        _final_source_guard(expected, consumer_record, expected_maple_root)  # last check before success
    except BaseException as exc:
        binding_path = target / INITIAL_BINDING_NAME
        if binding_path in written:
            binding_path.unlink()
        _write_new(target / INITIAL_REIMPORT_FAILED_NAME,
                   (json.dumps({'status': 'failed', 'error': str(exc), 'executable_binding_written': False},
                               sort_keys=True) + '\n').encode())
        raise
    return {'metadata_dir': str(target_resolved), 'initial_binding': new_identity,
            'original_initial_binding': initial_identity_record(before), 'transition_sha256': transition_sha256}


def _chastre_inputs(args):
    from maple_syrup.chastre_case import verify_chastre_case
    from maple_syrup.windowed_case import chastre_row_tiles

    verified = verify_chastre_case(args.case_dir, expected_maple_root=args.expected_maple_root,
                                   allow_maple_source_change=args.allow_case_maple_source_change,
                                   reload_tiles=False, progress=lambda m: print(m, file=sys.stderr))
    return (chastre_row_tiles(verified), verified.case.config.geometry, verified.case.config.grain_classes,
            float(verified.binding['grid']['datum_offset_m']))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog='python -m maple_syrup.initial_reimport', description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='command', required=True)
    propose = sub.add_parser('propose', help='write the UNREVIEWED transition artifact for root review')
    run = sub.add_parser('import', help='reimport a Chastre initial assembly under a reviewed transition')
    for q in (propose, run):
        q.add_argument('--store-dir', required=True)
        q.add_argument('--producer-package-dir', required=True)
        q.add_argument('--expected-maple-root')
    propose.add_argument('--out', required=True)
    run.add_argument('--case-dir', required=True)
    run.add_argument('--metadata-dir', required=True)
    run.add_argument('--transition', required=True)
    run.add_argument('--max-host-bytes', type=int, required=True)
    # The EXISTING Chastre case-verification option, passed to verify_chastre_case only; it never relaxes the
    # initial-assembly source check, which is satisfied solely by the reviewed transition.
    run.add_argument('--allow-case-maple-source-change', action='store_true')
    args = p.parse_args(argv)
    if args.command == 'propose':
        propose_source_transition(args.store_dir, args.producer_package_dir, args.out,
                                  expected_maple_root=args.expected_maple_root)
        print(json.dumps({'proposal': str(Path(args.out).resolve()), 'review': None}))
        return 0
    tiles, geometry, grain_classes, datum = _chastre_inputs(args)
    out = reimport_initial_assembly(tiles, geometry, grain_classes, store_dir=args.store_dir,
                                    metadata_dir=args.metadata_dir, transition_path=args.transition,
                                    producer_package_dir=args.producer_package_dir, datum_offset_m=datum,
                                    max_host_bytes=args.max_host_bytes, expected_maple_root=args.expected_maple_root)
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
