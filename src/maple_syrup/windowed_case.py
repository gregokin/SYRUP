"""Stream genuine compiled MAPLE row tiles into one resident-window water bed.

One compiled row tile and one deep-store chunk are live at a time. Initial
active mass, availability, water and committed baselines are copied verbatim;
MAPLE owns all buried mass and supplies the height/inventory implementations.
This is a constructor, not an evolving storm driver. Its successful receipt is
written last. A failure can leave an explicitly failed staging directory; it
never returns an executable partial bed or overwrites an existing store.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from maple_syrup.case_import import _json_safe
from maple_syrup.chastre_case import hash_tree
from maple_syrup.provenance import capture_maple_provenance
from maple_syrup.sediment_bed import BedContext, BedIntegrationError, BedState


@dataclass(frozen=True)
class CompiledRowTile:
    row_start: int
    row_stop: int
    case_dir: Path
    files: dict[str, str]


@dataclass(frozen=True)
class WindowedCaseAssembly:
    state: BedState
    context: BedContext
    receipt: dict[str, Any]
    # CW-B15: content identity of the persisted initial operands (`initial_binding_identity`), set by
    # `restore_initial_assembly`; a fresh assembly carries None until `persist_initial_assembly` returns it.
    initial_binding: dict[str, Any] | None = None


#: The persisted initial operands bound by `initial_binding.json` (CW-B15: all four files are made read-only 0444).
INITIAL_FILES = ('initial_resident.npz', 'initial_baseline.npz', 'assembly_receipt.json')
INITIAL_BINDING_NAME = 'initial_binding.json'
INITIAL_BINDING_SCHEMA = 'maple-syrup-initial-window-assembly/1'
#: CW-B21: failure record of an INITIAL-only reimport (`maple_syrup.initial_reimport`); restore refuses it.
INITIAL_REIMPORT_FAILED_NAME = 'REIMPORT_FAILED.json'
#: CW-B21: schema of the receipt's `initial_reimport` record (original producer, consumer, transition, identity).
INITIAL_REIMPORT_SCHEMA = 'maple-syrup-initial-reimport/1'
_READ_ONLY = 0o444


def _sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def initial_binding_identity(store_dir) -> dict[str, Any]:
    """Content identity of a persisted initial assembly: the SHA-256 of the exact `initial_binding.json` bytes
    (parsed from those same bytes) and the SHA-256 of every bound file, each re-hashed now and required to equal
    the binding. Returns `{'schema', 'initial_binding_sha256', 'initial_files_sha256', 'binding'}`; the caller
    puts the first three in the run identity. Refuses a missing / malformed binding or any changed file."""
    from maple_syrup.checkpoint import sha256_file

    root = Path(store_dir)
    path = root / INITIAL_BINDING_NAME
    if not path.is_file() or path.is_symlink():
        raise BedIntegrationError('initial assembly binding is missing (or a symlink)')
    data = path.read_bytes()
    try:
        binding = json.loads(data)
    except ValueError as exc:
        raise BedIntegrationError('initial assembly binding is not JSON') from exc
    if (not isinstance(binding, dict) or binding.get('schema') != INITIAL_BINDING_SCHEMA
            or not isinstance(binding.get('files'), dict) or set(binding['files']) != set(INITIAL_FILES)):
        raise BedIntegrationError('initial assembly binding schema/files mismatch')
    files = {}
    for name in INITIAL_FILES:
        file = root / name
        if not file.is_file() or file.is_symlink():
            raise BedIntegrationError(f'initial assembly file {name} is missing (or a symlink)')
        files[name] = sha256_file(file)
        if files[name] != binding['files'][name]:
            raise BedIntegrationError('initial assembly restart source changed')
    return {'schema': INITIAL_BINDING_SCHEMA, 'initial_binding_sha256': _sha256_bytes(data),
            'initial_files_sha256': files, 'binding': binding}


def initial_identity_record(identity: dict[str, Any]) -> dict[str, Any]:
    """The JSON part of `initial_binding_identity` that enters a run identity (the parsed binding is omitted;
    its bytes are covered by `initial_binding_sha256`)."""
    return {k: identity[k] for k in ('schema', 'initial_binding_sha256', 'initial_files_sha256')}


def chastre_row_tiles(verified) -> tuple[CompiledRowTile, ...]:
    """Descriptors from a freshly verified Chastre import; no fabricated case."""
    return tuple(CompiledRowTile(t['row_start'], t['row_stop'], verified.case_dir / t['dir'], dict(t['files']))
                 for t in verified.binding['tiles'])


def _payload(value):
    return json.dumps(_json_safe(dataclasses.asdict(value)), sort_keys=True, allow_nan=False)


def _check_rows(tiles, geometry):
    next_row = 0
    if not tiles:
        raise BedIntegrationError('no compiled row tiles')
    for t in tiles:
        if not isinstance(t, CompiledRowTile):
            raise BedIntegrationError('expected CompiledRowTile descriptors')
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (t.row_start, t.row_stop)):
            raise BedIntegrationError('row bounds must be non-Boolean integers')
        if t.row_start != next_row or not (t.row_start < t.row_stop <= geometry.ny):
            raise BedIntegrationError('row tiles must cover the geometry exactly once in contiguous order')
        if not t.files:
            raise BedIntegrationError('tile needs a complete source-file manifest')
        next_row = t.row_stop
    if next_row != geometry.ny:
        raise BedIntegrationError('row tiles do not cover the geometry')


def _check_tile(case, tile, geometry, grain_classes, n_voxels, *, datum_offset_m, signature=None):
    from maple.core.backend import array_leaves, array_namespace

    cfg = case.config
    elevation = case.case_import.elevation
    transforms = () if elevation is None else elevation.transforms
    datum_steps = [t for t in transforms if t.name == 'datum_offset']
    if len(datum_steps) > 1:
        raise BedIntegrationError('multiple imported datum transforms require a resolved datum binding')
    source_datum = 0.0
    if datum_steps:
        params = datum_steps[0].parameters
        if params.get('reference') != 'none':
            raise BedIntegrationError('datum transform needs an explicitly resolved absolute offset')
        source_datum = params['offset_m']
    if source_datum != datum_offset_m:
        raise BedIntegrationError('declared global datum differs from the actual compiled tile transform')
    expected_geometry = dataclasses.replace(geometry, ny=tile.row_stop - tile.row_start)
    if cfg.geometry != expected_geometry:
        raise BedIntegrationError('compiled tile geometry differs from the declared global geometry')
    if _payload(cfg.grain_classes) != _payload(grain_classes):
        raise BedIntegrationError('compiled tile class ordering or physical properties differ')
    expected = (tile.row_stop - tile.row_start, geometry.nx, n_voxels, len(grain_classes.classes))
    if case.voxel_column.mass_kg.shape != expected:
        raise BedIntegrationError('compiled tile voxel allocation differs')
    if case.water is None:
        raise BedIntegrationError('compiled tile needs actual MAPLE water state')
    if array_namespace(*array_leaves(case.voxel_column, case.active_layer, case.water, case.sediment_availability)) is not np:
        raise BedIntegrationError('compiled row tiles must be host MAPLE state')
    if any(np.any(a != 0) for a in array_leaves(case.sediment_ledger)):
        raise BedIntegrationError('constructor requires genuinely zero compiled ledger/history; nothing is cleared')
    if float(case.sediment_ledger.largest_residual_kg) != 0.0:
        raise BedIntegrationError('compiled ledger has a residual history')
    spec = case.sediment_availability_spec
    if spec.builder is not None or spec.builder_by_class:
        raise BedIntegrationError('availability builders require original resolved maps; unsupported by this constructor')
    keys = ('numerics', 'topographic_commit', 'avalanching', 'water_coupling', 'time_and_forcing')
    current = {k: _payload(getattr(cfg, k)) for k in keys}
    current['availability'] = _payload(spec)
    current['water_initial_depth'] = _payload(case.water_initial_depth_spec)
    current['last_commit_time_s'] = case.committed_topography.last_commit_time_s
    current['commit_count'] = case.committed_topography.commit_count
    if signature is not None and current != signature:
        raise BedIntegrationError('row tiles have inconsistent water/commit/availability/numerical configuration')
    return current


def assemble_windowed_case(tiles, geometry, grain_classes, n_voxels: int, *, store_dir: str | Path,
                          datum_offset_m: float, max_store_bytes: int, max_host_bytes: int,
                          window_voxels: int = 4, chunk_columns: int = 256,
                          mapping_cache_chunks: int = 8, progress=None) -> WindowedCaseAssembly:
    """Load one verified actual MAPLE compiled band at a time, and return one bed.

    max_host_bytes controls an explicit conservative planning estimate, not an
    OS allocation limit. Measured CPU/GPU peaks remain acceptance requirements.
    The source manifests, all scientific options, exact geometry/classes, zero
    ledger and initial physical state are checked before the final receipt.
    Source files are hashed again after load, so changed late tiles are refused.
    """
    from maple.case_tools.compilers.case_compiler import load_compiled_case
    from maple.core.parameters.case_initialization import (
        resolve_initial_available_fraction_array,
    )
    from maple.core.types.active_layer import ActiveLayerState
    from maple.core.types.sediment_availability import SedimentAvailabilityState
    from maple.core.types.sediment_ledger import (
        process_index,
        zeros_sediment_ledger_state,
    )
    from maple.core.types.topographic_commit import CommittedTopographyState
    from maple.core.types.water import WaterState
    from maple.surface.topographic_commit.windowed_commit import (
        windowed_committed_state_from_dense_baseline,
    )
    from maple.surface.voxels.deep_store import DeepStoreBuilder, DeepStoreHeader
    from maple.surface.voxels.windowed_inventory import windowed_column_from_version
    from maple.surface.voxels.windowed_prefix import packet_terms_for

    from maple_syrup.dependency import resolve_maple_dependency

    source_maple = capture_maple_provenance(resolve_maple_dependency())
    tiles = tuple(tiles)
    _check_rows(tiles, geometry)
    root = Path(store_dir)
    if root.exists():
        raise BedIntegrationError('refusing an existing store/staging directory')
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (max_host_bytes, max_store_bytes)):
        raise BedIntegrationError('memory and store budgets must be positive integers')
    header = DeepStoreHeader.from_geometry(geometry, n_voxels=n_voxels,
                                           class_ids=tuple(g.class_id for g in grain_classes.classes),
                                           window_voxels=window_voxels, chunk_columns=chunk_columns,
                                           datum_offset_m=datum_offset_m)
    n, nc = header.n_cells, header.n_classes
    band_bytes = max(t.row_stop-t.row_start for t in tiles) * geometry.nx * n_voxels * nc * 8
    window_bytes = n * window_voxels * nc * 8
    packet_bytes = n * packet_terms_for(n_voxels, nc) * 8
    # Planner covers column/active/availability/water/baselines/fractions,
    # compact ledger + conversion scratch, and the existing loader's tile peak.
    estimated_peak = window_bytes * 3 + packet_bytes * 3 + 30 * n * nc * 8 + 10 * band_bytes
    plan = {'logical_dense_voxel_bytes_not_allocated': header.logical_bytes,
            'source_band_voxel_bytes': band_bytes, 'resident_window_bytes': window_bytes,
            'prefix_packet_bytes': packet_bytes, 'chunk_buffer_bytes': header.chunk_columns*n_voxels*nc*8,
            'estimated_host_peak_bytes': estimated_peak, 'max_host_bytes': max_host_bytes,
            'max_store_bytes': max_store_bytes,
            'estimate_note': 'conservative planning estimate, not measured RSS or a GPU-fit claim'}
    if estimated_peak > max_host_bytes or header.logical_bytes > max_store_bytes:
        raise BedIntegrationError('assembly resource preflight exceeds the supplied memory/store budget')
    # Check all source files before staging output. Manifest size is tiny next to bands.
    for t in tiles:
        if hash_tree(t.case_dir) != t.files:
            raise BedIntegrationError(f'compiled source tile manifest changed: {t.case_dir}')
    root.parent.mkdir(parents=True, exist_ok=True)
    shape = (geometry.ny, geometry.nx)
    arrays = {k: np.empty((*shape, nc), dtype=np.float64) for k in
              ('active', 'available', 'bound', 'mobile', 'committed_active', 'committed_voxel')}
    arrays.update({k: np.empty(shape, dtype=np.float64) for k in ('depth', 'reference', 'offset')})
    signature = None
    config = availability_spec = None
    sources = []
    builder = None
    try:
        for index, t in enumerate(tiles):
            if progress is not None:
                progress(f'loading actual MAPLE row tile {index+1}/{len(tiles)}')
            case = load_compiled_case(t.case_dir)
            signature = _check_tile(case, t, geometry, grain_classes, n_voxels, datum_offset_m=datum_offset_m, signature=signature)
            if hash_tree(t.case_dir) != t.files:
                raise BedIntegrationError(f'compiled source tile changed during load: {t.case_dir}')
            if config is None:
                config, availability_spec = case.config, case.sediment_availability_spec
                builder = DeepStoreBuilder(root, header, geometry, config.numerics.mass_resolution_kg,
                                           max_store_bytes=max_store_bytes,
                                           mapping_cache_chunks=mapping_cache_chunks)
            builder.add_columns(case.voxel_column.mass_kg.reshape(-1, n_voxels, nc))
            sl = slice(t.row_start, t.row_stop)
            values = {'active': case.active_layer.mass_kg,
                      'available': case.sediment_availability.available_mass_kg,
                      'bound': case.sediment_availability.bound_mass_kg,
                      'mobile': case.water.mobile_mass_by_cell_class_kg, 'depth': case.water.depth_m,
                      'reference': case.committed_topography.reference_elevation_m,
                      'offset': case.committed_topography.elevation_offset_m,
                      'committed_active': case.committed_topography.committed_active_layer_mass_by_class_kg,
                      'committed_voxel': case.committed_topography.committed_voxel_mass_by_class_kg}
            for key, value in values.items():
                arrays[key][sl] = value
            sources.append({'row_start': t.row_start, 'row_stop': t.row_stop,
                            'case_dir': str(t.case_dir.resolve()), 'files': t.files,
                            'case_identity_sha256': case.provenance_record['case_identity_sha256'],
                            'artifact_sha256': case.provenance_record['artifact_sha256']})
            del values, case  # release this band before the next loader runs
        version = builder.finish()
        column = windowed_column_from_version(version, geometry, config.numerics.mass_resolution_kg,
                                               class_ids=header.class_ids, with_prefix_packets=True)
        active = ActiveLayerState(arrays['active'])
        committed = CommittedTopographyState(arrays['reference'], arrays['offset'], arrays['committed_active'],
                                             arrays['committed_voxel'], signature['last_commit_time_s'],
                                             signature['commit_count'])
        committed, regrouping = windowed_committed_state_from_dense_baseline(committed, column, active, geometry)
        ledger = zeros_sediment_ledger_state(geometry.ny, geometry.nx, nc,
                                             present_processes=(process_index('water_erosion_deposition'),))
        fractions = resolve_initial_available_fraction_array(availability_spec, geometry, list(header.class_ids))
        state = BedState(column, active, WaterState(arrays['depth'], arrays['mobile']), ledger, committed,
                         SedimentAvailabilityState(arrays['available'], arrays['bound']))
        context = BedContext(geometry, grain_classes, config.numerics.mass_resolution_kg,
                             config.topographic_commit, config.avalanching, fractions)
        from maple.core.parameters import config_to_dict

        receipt = {'maple_configuration': config_to_dict(dataclasses.replace(config, geometry=geometry)),
                   'schema': 'maple_syrup.windowed_case.v1', 'status': 'ready', 'plan': plan,
                   'source_tiles': sources, 'source_maple': source_maple,
                   'deep_manifest_digest': version.manifest_digest, 'deep_generation': version.version_id,
                   'datum_offset_m': datum_offset_m, 'baseline_regrouping': regrouping,
                   'initial_state_rule': 'actual compiled MAPLE fields copied verbatim; genuinely zero ledger',
                   'not_claimed': ['complete storm', 'restart', 'measured peak memory', 'GPU fit', 'wind handoff']}
        receipt=json.loads(json.dumps(_json_safe(receipt),allow_nan=False))
        (root/'assembly_receipt.json').write_text(json.dumps(_json_safe(receipt), sort_keys=True, indent=2)+'\n')
        return WindowedCaseAssembly(state, context, receipt)
    except BaseException as exc:
        if root.exists():
            (root/'ASSEMBLY_FAILED.json').write_text(json.dumps({'status': 'failed', 'error': str(exc),
                                                              'executable_bed_returned': False})+'\n')
        raise


def assembly_to_backend(assembly: WindowedCaseAssembly, xp) -> WindowedCaseAssembly:
    """Move resident MAPLE state explicitly; keep the opaque disk store on host."""
    from maple.core.backend import to_device, to_device_tree
    from maple.surface.voxels.windowed_pending import require_published_column
    from maple.surface.voxels.windowed_prefix import move_prefix_packets

    require_published_column(assembly.state.voxel_column, operation='assembly_to_backend')
    state = to_device_tree(assembly.state, xp)
    column = move_prefix_packets(state.voxel_column, assembly.state.voxel_column, xp)
    state = dataclasses.replace(state, voxel_column=column)
    context = dataclasses.replace(assembly.context,
                                  initial_available_fraction=to_device(assembly.context.initial_available_fraction, xp))
    return dataclasses.replace(assembly, state=state, context=context)


def persist_initial_assembly(assembly: WindowedCaseAssembly) -> dict[str, Any]:
    """Write bounded initial restart operands once, after valid assembly.

    CW-B15: the receipt, both operand files and the binding are made read-only (0444) and the content identity
    (`initial_binding_identity`, re-hashed from disk) is returned for the run identity. Existing deep versions and
    store files are never modified or removed."""
    from maple.core.backend import to_host
    from maple.surface.voxels.windowed_snapshot import save_windowed_state_snapshot

    state,context=assembly.state,assembly.context
    root=state.voxel_column.deep_store.store_dir
    snapshot=root/'initial_resident.npz'
    baseline=root/'initial_baseline.npz'
    manifest=root/'initial_binding.json'
    if any(p.exists() for p in (snapshot,baseline,manifest)):
        raise BedIntegrationError('initial assembly restart files already exist')
    save_windowed_state_snapshot(state.voxel_column,state.active_layer,0.,snapshot,step=0,
                  geometry=context.geometry,mass_resolution_kg=context.mass_resolution_kg,
                  sediment_availability=state.sediment_availability,water=state.water,
                  provenance={'assembly_receipt':'assembly_receipt.json'})
    fields=('reference_elevation_m','elevation_offset_m','committed_active_layer_mass_by_class_kg',
            'committed_voxel_mass_by_class_kg')
    arrays={n:np.asarray(to_host(getattr(state.committed_topography,n))) for n in fields}
    arrays['initial_available_fraction']=np.asarray(to_host(context.initial_available_fraction))
    np.savez(baseline,**arrays)
    from maple_syrup.checkpoint import sha256_file

    binding={'schema':'maple-syrup-initial-window-assembly/1',
             'files':{p.name:sha256_file(p) for p in (snapshot,baseline,root/'assembly_receipt.json')},
             'last_commit_time_s':state.committed_topography.last_commit_time_s,
             'commit_count':state.committed_topography.commit_count}
    manifest.write_text(json.dumps(binding,sort_keys=True,allow_nan=False)+'\n')
    for p in (root/'assembly_receipt.json',snapshot,baseline,manifest):
        p.chmod(_READ_ONLY)
    identity=initial_binding_identity(root)
    if identity['initial_files_sha256']!=binding['files']:
        raise BedIntegrationError('initial assembly files changed while being persisted')
    return identity


def restore_initial_assembly(tiles, geometry, grain_classes, *, store_dir, max_host_bytes,
                             datum_offset_m) -> WindowedCaseAssembly:
    """Rebind the immutable original bed without constructing all row tiles.

    All original source manifests and genuine current MAPLE source are checked.
    One actual first tile supplies validated configuration; persisted initial
    resident arrays and baselines supply the exact original numerical state.

    CW-B15: `datum_offset_m` is the caller's verified datum and must equal the
    receipt's before anything is loaded. The binding and its three files are
    hashed BEFORE use and again AFTER every read; any difference refuses. The
    returned `initial_binding` content identity must enter the run identity, so
    a coherently re-hashed edit of the persisted operands changes the identity
    (a checkpoint of the original run then refuses to resume) instead of
    silently passing as the same run.
    """
    from maple.case_tools.compilers.case_compiler import load_compiled_case
    from maple.core.parameters import config_to_dict
    from maple.core.types.sediment_ledger import (
        process_index,
        zeros_sediment_ledger_state,
    )
    from maple.core.types.topographic_commit import CommittedTopographyState
    from maple.surface.voxels.windowed_snapshot import load_windowed_state_snapshot

    from maple_syrup.dependency import resolve_maple_dependency

    tiles=tuple(tiles);_check_rows(tiles,geometry)
    if isinstance(datum_offset_m,bool) or not isinstance(datum_offset_m,(int,float)) or not np.isfinite(datum_offset_m):
        raise BedIntegrationError('datum_offset_m must be the finite verified case datum')
    root=Path(store_dir)
    before=initial_binding_identity(root)  # hash-before: binding bytes and every bound file
    binding=before['binding']
    receipt_bytes=(root/'assembly_receipt.json').read_bytes()
    if _sha256_bytes(receipt_bytes)!=before['initial_files_sha256']['assembly_receipt.json']:
        raise BedIntegrationError('initial assembly restart source changed')
    receipt=json.loads(receipt_bytes)
    if receipt.get('status')!='ready' or receipt['plan']['estimated_host_peak_bytes']>max_host_bytes:
        raise BedIntegrationError('initial assembly not ready or above supplied host budget')
    if (root/'ASSEMBLY_FAILED.json').exists() or (root/INITIAL_REIMPORT_FAILED_NAME).exists():
        raise BedIntegrationError('initial assembly store carries a failure record')
    if receipt.get('datum_offset_m')!=float(datum_offset_m):
        raise BedIntegrationError('initial assembly datum differs from the verified case datum; refused before use')
    source=capture_maple_provenance(resolve_maple_dependency())
    if source['package_source_digest']!=receipt['source_maple']['package_source_digest']:
        raise BedIntegrationError('initial assembly MAPLE source changed')
    if 'initial_reimport' in receipt:
        # CW-B21: a reimported receipt must name its consumer as source_maple and bind the copied operands.
        record=receipt['initial_reimport']
        if (not isinstance(record,dict) or record.get('schema')!=INITIAL_REIMPORT_SCHEMA
                or record.get('consumer',{}).get('package_source_digest')!=receipt['source_maple']['package_source_digest']
                or record.get('copied_operands')!={n:binding['files'][n] for n in ('initial_resident.npz','initial_baseline.npz')}):
            raise BedIntegrationError('initial reimport record is inconsistent with its receipt and binding')
    if len(tiles)!=len(receipt['source_tiles']):
        raise BedIntegrationError('initial assembly tile coverage changed')
    for tile,old in zip(tiles,receipt['source_tiles'],strict=True):
        if (tile.row_start,tile.row_stop,str(tile.case_dir.resolve()),tile.files)!=(
                old['row_start'],old['row_stop'],old['case_dir'],old['files']) or hash_tree(tile.case_dir)!=tile.files:
            raise BedIntegrationError('initial assembly original source tile changed')
    first=load_compiled_case(tiles[0].case_dir)
    snap=load_windowed_state_snapshot(root/'initial_resident.npz',geometry=geometry,grain_classes=grain_classes,
                                      mass_resolution_kg=first.config.numerics.mass_resolution_kg,max_bytes=max_host_bytes)
    _check_tile(first,tiles[0],geometry,grain_classes,snap['voxel_column'].n_voxels,
                 datum_offset_m=receipt['datum_offset_m'])
    if json.loads(json.dumps(_json_safe(config_to_dict(dataclasses.replace(first.config,geometry=geometry)))))!=receipt['maple_configuration']:
        raise BedIntegrationError('initial assembly configuration changed')
    if snap['voxel_column'].manifest_digest!=receipt['deep_manifest_digest']:
        raise BedIntegrationError('initial assembly deep version changed')
    fields=('reference_elevation_m','elevation_offset_m','committed_active_layer_mass_by_class_kg',
            'committed_voxel_mass_by_class_kg')
    import zipfile

    with zipfile.ZipFile(root/'initial_baseline.npz') as z:
        if sum(i.file_size for i in z.infolist())>max_host_bytes:
            raise BedIntegrationError('initial baseline exceeds host budget')
    with np.load(root/'initial_baseline.npz',allow_pickle=False) as z:
        if set(z.files)!=set(fields)|{'initial_available_fraction'}:
            raise BedIntegrationError('initial assembly baseline fields differ')
        arrays={n:z[n].copy() for n in z.files}
    committed=CommittedTopographyState(*(arrays[n] for n in fields),binding['last_commit_time_s'],binding['commit_count'])
    from maple.surface.topographic_commit.windowed_commit import (
        is_windowed_already_committed,
    )

    ledger=zeros_sediment_ledger_state(geometry.ny,geometry.nx,len(grain_classes.classes),
                                      present_processes=(process_index('water_erosion_deposition'),))
    state=BedState(snap['voxel_column'],snap['active_layer'],snap['water'],ledger,committed,snap['sediment_availability'])
    if not is_windowed_already_committed(ledger,committed,state.voxel_column,state.active_layer,geometry,
                                        first.config.numerics.mass_resolution_kg,binding['last_commit_time_s']):
        raise BedIntegrationError('initial saved baseline is not the actual committed bed')
    context=BedContext(geometry,grain_classes,first.config.numerics.mass_resolution_kg,
                       first.config.topographic_commit,first.config.avalanching,arrays['initial_available_fraction'])
    after=initial_binding_identity(root)  # hash-after: nothing changed while it was read
    if initial_identity_record(after)!=initial_identity_record(before):
        raise BedIntegrationError('initial assembly files changed during restore')
    return WindowedCaseAssembly(state,context,receipt,initial_binding=initial_identity_record(before))
