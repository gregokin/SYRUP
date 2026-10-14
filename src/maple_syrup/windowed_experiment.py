"""Explicit conservative water events on actual MAPLE resident-window beds.

Shares the complete-event solver, wet laws, rainfall factories, checkpoints
and accounting with the dense runner. Deep voxels stay in an immutable MAPLE
host store; final outputs and checkpoints contain resident arrays and version
references. This is not the unlimited-supply MAHLERAN legacy replay.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from maple_syrup.case_import import _refuse_output, verify_plot1_case
from maple_syrup.complete_event import CompletionPolicy
from maple_syrup.event_experiment import run_complete_case
from maple_syrup.sediment_event import SedimentEventError, initial_event_state
from maple_syrup.windowed_case import (
    CompiledRowTile,
    assemble_windowed_case,
    assembly_to_backend,
    chastre_row_tiles,
    persist_initial_assembly,
    restore_initial_assembly,
)

#: Fraction of the CURRENT free device memory used both by the planning pre-check and by the enforced CuPy pool hard
#: limit (the rule of the B13 phase profiler).
POOL_BUDGET_FRACTION = 0.85

PLAN_NOTE = ('planning estimate only, NOT a conservative envelope: it is known to undercount the measured pipeline '
             '(ROOT B12 Chastre GPU one-step profile: 17,078,447,104 bytes tracked live pool peak against an '
             '8,918,989,120-byte estimate on an 11,811,028,992-byte device). It is a necessary pre-check, never a bound '
             'or a fit claim; the enforced device bound is the CuPy pool hard limit (DevicePoolLimit). No full-column '
             'domain array is allocated.')


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise SedimentEventError(f'{name} must be a positive integer, got {value!r}')
    return int(value)


def window_memory_plan(geometry, n_voxels, *, n_classes, window_voxels, phase_max_nnz,
                       max_journal_records, device_id=None, mem_info=None):
    """Planning estimate of the resident-window run (see PLAN_NOTE: known to undercount; not a bound).

    Every dimension is validated: grid `ny`, `nx`, `n_voxels`, the ACTUAL class count `n_classes`, the window width
    (MAPLE `resolve_window_voxels` rule), `phase_max_nnz` and `max_journal_records` are positive integers. With a
    `device_id`, the estimate is refused when it alone exceeds 85% of the current free device memory (a necessary
    condition only). `mem_info` (a `() -> (free, total)` callable) replaces the CuPy device query in tests."""
    from maple.core.types.voxel import resolve_window_voxels
    from maple.surface.voxels.windowed_prefix import packet_terms_for

    ny = _positive_int('geometry.ny', getattr(geometry, 'ny', None))
    nx = _positive_int('geometry.nx', getattr(geometry, 'nx', None))
    n_voxels = _positive_int('n_voxels', n_voxels)
    nc = _positive_int('n_classes', n_classes)
    window_voxels = _positive_int('window_voxels', window_voxels)
    try:
        resolve_window_voxels(window_voxels, n_voxels)
    except ValueError as exc:
        raise SedimentEventError(str(exc)) from None
    phase_max_nnz = _positive_int('phase_max_nnz', phase_max_nnz)
    max_journal_records = _positive_int('max_journal_records', max_journal_records)
    n = ny*nx
    window = n*window_voxels*nc*8
    prefix = n*packet_terms_for(n_voxels,nc)*8
    # Versioned bed copies, history/ledger, transport, hydraulic arrays and CSR
    # merge buffers coexist. Host journal sorting has its own explicit bound.
    estimate = 4*window + 4*prefix + 40*n*nc*8 + 8*(n*nc+1)*8 + 6*phase_max_nnz*32 + n*192
    plan = {'estimated_device_or_array_peak_bytes':estimate,
            'resident_window_bytes':window,'prefix_bytes':prefix,
            'dense_voxel_bytes_not_allocated':n*n_voxels*nc*8,
            'journal_host_scratch_envelope_bytes':max_journal_records*(nc*8+16)*12,
            'n_cells':n,'n_voxels':n_voxels,'n_classes':nc,'window_voxels':window_voxels,
            'phase_max_nnz':phase_max_nnz,'max_journal_records':max_journal_records,
            'note':PLAN_NOTE}
    if device_id is not None:
        if mem_info is None:
            import cupy as cp

            def mem_info():
                return cp.cuda.Device(device_id).mem_info
        free,total=(int(v) for v in mem_info())
        plan.update({'device_free_bytes':free,'device_total_bytes':total,'device_budget_fraction':POOL_BUDGET_FRACTION})
        if estimate > POOL_BUDGET_FRACTION*free:
            raise SedimentEventError('resource refusal: the resident-window planning estimate alone exceeds 85% of '
                                     'current free device memory (the estimate undercounts; passing it is no fit claim)')
    return plan


class DevicePoolLimit:
    """The enforced device bound of a windowed CuPy run: a hard limit on the CuPy DEFAULT memory pool of the selected
    device, installed BEFORE any resident array is uploaded and restored after the invocation (success or failure).

    * limit = floor(0.85 x the device's CURRENT free memory) at installation; a stricter pre-existing pool limit is
      honoured (never loosened); the previous limit (0 = unlimited) is restored by `restore()`.
    * Refused (no upload) when the active CuPy allocator is not that default pool: the limit could not apply.
    * Scope (SCOPE): pooled CuPy allocations only (used plus cached pool blocks); NOT the CUDA context, kernel local
      memory, NVRTC/library workspaces or any non-pool allocation. Not a measured peak and not a GPU fit claim.
    * `refusal_for(exc)` turns a CuPy `OutOfMemoryError` anywhere in an exception chain into an explicit resource
      refusal naming the configured limit. The event loop publishes an accepted step only after a successful
      attempt and checkpoints only at accepted boundaries, so a refused attempt leaves earlier checkpoints valid and
      writes no new checkpoint or output; it is not retried and nothing is clipped.

    `cupy_module` injects a CuPy-compatible module (tests use a CPU stand-in; actual devices are the root's checks).
    """

    SCOPE = ('CuPy default memory pool of the selected device: pooled allocations (used plus cached blocks) only; '
             'excludes the CUDA context, kernel local memory, NVRTC/library workspaces and non-pool allocations; '
             'not a measured peak; no GPU fit claim')

    def __init__(self, fraction=POOL_BUDGET_FRACTION, *, cupy_module=None):
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not (0.0 < fraction <= 1.0):
            raise SedimentEventError(f'pool limit fraction must lie in (0, 1], got {fraction!r}')
        self.fraction = float(fraction)
        self._cp = cupy_module
        self._pool = None
        self._device_id = None
        self._prior = None
        self.record = None

    @property
    def installed(self) -> bool:
        return self._pool is not None

    def _cupy(self):
        if self._cp is None:
            from maple.core.backend import cupy_module

            self._cp = cupy_module()
        return self._cp

    def install(self, device_id) -> dict:
        if self.installed:
            raise SedimentEventError('device pool limit is already installed')
        if isinstance(device_id, bool) or not isinstance(device_id, (int, np.integer)) or int(device_id) < 0:
            raise SedimentEventError(f'device_id must be a nonnegative integer, got {device_id!r}')
        device_id = int(device_id)
        cp = self._cupy()
        with cp.cuda.Device(device_id):
            pool = cp.get_default_memory_pool()
            allocator = cp.cuda.get_allocator()
            if getattr(allocator, '__self__', None) is not pool:
                raise SedimentEventError('resource refusal: the active CuPy allocator is not the default memory pool, '
                                         'so the windowed runner cannot enforce its pool hard limit; nothing uploaded')
            free, total = (int(v) for v in cp.cuda.Device(device_id).mem_info)
            prior = int(pool.get_limit())
            used, held = int(pool.used_bytes()), int(pool.total_bytes())
            computed = math.floor(self.fraction * free)
            if computed < 1:
                raise SedimentEventError('resource refusal: no free device memory for the pool limit; nothing uploaded')
            stricter_prior = 0 < prior < computed
            effective = prior if stricter_prior else computed
            pool.set_limit(size=effective)
            try:
                applied = int(pool.get_limit())
                if applied != effective:
                    raise SedimentEventError(f'resource refusal: CuPy pool limit reads back {applied} bytes, '
                                             f'not the requested {effective}; nothing uploaded')
            except BaseException:
                pool.set_limit(size=prior)
                raise
        self._pool, self._device_id, self._prior = pool, device_id, prior
        self.record = {'enforced': True, 'device_id': device_id, 'fraction_of_current_free': self.fraction,
                       'device_free_bytes_at_install': free, 'device_total_bytes': total,
                       'computed_limit_bytes': computed, 'prior_limit_bytes': prior,
                       'effective_limit_bytes': effective,
                       'limit_source': ('stricter pre-existing pool limit honoured' if stricter_prior
                                        else '85% of current free device memory'),
                       'pool_used_bytes_at_install': used, 'pool_held_bytes_at_install': held,
                       'installed': 'before any resident array upload',
                       'restore_policy': 'prior limit restored after the invocation, including failures',
                       'scope': self.SCOPE}
        return self.record

    def restore(self):
        if not self.installed:
            return None
        cp = self._cupy()
        with cp.cuda.Device(self._device_id):
            self._pool.set_limit(size=self._prior)
        self._pool = None
        return self._prior

    def out_of_memory_types(self) -> tuple:
        cp = self._cp
        error = getattr(getattr(getattr(cp, 'cuda', None), 'memory', None), 'OutOfMemoryError', None)
        return (error,) if isinstance(error, type) else ()

    def refusal_for(self, exc):
        """A `SedimentEventError` resource refusal when `exc` (or its cause/context chain) is a CuPy
        `OutOfMemoryError`; otherwise None."""
        types = self.out_of_memory_types()
        if not types:
            return None
        seen = set()
        current = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, types):
                limit = None if self.record is None else self.record['effective_limit_bytes']
                return SedimentEventError(
                    f'resource refusal: CuPy device memory request exceeded the enforced pool hard limit '
                    f'({limit} bytes = {self.fraction:g} of free memory at installation, or a stricter prior limit) '
                    f'[{type(current).__name__}: {current}]. The failed attempt published no accepted step and wrote '
                    'no checkpoint or output; earlier checkpoints remain valid. Not retried, nothing clipped. A large '
                    'GPU run needs a measured pipeline that fits (estimates are not a fit claim).')
            current = current.__cause__ or current.__context__
        return None


def _require_xml(xml_path, xml_sha256, what):
    """CW-B15: hash the MAHLERAN XML and require its bound digest BEFORE (and, by the callers, after) a parse."""
    from maple_syrup.checkpoint import sha256_file

    if not isinstance(xml_sha256,str) or len(xml_sha256)!=64:
        raise SedimentEventError(f'{what}: the bound MAHLERAN XML digest is missing')
    if sha256_file(xml_path)!=xml_sha256:
        raise SedimentEventError(f'{what}: MAHLERAN XML {xml_path} does not hash to its bound digest; refused before use')


def pinned_rfid_report(verified):
    """CW-B15: the ORIGINAL RFID import report a Chastre case pins, read ONCE: the exact bytes are hashed against
    the binding's `source_rfid.report_sha256` and parsed from those same bytes. Returns `(report, record)`."""
    pins=verified.binding['source_rfid']
    path=Path(pins['case_dir'])/'syrup/rfid_import_report.json'
    data=path.read_bytes()
    digest=hashlib.sha256(data).hexdigest()
    if digest!=pins['report_sha256']:
        raise SedimentEventError('the pinned original RFID import report changed; refused before use')
    return json.loads(data),{'path':str(path.resolve()),'sha256':digest}


def prescribed_vegetation(verified,xml_path,xp,*,xml_sha256,rfid_report=None):
    """Read prescribed cover, never interpret the XML output flag as physics.

    CW-B15: the XML is hashed against `xml_sha256` BEFORE it is parsed and again after. A large (Chastre) case uses
    the `rfid_report` already read and pinned by `pinned_rfid_report` (read once); without it the report is read and
    pinned here."""
    from maple.core.backend import to_device

    from maple_syrup.case_import import _xml_value, parse_mahleran_xml
    from maple_syrup.checkpoint import sha256_file
    from maple_syrup.rfid_case import read_legacy_grid

    key='vegetation_cover_fraction'
    shape=(verified.case.config.geometry.ny,verified.case.config.geometry.nx)
    if key in verified.fields:
        cover=np.asarray(verified.fields[key],dtype=np.float64)
        record={'source':'verified bound sidecar cover map','growth':False}
    else:
        source_dir=verified.case_dir
        source_report=verified.report
        large='source_rfid' in verified.binding
        if large:
            pins=verified.binding['source_rfid'];source_dir=Path(pins['case_dir'])
            source_report=pinned_rfid_report(verified)[0] if rfid_report is None else rfid_report
        _require_xml(xml_path,xml_sha256,'prescribed vegetation')
        name=_xml_value(parse_mahleran_xml(xml_path),'vegetation-cover_map')
        _require_xml(xml_path,xml_sha256,'prescribed vegetation')
        if Path(name).name!=name or name not in source_report['staged_sources']:
            raise SedimentEventError('prescribed vegetation source is not a bound RFID input')
        bound=source_report['staged_sources'][name]
        path=source_dir/bound['relpath']
        if sha256_file(path)!=bound['sha256']:
            raise SedimentEventError('prescribed vegetation map changed')
        _header,north=read_legacy_grid(path)
        source_fields=verified.fields
        if large:
            fields_path=source_dir/'syrup/rfid_fields.npz'
            if sha256_file(fields_path)!=pins['fields_sha256']:
                raise SedimentEventError('prescribed vegetation source mask changed')
            with np.load(fields_path,allow_pickle=False) as z:
                source_active=z['active'].copy()
            if sha256_file(fields_path)!=pins['fields_sha256']:
                raise SedimentEventError('prescribed vegetation source mask changed during read')
        else:
            source_active=np.asarray(source_fields['active'],dtype=bool)
        values=north[::-1, :][1:-1,1:-1]
        if values.shape!=source_active.shape or not np.isfinite(values[source_active]).all():
            raise SedimentEventError('prescribed vegetation geometry/values differ from verified source')
        if np.any((values[source_active]<0)|(values[source_active]>100)):
            raise SedimentEventError('prescribed cover percent outside [0,100]')
        if large:
            unique=np.unique(values[source_active])
            if unique.size!=1:
                raise SedimentEventError('Chastre needs an explicit mapping policy for nonuniform prescribed vegetation')
            cover=np.where(verified.fields['active'],unique[0]*.01,0.)
        else:
            cover=np.where(source_active,values*.01,0.)
        record={'source':'bound original RFID vegetation-cover_map','map_sha256':bound['sha256'],
                'original_units':'percent','growth':False,
                'large_domain_rule':'same verified uniform active cover' if large else 'original active map cells'}
    if cover.shape!=shape or not np.isfinite(cover).all() or np.any((cover<0)|(cover>1)):
        raise SedimentEventError('invalid prescribed vegetation map')
    return to_device(np.ascontiguousarray(cover),xp),record


#: CW-F2: a direct `_setup_windowed` call that did not come from `run_complete_windowed` (no caller graph rebuild).
_NODATA_NOT_FORWARDED = object()


def _setup_windowed(case_dir, *, case_kind, store_dir, window_voxels, max_store_bytes,
                     max_host_bytes, max_journal_records, restore_initial=False, progress=None, pool_limit=None,
                     caller_nodata_value=_NODATA_NOT_FORWARDED, deep_chunk_encoding='full', deep_overlay_max_bytes=None,
                     **options):
    from maple.core.backend import resolve_backend, to_host
    from maple.core.types.topographic_commit import committed_elevation_m

    from maple_syrup.chastre_case import hash_tree, verify_chastre_case
    from maple_syrup.rfid_case import rfid_inputs, verify_rfid_case
    from maple_syrup.sediment_bed import (
        BedIntegrationError,
        check_deep_storage_options,
        check_deep_storage_source,
        terrain_reference,
        validate_deep_storage,
    )
    from maple_syrup.sediment_experiment import (
        plot1_sediment_parameters_from_xml,
        prepare_verified_sediment_case,
    )
    from maple_syrup.windowed_case import initial_identity_record

    try:  # CW-B108: pure option values first, before any verification, read or store access
        check_deep_storage_options(deep_chunk_encoding,deep_overlay_max_bytes)
    except BedIntegrationError as exc:
        raise SedimentEventError(str(exc)) from None
    backend=options['backend']; control=options['control']
    if control.phase_storage != 'csr' or not control.characteristic:
        raise SedimentEventError('windowed complete runner requires explicitly bounded characteristic CSR transport')
    if backend=='cupy' and not isinstance(pool_limit,DevicePoolLimit):
        raise SedimentEventError('a windowed CuPy run requires the enforced device pool limit (DevicePoolLimit)')
    common={k:options[k] for k in ('expected_maple_root','allow_maple_source_change')}
    rfid_report=None
    if case_kind == 'chastre':
        verified=verify_chastre_case(case_dir,reload_tiles=False,progress=progress,**common)
        nv=int(verified.binding['grid']['nz']);datum=float(verified.binding['grid']['datum_offset_m'])
        tiles=chastre_row_tiles(verified)
        # CW-B15: the original RFID report is read and pinned ONCE and reused (XML record, vegetation source).
        rfid_report,rfid_report_record=pinned_rfid_report(verified)
        xml_record=rfid_report['mahleran']
    elif case_kind in ('plot1','rfid'):
        verifier=verify_plot1_case if case_kind=='plot1' else verify_rfid_case
        verified=verifier(case_dir,mahleran_root=options['mahleran_root'],**common)
        nv=verified.case.voxel_column.mass_kg.shape[2]
        datum=float(verified.report['bed']['datum_offset_m'])
        tiles=(CompiledRowTile(0,verified.case.config.geometry.ny,verified.case_dir,hash_tree(verified.case_dir)),)
        xml_record=verified.report['mahleran']
        rfid_report_record=None
    else:
        raise SedimentEventError('case_kind must be plot1, rfid or chastre')
    if case_kind!='plot1' and caller_nodata_value is not _NODATA_NOT_FORWARDED:
        # CW-F2: the masked graph that run_complete_case rebuilds uses the CALLER's nodata_value, while the terrain
        # reference built here uses the verified grid value. They must be the same sentinel; a different value is
        # refused here, before any store assembly, restore or device upload (no inferred mask, no case rewrite).
        # `run_complete_windowed` always forwards the caller value; a direct setup call (no graph rebuild by the
        # runner) leaves it unforwarded and is unchanged.
        declared=float(verified.report['grid']['nodata_value'])
        if (caller_nodata_value is None or isinstance(caller_nodata_value,bool)
                or not isinstance(caller_nodata_value,(int,float,np.integer,np.floating))
                or float(caller_nodata_value)!=declared):
            raise SedimentEventError(f'nodata_value {caller_nodata_value!r} differs from the verified case grid '
                                     f'nodata_value {declared!r}; refused before assembly')
    # CW-B15: the bound XML is identified and hash-checked BEFORE any parameter evaluation, assembly or upload.
    if case_kind=='plot1':
        recipe_record=verified.report['recipe']
        xml_path=(Path(options['mahleran_root'] or recipe_record['mahleran_root']).resolve()/recipe_record['xml_name'])
    else:
        xml_path=Path(xml_record['xml_path'])
    xml_sha256=xml_record['xml_sha256']
    _require_xml(xml_path,xml_sha256,'windowed case setup')
    bound_xml={'path':str(xml_path.resolve()),'sha256':xml_sha256,
               'pinned_by':('original RFID import report (read once, pinned by the Chastre binding)' if case_kind=='chastre'
                            else 'verified case report')}
    if rfid_report_record is not None:
        bound_xml['rfid_report']=rfid_report_record
    _refuse_output((Path(store_dir)/".restart_path_check" if restore_initial else Path(store_dir)).resolve(),{'case':Path(case_dir).resolve(),
                   'MAPLE':verified.maple_dependency.source_root,'SYRUP package':Path(__file__).parent,
                   'MAHLERAN':Path(options['mahleran_root'] or verified.report['mahleran']['root']).resolve()})
    g=verified.case.config.geometry
    grain_classes=verified.case.config.grain_classes
    resolved=resolve_backend(backend);xp=resolved.xp
    plan=window_memory_plan(g,nv,n_classes=len(grain_classes.classes),window_voxels=window_voxels,
                            phase_max_nnz=control.phase_max_nnz,max_journal_records=max_journal_records,
                            device_id=resolved.device_id if backend=='cupy' else None)
    deep_option=(deep_chunk_encoding,deep_overlay_max_bytes)!=('full',None)
    if deep_option:
        # CW-B108: the bound MAPLE source and MAPLE's own overlay voxel rule, BEFORE any store assembly or restore.
        # CW-B108C1: the source is checked FIRST; only the accepted source defines OVERLAY_MAX_VOXELS (M94 does not).
        try:
            check_deep_storage_source(deep_chunk_encoding,deep_overlay_max_bytes)
        except BedIntegrationError as exc:
            raise SedimentEventError(str(exc)) from None
        from maple.surface.voxels.deep_store import OVERLAY_MAX_VOXELS
        if nv>OVERLAY_MAX_VOXELS:
            raise SedimentEventError(f'deep_chunk_encoding=\'overlay\' supports n_voxels <= {OVERLAY_MAX_VOXELS}, '
                                     f'got {nv}; refused before assembly (no fallback encoding)')
    if restore_initial:
        assembly=restore_initial_assembly(tiles,g,grain_classes,store_dir=store_dir,
                                           max_host_bytes=max_host_bytes,datum_offset_m=datum)
        initial=assembly.initial_binding
    else:
        assembly=assemble_windowed_case(tiles,g,grain_classes,nv,store_dir=store_dir,
                     datum_offset_m=datum,max_store_bytes=max_store_bytes,max_host_bytes=max_host_bytes,
                     window_voxels=window_voxels,progress=progress)
        initial=initial_identity_record(persist_initial_assembly(assembly))
    if assembly.state.voxel_column.window_voxels!=window_voxels or assembly.receipt['plan']['max_store_bytes']!=max_store_bytes:
        raise SedimentEventError('initial assembly window or storage cap differs from the run configuration')
    # Explicit per-run scope: no slope avalanching, compatible with the first
    # water-only masked/pit experiment. Original compiled options untouched.
    original_avalanching=dataclasses.asdict(assembly.context.avalanching_spec)
    context=dataclasses.replace(assembly.context,
             avalanching_spec=dataclasses.replace(assembly.context.avalanching_spec,enabled=False),
             window_max_journal_records=max_journal_records,
             deep_chunk_encoding=deep_chunk_encoding,deep_overlay_max_bytes=deep_overlay_max_bytes)
    try:  # CW-B108: the option against the actual host bed / store header, before any upload or event
        validate_deep_storage(context,assembly.state)
    except BedIntegrationError as exc:
        raise SedimentEventError(str(exc)) from None
    assembly=dataclasses.replace(assembly,context=context)
    if backend=='cupy':
        # CW-B15: the enforced pool hard limit is installed BEFORE the first resident array is uploaded.
        plan['device_pool_limit']=pool_limit.install(resolved.device_id)
    assembly=assembly_to_backend(assembly,xp)
    bed,context=assembly.state,assembly.context
    if case_kind=='plot1':
        prepared=prepare_verified_sediment_case(verified,backend='numpy',mahleran_root=options['mahleran_root'],
                                                end_s=options['end_s'],control=control)
        if Path(prepared['xml_path']).resolve()!=xml_path.resolve() or prepared['xml_sha_before']!=xml_sha256:
            raise SedimentEventError('the dense factory bound a different MAHLERAN XML than the windowed setup')
        # Keep rainfall, infiltration and geometry from the existing factory.
        # Move only its solver operands; its temporary dense bed is not used.
        from maple.core.backend import to_device_tree

        for name in ('column','field','soil0','vegetation'):
            moved=to_device_tree(prepared[name],xp)
            prepared[name]=dataclasses.replace(moved,xp=xp) if hasattr(moved,'xp') else moved
        from maple_syrup.routing import plot1_routing_graph

        prepared['graph']=plot1_routing_graph(verified.fields,verified.report,xp=xp)
        _require_xml(xml_path,xml_sha256,'windowed Plot1 sediment parameters')
        prepared['sediment'],prepared['sediment_record']=plot1_sediment_parameters_from_xml(
            xml_path,context.grain_classes,xp=xp)
        _require_xml(xml_path,xml_sha256,'windowed Plot1 sediment parameters')
    else:
        hyd=rfid_inputs(verified,backend,with_geometry=False)
        _require_xml(xml_path,xml_sha256,'windowed sediment parameters')
        sediment,sediment_record=plot1_sediment_parameters_from_xml(xml_path,context.grain_classes,xp=xp)
        _require_xml(xml_path,xml_sha256,'windowed sediment parameters')
        vegetation,vegetation_record=prescribed_vegetation(verified,xml_path,xp,xml_sha256=xml_sha256,
                                                           rfid_report=rfid_report)
        sediment_record['prescribed_vegetation']=vegetation_record
        prepared={'column':hyd.params,'field':hyd.field,'schedule':hyd.schedule,'soil0':hyd.soil0,
                  'vegetation':vegetation,
                  'sediment':sediment,'sediment_record':sediment_record,'graph':hyd.graph,
                  'parameter_record':hyd.parameter_record,'settings':hyd.settings,
                  'xml_path':xml_path,'xml_sha_before':xml_sha256,
                  'end':hyd.stormlength_s if options['end_s'] is None else options['end_s']}
    graph=prepared['graph']
    full=np.asarray(verified.fields['legacy_full_elevation_m'],dtype=np.float64)
    if case_kind=='plot1':
        exports=np.asarray(verified.fields['legacy_full_rainfall_scaling'])<0
        nodata=None
    else:
        exports=np.ones(full.shape,dtype=bool);exports[1:-1,1:-1]=False
        nodata=float(verified.report['grid']['nodata_value'])
    terrain=terrain_reference(full,exports,bed,graph,nodata_value=nodata)
    prepared.update({'state0':initial_event_state(graph,terrain,bed,context,prepared['soil0'],control=control),
                     'context':context,'xp':xp,'initial_elevation':to_host(committed_elevation_m(bed.committed_topography)).copy(),
                     'initial_active':to_host(bed.active_layer.mass_kg).copy(),
                     'initial_available':to_host(bed.sediment_availability.available_mass_kg).copy(),
                     'rainfall_source_path':(verified.case_dir/'syrup/forcing/applied_forcing.json'
                           if prepared['schedule'].provenance.kind=='native_applied_capture' else verified.rainfall_path)})
    identity=verified.binding.get('maple_case_identity_sha256')
    if identity is None:
        identity=hashlib.sha256(json.dumps(verified.binding,sort_keys=True).encode()).hexdigest()
    configuration={'scheme':'MAPLE resident-window/1','case_kind':case_kind,'window_voxels':window_voxels,
                   'max_journal_records':max_journal_records,'max_store_bytes':max_store_bytes,
                   'max_host_bytes':max_host_bytes,'original_avalanching':original_avalanching,
                   'run_avalanching_enabled':False,'source_tiles':assembly.receipt['source_tiles'],
                   'initial_deep_manifest_digest':assembly.receipt['deep_manifest_digest'],
                   'datum_offset_m':datum,
                   # CW-B15: the persisted initial operands (binding bytes and its full three-file map), identical
                   # on the fresh and the reuse path, and the actual bound XML.
                   'initial_assembly':initial,'bound_mahleran_xml':bound_xml}
    if deep_option:
        # CW-B108: present ONLY for a non-default encoding (this is the bed_storage identity; defaults unchanged)
        configuration['deep_storage']={'deep_chunk_encoding':deep_chunk_encoding,
                                       'deep_overlay_max_bytes':deep_overlay_max_bytes}
    return {'verified':verified,'prepared':prepared,'memory_preflight':plan,'run_configuration':configuration,
             'case_identity_sha256':identity,'maple_configuration':assembly.receipt['maple_configuration']}


def resolve_window_checkpoint_bytes(max_host_bytes, max_checkpoint_bytes=None):
    """CW-F2: the resume byte bound of a resident-window run and its recorded derivation.

    Default: the run's ALREADY explicit host cap `max_host_bytes` (40 GiB default), so the uncompressed resident
    arrays of a resumed bundle never exceed the host budget the caller declared for the run. A caller may tighten it
    (any positive integer up to `max_host_bytes`); a looser value is refused, so the single declared host cap stays
    the authority. Every value is validated (positive integer; no bool, float, NaN or None as an unbounded value)."""
    from maple_syrup.checkpoint import checkpoint_byte_bound

    host = _positive_int('max_host_bytes', max_host_bytes)
    if max_checkpoint_bytes is None:
        return host, {'max_checkpoint_uncompressed_bytes': host,
                      'resolution': 'resident-window default: equal to the explicit max_host_bytes cap',
                      'max_host_bytes': host}
    bound = checkpoint_byte_bound(max_checkpoint_bytes, 'max_checkpoint_bytes')
    if bound > host:
        raise SedimentEventError(f'max_checkpoint_bytes {bound} exceeds the explicit max_host_bytes cap {host}; '
                                 'a resident-window resume bound may only tighten the host cap')
    return bound, {'max_checkpoint_uncompressed_bytes': bound,
                   'resolution': 'caller-explicit, tightened within max_host_bytes', 'max_host_bytes': host}


def run_complete_windowed(case_dir, output_dir, *, case_kind, store_dir, window_voxels=4,
                           max_store_bytes=256*2**30,max_host_bytes=40*2**30,
                           max_journal_records=1_000_000,progress=None,reuse_initial=False,
                           max_checkpoint_bytes=None,deep_chunk_encoding='full',deep_overlay_max_bytes=None,
                           **options):
    """The shared complete-event loop, with real streamed MAPLE case setup.

    A CuPy run installs the enforced device pool limit (DevicePoolLimit) before any resident upload, converts a
    CuPy out-of-memory error into an explicit resource refusal and restores the previous pool limit after the
    invocation, on success and on every failure. CW-F2: a resume is bounded by `max_checkpoint_bytes`
    (`resolve_window_checkpoint_bytes`), validated before any setup and forwarded to the checkpoint loader."""
    if not isinstance(reuse_initial,bool):
        raise SedimentEventError('reuse_initial must be bool')
    from maple_syrup.sediment_bed import BedIntegrationError, check_deep_storage_source
    try:  # CW-B108: option values refused before any setup (default 'full' / None: the existing runner, no MAPLE read)
        # CW-B108C1: and an unsupported MAPLE source for a non-default option, before case verification / output /
        # store creation (the setup re-checks it before assembly).
        check_deep_storage_source(deep_chunk_encoding,deep_overlay_max_bytes)
    except BedIntegrationError as exc:
        raise SedimentEventError(str(exc)) from None
    checkpoint_bytes,checkpoint_record=resolve_window_checkpoint_bytes(max_host_bytes,max_checkpoint_bytes)
    options.setdefault('phase_storage','csr')
    options.setdefault('phase_max_nnz',1_000_000)
    if case_kind=='chastre':
        options.setdefault('allow_flat_storage',True)
    if case_kind!='plot1':
        options.setdefault('allow_pit_storage',True)
        options.setdefault('allow_masked_nodata',True)
        # Caller must pass the case's declared nodata, never infer a new mask.
        if options.get('nodata_value') is None:
            raise SedimentEventError('masked case requires explicit nodata_value')
    pool_limit=_new_pool_limit() if options.get('backend','numpy')=='cupy' else None
    def setup(**kw):
        result=_setup_windowed(**kw,case_kind=case_kind,store_dir=store_dir,window_voxels=window_voxels,
                max_store_bytes=max_store_bytes,max_host_bytes=max_host_bytes,
                max_journal_records=max_journal_records,restore_initial=reuse_initial or options.get("resume") is not None,
                progress=progress,pool_limit=pool_limit,caller_nodata_value=options.get('nodata_value'),
                deep_chunk_encoding=deep_chunk_encoding,deep_overlay_max_bytes=deep_overlay_max_bytes)
        # CW-F2: recorded in the summary, NOT in the run configuration (that is the bed_storage identity).
        result['checkpoint_byte_bound']=checkpoint_record
        return result
    try:
        return run_complete_case(case_dir,output_dir,case_setup=setup,max_checkpoint_bytes=checkpoint_bytes,**options)
    except BaseException as exc:
        refusal=None if pool_limit is None else pool_limit.refusal_for(exc)
        if refusal is not None:
            raise refusal from exc
        raise
    finally:
        if pool_limit is not None:
            pool_limit.restore()


def _new_pool_limit():
    """Factory of the enforced pool limit of one windowed CuPy invocation (tests substitute a CPU stand-in)."""
    return DevicePoolLimit()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case-dir',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--store-dir',required=True);p.add_argument('--case-kind',choices=('plot1','rfid','chastre'),required=True)
    p.add_argument('--backend',choices=('numpy','cupy'),default='numpy')
    p.add_argument('--bisection-iterations',type=int,default=40)
    p.add_argument('--root-solver',choices=('bisection','newton'),default='bisection')
    p.add_argument('--max-end-s',type=float);p.add_argument('--pause-at-s',type=float)
    p.add_argument('--reuse-initial',action='store_true');p.add_argument('--resume');p.add_argument('--checkpoint-every-s',type=float,default=600.)
    p.add_argument('--phase-max-nnz',type=int,default=1_000_000)
    p.add_argument('--max-journal-records',type=int,default=1_000_000)
    p.add_argument('--window-voxels',type=int,default=4)
    p.add_argument('--max-host-bytes',type=int,default=40*2**30)
    p.add_argument('--max-checkpoint-bytes',type=int,default=None,
                   help='resume bound on summed uncompressed bundle bytes; default = --max-host-bytes, may only tighten it')
    p.add_argument('--max-store-bytes',type=int,default=256*2**30)
    p.add_argument('--deep-chunk-encoding',choices=('full','overlay'),default='full',
                   help="CW-B108 deep-chunk storage: full (default) or exact overlay (accepted typed MAPLE source only)")
    p.add_argument('--deep-overlay-max-bytes',type=int,default=None,
                   help='overlay only: explicit per-publication overlay byte cap forwarded to MAPLE (default: MAPLE rule)')
    p.add_argument('--nodata-value',type=float)
    p.add_argument('--allow-flat-storage',action='store_true',default=None)
    p.add_argument('--terminal-storage-water',action='store_true')
    p.add_argument('--deposit-mobile-at-flow-end',action='store_true')
    p.add_argument('--record-exchange-history',action='store_true')
    p.add_argument('--allow-maple-source-change',action='store_true')
    a=vars(p.parse_args(argv));terminal=a.pop('terminal_storage_water');flow_end=a.pop('deposit_mobile_at_flow_end')
    if a['allow_flat_storage'] is None:
        a.pop('allow_flat_storage')
    a['implementation']='numba' if a['backend']=='numpy' else 'cuda'
    a['policy']=CompletionPolicy(terminal_storage_water=terminal,deposit_mobile_at_flow_end=flow_end)
    try:
        result=run_complete_windowed(**a,progress=lambda s:print(s,flush=True))
    except (ValueError,RuntimeError,OSError) as exc:
        print(f'{type(exc).__name__}: {exc}',flush=True)
        return 1
    print(json.dumps(result.summary,indent=2,allow_nan=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
