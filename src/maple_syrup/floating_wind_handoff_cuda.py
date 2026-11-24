"""Explicit Dense CUDA wind boundary adapter; native physics remains upstream.

A fresh native CUDA owner can publish its initial boundary without a fictional
rain event. A genuinely finalized, separately staged MAPLE wind bed is then
reconciled against CURRENT native water with the existing selected-column CUDA
reconcile. Wind carries surface water bitwise and takes no hydrology/E/T step.
The held vegetation generation, J/I and actual routing source survive.

Device-first, separate host/device admission precedes scans/copies. The dirty
comparison is one cold tiled resident byte comparison, including composition
changes at unchanged elevation. Cold post-stage SHA256 integrity uses native
bounded streaming, explicitly counted in receipts; no whole-volume host mirror
or hot hydrology-step transfer is introduced. This first Dense qualification
makes no large-domain/windowed or combined persisted-restart claim.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import threading
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any

import numpy as np
from maple_syrup import floating_wind_handoff as wh
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_event_lifecycle_cuda as flc
from maple_syrup import floating_event_refresh_cuda as erc
from maple_syrup import floating_matrix_infiltration as fm
from maple_syrup import floating_preferential_infiltration as fpi
from maple_syrup import floating_pond as fp
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import floating_soil_layers_cuda as fcu
from maple_syrup import native_floating_soil as nfs
from maple_syrup import native_floating_soil_cuda as ncuda
from maple_syrup import dense_cuda_source as dcs
from maple_syrup import sediment_bed as sb
from maple_syrup import vegetation_lifecycle as vl
from maple_syrup import water_vegetation_binding as wvb
from maple_syrup.sediment_bed import BedContext, BedState, is_windowed_bed

PROFILE = 'maple-syrup-floating-wind-handoff-cuda/1'
CONTEXT_SCHEMA = 'maple_syrup.floating_wind_context.cuda.v1'
RECEIPT_SCHEMA = 'maple_syrup.floating_wind_handoff_receipt.cuda.v1'
WindHandoffError = wh.WindHandoffError
WindHandoffLimitError = wh.WindHandoffLimitError
WindHandoffLimits = wh.WindHandoffLimits
_MINT = threading.local()
_CONTEXTS = weakref.WeakSet()
_CANDIDATES = weakref.WeakSet()
_INTEGRITY = weakref.WeakKeyDictionary()
_QUOTAS = weakref.WeakKeyDictionary()
_LAST = weakref.WeakKeyDictionary()

__all__ = ['PROFILE', 'CONTEXT_SCHEMA', 'RECEIPT_SCHEMA', 'WindContext', 'WindHandoffCandidate',
           'WindHandoffError', 'WindHandoffLimitError', 'WindHandoffLimits', 'start_wind_boundary',
           'stage_wind_handoff', 'adopt_wind_handoff', 'restore_wind_context']


@contextlib.contextmanager
def _minting():
    old = getattr(_MINT, 'ok', False)
    _MINT.ok = True
    try:
        yield
    finally:
        _MINT.ok = old


@dataclass(frozen=True, eq=False)
class WindContext(wh._Record):
    bed_context: Any
    shape: tuple
    n_classes: int
    descriptor_sha256: str
    source_witness: str
    config_sha256: str
    device_id: int
    # Terrain (default LAST): the sealed scalar identity of the bed context's fixed base / domain footprint (`sediment_bed.
    # terrain_identity`), None for a terrain-free context (its record is then byte-identical to before).
    terrain_identity: Any = None

    def __post_init__(self):
        if not getattr(_MINT, 'ok', False):
            raise WindHandoffError('WindContext is minted only by this CUDA adapter')
        _CONTEXTS.add(self)

    def record(self):
        rec = dict(schema=CONTEXT_SCHEMA, shape=list(self.shape), n_classes=self.n_classes,
                   descriptor_sha256=self.descriptor_sha256, source_witness=self.source_witness,
                   config_sha256=self.config_sha256, representation='dense', backend='cuda', device_id=self.device_id)
        if self.terrain_identity is not None:
            rec['terrain'] = self.terrain_identity
        return rec

    @property
    def geometry(self):
        return self.bed_context.geometry

    @property
    def grain_classes(self):
        return self.bed_context.grain_classes

    @property
    def mass_resolution_kg(self):
        return float(self.bed_context.mass_resolution_kg)


@dataclass(frozen=True, eq=False)
class _Parts:
    base: Any
    vegetation: Any
    host_active: Any
    edges_host: Any
    host_depth: Any


@dataclass(frozen=True, eq=False)
class WindHandoffCandidate(wh._Record):
    occurrence_id: str
    end_s: float
    role: str
    receipt_json: str
    _parent_ref: Any
    _lineage_ref: Any
    _cell: Any

    def __post_init__(self):
        if not getattr(_MINT, 'ok', False):
            raise WindHandoffError('WindHandoffCandidate is minted only by CUDA staging')
        _CANDIDATES.add(self)

    def receipt(self):
        return json.loads(self.receipt_json)

    @property
    def consumed(self):
        return bool(self._cell['consumed'])

    @property
    def accepted_boundary_ref(self):
        return self._cell['accepted']


def _q(available_bytes_fn, reserve_bytes, host_available_bytes_fn, host_reserve_bytes, memory_budget_bytes, host_memory_budget_bytes):
    if not callable(available_bytes_fn) or not callable(host_available_bytes_fn):
        raise WindHandoffError('explicit real device AND host readers are required')
    for v in (memory_budget_bytes, host_memory_budget_bytes):
        if type(v) is not int or v < 1:
            raise WindHandoffError('separate positive device and host byte caps are required')
    return flc.quota_of(available_bytes_fn, reserve_bytes, host_available_bytes_fn, host_reserve_bytes, memory_budget_bytes, host_memory_budget_bytes)


def _array(cp, dev, a, what):
    if type(a) is not cp.ndarray or int(a.device.id) != dev or a.dtype.hasobject or not a.flags.c_contiguous:
        raise WindHandoffError(f'{what} must be a resident C-contiguous array on the current native device')


def _terrain_pairing(cp, dev, bed_context, expected, terrain_source, ny, nx):
    """The ONLY admitted terrain pairing: a bed context carrying a fixed base / domain footprint requires the genuine sealed
    `dense_cuda_source.DenseCudaSource` of `PROFILE_NATIVE_TERRAIN_20261123` (hot identity guard: no hash, no read), its identity must equal the
    sealed/expected one, and the owners must be resident arrays of this device that pass the native structural validators. A
    terrain-free context refuses a `terrain_source`. Returns the terrain identity (None without terrain)."""
    tid = sb.terrain_identity(bed_context)
    if tid != expected:
        raise WindHandoffError('the bed context terrain (fixed base / domain) differs from the sealed context identity')
    if tid is None:
        if terrain_source is not None:
            raise WindHandoffError('terrain_source given for a bed context without a fixed base / domain footprint')
        return None
    try:
        dcs.check_dense_cuda_source_hot(terrain_source, dcs.PROFILE_NATIVE_TERRAIN_20261123)
    except dcs.DenseCudaSourceError as exc:
        raise WindHandoffError(f'terrain pairing refused (the explicit native_terrain_20261123 terrain source association is required): {exc}') from None
    from maple.core.types.domain import validate_domain_state
    from maple.core.types.fixed_base import validate_fixed_base_state
    try:
        if bed_context.fixed_base is not None:
            _array(cp, dev, bed_context.fixed_base.elevation_m, 'fixed base elevation')
            validate_fixed_base_state(bed_context.fixed_base, ny, nx)
        if bed_context.domain is not None:
            for name in ('valid_mask', 'step_east', 'step_west', 'step_north', 'step_south', 'box_clearance_cells'):
                _array(cp, dev, getattr(bed_context.domain, name), f'domain {name}')
            validate_domain_state(bed_context.domain, ny, nx)
    except ValueError as exc:
        raise WindHandoffError(f'native terrain validation refused: {exc}') from None
    return tid


def _terrain_water_dry(bed, bed_context, what):
    if sb.water_outside_domain_nonzero(bed, bed_context):
        raise WindHandoffError(f'{what}: water is nonzero on a NoData cell of the domain footprint')


def _bed_meta(cp, dev, bed, ny, nx, K):
    if type(bed) is not BedState or is_windowed_bed(bed):
        raise WindHandoffError('only a complete native Dense CUDA BedState is accepted')
    spec = ((bed.voxel_column.mass_kg, (ny,nx,None,K)), (bed.active_layer.mass_kg,(ny,nx,K)),
            (bed.water.depth_m,(ny,nx)),(bed.water.mobile_mass_by_cell_class_kg,(ny,nx,K)))
    for a, shape in spec:
        _array(cp,dev,a,'bed leaf')
        if a.dtype != cp.float64 or a.ndim != len(shape) or any(w is not None and g != w for g,w in zip(a.shape,shape)):
            raise WindHandoffError('a resident bed leaf has the wrong float64 shape')
    return int(bed.voxel_column.mass_kg.shape[2])


def _same(cp, a, b):
    return a.dtype == b.dtype and a.shape == b.shape and bool(flc._scalar(cp.array_equal(a.view(cp.uint8),b.view(cp.uint8))))


def _walk(v, cp, fn, name='', seen=None, depth=0):
    if seen is None:
        seen = set()
    if depth > 18:
        raise WindHandoffError('state nesting exceeds the admitted native layout')
    if type(v) is cp.ndarray:
        if id(v) not in seen:
            seen.add(id(v)); fn(name,v)
    elif type(v) is np.ndarray or hasattr(v,'__cuda_array_interface__'):
        raise WindHandoffError('mixed or foreign device array in native mobile pool')
    elif is_dataclass(v) and not isinstance(v,type):
        if id(v) in seen: return
        seen.add(id(v))
        for f in fields(v): _walk(getattr(v,f.name),cp,fn,name+'.'+f.name,seen,depth+1)
    elif isinstance(v,Mapping):
        for k in sorted(v,key=repr): _walk(v[k],cp,fn,name+'.'+str(k),seen,depth+1)
    elif isinstance(v,(list,tuple)):
        for i,x in enumerate(v): _walk(x,cp,fn,name+'.'+str(i),seen,depth+1)


def _integrity(cp, values):
    """Native SHA256 bounded cold stream. No retained host bed/W/J/I mirror."""
    h = hashlib.sha256(PROFILE.encode()); arrays=[]; seen={}
    def feed(v, depth=0):
        if depth > 18: raise WindHandoffError('integrity nesting exceeds native layout')
        if type(v) is vl.VegetationRecord:
            # The producer fingerprints every record field and the immutable
            # borrowed binding. Never traverse its projection or PlantSession.
            vl.verify_record(v)
            h.update(f'VegetationRecord|{id(v)}|{v.recipe_id}|{v.binding_sha256}|{v.applied_g_sha256}'.encode())
            for cover in (v.splash_cover, v.shade_cover, v.splash_cover_device, v.shade_cover_device):
                feed(cover,depth+1)
            return
        if type(v) is cp.ndarray:
            if id(v) in seen: h.update(f'R{seen[id(v)]}'.encode()); return
            seen[id(v)]=len(seen); key=str(len(arrays)); arrays.append((key,v)); h.update(f'A{key}|{v.dtype}|{v.shape}'.encode())
        elif is_dataclass(v) and not isinstance(v,type):
            if id(v) in seen: h.update(f'R{seen[id(v)]}'.encode()); return
            seen[id(v)]=len(seen); h.update(type(v).__qualname__.encode())
            for f in fields(v): h.update(f.name.encode()); feed(getattr(v,f.name),depth+1)
        elif isinstance(v,Mapping):
            for k in sorted(v,key=repr): h.update(repr(k).encode()); feed(v[k],depth+1)
        elif isinstance(v,(tuple,list)):
            h.update(f'T{len(v)}'.encode())
            for x in v: feed(x,depth+1)
        elif type(v) is np.ndarray:
            wh._feed(h,v,{})
        elif v is None or isinstance(v,(str,bytes,int,float,bool,np.generic)):
            wh._feed(h,v,{})
        else:
            # Native prepared records own logically immutable maps. Include the
            # actual maps as well as the record identity so mutable GPU inputs
            # cannot evade the post-stage content audit via an old sha256.
            h.update(f'O{type(v).__qualname__}|{id(v)}|{getattr(v,"sha256",None)}'.encode())
            for name in ('_maps',):
                if hasattr(v,name): feed(getattr(v,name),depth+1)
    feed(values)
    return fcu.stream_hash(cp,arrays,h.digest())


def _boundaries(parent,parts,oid,end,text):
    return (parent,parts,oid,end,text)


def _admit(cp,dev,model,q,limits,what,extra_d=0,extra_h=0,scan=False):
    d=int(model['device_payload_bytes'])+extra_d; h=int(model['host_payload_bytes'])+extra_h
    cap=limits.max_scan_bytes if scan else limits.max_candidate_bytes
    if d+int(q['reserve_bytes']) > cap or h+fcu.PY_ALLOWANCE_BYTES+int(q['host_reserve_bytes']) > cap:
        raise WindHandoffLimitError(f'{what}: separate device/host requirements {d}/{h} exceed stage cap {cap}')
    fcu.admit(cp,dev,d,h,q,what)
    return d,h


def _vegetation(parent,cp,dev):
    v=vl.current_record(parent)
    if v is None: return None
    if v.backend != 'cuda' or v.device_id != dev or (v.n_cells,v.n_layers) != (parent.root.column.n_cells,parent.root.column.n_layers):
        raise WindHandoffError('vegetation generation belongs to another backend/device/domain')
    g=fm.held_cell_vegetation_factor(parent.root.matrix_op); _array(cp,dev,g,'held vegetation factor')
    if g.dtype != cp.float64 or g.shape != (v.n_cells,):
        raise WindHandoffError('held per-cell vegetation law has a different domain')
    return v


def _wind_meta(cp,dev,final,child,parent,start,ctx,limits):
    from maple.aeolian.scheduler.finalization import WindEventFinalizationResult
    from maple.core.types.cohort import MobileCohortState
    from maple.core.backend import arrays_share_memory
    if type(final) is not WindEventFinalizationResult or final.completed is not True or final.water_handoff_ready is not True or final.status not in wh._STATUSES or tuple(final.reasons):
        raise WindHandoffError('a genuine completed native wind finalization is required')
    sched=final.state; phys=sched.physical
    if any(getattr(phys,n) is not getattr(child,n) for n in wh._SHARED):
        raise WindHandoffError('child fields are not the finalized native fields')
    end=wh._clock(final.time_s,'final.time_s')
    if end <= start or float(phys.time_s).hex()!=end.hex() or float(final.block_offset_s)!=0.0 or float(sched.block_offset_s)!=0.0 or int(final.block_index)!=int(sched.block_index) or int(sched.block_index)<1:
        raise WindHandoffError('native finalization clocks/cursor are incomplete')
    pool=phys.mobile_state
    if type(pool) is not MobileCohortState:
        raise WindHandoffError('this first Dense CUDA path requires the native flat cohort pool')
    b=[0]; slots=[0]
    def count(name,a):
        _array(cp,dev,a,'wind mobile pool'); b[0]+=int(a.nbytes)
        if name.endswith('.mass_kg'): slots[0]+=int(a.size)
    _walk(pool,cp,count)
    if slots[0] > limits.max_cohorts: raise WindHandoffLimitError('native cohort slots exceed handoff limit')
    vec={}
    for n in wh._CLASS_FIELDS:
        a=getattr(final,n)
        if type(a) is not np.ndarray or a.dtype!=np.float64 or a.shape!=(ctx.n_classes,):
            raise WindHandoffError('native finalizer class packet must be exact host float64 K-vector')
        vec[n]=a
    if child is parent or child.voxel_column is parent.voxel_column or child.active_layer is parent.active_layer or arrays_share_memory(child.voxel_column.mass_kg,parent.voxel_column.mass_kg) or arrays_share_memory(child.active_layer.mass_kg,parent.active_layer.mass_kg):
        raise WindHandoffError('staged wind must not alias the accepted parent bed')
    p,c=wh._stamp(parent,'parent'),wh._stamp(child,'child')
    if not(p<=start and p<=c<=end): raise WindHandoffError('native commit stamps are not ordered')
    return dict(end_s=end,phys=phys,vec=vec,pool_bytes=b[0],slots=slots[0])


def _dirty(cp,pnb,cnb,active,B):
    """Resident byte comparison, same canonical zero-padding rule as nfs._changed_rows."""
    from maple.core.backend import to_host,to_device
    K=pnb.active_layer.mass_kg.shape[-1]; out=[]
    pv=pnb.voxel_column.mass_kg.reshape(-1,pnb.voxel_column.mass_kg.shape[2],K)
    cv=cnb.voxel_column.mass_kg.reshape(-1,cnb.voxel_column.mass_kg.shape[2],K)
    pa=pnb.active_layer.mass_kg.reshape(-1,K); ca=cnb.active_layer.mass_kg.reshape(-1,K)
    for lo in range(0,len(active),B):
        ids=active[lo:lo+B]; ii=to_device(np.ascontiguousarray(ids),cp)
        a,b=cv[ii].view(cp.uint64),pv[ii].view(cp.uint64); nv=min(a.shape[1],b.shape[1]); n=len(ids)
        changed=cp.any((a[:,:nv]!=b[:,:nv]).reshape(n,-1),axis=1)
        if a.shape[1]>nv: changed |= cp.any((a[:,nv:]!=0).reshape(n,-1),axis=1)
        if b.shape[1]>nv: changed |= cp.any((b[:,nv:]!=0).reshape(n,-1),axis=1)
        changed |= cp.any(ca[ii].view(cp.uint64)!=pa[ii].view(cp.uint64),axis=1)
        out.append(ids[np.asarray(to_host(changed))])  # B bytes, never the voxel rows
    return np.concatenate(out).astype(np.int64,copy=False) if out else np.empty(0,dtype=np.int64)


def start_wind_boundary(owner,bed,event_factory,*,bed_context,limits,available_bytes_fn,preferential=None,reserve_bytes=0,batch_cells=nfs.DEFAULT_BATCH_CELLS,before_publish=None,
                        host_available_bytes_fn=None,host_reserve_bytes=0,memory_budget_bytes=None,host_memory_budget_bytes=None,terrain_source=None):
    limits=wh._limits(limits)
    if type(owner) is not nfs.NativeFloatingSoil or not owner.is_current or owner.column.backend!='cuda':
        raise WindHandoffError('a genuine current native CUDA owner is required')
    cp,dev=flc._gate(owner.column)
    q=_q(available_bytes_fn,reserve_bytes,host_available_bytes_fn,host_reserve_bytes,memory_budget_bytes,host_memory_budget_bytes)
    B=fsl._count(batch_cells,'batch_cells',minimum=1)
    if B>limits.max_batch_cells or not isinstance(bed_context,BedContext) or not callable(event_factory):
        raise WindHandoffError('genuine static context, factory and admitted batch required')
    before_publish=wh._callback(before_publish,'before_publish')
    ny,nx=int(bed_context.geometry.ny),int(bed_context.geometry.nx); K=len(bed_context.grain_classes.classes); N=ny*nx
    # Terrain: explicit pairing with the native_terrain_20261123 source association (refused for a terrain-free context); metadata/structural only.
    tid=_terrain_pairing(cp,dev,bed_context,sb.terrain_identity(bed_context),terrain_source,ny,nx)
    V=_bed_meta(cp,dev,bed,ny,nx,K)
    if N>limits.max_cells or V>limits.max_voxels or K>limits.max_classes or owner.column.n_cells!=N:
        raise WindHandoffLimitError('native case dimensions exceed explicit handoff limits')
    # Mask download itself is admitted before the first transfer (plus the footprint mask on a terrain context: one bounded N-byte download).
    fcu.admit(cp,dev,0,N*(2 if tid else 1)+fcu.HASH_CHUNK_BYTES,q,'wind start host mask')
    from maple_syrup import floating_checkpoint_cuda as fcc
    active=np.ascontiguousarray(fcc.d2h(owner.column._maps['active']).reshape(ny,nx))
    if tid is not None and bed_context.domain is not None:
        valid=np.ascontiguousarray(fcc.d2h(bed_context.domain.valid_mask)).reshape(ny,nx).astype(bool,copy=False)
        if np.any(active & ~valid):
            raise WindHandoffError('a hydraulically active cell lies outside the native domain footprint (NoData)')
        del valid
    nb=wh._native_bed(bed_context,bed,active); nfs._check_bed(nb,owner.config); ncuda._check_devices(nb,dev)
    model=ncuda.estimate_cuda_memory(None,nb,owner.config,batch_cells=B)
    terrain_d=(8*N+8*N*K) if tid else 0  # fused dry-NoData water predicate temporaries (device)
    _admit(cp,dev,model,q,limits,'wind initial boundary',extra_d=bed.voxel_column.mass_kg.nbytes+bed.active_layer.mass_kg.nbytes+terrain_d,extra_h=fcu.HASH_CHUNK_BYTES,scan=True)
    if bool(flc._scalar(cp.any(bed.water.mobile_mass_by_cell_class_kg!=0.0))):
        raise WindHandoffError('initial boundary carries unresolved water-borne sediment')
    if tid is not None: _terrain_water_dry(bed,bed_context,'initial boundary')
    t0=float(owner.state.t_s); wh._native_valid(bed,bed_context,t0,'initial CUDA')
    vert=fp.FloatingPondVertical(owner,preferential=preferential,batch_cells=B,**q)
    event=event_factory(vert); lc._admit_native(event,vert)
    if event.control.backend!='cuda' or event.control.bed_storage!='dense' or tuple(event.shape)!=(ny,nx):
        raise WindHandoffError('initial native event must be the same Dense CUDA case')
    state=event.initial_state(bed,vert.initial_soil_water(),t_s=t0); lc._admit_state(event,vert,state,bed,t0)
    held=vert.held_template()
    if nfs._descriptor(wh._native_bed(event.context,bed,active))!=nfs._descriptor(nb):
        raise WindHandoffError('static event and wind context describe different native cases')
    storage=flc._storage_m3(vert,state,state.hydraulic.depth_m,q,cp,dev,'wind initial storage')
    led=dict(storage0_m3=storage,rain_m3=0.0,drainage_m3=0.0,outlet_export_m3=0.0,bottom_export_m3=0.0,surface_return_m3=0.0,bound_m3=0.0,n_refreshes=0,n_occurrences=0)
    prov=dict(cold_anchor_witness=held.source_witness,property_basis='initial_full',execution_epoch=None,cold_content_hash=None,area_m2=float(owner.column.cell_area_m2),backend='cuda')
    lineage=lc.FloatingEventLineage()
    with _minting(): ctx=WindContext(bed_context,(ny,nx),K,nfs._descriptor(nb),held.source_witness,owner.config.sha256,dev,tid)
    d=owner.column._maps['soil_depth']
    with lc._minting():
        root=lc.BoundaryRoot(owner.column,owner.matrix_op,preferential,owner.ks_m_s,d,d,owner.matrix_history,owner.state,owner.config,int(owner.binding.generation),held.source_witness,bed,vert._host_active,vert._edges_host,owner.depth_m)
        boundary=lc.Boundary('boundary',root,bed,state.hydraulic.depth_m,state.hydraulic.qx_m2_s,state.hydraulic.qy_m2_s,(state.topology,state.held_source_m,state.datum),t0,led,prov,0,None,None,None)
    proposal=(lineage,boundary,ctx)
    if before_publish is not None:
        expected=_integrity(cp,proposal); before_publish(*proposal)
        if _integrity(cp,proposal)!=expected: raise WindHandoffError('initial proposal changed during callback')
    if not owner.is_current: raise WindHandoffError('owner superseded during initial publication')
    stale=False
    try: nfs.retire_native_owner_to_lineage(owner)
    except nfs.NativeStaleError: stale=True; raise
    finally:
        if not stale and owner._authority.current is None: lineage._head=boundary
    return proposal


def stage_wind_handoff(lineage,parent,final,child,*,context,occurrence_id,start_s,limits,available_bytes_fn,occurrence_sha256=None,reserve_bytes=0,batch_cells=nfs.DEFAULT_BATCH_CELLS,
                       host_available_bytes_fn=None,host_reserve_bytes=0,memory_budget_bytes=None,host_memory_budget_bytes=None,terrain_source=None):
    lc._expect(lineage,parent,lc.Boundary)
    if parent.root.column.backend!='cuda': raise WindHandoffError('CUDA boundary required')
    cp,dev=flc._gate(parent.root.column)
    q=_q(available_bytes_fn,reserve_bytes,host_available_bytes_fn,host_reserve_bytes,memory_budget_bytes,host_memory_budget_bytes); comp=flc._component(q)
    limits=wh._limits(limits); oid,osha=wh._occurrence(occurrence_id,occurrence_sha256); start=wh._clock(start_s,'start_s')
    if type(context) is not WindContext or context not in _CONTEXTS or context.device_id!=dev:
        raise WindHandoffError('genuine matching CUDA WindContext required')
    tid=_terrain_pairing(cp,dev,context.bed_context,context.terrain_identity,terrain_source,*context.shape)
    if not parent.eligible_for_interevent or parent.pending_forcing is not None or parent.storm_match is not None or (parent.cursor is not None and parent.cursor.pending is not None):
        raise WindHandoffError('unfinished rain/recession is not wind eligible')
    if _LAST.get(lineage,[None])[0]==oid: raise WindHandoffError('immediately repeated occurrence')
    root=parent.root; cfg=root.config; ny,nx=context.shape; N=ny*nx; K=context.n_classes; L=root.column.n_layers
    if root.source_witness!=context.source_witness or parent.provenance['cold_anchor_witness']!=context.source_witness or cfg.sha256!=context.config_sha256:
        raise WindHandoffError('context differs from current cold witness/config')
    if start.hex()!=float(parent.t_s).hex() or float(root.water_state.t_s).hex()!=start.hex(): raise WindHandoffError('wind and native soil clocks differ')
    B=fsl._count(batch_cells,'batch_cells',minimum=1)
    if B>limits.max_batch_cells or N>limits.max_cells or K>limits.max_classes: raise WindHandoffLimitError('handoff dimensions/batch exceed limits')
    V=max(_bed_meta(cp,dev,parent.bed,ny,nx,K),_bed_meta(cp,dev,child,ny,nx,K))
    if V>limits.max_voxels: raise WindHandoffLimitError('native voxels exceed explicit handoff limit')
    meta=_wind_meta(cp,dev,final,child,parent.bed,start,context,limits); end=meta['end_s']
    v=_vegetation(parent,cp,dev)
    active=root.host_active
    if type(active) is not np.ndarray or active.dtype!=np.bool_ or active.size!=N or root.host_depth is None:
        raise WindHandoffError('sealed native CUDA host mask/depth records required')
    pnb,cnb=(wh._native_bed(context,b,active.reshape(ny,nx)) for b in (parent.bed,child))
    nfs._check_bed(pnb,cfg); nfs._check_bed(cnb,cfg); ncuda._check_devices(pnb,dev); ncuda._check_devices(cnb,dev)
    base=ncuda.estimate_cuda_memory(pnb,cnb,cfg,batch_cells=B)
    validity=child.voxel_column.mass_kg.nbytes+child.active_layer.mass_kg.nbytes
    extra_d=2*B*V*K*8+2*B*K*8+meta['pool_bytes']+validity+9*N*K+9*N+B*(V+1)*K+((8*N+8*N*K) if tid else 0)
    d0,h0=_admit(cp,dev,base,q,limits,'CUDA wind cold comparison',extra_d=extra_d,extra_h=fcu.HASH_CHUNK_BYTES+32*N,scan=True)
    from maple.core.backend import read_transfer_counters
    counter=read_transfer_counters()
    cls=wh._wind_values(final,meta)
    if not _same(cp,parent.bed.water.depth_m,child.water.depth_m) or not _same(cp,parent.bed.water.mobile_mass_by_cell_class_kg,child.water.mobile_mass_by_cell_class_kg):
        raise WindHandoffError('wind changed a surface water leaf')
    if bool(flc._scalar(cp.any(parent.bed.water.mobile_mass_by_cell_class_kg!=0.0))): raise WindHandoffError('parent carries unresolved water sediment')
    if tid is not None: _terrain_water_dry(parent.bed,context.bed_context,'parent boundary')  # the child carries the same leaves bitwise
    wh._native_valid(child,context.bed_context,end,'CUDA wind child'); wh._descriptor(context,pnb,'parent'); wh._descriptor(context,cnb,'child')
    factor=fm.held_cell_vegetation_factor(root.matrix_op)
    factor_sha=None
    if v is not None:
        from maple_syrup import floating_checkpoint_cuda as fcc
        hasher=hashlib.sha256();wvb._hash_chunked(hasher,np.ascontiguousarray(fcc.d2h(factor)));factor_sha=hasher.hexdigest()
        if factor_sha!=v.applied_g_sha256: raise WindHandoffError('held vegetation factor differs from accepted generation')
    dirty=_dirty(cp,pnb,cnb,np.flatnonzero(active.reshape(-1)).astype(np.int64),B)
    model=erc.event_refresh_model_cuda(cnb,cfg,n_selected=len(dirty),batch_cells=B,has_pref=root.pref_op is not None)
    d1,h1=_admit(cp,dev,model,q,limits,'CUDA wind selected reconcile',extra_d=validity,extra_h=fcu.HASH_CHUNK_BYTES+limits.max_receipt_bytes)
    maps=root.column._maps; col,op,hist,pref,ks,depth=root.column,root.matrix_op,root.history,root.pref_op,root.ks_m_s,root.host_depth
    water=root.water_state.water_m; receipt=None; changed=np.empty(0,dtype=np.int64); dc=changed; kc=changed; surf=None
    if dirty.size:
        fc,sa=cp.array(maps['theta_field_capacity'],copy=True),cp.array(maps['theta_saturation'],copy=True)
        newks=None if ks is None else cp.array(ks,copy=True); newdepth=np.array(depth,copy=True)
        old=dict(depth=depth,fc=maps['theta_field_capacity'],sat=maps['theta_saturation'],ks=ks)
        changed,kc,dc=ncuda._scatter_tiles(cp,cnb,cfg,dirty,fc,sa,newdepth,newks,model=model,B=int(model['batch_cells']),update_properties=True,old=old)
        if changed.size or dc.size or kc.size:
            area=nfs._area(cnb)
            if changed.size:
                water,surf,exp,acc=ncuda.reconcile_selected_columns(cp,cfg,root.column,root.water_state,depth,fc,sa,newdepth,changed,area,B,comp)
            col=ncuda._device_column(cp,cfg,fc,sa,ncuda._up(cp,newdepth),cp.array(maps['active'],copy=True),area,comp)
            op=ncuda._device_op(cfg,col,newks,comp,vegetation=factor)
            hist=fm.rebind_floating_matrix_history(root.matrix_op,root.history,op,**comp).history
            pref=None if root.pref_op is None else fpi.rebind_preferential_column(root.pref_op,col,**comp)
            ks,depth=newks,wh._seal(newdepth)
    state=fsl.floating_continued_state(col,water,end,**comp)
    if changed.size:
        vals=ncuda.inventory_closure_cuda(cp,cfg,root.column,root.water_state,col,state,water,changed,B,acc,comp,check_unrefreshed=True)
        receipt=nfs.NativeReconcileReceipt(whole_before_m3=vals[0],whole_after_m3=vals[1],whole_residual_m3=vals[2],whole_bound_m3=vals[3],old_column_sha256=root.column.sha256,new_column_sha256=col.sha256,**acc)
    if not _same(cp,root.history.wetting_m,hist.wetting_m) or not _same(cp,root.history.cumulative_intake_m,hist.cumulative_intake_m): raise WindHandoffError('wind reconcile changed J/I')
    if factor is not None and not _same(cp,factor,fm.held_cell_vegetation_factor(op)): raise WindHandoffError('wind reconcile changed held vegetation law')
    ret=0.0 if receipt is None else float(receipt.surface_return_m3)
    export=0.0 if receipt is None else float(receipt.bottom_export_m3)
    surface=child.water.depth_m; bed=child
    if ret>0.0:
        surface=surface+surf.reshape(ny,nx); bed=replace(child,water=replace(child.water,depth_m=surface))
    role='pending_recession' if ret>0.0 else 'boundary'
    soil=float(fsl.floating_storage_m3(col,state,**comp)); area=float(col.cell_area_m2)
    led=dict(parent.ledger); led['n_refreshes']+=1; led['bottom_export_m3']+=export; led['surface_return_m3']+=ret
    led['bound_m3']+=0.0 if receipt is None else float(receipt.whole_bound_m3)
    resid,bound=lc._balance_check(led,soil+flc._scalar(cp.sum(surface))*area); led['residual_m3'],led['residual_bound_m3']=resid,bound
    replaced=col is not root.column or op is not root.matrix_op; generation=int(root.generation)+int(replaced)
    prev=parent.provenance.get('execution_epoch') or ''; epoch=hashlib.sha256(f'{erc.EPOCH_KIND}|{prev}|wind|{oid}|{start.hex()}|{end.hex()}|{col.sha256}'.encode()).hexdigest()
    prov={**parent.provenance,'execution_epoch':epoch,'area_m2':area,'cold_content_hash':None}
    if replaced:
        from maple_syrup import floating_held_context_cuda as fhc
        prov.update(property_basis=erc.PROPERTY_BASIS,kernel=fhc.kernel_descriptor())
    lc._check_provenance_cuda(prov,root.source_witness)
    delta=read_transfer_counters().delta(counter)
    rec=dict(schema=RECEIPT_SCHEMA,profile=PROFILE,occurrence=dict(id=oid,sha256=osha),clocks=dict(start_s_hex=start.hex(),end_s_hex=end.hex(),elapsed_s=end-start),
        water=dict(wind_water_physics='none',surface_leaves_bitwise_equal=True,zero_time_reconcile=dict(performed=bool(changed.size),n_columns=len(changed),surface_return_m3=ret,bottom_export_m3=export,whole_bound_m3=0.0 if receipt is None else receipt.whole_bound_m3),soil_after_m3=soil,lifecycle_residual_m3=resid,lifecycle_bound_m3=bound,role=role),
        properties=dict(n_dirty=len(dirty),n_changed=len(changed),n_depth_changed=len(dc),ks_changed=bool(kc.size),root_replaced=replaced,new_generation=generation,new_column_sha256=col.sha256),
        vegetation=None if v is None else dict(generation=v.generation,recipe_id=v.recipe_id,binding_sha256=v.binding_sha256,held_factor_sha256_before=factor_sha,held_factor_sha256_after=factor_sha,covers='unchanged'),
        plan=dict(cold_scan_device_bytes=d0,cold_scan_host_bytes=h0,candidate_device_bytes=d1,candidate_host_bytes=h1,batch_cells=B,pool_bytes=meta['pool_bytes'],pool_slots=meta['slots'],n_cells=N,n_layers=L,dirty_comparison='resident_tiled_byte_comparison',integrity='native_bounded_cold_sha256_at_stage_and_adopt'),
        transfers_before_integrity={f.name:int(getattr(delta,f.name)) for f in fields(delta)},limits=limits.as_record(),scope='Dense CUDA; no hydrology/E/T during wind, no reroute, no reset; native selected capacity reconcile only',
        terrain=tid,terrain_source=None if tid is None else dict(profile=dcs.PROFILE_NATIVE_TERRAIN_20261123,package_sha256=terrain_source.package_sha256))
    text=wh._json(rec)
    if len(text.encode())>limits.max_receipt_bytes: raise WindHandoffLimitError('wind scalar receipt exceeds explicit cap')
    d=col._maps['soil_depth']
    p=wh._Parts(col,op,pref,ks,d,hist,state,cfg,generation,root.source_witness,bed,surface,parent.qx_m2_s,parent.qy_m2_s,parent.routing,end,led,prov,int(parent.seq)+1,parent.cursor,role)
    parts=_Parts(p,v,root.host_active,root.edges_host,depth)
    expected=_integrity(cp,_boundaries(parent,parts,oid,end,text))
    with _minting(): cand=WindHandoffCandidate(oid,end,role,text,weakref.ref(parent),weakref.ref(lineage),dict(parts=parts,consumed=False,accepted=None))
    _INTEGRITY[cand]=expected; _QUOTAS[cand]=(q,limits,validity,tid)
    return cand


def adopt_wind_handoff(lineage,parent,candidate,*,available_bytes_fn=None,reserve_bytes=0,before_publish=None,host_available_bytes_fn=None,host_reserve_bytes=0,memory_budget_bytes=None,host_memory_budget_bytes=None,terrain_source=None):
    lc._expect(lineage,parent,lc.Boundary)
    cp,dev=flc._gate(parent.root.column)
    q=_q(available_bytes_fn,reserve_bytes,host_available_bytes_fn,host_reserve_bytes,memory_budget_bytes,host_memory_budget_bytes)
    if type(candidate) is not WindHandoffCandidate or candidate not in _CANDIDATES: raise WindHandoffError('genuine CUDA candidate required')
    cell=candidate._cell
    if cell['consumed'] or cell['parts'] is None: raise WindHandoffError('wind candidate already consumed')
    if candidate._parent_ref() is not parent or candidate._lineage_ref() is not lineage: raise WindHandoffError('candidate belongs to another parent/lineage')
    parts=cell['parts']; quota=_QUOTAS.get(candidate)
    if type(parts) is not _Parts or quota is None or _INTEGRITY.get(candidate) is None: raise WindHandoffError('candidate has no authentic staging receipt')
    limits=quota[1]
    # Terrain pairing at adoption: the same explicit native_terrain_20261123 source association the staging was paired with (hot guard), or none.
    if quota[3] is None:
        if terrain_source is not None: raise WindHandoffError('terrain_source given for a terrain-free staged handoff')
    else:
        try: dcs.check_dense_cuda_source_hot(terrain_source,dcs.PROFILE_NATIVE_TERRAIN_20261123)
        except dcs.DenseCudaSourceError as exc: raise WindHandoffError(f'terrain pairing refused at adoption: {exc}') from None
    if type(candidate.receipt_json) is not str or len(candidate.receipt_json.encode())>limits.max_receipt_bytes: raise WindHandoffError('candidate scalar receipt changed shape')
    fcu.admit(cp,dev,0,fcu.HASH_CHUNK_BYTES+limits.max_receipt_bytes,q,'CUDA wind adoption integrity')
    callback=wh._callback(before_publish,'before_publish')
    if callback is not None: callback(); lc._expect(lineage,parent,lc.Boundary)
    if cell['consumed'] or cell['parts'] is not parts: raise WindHandoffError('candidate changed during callback')
    if _integrity(cp,_boundaries(parent,parts,candidate.occurrence_id,candidate.end_s,candidate.receipt_json))!=_INTEGRITY[candidate]:
        raise WindHandoffError('post-stage content integrity failed; nothing published')
    last=_LAST.get(lineage)
    if last is None: last=[None]; _LAST[lineage]=last
    if last[0]==candidate.occurrence_id: raise WindHandoffError('immediately repeated wind occurrence')
    p=parts.base
    with lc._minting():
        root=lc.BoundaryRoot(p.column,p.matrix_op,p.pref_op,p.ks_m_s,p.depth_m,p.depth_m,p.history,p.water_state,p.config,p.generation,p.source_witness,p.bed,parts.host_active,parts.edges_host,parts.host_depth,vegetation=parts.vegetation)
        boundary=lc.Boundary(p.role,root,p.bed,p.surface_depth_m,p.qx_m2_s,p.qy_m2_s,p.routing,p.t_s,p.ledger,p.provenance,p.seq,p.cursor,None,None)
    ref=weakref.ref(boundary); cell['accepted']=ref
    try: lineage._head=boundary
    finally:
        if lineage.head is boundary: cell['consumed']=True; cell['parts']=None; last[0]=candidate.occurrence_id
    return boundary


def restore_wind_context(*args,**kwargs):
    raise WindHandoffError('combined Dense CUDA wind context restore requires separate qualification')
