'''Explicit transpiration withdrawal for the native layered soil column (maple_syrup.subsurface).

Opt-in extension: the existing `subsurface.step` (intake -> capacity return -> drainage -> soil
evaporation -> diffusion -> closure) is reused unchanged; transpiration is a SEPARATE, explicit
withdrawal applied to the hydro result (first-order hydro -> T operator splitting).

Units: water is water-equivalent DEPTH in m on the [L, N] column grid; `requested_water_m` is the
depth demanded over the ALREADY DEFINED interval (the caller fixes dt; this module never scales a rate).
Volumes are depth * column.cell_area_m2 (multiplied exactly once, in the budget) summed over ACTIVE
cells only (same definition as the native storage_m3 / StepBudget).

Withdrawal law (per layer/cell): available = max(w - residual_m[l], 0); actual = min(requested,
available); water' = w - actual; unmet = requested - actual; scale = actual / requested (0 for zero
demand). Inactive cells carry zero requests, so their water is preserved exactly. No wetter-layer
compensation, no species wilting law, no clipping after subtraction.

Order of every public operation: metadata/time checks (pinned native helpers, no allocation) ->
scalar guards -> scalar memory admission -> allocation -> fused compiled kernel that computes the
fresh outputs AND per-cell validity flags/reductions -> ONE host packet -> reject or publish.
'''
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from maple_syrup import subsurface as ss
from maple_syrup.conservation import volume_roundoff_bound_m3

try:  # GPU host transfer helper (single small packet per withdrawal)
    from maple.core.backend import to_host as _to_host
except ImportError:  # pragma: no cover - environment dependent
    def _to_host(a):
        return np.asarray(a.get() if hasattr(a, 'get') else a)

SubsurfaceError = ss.SubsurfaceError
SubsurfaceMemoryError = ss.SubsurfaceMemoryError
SubsurfaceBackendUnavailableError = ss.SubsurfaceBackendUnavailableError

VERSION = '0.2.0'
THREADS = 256
_F8 = 8
_N_ROWS = 8        # per-cell reduction rows written by the fused kernel (see _ROW_* below)
_PACKET = _N_ROWS  # host packet: one scalar per row
_ROW_W_BEFORE, _ROW_W_AFTER, _ROW_ACTUAL, _ROW_UNMET, _ROW_REQ, _ROW_MIN_AFTER, _ROW_FLAGS, _ROW_QMAX = range(8)
_FLAG_WATER, _FLAG_REQUEST, _FLAG_INACTIVE = 1.0, 2.0, 4.0
_FLAG_TEXT = {1: 'soil water not finite, negative or above saturation_m + capacity_tolerance_m',
              2: 'requested_water_m not finite or negative', 4: 'positive requested_water_m on an inactive cell'}


# ------------------------------------------------------------------ strict scalar guards
def _as_size(v, name):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or int(v) < 1:
        raise SubsurfaceError(f'{name} must be a non-boolean integer >= 1, got {v!r}')
    return int(v)


def _as_bytes(v, name):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or int(v) < 0:
        raise SubsurfaceError(f'{name} must be a non-boolean integer >= 0, got {v!r}')
    return int(v)


def _check_backend(backend):
    if backend not in ('numpy', 'numba', 'cuda'):
        raise SubsurfaceError(f'unknown backend {backend!r}')
    return backend


# ------------------------------------------------------------------ memory estimates / admission
@dataclass(frozen=True)
class TranspirationMemory:
    '''Byte estimate for one withdrawal. parameter_bytes are the operator's persistent per-layer vectors
    (residual, saturation bound); the active mask is the column's own array (reused, not duplicated).'''
    backend: str
    n_layers: int
    n_cells: int
    parameter_bytes: int      # persistent residual[L] + saturation bound[L] on the compute side
    state_bytes: int          # caller soil water [L,N] live during the operation
    input_bytes: int          # requested_water_m [L,N]
    output_bytes: int         # water' + actual + unmet + scale, all [L,N]
    row_bytes: int            # fused per-cell reduction rows [8, N]
    scratch_bytes: int        # backend temporaries (numpy reference: [L,N]-sized) + packet
    packet_bytes: int         # host packet (8 doubles)
    host_bytes: int
    device_bytes: int
    reserve_bytes: int        # column.reserve_bytes
    extra_reserve_bytes: int  # caller-owned simultaneously live bytes
    required_bytes: int       # admitted side (host for numpy/numba, device for cuda) + reserves
    total_bytes: int          # host + device + reserves (native convention)

    def as_dict(self):
        return dict(self.__dict__)


@dataclass(frozen=True)
class CombinedMemory:
    '''Combined admission for step_with_transpiration: original state + forcing + hydro outputs still
    live + T parameters/inputs/outputs/rows/scratch + column reserve + caller reserve.'''
    backend: str
    hydro: object
    transpiration: TranspirationMemory
    state_bytes: int
    host_bytes: int
    device_bytes: int
    reserve_bytes: int
    extra_reserve_bytes: int
    required_bytes: int
    available_bytes: int   # -1 when the column has no available_bytes_fn
    peak_bytes: int
    total_bytes: int

    def as_dict(self):
        d = dict(self.__dict__)
        d['hydro'] = dict(getattr(self.hydro, '__dict__', {})) or repr(self.hydro)
        d['transpiration'] = self.transpiration.as_dict()
        return d


def estimate_transpiration_memory(n_layers, n_cells, *, backend, reserve_bytes=0, extra_reserve_bytes=0,
                                  include_state=True):
    '''Scalar byte estimate of a withdrawal (no allocation). Numba/CUDA kernels need only the [8, N]
    rows; the NumPy reference additionally forms [L,N]-sized temporaries (available depth, actual mask,
    masked water products, validity masks): 6 FP64 [L,N] equivalents are reserved for it.
    scratch_bytes (documented, conservative): backend temporaries + reduction workspace for the eight
    row reductions (one FP64 N-row equivalent) + the eight 0-d reduction results, the stacked compute-side
    packet and the host packet all simultaneously live (3 packets).'''
    L, N = _as_size(n_layers, 'n_layers'), _as_size(n_cells, 'n_cells')
    _check_backend(backend)
    reserve, extra = _as_bytes(reserve_bytes, 'reserve_bytes'), _as_bytes(extra_reserve_bytes, 'extra_reserve_bytes')
    LN = _F8 * L * N
    param, inp, out, rows, packet = 2 * _F8 * L, LN, 4 * LN, _N_ROWS * _F8 * N, _F8 * _PACKET
    state = LN if include_state else 0
    scratch = (6 * LN if backend == 'numpy' else 0) + _F8 * N + 3 * packet
    compute = param + state + inp + out + rows + scratch
    host = compute if backend != 'cuda' else packet
    device = compute if backend == 'cuda' else 0
    side = device if backend == 'cuda' else host
    required = side + reserve + extra
    return TranspirationMemory(backend, L, N, param, state, inp, out, rows, scratch, packet, host, device,
                               reserve, extra, required, host + device + reserve + extra)


def _available(column):
    fn = getattr(column, 'available_bytes_fn', None)
    if fn is None:
        return None
    v = fn()
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or int(v) < 0:
        raise SubsurfaceError(f'available_bytes_fn must return a non-boolean integer >= 0, got {v!r}')
    return int(v)


def _admit(column, required_bytes, stage):
    '''Scalar admission BEFORE any allocation; refuses without mutating anything or raising budgets.'''
    avail = _available(column)
    if avail is not None and int(required_bytes) > avail:
        raise SubsurfaceMemoryError(f'{stage}: required {int(required_bytes)} bytes exceed available {avail} bytes '
                                    f'(column reserve {int(column.reserve_bytes)} bytes included); nothing allocated')
    return avail


# ------------------------------------------------------------------ fused kernels (outputs + flags + rows)
def _fused_numpy(w, q, residual, satb, active_u8, out_w, actual, unmet, scale, rows):
    '''NumPy reference of the fused kernel (vectorized; [L,N] temporaries are counted as scratch).'''
    act = active_u8 != 0
    avail = w - residual[:, None]
    np.maximum(avail, 0.0, out=avail)
    np.minimum(q, avail, out=actual)
    np.subtract(w, actual, out=out_w)
    np.subtract(q, actual, out=unmet)
    scale.fill(0.0)
    np.divide(actual, q, out=scale, where=q > 0.0)
    with np.errstate(invalid='ignore'):
        bad_w = ~((w >= 0.0) & (w <= satb[:, None]))
        bad_q = ~(q >= 0.0) | (q == np.inf)
        bad_inactive = (q > 0.0) & ~act[None, :]
        rows[_ROW_W_BEFORE] = np.where(act, w.sum(axis=0), 0.0)
        rows[_ROW_W_AFTER] = np.where(act, out_w.sum(axis=0), 0.0)
        rows[_ROW_ACTUAL] = np.where(act, actual.sum(axis=0), 0.0)
        rows[_ROW_UNMET] = np.where(act, unmet.sum(axis=0), 0.0)
        rows[_ROW_REQ] = np.where(act, q.sum(axis=0), 0.0)
        rows[_ROW_MIN_AFTER] = out_w.min(axis=0)
        rows[_ROW_FLAGS] = (bad_w.any(axis=0) * _FLAG_WATER + bad_q.any(axis=0) * _FLAG_REQUEST
                            + bad_inactive.any(axis=0) * _FLAG_INACTIVE)
        rows[_ROW_QMAX] = q.max(axis=0)


try:
    from numba import njit as _njit

    @_njit(cache=True)
    def _fused_numba(w, q, residual, satb, active_u8, out_w, actual, unmet, scale, rows):
        L, N = w.shape
        inf = np.inf
        for j in range(N):
            rows[0, j] = 0.0
            rows[1, j] = 0.0
            rows[2, j] = 0.0
            rows[3, j] = 0.0
            rows[4, j] = 0.0
            rows[5, j] = inf
            rows[6, j] = 0.0
            rows[7, j] = 0.0
        for l in range(L):
            r = residual[l]
            sb = satb[l]
            for j in range(N):
                wv = w[l, j]
                qv = q[l, j]
                act = active_u8[j] != 0
                f = rows[6, j]
                if not (wv >= 0.0 and wv <= sb):
                    f = f if (int(f) & 1) else f + 1.0
                if not (qv >= 0.0) or qv == inf:
                    f = f if (int(f) & 2) else f + 2.0
                if (not act) and qv > 0.0:
                    f = f if (int(f) & 4) else f + 4.0
                rows[6, j] = f
                a = wv - r
                if a < 0.0:
                    a = 0.0
                t = qv if qv < a else a
                o = wv - t
                out_w[l, j] = o
                actual[l, j] = t
                unmet[l, j] = qv - t
                scale[l, j] = t / qv if qv > 0.0 else 0.0
                if act:
                    rows[0, j] += wv
                    rows[1, j] += o
                    rows[2, j] += t
                    rows[3, j] += qv - t
                    rows[4, j] += qv
                if o < rows[5, j]:
                    rows[5, j] = o
                if qv > rows[7, j]:
                    rows[7, j] = qv
except ImportError:  # pragma: no cover
    _fused_numba = None

_CUDA_SRC = r'''
extern "C" __global__ void withdraw_fused(const double* __restrict__ w, const double* __restrict__ q,
                                          const double* __restrict__ residual, const double* __restrict__ satb,
                                          const unsigned char* __restrict__ active_u8,
                                          double* __restrict__ out_w, double* __restrict__ actual,
                                          double* __restrict__ unmet, double* __restrict__ scale,
                                          double* __restrict__ rows, const int L, const long long N) {
    long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    /* IEEE-754 positive infinity via a header-free CUDA device intrinsic (NVRTC has no INFINITY/math.h) */
    const double positive_inf = __longlong_as_double(0x7ff0000000000000LL);
    const bool act = active_u8[j] != 0;
    double sw = 0.0, sa = 0.0, st = 0.0, su = 0.0, sq = 0.0, mn = positive_inf, qm = 0.0;
    int flags = 0;
    for (int l = 0; l < L; ++l) {
        long long i = (long long)l * N + j;
        double wv = w[i];
        double qv = q[i];
        if (!(wv >= 0.0 && wv <= satb[l])) flags |= 1;
        if (!(qv >= 0.0) || qv == positive_inf) flags |= 2;
        if (!act && qv > 0.0) flags |= 4;
        double a = wv - residual[l];
        if (a < 0.0) a = 0.0;
        double t = (qv < a) ? qv : a;
        double o = wv - t;
        out_w[i] = o; actual[i] = t; unmet[i] = qv - t; scale[i] = (qv > 0.0) ? (t / qv) : 0.0;
        if (act) { sw += wv; sa += o; st += t; su += qv - t; sq += qv; }
        if (o < mn) mn = o;
        if (qv > qm) qm = qv;
    }
    rows[0 * N + j] = sw; rows[1 * N + j] = sa; rows[2 * N + j] = st; rows[3 * N + j] = su; rows[4 * N + j] = sq;
    rows[5 * N + j] = mn; rows[6 * N + j] = (double)flags; rows[7 * N + j] = qm;
}
'''


# ------------------------------------------------------------------ operator
@dataclass(frozen=True, eq=False)
class TranspirationOperator:
    '''Prepared (compiled) withdrawal operator bound to one native column. Persistent compute-side
    parameters: residual depth [L] and saturation bound [L]; the active mask is column.active_u8.'''
    column: object
    backend: str
    xp: object
    n_layers: int
    n_cells: int
    residual_xp: object
    saturation_xp: object
    kernel: object
    memory: TranspirationMemory
    prepare_bytes: int

    def check_device(self):
        if self.backend == 'cuda':
            dev = int(self.xp.cuda.Device().id)
            if self.column.device_id is not None and dev != int(self.column.device_id):
                raise SubsurfaceBackendUnavailableError(
                    f'current CUDA device {dev} differs from column device {self.column.device_id}')
        pf = getattr(self.column, 'preflight', None)
        if callable(pf) and pf() is False:
            raise SubsurfaceBackendUnavailableError('column preflight failed')


def prepare_transpiration(column, *, extra_reserve_bytes=0):
    '''Compile/prepare outside daily work. Scalar admission (parameter vectors + tiny warm-up) precedes
    every allocation, including the small per-layer host vectors. Columns are never mutated.'''
    ss._check_column(column)
    backend = _check_backend(column.backend)
    extra = _as_bytes(extra_reserve_bytes, 'extra_reserve_bytes')
    L, N = _as_size(column.n_layers, 'column.n_layers'), _as_size(column.n_cells, 'column.n_cells')
    reserve = _as_bytes(column.reserve_bytes, 'column.reserve_bytes')
    xp = column.xp
    if backend == 'numba' and _fused_numba is None:
        raise SubsurfaceBackendUnavailableError('numba backend requested but numba is not importable')
    if backend == 'cuda':
        dev = int(xp.cuda.Device().id)
        if column.device_id is not None and dev != int(column.device_id):
            raise SubsurfaceBackendUnavailableError(
                f'current CUDA device {dev} differs from column device {column.device_id}')
    mem = estimate_transpiration_memory(L, N, backend=backend, reserve_bytes=reserve, extra_reserve_bytes=extra)
    warm = estimate_transpiration_memory(1, 1, backend=backend)  # (1,1) compiled warm-up launch
    host_param = 2 * _F8 * L + 4 * _F8 * L                       # host residual/saturation vectors + their
    prepare_bytes = host_param + mem.parameter_bytes + warm.total_bytes + 1  # finite/sign validation temporaries
    _admit(column, prepare_bytes + reserve + extra, 'prepare_transpiration')
    layers = column.layers
    residual = np.ascontiguousarray(np.asarray(layers.residual_m, dtype=np.float64))
    satb = np.ascontiguousarray(np.asarray(layers.saturation_m, dtype=np.float64)
                                + np.asarray(layers.capacity_tolerance_m, dtype=np.float64))
    if residual.shape != (L,) or satb.shape != (L,):
        raise SubsurfaceError('layer residual/saturation/tolerance arrays must have shape [n_layers]')
    if not (np.all(np.isfinite(residual)) and np.all(residual >= 0.0) and np.all(np.isfinite(satb))):
        raise SubsurfaceError('layer residual_m must be finite and non-negative; saturation/tolerance finite')
    kernel = None
    if backend == 'cuda':
        kernel = xp.RawKernel(_CUDA_SRC, 'withdraw_fused', options=('-std=c++11',))
        kernel.compile()
    elif backend == 'numba':
        kernel = _fused_numba
    residual_xp = xp.ascontiguousarray(xp.asarray(residual, dtype=xp.float64))
    saturation_xp = xp.ascontiguousarray(xp.asarray(satb, dtype=xp.float64))
    op = TranspirationOperator(column, backend, xp, L, N, residual_xp, saturation_xp, kernel, mem, prepare_bytes)
    one = xp.ones((1, 1), dtype=xp.float64)
    outs = [xp.empty((1, 1), dtype=xp.float64) for _ in range(4)]
    _launch(op, one, xp.zeros((1, 1), dtype=xp.float64), residual_xp[:1], saturation_xp[:1],
            xp.ones(1, dtype=xp.uint8), *outs, xp.empty((_N_ROWS, 1), dtype=xp.float64))
    if backend == 'cuda':
        xp.cuda.Device().synchronize()
    return op


def _launch(op, w, q, residual, satb, active_u8, out_w, actual, unmet, scale, rows):
    if op.backend == 'numpy':
        _fused_numpy(w, q, residual, satb, active_u8, out_w, actual, unmet, scale, rows)
    elif op.backend == 'numba':
        op.kernel(w, q, residual, satb, active_u8, out_w, actual, unmet, scale, rows)
    else:
        N = int(w.shape[1])
        op.kernel(((N + THREADS - 1) // THREADS,), (THREADS,),
                  (w, q, residual, satb, active_u8, out_w, actual, unmet, scale, rows, np.int32(w.shape[0]), np.int64(N)))


# ------------------------------------------------------------------ withdrawal
@dataclass(frozen=True)
class TranspirationBudget:
    '''All volumes over ACTIVE cells (native storage definition); area multiplied exactly once.'''
    storage_before_m3: float
    storage_after_m3: float
    requested_m3: float
    actual_m3: float
    unmet_m3: float
    residual_m3: float          # after - before + actual
    request_residual_m3: float  # requested - actual - unmet
    bound_m3: float             # volume_roundoff_bound_m3 (MAPLE FP64 summation policy)
    n_terms: int
    min_water_after_m: float    # over all cells (inactive included)
    max_requested_m: float

    def as_dict(self):
        return dict(self.__dict__)


@dataclass(frozen=True, eq=False)
class TranspirationResult:
    state: object        # fresh authoritative SoilColumnState (time preserved)
    actual_m: object
    unmet_m: object
    scale: object
    budget: TranspirationBudget
    memory: TranspirationMemory


def _run_withdrawal(op, state, requested, extra, stage):
    '''Shared body: metadata -> admission -> allocation -> fused kernel -> ONE packet -> reject/return.'''
    column, xp = op.column, op.xp
    t_s = ss._check_state(column, state)
    ss._check_dynamic(column, 'requested_water_m', requested, (op.n_layers, op.n_cells))
    mem = estimate_transpiration_memory(op.n_layers, op.n_cells, backend=op.backend,
                                        reserve_bytes=column.reserve_bytes, extra_reserve_bytes=extra)
    _admit(column, mem.required_bytes, stage)
    shape = (op.n_layers, op.n_cells)
    out_w, actual, unmet, scale = (xp.empty(shape, dtype=xp.float64) for _ in range(4))
    rows = xp.empty((_N_ROWS, op.n_cells), dtype=xp.float64)
    w = state.water_m
    _launch(op, w, requested, op.residual_xp, op.saturation_xp, column.active_u8, out_w, actual, unmet, scale, rows)
    packet = xp.stack([rows[_ROW_W_BEFORE].sum(), rows[_ROW_W_AFTER].sum(), rows[_ROW_ACTUAL].sum(),
                       rows[_ROW_UNMET].sum(), rows[_ROW_REQ].sum(), rows[_ROW_MIN_AFTER].min(),
                       rows[_ROW_FLAGS].max(), rows[_ROW_QMAX].max()])
    p = np.asarray(_to_host(packet), dtype=np.float64)
    del rows
    flags = p[_ROW_FLAGS]
    if not (flags == 0.0):  # NaN flags are impossible (integers); any nonzero => reject, nothing published
        bits = int(flags) if math.isfinite(flags) else 7
        why = '; '.join(t for b, t in _FLAG_TEXT.items() if bits & b) or 'invalid values'
        raise SubsurfaceError(f'{stage}: {why} (checked on every cell, inactive included)')
    area = float(column.cell_area_m2)
    before, after, act, unm, req = (float(v) * area for v in p[:5])
    wmin, qmax = float(p[_ROW_MIN_AFTER]), float(p[_ROW_QMAX])
    n_terms = 3 * op.n_layers * int(column.n_active) + _PACKET
    bound = float(volume_roundoff_bound_m3(n_terms, max(before + req, 0.0)))
    budget = TranspirationBudget(before, after, req, act, unm, after - before + act, req - act - unm, bound,
                                 n_terms, wmin, qmax)
    if not (math.isfinite(after) and math.isfinite(act) and math.isfinite(unm) and math.isfinite(wmin)) or wmin < 0.0:
        raise SubsurfaceError(f'{stage}: withdrawal produced non-finite or negative water: {budget}')
    if abs(budget.residual_m3) > bound or abs(budget.request_residual_m3) > bound:
        raise SubsurfaceError(f'{stage}: transpiration closure violated beyond the FP64 bound: {budget}')
    return TranspirationResult(ss.SoilColumnState(out_w, t_s), actual, unmet, scale, budget, mem)


def withdraw_transpiration(operator, state, requested_water_m, *, extra_reserve_bytes=0):
    '''Pure native withdrawal (see module docstring). Inputs are never modified; on any rejection nothing
    is published. Exactly one host packet.'''
    op = operator
    if not isinstance(op, TranspirationOperator):
        raise SubsurfaceError('operator must come from prepare_transpiration')
    extra = _as_bytes(extra_reserve_bytes, 'extra_reserve_bytes')
    op.check_device()
    return _run_withdrawal(op, state, requested_water_m, extra, 'withdraw_transpiration')


# ------------------------------------------------------------------ combined step
@dataclass(frozen=True)
class CoupledBudget:
    '''Active-cell volumes. after - before = added - surface_return - drainage - evaporation - transpiration
    (E and T distinct); before/after are the true endpoints (native hydro before, T after). residual is the
    closure of that equation; bound from volume_roundoff_bound_m3 (no new epsilon).'''
    storage_before_m3: float
    storage_after_m3: float
    added_m3: float
    surface_return_m3: float
    drainage_m3: float
    evaporation_m3: float
    evaporation_unmet_m3: float
    transpiration_m3: float
    transpiration_unmet_m3: float
    transpiration_requested_m3: float
    residual_m3: float
    bound_m3: float
    n_terms: int
    intermediate_mismatch_m3: float  # hydro after vs T before (same intermediate state, two reductions)

    def as_dict(self):
        return dict(self.__dict__)


@dataclass(frozen=True, eq=False)
class CoupledStep:
    '''Final authoritative state plus hydro diagnostics and T outputs. The intermediate (post-hydro,
    pre-T) water is NOT retained.'''
    state: object
    dt_s: float
    surface_return_m: object
    drainage_m: object
    evaporation_m: object
    evaporation_unmet_m: object
    percolation_m: object
    diffusion_m: object
    hydro_budget: object
    actual_m: object
    unmet_m: object
    scale: object
    transpiration_budget: TranspirationBudget
    budget: CoupledBudget
    memory: CombinedMemory

    def diagnostic_bytes(self):
        return int(sum(int(a.nbytes) for a in (self.surface_return_m, self.drainage_m, self.evaporation_m,
                                                self.evaporation_unmet_m, self.percolation_m, self.diffusion_m,
                                                self.actual_m, self.unmet_m, self.scale)))


def estimate_combined_memory(column, *, extra_reserve_bytes=0, query_available=True, with_evaporation_share=False):
    '''Scalar combined estimate (no allocation): hydro step estimate + T estimate (state counted once)
    + column reserve + caller reserve. query_available=False never calls available_bytes_fn.
    with_evaporation_share=True counts the caller's optional (N,) post-cap evaporation share field.'''
    L, N = _as_size(column.n_layers, 'column.n_layers'), _as_size(column.n_cells, 'column.n_cells')
    reserve, extra = _as_bytes(column.reserve_bytes, 'column.reserve_bytes'), _as_bytes(extra_reserve_bytes, 'extra_reserve_bytes')
    backend = _check_backend(column.backend)
    if type(with_evaporation_share) is not bool:
        raise SubsurfaceError(f'with_evaporation_share must be a bool, got {with_evaporation_share!r}')
    # the hydrology estimate must follow the column's ACTUAL diffusion mode (the donor-depth mode carries extra
    # parameter vectors and a second factor scratch); the historical constant mode is the default and unchanged
    hm = ss.estimate_step_memory(L, N, backend=backend, reserve_bytes=0,
                                 diffusion_mode=getattr(column.layers, "diffusion_mode", "constant"),
                                 with_evaporation_share=with_evaporation_share)
    tm = estimate_transpiration_memory(L, N, backend=backend, reserve_bytes=0, include_state=False)
    state_bytes = _F8 * L * N
    host = int(hm.host_bytes) + tm.host_bytes + (0 if backend == 'cuda' else state_bytes)
    device = int(hm.device_bytes) + tm.device_bytes + (state_bytes if backend == 'cuda' else 0)
    side = device if backend == 'cuda' else host
    required = side + reserve + extra
    avail = _available(column) if query_available else None
    return CombinedMemory(backend, hm, tm, state_bytes, host, device, reserve, extra, required,
                          -1 if avail is None else avail, required, host + device + reserve + extra)


def step_with_transpiration(operator, state, dt_s, *, added_water_m, potential_evaporation_m_s,
                            requested_transpiration_m, extra_reserve_bytes=0, evaporation_share=None):
    '''First-order splitting: native hydro `step` (advances t_s by dt once; its own packet) followed by
    the fused withdrawal (time preserved; ONE packet). dt=0 is an identity and rejects any positive
    requested transpiration (flagged from the fused packet, nothing published). Combined admission
    precedes any work. `evaporation_share` (optional (N,) field) is passed through to the native step's
    post-cap evaporation share law (see subsurface.step); None keeps the historical law.'''
    op = operator
    if not isinstance(op, TranspirationOperator):
        raise SubsurfaceError('operator must come from prepare_transpiration')
    extra = _as_bytes(extra_reserve_bytes, 'extra_reserve_bytes')
    op.check_device()
    column = op.column
    dt = ss._check_dt(dt_s)
    ss._check_state(column, state)
    ss._check_dynamic(column, 'requested_transpiration_m', requested_transpiration_m, (op.n_layers, op.n_cells))
    if evaporation_share is not None:
        ss._check_dynamic(column, 'evaporation_share', evaporation_share, (op.n_cells,))
    mem = estimate_combined_memory(column, extra_reserve_bytes=extra, with_evaporation_share=evaporation_share is not None)
    _admit(column, mem.required_bytes, 'step_with_transpiration')
    hydro = ss.step(column, state, dt, added_water_m=added_water_m, potential_evaporation_m_s=potential_evaporation_m_s,
                    evaporation_share=evaporation_share)
    tr = _run_withdrawal(op, hydro.state, requested_transpiration_m, extra, 'step_with_transpiration')
    hb, tb = hydro.budget, tr.budget
    if dt == 0.0 and tb.max_requested_m > 0.0:
        raise SubsurfaceError('dt_s=0 is an identity step and rejects any positive requested transpiration')
    before, after = float(hb.storage_before_m3), tb.storage_after_m3
    added, ret, drain, evap = (float(v) for v in (hb.added_m3, hb.surface_return_m3, hb.drainage_m3, hb.evaporation_m3))
    residual = after - before - (added - ret - drain - evap - tb.actual_m3)
    n_terms = int(hb.n_terms) + tb.n_terms
    bound = float(volume_roundoff_bound_m3(n_terms, max(before + added + tb.requested_m3, 0.0)))
    mismatch = float(hb.storage_after_m3) - tb.storage_before_m3
    budget = CoupledBudget(before, after, added, ret, drain, evap, float(hb.evaporation_unmet_m3), tb.actual_m3,
                           tb.unmet_m3, tb.requested_m3, residual, bound, n_terms, mismatch)
    if abs(residual) > bound or abs(mismatch) > bound:
        raise SubsurfaceError(f'combined hydro+transpiration closure violated beyond the FP64 bound: {budget}')
    return CoupledStep(tr.state, dt, hydro.surface_return_m, hydro.drainage_m, hydro.evaporation_m,
                       hydro.evaporation_unmet_m, hydro.percolation_m, hydro.diffusion_m, hb,
                       tr.actual_m, tr.unmet_m, tr.scale, tb, budget, mem)


__all__ = ['prepare_transpiration', 'withdraw_transpiration', 'step_with_transpiration',
           'estimate_transpiration_memory', 'estimate_combined_memory', 'TranspirationOperator',
           'TranspirationResult', 'TranspirationBudget', 'TranspirationMemory', 'CombinedMemory',
           'CoupledStep', 'CoupledBudget', 'SubsurfaceError', 'SubsurfaceMemoryError',
           'SubsurfaceBackendUnavailableError', 'VERSION']
