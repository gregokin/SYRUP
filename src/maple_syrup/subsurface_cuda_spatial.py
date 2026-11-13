"""CuPy RawKernel CUDA evolution of an explicit spatial parameter set (`spatial_soil_layers`, backend "cuda"). Trial component only (S2-A.4.2).

ONE physical body: the stage statements are `subsurface_cuda._BODY`, rendered with the SPATIAL reader macros below through the pure `subsurface_cuda.render_body_source`. The body becomes the
`__device__` function `spatial_column` (its early `return` is the invalid-lane guard of a padded lane); a wrapper kernel then lets EVERY lane, padded or inactive, reach the block reduction. The
readers are the A1 strided raw maps `p[l * stride_layer + i * stride_cell]` (scalar 0/0, shared 1/0, cell N/1): `DZ = dz`, `RES_M = theta_res * dz`, `FC_M = theta_fc * dz`, `SAT_M = theta_sat * dz`,
`CAP_TOL = coef * SAT_M` in exactly the product order of `soil_layers`/A1, evaluated where used (registers; no per-thread array beyond the 7 contribution doubles, no `(L, N)` derived tensor, no host
mirror). Interface factors: when the conductance maps are SHARED (`(L-1,)`), the existing common factor kernel is launched on them and the column kernel reads the scratch (`hoisted = 1`); per-cell
maps evaluate `relax_factor(G, 1/pair, dt)` per cell and interface. Constant mode passes the downward conductance as the upward one (alias); a one-layer stack has no interface maps (dummy never-read pointers).

Launches on the current stream, in order: [factor kernel if hoisted] -> `maple_syrup_spatial_step` (per-lane contributions: before, after, added, surface return, drainage, evaporation, unmet, each 0 for a
padded lane or an inactive cell; a block tree in STATIC shared memory `7 * 8 * THREADS` B; block partials `7 * nb` doubles; integer atomics OR the flag words and count flagged cells) ->
`maple_syrup_spatial_step_finalize` (one block, `FINAL_THREADS` threads, fixed tree) -> ONE `PACKET_BYTES` = 72 B read of the 9-word uint64 packet [OR of flags, flagged count, 7 total bit patterns]. No CUB, no
CuPy reduction, no scalar array. The host rebuilds the original nine-value float packet (lowest set bit of the OR = the minimum per-cell lowest bit; exact count; totals) and `subsurface._resolve`
applies the original closure coefficient, term count and typed errors. No `rows` array exists. Also: `maple_syrup_spatial_uniform` (host L-vector -> `(L, N)` water, threshold checks in EVERY cell, 4-word
diagnostic), `maple_syrup_spatial_moisture`, `maple_syrup_spatial_storage` (+ the generic finalize with one row).

Memory/lifetimes (see `operation_terms`): scratch (partials, packet, factor arrays, the uniform theta copy) are Python locals held until after the synchronising packet read, then released to the CuPy
pool; outputs are fresh pool arrays. Static shared memory is reported, not allocated globally. Excluded and declared: NVRTC/module load, CUDA context, allocator pool policy/rounding, interpreter. The state copy and
moisture query may complete asynchronously on the current stream (the caller owns later cross-stream ordering). Written by its author without a GPU; Root qualifies it on hardware.
"""

from __future__ import annotations

import hashlib
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np

from maple_syrup import subsurface as ss
from maple_syrup import subsurface_cuda as sc

__all__ = ["COMPILE_OPTIONS", "FINAL_THREADS", "MAX_GRID", "PACKET_BYTES", "PACKET_WORDS", "STEP_HOST_PACKET_BYTES", "STEP_ROWS", "THREADS", "UNIFORM_DIAG_WORDS", "factor_arrays", "kernel_provenance",
           "kernel_source", "load_kernels", "moisture", "n_blocks", "operation_terms", "render_source", "run_step", "storage_depth", "uniform_water"]

COMPILE_OPTIONS = sc.COMPILE_OPTIONS
THREADS = sc.THREADS  # the static shared arrays and the finalize tree assume this block size
FINAL_THREADS = 256
STEP_ROWS = 7  # before, after, added, surface return, drainage, evaporation, unmet
PACKET_WORDS = 9  # uint64: [0] OR of cell flag words, [1] flagged cells, [2..8] bit patterns of the 7 totals
PACKET_BYTES = PACKET_WORDS * 8
STEP_HOST_PACKET_BYTES = PACKET_BYTES + ss.PACKET_LENGTH * 8  # host peak payload of a step: the raw uint64 packet read back + the float packet rebuilt for `_resolve` (both live in `run_step`)
UNIFORM_DIAG_WORDS = 4  # uint64: [0] n_layers - first offending layer (0 = none), [1] offending lanes, [2..3] unused
STORAGE_WORDS = 1
MAX_GRID = 2**31 - 1
STEP_KERNEL = "maple_syrup_spatial_step"
STEP_FINAL = "maple_syrup_spatial_step_finalize"
STORAGE_KERNEL = "maple_syrup_spatial_storage"
STORAGE_FINAL = "maple_syrup_spatial_storage_finalize"
UNIFORM_KERNEL = "maple_syrup_spatial_uniform"
MOISTURE_KERNEL = "maple_syrup_spatial_moisture"
_ALL_KERNELS = (sc.FACTOR_KERNEL_NAME, STEP_KERNEL, STEP_FINAL, STORAGE_KERNEL, STORAGE_FINAL, UNIFORM_KERNEL, MOISTURE_KERNEL)

# Reader macros of the spatial variant (see the module docstring). `i` is the lane's cell index (a parameter of `spatial_column`).
SPATIAL_READERS = r"""
#define BLOCK 128
#define SP(p, sl, sc, l) (p)[(long long)(l) * (sl) + i * (sc)]
#define GUARD_INVALID_LANE if (i >= n) return;
#define DZ(l) SP(dz_p, dz_sl, dz_sc, l)
#define RES_M(l) (SP(tr_p, tr_sl, tr_sc, l) * DZ(l))
#define FC_M(l) (SP(tf_p, tf_sl, tf_sc, l) * DZ(l))
#define SAT_M(l) (SP(ts_p, ts_sl, ts_sc, l) * DZ(l))
#define CAP_TOL(l) (coef * SAT_M(l))
#define PAIR(k) SP(pv_p, pv_sl, pv_sc, k)
#define FAC(k) (hoisted != 0 ? factor[k] : relax_factor(SP(gd_p, g_sl, g_sc, k), PAIR(k), dt))
#define FAC_UP(k) (hoisted != 0 ? factor_up[k] : relax_factor(SP(gu_p, g_sl, g_sc, k), PAIR(k), dt))
#define RES_THETA0 SP(tr_p, tr_sl, tr_sc, 0)
#define STORE_ROWS(a_, before_, after_, added_, f_) \
    contrib[0] = a_ ? before_ : 0.0; \
    contrib[1] = a_ ? after_ : 0.0; \
    contrib[2] = a_ ? added_ : 0.0; \
    contrib[3] = ret_v; \
    contrib[4] = drain_v; \
    contrib[5] = evap_v; \
    contrib[6] = unmet_v; \
    *flags_out = f_;

"""

_MAP_PARAMS = r"""
    const double* dz_p, const long long dz_sl, const long long dz_sc,
    const double* tr_p, const long long tr_sl, const long long tr_sc,
    const double* tf_p, const long long tf_sl, const long long tf_sc,
    const double* ts_p, const long long ts_sl, const long long ts_sc,
    const double* pv_p, const long long pv_sl, const long long pv_sc,
    const double* gd_p, const double* gu_p, const long long g_sl, const long long g_sc,
    const double* factor, const double* factor_up, const int hoisted, const double coef,
    const double dt, const double stress, const double fct, const int gravity, const double cell_tol,
    const int bottom, const double* share, const int has_share, const double pulse, const int pulse_on,
    double* w_out, double* perc, double* diff, double* ret, double* drain, double* evap, double* unmet"""

SPATIAL_HEAD = ('__device__ __forceinline__ void spatial_column(\n    const long long i, const long long n, const int nl,\n'
                '    const double* w_in, const double* add, const double* pet, const unsigned char* active,' + _MAP_PARAMS + ',\n    double* contrib, unsigned int* flags_out)\n{\n')

_ARG_NAMES = ("dz_p, dz_sl, dz_sc, tr_p, tr_sl, tr_sc, tf_p, tf_sl, tf_sc, ts_p, ts_sl, ts_sc, pv_p, pv_sl, pv_sc, gd_p, gu_p, g_sl, g_sc, factor, factor_up, hoisted, coef, dt, stress, fct, gravity, "
              "cell_tol, bottom, share, has_share, pulse, pulse_on, w_out, perc, diff, ret, drain, evap, unmet")

SPATIAL_TAIL = ('''
extern "C" __global__ void maple_syrup_spatial_step(
    const long long n, const int nl,
    const double* w_in, const double* add, const double* pet, const unsigned char* active,''' + _MAP_PARAMS + r''',
    double* partials, unsigned long long* diag)
{
    __shared__ double sh[7][BLOCK];
    const int tid = threadIdx.x;
    const long long i = (long long)blockIdx.x * (long long)blockDim.x + (long long)tid;
    double contrib[7];
    for (int r = 0; r < 7; ++r) contrib[r] = 0.0;  /* a padded lane or an inactive cell contributes 0 */
    unsigned int f = 0u;
    spatial_column(i, n, nl, w_in, add, pet, active, ''' + _ARG_NAMES + r''', contrib, &f);  /* returns at once for i >= n */
    if (i < n && f != 0u) {
        atomicOr(&diag[0], (unsigned long long)f);
        atomicAdd(&diag[1], 1ULL);
    }
    for (int r = 0; r < 7; ++r) sh[r][tid] = contrib[r];
    __syncthreads();  /* every lane, padded or not, reaches every barrier */
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) { for (int r = 0; r < 7; ++r) sh[r][tid] = sh[r][tid] + sh[r][tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { for (int r = 0; r < 7; ++r) partials[(long long)r * (long long)gridDim.x + (long long)blockIdx.x] = sh[r][0]; }
}

__device__ __forceinline__ double rd(const double* p, const long long sl, const long long sc, const long long l, const long long i)
{
    return p[l * sl + i * sc];
}

extern "C" __global__ void maple_syrup_spatial_storage(const long long n, const int nl, const double* w, const unsigned char* active, double* partials)
{
    __shared__ double sh[BLOCK];
    const int tid = threadIdx.x;
    const long long i = (long long)blockIdx.x * (long long)blockDim.x + (long long)tid;
    double tot = 0.0;
    if (i < n) {
        for (int l = 0; l < nl; ++l) tot = tot + w[(long long)l * n + i];
        if (active[i] == 0) tot = 0.0;  /* inactive cells are zeroed, never clipped or counted */
    }
    sh[tid] = tot;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] = sh[tid] + sh[tid + s];
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x] = sh[0];
}

extern "C" __global__ void maple_syrup_spatial_moisture(
    const long long n, const int nl, const double* w, const double* dz_p, const long long dz_sl, const long long dz_sc, double* out)
{
    const long long i = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (i >= n) return;
    for (int l = 0; l < nl; ++l) out[(long long)l * n + i] = w[(long long)l * n + i] / rd(dz_p, dz_sl, dz_sc, l, i);
}

extern "C" __global__ void maple_syrup_spatial_uniform(
    const long long n, const int nl, const double* theta, const double* dz_p, const long long dz_sl, const long long dz_sc,
    const double* ts_p, const long long ts_sl, const long long ts_sc, double* water, unsigned long long* diag)
{
    const long long i = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (i >= n) return;
    for (int l = 0; l < nl; ++l) {
        const double t = theta[l];
        if (t < 0.0 || t > rd(ts_p, ts_sl, ts_sc, l, i)) {  /* every cell, inactive placeholders included */
            atomicMax(&diag[0], (unsigned long long)(nl - l));
            atomicAdd(&diag[1], 1ULL);
        }
        water[(long long)l * n + i] = t * rd(dz_p, dz_sl, dz_sc, l, i);
    }
}
''')

_FINALIZE = r"""
extern "C" __global__ void @NAME@(const double* partials, const long long n_blocks, unsigned long long* diag)
{
    __shared__ double sh[@NROWS@][@FINAL@];
    const int tid = threadIdx.x;
    for (int r = 0; r < @NROWS@; ++r) {
        double acc = 0.0;
        for (long long b = tid; b < n_blocks; b += @FINAL@) acc = acc + partials[(long long)r * n_blocks + b];
        sh[r][tid] = acc;
    }
    __syncthreads();
    for (int s = @FINAL@ / 2; s > 0; s >>= 1) {
        if (tid < s) { for (int r = 0; r < @NROWS@; ++r) sh[r][tid] = sh[r][tid] + sh[r][tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { for (int r = 0; r < @NROWS@; ++r) diag[@OFFSET@ + r] = (unsigned long long)__double_as_longlong(sh[r][0]); }
}
"""


def _finalize(name: str, nrows: int, offset: int) -> str:
    return _FINALIZE.replace("@NAME@", name).replace("@NROWS@", str(nrows)).replace("@FINAL@", str(FINAL_THREADS)).replace("@OFFSET@", str(offset))


def render_source() -> str:
    """The spatial CUDA source: the shared body rendered with the spatial readers/head and this module's wrapper/utility kernels (pure, stateless)."""
    return sc.render_body_source(SPATIAL_READERS, SPATIAL_HEAD, SPATIAL_TAIL + _finalize(STEP_FINAL, STEP_ROWS, 2) + _finalize(STORAGE_FINAL, 1, 0))


def kernel_source() -> str:
    return render_source()


def kernel_provenance() -> dict[str, Any]:
    src = render_source()
    return {"module": "maple_syrup.subsurface_cuda_spatial", "kernels": list(_ALL_KERNELS), "source_sha256": hashlib.sha256(src.encode()).hexdigest(),
            "shared_body_sha256": hashlib.sha256(sc._BODY.encode()).hexdigest(), "spatial_readers_sha256": hashlib.sha256(SPATIAL_READERS.encode()).hexdigest(),
            "compile_options": list(COMPILE_OPTIONS), "threads_per_block": THREADS, "final_threads": FINAL_THREADS, "packet_bytes": PACKET_BYTES,
            "reductions": "explicit fused block tree (static shared memory) + one-block finalize; integer atomics for the flag OR and the flagged count; no CUB/CuPy reduction, no scalar array",
            "fastmath": False, "libm": "device expm1 (may differ from NumPy by an ulp); declared bound rtol 2e-12 / atol 1e-14 (bitwise where the diffusivity is 0)"}


_LOCK = threading.Lock()
_FUNCTIONS: dict[int, SimpleNamespace] = {}  # device id -> compiled functions only (never an owner, state or array)


def load_kernels(cp, device_id: int) -> SimpleNamespace:
    """Compile/load the spatial module once per process and device. Failure is a typed setup error, never a fallback."""
    with _LOCK:
        fns = _FUNCTIONS.get(device_id)
        if fns is not None:
            return fns
        try:
            module = cp.RawModule(code=render_source(), options=COMPILE_OPTIONS, backend="nvrtc")
            got = {name: module.get_function(name) for name in _ALL_KERNELS}
            for fn in got.values():
                if int(dict(fn.attributes).get("max_threads_per_block", 0)) < THREADS:
                    raise ss.SubsurfaceBackendUnavailableError(f"a spatial kernel supports fewer than {THREADS} threads per block")
        except ss.SubsurfaceBackendUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ss.SubsurfaceBackendUnavailableError(f"the CUDA spatial kernels failed to compile/load ({type(exc).__name__}: {exc}); no fallback") from exc
        ns = SimpleNamespace(factor=got[sc.FACTOR_KERNEL_NAME], step=got[STEP_KERNEL], step_final=got[STEP_FINAL], storage=got[STORAGE_KERNEL], storage_final=got[STORAGE_FINAL],
                             uniform=got[UNIFORM_KERNEL], moisture=got[MOISTURE_KERNEL])
        _FUNCTIONS[device_id] = ns
        return ns


# ---- memory --------------------------------------------------------------------------------------------------------------------------------
def n_blocks(n_cells: int) -> int:
    return (int(n_cells) + THREADS - 1) // THREADS


def check_launch_ranges(n_layers: int, n_cells: int) -> None:
    """Integer launch widths are validated BEFORE any cast or allocation: `nl` is a 32-bit kernel argument, the grid is limited to 2^31 - 1 blocks."""
    if not (1 <= n_layers <= 2**31 - 1):
        raise ss.SubsurfaceError(f"n_layers = {n_layers} does not fit the 32-bit kernel argument")
    if not (1 <= n_cells and n_blocks(n_cells) <= MAX_GRID):
        raise ss.SubsurfaceError(f"n_cells = {n_cells} exceeds the CUDA launch grid limit of {MAX_GRID} blocks of {THREADS} threads")


def factor_arrays(n_layers: int, hoisted: bool, directional: bool) -> int:
    """Per-step device factor arrays: 0 when the factors are evaluated per cell, otherwise one (constant) or two (donor-depth) `(max(L-1, 1),)` arrays."""
    return 0 if not hoisted or n_layers < 2 else (2 if directional else 1)


def operation_terms(operation: str, n_layers: int, n_cells: int, *, factors: int = 0) -> dict[str, int]:
    """Exact byte terms of one CUDA operation (payload model: the call's own arrays; allocator rounding, NVRTC, the context and the interpreter are excluded).

    Device (main quota): `outputs` survive the call, `scratch` is released after the synchronising read; `host` is the call's own host allocations without the Python allowance.
      state      device 8LN (device copy); host 0
      uniform    device 8LN + 8L (theta) + 8*UNIFORM_DIAG_WORDS; host 24L (the `_layer_array` conversion lifetimes, the owned theta being the transfer source) + 8*UNIFORM_DIAG_WORDS (diagnostic read)
      moisture   device 8LN; host 0
      storage    device 8*nb (partials) + 8*STORAGE_WORDS; host 8*STORAGE_WORDS (the one-word packet read)
      step       device outputs (LN + 2(L-1)N + 4N)*8 + scratch 7*8*nb + PACKET_BYTES + factors*max(L-1,1)*8; host STEP_HOST_PACKET_BYTES = 144: the raw `PACKET_BYTES` (72 B) array returned by
                 the counted read AND the reconstructed nine-value float packet (72 B) are live together in `run_step` (wire traffic is still ONE 72-byte read)
    `shared` is the static shared memory per block (not a global allocation)."""
    L, N = int(n_layers), int(n_cells)
    nb, f8 = n_blocks(N), 8
    if operation == "state":
        t = dict(outputs=f8 * L * N, scratch=0, host=0, shared=0)
    elif operation == "uniform_state":
        t = dict(outputs=f8 * L * N, scratch=f8 * L + f8 * UNIFORM_DIAG_WORDS, host=3 * f8 * L + f8 * UNIFORM_DIAG_WORDS, shared=0)
    elif operation == "moisture":
        t = dict(outputs=f8 * L * N, scratch=0, host=0, shared=0)
    elif operation == "storage":
        t = dict(outputs=0, scratch=f8 * nb + f8 * STORAGE_WORDS, host=f8 * STORAGE_WORDS, shared=f8 * THREADS)
    elif operation in ("step", "step_with_recharge"):
        t = dict(outputs=(L * N + 2 * (L - 1) * N + 4 * N) * f8, scratch=STEP_ROWS * f8 * nb + PACKET_BYTES + factors * max(L - 1, 1) * f8, host=STEP_HOST_PACKET_BYTES, shared=STEP_ROWS * f8 * THREADS)
    else:
        raise ss.SubsurfaceError(f"unknown CUDA operation {operation!r}")
    t["n_blocks"] = nb
    t["factor_arrays"] = factors if operation in ("step", "step_with_recharge") else 0
    return t


# ---- launches ------------------------------------------------------------------------------------------------------------------------------
_KIND_STRIDES = {"scalar": lambda n: (0, 0), "shared": lambda n: (1, 0), "cell": lambda n: (n, 1)}


def _strides(kind: str, n_cells: int) -> tuple[Any, Any]:
    sl, sc_ = _KIND_STRIDES[kind](int(n_cells))
    return np.int64(sl), np.int64(sc_)


def _map_args(column, maps: dict, kinds: dict) -> dict:
    n = column.n_cells
    dz = maps["dz"]
    out = {"dz": (dz, *_strides(kinds["dz"], n)), "tr": (maps["theta_residual"], *_strides(kinds["theta_residual"], n)), "tf": (maps["theta_field_capacity"], *_strides(kinds["theta_field_capacity"], n)),
           "ts": (maps["theta_saturation"], *_strides(kinds["theta_saturation"], n))}
    if column.n_layers > 1:
        gd = maps["interface_conductance_down"]
        out["pv"] = (maps["pair_inverse_thickness"], *_strides(kinds["pair_inverse_thickness"], n))
        out["gd"], out["gu"] = gd, maps.get("interface_conductance_up", gd)  # constant mode: the upward conductance IS the downward one
        out["g_strides"] = _strides(kinds["interface_conductance_down"], n)
    else:  # no interface maps exist: dummy never-read pointers
        out["pv"] = (dz, np.int64(0), np.int64(0))
        out["gd"] = out["gu"] = dz
        out["g_strides"] = (np.int64(0), np.int64(0))
    return out


def run_step(cp, column, water_in, added, pet, share, dt: float, pulse: float | None):
    """One spatial CUDA step (admission, the device/current-device/type checks and the time checks were done by the caller). Returns `(StepArrays with rows=None, nine-value float packet)`.
    Fresh outputs; the state, forcing, retained maps and mask are only read."""
    from maple.core.backend import to_host

    L, N = int(column.n_layers), int(column.n_cells)
    check_launch_ranges(L, N)
    kinds = {n: v["kind"] for n, v in column.estimate.retained.items()}
    maps = column._maps
    n_if = L - 1
    directional = column.diffusion_mode != "constant"
    hoisted = L > 1 and kinds["interface_conductance_down"] == "shared"
    fns = load_kernels(cp, column.device_id)
    from maple_syrup.conservation import volume_roundoff_bound_m3

    coef = volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0)
    pulse_on = 0 if pulse is None else 1
    cell_tol = ss.cell_closure_coefficient(L, pulse_on == 1)
    a = _map_args(column, maps, kinds)
    water_out = cp.empty((L, N), dtype=np.float64)
    percolation = cp.empty((n_if, N), dtype=np.float64)
    diffusion = cp.empty((n_if, N), dtype=np.float64)
    ret, drain, evap, unmet = (cp.empty(N, dtype=np.float64) for _ in range(4))
    nb = n_blocks(N)
    partials = cp.empty(STEP_ROWS * nb, dtype=np.float64)
    diag = cp.zeros(PACKET_WORDS, dtype=np.uint64)
    if hoisted:
        factor = cp.empty(max(n_if, 1), dtype=np.float64)
        factor_up = cp.empty(max(n_if, 1), dtype=np.float64) if directional else factor
        fns.factor(((n_if + THREADS - 1) // THREADS,), (THREADS,), (np.int32(n_if), a["gd"], a["gu"], a["pv"][0], np.float64(dt), np.int32(1 if directional else 0), factor, factor_up))
    else:
        factor = factor_up = pet  # never read (hoisted == 0)
    share_arr, has_share = (pet, 0) if share is None else (share, 1)
    fns.step((nb,), (THREADS,), (
        np.int64(N), np.int32(L), water_in, water_in if pulse is not None else added, pet, column._mask,
        *a["dz"], *a["tr"], *a["tf"], *a["ts"], *a["pv"], a["gd"], a["gu"], *a["g_strides"],
        factor, factor_up, np.int32(1 if hoisted else 0), np.float64(coef),
        np.float64(dt), np.float64(column.theta_evaporation_stress), np.float64(column.evaporation_factor), np.int32(1 if column.gravity_drainage else 0), np.float64(cell_tol),
        np.int32(0 if column.bottom_drainage == "closed" else 1), share_arr, np.int32(has_share), np.float64(0.0 if pulse is None else pulse), np.int32(pulse_on),
        water_out, percolation, diffusion, ret, drain, evap, unmet, partials, diag))
    fns.step_final((1,), (FINAL_THREADS,), (partials, np.int64(nb), diag))
    words = np.asarray(to_host(diag), dtype=np.uint64)  # the ONE counted PACKET_BYTES read; synchronises the stream: partials/factors are released only afterwards
    orf, count = int(words[0]), int(words[1])
    packet = np.empty(ss.PACKET_LENGTH, dtype=np.float64)
    packet[:7] = words[2:9].view(np.float64)
    packet[7] = float(orf & -orf) if orf else ss.NO_FLAG  # the lowest set bit of the OR == the minimum over cells of each cell's lowest bit
    packet[8] = float(count)
    return ss.StepArrays(water_out, percolation, diffusion, ret, drain, evap, unmet, None), packet


def uniform_water(cp, column, theta_host: np.ndarray):
    """`(L, N)` water = theta[k] * dz[k, c] on the device from the validated host L-vector (ONE counted host-to-device copy of 8L bytes). Returns `(water, offending_layer or None)`; the
    caller discards the water when a layer is flagged (the threshold is checked in EVERY cell, inactive placeholders included)."""
    from maple.core.backend import to_device, to_host

    L, N = int(column.n_layers), int(column.n_cells)
    check_launch_ranges(L, N)
    kinds = {n: v["kind"] for n, v in column.estimate.retained.items()}
    fns = load_kernels(cp, column.device_id)
    theta_dev = to_device(np.ascontiguousarray(theta_host, dtype=np.float64), cp)
    water = cp.empty((L, N), dtype=np.float64)
    diag = cp.zeros(UNIFORM_DIAG_WORDS, dtype=np.uint64)
    fns.uniform((n_blocks(N),), (THREADS,), (np.int64(N), np.int32(L), theta_dev, column._maps["dz"], *_strides(kinds["dz"], N), column._maps["theta_saturation"],
                                             *_strides(kinds["theta_saturation"], N), water, diag))
    words = np.asarray(to_host(diag), dtype=np.uint64)
    code = int(words[0])
    return water, (None if code == 0 else L - code)


def moisture(cp, column, water):
    L, N = int(column.n_layers), int(column.n_cells)
    check_launch_ranges(L, N)
    kinds = {n: v["kind"] for n, v in column.estimate.retained.items()}
    fns = load_kernels(cp, column.device_id)
    out = cp.empty((L, N), dtype=np.float64)
    fns.moisture((n_blocks(N),), (THREADS,), (np.int64(N), np.int32(L), water, column._maps["dz"], *_strides(kinds["dz"], N), out))
    return out


def storage_depth(cp, column, water) -> float:
    """Sum over ACTIVE cells of the layer-sequential per-cell water depth (m): block tree + one-block finalize + one 8-byte read. Not clipped, not rounded."""
    from maple.core.backend import to_host

    L, N = int(column.n_layers), int(column.n_cells)
    check_launch_ranges(L, N)
    fns = load_kernels(cp, column.device_id)
    nb = n_blocks(N)
    partials = cp.empty(nb, dtype=np.float64)
    diag = cp.zeros(STORAGE_WORDS, dtype=np.uint64)
    fns.storage((nb,), (THREADS,), (np.int64(N), np.int32(L), water, column._mask, partials))
    fns.storage_final((1,), (FINAL_THREADS,), (partials, np.int64(nb), diag))
    words = np.asarray(to_host(diag), dtype=np.uint64)
    return float(words[:1].view(np.float64)[0])
