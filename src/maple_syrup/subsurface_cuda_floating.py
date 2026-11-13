"""CUDA SOURCE RENDERER of the FLOATING porous column (`maple_syrup.floating_soil_layers`). AUTHOR-WRITTEN, NOT RUN ON A DEVICE. There is NO launcher, owner, device validation or reduction packet yet (the "cuda" backend of
`floating_soil_layers` is refused with a typed error): next bounded slice D1b. This module delivers ONLY a pure source renderer, verified by host-compiler execution (not hardware evidence).

Engineering choice (documented): NO physical stage text is copied. The step source is `subsurface_cuda.render_body_source(FLOATING_READERS, FLOATING_STEP_HEAD)`: the ONE shared body `subsurface_cuda._BODY`, untouched, read through
FLOATING reader macros exactly as the spatial variant reads through its own. `subsurface_cuda.py`, `subsurface_cuda_spatial.py` and the emitted COMMON source/ABI/provenance are not edited and stay byte exact.

* Geometry in registers, no per-thread arrays and no layer cap: `HP(l) = max(0, min(D_i, e_{l+1}) - e_l)` from the O(L) edges and the O(N) soil depth, evaluated where used. `DZ(l)` is `HP(l)` or the finite stand-in 1.0 where `HP(l) = 0`
  (arithmetic only: never returned as a thickness or capacity; a zero slot has `RES_M = FC_M = SAT_M = theta * 0 = 0` and water exactly 0). `PAIR(k) = 1/DZ(k) + 1/DZ(k+1)`.
* Zero-porosity interfaces: `FAC(k)`/`FAC_UP(k)` come from `fl_relax`, which returns the factor 0.0 BEFORE any division when either layer has `HP = 0`, so the shared body computes `amount = g * 0 = 0` and exchanges nothing (the NumPy reference reaches the
  same exact 0 through its donor/receiver limits). Otherwise `sep = sep_nominal[k]` for a fully-soil pair (`D >= e_{k+2}`; the historical centre difference, bit-identical to the spatial owner) else `(h_k + h_{k+1})/2`; `G_down = ((d K) h_k)/sep`,
  `G_up = ((d K) h_{k+1})/sep` (donor-depth law with the actual donor depth) or `K/sep` (constant), factor `relax_factor(G, 1/h_k + 1/h_{k+1}, dt)`.
* Per-cell lower boundary: `GUARD_INVALID_LANE` also declares `const int bottom = bopen[i] != 0` (configured open AND `D > H`), which the shared gravity stage reads. Inactive placeholders: `SAT_M(l)` is `DBL_MAX` for an inactive cell so the body's
  input capacity flag never fires there (published inactive policy); the physics stages never run on inactive cells.
* Staged reconciliation (`maple_syrup_floating_reconcile`) is assembled from FAIL-CLOSED anchored slices of the shared body: the gravity-cascade block and the saturation-return loop (with `pulse_on = 1`, i.e. zero added water: the return statements
  still run) are cut out of `_BODY` by exact anchors (each must occur exactly once; otherwise rendering raises) and emitted in the order gravity THEN saturation return; no copy of the cascade equations is maintained here. `provenance()` hashes the
  shared body, each slice, the readers and the rendered source.

Rows written like the common step kernel (`rows[0..3]`: storage before/after, added, lowest failure bit). Importing this module never imports CuPy or touches a device. Compiled with the intended `--fmad=false --prec-div=true --prec-sqrt=true --ftz=false`.
"""

from __future__ import annotations

import hashlib

from maple_syrup import subsurface_cuda as sc

__all__ = ["CHECK_WORDS", "COMPILE_OPTIONS", "FINAL_THREADS", "FLOATING_READERS", "FLOATING_STEP_HEAD", "RECONCILE_KERNEL", "RECONCILE_PACKET_WORDS", "RESIDENT_KERNELS", "RESIDENT_READERS", "STEP_KERNEL",
           "STEP_PACKET_WORDS", "THREADS", "provenance", "render_reconcile_source", "render_resident_source", "render_source", "render_step_source", "resident_provenance"]

COMPILE_OPTIONS = sc.COMPILE_OPTIONS
STEP_KERNEL = "maple_syrup_floating_step"
RECONCILE_KERNEL = "maple_syrup_floating_reconcile"

_GEOMETRY_READERS = r"""
#define SP(p, sl, sc_, l) (p)[(long long)(l) * (sl) + i * (sc_)]
#define HP(l) fmax(0.0, fmin(depth_p[i], edges_p[(l) + 1]) - edges_p[(l)])
#define GUARD_INVALID_LANE if (i >= n) return; const int bottom = (bopen[i] != 0) ? 1 : 0;
#define DZ(l) (HP(l) > 0.0 ? HP(l) : 1.0)
#define RES_M(l) (SP(tr_p, tr_sl, tr_sc, l) * HP(l))
#define FC_M(l) (SP(tf_p, tf_sl, tf_sc, l) * HP(l))
#define SAT_M(l) (a ? (SP(ts_p, ts_sl, ts_sc, l) * HP(l)) : 1.7976931348623157e308)
#define CAP_TOL(l) (coef * SAT_M(l))
#define PAIR(k) (1.0 / DZ(k) + 1.0 / DZ((k) + 1))
#define FAC(k) fl_relax(i, (k), 0, edges_p, depth_p, sepn, kif_p, kif_sl, kif_sc, dcoef, directional, dt)
#define FAC_UP(k) fl_relax(i, (k), 1, edges_p, depth_p, sepn, kif_p, kif_sl, kif_sc, dcoef, directional, dt)
#define RES_THETA0 SP(tr_p, tr_sl, tr_sc, 0)
// Exchange factor of interface k for cell i (up != 0: the upward-donor factor). 0.0 BEFORE any division when either layer has no porous thickness: no exchange across a zero slot.
__device__ __forceinline__ double fl_relax(const long long i, const int k, const int up, const double* edges_p, const double* depth_p, const double* sepn, const double* kif_p, const long long kif_sl,
                                           const long long kif_sc, const double dcoef, const int directional, const double dt)
{
    const double d = depth_p[i];
    const double hk = fmax(0.0, fmin(d, edges_p[k + 1]) - edges_p[k]);
    const double hk1 = fmax(0.0, fmin(d, edges_p[k + 2]) - edges_p[k + 1]);
    if (!(hk > 0.0 && hk1 > 0.0)) return 0.0;
    double sp = (d >= edges_p[k + 2]) ? sepn[k] : (hk + hk1) / 2.0;
    if (!(sp > 0.0)) sp = 1.0;
    const double v = kif_p[(long long)k * kif_sl + i * kif_sc];
    double g;
    if (directional != 0) g = (up != 0) ? (((dcoef * v) * hk1) / sp) : (((dcoef * v) * hk) / sp);
    else g = v / sp;
    return relax_factor(g, 1.0 / hk + 1.0 / hk1, dt);
}

"""


_ROWS_EPILOGUE = r"""
#define STORE_ROWS(a_, before_, after_, added_, f_) \
    rows[i] = a_ ? before_ : 0.0; \
    rows[n + i] = a_ ? after_ : 0.0; \
    rows[2 * n + i] = a_ ? added_ : 0.0; \
    rows[3 * n + i] = (f_ == 0u) ? no_flag : (double)(f_ & (0u - f_));
"""

# The resident (production) variant stores the seven per-lane contributions and the cell flag word instead of four full-width rows (the spatial S2 pattern).
_CONTRIB_EPILOGUE = r"""
#define BLOCK 128
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

FLOATING_READERS = _GEOMETRY_READERS + _ROWS_EPILOGUE  # host-test/render variant: four full rows
RESIDENT_READERS = _GEOMETRY_READERS + _CONTRIB_EPILOGUE  # production variant: fixed 72-byte reduction packet

_MAP_PARAMS = r"""
    const double* edges_p, const double* depth_p, const double* sepn,
    const double* tr_p, const long long tr_sl, const long long tr_sc,
    const double* tf_p, const long long tf_sl, const long long tf_sc,
    const double* ts_p, const long long ts_sl, const long long ts_sc,"""

FLOATING_STEP_HEAD = (r"""extern "C" __global__ void maple_syrup_floating_step(
    const long long n, const int nl,
    const double* w_in, const double* add, const double* pet, const unsigned char* active, const unsigned char* bopen,""" + _MAP_PARAMS + r"""
    const double* kif_p, const long long kif_sl, const long long kif_sc, const double dcoef, const int directional,
    const double coef, const double dt, const double stress, const double fct, const int gravity, const double cell_tol, const double no_flag,
    const double* share, const int has_share, const double pulse, const int pulse_on,
    double* w_out, double* perc, double* diff, double* ret, double* drain, double* evap, double* unmet, double* rows)
{
    const long long i = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
""")

# fail-closed anchors into the shared body (each must occur exactly once)
_GRAV_BEGIN = "        // gravity cascade, top to bottom within the step\n        if (gravity != 0) {\n"
_GRAV_END = "            drain_v = carry;  // exactly 0 with a closed bottom\n        }\n"
_SAT_BEGIN = "        // addition with saturation return\n        for (int l = 0; l < nl; ++l) {\n"
_SAT_END = "            w_out[p] = w;\n        }\n"


class FloatingSourceError(ValueError):
    """The shared body no longer matches the renderer's anchors (fail closed: nothing is rendered)."""


def _slice(body: str, begin: str, end: str, what: str) -> str:
    for marker in (begin, end):
        if body.count(marker) != 1:
            raise FloatingSourceError(f"the shared CUDA body does not contain the {what} anchor exactly once ({marker[:50]!r}: {body.count(marker)})")
    a = body.index(begin)
    b = body.index(end, a) + len(end)
    return body[a:b]


def _slices() -> tuple[str, str]:
    body = sc._BODY
    return _slice(body, _GRAV_BEGIN, _GRAV_END, "gravity-stage"), _slice(body, _SAT_BEGIN, _SAT_END, "saturation-return")


def render_step_source() -> str:
    """The floating STEP source: the shared body through the floating reader macros (pure rendering; `sc.render_body_source` checks every body macro is defined)."""
    return sc.render_body_source(FLOATING_READERS, FLOATING_STEP_HEAD)


def render_reconcile_source() -> str:
    """The staged reconciliation kernel: the shared gravity-stage slice, THEN the shared saturation-return slice (zero added water), from the one body."""
    grav, sat = _slices()
    return (r"""
extern "C" __global__ void maple_syrup_floating_reconcile(
    const long long n, const int nl, const double* w_in, const unsigned char* active, const unsigned char* bopen,""" + _MAP_PARAMS + r"""
    const double coef, double* w_out, double* perc, double* surface, double* export_m)
{
    const long long i = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (i >= n) return;
    const int n_if = nl - 1;
    const bool a = active[i] != 0;
    const int bottom = (bopen[i] != 0) ? 1 : 0;
    const int gravity = 1;
    const int pulse_on = 1;           /* zero added water: the saturation-return statements still run */
    const double* add = w_in;         /* never read (pulse_on != 0) */
    for (int l = 0; l < nl; ++l) w_out[(long long)l * n + i] = w_in[(long long)l * n + i];
    for (int k = 0; k < n_if; ++k) perc[(long long)k * n + i] = 0.0;
    surface[i] = 0.0;
    export_m[i] = 0.0;
    if (!a) return;
    double ret_v = 0.0, drain_v = 0.0;
    {
""" + grav + r"""
""" + sat + r"""
    }
    surface[i] = ret_v;
    export_m[i] = drain_v;
}
""")


def render_source() -> str:
    """The complete floating source: step kernel (shared body) + reconciliation kernel (shared slices)."""
    return render_step_source() + render_reconcile_source()


def provenance() -> dict:
    grav, sat = _slices()
    h = lambda t: hashlib.sha256(t.encode()).hexdigest()  # noqa: E731
    return {"module": "maple_syrup.subsurface_cuda_floating", "kernels": [STEP_KERNEL, RECONCILE_KERNEL], "shared_body_sha256": h(sc._BODY), "gravity_slice_sha256": h(grav), "saturation_slice_sha256": h(sat),
            "readers_sha256": h(FLOATING_READERS), "source_sha256": h(render_source()), "compile_options": list(COMPILE_OPTIONS), "status": "source renderer only: no launcher/owner/device validation (D1b)"}


# =====================================================================================================================================
# D1b: the RESIDENT (production) source: the SAME shared body through the SAME geometry readers, an all-lane wrapper with the fixed block reduction (7 contributions -> 72-byte packet), staged reconciliation with a 4-total
# packet, resident validation kernels and utilities. No second copy of any stage equation (the reconcile uses the anchored slices); utilities contain no physics.
# =====================================================================================================================================
THREADS = sc.THREADS
FINAL_THREADS = 256
STEP_ROWS = 7
STEP_PACKET_WORDS = 9  # [0] OR of cell flag words, [1] flagged cells, [2..8] bit patterns of the 7 totals
RECONCILE_ROWS = 5  # before, after, surface return, bottom export, moved percolation (sum of the interface forwards)
RECONCILE_PACKET_WORDS = 7  # [0] OR of flags, [1] flagged cells, [2..6] the five totals
CHECK_WORDS = 4  # prepare / state-check diagnostic: [0] OR of flags, [1] flagged cells, [2] min(cell << 8 | flags), [3] unused
RES_STEP = "maple_syrup_floating_resident_step"
RES_STEP_FINAL = "maple_syrup_floating_resident_step_finalize"
RES_RECONCILE = "maple_syrup_floating_resident_reconcile"
RES_RECONCILE_FINAL = "maple_syrup_floating_resident_reconcile_finalize"
PREPARE_KERNEL = "maple_syrup_floating_prepare"
STATE_CHECK_KERNEL = "maple_syrup_floating_state_check"
MOISTURE_KERNEL = "maple_syrup_floating_moisture"
STORAGE_KERNEL = "maple_syrup_floating_storage"
STORAGE_FINAL = "maple_syrup_floating_storage_finalize"
RESIDENT_KERNELS = (RES_STEP, RES_STEP_FINAL, RES_RECONCILE, RES_RECONCILE_FINAL, PREPARE_KERNEL, STATE_CHECK_KERNEL, MOISTURE_KERNEL, STORAGE_KERNEL, STORAGE_FINAL)

_KIF_PARAMS = "const double* kif_p, const long long kif_sl, const long long kif_sc, const double dcoef, const int directional,"
_STEP_PARAMS = (r"""
    const double* w_in, const double* add, const double* pet, const unsigned char* active, const unsigned char* bopen,""" + _MAP_PARAMS + "\n    " + _KIF_PARAMS + r"""
    const double coef, const double dt, const double stress, const double fct, const int gravity, const double cell_tol,
    const double* share, const int has_share, const double pulse, const int pulse_on,
    double* w_out, double* perc, double* diff, double* ret, double* drain, double* evap, double* unmet""")
_STEP_ARGS = ("w_in, add, pet, active, bopen, edges_p, depth_p, sepn, tr_p, tr_sl, tr_sc, tf_p, tf_sl, tf_sc, ts_p, ts_sl, ts_sc, kif_p, kif_sl, kif_sc, dcoef, directional, coef, dt, stress, fct, gravity, "
              "cell_tol, share, has_share, pulse, pulse_on, w_out, perc, diff, ret, drain, evap, unmet")

RESIDENT_STEP_HEAD = ("__device__ __forceinline__ void floating_column(\n    const long long i, const long long n, const int nl," + _STEP_PARAMS + ",\n    double* contrib, unsigned int* flags_out)\n{\n")

_TREE = r"""
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) { for (int r = 0; r < @ROWS@; ++r) sh[r][tid] = sh[r][tid] + sh[r][tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { for (int r = 0; r < @ROWS@; ++r) partials[(long long)r * (long long)gridDim.x + (long long)blockIdx.x] = sh[r][0]; }
"""

RESIDENT_STEP_TAIL = (r"""
extern "C" __global__ void maple_syrup_floating_resident_step(
    const long long n, const int nl,""" + _STEP_PARAMS + r""",
    double* partials, unsigned long long* diag)
{
    __shared__ double sh[7][BLOCK];
    const int tid = threadIdx.x;
    const long long i = (long long)blockIdx.x * (long long)blockDim.x + (long long)tid;
    double contrib[7];
    for (int r = 0; r < 7; ++r) contrib[r] = 0.0;  /* a padded lane or an inactive cell contributes 0 */
    unsigned int f = 0u;
    floating_column(i, n, nl, """ + _STEP_ARGS + r""", contrib, &f);  /* returns at once for i >= n */
    if (i < n && f != 0u) {
        atomicOr(&diag[0], (unsigned long long)f);
        atomicAdd(&diag[1], 1ULL);
    }
    for (int r = 0; r < 7; ++r) sh[r][tid] = contrib[r];
    __syncthreads();  /* every lane, padded or not, reaches every barrier */
""" + _TREE.replace("@ROWS@", "7") + "}\n")

_RECON_PARAMS = (r"""
    const double* w_in, const unsigned char* active, const unsigned char* bopen,""" + _MAP_PARAMS + r"""
    const double* odepth_p, const double* ots_p, const long long ots_sl, const long long ots_sc,
    const double coef, double* w_out, double* perc, double* surface, double* export_m""")
_RECON_ARGS = ("w_in, active, bopen, edges_p, depth_p, sepn, tr_p, tr_sl, tr_sc, tf_p, tf_sl, tf_sc, ts_p, ts_sl, ts_sc, odepth_p, ots_p, ots_sl, ots_sc, coef, w_out, perc, surface, export_m")


def _reconcile_device_function() -> str:
    grav, sat = _slices()
    return (r"""
__device__ __forceinline__ void floating_reconcile_column(
    const long long i, const long long n, const int nl,""" + _RECON_PARAMS + r""",
    double* contrib, unsigned int* flags_out)
{
    if (i >= n) return;
    const int n_if = nl - 1;
    const bool a = active[i] != 0;
    const int bottom = (bopen[i] != 0) ? 1 : 0;
    const int gravity = 1;
    const int pulse_on = 1;           /* zero added water: the saturation-return statements still run */
    const double* add = w_in;         /* never read (pulse_on != 0) */
    /* The OLD state must be valid for its OWN owner (finite, non-negative, within the OLD porous capacity, rock slots empty on active cells): exceeding the NEW capacity is legal only inside this stage.
       Evaluated in the same per-cell walk, from the OLD soil depth and OLD saturation map; bits 20/21 are the old-state refusal. */
    unsigned int fo0 = 0u;
    for (int l = 0; l < nl; ++l) {
        const double wl = w_in[(long long)l * n + i];
        if (!(isfinite(wl) && wl >= 0.0)) { fo0 |= 1u << 20; continue; }
        if (!a) continue;
        const double ho = fmax(0.0, fmin(odepth_p[i], edges_p[l + 1]) - edges_p[l]);
        const double sato = SP(ots_p, ots_sl, ots_sc, l) * ho;
        if (!(wl <= sato + coef * sato)) fo0 |= 1u << 21;
    }
    if (fo0 != 0u) { *flags_out = fo0; return; }   /* nothing moves; the host raises before any state is minted */
    for (int l = 0; l < nl; ++l) w_out[(long long)l * n + i] = w_in[(long long)l * n + i];
    for (int k = 0; k < n_if; ++k) perc[(long long)k * n + i] = 0.0;
    surface[i] = 0.0;
    export_m[i] = 0.0;
    if (!a) return;                   /* an inactive cell is copied through: zero contributions, no law applied */
    double ret_v = 0.0, drain_v = 0.0;
    {
""" + grav + r"""
""" + sat + r"""
    }
    surface[i] = ret_v;
    export_m[i] = drain_v;
    double before = 0.0, after = 0.0;
    unsigned int fo = 0u;
    for (int l = 0; l < nl; ++l) {
        before = before + w_in[(long long)l * n + i];
        const double w = w_out[(long long)l * n + i];
        if (!isfinite(w)) fo |= 1u << 10;
        if (w < 0.0) fo |= 1u << 11;
        if (w > SAT_M(l) + CAP_TOL(l)) fo |= 1u << 12;   /* the minted candidate must be a valid continued state */
        after = after + w;
    }
    double moved = 0.0;                /* actual moved percolation: the sum of the interface forwards of this cell (one more scalar in the SAME block reduction) */
    for (int k = 0; k < n_if; ++k) moved = moved + fabs(perc[(long long)k * n + i]);
    contrib[0] = before; contrib[1] = after; contrib[2] = ret_v; contrib[3] = drain_v; contrib[4] = moved;
    *flags_out = fo;
}

extern "C" __global__ void maple_syrup_floating_resident_reconcile(
    const long long n, const int nl,""" + _RECON_PARAMS + r""",
    double* partials, unsigned long long* diag)
{
    __shared__ double sh[5][BLOCK];
    const int tid = threadIdx.x;
    const long long i = (long long)blockIdx.x * (long long)blockDim.x + (long long)tid;
    double contrib[5];
    for (int r = 0; r < 5; ++r) contrib[r] = 0.0;
    unsigned int f = 0u;
    floating_reconcile_column(i, n, nl, """ + _RECON_ARGS + r""", contrib, &f);
    if (i < n && f != 0u) {
        atomicOr(&diag[0], (unsigned long long)f);
        atomicAdd(&diag[1], 1ULL);
    }
    for (int r = 0; r < 5; ++r) sh[r][tid] = contrib[r];
    __syncthreads();
""" + _TREE.replace("@ROWS@", "5") + "}\n")


_PREPARE_AND_UTILITIES = r"""
/* ---- resident validation of the RAW maps (cold, once per column; flags per cell, packet-only report) ---- */
#define FL_BAD_THETA   1u   /* a theta input is not finite                                           */
#define FL_BAD_REL     2u   /* 0 <= residual <= field capacity <= saturation <= 1, residual < saturation */
#define FL_BAD_STRESS  4u   /* theta_evaporation_stress must exceed the layer-0 residual              */
#define FL_BAD_IFACE   8u   /* interface input finite and >= 0                                        */
#define FL_BAD_DERIVED 16u  /* a porous slab/pair is unrepresentable (capacity tolerance, pair derived) */
#define FL_BAD_DEPTH   32u  /* soil depth not finite or negative                                      */
extern "C" __global__ void maple_syrup_floating_prepare(
    const long long n, const int nl, const unsigned char* active, const double* edges_p, const double* depth_p, const double* sepn,
    const double* tr_p, const long long tr_sl, const long long tr_sc, const double* tf_p, const long long tf_sl, const long long tf_sc,
    const double* ts_p, const long long ts_sl, const long long ts_sc, const double* kif_p, const long long kif_sl, const long long kif_sc, const double dcoef, const int directional,
    const double coef, const double stress, const int config_open, unsigned char* bopen_out, unsigned long long* diag)
{
    const long long i = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (i >= n) return;
    unsigned int f = 0u;
    const double d = depth_p[i];
    if (!(isfinite(d) && d >= 0.0)) f |= FL_BAD_DEPTH;
    for (int l = 0; l < nl; ++l) {
        const double r = SP(tr_p, tr_sl, tr_sc, l), fc = SP(tf_p, tf_sl, tf_sc, l), s = SP(ts_p, ts_sl, ts_sc, l);
        if (!(isfinite(r) && isfinite(fc) && isfinite(s))) { f |= FL_BAD_THETA; continue; }
        if (!(r >= 0.0 && fc >= r && s >= fc && s <= 1.0 && s > r)) f |= FL_BAD_REL;
        if (l == 0 && !(stress > r)) f |= FL_BAD_STRESS;
        if (l < nl - 1) { const double v = SP(kif_p, kif_sl, kif_sc, l); if (!(isfinite(v) && v >= 0.0)) f |= FL_BAD_IFACE; }
    }
    if (f == 0u) {  /* derived quantities of every POROUS slab/pair (rock slots carry no property), only for cells whose inputs passed */
        for (int l = 0; l < nl; ++l) {
            const double h = HP(l);
            if (!(h > 0.0)) continue;
            const double sat_m = SP(ts_p, ts_sl, ts_sc, l) * h;
            const double tol = coef * sat_m;
            if (!(isfinite(sat_m) && sat_m > 0.0 && isfinite(tol) && (sat_m + tol) > sat_m)) f |= FL_BAD_DERIVED;
            if (l < nl - 1) {
                const double h2 = HP(l + 1);
                if (h2 > 0.0) {
                    const double inv = 1.0 / h + 1.0 / h2;
                    double sp = (d >= edges_p[l + 2]) ? sepn[l] : (h + h2) / 2.0;
                    const double v = SP(kif_p, kif_sl, kif_sc, l);
                    double g, gu;
                    if (directional != 0) { g = ((dcoef * v) * h) / sp; gu = ((dcoef * v) * h2) / sp; }
                    else { g = v / sp; gu = g; }
                    if (!(isfinite(inv) && isfinite(sp) && sp > 0.0 && isfinite(g) && isfinite(gu) && isfinite(g * inv) && isfinite(gu * inv))) f |= FL_BAD_DERIVED;
                }
            }
        }
    }
    bopen_out[i] = (active[i] != 0 && config_open != 0 && d > edges_p[nl]) ? 1 : 0;  /* strict: D == H is physically closed rock */
    if (f != 0u) {
        atomicOr(&diag[0], (unsigned long long)f);
        atomicAdd(&diag[1], 1ULL);
        atomicMin(&diag[2], ((unsigned long long)i << 8) | (unsigned long long)f);
    }
}

/* ---- resident state validation: mode 0 initial (strictly unsaturated porous layers, rock exactly 0), mode 1 continued (<= sat + tol). Inactive placeholders: finite and >= 0 only. ---- */
#define FL_ST_NEG 1u    /* water not finite or negative (every cell)                                */
#define FL_ST_CAP 2u    /* active: above capacity (continued) / not strictly below saturation (initial, porous) */
#define FL_ST_ROCK 4u   /* active: a rock slot holds water (initial)                                */
extern "C" __global__ void maple_syrup_floating_state_check(
    const long long n, const int nl, const double* w, const unsigned char* active, const double* edges_p, const double* depth_p,
    const double* ts_p, const long long ts_sl, const long long ts_sc, const double coef, const int mode, unsigned long long* diag)
{
    const long long i = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (i >= n) return;
    unsigned int f = 0u;
    const bool a = active[i] != 0;
    for (int l = 0; l < nl; ++l) {
        const double wl = w[(long long)l * n + i];
        if (!(isfinite(wl) && wl >= 0.0)) { f |= FL_ST_NEG; continue; }
        if (!a) continue;
        const double h = HP(l);
        const double sat = SP(ts_p, ts_sl, ts_sc, l) * h;
        if (mode == 0) { if (h > 0.0) { if (!(wl < sat)) f |= FL_ST_CAP; } else if (wl != 0.0) f |= FL_ST_ROCK; }
        else if (!(wl <= sat + coef * sat)) f |= FL_ST_CAP;
    }
    if (f != 0u) {
        atomicOr(&diag[0], (unsigned long long)f);
        atomicAdd(&diag[1], 1ULL);
        atomicMin(&diag[2], ((unsigned long long)i << 8) | (unsigned long long)f);
    }
}

extern "C" __global__ void maple_syrup_floating_moisture(
    const long long n, const int nl, const double* w, const double* edges_p, const double* depth_p, double* out)
{
    const long long i = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (i >= n) return;
    for (int l = 0; l < nl; ++l) { const double h = HP(l); out[(long long)l * n + i] = h > 0.0 ? w[(long long)l * n + i] / h : 0.0; }  /* actual porous theta; 0 where there is no soil, no division by zero */
}

extern "C" __global__ void maple_syrup_floating_storage(const long long n, const int nl, const double* w, const unsigned char* active, double* partials)
{
    __shared__ double sh[BLOCK];
    const int tid = threadIdx.x;
    const long long i = (long long)blockIdx.x * (long long)blockDim.x + (long long)tid;
    double tot = 0.0;
    if (i < n) {
        for (int l = 0; l < nl; ++l) tot = tot + w[(long long)l * n + i];
        if (active[i] == 0) tot = 0.0;  /* inactive placeholders are never counted */
    }
    sh[tid] = tot;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] = sh[tid] + sh[tid + s];
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x] = sh[0];
}
"""


def render_resident_source() -> str:
    """The production source: the shared body (floating geometry readers, contribution epilogue) + all-lane wrapper + staged reconciliation (anchored slices) + resident validation and utilities + the spatial module's
    fixed-tree finalize kernel text for the 7-row step, 4-row reconcile and 1-row storage reductions. Pure and stateless."""
    from maple_syrup import subsurface_cuda_spatial as scs

    tail = (RESIDENT_STEP_TAIL + scs._finalize(RES_STEP_FINAL, STEP_ROWS, 2) + _reconcile_device_function() + scs._finalize(RES_RECONCILE_FINAL, RECONCILE_ROWS, 2) + _PREPARE_AND_UTILITIES
            + scs._finalize(STORAGE_FINAL, 1, 0))
    return sc.render_body_source(RESIDENT_READERS, RESIDENT_STEP_HEAD, tail)


def resident_provenance() -> dict:
    grav, sat = _slices()
    h = lambda t: hashlib.sha256(t.encode()).hexdigest()  # noqa: E731
    return {"module": "maple_syrup.subsurface_cuda_floating", "kernels": list(RESIDENT_KERNELS), "shared_body_sha256": h(sc._BODY), "gravity_slice_sha256": h(grav), "saturation_slice_sha256": h(sat),
            "readers_sha256": h(RESIDENT_READERS), "source_sha256": h(render_resident_source()), "compile_options": list(COMPILE_OPTIONS), "threads_per_block": THREADS, "final_threads": FINAL_THREADS,
            "step_packet_bytes": 8 * STEP_PACKET_WORDS, "reconcile_packet_bytes": 8 * RECONCILE_PACKET_WORDS,
            "reductions": "all-lane fixed block tree (static shared memory) + one-block fixed finalize; integer atomics for the flag OR/count; no CUB/CuPy reduction, no full rows, no per-thread layer arrays"}


def static_resource_report() -> dict:
    """SOURCE-derived resource report (not a compiler report): per kernel, the static shared-memory bytes (`__shared__ double sh[R][BLOCK]`) and the bytes of fixed-size per-thread arrays declared in the kernel/device-function text. The ACTUAL
    registers/local bytes/shared bytes are read from the compiled function attributes by the gated hardware definitions (Root). No array is sized by the layer count."""
    import re

    src = render_resident_source()
    out = {}
    for m in re.finditer(r'extern "C" __global__ void (\w+)\(.*?\n}\n', src, re.S):
        body = m.group(0)
        shared = 0
        for sm in re.finditer(r"__shared__ double sh\[([^\]]+)\](?:\[([^\]]+)\])?", body):
            dims = [int(x.replace("BLOCK", str(THREADS))) if x.strip() != "BLOCK" else THREADS for x in sm.groups() if x]
            n = 1
            for d in dims:
                n *= d
            shared += 8 * n
        local = sum(8 * int(x) for x in re.findall(r"double contrib\[(\d+)\]", body))
        out[m.group(1)] = {"static_shared_bytes": shared, "per_thread_local_array_bytes": local}
    return out
