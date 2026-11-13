"""H0: water-free held law-context export/reconstruction (CPU/host; NumPy and Numba). Real native beds/owners/verticals/preferential operators; reconstruction touches no bed, texture reader, provider or owner."""
# ruff: noqa: RUF059
from __future__ import annotations

import copy
import dataclasses
import pickle
import tracemalloc

import builders as b
import hbuild as hb
import numpy as np
import pytest
from hbuild import AVAIL, expected_of, make, pref_spec

from maple_syrup import floating_held_context as fh
from maple_syrup import floating_matrix_infiltration as fmi
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import native_floating_soil as nfs
from maple_syrup import soil_texture_properties as stp


@pytest.fixture(params=hb.BACKENDS)
def backend(request):
    return request.param


def roundtrip(vert, spec=None):
    rec = fh.export_held_context(vert, spec, available_bytes_fn=AVAIL)
    meta, arrays = hb.deserialise(*hb.serialise(rec))  # through plain parts, as a file codec would
    parsed = fh.record_from_parts(meta, arrays, available_bytes_fn=AVAIL)
    return rec, parsed, fh.reconstruct_held_context(parsed, expected_of(vert), available_bytes_fn=AVAIL)


# ---- exact identities across configurations ------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("L", [1, 3, 65])
@pytest.mark.parametrize("mode", ["explicit", "texture_ks"])
@pytest.mark.parametrize("overrides", [None, {}])
@pytest.mark.parametrize("bottom", ["open", "closed"])
def test_reconstruction_reproduces_config_column_matrix_identities(backend, L, mode, overrides, bottom):
    vert, owner, bed = make(L=L, backend=backend, mode=mode, overrides=overrides, bottom=bottom)
    rec, parsed, ctx = roundtrip(vert)
    t = vert.held_template()
    assert ctx.config.sha256 == owner.config.sha256 and ctx.column.sha256 == t.column.sha256 and ctx.matrix_op.sha256 == t.matrix_op.sha256 and ctx.preferential is None
    assert (ctx.config.overrides is None) == (overrides is None)  # None vs {} preserved
    assert ctx.config.layers.thickness_m.tobytes() == owner.config.layers.thickness_m.tobytes()  # the ORIGINAL thicknesses, not diff(edges)
    assert ctx.depth0_m.tobytes() == t.depth0_m.tobytes() and ctx.active.tobytes() == t.column._maps["active"].tobytes() and ctx.area_m2 == t.column.cell_area_m2
    assert (ctx.ks_m_s is None) == (mode == "explicit") and (ctx.ks_m_s is None or ctx.ks_m_s.tobytes() == owner.ks_m_s.tobytes())
    assert ctx.generation == owner.generation and ctx.source_witness == owner.binding.source_witness
    for name in ("water", "state", "history", "j", "i"):
        assert not hasattr(ctx, name)  # water-free by construction
    assert ctx.column.backend == backend and bool(ctx.column._maps["theta_field_capacity"].flags.writeable) is False


def test_true_overrides_are_restored_byte_exact(backend):
    ov = {"theta_33_override": np.full((3, 3), np.nan)}
    ov["theta_33_override"][1, 1] = 0.3
    vert, owner, _ = make(backend=backend, overrides=ov)
    rec, parsed, ctx = roundtrip(vert)
    assert set(ctx.config.overrides) == {"theta_33_override"} and ctx.config.overrides["theta_33_override"].tobytes() == owner.config.overrides["theta_33_override"].tobytes()
    assert ctx.config.sha256 == owner.config.sha256
    assert ctx.column.sha256 == vert.held_template().column.sha256  # the held FC/SAT maps carry the measured value (no fake override re-derivation)


@pytest.mark.parametrize("kind,rate,species", [("cell", "scalar", 0), ("scalar", "scalar", 0), ("cell", "cell", 0), ("scalar", "inf", 0), ("cell", "scalar", 2)])
@pytest.mark.parametrize("L", [1, 3])
def test_preferential_is_reprepared_from_the_explicit_spec_with_the_exact_sha(backend, kind, rate, species, L):
    N = 3
    spec = pref_spec(L, N, kind=kind, rate=rate, species=species)
    vert, owner, _ = make(L=L, backend=backend, spec=spec)
    rec, parsed, ctx = roundtrip(vert, spec)
    assert ctx.preferential.sha256 == vert.held_template().preferential.sha256 and ctx.preferential.column is ctx.column
    assert ctx.preferential.n_species == species
    assert parsed.meta()["pref"]["max_rate_kind"] == ("cell" if rate == "cell" else "scalar")


def test_misstated_or_missing_preferential_spec_is_refused_at_export(backend):
    spec = pref_spec(3, 3, species=1)
    vert, *_ = make(backend=backend, spec=spec)
    wrong = fh.PreferentialSpec(np.full(3, 3.0e-3), spec.max_fraction_per_hour, spec.max_intake_rate_m_s, spec.background_receiver_weights, spec.species_receiver_weights, spec.species_contributions)
    with pytest.raises(fh.HeldContextError, match="sha256"):
        fh.export_held_context(vert, wrong, available_bytes_fn=AVAIL)
    with pytest.raises(fh.HeldContextError, match="explicit PreferentialSpec"):
        fh.export_held_context(vert, None, available_bytes_fn=AVAIL)
    plain, *_ = make(backend=backend)
    with pytest.raises(fh.HeldContextError, match="no preferential"):
        fh.export_held_context(plain, spec, available_bytes_fn=AVAIL)


def test_the_context_for_a_partial_and_rock_profile_is_a_law_context_not_a_water_state(backend):
    vert, owner, bed = make(backend=backend, bed=b.stratified(shallow=((0, 2),)))
    rec, parsed, ctx = roundtrip(vert)
    d0 = ctx.depth0_m
    assert float(d0.min()) < float(owner.column._maps["edges"][-1]) <= float(d0.max()) or float(d0.min()) >= 0.0  # shallow (partial/rock) and deeper cells coexist in the held D0
    assert ctx.column.sha256 == owner.column.sha256


# ---- nothing is recomputed from a bed, a texture or the property provider -----------------------------------------------------------------------------
def test_restore_never_touches_a_bed_texture_provider_or_owner_factory(backend, monkeypatch):
    spec = pref_spec(3, 3)
    vert, owner, bed = make(backend=backend, spec=spec, mode="texture_ks")
    rec = fh.export_held_context(vert, spec, available_bytes_fn=AVAIL)
    meta, arrays = hb.deserialise(*hb.serialise(rec))
    expected = expected_of(vert)

    def boom(*a, **k):
        raise AssertionError("must not run during parse/reconstruct")

    for obj, name in ((stp, "estimate_soil_texture_properties"), (nfs, "_read_texture"), (nfs, "initialize_native_floating_soil"), (nfs, "native_source_witness"), (nfs, "stage_native_refresh"),
                      (nfs, "read_canonical_depth"), (nfs, "NativeBed")):
        monkeypatch.setattr(obj, name, boom)
    parsed = fh.record_from_parts(meta, arrays, available_bytes_fn=AVAIL)
    ctx = fh.reconstruct_held_context(parsed, expected, available_bytes_fn=AVAIL)
    assert ctx.column.sha256 == expected.root_column_sha256


# ---- layouts (scalar / shared / cell), through the PUBLIC factories ------------------------------------------------------------------------------
@pytest.mark.parametrize("layout", ["scalar", "shared", "cell"])
def test_scalar_shared_and_cell_raw_map_layouts_round_trip(backend, layout):
    vert, owner, bed = make(backend=backend)
    cfg, t = owner.config, vert.held_template()
    N = 3
    fc = {"scalar": 0.25, "shared": np.array([0.25, 0.26, 0.27]), "cell": np.tile(np.array([0.25, 0.26, 0.27])[:, None], (1, N))}[layout]
    sat = {"scalar": 0.42, "shared": np.array([0.42, 0.43, 0.44]), "cell": np.tile(np.array([0.42, 0.43, 0.44])[:, None], (1, N))}[layout]
    col = fsl.prepare_floating_column(cfg.nominal_edges_m, theta_residual=cfg.theta_residual, theta_field_capacity=fc, theta_saturation=sat, diffusivity_m2_s=cfg.diffusivity_m2_s, theta_evaporation_stress=cfg.theta_evaporation_stress,
                                      evaporation_factor=cfg.evaporation_factor, diffusion_mode="constant", bottom_drainage=cfg.bottom_drainage, soil_depth_m=np.array(t.depth0_m), active_mask=np.array(t.column._maps["active"]),
                                      cell_area_m2=t.column.cell_area_m2, backend=backend, available_bytes_fn=AVAIL)
    op = fmi.prepare_floating_matrix(col, conductivity_m_s=cfg.matrix_conductivity_m_s, suction_m=cfg.matrix_suction_m, vegetation_factor=cfg.matrix_vegetation_factor, available_bytes_fn=AVAIL)
    rec = fh.export_held_context(vert, None, available_bytes_fn=AVAIL, verify=False)
    meta, arrays = rec.meta(), {k: np.array(v) for k, v in rec.arrays().items()}
    meta["layouts"] = {"theta_field_capacity": layout, "theta_saturation": layout, "all": {k: str(v) for k, v in dict(col.layouts).items()}}
    meta["identities"] = {**meta["identities"], "root_column_sha256": col.sha256, "matrix_op_sha256": op.sha256}
    for name, raw in (("theta_field_capacity", fc), ("theta_saturation", sat)):
        arrays[name] = np.ascontiguousarray(np.array([raw]) if layout == "scalar" else raw, dtype=np.float64)
    parsed = fh.record_from_parts(meta, arrays, available_bytes_fn=AVAIL)
    ctx = fh.reconstruct_held_context(parsed, fh.ExpectedHeldIdentities(cfg.sha256, col.sha256, op.sha256, None), available_bytes_fn=AVAIL)
    assert ctx.column.sha256 == col.sha256 and dict(ctx.column.layouts)["theta_field_capacity"] == layout


# ---- refusals ----------------------------------------------------------------------------------------------------------------------------------------
def parts(backend="numpy", **kw):
    vert, owner, bed = make(backend=backend, **kw)
    rec = fh.export_held_context(vert, kw.get("spec"), available_bytes_fn=AVAIL)
    meta, arrays = hb.deserialise(*hb.serialise(rec))
    return vert, rec, meta, arrays


def test_misstated_config_root_sha_ks_and_expected_identities_are_refused(backend):
    vert, rec, meta, arrays = parts(backend, mode="texture_ks")
    good = expected_of(vert)
    for field in ("config_sha256", "root_column_sha256", "matrix_op_sha256"):
        bad = dataclasses.replace(good, **{field: "0" * 64})
        with pytest.raises(fh.HeldContextError, match="expected identities"):
            fh.reconstruct_held_context(rec, bad, available_bytes_fn=AVAIL)
    for key, mutate in (("scalars.matrix_suction_m", lambda m: m["scalars"].__setitem__("matrix_suction_m", (0.2).hex())), ("scalars.initial_wetting_m", lambda m: m["scalars"].__setitem__("initial_wetting_m", (0.5).hex()))):
        m2 = copy.deepcopy(meta)
        mutate(m2)
        p = fh.record_from_parts(m2, arrays, available_bytes_fn=AVAIL)
        with pytest.raises(fh.HeldContextError):
            fh.reconstruct_held_context(p, good, available_bytes_fn=AVAIL)  # the rebuilt config/operator no longer reproduces the recorded identity
    a2 = {k: v.copy() for k, v in arrays.items()}
    a2["ks_m_s"][0] *= 1.5
    with pytest.raises(fh.HeldContextError, match="matrix operator"):
        fh.reconstruct_held_context(fh.record_from_parts(meta, a2, available_bytes_fn=AVAIL), good, available_bytes_fn=AVAIL)
    a3 = {k: v.copy() for k, v in arrays.items()}
    a3["theta_field_capacity"][0, 0] *= 0.99
    with pytest.raises(fh.HeldContextError, match="root column"):
        fh.reconstruct_held_context(fh.record_from_parts(meta, a3, available_bytes_fn=AVAIL), good, available_bytes_fn=AVAIL)
    spec = pref_spec(3, 3)
    pv, prec, pmeta, parrays = parts(backend, spec=spec)
    p4 = {k: v.copy() for k, v in parrays.items()}
    p4["pref_a3"][0] = 3e-3
    with pytest.raises(fh.HeldContextError, match="preferential"):
        fh.reconstruct_held_context(fh.record_from_parts(pmeta, p4, available_bytes_fn=AVAIL), expected_of(pv), available_bytes_fn=AVAIL)


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
@pytest.mark.parametrize("what", ["extra_key", "missing_key", "schema", "cuda", "bad_hex", "nan", "extra_array", "missing_array", "dtype", "shape", "noncontiguous", "list", "subclass"])
def test_malformed_parts_are_refused_before_any_copy(backend, what, monkeypatch):
    vert, rec, meta, arrays = parts(backend, mode="texture_ks")
    m, a = copy.deepcopy(meta), dict(arrays)
    if what == "extra_key":
        m["extra"] = 1
    elif what == "missing_key":
        del m["facts"]
    elif what == "schema":
        m["schema"] = "other/1"
    elif what == "cuda":
        m["backend"] = "cuda"
    elif what == "bad_hex":
        m["scalars"]["matrix_suction_m"] = "0.1"
    elif what == "nan":
        m["scalars"]["matrix_suction_m"] = "nan"
    elif what == "extra_array":
        a["junk"] = np.zeros(2)
    elif what == "missing_array":
        del a["ks_m_s"]
    elif what == "dtype":
        a["depth0_m"] = a["depth0_m"].astype(np.float32)
    elif what == "shape":
        a["depth0_m"] = a["depth0_m"][:-1].copy()
    elif what == "noncontiguous":
        wide = np.zeros((a["theta_field_capacity"].shape[0], a["theta_field_capacity"].shape[1] * 2))
        wide[:, ::2] = a["theta_field_capacity"]
        a["theta_field_capacity"] = wide[:, ::2]
    elif what == "list":
        a["depth0_m"] = list(a["depth0_m"])
    elif what == "subclass":
        a["depth0_m"] = np.matrix(a["depth0_m"])
    copies = []
    real = np.ascontiguousarray
    monkeypatch.setattr(np, "ascontiguousarray", lambda *x, **k: copies.append(1) or real(*x, **k))
    reads = []
    with pytest.raises(fh.HeldContextError):
        fh.record_from_parts(m, a, available_bytes_fn=lambda: reads.append(1) or BIG)
    assert copies == [] and reads == []  # refused on metadata alone: no conversion copy, no reader call


BIG = hb.BIG


def test_private_alteration_of_a_minted_record_is_refused(backend):
    vert, rec, meta, arrays = parts(backend)
    expected = expected_of(vert)
    arr = rec._arrays["depth0_m"]
    keep = arr.copy()
    arr.flags.writeable = True  # low-level re-enabling of a sealed array
    arr[0] += 1e-9
    with pytest.raises(fh.HeldContextError, match="altered"):
        fh.reconstruct_held_context(rec, expected, available_bytes_fn=AVAIL)
    arr[...] = keep
    arr.flags.writeable = False
    assert fh.reconstruct_held_context(rec, expected, available_bytes_fn=AVAIL).column.sha256 == expected.root_column_sha256  # the digest binds the VALUES: restoring them restores the record
    object.__setattr__(rec, "_arrays", dict(rec._arrays))
    with pytest.raises(fh.HeldContextError, match="replaced"):
        rec.arrays()
    for bad in (lambda: copy.copy(rec), lambda: copy.deepcopy(rec), lambda: pickle.dumps(rec), lambda: dataclasses.replace(rec, _meta={}), lambda: fh.HeldContextRecord({}, {})):
        with pytest.raises(fh.HeldContextError):
            bad()


def test_minted_context_is_uncopyable_and_the_record_is_required(backend):
    vert, rec, meta, arrays = parts(backend)
    ctx = fh.reconstruct_held_context(rec, expected_of(vert), available_bytes_fn=AVAIL)
    for bad in (lambda: copy.copy(ctx), lambda: copy.deepcopy(ctx), lambda: pickle.dumps(ctx), lambda: fh.HeldContext(*([None] * 12))):
        with pytest.raises(fh.HeldContextError):
            bad()
    for plain in ({"meta": meta}, None, "x"):
        with pytest.raises(fh.HeldContextError):
            fh.reconstruct_held_context(plain, expected_of(vert), available_bytes_fn=AVAIL)
    with pytest.raises(fh.HeldContextError):
        fh.reconstruct_held_context(rec, {"config_sha256": "x"}, available_bytes_fn=AVAIL)


# ---- admission: reserves, zero / need-1 / exact, two stages ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [-1, True, 1.5, "0", None, np.float64(2.0)])
def test_reserve_is_a_strict_nonboolean_integer(backend, bad):
    vert, rec, meta, arrays = parts(backend)
    reads = []
    fn = lambda: reads.append(1) or BIG
    for call in (lambda: fh.record_from_parts(meta, arrays, available_bytes_fn=fn, reserve_bytes=bad), lambda: fh.reconstruct_held_context(rec, expected_of(vert), available_bytes_fn=fn, reserve_bytes=bad),
                 lambda: fh.export_held_context(vert, None, available_bytes_fn=fn, reserve_bytes=bad)):
        with pytest.raises(fh.HeldContextError):
            call()
    assert reads == []


@pytest.mark.parametrize("L", [1, 3, 65])
def test_two_stage_admission_zero_need_minus_one_exact_and_reserve_once(backend, L):
    spec = pref_spec(L, 3, species=1)
    vert, rec, meta, arrays = parts(backend, L=L, spec=spec, mode="texture_ks", overrides={"theta_33_override": np.full((L, 3), np.nan)})
    m = fh._model(meta)
    expected = expected_of(vert)
    # stage 1: parse (the sealed copies)
    parse_need = sum(int(a.nbytes) for a in arrays.values()) + fh.ALLOWANCE_BYTES
    for avail in (0, parse_need - 1):
        with pytest.raises(fh.HeldContextError, match="reported available"):
            fh.record_from_parts(meta, arrays, available_bytes_fn=lambda a=avail: a)
    fh.record_from_parts(meta, arrays, available_bytes_fn=lambda: parse_need)
    # stage 2: reconstruct
    for avail in (0, m.reconstruct_bytes - 1):
        with pytest.raises(fh.HeldContextError, match="reported available"):
            fh.reconstruct_held_context(rec, expected, available_bytes_fn=lambda a=avail: a)
    with pytest.raises(fh.HeldContextError):
        fh.reconstruct_held_context(rec, expected, available_bytes_fn=lambda: m.reconstruct_bytes, reserve_bytes=1)  # the reserve is counted once on top of the declared need
    ctx = fh.reconstruct_held_context(rec, expected, available_bytes_fn=lambda: BIG, reserve_bytes=5)
    assert ctx.column.sha256 == expected.root_column_sha256
    # export: the sealed copies and the verification reconstruction are in its need
    with pytest.raises(fh.HeldContextError, match="reported available"):
        fh.export_held_context(vert, spec, available_bytes_fn=lambda: 0)
    assert m.export_bytes == m.record_bytes + m.reconstruct_bytes


def test_default_real_host_reader_is_used_when_none_is_given(backend, monkeypatch):
    from maple_syrup import spatial_soil_layers as sp

    vert, rec, meta, arrays = parts(backend)
    seen = []
    monkeypatch.setattr(sp, "default_host_available_bytes", lambda: seen.append(1) or BIG)
    fh.reconstruct_held_context(rec, expected_of(vert))
    assert seen
    monkeypatch.setattr(sp, "default_host_available_bytes", lambda: 0)
    with pytest.raises(fh.HeldContextError):
        fh.reconstruct_held_context(rec, expected_of(vert))


def test_modelled_bytes_bound_the_traced_peak_of_a_moderate_reconstruction(backend):
    N = 4096
    bed = b.bed_from(b.fx.stratified_dense(shallow_cells=(), ny=1, nx=N))
    spec = pref_spec(3, N, kind="cell", species=1)
    vert, owner, _ = make(backend=backend, bed=bed, spec=spec, mode="texture_ks", overrides={"theta_33_override": np.full((3, N), np.nan)})
    rec = fh.export_held_context(vert, spec, available_bytes_fn=AVAIL)
    expected = expected_of(vert)
    m = fh._model(rec.meta())
    fh.reconstruct_held_context(rec, expected, available_bytes_fn=AVAIL)  # warm (compile)
    tracemalloc.start()
    tracemalloc.reset_peak()
    t0 = tracemalloc.get_traced_memory()[0]
    ctx = fh.reconstruct_held_context(rec, expected, available_bytes_fn=lambda: m.reconstruct_bytes)
    peak = tracemalloc.get_traced_memory()[1] - t0
    tracemalloc.stop()
    assert ctx.column.n_cells == N
    assert peak <= m.reconstruct_bytes, (peak, m)
    assert peak >= 3 * N * 8  # non-vacuous: at least the prepared maps
