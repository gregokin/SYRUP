"""H0 correction01: the three independently reproduced metadata/admission findings as regressions (CPU/host; NumPy and Numba), plus the preserved coverage (S1000 L3 N3, analytic counts equal the real arrays)."""
# ruff: noqa: RUF059, C408
from __future__ import annotations

import copy
import json
import tracemalloc

import hbuild as hb
import numpy as np
import pytest
from hbuild import AVAIL, expected_of, make, pref_spec

from maple_syrup import floating_held_context as fh


@pytest.fixture(params=hb.BACKENDS)
def backend(request):
    return request.param


def parts(backend="numpy", **kw):
    vert, owner, bed = make(backend=backend, **kw)
    rec = fh.export_held_context(vert, kw.get("spec"), available_bytes_fn=AVAIL)
    meta, arrays = hb.deserialise(*hb.serialise(rec))
    return vert, rec, meta, arrays


def spy_copies(monkeypatch):
    calls = {"dumps": 0, "deepcopy": 0, "contig": 0, "seal": 0, "stack": 0, "names": 0}
    for obj, name, key in ((json, "dumps", "dumps"), (copy, "deepcopy", "deepcopy"), (np, "ascontiguousarray", "contig"), (fh, "_seal", "seal"), (np, "stack", "stack"), (fh, "_expected_arrays", "names")):
        real = getattr(obj, name)
        monkeypatch.setattr(obj, name, lambda *a, _r=real, _k=key, **k: calls.__setitem__(_k, calls[_k] + 1) or _r(*a, **k))
    return calls


# ---- 1. metadata is validated BEFORE any serialization/deep copy/allocation proportional to the input --------------------------------------------------
def big_mutations(meta):
    big = "x" * 2_000_000
    out = {}
    m = copy.deepcopy(meta)
    m["unknown_payload"] = big
    out["top_unknown"] = m
    for section in ("scalars", "layouts", "identities", "facts"):
        m = copy.deepcopy(meta)
        m[section]["unknown_payload"] = big
        out[f"{section}_unknown"] = m
    m = copy.deepcopy(meta)
    m["layouts"]["all"]["unknown_payload"] = big
    out["layouts_all_unknown"] = m
    m = copy.deepcopy(meta)
    m["schema"] = big
    out["huge_schema"] = m
    m = copy.deepcopy(meta)
    m["scalars"]["matrix_suction_m"] = big
    out["huge_scalar"] = m
    m = copy.deepcopy(meta)
    m["identities"]["config_sha256"] = big
    out["huge_identity"] = m
    m = copy.deepcopy(meta)
    m["overrides"] = [big]
    out["huge_override_name"] = m
    m = copy.deepcopy(meta)
    m["overrides"] = [f"name_{i}" for i in range(5000)]
    out["many_unknown_overrides"] = m
    return out


def test_oversized_or_unknown_metadata_is_refused_before_any_serialization_copy_or_reader(backend, monkeypatch):
    vert, rec, meta, arrays = parts(backend, spec=pref_spec(3, 3, species=1))
    muts = big_mutations(meta)
    muts["bad_layout_kind"] = {**copy.deepcopy(meta), "layouts": {**copy.deepcopy(meta["layouts"]), "all": {**meta["layouts"]["all"], "interface": "diagonal"}}}
    m2 = copy.deepcopy(meta)
    m2["pref"]["n_species"] = 10**9  # a huge declared species count with a handful of arrays
    muts["huge_species"] = m2
    m3 = copy.deepcopy(meta)
    m3["overrides"] = ["theta_33_override"]  # a REAL provider name that is declared but has no array
    muts["declared_override_without_array"] = m3
    m4 = copy.deepcopy(meta)
    m4["n_layers"] = 10**18
    muts["huge_layers"] = m4
    reads = []
    calls = spy_copies(monkeypatch)
    tracemalloc.start()
    tracemalloc.reset_peak()
    t0 = tracemalloc.get_traced_memory()[0]
    for m in muts.values():
        with pytest.raises(fh.HeldContextError):
            fh.record_from_parts(m, arrays, available_bytes_fn=lambda: reads.append(1) or hb.BIG)
    peak = tracemalloc.get_traced_memory()[1] - t0
    tracemalloc.stop()
    assert reads == [] and calls == {"dumps": 0, "deepcopy": 0, "contig": 0, "seal": 0, "stack": 0, "names": 0}, calls  # no reader, no serialization, no copy, no array-name expansion
    assert peak < 200_000, peak  # the 2 MB payloads were never serialized or copied (the reproduced run peaked at 4.5 MB)


def test_a_minted_record_with_replaced_oversized_metadata_is_refused_without_serialization(backend, monkeypatch):
    vert, rec, meta, arrays = parts(backend)
    bad = copy.deepcopy(rec._meta)
    bad["unknown_payload"] = "x" * 2_000_000
    object.__setattr__(rec, "_meta", bad)
    calls = spy_copies(monkeypatch)
    for call in (rec.meta, rec.arrays, lambda: fh.reconstruct_held_context(rec, expected_of(vert), available_bytes_fn=AVAIL)):
        with pytest.raises(fh.HeldContextError):
            call()
    assert calls["dumps"] == 0 and calls["deepcopy"] == 0  # `_verify` validates in place before any JSON/digest work


def test_the_analytic_count_and_bytes_equal_the_real_arrays(backend):
    for kw in ({}, {"mode": "texture_ks"}, {"L": 1}, {"spec": pref_spec(3, 3, kind="cell", rate="cell", species=2)},
               {"overrides": {"theta_33_override": np.full((3, 3), np.nan)}, "mode": "texture_ks"}, {"spec": pref_spec(1, 3, kind="scalar", species=0), "L": 1}):
        vert, rec, meta, arrays = parts(backend, **kw)
        count, nbytes = fh._analytic(meta)
        assert count == len(arrays) and nbytes == sum(int(a.nbytes) for a in arrays.values()), kw
    # the declared-resource cap rejects an unsupported record before any name/shape/copy is built
    with pytest.raises(fh.HeldContextError, match="above max_bytes"):
        fh.record_from_parts(meta, arrays, max_bytes=nbytes - 1, available_bytes_fn=AVAIL)
    short = dict(arrays)
    short.pop(next(iter(short)))
    with pytest.raises(fh.HeldContextError, match="arrays but"):
        fh.record_from_parts(meta, short, available_bytes_fn=AVAIL)


# ---- 2. mandatory scalars --------------------------------------------------------------------------------------------------------------------------------
REQUIRED = ("bulk_density_kg_m3", "theta_residual", "theta_evaporation_stress", "evaporation_factor", "matrix_suction_m", "matrix_vegetation_factor", "initial_wetting_m", "initial_cumulative_intake_m", "cell_area_m2")


@pytest.mark.parametrize("name", REQUIRED)
@pytest.mark.parametrize("bad", [None, "inf", "-inf", "nan", "0.1", 5, True, ""])
def test_required_scalars_must_be_canonical_finite_hex(backend, name, bad, monkeypatch):
    vert, rec, meta, arrays = parts(backend)
    m = copy.deepcopy(meta)
    m["scalars"][name] = (float("inf").hex() if bad == "inf" else float("-inf").hex() if bad == "-inf" else float("nan").hex() if bad == "nan" else bad)
    calls = spy_copies(monkeypatch)
    reads = []
    with pytest.raises(fh.HeldContextError):  # a TYPED refusal at the metadata, never a TypeError out of the reconstruction
        fh.record_from_parts(m, arrays, available_bytes_fn=lambda: reads.append(1) or hb.BIG)
    assert reads == [] and calls["seal"] == 0 and calls["contig"] == 0


def test_mode_specific_conductivities_and_the_preferential_scalars(backend):
    vert, rec, meta, arrays = parts(backend, mode="texture_ks")
    for key, value in (("matrix_conductivity_m_s", (1e-6).hex()), ("latent_saturated_conductivity_m_s", None), ("latent_saturated_conductivity_m_s", float("inf").hex())):
        m = copy.deepcopy(meta)
        m["scalars"][key] = value
        with pytest.raises(fh.HeldContextError):
            fh.record_from_parts(m, arrays, available_bytes_fn=AVAIL)
    vert, rec, meta, arrays = parts(backend)
    m = copy.deepcopy(meta)
    m["scalars"]["matrix_conductivity_m_s"] = None
    with pytest.raises(fh.HeldContextError):
        fh.record_from_parts(m, arrays, available_bytes_fn=AVAIL)
    spec = pref_spec(3, 3, kind="scalar", rate="inf")
    pv, prec, pmeta, parrays = parts(backend, spec=spec)
    assert pmeta["pref"]["max_rate_value"] == float("inf").hex()  # inf IS allowed for the scalar maximum intake rate (the physics permits an unbounded rate) ...
    fh.record_from_parts(pmeta, parrays, available_bytes_fn=AVAIL)
    for key in ("a3_value", "max_fraction_per_hour"):  # ... and only there
        m = copy.deepcopy(pmeta)
        m["pref"][key] = float("inf").hex()
        with pytest.raises(fh.HeldContextError):
            fh.record_from_parts(m, parrays, available_bytes_fn=AVAIL)
    for key in ("a3_value", "max_rate_value", "max_fraction_per_hour"):
        m = copy.deepcopy(pmeta)
        m["pref"][key] = None
        with pytest.raises(fh.HeldContextError):
            fh.record_from_parts(m, parrays, available_bytes_fn=AVAIL)


def test_wrong_but_well_formed_values_still_meet_the_real_public_factory_bounds(backend):
    vert, rec, meta, arrays = parts(backend)
    m = copy.deepcopy(meta)
    m["scalars"]["matrix_vegetation_factor"] = (2.0).hex()  # canonical and finite, but outside the operator's [0, 1]
    p = fh.record_from_parts(m, arrays, available_bytes_fn=AVAIL)
    with pytest.raises(fh.HeldContextError, match="refused the record|reproduce"):
        fh.reconstruct_held_context(p, expected_of(vert), available_bytes_fn=AVAIL)


# ---- 3. export: nothing proportional is allocated before the admission --------------------------------------------------------------------------------
def test_export_with_two_species_allocates_nothing_before_the_reader_says_yes(backend, monkeypatch):
    spec = pref_spec(3, 3, kind="cell", species=2)
    vert, owner, bed = make(backend=backend, spec=spec, mode="texture_ks", overrides={"theta_33_override": np.full((3, 3), np.nan)})
    calls = spy_copies(monkeypatch)
    reads = []
    tracemalloc.start()
    tracemalloc.reset_peak()
    t0 = tracemalloc.get_traced_memory()[0]
    with pytest.raises(fh.HeldContextError, match="reported available"):
        fh.export_held_context(vert, spec, available_bytes_fn=lambda: reads.append(1) or 0)
    peak = tracemalloc.get_traced_memory()[1] - t0
    tracemalloc.stop()
    assert reads == [1] and calls["stack"] == 0 and calls["seal"] == 0 and calls["contig"] == 0 and peak < 20_000, (calls, peak)  # the (S, L) stack and every sealed copy come AFTER the admission
    m = fh._model(*[fh._snapshot(vert, spec)[0]])
    ok = fh.export_held_context(vert, spec, available_bytes_fn=lambda: m.export_bytes)  # the exact declared need admits
    assert ok.arrays()["pref_species_weights"].shape == (2, 3)


def test_the_preferential_spec_can_be_constructed_behind_the_real_reader(monkeypatch):
    calls = spy_copies(monkeypatch)
    kw = dict(a3=np.full(50, 2e-3), max_fraction_per_hour=0.5, max_intake_rate_m_s=1e-5, background_receiver_weights=np.ones(3), species_receiver_weights=(np.ones(3),), species_contributions=(np.full(50, 0.5),))
    need = (50 + 3 + 3 + 50) * 8 + 4 * fh.SPEC_ARRAY_OVERHEAD_BYTES + fh.ALLOWANCE_BYTES  # raw payload + the declared per-array metadata allowance (4 arrays + 2 species arrays: a3, background, weights, contribution)
    assert need == fh.PreferentialSpec.construction_need(kw["a3"], 1e-5, kw["background_receiver_weights"], kw["species_receiver_weights"], kw["species_contributions"])
    with pytest.raises(fh.HeldContextError):
        fh.PreferentialSpec.admitted(**kw, available_bytes_fn=lambda: need - 1)
    assert calls["seal"] == 0  # nothing was copied before the reader spoke
    spec = fh.PreferentialSpec.admitted(**kw, available_bytes_fn=lambda: need)
    assert spec.species_receiver_weights[0].flags.writeable is False
    for bad in (-1, True, 1.5):
        with pytest.raises(fh.HeldContextError):
            fh.PreferentialSpec.admitted(**kw, available_bytes_fn=lambda: need, reserve_bytes=bad)


# ---- preserved coverage: S1000 L3 N3 ---------------------------------------------------------------------------------------------------------------------
def test_one_thousand_species_l3_n3_still_reconstructs_within_the_declared_model(backend):
    spec = pref_spec(3, 3, kind="cell", species=1000)
    vert, owner, bed = make(backend=backend, spec=spec)
    rec = fh.export_held_context(vert, spec, available_bytes_fn=AVAIL)
    m = fh._model(rec.meta())
    expected = expected_of(vert)
    fh.reconstruct_held_context(rec, expected, available_bytes_fn=AVAIL)  # warm
    tracemalloc.start()
    tracemalloc.reset_peak()
    t0 = tracemalloc.get_traced_memory()[0]
    ctx = fh.reconstruct_held_context(rec, expected, available_bytes_fn=lambda: m.reconstruct_bytes)
    peak = tracemalloc.get_traced_memory()[1] - t0
    tracemalloc.stop()
    assert ctx.preferential.n_species == 1000 and peak <= m.reconstruct_bytes, (peak, m.reconstruct_bytes)
