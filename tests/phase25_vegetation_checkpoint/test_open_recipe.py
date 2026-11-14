"""CPU proof of the NEW open-plane static recipe helper (`veg_open_recipe`, correction01 C2): the same code the GPU fresh worker uses (backend `cuda` is only a lazy branch) rebuilds the original generation-0 topology through the actual API, the
static event reproduces ALL original binding digests including the BOUND cover, the old closed-lake helper keeps refusing the open plane, tampered recipes are refused, and a physical vegetation checkpoint restores through the recipe-built static
event and continues bit-exactly like the uninterrupted run. Real native states; no fake objects."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import veg_open_recipe as vr
from vck_builders import boundary_vns, load, save, storm

from maple_syrup import vegetation_lifecycle as vl


@pytest.fixture(scope="module")
def open_ns(tmp_path_factory, leaf):
    tmp = tmp_path_factory.mktemp("open")
    ns = boundary_vns(tmp / "a", leaf, "numpy", False, deep_dense=True, splash=False)
    captured = []
    real = ns.factory

    def cap(vert, topology=None, cover=None, schedule=None):
        ev = real(vert, topology, cover, schedule)
        captured.append(ev)
        return ev

    ns.factory = cap
    saved = save(ns, tmp / "ck")
    return ns, saved, captured[0], real, tmp


def test_the_old_helper_still_serves_the_closed_lake_only_and_the_new_one_serves_the_open_plane(open_ns):
    ns, _saved, event, _real, _t = open_ns
    import cc2_static_case as sc2

    base = sc2.make_recipe(event, ns.original)
    assert base["closed"] is False
    with pytest.raises(sc2.RecipeError, match="closed lake only"):
        sc2.original_topology(base)  # the historical guard is preserved untouched
    rec = vr.make_open_recipe(event, ns.original)
    top = vr.original_topology(rec)
    assert (top.identity_sha256, top.source_sha256, top.topology_sha256, int(top.n_closed_components)) == (ns.original.identity_sha256, ns.original.source_sha256, ns.original.topology_sha256, 0)


def test_tampered_open_recipes_are_refused(open_ns, tmp_path):
    ns, _saved, event, _real, _t = open_ns
    rec = vr.make_open_recipe(event, ns.original)
    vr.write_open_recipe(tmp_path / "r.json", rec)
    assert vr.read_open_recipe(tmp_path / "r.json") == rec
    bad = {
        "friction": lambda r: r["base"].update(friction=r["base"]["friction"] * 2.0),
        "datum": lambda r: r["base"]["original_source_full"][1].__setitem__(1, r["base"]["original_source_full"][1][1] + 1e-3),
        "dx": lambda r: r["base"].update(dx_m=r["base"]["dx_m"] * 2.0),
        "topo_digest": lambda r: r["base"]["original_topology"].update(topology_sha256="0" * 64),
    }
    for edit in bad.values():
        r = copy.deepcopy(rec)
        edit(r)
        with pytest.raises((vr.OpenRecipeError, ValueError, RuntimeError)):
            vr.original_topology(r)
    for edit in (lambda r: r.update(outlets="west"), lambda r: r.update(schema="x"), lambda r: r.update(extra=1), lambda r: r["base"].update(closed=True)):
        r = copy.deepcopy(rec)
        edit(r)
        with pytest.raises(vr.OpenRecipeError, match="mismatch|non-closed|keys"):
            vr.check_open_recipe(r)
    raw = json.loads((tmp_path / "r.json").read_text())
    raw["base"]["rain_mm_h"] = "x"
    (tmp_path / "bad.json").write_text(json.dumps(raw))
    with pytest.raises((vr.OpenRecipeError, RuntimeError)):
        vr.read_open_recipe(tmp_path / "bad.json")


def test_a_static_event_reproducing_every_binding_digest_restores_and_continues_bit_exactly(open_ns):
    ns, saved, event, real, _t = open_ns
    rec = vr.make_open_recipe(event, ns.original)
    original = vr.original_topology(rec)

    def static(vert, topology, cover, schedule=None):
        return vr.static_event(rec, vert, topology, cover, "numpy", schedule)

    r = load(ns, saved.path, saved, factory=static)
    ns2 = type(ns)(**{**vars(ns), "factory": static, "original": original})
    assert vl.current_record(r.head).binding.origin == "cached_derived"
    a = storm(ns.lin, ns.lin.head, real)
    b = storm(r.lineage, r.head, static)
    assert np.array_equal(a.root.water_state.water_m, b.root.water_state.water_m) and a.ledger == b.ledger and a.t_s == b.t_s
    assert a.root.matrix_op.sha256 == b.root.matrix_op.sha256 and ns2.original is original
    # a recipe whose digests do not match the event it builds is refused at construction
    bad = copy.deepcopy(rec)
    bad["base"]["digests"]["context"] = "0" * 64
    with pytest.raises(vr.OpenRecipeError, match="bindings"):
        load(ns, saved.path, saved, factory=lambda vert, topo, cover: vr.static_event(bad, vert, topo, cover, "numpy"))
