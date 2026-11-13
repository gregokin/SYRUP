"""CC2 static-recipe HOST tests: the no-original-bed recipe, built on the CPU backend, reproduces the matched DEEP-pair original CPU case's SourceBinding exactly, restores a genuine continued checkpoint, and is closed/bounded."""
from __future__ import annotations

import copy
import json

import cc2_fixtures as f
import cc2_static_case as sc2
import numpy as np
import pytest
from cont_builders import AVAIL, IDENT, cc, fc, save, snapshot

from maple_syrup import floating_context_checkpoint as fcx
from maple_syrup import floating_held_context as fh
from maple_syrup import pond_checkpoint as pc


@pytest.fixture(params=[False, True], ids=["dense", "windowed"])
def case(request, tmp_path):
    ns = f.cpu_control(tmp_path / "a", request.param)
    return ns, sc2.make_recipe(ns.ev0, ns.original)


def _ctx(ns, record=None):
    rec = ns.record if record is None else record
    return fh.reconstruct_held_context(rec, fcx._expected_from(rec.meta()), available_bytes_fn=AVAIL)


def test_the_recipe_event_reproduces_the_original_cases_source_binding_exactly(case):
    ns, r = case
    make = sc2.factory(r, "numpy")
    ev = make(_ctx(ns, ns.record0))  # the ORIGINAL role on the verified ORIGINAL-root H0 context: no bed, owner or probe
    sc2.verify_event(r, ev)  # context / sediment / schedule / options / config / forcing digests (config includes the original root identity)
    a, b = pc._binding(ev, pc.SCHEMA), pc._binding(ns.ev0, pc.SCHEMA)
    assert a["context_sha256"] == b["context_sha256"] and a["initial_topology"] == b["initial_topology"] and a["options_sha256"] == b["options_sha256"]
    top = sc2.original_topology(r)
    assert top.source_sha256 == ns.original.source_sha256 and np.array_equal(top.source_z_full, ns.original.source_z_full)
    assert ev._initial_topology.topology_sha256 == ns.original.topology_sha256


def test_save_refuses_a_drifted_recipe(case):
    ns, r = case
    for edit in (lambda x: x["digests"].update(context="0" * 64), lambda x: x.update(rain_mm_h=1.0), lambda x: x["policy"].update(hold_s=2.0), lambda x: x["control"].update(max_dt_s=0.5)):
        bad = copy.deepcopy(r)
        edit(bad)
        ev = sc2.factory(bad, "numpy")(_ctx(ns, ns.record0))
        with pytest.raises(sc2.RecipeError, match="does not reproduce"):
            sc2.verify_event(bad, ev)


def test_the_recipe_is_closed_bounded_json_and_a_wrong_datum_is_refused(case, tmp_path):
    _ns, r = case
    p = tmp_path / "recipe.json"
    sc2.write_recipe(p, r)
    assert sc2.read_recipe(p) == json.loads(json.dumps(r))
    for edit in (lambda x: x.update(extra=1), lambda x: x.update(ny=10**6), lambda x: x["original_source_full"][0].__setitem__(0, float("nan")), lambda x: x["policy"].update(unknown=1),
                 lambda x: x["original_topology"].update(generation=1), lambda x: x["digests"].update(config="zz"), lambda x: x.update(schema="other")):
        bad = copy.deepcopy(r)
        edit(bad)
        with pytest.raises(sc2.RecipeError):
            sc2.check_recipe(bad)
    shifted = copy.deepcopy(r)
    shifted["original_source_full"][1][1] += 1e-9  # a different datum
    with pytest.raises(sc2.RecipeError, match="differs from the recorded"):
        sc2.original_topology(shifted)


def test_a_genuine_continued_checkpoint_restores_through_the_recipe_with_every_original_constructor_disabled(case, tmp_path, monkeypatch):
    ns, r = case
    ck = save(ns, tmp_path / "ck")
    ref = ns.state
    from maple_syrup import native_floating_soil as nfs
    from maple_syrup import probe

    def boom(*a, **k):
        raise AssertionError("an original bed/owner/probe constructor ran on the restore path")

    import fpond_builders as fb

    for mod, name in ((nfs, "initialize_native_floating_soil"), (nfs, "stage_native_refresh"), (probe, "build_minimal_state"), (fb, "make"), (fb, "make_pair"), (fb.ph, "make_bed")):  # BEFORE any recipe preparation
        monkeypatch.setattr(mod, name, boom)
    origin = cc.read_continued_origin(ck, IDENT, available_bytes_fn=AVAIL)
    orig_top = sc2.original_topology(r)
    ev = sc2.factory(r, "numpy")(_ctx(ns), cc.origin_topology(origin, orig_top))
    st = cc.load_floating_continued_checkpoint(ck, ev, IDENT, original_topology=orig_top, available_bytes_fn=AVAIL)
    assert st.topology.generation == ref.topology.generation >= 1 and st.counters.n_refreshes == 0 and ev.vertical._owner is None
    a, b = ref, st
    for _ in range(3):
        a = ns.event.step(a, 30.0 / 3.6e6, 0.1).state
        b = ev.step(b, 30.0 / 3.6e6, 0.1).state
    sa, sb = snapshot(ns.event, a, ns.occ.vertical), snapshot(ev, b, ev.vertical)
    assert set(sa) == set(sb) and all(sa[k].tobytes() == sb[k].tobytes() for k in sa if "window_cache" not in k)
    assert fc.CONTINUED_SCHEMAS


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("windowed", [False, True], ids=["dense", "windowed"])
def test_fresh_process_restore_from_recipe_h0_and_file_alone_with_the_constructors_disabled_first(tmp_path, backend, windowed):
    """The CPU analogue of the GPU fresh worker, in NEW interpreters: LOAD installs the original-constructor tripwires BEFORE reading the recipe and never builds a bed/owner/probe; bit-exact continuation vs the saving process."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    if backend == "numba":
        pytest.importorskip("numba")
    worker = Path(__file__).resolve().parent / "cpu_recipe_worker.py"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", NUMBA_NUM_THREADS="1")
    for mode in ("save", "load"):
        r = subprocess.run([sys.executable, "-B", str(worker), mode, str(tmp_path), backend, "1" if windowed else "0", "4"], capture_output=True, text=True, env=env, timeout=900, check=False)
        assert r.returncode == 0, r.stderr[-3000:]
    a, b = np.load(tmp_path / "save_final.npz"), np.load(tmp_path / "load_final.npz")
    assert set(a.files) == set(b.files) and len(a.files) > 40 and all(a[k].tobytes() == b[k].tobytes() for k in a.files if "window_cache" not in k)
    tw = json.loads((tmp_path / "tripwires.json").read_text())["disabled_before_recipe"]
    assert {"make", "make_pair", "make_bed", "build_minimal_state", "initialize_native_floating_soil", "stage_native_refresh", "_windowed_helpers"} == set(tw)


def test_the_cli_workers_explicit_path_list_covers_the_whole_import_closure_without_cupy(tmp_path):
    """A NEW interpreter that has ONLY the worker's own `sys.path` edits (not conftest) imports everything the SAVE and LOAD paths import (the static recipe, the accepted `static_case`, `refresh_fixtures`, `case`, `cont_builders`, the
    GPU definitions modules by path) -- and never imports CuPy."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    here = Path(__file__).resolve().parent
    code = (
        "import sys, importlib.util\n"
        "import gpu_continued_worker as w\n"
        "import cc2_static_case, static_case, refresh_fixtures, case, fpond_builders, test_event_refresh\n"
        "import cc2_fixtures, cont_builders, test_gpu_definitions as defs\n"
        "for name, rel in (('r', 'phase2_floating_event_refresh_cuda/test_gpu_definitions.py'), ('o', 'phase2_floating_checkpoint_cuda/test_gpu_definitions.py')):\n"
        "    defs._load('cc2_closure_' + name, rel)\n"
        "assert 'cupy' not in sys.modules, 'cupy was imported'\n"
        "print('closure ok')\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH=os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, env=env, cwd=str(here), timeout=300, check=False)
    assert r.returncode == 0 and "closure ok" in r.stdout, r.stderr[-3000:]
