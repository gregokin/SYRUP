"""CW-B109: the strict comparator / check functions of `b109_restart` on small synthetic records (host only; the
actual-output negative controls run as the helper's `controls` stage on real reference / pause / resume outputs)."""
import copy

import numpy as np

import b109_restart as h


def meta():
    return {"fields": {"n_reroutes": 1, "routing_source_t_s": None}, "deep_publication": None,
            "final_store": {"store_format_version": 2, "generation": 2}, "closure": {"closed": True},
            "roots": {"pid": 1}, "stage_wall_s": 1.0, "store_dir": "/a"}


def test_exact_metadata_and_named_qualifications():
    a, b = meta(), meta()
    b.update(roots={"pid": 2}, stage_wall_s=9.0, store_dir="/b", post_restart={"x": 1})
    assert h.compare_metadata(a, b) == []
    assert h.compare_metadata(a, {k: v for k, v in b.items() if k != "post_restart"})  # resume-only missing
    assert h.compare_metadata({**a, "post_restart": {}}, b)  # misplaced on the reference
    assert h.compare_metadata(a, {k: v for k, v in b.items() if k != "store_dir"})  # qualified field missing


def test_missing_key_never_equals_none_and_types_are_exact():
    a, b = meta(), meta()
    b["post_restart"] = {}
    del b["deep_publication"]
    assert any("deep_publication" in d for d in h.compare_metadata(a, b))
    b = meta(); b["post_restart"] = {}
    del b["fields"]["routing_source_t_s"]
    assert h.compare_metadata(a, b)
    b = meta(); b["post_restart"] = {}
    b["final_store"]["store_format_version"] = 2.0
    assert h.compare_metadata(a, b)
    b = meta(); b["post_restart"] = {}
    b["fields"]["n_reroutes"] = True  # bool is not int
    assert h.compare_metadata(a, b)
    b = meta(); b["post_restart"] = {}
    b["extra"] = None
    assert h.compare_metadata(a, b)
    assert h.compare_values([1, 2], [1, 2, 3]) and h.compare_values({"a": [1.0]}, {"a": [1.0]}) == []


def test_arrays_exact_dtype_shape_bytes_and_names():
    ref = {"x": np.arange(3.0), "y": np.zeros(2, dtype=np.int64)}
    assert h.compare_arrays(ref, {k: v.copy() for k, v in ref.items()}) == []
    bad = dict(ref); bad["x"] = ref["x"].copy(); bad["x"][1] = np.nextafter(1.0, 2.0)
    assert h.compare_arrays(ref, bad) == ["x"]
    assert h.compare_arrays(ref, {**ref, "y": ref["y"].astype(np.int32)}) == ["y"]
    assert h.compare_arrays(ref, {**ref, "x": ref["x"].reshape(3, 1)}) == ["x"]
    assert h.compare_arrays(ref, {"x": ref["x"]})
    assert h.compare_arrays(ref, {**ref, "z": ref["x"]})
    assert h.compare_arrays({"z": np.array([0.0])}, {"z": np.array([-0.0])}) == ["z"]  # signed zero is a difference


def pause_post():
    pause = {"facts": {"generation": 1, "version_id": 1, "physical_manifest_digest": "p" * 64, "mobile_kg": 0.5,
                       "n_accepted_steps": 1, "cumulative_pickup_kg": 1.0, "cumulative_deposition_kg": 1.0},
             "store_files": {"versions/old": "h", "chunks/old": "h"}}
    post = {"final_generation": 2, "final_version_id": 2, "final_store_format_version": 2, "final_reload_ok": True,
            "lineage": ["p" * 64], "new_store_files": ["versions/new", "chunks/new"],
            "referenced_data_files": ["chunks/new", "chunks/old"], "changed_or_missing_paused_files": [],
            "store_files_after": {"versions/old": "h", "chunks/old": "h", "versions/new": "n", "chunks/new": "n"},
            "after": {"n_accepted_steps": 5, "cumulative_pickup_kg": 2.0, "cumulative_deposition_kg": 2.0}}
    return pause, post


def test_post_restart_publication_checks_and_their_failures():
    pause, post = pause_post()
    assert h.check_post_restart_publication(pause, post) == []
    edits = [("final_generation", 1), ("final_version_id", 1), ("final_store_format_version", 1),
             ("final_reload_ok", False), ("lineage", []), ("lineage", ["q" * 64]), ("new_store_files", ["chunks/new"]),
             ("referenced_data_files", ["chunks/old"]), ("changed_or_missing_paused_files", ["chunks/old"]),
             ("store_files_after", {"versions/new": "n"})]
    for key, value in edits:
        p = copy.deepcopy(post); p[key] = value
        assert h.check_post_restart_publication(pause, p), key
    for key in ("cumulative_pickup_kg", "cumulative_deposition_kg", "n_accepted_steps"):
        p = copy.deepcopy(post); p["after"][key] = pause["facts"][key]
        assert h.check_post_restart_publication(pause, p), key
    dry = copy.deepcopy(pause); dry["facts"]["mobile_kg"] = 0.0
    assert h.check_post_restart_publication(dry, post)


def test_paused_binding_checks():
    arrays = {"routing_source_elevation_m": np.ones(3), "bed.x": np.zeros(2)}
    facts = {"physical_manifest_digest": "a" * 64, "logical_manifest_digest": "b" * 64, "generation": 1,
             "version_id": 1, "store_format_version": 2, "store_dir": "/s"}
    assert h.check_paused_binding(arrays, {k: v.copy() for k, v in arrays.items()}, facts, dict(facts)) == []
    held = dict(arrays); held["routing_source_elevation_m"] = np.array([1.0, np.nextafter(1.0, 2.0), 1.0])
    assert h.check_paused_binding(arrays, held, facts, facts)
    for key, value in (("generation", 2), ("version_id", 2), ("logical_manifest_digest", "c" * 64), ("store_dir", "/t")):
        assert h.check_paused_binding(arrays, arrays, facts, {**facts, key: value}), key
    assert h.check_paused_binding(arrays, arrays, facts, {k: v for k, v in facts.items() if k != "generation"})
