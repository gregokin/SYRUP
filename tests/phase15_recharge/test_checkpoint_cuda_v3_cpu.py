"""Phase 1.5 S4: CPU-side controls of the CUDA V3 daily codec (NO GPU, NO CuPy import): the registered policy and its helper-identity set, the old-policy identities/pins, the pure V3
planners, the typed refusals that precede any device work, and the expected-plan/counter-bound validation shared with the CPU codec. The genuine device save/load/continuation controls are
`test_checkpoint_cuda_v3.py` (written, UNRUN until Root's guarded GPU run). Needs SYRUP_NCP_TARGET (CPU 9f bundle) like the file-case lane: skip only if unset, FAIL if configured-unusable."""
from __future__ import annotations

import copy
import dataclasses
import pickle
from pathlib import Path
from types import SimpleNamespace

import ncpcases as nc
import pytest
import s4_worker as w
from dcases import RUN, rc

from maple_syrup import layered_checkpoint as lc
from maple_syrup import lifecycle_checkpoint as lk
from maple_syrup import lifecycle_checkpoint_cuda as cs
from maple_syrup import lifecycle_checkpoint_cuda_load_plan as lp
from maple_syrup import lifecycle_checkpoint_cuda_plan as sp
from maple_syrup import lifecycle_checkpoint_cuda_policy as pol_mod
from maple_syrup import lifecycle_checkpoint_cuda_v2_plan as v2plan
from maple_syrup import lifecycle_checkpoint_cuda_v3 as v3
from maple_syrup import lifecycle_checkpoint_cuda_v3_plan as v3plan
from maple_syrup import provenance as pv
from maple_syrup import water_lifecycle_cuda as wlc

pytestmark = [pytest.mark.skipif(nc.skip_reason() is not None, reason=nc.skip_reason() or "")]
LIMITS = lc.CheckpointLimits(1 << 30, 1 << 30, 1 << 30, 1 << 30)
SRC = Path(lk.__file__).resolve().parent


@pytest.fixture(autouse=True)
def _configured_bundle_or_fail():
    if nc.skip_reason() is None:
        nc.require_configured_bundle()


# --- policy / pins -------------------------------------------------------------------------------------------------------------------
def test_three_registered_literals_v3_is_daily_and_partial_and_v1_v2_keep_their_meaning():
    P = pol_mod
    assert pol_mod.policy_of(None) is P.V1 and pol_mod.policy_of(P.V2) is P.V2 and pol_mod.policy_of(P.V3) is P.V3
    assert P._REGISTERED == (P.V1, P.V2, P.V3)
    assert (P.V1.daily, P.V1.partial_storm, P.V1.helper_pins) == (False, False, None) and (P.V2.daily, P.V2.partial_storm, P.V2.helper_pins) == (False, False, None)
    assert P.V2.storm_allowed and P.V2.roles == ("boundary", "interevent", "storm") and not P.V1.storm_allowed  # V2 stays completed-storm-only (its engine checks `completed`)
    assert (P.V3.daily, P.V3.partial_storm, P.V3.storm_allowed) == (True, True, True)
    assert P.V3.schema.endswith("/3") and P.V3.profile.endswith("_cuda_v3") and P.V3.manifest_name == "lifecycle_checkpoint_cuda_v3.json"
    assert len({P.V1.schema, P.V2.schema, P.V3.schema}) == 3 and len({P.V1.manifest_name, P.V2.manifest_name, P.V3.manifest_name}) == 3
    for pairs in (P.V3.scalar_parts.values(),):
        assert all("daily" in parts for parts in pairs)
    assert all("daily" not in parts for parts in P.V2.scalar_parts.values())
    for bad in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(P.CodecPolicyError):
            bad(P.V3)
    fields = {k: getattr(P.V3, k) for k in P.V3.__slots__}
    for forged in (SimpleNamespace(**fields), "v3", object()):
        with pytest.raises(P.CodecPolicyError):
            pol_mod.policy_of(forged)
    with pytest.raises(P.CodecPolicyError):
        P._CodecPolicy(object(), **fields)  # only the module's own seal constructs a policy
    with pytest.raises(AttributeError):
        P.V3.daily = False


def test_c1_helper_pins_of_v1_v2_are_unchanged_and_v3_has_its_own_reviewed_set_equal_to_the_files_on_disk():
    assert cs.HELPER_MODULE_PINS["lifecycle_checkpoint.py"] == "ff4d228e9464a29333a544e3ca098577df492d02661b9faaa7c7278c98e6d085"  # the C1 literal, never overwritten
    assert cs.HELPER_MODULE_PINS == {"layered_checkpoint.py": "4f4336d98f40febb56e15bcdb5a4c06e2e1dc1187c330d08b79849c4c0d72d51",
                                     "pond_checkpoint.py": "94cd2339566646ba44f2015dad13a67194c26a8dd3cc85b56916f62d2b4a704f",
                                     "lifecycle_checkpoint.py": "ff4d228e9464a29333a544e3ca098577df492d02661b9faaa7c7278c98e6d085",
                                     "checkpoint_capability.py": "36dc5d6a07245fca4badbadd0d754f2a250e688c2cc8b0b30d3a8c37c55ccbd6",
                                     "layered_storm.py": "bd125b743e58c588619a86ee952934deccea360ef3af215ae2b230af60d73c78"}
    pins = dict(pol_mod.V3.helper_pins)
    assert set(pins) == {"layered_checkpoint.py", "pond_checkpoint.py", "lifecycle_checkpoint.py", "checkpoint_capability.py", "layered_storm.py", "recharge.py", "water_lifecycle.py"}
    files = dict(pv.source_tree_digest(SRC).files)
    assert {name: files[name] for name in pins} == pins  # the FINAL reviewed helper SHAs equal the V3 literal exactly (re-run s4_repin.py after any helper edit)
    assert pins["lifecycle_checkpoint.py"] != cs.HELPER_MODULE_PINS["lifecycle_checkpoint.py"]  # the CPU helper changed (daily engine): V1/V2 archives stay tied to the C1 sources


def test_fresh_source_pair_v3_accepts_the_current_sources_and_v1_v2_refuse_the_changed_helper(monkeypatch):
    pair = cs._fresh_source_pair(pol_mod.V3)
    assert pair["maple_package_sha256"] == cs.NATIVE9F_PACKAGE_SHA256 and pair["maple_syrup_n_files"] > 0
    for old in (pol_mod.V1, pol_mod.V2):  # not silently rebound to the new source
        with pytest.raises(cs.LifecycleCheckpointCudaError, match="differs from its pinned digest"):
            cs._fresh_source_pair(old)
    real = pv.source_tree_digest

    def altered(root):
        t = real(root)
        if Path(root).name == "maple_syrup":
            files = dict(t.files)
            files["recharge.py"] = "0" * 64
            return SimpleNamespace(files=files, digest_sha256=t.digest_sha256, file_count=t.file_count, algorithm=t.algorithm)
        return t

    monkeypatch.setattr(pv, "source_tree_digest", altered)  # INJECTION control of the pin check (a changed pinned helper must be refused)
    with pytest.raises(cs.LifecycleCheckpointCudaError, match="recharge.py differs from its pinned digest"):
        cs._fresh_source_pair(pol_mod.V3)


# --- pure planners -----------------------------------------------------------------------------------------------------------------------
def meta_for(run, role, kind=None):
    c = run.c
    bed = c.s0.payload.bed
    return v2plan.extract_save_metadata_v2(bed=bed, context_fraction=run.lop.context.initial_available_fraction, role=role, source_kind=kind, n_cells=c.n, n_layers=c.L,
                                    n_classes=run.lop.n_classes, shape=tuple(run.lop.shape), wrapper_retained_host_bytes=wlc.wrapper_retained_host_bytes(c.L),
                                    owner_audit_device_bytes=int(run.lop.memory.audit_scratch_bytes))


@pytest.mark.parametrize("role,kind", [("boundary", None), ("interevent", "storm_end"), ("interevent", "boundary_origin"), ("storm", None)])
def test_v3_save_plan_is_the_v2_plan_plus_exactly_the_derived_daily_allowance(role, kind, tmp_path):
    run = w.Run(tmp_path, "storm")
    meta = meta_for(run, role, kind)
    storm = v2plan.StormSaveTerms() if role == "storm" else None
    b2 = v2plan.plan_save_v2(meta, caller_host_reserve_bytes=1 << 20, caller_device_reserve_bytes=1 << 20, extra_reserve_bytes=7, storm=storm)
    b3 = v3plan.plan_save_v3(meta, caller_host_reserve_bytes=1 << 20, caller_device_reserve_bytes=1 << 20, extra_reserve_bytes=7, storm=storm)
    x = v3plan.PARTIAL_FLAG_BYTES if role == "storm" else 0  # the V3-only partial-storm flag delta (storm role only)
    assert v3plan.PARTIAL_FLAG_BYTES == 1
    delta = 3 * v3plan.DAILY_RECORD_BYTES + v3plan.DAILY_NODES * lc.NODE_BYTES + 3 * x
    assert b3.profile == "native9f_dense_lifecycle_cuda_v3" and b3.manifest_cap_bytes == b2.manifest_cap_bytes + v3plan.DAILY_RECORD_BYTES + x
    assert b3.manifest_max_nodes == b2.manifest_max_nodes + v3plan.DAILY_NODES and b3.metadata_bytes == b2.metadata_bytes + delta
    assert b3.serialized_bytes == b2.serialized_bytes + v3plan.DAILY_RECORD_BYTES + x and b3.disk_bytes == b2.disk_bytes + v3plan.DAILY_RECORD_BYTES + x
    assert b3.scratch_bytes == b2.scratch_bytes + delta
    assert b3.device_save_bytes == b2.device_save_bytes and b3.device_required_with_reserves == b2.device_required_with_reserves  # host-only metadata term: device figures unchanged
    assert b3.host_required_with_reserves > b2.host_required_with_reserves
    for stage, req in b2.stage_requirements.items():
        assert b3.stage_requirements[stage]["host_with_reserves"] == req["host_with_reserves"] + delta
    assert v3plan.DAILY_RECORD_BYTES == lk._DAILY_BYTES and v3plan.DAILY_NODES == lk._DAILY_NODES  # one derivation shared with the CPU codec


def test_daily_allowance_derivation_uses_the_declared_counter_bound_and_encode_decode_refuse_beyond_it():
    big = rc.MAX_RECORD_COUNT
    assert big == 2 ** 63 - 1
    plan = rc.plan_daily_rain(1.0, 10.0, 5.0, run_digest=RUN, ordinal=big)  # max accepted
    assert plan.ordinal == big
    with pytest.raises(rc.RechargeError):
        rc.plan_daily_rain(1.0, 10.0, 5.0, run_digest=RUN, ordinal=big + 1)  # max + 1 refused
    worst = rc.DailyCursor(RUN, big, rc.PendingStorm(big, "ab" * 32, "storm_fallback", "cd" * 32, "ef" * 32, big), "12" * 32, "storm_fallback", "34" * 32)
    rec = rc.cursor_record(worst)  # encodes at the bound ...
    import json
    from maple_syrup import checkpoint_capability as cc

    text = json.dumps({"daily": cc.encode_tree(rec)}, indent=1, sort_keys=True)
    assert len(text) + 3 * text.count("\n") + len('"daily": ') + 2 <= lk._DAILY_BYTES  # ... and its ACTUAL encoded size fits the derived allowance
    assert rc.cursor_from_record(rec).digest() == worst.digest()
    for key, value in (("ordinal", big + 1), ("ordinal", 10 ** 4000)):
        with pytest.raises(rc.RechargeError):
            rc.cursor_from_record({**rec, key: value})  # decode refuses beyond the bound
    pend = dict(rec["pending"])
    for key in ("ordinal", "capacity_failing_columns"):
        with pytest.raises(rc.RechargeError):
            rc.cursor_from_record({**rec, "pending": {**pend, key: big + 1}})
    with pytest.raises(rc.RechargeError):
        rc.PendingStorm(big + 1, "ab" * 32, "storm_fallback", "cd" * 32, "ef" * 32, 1)
    with pytest.raises(rc.RechargeError):
        rc.DailyCursor(RUN, big).consume(rc.plan_daily_rain(1.0, 10.0, 5.0, run_digest=RUN, ordinal=big), "recharge", "56" * 32)  # the NEXT ordinal would exceed the bound


# --- typed refusals before any device work / expected-plan validation ------------------------------------------------------------------
def test_cuda_v3_entry_points_refuse_a_cpu_lifecycle_and_a_cursor_free_state_before_any_io(tmp_path):
    run = w.Run(tmp_path / "r", "storm")
    run.step_start()
    kw = dict(limits=LIMITS, caller_host_reserve_bytes=0, caller_device_reserve_bytes=0)
    target = tmp_path / "ck"
    with pytest.raises(cs.LifecycleCheckpointCudaError, match="CUDA lifecycle operator"):
        v3.save_lifecycle_checkpoint_cuda_v3(target, run.lop, run.state, w.IDENT, **kw)  # a daily CPU lifecycle is not a CUDA lifecycle
    with pytest.raises(cs.LifecycleCheckpointCudaError, match="no cursor"):
        v3.save_lifecycle_checkpoint_cuda_v3(target, run.lop, run.c.s0, w.IDENT, **kw)  # a cursor-free state
    assert not target.exists()
    with pytest.raises(cs.LifecycleCheckpointCudaError, match="daily"):  # V1 and V2 CUDA writers still refuse a cursor state before any I/O
        cs.save_lifecycle_checkpoint_cuda(target, run.lop, run.state, w.IDENT, **kw)
    from maple_syrup import lifecycle_checkpoint_cuda_v2 as v2

    with pytest.raises(cs.LifecycleCheckpointCudaError, match="daily"):
        v2.save_lifecycle_checkpoint_cuda_v2(target, run.lop, run.state, w.IDENT, **kw)
    assert not target.exists()


@pytest.mark.parametrize("bad", ["plan", SimpleNamespace(digest="0" * 64), 3, object()])
def test_every_wrapper_revalidates_a_supplied_expected_plan_before_any_work(bad, tmp_path):
    run = w.Run(tmp_path / "r", "storm")
    run.step_start()
    with pytest.raises(lk.LifecycleCheckpointError, match="expected_plan"):
        lk._daily_cursor_checked(lk._V3_9F, "boundary", {"daily": None}, None, bad)  # the shared helper types the refusal (no raw AttributeError, no digest stand-in accepted)
    with pytest.raises(lk.LifecycleCheckpointError, match="expected_plan"):
        from maple_syrup import lifecycle_checkpoint_v3 as cpu_v3

        cpu_v3.load_lifecycle_checkpoint_v3(tmp_path / "none", run.lop, w.IDENT, expected_role="boundary", limits=LIMITS, host_available_bytes_fn=lambda: 1 << 40,
                                            reference_bed=run.c.s0.payload.bed, expected_plan=bad)
    with pytest.raises(cs.LifecycleCheckpointCudaError, match="expected_plan"):
        v3.load_lifecycle_checkpoint_cuda_v3(tmp_path / "none", run.lop, w.IDENT, expected_role="boundary", limits=LIMITS, reference_bed=None, caller_host_reserve_bytes=0,
                                             caller_device_reserve_bytes=0, expected_plan=bad)


def test_dataclass_forged_canonical_plans_are_refused_on_both_codecs(tmp_path):
    run = w.Run(tmp_path / "r", "storm")
    plan = rc.plan_daily_rain(*w.P_STORM, run_digest=RUN, ordinal=0)
    forged = [dataclasses.replace(plan, depth_m=0.5), dataclasses.replace(plan, selection="recharge"), dataclasses.replace(plan, digest="ee" * 32),
              dataclasses.replace(plan, ordinal=1)]
    for bad in forged:
        with pytest.raises(lk.LifecycleCheckpointError, match="expected_plan"):
            lk._validate_expected_plan(bad)
        with pytest.raises(cs.LifecycleCheckpointCudaError, match="expected_plan"):
            v3.load_lifecycle_checkpoint_cuda_v3(tmp_path / "none", run.lop, w.IDENT, expected_role="boundary", limits=LIMITS, reference_bed=None, caller_host_reserve_bytes=0,
                                                 caller_device_reserve_bytes=0, expected_plan=bad)
    lk._validate_expected_plan(None)  # the documented optional case
    lk._validate_expected_plan(plan)  # a genuine canonical plan


def test_load_engine_rejects_expected_plan_for_the_old_codecs():
    from maple_syrup import lifecycle_checkpoint_cuda_v2 as v2  # noqa: F401
    from maple_syrup import lifecycle_checkpoint_cuda_load as ld

    plan = rc.plan_daily_rain(*w.P_STORM, run_digest=RUN, ordinal=0)
    for pol in (pol_mod.V1, pol_mod.V2):
        with pytest.raises(cs.LifecycleCheckpointCudaError, match="V3 daily codec"):
            ld._load_engine("/nonexistent", None, {}, expected_role="boundary", limits=LIMITS, reference_bed=None, caller_host_reserve_bytes=0, caller_device_reserve_bytes=0,
                            _policy=pol, _expected_plan=plan)


def test_cpu_policy_registry_has_exactly_four_literals_and_the_old_three_keep_their_identities_pins_and_immutability():
    """Focused replacement for the two historical assertions of the former three-policy tuple (obsolete by design once V3 exists): V1/V2/V2-9F identities, pins, closed
    parts and immutability are unchanged and V3 is the fourth registered literal selected by identity only."""
    assert lk._POLICIES == (lk._V1, lk._V2, lk._V2_9F, lk._V3_9F) and lk._policy_of(None) is lk._V1
    expect = {"_V1": ("v1", "maple-syrup-water-lifecycle-checkpoint/1", False), "_V2": ("v2", "maple-syrup-water-lifecycle-checkpoint/2", False),
              "_V2_9F": ("v2-native9f", "maple-syrup-water-lifecycle-checkpoint/2", False), "_V3_9F": ("v3-native9f", "maple-syrup-water-lifecycle-checkpoint/3", True)}
    for attr, (name, schema, daily) in expect.items():
        pol = getattr(lk, attr)
        assert (pol.name, pol.schema, pol.daily) == (name, schema, daily)
        assert all(("daily" in parts) == daily for parts in pol.scalar_parts.values())
        for field in ("schema", "files", "package_pins", "v2", "daily"):
            with pytest.raises(AttributeError):
                setattr(pol, field, 1)
            with pytest.raises(AttributeError):
                delattr(pol, field)
    assert lk._V2_9F.package_pins == {lk.NATIVE9F_PROFILE_V2: lk.NATIVE9F_PACKAGE_SHA256} and lk._V3_9F.package_pins == {lk.NATIVE9F_PROFILE_V3: lk.NATIVE9F_PACKAGE_SHA256}
    assert lk._V1.package_pins == {lk.NATIVE_PROFILE: lk.NATIVE39_PACKAGE_SHA256} and lk._V2.package_pins == {lk.NATIVE_PROFILE_V2: lk.NATIVE39_PACKAGE_SHA256}
    fields = {k: getattr(lk._V3_9F, k) for k in lk._Policy.__slots__}
    for forged in (SimpleNamespace(**fields), "v3", object(), 3):
        with pytest.raises(lk.LifecycleCheckpointError):
            lk._policy_of(forged)
    with pytest.raises(lk.LifecycleCheckpointError):
        lk._Policy(object(), **fields)


def test_partial_storm_record_is_exactly_one_byte_larger_than_the_v2_derivation_and_v3_accounts_for_it():
    """`false` (5 chars) vs the V2 worst case's `true` (4 chars): the V2-derived storm term does not cover a PARTIAL storm record by itself; V3 adds the explicit V3-only delta. Also the
    ACTUAL worst-case partial record text fits the V3 storm manifest figure (V2 storm envelope + the delta), with V2 meaning/caps unchanged."""
    import json

    from maple_syrup import checkpoint_capability as cc

    done = v2plan.storm_record_worst_case()
    partial = {**done, "completed": False}
    cost = lambda rec: v2plan._text_cost(cc.encode_tree(rec), '"storm": ')  # noqa: E731
    (b_true, n_true), (b_false, n_false) = cost(done), cost(partial)
    assert b_false == b_true + 1 and n_false == n_true  # exactly ONE byte, no extra node
    assert json.dumps(cc.encode_tree(partial), indent=1, sort_keys=True).count("false") == 1
    assert v3plan.PARTIAL_FLAG_BYTES == b_false - b_true
    assert v2plan.ENVELOPE_STORM[0] >= b_true  # the V2 term was derived from the `true` record ...
    assert v2plan.ENVELOPE_STORM[0] + v3plan.PARTIAL_FLAG_BYTES >= b_false  # ... and V2 + the V3 delta covers the partial record (the combined derived bound)
    assert pol_mod.V2.helper_pins is None and v2plan.PROFILE_V2.endswith("_v2")  # V2 meaning unchanged
