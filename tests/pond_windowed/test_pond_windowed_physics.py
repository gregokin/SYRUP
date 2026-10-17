"""Dense / windowed physical correspondence of the pond event on ACTUAL MAPLE beds (wet transport, dry splash, deep refill and
deposition flush, net-elevation geometry, completion). Strict policies: hydraulic 2e-12 / 1e-14, sediment 1e-10 / 1e-14.
Nothing here was run by its author (file-only tools); the root executes it."""
import numpy as np
import pytest
from pond_helpers import NC, basin, host
from windowed_helpers import (
    HYD,
    LOAD_KG,
    NV,
    SED,
    assert_physical_match,
    dense_deep_bed,
    heavy_mobile,
    logical_column,
    lower_cell,
    pair,
    windowed_counterpart,
)


def test_wet_transport_matches_dense_through_deep_columns(be, tmp_path):
    d_ev, d0, w_ev, w0 = pair(be, tmp_path, mobile=heavy_mobile())
    column = w0.bed.voxel_column
    assert column.n_voxels == NV and column.window_kg.shape[2] < NV and (host(column.base) > 0).all()
    d = d_ev.evolve(d0, 2.0).state
    r = w_ev.evolve(w0, 2.0)
    w = r.state
    assert_physical_match(d, w)
    assert w.counters.max_carried_records > 0, "the heavy load must overflow the window into the deep store"
    assert w.bed.voxel_column.deep_store.version_id > w0.bed.voxel_column.deep_store.version_id
    assert w.bed.voxel_column.pending_deep is None  # 'step' policy: every accepted step published
    audit = w_ev.audit(w)
    assert audit["water"]["closes"] and audit["sediment"]["closes"], audit["sediment"]
    assert audit["storage"]["pending_journal_records"] == 0
    # wet detachment / transport / export actually happened and agrees with the dense run
    assert host(w.sediment.wet_actual).sum() > 0.0
    np.testing.assert_allclose(host(w.sediment.export_wet), host(d.sediment.export_wet), **SED)


def test_dry_splash_with_finite_supply_matches_dense(be, tmp_path):
    available = np.ones((3, 4, NC))
    available[:, 0, :] = 0.0  # the west column has nothing available: requested > actual (finite supply / unmet)
    d_ev, d0, w_ev, w0 = pair(be, tmp_path, depth=0.0, rain_mm_h=60.0, splash=True, ksat=1e-3, available=available,
                              mobile=heavy_mobile(cells=((1, 2),)))
    d = d_ev.evolve(d0, 1.0).state
    w = w_ev.evolve(w0, 1.0).state
    assert_physical_match(d, w)
    assert w.counters.n_splash_exchanges > 0
    requested, actual = host(w.sediment.splash_requested).sum(), host(w.sediment.splash_actual).sum()
    assert requested > actual > 0.0
    for name in ("x_gross", "y_gross", "x_net", "y_net", "pickup_cell", "deposition_cell"):  # all face / cell accounting
        np.testing.assert_allclose(host(getattr(w.sediment, name)), host(getattr(d.sediment, name)), err_msg=name, **SED)
    # the pre-existing mobile pool of the heavy cell is unchanged by the splash exchange (no wet law acts on dry cells)
    assert w_ev.audit(w)["sediment"]["closes"]


def test_windowed_splash_exchange_defines_landed_mass_and_matches_dense(cpu, tmp_path):
    """Direct regression of the windowed `apply_splash_exchange` branch (the common tail reads `landed`)."""
    from maple_syrup.splash import apply_splash_exchange

    d_ev, d0, w_ev, w0 = pair(cpu, tmp_path, depth=0.0, rain_mm_h=60.0, splash=True, ksat=1e-3,
                              mobile=heavy_mobile(cells=((0, 0),)))
    request = np.zeros((3, 4, NC))
    request[1, 1, :] = 1e-6
    request[1, 2, :] = 2e-6
    out = []
    for ev, st in ((d_ev, d0), (w_ev, w0)):
        rt = ev._check_state(st)
        out.append(apply_splash_exchange(st.bed, ev.context, ev.splash, rt.splash_grid, ev.xp.asarray(request),
                                         implementation="array"))
    dense, windowed = out
    np.testing.assert_allclose(host(windowed.bed.water.mobile_mass_by_cell_class_kg),
                               host(dense.bed.water.mobile_mass_by_cell_class_kg), **SED)
    assert windowed.audit.net_bed_change_by_class_kg is None  # PickupView has no bed change: unavailable, not invented
    totals_d, totals_w = dense.audit.scalar_totals(), windowed.audit.scalar_totals()
    for name, value in totals_w.items():
        assert value == pytest.approx(totals_d[name], rel=1e-9, abs=1e-14), name
    np.testing.assert_allclose(host(windowed.audit.mobile_change_by_cell_class_kg),
                               host(dense.audit.mobile_change_by_cell_class_kg), **SED)


def test_deep_refill_deposition_flush_and_net_elevation_refresh(cpu, tmp_path):
    cell = (1, 0)
    d_ev, d0, w_ev, w0 = pair(cpu, tmp_path, prepare=lower_cell(cell, 0.0095))
    base_after_pickup = host(w0.bed.voxel_column.base)[cell]
    (tmp_path / "ref").mkdir()
    _, _, _, w_ref = pair(cpu, tmp_path / "ref")
    assert base_after_pickup < host(w_ref.bed.voxel_column.base)[cell]  # the window was refilled from the deep store
    d = d_ev.evolve(d0, 2.0).state
    w = w_ev.evolve(w0, 2.0).state
    assert_physical_match(d, w)
    for state in (d, w):  # a measured net 9.5 mm change crosses the 0.75 mm threshold: a genuine net refresh. MAPLE's own
        # trigger may commit first, so only the refresh (the actual-surface gate) is essential, not the commit reason.
        assert state.counters.n_refreshes >= 1 and state.topology.generation == state.counters.n_refreshes
        assert state.counters.n_commits >= 1
    np.testing.assert_allclose(host(w.held_source_m), host(d.held_source_m), **HYD)
    assert w_ev.audit(w)["sediment"]["closes"]


def test_layered_columns_are_heterogeneous():
    bed, _ = dense_deep_bed(np.full((3, 4), 0.2))
    mass = host(bed.voxel_column.mass_kg)
    low, high = mass[1, 1, 10], mass[1, 1, 190]
    assert not np.allclose(low / low.sum(), high / high.sum())  # lower and upper strata differ in grain classes


def test_large_terminal_deposition_spills_to_the_deep_store_with_a_truthful_receipt(cpu, tmp_path):
    """3 kg of real mobile load settles at flow end: 9.6 mm net change (a genuine refresh) and real deep-store spills. The
    returned bed is published; the final audit's storage block equals a fresh audit of the returned state; the completed
    handoff survives a checkpoint round trip."""
    from pond_helpers import make_event, make_topology
    from windowed_helpers import control_for, physical_bytes, store_identity

    from maple_syrup.pond_checkpoint import load_pond_checkpoint, save_pond_checkpoint
    from maple_syrup.pond_event import PondCompletionPolicy

    mobile = np.zeros((3, 5, NC))
    mobile[1, 2, :] = 0.5  # 3 kg
    dense, ctx = dense_deep_bed(basin(), depth=np.maximum(0.215 - basin(), 0.0), mobile=mobile)
    wbed, wctx = windowed_counterpart(dense, ctx, tmp_path / "store")
    topo = make_topology(dense, [], allow_closed=True)
    policy = PondCompletionPolicy(hold_s=1.0, deposit_mobile_at_flow_end=True)

    def build(bed, c, windowed):
        return make_event(bed, c, topo, policy=policy, control=control_for(
            cpu, windowed=windowed, deep_publication="boundary" if windowed else "step", high_water=3 if windowed else None))

    d_ev, w_ev = build(dense, ctx, False), build(wbed, wctx, True)
    zeros = np.zeros(w_ev.shape)
    w0 = w_ev.initial_state(wbed, zeros)
    version0 = store_identity(w0)[0]
    d = d_ev.complete(d_ev.initial_state(dense, zeros), 20.0)
    w = w_ev.complete(w0, 20.0)
    dry, acc = w.dry_state, w.accounting
    assert sum(acc["terminal_deposition_by_class_kg"]) == pytest.approx(3.0, abs=1e-8)
    np.testing.assert_allclose(acc["terminal_deposition_by_class_kg"], d.accounting["terminal_deposition_by_class_kg"], **SED)
    assert dry.counters.n_refreshes == d.dry_state.counters.n_refreshes == 1 and acc["geometry_refreshed_at_final"]
    assert acc["final_commit"]["net_after_m"] >= 0.009  # ~9.6 mm
    assert dry.bed.voxel_column.pending_deep is None and store_identity(dry)[0] > version0  # real spills, published
    assert dry.counters.n_storage_publications >= 1
    storage = acc["final_audit"]["storage"]
    assert storage["max_carried_records"] > 0  # terminal exchange created a genuine deep journal
    assert storage == w_ev.audit(dry)["storage"]  # the receipt describes the RETURNED (published) state
    assert storage["pending_journal_records"] == 0 and storage["deep_store_version_id"] == store_identity(dry)[0]
    assert acc["final_audit"]["water"]["closes"] and acc["final_audit"]["sediment"]["closes"]
    np.testing.assert_allclose(logical_column(dry.bed), host(d.dry_state.bed.voxel_column.mass_kg), **SED)
    identity = {"case": "controlled actual MAPLE pond bed", "forcing": "closed lake", "source": "test", "options": {}}
    path = save_pond_checkpoint(tmp_path / "ck", w, w_ev, identity)
    loaded = load_pond_checkpoint(path, build(wbed, wctx, True), identity, allow_complete=True)
    assert physical_bytes(loaded.dry_state) == physical_bytes(dry)


@pytest.mark.parametrize("load_kg,net_refresh", [(0.1, False), (LOAD_KG, True)])
def test_completion_settles_terminal_deposition_and_resets(cpu, tmp_path, load_kg, net_refresh):
    """Terminal deposition of `load_kg` raises the cell by load / (rho dx dy) = load / 312.5 m: 0.32 mm (below the 0.75 mm
    threshold: a plain forced final commit, no refresh) or 0.96 mm (a genuine net-elevation refresh at the final commit)."""
    from pond_helpers import make_event, make_topology
    from windowed_helpers import control_for

    from maple_syrup.pond_event import PondCompletionPolicy

    z = basin()
    mobile = np.zeros((3, 5, NC))
    mobile[1, 2, 2] = load_kg
    eta = 0.215
    dense, ctx = dense_deep_bed(z, depth=np.maximum(eta - z, 0.0), mobile=mobile)
    wbed, wctx = windowed_counterpart(dense, ctx, tmp_path / "store")
    topo = make_topology(dense, [], allow_closed=True)
    policy = PondCompletionPolicy(hold_s=1.0, deposit_mobile_at_flow_end=True)
    handoffs = {}
    for name, bed, c, windowed in (("dense", dense, ctx, False), ("windowed", wbed, wctx, True)):
        event = make_event(bed, c, topo, policy=policy, control=control_for(cpu, windowed=windowed,
                           deep_publication="boundary" if windowed else "step", high_water=3 if windowed else None))
        handoffs[name] = event.complete(event.initial_state(bed, np.zeros(event.shape)), 20.0)
    d, w = handoffs["dense"], handoffs["windowed"]
    assert w.status == "complete" and w.dry_state.status == "dry"
    assert w.dry_state.bed.voxel_column.pending_deep is None  # no unpersisted deep state in a completed handoff
    assert not np.any(host(w.dry_state.hydraulic.depth_m)) and not np.any(host(w.dry_state.hydraulic.soil_water_m))
    assert not np.any(host(w.dry_state.bed.water.mobile_mass_by_cell_class_kg))
    acc_d, acc_w = d.accounting, w.accounting
    assert acc_w["terminal_deposition_by_class_kg"][2] == pytest.approx(load_kg, abs=1e-9)
    np.testing.assert_allclose(acc_w["terminal_deposition_by_class_kg"], acc_d["terminal_deposition_by_class_kg"], **SED)
    assert acc_w["surface_removed_m3"] == pytest.approx(acc_d["surface_removed_m3"], rel=1e-12)
    assert acc_w["final_audit"]["water"]["closes"] and acc_w["final_audit"]["sediment"]["closes"]
    cd, cw = d.dry_state.counters, w.dry_state.counters
    for name in ("n_commits", "n_final_commits", "n_net_elevation_commits", "n_maple_trigger_commits", "n_refreshes"):
        assert getattr(cw, name) == getattr(cd, name), name  # the windowed bookkeeping equals the dense one
    assert acc_w["geometry_refreshed_at_final"] == acc_d["geometry_refreshed_at_final"] == net_refresh
    assert cw.n_refreshes == int(net_refresh) and w.dry_state.topology.generation == int(net_refresh)
    reason = acc_w["final_commit"]["reason"]
    if net_refresh:  # the actual-surface gate, not the commit reason, decides the refresh
        assert reason in ("net_elevation", "maple_trigger") and acc_w["final_commit"]["net_after_m"] >= 0.00075
    else:  # no refresh; a genuine MAPLE trigger may already have made the accounting commit, so the reason is not fixed
        from maple.surface.topographic_commit.commit import ledger_is_empty

        assert acc_w["final_commit"]["committed"] and reason in ("final_forced", "maple_trigger")
        assert cw.n_commits >= 1 and ledger_is_empty(w.dry_state.bed.ledger)
        assert acc_w["final_commit"]["net_after_m"] < 0.00075
    np.testing.assert_allclose(host(w.dry_state.bed.active_layer.mass_kg), host(d.dry_state.bed.active_layer.mass_kg),
                               **SED)
