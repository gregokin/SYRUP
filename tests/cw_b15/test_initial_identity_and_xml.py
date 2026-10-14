"""CW-B15 items 3 and 4 on a genuine compiled synthetic Chastre case (actual MAPLE row tiles, CPU).

Item 3: the persisted initial operands are bound into the run identity on the fresh AND the reuse path; files are
read-only; hash-before-and-after detects edits during a restore; a coherent, fully re-hashed edit of the resident
water changes the identity (so a checkpoint of the original run would not resume under it); the receipt datum must
equal the verified case datum before anything is loaded. Item 4: the bound XML and the pinned original RFID report
are checked BEFORE parameter evaluation, assembly or upload; the report is read once."""
from __future__ import annotations

import hashlib
import json
import stat

import numpy as np
import pytest

pytest.importorskip("maple")

import maple_syrup.windowed_experiment as we
from maple_syrup.sediment_bed import BedIntegrationError
from maple_syrup.sediment_event import SedimentEventControl, SedimentEventError
from maple_syrup.windowed_case import (
    INITIAL_BINDING_NAME,
    INITIAL_FILES,
    assemble_windowed_case,
    chastre_row_tiles,
    initial_binding_identity,
    initial_identity_record,
    persist_initial_assembly,
    restore_initial_assembly,
)


@pytest.fixture(scope="module")
def verified(chastre_case_dir):
    from maple_syrup.chastre_case import verify_chastre_case

    return verify_chastre_case(chastre_case_dir, allow_maple_source_change=True)


def assemble(verified, store):
    grid = verified.binding["grid"]
    return assemble_windowed_case(chastre_row_tiles(verified), verified.case.config.geometry,
                                  verified.case.config.grain_classes, grid["nz"], store_dir=store,
                                  datum_offset_m=grid["datum_offset_m"], max_store_bytes=100_000_000,
                                  max_host_bytes=2_000_000_000)


def restore(verified, store, **kw):
    options = {"store_dir": store, "max_host_bytes": 2_000_000_000,
               "datum_offset_m": float(verified.binding["grid"]["datum_offset_m"])}
    options.update(kw)
    return restore_initial_assembly(chastre_row_tiles(verified), verified.case.config.geometry,
                                    verified.case.config.grain_classes, **options)


def writable(path):
    path.chmod(path.stat().st_mode | stat.S_IWUSR)


def rewrite_binding(store, name):
    """An adversary's coherent re-hash: the binding's digest of `name` is updated to the edited file."""
    path = store / INITIAL_BINDING_NAME
    binding = json.loads(path.read_text())
    binding["files"][name] = hashlib.sha256((store / name).read_bytes()).hexdigest()
    writable(path)
    path.write_text(json.dumps(binding, sort_keys=True) + "\n")


@pytest.fixture()
def store(verified, tmp_path):
    path = tmp_path / "store"
    identity = persist_initial_assembly(assemble(verified, path))
    return path, identity


def test_persist_returns_the_content_identity_and_makes_the_files_read_only(store):
    path, identity = store
    for name in (*INITIAL_FILES, INITIAL_BINDING_NAME):
        assert stat.S_IMODE((path / name).stat().st_mode) == 0o444, name
    assert identity["initial_binding_sha256"] == hashlib.sha256((path / INITIAL_BINDING_NAME).read_bytes()).hexdigest()
    assert identity["initial_files_sha256"] == {n: hashlib.sha256((path / n).read_bytes()).hexdigest() for n in INITIAL_FILES}


def test_restore_reproduces_the_fresh_identity_and_arrays(verified, store):
    from maple.core.backend import array_leaves

    path, identity = store
    again = restore(verified, path)
    assert again.initial_binding == initial_identity_record(identity)
    fresh = assemble(verified, path.parent / "fresh")
    for u, v in zip(array_leaves(fresh.state), array_leaves(again.state), strict=True):
        assert u.dtype == v.dtype and u.shape == v.shape and u.tobytes() == v.tobytes()


def test_coherent_rehashed_resident_water_edit_changes_the_identity(verified, store):
    """Before B15 this edit restored silently under an unchanged run identity (review D3)."""
    path, identity = store
    resident = path / "initial_resident.npz"
    with np.load(resident, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    active = np.asarray(verified.fields["active"], dtype=bool)
    cell = tuple(np.argwhere(active)[0])
    original = float(arrays["water_depth_m"][cell])
    arrays["water_depth_m"][cell] += 0.01  # a valid (finite, nonnegative) but different initial water state
    writable(resident)
    np.savez(resident, **arrays)
    rewrite_binding(path, "initial_resident.npz")
    edited = restore(verified, path)  # every internal consistency check still passes
    assert float(edited.state.water.depth_m[cell]) == original + 0.01  # the edited state really is restored
    assert edited.initial_binding != initial_identity_record(identity)
    assert edited.initial_binding["initial_files_sha256"]["initial_resident.npz"] != \
        identity["initial_files_sha256"]["initial_resident.npz"]


def test_unrehashed_edit_is_refused(verified, store):
    path, _ = store
    baseline = path / "initial_baseline.npz"
    writable(baseline)
    with baseline.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(BedIntegrationError, match="restart source changed"):
        restore(verified, path)


def test_an_edit_during_the_restore_read_is_refused(verified, store, monkeypatch):
    import maple.surface.voxels.windowed_snapshot as snapshot_module

    path, _ = store
    real = snapshot_module.load_windowed_state_snapshot

    def load_then_edit(*args, **kwargs):
        out = real(*args, **kwargs)
        baseline = path / "initial_baseline.npz"
        writable(baseline)
        with baseline.open("ab") as handle:
            handle.write(b"x")
        return out

    monkeypatch.setattr(snapshot_module, "load_windowed_state_snapshot", load_then_edit)
    with pytest.raises(BedIntegrationError, match="changed"):
        restore(verified, path)


def test_receipt_datum_must_equal_the_verified_datum_before_use(verified, store, monkeypatch):
    import maple.surface.voxels.windowed_snapshot as snapshot_module

    path, _ = store
    monkeypatch.setattr(snapshot_module, "load_windowed_state_snapshot",
                        lambda *a, **k: pytest.fail("snapshot loaded before the datum check"))
    datum = float(verified.binding["grid"]["datum_offset_m"])
    for bad in (datum + 1.0, datum + 1e-12):
        with pytest.raises(BedIntegrationError, match="datum"):
            restore(verified, path, datum_offset_m=bad)
    for bad in (True, float("nan"), None):
        with pytest.raises(BedIntegrationError, match="datum"):
            restore(verified, path, datum_offset_m=bad)


def test_failure_record_and_missing_binding_are_refused(verified, store):
    path, _ = store
    marker = path / "ASSEMBLY_FAILED.json"
    marker.write_text("{}")
    with pytest.raises(BedIntegrationError, match="failure record"):
        restore(verified, path)
    marker.unlink()
    binding = path / INITIAL_BINDING_NAME
    writable(path)
    binding.unlink()
    with pytest.raises(BedIntegrationError, match="missing"):
        initial_binding_identity(path)


# --- the setup wiring: fresh and reuse identities, bound XML, refusal order -----------------------------------------
def setup(verified_case_dir, store, **kw):
    control = SedimentEventControl(phase_storage="csr", phase_max_nnz=100_000).validated()
    options = {"case_kind": "chastre", "store_dir": store, "window_voxels": 4, "max_store_bytes": 100_000_000,
               "max_host_bytes": 2_000_000_000, "max_journal_records": 10_000, "backend": "numpy",
               "control": control, "expected_maple_root": None, "allow_maple_source_change": True,
               "mahleran_root": None, "end_s": None}
    options.update(kw)
    return we._setup_windowed(verified_case_dir, **options)


def test_fresh_and_reuse_setups_carry_the_same_initial_identity_and_bound_xml(chastre_case_dir, tmp_path):
    store = tmp_path / "store"
    fresh = setup(chastre_case_dir, store)
    reuse = setup(chastre_case_dir, store, restore_initial=True)
    a, b = fresh["run_configuration"], reuse["run_configuration"]
    assert a["initial_assembly"] == b["initial_assembly"] and a["initial_assembly"]["initial_binding_sha256"]
    assert a["bound_mahleran_xml"] == b["bound_mahleran_xml"]
    xml = a["bound_mahleran_xml"]
    with open(xml["path"], "rb") as handle:
        assert hashlib.sha256(handle.read()).hexdigest() == xml["sha256"]
    assert xml["rfid_report"]["sha256"] == fresh["verified"].binding["source_rfid"]["report_sha256"]
    assert fresh["prepared"]["xml_sha_before"] == xml["sha256"]
    plan = fresh["memory_preflight"]
    assert plan["n_classes"] == len(fresh["verified"].case.config.grain_classes.classes)
    assert "device_pool_limit" not in plan  # host run: no device bound installed or claimed


def test_changed_xml_is_refused_before_parameters_assembly_or_upload(chastre_case_dir, tmp_path, monkeypatch):
    real = we.pinned_rfid_report

    def report_with_changed_xml(verified):
        report, record = real(verified)
        report = json.loads(json.dumps(report))
        report["mahleran"]["xml_sha256"] = "0" * 64  # the file on disk no longer hashes to its pin
        return report, record

    called = []
    monkeypatch.setattr(we, "pinned_rfid_report", report_with_changed_xml)
    for name in ("window_memory_plan", "assemble_windowed_case", "restore_initial_assembly", "assembly_to_backend"):
        monkeypatch.setattr(we, name, lambda *a, _n=name, **k: called.append(_n))
    import maple_syrup.sediment_experiment as se

    monkeypatch.setattr(se, "plot1_sediment_parameters_from_xml", lambda *a, **k: called.append("sediment"))
    with pytest.raises(SedimentEventError, match="does not hash to its bound digest"):
        setup(chastre_case_dir, tmp_path / "store")
    assert called == [] and not (tmp_path / "store").exists()


def test_pinned_rfid_report_is_read_once_and_refused_when_changed(verified, tmp_path):
    from types import SimpleNamespace

    source = tmp_path / "rfid" / "syrup"
    source.mkdir(parents=True)
    path = source / "rfid_import_report.json"
    path.write_text(json.dumps({"mahleran": {"xml_sha256": "1" * 64}}))
    pins = {"case_dir": str(tmp_path / "rfid"), "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    v = SimpleNamespace(binding={"source_rfid": pins})
    report, record = we.pinned_rfid_report(v)
    assert report["mahleran"]["xml_sha256"] == "1" * 64 and record["sha256"] == pins["report_sha256"]
    path.write_text(path.read_text() + " ")
    with pytest.raises(SedimentEventError, match="report changed"):
        we.pinned_rfid_report(v)
