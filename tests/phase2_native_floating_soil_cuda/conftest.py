"""Host-double plumbing for the E3b CUDA branch (NOT hardware). Loads the D2 hostd2/builders and the E3 builders by path; fixtures mirror the D2 ones."""
import sys
from pathlib import Path

import numpy as np
import pytest

T = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T / "phase2_floating_intake_withdrawal"))  # hostd2 only (the two `builders.py` modules would collide)

import hostd2

needs_host_cuda = pytest.mark.skipif(not hostd2.have_toolchain(), reason="no C++20 g++ toolchain with <barrier>: no host execution of the CUDA source (no claim)")


class Counter:
    def __init__(self):
        self.h2d = self.d2h = self.h2d_bytes = self.d2h_bytes = 0
        self.d2h_sizes = []


@pytest.fixture(scope="session")
def libs():
    from maple_syrup import floating_matrix_infiltration_cuda as fmc
    from maple_syrup import floating_transpiration_cuda as ftc
    from maple_syrup import subsurface_cuda_floating as cf

    return [("maple_syrup_floating_resident_step", hostd2.build(cf.render_resident_source())), ("maple_syrup_floating_withdraw", hostd2.build(ftc.render_source())),
            ("maple_syrup_floating_matrix_step", hostd2.build(fmc.render_source()))]


@pytest.fixture()
def cp(libs, monkeypatch):
    import maple.core.backend as mb

    from maple_syrup import floating_matrix_infiltration_cuda as fmc
    from maple_syrup import floating_soil_layers_cuda as fcu

    c = hostd2.MultiFakeCupy(libs)
    c.dev = {"id": 0}
    c.count = Counter()
    rc = type("RC", (), {})()
    rc.require_cuda = lambda: c
    rc.current_device_id = lambda cc: c.dev["id"]
    rc.default_device_available_bytes_fn = lambda cc: (lambda: 1 << 40)
    monkeypatch.setattr(fcu, "_rc", lambda: rc)
    for mod in (fcu, fmc):
        with mod._LOCK:
            mod._FUNCTIONS.clear()

    def to_host(a):
        out = np.array(a)
        c.count.d2h += 1
        c.count.d2h_bytes += out.nbytes
        c.count.d2h_sizes.append(out.nbytes)
        return out

    def to_device(a, xp):
        c.count.h2d += 1
        c.count.h2d_bytes += int(np.asarray(a).nbytes)
        return hostd2.fake(np.ascontiguousarray(a), c.dev["id"])

    monkeypatch.setattr(mb, "to_host", to_host)
    monkeypatch.setattr(mb, "to_device", to_device)
    monkeypatch.setattr(mb, "is_device_array", lambda a: isinstance(a, hostd2.FakeArray))
    yield c
