import pytest
from builders import BACKENDS


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param


# ---- host-compiled CUDA plumbing (NOT hardware) ------------------------------------------------------------------------------------------------------------------------------------------
import hostd2
import numpy as np

needs_host_cuda = pytest.mark.skipif(not hostd2.have_toolchain(), reason="no C++20 g++ toolchain with <barrier>: the host execution of the CUDA source is not run (no claim)")


class Counter:
    def __init__(self):
        self.h2d = self.d2h = self.h2d_bytes = self.d2h_bytes = 0
        self.d2h_sizes = []


@pytest.fixture(scope="session")
def libs():
    from maple_syrup import floating_transpiration_cuda as ftc
    from maple_syrup import subsurface_cuda_floating as cf

    out = [("maple_syrup_floating_resident_step", hostd2.build(cf.render_resident_source())), ("maple_syrup_floating_withdraw", hostd2.build(ftc.render_source()))]
    try:
        from maple_syrup import floating_matrix_infiltration_cuda as fmc

        out.append(("maple_syrup_floating_matrix_step", hostd2.build(fmc.render_source())))
    except ImportError:
        pass
    try:
        from maple_syrup import floating_preferential_infiltration_cuda as fpc

        out.append(("maple_syrup_floating_pref_step", hostd2.build(fpc.render_source())))
    except ImportError:
        pass
    return out


@pytest.fixture()
def cp(libs, monkeypatch):
    import maple.core.backend as mb

    from maple_syrup import floating_soil_layers_cuda as fcu
    from maple_syrup import floating_transpiration_cuda as ftc

    c = hostd2.MultiFakeCupy(libs)
    c.dev = {"id": 0}
    c.count = Counter()
    rc = type("RC", (), {})()
    rc.require_cuda = lambda: c
    rc.current_device_id = lambda cc: c.dev["id"]
    rc.default_device_available_bytes_fn = lambda cc: (lambda: 1 << 40)
    monkeypatch.setattr(fcu, "_rc", lambda: rc)
    for mod in (fcu, ftc):
        with mod._LOCK:
            mod._FUNCTIONS.clear()
    for name in ("floating_matrix_infiltration_cuda", "floating_preferential_infiltration_cuda"):
        try:
            m = __import__(f"maple_syrup.{name}", fromlist=["_LOCK"])
        except ImportError:
            continue
        with m._LOCK:
            m._FUNCTIONS.clear()

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
    yield c
