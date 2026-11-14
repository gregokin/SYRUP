"""HOST-compiled CUDA plumbing double for the S0 CUDA path (NOT hardware, NOT a GPU, no CuPy): the same numpy-backed CuPy double and host-compiled rendered kernels the existing floating suites use,
loaded by path under distinct module names. A skipped lane makes no claim."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_DIR = Path(__file__).resolve().parents[1] / "phase2_floating_intake_withdrawal"
_spec = importlib.util.spec_from_file_location("hostd2_s0_double", _DIR / "hostd2.py")
hostd2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hostd2)

from maple_syrup import floating_soil_layers as fl


class Counter:
    def __init__(self):
        self.h2d = self.d2h = self.h2d_bytes = self.d2h_bytes = 0
        self.d2h_sizes = []


def have_toolchain() -> bool:
    return hostd2.have_toolchain()


def build_libs():
    from maple_syrup import floating_matrix_infiltration_cuda as fmc
    from maple_syrup import floating_preferential_infiltration_cuda as fpc
    from maple_syrup import floating_transpiration_cuda as ftc
    from maple_syrup import subsurface_cuda_floating as cf

    return [("maple_syrup_floating_resident_step", hostd2.build(cf.render_resident_source())), ("maple_syrup_floating_withdraw", hostd2.build(ftc.render_source())),
            ("maple_syrup_floating_matrix_step", hostd2.build(fmc.render_source())), ("maple_syrup_floating_pref_step", hostd2.build(fpc.render_source()))]


def make_cp(libs, monkeypatch):
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
        m = __import__(f"maple_syrup.{name}", fromlist=["_LOCK"])
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

    orig_get = hostd2.FakeArray.get

    def counted_get(self, out=None):
        """`.get()` is a real device-to-host copy on hardware (the native cold identity hashes stream through it): count it, exactly like `to_host`."""
        c.count.d2h += 1
        c.count.d2h_bytes += int(self.nbytes)
        c.count.d2h_sizes.append(int(self.nbytes))
        return orig_get(self, out)

    monkeypatch.setattr(hostd2.FakeArray, "get", counted_get)
    monkeypatch.setattr(mb, "to_host", to_host)
    monkeypatch.setattr(mb, "to_device", to_device)
    return c


def pair(cp, edges, depth, *, active):
    """The same inputs as a NumPy column and as a (fake) CUDA column."""
    fake = hostd2.fake
    edges = np.asarray(edges, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    L = len(edges) - 1
    dev = lambda x: fake(x) if isinstance(x, np.ndarray) else x
    common = {"theta_evaporation_stress": 0.2, "diffusion_mode": "constant", "bottom_drainage": "open", "cell_area_m2": 1.0}
    cpu = fl.prepare_floating_column(edges, theta_residual=0.1, theta_field_capacity=0.3, theta_saturation=0.5, soil_depth_m=depth, active_mask=active, diffusivity_m2_s=np.zeros(L - 1), **common)
    gpu = fl.prepare_floating_column(edges, theta_residual=0.1, theta_field_capacity=0.3, theta_saturation=0.5, soil_depth_m=dev(depth), active_mask=dev(active), backend="cuda",
                                     diffusivity_m2_s=dev(np.zeros(L - 1)), **common)
    return cpu, gpu
