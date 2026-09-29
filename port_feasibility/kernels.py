"""CPU feasibility kernels, not production MAPLE or a complete MAHLERAN port.

SI form of constant-friction iroute=5 balance and method-2 sediment CN.
Fixed square cells; one cardinal downstream receiver per cell; -1 is outlet.
Ordering is explicit. No infiltration, splash, entrainment law, or moving DEM.
Unlike legacy: bounded monotone bracket; reject negative RHS rather than clip.
"""
from dataclasses import dataclass
import numpy as np


class StepRejected(ValueError):
    """No state has been published; a caller may choose a smaller timestep."""


@dataclass(frozen=True)
class Network:
    receiver: np.ndarray
    order: np.ndarray
    dx_m: float

    def __post_init__(self):
        r, o = np.asarray(self.receiver), np.asarray(self.order)
        n = r.size
        if r.ndim != 1 or o.shape != (n,) or n == 0:
            raise ValueError('network arrays must be nonempty matching vectors')
        if r.dtype.kind not in 'iu' or o.dtype.kind not in 'iu':
            raise ValueError('integer indices required')
        if not np.array_equal(np.sort(o), np.arange(n)):
            raise ValueError('order must be a permutation')
        if np.any((r < -1) | (r >= n)) or not np.isfinite(self.dx_m) or self.dx_m <= 0:
            raise ValueError('invalid receiver or cell width')
        rank = np.empty(n, dtype=int)
        rank[o] = np.arange(n)
        internal = r >= 0
        if np.any(rank[r[internal]] <= rank[np.flatnonzero(internal)]):
            raise ValueError('order must be strictly upstream to downstream; cycles forbidden')
        # Own immutable topology, rather than trust later caller mutation.
        r, o = r.copy(), o.copy()
        r.flags.writeable = o.flags.writeable = False
        object.__setattr__(self, 'receiver', r)
        object.__setattr__(self, 'order', o)


def _array(value, shape, name):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError(f'{name}: expected finite nonnegative shape {shape}')
    return a


def _dt(dt):
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('positive finite dt required')


def incoming(outflow, network):
    result = np.zeros_like(outflow)
    internal = network.receiver >= 0
    np.add.at(result, network.receiver[internal], outflow[internal])
    return result


@dataclass(frozen=True)
class WaterResult:
    depth_m: np.ndarray
    discharge_m2_s: np.ndarray
    export_m3: float
    balance_by_cell_m3: np.ndarray


def water_step(network, depth_m, discharge_m2_s, rain_m_s, slope, friction, dt_s):
    _dt(dt_s)
    shape = (len(network.receiver),)
    h = _array(depth_m, shape, 'depth')
    q = _array(discharge_m2_s, shape, 'discharge')
    rain = _array(rain_m_s, shape, 'rain')
    slope = _array(slope, shape, 'slope')
    friction = _array(friction, shape, 'friction')
    if np.any(friction == 0):
        raise ValueError('friction must be positive')
    k = np.sqrt(8 * 9.81 * slope / friction)
    old_in = incoming(q, network)
    new_in, new_q, new_h = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    factor = dt_s / (2 * network.dx_m)
    for cell in network.order:
        rhs = h[cell] + dt_s * rain[cell] + factor * (old_in[cell] - q[cell] + new_in[cell])
        if rhs < 0:
            raise StepRejected(f'negative water RHS at cell {cell}: {rhs}')
        lo, hi = 0., rhs  # h + factor*k*h**1.5 = rhs; root lies in [0,rhs]
        for _ in range(80):
            mid = (lo + hi) * .5
            if mid == lo or mid == hi:
                break
            if mid + factor * k[cell] * mid**1.5 > rhs:
                hi = mid
            else:
                lo = mid
        new_h[cell] = (lo + hi) * .5
        new_q[cell] = k[cell] * new_h[cell]**1.5
        dst = network.receiver[cell]
        if dst >= 0:
            new_in[dst] += new_q[cell]
    dx = network.dx_m
    balance = (new_h-h-dt_s*rain)*dx**2 - .5*dt_s*dx*(old_in+new_in-q-new_q)
    export = float(.5*dt_s*dx*np.sum((q+new_q)[network.receiver == -1]))
    return WaterResult(new_h, new_q, export, balance)


@dataclass(frozen=True)
class SedimentResult:
    mobile_kg: np.ndarray
    outflow_kg_s: np.ndarray
    export_by_class_kg: np.ndarray
    balance_by_cell_class_kg: np.ndarray


def sediment_step(network, mobile_kg, outflow_kg_s, velocity_m_s, dt_s):
    """Source-free CN advection after pickup; deposition occurs separately.

    M_new = (M_old + dt/2*(in_old-out_old+in_new))/(1+dt*v/(2dx)).
    Q_new = v/dx*M_new. This is an explicitly declared operator split.
    """
    _dt(dt_s)
    raw = np.asarray(mobile_kg)
    if raw.ndim != 2 or raw.shape[0] != len(network.receiver):
        raise ValueError('mobile shape must be (n_cells,n_classes)')
    m = _array(raw, raw.shape, 'mobile')
    q = _array(outflow_kg_s, raw.shape, 'sediment flux')
    v = _array(velocity_m_s, raw.shape, 'sediment velocity')
    old_in = incoming(q, network)
    new_in, new_q, new_m = np.zeros_like(m), np.zeros_like(m), np.zeros_like(m)
    for cell in network.order:
        rhs = m[cell] + .5*dt_s*(old_in[cell]-q[cell]+new_in[cell])
        if np.any(rhs < 0):
            raise StepRejected(f'negative sediment RHS at cell {cell}')
        new_m[cell] = rhs / (1+.5*dt_s*v[cell]/network.dx_m)
        new_q[cell] = new_m[cell]*v[cell]/network.dx_m
        dst = network.receiver[cell]
        if dst >= 0:
            new_in[dst] += new_q[cell]
    balance = new_m-m-.5*dt_s*(old_in+new_in-q-new_q)
    export = .5*dt_s*np.sum((q+new_q)[network.receiver == -1], axis=0)
    return SedimentResult(new_m, new_q, export, balance)
