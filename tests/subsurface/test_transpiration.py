'''Isolated tests of maple_syrup.subsurface_transpiration against the actual native column API.'''
import unittest
from unittest import mock

import numpy as np

from maple_syrup import subsurface as ss
from maple_syrup import subsurface_transpiration as st  # opt-in branch: installed/overlaid alongside subsurface

THETA_R, THETA_FC, THETA_SAT, THICK = 0.05, 0.25, 0.40, 0.1
RES_M = THETA_R * THICK  # residual depth per layer (0.005 m)


def cuda_hardware():
    '''Hardware/driver probe ONLY; construction or implementation errors must fail, not skip.'''
    try:
        import cupy
        return int(cupy.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def xp_for(backend):
    if backend == 'cuda':
        import cupy
        return cupy
    return np


def make_column(L=3, N=4, area=1.0, backend='numpy', active=None, available=None, reserve_bytes=0, diffusivity=0.0):
    edges = np.linspace(0.0, THICK * L, L + 1)
    layers = ss.soil_layers(edges, theta_residual=THETA_R, theta_field_capacity=THETA_FC, theta_saturation=THETA_SAT,
                            diffusivity_m2_s=diffusivity, theta_evaporation_stress=0.10)
    mask = np.ones(N, bool) if active is None else np.asarray(active, bool)
    mask = xp_for(backend).asarray(mask)  # explicit setup mask in the selected namespace
    fn = None if available is None else (lambda: available['bytes'])
    return ss.prepare_column(layers, mask, area, backend=backend, reserve_bytes=reserve_bytes, available_bytes_fn=fn)


def host(a):
    return np.asarray(a.get() if hasattr(a, 'get') else a)


class WithdrawalTests(unittest.TestCase):
    def setUp(self):
        self.col = make_column()
        self.op = st.prepare_transpiration(self.col)
        self.w0 = np.full((3, 4), 0.02)  # theta 0.2 < field capacity: 0.015 m available per layer/cell
        self.state = ss.column_state(self.col, self.w0, t_s=1234.5)

    def test_analytic_finite_and_residual_limited(self):
        req = np.zeros((3, 4))
        req[0, :] = 0.010  # within availability
        req[1, :] = 0.020  # exceeds 0.015 -> residual-limited
        req_copy = req.copy()
        r = st.withdraw_transpiration(self.op, self.state, req)
        np.testing.assert_array_equal(req, req_copy)
        np.testing.assert_array_equal(self.state.water_m, self.w0)  # input state untouched
        self.assertEqual(r.state.t_s, 1234.5)  # time preserved
        np.testing.assert_allclose(r.actual_m[0], 0.010, rtol=1e-15)
        np.testing.assert_allclose(r.actual_m[1], 0.015, rtol=1e-15)
        np.testing.assert_array_equal(r.actual_m[2], 0.0)
        np.testing.assert_allclose(r.unmet_m[1], 0.005, rtol=1e-12)
        np.testing.assert_allclose(r.scale[1], 0.75, rtol=1e-15)
        np.testing.assert_array_equal(r.scale[2], 0.0)
        np.testing.assert_allclose(r.state.water_m[1], RES_M, rtol=1e-12)
        self.assertTrue((r.state.water_m >= 0).all())
        b = r.budget
        self.assertLessEqual(abs(b.residual_m3), b.bound_m3)
        self.assertLessEqual(abs(b.request_residual_m3), b.bound_m3)
        np.testing.assert_allclose(b.actual_m3, 4 * (0.010 + 0.015), rtol=1e-12)
        np.testing.assert_allclose(b.storage_before_m3 - b.storage_after_m3, b.actual_m3, rtol=1e-12)
        np.testing.assert_allclose(b.storage_before_m3, float(ss.storage_m3(self.col, self.state)), rtol=1e-12)
        np.testing.assert_allclose(b.storage_after_m3, float(ss.storage_m3(self.col, r.state)), rtol=1e-12)

    def test_single_host_packet_per_withdrawal(self):
        with mock.patch.object(st, '_to_host', wraps=st._to_host) as m:
            st.withdraw_transpiration(self.op, self.state, np.full((3, 4), 0.01))
        self.assertEqual(m.call_count, 1)
        with mock.patch.object(st, '_to_host', wraps=st._to_host) as m:
            st.step_with_transpiration(self.op, self.state, 86400.0, added_water_m=np.zeros((3, 4)),
                                       potential_evaporation_m_s=np.zeros(4), requested_transpiration_m=np.full((3, 4), 0.01))
        self.assertEqual(m.call_count, 1)  # T packet only; hydro uses its own native packet
        with mock.patch.object(st, '_to_host', wraps=st._to_host) as m:
            with self.assertRaises(ss.SubsurfaceError):  # invalid values are flagged from the same single packet
                st.withdraw_transpiration(self.op, self.state, np.full((3, 4), -1.0))
        self.assertEqual(m.call_count, 1)

    def test_area_multiplied_once(self):
        col = make_column(area=0.0625)
        op = st.prepare_transpiration(col)
        state = ss.column_state(col, self.w0)
        req = np.full((3, 4), 0.004)
        b = st.withdraw_transpiration(op, state, req).budget
        np.testing.assert_allclose(b.actual_m3, 12 * 0.004 * 0.0625, rtol=1e-12)
        np.testing.assert_allclose(b.requested_m3, 12 * 0.004 * 0.0625, rtol=1e-12)

    def test_already_below_residual_unchanged(self):
        w = np.full((3, 4), 0.002)  # below residual depth 0.005
        state = ss.column_state(self.col, w)
        r = st.withdraw_transpiration(self.op, state, np.full((3, 4), 0.01))
        np.testing.assert_array_equal(r.state.water_m, w)
        np.testing.assert_array_equal(r.actual_m, 0.0)
        np.testing.assert_array_equal(r.scale, 0.0)
        np.testing.assert_allclose(r.unmet_m, 0.01)

    def test_zero_demand_and_dry(self):
        r = st.withdraw_transpiration(self.op, self.state, np.zeros((3, 4)))
        np.testing.assert_array_equal(r.state.water_m, self.w0)
        np.testing.assert_array_equal(r.scale, 0.0)
        self.assertEqual(r.budget.actual_m3, 0.0)
        dry = ss.column_state(self.col, np.zeros((3, 4)))
        r = st.withdraw_transpiration(self.op, dry, np.full((3, 4), 0.01))
        np.testing.assert_array_equal(r.state.water_m, 0.0)
        self.assertEqual(r.budget.actual_m3, 0.0)

    def test_inactive_cells_and_active_only_budget(self):
        col = make_column(L=1, N=2, active=[True, False])
        op = st.prepare_transpiration(col)
        state = ss.column_state(col, np.array([[0.02, 0.03]]))
        req = np.array([[0.01, 0.0]])
        r = st.withdraw_transpiration(op, state, req)
        b = r.budget
        np.testing.assert_allclose(b.storage_before_m3, 0.02, rtol=1e-15)  # inactive 0.03 excluded
        np.testing.assert_allclose(b.storage_after_m3, 0.01, rtol=1e-15)
        np.testing.assert_allclose((b.requested_m3, b.actual_m3, b.unmet_m3), (0.01, 0.01, 0.0), rtol=1e-15)
        np.testing.assert_allclose(b.storage_before_m3, float(ss.storage_m3(col, state)), rtol=1e-12)
        np.testing.assert_allclose(b.storage_after_m3, float(ss.storage_m3(col, r.state)), rtol=1e-12)
        self.assertEqual(float(r.state.water_m[0, 1]), 0.03)  # preserved exactly
        np.testing.assert_array_equal(r.scale[:, 1], 0.0)
        with self.assertRaises(ss.SubsurfaceError):  # positive request on the inactive cell
            st.withdraw_transpiration(op, state, np.array([[0.01, 0.01]]))
        with self.assertRaises(ss.SubsurfaceError):  # invalid water on an inactive cell is still invalid
            st.withdraw_transpiration(op, ss.column_state(col, np.array([[0.02, np.nan]])), req)
        with self.assertRaises(ss.SubsurfaceError):
            st.withdraw_transpiration(op, ss.column_state(col, np.array([[0.02, -0.01]])), req)

    def test_mixed_mask_budget_regression(self):
        col = make_column(L=2, N=5, area=2.5, active=[True, False, True, False, True])
        op = st.prepare_transpiration(col)
        rng = np.random.default_rng(11)
        w = rng.uniform(0.0, 0.035, (2, 5))
        req = rng.uniform(0.0, 0.03, (2, 5))
        req[:, [1, 3]] = 0.0
        state = ss.column_state(col, w)
        r = st.withdraw_transpiration(op, state, req)
        act = np.array([True, False, True, False, True])
        exp_actual = np.minimum(req, np.maximum(w - RES_M, 0.0))
        np.testing.assert_allclose(r.actual_m, exp_actual, rtol=1e-15)
        np.testing.assert_allclose(r.budget.actual_m3, exp_actual[:, act].sum() * 2.5, rtol=1e-12)
        np.testing.assert_allclose(r.budget.storage_before_m3, w[:, act].sum() * 2.5, rtol=1e-12)
        np.testing.assert_allclose(r.budget.requested_m3, req[:, act].sum() * 2.5, rtol=1e-12)
        np.testing.assert_array_equal(r.state.water_m[:, ~act], w[:, ~act])
        self.assertLessEqual(abs(r.budget.residual_m3), r.budget.bound_m3)

    def test_validation(self):
        for bad in (np.full((3, 4), -1e-3), np.full((3, 4), np.nan), np.full((3, 4), np.inf),
                    np.zeros((3, 5)), np.zeros((3, 4), np.float32), np.zeros((4, 3)).T):
            with self.assertRaises((ss.SubsurfaceError, TypeError, ValueError), msg=repr(bad.dtype)):
                st.withdraw_transpiration(self.op, self.state, bad)
        sat = THETA_SAT * THICK
        for w in (np.full((3, 4), -1e-6), np.full((3, 4), np.nan), np.full((3, 4), sat * 1.5)):
            with self.assertRaises(ss.SubsurfaceError):
                st.withdraw_transpiration(self.op, ss.column_state(self.col, w), np.zeros((3, 4)))
        with self.assertRaises(ss.SubsurfaceError):
            st.withdraw_transpiration(object(), self.state, np.zeros((3, 4)))

    def test_strict_scalar_guards(self):
        zeros = np.zeros((3, 4))
        for bad in (-1, -10 ** 12, True, 1.5, None):
            with self.assertRaises((ss.SubsurfaceError, TypeError), msg=repr(bad)):
                st.withdraw_transpiration(self.op, self.state, zeros, extra_reserve_bytes=bad)
            with self.assertRaises((ss.SubsurfaceError, TypeError), msg=repr(bad)):
                st.step_with_transpiration(self.op, self.state, 0.0, added_water_m=zeros,
                                           potential_evaporation_m_s=np.zeros(4), requested_transpiration_m=zeros,
                                           extra_reserve_bytes=bad)
            with self.assertRaises((ss.SubsurfaceError, TypeError), msg=repr(bad)):
                st.prepare_transpiration(self.col, extra_reserve_bytes=bad)
            with self.assertRaises((ss.SubsurfaceError, TypeError), msg=repr(bad)):
                st.estimate_transpiration_memory(3, 4, backend='numpy', extra_reserve_bytes=bad)
            with self.assertRaises((ss.SubsurfaceError, TypeError), msg=repr(bad)):
                st.estimate_combined_memory(self.col, extra_reserve_bytes=bad)
        for bad in (0, -1, True, 2.0):
            with self.assertRaises((ss.SubsurfaceError, TypeError), msg=repr(bad)):
                st.estimate_transpiration_memory(bad, 4, backend='numpy')
            with self.assertRaises((ss.SubsurfaceError, TypeError), msg=repr(bad)):
                st.estimate_transpiration_memory(3, bad, backend='numpy')
        # zero available memory cannot be bypassed by a negative caller reserve
        avail = {'bytes': 1 << 30}
        col = make_column(available=avail)
        op = st.prepare_transpiration(col)
        avail['bytes'] = 0
        with self.assertRaises((ss.SubsurfaceError, ss.SubsurfaceMemoryError)):
            st.withdraw_transpiration(op, ss.column_state(col, self.w0), zeros, extra_reserve_bytes=-10 ** 12)
        for v in (True, -5, 1.0):  # invalid availability readings are errors, not admissions
            avail['bytes'] = v
            with self.assertRaises(ss.SubsurfaceError):
                st.withdraw_transpiration(op, ss.column_state(col, self.w0), zeros)

    def test_memory_refusal_before_allocation(self):
        avail = {'bytes': 1 << 30}
        col = make_column(available=avail, reserve_bytes=1024)
        op = st.prepare_transpiration(col)
        state = ss.column_state(col, self.w0)
        avail['bytes'] = 64  # no room for outputs
        with self.assertRaises(ss.SubsurfaceMemoryError):
            st.withdraw_transpiration(op, state, np.full((3, 4), 0.01))
        with self.assertRaises(ss.SubsurfaceMemoryError):
            st.step_with_transpiration(op, state, 86400.0, added_water_m=np.zeros((3, 4)),
                                       potential_evaporation_m_s=np.zeros(4), requested_transpiration_m=np.zeros((3, 4)))
        np.testing.assert_array_equal(state.water_m, self.w0)
        with mock.patch.object(col.xp if hasattr(col, 'xp') else np, 'empty', side_effect=AssertionError('allocated')):
            with self.assertRaises(ss.SubsurfaceMemoryError):  # refused before ANY allocation
                st.withdraw_transpiration(op, state, np.full((3, 4), 0.01))
            with self.assertRaises(ss.SubsurfaceMemoryError):
                st.prepare_transpiration(col)
        self.assertEqual(int(col.reserve_bytes), 1024)  # budgets never raised/mutated
        m = st.estimate_transpiration_memory(3, 4, backend='numpy', reserve_bytes=10, extra_reserve_bytes=5)
        self.assertEqual(m.required_bytes, m.host_bytes + 15)
        self.assertEqual(m.total_bytes, m.host_bytes + m.device_bytes + 15)
        self.assertEqual(m.output_bytes, 4 * 8 * 12)
        self.assertEqual(m.parameter_bytes, 2 * 8 * 3)
        self.assertEqual(m.row_bytes, 8 * 8 * 4)
        self.assertEqual(m.host_bytes, m.parameter_bytes + m.state_bytes + m.input_bytes + m.output_bytes
                         + m.row_bytes + m.scratch_bytes)
        self.assertGreaterEqual(m.scratch_bytes, 8 * 4 + 3 * m.packet_bytes)  # reduction workspace + 3 live packets
        mc = st.estimate_transpiration_memory(3, 4, backend='numba')
        self.assertEqual(mc.scratch_bytes, 8 * 4 + 3 * mc.packet_bytes)
        self.assertGreater(op.prepare_bytes, m.parameter_bytes + 6 * 8 * 3)  # host vectors + validation temporaries


class CombinedStepTests(unittest.TestCase):
    def setUp(self):
        self.col = make_column()
        self.op = st.prepare_transpiration(self.col)
        self.w0 = np.full((3, 4), 0.02)
        self.state = ss.column_state(self.col, self.w0, t_s=100.0)
        self.zeros, self.pet0 = np.zeros((3, 4)), np.zeros(4)

    def test_zero_time_identity_and_no_double_time(self):
        r = st.step_with_transpiration(self.op, self.state, 0.0, added_water_m=self.zeros,
                                       potential_evaporation_m_s=self.pet0, requested_transpiration_m=self.zeros)
        self.assertEqual(r.state.t_s, 100.0)
        np.testing.assert_array_equal(r.state.water_m, self.w0)
        with self.assertRaises(ss.SubsurfaceError):
            st.step_with_transpiration(self.op, self.state, 0.0, added_water_m=self.zeros,
                                       potential_evaporation_m_s=self.pet0, requested_transpiration_m=np.full((3, 4), 1e-3))
        dry = ss.column_state(self.col, np.zeros((3, 4)), t_s=100.0)
        with self.assertRaises(ss.SubsurfaceError):  # positive request rejected at dt=0 even when physically dry
            st.step_with_transpiration(self.op, dry, 0.0, added_water_m=self.zeros,
                                       potential_evaporation_m_s=self.pet0, requested_transpiration_m=np.full((3, 4), 1e-3))
        r = st.step_with_transpiration(self.op, self.state, 86400.0, added_water_m=self.zeros,
                                       potential_evaporation_m_s=self.pet0, requested_transpiration_m=np.full((3, 4), 1e-3))
        self.assertEqual(r.state.t_s, 100.0 + 86400.0)  # hydro advances once, T preserves
        self.assertEqual(self.state.t_s, 100.0)
        np.testing.assert_allclose(r.actual_m, 1e-3, rtol=1e-9)
        np.testing.assert_allclose(r.budget.transpiration_m3, 12e-3, rtol=1e-9)

    def test_combined_budget_with_intake_evaporation_and_transpiration(self):
        col = make_column(diffusivity=1e-9)  # real redistribution exercised here
        op = st.prepare_transpiration(col)
        w = np.array([[0.03], [0.02], [0.01]]) * np.ones((1, 4))
        state = ss.column_state(col, w, t_s=100.0)
        added = np.zeros((3, 4))
        added[0, :] = 0.004
        pet = np.full(4, 1e-8)  # ~0.86 mm/day potential soil evaporation
        req = np.zeros((3, 4))
        req[1, :] = 0.020  # residual-limited in layer 1
        r = st.step_with_transpiration(op, state, 86400.0, added_water_m=added, potential_evaporation_m_s=pet,
                                       requested_transpiration_m=req)
        b = r.budget
        self.assertGreater(b.added_m3, 0.0)
        self.assertGreater(b.transpiration_m3, 0.0)
        self.assertGreaterEqual(b.evaporation_m3, 0.0)
        self.assertNotEqual(b.transpiration_m3, b.evaporation_m3)
        lhs = b.storage_after_m3 - b.storage_before_m3
        rhs = b.added_m3 - b.surface_return_m3 - b.drainage_m3 - b.evaporation_m3 - b.transpiration_m3
        self.assertLessEqual(abs(lhs - rhs), b.bound_m3)
        self.assertLessEqual(abs(b.residual_m3), b.bound_m3)
        self.assertLessEqual(abs(b.intermediate_mismatch_m3), b.bound_m3)
        np.testing.assert_allclose(b.storage_before_m3, float(ss.storage_m3(col, state)), rtol=1e-12)
        np.testing.assert_allclose(b.storage_after_m3, float(ss.storage_m3(col, r.state)), rtol=1e-12)
        self.assertGreater(b.transpiration_unmet_m3, 0.0)
        np.testing.assert_allclose(b.transpiration_requested_m3, b.transpiration_m3 + b.transpiration_unmet_m3, rtol=1e-12)
        np.testing.assert_array_equal(state.water_m, w)  # original untouched
        self.assertEqual(r.scale.shape, (3, 4))

    def test_combined_with_inactive_cells(self):
        col = make_column(L=2, N=3, active=[True, False, True], diffusivity=1e-9)
        op = st.prepare_transpiration(col)
        w = np.array([[0.02, 0.03, 0.025], [0.01, 0.03, 0.015]])
        state = ss.column_state(col, w)
        added = np.zeros((2, 3))
        added[0, [0, 2]] = 0.003
        pet = np.array([1e-8, 0.0, 1e-8])
        req = np.array([[0.004, 0.0, 0.004], [0.001, 0.0, 0.03]])
        r = st.step_with_transpiration(op, state, 86400.0, added_water_m=added, potential_evaporation_m_s=pet,
                                       requested_transpiration_m=req)
        b = r.budget
        np.testing.assert_allclose(b.storage_before_m3, w[:, [0, 2]].sum(), rtol=1e-12)
        np.testing.assert_allclose(b.storage_after_m3, host(r.state.water_m)[:, [0, 2]].sum(), rtol=1e-12)
        np.testing.assert_array_equal(host(r.state.water_m)[:, 1], w[:, 1])  # inactive preserved through hydro + T
        self.assertLessEqual(abs(b.residual_m3), b.bound_m3)
        self.assertGreater(b.transpiration_unmet_m3, 0.0)


class BackendAgreementTests(unittest.TestCase):
    def _run(self, backend):
        col = make_column(L=3, N=6, backend=backend, active=[True, True, False, True, True, True])
        op = st.prepare_transpiration(col)
        xp = col.xp
        rng = np.random.default_rng(5)
        w = rng.uniform(0.0, 0.035, (3, 6))
        req = rng.uniform(0.0, 0.03, (3, 6))
        req[:, 2] = 0.0
        state = ss.column_state(col, xp.asarray(w))
        r = st.withdraw_transpiration(op, state, xp.ascontiguousarray(xp.asarray(req)))
        out = {k: host(getattr(r, k)) for k in ('actual_m', 'unmet_m', 'scale')}
        out['w'], out['b'] = host(r.state.water_m), r.budget
        with self.assertRaises(ss.SubsurfaceError):  # flags work on every backend
            st.withdraw_transpiration(op, state, xp.ascontiguousarray(xp.asarray(np.where(req == 0.0, 0.01, req))))
        return out

    def test_numpy_numba_agree(self):
        a, b = self._run('numpy'), self._run('numba')
        for k in ('actual_m', 'unmet_m', 'scale', 'w'):
            np.testing.assert_allclose(b[k], a[k], rtol=1e-15, atol=0.0)
        self.assertLessEqual(abs(a['b'].actual_m3 - b['b'].actual_m3), a['b'].bound_m3)
        self.assertLessEqual(abs(a['b'].storage_before_m3 - b['b'].storage_before_m3), a['b'].bound_m3)

    @unittest.skipUnless(cuda_hardware(), 'no CUDA device visible to CuPy')
    def test_cuda_agrees_with_numpy(self):
        a, c = self._run('numpy'), self._run('cuda')  # construction/implementation errors fail, never skip
        for k in ('actual_m', 'unmet_m', 'scale', 'w'):
            np.testing.assert_allclose(c[k], a[k], rtol=1e-15, atol=0.0)
        self.assertLessEqual(abs(a['b'].actual_m3 - c['b'].actual_m3), a['b'].bound_m3 + c['b'].bound_m3)
        self.assertLessEqual(abs(a['b'].storage_before_m3 - c['b'].storage_before_m3), a['b'].bound_m3 + c['b'].bound_m3)

    @unittest.skipUnless(cuda_hardware(), 'no CUDA device visible to CuPy')
    def test_cuda_combined_single_packet_and_budget(self):
        import cupy as cp
        col = make_column(L=2, N=3, backend='cuda', active=[True, False, True], diffusivity=1e-9)
        op = st.prepare_transpiration(col)
        w = np.array([[0.02, 0.03, 0.025], [0.01, 0.03, 0.015]])
        state = ss.column_state(col, cp.asarray(w))
        added = cp.zeros((2, 3))
        added[0, 0] = 0.003
        added[0, 2] = 0.003
        with mock.patch.object(st, '_to_host', wraps=st._to_host) as m:
            r = st.step_with_transpiration(op, state, 86400.0, added_water_m=added,
                                           potential_evaporation_m_s=cp.asarray([1e-8, 0.0, 1e-8]),
                                           requested_transpiration_m=cp.asarray([[0.004, 0.0, 0.004], [0.001, 0.0, 0.03]]))
        self.assertEqual(m.call_count, 1)
        self.assertLessEqual(abs(r.budget.residual_m3), r.budget.bound_m3)
        np.testing.assert_array_equal(host(r.state.water_m)[:, 1], w[:, 1])
        self.assertIs(type(r.scale), cp.ndarray)


if __name__ == '__main__':
    unittest.main()
