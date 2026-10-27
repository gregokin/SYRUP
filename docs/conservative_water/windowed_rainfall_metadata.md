# HOST rainfall metadata in the device water bridge

The device water bridge admits the three HOST arrays owned by the exact native
`RainfallSchedule`: `edges_s`, `intensity_mm_per_h` and `rate_m_per_s`. The native
constructor keeps these arrays on NumPy even when the landscape uses CuPy.
Admission checks their native one-dimensional FP64, readonly, contiguous private
ownership and lengths (`n + 1`, `n`, `n`, with `n >= 1`). The existing rainfall
constructor and consumers retain authority over forcing values, conversion,
integration and solver time.

The census authenticates the schedule holder before accepting its exact field
identities. A same-named dictionary, foreign dataclass, subclass, direct leaf
without its owner, additional field, mutated array metadata or unrelated HOST
payload refuses. It neither converts arrays nor transfers data. There is no new
interval-count ceiling derived from the separate report-clock limit.

Each array occurrence retains its complete backing-byte charge. Repeated owners
receive repeated charges; existing leaf, depth, frontier, individual backing and
whole-backing occurrence limits remain active. All Native resource admission,
runtime/Source/currentness checks, safety factors and water physics remain in
their owning paths. New Source/code identities require fresh immutable runtime
and Source accounting inputs before qualification.

Task979 supplies focused metadata tests in
`tests/conservative_water/test_windowed_rainfall_metadata.py`. The author has not
run tests, providers or a water event. Source acceptance of this correction does
not establish complete GPU water execution, fit, numerical equivalence or
calibration.
