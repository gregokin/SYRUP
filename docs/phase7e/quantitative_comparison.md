# Quantitative legacy versus bin comparison

Frozen Plot 1, no splash, 5400-second run, 1-second steps. The legacy replay uses
MAHLERAN's single-pool Crank–Nicolson transport and source-based deposition;
characteristic bins discretize mobile sediment position in the conservative SYRUP
transport. Bin convergence and reproduction of the legacy algorithm are separate
questions. Sources: `bin_sweep_qualification.json` and `legacy_comparison_numba.json`
in `benchmarks/phase 7e/`.

| Transport | Export (g) | Class 1 share (%) | One-second peak (g/s) | Time to 50% export (s) |
|---|---:|---:|---:|---:|
| MAHLERAN Fortran | 9.2233 | 80.93 | 0.0692 | 1285 |
| Python legacy replay | 9.2163 | 80.92 | 0.0692 | 1285 |
| 1 bins | 8.5289 | 100.00 | 1.6803 | 1158 |
| 2 bins | 13.8477 | 67.62 | 1.0995 | 1137 |
| 4 bins | 13.6723 | 69.56 | 0.7508 | 1148 |
| 8 bins | 13.7754 | 69.57 | 0.4101 | 1143 |
| 16 bins | 13.7733 | 69.68 | 0.3347 | 1144 |
| 32 bins | 13.7640 | 69.71 | 0.2181 | 1143 |
| 64 bins | 13.7632 | 69.71 | 0.1767 | 1143 |
| 128 bins | 13.7626 | 69.71 | 0.1833 | 1143 |

Class 2 supplies almost all remaining export; legacy class 3 export is only
0.0000216 g. At 32 bins, class 1 exports 9.5946 g versus legacy 7.4640 g (+28.5%),
and class 2 exports 4.1693 g versus 1.7593 g (+137.0%). Total bin export converges near
13.763 g, about 49.2% above legacy. Increasing bins does not remove that difference.

The 32-bin export centroid is 1133.3 s versus legacy 1315.9 s (182.6 s earlier),
and 90% export is reached at 1195 versus 1548 s (353 s earlier). Peak flux remains
higher after smoothing: the 60-second peak is 0.1310 versus 0.04495 g/s (2.91x).
Thus the disparity is more than one-second discretization noise, although raw
peaks are particularly sensitive to bin count and are not monotonic at 64/128.

Across 1–128 bins, gross pickup is 411.06339–411.06344 kg and gross deposition
411.04958–411.05486 kg. Legacy Fortran has 452.85733 kg pickup and 452.83221 kg
active-cell deposition (Python replay 452.83728 and 452.81217 kg). These are summed
transfers, not unique eroded mass or net landscape loss; repeated/local exchanges
can greatly exceed exported mass.

Accounting differs too: the legacy replay has 0.29193 kg effective artificial
source from negative-pool clipping and 0.30782 kg final mobile inventory. Binned
SYRUP ends with zero mobile inventory and closes bed/mobile/export budgets. The
legacy ring-walk diagnostic is not an extra sink to add to outlet export.

This comparison does not isolate bins as the cause of the legacy discrepancy:
legacy also uses fixed composition, unlimited supply and different pool/deposition
timing, while SYRUP uses actual MAPLE availability and evolving composition.
One bin happens to export a total closer to legacy, but loses class 2 entirely and
has a much larger spurious peak. It is not an equivalent approximation.

Eight bins are promising for Plot 1 cumulative/class results within 5% of the
128-bin reference, but that is convergence within SYRUP, not 5% agreement with
MAHLERAN. Default 32 remains unchanged.
