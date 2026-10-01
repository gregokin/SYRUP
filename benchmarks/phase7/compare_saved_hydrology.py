"""Compare saved MAHLERAN initialization experiments and earlier SYRUP runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/phase7"


def main():
    hashes = {}

    def read(name, **kwargs):
        p = ROOT / name
        hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        return np.loadtxt(p, **kwargs)

    curves = {}
    for label, folder in [
        ("Full MAHLERAN: sampled K", "mahleran_fixed_no_splash"),
        ("Full MAHLERAN: constant K", "mahleran_deterministic_ksat_run"),
    ]:
        a = read(f"outputs/phase7/{folder}/Output/hydro001.dat")
        assert a.shape == (5400, 12) and np.isfinite(a).all()
        curves[label] = (a[:, 0], a[:, 2] * 1e-9)
    ledger = {}
    cumulative = {}
    for label, folder in [
        ("Controlled Fortran: original rain", "outputs/phase4_reference_final/dt1"),
        (
            "Controlled Fortran: applied rain",
            "outputs/phase7/controlled_fortran_applied_rain",
        ),
    ]:
        a = read(folder + "/hydrograph.dat", skiprows=1, max_rows=5400)
        curves[label] = (a[:, 0], a[:, 1])
        ledger[label] = float(a[-1, 2])
    for label, folder in [
        ("SYRUP: fixed-terrain water only", "phase4_storm/numba_dt1"),
        ("SYRUP: evolving terrain/sediment", "phase5_storm/dt1"),
    ]:
        name = f"outputs/{folder}/hydrograph.npz"
        p = ROOT / name
        hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        with np.load(p) as a:
            curves[label] = (a["t_s"].copy(), a["outlet_discharge_m3_s"].copy())
            ledger[label] = float(a["cumulative_export_m3"][-1])
            cumulative[label] = a["cumulative_export_m3"].copy()
    summaries = {}
    for label, (t, q) in curves.items():
        assert np.isfinite(q).all()
        dt = np.diff(t, prepend=0)
        assert np.all(dt > 0)
        dense = bool(np.all(dt == 1))
        summaries[label] = {
            "sum_endpoint_Q_dt_m3": float(np.sum(q * dt)) if dense else None,
            "saved_output_interval_s": np.unique(dt).tolist(),
            "conservative_or_CN_ledger_export_m3": ledger.get(label),
            "endpoint_Q_integral_through_1620s_m3": float(
                np.sum(q[t <= 1620] * dt[t <= 1620])
            )
            if dense
            else None,
            "endpoint_Q_integral_after_1620s_m3": float(
                np.sum(q[t > 1620] * dt[t > 1620])
            )
            if dense
            else None,
            "peak_Q_m3_s": float(q.max()),
            "first_printed_peak_s": float(t[q.argmax()]),
        }
    x = curves["Full MAHLERAN: constant K"][1]
    y = curves["Controlled Fortran: applied rain"][1]
    ks = read(
        "outputs/phase7/mahleran_fixed_no_splash/Output/ksat_001.asc", skiprows=6
    )[1:-1, 1:-1]
    kd = read(
        "outputs/phase7/mahleran_deterministic_ksat_run/Output/ksat_001.asc", skiprows=6
    )[1:-1, 1:-1]
    assert np.all(kd == 0.00025)
    report = {
        "scope": "Saved-output comparison and MAHLERAN-only experiments; no new SYRUP run.",
        "input_sha256": hashes,
        "curves": summaries,
        "sampled_K_mm_s": {
            "min": float(ks.min()),
            "mean": float(ks.mean()),
            "max": float(ks.max()),
        },
        "constant_K_mm_s": float(kd[0, 0]),
        "full_constant_vs_controlled_applied_rain": {
            "max_absolute_Q_difference_m3_s": float(np.max(np.abs(x - y))),
            "Q_rmse_m3_s": float(np.sqrt(np.mean((x - y) ** 2))),
            "endpoint_integral_relative_difference": float(
                (x.sum() - y.sum()) / y.sum()
            ),
            "full_printed_peak_times_s": curves["Full MAHLERAN: constant K"][0][
                x == x.max()
            ].tolist(),
        },
        "limitations": [
            "MAHLERAN prints four significant figures; comparisons are not bitwise.",
            "Endpoint discharge integrals and CN/conservative export ledgers are distinct.",
            "SYRUP curves use original interval rainfall, not legacy applied rainfall.",
            "No full sediment-storm equivalence or GPU evidence from these experiments.",
        ],
    }
    (OUT / "initialization_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for label, (t, q) in curves.items():
        axes[0].plot(t, q * 1000, label=label, linewidth=1.3)
        axes[1].plot(
            t,
            cumulative.get(label, np.cumsum(q * np.diff(t, prepend=0))),
            linewidth=1.3,
        )
    axes[0].set_ylabel("Outlet discharge (L/s)")
    axes[1].set_ylabel("Outlet volume (m³; see report for definitions)")
    axes[1].set_xlabel("Time (s)")
    for ax in axes:
        ax.axvline(1620, color="gray", linestyle=":", linewidth=1)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Plot1: conductivity initialization explains the large runoff difference"
    )
    fig.tight_layout()
    fig.savefig(OUT / "initialization_comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
