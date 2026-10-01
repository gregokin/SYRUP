"""Measure first-cell survival of the actual operator against an exponential walk.

Controlled constant-law impulse, not a Plot1 simulation or a legacy execution.
A unit impulse starts in the outlet cell of a three-cell chain; export is the probability of crossing its downstream
face before deposition. The legacy flow_distrib convention assigns distance dx
inside the pickup cell, with exact survival exp(-dx/L). Upwind cell mixing has
the different continuous-time limit L/(L+dx). Run with the pinned MAPLE env.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from maple_syrup.provenance import source_tree_digest
from maple_syrup.routing import build_routing_graph
from maple_syrup.sediment_transport import transport_network, transport_step


def measure(dx: float, distance: float, dt: float) -> dict:
    z = np.repeat(np.arange(5, dtype=float)[:, None] * 0.015625, 3, axis=1)
    z[:, [0, -1]] += 1.0
    outlets = np.zeros((5, 3), dtype=bool)
    outlets[0] = True
    graph = build_routing_graph(z, outlets, np.ones((3, 1)), dx)
    network = transport_network(graph)
    mobile = np.zeros((3, 1, 1))
    mobile[0] = 1.0
    velocity = np.full_like(mobile, 0.01)
    rate = np.full_like(mobile, 1.0 / distance)
    settle = np.zeros_like(mobile, dtype=bool)
    exported = deposited = 0.0
    steps = 0
    while mobile.sum() > 1e-14:
        result = transport_step(network, mobile, velocity, rate, settle, dt)
        exported += float(result.export_request_kg.sum())
        deposited += float(result.deposition_request_kg.sum())
        mobile = result.mobile_after_transfer_kg - result.export_request_kg - result.deposition_request_kg
        steps += 1
        if steps > 100000:
            raise RuntimeError("impulse did not settle")
    # Independent scalar infinite geometric series for reaction/advection/reaction.
    courant = 0.01 * dt / dx
    half_survival = np.exp(-0.01 * dt / (2 * distance))
    geometric = courant * half_survival / (1 - (1 - courant) * half_survival ** 2)
    np.testing.assert_allclose(exported, geometric, rtol=1e-11, atol=1e-14)
    np.testing.assert_allclose(exported + deposited + mobile.sum(), 1.0, rtol=0, atol=2e-13)
    return {"dx_m": dx, "mean_distance_m": distance, "dt_s": dt, "steps": steps,
            "actual_export_fraction": exported, "actual_deposited_fraction": deposited,
            "remaining_fraction": float(mobile.sum()), "geometric_series_export_fraction": float(geometric),
            "legacy_distance_convention_survival": float(np.exp(-dx / distance)),
            "upwind_dt_to_zero_survival": distance / (distance + dx)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    rows = [measure(0.5, distance, dt) for distance in (0.01, 0.05, 0.1) for dt in (1.0, 0.25)]
    report = {"scope": __doc__, "source_sha256": source_tree_digest(Path("src/maple_syrup")).digest_sha256,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "rows": rows}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
