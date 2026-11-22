"""MOAB soil properties for the dry-window comparison: the CURRENT native Saxton-Rawls provider (default 33 kPa field capacity) applied to the ORIGINAL MOAB texture, organic matter and measured bulk density.

MOAB keeps its OWN field capacity (not JORN's 10 vol. %) and its historical SR-derived Ks, saturation and wilting point. Inputs are the frozen audit rows (`site_properties_mixing/data/property_audit/root_probe.py`, SHA pinned
below): MOAB_A and MOAB_Bw fractions/OM/density, layer 0 = the 0.6 depth-fraction dry-mass mix of MOAB_A and MOAB_Bw (receipt record MOAB_0_10_dry_mass), layers 1-2 = MOAB_Bw. The raw MOAB air entries are positive
and valid, so no air-entry override or floor is needed (the zero floor of the JORN studies applies only where the raw value is negative); the measured densities are above the 0.901 x normal-density rule, so the
effective density equals the raw one. The result is compared with the historical template and with the frozen receipt's theta33; it is NOT the old 6 kPa field capacity of the historical templates.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

CANDIDATE = Path(__file__).resolve().parents[2]
AUDIT = CANDIDATE / "docs/subsurface_hydrology/site_properties_mixing/data/property_audit/root_probe.py"
AUDIT_SHA256 = "2218c3caf583c298350b9057d2fc9e170810e778fe0986c19621819e1d3f960e"
RECEIPT = CANDIDATE / "docs/subsurface_hydrology/site_properties_mixing/data/root_relative_density_probe.json"
RECEIPT_SHA256 = "9ad303384f26116f37088aa6609d3519c7396d22d46d8d0ac42919358165675f"
MOAB_A = {"S": 0.730, "Si": 0.197, "C": 0.073, "OM": 1.0344, "rho": 1519.7}       # frozen audit row MOAB_A (fractions; OM percent by weight; kg/m3)
MOAB_BW = {"S": 0.712, "Si": 0.170, "C": 0.118, "OM": 0.5172, "rho": 1481.8}      # frozen audit row MOAB_Bw
DEPTH_FRACTION_A = 0.6                                                            # 0-10 cm layer = 0.6 MOAB_A + 0.4 MOAB_Bw by depth, dry-mass weighted (record MOAB_0_10_dry_mass)


def layer_inputs() -> list:
    rho = DEPTH_FRACTION_A * MOAB_A["rho"] + (1 - DEPTH_FRACTION_A) * MOAB_BW["rho"]
    wa = DEPTH_FRACTION_A * MOAB_A["rho"] / rho
    mix = {k: wa * MOAB_A[k] + (1 - wa) * MOAB_BW[k] for k in ("S", "Si", "C", "OM")} | {"rho": rho}
    return [mix, dict(MOAB_BW), dict(MOAB_BW)]


def moab_properties() -> dict:
    """The native provider's default-method outputs for the three MOAB layers (numpy reference backend)."""
    assert hashlib.sha256(AUDIT.read_bytes()).hexdigest() == AUDIT_SHA256 and hashlib.sha256(RECEIPT.read_bytes()).hexdigest() == RECEIPT_SHA256
    from maple_syrup import soil_texture_properties as st
    L = layer_inputs()
    r = st.estimate_soil_texture_properties(sand_fraction=np.array([x["S"] for x in L]), silt_fraction=np.array([x["Si"] for x in L]), clay_fraction=np.array([x["C"] for x in L]),
                                            organic_matter_percent=np.array([x["OM"] for x in L]), bulk_density_kg_m3=np.array([x["rho"] for x in L]), n_layers=3, n_cells=1, backend="numpy", diagnostics=True,
                                            memory_budget_bytes=10**8)
    out = {n: r.array(n).tolist() for n in ("theta_field_capacity", "theta_saturation", "theta_1500", "saturated_conductivity_m_s", "air_entry_kpa")}
    out["inputs"], out["provider_input_sha256"], out["recipe"] = L, r.sha256, st.recipe_record()["recipe"]
    return out
