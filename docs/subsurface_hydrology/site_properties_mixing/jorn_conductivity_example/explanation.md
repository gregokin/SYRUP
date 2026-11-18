# How SYRUP gets JORN's conductivities

*Saxton–Rawls Ks, the EcoHyD within-layer adjustment, and layer-to-layer averaging, with one worked example · JORN 0–10 cm A horizon and 10–20 cm layer · documentation only, no model run · 2026-11-17*

## 1. What this note answers

Three different conductivities appear in the SYRUP JORN work, and they are easy to mix up. They are computed in sequence, and each is a distinct physical quantity: (a) the saturated conductivity Ks of each layer, from the Saxton and Rawls (2006) pedotransfer equations; (b) a representative unsaturated conductivity K inside each layer, from the Tietjen/EcoHyD rule K = f·Ks; and (c) one conductivity per interface between adjacent layers, from a SYRUP experiment rule (weighted harmonic or geometric mean). Only (a) is Saxton–Rawls. Step (b) is Tietjen's, step (c) is neither.

- **Surface Ks:** 120.78652 mm/h = 3.3552×10^-5 m/s for the A horizon (0–10 cm).
- **Within-layer K:** f = 0.31635739, so K₁ = 38.2117 mm/h (layer 1) and K₂ = 37.8704 mm/h (layer 2).
- **Interface:** harmonic 38.0403061 mm/h, geometric 38.0406888 mm/h. They are nearly identical because the two neighbours are nearly identical.
- **Texture alone is not what this example uses.** The density-adjusted Ks below uses texture plus an organic-matter value plus a reported bulk density. Without a reported density one could use the regression's own normal density, but that is a different assumption and a different Ks. An interface value further needs the neighbouring layer's properties and the layer geometry.

## 2. Where the Saxton–Rawls equations are

Saxton KE, Rawls WJ (2006), Soil Water Characteristic Estimates by Texture and Organic Matter for Hydrologic Solutions, Soil Sci. Soc. Am. J. 70:1569–1578, DOI 10.2136/sssaj2005.0117. The equations used here are in the paper's Table 1 (printed page 1571, PDF page 3). The table below maps each equation number to its role and to the line in SYRUP's accepted recipe that evaluates it. Nothing in this note reproduces the paper's figures or text; equations are restated with their coefficients, as the code uses them.

*Table 1. Formula map. Eq numbers are the paper's. “Code” is soil_texture_properties.py in this candidate.*

| Eq | Quantity | Role in Ks chain | Code |
|---|---|---|---|
| 1 | θ₁₅₀₀ wilting (1500 kPa) | Lower anchor | EQ1_* |
| 2 | θ₃₃,N normal 33 kPa | Upper anchor, via Eq 9 | EQ2_* |
| 3 | g (intermediate) | Ingredient of Eq 5 only | EQ3_* |
| 5 | θS,N normal saturation | Gives Eq 6, Eq 9 | EQ3_SAT_TERMS |
| 6 | ρN normal density | Reference for DF | rho_n |
| 7 | DF = ρDF/ρN | Reported, range check | df |
| 8 | θS,DF density-adjusted | Saturation in Eq 16 | sat_d |
| 9 | θ33,DF density-adjusted | 33 kPa anchor | fc_d |
| 15 | B retention slope | From 33 and 1500 kPa | B |
| 18 | λ = 1/B | Exponent 3 − λ | lam |
| 16 | Ks (mm/h) | Result | ks_est |
| 4, 17 | Air entry; K(θ) | Not in Ks chain | — |

Two further paper equations are deliberately outside the Ks chain: Eq 4 (air-entry tension) and Eq 17 (moisture-dependent conductivity K(θ)). Section 6 explains why.

## 3. Inputs for the JORN surface layer

The whole 0–10 cm layer is the A horizon. Texture (sand, silt, clay) and bulk density are the reported inputs. Organic matter is not a measurement: 0.3448 % is a provisional derived assumption, converted as 1.724 × estimated organic carbon (g/kg) / 10. No hydraulic property here is measured. OM enters the paper's equations as percent of dry mass, so the number 0.3448 is used (not 0.003448). Soil-moisture quantities (θ) are volume fractions in m³/m³. The code takes density in kg/m³ and divides by 1000 to the paper's g/cm³.

*Table 2. Inputs, JORN A horizon: texture and density as reported; OM a provisional derived assumption, not a measurement.*

| Quantity | Value | Unit / note |
|---|---|---|
| Sand S | 0.867 | mass fraction |
| Silt Si | 0.062 | not used by Eq 1–16 |
| Clay C | 0.071 | mass fraction |
| Organic matter OM | 0.3448 | derived, provisional: 1.724 × est. OC (g/kg) / 10; percent of dry mass |
| Bulk density | 1419.6 | reported; kg/m³ = 1.4196 g/cm³ |

## 4. Step A: retention anchors and density adjustment

### 4.1 Regression intermediates (Eq 1, 2, 3)

Each equation forms a first estimate t from the same seven terms (S, C, OM, their products S·OM, C·OM, S·C, and a constant); only the coefficients differ. The first estimate is then corrected to the final retention value. Table 3 shows every term for the JORN A horizon.

    t_1500 =−0.024 S + 0.487 C + 0.006 OM
    + 0.005 S·OM − 0.013 C·OM + 0.068 S·C + 0.031    [Eq 1a]

    θ_1500 = 1.14 t_1500 − 0.02    [Eq 1b]

    t_33 =−0.251 S + 0.195 C + 0.011 OM
    + 0.006 S·OM − 0.027 C·OM + 0.452 S·C + 0.299    [Eq 2a]

    θ_33_N = t_33 + 1.283 t_33^(2) − 0.374 t_33 − 0.015    [Eq 2b]

    t_g =0.278 S + 0.034 C + 0.022 OM
    − 0.018 S·OM − 0.027 C·OM − 0.584 S·C + 0.078    [Eq 3a]

    g = 1.636 t_g − 0.107    [Eq 3b]

*Table 3. Term-by-term evaluation (S = 0.867, C = 0.071, OM = 0.3448, S·OM = 0.298942, C·OM = 0.024481, S·C = 0.061557).*

| Term | Value | Eq 1 coef → term | Eq 2 coef → term | Eq 3 coef → term |
|---|---|---|---|---|
| S | 0.867000 | -0.024 → -0.020808 | -0.251 → -0.217617 | +0.278 → 0.241026 |
| C | 0.071000 | +0.487 → 0.034577 | +0.195 → 0.013845 | +0.034 → 0.002414 |
| OM | 0.344800 | +0.006 → 0.002069 | +0.011 → 0.003793 | +0.022 → 0.007586 |
| S·OM | 0.298942 | +0.005 → 0.001495 | +0.006 → 0.001794 | -0.018 → -0.005381 |
| C·OM | 0.024481 | -0.013 → -0.000318 | -0.027 → -0.000661 | -0.027 → -0.000661 |
| S·C | 0.061557 | +0.068 → 0.004186 | +0.452 → 0.027824 | -0.584 → -0.035949 |
| const. | 1.000000 | +0.031 → 0.031000 | +0.299 → 0.299000 | +0.078 → 0.078000 |
| sum |  | t₁₅₀₀ = 0.05220013 | t₃₃ = 0.12797723 | t_g = 0.28703438 |

Eq 1b gives the permanent-wilting-point fraction θ₁₅₀₀ (water held at 1500 kPa). With t₁₅₀₀ = 0.0522001336:

    θ_1500 = 1.14 × 0.0522001336 − 0.02 = 0.039508152304    [substituted]

Eq 2b gives the normal-density 33 kPa (field-capacity-type) content θ₃₃ₙ. With t₃₃ = 0.1279772320:

    θ_33_N =0.1279772320 + 1.283 × 0.1279772320^(2)
     − 0.374 × 0.1279772320 − 0.015 = 0.086126941793    [substituted]

Eq 3b gives g, the paper's intermediate for the saturation-minus-33 kPa gap at normal density. Its value is g = 0.3625882482976. It is only an ingredient of Eq 5. It is not the final θS − θ₃₃: after the density adjustment below, the final difference is a different number: 0.366837805659 here, against g = 0.362588248298.

    θ_S_N = θ_33_N + g − 0.097 S + 0.043    [Eq 5]

    θ_S_N =0.086126941793 + 0.3625882482976
     − 0.097 × 0.867 + 0.043 = 0.407616190091    [substituted]

In words: Eq 5 gives the saturated water content a soil of this texture would have at the paper's “normal” density. Eq 6 converts it to that normal bulk density:

    ρ_N = (1 − θ_S_N) × 2.65    [Eq 6]

    ρ_N = (1 − 0.407616190091) × 2.65 = 1.5698170962599 g/cm³    [substituted]

### 4.2 Density adjustment (Eq 7, 8, 9) and the effective density

The regression assumes normal density. The actual soil is looser or tighter, so saturation and field capacity are shifted. SYRUP uses the effective density, the largest of the raw density, 1000 kg/m³, and 0.901 times the normal density. This floor is applied after dry-mass mixing of the raw layer density, and the reported top-layer density is left unchanged when it is already above the floor. For the A horizon, 0.901 × 1.5698 = 1.4144 g/cm³ is below the reported 1.4196 g/cm³, so the reported value stands.

    (ρ_DF)/(ρ_N) = DF    [Eq 7 (reported only)]

    θ_S_DF = 1 − (ρ_DF)/(2.65)    [Eq 8]

    θ_33_DF = θ_33_N − 0.2 (θ_S_N − θ_S_DF)    [Eq 9]

In words: DF compares actual to normal density; Eq 8 turns the actual density into a porosity-based saturation (2.65 g/cm³ is particle density); Eq 9 lowers or raises field capacity by 20 % of the saturation shift.

    θ_S_DF = 1 − (1.4196)/(2.65) = 0.4643018867925    [substituted]

    θ_33_DF =0.086126941793 − 0.2 × (
    0.407616190091 − 0.464301886792) = 0.09746408113339    [substituted]

The density factor is DF = 0.9043091729490 (within the accepted 0.9–1.3 range). The retention values that feed conductivity are therefore θ₁₅₀₀ = 0.039508152304, θ₃₃,DF = 0.09746408113339, θS,DF = 0.4643018867925.

## 5. Step B: the saturated conductivity Ks

The paper fits a power-law retention curve through two anchors: 1500 kPa and 33 kPa. Eq 15 gives its slope B from them, Eq 18 inverts it to λ = 1/B, and Eq 16 turns the wet-end gap into Ks in mm/h.

    B = (ln1500 − ln33)/(lnθ_33_DF − lnθ_1500)    [Eq 15]

    λ = (1)/(B)    [Eq 18]

    K_s = 1930 (θ_S_DF − θ_33_DF)^(3 − λ) (mm/h)    [Eq 16]

In words: a wider gap between saturation and 33 kPa content means more water-filled pore space that drains freely, so Ks rises; because the gap is below 1, a larger exponent 3 − λ gives a smaller Ks. The 33 kPa anchor is the one the paper uses here; SYRUP's separate 6 kPa field capacity is not in this formula.

    B =(7.313220387 − 3.496507561)/(-2.328271367 − (-3.231248241))
     = (3.81671283)/(0.90297687) = 4.22681127    [substituted]

    λ = (1)/(4.22681127) = 0.23658497;  3 − λ = 2.76341503    [substituted]

    K_s =1930 × (0.464301887 − 0.097464081)^(2.763415)
    = 1930 × 0.366837806^(2.763415) = 120.78651559 mm/h    [substituted]

SYRUP stores conductivity in m/s. Numerically, divide the value in mm/h by 3 600 000 to get m/s (one mm/h is 10⁻³ m per 3600 s); the subscripts name the unit of each number.

    K_s,m/s = (K_s,mm/h)/(3 600 000) = (120.78651559)/(3 600 000) = 3.3552×10^(-5)    [unit conversion]

## 6. What does not enter Ks, and how bounds are treated

- **Field capacity at 6 kPa** (θ₆ = 0.3976041039) is a separate SYRUP quantity built from the air-entry estimate. It is not an input to Ks, which uses 33 kPa.
- **Air entry.** The Eq 4 prediction for this soil is -0.607908684 kPa, which is negative and physically meaningless. For the approved prescribed-surrogate case it is floored at 0 as an independent what-if. The floored value does not enter Ks.
- **Native strict provider is unchanged.** It still refuses a negative air-entry prediction. These JORN parameters are prescribed independently; they are not an automatic acceptance by the native provider, not measured hydraulics, and not a hidden override.
- **Eq 17**, the Saxton–Rawls moisture-dependent conductivity K(θ), is not used in this diagnostic, and nothing here asserts a full unsaturated conductivity curve or a dynamic K.

## 7. Step C: the Tietjen/EcoHyD within-layer adjustment

The next step is distinct from Saxton–Rawls. In the later EcoHyD model (WaterCell.cpp, which follows the conductivity discussion in Section 2.1 of Tietjen et al. 2009, DOI 10.1029/2007WR006589; code downloaded at commit 80b10e6, lines 549–552), the conductivity of a layer is the geometric mean of the saturated Ks and a representative unsaturated value Kᵤ, using Gardner's (1958) form with reference ratio 30/26 and exponent 3. This is a geometric adjustment between saturated and unsaturated conductivity within one layer. It is not the averaging of adjacent layers (Section 8).

    K_u = (K_s)/((1 + (30)/(26))^(3))    [EcoHyD]

    K = sqrt(K_sK_u) = f K_s,   f = (1 + (30)/(26))^(−3/2)    [EcoHyD]

Here f = (56/26)^(−1.5) = 0.31635738822500749 (exact arithmetic; double precision prints 0.31635738822500753). The model's separate dimensionless multiplier d is not in this reported K. Applying f to the surface Ks:

    K_1 = 0.3163573882 × 120.78651559 = 38.21170660 mm/h    [substituted]

## 8. Step D: conductivity between adjacent layers

Redistribution between two layers needs one K at their interface. This rule is a SYRUP experiment extension. It is not a Saxton–Rawls formula and is not the original Tietjen heterogeneous-layer rule. The weights come from half-layer distances: with thicknesses hᵢ and hⱼ, wᵢ = hᵢ/(hᵢ + hⱼ). The physical motivation of the harmonic form is flow in series through layers, where resistances add (e.g. Hoffmann, Ripich and Ellett 2002, USGS WRI 01-4257, Eq 5, p. 19; DOI 10.3133/wri014257). The geometric form is the log-average alternative. Both apply to the within-layer K values of Step C.

    w_i = (h_i)/(h_i + h_j),   w_j = (h_j)/(h_i + h_j)    [weights]

    K_H = (1)/((w_i)/(K_i) + (w_j)/(K_j))    [harmonic]

    K_G = exp(w_i ln K_i + w_j ln K_j)    [geometric]

In words: the harmonic mean is dominated by the less conductive layer; the geometric mean is the log-midpoint. For equal weights they reduce to 2KᵢKⱼ/(Kᵢ + Kⱼ) and √(KᵢKⱼ), and always satisfy harmonic ≤ geometric.

### 8.1 Building layer 2 (10–20 cm) from the A and Bw horizons

Layer 2 holds 8 cm of A horizon and 2 cm of Bw (S 0.837, Si 0.079, C 0.084, OM 0.1724 %, raw density 0.9035 g/cm³). Texture and OM are averaged with dry-mass weights, not 80/20 by depth: the A share is wA = 0.8 × 1.4196 / (0.8 × 1.4196 + 0.2 × 0.9035) = 0.8627296070. The raw layer density is the depth-weighted 0.8 × 1.4196 + 0.2 × 0.9035 = 1.31638 g/cm³ (1316.38 kg/m³).

*Table 4. Layer 2 mixed properties and Ks (same Eq 1–16 chain as Section 4–5).*

| Quantity | A horizon (layer 1) | Layer 2 (mixed) |
|---|---|---|
| Sand S | 0.867000 | 0.86288189 |
| Clay C | 0.071000 | 0.07278452 |
| OM (%) | 0.344800 | 0.32113458 |
| Raw density (kg/m³) | 1419.6 | 1316.38 |
| Normal density ρN (g/cm³) | 1.56981710 | 1.57302571 |
| Effective density (kg/m³) | 1419.6000 | 1417.2961608 |
| θ₁₅₀₀ | 0.039508152 | 0.040438299 |
| θ₃₃,DF | 0.097464081 | 0.099395088 |
| θS,DF | 0.464301887 | 0.465171260 |
| Ks (mm/h) | 120.78651559 | 119.70776661 |
| K = f·Ks (mm/h) | 38.21170660 | 37.87043640 |

The effective density rule acts here: 0.901 × 1.5730257056633 g/cm³ = 1.4172961608026 g/cm³ = 1.4172961608026 × 1000 = 1417.2961608026 kg/m³. This exceeds the mixed raw density 1.31638 g/cm³ (1316.38 kg/m³) and the 1000 kg/m³ floor, so the effective density is 1417.2961608026 kg/m³. The resulting Ks₂ is 119.70776661 mm/h, and K₂ = f·Ks₂ = 37.87043640 mm/h.

### 8.2 The interface

Both layers are 10 cm thick, so wᵢ = wⱼ = 0.5.

    K_H =(1)/((0.5)/(38.211707) + (0.5)/(37.870436))
    = 38.0403061077 mm/h    [substituted]

    K_G = exp(0.5 ln 38.211707 + 0.5 ln 37.870436) = 38.0406888019 mm/h    [substituted]

The two results differ by only 0.000383 mm/h (0.00101 %) because the neighbours are alike. The choice of mean matters only for contrast. As pure arithmetic, not a site property: 38.2 mm/h above 3.82 mm/h gives harmonic 6.945 and geometric 12.080 mm/h. An interface value cannot be derived from the surface texture alone: it needs the lower layer's properties and the layer geometry.

## 9. Note on the 0.18 wilting-point parameter at Tabernas

Tietjen et al. (2009, Section 3.1) tied that model's site thresholds to −1500 kPa. The later Tabernas wilting value 0.18 is explicitly labelled a calibration in Baldauf's dissertation (Table A4.3, printed p. 124; bounds 0.08–0.18 in Table A4.1, p. 123, described as literature and standard values for silty loam), so it should not be read as an established measured θ₁₅₀₀. It does not enter Ks. The parameter file's “loamy sand” header also conflicts with field literature describing predominantly silty soils; the header is not verified texture. Baldauf et al. (2023, Front. Microbiol., Sec. 2.3.3) calibrate the wilting point against soil moisture at 3 and 10 cm. This note does not fit soil parameters and makes no claim about the exact pressure behind 0.18.

## 10. Limits

- Arithmetic and documentation only. No model, GPU or soil-water simulation was run, and no production source, test, reference or raw data were changed.
- The Ks values are pedotransfer estimates, not measured conductivity. Saxton–Rawls is an empirical regression; its Ks has large uncertainty for individual soils.
- Admission issue: the bulk density and density factor DF are inside the accepted bounds after the declared effective-density floor, but the Eq 4 regression predicts a negative air-entry tension for this soil, so the full native provider refuses the curve. The explicit zero floor is a labelled case bound for this prescribed surrogate; it does not enter Ks and does not make the native provider accept the soil.
- The interface rule is an experiment choice; its adequacy for JORN is not established.

## Appendix. Source identities

*Table 5. Sources, lines and SHA-256. Paths are relative to the candidate root unless absolute.*

| Item | Path | Lines / note | SHA-256 |
|---|---|---|---|
| Saxton–Rawls 2006 PDF (Table 1, printed p.1571 = PDF p.3) | /home/okin/SYRUP/agent_handoffs/tasks/neon_texture_failure_audit_20261117/saxton_rawls_2006.pdf | hash verified; location per assignment | 24038366b95d508d…7186e5 |
| Coefficients Eq 1–4, constants | src/maple_syrup/soil_texture_properties.py | L106–125 | ff737938a1c3fc4c…c20481 |
| Eq 1–6 normal regressions | same | L376–386 | (same file) |
| Eq 7–9 density adjustment | same | L388–390 | (same file) |
| Eq 15, 18, 16 Ks | same | L404–406 | (same file) |
| Interface rule, EcoHyD factor | benchmarks/subsurface_site_comparison/interfaces.py | L16, L39–52 | 02dfd1bd6f76d14d…f2c09b |
| EcoHyD Ku, K_geom | /home/okin/SYRUP/reference_models/ecohyd/github_baldauf_2023/source/model/src/WaterCell.cpp | L549–552 | af476e1aa727ed51…01d9e9 |
| Prior probe values (cross-check) | docs/subsurface_hydrology/site_properties_mixing/data/root_relative_density_probe.json | records JORN_A, JORN_10_20_dry_mass | 9ad303384f26116f…65675f |
| Property audit probe (cross-check) | …/data/property_audit/root_probe.json |  | 7b18b863faace9c9…5bcf36 |

Saxton KE, Rawls WJ (2006). SSSAJ 70:1569–1578, DOI 10.2136/sssaj2005.0117; https://rashms.com/wp-content/uploads/2021/08/soil-characteristics-by-texture-and-organic-rawls.pdf. Tietjen B et al. (2009) DOI 10.1029/2007WR006589. Baldauf S (dissertation), https://refubium.fu-berlin.de/bitstream/handle/fub188/37307/Dissertation_SelinaBaldauf.pdf. Baldauf et al. (2023) DOI 10.3389/fmicb.2023.1179291. Hoffmann JP, Ripich MA, Ellett KM (2002) USGS WRI 01-4257, DOI 10.3133/wri014257.
