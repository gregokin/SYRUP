# Scientific source check

Read-only audit, 20 November 2026. No hydrology rerun or parameter adoption.

Saxton & Rawls (2006), https://doi.org/10.2136/sssaj2005.0117,
Table 1, equations 2–12: the implemented density-adjusted 33-kPa anchor
and the linear 33-kPa-to-air-entry branch agree with the printed equations.
Table 2 explicitly identifies field capacity with 33-kPa water content.
Table 3's sand and loamy-sand examples give approximately 10% and 12%
at that pressure, with 2.5% organic matter; these are not JORN measurements.
Root checked the original local PDF text as well as the primary PDF web
extraction. The web extraction has sign errors in equation 3; it was not
used to diagnose a model defect. No PDF page image or Word pagination
renderer was available; no visual page-rendering claim is made.

Tietjen, Zehe & Jeltsch (2009), https://doi.org/10.1029/2007WR006589,
section 3.1, paragraph 25: field capacity was supplied from similar-site
field observations and associated with matric potential −6 kPa. Tietjen
did not prescribe the Saxton–Rawls wet-end interpolation for these sites.

NEON DP1.00096.001, RELEASE-2026:
https://data.neonscience.org/data-products/DP1.00096.001/RELEASE-2026
Original downloaded tables and their variable definitions establish the
actual JORN texture, density, estimated carbon, horizon depths and sand
size boundaries. Exact selected rows, variable units and SHA256 hashes
are recorded in root_audit.json. The middle model layer is a derived
raw-density-weighted mixture, not a third measured sample.

Inference from our arithmetic: combining the 6-kPa convention with the
linear wet branch and study-specific zero air-entry floor puts the
drainage threshold near saturation. This reproduces the archived inputs
but does not establish a physically credible JORN field capacity. The
site parameter/curve choice requires qualification before further field
interpretation. Neither the scientific pressure choice nor earlier
calibrations are changed by this audit.
