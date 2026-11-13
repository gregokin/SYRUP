# Portable independent controls (byte copies, not re-derived)

`root_capacity_exact_controls.json` (exact rational FC-cascade/saturation-return controls) and `root_d1_porous_geometry_controls.json` (exact Fraction gravity controls over partial/zero porous
slots and Decimal80 partial-pair relaxation controls) were written by Root before the D1 implementation, without model code, from the approved decisions. They are mathematical controls of the
agreed laws, not field validation. The tests read only these copies (no dependency on the task directory).
