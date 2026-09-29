"""Bounded diagnostic of the existing local water accounting seam."""
import sys
sys.path.insert(0, '/home/okin/MAPLE')
import numpy as np
from tests.unit._phase19_helpers import build_geometry, build_bed, MASS_RESOLUTION_KG
from maple.water.interfaces import WaterProcessDemand
from maple.water.step import apply_water_process_demand

geometry = build_geometry(ny=1, nx=2)
classes, column, layer, ledger, water = build_bed(geometry)
removal = np.array([[[0.01], [0.0]]])
deposition = np.array([[[0.0], [0.01]]])
result = apply_water_process_demand(column, layer, water, ledger,
    WaterProcessDemand(removal, deposition), geometry, classes, MASS_RESOLUTION_KG)
print('actual removal by cell:', result.actual_removal_by_cell_class_kg.ravel().tolist())
print('actual deposition by cell:', result.deposition_by_cell_class_kg.ravel().tolist())
print('remaining mobile by cell:', result.new_water.mobile_mass_by_cell_class_kg.ravel().tolist())
assert np.allclose(result.actual_removal_by_cell_class_kg, removal)
assert np.all(result.deposition_by_cell_class_kg == 0)
assert np.allclose(result.new_water.mobile_mass_by_cell_class_kg, removal)
