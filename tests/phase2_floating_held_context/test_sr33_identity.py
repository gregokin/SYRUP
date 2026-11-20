"""NEW (task sr33_default_20261120): an old/unversioned held-context record cannot be reinterpreted under the SR33 default (no migration, no rebinding), and a new-default record round-trips."""
from __future__ import annotations

import copy
import hashlib

import hbuild as hb
import numpy as np
import pytest
from hbuild import AVAIL, expected_of, make

from maple_syrup import floating_held_context as fh
from maple_syrup import native_floating_soil as nfs
from maple_syrup import soil_texture_properties as stp


def pre_version_sha(cfg):
    """The config sha256 recipe of the source BEFORE the field-capacity version tag existed (same inputs, no `|field_capacity_method:` suffix)."""
    h = hashlib.sha256()
    for a in (cfg.layers.depth_edges_m, cfg.grain_texture, cfg.organic_matter_percent, cfg.latent_theta_field_capacity, cfg.latent_theta_saturation, cfg.diffusivity_m2_s,
              np.array([cfg.bulk_density_kg_m3, cfg.theta_residual, cfg.theta_evaporation_stress, cfg.evaporation_factor, cfg.matrix_conductivity_m_s, cfg.matrix_suction_m,
                        cfg.matrix_vegetation_factor, cfg.initial_wetting_m, cfg.initial_cumulative_intake_m])):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(f"{nfs.SCHEMA}|{cfg.bottom_drainage}|{cfg.backend}".encode())
    return h.hexdigest()


def test_new_default_record_round_trips_and_an_old_unversioned_identity_is_refused():
    vert, owner, _bed = make()
    rec = fh.export_held_context(vert, None, available_bytes_fn=AVAIL)
    meta, arrays = hb.deserialise(*hb.serialise(rec))
    parsed = fh.record_from_parts(meta, arrays, available_bytes_fn=AVAIL)
    ctx = fh.reconstruct_held_context(parsed, expected_of(vert), available_bytes_fn=AVAIL)
    assert ctx.config.sha256 == owner.config.sha256 and ctx.config.field_capacity_method == stp.METHOD_SR33
    old = pre_version_sha(owner.config)
    assert old != owner.config.sha256
    m2 = copy.deepcopy(meta)
    m2["identities"]["config_sha256"] = old  # a record written before the version tag: its recorded identity cannot be reproduced under the new default
    p2 = fh.record_from_parts(m2, arrays, available_bytes_fn=AVAIL)
    import dataclasses

    exp_old = dataclasses.replace(expected_of(vert), config_sha256=old)
    with pytest.raises(fh.HeldContextError):
        fh.reconstruct_held_context(p2, exp_old, available_bytes_fn=AVAIL)
    with pytest.raises(fh.HeldContextError):  # and the old identity never satisfies the NEW expected identities
        fh.reconstruct_held_context(p2, expected_of(vert), available_bytes_fn=AVAIL)
