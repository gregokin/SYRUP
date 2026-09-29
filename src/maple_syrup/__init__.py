"""MAPLE-SYRUP: Sediment Yield, Runoff, and Uptake by Plants.

A water extension that depends on the actual MAPLE framework. Wind
physics, the shared sediment bed, grain classes, ledger, topographic
commits, snapshots and backend all stay in MAPLE; this package adds
water-specific processes on top of them.

Phase 1 content only: dependency resolution and identity checks
(`dependency`), bounded source provenance (`provenance`), and a minimal
integration probe that builds real MAPLE state (`probe`). No rainfall,
infiltration, routing or sediment physics exists here yet.
"""

__version__ = "0.0.1.dev0"
