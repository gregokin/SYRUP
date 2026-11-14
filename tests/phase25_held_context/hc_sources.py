"""Pinned source constants of the dynamic-vegetation held-context suite (a UNIQUE module name: never imported through `conftest`, so another suite's conftest cannot shadow it under any import mode). The expected MAPLE root /
package are configurable (`HELD_CONTEXT_MAPLE_ROOT`, `HELD_CONTEXT_MAPLE_PACKAGE`; default the CPU pin; the actual-GPU pin has the same package digest), the LEAF package is the FINAL frozen H3."""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ROOT = Path("/home/okin/SYRUP/agent_handoffs/tasks/phase1_publication_c1_20261112/runs/runtime-store-cpu/objects/maple-materialized/5d850789f9e82dc456227cb1eb4dfcc255b3ee4dfb0ed74883813c93e98d0ab3")
MAPLE_ROOT = Path(os.environ.get("HELD_CONTEXT_MAPLE_ROOT", str(DEFAULT_ROOT)))
MAPLE_PACKAGE = os.environ.get("HELD_CONTEXT_MAPLE_PACKAGE", "fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e")
LEAF_PACKAGE = "38009872f8969f86b56c720adc1b5d9ffefc554fb9cacf90855d999db604530f"
