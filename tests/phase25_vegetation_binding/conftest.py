"""S0 vegetation-binding fixtures. LEAF is NEVER faked: when the actual package cannot be imported the consumer tests skip (no claim)."""
from __future__ import annotations

import pytest
import veg_binding_support  # noqa: F401  (puts the actual LEAF root on sys.path)


@pytest.fixture(scope="session")
def leaf():
    try:
        import leaf as pkg  # noqa: F401
        from leaf import state, traits
        from leaf import water_vegetation as wv
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the actual LEAF package is not importable ({type(exc).__name__}: {exc}); the consumer is UNQUALIFIED (no claim)")
    return {"state": state, "traits": traits, "wv": wv}
