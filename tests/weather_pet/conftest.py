"""Fixtures of the weather/PET suite. The `gpu` fixture skips honestly (no claim) when CuPy or a CUDA device is
unavailable; it imports CuPy directly so that this suite never imports the native MAPLE package. Select the device
with CUDA_VISIBLE_DEVICES before the process starts; GPU 0 is the user allocation and launches are Codex's call."""
from __future__ import annotations

import pytest


def _cupy_with_device():
    try:
        import cupy  # type: ignore[import-not-found]
    except ImportError:
        return None, "CuPy is not importable"
    try:
        count = cupy.cuda.runtime.getDeviceCount()
    except Exception as exc:  # noqa: BLE001 - CuPy raises its own runtime error types; any of them means "no device"
        return None, f"CUDA runtime unavailable: {exc}"
    if count < 1:
        return None, "no CUDA device is visible"
    return cupy, ""


@pytest.fixture(scope="session")
def gpu():
    cupy, reason = _cupy_with_device()
    if cupy is None:
        pytest.skip(f"{reason}; the CUDA weather/PET path is not exercised (no claim)")
    return cupy


@pytest.fixture(scope="session")
def gpu_unavailable_reason() -> str:
    """Non-empty exactly when no GPU is available (for the one refusal test that must run WITHOUT a device)."""
    cupy, reason = _cupy_with_device()
    return reason if cupy is None else ""
