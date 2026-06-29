"""PyNTT backend lookup."""

from __future__ import annotations

from functools import lru_cache

from pyntt.runtime.errors import PyNTTBackendError


@lru_cache(maxsize=None)
def get_backend(name: str):
    """Return a PyNTT backend implementation by name."""
    if name == "triton":
        from .triton.backend import TritonBackend

        return TritonBackend()

    raise PyNTTBackendError(f"Unsupported PyNTT backend: {name}")


__all__ = [
    "get_backend",
]
