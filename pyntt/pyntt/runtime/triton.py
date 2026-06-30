"""Triton runtime helpers for generated PyNTT models."""

from __future__ import annotations

from typing import Optional

_TRITON_ALLOCATOR_INSTALLED = False


def ensure_triton_allocator(device: Optional[object] = None) -> None:
    """Install a default Triton scratch allocator backed by torch tensors."""
    global _TRITON_ALLOCATOR_INSTALLED
    if _TRITON_ALLOCATOR_INSTALLED:
        return

    import torch
    import triton

    allocation_device = torch.device(device) if device is not None else None

    def _alloc(size: int, _alignment: int, _stream: Optional[int]):
        target_device = allocation_device
        if target_device is None:
            if not torch.cuda.is_available():
                raise RuntimeError("Triton scratch allocation requires a CUDA device.")
            target_device = torch.device("cuda", torch.cuda.current_device())
        return torch.empty((size,), dtype=torch.uint8, device=target_device)

    triton.set_allocator(_alloc)
    _TRITON_ALLOCATOR_INSTALLED = True
