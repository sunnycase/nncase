"""Workspace allocation helpers for generated PyNTT models."""

from __future__ import annotations

import base64
from typing import Any

from pyntt.runtime.errors import PyNTTSpecError
from pyntt.runtime.tensor import _torch_dtype


def allocate_workspace(inputs: tuple[Any, ...], elements: int, dtype: str):
    """Allocate a one-dimensional workspace tensor next to runtime inputs."""
    import torch

    device = inputs[0].device if inputs else "cuda"
    return torch.empty((int(elements),), dtype=_torch_dtype(torch, dtype), device=device)


def materialize_rdata(inputs: tuple[Any, ...], payload: str, byte_count: int):
    """Materialize one readonly data payload as a CUDA uint8 tensor."""
    raw = _decode_payload(payload, byte_count)
    return _bytes_to_tensor(inputs, raw)


def materialize_rdata_table(inputs: tuple[Any, ...], payloads: tuple[str, ...], bytes_per_entry: int):
    """Materialize per-shard readonly data payloads as one flat uint8 tensor."""
    if bytes_per_entry == 0:
        return allocate_workspace(inputs, 0, "uint8")

    entries = [_decode_payload(payload, bytes_per_entry) for payload in payloads]
    return _bytes_to_tensor(inputs, b"".join(entries))


def _decode_payload(payload: str, byte_count: int) -> bytes:
    if byte_count == 0:
        if payload:
            raise PyNTTSpecError("PyNTT rdata payload is non-empty for a zero-sized section.")
        return b""

    raw = base64.b64decode(payload.encode("ascii"))
    if len(raw) != int(byte_count):
        raise PyNTTSpecError(
            f"PyNTT rdata payload size mismatch: expected {byte_count} bytes, got {len(raw)}."
        )
    return raw


def _bytes_to_tensor(inputs: tuple[Any, ...], raw: bytes):
    import torch

    device = inputs[0].device if inputs else "cuda"
    if not raw:
        return torch.empty((0,), dtype=torch.uint8, device=device)

    host = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
    return host.to(device=device)
