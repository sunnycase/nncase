"""Workspace and readonly-data helpers for generated PyNTT models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pyntt.runtime.errors import PyNTTSpecError
from pyntt.runtime.tensor import _torch_dtype


def _import_torch():
    import torch

    return torch


def _normalize_device(inputs: tuple[Any, ...] = (), device: Any | None = None):
    torch = _import_torch()
    if device is not None:
        resolved = torch.device(device)
    else:
        resolved = None
        for value in inputs:
            input_device = getattr(value, "device", None)
            if input_device is not None:
                resolved = torch.device(input_device)
                break
        if resolved is None:
            if torch.cuda.is_available():
                resolved = torch.device("cuda", torch.cuda.current_device())
            else:
                resolved = torch.device("cpu")

    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def _device_key(device: Any) -> str:
    resolved = _normalize_device(device=device)
    if resolved.index is None:
        return resolved.type
    return f"{resolved.type}:{resolved.index}"


class WorkspacePool:
    """Reusable torch workspace buffers keyed by logical workspace name."""

    def __init__(self):
        self._buffers: dict[tuple[str, str, str], Any] = {}

    def allocate(
        self,
        inputs: tuple[Any, ...],
        key: str,
        elements: int,
        dtype: str,
        device: Any | None = None,
    ):
        torch = _import_torch()
        resolved_device = _normalize_device(inputs, device)
        torch_dtype = _torch_dtype(torch, dtype)
        element_count = int(elements)
        cache_key = (str(key), _device_key(resolved_device), str(torch_dtype))
        existing = self._buffers.get(cache_key)
        if existing is None or existing.numel() < element_count:
            existing = torch.empty((element_count,), dtype=torch_dtype, device=resolved_device)
            self._buffers[cache_key] = existing
        if existing.numel() == element_count:
            return existing
        return existing.narrow(0, 0, element_count)


class RDataCache:
    """Readonly data cache with separate host-load and device-materialize stages."""

    def __init__(self):
        self._host_payloads: dict[tuple[str, int], bytearray] = {}
        self._host_tables: dict[tuple[tuple[str, ...], int], bytearray] = {}
        self._device_payloads: dict[tuple[str, int, str], Any] = {}
        self._device_tables: dict[tuple[tuple[str, ...], int, str], Any] = {}

    def prepare_bundle(self, bundle: dict[str, Any]) -> None:
        self.prepare_payload(bundle["rdata"], bundle["rdata_bytes"])
        self.prepare_table(bundle["thread_local_rdata"], bundle["thread_local_rdata_bytes"])
        self.prepare_table(bundle["warp_local_rdata"], bundle["warp_local_rdata_bytes"])
        self.prepare_table(bundle["block_local_rdata"], bundle["block_local_rdata_bytes"])

    def materialize_bundle(
        self,
        inputs: tuple[Any, ...],
        bundle: dict[str, Any],
        device: Any | None = None,
    ):
        return (
            self.materialize_payload(inputs, bundle["rdata"], bundle["rdata_bytes"], device),
            self.materialize_table(inputs, bundle["thread_local_rdata"], bundle["thread_local_rdata_bytes"], device),
            self.materialize_table(inputs, bundle["warp_local_rdata"], bundle["warp_local_rdata_bytes"], device),
            self.materialize_table(inputs, bundle["block_local_rdata"], bundle["block_local_rdata_bytes"], device),
        )

    def prepare_payload(self, payload: str, byte_count: int) -> bytearray:
        key = (payload, int(byte_count))
        raw = self._host_payloads.get(key)
        if raw is None:
            raw = _decode_payload(payload, byte_count)
            self._host_payloads[key] = raw
        return raw

    def prepare_table(self, payloads: tuple[str, ...], bytes_per_entry: int) -> bytearray:
        payload_tuple = tuple(payloads)
        key = (payload_tuple, int(bytes_per_entry))
        raw = self._host_tables.get(key)
        if raw is None:
            raw = _decode_payload_table(payload_tuple, bytes_per_entry)
            self._host_tables[key] = raw
        return raw

    def materialize_payload(
        self,
        inputs: tuple[Any, ...],
        payload: str,
        byte_count: int,
        device: Any | None = None,
    ):
        resolved_device = _normalize_device(inputs, device)
        cache_key = (payload, int(byte_count), _device_key(resolved_device))
        tensor = self._device_payloads.get(cache_key)
        if tensor is None:
            raw = self.prepare_payload(payload, byte_count)
            tensor = _bytes_to_tensor(raw, resolved_device)
            self._device_payloads[cache_key] = tensor
        return tensor

    def materialize_table(
        self,
        inputs: tuple[Any, ...],
        payloads: tuple[str, ...],
        bytes_per_entry: int,
        device: Any | None = None,
    ):
        resolved_device = _normalize_device(inputs, device)
        payload_tuple = tuple(payloads)
        cache_key = (payload_tuple, int(bytes_per_entry), _device_key(resolved_device))
        tensor = self._device_tables.get(cache_key)
        if tensor is None:
            raw = self.prepare_table(payload_tuple, bytes_per_entry)
            tensor = _bytes_to_tensor(raw, resolved_device)
            self._device_tables[cache_key] = tensor
        return tensor


_GLOBAL_WORKSPACE_POOL = WorkspacePool()
_GLOBAL_RDATA_CACHE = RDataCache()


def allocate_workspace(inputs: tuple[Any, ...], elements: int, dtype: str):
    """Return a reusable one-dimensional workspace tensor next to runtime inputs."""
    return _GLOBAL_WORKSPACE_POOL.allocate(inputs, "global", elements, dtype)


def materialize_rdata(inputs: tuple[Any, ...], payload: str, byte_count: int):
    """Materialize one readonly data payload as a CUDA uint8 tensor."""
    return _GLOBAL_RDATA_CACHE.materialize_payload(inputs, payload, byte_count)


def materialize_rdata_table(inputs: tuple[Any, ...], payloads: tuple[str, ...], bytes_per_entry: int):
    """Materialize per-shard readonly data payloads as one flat uint8 tensor."""
    return _GLOBAL_RDATA_CACHE.materialize_table(inputs, tuple(payloads), bytes_per_entry)


def _decode_payload(payload: str, byte_count: int) -> bytearray:
    if byte_count == 0:
        if payload:
            raise PyNTTSpecError("PyNTT rdata payload is non-empty for a zero-sized section.")
        return bytearray()

    if payload.startswith("file:"):
        raw = bytearray(Path(payload[5:]).read_bytes())
    else:
        raise PyNTTSpecError("PyNTT rdata payloads must be binary files.")
    if len(raw) != int(byte_count):
        raise PyNTTSpecError(
            f"PyNTT rdata payload size mismatch: expected {byte_count} bytes, got {len(raw)}."
        )
    return raw


def _decode_payload_table(payloads: tuple[str, ...], bytes_per_entry: int) -> bytearray:
    entry_bytes = int(bytes_per_entry)
    if entry_bytes == 0:
        if payloads:
            raise PyNTTSpecError("PyNTT rdata table has payloads for a zero-sized section.")
        return bytearray()

    raw = bytearray(checked_total := checked_len(len(payloads), entry_bytes))
    offset = 0
    for payload in payloads:
        entry = _decode_payload(payload, entry_bytes)
        raw[offset:offset + entry_bytes] = entry
        offset += entry_bytes
    if len(raw) != checked_total:
        raise PyNTTSpecError("PyNTT rdata table payload size mismatch.")
    return raw


def checked_len(count: int, size: int) -> int:
    return int(count) * int(size)


def _bytes_to_tensor(raw: bytearray, device: Any):
    torch = _import_torch()
    if not raw:
        return torch.empty((0,), dtype=torch.uint8, device=device)

    host = torch.frombuffer(raw, dtype=torch.uint8)
    if torch.device(device).type == "cpu":
        return host
    return host.to(device=device)
