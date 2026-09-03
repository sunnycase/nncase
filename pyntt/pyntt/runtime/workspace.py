"""Workspace and readonly-data helpers for generated PyNTT models."""

from __future__ import annotations

from collections.abc import Mapping
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
        self._host_files: dict[tuple[str, int], Any] = {}
        self._host_payloads: dict[tuple[Any, ...], Any] = {}
        self._host_tables: dict[tuple[tuple[tuple[Any, ...], ...], int], Any] = {}
        self._device_payloads: dict[tuple[tuple[Any, ...], str], Any] = {}
        self._device_tables: dict[tuple[tuple[tuple[Any, ...], ...], int, str], Any] = {}
        self._materialized_bundles: dict[
            tuple[int, str], tuple[Mapping[str, Any], tuple[Any, Any, Any]]
        ] = {}

    def prepare_bundle(self, bundle: dict[str, Any]) -> None:
        self.prepare_payload(bundle["rdata"], bundle["rdata_bytes"])
        self.prepare_payload(bundle.get("chip_local_rdata"), bundle.get("chip_local_rdata_bytes", 0))
        self.prepare_table(bundle["block_local_rdata"], bundle["block_local_rdata_bytes"])

    def materialize_bundle(
        self,
        inputs: tuple[Any, ...],
        bundle: Mapping[str, Any],
        device: Any | None = None,
    ):
        resolved_device = _normalize_device(inputs, device)
        cache_key = (id(bundle), _device_key(resolved_device))
        cached = self._materialized_bundles.get(cache_key)
        if cached is not None and cached[0] is bundle:
            return cached[1]

        result = (
            self.materialize_payload(inputs, bundle["rdata"], bundle["rdata_bytes"], device),
            self.materialize_payload(inputs, bundle.get("chip_local_rdata"), bundle.get("chip_local_rdata_bytes", 0), device),
            self.materialize_table(inputs, bundle["block_local_rdata"], bundle["block_local_rdata_bytes"], device),
        )
        self._materialized_bundles[cache_key] = (bundle, result)
        return result

    def prepare_payload(self, payload: Mapping[str, Any] | None, byte_count: int):
        spec = _normalize_payload_spec(payload, byte_count)
        if spec is None:
            return _empty_host_payload()

        key = _payload_identity(spec)
        raw = self._host_payloads.get(key)
        if raw is None:
            raw = self._map_payload(spec)
            self._host_payloads[key] = raw
        return raw

    def prepare_table(self, payloads: tuple[Mapping[str, Any], ...], bytes_per_entry: int):
        payload_tuple = tuple(payloads)
        specs = tuple(
            _require_payload_spec(payload, bytes_per_entry) for payload in payload_tuple
        )
        identities = tuple(_payload_identity(spec) for spec in specs)
        key = (identities, int(bytes_per_entry))
        raw = self._host_tables.get(key)
        if raw is None:
            raw = self._prepare_payload_table(specs, bytes_per_entry)
            self._host_tables[key] = raw
        return raw

    def materialize_payload(
        self,
        inputs: tuple[Any, ...],
        payload: Mapping[str, Any] | None,
        byte_count: int,
        device: Any | None = None,
    ):
        resolved_device = _normalize_device(inputs, device)
        spec = _normalize_payload_spec(payload, byte_count)
        if spec is None:
            return _empty_device_payload(resolved_device)

        identity = _payload_identity(spec)
        cache_key = (identity, _device_key(resolved_device))
        tensor = self._device_payloads.get(cache_key)
        if tensor is None:
            raw = self.prepare_payload(payload, byte_count)
            tensor = _bytes_to_tensor(raw, resolved_device)
            self._device_payloads[cache_key] = tensor
        return tensor

    def materialize_table(
        self,
        inputs: tuple[Any, ...],
        payloads: tuple[Mapping[str, Any], ...],
        bytes_per_entry: int,
        device: Any | None = None,
    ):
        resolved_device = _normalize_device(inputs, device)
        payload_tuple = tuple(payloads)
        identities = tuple(
            _payload_identity(_require_payload_spec(payload, bytes_per_entry))
            for payload in payload_tuple
        )
        cache_key = (identities, int(bytes_per_entry), _device_key(resolved_device))
        tensor = self._device_tables.get(cache_key)
        if tensor is None:
            raw = self.prepare_table(payload_tuple, bytes_per_entry)
            tensor = _bytes_to_tensor(raw, resolved_device)
            self._device_tables[cache_key] = tensor
        return tensor

    def _map_payload(self, spec: dict[str, Any]):
        torch = _import_torch()
        source = spec["source"]
        if not source.startswith("file:"):
            raise PyNTTSpecError("PyNTT rdata payload sources must be binary files.")

        path = Path(source[5:])
        file_bytes = path.stat().st_size
        source_offset = spec["source_offset_bytes"]
        payload_bytes = spec["bytes"]
        if source_offset + payload_bytes > file_bytes:
            raise PyNTTSpecError(
                "PyNTT rdata payload slice exceeds its source file: "
                f"offset {source_offset}, bytes {payload_bytes}, file bytes {file_bytes}."
            )

        file_key = (str(path.resolve()), file_bytes)
        mapped = self._host_files.get(file_key)
        if mapped is None:
            mapped = torch.from_file(
                str(path), shared=False, size=file_bytes, dtype=torch.uint8
            )
            self._host_files[file_key] = mapped
        return mapped.narrow(0, source_offset, payload_bytes)

    def _prepare_payload_table(
        self, specs: tuple[dict[str, Any], ...], bytes_per_entry: int
    ):
        torch = _import_torch()
        entry_bytes = int(bytes_per_entry)
        if entry_bytes == 0:
            if specs:
                raise PyNTTSpecError(
                    "PyNTT rdata table has payloads for a zero-sized section."
                )
            return torch.empty((0,), dtype=torch.uint8)

        raw = torch.empty((checked_len(len(specs), entry_bytes),), dtype=torch.uint8)
        for index, spec in enumerate(specs):
            raw.narrow(0, index * entry_bytes, entry_bytes).copy_(
                self._map_payload(spec)
            )
        return raw

_GLOBAL_WORKSPACE_POOL = WorkspacePool()
_GLOBAL_RDATA_CACHE = RDataCache()


def allocate_workspace(inputs: tuple[Any, ...], elements: int, dtype: str):
    """Return a reusable one-dimensional workspace tensor next to runtime inputs."""
    return _GLOBAL_WORKSPACE_POOL.allocate(inputs, "global", elements, dtype)


def materialize_rdata(
    inputs: tuple[Any, ...], payload: Mapping[str, Any] | None, byte_count: int
):
    """Materialize one readonly data payload as a CUDA uint8 tensor."""
    return _GLOBAL_RDATA_CACHE.materialize_payload(inputs, payload, byte_count)


def materialize_rdata_table(
    inputs: tuple[Any, ...],
    payloads: tuple[Mapping[str, Any], ...],
    bytes_per_entry: int,
):
    """Materialize per-shard readonly data payloads as one flat uint8 tensor."""
    return _GLOBAL_RDATA_CACHE.materialize_table(inputs, tuple(payloads), bytes_per_entry)


_PAYLOAD_SPEC_KEYS = {
    "source",
    "source_offset_bytes",
    "bytes",
}


def _normalize_payload_spec(
    payload: Mapping[str, Any] | None, byte_count: int
) -> dict[str, Any] | None:
    expected_bytes = int(byte_count)
    if expected_bytes < 0:
        raise PyNTTSpecError(
            f"PyNTT rdata byte count must be non-negative, got {expected_bytes}."
        )
    if expected_bytes == 0:
        if payload is not None:
            raise PyNTTSpecError(
                "PyNTT rdata payload must be None for a zero-sized section."
            )
        return None
    return _require_payload_spec(payload, expected_bytes)


def _require_payload_spec(
    payload: Mapping[str, Any] | None, expected_bytes: int
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PyNTTSpecError(
            "PyNTT non-empty rdata payload must be a structured payload spec."
        )
    unknown = set(payload) - _PAYLOAD_SPEC_KEYS
    missing = _PAYLOAD_SPEC_KEYS - set(payload)
    if unknown or missing:
        raise PyNTTSpecError(
            "PyNTT rdata payload spec fields mismatch: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}."
        )

    source = payload["source"]
    source_offset = int(payload["source_offset_bytes"])
    payload_bytes = int(payload["bytes"])
    if not isinstance(source, str) or not source:
        raise PyNTTSpecError("PyNTT rdata payload source must be a non-empty string.")
    if source_offset < 0:
        raise PyNTTSpecError("PyNTT rdata payload source offset must be non-negative.")
    if payload_bytes != int(expected_bytes):
        raise PyNTTSpecError(
            f"PyNTT rdata payload size mismatch: expected {expected_bytes} bytes, "
            f"got {payload_bytes}."
        )
    return {
        "source": source,
        "source_offset_bytes": source_offset,
        "bytes": payload_bytes,
    }


def _payload_identity(spec: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(spec[key] for key in sorted(_PAYLOAD_SPEC_KEYS))


def _empty_host_payload():
    torch = _import_torch()
    return torch.empty((0,), dtype=torch.uint8)


def _empty_device_payload(device: Any):
    torch = _import_torch()
    return torch.empty((0,), dtype=torch.uint8, device=device)


def checked_len(count: int, size: int) -> int:
    return int(count) * int(size)


def _bytes_to_tensor(host: Any, device: Any):
    if host.numel() == 0:
        return _empty_device_payload(device)
    if _import_torch().device(device).type == "cpu":
        return host
    return host.to(device=device)
