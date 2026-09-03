"""Pinned UVA channels shared by heterogeneous persistent workers."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field

import torch


PIPELINE_CHANNEL_HEADER_BYTES = 64
_PIPELINE_CHANNEL_STATE_WORDS = 3
_PipelineChannelState = ctypes.c_uint32 * _PIPELINE_CHANNEL_STATE_WORDS


@dataclass(frozen=True)
class PipelineChannel:
    """Own one cache-line-aligned synchronization header and tensor payload."""

    tensor: torch.Tensor
    _allocation: torch.Tensor
    payload_bytes: int
    _state: _PipelineChannelState = field(repr=False)

    @classmethod
    def allocate(cls, payload_bytes: int) -> "PipelineChannel":
        payload_bytes = int(payload_bytes)
        if payload_bytes < 0:
            raise ValueError(
                f"Pipeline channel payload size must be non-negative, got {payload_bytes}."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Heterogeneous PyNTT pipeline channels require CUDA pinned host memory."
            )

        total_bytes = PIPELINE_CHANNEL_HEADER_BYTES + payload_bytes
        allocation = torch.empty(
            total_bytes + PIPELINE_CHANNEL_HEADER_BYTES - 1,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        offset = (-int(allocation.data_ptr())) % PIPELINE_CHANNEL_HEADER_BYTES
        tensor = allocation.narrow(0, offset, total_bytes)
        if int(tensor.data_ptr()) % PIPELINE_CHANNEL_HEADER_BYTES != 0:
            raise RuntimeError("Pipeline channel allocation is not cache-line aligned.")
        state = _PipelineChannelState.from_address(int(tensor.data_ptr()))
        channel = cls(
            tensor=tensor,
            _allocation=allocation,
            payload_bytes=payload_bytes,
            _state=state,
        )
        channel.reset()
        return channel

    @property
    def data_ptr(self) -> int:
        return int(self.tensor.data_ptr())

    @property
    def total_bytes(self) -> int:
        return int(self.tensor.numel())

    def reset(self) -> None:
        """Reset only synchronization state; payload contents are intentionally retained."""
        self._state[0] = 0
        self._state[1] = 0
        self._state[2] = 0
