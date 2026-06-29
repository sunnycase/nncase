"""Sharding helpers shared by generated PyNTT runtimes and tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class LocalShard:
    """A contiguous 1-D local shard region."""

    offset: int
    extent: int


@dataclass(frozen=True)
class ShardedTensorSpec:
    """Static sharding metadata for a tensor axis split by a placement axis."""

    shape: tuple[int, ...]
    tensor_axis: int
    placement_axis: str = "b"

    def local_shape(self, shard_index: int, shard_count: int) -> tuple[int, ...]:
        shard = local_shard_1d(self.shape[self.tensor_axis], shard_index, shard_count)
        local = list(self.shape)
        local[self.tensor_axis] = shard.extent
        return tuple(local)

    def local_offsets(self, shard_index: int, shard_count: int) -> tuple[int, ...]:
        shard = local_shard_1d(self.shape[self.tensor_axis], shard_index, shard_count)
        offsets = [0] * len(self.shape)
        offsets[self.tensor_axis] = shard.offset
        return tuple(offsets)


def local_shard_1d(numel: int, shard_index: int, shard_count: int) -> LocalShard:
    if numel < 0:
        raise ValueError("numel must be non-negative.")
    if shard_count <= 0:
        raise ValueError("shard_count must be positive.")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must be in [0, shard_count).")

    local_extent = (numel + shard_count - 1) // shard_count
    offset = shard_index * local_extent
    extent = min(local_extent, max(numel - offset, 0))
    return LocalShard(offset=offset, extent=extent)


def sharded_tensor(
    shape: Sequence[int],
    tensor_axis: int,
    placement_axis: str = "b",
) -> ShardedTensorSpec:
    rank = len(shape)
    if tensor_axis < 0:
        tensor_axis += rank
    if tensor_axis < 0 or tensor_axis >= rank:
        raise ValueError("tensor_axis is out of range.")

    return ShardedTensorSpec(
        shape=tuple(int(dim) for dim in shape),
        tensor_axis=tensor_axis,
        placement_axis=placement_axis,
    )
