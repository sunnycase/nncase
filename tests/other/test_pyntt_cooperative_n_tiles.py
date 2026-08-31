from __future__ import annotations

import sys
from pathlib import Path


def _add_pyntt_to_path() -> None:
    pyntt_root = Path(__file__).resolve().parents[2] / "pyntt"
    if str(pyntt_root) not in sys.path:
        sys.path.insert(0, str(pyntt_root))


def test_renderer_marks_only_consumer_calls_closed_by_full_grid_barrier():
    _add_pyntt_to_path()
    from pyntt.codegen.render import _annotate_barrier_synchronized_helpers

    synchronized = {"FunctionName": "synchronized", "MicroKernel": {}}
    unsynchronized = {"FunctionName": "unsynchronized", "MicroKernel": {}}
    kernel = {
        "helpers": [
            {"template": "template", "model": synchronized},
            {"template": "template", "model": unsynchronized},
        ],
        "body_source": (
            "synchronized__consumer()\n"
            "tle.distributed_barrier(PYNTT_GRID_MESH)\n"
            "unsynchronized__consumer()\n"
            "tl.debug_barrier()\n"
        ),
    }

    _annotate_barrier_synchronized_helpers(kernel)

    assert synchronized["_BarrierSynchronizedCooperativeNTiles"] is True
    assert "_BarrierSynchronizedCooperativeNTiles" not in unsynchronized


def test_renderer_balances_block_cyclic_glu_tail_tiles_within_inner_mesh_axis():
    _add_pyntt_to_path()
    from pyntt.codegen.render import _matmul_glu_cooperative_n_tile_plan

    def fixed(value: int) -> dict[str, int | str]:
        return {"TritonExpression": str(value), "FixedValue": value}

    def pointer(
        global_shape: list[int], n_axis: int, block_size: int
    ) -> dict[str, object]:
        shard_axes = [{"Stages": []} for _ in global_shape]
        shard_axes[n_axis] = {
            "Stages": [
                {
                    "HierarchyAxes": [0, 1],
                    "Distribution": "BlockCyclic",
                    "Granularity": fixed(17 if n_axis == 1 else 136),
                    "BlockSize": block_size,
                }
            ]
        }
        return {
            "DistributedStorageKind": "CanonicalGlobal",
            "GlobalShape": [fixed(value) for value in global_shape],
            "GlobalOffsets": [fixed(0) for _ in global_shape],
            "ShardAxes": shard_axes,
            "Hierarchy": [8, 16],
        }

    model = {
        "_BarrierSynchronizedCooperativeNTiles": True,
        "MicroKernel": {
            "Parameters": {"block_n": 64},
            "AuxiliaryConsumer": {
                "TransferPipelineChannelNames": ["weight"],
                "SharedWorkspaceNames": ["lhs_quantized", "lhs_scale"],
            },
        },
        "EmitPartialResults": False,
        "HasGateBias": False,
        "HasUpBias": False,
        "NVectorLaneCount": 8,
        "OutputShape": [fixed(1), fixed(17)],
        "Output": pointer([1, 2176], 1, 1),
        "GateWeight": pointer([17408, 160], 0, 8),
        "UpWeight": pointer([17408, 160], 0, 8),
        "GateWeightDescriptorOriginElements": "0",
        "UpWeightDescriptorOriginElements": "0",
    }

    plan = _matmul_glu_cooperative_n_tile_plan(model, 64)

    assert plan == {
        "owner_count": 128,
        "group_size": 16,
        "group_tiles": 34,
        "base_tiles": 2,
        "extra_tiles": 2,
        "tiles_per_owner": 3,
        "block_n": 64,
        "global_n": 17408,
        "runtime_active_n": (
            "(128 + ((shard_index % 16) < 2).to(tl.int32) * 64)"
        ),
        "global_output_n": (
            "((((shard_index // 16) * 34 + n_tile * 16 + "
            "(shard_index % 16)) * 64) + local_n)"
        ),
    }


def test_renderer_reorders_cooperative_glu_descriptor_entries_by_tile_wave():
    _add_pyntt_to_path()
    from pyntt.codegen.render import (
        _cooperative_n_major_k_packed_descriptor_spec,
    )

    prototype = {
        "offset_bytes": 100,
        "shape": (40, 64, 128),
        "strides": (128, 5120, 1),
        "source_shape_axes": ((), (), ()),
    }
    spec = {
        "block_shape": (2, 64, 128),
        "entries": (prototype,),
    }
    backing = {
        "offset_bytes": 100,
        "scalar_dtype": "float8e4m3fn",
        "logical_shape": [17408, 160],
        "logical_strides": [160, 1],
        "vector_lane_shape": [2, 16],
    }
    plan = {
        "owner_count": 128,
        "group_size": 16,
        "group_tiles": 34,
        "tiles_per_owner": 3,
        "block_n": 64,
    }

    result = _cooperative_n_major_k_packed_descriptor_spec(
        spec, backing, plan
    )

    scalar_tile_bytes = 64 * 160 * 32
    assert len(result["entries"]) == 384
    assert result["entries"][0]["offset_bytes"] == 100
    assert result["entries"][1]["offset_bytes"] == 100 + 16 * scalar_tile_bytes
    assert result["entries"][2]["offset_bytes"] == 100 + 32 * scalar_tile_bytes
    assert result["entries"][48]["offset_bytes"] == 100 + 34 * scalar_tile_bytes


def test_renderer_canonicalizes_fragmented_cooperative_glu_descriptors():
    _add_pyntt_to_path()
    from pyntt.codegen.render import (
        _cooperative_n_major_k_packed_descriptor_spec,
    )

    prototype = {
        "offset_bytes": 100,
        "shape": (40, 8, 8, 128),
        "strides": (128, 5_242_880, 5120, 1),
        "source_shape_axes": ((), (), (), ()),
    }
    spec = {
        "block_shape": (2, 8, 8, 128),
        "swizzle_mode": 3,
        "entries": (prototype,),
    }
    backing = {
        "offset_bytes": 100,
        "scalar_dtype": "float8e4m3fn",
        "logical_shape": [17408, 160],
        "logical_strides": [160, 1],
        "vector_lane_shape": [2, 16],
    }
    plan = {
        "owner_count": 128,
        "group_size": 16,
        "group_tiles": 34,
        "tiles_per_owner": 3,
        "block_n": 64,
    }

    result = _cooperative_n_major_k_packed_descriptor_spec(
        spec, backing, plan
    )

    assert result["block_shape"] == (2, 64, 128)
    assert result["entries"][0]["shape"] == (40, 64, 128)
    assert result["entries"][0]["strides"] == (128, 5120, 1)
    assert result["entries"][0]["source_shape_axes"] == ((), (), ())
