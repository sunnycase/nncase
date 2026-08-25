import copy
import sys
from pathlib import Path

import pytest
from jinja2 import meta


def _add_pyntt_to_path():
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "pyntt"))


def _test_pyntt_codegen_manifest(render_kernel):
    metadata = render_kernel["metadata"]
    attrs = dict(metadata.get("attrs", {}))
    attrs.setdefault("target_worker_width", 32)
    attrs.setdefault("target_threads_per_block", 256)
    attrs.setdefault("target_resident_blocks_per_compute_unit", 1)
    attrs.setdefault("register_file_capacity_units", 255 * 8 * 32)
    attrs.setdefault("register_file_allocation_granularity_units", 8 * 32)
    attrs.setdefault("registers_per_thread_limit", 255)
    attrs.setdefault("shared_memory_capacity_bytes", 101376)
    strict_metadata = {
        "name": metadata["name"],
        "op_kind": metadata.get("op_kind", "test"),
        "inputs": metadata.get("inputs", []),
        "outputs": metadata.get("outputs", []),
        "attrs": attrs,
        "launch": {
            "meta": {
                "shared_data_pool_bytes": 0,
                "shared_data_pool_alignment_bytes": 8,
            },
            "host_tensor_descriptors": [],
            "sharding": {
                "strategy": "replicated",
                "placement_axis": "b",
                "tensor_axis": 0,
                "extent": "1",
                "hierarchy": [1],
                "hierarchy_levels": "b",
                "global_shape": [],
            },
        },
    }
    return {
        "pyntt_codegen_manifest_version": 9,
        "target_kind": "pyntt",
        "backend": "triton",
        "functions": [
            {
                "id": 0,
                "name": "main",
                "module_kind": "pyntt",
                "is_entry": True,
                "render_kernels": [
                    {
                        "metadata": strict_metadata,
                        "helpers": copy.deepcopy(render_kernel.get("helpers", [])),
                        "device_functions": copy.deepcopy(
                            render_kernel.get("device_functions", [])
                        ),
                        "body_source": render_kernel.get("body_source", ""),
                    }
                ],
            }
        ],
    }


def _render_test_pyntt_manifest(render_manifest, partial_manifest):
    render_kernel = partial_manifest["functions"][0]["render_kernels"][0]
    return render_manifest(_test_pyntt_codegen_manifest(render_kernel))


def _device_function(name, body, extra_parameters=()):
    return {
        "name": name,
        "noinline": True,
        "helpers": [],
        "body_source": body,
        "parameter_overrides": {},
        "extra_parameters": list(extra_parameters),
        "extra_parameter_arguments": {},
    }


def test_pyntt_target_options_do_not_expose_removed_pipeline_policy():
    import nncase

    options = nncase.PyNTTTargetOptions()
    assert not hasattr(options, "PipelinePolicy")


def test_pyntt_elementwise_accesses_preserve_their_natural_broadcast_domain():
    _add_pyntt_to_path()
    from pyntt.codegen.render import _elementwise_binary_template_context

    context = _elementwise_binary_template_context(
        {
            "LhsShape": [1, 1],
            "LhsStrides": [0, 0],
            "LhsVectorLaneShape": [],
            "RhsShape": [4, 16],
            "RhsStrides": [16, 1],
            "RhsVectorLaneShape": [8],
            "OutputShape": [4, 16],
            "OutputStrides": [16, 1],
            "OutputVectorLaneShape": [8],
        }
    )

    assert context["lane_count"] == 8
    assert context["tile_shape"] == "((block_size // 8), 8)"
    assert context["lhs_access"]["ScalarOffset"] == "0"
    assert context["lhs_access"]["BoundaryMask"] == "True"
    assert context["rhs_access"]["BoundaryMask"] == "mask"
    assert context["output_access"]["BoundaryMask"] == "mask"
    assert context["tensor_coordinates"] == ("index0", "major_raw")
    assert context["lane_coordinates"] == ("lane_raw0",)
    assert "major_raw" in context["rhs_access"]["ScalarOffset"]
    assert "lane_raw0" in context["rhs_access"]["ScalarOffset"]
    assert "tl.broadcast_to" not in context["rhs_access"]["ScalarOffset"]
    assert "tl.broadcast_to" not in context["output_access"]["ScalarOffset"]


def test_pyntt_elementwise_scalar_domain_masks_the_physical_tile_tail():
    _add_pyntt_to_path()
    from pyntt.codegen.render import _elementwise_unary_template_context

    context = _elementwise_unary_template_context(
        {
            "InputShape": [1, 1],
            "InputStrides": [0, 1],
            "InputVectorLaneShape": [],
            "OutputShape": [1, 1],
            "OutputStrides": [0, 1],
            "OutputVectorLaneShape": [],
        }
    )

    assert context["major_axis"] == 1
    assert context["input_access"]["BoundaryMask"] == "True"
    assert context["output_access"]["BoundaryMask"] == "mask"


def test_pyntt_elementwise_iteration_uses_explicit_physical_tile_width():
    template_dir = (
        Path(__file__).resolve().parents[2]
        / "pyntt/pyntt/codegen/templates/triton/kernels"
    )
    template = (template_dir / "_elementwise.py.jinja").read_text(encoding="utf-8")

    assert 'physical_width_divisor="1"' in template
    assert '"max(1, block_size // "' in template
    assert "_physical_tile_width: tl.constexpr = min(" in template
    assert "tl.arange(0, {{ scope }}_physical_tile_width)" in template
    assert "block_size >>" not in template
    assert "tl.broadcast_to" not in template
    assert "zero_coord" not in template
    assert "raw_coord" not in template
    assert "raw_lane_coord" not in template


def test_pyntt_inline_norm_reduction_caps_its_physical_tile_to_the_extent():
    _add_pyntt_to_path()
    from pyntt.codegen.render import _norm_rope_template_context

    norm = {
        "InputShape": [1, 1, 16],
        "InputGlobalShape": [1, 16, 16],
        "InputStrides": [0, 16, 1],
        "ScaleShape": [16],
        "ScaleStrides": [1],
        "BiasShape": [16],
        "BiasStrides": [1],
        "InputVectorLaneShape": [8],
        "InputVectorLaneCount": 8,
        "ScaleVectorLaneShape": [8],
        "ScaleVectorLaneCount": 8,
        "BiasVectorLaneShape": [8],
        "BiasVectorLaneCount": 8,
        "Axis": 2,
        "Epsilon": 1e-6,
        "UseMean": False,
    }
    rope = {
        "InputShape": [1, 1, 16],
        "CosShape": [1, 1, 8],
        "SinShape": [1, 1, 8],
        "OutputShape": [1, 1, 16],
        "InputStrides": [0, 16, 1],
        "CosStrides": [0, 0, 1],
        "SinStrides": [0, 0, 1],
        "OutputStrides": [0, 16, 1],
        "InputVectorLaneShape": [8],
        "CosVectorLaneShape": [2, 8],
        "SinVectorLaneShape": [2, 8],
        "OutputVectorLaneShape": [8],
        "InputVectorLaneCount": 8,
        "CosVectorLaneCount": 16,
        "SinVectorLaneCount": 16,
        "OutputVectorLaneCount": 8,
        "SinCosVectorPackFactor": 2,
        "RotaryAxis": 2,
    }

    context = _norm_rope_template_context(norm, rope, "test norm", "q")

    assert context["reduction_context"]["physical_tile_width_cap"] == 16
    assert context["reduction_context"]["major_variable"] == "q_reduce_major_raw"


def test_pyntt_package_imports():
    _add_pyntt_to_path()

    import pyntt
    from pyntt.backends import get_backend
    from pyntt.ir import FunctionSpec, ModuleSpec, TensorResultSpec, TensorSpec
    from pyntt.runtime import (
        LocalShard,
        PyNTTInterpreter,
        PyNTTModule,
        local_shard_1d,
        select_tuning_parameter,
        sharded_tensor,
    )

    spec = ModuleSpec(
        name="smoke",
        backend="triton",
        functions=(
            FunctionSpec(
                "main",
                "pyntt",
                True,
                inputs=(TensorSpec("x", "float32", (1,)),),
                outputs=(TensorSpec("output0", "float32", (1,), role="output"),),
                results=(
                    TensorResultSpec(
                        TensorSpec("result0", "float32", (1,), role="result"),
                        "output",
                        0,
                    ),
                ),
            ),
        ),
    )
    module = PyNTTModule(spec)
    interpreter = PyNTTInterpreter(spec).load()

    assert pyntt.__version__ == "0.0.0"
    assert type(get_backend("triton")).__name__ == "TritonBackend"
    assert module.spec.entry is not None
    assert interpreter.spec.entry is not None
    assert interpreter.loaded
    assert module.spec.entry.name == "main"
    assert module.spec.entry.parameters == ("x",)
    assert module.spec.entry.outputs[0].name == "output0"
    assert (
        select_tuning_parameter("main", "block_size", (128, 256), source="search_space")
        == 256
    )
    assert local_shard_1d(33, 0, 2) == LocalShard(offset=0, extent=17)
    assert local_shard_1d(33, 1, 2) == LocalShard(offset=17, extent=16)
    assert local_shard_1d(2, 3, 4) == LocalShard(offset=3, extent=0)

    sharded = sharded_tensor((4, 8), tensor_axis=1)
    assert sharded.placement_axis == "b"
    assert sharded.local_offsets(1, 3) == (0, 3)
    assert sharded.local_shape(1, 3) == (4, 3)


def test_pyntt_runtime_validates_torch_inputs_and_reuses_storage(tmp_path):
    torch = pytest.importorskip("torch")
    _add_pyntt_to_path()

    from pyntt.ir import FunctionSpec, ModuleSpec, TensorResultSpec, TensorSpec
    from pyntt.runtime import PyNTTArgumentError, PyNTTInterpreter, PyNTTModule
    from pyntt.runtime import (
        allocate_workspace,
        materialize_rdata,
        materialize_rdata_table,
    )

    spec = ModuleSpec(
        name="runtime",
        backend="triton",
        functions=(
            FunctionSpec(
                "main",
                "pyntt",
                True,
                inputs=(TensorSpec("x", "float32", (2, 3), strides=(3, 1)),),
                outputs=(
                    TensorSpec(
                        "output0",
                        "float32",
                        (2, 3),
                        role="output",
                        device="like_input",
                    ),
                ),
                results=(
                    TensorResultSpec(
                        TensorSpec("result0", "float32", (2, 3), role="result"),
                        "output",
                        0,
                    ),
                ),
            ),
        ),
    )
    module = PyNTTModule(spec)
    class CopyInterpreter(PyNTTInterpreter):
        def _run_entry(self, inputs, outputs, shape_env):
            outputs[0].copy_(inputs[0])

    interpreter = CopyInterpreter(spec).load()

    x = torch.ones((2, 3), dtype=torch.float32)
    y = module(x)
    z = interpreter.run(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.device == x.device
    assert z.shape == x.shape

    caller_output = torch.empty_like(x)
    assert interpreter.run_into((caller_output,), x) is None
    torch.testing.assert_close(caller_output, x)
    with pytest.raises(PyNTTArgumentError, match="output 0.*dtype"):
        interpreter.run_into(
            (torch.empty_like(x, dtype=torch.float16),),
            x,
        )
    if torch.cuda.is_available():
        unaligned = torch.empty((7,), dtype=torch.float32, device="cuda")[1:].view(2, 3)
        with pytest.raises(PyNTTArgumentError, match="16-byte aligned"):
            interpreter.run_into((torch.empty_like(unaligned),), unaligned)

    workspace = allocate_workspace((x,), 8, "uint8")
    workspace.fill_(7)
    reused_workspace = allocate_workspace((x,), 8, "uint8")
    assert reused_workspace.data_ptr() == workspace.data_ptr()
    assert int(reused_workspace[0].item()) == 7

    with pytest.raises(PyNTTArgumentError, match="shape"):
        module(torch.ones((3, 2), dtype=torch.float32))
    with pytest.raises(PyNTTArgumentError, match="dtype"):
        module(torch.ones((2, 3), dtype=torch.float16))

    rdata_path = tmp_path / "rdata.bin"
    rdata_path.write_bytes(bytes([1, 2, 3]))
    rdata = materialize_rdata((x,), f"file:{rdata_path}", 3)
    rdata_again = materialize_rdata((x,), f"file:{rdata_path}", 3)
    assert rdata.dtype == torch.uint8
    assert rdata.tolist() == [1, 2, 3]
    assert rdata_again.data_ptr() == rdata.data_ptr()

    table_paths = [tmp_path / f"rdata_table_{index}.bin" for index in range(2)]
    for index, path in enumerate(table_paths, start=1):
        path.write_bytes(bytes([index]))
    sources = tuple(f"file:{path}" for path in table_paths)
    rdata_table = materialize_rdata_table((x,), sources, 1)
    rdata_table_again = materialize_rdata_table((x,), sources, 1)
    assert rdata_table.tolist() == [1, 2]
    assert rdata_table_again.data_ptr() == rdata_table.data_ptr()


def test_pyntt_runtime_uses_authoritative_kv_cache_capacity_without_scanning():
    torch = pytest.importorskip("torch")
    _add_pyntt_to_path()

    from pyntt.runtime.tensor import (
        get_kv_cache_num_seqs,
        materialize_kv_cache_blocks_per_shard,
        materialize_kv_cache_storage,
    )
    from pyntt.runtime.errors import PyNTTArgumentError

    class DirectKVCache:
        num_blocks = 8
        num_seqs = 3

        def __init__(self):
            self.kv_caches = torch.empty((8, 16), dtype=torch.bfloat16)

        @property
        def slot_mapping(self):
            raise AssertionError("explicit capacity must not scan slot_mapping")

        @property
        def block_table(self):
            raise AssertionError("explicit capacity must not scan block_table")

        @property
        def query_start_loc(self):
            raise AssertionError("explicit num_seqs must not inspect query metadata")

        @property
        def seq_lens(self):
            raise AssertionError("explicit num_seqs must not inspect query metadata")

    cache = DirectKVCache()
    storage = materialize_kv_cache_storage(
        cache,
        dtype="bfloat16",
        topology_shape=(),
        key_tail_shape=(8,),
        value_tail_shape=(8,),
        key_section_elements=8,
        value_section_elements=8,
        block_elements=16,
        block_size=4,
    )

    assert storage is cache.kv_caches
    assert materialize_kv_cache_blocks_per_shard(
        cache, topology_shape=(), block_size=4
    ) == 8
    assert get_kv_cache_num_seqs(cache) == 3

    cache.num_blocks = 9
    with pytest.raises(PyNTTArgumentError, match="fewer blocks"):
        materialize_kv_cache_storage(
            cache,
            dtype="bfloat16",
            topology_shape=(),
            key_tail_shape=(8,),
            value_tail_shape=(8,),
            key_section_elements=8,
            value_section_elements=8,
            block_elements=16,
            block_size=4,
        )


def test_pyntt_runtime_reuses_bounded_host_tensor_descriptors():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton.tools.tensor_descriptor")
    _add_pyntt_to_path()

    from pyntt.runtime.triton import TritonTensorDescriptorCache

    cache = TritonTensorDescriptorCache()
    spec = {
        "kind": "single",
        "name": "rhs_descriptor",
        "source": "rdata",
        "offset_bytes": 0,
        "dtype": "uint8",
        "shape": (4, 16),
        "strides": (16, 1),
        "block_shape": (4, 16),
        "source_shape_axes": ((), ()),
        "padding": "zero",
    }
    storage = torch.empty((64,), dtype=torch.uint8)
    first = cache.materialize_many("kernel", (spec,), {"rdata": storage})[0]
    second = cache.materialize_many("kernel", (spec,), {"rdata": storage})[0]
    assert second is first

    replacement_storage = torch.empty_like(storage)
    replacement = cache.materialize_many(
        "kernel", (spec,), {"rdata": replacement_storage}
    )[0]
    assert replacement is not first


def test_pyntt_runtime_reuses_per_owner_tensor_map_tables():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("tensor-map tables require CUDA")
    _add_pyntt_to_path()

    from pyntt.runtime.triton import TritonTensorDescriptorCache

    cache = TritonTensorDescriptorCache()
    entries = tuple(
        {
            "offset_bytes": owner * 64,
            "shape": (4, 16),
            "strides": (16, 1),
            "source_shape_axes": ((), ()),
        }
        for owner in range(2)
    )
    spec = {
        "kind": "table",
        "name": "rhs_descriptor_table",
        "source": "rdata",
        "dtype": "uint8",
        "block_shape": (4, 16),
        "padding": "zero",
        "swizzle_mode": 0,
        "entry_size_bytes": 128,
        "entries": entries,
    }
    storage = torch.empty((128,), dtype=torch.uint8, device="cuda")
    first = cache.materialize_many("kernel", (spec,), {"rdata": storage})[0]
    second = cache.materialize_many("kernel", (spec,), {"rdata": storage})[0]
    assert second is first
    assert first.dtype == torch.uint8
    assert first.numel() == 256
    assert first.data_ptr() % 128 == 0

    replacement_storage = torch.empty_like(storage)
    replacement = cache.materialize_many(
        "kernel", (spec,), {"rdata": replacement_storage}
    )[0]
    assert replacement is not first


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("strides", (16, 2), "contiguous last dimension"),
        ("strides", (15, 1), "16-byte aligned"),
        ("block_shape", (3, 16), "power of two"),
    ],
)
def test_pyntt_runtime_validates_host_tensor_descriptor_layout(
    field, value, message
):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton.tools.tensor_descriptor")
    _add_pyntt_to_path()

    from pyntt.runtime.triton import TritonTensorDescriptorCache

    spec = {
        "kind": "single",
        "name": "rhs_descriptor",
        "source": "rdata",
        "offset_bytes": 0,
        "dtype": "uint8",
        "shape": (4, 16),
        "strides": (16, 1),
        "block_shape": (4, 16),
        "source_shape_axes": ((), ()),
        "padding": "zero",
    }
    spec[field] = value
    with pytest.raises(ValueError, match=message):
        TritonTensorDescriptorCache().materialize_many(
            "kernel",
            (spec,),
            {"rdata": torch.empty((64,), dtype=torch.uint8)},
        )


def test_pyntt_runtime_materializes_zero_copy_input_result_views():
    torch = pytest.importorskip("torch")
    _add_pyntt_to_path()

    from pyntt.ir import FunctionSpec, ModuleSpec, TensorResultSpec, TensorSpec
    from pyntt.runtime import PyNTTModule

    spec = ModuleSpec(
        name="views",
        backend="triton",
        functions=(
            FunctionSpec(
                "main",
                "pyntt",
                True,
                inputs=(TensorSpec("x", "float32", (2, 2)),),
                outputs=(),
                results=(
                    TensorResultSpec(
                        TensorSpec("reshaped", "float32", (4,), role="result"),
                        "input",
                        0,
                    ),
                    TensorResultSpec(
                        TensorSpec("bytes", "uint8", (16,), role="result"),
                        "input",
                        0,
                    ),
                ),
            ),
        ),
    )

    x = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    reshaped, bytes_view = PyNTTModule(spec)(x)
    assert reshaped.shape == (4,)
    assert bytes_view.shape == (16,)
    assert bytes_view.dtype == torch.uint8
    assert reshaped.data_ptr() == x.data_ptr()
    assert bytes_view.data_ptr() == x.data_ptr()


def test_pyntt_renderer_requires_manifest_v9():
    _add_pyntt_to_path()
    from pyntt.codegen.render import render_manifest

    with pytest.raises(ValueError, match="expected 9"):
        render_manifest({"pyntt_codegen_manifest_version": 8, "functions": []})


@pytest.mark.parametrize("alignment", [None, 3])
def test_pyntt_renderer_requires_valid_shared_arena_alignment(alignment):
    _add_pyntt_to_path()
    from pyntt.codegen.render import render_manifest

    manifest = _test_pyntt_codegen_manifest(
        {"metadata": {"name": "top"}, "body_source": "pass"}
    )
    meta = manifest["functions"][0]["render_kernels"][0]["metadata"]["launch"]["meta"]
    if alignment is None:
        meta.pop("shared_data_pool_alignment_bytes")
        message = "must be an integer"
    else:
        meta["shared_data_pool_alignment_bytes"] = alignment
        message = "positive power of two"

    with pytest.raises(ValueError, match=message):
        render_manifest(manifest)


def test_pyntt_renderer_preserves_shared_arena_alignment():
    _add_pyntt_to_path()
    from pyntt.codegen.render import render_manifest

    manifest = _test_pyntt_codegen_manifest(
        {"metadata": {"name": "top"}, "body_source": "pass"}
    )
    meta = manifest["functions"][0]["render_kernels"][0]["metadata"]["launch"]["meta"]
    meta["shared_data_pool_bytes"] = 65536
    meta["shared_data_pool_alignment_bytes"] = 1024

    source = render_manifest(manifest)
    assert "alignment=1024" in source


def test_pyntt_renderer_maps_staged_block_cyclic_tma_local_coordinates():
    _add_pyntt_to_path()
    from pyntt.codegen.render import _tma_local_axis_transfer

    def fixed(value):
        return {"TritonExpression": str(value), "FixedValue": value}

    pointer = {
        "DistributedStorageKind": "CanonicalGlobal",
        "GlobalShape": [fixed(8192)],
        "ShardAxes": [
            {
                "Stages": [
                    {
                        "HierarchyAxes": [0],
                        "Distribution": "Contiguous",
                        "Granularity": None,
                        "BlockSize": 0,
                    },
                    {
                        "HierarchyAxes": [1, 2],
                        "Distribution": "BlockCyclic",
                        "Granularity": None,
                        "BlockSize": 8,
                    },
                ]
            }
        ],
        "Hierarchy": [2, 4, 8],
    }

    transfer = _tma_local_axis_transfer(
        pointer,
        0,
        fixed(0),
        local_offset=0,
        tile_index="n_tile",
        tile_stride=8,
        tile_extent=8,
        context="test weight N",
    )

    coordinates = transfer["coordinates"]
    assert transfer["block_shape"] == (1, 8)
    assert "shard_coord" not in coordinates[0]
    assert "n_tile" in coordinates[0]
    with pytest.raises(ValueError, match="must start and advance"):
        _tma_local_axis_transfer(
            pointer,
            0,
            fixed(0),
            local_offset=0,
            tile_index="n_tile",
            tile_stride=4,
            tile_extent=8,
            context="crossing weight N",
        )

    packed_pointer = {
        "DistributedStorageKind": "CanonicalGlobal",
        "GlobalShape": [fixed(256)],
        "ShardAxes": [
            {
                "Stages": [
                    {
                        "HierarchyAxes": [0, 1],
                        "Distribution": "BlockCyclic",
                        "Granularity": None,
                        "BlockSize": 1,
                    }
                ]
            }
        ],
        "Hierarchy": [4, 8],
    }
    packed_transfer = _tma_local_axis_transfer(
        packed_pointer,
        0,
        fixed(0),
        local_offset=0,
        tile_index="n_tile",
        tile_stride=8,
        tile_extent=8,
        context="packed weight N",
    )
    assert packed_transfer["block_shape"] == (8,)
    assert len(packed_transfer["coordinates"]) == 1


def test_pyntt_renderer_builds_exact_ragged_block_cyclic_descriptor_entries():
    _add_pyntt_to_path()
    from pyntt.codegen.render import _tma_descriptor_table_axis_entry

    def fixed(value):
        return {"TritonExpression": str(value), "FixedValue": value}

    pointer = {
        "DistributedStorageKind": "CanonicalGlobal",
        "GlobalShape": [fixed(18992)],
        "ShardAxes": [
            {
                "Stages": [
                    {
                        "HierarchyAxes": [0, 1],
                        "Distribution": "BlockCyclic",
                        "Granularity": None,
                        "BlockSize": 1,
                    }
                ]
            }
        ],
        "Hierarchy": [4, 8],
    }

    owner0 = _tma_descriptor_table_axis_entry(
        pointer, 0, (0, 0), tile_extent=8, context="lm_head N"
    )
    owner15 = _tma_descriptor_table_axis_entry(
        pointer, 0, (1, 7), tile_extent=8, context="lm_head N"
    )
    owner16 = _tma_descriptor_table_axis_entry(
        pointer, 0, (2, 0), tile_extent=8, context="lm_head N"
    )
    owner31 = _tma_descriptor_table_axis_entry(
        pointer, 0, (3, 7), tile_extent=8, context="lm_head N"
    )

    assert owner0["descriptor_shape"] == (594,)
    assert owner15["descriptor_shape"] == (594,)
    assert owner16["descriptor_shape"] == (593,)
    assert owner31["descriptor_shape"] == (593,)
    assert owner0["stride_multipliers"] == (32,)
    assert owner0["base"] == 0
    assert owner15["base"] == 15
    assert owner16["base"] == 16
    assert owner31["base"] == 31


def test_pyntt_renderer_builds_tile_sized_descriptor_for_inactive_owner():
    _add_pyntt_to_path()
    from pyntt.codegen.render import _tma_descriptor_table_axis_entry

    def fixed(value):
        return {"TritonExpression": str(value), "FixedValue": value}

    pointer = {
        "DistributedStorageKind": "CanonicalGlobal",
        "GlobalShape": [fixed(48)],
        "ShardAxes": [
            {
                "Stages": [
                    {
                        "HierarchyAxes": [0, 1],
                        "Distribution": "Contiguous",
                        "Granularity": fixed(2),
                        "BlockSize": 0,
                    }
                ]
            }
        ],
        "Hierarchy": [4, 8],
    }

    active_owner = _tma_descriptor_table_axis_entry(
        pointer, 0, (2, 7), tile_extent=2, context="GDN value heads"
    )
    inactive_owner = _tma_descriptor_table_axis_entry(
        pointer, 0, (3, 0), tile_extent=2, context="GDN value heads"
    )

    assert active_owner["active"]
    assert active_owner["base"] == 46
    assert active_owner["descriptor_shape"] == (2,)
    assert not inactive_owner["active"]
    assert inactive_owner["base"] == 0
    assert inactive_owner["descriptor_shape"] == (2,)
    assert inactive_owner["stride_multipliers"] == (1,)


def test_pyntt_renderer_canonicalizes_unit_block_cyclic_tma_dimensions():
    _add_pyntt_to_path()
    from pyntt.codegen.render import (
        _k_major_gemv_host_descriptor_spec,
        _tma_local_axis_transfer,
    )

    def fixed(value):
        return {"TritonExpression": str(value), "FixedValue": value}

    pointer = {
        "DistributedStorageKind": "CanonicalGlobal",
        "GlobalShape": [fixed(128), fixed(128)],
        "ShardAxes": [
            {
                "Stages": [
                    {
                        "HierarchyAxes": [1],
                        "Distribution": "BlockCyclic",
                        "Granularity": None,
                        "BlockSize": 16,
                    }
                ]
            },
            {
                "Stages": [
                    {
                        "HierarchyAxes": [0],
                        "Distribution": "BlockCyclic",
                        "Granularity": None,
                        "BlockSize": 8,
                    }
                ]
            },
        ],
        "Hierarchy": [4, 8],
    }
    backing = {
        "name": "weight_descriptor",
        "source": "chip_local_rdata",
        "offset_bytes": 0,
        "scalar_dtype": "bfloat16",
        "logical_shape": [128, 128],
        "logical_strides": [128, 1],
        "vector_lane_shape": [8, 2, 8],
        "contiguous_rebase_extent_elements": 0,
        "owner_stride_bytes": 0,
    }

    spec = _k_major_gemv_host_descriptor_spec(
        {},
        backing,
        block_n=64,
        block_k=256,
        pointer=pointer,
        expected_vector_lane_shape=(8, 2, 8),
    )
    assert spec["block_shape"] == (16, 1, 8, 2, 64)
    assert len(spec["entries"]) == 32
    assert spec["entries"][0]["shape"] == (16, 4, 8, 2, 64)
    assert spec["entries"][0]["strides"] == (16384, 4096, 128, 64, 1)

    transfer = _tma_local_axis_transfer(
        pointer,
        0,
        fixed(0),
        local_offset=0,
        tile_index="k_tile",
        tile_stride=16,
        tile_extent=16,
        context="partial projection K",
    )
    assert transfer["block_shape"] == (16,)
    assert len(transfer["coordinates"]) == 1


def test_pyntt_renderer_accepts_selected_fp8_packed_gemv_lane_shape():
    _add_pyntt_to_path()
    from pyntt.codegen.render import _k_major_gemv_host_descriptor_spec

    def fixed(value):
        return {"TritonExpression": str(value), "FixedValue": value}

    pointer = {
        "DistributedStorageKind": "CanonicalGlobal",
        "GlobalShape": [fixed(128), fixed(128)],
        "ShardAxes": [{"Stages": []}, {"Stages": []}],
        "Hierarchy": [4, 8],
    }
    backing = {
        "name": "weight_descriptor",
        "source": "chip_local_rdata",
        "offset_bytes": 0,
        "scalar_dtype": "float8e4m3fn",
        "logical_shape": [128, 128],
        "logical_strides": [128, 1],
        "vector_lane_shape": [8, 2, 16],
        "contiguous_rebase_extent_elements": 0,
        "owner_stride_bytes": 0,
    }

    spec = _k_major_gemv_host_descriptor_spec(
        {},
        backing,
        block_n=64,
        block_k=256,
        pointer=pointer,
        expected_vector_lane_shape=(8, 2, 16),
    )

    assert spec["dtype"] == "float8e4m3fn"
    assert spec["block_shape"][-2:] == (2, 128)


def test_pyntt_renderer_preserves_one_block_cyclic_n_block_per_mma_tile():
    _add_pyntt_to_path()
    from pyntt.codegen.render import _n_major_k_packed_gemv_host_descriptor_spec

    def fixed(value):
        return {"TritonExpression": str(value), "FixedValue": value}

    pointer = {
        "DistributedStorageKind": "CanonicalGlobal",
        "GlobalShape": [fixed(512), fixed(16)],
        "ShardAxes": [
            {
                "Stages": [
                    {
                        "HierarchyAxes": [0, 1],
                        "Distribution": "BlockCyclic",
                        "Granularity": None,
                        "BlockSize": 64,
                    }
                ]
            },
            {"Stages": []},
        ],
        "Hierarchy": [2, 2],
    }
    model = {
        "MicroKernel": {
            "Family": "triton.matmul",
            "Variant": "mma_block_fp8_smem_pipeline",
            "Parameters": {
                "block_m": 1,
                "block_n": 64,
                "block_k": 512,
                "num_stages": 2,
            },
            "SharedWorkspaceOffsets": {
                "rhs_stage": "0",
                "lhs_quantized": "65536",
                "lhs_scale": "66048",
            },
            "SharedWorkspaceShapes": {
                "rhs_stage": [2, 4, 64, 128],
                "lhs_quantized": [4, 128],
                "lhs_scale": [4, 1],
            },
        },
        "WeightBlockK": 128,
        "Rhs": pointer,
        "OutputShape": [fixed(1), fixed(16)],
        "OutputNVectorLaneCount": 8,
    }
    backing = {
        "name": "weight_descriptor",
        "source": "chip_local_rdata",
        "offset_bytes": 256,
        "scalar_dtype": "float8e4m3fn",
        "logical_shape": [512, 16],
        "logical_strides": [16, 1],
        "vector_lane_shape": [2, 16],
        "contiguous_rebase_extent_elements": 0,
        "owner_stride_bytes": 0,
    }

    spec = _n_major_k_packed_gemv_host_descriptor_spec(model, backing)

    assert spec["block_shape"] == (64, 128)
    assert spec["swizzle_mode"] == 3
    assert len(spec["entries"]) == 8
    assert spec["entries"][0] == {
        "offset_bytes": 256,
        "shape": (64, 512),
        "strides": (512, 1),
        "source_shape_axes": ((), ()),
    }
    assert spec["entries"][1]["offset_bytes"] == 256 + 256 * 512
    assert spec["entries"][2]["offset_bytes"] == 256 + 64 * 512


def test_pyntt_renderer_uses_one_rank3_descriptor_for_block_cyclic_mma_tile():
    _add_pyntt_to_path()
    from pyntt.codegen.render import _n_major_k_packed_gemv_host_descriptor_spec

    def fixed(value):
        return {"TritonExpression": str(value), "FixedValue": value}

    pointer = {
        "DistributedStorageKind": "CanonicalGlobal",
        "GlobalShape": [fixed(640), fixed(16)],
        "ShardAxes": [
            {
                "Stages": [
                    {
                        "HierarchyAxes": [0],
                        "Distribution": "BlockCyclic",
                        "Granularity": None,
                        "BlockSize": 64,
                    }
                ]
            },
            {"Stages": []},
        ],
        "Hierarchy": [2],
    }
    model = {
        "MicroKernel": {
            "Family": "triton.matmul",
            "Variant": "mma_block_fp8_smem_pipeline",
            "Parameters": {
                "block_m": 1,
                "block_n": 128,
                "block_k": 256,
                "num_stages": 2,
            },
            "SharedWorkspaceOffsets": {
                "rhs_stage": "0",
                "lhs_quantized": "65536",
                "lhs_scale": "65792",
            },
            "SharedWorkspaceShapes": {
                "rhs_stage": [2, 2, 128, 128],
                "lhs_quantized": [2, 128],
                "lhs_scale": [2, 1],
            },
        },
        "WeightBlockK": 128,
        "Rhs": pointer,
        "OutputShape": [fixed(1), fixed(40)],
        "OutputNVectorLaneCount": 8,
    }
    backing = {
        "name": "weight_descriptor",
        "source": "chip_local_rdata",
        "offset_bytes": 256,
        "scalar_dtype": "float8e4m3fn",
        "logical_shape": [640, 16],
        "logical_strides": [16, 1],
        "vector_lane_shape": [2, 16],
        "contiguous_rebase_extent_elements": 0,
        "owner_stride_bytes": 0,
    }

    spec = _n_major_k_packed_gemv_host_descriptor_spec(model, backing)

    assert spec["block_shape"] == (2, 64, 128)
    assert len(spec["entries"]) == 6
    assert [entry["shape"][:2] for entry in spec["entries"]] == [
        (2, 64),
        (2, 64),
        (1, 64),
        (2, 64),
        (2, 64),
        (1, 64),
    ]
    assert spec["entries"][0]["offset_bytes"] == 256
    assert spec["entries"][1]["offset_bytes"] == 256 + 256 * 512
    assert spec["entries"][3]["offset_bytes"] == 256 + 64 * 512


def test_pyntt_renderer_uses_dense_coordinates_for_compact_local_tma():
    _add_pyntt_to_path()
    from pyntt.codegen.render import _k_major_gemv_host_descriptor_spec

    def fixed(value):
        return {"TritonExpression": str(value), "FixedValue": value}

    pointer = {
        "DistributedStorageKind": "CompactLocal",
        "GlobalShape": [fixed(96), fixed(16)],
        "ShardAxes": [
            {
                "Stages": [
                    {
                        "HierarchyAxes": [0],
                        "Distribution": "BlockCyclic",
                        "Granularity": None,
                        "BlockSize": 4,
                    }
                ]
            },
            {
                "Stages": [
                    {
                        "HierarchyAxes": [1],
                        "Distribution": "BlockCyclic",
                        "Granularity": None,
                        "BlockSize": 2,
                    }
                ]
            },
        ],
        "Hierarchy": [4, 8],
    }
    backing = {
        "name": "weight_descriptor",
        "source": "data",
        "offset_bytes": 0,
        "scalar_dtype": "bfloat16",
        "logical_shape": [96, 16],
        "logical_strides": [2, 1],
        "vector_lane_shape": [8, 2, 8],
        "contiguous_rebase_extent_elements": 0,
        "owner_stride_bytes": 24576,
    }

    spec = _k_major_gemv_host_descriptor_spec(
        {},
        backing,
        block_n=64,
        block_k=256,
        pointer=pointer,
        expected_vector_lane_shape=(8, 2, 8),
    )

    assert spec["block_shape"] == (16, 8, 2, 64)
    assert len(spec["entries"]) == 32
    assert spec["entries"][0]["shape"] == (24, 2, 2, 64)
    assert spec["entries"][0]["strides"] == (256, 128, 64, 1)
    for owner, entry in enumerate(spec["entries"]):
        span_elements = 1 + sum(
            (extent - 1) * stride
            for extent, stride in zip(entry["shape"], entry["strides"])
        )
        assert entry["offset_bytes"] == owner * backing["owner_stride_bytes"]
        assert span_elements * 2 <= backing["owner_stride_bytes"]


def test_pyntt_renderer_maps_local_kv_head_to_block_cyclic_global_head():
    _add_pyntt_to_path()
    from pyntt.codegen.render import _distributed_local_to_global_coordinates

    def fixed(value):
        return {"TritonExpression": str(value), "FixedValue": value}

    coordinates = _distributed_local_to_global_coordinates(
        ("token_id", "head_id", "dim_block"),
        [fixed(1), fixed(8), fixed(16)],
        [fixed(0), fixed(0), fixed(0)],
        [
            {"Stages": []},
            {
                "Stages": [
                    {
                        "HierarchyAxes": [1],
                        "Distribution": "BlockCyclic",
                        "Granularity": None,
                        "BlockSize": 1,
                    }
                ]
            },
            {"Stages": []},
        ],
        [4, 8],
    )

    assert coordinates[0] == "token_id"
    assert "head_id" in coordinates[1]
    assert "shard_coord1" in coordinates[1]
    assert "* 8" in coordinates[1]
    assert coordinates[2] == "dim_block"

    offset_coordinates = _distributed_local_to_global_coordinates(
        ("0",),
        [fixed(32)],
        [fixed(1)],
        [
            {
                "Stages": [
                    {
                        "HierarchyAxes": [1],
                        "Distribution": "BlockCyclic",
                        "Granularity": None,
                        "BlockSize": 1,
                    }
                ]
            }
        ],
        [4, 8],
    )
    assert "((1) // 1) * 8" in offset_coordinates[0]
    assert "shard_coord1" in offset_coordinates[0]


def test_pyntt_renderer_maps_local_packed_vector_coordinate_to_global_scalar():
    _add_pyntt_to_path()
    from pyntt.codegen.render import (
        _pointer_local_vector_to_global_scalar_coordinate,
    )

    def fixed(value):
        return {"TritonExpression": str(value), "FixedValue": value}

    pointer = {
        "DistributedStorageKind": "CompactLocal",
        "GlobalShape": [fixed(1), fixed(640)],
        "GlobalOffsets": [fixed(0), fixed(0)],
        "ShardAxes": [
            {"Stages": []},
            {
                "Stages": [
                    {
                        "HierarchyAxes": [0, 1],
                        "Distribution": "BlockCyclic",
                        "Granularity": None,
                        "BlockSize": 1,
                    }
                ]
            },
        ],
        "Hierarchy": [4, 8],
    }

    coordinate = _pointer_local_vector_to_global_scalar_coordinate(
        pointer, 1, "local_n", 8
    )

    assert "((local_n) // 8)" in coordinate
    assert "shard_coord0 * 8 + shard_coord1" in coordinate
    assert "* 32" in coordinate
    assert "* 8" in coordinate
    assert "((local_n) % 8)" in coordinate


def test_pyntt_renderer_composes_bounded_block_cyclic_coordinates():
    _add_pyntt_to_path()
    from pyntt.codegen.render import (
        _local_to_global_coordinate,
        _remap_local_coordinate,
        _scale_shard_axis_mapping,
    )

    def fixed(value):
        return {"TritonExpression": str(value), "FixedValue": value}

    hierarchy = [4, 8]
    head_mapping = {
        "Stages": [
            {
                "HierarchyAxes": [1],
                "Distribution": "BlockCyclic",
                "Granularity": None,
                "BlockSize": 2,
            }
        ]
    }
    assert _local_to_global_coordinate(
        "q_head", fixed(16), head_mapping, hierarchy, local_extent=2
    ) == "((shard_coord1) * 2 + (q_head))"
    assert (
        _remap_local_coordinate(
            "q_head",
            fixed(16),
            head_mapping,
            fixed(16),
            head_mapping,
            hierarchy,
            local_extent=2,
        )
        == "q_head"
    )

    physical_dim_mapping = {
        "Stages": [
            {
                "HierarchyAxes": [0],
                "Distribution": "BlockCyclic",
                "Granularity": None,
                "BlockSize": 4,
            }
        ]
    }
    scalar_dim_mapping = _scale_shard_axis_mapping(physical_dim_mapping, 8)
    assert scalar_dim_mapping["Stages"][0]["BlockSize"] == 32
    assert _local_to_global_coordinate(
        "dim_grid", fixed(128), scalar_dim_mapping, hierarchy, local_extent=32
    ) == "((shard_coord0) * 32 + (dim_grid))"


def test_pyntt_renderer_maps_gather_global_index_and_local_shard_coordinates():
    _add_pyntt_to_path()
    from pyntt.codegen.render import _access_pointer, _gather_template_context

    def fixed(value):
        return {"TritonExpression": str(value), "FixedValue": value}

    shard_axes = [
        {"Stages": []},
        {
            "Stages": [
                {
                    "HierarchyAxes": [0, 1],
                    "Distribution": "BlockCyclic",
                    "Granularity": None,
                    "BlockSize": 8,
                }
            ]
        },
    ]
    input_strides = [fixed(256), fixed(1)]
    model = {
        "Input": {
            "DistributedStorageKind": "CanonicalGlobal",
            "GlobalShape": [fixed(151936), fixed(256)],
            "GlobalOffsets": [fixed(0), fixed(0)],
            "Strides": input_strides,
            "ShardAxes": shard_axes,
            "Hierarchy": [4, 8],
        },
        "InputShape": [fixed(151936), fixed(8)],
        "InputGlobalShape": [fixed(151936), fixed(256)],
        "InputStrides": input_strides,
        "InputShardAxes": shard_axes,
        "IndexShape": [fixed(1)],
        "IndexStrides": [fixed(0)],
        "OutputShape": [fixed(1), fixed(8)],
        "OutputStrides": [fixed(0), fixed(1)],
        "Axis": 0,
        "ValueVectorLaneCount": 8,
        "ValueVectorLaneShape": [8],
        "Hierarchy": [4, 8],
        "IndexDType": "int32",
    }
    context = _gather_template_context(model)

    assert context["input_access"]["GlobalCoordinateAxes"] == (0,)
    pointer = _access_pointer(model, "Input", "input", context["input_access"])

    assert "gather_index" in pointer
    assert "shard_coord0" in pointer
    assert "shard_coord1" in pointer
    assert "major_raw" in pointer
    assert "((gather_index) //" not in pointer


def test_pyntt_renderer_canonical_pointer_applies_view_origin_before_sharding():
    _add_pyntt_to_path()
    from pyntt.codegen.render import _access_pointer, _tensor_access

    def fixed(value):
        return {"TritonExpression": str(value), "FixedValue": value}

    pointer_model = {
        "Input": {
            "DistributedStorageKind": "CanonicalGlobal",
            "GlobalShape": [fixed(1), fixed(64)],
            "GlobalOffsets": [fixed(0), fixed(2)],
            "Strides": [fixed(64), fixed(1)],
            "ShardAxes": [
                {"Stages": []},
                {
                    "Stages": [
                        {
                            "HierarchyAxes": [1],
                            "Distribution": "BlockCyclic",
                            "Granularity": None,
                            "BlockSize": 1,
                        }
                    ]
                },
            ],
            "Hierarchy": [4, 8],
        }
    }
    access = _tensor_access(("0", "local_index"), (fixed(64), fixed(1)))

    pointer = _access_pointer(pointer_model, "Input", "input", access)

    assert "(2) + (local_index)" in pointer
    assert pointer.count("shard_coord1") == 1


def test_pyntt_softmax_uses_coordinate_native_canonical_global_accesses():
    _add_pyntt_to_path()
    from pyntt.codegen.render import _access_pointer, _softmax_template_context

    def fixed(value):
        return {
            "TritonExpression": str(value),
            "FixedValue": value,
            "MaxValue": value,
        }

    strides = [fixed(0), fixed(1)]
    model = {
        "Input": {
            "DistributedStorageKind": "CanonicalGlobal",
            "GlobalShape": [fixed(1), fixed(256)],
            "GlobalOffsets": [fixed(0), fixed(0)],
            "Strides": strides,
            "ShardAxes": [{"Stages": []}, {"Stages": []}],
            "Hierarchy": [4, 8],
        },
        "Shape": [fixed(1), fixed(256)],
        "InputStrides": strides,
        "OutputStrides": strides,
        "Axis": 1,
    }

    context = _softmax_template_context(model)
    input_pointer = _access_pointer(
        model, "Input", "input", context["input_access"]
    )

    assert context["axis_block_size"] == 256
    assert context["loop_axes"] == (0,)
    assert context["input_access"]["TensorIndices"] == ("idx0", "lane")
    assert "lane" in input_pointer


def test_pyntt_softmax_uses_dynamic_axis_maximum_for_static_reduction_capacity():
    _add_pyntt_to_path()
    from pyntt.codegen.render import _softmax_template_context

    def fixed(value):
        return {
            "TritonExpression": str(value),
            "FixedValue": value,
            "MaxValue": value,
        }

    dynamic_axis = {
        "TritonExpression": "sequence_length",
        "FixedValue": None,
        "MaxValue": 300,
    }
    context = _softmax_template_context(
        {
            "Shape": [fixed(2), dynamic_axis],
            "InputStrides": [dynamic_axis, fixed(1)],
            "OutputStrides": [dynamic_axis, fixed(1)],
            "Axis": 1,
        }
    )

    assert context["axis_block_size"] == 512
    assert context["axis_extent"] is dynamic_axis


def test_pyntt_renderer_owns_block_schedule_config():
    _add_pyntt_to_path()
    from pyntt.codegen.render import render_manifest

    manifest = _test_pyntt_codegen_manifest(
        {"metadata": {"name": "top"}, "body_source": "pass"}
    )
    launch = manifest["functions"][0]["render_kernels"][0]["metadata"]["launch"]
    source = render_manifest(manifest)
    assert "PYNTT_KERNEL_CONFIGS" in source
    assert "'source': 'autotune'" in source
    assert "'candidates': (32, 128, 256, 512, 1024)" in source
    assert "'num_warps': 8" in source
    assert "'num_stages': 1" in source
    assert "'resident_blocks_per_compute_unit': 1" in source

    launch["num_warps"] = 8
    with pytest.raises(ValueError, match=r"unexpected fields \['num_warps'\]"):
        render_manifest(manifest)
    launch.pop("num_warps")
    launch["num_stages"] = 1
    with pytest.raises(ValueError, match=r"unexpected fields \['num_stages'\]"):
        render_manifest(manifest)
    launch.pop("num_stages")
    launch["tuning"] = {"parameters": {}}
    with pytest.raises(ValueError, match=r"unexpected fields \['tuning'\]"):
        render_manifest(manifest)


@pytest.mark.parametrize(
    ("block_k", "expected"),
    ((256, False), (512, True), (1024, True)),
)
def test_pyntt_renderer_outlines_only_large_packed_gemv_stages(block_k, expected):
    _add_pyntt_to_path()
    from pyntt.codegen.render import _should_outline_packed_gemv_consumer_stage

    assert (
        _should_outline_packed_gemv_consumer_stage(
            block_k=block_k,
            reduction_group=32,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("cache_block_size", "expected_block_n", "expected_candidates"),
    [(32, 32, (32,)), (256, 64, (32, 64))],
)
def test_pyntt_renderer_constrains_attention_tile_to_cache_page(
    cache_block_size, expected_block_n, expected_candidates
):
    _add_pyntt_to_path()
    from pyntt.codegen.render import _paged_attention_backend_config

    kernel = {
        "metadata": {"name": "top"},
        "helpers": [],
        "device_functions": [
            {
                "helpers": [
                    {
                        "template": (
                            "triton/kernels/paged_attention/mma_direct.py.jinja"
                        ),
                        "model": {"Cache": {"BlockSize": cache_block_size}},
                    }
                ]
            }
        ],
    }

    config = _paged_attention_backend_config(kernel)
    assert config["block_n"] == expected_block_n
    assert config["block_n_candidates"] == expected_candidates


@pytest.mark.parametrize(
    ("block_n", "page_size", "expected"),
    [
        (64, 32, (32, 2)),
        (128, 32, (32, 4)),
        (128, 256, (128, 1)),
    ],
)
def test_pyntt_paged_attention_tma_tile_can_span_integral_cache_pages(
    block_n, page_size, expected
):
    _add_pyntt_to_path()
    from pyntt.codegen.render import _paged_attention_tile_geometry

    assert _paged_attention_tile_geometry(
        block_n, page_size, allow_cross_page=True
    ) == expected


def test_pyntt_paged_attention_direct_tile_cannot_span_cache_pages():
    _add_pyntt_to_path()
    from pyntt.codegen.render import _paged_attention_tile_geometry

    with pytest.raises(ValueError, match="requires a transfer pipeline"):
        _paged_attention_tile_geometry(64, 32, allow_cross_page=False)


def test_pyntt_paged_attention_detects_noncontiguous_gqa_head_shards():
    _add_pyntt_to_path()
    from pyntt.codegen.render import _query_heads_form_contiguous_gqa_groups

    def block_cyclic(block_size):
        return {
            "Stages": [
                {
                    "HierarchyAxes": [1],
                    "Distribution": "BlockCyclic",
                    "Granularity": None,
                    "BlockSize": block_size,
                }
            ]
        }

    assert _query_heads_form_contiguous_gqa_groups(
        block_cyclic(2),
        [4, 8],
        global_query_heads=16,
        local_query_heads=2,
        query_heads_per_kv_head=2,
    )
    assert not _query_heads_form_contiguous_gqa_groups(
        block_cyclic(1),
        [4, 9],
        global_query_heads=16,
        local_query_heads=2,
        query_heads_per_kv_head=2,
    )


@pytest.mark.parametrize(
    "template_name",
    (
        "simt_direct.py.jinja",
        "mma_direct.py.jinja",
        "simt_tma_smem_pipeline.py.jinja",
        "mma_tma_smem_pipeline.py.jinja",
    ),
)
def test_pyntt_paged_attention_narrows_bounded_split_indices(template_name):
    template = (
        Path(__file__).parents[2]
        / "pyntt/pyntt/codegen/templates/triton/kernels/paged_attention"
        / template_name
    ).read_text(encoding="utf-8")
    lines = tuple(line.strip() for line in template.splitlines())
    split_begin_lines = tuple(
        line for line in lines if line.startswith("split_begin =")
    )
    split_end_lines = tuple(
        line for line in lines if line.startswith("split_end =")
    )
    causal_end_lines = tuple(
        line for line in lines if line.startswith("causal_end =")
    )

    assert split_begin_lines
    assert len(split_begin_lines) == len(split_end_lines)
    assert all(line.endswith(".to(tl.int32)") for line in split_begin_lines)
    assert all(line.endswith(".to(tl.int32)") for line in split_end_lines)
    assert all(line.endswith(".to(tl.int32)") for line in causal_end_lines)


def test_pyntt_qkv_pipeline_uses_one_canonical_fused_n_transfer():
    _add_pyntt_to_path()
    from pyntt.codegen.render import _packed_qkv_fixed_projection_ns

    projection_ns = _packed_qkv_fixed_projection_ns(
        {"ProjectionNCapacities": [96, 32, 32]}
    )
    template = (
        Path(__file__).resolve().parents[2]
        / "pyntt/pyntt/codegen/templates/triton/kernels/qkv_parallel_linear"
        / "simt_fma_smem_pipeline.py.jinja"
    ).read_text(encoding="utf-8")

    assert projection_ns == {"Q": 96, "K": 32, "V": 32}
    assert template.count("tle.gpu.copy(") == 1
    assert "for n_tile in tl.static_range(" in template
    assert "for q_tile" not in template
    assert "for k_projection_tile" not in template
    assert "for v_tile" not in template


def test_pyntt_qkv_pipeline_requires_three_projection_capacities():
    _add_pyntt_to_path()
    from pyntt.codegen.render import _packed_qkv_fixed_projection_ns

    with pytest.raises(ValueError, match="three projection N capacities"):
        _packed_qkv_fixed_projection_ns({"ProjectionNCapacities": [64, 32]})


def test_pyntt_renderer_marks_dynamic_top_kernel_scalars_non_specializing():
    _add_pyntt_to_path()
    from pyntt.codegen.render import render_manifest

    manifest = _test_pyntt_codegen_manifest(
        {
            "metadata": {
                "name": "top",
                "attrs": {
                    "runtime_scalar_input_args": ["input0"],
                    "runtime_shape_args": ["sequence_length"],
                    "abi_view_stride_args": ["input0_scalar_stride0"],
                },
            },
            "body_source": "pass",
        }
    )

    source = render_manifest(manifest)
    assert (
        "@triton.jit(do_not_specialize=('input0', 'input0_scalar_stride0', "
        "'sequence_length', 'numel'))" in source
    )


def test_pyntt_renderer_rejects_non_integral_target_worker_geometry():
    _add_pyntt_to_path()
    from pyntt.codegen.render import render_manifest

    manifest = _test_pyntt_codegen_manifest(
        {
            "metadata": {
                "name": "top",
                "attrs": {
                    "target_worker_width": 32,
                    "target_threads_per_block": 130,
                },
            },
            "body_source": "pass",
        }
    )
    with pytest.raises(ValueError, match="must be divisible"):
        render_manifest(manifest)


@pytest.mark.parametrize(
    "removed_field",
    ["pipeline_executions", "shared_arena", "local_buffers", "microkernels"],
)
def test_pyntt_renderer_rejects_removed_scheduling_fields(removed_field):
    _add_pyntt_to_path()
    from pyntt.codegen.render import render_manifest

    manifest = _test_pyntt_codegen_manifest(
        {"metadata": {"name": "top"}, "body_source": "pass"}
    )
    kernel = manifest["functions"][0]["render_kernels"][0]
    kernel[removed_field] = []
    with pytest.raises(ValueError, match="unexpected fields"):
        render_manifest(manifest)


def test_pyntt_renderer_rejects_unknown_manifest_fields():
    _add_pyntt_to_path()
    from pyntt.codegen.render import render_manifest

    manifest = _test_pyntt_codegen_manifest(
        {"metadata": {"name": "top"}, "body_source": "pass"}
    )
    manifest["legacy_pipeline_stages"] = 3
    with pytest.raises(ValueError, match="unexpected fields"):
        render_manifest(manifest)


def test_pyntt_kernel_templates_own_their_triton_source():
    _add_pyntt_to_path()
    from pyntt.codegen.render import _make_env

    template_dir = (
        Path(__file__).resolve().parents[2]
        / "pyntt/pyntt/codegen/templates/triton/kernels"
    )
    env = _make_env()
    for template_path in template_dir.rglob("*.jinja"):
        source = template_path.read_text(encoding="utf-8")
        assert "{{ emit(" not in source
        assert "pipeline_executions" not in source
        assert "shared_arena" not in source.replace("pyntt_shared_arena", "")
        relative_path = template_path.relative_to(template_dir).as_posix()
        env.get_template(f"triton/kernels/{relative_path}")

    algorithm_dirs = (
        template_dir / "matmul",
        template_dir / "qkv_parallel_linear",
        template_dir / "matmul_glu",
    )
    for algorithm_dir in algorithm_dirs:
        for template_path in algorithm_dir.glob("*.py.jinja"):
            source = template_path.read_text(encoding="utf-8")
            parsed = env.parse(source)
            references = set(meta.find_referenced_templates(parsed))
            assert references <= {
                "triton/kernels/_common.py.jinja",
                "triton/kernels/_producer_consumer.py.jinja",
            }

    executable_templates = (
        template_path
        for template_path in template_dir.rglob("*.py.jinja")
        if not template_path.name.startswith("_")
    )
    for template_path in executable_templates:
        source = template_path.read_text(encoding="utf-8")
        assert (
            "phases.dispatch(model" in source
            or 'model["FunctionName"] }}__producer' in source
        ), f"{template_path} does not implement the producer/consumer contract"

    common_source = (template_dir / "_common.py.jinja").read_text(encoding="utf-8")
    assert "alias=pyntt_shared_arena" in common_source
    assert 'ctx["microkernel"]["shared_workspace_offsets"]' in common_source


def test_pyntt_renderer_preserves_codegen_scope_device_boundary():
    _add_pyntt_to_path()
    from pyntt.codegen.render import render_manifest

    source = _render_test_pyntt_manifest(
        render_manifest,
        {
            "functions": [
                {
                    "render_kernels": [
                        {
                            "metadata": {"name": "top"},
                            "body_source": (
                                "__pyntt_device_call__child(shard_index)\n"
                                "__pyntt_device_call__child(shard_index)"
                            ),
                            "device_functions": [
                                _device_function(
                                    "child", "child_shard_index", ("child_shard_index",)
                                )
                            ],
                        }
                    ]
                }
            ]
        },
    )
    assert "tl.program_id(0)" not in source
    assert source.count(
        "tle.shard_id(PYNTT_GRID_MESH, 'block_b').to(tl.int64)"
    ) == 2
    assert source.count("child(shard_index)") == 2
    assert "def child(child_shard_index):" in source
    assert "pyntt_call_frame" not in source


def test_pyntt_renderer_materializes_named_mesh_coordinates():
    _add_pyntt_to_path()
    from pyntt.codegen.render import render_manifest

    manifest = _test_pyntt_codegen_manifest(
        {
            "metadata": {"name": "top"},
            "body_source": "value = shard_coord0 + shard_coord1 + shard_index",
        }
    )
    sharding = manifest["functions"][0]["render_kernels"][0]["metadata"][
        "launch"
    ]["sharding"]
    sharding["placement_axis"] = "yx"
    sharding["hierarchy"] = [4, 8]
    sharding["hierarchy_levels"] = "bb"

    source = render_manifest(manifest)

    assert '_PYNTT_GRID_MESH_VALUE = tle.device_mesh({"block": [(\'block_y\', 4), (\'block_x\', 8)]})' in source
    assert "shard_coord0 = tle.shard_id(PYNTT_GRID_MESH, 'block_y')" in source
    assert "shard_coord1 = tle.shard_id(PYNTT_GRID_MESH, 'block_x')" in source
    assert "shard_index = (shard_coord0 * 8 + shard_coord1)" in source
    assert "tl.program_id(0)" not in source
    assert "shard_index //" not in source
    assert "shard_index %" not in source


def test_pyntt_renderer_passes_nested_device_arguments_directly():
    _add_pyntt_to_path()
    from pyntt.codegen.render import render_manifest

    source = _render_test_pyntt_manifest(
        render_manifest,
        {
            "functions": [
                {
                    "render_kernels": [
                        {
                            "metadata": {"name": "top"},
                            "body_source": "__pyntt_device_call__parent(data)",
                            "device_functions": [
                                _device_function(
                                    "parent",
                                    "__pyntt_device_call__child(parent_ptr)",
                                    ("parent_ptr",),
                                ),
                                _device_function("child", "pass", ("child_ptr",)),
                            ],
                        }
                    ]
                }
            ]
        },
    )
    assert "def parent(parent_ptr):" in source
    assert "parent(data)" in source
    assert "child(parent_ptr)" in source
    assert "pyntt_call_frame" not in source


def test_pyntt_renderer_marks_nested_device_helpers_noinline(monkeypatch):
    _add_pyntt_to_path()
    import pyntt.codegen.render as render

    captured = {}

    def capture_helper_options(*args, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(render, "_render_helper_sources", capture_helper_options)
    device_function = _device_function("child", "pass")
    device_function["direct_parameters"] = []
    device_function["direct_extra_parameter_declarations"] = []
    render._render_device_function(
        render._make_env(),
        device_function,
        {},
        8,
        32,
        1,
        24,
        8,
        255,
        {},
        (),
    )

    assert captured["noinline"] is True


def test_pyntt_renderer_reduces_contiguous_axes_first():
    _add_pyntt_to_path()
    from pyntt.codegen.render import _norm_stats_template_context, _reduce_all_axes

    assert _reduce_all_axes("value", 3) == (
        "tl.sum(tl.sum(tl.sum(value, axis=2), axis=1), axis=0)"
    )
    context = _norm_stats_template_context(
        {
            "InputShape": [1, 256],
            "InputStrides": [256, 1],
            "InputVectorLaneShape": [8],
            "InputVectorLaneCount": 8,
            "OutputShape": [1, 1, 1],
            "OutputStrides": [1, 1, 1],
            "OutputVectorLaneShape": [],
            "Axis": 1,
        }
    )
    assert context["flat_reduction"] is True
    assert context["reduction_extent"] == "((256)) * 8"
    assert context["reduction"] == "tl.sum(mean_partial, axis=0)"
    assert context["square_reduction"] == "tl.sum(square_partial, axis=0)"

    template = Path(
        "pyntt/pyntt/codegen/templates/triton/kernels/NormStats.py.jinja"
    ).read_text()
    assert "loop_unroll_factor=1" in template

    strided = _norm_stats_template_context(
        {
            "InputShape": [1, 256],
            "InputStrides": [512, 2],
            "InputVectorLaneShape": [8],
            "InputVectorLaneCount": 8,
            "OutputShape": [1, 1, 1],
            "OutputStrides": [1, 1, 1],
            "OutputVectorLaneShape": [],
            "Axis": 1,
        }
    )
    assert strided["flat_reduction"] is False
    assert strided["square_reduction"] == (
        "tl.sum(tl.sum(square_partial, axis=1), axis=0)"
    )


def test_pyntt_renderer_propagates_only_live_canonical_device_parameters():
    _add_pyntt_to_path()
    from pyntt.codegen.render import render_manifest

    source = _render_test_pyntt_manifest(
        render_manifest,
        {
            "functions": [
                {
                    "render_kernels": [
                        {
                            "metadata": {
                                "name": "top",
                                "inputs": ["unused", "live"],
                                "attrs": {"runtime_shape_args": ["extent"]},
                            },
                            "body_source": "__pyntt_device_call__parent()",
                            "device_functions": [
                                _device_function(
                                    "parent",
                                    "tl.load(rdata)\n__pyntt_device_call__child()",
                                ),
                                _device_function(
                                    "child", "tl.load(input1) + extent + tl.load(rdata)"
                                ),
                            ],
                        }
                    ]
                }
            ]
        },
    )
    parent_parameters = source.split("def parent(", 1)[1].split("):", 1)[0]
    child_parameters = source.split("def child(", 1)[1].split("):", 1)[0]
    assert parent_parameters == "input1, rdata, extent"
    assert child_parameters == "input1, rdata, extent"
    assert "parent(input1, rdata, extent)" in source
    assert "child(input1, rdata, extent)" in source
    assert "pyntt_call_frame" not in source


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "(tl.full((), 0, tl.int32) + (0)).to(tl.int32)"),
        (
            "descriptor_origin",
            "(tl.full((), 0, tl.int32) + (descriptor_origin)).to(tl.int32)",
        ),
    ],
)
def test_pyntt_tma_offset_is_type_stable_after_specialization(value, expected):
    _add_pyntt_to_path()
    from pyntt.codegen.render import _tma_offset

    assert _tma_offset(value) == expected


class _FakeCompiledKernel:
    def __init__(
        self,
        *,
        num_warps=8,
        shared=0,
        registers=32,
        spill_stores=0,
        spill_loads=0,
        stack=0,
        local=0,
    ):
        self.hash = object()
        self.name = "fake_kernel"
        self.metadata = type(
            "Metadata", (), {"num_warps": num_warps, "shared": shared}
        )()
        self.n_regs = registers
        self.n_spill_stores = spill_stores
        self.n_spill_loads = spill_loads
        self.n_stack_bytes = stack
        self.n_local_bytes = local

    def _init_handles(self):
        pass


class _FakeJitKernel:
    def __init__(self, compiled):
        self.compiled = compiled
        self.calls = []

    def run(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.compiled


class _FakeTunableJitKernel:
    def __init__(self, compiled_by_candidate):
        self.compiled_by_candidate = compiled_by_candidate
        self.attempts = []
        self.prepare_calls = []
        self.prepared = []

    def prepare(self, *args, **kwargs):
        candidate = int(args[-1])
        self.attempts.append(candidate)
        self.prepare_calls.append((args, kwargs))
        result = self.compiled_by_candidate[candidate]
        if isinstance(result, BaseException):
            raise result
        prepared = _FakePreparedKernel(result)
        self.prepared.append(prepared)
        return prepared


class _FakePreparedKernel:
    def __init__(self, compiled):
        self.compiled_kernel = compiled
        self.launches = []

    def launch(self, *args, **kwargs):
        self.launches.append((args, kwargs))


def test_pyntt_runtime_accepts_kernel_within_fixed_resource_budget():
    _add_pyntt_to_path()
    from pyntt.runtime.triton import validate_triton_kernel_resources

    argument = object()
    kernel = _FakeJitKernel(
        _FakeCompiledKernel(shared=4096, registers=64, stack=32, local=8)
    )
    validate_triton_kernel_resources(
        kernel,
        argument,
        grid=(36,),
        expected_compute_num_warps=8,
        expected_resident_blocks_per_compute_unit=1,
        registers_per_thread_limit=255,
        shared_memory_capacity_bytes=101_376,
        forbid_spills=True,
        min_ctas_per_sm=1,
    )
    assert kernel.calls[0][0][0] is argument
    assert kernel.calls[0][1]["warmup"] is True


def test_pyntt_runtime_accepts_backend_warp_specialization_workers():
    _add_pyntt_to_path()
    from pyntt.runtime.triton import validate_triton_kernel_resources

    kernel = _FakeJitKernel(_FakeCompiledKernel(num_warps=12))
    validate_triton_kernel_resources(
        kernel,
        grid=(36,),
        expected_compute_num_warps=8,
        expected_resident_blocks_per_compute_unit=1,
        registers_per_thread_limit=255,
        shared_memory_capacity_bytes=101_376,
        forbid_spills=True,
        num_warps=8,
        min_ctas_per_sm=1,
    )


@pytest.mark.parametrize(
    ("launch_options", "message"),
    [
        ({}, "does not specify min_ctas_per_sm"),
        ({"min_ctas_per_sm": 2}, "requires 1 resident blocks"),
    ],
)
def test_pyntt_runtime_requires_target_resident_block_contract(
    launch_options, message
):
    _add_pyntt_to_path()
    from pyntt.runtime.triton import validate_triton_kernel_resources

    with pytest.raises(RuntimeError, match=message):
        validate_triton_kernel_resources(
            _FakeJitKernel(_FakeCompiledKernel()),
            grid=(36,),
            expected_compute_num_warps=8,
            expected_resident_blocks_per_compute_unit=1,
            registers_per_thread_limit=255,
            shared_memory_capacity_bytes=101_376,
            forbid_spills=True,
            **launch_options,
        )


@pytest.mark.parametrize(
    ("compiled", "message"),
    [
        (_FakeCompiledKernel(num_warps=4), "requires at least 8"),
        (_FakeCompiledKernel(registers=256), "registers per thread"),
        (_FakeCompiledKernel(shared=1024), "shared-memory bytes"),
        (_FakeCompiledKernel(spill_stores=4), "forbids register spilling"),
    ],
)
def test_pyntt_runtime_rejects_kernel_outside_fixed_resource_budget(compiled, message):
    _add_pyntt_to_path()
    from pyntt.runtime.triton import validate_triton_kernel_resources

    shared_capacity = 512 if compiled.metadata.shared else 101_376
    with pytest.raises(RuntimeError, match=message):
        validate_triton_kernel_resources(
            _FakeJitKernel(compiled),
            grid=(36,),
            expected_compute_num_warps=8,
            expected_resident_blocks_per_compute_unit=1,
            registers_per_thread_limit=255,
            shared_memory_capacity_bytes=shared_capacity,
            forbid_spills=True,
            min_ctas_per_sm=1,
        )


def test_pyntt_runtime_prepares_first_resource_feasible_tuning_candidate():
    _add_pyntt_to_path()
    from pyntt.runtime.triton import prepare_and_validate_triton_kernel
    from triton.runtime.errors import OutOfResources

    kernel = _FakeTunableJitKernel(
        {
            128: _FakeCompiledKernel(registers=64),
            256: _FakeCompiledKernel(spill_stores=4),
            512: OutOfResources(128 * 1024, 96 * 1024, "shared memory"),
        }
    )
    kwargs = {
        "source": "search_space",
        "kernel": kernel,
        "kernel_args": (),
        "grid_for_candidate": lambda _: (1,),
        "expected_compute_num_warps": 8,
        "expected_resident_blocks_per_compute_unit": 1,
        "registers_per_thread_limit": 255,
        "shared_memory_capacity_bytes": 101_376,
        "forbid_spills": True,
        "num_warps": 8,
        "min_ctas_per_sm": 1,
    }
    prepared = prepare_and_validate_triton_kernel(
        "test_kernel", "block_size", (128, 256, 512), **kwargs
    )
    assert prepared.parameter_value == 128
    assert kernel.attempts == [512, 256, 128]
    assert all(
        call_kwargs["min_ctas_per_sm"] == 1
        for _, call_kwargs in kernel.prepare_calls
    )

    prepared.launch(grid=(4,))
    assert kernel.prepared[-1].launches == [((), {"grid": (4,), "stream": None})]


def test_pyntt_runtime_requires_prepared_triton_abi():
    _add_pyntt_to_path()
    from pyntt.runtime.triton import prepare_and_validate_triton_kernel

    with pytest.raises(RuntimeError, match=r"JITFunction\.prepare"):
        prepare_and_validate_triton_kernel(
            "test_kernel",
            "block_size",
            (128,),
            source="search_space",
            kernel=_FakeJitKernel(_FakeCompiledKernel()),
            kernel_args=(),
            grid_for_candidate=lambda _: (1,),
            expected_compute_num_warps=8,
            expected_resident_blocks_per_compute_unit=1,
            registers_per_thread_limit=255,
            shared_memory_capacity_bytes=101_376,
            forbid_spills=True,
            num_warps=8,
            min_ctas_per_sm=1,
        )


def test_pyntt_interpreter_isolates_prepared_state_by_execution_device():
    _add_pyntt_to_path()
    from pyntt.ir import ModuleSpec
    from pyntt.runtime.interpreter import PyNTTInterpreter

    interpreter = PyNTTInterpreter(ModuleSpec("test", "triton", ()))
    prepared0 = object()
    prepared1 = object()
    resources0 = (object(),)
    resources1 = (object(),)

    interpreter.store_prepared_triton_kernel("top", "cuda:0", prepared0)
    interpreter.store_prepared_triton_kernel("top", "cuda:1", prepared1)
    interpreter.store_launch_resources("top", "cuda:0", resources0)
    interpreter.store_launch_resources("top", "cuda:1", resources1)

    assert interpreter.lookup_prepared_triton_kernel("top", "cuda:0") is prepared0
    assert interpreter.lookup_prepared_triton_kernel("top", "cuda:1") is prepared1
    assert interpreter.lookup_launch_resources("top", "cuda:0") is resources0
    assert interpreter.lookup_launch_resources("top", "cuda:1") is resources1
