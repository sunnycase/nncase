import sys
from pathlib import Path

import pytest


def _add_pyntt_to_path():
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "pyntt"))


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
    assert select_tuning_parameter("main", "block_size", (128, 256), source="search_space") == 256
    assert local_shard_1d(33, 0, 2) == LocalShard(offset=0, extent=17)
    assert local_shard_1d(33, 1, 2) == LocalShard(offset=17, extent=16)
    assert local_shard_1d(2, 3, 4) == LocalShard(offset=3, extent=0)

    sharded = sharded_tensor((4, 8), tensor_axis=1)
    assert sharded.placement_axis == "b"
    assert sharded.local_offsets(1, 3) == (0, 3)
    assert sharded.local_shape(1, 3) == (4, 3)


def test_pyntt_runtime_validates_torch_inputs_and_allocates_outputs(tmp_path):
    torch = pytest.importorskip("torch")
    _add_pyntt_to_path()

    from pyntt.ir import FunctionSpec, ModuleSpec, TensorResultSpec, TensorSpec
    from pyntt.runtime import PyNTTArgumentError, PyNTTInterpreter, PyNTTModule
    from pyntt.runtime import allocate_workspace, materialize_rdata, materialize_rdata_table

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
    interpreter = PyNTTInterpreter(spec).load()

    x = torch.ones((2, 3), dtype=torch.float32)
    y = module(x)
    z = interpreter.run(x)

    assert isinstance(y, torch.Tensor)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.device == x.device
    assert isinstance(z, torch.Tensor)
    assert z.shape == x.shape

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
    assert rdata.device == x.device
    assert rdata.tolist() == [1, 2, 3]
    assert rdata_again.data_ptr() == rdata.data_ptr()

    rdata_table_path0 = tmp_path / "rdata_table_0.bin"
    rdata_table_path1 = tmp_path / "rdata_table_1.bin"
    rdata_table_path0.write_bytes(bytes([1]))
    rdata_table_path1.write_bytes(bytes([2]))
    rdata_table = materialize_rdata_table(
        (x,), (f"file:{rdata_table_path0}", f"file:{rdata_table_path1}"), 1
    )
    rdata_table_again = materialize_rdata_table(
        (x,), (f"file:{rdata_table_path0}", f"file:{rdata_table_path1}"), 1
    )
    assert rdata_table.dtype == torch.uint8
    assert rdata_table.tolist() == [1, 2]
    assert rdata_table_again.data_ptr() == rdata_table.data_ptr()


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


def test_pyntt_renderer_rejects_incompatible_manifest_version():
    _add_pyntt_to_path()

    from pyntt.codegen.render import render_manifest

    with pytest.raises(ValueError, match="expected 3"):
        render_manifest({"pyntt_codegen_manifest_version": 1, "functions": []})


def test_pyntt_renderer_uses_explicit_device_arguments_and_tiling_shared_arena():
    _add_pyntt_to_path()

    from pyntt.codegen.render import render_manifest

    source = render_manifest(
        {
            "pyntt_codegen_manifest_version": 3,
            "functions": [
                {
                    "render_kernels": [
                        {
                            "metadata": {
                                "name": "top",
                                "inputs": [],
                                "outputs": [],
                                "attrs": {
                                    "shared_memory_bytes": 65_536,
                                    "shared_memory_capacity_bytes": 101_376,
                                    "shared_memory_allocation_size_policy": "power_of_two",
                                    "shared_memory_allocation_granularity_bytes": 16,
                                },
                                "launch": {"num_warps": 8},
                            },
                            "body_source": "__pyntt_device_call__child(data, numel)",
                            "device_functions": [
                                {
                                    "name": "child",
                                    "noinline": True,
                                    "preserve_helper_call_boundaries": False,
                                    "helpers": [],
                                    "body_source": "tl.load(child_data) + child_extent",
                                    "parameter_overrides": {},
                                    "extra_parameters": [
                                        "child_data",
                                        "child_extent",
                                    ],
                                    "extra_parameter_arguments": {},
                                }
                            ],
                        }
                    ]
                }
            ]
        }
    )

    assert "import triton.experimental.tle.language as tle" in source
    assert source.count("pyntt_shared_arena = tle.gpu.alloc([65536]") == 1
    assert "@triton.jit(noinline=True)\ndef child(" in source
    assert "def child(pyntt_shared_arena, child_data, child_extent):" in source
    assert "child(pyntt_shared_arena, data, numel)" in source
    assert "tl.load(" in source
    assert "pyntt_shared_arena" in source
    assert "pyntt_call_frame" not in source


def test_pyntt_renderer_inlines_helpers_with_tensor_value_abi():
    _add_pyntt_to_path()

    from pyntt.codegen.render import _render_helper_sources

    rendered_models = []

    class CapturingTemplate:
        def render(self, *, model):
            rendered_models.append(model)
            return "helper"

    class CapturingEnvironment:
        def get_template(self, _):
            return CapturingTemplate()

    helpers = (
        {
            "template": "unused",
            "model": {},
            "arguments": (),
            "requires_inline": True,
        },
        {
            "template": "unused",
            "model": {},
            "arguments": (),
            "requires_inline": False,
        },
    )
    _render_helper_sources(
        CapturingEnvironment(), helpers, noinline=True, num_warps=8
    )

    assert [model["NoInline"] for model in rendered_models] == [False, True]
    assert [model["NumWarps"] for model in rendered_models] == [8, 8]


def test_pyntt_gemv_register_accumulator_is_lowered_by_the_selected_template():
    _add_pyntt_to_path()

    from pyntt.codegen.render import _make_env

    def fixed(value):
        return {
            "PythonExpression": str(value),
            "TritonExpression": str(value),
            "FixedValue": value,
            "RangeMin": None,
            "RangeMax": None,
            "IsFixed": True,
            "IsFixedOne": value == 1,
            "IsFixedNonOne": value != 1,
            "MinValue": value,
            "MaxValue": value,
        }

    exact_dynamic_k = {
        "PythonExpression": "min(256, remaining_k)",
        "TritonExpression": "tl.minimum(256, remaining_k)",
        "FixedValue": None,
        "RangeMin": 256,
        "RangeMax": 256,
        "MinValue": 256,
        "MaxValue": 256,
    }
    pointer = {
        "Expression": "unused",
        "ShardCoordHierarchy": None,
        "AddressSpace": 1,
        "LocalBuffer": None,
    }
    model = {
        "FunctionName": "register_gemv_accumulate",
        "Lhs": pointer,
        "Rhs": pointer,
        "Output": pointer,
        "LhsDType": "bfloat16",
        "RhsDType": "bfloat16",
        "OutputDType": "bfloat16",
        "LhsTritonDType": "tl.bfloat16",
        "RhsTritonDType": "tl.bfloat16",
        "OutputTritonDType": "tl.bfloat16",
        "LhsShape": [fixed(1), exact_dynamic_k],
        "RhsShape": [fixed(256), exact_dynamic_k],
        "OutputShape": [fixed(1), fixed(256)],
        "LhsStrides": [fixed(256), fixed(1)],
        "RhsStrides": [fixed(256), fixed(1)],
        "OutputStrides": [fixed(256), fixed(1)],
        "TransposeA": False,
        "TransposeB": True,
        "Hierarchy": [4, 8],
        "RhsNVectorLaneCount": 1,
        "OutputNVectorLaneCount": 1,
        "RhsNPackedLaneCount": 1,
        "OutputNPackedLaneCount": 1,
        "Scale": "1",
        "Comment": "register GEMV regression",
        "RuntimeShapeArgs": [],
        "LoadCExpression": "False",
        "ReductionPhase": "accumulate",
        "ReductionBlockM": 1,
        "ReductionBlockN": 256,
        "ReductionBlockK": 256,
        "MicroKernelFamily": "triton.gemv",
        "MicroKernelVariant": "register_mma_accumulator",
        "MicroKernelParameters": {
            "contract_version": 3,
            "state_block_m": 1,
            "state_block_n": 256,
            "state_block_k": 256,
            "inner_m": 8,
            "inner_n": 128,
            "inner_k": 16,
            "pipeline_stages": 2,
            "mma_m": 16,
            "mma_n": 8,
            "mma_k": 16,
        },
        "NumWarps": 8,
        "NoInline": False,
    }

    source = _make_env().get_template("triton/kernels/Gemv.py.jinja").render(
        model=model
    )

    assert "for acc_n_start" not in source
    assert "rhs_n_physical = tl.arange(0, 256)[:, None]" in source
    assert "rhs_n_logical = rhs_n_physical" in source
    assert "offs_n" not in source
    assert "offs_m = tl.arange(0, 8)" in source
    assert "lhs_columns = tl.where(offs_m[None, :] == 0, lhs_values[:, None], 0.0)" in source
    assert "partial_transposed = tl.dot(rhs_values, lhs_columns)" in source
    assert "acc += tl.sum(partial_transposed, axis=1)" in source
    assert "tl.sum(rhs_values * lhs_values[None, :], axis=1)" not in source
    assert "tle.gpu.local_ptr(acc" not in source
    assert "mask=" not in source
    assert "return acc" in source

    simt_model = dict(model)
    simt_model["MicroKernelVariant"] = "register_simt_accumulator"
    simt_model["MicroKernelParameters"] = {
        "contract_version": 3,
        "state_block_m": 1,
        "state_block_n": 256,
        "state_block_k": 256,
        "inner_m": 1,
        "inner_n": 256,
        "inner_k": 256,
        "pipeline_stages": 1,
    }
    simt_source = _make_env().get_template(
        "triton/kernels/Gemv.py.jinja"
    ).render(model=simt_model)
    assert "tl.dot(" not in simt_source
    assert "for reduction_k_start" not in simt_source
    assert "partial = acc" in simt_source
    assert "offs_k = tl.arange(0, 256)" in simt_source
    assert "tl.sum(rhs_values * lhs_values[None, :], axis=1)" in simt_source
    assert "tle.gpu.local_ptr(acc" not in simt_source
    assert "return partial" in simt_source

    packed_simt_model = dict(simt_model)
    packed_simt_model.update(
        Rhs={
            "Expression": "packed_rhs",
            "ShardCoordHierarchy": None,
            "AddressSpace": 3,
            "LocalBuffer": {
                "DescriptorExpression": "packed_rhs",
                "DescriptorShape": [1, 256, 32],
                "LogicalShape": [1, 256],
                "LogicalStrides": [256, 1],
                "BaseCoordinates": [fixed(0), fixed(0)],
                "VectorLaneShape": [4, 8],
                "AvailableBytes": 256 * 4 * 8 * 2,
                "ScalarElementSizeBytes": 2,
                "StorageEncoding": "triton.shared.k-major-packed-n",
            },
        },
        LhsShape=[fixed(1), exact_dynamic_k],
        RhsShape=[fixed(1), exact_dynamic_k],
        RhsStrides=[fixed(256), fixed(1)],
        RhsNVectorLaneCount=8,
        RhsNPackedLaneCount=4,
        OutputNVectorLaneCount=8,
        OutputNPackedLaneCount=4,
        OutputShape=[fixed(1), fixed(1)],
        OutputStrides=[fixed(1), fixed(1)],
        ReductionBlockN=32,
        MicroKernelParameters=dict(
            simt_model["MicroKernelParameters"], state_block_n=32, inner_n=32
        ),
    )
    packed_simt_source = _make_env().get_template(
        "triton/kernels/Gemv.py.jinja"
    ).render(model=packed_simt_model)
    assert "rhs_n_physical" not in packed_simt_source
    assert "shape=(256, 1, 32)" in packed_simt_source
    assert "tl.reshape(" not in packed_simt_source
    assert "mask=" not in packed_simt_source
    assert ").to(tl.float32)" in packed_simt_source
    assert "tl.sum(rhs_values * lhs_values[:, None, None], axis=0)" in packed_simt_source
    assert "tl.sum(rhs_values * lhs_values[None, :], axis=1)" not in packed_simt_source

    tail_dynamic_k = dict(
        exact_dynamic_k,
        RangeMin=1,
        MinValue=1,
    )
    mma_tail_model = dict(
        model,
        LhsShape=[fixed(1), tail_dynamic_k],
        RhsShape=[fixed(256), tail_dynamic_k],
    )
    mma_tail_source = _make_env().get_template(
        "triton/kernels/Gemv.py.jinja"
    ).render(model=mma_tail_model)
    assert mma_tail_source.count("mask=") == 2
    assert mma_tail_source.count("tl.minimum(256, remaining_k)") == 2

    packed_tail_model = dict(
        packed_simt_model,
        LhsShape=[fixed(1), tail_dynamic_k],
        RhsShape=[fixed(1), tail_dynamic_k],
    )
    packed_tail_source = _make_env().get_template(
        "triton/kernels/Gemv.py.jinja"
    ).render(model=packed_tail_model)
    assert "mask=(offs_k[:, None, None] < tl.minimum(256, remaining_k))" in packed_tail_source
    assert "mask=(offs_k < tl.minimum(256, remaining_k))" in packed_tail_source

    simt_finalize_model = dict(
        simt_model,
        FunctionName="register_gemv_finalize",
        ReductionPhase="finalize",
    )
    simt_finalize_source = _make_env().get_template(
        "triton/kernels/Gemv.py.jinja"
    ).render(model=simt_finalize_model)
    assert "tl.inline_asm_elementwise(" in simt_finalize_source
    assert "st.global.b16" in simt_finalize_source
    assert "pyntt_store_offsets" in simt_finalize_source

    local_finalize_model = dict(simt_finalize_model)
    local_finalize_model["Output"] = {
        "Expression": "unused",
        "ShardCoordHierarchy": None,
        "AddressSpace": 3,
        "LocalBuffer": {
            "DescriptorExpression": "output_tile",
            "DescriptorShape": [1, 1, 8, 32],
            "LogicalShape": [1, 256],
            "LogicalStrides": [256, 1],
            "BaseCoordinates": [fixed(0), fixed(0)],
            "VectorLaneShape": [],
            "AvailableBytes": 512,
            "ScalarElementSizeBytes": 2,
            "StorageEncoding": "triton.shared.swizzled",
        },
    }
    local_finalize_source = _make_env().get_template(
        "triton/kernels/Gemv.py.jinja"
    ).render(model=local_finalize_model)
    assert "tle.gpu.local_ptr(output_tile)" in local_finalize_source
    assert "tl.reshape(pyntt_store_value, (1, 1, 8, 32))" in local_finalize_source
    assert "tl.inline_asm_elementwise(" not in local_finalize_source

    for removed_variant in (
        "shared_simt_accumulator",
        "shared_mma_accumulator",
    ):
        removed_model = dict(model, MicroKernelVariant=removed_variant)
        with pytest.raises(ValueError, match="Unsupported Matmul block microkernel"):
            _make_env().get_template("triton/kernels/Gemv.py.jinja").render(
                model=removed_model
            )

    invalid_parameter_model = dict(simt_model)
    invalid_parameter_model["MicroKernelParameters"] = dict(
        simt_model["MicroKernelParameters"], inner_k="32"
    )
    with pytest.raises(ValueError, match="must be an integer"):
        _make_env().get_template("triton/kernels/Gemv.py.jinja").render(
            model=invalid_parameter_model
        )

    tail_model = dict(model)
    tail_model["RhsShape"] = [fixed(32), fixed(256)]
    tail_model["OutputShape"] = [fixed(1), fixed(32)]
    tail_model["OutputStrides"] = [fixed(32), fixed(1)]
    tail_model["ReductionBlockN"] = 32
    tail_model["MicroKernelParameters"] = dict(
        model["MicroKernelParameters"], state_block_n=32, inner_n=32
    )
    tail_source = _make_env().get_template(
        "triton/kernels/Gemv.py.jinja"
    ).render(model=tail_model)
    assert "tl.arange(0, 32)" in tail_source
    assert "for acc_n_start" not in tail_source
    assert "partial_transposed = tl.dot(rhs_values, lhs_columns)" in tail_source


def test_pyntt_simt_gemv_contract_lowers_complete_block_k(tmp_path):
    import importlib.util
    import re

    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to inspect compiled Triton layouts")

    _add_pyntt_to_path()

    from pyntt.codegen.render import _make_env

    block_n = 32
    block_k = 512
    inner_k = block_k

    def fixed(value):
        return {
            "PythonExpression": str(value),
            "TritonExpression": str(value),
            "FixedValue": value,
            "RangeMin": None,
            "RangeMax": None,
            "IsFixed": True,
            "IsFixedOne": value == 1,
            "IsFixedNonOne": value != 1,
            "MinValue": value,
            "MaxValue": value,
        }

    def pointer(expression):
        return {
            "Expression": expression,
            "ShardCoordHierarchy": None,
            "AddressSpace": 1,
            "LocalBuffer": None,
        }

    model = {
        "FunctionName": "gemv_contract",
        "Lhs": pointer("data"),
        "Rhs": pointer("rdata"),
        "Output": pointer("data"),
        "LhsDType": "bfloat16",
        "RhsDType": "bfloat16",
        "OutputDType": "float32",
        "LhsTritonDType": "tl.bfloat16",
        "RhsTritonDType": "tl.bfloat16",
        "OutputTritonDType": "tl.float32",
        "LhsShape": [fixed(1), fixed(block_k)],
        "RhsShape": [fixed(block_n), fixed(block_k)],
        "OutputShape": [fixed(1), fixed(block_n)],
        "LhsStrides": [fixed(block_k), fixed(1)],
        "RhsStrides": [fixed(block_k), fixed(1)],
        "OutputStrides": [fixed(block_n), fixed(1)],
        "TransposeA": False,
        "TransposeB": True,
        "Hierarchy": [1],
        "RhsNVectorLaneCount": 1,
        "OutputNVectorLaneCount": 1,
        "RhsNPackedLaneCount": 1,
        "OutputNPackedLaneCount": 1,
        "Scale": "1",
        "Comment": "SIMT GEMV execution-contract regression",
        "RuntimeShapeArgs": [],
        "LoadCExpression": "False",
        "ReductionPhase": "accumulate",
        "ReductionBlockM": 1,
        "ReductionBlockN": block_n,
        "ReductionBlockK": block_k,
        "MicroKernelFamily": "triton.gemv",
        "MicroKernelVariant": "register_simt_accumulator",
        "MicroKernelParameters": {
            "contract_version": 3,
            "state_block_m": 1,
            "state_block_n": block_n,
            "state_block_k": block_k,
            "inner_m": 1,
            "inner_n": block_n,
            "inner_k": inner_k,
            "pipeline_stages": 1,
        },
        "NumWarps": 8,
        "NoInline": False,
    }
    helper_source = _make_env().get_template(
        "triton/kernels/Gemv.py.jinja"
    ).render(model=model)
    assert "tle.gpu.local_ptr(acc" not in helper_source
    finalize_model = dict(
        model,
        FunctionName="gemv_contract_finalize",
        ReductionPhase="finalize",
    )
    finalize_source = _make_env().get_template(
        "triton/kernels/Gemv.py.jinja"
    ).render(model=finalize_model)
    assert "acc_values = acc" in finalize_source
    assert "tle.gpu.local_ptr(acc" not in finalize_source
    assert "tl.inline_asm_elementwise(" in finalize_source
    assert "st.global.b32" in finalize_source
    module_source = f"""
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

{helper_source}
{finalize_source}

@triton.jit
def gemv_contract_top(data, rdata, output, block_size: tl.constexpr):
    acc = tl.zeros(({block_n},), tl.float32)
    for _ in tl.range(0, 2):
        acc = gemv_contract(
            acc,
            data,
            rdata,
            data,
            data,
            data,
            data,
            0,
            0,
            block_size,
        )
    gemv_contract_finalize(
        acc,
        output,
        rdata,
        output,
        output,
        output,
        output,
        0,
        0,
        block_size,
    )
"""
    module_path = tmp_path / "gemv_contract_module.py"
    module_path.write_text(module_source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_gemv_contract_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    lhs = torch.linspace(
        -0.5, 0.5, block_k, device="cuda", dtype=torch.bfloat16
    )
    rhs = torch.linspace(
        -0.25,
        0.25,
        block_n * block_k,
        device="cuda",
        dtype=torch.bfloat16,
    ).reshape(block_n, block_k)
    output = torch.empty((block_n,), device="cuda", dtype=torch.float32)
    compiled = module.gemv_contract_top.warmup(
        lhs, rhs, output, grid=(1,), block_size=1, num_warps=8
    )
    compiled._init_handles()
    ttgir = compiled.asm["ttgir"]
    ptx = compiled.asm["ptx"]

    reduction = re.search(
        rf'"tt\.reduce".*?: \(tensor<{block_n}x{inner_k}xf32, #(?P<layout>[A-Za-z0-9_]+)>\)',
        ttgir,
        re.DOTALL,
    )
    assert reduction is not None, ttgir
    layout = re.escape(reduction.group("layout"))
    layout_definition = re.search(
        rf"#{layout} = #ttg\.blocked<\{{"
        r"sizePerThread = \[(?P<size>[^]]+)\], "
        r"threadsPerWarp = \[(?P<threads>[^]]+)\], "
        r"warpsPerCTA = \[(?P<warps>[^]]+)\]",
        ttgir,
    )
    assert layout_definition is not None, ttgir
    size_per_thread = [
        int(value) for value in layout_definition.group("size").split(", ")
    ]
    threads_per_warp = [
        int(value) for value in layout_definition.group("threads").split(", ")
    ]
    warps_per_cta = [
        int(value) for value in layout_definition.group("warps").split(", ")
    ]
    assert warps_per_cta[0] * warps_per_cta[1] == 8
    assert threads_per_warp[0] * threads_per_warp[1] == 32
    assert (
        size_per_thread[1] * threads_per_warp[1] * warps_per_cta[1]
        == inner_k
    )
    assert "shfl.sync" in ptx
    assert "ttg.local_load" not in ttgir
    assert "ttg.local_store" not in ttgir
    assert compiled.n_spill_loads == 0
    assert compiled.n_spill_stores == 0

    module.gemv_contract_top[(1,)](
        lhs, rhs, output, block_size=1, num_warps=8
    )
    torch.testing.assert_close(
        output,
        2 * (rhs.float() @ lhs.float()),
        rtol=2e-3,
        atol=2e-2,
    )


def test_pyntt_grouped_simt_packed_gemv_consumes_g_k_lanes_shared_layout(
    tmp_path,
):
    import importlib.util

    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to compile grouped packed GEMV")

    _add_pyntt_to_path()

    from pyntt.codegen.render import _make_env

    group_count = 64
    block_k = 8
    lane_count = 32
    block_n = group_count * lane_count

    def fixed(value):
        return {
            "PythonExpression": str(value),
            "TritonExpression": str(value),
            "FixedValue": value,
            "RangeMin": None,
            "RangeMax": None,
            "IsFixed": True,
            "IsFixedOne": value == 1,
            "IsFixedNonOne": value != 1,
            "MinValue": value,
            "MaxValue": value,
        }

    def pointer(expression):
        return {
            "Expression": expression,
            "ShardCoordHierarchy": None,
            "AddressSpace": 1,
            "LocalBuffer": None,
        }

    model = {
        "FunctionName": "grouped_packed_gemv",
        "Arguments": ["packed_rhs"],
        "Lhs": pointer("data"),
        "Rhs": {
            "Expression": "unused",
            "ShardCoordHierarchy": None,
            "AddressSpace": 3,
            "LocalBuffer": {
                "DescriptorExpression": "packed_rhs",
                "DescriptorShape": [group_count, block_k, lane_count],
                "LogicalShape": [group_count, block_k],
                "LogicalStrides": [block_k, 1],
                "BaseCoordinates": [fixed(0), fixed(0)],
                "VectorLaneShape": [4, 8],
                "AvailableBytes": group_count * block_k * lane_count * 2,
                "ScalarElementSizeBytes": 2,
                "StorageEncoding": "triton.shared.k-major-packed-n",
            },
        },
        "Output": pointer("chip_local_rdata"),
        "LhsDType": "bfloat16",
        "RhsDType": "bfloat16",
        "OutputDType": "float32",
        "LhsTritonDType": "tl.bfloat16",
        "RhsTritonDType": "tl.bfloat16",
        "OutputTritonDType": "tl.float32",
        "LhsShape": [fixed(1), fixed(block_k)],
        "RhsShape": [fixed(group_count), fixed(block_k)],
        "OutputShape": [fixed(1), fixed(group_count)],
        "LhsStrides": [fixed(block_k), fixed(1)],
        "RhsStrides": [fixed(block_k), fixed(1)],
        "OutputStrides": [fixed(group_count), fixed(1)],
        "TransposeA": False,
        "TransposeB": True,
        "Hierarchy": [1],
        "RhsNVectorLaneCount": 8,
        "OutputNVectorLaneCount": 8,
        "RhsNPackedLaneCount": 4,
        "OutputNPackedLaneCount": 4,
        "Scale": "1",
        "Comment": "Grouped packed SIMT GEMV shared-layout regression",
        "RuntimeShapeArgs": [],
        "LoadCExpression": "False",
        "ReductionPhase": "accumulate",
        "ReductionBlockM": 1,
        "ReductionBlockN": block_n,
        "ReductionBlockK": block_k,
        "MicroKernelFamily": "triton.gemv",
        "MicroKernelVariant": "register_simt_accumulator",
        "MicroKernelParameters": {
            "contract_version": 3,
            "state_block_m": 1,
            "state_block_n": block_n,
            "state_block_k": block_k,
            "inner_m": 1,
            "inner_n": block_n,
            "inner_k": block_k,
            "pipeline_stages": 1,
        },
        "NumWarps": 8,
        "NoInline": False,
    }
    helper_source = _make_env().get_template(
        "triton/kernels/Gemv.py.jinja"
    ).render(model=model)
    assert f"shape=({block_k}, {group_count}, {lane_count})" in helper_source
    assert "tl.reshape(" not in helper_source
    assert "axis=0" in helper_source

    module_source = f"""
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

{helper_source}

@triton.jit
def grouped_packed_gemv_top(lhs, rhs, output, block_size: tl.constexpr):
    packed_rhs = tle.gpu.alloc(
        [{group_count}, {block_k}, {lane_count}],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    rhs_g = tl.arange(0, {group_count})[:, None, None]
    rhs_k = tl.arange(0, {block_k})[None, :, None]
    rhs_lane = tl.arange(0, {lane_count})[None, None, :]
    rhs_offsets = (rhs_g * {block_k} + rhs_k) * {lane_count} + rhs_lane
    tle.gpu.copy(
        rhs + rhs_offsets,
        packed_rhs,
        [{group_count}, {block_k}, {lane_count}],
    )
    acc = tl.zeros(({group_count}, {lane_count}), tl.float32)
    acc = grouped_packed_gemv(
        acc,
        packed_rhs,
        lhs,
        rhs,
        output,
        output,
        output,
        output,
        0,
        0,
        block_size,
    )
    tl.store(output + tl.arange(0, {block_n}), tl.reshape(acc, ({block_n},)))
"""
    module_path = tmp_path / "grouped_packed_gemv_module.py"
    module_path.write_text(module_source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(
        "_grouped_packed_gemv_module", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    lhs = torch.linspace(
        -0.5, 0.5, block_k, device="cuda", dtype=torch.bfloat16
    )
    rhs = torch.linspace(
        -0.25,
        0.25,
        group_count * block_k * lane_count,
        device="cuda",
        dtype=torch.bfloat16,
    ).reshape(group_count, block_k, lane_count)
    output = torch.empty((block_n,), device="cuda", dtype=torch.float32)
    compiled = module.grouped_packed_gemv_top.warmup(
        lhs, rhs, output, grid=(1,), block_size=1, num_warps=8
    )
    compiled._init_handles()
    assert "warpsPerCTA = [1, 8, 1]" in compiled.asm["ttgir"]
    assert compiled.metadata.shared == group_count * block_k * lane_count * 2
    assert compiled.n_spill_loads == 0
    assert compiled.n_spill_stores == 0

    module.grouped_packed_gemv_top[(1,)](
        lhs, rhs, output, block_size=1, num_warps=8
    )
    expected = torch.einsum("gkl,k->gl", rhs.float(), lhs.float()).reshape(-1)
    torch.testing.assert_close(output, expected, rtol=2e-3, atol=2e-2)


def test_pyntt_mma_shared_load_uses_only_exact_full_descriptors():
    from copy import deepcopy

    _add_pyntt_to_path()

    from pyntt.codegen.render import _direct_mma_shared_load

    pointer = {
        "Expression": "unused",
        "AddressSpace": 3,
        "LocalBuffer": {
            "DescriptorExpression": "packed_weight",
            "DescriptorShape": [128, 32],
            "LogicalShape": [1, 128],
            "LogicalStrides": [128, 1],
            "BaseCoordinates": [
                {"TritonExpression": "0", "FixedValue": 0},
                {"TritonExpression": "0", "FixedValue": 0},
            ],
            "VectorLaneShape": [4, 8],
            "AvailableBytes": 8192,
            "ScalarElementSizeBytes": 2,
            "StorageEncoding": "triton.nvidia.mma-shared",
        },
    }

    assert _direct_mma_shared_load(pointer, (128, 32)) == (
        "tl.load(tle.gpu.local_ptr(packed_weight))"
    )
    assert _direct_mma_shared_load(pointer, (128, 32), transpose=True) == (
        "tl.trans(tl.load(tle.gpu.local_ptr(packed_weight)))"
    )
    assert _direct_mma_shared_load(pointer, (32, 128)) is None

    offset_pointer = deepcopy(pointer)
    offset_pointer["LocalBuffer"]["BaseCoordinates"][1] = {
        "TritonExpression": "1",
        "FixedValue": 1,
    }
    assert _direct_mma_shared_load(offset_pointer, (128, 32)) is None


def test_pyntt_simt_packed_rhs_uses_exact_k_major_descriptor():
    _add_pyntt_to_path()

    from pyntt.codegen.render import _direct_simt_packed_rhs_pointer

    pointer = {
        "Expression": "unused",
        "AddressSpace": 3,
        "LocalBuffer": {
            "DescriptorExpression": "packed_weight",
            "DescriptorShape": [8, 64, 32],
            "LogicalShape": [8, 64],
            "LogicalStrides": [64, 1],
            "BaseCoordinates": [
                {"TritonExpression": "0", "FixedValue": 0},
                {"TritonExpression": "0", "FixedValue": 0},
            ],
            "VectorLaneShape": [4, 8],
            "AvailableBytes": 32768,
            "ScalarElementSizeBytes": 2,
            "StorageEncoding": "triton.shared.k-major-packed-n",
        },
    }

    assert _direct_simt_packed_rhs_pointer(pointer, 64, 256) == (
        "tle.gpu.local_ptr(packed_weight, "
        "(tl.arange(0, 8)[None, :, None], offs_k[:, None, None], "
        "tl.arange(0, 32)[None, None, :]), shape=(64, 8, 32))"
    )
    assert _direct_simt_packed_rhs_pointer(pointer, 32, 256) is None


def test_pyntt_tensor_region_copy_selects_tle_copy_only_for_full_shared_tiles():
    from copy import deepcopy

    _add_pyntt_to_path()

    from pyntt.codegen.render import _make_env

    template = _make_env().get_template(
        "triton/kernels/TensorRegionCopy.py.jinja"
    )

    def render(model):
        return template.render(model=model).strip()

    def fixed(value):
        return {
            "PythonExpression": str(value),
            "TritonExpression": str(value),
            "FixedValue": value,
        }

    def copy_plan(
        extents,
        *,
        source_origins=None,
        destination_origins=None,
        covers_source=True,
        covers_destination=True,
    ):
        extents = extents if isinstance(extents, list) else [extents]
        logical_rank = len(source_origins or destination_origins or [0])
        source_origins = source_origins or [0] * logical_rank
        destination_origins = destination_origins or [0] * logical_rank
        return {
            "SourceOrigins": [fixed(value) for value in source_origins],
            "DestinationOrigins": [fixed(value) for value in destination_origins],
            "Extents": [
                fixed(extent) if isinstance(extent, int) else extent
                for extent in extents
            ],
            "CoversWholeSource": covers_source,
            "CoversWholeDestination": covers_destination,
        }

    full_tile_model = {
        "FunctionName": "tile_load",
        "Source": {
            "Expression": "source_ptr",
            "AddressSpace": 1,
            "LocalBuffer": None,
        },
        "Destination": {
            "Expression": "destination_ptr",
            "AddressSpace": 3,
            "LocalBuffer": {
                "DescriptorExpression": "destination_storage",
                "DescriptorShape": [32],
                "LogicalShape": [32],
                "LogicalStrides": [1],
                "BaseCoordinates": [fixed(0)],
                "VectorLaneShape": [],
                "AvailableBytes": 128,
                "ScalarElementSizeBytes": 4,
            },
        },
        "DType": "float32",
        "TritonDType": "tl.float32",
        "SourceShape": [fixed(32)],
        "DestinationShape": [fixed(32)],
        "SourceGlobalOffsets": [fixed(0)],
        "DestinationGlobalOffsets": [fixed(0)],
        "SourceStrides": [fixed(1)],
        "DestinationStrides": [fixed(1)],
        "VectorLaneShape": [],
        "OperationKind": "TileLoad",
        "CopyPlan": copy_plan(32),
        "Comment": "TileLoad: source -> destination",
        "RuntimeShapeArgs": [],
        "Arguments": ["destination_storage"],
        "NoInline": False,
    }

    full_tile_source = render(full_tile_model)
    assert "tle.gpu.alloc" not in full_tile_source
    assert "tle.gpu.copy(source + copy_global_offset, destination_storage, [32])" in full_tile_source
    assert "if full_tile:" not in full_tile_source
    assert "value = tl.load(source + source_offset, mask=mask)" not in full_tile_source

    vector_model = deepcopy(full_tile_model)
    vector_model["SourceShape"] = [fixed(4)]
    vector_model["DestinationShape"] = [fixed(4)]
    vector_model["VectorLaneShape"] = [8]
    vector_model["CopyPlan"] = copy_plan([4, 8])
    vector_model["Destination"]["LocalBuffer"].update(
        DescriptorShape=[4, 8],
        LogicalShape=[4],
        LogicalStrides=[1],
        VectorLaneShape=[8],
    )
    vector_source = render(vector_model)
    assert "copy_tensor_linear" not in vector_source
    assert "copy_vector_lane" not in vector_source
    assert "copy_global_offset = copy_linear" in vector_source

    packed_vector_model = deepcopy(full_tile_model)
    packed_vector_model["SourceShape"] = [fixed(1), fixed(256)]
    packed_vector_model["DestinationShape"] = [fixed(1), fixed(256)]
    packed_vector_model["SourceGlobalOffsets"] = [fixed(0), fixed(0)]
    packed_vector_model["DestinationGlobalOffsets"] = [fixed(0), fixed(0)]
    packed_vector_model["SourceStrides"] = [fixed(256), fixed(1)]
    packed_vector_model["DestinationStrides"] = [fixed(256), fixed(1)]
    packed_vector_model["VectorLaneShape"] = [4, 8]
    packed_vector_model["CopyPlan"] = copy_plan(
        [1, 256, 4, 8], source_origins=[0, 0]
    )
    packed_vector_model["Destination"]["LocalBuffer"]["DescriptorShape"] = [
        1,
        256,
        4,
        8,
    ]
    packed_vector_model["Destination"]["LocalBuffer"]["AvailableBytes"] = 32768
    packed_vector_model["Destination"]["LocalBuffer"].update(
        LogicalShape=[1, 256],
        LogicalStrides=[256, 1],
        BaseCoordinates=[fixed(0), fixed(0)],
        VectorLaneShape=[4, 8],
    )
    packed_vector_source = render(packed_vector_model)
    assert "copy_global_offset = copy_linear" in packed_vector_source
    assert "copy_tensor_linear" not in packed_vector_source
    assert "copy_vector_lane" not in packed_vector_source

    singleton_packed_model = deepcopy(packed_vector_model)
    exact_dynamic_extent = {
        "PythonExpression": "tl.minimum(512, remaining_k)",
        "TritonExpression": "tl.minimum(512, remaining_k)",
        "FixedValue": None,
        "RangeMin": 512,
        "RangeMax": 512,
    }
    singleton_packed_model["SourceShape"] = [fixed(1), exact_dynamic_extent]
    singleton_packed_model["DestinationShape"] = [fixed(1), exact_dynamic_extent]
    singleton_packed_model["SourceStrides"] = [fixed(2048), fixed(1)]
    singleton_packed_model["DestinationStrides"] = [fixed(0), fixed(1)]
    singleton_packed_model["CopyPlan"] = copy_plan(
        [1, exact_dynamic_extent, 4, 8], source_origins=[0, 0]
    )
    singleton_packed_model["Destination"]["LocalBuffer"].update(
        DescriptorShape=[1, 512, 32],
        LogicalShape=[1, 512],
        LogicalStrides=[0, 1],
        VectorLaneShape=[4, 8],
        ScalarElementSizeBytes=2,
        StorageEncoding="triton.shared.k-major-packed-n",
    )
    singleton_packed_source = render(singleton_packed_model)
    assert (
        "tle.gpu.copy(source + copy_global_offset, destination_storage, [1, 512, 32])"
        in singleton_packed_source
    )

    matrix_model = deepcopy(full_tile_model)
    matrix_model["SourceShape"] = [fixed(4), fixed(8)]
    matrix_model["DestinationShape"] = [fixed(4), fixed(8)]
    matrix_model["SourceGlobalOffsets"] = [fixed(0), fixed(0)]
    matrix_model["DestinationGlobalOffsets"] = [fixed(0), fixed(0)]
    matrix_model["SourceStrides"] = [fixed(8), fixed(1)]
    matrix_model["DestinationStrides"] = [fixed(8), fixed(1)]
    matrix_model["CopyPlan"] = copy_plan(
        [4, 8], source_origins=[0, 0]
    )
    matrix_model["Destination"]["LocalBuffer"]["DescriptorShape"] = [4, 8]
    matrix_model["Destination"]["LocalBuffer"].update(
        LogicalShape=[4, 8],
        LogicalStrides=[8, 1],
        BaseCoordinates=[fixed(0), fixed(0)],
    )
    matrix_source = render(matrix_model)
    assert "copy_desc_idx0 = tl.arange(0, 4)[:, None]" in matrix_source
    assert "copy_desc_idx1 = tl.arange(0, 8)[None, :]" in matrix_source
    assert (
        "tle.gpu.copy(source + copy_global_offset, destination_storage, [4, 8])"
        in matrix_source
    )

    offset_model = deepcopy(full_tile_model)
    offset_model["CopyPlan"] = copy_plan(32, source_origins=[64])
    offset_source = render(offset_model)
    assert "copy_global_offset = (64) + copy_linear" in offset_source

    tail_model = deepcopy(full_tile_model)
    dynamic_extent = {
        "PythonExpression": "extent",
        "TritonExpression": "extent",
        "FixedValue": None,
        "RangeMin": 1,
        "RangeMax": 32,
    }
    tail_model["SourceShape"] = [dynamic_extent]
    tail_model["DestinationShape"] = [dynamic_extent]
    tail_model["CopyPlan"] = copy_plan(dynamic_extent)
    tail_model["RuntimeShapeArgs"] = ["extent"]
    tail_source = render(tail_model)
    assert "full_tile" not in tail_source
    assert "tle.gpu.copy" not in tail_source
    assert (
        "value = tl.load(source + tl.broadcast_to(copy_idx0, (32,)), mask=mask)"
        in tail_source
    )
    assert (
        "tle.gpu.local_ptr(destination_storage, "
        "(copy_idx0,), shape=(32,))"
        in tail_source
    )
    assert "tle.gpu.local_ptr(destination_storage, (tl.broadcast_to" not in tail_source
    assert "= tle.gpu.local_ptr" not in tail_source
    assert " // " not in tail_source
    assert " % " not in tail_source

    loose_extent_model = deepcopy(tail_model)
    loose_extent = deepcopy(dynamic_extent)
    loose_extent["RangeMax"] = 256
    loose_extent_model["CopyPlan"] = copy_plan(loose_extent)
    loose_extent_source = render(loose_extent_model)
    assert "copy_idx0 = tl.arange(0, 32)" in loose_extent_source
    assert "copy_idx0 = tl.arange(0, 256)" not in loose_extent_source

    mma_tail_model = deepcopy(tail_model)
    mma_tail_model["Destination"]["LocalBuffer"]["StorageEncoding"] = (
        "triton.nvidia.mma-shared"
    )
    mma_tail_source = render(mma_tail_model)
    assert "tl.store(tle.gpu.local_ptr(destination_storage), 0.0)" in mma_tail_source
    assert (
        "value = tl.load(source + tl.broadcast_to(copy_idx0, (32,)), mask=mask)"
        in mma_tail_source
    )

    noncoincident_model = deepcopy(full_tile_model)
    noncoincident_model["CopyPlan"] = copy_plan(
        32, covers_destination=False
    )
    noncoincident_source = render(noncoincident_model)
    assert "tle.gpu.copy" not in noncoincident_source
    assert "tl.broadcast_to(copy_idx0, (32,))" in noncoincident_source
    assert "tl.store(tle.gpu.local_ptr(destination_storage" in noncoincident_source
    assert "= tle.gpu.local_ptr" not in noncoincident_source

    noncompact_model = deepcopy(full_tile_model)
    noncompact_model["DestinationStrides"] = [fixed(2)]
    noncompact_model["Destination"]["LocalBuffer"]["LogicalStrides"] = [2]
    with pytest.raises(ValueError, match="descriptor strides"):
        render(noncompact_model)

    noncompact_global_model = deepcopy(full_tile_model)
    noncompact_global_model["SourceStrides"] = [fixed(2)]
    noncompact_global_source = render(noncompact_global_model)
    assert "tle.gpu.copy" not in noncompact_global_source
    assert (
        "tl.load(source + tl.broadcast_to((copy_idx0) * (2), (32,)), mask=mask)"
        in noncompact_global_source
    )

    strided_model = deepcopy(full_tile_model)
    strided_model["Destination"] = {
        "Expression": "destination_ptr",
        "AddressSpace": 1,
        "LocalBuffer": None,
    }
    strided_model["Arguments"] = ["source_ptr", "destination_ptr"]
    strided_model["SourceShape"] = [fixed(4), fixed(8)]
    strided_model["DestinationShape"] = [fixed(4), fixed(8)]
    strided_model["SourceGlobalOffsets"] = [fixed(0), fixed(0)]
    strided_model["DestinationGlobalOffsets"] = [fixed(0), fixed(0)]
    strided_model["SourceStrides"] = [fixed(16), fixed(1)]
    strided_model["DestinationStrides"] = [fixed(8), fixed(1)]
    strided_model["CopyPlan"] = {
        "SourceOrigins": [fixed(0), fixed(3)],
        "DestinationOrigins": [fixed(0), fixed(5)],
        "Extents": [fixed(4), fixed(8)],
        "CoversWholeSource": True,
        "CoversWholeDestination": True,
    }
    strided_source = render(strided_model)
    assert "for copy_idx0 in tl.range(0, 4):" in strided_source
    assert "tl.full((block_size,), copy_idx0, tl.int64)" not in strided_source
    assert (
        "tl.broadcast_to((copy_idx0) * (16) + (3) + (copy_idx1), "
        "(block_size,))"
        in strided_source
    )
    assert (
        "tl.broadcast_to((copy_idx0) * (8) + (5) + (copy_idx1), "
        "(block_size,))"
        in strided_source
    )
    assert "copy_remaining" not in strided_source
    assert "tensor_linear" not in strided_source
    assert " // " not in strided_source
    assert " % " not in strided_source

    zero_stride_model = deepcopy(strided_model)
    zero_stride_model["SourceStrides"][1] = fixed(0)
    zero_stride_source = render(zero_stride_model)
    assert "* 0" not in zero_stride_source
    assert "(3) + (copy_idx1)" not in zero_stride_source


def test_pyntt_kernel_templates_own_their_triton_source():
    _add_pyntt_to_path()

    from pyntt.codegen.render import _make_env

    template_dir = (
        Path(__file__).resolve().parents[2]
        / "pyntt/pyntt/codegen/templates/triton/kernels"
    )
    public_templates = {
        path.name.removesuffix(".py.jinja")
        for path in template_dir.glob("*.py.jinja")
        if not path.name.startswith("_")
    }
    env = _make_env()
    for name in public_templates:
        source = (template_dir / f"{name}.py.jinja").read_text(encoding="utf-8")
        assert "{{ emit(" not in source
        env.get_template(f"triton/kernels/{name}.py.jinja")


def test_pyntt_renderer_passes_one_materialized_shard_index_to_device_calls():
    _add_pyntt_to_path()

    from pyntt.codegen.render import render_manifest

    source = render_manifest(
        {
            "pyntt_codegen_manifest_version": 3,
            "functions": [
                {
                    "render_kernels": [
                        {
                            "metadata": {
                                "name": "top",
                                "inputs": [],
                                "outputs": [],
                                "attrs": {
                                    "shared_memory_capacity_bytes": 64,
                                    "shared_memory_allocation_size_policy": "power_of_two",
                                    "shared_memory_allocation_granularity_bytes": 8,
                                },
                                "launch": {"num_warps": 8},
                            },
                            "body_source": (
                                "__pyntt_device_call__child(shard_index)\n"
                                "__pyntt_device_call__child(shard_index)"
                            ),
                            "device_functions": [
                                {
                                    "name": "child",
                                    "noinline": True,
                                    "preserve_helper_call_boundaries": False,
                                    "helpers": [],
                                    "body_source": "child_shard_index",
                                    "parameter_overrides": {},
                                    "extra_parameters": ["child_shard_index"],
                                    "extra_parameter_arguments": {},
                                }
                            ],
                        }
                    ]
                }
            ]
        }
    )

    assert source.count("tl.program_id(0).to(tl.int64)") == 1
    assert source.count("child(shard_index)") == 2
    assert "def child(child_shard_index):" in source
    assert "pyntt_call_frame" not in source


def test_pyntt_renderer_passes_nested_device_arguments_directly():
    _add_pyntt_to_path()

    from pyntt.codegen.render import render_manifest

    def device_function(name, body, parameter_count):
        pointer_names = [f"{name}_ptr{index}" for index in range(parameter_count)]
        return {
            "name": name,
            "noinline": True,
            "preserve_helper_call_boundaries": False,
            "helpers": [],
            "body_source": body,
            "parameter_overrides": {},
            "extra_parameters": pointer_names,
            "extra_parameter_arguments": {},
        }

    source = render_manifest(
        {
            "pyntt_codegen_manifest_version": 3,
            "functions": [
                {
                    "render_kernels": [
                        {
                            "metadata": {
                                "name": "top",
                                "inputs": [],
                                "outputs": [],
                                "attrs": {
                                    "shared_memory_bytes": 0,
                                    "shared_memory_capacity_bytes": 64,
                                    "shared_memory_allocation_size_policy": "power_of_two",
                                    "shared_memory_allocation_granularity_bytes": 8,
                                },
                                "launch": {"num_warps": 8},
                            },
                            "body_source": "__pyntt_device_call__parent(data)",
                            "device_functions": [
                                device_function(
                                    "parent",
                                    "__pyntt_device_call__left(parent_ptr0, parent_ptr0, parent_ptr0, parent_ptr0)\n"
                                    "__pyntt_device_call__right(parent_ptr0, parent_ptr0, parent_ptr0, parent_ptr0)",
                                    1,
                                ),
                                device_function("left", "pass", 4),
                                device_function("right", "pass", 4),
                            ],
                        }
                    ]
                }
            ]
        }
    )

    assert "def parent(parent_ptr0):" in source
    assert "parent(data)" in source
    assert "left(parent_ptr0, parent_ptr0, parent_ptr0, parent_ptr0)" in source
    assert "right(parent_ptr0, parent_ptr0, parent_ptr0, parent_ptr0)" in source
    assert "pyntt_call_frame" not in source


def test_pyntt_renderer_propagates_only_live_canonical_device_parameters():
    _add_pyntt_to_path()

    from pyntt.codegen.render import render_manifest

    def device_function(name, body):
        return {
            "name": name,
            "noinline": True,
            "preserve_helper_call_boundaries": False,
            "helpers": [],
            "body_source": body,
            "parameter_overrides": {},
            "extra_parameters": [],
            "extra_parameter_arguments": {},
        }

    source = render_manifest(
        {
            "pyntt_codegen_manifest_version": 3,
            "functions": [
                {
                    "render_kernels": [
                        {
                            "metadata": {
                                "name": "top",
                                "inputs": ["unused", "live"],
                                "outputs": [],
                                "attrs": {
                                    "runtime_shape_args": ["extent"],
                                    "shared_memory_capacity_bytes": 128,
                                    "shared_memory_allocation_size_policy": "power_of_two",
                                    "shared_memory_allocation_granularity_bytes": 8,
                                },
                                "launch": {"num_warps": 8},
                            },
                            "body_source": "__pyntt_device_call__parent()",
                            "device_functions": [
                                device_function(
                                    "parent",
                                    "tl.load(rdata)\n"
                                    "__pyntt_device_call__child()",
                                ),
                                device_function(
                                    "child", "tl.load(input1) + extent + tl.load(rdata)"
                                ),
                            ],
                        }
                    ]
                }
            ]
        }
    )

    parent_parameters = source.split("def parent(", 1)[1].split("):", 1)[0]
    child_parameters = source.split("def child(", 1)[1].split("):", 1)[0]
    assert parent_parameters == "input1, rdata, extent"
    assert child_parameters == "input1, rdata, extent"
    assert "parent(input1, rdata, extent)" in source
    assert "child(input1, rdata, extent)" in source
    assert "tl.load(" in source
    assert "tl.store(" not in source
    assert "volatile=True" not in source
    assert "pyntt_call_frame" not in source


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

    def run(self, *args, **kwargs):
        candidate = int(args[-1])
        self.attempts.append(candidate)
        result = self.compiled_by_candidate[candidate]
        if isinstance(result, BaseException):
            raise result
        return result


def test_pyntt_runtime_accepts_kernel_within_fixed_resource_budget():
    _add_pyntt_to_path()

    from pyntt.runtime.triton import validate_triton_kernel_resources

    argument = object()
    kernel = _FakeJitKernel(
            _FakeCompiledKernel(
                shared=4096, registers=64, stack=32, local=8
            )
        )
    validate_triton_kernel_resources(
        kernel,
        argument,
        grid=(36,),
        expected_num_warps=8,
        registers_per_thread_limit=255,
        shared_memory_capacity_bytes=101_376,
        forbid_spills=True,
    )
    assert kernel.calls[0][0][0] is argument
    assert kernel.calls[0][1]["warmup"] is True


@pytest.mark.parametrize(
    ("compiled", "message"),
    [
        (_FakeCompiledKernel(num_warps=4), "requires 8"),
        (_FakeCompiledKernel(registers=256), "registers per thread"),
        (_FakeCompiledKernel(shared=1024), "shared-memory bytes"),
        (_FakeCompiledKernel(spill_stores=4), "forbids register spilling"),
    ],
)
def test_pyntt_runtime_rejects_kernel_outside_fixed_resource_budget(
    compiled, message
):
    _add_pyntt_to_path()

    from pyntt.runtime.triton import validate_triton_kernel_resources

    shared_capacity = 512 if compiled.metadata.shared else 101_376
    with pytest.raises(RuntimeError, match=message):
        validate_triton_kernel_resources(
            _FakeJitKernel(compiled),
            grid=(36,),
            expected_num_warps=8,
            registers_per_thread_limit=255,
            shared_memory_capacity_bytes=shared_capacity,
            forbid_spills=True,
        )


def test_pyntt_runtime_selects_first_resource_feasible_tuning_candidate():
    _add_pyntt_to_path()

    from pyntt.runtime.triton import (
        select_and_validate_triton_tuning_parameter,
    )
    from triton.runtime.errors import OutOfResources

    kernel = _FakeTunableJitKernel(
        {
            128: _FakeCompiledKernel(registers=64),
            256: _FakeCompiledKernel(spill_stores=4),
            512: OutOfResources(128 * 1024, 96 * 1024, "shared memory"),
        }
    )
    selected = select_and_validate_triton_tuning_parameter(
        "test_kernel",
        "block_size",
        (128, 256, 512),
        source="search_space",
        kernel=kernel,
        kernel_args=(),
        grid_for_candidate=lambda _: (1,),
        expected_num_warps=8,
        registers_per_thread_limit=255,
        shared_memory_capacity_bytes=101_376,
        forbid_spills=True,
        num_warps=8,
    )

    assert selected == 128
    assert kernel.attempts == [512, 256, 128]

    selected_again = select_and_validate_triton_tuning_parameter(
        "test_kernel",
        "block_size",
        (128, 256, 512),
        source="search_space",
        kernel=kernel,
        kernel_args=(),
        grid_for_candidate=lambda _: (1,),
        expected_num_warps=8,
        registers_per_thread_limit=255,
        shared_memory_capacity_bytes=101_376,
        forbid_spills=True,
        num_warps=8,
    )

    assert selected_again == 128
    assert kernel.attempts == [512, 256, 128]
