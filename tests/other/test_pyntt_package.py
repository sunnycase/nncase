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


def test_pyntt_renderer_uses_explicit_device_arguments_and_tiling_shared_arena():
    _add_pyntt_to_path()

    from pyntt.codegen.render import render_manifest

    source = render_manifest(
        {
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
    assert source.count("pyntt_shared_storage = tle.gpu.alloc([65536]") == 1
    assert "@triton.jit(noinline=True)\ndef child(" in source
    assert "def child(pyntt_shared_base, child_data, child_extent):" in source
    assert "child(pyntt_shared_base, data, numel)" in source
    assert "tl.load(" in source
    assert "pyntt_shared_base" in source
    assert "pyntt_call_frame" not in source


def test_pyntt_renderer_passes_one_materialized_shard_index_to_device_calls():
    _add_pyntt_to_path()

    from pyntt.codegen.render import render_manifest

    source = render_manifest(
        {
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
        shared_memory_capacity_bytes=101_376,
        forbid_spills=True,
    )
    assert kernel.calls[0][0][0] is argument
    assert kernel.calls[0][1]["warmup"] is True


@pytest.mark.parametrize(
    ("compiled", "message"),
    [
        (_FakeCompiledKernel(num_warps=4), "requires 8"),
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
        shared_memory_capacity_bytes=101_376,
        forbid_spills=True,
        num_warps=8,
    )

    assert selected_again == 128
    assert kernel.attempts == [512, 256, 128]
