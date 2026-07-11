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


def test_pyntt_renderer_threads_one_shared_pool_through_device_functions():
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
                                "attrs": {"shared_memory_bytes": 128},
                            },
                            "body_source": "__pyntt_device_call__child(data)",
                            "device_functions": [
                                {
                                    "name": "child",
                                    "noinline": True,
                                    "preserve_helper_call_boundaries": False,
                                    "pointer_call_frame": [
                                        {
                                            "name": "child_data",
                                            "triton_dtype": "tl.uint8",
                                        }
                                    ],
                                    "helpers": [],
                                    "body_source": "tl.load(child_data)",
                                    "parameter_overrides": {},
                                    "extra_parameters": ["child_data"],
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
    assert source.count("pyntt_shared_storage = tle.gpu.alloc([256]") == 1
    assert "@triton.jit(noinline=True)\ndef child(" in source
    assert "child_data" not in source.split("def child(", 1)[1].split("):", 1)[0]
    assert "tl.pointer_type(tl.uint64, 3)" in source
    assert "tl.store(" in source
    assert "tl.load(" in source
    assert "pyntt_shared_base" in source
    assert "child(" in source


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

    def warmup(self, *args, **kwargs):
        return self.compiled


class _FakeTunableJitKernel:
    def __init__(self, compiled_by_candidate):
        self.compiled_by_candidate = compiled_by_candidate
        self.attempts = []

    def warmup(self, *args, **kwargs):
        candidate = int(args[-1])
        self.attempts.append(candidate)
        result = self.compiled_by_candidate[candidate]
        if isinstance(result, BaseException):
            raise result
        return result


def test_pyntt_runtime_accepts_kernel_within_fixed_resource_budget():
    _add_pyntt_to_path()

    from pyntt.runtime.triton import validate_triton_kernel_resources

    validate_triton_kernel_resources(
        _FakeJitKernel(
            _FakeCompiledKernel(
                shared=4096, registers=64, stack=32, local=8
            )
        ),
        grid=(36,),
        expected_num_warps=8,
        worker_width=32,
        register_capacity_bytes=256 * 1024,
        shared_memory_capacity_bytes=101_376,
        forbid_spills=True,
    )


@pytest.mark.parametrize(
    ("compiled", "message"),
    [
        (_FakeCompiledKernel(num_warps=4), "requires 8"),
        (_FakeCompiledKernel(shared=1024), "shared-memory bytes"),
        (_FakeCompiledKernel(registers=257), "register bytes"),
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
            worker_width=32,
            register_capacity_bytes=256 * 1024,
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
        worker_width=32,
        register_capacity_bytes=256 * 1024,
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
        worker_width=32,
        register_capacity_bytes=256 * 1024,
        shared_memory_capacity_bytes=101_376,
        forbid_spills=True,
        num_warps=8,
    )

    assert selected_again == 128
    assert kernel.attempts == [512, 256, 128]
