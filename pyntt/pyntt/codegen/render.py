"""Render generated PyNTT Triton kernels from a nncase codegen manifest."""

from __future__ import annotations

import ast
import importlib
import json
import re
import sys
from math import gcd, prod
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, StrictUndefined


WORKSPACE_PARAMETERS = (
    "data",
    "rdata",
    "chip_local_rdata",
    "chip_local_data",
    "block_local_rdata",
    "block_local_data",
)

WORKSPACE_STRIDE_PARAMETERS = (
    "data_pool_stride_bytes: tl.constexpr",
    "block_local_data_pool_stride_bytes: tl.constexpr",
)

PYNTT_CODEGEN_MANIFEST_VERSION = 9
THROUGHPUT_BLOCK_SIZE_CANDIDATES = (128, 256, 512, 1024)
WARP_SPECIALIZATION_PRODUCER_WARPS = 1
WARP_SPECIALIZATION_WORKER_ALLOCATION_WARPS = 4
WARP_SPECIALIZATION_MINIMUM_PARTITION_REGISTERS = 24
PAGED_ATTENTION_BLOCK_N_CANDIDATES = (32, 64)
PAGED_ATTENTION_NUM_STAGES_CANDIDATES = (2, 3)
PAGED_ATTENTION_BLOCK_N = 64
PAGED_ATTENTION_NUM_STAGES = 2
TENSOR_MAP_ENTRY_BYTES = 128
TMA_MAXIMUM_SWIZZLE_BYTES = 128
TMA_DTYPE_ITEM_SIZES = {
    "uint8": 1,
    "int8": 1,
    "float8e4m3fn": 1,
    "float8e5m2": 1,
    "uint16": 2,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "uint32": 4,
    "int32": 4,
    "float32": 4,
    "uint64": 8,
    "int64": 8,
    "float64": 8,
}

PACKED_GEMV_MAXIMUM_INLINE_REDUCTION_GROUPS = 8

DEVICE_CALL_RE = re.compile(
    r"(?m)^(?P<indent>[ \t]*)__pyntt_device_call__(?P<name>[A-Za-z_]\w*)\((?P<args>.*)\)$"
)
DEVICE_CALL_NAME_RE = re.compile(r"__pyntt_device_call__(?P<name>[A-Za-z_]\w*)\(")


def _require_exact_object(
    value: Any, path: str, expected_keys: set[str]
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a JSON object.")
    actual_keys = set(value)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing fields {missing}")
        if unexpected:
            details.append(f"unexpected fields {unexpected}")
        raise ValueError(f"{path} has {' and '.join(details)}.")
    return value


def _require_int(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path} must be at least {minimum}, got {value}.")
    return value


def _require_string(value: Any, path: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        suffix = " a non-empty string" if nonempty else " a string"
        raise ValueError(f"{path} must be{suffix}.")
    return value


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a JSON array.")
    return value


def _require_string_list(value: Any, path: str) -> list[str]:
    values = _require_list(value, path)
    for index, item in enumerate(values):
        _require_string(item, f"{path}[{index}]")
    return values


def _parameter_name(declaration: str, path: str) -> str:
    name = declaration.split(":", 1)[0].strip()
    if not name.isidentifier():
        raise ValueError(f"{path} has invalid parameter declaration {declaration!r}.")
    return name


def _require_string_map(value: Any, path: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a JSON object.")
    for key, item in value.items():
        _require_string(key, f"{path} key", nonempty=True)
        _require_string(item, f"{path}[{key!r}]")
    return value


def _validate_helper(helper: Any, path: str) -> None:
    helper = _require_exact_object(
        helper,
        path,
        {"template", "model", "arguments", "workspace_arguments"},
    )
    _require_string(helper["template"], f"{path}.template", nonempty=True)
    if not isinstance(helper["model"], dict):
        raise ValueError(f"{path}.model must be a JSON object.")
    arguments = _require_string_list(helper["arguments"], f"{path}.arguments")
    workspace_arguments = _require_string_list(
        helper["workspace_arguments"], f"{path}.workspace_arguments"
    )
    parameter_names = [
        _parameter_name(declaration, f"{path}.{field}[{index}]")
        for field, declarations in (
            ("arguments", arguments),
            ("workspace_arguments", workspace_arguments),
        )
        for index, declaration in enumerate(declarations)
    ]
    duplicates = sorted(
        name for name in set(parameter_names) if parameter_names.count(name) > 1
    )
    if duplicates:
        raise ValueError(f"{path} has duplicate parameters {duplicates}.")


def _validate_device_function(device_function: Any, path: str) -> None:
    device_function = _require_exact_object(
        device_function,
        path,
        {
            "name",
            "noinline",
            "helpers",
            "body_source",
            "parameter_overrides",
            "extra_parameters",
            "extra_parameter_arguments",
        },
    )
    _require_string(device_function["name"], f"{path}.name", nonempty=True)
    if not isinstance(device_function["noinline"], bool):
        raise ValueError(f"{path}.noinline must be a boolean.")
    helpers = _require_list(device_function["helpers"], f"{path}.helpers")
    for index, helper in enumerate(helpers):
        _validate_helper(helper, f"{path}.helpers[{index}]")
    body_source = _require_string(device_function["body_source"], f"{path}.body_source")
    if body_source.strip():
        _validate_python_statements(body_source, f"{path}.body_source")
    _require_string_map(
        device_function["parameter_overrides"], f"{path}.parameter_overrides"
    )
    _require_string_list(
        device_function["extra_parameters"], f"{path}.extra_parameters"
    )
    _require_string_map(
        device_function["extra_parameter_arguments"],
        f"{path}.extra_parameter_arguments",
    )


def _require_positive_int_list(value: Any, path: str) -> list[int]:
    values = _require_list(value, path)
    for index, item in enumerate(values):
        _require_int(item, f"{path}[{index}]", minimum=1)
    return values


def _normalize_singleton_strides(
    shape: tuple[int, ...], strides: tuple[int, ...]
) -> tuple[int, ...]:
    """Give zero-stride singleton dimensions an equivalent positive stride."""

    if len(shape) != len(strides):
        raise ValueError(
            "PyNTT tensor descriptor shape/stride ranks differ: "
            f"{len(shape)} and {len(strides)}."
        )
    normalized = list(strides)
    trailing_span = 1
    for axis in range(len(shape) - 1, -1, -1):
        extent = shape[axis]
        stride = normalized[axis]
        if stride == 0:
            if extent != 1:
                raise ValueError(
                    "PyNTT tensor descriptor has a zero stride on non-singleton "
                    f"axis {axis}: shape={shape}, strides={strides}."
                )
            stride = trailing_span
            normalized[axis] = stride
        trailing_span = max(trailing_span, stride * extent)
    return tuple(normalized)


def _validate_launch(launch: Any, path: str) -> None:
    launch = _require_exact_object(
        launch,
        path,
        {"meta", "sharding", "host_tensor_descriptors"},
    )
    if not isinstance(launch["meta"], dict):
        raise ValueError(f"{path}.meta must be a JSON object.")

    descriptor_names: set[str] = set()
    descriptors = _require_list(
        launch["host_tensor_descriptors"],
        f"{path}.host_tensor_descriptors",
    )
    for index, descriptor in enumerate(descriptors):
        descriptor_path = f"{path}.host_tensor_descriptors[{index}]"
        descriptor = _require_exact_object(
            descriptor,
            descriptor_path,
            {
                "name",
                "source",
                "offset_bytes",
                "owner_stride_bytes",
                "scalar_dtype",
                "logical_shape",
                "logical_strides",
                "vector_lane_shape",
                "contiguous_rebase_extent_elements",
                "source_shape_axes",
            },
        )
        name = _require_string(
            descriptor["name"], f"{descriptor_path}.name", nonempty=True
        )
        if not name.isidentifier():
            raise ValueError(
                f"{descriptor_path}.name must be a Python identifier, got {name!r}."
            )
        if name in descriptor_names:
            raise ValueError(f"{path} has duplicate host tensor descriptor {name!r}.")
        descriptor_names.add(name)
        source = _require_string(
            descriptor["source"], f"{descriptor_path}.source", nonempty=True
        )
        if not source.isidentifier():
            raise ValueError(
                f"{descriptor_path}.source must be a kernel ABI identifier, "
                f"got {source!r}."
            )
        _require_int(
            descriptor["offset_bytes"],
            f"{descriptor_path}.offset_bytes",
            minimum=0,
        )
        _require_int(
            descriptor["owner_stride_bytes"],
            f"{descriptor_path}.owner_stride_bytes",
            minimum=0,
        )
        _require_string(
            descriptor["scalar_dtype"],
            f"{descriptor_path}.scalar_dtype",
            nonempty=True,
        )
        logical_shape = _require_positive_int_list(
            descriptor["logical_shape"], f"{descriptor_path}.logical_shape"
        )
        logical_strides = _require_list(
            descriptor["logical_strides"], f"{descriptor_path}.logical_strides"
        )
        for axis, stride in enumerate(logical_strides):
            _require_int(
                stride,
                f"{descriptor_path}.logical_strides[{axis}]",
                minimum=0,
            )
        if len(logical_shape) != len(logical_strides):
            raise ValueError(
                f"{descriptor_path} logical shape/stride ranks differ: "
                f"{len(logical_shape)} and {len(logical_strides)}."
            )
        _normalize_singleton_strides(
            tuple(logical_shape), tuple(logical_strides)
        )
        source_shape_axes = _require_list(
            descriptor["source_shape_axes"],
            f"{descriptor_path}.source_shape_axes",
        )
        if len(source_shape_axes) != len(logical_shape):
            raise ValueError(
                f"{descriptor_path} logical/source-shape ranks differ: "
                f"{len(logical_shape)} and {len(source_shape_axes)}."
            )
        used_source_axes: set[int] = set()
        for descriptor_axis, source_axes in enumerate(source_shape_axes):
            source_axes = _require_list(
                source_axes,
                f"{descriptor_path}.source_shape_axes[{descriptor_axis}]",
            )
            for source_axis_index, source_axis in enumerate(source_axes):
                source_axis = _require_int(
                    source_axis,
                    f"{descriptor_path}.source_shape_axes[{descriptor_axis}]"
                    f"[{source_axis_index}]",
                    minimum=0,
                )
                if source_axis in used_source_axes:
                    raise ValueError(
                        f"{descriptor_path} source tensor axis {source_axis} "
                        "is used by more than one descriptor dimension."
                    )
                used_source_axes.add(source_axis)
        _require_positive_int_list(
            descriptor["vector_lane_shape"],
            f"{descriptor_path}.vector_lane_shape",
        )
        _require_int(
            descriptor["contiguous_rebase_extent_elements"],
            f"{descriptor_path}.contiguous_rebase_extent_elements",
            minimum=0,
        )

    sharding = _require_exact_object(
        launch["sharding"],
        f"{path}.sharding",
        {
            "strategy",
            "placement_axis",
            "tensor_axis",
            "extent",
            "hierarchy",
            "hierarchy_levels",
            "global_shape",
        },
    )
    for field in ("strategy", "placement_axis", "extent", "hierarchy_levels"):
        _require_string(sharding[field], f"{path}.sharding.{field}")
    _require_int(sharding["tensor_axis"], f"{path}.sharding.tensor_axis")
    hierarchy = _require_list(sharding["hierarchy"], f"{path}.sharding.hierarchy")
    for index, extent in enumerate(hierarchy):
        _require_int(extent, f"{path}.sharding.hierarchy[{index}]", minimum=1)
    _require_string_list(sharding["global_shape"], f"{path}.sharding.global_shape")


def _validate_python_expression(value: Any, path: str) -> str:
    source = _require_string(value, path, nonempty=True)
    try:
        ast.parse(source, mode="eval")
    except SyntaxError as ex:
        raise ValueError(f"{path} is not a valid Python expression.") from ex
    return source


def _validate_python_statements(value: Any, path: str) -> str:
    source = _require_string(value, path, nonempty=True)
    try:
        ast.parse(source)
    except SyntaxError as ex:
        raise ValueError(f"{path} is not valid Python source.") from ex
    return source


def _validate_codegen_manifest(manifest: dict[str, Any]) -> None:
    manifest = _require_exact_object(
        manifest,
        "PyNTT codegen manifest",
        {"pyntt_codegen_manifest_version", "target_kind", "backend", "functions"},
    )
    _require_string(manifest["target_kind"], "manifest.target_kind", nonempty=True)
    _require_string(manifest["backend"], "manifest.backend", nonempty=True)
    functions = _require_list(manifest["functions"], "manifest.functions")
    for function_index, function in enumerate(functions):
        function_path = f"manifest.functions[{function_index}]"
        function = _require_exact_object(
            function,
            function_path,
            {"id", "name", "module_kind", "is_entry", "render_kernels"},
        )
        _require_int(function["id"], f"{function_path}.id", minimum=0)
        _require_string(function["name"], f"{function_path}.name", nonempty=True)
        _require_string(
            function["module_kind"], f"{function_path}.module_kind", nonempty=True
        )
        if not isinstance(function["is_entry"], bool):
            raise ValueError(f"{function_path}.is_entry must be a boolean.")
        kernels = _require_list(
            function["render_kernels"], f"{function_path}.render_kernels"
        )
        for kernel_index, kernel in enumerate(kernels):
            kernel_path = f"{function_path}.render_kernels[{kernel_index}]"
            kernel = _require_exact_object(
                kernel,
                kernel_path,
                {
                    "metadata",
                    "helpers",
                    "device_functions",
                    "body_source",
                },
            )
            metadata_path = f"{kernel_path}.metadata"
            metadata = _require_exact_object(
                kernel["metadata"],
                metadata_path,
                {"name", "op_kind", "inputs", "outputs", "attrs", "launch"},
            )
            _require_string(metadata["name"], f"{metadata_path}.name", nonempty=True)
            _require_string(
                metadata["op_kind"], f"{metadata_path}.op_kind", nonempty=True
            )
            _require_string_list(metadata["inputs"], f"{metadata_path}.inputs")
            _require_string_list(metadata["outputs"], f"{metadata_path}.outputs")
            if not isinstance(metadata["attrs"], dict):
                raise ValueError(f"{metadata_path}.attrs must be a JSON object.")
            _validate_launch(metadata["launch"], f"{metadata_path}.launch")
            helpers = _require_list(kernel["helpers"], f"{kernel_path}.helpers")
            for helper_index, helper in enumerate(helpers):
                _validate_helper(helper, f"{kernel_path}.helpers[{helper_index}]")
            device_functions = _require_list(
                kernel["device_functions"], f"{kernel_path}.device_functions"
            )
            for device_index, device_function in enumerate(device_functions):
                _validate_device_function(
                    device_function,
                    f"{kernel_path}.device_functions[{device_index}]",
                )
            body_source = _require_string(
                kernel["body_source"], f"{kernel_path}.body_source"
            )
            if body_source.strip():
                _validate_python_statements(body_source, f"{kernel_path}.body_source")


def render_generated_kernels(
    model_dir: str | Path,
    *,
    package: str | None = None,
    manifest_name: str = "kernel_params.json",
    output_name: str = "generated_kernels.py",
) -> Path:
    """Render ``generated_kernels.py`` from ``kernel_params.json``.

    The nncase compiler emits the manifest. PyNTT owns the Jinja templates and
    this renderer so kernel-template changes do not require recompiling nncase
    or recompiling the model.
    """

    model_dir = Path(model_dir)
    manifest_path = model_dir / manifest_name
    output_path = model_dir / output_name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    source = render_manifest(manifest)
    output_path.write_text(source, encoding="utf-8")

    if package:
        sys.modules.pop(f"{package}.generated_kernels", None)
        importlib.invalidate_caches()

    return output_path


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate the compiler-to-PyNTT manifest reader contract."""

    if not isinstance(manifest, dict):
        raise ValueError("The PyNTT codegen manifest must be a JSON object.")
    manifest_version = manifest.get("pyntt_codegen_manifest_version")
    if (
        isinstance(manifest_version, bool)
        or not isinstance(manifest_version, int)
        or manifest_version != PYNTT_CODEGEN_MANIFEST_VERSION
    ):
        raise ValueError(
            "Unsupported PyNTT codegen manifest version "
            f"{manifest_version!r}; expected {PYNTT_CODEGEN_MANIFEST_VERSION}."
        )
    _validate_codegen_manifest(manifest)


def render_manifest(manifest: dict[str, Any]) -> str:
    validate_manifest(manifest)

    render_kernels = [
        kernel
        for function in manifest.get("functions", ())
        for kernel in function.get("render_kernels", ())
    ]
    grid_mesh_topology = _grid_mesh_topology(manifest)
    kernels = [
        _render_kernel(kernel, grid_mesh_topology) for kernel in render_kernels
    ]
    kernel_configs: dict[str, dict[str, Any]] = {}
    host_tensor_descriptor_specs: dict[str, tuple[dict[str, Any], ...]] = {}
    for kernel in render_kernels:
        name = kernel["metadata"]["name"]
        if name in kernel_configs:
            raise ValueError(f"PyNTT manifest contains duplicate kernel name {name!r}.")
        kernel_configs[name] = _kernel_backend_config(kernel)
        host_tensor_descriptor_specs[name] = _kernel_host_tensor_descriptor_specs(
            kernel
        )
    grid_barrier_axis_groups = _grid_barrier_axis_groups(
        manifest, grid_mesh_topology
    )
    env = _make_env()
    return env.get_template("triton/module.py.jinja").render(
        kernels=kernels,
        kernel_configs=kernel_configs,
        host_tensor_descriptor_specs=host_tensor_descriptor_specs,
        grid_mesh_axes=[
            (axis["name"], axis["size"])
            for axis in grid_mesh_topology
            if axis["level"] == "b"
        ],
        grid_barrier_axis_groups=grid_barrier_axis_groups,
    )


def _paged_attention_backend_config(kernel: dict[str, Any]) -> dict[str, Any]:
    cache_block_sizes: set[int] = set()
    helper_groups = [kernel.get("helpers", ())]
    helper_groups.extend(
        device_function.get("helpers", ())
        for device_function in kernel.get("device_functions", ())
    )
    for helpers in helper_groups:
        for helper in helpers:
            template = helper.get("template")
            if not isinstance(template, str) or not template.startswith(
                "triton/kernels/paged_attention/"
            ):
                continue
            model = helper.get("model")
            cache = model.get("Cache") if isinstance(model, dict) else None
            if not isinstance(cache, dict):
                continue
            cache_block_sizes.add(
                _require_int(
                    cache.get("BlockSize"),
                    f"PyNTT kernel {kernel['metadata']['name']} PagedAttention "
                    "cache block size",
                    minimum=1,
                )
            )

    block_n_candidates = tuple(
        candidate
        for candidate in PAGED_ATTENTION_BLOCK_N_CANDIDATES
        if all(
            candidate <= cache_block_size
            and cache_block_size % candidate == 0
            for cache_block_size in cache_block_sizes
        )
    )
    if cache_block_sizes and not block_n_candidates:
        raise ValueError(
            f"PyNTT kernel {kernel['metadata']['name']} has PagedAttention cache "
            f"block sizes {sorted(cache_block_sizes)} but no legal block_n in "
            f"{PAGED_ATTENTION_BLOCK_N_CANDIDATES}."
        )
    if not block_n_candidates:
        block_n_candidates = PAGED_ATTENTION_BLOCK_N_CANDIDATES

    block_n = min(PAGED_ATTENTION_BLOCK_N, max(block_n_candidates))
    return {
        "block_n": block_n,
        "block_n_candidates": block_n_candidates,
        "num_stages": PAGED_ATTENTION_NUM_STAGES,
        "num_stages_candidates": PAGED_ATTENTION_NUM_STAGES_CANDIDATES,
    }


def _kernel_backend_config(kernel: dict[str, Any]) -> dict[str, Any]:
    metadata = kernel["metadata"]
    attrs = _attrs(metadata)
    worker_width = _require_int(
        attrs.get("target_worker_width"),
        f"PyNTT kernel {metadata['name']} attrs.target_worker_width",
        minimum=1,
    )
    threads_per_block = _require_int(
        attrs.get("target_threads_per_block"),
        f"PyNTT kernel {metadata['name']} attrs.target_threads_per_block",
        minimum=1,
    )
    resident_blocks_per_compute_unit = _require_int(
        attrs.get("target_resident_blocks_per_compute_unit"),
        f"PyNTT kernel {metadata['name']} "
        "attrs.target_resident_blocks_per_compute_unit",
        minimum=1,
    )
    num_warps, remainder = divmod(threads_per_block, worker_width)
    if remainder:
        raise ValueError(
            f"PyNTT kernel {metadata['name']} target execution geometry is invalid: "
            "target_threads_per_block must be divisible by target_worker_width, got "
            f"{threads_per_block} and {worker_width}."
        )
    register_file_capacity_units = _require_int(
        attrs.get("register_file_capacity_units"),
        f"PyNTT kernel {metadata['name']} attrs.register_file_capacity_units",
        minimum=1,
    )
    register_file_allocation_granularity_units = _require_int(
        attrs.get("register_file_allocation_granularity_units"),
        f"PyNTT kernel {metadata['name']} "
        "attrs.register_file_allocation_granularity_units",
        minimum=1,
    )
    register_granularity, remainder = divmod(
        register_file_allocation_granularity_units, worker_width
    )
    if remainder:
        raise ValueError(
            f"PyNTT kernel {metadata['name']} register allocation granularity "
            f"{register_file_allocation_granularity_units} is not divisible by "
            f"worker width {worker_width}."
        )
    physical_warps = num_warps + WARP_SPECIALIZATION_WORKER_ALLOCATION_WARPS
    uniform_register_limit = register_file_capacity_units // (
        physical_warps * worker_width
    )
    uniform_register_limit = (
        uniform_register_limit // register_granularity
    ) * register_granularity
    producer_registers = (
        (
            WARP_SPECIALIZATION_MINIMUM_PARTITION_REGISTERS
            + register_granularity
            - 1
        )
        // register_granularity
    ) * register_granularity
    if uniform_register_limit < producer_registers:
        raise ValueError(
            f"PyNTT kernel {metadata['name']} has insufficient register capacity "
            "for its fixed producer/consumer execution geometry."
        )
    reclaimed_registers = (
        uniform_register_limit - producer_registers
    ) * WARP_SPECIALIZATION_WORKER_ALLOCATION_WARPS
    consumer_registers = uniform_register_limit + (
        reclaimed_registers // num_warps // register_granularity
    ) * register_granularity
    registers_per_thread_limit = _require_int(
        attrs.get("registers_per_thread_limit"),
        f"PyNTT kernel {metadata['name']} attrs.registers_per_thread_limit",
        minimum=1,
    )
    if consumer_registers > registers_per_thread_limit:
        consumer_registers = (
            registers_per_thread_limit // register_granularity
        ) * register_granularity
    block_size_candidates = tuple(
        sorted(
            {
                worker_width,
                *(
                    candidate
                    for candidate in THROUGHPUT_BLOCK_SIZE_CANDIDATES
                    if candidate >= worker_width and candidate % worker_width == 0
                ),
            }
        )
    )
    return {
        "block_size": {
            "source": "autotune",
            "candidates": block_size_candidates,
        },
        "num_warps": num_warps,
        "num_stages": 1,
        "resident_blocks_per_compute_unit": resident_blocks_per_compute_unit,
        "paged_attention": _paged_attention_backend_config(kernel),
        "producer_warps": WARP_SPECIALIZATION_PRODUCER_WARPS,
        "producer_registers": producer_registers,
        "consumer_registers": consumer_registers,
        "register_granularity": register_granularity,
        "registers_per_thread_limit": registers_per_thread_limit,
    }


def _kernel_parameters(metadata: dict[str, Any]) -> tuple[str, ...]:
    host_tensor_descriptor_parameters = tuple(
        descriptor["name"]
        for descriptor in metadata.get("launch", {}).get(
            "host_tensor_descriptors", ()
        )
    )
    return (
        tuple(f"input{index}" for index, _ in enumerate(metadata.get("inputs", ())))
        + tuple(f"output{index}" for index, _ in enumerate(metadata.get("outputs", ())))
        + tuple(
            f"input{index}_pool_stride_elements: tl.constexpr"
            for index, _ in enumerate(metadata.get("inputs", ()))
        )
        + tuple(
            f"output{index}_pool_stride_elements: tl.constexpr"
            for index, _ in enumerate(metadata.get("outputs", ()))
        )
        + _abi_view_stride_args(metadata)
        + WORKSPACE_PARAMETERS
        + WORKSPACE_STRIDE_PARAMETERS
        + tuple(_runtime_shape_args(metadata))
        + host_tensor_descriptor_parameters
        + ("numel", "block_size: tl.constexpr")
    )


def _kernel_host_tensor_descriptor_specs(
    kernel: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    metadata = kernel["metadata"]
    backings = tuple(
        metadata.get("launch", {}).get("host_tensor_descriptors", ())
    )
    if not backings:
        return ()

    backing_by_name = {backing["name"]: backing for backing in backings}
    specs_by_name: dict[str, dict[str, Any]] = {}
    device_functions = {
        function["name"]: function
        for function in kernel.get("device_functions", ())
    }
    active_functions: set[str] = set()

    def resolve_descriptor_name(
        formal_name: str,
        environment: dict[str, str],
        context: str,
    ) -> str:
        actual_name = environment.get(formal_name, formal_name)
        if actual_name not in backing_by_name:
            raise ValueError(
                f"{context} binds host descriptor {formal_name!r} to unknown "
                f"backing {actual_name!r}."
            )
        return actual_name

    def consume_helpers(
        helpers: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        environment: dict[str, str],
        owner: str,
    ) -> None:
        for helper in helpers:
            model = helper["model"]
            for formal_name, make_spec in _pipeline_helper_descriptor_specs(helper):
                actual_name = resolve_descriptor_name(
                    formal_name,
                    environment,
                    f"PyNTT helper {model.get('FunctionName')} in {owner}",
                )
                spec = make_spec(model, backing_by_name[actual_name])
                previous = specs_by_name.get(actual_name)
                if previous is not None and previous != spec:
                    raise ValueError(
                        f"PyNTT host descriptor {actual_name!r} has "
                        "incompatible consumers."
                    )
                specs_by_name[actual_name] = spec
            for body_key in ("ConsumerBodySource", "ProducerBodySource"):
                body_source = model.get(body_key)
                if isinstance(body_source, str) and body_source.strip():
                    walk_calls(
                        body_source,
                        environment,
                        f"{owner}.{model.get('FunctionName')}.{body_key}",
                    )

    def walk_calls(
        source: str,
        environment: dict[str, str],
        owner: str,
    ) -> None:
        for match in DEVICE_CALL_RE.finditer(source):
            callee_name = match.group("name")
            try:
                callee = device_functions[callee_name]
            except KeyError as ex:
                raise ValueError(
                    f"PyNTT {owner} calls unknown device function "
                    f"{callee_name!r} while resolving host descriptors."
                ) from ex
            if callee_name in active_functions:
                raise ValueError(
                    f"Recursive PyNTT device call involving {callee_name!r}."
                )

            explicit_arguments = _split_expression_arguments(match.group("args"))
            formal_parameters = _parameter_call_arguments(
                tuple(callee["extra_parameters"])
            )
            if explicit_arguments:
                if len(explicit_arguments) != len(formal_parameters):
                    raise ValueError(
                        f"PyNTT call to {callee_name} passes "
                        f"{len(explicit_arguments)} explicit parameters, expected "
                        f"{len(formal_parameters)}."
                    )
                raw_bindings = dict(zip(formal_parameters, explicit_arguments))
            else:
                raw_bindings = dict(callee["extra_parameter_arguments"])

            callee_environment: dict[str, str] = {}
            for formal_name, expression in raw_bindings.items():
                if not isinstance(expression, str):
                    continue
                actual_name = environment.get(expression, expression)
                if actual_name in backing_by_name:
                    callee_environment[formal_name] = actual_name

            active_functions.add(callee_name)
            try:
                consume_helpers(
                    callee.get("helpers", ()),
                    callee_environment,
                    callee_name,
                )
                walk_calls(
                    callee.get("body_source", ""),
                    callee_environment,
                    callee_name,
                )
            finally:
                active_functions.remove(callee_name)

    root_environment = {name: name for name in backing_by_name}
    consume_helpers(
        kernel.get("helpers", ()),
        root_environment,
        kernel["metadata"]["name"],
    )
    walk_calls(
        kernel.get("body_source", ""),
        root_environment,
        kernel["metadata"]["name"],
    )

    unused = sorted(set(backing_by_name) - set(specs_by_name))
    if unused:
        raise ValueError(
            f"PyNTT kernel {metadata['name']} has unconsumed host tensor "
            f"descriptor backing(s): {unused}."
        )
    return tuple(specs_by_name[backing["name"]] for backing in backings)


def _pipeline_helper_descriptor_specs(
    helper: dict[str, Any],
) -> tuple[tuple[str, Any], ...]:
    model = helper["model"]
    template = helper["template"]
    if template in (
        "triton/kernels/matmul/simt_fma_smem_pipeline.py.jinja",
        "triton/kernels/matmul/simt_fp8_fma_smem_pipeline.py.jinja",
        "triton/kernels/matmul/simt_block_fp8_fma_smem_pipeline.py.jinja",
        "triton/kernels/matmul/mma_block_fp8_smem_pipeline.py.jinja",
    ):
        descriptor_names = (model.get("RhsDescriptorName"),)
        descriptor_specs = (
            _n_major_k_packed_gemv_host_descriptor_spec
            if template.endswith("/mma_block_fp8_smem_pipeline.py.jinja")
            else _packed_gemv_host_descriptor_spec,
        )
    elif (
        template
        == "triton/kernels/paged_attention_merge_matmul/simt_fma_smem_pipeline.py.jinja"
    ):
        matmul = model.get("Matmul")
        if not isinstance(matmul, dict):
            raise ValueError(
                "PyNTT fused paged-attention merge/matmul is missing Matmul metadata."
            )
        descriptor_names = (matmul.get("RhsDescriptorName"),)
        descriptor_specs = (
            _paged_attention_merge_matmul_host_descriptor_spec,
        )
    elif (
        template
        == "triton/kernels/matmul_sampling_partial/simt_fma_smem_pipeline.py.jinja"
    ):
        matmul = model.get("Matmul")
        if not isinstance(matmul, dict):
            raise ValueError("PyNTT packed matmul sampling is missing Matmul metadata.")
        descriptor_names = (matmul.get("RhsDescriptorName"),)
        descriptor_specs = (_packed_matmul_sampling_partial_host_descriptor_spec,)
    elif (
        template in (
            "triton/kernels/qkv_parallel_linear/simt_fma_smem_pipeline.py.jinja",
            "triton/kernels/qkv_parallel_linear/simt_fp8_fma_smem_pipeline.py.jinja",
            "triton/kernels/qkv_parallel_linear/mma_smem_pipeline.py.jinja",
        )
    ):
        descriptor_names = (model.get("WeightDescriptorName"),)
        descriptor_specs = (_packed_qkv_gemv_host_descriptor_spec,)
    elif (
        template in (
            "triton/kernels/matmul_glu/simt_fma_smem_pipeline.py.jinja",
            "triton/kernels/matmul_glu/simt_fp8_fma_smem_pipeline.py.jinja",
            "triton/kernels/matmul_glu/simt_block_fp8_fma_smem_pipeline.py.jinja",
        )
    ):
        descriptor_names = tuple(
            model.get(f"{prefix}WeightDescriptorName")
            for prefix in ("Gate", "Up")
        )
        descriptor_specs = tuple(
            (
                lambda current_model, backing, current_prefix=prefix:
                _packed_matmul_glu_gemv_host_descriptor_spec(
                    current_model, backing, current_prefix
                )
            )
            for prefix in ("Gate", "Up")
        )
    elif template in (
        "triton/kernels/paged_attention/mma_tma_smem_pipeline.py.jinja",
        "triton/kernels/paged_attention/simt_tma_smem_pipeline.py.jinja",
    ):
        descriptor_names = (
            model.get("KeyDescriptorName"),
            model.get("ValueDescriptorName"),
        )
        descriptor_specs = (
            _paged_attention_host_descriptor_spec,
            _paged_attention_host_descriptor_spec,
        )
    elif (
        template
        == "triton/kernels/gated_delta_net/recurrent_core.py.jinja"
    ):
        descriptor_names = (
            model.get("BWeightDescriptorName"),
            model.get("AWeightDescriptorName"),
        )
        descriptor_specs = (
            _gated_delta_net_projection_host_descriptor_spec,
            _gated_delta_net_projection_host_descriptor_spec,
        )
    elif template == "triton/kernels/nvfp4_matmul/mma_tma_smem_pipeline.py.jinja":
        descriptor_names = (model.get("RhsPackedDescriptorName"),)
        descriptor_specs = (
            lambda current_model, backing: _nvfp4_n_major_host_descriptor_spec(
                current_model,
                backing,
                pointer_key="RhsPacked",
            ),
        )
    elif template == "triton/kernels/nvfp4_matmul_glu/mma_tma_smem_pipeline.py.jinja":
        descriptor_names = tuple(
            model.get(f"{prefix}WeightPackedDescriptorName")
            for prefix in ("Gate", "Up")
        )
        descriptor_specs = tuple(
            lambda current_model, backing, current_prefix=prefix:
                _nvfp4_n_major_host_descriptor_spec(
                    current_model,
                    backing,
                    pointer_key=f"{current_prefix}WeightPacked",
                )
            for prefix in ("Gate", "Up")
        )
    else:
        return ()

    result = []
    for descriptor_name, make_spec in zip(descriptor_names, descriptor_specs):
        if not isinstance(descriptor_name, str) or not descriptor_name:
            raise ValueError(
                f"PyNTT pipeline helper {model.get('FunctionName')} requires "
                "a host tensor descriptor name."
            )
        result.append((descriptor_name, make_spec))
    return tuple(result)


def _gated_delta_net_projection_host_descriptor_spec(
    model: dict[str, Any],
    backing: dict[str, Any],
) -> dict[str, Any]:
    microkernel = _microkernel_context(
        model,
        "triton.gated_delta_net",
        "recurrent_core",
        required_workspace_names=(
            "b_projection_stage",
            "a_projection_stage",
            "projection_stage",
        ),
    )
    parameters = microkernel["parameters"]
    block_k = parameters["block_k"]
    head_capacity = _require_int(
        parameters.get("projection_head_capacity"),
        "projection_head_capacity",
        minimum=1,
    )
    k_atom = _require_int(
        parameters.get("projection_tma_k_atom"),
        "projection_tma_k_atom",
        minimum=1,
    )
    hidden_size = _require_int(model.get("HiddenSize"), "HiddenSize", minimum=1)
    num_value_heads = _require_int(
        model.get("NumValueHeads"), "NumValueHeads", minimum=1
    )
    descriptor_name = backing["name"]
    if descriptor_name == model.get("BWeightDescriptorName"):
        owner_indexed = model.get("BWeightDescriptorOwnerIndexed")
    elif descriptor_name == model.get("AWeightDescriptorName"):
        owner_indexed = model.get("AWeightDescriptorOwnerIndexed")
    else:
        raise ValueError(
            "PyNTT GatedDeltaNet projection descriptor backing does not match "
            f"the helper contract: {descriptor_name!r}."
        )
    if not isinstance(owner_indexed, bool):
        raise ValueError(
            "PyNTT GatedDeltaNet projection descriptor owner-indexed flag must "
            f"be bool, got {owner_indexed!r}."
        )
    owner_stride_bytes = int(backing["owner_stride_bytes"])
    if owner_indexed != (owner_stride_bytes > 0):
        raise ValueError(
            "PyNTT GatedDeltaNet projection descriptor addressing disagrees "
            f"with its backing: owner_indexed={owner_indexed}, "
            f"owner_stride_bytes={owner_stride_bytes}."
        )
    logical_shape = tuple(int(value) for value in backing["logical_shape"])
    logical_strides = _normalize_singleton_strides(
        logical_shape,
        tuple(int(value) for value in backing["logical_strides"]),
    )
    vector_lane_shape = tuple(int(value) for value in backing["vector_lane_shape"])
    if (
        backing["scalar_dtype"] != "bfloat16"
        or vector_lane_shape
        or logical_shape != (num_value_heads, hidden_size)
        or logical_strides != (hidden_size, 1)
        or hidden_size % k_atom != 0
        or block_k % k_atom != 0
    ):
        raise ValueError(
            "PyNTT GatedDeltaNet projection TMA requires contiguous scalar BF16 "
            f"weights [{num_value_heads}, {hidden_size}], block_k={block_k}, "
            f"k_atom={k_atom}; got shape={logical_shape}, strides={logical_strides}, "
            f"lanes={vector_lane_shape}, dtype={backing['scalar_dtype']!r}."
        )
    contiguous_rebase_extent = int(backing["contiguous_rebase_extent_elements"])
    descriptor_shape = (
        num_value_heads,
        hidden_size // k_atom,
        k_atom + contiguous_rebase_extent,
    )
    descriptor_strides = (hidden_size, k_atom, 1)
    block_shape = (head_capacity, block_k // k_atom, k_atom)
    common = {
        "name": backing["name"],
        "source": backing["source"],
        "dtype": backing["scalar_dtype"],
        "block_shape": block_shape,
        "padding": "zero",
    }
    if not owner_indexed:
        return {
            "kind": "single",
            **common,
            "offset_bytes": int(backing["offset_bytes"]),
            "shape": descriptor_shape,
            "strides": descriptor_strides,
            "source_shape_axes": ((), (), ()),
        }

    pointer_name = (
        "BWeight"
        if descriptor_name == model.get("BWeightDescriptorName")
        else "AWeight"
    )
    pointer = model.get(pointer_name)
    hierarchy = pointer.get("Hierarchy") if isinstance(pointer, dict) else None
    if not isinstance(hierarchy, list) or not hierarchy:
        raise ValueError(
            "PyNTT GatedDeltaNet owner-indexed projection descriptor requires "
            f"a non-empty {pointer_name} hierarchy."
        )
    owner_count = _product_int(
        [
            _require_int(extent, f"{pointer_name} hierarchy extent", minimum=1)
            for extent in hierarchy
        ]
    )
    entries = tuple(
        {
            "offset_bytes": int(backing["offset_bytes"])
            + owner * owner_stride_bytes,
            "shape": descriptor_shape,
            "strides": descriptor_strides,
            "source_shape_axes": ((), (), ()),
        }
        for owner in range(owner_count)
    )
    return {
        "kind": "table",
        **common,
        "swizzle_mode": _nv_tma_swizzle_mode(
            block_shape, backing["scalar_dtype"]
        ),
        "entry_size_bytes": TENSOR_MAP_ENTRY_BYTES,
        "entries": entries,
    }


def _paged_attention_host_descriptor_spec(
    model: dict[str, Any],
    backing: dict[str, Any],
) -> dict[str, Any]:
    microkernel = _microkernel_context(
        model,
        "triton.paged_attention_partial",
    )
    if microkernel["variant"] not in (
        "mma_tma_smem_pipeline",
        "simt_tma_smem_pipeline",
    ):
        raise ValueError(
            "PyNTT PagedAttention host descriptor requires a TMA shared "
            f"pipeline variant, got {microkernel['variant']!r}."
        )
    block_n = microkernel["parameters"]["block_n"]
    head_dim = microkernel["parameters"]["head_dim"]
    copy_block_n, _ = _paged_attention_tile_geometry(
        block_n,
        int(model["Cache"]["BlockSize"]),
        allow_cross_page=True,
    )
    shape = tuple(int(value) for value in backing["logical_shape"])
    strides = _normalize_singleton_strides(
        shape,
        tuple(int(value) for value in backing["logical_strides"]),
    )
    source_shape_axes = tuple(
        tuple(int(axis) for axis in axes)
        for axes in backing["source_shape_axes"]
    )
    if len(shape) != 5 or len(strides) != 5 or len(source_shape_axes) != 5:
        raise ValueError(
            "PyNTT PagedAttention host descriptor requires rank-5 "
            "shape/stride/source-shape metadata."
        )
    if shape[-1] != head_dim or shape[2] % copy_block_n != 0:
        raise ValueError(
            "PyNTT PagedAttention host descriptor shape is incompatible with "
            f"copy_block_n={copy_block_n}, head_dim={head_dim}: {shape}."
        )
    return {
        "kind": "single",
        "name": backing["name"],
        "source": backing["source"],
        "offset_bytes": int(backing["offset_bytes"]),
        "dtype": backing["scalar_dtype"],
        "shape": shape,
        "strides": strides,
        "block_shape": (1, 1, copy_block_n, 1, head_dim),
        "source_shape_axes": source_shape_axes,
        "padding": "zero",
    }


def _packed_gemv_host_descriptor_spec(
    model: dict[str, Any],
    backing: dict[str, Any],
) -> dict[str, Any]:
    microkernel = _microkernel_context(
        model, "triton.matmul", str(model["MicroKernel"]["Variant"])
    )
    return _k_major_gemv_host_descriptor_spec(
        model,
        backing,
        block_n=microkernel["parameters"]["block_n"],
        block_k=microkernel["parameters"]["block_k"],
        pointer=model["Rhs"],
        expected_vector_lane_shape=_packed_gemv_vector_lane_shape(
            microkernel["variant"]
        ),
    )


def _n_major_k_packed_gemv_descriptor_n_plan(
    pointer: dict[str, Any],
    logical_block_n: int,
) -> dict[str, Any]:
    """Preserve the distributed N-axis rectangle in one TMA transaction."""

    plan = _tma_local_axis_plan(
        pointer,
        0,
        tile_extent=logical_block_n,
        context="N-major K-packed GEMV descriptor N",
    )
    if (
        plan["is_block_cyclic"]
        and plan["block_size"] != 1
        and logical_block_n <= int(plan["block_size"])
    ):
        retained_dimensions = (1,)
    else:
        retained_dimensions = tuple(range(len(plan["block_shape"])))
    return {
        **plan,
        "raw_block_shape": tuple(plan["block_shape"]),
        "block_shape": tuple(
            plan["block_shape"][dimension]
            for dimension in retained_dimensions
        ),
        "retained_dimensions": retained_dimensions,
    }


def _n_major_k_packed_gemv_host_descriptor_spec(
    model: dict[str, Any],
    backing: dict[str, Any],
) -> dict[str, Any]:
    """Build owner/tile-indexed rank-2 TMA views of scalar [N, K] weights."""

    microkernel = _microkernel_context(
        model, "triton.matmul", "mma_block_fp8_smem_pipeline"
    )
    logical_block_n = microkernel["parameters"]["block_n"]
    transfer_block_k = _require_int(
        microkernel["parameters"].get("transfer_block_k"),
        "microkernel.transfer_block_k",
        minimum=1,
    )
    reduction_group = _require_int(
        model.get("WeightBlockK"), "WeightBlockK", minimum=1
    )
    logical_shape = tuple(int(value) for value in backing["logical_shape"])
    logical_strides = _normalize_singleton_strides(
        logical_shape,
        tuple(int(value) for value in backing["logical_strides"]),
    )
    vector_lane_shape = tuple(int(value) for value in backing["vector_lane_shape"])
    if len(logical_shape) != 2 or len(logical_strides) != 2:
        raise ValueError(
            "PyNTT N-major K-packed GEMV descriptor requires a rank-2 "
            f"logical RHS backing, got {logical_shape}/{logical_strides}."
        )
    if vector_lane_shape != (2, 16):
        raise ValueError(
            "PyNTT N-major K-packed block-FP8 GEMV requires RHS vector lanes "
            f"(2, 16), got {vector_lane_shape}."
        )

    k_atom = _product_int(list(vector_lane_shape))
    if (
        logical_block_n <= 0
        or transfer_block_k % k_atom != 0
        or reduction_group % transfer_block_k != 0
    ):
        raise ValueError(
            "PyNTT N-major K-packed GEMV descriptor tile is incompatible with "
            f"block_n={logical_block_n}, reduction_group={reduction_group}, "
            f"transfer_block_k={transfer_block_k}, k_atom={k_atom}."
        )
    k_outer_extent = transfer_block_k // k_atom

    pointer = model["Rhs"]
    n_plan = _n_major_k_packed_gemv_descriptor_n_plan(pointer, logical_block_n)
    descriptor_block_shape = tuple(n_plan["block_shape"]) + (transfer_block_k,)
    hierarchy = pointer.get("Hierarchy")
    pointer_global_shape = pointer.get("GlobalShape")
    if not isinstance(hierarchy, list) or not hierarchy:
        raise ValueError(
            "PyNTT N-major K-packed GEMV descriptor requires a hierarchy."
        )
    if not isinstance(pointer_global_shape, list) or len(pointer_global_shape) != 2:
        raise ValueError(
            "PyNTT N-major K-packed GEMV descriptor requires a rank-2 global pointer."
        )
    fixed_pointer_shape = tuple(
        _require_fixed_positive_dim(
            extent, f"PyNTT N-major K-packed GEMV global axis {axis}"
        )
        for axis, extent in enumerate(pointer_global_shape)
    )
    if fixed_pointer_shape != logical_shape:
        raise ValueError(
            "PyNTT N-major K-packed GEMV descriptor backing/pointer shapes "
            f"differ: {logical_shape}/{fixed_pointer_shape}."
        )

    item_size = TMA_DTYPE_ITEM_SIZES.get(backing["scalar_dtype"])
    if item_size is None:
        raise ValueError(
            "PyNTT N-major K-packed GEMV descriptor does not support scalar "
            f"dtype {backing['scalar_dtype']!r}."
        )
    contiguous_rebase_extent = int(backing["contiguous_rebase_extent_elements"])
    owner_count = _product_int([int(value) for value in hierarchy])
    owner_entries: list[list[dict[str, Any]]] = []
    maximum_tiles_per_owner = 0
    for linear_owner in range(owner_count):
        owner = _unflatten_hierarchy_owner(linear_owner, hierarchy)
        n_entry = _tma_descriptor_table_axis_entry(
            pointer,
            0,
            owner,
            tile_extent=logical_block_n,
            context="N-major K-packed GEMV descriptor N",
        )
        k_entry = _tma_descriptor_table_axis_entry(
            pointer,
            1,
            owner,
            tile_extent=k_outer_extent,
            context="N-major K-packed GEMV descriptor K",
        )

        if not n_entry["active"] or not k_entry["active"]:
            owner_entries.append([])
            continue
        if (
            k_entry["is_block_cyclic"]
            or tuple(k_entry["stride_multipliers"]) != (1,)
            or logical_strides[1] != 1
        ):
            raise ValueError(
                "PyNTT N-major K-packed GEMV requires a contiguous local K axis."
            )

        local_n_extent = _product_int(list(n_entry["descriptor_shape"]))
        local_k_extent = (
            _product_int(list(k_entry["descriptor_shape"])) * k_atom
            + contiguous_rebase_extent
        )
        descriptor_tile_count = (
            local_n_extent + logical_block_n - 1
        ) // logical_block_n
        maximum_tiles_per_owner = max(
            maximum_tiles_per_owner, descriptor_tile_count
        )
        base_scalar_elements = _tma_owner_backing_base_elements(
            pointer,
            (n_entry, k_entry),
            logical_strides,
            scalar_lanes_per_logical_element=k_atom,
            context="N-major K-packed GEMV",
        )
        entries_for_owner = []
        for tile_index in range(descriptor_tile_count):
            local_n_offset = tile_index * logical_block_n
            if n_entry["is_block_cyclic"] and n_entry["block_size"] != 1:
                block_size = int(n_entry["block_size"])
                raw_coordinates = (
                    local_n_offset // block_size,
                    local_n_offset % block_size,
                )
            else:
                raw_coordinates = (local_n_offset,)
            if len(raw_coordinates) != len(n_entry["stride_multipliers"]):
                raise ValueError(
                    "PyNTT N-major K-packed GEMV N coordinate/stride ranks "
                    f"differ: {raw_coordinates}/{n_entry['stride_multipliers']}."
                )
            physical_n_offset = sum(
                coordinate * int(multiplier)
                for coordinate, multiplier in zip(
                    raw_coordinates, n_entry["stride_multipliers"]
                )
            )
            tile_n_extent = min(
                logical_block_n, local_n_extent - local_n_offset
            )
            if n_entry["is_block_cyclic"] and n_entry["block_size"] != 1:
                block_size = int(n_entry["block_size"])
                if logical_block_n <= block_size:
                    raw_descriptor_n_shape = (1, tile_n_extent)
                elif tile_n_extent % block_size != 0:
                    raise ValueError(
                        "PyNTT N-major K-packed GEMV cannot encode a partial "
                        f"block-cyclic N block: extent={tile_n_extent}, "
                        f"block_size={block_size}."
                    )
                else:
                    raw_descriptor_n_shape = (
                        tile_n_extent // block_size,
                        block_size,
                    )
            else:
                raw_descriptor_n_shape = (tile_n_extent,)
            descriptor_row_stride = logical_strides[0] * k_atom
            descriptor_n_shape = tuple(
                raw_descriptor_n_shape[dimension]
                for dimension in n_plan["retained_dimensions"]
            )
            descriptor_n_strides = tuple(
                int(n_entry["stride_multipliers"][dimension])
                * descriptor_row_stride
                for dimension in n_plan["retained_dimensions"]
            )
            descriptor_shape = descriptor_n_shape + (local_k_extent,)
            descriptor_strides = descriptor_n_strides + (1,)
            entries_for_owner.append(
                {
                    "offset_bytes": int(backing["offset_bytes"])
                    + linear_owner * int(backing["owner_stride_bytes"])
                    + (
                        base_scalar_elements
                        + physical_n_offset * logical_strides[0] * k_atom
                    )
                    * item_size,
                    "shape": descriptor_shape,
                    "strides": descriptor_strides,
                    "source_shape_axes": tuple(() for _ in descriptor_shape),
                }
            )
        owner_entries.append(entries_for_owner)

    if maximum_tiles_per_owner <= 0:
        raise ValueError("PyNTT N-major K-packed GEMV has no active RHS tiles.")
    output_outer_n = _max_value(model["OutputShape"][-1])
    if output_outer_n is None:
        raise ValueError(
            "PyNTT N-major K-packed GEMV requires a bounded local output N axis."
        )
    output_n_lanes = int(model["OutputNVectorLaneCount"])
    expected_tiles_per_owner = (
        (output_outer_n * output_n_lanes + logical_block_n - 1)
        // logical_block_n
    )
    if maximum_tiles_per_owner != expected_tiles_per_owner:
        raise ValueError(
            "PyNTT N-major K-packed GEMV descriptor/output tile counts differ: "
            f"{maximum_tiles_per_owner}/{expected_tiles_per_owner}."
        )

    first_entry = next(
        entry for entries_for_owner in owner_entries for entry in entries_for_owner
    )
    padding_entry = {
        **first_entry,
        "shape": tuple(1 for _ in first_entry["shape"][:-1])
        + (first_entry["shape"][-1],),
    }
    entries = tuple(
        entry
        for entries_for_owner in owner_entries
        for entry in (
            entries_for_owner
            + [padding_entry]
            * (maximum_tiles_per_owner - len(entries_for_owner))
        )
    )

    return {
        "kind": "table",
        "name": backing["name"],
        "source": backing["source"],
        "dtype": backing["scalar_dtype"],
        "block_shape": descriptor_block_shape,
        "padding": "zero",
        "swizzle_mode": _nv_tma_swizzle_mode(
            descriptor_block_shape, backing["scalar_dtype"]
        ),
        "entry_size_bytes": TENSOR_MAP_ENTRY_BYTES,
        "entries": entries,
    }


def _tma_owner_backing_base_elements(
    pointer: dict[str, Any],
    axis_entries: tuple[dict[str, Any], ...],
    logical_strides: tuple[int, ...],
    *,
    scalar_lanes_per_logical_element: int = 1,
    context: str,
) -> int:
    """Map a logical owner origin into the selected physical backing."""

    if len(axis_entries) != len(logical_strides):
        raise ValueError(
            f"PyNTT {context} descriptor axis/stride ranks differ: "
            f"{len(axis_entries)}/{len(logical_strides)}."
        )
    storage_kind = _require_string(
        pointer.get("DistributedStorageKind"),
        f"PyNTT {context} distributed storage kind",
        nonempty=True,
    )
    if storage_kind in ("CompactLocal", "CompactPerOwner"):
        return 0
    if storage_kind != "CanonicalGlobal":
        raise ValueError(
            f"PyNTT {context} descriptor does not support distributed storage "
            f"kind {storage_kind!r}."
        )
    return (
        sum(
            int(entry["base"]) * int(stride)
            for entry, stride in zip(axis_entries, logical_strides)
        )
        * scalar_lanes_per_logical_element
    )


def _nvfp4_n_major_host_descriptor_spec(
    model: dict[str, Any],
    backing: dict[str, Any],
    *,
    pointer_key: str,
) -> dict[str, Any]:
    """Build scalar TMA views over target-packed N-major NVFP4 payloads."""

    family = _require_string(
        model.get("MicroKernel", {}).get("Family"),
        "microkernel.Family",
        nonempty=True,
    )
    if family == "triton.nvfp4_matmul":
        context = _nvfp4_matmul_template_context(model)
    elif family == "triton.nvfp4_matmul_glu":
        context = _nvfp4_matmul_glu_template_context(model)
    else:
        raise ValueError(
            f"PyNTT NVFP4 descriptor does not support family {family!r}."
        )
    pointer = model[pointer_key]
    logical_shape = tuple(int(value) for value in backing["logical_shape"])
    logical_strides = _normalize_singleton_strides(
        logical_shape,
        tuple(int(value) for value in backing["logical_strides"]),
    )
    vector_lane_shape = tuple(int(value) for value in backing["vector_lane_shape"])
    expected_dtype = "uint8"
    expected_vector_lane_shape = (2, 16)
    k_atom = _product_int(list(expected_vector_lane_shape))
    expected_local_k_outer = context["fixed_k"] // 2 // k_atom
    if (
        backing["scalar_dtype"] != expected_dtype
        or vector_lane_shape != expected_vector_lane_shape
        or len(logical_shape) != 2
        or len(logical_strides) != 2
        or logical_strides[1] != 1
    ):
        raise ValueError(
            "PyNTT NVFP4 TMA requires a target-packed N-major backing with a "
            "unit outer-K stride, got "
            f"shape={logical_shape}, strides={logical_strides}, "
            f"lanes={vector_lane_shape}, dtype={backing['scalar_dtype']!r}, "
            f"expected_lanes={expected_vector_lane_shape}, "
            f"expected_dtype={expected_dtype!r}."
        )

    hierarchy = pointer.get("Hierarchy")
    pointer_global_shape = pointer.get("GlobalShape")
    if not isinstance(hierarchy, list) or not hierarchy:
        raise ValueError("PyNTT NVFP4 TMA descriptor requires a hierarchy.")
    if not isinstance(pointer_global_shape, list) or len(pointer_global_shape) != 2:
        raise ValueError("PyNTT NVFP4 TMA descriptor requires a rank-2 global pointer.")
    fixed_pointer_shape = tuple(
        _require_fixed_positive_dim(
            extent, f"PyNTT NVFP4 TMA pointer global axis {axis}"
        )
        for axis, extent in enumerate(pointer_global_shape)
    )
    if fixed_pointer_shape != logical_shape:
        raise ValueError(
            "PyNTT NVFP4 TMA backing/pointer shapes differ: "
            f"{logical_shape}/{fixed_pointer_shape}."
        )

    block_n = context["block_n"]
    transfer_k = context["block_k"] // 2
    if transfer_k % k_atom != 0:
        raise ValueError(
            "PyNTT NVFP4 TMA transfer K must be divisible by the packed K atom, "
            f"got transfer_k={transfer_k}, k_atom={k_atom}."
        )
    transfer_k_outer = transfer_k // k_atom
    n_plan = _n_major_k_packed_gemv_descriptor_n_plan(pointer, block_n)
    k_payload_plan = _tma_packed_atom_axis_plan(
        pointer,
        1,
        tile_extent=transfer_k_outer,
        atom_extent=k_atom,
        logical_axis_stride=logical_strides[1],
        context=f"NVFP4 {pointer_key} TMA descriptor K",
    )
    k_plan = k_payload_plan["axis_plan"]
    descriptor_block_shape = context["packed_tma_block_shape"]
    pointer_block_shape = (
        tuple(n_plan["block_shape"])
        + tuple(k_payload_plan["block_shape"])
    )
    if pointer_block_shape != descriptor_block_shape:
        raise ValueError(
            "PyNTT NVFP4 descriptor and Shared tile ABIs differ: "
            f"pointer={pointer_key}, descriptor={pointer_block_shape}, "
            f"shared={descriptor_block_shape}."
        )

    item_size = TMA_DTYPE_ITEM_SIZES[expected_dtype]
    owner_count = _product_int([int(value) for value in hierarchy])
    owner_entries: list[list[dict[str, Any]]] = []
    maximum_tiles_per_owner = 0
    maximum_local_k_outer = 0
    for linear_owner in range(owner_count):
        owner = _unflatten_hierarchy_owner(linear_owner, hierarchy)
        n_entry = _tma_descriptor_table_axis_entry(
            pointer,
            0,
            owner,
            tile_extent=block_n,
            context="NVFP4 TMA descriptor N",
        )
        k_entry = _tma_descriptor_table_axis_entry(
            pointer,
            1,
            owner,
            tile_extent=transfer_k_outer,
            context="NVFP4 TMA descriptor K",
        )
        if not n_entry["active"] or not k_entry["active"]:
            owner_entries.append([])
            continue
        local_n_extent = _product_int(list(n_entry["descriptor_shape"]))
        local_k_outer = _product_int(list(k_entry["descriptor_shape"]))
        if local_k_outer > expected_local_k_outer:
            raise ValueError(
                "PyNTT NVFP4 TMA local K exceeds the selected local projection K: "
                f"owner={owner}, descriptor={local_k_outer}, "
                f"selected={expected_local_k_outer}."
            )
        maximum_local_k_outer = max(maximum_local_k_outer, local_k_outer)
        descriptor_tile_count = (local_n_extent + block_n - 1) // block_n
        maximum_tiles_per_owner = max(maximum_tiles_per_owner, descriptor_tile_count)
        base_scalar_elements = _tma_owner_backing_base_elements(
            pointer,
            (n_entry, k_entry),
            logical_strides,
            scalar_lanes_per_logical_element=k_atom,
            context="NVFP4 TMA",
        )
        entries_for_owner = []
        for tile_index in range(descriptor_tile_count):
            local_n_offset = tile_index * block_n
            if n_entry["is_block_cyclic"] and n_entry["block_size"] != 1:
                block_size = int(n_entry["block_size"])
                raw_coordinates = (
                    local_n_offset // block_size,
                    local_n_offset % block_size,
                )
            else:
                raw_coordinates = (local_n_offset,)
            if len(raw_coordinates) != len(n_entry["stride_multipliers"]):
                raise ValueError(
                    "PyNTT NVFP4 TMA N coordinate/stride ranks differ: "
                    f"{raw_coordinates}/{n_entry['stride_multipliers']}."
                )
            physical_n_offset = sum(
                coordinate * int(multiplier)
                for coordinate, multiplier in zip(
                    raw_coordinates, n_entry["stride_multipliers"]
                )
            )
            tile_n_extent = min(block_n, local_n_extent - local_n_offset)
            if n_entry["is_block_cyclic"] and n_entry["block_size"] != 1:
                block_size = int(n_entry["block_size"])
                if block_n <= block_size:
                    raw_descriptor_n_shape = (1, tile_n_extent)
                elif tile_n_extent % block_size == 0:
                    raw_descriptor_n_shape = (
                        tile_n_extent // block_size,
                        block_size,
                    )
                else:
                    raise ValueError(
                        "PyNTT NVFP4 TMA cannot encode a partial block-cyclic "
                        f"N block: extent={tile_n_extent}, block_size={block_size}."
                    )
            else:
                raw_descriptor_n_shape = (tile_n_extent,)
            descriptor_n_shape = tuple(
                raw_descriptor_n_shape[dimension]
                for dimension in n_plan["retained_dimensions"]
            )
            descriptor_row_stride = logical_strides[0] * k_atom
            descriptor_n_strides = tuple(
                int(n_entry["stride_multipliers"][dimension])
                * descriptor_row_stride
                for dimension in n_plan["retained_dimensions"]
            )
            descriptor_k_shape, descriptor_k_strides = (
                _tma_packed_atom_entry(k_entry, k_payload_plan)
            )
            descriptor_shape = descriptor_n_shape + descriptor_k_shape
            entries_for_owner.append(
                {
                    "offset_bytes": int(backing["offset_bytes"])
                    + linear_owner * int(backing["owner_stride_bytes"])
                    + (
                        base_scalar_elements
                        + physical_n_offset * logical_strides[0] * k_atom
                    )
                    * item_size,
                    "shape": descriptor_shape,
                    "strides": descriptor_n_strides
                    + descriptor_k_strides,
                    "source_shape_axes": tuple(() for _ in descriptor_shape),
                }
            )
        owner_entries.append(entries_for_owner)

    if maximum_local_k_outer != expected_local_k_outer:
        raise ValueError(
            "PyNTT NVFP4 descriptor/local projection K extents differ: "
            f"{maximum_local_k_outer}/{expected_local_k_outer}."
        )
    expected_tiles = context["num_n_tiles"]
    if maximum_tiles_per_owner != expected_tiles:
        raise ValueError(
            "PyNTT NVFP4 descriptor/output tile counts differ: "
            f"{maximum_tiles_per_owner}/{expected_tiles}."
        )
    first_entry = next(
        entry for entries_for_owner in owner_entries for entry in entries_for_owner
    )
    padding_entry = {
        **first_entry,
        "shape": tuple(1 for _ in n_plan["block_shape"])
        + first_entry["shape"][len(n_plan["block_shape"]):],
    }
    entries = tuple(
        entry
        for entries_for_owner in owner_entries
        for entry in (
            entries_for_owner
            + [padding_entry] * (expected_tiles - len(entries_for_owner))
        )
    )
    return {
        "kind": "table",
        "name": backing["name"],
        "source": backing["source"],
        "dtype": expected_dtype,
        "block_shape": descriptor_block_shape,
        "padding": "zero",
        "swizzle_mode": _nv_tma_swizzle_mode(descriptor_block_shape, expected_dtype),
        "entry_size_bytes": TENSOR_MAP_ENTRY_BYTES,
        "entries": entries,
    }


def _paged_attention_merge_matmul_host_descriptor_spec(
    model: dict[str, Any],
    backing: dict[str, Any],
) -> dict[str, Any]:
    matmul = model.get("Matmul")
    if not isinstance(matmul, dict):
        raise ValueError(
            "PyNTT fused paged-attention merge/matmul is missing Matmul metadata."
        )
    microkernel = _microkernel_context(
        matmul,
        "triton.paged_attention_merge_matmul",
        "simt_fma_smem_pipeline",
    )
    return _k_major_gemv_host_descriptor_spec(
        matmul,
        backing,
        block_n=microkernel["parameters"]["block_n"],
        block_k=microkernel["parameters"]["block_k"],
        pointer=matmul["Rhs"],
    )


def _packed_matmul_sampling_partial_host_descriptor_spec(
    model: dict[str, Any],
    backing: dict[str, Any],
) -> dict[str, Any]:
    matmul = model.get("Matmul")
    if not isinstance(matmul, dict):
        raise ValueError("PyNTT packed matmul sampling is missing Matmul metadata.")
    microkernel = _microkernel_context(
        matmul,
        "triton.matmul_sampling_partial",
        "simt_fma_smem_pipeline",
        required_workspace_names=_packed_matmul_sampling_partial_workspace_names(),
    )
    return _k_major_gemv_host_descriptor_spec(
        matmul,
        backing,
        block_n=microkernel["parameters"]["block_n"],
        block_k=microkernel["parameters"]["block_k"],
        pointer=matmul["Rhs"],
        expected_vector_lane_shape=_packed_gemv_vector_lane_shape(
            microkernel["variant"]
        ),
    )


def _packed_qkv_gemv_host_descriptor_spec(
    model: dict[str, Any],
    backing: dict[str, Any],
) -> dict[str, Any]:
    microkernel = _microkernel_context(
        model,
        "triton.qkv_parallel_linear",
        str(model["MicroKernel"]["Variant"]),
    )
    if microkernel["variant"] not in (
        "simt_fma_smem_pipeline",
        "simt_fp8_fma_smem_pipeline",
        "mma_smem_pipeline",
    ):
        raise ValueError(
            "PyNTT packed QKV host descriptor requires a shared-memory "
            f"pipeline variant, got {microkernel['variant']!r}."
        )
    return _k_major_gemv_host_descriptor_spec(
        model,
        backing,
        block_n=microkernel["parameters"]["block_n"],
        block_k=microkernel["parameters"]["block_k"],
        pointer=model["Weight"],
        expected_vector_lane_shape=_packed_gemv_vector_lane_shape(
            microkernel["variant"]
        ),
        transpose_kn=True,
    )


def _packed_matmul_glu_gemv_host_descriptor_spec(
    model: dict[str, Any],
    backing: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    microkernel = _microkernel_context(
        model, "triton.matmul_glu", str(model["MicroKernel"]["Variant"])
    )
    return _k_major_gemv_host_descriptor_spec(
        model,
        backing,
        block_n=microkernel["parameters"]["block_n"],
        block_k=microkernel["parameters"]["block_k"],
        pointer=model[f"{prefix}Weight"],
        expected_vector_lane_shape=_packed_gemv_vector_lane_shape(
            microkernel["variant"]
        ),
    )


def _packed_qkv_fixed_projection_ns(model: dict[str, Any]) -> dict[str, int]:
    capacities = model.get("ProjectionNCapacities")
    if not isinstance(capacities, list) or len(capacities) != 3:
        raise ValueError(
            "PyNTT fused packed QKV requires three projection N capacities."
        )
    projection_ns = {
        prefix: _require_int(
            capacities[index],
            f"ProjectionNCapacities[{index}]",
            minimum=1,
        )
        for index, prefix in enumerate(("Q", "K", "V"))
    }
    return projection_ns


def _packed_gemv_vector_lane_shape(variant: str) -> tuple[int, int, int]:
    shapes = {
        "simt_fma_smem_pipeline": (8, 2, 8),
        "mma_smem_pipeline": (8, 2, 8),
        "simt_fp8_fma_smem_pipeline": (8, 2, 16),
        "simt_block_fp8_fma_smem_pipeline": (8, 2, 16),
    }
    try:
        return shapes[variant]
    except KeyError as error:
        raise ValueError(
            "PyNTT packed GEMV host descriptor does not support microkernel "
            f"variant {variant!r}."
        ) from error


def _k_major_gemv_host_descriptor_spec(
    model: dict[str, Any],
    backing: dict[str, Any],
    *,
    block_n: int,
    block_k: int,
    pointer: dict[str, Any],
    expected_vector_lane_shape: tuple[int, int, int],
    transpose_kn: bool = False,
) -> dict[str, Any]:
    logical_shape = tuple(int(value) for value in backing["logical_shape"])
    logical_strides = _normalize_singleton_strides(
        logical_shape,
        tuple(int(value) for value in backing["logical_strides"]),
    )
    vector_lane_shape = tuple(int(value) for value in backing["vector_lane_shape"])
    if len(logical_shape) != 2 or len(logical_strides) != 2:
        raise ValueError(
            "PyNTT packed GEMV host descriptor requires a rank-2 logical RHS "
            f"backing, got ranks {len(logical_shape)}/{len(logical_strides)}."
        )
    if vector_lane_shape != expected_vector_lane_shape:
        raise ValueError(
            "PyNTT packed GEMV host descriptor requires vector lane shape "
            f"{expected_vector_lane_shape}, got {vector_lane_shape}."
        )

    n_lane, k_pack, k_lane = vector_lane_shape
    k_atom = k_pack * k_lane
    if block_n % n_lane != 0 or block_k % k_atom != 0:
        raise ValueError(
            "PyNTT packed GEMV host descriptor block shape is incompatible "
            f"with vector lanes: block_n={block_n}, block_k={block_k}, "
            f"lanes={vector_lane_shape}."
        )
    scalar_lanes_per_logical_element = n_lane * k_pack * k_lane
    contiguous_extent = n_lane * k_lane
    contiguous_rebase_extent = int(
        backing["contiguous_rebase_extent_elements"]
    )
    descriptor_contiguous_extent = contiguous_extent + contiguous_rebase_extent
    if descriptor_contiguous_extent > 2**31 - 1:
        raise ValueError(
            "PyNTT packed GEMV host descriptor contiguous rebase domain "
            f"exceeds signed int32 coordinates: {descriptor_contiguous_extent}."
        )

    packed_k_outer = block_k // k_atom
    packed_n_outer = block_n // n_lane
    k_plan = _tma_canonical_axis_plan(
        pointer,
        0,
        tile_extent=packed_k_outer,
        context="packed GEMV descriptor K",
    )
    n_plan = _tma_canonical_axis_plan(
        pointer,
        1,
        tile_extent=packed_n_outer,
        context="packed GEMV descriptor N",
    )

    hierarchy = pointer.get("Hierarchy")
    if not isinstance(hierarchy, list) or not hierarchy:
        raise ValueError("PyNTT packed GEMV descriptor table requires a hierarchy")
    pointer_global_shape = pointer.get("GlobalShape")
    if not isinstance(pointer_global_shape, list) or len(pointer_global_shape) != 2:
        raise ValueError(
            "PyNTT packed GEMV descriptor table requires a rank-2 global pointer"
        )
    fixed_pointer_shape = tuple(
        _require_fixed_positive_dim(
            extent, f"PyNTT packed GEMV pointer global axis {axis}"
        )
        for axis, extent in enumerate(pointer_global_shape)
    )
    if fixed_pointer_shape != logical_shape:
        raise ValueError(
            "PyNTT packed GEMV descriptor backing/pointer shapes differ: "
            f"{logical_shape}/{fixed_pointer_shape}."
        )

    ordered_plans = (n_plan, k_plan) if transpose_kn else (k_plan, n_plan)
    descriptor_block_shape = tuple(
        value for plan in ordered_plans for value in plan["block_shape"]
    ) + (k_pack, contiguous_extent)
    if len(descriptor_block_shape) > 5:
        raise ValueError(
            "PyNTT packed GEMV TMA descriptor exceeds the hardware rank-5 "
            f"limit: block_shape={descriptor_block_shape}."
        )

    item_size = TMA_DTYPE_ITEM_SIZES.get(backing["scalar_dtype"])
    if item_size is None:
        raise ValueError(
            "PyNTT packed GEMV descriptor table does not support scalar dtype "
            f"{backing['scalar_dtype']!r}."
        )
    owner_count = _product_int([int(value) for value in hierarchy])
    entries = []
    for linear_owner in range(owner_count):
        owner = _unflatten_hierarchy_owner(linear_owner, hierarchy)
        k_entry = _tma_descriptor_table_axis_entry(
            pointer,
            0,
            owner,
            tile_extent=packed_k_outer,
            context="packed GEMV descriptor K",
        )
        n_entry = _tma_descriptor_table_axis_entry(
            pointer,
            1,
            owner,
            tile_extent=packed_n_outer,
            context="packed GEMV descriptor N",
        )

        def axis_group(
            axis: int, entry: dict[str, Any], plan: dict[str, Any]
        ) -> tuple[tuple[int, ...], tuple[int, ...]]:
            retained_dimensions = plan["retained_dimensions"]
            return (
                tuple(
                    int(entry["descriptor_shape"][dimension])
                    for dimension in retained_dimensions
                ),
                tuple(
                    logical_strides[axis]
                    * scalar_lanes_per_logical_element
                    * int(entry["stride_multipliers"][dimension])
                    for dimension in retained_dimensions
                ),
            )

        k_group = axis_group(0, k_entry, k_plan)
        n_group = axis_group(1, n_entry, n_plan)
        ordered_groups = (n_group, k_group) if transpose_kn else (k_group, n_group)
        descriptor_shape = tuple(
            value for group in ordered_groups for value in group[0]
        ) + (k_pack, descriptor_contiguous_extent)
        descriptor_strides = tuple(
            value for group in ordered_groups for value in group[1]
        ) + (contiguous_extent, 1)
        base_scalar_elements = _tma_owner_backing_base_elements(
            pointer,
            (k_entry, n_entry),
            logical_strides,
            scalar_lanes_per_logical_element=scalar_lanes_per_logical_element,
            context="packed GEMV",
        )
        entries.append(
            {
                "offset_bytes": int(backing["offset_bytes"])
                + linear_owner * int(backing["owner_stride_bytes"])
                + base_scalar_elements * item_size,
                "shape": descriptor_shape,
                "strides": descriptor_strides,
                "source_shape_axes": tuple(() for _ in descriptor_shape),
            }
        )

    return {
        "kind": "table",
        "name": backing["name"],
        "source": backing["source"],
        "dtype": backing["scalar_dtype"],
        "block_shape": descriptor_block_shape,
        "padding": "zero",
        "swizzle_mode": _nv_tma_swizzle_mode(
            descriptor_block_shape, backing["scalar_dtype"]
        ),
        "entry_size_bytes": TENSOR_MAP_ENTRY_BYTES,
        "entries": tuple(entries),
    }


def _render_kernel(
    kernel: dict[str, Any], mesh_axes: tuple[dict[str, Any], ...]
) -> str:
    env = _make_env()
    metadata = kernel["metadata"]
    kernel_attrs = _attrs(metadata)
    backend_config = _kernel_backend_config(kernel)
    num_warps = backend_config["num_warps"]
    target_worker_width = _require_int(
        kernel_attrs.get("target_worker_width"),
        f"PyNTT kernel {metadata['name']} attrs.target_worker_width",
        minimum=1,
    )
    shared_allocation_bytes = _require_int(
        metadata.get("launch", {})
        .get("meta", {})
        .get("shared_data_pool_bytes", 0),
        f"PyNTT kernel {metadata['name']} launch.meta.shared_data_pool_bytes",
        minimum=0,
    )
    shared_allocation_alignment_bytes = _require_int(
        metadata.get("launch", {})
        .get("meta", {})
        .get("shared_data_pool_alignment_bytes"),
        f"PyNTT kernel {metadata['name']} "
        "launch.meta.shared_data_pool_alignment_bytes",
        minimum=1,
    )
    if not _is_positive_power_of_two(shared_allocation_alignment_bytes):
        raise ValueError(
            f"PyNTT kernel {metadata['name']} shared allocation alignment must "
            f"be a positive power of two, got "
            f"{shared_allocation_alignment_bytes}."
        )
    shared_memory_capacity_bytes = _require_int(
        kernel_attrs.get("shared_memory_capacity_bytes", 0),
        f"PyNTT kernel {metadata['name']} attrs.shared_memory_capacity_bytes",
        minimum=0,
    )
    if (
        shared_memory_capacity_bytes > 0
        and shared_allocation_bytes > shared_memory_capacity_bytes
    ):
        raise ValueError(
            f"PyNTT kernel {metadata['name']} requires "
            f"{shared_allocation_bytes} shared-memory bytes, exceeding target "
            f"capacity {shared_memory_capacity_bytes}."
        )
    parameters = _kernel_parameters(metadata)
    raw_device_functions = tuple(kernel["device_functions"])
    device_functions = _prepare_device_functions(raw_device_functions, parameters)
    device_functions_by_name = {
        device_function["name"]: device_function for device_function in device_functions
    }
    helper_sources = _render_helper_sources(
        env,
        kernel.get("helpers", ()),
        noinline=False,
        num_warps=num_warps,
        target_worker_width=target_worker_width,
        producer_warps=backend_config["producer_warps"],
        producer_registers=backend_config["producer_registers"],
        register_granularity=backend_config["register_granularity"],
        registers_per_thread_limit=backend_config["registers_per_thread_limit"],
        kernel_config=backend_config,
        device_functions_by_name=device_functions_by_name,
        mesh_axes=mesh_axes,
    )
    device_function_sources = [
        _render_device_function(
            env,
            device_function,
            device_functions_by_name,
            num_warps,
            target_worker_width,
            backend_config["producer_warps"],
            backend_config["producer_registers"],
            backend_config["register_granularity"],
            backend_config["registers_per_thread_limit"],
            backend_config,
            mesh_axes,
        )
        for device_function in device_functions
    ]
    body_source = kernel.get("body_source", "")
    body_source = _replace_device_function_calls(
        body_source,
        device_functions_by_name,
    )
    top_kernel = (
        env.get_template("triton/top_kernel.py.jinja")
        .render(
            name=metadata["name"],
            parameters=", ".join(parameters),
            do_not_specialize=repr(
                tuple(
                    dict.fromkeys(
                        (
                            *_runtime_scalar_input_args(metadata),
                            *_abi_view_stride_args(metadata),
                            *_runtime_shape_args(metadata),
                            "numel",
                        )
                    )
                )
            ),
            body_source=body_source.rstrip(),
            mesh_axes=mesh_axes,
            shared_allocation_bytes=shared_allocation_bytes,
            shared_allocation_alignment_bytes=shared_allocation_alignment_bytes,
            noinline=False,
        )
        .strip()
    )
    parts = [source for source in helper_sources if source]
    parts.extend(source for source in device_function_sources if source)
    parts.append(top_kernel)
    return "\n\n".join(parts)


def _render_device_function(
    env: Environment,
    device_function: dict[str, Any],
    device_functions_by_name: dict[str, dict[str, Any]],
    num_warps: int,
    target_worker_width: int,
    producer_warps: int,
    producer_registers: int,
    register_granularity: int,
    registers_per_thread_limit: int,
    kernel_config: dict[str, Any],
    mesh_axes: tuple[dict[str, Any], ...],
) -> str:
    helper_sources = _render_helper_sources(
        env,
        device_function.get("helpers", ()),
        noinline=False,
        num_warps=num_warps,
        target_worker_width=target_worker_width,
        producer_warps=producer_warps,
        producer_registers=producer_registers,
        register_granularity=register_granularity,
        registers_per_thread_limit=registers_per_thread_limit,
        kernel_config=kernel_config,
        device_functions_by_name=device_functions_by_name,
        mesh_axes=mesh_axes,
    )
    parts = [source for source in helper_sources if source]
    device_parameters = tuple(device_function["direct_parameters"]) + tuple(
        device_function["direct_extra_parameter_declarations"]
    )
    dynamic_device_parameters = tuple(
        name
        for declaration, name in zip(
            device_parameters,
            _parameter_call_arguments(device_parameters),
        )
        if ": tl.constexpr" not in declaration
    )
    body_source = _replace_device_function_calls(
        device_function["body_source"],
        device_functions_by_name,
    )
    parts.append(
        env.get_template("triton/top_kernel.py.jinja")
        .render(
            name=device_function["name"],
            parameters=", ".join(device_parameters),
            do_not_specialize=repr(dynamic_device_parameters),
            body_source=body_source.rstrip(),
            mesh_axes=mesh_axes,
            shared_allocation_bytes=0,
            noinline=device_function["noinline"],
        )
        .strip()
    )
    return "\n\n".join(parts)


def _prepare_device_functions(
    device_functions: tuple[dict[str, Any], ...],
    parameters: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    parameter_names = _parameter_call_arguments(parameters)
    parameter_by_name = dict(zip(parameter_names, parameters))
    prepared_functions = []
    for device_function in device_functions:
        prepared = dict(device_function)
        extra_parameter_declarations = tuple(device_function["extra_parameters"])
        extra_parameters = _parameter_call_arguments(extra_parameter_declarations)
        prepared["direct_extra_parameters"] = extra_parameters
        prepared["direct_extra_parameter_declarations"] = (
            extra_parameter_declarations
        )
        body_source = device_function.get("body_source", "").rstrip() or "pass"
        prepared["body_source"] = body_source
        prepared["liveness_source"] = body_source
        prepared_functions.append(prepared)

    functions_by_name = {
        device_function["name"]: device_function
        for device_function in prepared_functions
    }
    required_parameters = {
        name: _referenced_parameter_names(
            device_function["liveness_source"], parameter_names
        )
        for name, device_function in functions_by_name.items()
    }

    # Keep only canonical top-kernel parameters used by this private function
    # or a transitive callee. PrimFunc descriptors are explicit parameters.
    changed = True
    while changed:
        changed = False
        for name, device_function in functions_by_name.items():
            for match in DEVICE_CALL_NAME_RE.finditer(
                device_function["liveness_source"]
            ):
                callee_name = match.group("name")
                callee = functions_by_name.get(callee_name)
                if callee is None:
                    raise RuntimeError(
                        f"PyNTT device function {name} calls unknown device function "
                        f"{callee_name}."
                    )
                overrides = dict(callee["parameter_overrides"])
                for parameter in required_parameters[callee_name]:
                    expression = overrides.get(parameter, parameter)
                    for dependency in _referenced_parameter_names(
                        expression, parameter_names
                    ):
                        if dependency not in required_parameters[name]:
                            required_parameters[name].add(dependency)
                            changed = True

    for device_function in prepared_functions:
        device_function["direct_parameters"] = tuple(
            parameter_by_name[name]
            for name in parameter_names
            if name in required_parameters[device_function["name"]]
        )
    return tuple(prepared_functions)


def _referenced_parameter_names(
    source: str, parameter_names: tuple[str, ...]
) -> set[str]:
    if not source.strip():
        return set()
    try:
        tree = ast.parse(source)
    except SyntaxError as ex:
        raise RuntimeError(
            "Invalid PyNTT device-function body while computing ABI liveness."
        ) from ex
    candidates = set(parameter_names)
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id in candidates
    }


def _render_helper_sources(
    env: Environment,
    helpers: Any,
    *,
    noinline: bool = False,
    num_warps: int | None = None,
    target_worker_width: int | None = None,
    producer_warps: int | None = None,
    producer_registers: int | None = None,
    register_granularity: int | None = None,
    registers_per_thread_limit: int | None = None,
    kernel_config: dict[str, Any] | None = None,
    device_functions_by_name: dict[str, dict[str, Any]] | None = None,
    mesh_axes: tuple[dict[str, Any], ...] = (),
) -> list[str]:
    helper_sources = []
    for helper in helpers:
        model = _prepare_helper_model(
            helper["model"],
            noinline=noinline,
            num_warps=num_warps,
            target_worker_width=target_worker_width,
            producer_warps=producer_warps,
            producer_registers=producer_registers,
            register_granularity=register_granularity,
            registers_per_thread_limit=registers_per_thread_limit,
            kernel_config=kernel_config,
            device_functions_by_name=device_functions_by_name,
            mesh_axes=mesh_axes,
        )
        model["Arguments"] = tuple(helper["arguments"])
        model["WorkspaceArguments"] = tuple(helper["workspace_arguments"])
        helper_sources.append(
            env.get_template(helper["template"]).render(model=model).strip()
        )
    return helper_sources


def _prepare_helper_model(
    raw_model: dict[str, Any],
    *,
    noinline: bool,
    num_warps: int | None,
    target_worker_width: int | None,
    producer_warps: int | None,
    producer_registers: int | None,
    register_granularity: int | None,
    registers_per_thread_limit: int | None,
    kernel_config: dict[str, Any] | None,
    device_functions_by_name: dict[str, dict[str, Any]] | None,
    mesh_axes: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    model = dict(raw_model)
    model["NoInline"] = bool(noinline)
    model["MeshAxes"] = mesh_axes
    if num_warps is not None:
        model["NumWarps"] = num_warps
    if target_worker_width is not None:
        model["TargetWorkerWidth"] = target_worker_width
    if producer_warps is not None:
        model["ProducerWarps"] = producer_warps
    if producer_registers is not None:
        model["ProducerRegisters"] = producer_registers
    if register_granularity is not None:
        model["RegisterGranularity"] = register_granularity
    if registers_per_thread_limit is not None:
        model["RegistersPerThreadLimit"] = registers_per_thread_limit
    if kernel_config is not None:
        model["KernelConfig"] = kernel_config

    if "Stages" in model:
        stages = []
        for raw_stage in model["Stages"]:
            stage = dict(raw_stage)
            stage["Model"] = _prepare_helper_model(
                stage["Model"],
                noinline=noinline,
                num_warps=num_warps,
                target_worker_width=target_worker_width,
                producer_warps=producer_warps,
                producer_registers=producer_registers,
                register_granularity=register_granularity,
                registers_per_thread_limit=registers_per_thread_limit,
                kernel_config=kernel_config,
                device_functions_by_name=device_functions_by_name,
                mesh_axes=mesh_axes,
            )
            stages.append(stage)
        model["Stages"] = tuple(stages)
    if device_functions_by_name is not None:
        for body_key in ("ConsumerBodySource", "ProducerBodySource"):
            body_source = model.get(body_key)
            if isinstance(body_source, str):
                model[body_key] = _replace_device_function_calls(
                    body_source,
                    device_functions_by_name,
                )
    return model


def _parameter_call_arguments(parameters: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(parameter.split(":", 1)[0].strip() for parameter in parameters)


def _split_expression_arguments(source: str) -> tuple[str, ...]:
    if not source.strip():
        return ()
    wrapped = f"_pyntt_call({source})"
    try:
        expression = ast.parse(wrapped, mode="eval").body
    except SyntaxError as ex:
        raise RuntimeError(f"Invalid PyNTT device-function arguments: {source}") from ex
    if not isinstance(expression, ast.Call):
        raise RuntimeError(f"Invalid PyNTT device-function arguments: {source}")
    return tuple(
        ast.get_source_segment(wrapped, argument) or ast.unparse(argument)
        for argument in expression.args
    )


def _bind_device_function_extra_arguments(
    device_function: dict[str, Any],
    explicit_extra_arguments: tuple[str, ...],
) -> dict[str, str]:
    extra_parameters = _parameter_call_arguments(
        tuple(device_function["extra_parameters"])
    )
    if explicit_extra_arguments:
        if len(explicit_extra_arguments) != len(extra_parameters):
            raise RuntimeError(
                f"PyNTT call to {device_function['name']} passes "
                f"{len(explicit_extra_arguments)} extra arguments, expected "
                f"{len(extra_parameters)}."
            )
        return dict(zip(extra_parameters, explicit_extra_arguments))

    defaults = dict(device_function["extra_parameter_arguments"])
    missing = [parameter for parameter in extra_parameters if parameter not in defaults]
    if missing:
        raise RuntimeError(
            f"PyNTT call to {device_function['name']} is missing extra arguments "
            f"{missing}."
        )
    return {parameter: defaults[parameter] for parameter in extra_parameters}


def _build_device_function_call(
    device_function: dict[str, Any],
    explicit_extra_arguments: tuple[str, ...],
) -> str:
    call_arguments = _build_device_function_arguments(
        device_function, explicit_extra_arguments
    )
    return f"{device_function['name']}({', '.join(call_arguments)})"


def _build_device_function_arguments(
    device_function: dict[str, Any],
    explicit_extra_arguments: tuple[str, ...],
) -> tuple[str, ...]:
    extra_arguments = _bind_device_function_extra_arguments(
        device_function, explicit_extra_arguments
    )

    parameter_overrides = dict(device_function["parameter_overrides"])
    return (
        tuple(
            parameter_overrides.get(argument, argument)
            for argument in _parameter_call_arguments(
                tuple(device_function["direct_parameters"])
            )
        )
        + tuple(
            extra_arguments[parameter]
            for parameter in device_function["direct_extra_parameters"]
        )
    )


def _replace_device_function_calls(
    source: str,
    device_functions: dict[str, dict[str, Any]],
) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        if name not in device_functions:
            raise RuntimeError(
                f"PyNTT kernel references unknown device function {name}."
            )
        indent = match.group("indent")
        extra_arguments = _split_expression_arguments(match.group("args"))
        call_source = _build_device_function_call(
            device_functions[name],
            extra_arguments,
        )

        return "\n".join(
            f"{indent}{line}" if line else line for line in call_source.splitlines()
        )

    return DEVICE_CALL_RE.sub(replace, source)


def _make_env() -> Environment:
    env = Environment(
        loader=PackageLoader("pyntt", "codegen/templates"),
        undefined=StrictUndefined,
        extensions=("jinja2.ext.do",),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.globals.update(
        access_boundary_mask=_access_boundary_mask,
        access_pointer=_access_pointer,
        axes_except=lambda rank, excluded: tuple(
            axis for axis in range(rank) if axis != excluded
        ),
        concat_context=_concat_template_context,
        conv2d_context=_conv2d_template_context,
        dim=_dim,
        elementwise_binary_context=_elementwise_binary_template_context,
        elementwise_cast_context=_elementwise_cast_template_context,
        elementwise_unary_context=_elementwise_unary_template_context,
        elementwise_where_context=_elementwise_where_template_context,
        fixed=_fixed,
        gather_context=_gather_template_context,
        gather_reduce_norm_apply_context=(
            _gather_reduce_norm_apply_template_context
        ),
        gated_delta_net_convolution_context=(
            _gated_delta_net_convolution_template_context
        ),
        gated_delta_net_recurrent_core_context=(
            _gated_delta_net_recurrent_core_template_context
        ),
        gather_reduce_add_norm_context=_gather_reduce_add_norm_template_context,
        helper_argument_names=_helper_argument_names,
        helper_parameters=_helper_parameters,
        is_bool_dtype=_is_bool_dtype,
        is_fixed_one=_is_fixed_one,
        layer_norm_context=_layer_norm_template_context,
        logical_shape=_logical_shape,
        logical_strides=_logical_strides,
        memcopy_context=_memcopy_template_context,
        matmul_glu_context=_matmul_glu_template_context,
        packed_matmul_glu_gemv_pipeline_context=(
            _packed_matmul_glu_gemv_pipeline_template_context
        ),
        packed_fp8_matmul_glu_gemv_pipeline_context=(
            _packed_fp8_matmul_glu_gemv_pipeline_template_context
        ),
        packed_block_fp8_matmul_glu_gemv_pipeline_context=(
            _packed_block_fp8_matmul_glu_gemv_pipeline_template_context
        ),
        matmul_context=_matmul_template_context,
        nvfp4_matmul_context=_nvfp4_matmul_template_context,
        nvfp4_matmul_glu_context=_nvfp4_matmul_glu_template_context,
        multiply_expr=_multiply_expr,
        norm_apply_context=_norm_apply_template_context,
        norm_stats_context=_norm_stats_template_context,
        paged_attention_context=_paged_attention_template_context,
        paged_attention_merge_context=_paged_attention_merge_template_context,
        paged_attention_merge_matmul_context=(
            _paged_attention_merge_matmul_template_context
        ),
        paged_attention_partial_context=_paged_attention_partial_template_context,
        packed_gemv_pipeline_context=_packed_gemv_pipeline_template_context,
        packed_fp8_gemv_pipeline_context=(
            _packed_fp8_gemv_pipeline_template_context
        ),
        packed_block_fp8_gemv_pipeline_context=(
            _packed_block_fp8_gemv_pipeline_template_context
        ),
        packed_block_fp8_mma_gemv_pipeline_context=(
            _packed_block_fp8_mma_gemv_pipeline_template_context
        ),
        packed_matmul_sampling_partial_context=(
            _packed_matmul_sampling_partial_template_context
        ),
        packed_qkv_gemv_pipeline_context=_packed_qkv_gemv_pipeline_template_context,
        packed_fp8_qkv_gemv_pipeline_context=(
            _packed_fp8_qkv_gemv_pipeline_template_context
        ),
        pointer_local_to_global_coordinate=_pointer_local_to_global_coordinate,
        packed_qkv_mma_pipeline_context=_packed_qkv_mma_pipeline_template_context,
        product=_product,
        qkv_rope_with_cache_context=_qkv_rope_with_cache_template_context,
        qkv_parallel_linear_context=_qkv_parallel_linear_template_context,
        reduce_context=_reduce_template_context,
        ptr=_ptr,
        pyrepr=repr,
        reshard_context=_reshard_template_context,
        rope_context=_rope_template_context,
        sampling_context=_sampling_template_context,
        scatter_nd_context=_scatter_nd_template_context,
        select_block_axis=_select_block_axis,
        shape_tuple=_shape_tuple,
        softmax_context=_softmax_template_context,
        sparse_experts_down_pipeline_context=(
            _sparse_experts_down_pipeline_template_context
        ),
        summa_context=_summa_template_context,
        tensor_copy_context=_tensor_copy_template_context,
        tensor_access=_tensor_access,
        contiguous_vector_axis_access=_contiguous_vector_axis_access,
        tensor_region_copy_context=_tensor_region_copy_template_context,
        tma_offset=_tma_offset,
        transpose_context=_transpose_template_context,
        update_paged_attention_kv_cache_context=(
            _update_paged_attention_kv_cache_template_context
        ),
        vector_layout_context=_vector_layout_template_context,
    )
    return env


def _grid_mesh_topology(manifest: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    topologies: set[tuple[tuple[int, str, int, str], ...]] = set()
    for function in manifest.get("functions", ()):
        for kernel in function.get("render_kernels", ()):
            metadata = kernel.get("metadata", {})
            sharding = metadata.get("launch", {}).get("sharding", {})
            hierarchy = tuple(
                _require_int(value, "launch.sharding.hierarchy", minimum=1)
                for value in sharding.get("hierarchy", ())
            )
            names = _require_string(
                sharding.get("placement_axis"),
                "launch.sharding.placement_axis",
                nonempty=True,
            )
            levels = _require_string(
                sharding.get("hierarchy_levels"),
                "launch.sharding.hierarchy_levels",
                nonempty=True,
            ).lower()
            if len(hierarchy) != len(names) or len(hierarchy) != len(levels):
                raise RuntimeError(
                    "PyNTT grid mesh hierarchy, placement-axis names, and hierarchy "
                    f"levels must have equal rank, got {hierarchy}, {names!r}, {levels!r}."
                )
            if any(level != "b" and hierarchy[index] != 1 for index, level in enumerate(levels)):
                raise RuntimeError(
                    "PyNTT launches can only materialize non-trivial physical block "
                    f"axes, got hierarchy {hierarchy} with levels {levels!r}."
                )
            topology = tuple(
                (
                    index,
                    f"block_{names[index]}",
                    hierarchy[index],
                    level,
                )
                for index, level in enumerate(levels)
            )
            physical_axes = tuple(item for item in topology if item[3] == "b")
            if not physical_axes:
                raise RuntimeError("PyNTT launch has no physical block axes.")
            if len({item[1] for item in physical_axes}) != len(physical_axes):
                raise RuntimeError(f"PyNTT grid mesh axis names must be unique, got {topology}.")
            topologies.add(topology)
    if not topologies:
        return (
            {
                "placement_axis": 0,
                "name": "block_b",
                "size": 1,
                "level": "b",
            },
        )
    if len(topologies) != 1:
        raise RuntimeError(
            "PyNTT generated kernels must use one grid mesh "
            f"topology, got {sorted(topologies)}."
        )
    return tuple(
        {
            "placement_axis": axis,
            "name": name,
            "size": size,
            "level": level,
        }
        for axis, name, size, level in next(iter(topologies))
    )


def _grid_barrier_axis_groups(
    manifest: dict[str, Any],
    topology: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    physical_axis_sizes = {
        int(axis["placement_axis"]): int(axis["size"])
        for axis in topology
        if axis["level"] == "b"
    }
    physical_axes = set(physical_axis_sizes)
    axis_groups: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for function in manifest.get("functions", ()):
        for kernel in function.get("render_kernels", ()):
            metadata = kernel.get("metadata", {})
            raw_groups = _attrs(metadata).get("grid_barrier_axis_groups", ())
            if not isinstance(raw_groups, (list, tuple)):
                raise ValueError("attrs.grid_barrier_axis_groups must be an array.")
            for group_index, raw_group in enumerate(raw_groups):
                if not isinstance(raw_group, dict):
                    raise ValueError(
                        f"attrs.grid_barrier_axis_groups[{group_index}] must be an object."
                    )
                axes = tuple(
                    _require_int(
                        axis,
                        f"attrs.grid_barrier_axis_groups[{group_index}].axes",
                        minimum=0,
                    )
                    for axis in raw_group.get("axes", ())
                )
                if not axes:
                    raise ValueError("A grid barrier axis group cannot be empty.")
                if len(set(axes)) != len(axes) or tuple(sorted(axes)) != axes:
                    raise ValueError(
                        f"Grid barrier axis-group axes must be unique and sorted, got {axes}."
                    )
                unknown = set(axes) - physical_axes
                if unknown:
                    raise ValueError(
                        f"Grid barrier axis-group axes {sorted(unknown)} are not block axes in {topology}."
                    )
                raw_shape = raw_group.get("shape", ())
                if not isinstance(raw_shape, (list, tuple)):
                    raise ValueError(
                        f"attrs.grid_barrier_axis_groups[{group_index}].shape must be an array."
                    )
                shape = tuple(
                    _require_int(
                        extent,
                        f"attrs.grid_barrier_axis_groups[{group_index}].shape",
                        minimum=1,
                    )
                    for extent in raw_shape
                )
                if len(shape) != len(axes):
                    raise ValueError(
                        f"Grid barrier axis-group shape {shape} must match axes {axes}."
                    )
                for axis, extent in zip(axes, shape):
                    domain_extent = physical_axis_sizes[axis]
                    if domain_extent % extent != 0:
                        raise ValueError(
                            f"Grid barrier group extent {extent} must divide axis {axis} "
                            f"domain extent {domain_extent}."
                        )
                if set(axes) == physical_axes and all(
                    extent == physical_axis_sizes[axis]
                    for axis, extent in zip(axes, shape)
                ):
                    raise ValueError(
                        f"Full-mesh barrier axes {axes} with shape {shape} must use the canonical full grid barrier."
                    )
                axis_groups.add((axes, shape))

    groups: list[dict[str, Any]] = []
    for axes, shape in sorted(axis_groups):
        key = "_".join(
            f"{axis}x{extent}" for axis, extent in zip(axes, shape)
        )
        axis_names = tuple(
            axis["name"]
            for axis in topology
            if int(axis["placement_axis"]) in axes
        )
        groups.append(
            {
                "key": key,
                "axis_names": axis_names,
                "shape": shape,
            }
        )
    return tuple(groups)


def _attrs(metadata: dict[str, Any]) -> dict[str, Any]:
    return metadata.get("attrs") or metadata.get("Attrs") or {}


def _runtime_shape_args(metadata: dict[str, Any]) -> tuple[str, ...]:
    value = _attrs(metadata).get("runtime_shape_args", ())
    return tuple(value or ())


def _runtime_scalar_input_args(metadata: dict[str, Any]) -> tuple[str, ...]:
    value = _attrs(metadata).get("runtime_scalar_input_args", ())
    return tuple(value or ())


def _abi_view_stride_args(metadata: dict[str, Any]) -> tuple[str, ...]:
    value = _attrs(metadata).get("abi_view_stride_args", ())
    return tuple(value or ())


def _dim(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("TritonExpression", value.get("triton_expression", "0")))
    return str(value)


def _tma_offset(value: Any) -> str:
    expression = str(value).strip()
    return f"(tl.full((), 0, tl.int32) + ({expression})).to(tl.int32)"


def _pointer_type(dtype: str, address_space: Any) -> str:
    address_space = int(address_space)
    if address_space <= 0:
        raise ValueError(f"Pointer address space must be positive, got {address_space}")
    return (
        f"tl.pointer_type({dtype})"
        if address_space == 1
        else f"tl.pointer_type({dtype}, {address_space})"
    )


def _py_dim(value: Any) -> str:
    if isinstance(value, dict):
        return str(
            value.get("PythonExpression", value.get("python_expression", _dim(value)))
        )
    return str(value)


def _fixed(value: Any) -> int | None:
    if not isinstance(value, dict):
        return value if isinstance(value, int) else None
    fixed = value.get("FixedValue", value.get("fixed_value"))
    return None if fixed is None else int(fixed)


def _require_fixed_positive_dim(value: Any, context: str) -> int:
    fixed = _fixed(value)
    if fixed is None or fixed <= 0:
        raise ValueError(f"{context} must be fixed and positive, got {_dim(value)}")
    return fixed


def _min_value(value: Any) -> int | None:
    fixed = _fixed(value)
    if fixed is not None:
        return fixed
    if not isinstance(value, dict):
        return None
    value = value.get("RangeMin", value.get("MinValue", value.get("range_min")))
    return None if value is None else int(value)


def _max_value(value: Any) -> int | None:
    fixed = _fixed(value)
    if fixed is not None:
        return fixed
    if not isinstance(value, dict):
        return None
    value = value.get("RangeMax", value.get("MaxValue", value.get("range_max")))
    return None if value is None else int(value)


def _is_fixed_one(value: Any) -> bool:
    return _fixed(value) == 1


def _one() -> dict[str, Any]:
    return {"PythonExpression": "1", "TritonExpression": "1", "FixedValue": 1}


def _zero() -> dict[str, Any]:
    return {"PythonExpression": "0", "TritonExpression": "0", "FixedValue": 0}


def _multiply_dim(dim: Any, lane: int) -> dict[str, Any]:
    if lane == 1:
        return (
            dict(dim)
            if isinstance(dim, dict)
            else {
                "PythonExpression": str(dim),
                "TritonExpression": str(dim),
                "FixedValue": dim if isinstance(dim, int) else None,
            }
        )
    fixed = _fixed(dim)
    range_min = _min_value(dim)
    range_max = _max_value(dim)
    result: dict[str, Any] = {
        "PythonExpression": f"({_py_dim(dim)} * {lane})",
        "TritonExpression": f"({_dim(dim)} * {lane})",
    }
    if fixed is not None:
        result["FixedValue"] = fixed * lane
    if range_min is not None:
        result["RangeMin"] = range_min * lane
    if range_max is not None:
        result["RangeMax"] = range_max * lane
    return result


def _add_dims(lhs: Any, rhs: Any) -> dict[str, Any]:
    if _fixed(lhs) == 0:
        return dict(rhs)
    if _fixed(rhs) == 0:
        return dict(lhs)
    fixed = (
        _fixed(lhs) + _fixed(rhs)
        if _fixed(lhs) is not None and _fixed(rhs) is not None
        else None
    )
    result = {
        "PythonExpression": f"({_py_dim(lhs)} + {_py_dim(rhs)})",
        "TritonExpression": f"({_dim(lhs)} + {_dim(rhs)})",
    }
    if fixed is not None:
        result["FixedValue"] = fixed
    return result


def _product(values: list[Any]) -> str:
    if not values:
        return "1"
    return " * ".join(f"({_dim(value)})" for value in values)


def _product_int(values: list[int]) -> int:
    product = 1
    for value in values:
        product *= int(value)
    return product


def _multiply_expr(lhs: str, rhs: str | int) -> str:
    rhs = str(rhs)
    return lhs if rhs == "1" else f"({lhs}) * {rhs}"


def _shape_tuple(shape: list[Any]) -> str:
    suffix = "," if len(shape) == 1 else ""
    return f"({', '.join(_dim(dim) for dim in shape)}{suffix})"


def _coordinate_shape(shape: tuple[Any, ...] | list[Any]) -> str:
    if not shape:
        raise ValueError("A PyNTT block access requires a non-empty coordinate shape.")
    return _shape_tuple(list(shape))


def _ptr(model: dict[str, Any], name: str) -> str:
    value = model[name]
    if isinstance(value, dict):
        return value.get("Expression", value.get("expression"))
    return str(value)


def _join_index_terms(terms: list[str]) -> str:
    return "0" if not terms else " + ".join(terms)


def _validate_global_coordinate_axes(
    axes: tuple[int, ...] | list[int], rank: int, context: str
) -> tuple[int, ...]:
    axes = tuple(axes)
    if any(not isinstance(axis, int) or isinstance(axis, bool) for axis in axes):
        raise ValueError(f"{context} global coordinate axes must be integers: {axes}")
    if len(set(axes)) != len(axes):
        raise ValueError(f"{context} global coordinate axes must be unique: {axes}")
    if any(axis < 0 or axis >= rank for axis in axes):
        raise ValueError(
            f"{context} global coordinate axis is outside rank {rank}: {axes}"
        )
    return axes


def _tensor_access(
    tensor_indices: tuple[str, ...] | list[str],
    strides: list[Any],
    lane_indices: tuple[str, ...] | list[str] = (),
    lane_shape: tuple[int, ...] | list[int] = (),
    coordinate_shape: str | None = None,
    *,
    global_coordinate_axes: tuple[int, ...] | list[int] = (),
) -> dict[str, Any]:
    """Build one coordinate-preserving tensor access at render time."""

    tensor_indices = tuple(str(value) for value in tensor_indices)
    lane_indices = tuple(str(value) for value in lane_indices)
    lane_shape = tuple(int(value) for value in lane_shape)
    global_coordinate_axes = _validate_global_coordinate_axes(
        global_coordinate_axes, len(tensor_indices), "PyNTT tensor access"
    )
    if len(tensor_indices) != len(strides):
        raise ValueError(
            "PyNTT tensor access index/stride rank mismatch: "
            f"indices={len(tensor_indices)}, strides={len(strides)}"
        )
    if len(lane_indices) != len(lane_shape):
        raise ValueError(
            "PyNTT tensor access lane rank mismatch: "
            f"indices={len(lane_indices)}, shape={len(lane_shape)}"
        )
    if any(value <= 0 for value in lane_shape):
        raise ValueError(
            f"PyNTT tensor access lane shape must be positive: {lane_shape}"
        )
    tensor_terms = [
        index if _fixed(stride) == 1 else f"({index}) * ({_dim(stride)})"
        for index, stride in zip(tensor_indices, strides)
        if _fixed(stride) != 0 and index != "0"
    ]
    tensor_offset = _join_index_terms(tensor_terms)
    lane_terms: list[str] = []
    lane_stride = 1
    for index, extent in reversed(tuple(zip(lane_indices, lane_shape))):
        if index != "0":
            lane_terms.append(
                index if lane_stride == 1 else f"({index}) * {lane_stride}"
            )
        lane_stride *= extent
    lane_offset = _join_index_terms(list(reversed(lane_terms)))
    scalar_offset = tensor_offset
    if lane_stride != 1:
        scalar_offset = (
            "0" if tensor_offset == "0" else f"({tensor_offset}) * {lane_stride}"
        )
        if lane_offset != "0":
            scalar_offset = (
                lane_offset
                if scalar_offset == "0"
                else f"{scalar_offset} + {lane_offset}"
            )
    raw_scalar_offset = scalar_offset
    if coordinate_shape is not None:
        scalar_offset = f"tl.broadcast_to({scalar_offset}, {coordinate_shape})"

    result = {
        "CoordinateShape": coordinate_shape,
        "RawScalarOffset": raw_scalar_offset,
        "ScalarOffset": scalar_offset,
        "TensorIndices": tensor_indices,
        "TensorStrides": tuple(strides),
        "LaneIndices": lane_indices,
        "LaneShape": lane_shape,
    }
    if global_coordinate_axes:
        result["GlobalCoordinateAxes"] = global_coordinate_axes
    return result


def _contiguous_vector_axis_access(
    tensor_indices: tuple[str, ...] | list[str],
    strides: list[Any],
    *,
    tensor_shape: list[Any] | None = None,
    packed_axis: int,
    logical_index: str,
    lane_count: int,
    coordinate_shape: str | None = None,
    global_coordinate_axes: tuple[int, ...] | list[int] = (),
) -> dict[str, Any]:
    """Build scalar coordinates for a contiguous vectorized tensor axis."""

    tensor_indices = tuple(str(value) for value in tensor_indices)
    if len(tensor_indices) != len(strides):
        raise ValueError(
            "PyNTT contiguous vector access index/stride rank mismatch: "
            f"indices={len(tensor_indices)}, strides={len(strides)}"
        )
    if packed_axis != len(strides) - 1:
        raise ValueError(
            "PyNTT contiguous vector access requires the vectorized tensor "
            f"axis to be innermost, got axis {packed_axis} for rank {len(strides)}."
        )
    packed_stride = _fixed(strides[packed_axis])
    singleton_zero_stride = (
        packed_stride == 0
        and tensor_shape is not None
        and len(tensor_shape) == len(strides)
        and _fixed(tensor_shape[packed_axis]) == 1
    )
    if packed_stride != 1 and not singleton_zero_stride:
        raise ValueError(
            "PyNTT contiguous vector access requires unit stride on the "
            f"vectorized tensor axis, got {_dim(strides[packed_axis])}."
        )
    if lane_count <= 0:
        raise ValueError(
            f"PyNTT contiguous vector access lane count must be positive, got {lane_count}."
        )

    outer_indices = list(tensor_indices)
    outer_indices[packed_axis] = "0"
    outer_access = _tensor_access(
        outer_indices,
        strides,
        ("0",),
        (lane_count,),
    )
    outer_offset = _access_raw_scalar_offset(outer_access)
    raw_scalar_offset = _join_index_terms(
        [term for term in (outer_offset, str(logical_index)) if term != "0"]
    )
    scalar_offset = raw_scalar_offset
    if coordinate_shape is not None:
        scalar_offset = f"tl.broadcast_to({raw_scalar_offset}, {coordinate_shape})"
    result = {
        "CoordinateShape": coordinate_shape,
        "RawScalarOffset": raw_scalar_offset,
        "ScalarOffset": scalar_offset,
        "TensorIndices": tensor_indices,
        "TensorStrides": tuple(strides),
        "ContiguousVectorAxis": packed_axis,
        "LogicalIndex": str(logical_index),
        "LaneCount": lane_count,
    }
    if global_coordinate_axes:
        result["GlobalCoordinateAxes"] = _validate_global_coordinate_axes(
            global_coordinate_axes,
            len(tensor_indices),
            "PyNTT contiguous vector access",
        )
    return result


def _access_scalar_offset(access: Any) -> str:
    if access is None:
        return "0"
    if isinstance(access, dict):
        value = access.get("ScalarOffset", access.get("scalar_offset"))
        if value is None:
            raise ValueError("PyNTT structured access requires ScalarOffset")
        return str(value)
    return str(access)


def _access_raw_scalar_offset(access: Any) -> str:
    if not isinstance(access, dict):
        return _access_scalar_offset(access)
    value = access.get("RawScalarOffset", access.get("raw_scalar_offset"))
    return str(value) if value is not None else _access_scalar_offset(access)


def _with_access_boundary_mask(
    access: dict[str, Any], boundary_mask: str
) -> dict[str, Any]:
    if not boundary_mask:
        raise ValueError("PyNTT tensor access boundary mask cannot be empty")
    result = dict(access)
    result["BoundaryMask"] = boundary_mask
    return result


def _with_major_boundary_mask(
    access: dict[str, Any],
    shape: list[Any],
    strides: list[Any],
    major_axis: int,
) -> dict[str, Any]:
    if len(shape) != len(strides):
        raise ValueError(
            "PyNTT access boundary shape/stride rank mismatch: "
            f"shape={len(shape)}, strides={len(strides)}"
        )
    tensor_indices = access.get("TensorIndices", ())
    varies_on_major = (
        0 <= major_axis < len(shape)
        and major_axis < len(tensor_indices)
        and tensor_indices[major_axis] != "0"
        and _fixed(strides[major_axis]) != 0
    )
    return _with_access_boundary_mask(access, "mask" if varies_on_major else "True")


def _access_boundary_mask(access: Any, fallback: str = "mask") -> str:
    if isinstance(access, dict):
        value = access.get("BoundaryMask", access.get("boundary_mask"))
        if value is not None:
            return str(value)
    return fallback


def _add_coordinate(base: Any, index: str) -> str:
    if _fixed(base) == 0:
        return index
    if index == "0":
        return _dim(base)
    return f"({_dim(base)}) + ({index})"


def _shard_axis_stages(axis_mapping: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(axis_mapping, dict):
        raise ValueError(f"PyNTT shard-axis mapping must be an object, got {axis_mapping!r}")
    stages = axis_mapping.get("Stages")
    if not isinstance(stages, list):
        raise ValueError("PyNTT shard-axis mapping requires a Stages array")
    return tuple(stages)


def _shard_axis_hierarchy_axes(axis_mapping: Any) -> tuple[int, ...]:
    result: list[int] = []
    for stage in _shard_axis_stages(axis_mapping):
        axes = stage.get("HierarchyAxes")
        if not isinstance(axes, list) or not axes:
            raise ValueError("PyNTT split stage requires non-empty HierarchyAxes")
        result.extend(int(axis) for axis in axes)
    if len(set(result)) != len(result):
        raise ValueError(
            f"PyNTT shard-axis mapping reuses hierarchy axes: {result}"
        )
    return tuple(result)


def _shard_axes_hierarchy_axes(shard_axes: Any) -> tuple[int, ...]:
    if not isinstance(shard_axes, list):
        raise ValueError("PyNTT tensor shard mapping must be an array")
    result = tuple(
        axis
        for axis_mapping in shard_axes
        for axis in _shard_axis_hierarchy_axes(axis_mapping)
    )
    if len(set(result)) != len(result):
        raise ValueError(
            f"PyNTT tensor shard mapping reuses hierarchy axes: {result}"
        )
    return result


def _stage_linear_expression(
    stage: dict[str, Any],
    hierarchy: list[int],
    coord_prefix: str = "shard_coord",
) -> str:
    axes = stage.get("HierarchyAxes")
    if not isinstance(axes, list) or not axes:
        raise ValueError("PyNTT split stage requires non-empty HierarchyAxes")
    if len(set(axes)) != len(axes):
        raise ValueError(f"PyNTT split stage contains duplicate hierarchy axes: {axes}")
    for axis in axes:
        if not isinstance(axis, int) or axis < 0 or axis >= len(hierarchy):
            raise ValueError(
                f"PyNTT split stage axis {axis!r} is outside hierarchy rank {len(hierarchy)}"
            )
    return _split_linear_expression(axes, hierarchy, coord_prefix)


def _stage_shard_count(stage: dict[str, Any], hierarchy: list[int]) -> int:
    axes = stage.get("HierarchyAxes")
    if not isinstance(axes, list) or not axes:
        raise ValueError("PyNTT split stage requires non-empty HierarchyAxes")
    return _split_divisor(axes, hierarchy)


def _local_to_global_coordinate(
    local_coordinate: str,
    global_extent: Any,
    axis_mapping: Any,
    hierarchy: list[int],
    coord_prefix: str = "shard_coord",
    *,
    local_extent: int | None = None,
) -> str:
    """Compose ordered SplitStage mappings from a dense local coordinate."""

    if local_extent is not None and local_extent <= 0:
        raise ValueError(
            f"PyNTT local coordinate extent must be positive, got {local_extent}"
        )
    stages = _shard_axis_stages(axis_mapping)
    if not stages:
        return str(local_coordinate)
    parent_extent = _dim(global_extent)
    stage_infos: list[dict[str, Any]] = []
    for stage in stages:
        distribution = stage.get("Distribution")
        shard_count = _stage_shard_count(stage, hierarchy)
        shard_index = _stage_linear_expression(stage, hierarchy, coord_prefix)
        if distribution == "Contiguous":
            granularity = stage.get("Granularity")
            capacity = (
                _dim(granularity)
                if granularity is not None
                else f"tl.cdiv(({parent_extent}), {shard_count})"
            )
            active_extent = (
                f"tl.maximum(0, tl.minimum(({capacity}), "
                f"({parent_extent}) - ({shard_index}) * ({capacity})))"
            )
            stage_infos.append(
                {
                    "distribution": distribution,
                    "capacity": capacity,
                    "shard_index": shard_index,
                }
            )
        elif distribution == "BlockCyclic":
            block_size = stage.get("BlockSize")
            if not isinstance(block_size, int) or block_size <= 0:
                raise ValueError(
                    f"PyNTT block-cyclic split requires a positive BlockSize, got {block_size!r}"
                )
            period = shard_count * block_size
            active_extent = (
                f"((({parent_extent}) // {period}) * {block_size} + "
                f"tl.maximum(0, tl.minimum({block_size}, "
                f"(({parent_extent}) % {period}) - ({shard_index}) * {block_size})))"
            )
            stage_infos.append(
                {
                    "distribution": distribution,
                    "block_size": block_size,
                    "period": period,
                    "shard_index": shard_index,
                }
            )
        else:
            raise ValueError(
                f"Unsupported PyNTT split-stage distribution {distribution!r}"
            )
        parent_extent = active_extent

    coordinate = str(local_coordinate)
    coordinate_extent = local_extent
    for stage_info in reversed(stage_infos):
        shard_index = stage_info["shard_index"]
        if stage_info["distribution"] == "BlockCyclic":
            block_size = stage_info["block_size"]
            period = stage_info["period"]
            if coordinate_extent is not None and coordinate_extent <= block_size:
                coordinate = (
                    f"(({shard_index}) * {block_size} + ({coordinate}))"
                )
            else:
                coordinate = (
                    f"((({coordinate}) // {block_size}) * {period} + "
                    f"({shard_index}) * {block_size} + (({coordinate}) % {block_size}))"
                )
        else:
            capacity = stage_info["capacity"]
            coordinate = f"(({shard_index}) * ({capacity}) + ({coordinate}))"
        # The next outer stage sees this stage's parent domain. Its exact local
        # bound is not represented by the flat coordinate expression, so only
        # the innermost stage may consume the caller-provided bound.
        coordinate_extent = None
    return coordinate


def _dimensions_equivalent(lhs: Any, rhs: Any) -> bool:
    lhs_fixed = _fixed(lhs)
    rhs_fixed = _fixed(rhs)
    if lhs_fixed is not None or rhs_fixed is not None:
        return lhs_fixed is not None and lhs_fixed == rhs_fixed
    return _dim(lhs) == _dim(rhs)


def _scale_shard_axis_mapping(axis_mapping: Any, factor: int) -> dict[str, Any]:
    """Lift a physical-axis shard mapping into its scalar vector-lane domain."""

    if factor <= 0:
        raise ValueError(f"PyNTT shard-axis scale must be positive, got {factor}")
    stages = []
    for stage in _shard_axis_stages(axis_mapping):
        scaled = dict(stage)
        distribution = stage.get("Distribution")
        if distribution == "BlockCyclic":
            block_size = stage.get("BlockSize")
            if not isinstance(block_size, int) or block_size <= 0:
                raise ValueError(
                    "PyNTT block-cyclic split requires a positive BlockSize, "
                    f"got {block_size!r}"
                )
            scaled["BlockSize"] = block_size * factor
        elif distribution == "Contiguous":
            granularity = stage.get("Granularity")
            if granularity is not None:
                scaled["Granularity"] = _multiply_dim(granularity, factor)
        else:
            raise ValueError(
                f"Unsupported PyNTT split-stage distribution {distribution!r}"
            )
        stages.append(scaled)
    return {**axis_mapping, "Stages": stages}


def _remap_local_coordinate(
    local_coordinate: str,
    source_global_extent: Any,
    source_axis_mapping: Any,
    destination_global_extent: Any,
    destination_axis_mapping: Any,
    hierarchy: list[int],
    *,
    local_extent: int | None = None,
) -> str:
    """Compose local->global->local without expanding identical shard maps."""

    if not _dimensions_equivalent(source_global_extent, destination_global_extent):
        raise ValueError(
            "PyNTT coordinate remapping requires equivalent logical extents: "
            f"source={_dim(source_global_extent)}, "
            f"destination={_dim(destination_global_extent)}"
        )
    if source_axis_mapping == destination_axis_mapping:
        return str(local_coordinate)
    global_coordinate = _local_to_global_coordinate(
        local_coordinate,
        source_global_extent,
        source_axis_mapping,
        hierarchy,
        local_extent=local_extent,
    )
    return _global_to_local_coordinate(
        global_coordinate,
        destination_global_extent,
        destination_axis_mapping,
        hierarchy,
    )["local_coordinate"]


def _distributed_local_to_global_coordinates(
    local_coordinates: tuple[str, ...],
    global_shape: list[Any],
    local_origins: list[Any],
    shard_axes: list[Any],
    hierarchy: list[int],
) -> tuple[str, ...]:
    """Map dense local tensor coordinates into the logical global domain."""

    rank = len(local_coordinates)
    if not (
        len(global_shape) == rank
        and len(local_origins) == rank
        and len(shard_axes) == rank
    ):
        raise ValueError(
            "PyNTT distributed coordinate rank mismatch: "
            f"coordinates={rank}, shape={len(global_shape)}, "
            f"origins={len(local_origins)}, shard_axes={len(shard_axes)}"
        )
    return tuple(
        _local_to_global_coordinate(
            _add_coordinate(local_origins[axis], local_coordinates[axis]),
            global_shape[axis],
            shard_axes[axis],
            hierarchy,
        )
        for axis in range(rank)
    )


def _pointer_local_to_global_coordinate(
    pointer: dict[str, Any], axis: int, local_coordinate: str
) -> str:
    """Map one dense local pointer coordinate into its logical tensor domain."""

    if not isinstance(pointer, dict):
        raise ValueError("PyNTT distributed pointer metadata must be an object")
    global_shape = pointer.get("GlobalShape")
    global_offsets = pointer.get("GlobalOffsets")
    shard_axes = pointer.get("ShardAxes")
    hierarchy = pointer.get("Hierarchy")
    if not all(
        isinstance(value, list)
        for value in (global_shape, global_offsets, shard_axes, hierarchy)
    ):
        raise ValueError("PyNTT distributed pointer has incomplete shard metadata")
    rank = len(global_shape)
    if not (
        len(global_offsets) == rank
        and len(shard_axes) == rank
        and isinstance(axis, int)
        and 0 <= axis < rank
    ):
        raise ValueError(
            "PyNTT distributed pointer coordinate rank mismatch: "
            f"shape={rank}, origins={len(global_offsets)}, "
            f"mappings={len(shard_axes)}, axis={axis}"
        )
    return _local_to_global_coordinate(
        _add_coordinate(global_offsets[axis], str(local_coordinate)),
        global_shape[axis],
        shard_axes[axis],
        hierarchy,
    )


def _pointer_local_vector_to_global_scalar_coordinate(
    pointer: dict[str, Any], axis: int, local_coordinate: str, lane_count: int
) -> str:
    """Map a scalar coordinate on a contiguous packed axis to the global domain."""

    if lane_count <= 0:
        raise ValueError(
            "PyNTT vector coordinate mapping requires a positive lane count, "
            f"got {lane_count}."
        )
    global_packed_coordinate = _pointer_local_to_global_coordinate(
        pointer,
        axis,
        f"(({local_coordinate}) // {lane_count})",
    )
    return (
        f"(({global_packed_coordinate}) * {lane_count} + "
        f"(({local_coordinate}) % {lane_count}))"
    )


def _local_axis_active_extent(
    global_extent: Any,
    axis_mapping: Any,
    hierarchy: list[int],
    coord_prefix: str = "shard_coord",
) -> str:
    parent_extent = _dim(global_extent)
    for stage in _shard_axis_stages(axis_mapping):
        distribution = stage.get("Distribution")
        shard_count = _stage_shard_count(stage, hierarchy)
        shard_index = _stage_linear_expression(stage, hierarchy, coord_prefix)
        if distribution == "Contiguous":
            granularity = stage.get("Granularity")
            capacity = (
                _dim(granularity)
                if granularity is not None
                else f"tl.cdiv(({parent_extent}), {shard_count})"
            )
            parent_extent = (
                f"tl.maximum(0, tl.minimum(({capacity}), "
                f"({parent_extent}) - ({shard_index}) * ({capacity})))"
            )
        elif distribution == "BlockCyclic":
            block_size = stage.get("BlockSize")
            if not isinstance(block_size, int) or block_size <= 0:
                raise ValueError(
                    "PyNTT block-cyclic split requires a positive BlockSize, "
                    f"got {block_size!r}"
                )
            period = shard_count * block_size
            parent_extent = (
                f"((({parent_extent}) // {period}) * {block_size} + "
                f"tl.maximum(0, tl.minimum({block_size}, "
                f"(({parent_extent}) % {period}) - ({shard_index}) * {block_size})))"
            )
        else:
            raise ValueError(
                f"Unsupported PyNTT split-stage distribution {distribution!r}"
            )
    return parent_extent


def _tma_local_axis_plan(
    pointer: Any,
    axis: int,
    *,
    tile_extent: int,
    context: str,
) -> dict[str, Any]:
    """Describe one dense local shard axis in a per-owner TMA descriptor."""

    if not isinstance(pointer, dict):
        raise ValueError(f"PyNTT {context} pointer metadata must be an object")
    global_shape = pointer.get("GlobalShape")
    shard_axes = pointer.get("ShardAxes")
    hierarchy = pointer.get("Hierarchy")
    if not all(isinstance(value, list) for value in (global_shape, shard_axes, hierarchy)):
        raise ValueError(f"PyNTT {context} pointer has incomplete shard metadata")
    if len(global_shape) != len(shard_axes):
        raise ValueError(
            f"PyNTT {context} pointer shape/mapping rank mismatch: "
            f"{len(global_shape)}/{len(shard_axes)}"
        )
    normalized_axis = axis if axis >= 0 else len(global_shape) + axis
    if normalized_axis < 0 or normalized_axis >= len(global_shape):
        raise ValueError(
            f"PyNTT {context} descriptor axis {axis} is outside rank {len(global_shape)}"
        )
    if tile_extent <= 0:
        raise ValueError(f"PyNTT {context} requires a positive tile extent")

    axis_mapping = shard_axes[normalized_axis]
    storage_kind = _require_string(
        pointer.get("DistributedStorageKind"),
        f"PyNTT {context} distributed storage kind",
        nonempty=True,
    )
    if storage_kind in ("CompactLocal", "CompactPerOwner"):
        return {
            "axis": normalized_axis,
            "axis_mapping": axis_mapping,
            "hierarchy": hierarchy,
            "is_block_cyclic": False,
            "is_compact_local": True,
            "block_shape": (tile_extent,),
            "block_size": None,
            "period": None,
        }
    if storage_kind != "CanonicalGlobal":
        raise ValueError(
            f"PyNTT {context} TMA does not support distributed storage "
            f"kind {storage_kind!r}."
        )

    stages = _shard_axis_stages(axis_mapping)
    cyclic_stage_indexes = [
        index
        for index, stage in enumerate(stages)
        if stage.get("Distribution") == "BlockCyclic"
    ]
    if len(cyclic_stage_indexes) > 1 or (
        cyclic_stage_indexes and cyclic_stage_indexes[0] != len(stages) - 1
    ):
        raise ValueError(
            f"PyNTT {context} TMA supports at most one innermost block-cyclic "
            "split stage"
        )

    if not cyclic_stage_indexes:
        return {
            "axis": normalized_axis,
            "axis_mapping": axis_mapping,
            "hierarchy": hierarchy,
            "is_block_cyclic": False,
            "is_compact_local": False,
            "block_shape": (tile_extent,),
            "block_size": None,
            "period": None,
        }

    cyclic_index = cyclic_stage_indexes[0]
    cyclic_stage = stages[cyclic_index]
    block_size = cyclic_stage.get("BlockSize")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError(
            f"PyNTT {context} block-cyclic TMA requires a positive BlockSize"
        )
    shard_count = _stage_shard_count(cyclic_stage, hierarchy)
    period = shard_count * block_size
    for stage in stages[:cyclic_index]:
        distribution = stage.get("Distribution")
        if distribution != "Contiguous":
            raise ValueError(
                f"PyNTT {context} block-cyclic TMA requires all outer split "
                "stages to be contiguous"
            )

    if block_size == 1:
        block_shape = (tile_extent,)
    elif tile_extent >= block_size:
        if tile_extent % block_size != 0:
            raise ValueError(
                f"PyNTT {context} TMA tile extent {tile_extent} must be a "
                f"multiple of block-cyclic BlockSize {block_size}"
            )
        block_shape = (tile_extent // block_size, block_size)
    else:
        if block_size % tile_extent != 0:
            raise ValueError(
                f"PyNTT {context} block-cyclic BlockSize {block_size} must be a "
                f"multiple of TMA tile extent {tile_extent}"
            )
        block_shape = (1, tile_extent)

    return {
        "axis": normalized_axis,
        "axis_mapping": axis_mapping,
        "hierarchy": hierarchy,
        "is_block_cyclic": True,
        "is_compact_local": False,
        "block_shape": block_shape,
        "block_size": block_size,
        "period": period,
    }


def _tma_local_axis_transfer(
    pointer: Any,
    axis: int,
    local_origin: Any,
    *,
    local_offset: int,
    tile_index: int | str,
    tile_stride: int,
    tile_extent: int,
    context: str,
) -> dict[str, Any]:
    """Map one dense-local tile to its per-owner descriptor coordinates."""

    if local_offset < 0 or tile_stride <= 0:
        raise ValueError(
            f"PyNTT {context} requires a non-negative local_offset and a "
            "positive tile_stride"
        )
    plan = _tma_canonical_axis_plan(
        pointer,
        axis,
        tile_extent=tile_extent,
        context=context,
    )

    if plan["is_block_cyclic"]:
        origin = _fixed(local_origin)
        if origin is None:
            raise ValueError(
                f"PyNTT {context} block-cyclic TMA requires a fixed local origin"
            )
        base = origin + local_offset
        block_size = plan["block_size"]
        if tile_extent >= block_size:
            if base % block_size != 0 or tile_stride % block_size != 0:
                raise ValueError(
                    f"PyNTT {context} TMA tiles spanning block-cyclic blocks "
                    f"must start and advance on BlockSize {block_size} boundaries"
                )
        elif isinstance(tile_index, int):
            start = base + tile_index * tile_stride
            if start % block_size + tile_extent > block_size:
                raise ValueError(
                    f"PyNTT {context} TMA tile [{start}, {start + tile_extent}) "
                    f"crosses a block-cyclic block of {block_size} elements"
                )
        else:
            phase = gcd(block_size, tile_stride)
            if tile_extent > phase or base % phase + tile_extent > phase:
                raise ValueError(
                    f"PyNTT {context} dynamic TMA tiles cannot be proven to stay "
                    f"inside block-cyclic blocks: block_size={block_size}, "
                    f"base={base}, stride={tile_stride}, extent={tile_extent}"
                )

    tile_delta = local_offset
    if isinstance(tile_index, int):
        tile_delta += tile_index * tile_stride
        local_coordinate = _add_coordinate(local_origin, str(tile_delta))
    else:
        dynamic_delta = f"({tile_index}) * {tile_stride}"
        if tile_delta:
            dynamic_delta = f"{tile_delta} + ({dynamic_delta})"
        local_coordinate = _add_coordinate(local_origin, dynamic_delta)
    if plan["is_block_cyclic"] and plan["block_size"] != 1:
        block_size = plan["block_size"]
        raw_coordinates = (
            f"(({local_coordinate}) // {block_size})",
            f"(({local_coordinate}) % {block_size})",
        )
    else:
        raw_coordinates = (local_coordinate,)
    coordinates = tuple(
        raw_coordinates[dimension]
        for dimension in plan["retained_dimensions"]
    )
    return {**plan, "coordinates": coordinates}


def _tma_shared_axis_coordinates(
    local_coordinate: str,
    plan: dict[str, Any],
) -> tuple[str, ...]:
    if not plan["is_block_cyclic"] or plan["block_size"] == 1:
        raw_coordinates = (local_coordinate,)
    else:
        block_size = plan["block_size"]
        try:
            fixed_coordinate = int(local_coordinate)
        except ValueError:
            raw_coordinates = (
                f"(({local_coordinate}) // {block_size})",
                f"(({local_coordinate}) % {block_size})",
            )
        else:
            raw_coordinates = (
                str(fixed_coordinate // block_size),
                str(fixed_coordinate % block_size),
            )
    return tuple(
        raw_coordinates[dimension]
        for dimension in plan["retained_dimensions"]
    )


def _unflatten_hierarchy_owner(
    linear_owner: int, hierarchy: list[int]
) -> tuple[int, ...]:
    if linear_owner < 0:
        raise ValueError("PyNTT hierarchy owner must be non-negative")
    coordinates = [0] * len(hierarchy)
    remainder = linear_owner
    for axis in range(len(hierarchy) - 1, -1, -1):
        extent = int(hierarchy[axis])
        if extent <= 0:
            raise ValueError(f"PyNTT hierarchy axis {axis} has invalid extent {extent}")
        coordinates[axis] = remainder % extent
        remainder //= extent
    if remainder:
        raise ValueError(
            f"PyNTT hierarchy owner {linear_owner} exceeds hierarchy {hierarchy}"
        )
    return tuple(coordinates)


def _fixed_stage_owner(
    stage: dict[str, Any], hierarchy: list[int], owner: tuple[int, ...]
) -> int:
    axes = stage.get("HierarchyAxes")
    if not isinstance(axes, list) or not axes:
        raise ValueError("PyNTT split stage requires non-empty HierarchyAxes")
    linear = 0
    for axis in axes:
        if not isinstance(axis, int) or axis < 0 or axis >= len(hierarchy):
            raise ValueError(
                f"PyNTT split stage axis {axis!r} is outside hierarchy rank {len(hierarchy)}"
            )
        linear = linear * hierarchy[axis] + owner[axis]
    return linear


def _tma_descriptor_table_axis_entry(
    pointer: Any,
    axis: int,
    owner: tuple[int, ...],
    *,
    tile_extent: int,
    context: str,
) -> dict[str, Any]:
    """Build one exact rectangular descriptor axis for a fixed shard owner."""

    plan = _tma_local_axis_plan(
        pointer,
        axis,
        tile_extent=tile_extent,
        context=context,
    )
    global_extent = _require_fixed_positive_dim(
        pointer["GlobalShape"][plan["axis"]], f"PyNTT {context} global extent"
    )
    parent_extent = global_extent
    base = 0
    descriptor_shape: tuple[int, ...] | None = None
    stride_multipliers: tuple[int, ...] | None = None
    is_compact_local = bool(plan["is_compact_local"])

    for stage in _shard_axis_stages(plan["axis_mapping"]):
        shard_count = _stage_shard_count(stage, plan["hierarchy"])
        shard_owner = _fixed_stage_owner(stage, plan["hierarchy"], owner)
        distribution = stage.get("Distribution")
        if distribution == "Contiguous":
            granularity = stage.get("Granularity")
            capacity = (
                (parent_extent + shard_count - 1) // shard_count
                if granularity is None
                else _require_fixed_positive_dim(
                    granularity, f"PyNTT {context} contiguous-stage granularity"
                )
            )
            start = shard_owner * capacity
            active_extent = max(0, min(capacity, parent_extent - start))
            if not is_compact_local:
                base += start
            parent_extent = active_extent
        elif distribution == "BlockCyclic":
            block_size = stage.get("BlockSize")
            if not isinstance(block_size, int) or block_size <= 0:
                raise ValueError(
                    f"PyNTT {context} block-cyclic split requires a positive BlockSize"
                )
            period = shard_count * block_size
            active_extent = (
                (parent_extent // period) * block_size
                + max(
                    0,
                    min(
                        block_size,
                        (parent_extent % period) - shard_owner * block_size,
                    ),
                )
            )
            if not is_compact_local:
                base += shard_owner * block_size
            parent_extent = active_extent
            if not is_compact_local:
                if block_size == 1:
                    descriptor_shape = (active_extent,)
                    stride_multipliers = (period,)
                else:
                    if active_extent % block_size != 0:
                        raise ValueError(
                            f"PyNTT {context} owner {owner} has a partial "
                            f"block-cyclic block ({active_extent} elements, "
                            f"BlockSize={block_size}); one rectangular tensor map "
                            "cannot represent that shard exactly"
                        )
                    descriptor_shape = (active_extent // block_size, block_size)
                    stride_multipliers = (period, 1)
        else:
            raise ValueError(
                f"Unsupported PyNTT split-stage distribution {distribution!r}"
            )

    if descriptor_shape is None:
        descriptor_shape = (parent_extent,)
        stride_multipliers = (1,)

    active = parent_extent > 0
    if not active:
        return {
            **plan,
            "active": False,
            "base": 0,
            "descriptor_shape": tuple(plan["block_shape"]),
            "stride_multipliers": stride_multipliers,
        }
    return {
        **plan,
        "active": True,
        "base": base,
        "descriptor_shape": descriptor_shape,
        "stride_multipliers": stride_multipliers,
    }


def _tma_canonical_axis_plan(
    pointer: Any,
    axis: int,
    *,
    tile_extent: int,
    context: str,
) -> dict[str, Any]:
    """Remove descriptor-axis dimensions that are provably unit for every owner."""

    plan = _tma_local_axis_plan(
        pointer,
        axis,
        tile_extent=tile_extent,
        context=context,
    )
    hierarchy = plan["hierarchy"]
    owner_count = _product_int([int(value) for value in hierarchy])
    entries = tuple(
        _tma_descriptor_table_axis_entry(
            pointer,
            axis,
            _unflatten_hierarchy_owner(linear_owner, hierarchy),
            tile_extent=tile_extent,
            context=context,
        )
        for linear_owner in range(owner_count)
    )
    raw_block_shape = tuple(plan["block_shape"])
    if any(len(entry["descriptor_shape"]) != len(raw_block_shape) for entry in entries):
        raise ValueError(
            f"PyNTT {context} descriptor shape rank differs from its block rank."
        )
    retained_dimensions = tuple(
        dimension
        for dimension, block_extent in enumerate(raw_block_shape)
        if block_extent != 1
        or any(entry["descriptor_shape"][dimension] != 1 for entry in entries)
    )
    return {
        **plan,
        "raw_block_shape": raw_block_shape,
        "block_shape": tuple(
            raw_block_shape[dimension] for dimension in retained_dimensions
        ),
        "retained_dimensions": retained_dimensions,
    }


def _tma_packed_atom_axis_plan(
    pointer: Any,
    axis: int,
    *,
    tile_extent: int,
    atom_extent: int,
    logical_axis_stride: int,
    context: str,
) -> dict[str, Any]:
    """Preserve a target-packed scalar atom as an explicit TMA layout axis."""

    if atom_extent <= 0 or logical_axis_stride <= 0:
        raise ValueError(
            f"PyNTT {context} requires positive atom extent and logical stride."
        )
    axis_plan = _tma_canonical_axis_plan(
        pointer,
        axis,
        tile_extent=tile_extent,
        context=context,
    )
    axis_block_shape = tuple(axis_plan["block_shape"])
    return {
        "axis_plan": axis_plan,
        "atom_extent": atom_extent,
        "logical_axis_stride": logical_axis_stride,
        "block_shape": axis_block_shape + (atom_extent,),
    }


def _tma_packed_atom_coordinates(
    local_coordinate: str,
    plan: dict[str, Any],
) -> tuple[str, ...]:
    coordinates = _tma_shared_axis_coordinates(
        local_coordinate, plan["axis_plan"]
    )
    return coordinates + ("0",)


def _tma_packed_atom_entry(
    entry: dict[str, Any],
    plan: dict[str, Any],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    axis_plan = plan["axis_plan"]
    retained_dimensions = tuple(axis_plan["retained_dimensions"])
    atom_extent = int(plan["atom_extent"])
    logical_axis_stride = int(plan["logical_axis_stride"])
    shape = tuple(
        int(entry["descriptor_shape"][dimension])
        for dimension in retained_dimensions
    )
    strides = tuple(
        int(entry["stride_multipliers"][dimension])
        * logical_axis_stride
        * atom_extent
        for dimension in retained_dimensions
    )
    return shape + (atom_extent,), strides + (1,)


def _nv_tma_swizzle_mode(block_shape: tuple[int, ...], dtype: str) -> int:
    try:
        item_size = TMA_DTYPE_ITEM_SIZES[dtype]
    except KeyError as ex:
        raise ValueError(f"PyNTT TMA does not support descriptor dtype {dtype!r}") from ex
    if len(block_shape) < 2 or _product_int(list(block_shape[:-1])) < 8:
        return 0
    contiguous_bytes = block_shape[-1] * item_size
    for width, mode in ((128, 3), (64, 2), (32, 1)):
        if contiguous_bytes == width:
            return mode
    return 0


def _tma_split_contiguous_terminal_block(
    block_shape: tuple[int, ...],
    dtype: str,
    *,
    context: str,
) -> tuple[tuple[int, ...], int | None]:
    """Factor a wide contiguous TMA dimension into one 128-byte atom."""

    if not block_shape:
        raise ValueError(f"PyNTT {context} TMA block shape cannot be empty.")
    try:
        item_size = TMA_DTYPE_ITEM_SIZES[dtype]
    except KeyError as ex:
        raise ValueError(
            f"PyNTT {context} TMA does not support descriptor dtype {dtype!r}."
        ) from ex
    terminal_extent = int(block_shape[-1])
    maximum_atom = TMA_MAXIMUM_SWIZZLE_BYTES // item_size
    if terminal_extent <= maximum_atom:
        return block_shape, None
    if terminal_extent % maximum_atom != 0:
        raise ValueError(
            f"PyNTT {context} contiguous block extent {terminal_extent} cannot "
            f"be factored into {maximum_atom}-element TMA swizzle atoms."
        )
    return (
        block_shape[:-1]
        + (terminal_extent // maximum_atom, maximum_atom),
        maximum_atom,
    )


def _tma_dtype_from_triton_dtype(dtype: Any, *, context: str) -> str:
    mapping = {
        "tl.float16": "float16",
        "tl.bfloat16": "bfloat16",
        "tl.float8e4nv": "float8e4m3fn",
        "tl.float8e5": "float8e5m2",
        "tl.float32": "float32",
        "tl.int8": "int8",
        "tl.uint8": "uint8",
    }
    try:
        return mapping[str(dtype)]
    except KeyError as ex:
        raise ValueError(
            f"PyNTT {context} does not support Triton descriptor dtype {dtype!r}."
        ) from ex


def _global_to_local_coordinate(
    global_coordinate: str,
    global_extent: Any,
    axis_mapping: Any,
    hierarchy: list[int],
) -> dict[str, Any]:
    """Map one global coordinate to its dense local coordinate and stage owners."""

    coordinate = str(global_coordinate)
    parent_extent = _dim(global_extent)
    owners: dict[int, str] = {}
    for stage in _shard_axis_stages(axis_mapping):
        distribution = stage.get("Distribution")
        shard_count = _stage_shard_count(stage, hierarchy)
        if distribution == "Contiguous":
            granularity = stage.get("Granularity")
            capacity = (
                _dim(granularity)
                if granularity is not None
                else f"tl.cdiv(({parent_extent}), {shard_count})"
            )
            linear_owner = f"(({coordinate}) // ({capacity}))"
            local_coordinate = (
                f"(({coordinate}) - ({linear_owner}) * ({capacity}))"
            )
            active_extent = (
                f"tl.maximum(0, tl.minimum(({capacity}), "
                f"({parent_extent}) - ({linear_owner}) * ({capacity})))"
            )
        elif distribution == "BlockCyclic":
            block_size = stage.get("BlockSize")
            if not isinstance(block_size, int) or block_size <= 0:
                raise ValueError(
                    "PyNTT block-cyclic split requires a positive BlockSize, "
                    f"got {block_size!r}"
                )
            period = shard_count * block_size
            linear_owner = f"((({coordinate}) // {block_size}) % {shard_count})"
            local_coordinate = (
                f"((({coordinate}) // {period}) * {block_size} + "
                f"(({coordinate}) % {block_size}))"
            )
            active_extent = (
                f"((({parent_extent}) // {period}) * {block_size} + "
                f"tl.maximum(0, tl.minimum({block_size}, "
                f"(({parent_extent}) % {period}) - ({linear_owner}) * {block_size})))"
            )
        else:
            raise ValueError(
                f"Unsupported PyNTT split-stage distribution {distribution!r}"
            )

        remaining = linear_owner
        axes = stage["HierarchyAxes"]
        for hierarchy_axis in reversed(axes):
            owners[int(hierarchy_axis)] = (
                f"(({remaining}) % {int(hierarchy[hierarchy_axis])})"
            )
            remaining = f"(({remaining}) // {int(hierarchy[hierarchy_axis])})"
        coordinate = local_coordinate
        parent_extent = active_extent

    return {
        "local_coordinate": coordinate,
        "owners": owners,
        "active_extent": parent_extent,
    }


def _canonicalize_access(pointer: dict[str, Any], access: Any) -> Any:
    if pointer.get("DistributedStorageKind") != "CanonicalGlobal":
        return access
    global_shape = pointer.get("GlobalShape")
    global_offsets = pointer.get("GlobalOffsets")
    shard_axes = pointer.get("ShardAxes")
    hierarchy = pointer.get("Hierarchy")
    strides = pointer.get("Strides")
    if not all(
        isinstance(value, list)
        for value in (global_shape, global_offsets, shard_axes, hierarchy, strides)
    ):
        raise ValueError("PyNTT canonical-global pointer has incomplete shard metadata")
    if not (
        len(global_shape) == len(global_offsets) == len(shard_axes) == len(strides)
    ):
        raise ValueError(
            "PyNTT canonical-global pointer shape/origin/mapping/stride rank mismatch: "
            f"{len(global_shape)}/{len(global_offsets)}/{len(shard_axes)}/{len(strides)}"
        )

    if access is None:
        access = _tensor_access(["0"] * len(global_shape), strides)
    if not isinstance(access, dict) or "TensorIndices" not in access:
        raise ValueError(
            "PyNTT canonical-global buffers require coordinate-native structured accesses"
        )

    original_indices = list(access["TensorIndices"])
    if len(original_indices) != len(global_shape):
        raise ValueError(
            "PyNTT canonical-global access rank mismatch: "
            f"indices={len(original_indices)}, shape={len(global_shape)}"
        )
    global_coordinate_axes = _validate_global_coordinate_axes(
        access.get("GlobalCoordinateAxes", ()),
        len(original_indices),
        "PyNTT canonical-global access",
    )
    global_coordinate_axis_set = set(global_coordinate_axes)

    def canonical_coordinate(axis: int, coordinate: str) -> str:
        if axis in global_coordinate_axis_set:
            return coordinate
        return _local_to_global_coordinate(
            _add_coordinate(global_offsets[axis], coordinate),
            global_shape[axis],
            shard_axes[axis],
            hierarchy,
        )

    tensor_indices = [
        canonical_coordinate(axis, original_indices[axis])
        for axis in range(len(original_indices))
    ]

    if "ContiguousVectorAxis" in access:
        packed_axis = int(access["ContiguousVectorAxis"])
        lane_count = int(access["LaneCount"])
        logical_index = str(access["LogicalIndex"])
        tensor_indices[packed_axis] = canonical_coordinate(
            packed_axis, f"(({logical_index}) // {lane_count})"
        )
        outer_access = _tensor_access(
            tensor_indices,
            strides,
            ("0",),
            (lane_count,),
        )
        raw_outer = _access_raw_scalar_offset(outer_access)
        lane_index = f"(({logical_index}) % {lane_count})"
        raw_scalar_offset = f"({raw_outer}) + ({lane_index})"
        coordinate_shape = access.get("CoordinateShape")
        scalar_offset = (
            f"tl.broadcast_to({raw_scalar_offset}, {coordinate_shape})"
            if coordinate_shape is not None
            else raw_scalar_offset
        )
        result = dict(access)
        result["RawScalarOffset"] = raw_scalar_offset
        result["ScalarOffset"] = scalar_offset
        result["TensorIndices"] = tuple(tensor_indices)
        result.pop("GlobalCoordinateAxes", None)
        return result

    result = _tensor_access(
        tensor_indices,
        strides,
        access.get("LaneIndices", ()),
        access.get("LaneShape", ()),
        access.get("CoordinateShape"),
    )
    if "BoundaryMask" in access:
        result["BoundaryMask"] = access["BoundaryMask"]
    return result


def _access_pointer(
    model: dict[str, Any],
    name: str,
    local_name: str,
    access: Any = None,
) -> str:
    pointer = model.get(name)
    if isinstance(pointer, dict):
        access = _canonicalize_access(pointer, access)
    scalar_offset = _access_scalar_offset(access)
    return local_name if scalar_offset == "0" else f"{local_name} + {scalar_offset}"


def _select_block_axis(shape: list[Any], strides: list[Any]) -> int:
    if not shape:
        return 0
    for axis in range(len(shape) - 1, -1, -1):
        if not _is_fixed_one(shape[axis]) and _is_fixed_one(strides[axis]):
            return axis
    for axis in range(len(shape) - 1, -1, -1):
        if not _is_fixed_one(shape[axis]):
            return axis
    return len(shape) - 1


def _contiguous_strides(shape: list[Any]) -> list[dict[str, Any]]:
    strides: list[dict[str, Any]] = [_one() for _ in shape]
    stride = _one()
    for axis in range(len(shape) - 1, -1, -1):
        strides[axis] = stride
        fixed = _fixed(stride)
        dim_fixed = _fixed(shape[axis])
        next_stride = {
            "PythonExpression": f"({_py_dim(stride)} * {_py_dim(shape[axis])})",
            "TritonExpression": f"({_dim(stride)} * {_dim(shape[axis])})",
        }
        if fixed is not None and dim_fixed is not None:
            next_stride["FixedValue"] = fixed * dim_fixed
        stride = next_stride
    return strides


def _split_linear_expression(
    split_axes: list[int], hierarchy: list[int], coord_prefix: str = "shard_coord"
) -> str:
    if not split_axes:
        return "0"
    terms = []
    for index, placement_axis in enumerate(split_axes):
        stride = 1
        for axis in split_axes[index + 1 :]:
            stride *= hierarchy[axis]
        coord = f"{coord_prefix}{placement_axis}"
        terms.append(coord if stride == 1 else f"{coord} * {stride}")
    return " + ".join(terms)


def _split_divisor(split_axes: list[int], hierarchy: list[int]) -> int:
    divisor = 1
    for axis in split_axes:
        divisor *= hierarchy[axis]
    return divisor


def _helper_parameter_declarations(
    model: dict[str, Any], args: tuple[str, ...] | list[str] = ()
) -> tuple[str, ...]:
    abi_args = tuple(model.get("Arguments", ()) or ())
    workspace_args = tuple(model["WorkspaceArguments"])
    declarations = (
        tuple(args)
        + abi_args
        + workspace_args
        + tuple(model.get("RuntimeShapeArgs", ()) or ())
        + ("block_size: tl.constexpr",)
    )
    names = _parameter_call_arguments(declarations)
    duplicates = sorted(name for name in set(names) if names.count(name) > 1)
    if duplicates:
        raise ValueError(
            f"PyNTT helper {model.get('FunctionName', '<unknown>')} has duplicate "
            f"parameters {duplicates}."
        )
    return declarations


def _helper_parameters(
    model: dict[str, Any], args: tuple[str, ...] | list[str] = ()
) -> str:
    """Build the stable helper ABI; kernel control flow belongs to Jinja."""

    return ", ".join(_helper_parameter_declarations(model, args))


def _helper_argument_names(
    model: dict[str, Any], args: tuple[str, ...] | list[str] = ()
) -> str:
    return ", ".join(_parameter_call_arguments(_helper_parameter_declarations(model, args)))


def _is_bool_dtype(dtype: Any) -> bool:
    return str(dtype) == "bool"


def _constant_dim_value(value: Any) -> int | None:
    fixed = _fixed(value)
    if fixed is not None:
        return fixed
    minimum = _min_value(value)
    maximum = _max_value(value)
    return minimum if minimum is not None and minimum == maximum else None


def _tile_bounds_mask(
    bounds: tuple[tuple[str, Any, int], ...],
) -> str | None:
    """Build only predicates not discharged by manifest range bounds."""

    predicates = []
    for coordinate, extent, required_extent in bounds:
        minimum = _min_value(extent)
        if minimum is not None and minimum >= required_extent:
            continue
        predicate = f"({coordinate} < {_dim(extent)})"
        if predicate not in predicates:
            predicates.append(predicate)
    return " & ".join(predicates) or None


def _region_copy_plan(
    model: dict[str, Any],
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    plan = model.get("CopyPlan")
    if not isinstance(plan, dict):
        raise ValueError("TensorRegionCopy requires a coordinate CopyPlan")
    extents_value = plan.get("Extents")
    if not isinstance(extents_value, list) or not extents_value:
        raise ValueError("TensorRegionCopy CopyPlan requires at least one extent")
    extents = tuple(extents_value)
    for field in (
        "SourceOrigins",
        "DestinationOrigins",
        "CoversWholeSource",
        "CoversWholeDestination",
    ):
        if field not in plan:
            raise ValueError(f"TensorRegionCopy CopyPlan is missing {field}")
    rank = len(model.get("SourceShape", ()))
    lane_rank = len(model.get("VectorLaneShape", ()))
    if len(extents) != rank + lane_rank:
        raise ValueError(
            "TensorRegionCopy extent rank must equal logical plus vector-lane rank: "
            f"extents={len(extents)}, logical={rank}, lanes={lane_rank}"
        )
    for side in ("Source", "Destination"):
        origins = plan[f"{side}Origins"]
        if not isinstance(origins, list) or len(origins) != rank:
            raise ValueError(
                f"TensorRegionCopy {side}Origins rank must be {rank}, got "
                f"{len(origins) if isinstance(origins, list) else 'non-list'}"
            )
    return plan, extents


def _tensor_region_copy_template_context(model: dict[str, Any]) -> dict[str, Any]:
    """Prepare a global-memory region copy for its Jinja template."""

    rank = len(model["SourceShape"])
    if rank != len(model["DestinationShape"]):
        raise ValueError("TensorRegionCopy source and destination ranks must match")
    copy_plan, extents = _region_copy_plan(model)
    non_unit_axes = tuple(
        axis for axis, extent in enumerate(extents) if _max_value(extent) != 1
    )
    inner_axis = non_unit_axes[-1] if non_unit_axes else len(extents) - 1
    coordinate_shape = "(block_size,)"
    coordinate_expressions = tuple(
        f"copy_idx{axis}" for axis in range(len(extents))
    )
    lane_shape = tuple(int(value) for value in model.get("VectorLaneShape", ()))

    def build_access(side: str) -> dict[str, Any]:
        origins = copy_plan[f"{side}Origins"]
        tensor_indices = tuple(
            _add_coordinate(origins[axis], coordinate_expressions[axis])
            for axis in range(rank)
        )
        return _tensor_access(
            tensor_indices,
            model[f"{side}Strides"],
            coordinate_expressions[rank:],
            lane_shape,
            coordinate_shape,
        )

    return {
        "destination_access": build_access("Destination"),
        "extents": extents,
        "inner_axis": inner_axis,
        "inner_extent": extents[inner_axis],
        "outer_axes": tuple(
            axis for axis in range(len(extents)) if axis != inner_axis
        ),
        "pointer_values": (model["Source"], model["Destination"]),
        "source_access": build_access("Source"),
    }


def _tensor_copy_template_context(
    model: dict[str, Any], *, is_load: bool
) -> dict[str, Any]:
    """Prepare coordinate-native TensorLoad/TensorStore accesses for Jinja."""

    local_shape = model["LocalShape"]
    global_shape = model["GlobalShape"]
    local_strides = model["DestinationStrides" if is_load else "SourceStrides"]
    explicit_global_strides = model.get(
        "SourceStrides" if is_load else "DestinationStrides"
    )
    global_strides = explicit_global_strides or _contiguous_strides(global_shape)
    if len(local_shape) != len(global_shape):
        raise ValueError(
            "PyNTT TensorLoad/TensorStore local/global rank mismatch: "
            f"local={len(local_shape)}, global={len(global_shape)}"
        )
    lane_shape = model.get("VectorLaneShape", ())
    major_axis = _select_block_axis(local_shape, local_strides)
    if all(_is_fixed_one(dim) for dim in local_shape) and _fixed(
        local_strides[major_axis]
    ) == 0:
        major_axis = next(
            (
                axis
                for axis in range(len(local_shape) - 1, -1, -1)
                if _fixed(local_strides[axis]) != 0
            ),
            major_axis,
        )
    ctx = _coordinate_iteration_context(
        local_shape,
        local_strides,
        lane_shape,
        "PyNTT TensorLoad" if is_load else "PyNTT TensorStore",
        major_axis=major_axis,
    )
    local_access = _tensor_access(
        ctx["tensor_coordinates"],
        local_strides,
        ctx["lane_coordinates"],
        ctx["lane_shape"],
    )
    global_coordinates = tuple(
        _local_to_global_coordinate(
            _add_coordinate(model["GlobalOffsets"][axis], coordinate),
            global_shape[axis],
            model["ShardAxes"][axis],
            model["Hierarchy"],
        )
        for axis, coordinate in enumerate(ctx["tensor_coordinates"])
    )
    global_access = _tensor_access(
        global_coordinates,
        global_strides,
        ctx["lane_coordinates"],
        ctx["lane_shape"],
    )

    def add_external_base(access: dict[str, Any], base: str) -> dict[str, Any]:
        result = dict(access)
        scalar_offset = _access_scalar_offset(access)
        result["ScalarOffset"] = (
            base if scalar_offset == "0" else f"({base}) + ({scalar_offset})"
        )
        return result

    internal_source = model.get("Source") if is_load else None
    internal_destination = model.get("Destination") if not is_load else None
    if is_load:
        source_access = (
            global_access
            if internal_source is not None
            else add_external_base(
                global_access,
                f"source_pool_stride_elements * shard_index + {model['SourceOffset']}",
            )
        )
        destination_access = local_access
    else:
        source_access = local_access
        destination_access = (
            global_access
            if internal_destination is not None
            else add_external_base(
                global_access,
                f"destination_pool_stride_elements * shard_index + {model['DestinationOffset']}",
            )
        )

    ctx.update(
        destination_access=destination_access,
        global_coordinates=global_coordinates,
        global_shape=global_shape,
        internal_destination=internal_destination,
        internal_source=internal_source,
        is_load=is_load,
        local_shape=local_shape,
        source_access=source_access,
    )
    return ctx


def _logical_shape(shape: list[Any], lane_count: int) -> list[Any]:
    result = [dict(dim) if isinstance(dim, dict) else dim for dim in shape]
    if lane_count > 1:
        result[-1] = _multiply_dim(result[-1], lane_count)
    return result


def _logical_strides(strides: list[Any], lane_count: int) -> list[Any]:
    result = [dict(dim) if isinstance(dim, dict) else dim for dim in strides]
    if lane_count > 1:
        result[-1] = _one()
    return result


def _validate_coordinate_lane_shape(
    lane_shape: list[int], context: str
) -> tuple[int, ...]:
    lanes = tuple(int(value) for value in lane_shape)
    if any(value <= 0 or value & (value - 1) for value in lanes):
        raise ValueError(
            f"{context} lane dimensions must be positive powers of two, got {lanes}."
        )
    return lanes


def _flatten_coordinates(indices: tuple[str, ...], shape: tuple[int, ...]) -> str:
    if len(indices) != len(shape):
        raise ValueError(
            "PyNTT coordinate flatten rank mismatch: "
            f"indices={len(indices)}, shape={len(shape)}"
        )
    terms: list[str] = []
    stride = 1
    for index, extent in reversed(tuple(zip(indices, shape))):
        if index != "0":
            terms.append(index if stride == 1 else f"({index}) * {stride}")
        stride *= extent
    return _join_index_terms(list(reversed(terms)))


def _coordinate_tile_shape(
    physical_tile_extent: str, lane_shape: tuple[int, ...] | list[int]
) -> str:
    tile_extents = (physical_tile_extent,) + tuple(str(value) for value in lane_shape)
    return f"({', '.join(tile_extents)}{',' if len(tile_extents) == 1 else ''})"


def _coordinate_iteration_context(
    tensor_shape: list[Any],
    tensor_strides: list[Any],
    lane_shape: list[int],
    context: str = "PyNTT elementwise",
    variable_prefix: str = "",
    major_axis: int | None = None,
) -> dict[str, Any]:
    """Build a coordinate-native block tile without scalar unflattening."""

    if len(tensor_shape) != len(tensor_strides):
        raise ValueError(
            "PyNTT coordinate iteration shape/stride rank mismatch: "
            f"shape={len(tensor_shape)}, strides={len(tensor_strides)}"
        )
    lanes = _validate_coordinate_lane_shape(lane_shape, context)
    lane_count = _product_int(list(lanes)) if lanes else 1
    major_variable = f"{variable_prefix}major_raw"
    index_variables = tuple(
        f"{variable_prefix}index{axis}" for axis in range(len(tensor_shape))
    )

    if tensor_shape:
        if major_axis is None:
            major_axis = _select_block_axis(tensor_shape, tensor_strides)
        elif major_axis < 0 or major_axis >= len(tensor_shape):
            raise ValueError(
                f"{context} major axis {major_axis} is outside rank {len(tensor_shape)}"
            )
        major_extent = tensor_shape[major_axis]
        loop_axes = tuple(
            axis for axis in range(len(tensor_shape)) if axis != major_axis
        )
        tensor_coordinates = tuple(
            major_variable if axis == major_axis else index_variables[axis]
            for axis in range(len(tensor_shape))
        )
    else:
        major_axis = -1
        major_extent = _one()
        loop_axes = ()
        tensor_coordinates = ()

    lane_coordinates = tuple(
        f"{variable_prefix}lane_raw{axis}" for axis in range(len(lanes))
    )
    physical_tile_extent = (
        "block_size" if lane_count == 1 else f"(block_size // {lane_count})"
    )
    major_reshape = "" if not lanes else "[:, " + ", ".join("None" for _ in lanes) + "]"
    lane_reshapes = []
    for axis in range(len(lanes)):
        dimensions = ["None"] * (len(lanes) + 1)
        dimensions[axis + 1] = ":"
        lane_reshapes.append("[" + ", ".join(dimensions) + "]")

    return {
        "lane_count": lane_count,
        "lane_coordinates": lane_coordinates,
        "lane_reshapes": tuple(lane_reshapes),
        "lane_shape": lanes,
        "loop_axes": loop_axes,
        "index_variables": index_variables,
        "major_axis": major_axis,
        "major_extent": major_extent,
        "major_reshape": major_reshape,
        "major_variable": major_variable,
        "tensor_coordinates": tensor_coordinates,
        "tensor_shape": tuple(tensor_shape),
        "tile_shape": _coordinate_tile_shape(physical_tile_extent, lanes),
    }


def _broadcast_physical_access(
    shape: list[Any],
    strides: list[Any],
    lane_shape: list[int],
    output_shape: list[Any],
    output_lane_shape: tuple[int, ...],
    output_tensor_coordinates: tuple[str, ...],
    output_lane_coordinates: tuple[str, ...],
    output_major_axis: int,
) -> dict[str, Any]:
    lanes = tuple(int(value) for value in lane_shape)
    if len(shape) > len(output_shape):
        raise ValueError(
            "PyNTT elementwise operand rank exceeds output rank: "
            f"operand={len(shape)}, output={len(output_shape)}"
        )
    axis_offset = len(output_shape) - len(shape)
    tensor_coordinates = tuple(
        "0"
        if _is_fixed_one(extent)
        else output_tensor_coordinates[axis_offset + axis]
        for axis, extent in enumerate(shape)
    )
    if lanes:
        if lanes != output_lane_shape:
            raise ValueError(
                "PyNTT elementwise vector operand lanes must match output lanes: "
                f"operand={lanes}, output={output_lane_shape}"
            )
        lane_coordinates = output_lane_coordinates
    else:
        lane_coordinates = ()
    access = _tensor_access(
        tensor_coordinates, strides, lane_coordinates, lanes
    )
    return _with_major_boundary_mask(
        access, shape, strides, output_major_axis - axis_offset
    )


def _memcopy_template_context(model: dict[str, Any]) -> dict[str, Any]:
    """Prepare a coordinate-native copy over the destination buffer domain."""

    lanes = _validate_coordinate_lane_shape(model["VectorLaneShape"], "PyNTT Memcopy")
    lane_count = _product_int(list(lanes)) if lanes else 1
    if lane_count != int(model["VectorLaneCount"]):
        raise ValueError(
            "PyNTT Memcopy vector lane shape/count mismatch: "
            f"shape={lanes}, count={model['VectorLaneCount']}."
        )
    ctx = _coordinate_iteration_context(
        model["Shape"],
        model["DestinationStrides"],
        model["VectorLaneShape"],
        "PyNTT Memcopy",
    )
    ctx["source_access"] = _with_major_boundary_mask(
        _tensor_access(
            ctx["tensor_coordinates"],
            model["SourceStrides"],
            ctx["lane_coordinates"],
            lanes,
        ),
        model["Shape"],
        model["SourceStrides"],
        ctx["major_axis"],
    )
    ctx["destination_access"] = _with_major_boundary_mask(
        _tensor_access(
            ctx["tensor_coordinates"],
            model["DestinationStrides"],
            ctx["lane_coordinates"],
            lanes,
        ),
        model["Shape"],
        model["DestinationStrides"],
        ctx["major_axis"],
    )
    return ctx


def _elementwise_unary_template_context(model: dict[str, Any]) -> dict[str, Any]:
    ctx = _coordinate_iteration_context(
        model["OutputShape"],
        model["OutputStrides"],
        model["OutputVectorLaneShape"],
    )
    ctx["input_access"] = _broadcast_physical_access(
        model["InputShape"],
        model["InputStrides"],
        model["InputVectorLaneShape"],
        model["OutputShape"],
        ctx["lane_shape"],
        ctx["tensor_coordinates"],
        ctx["lane_coordinates"],
        ctx["major_axis"],
    )
    ctx["output_access"] = _with_major_boundary_mask(
        _tensor_access(
            ctx["tensor_coordinates"],
            model["OutputStrides"],
            ctx["lane_coordinates"],
            ctx["lane_shape"],
        ),
        model["OutputShape"],
        model["OutputStrides"],
        ctx["major_axis"],
    )
    return ctx


def _elementwise_binary_template_context(model: dict[str, Any]) -> dict[str, Any]:
    ctx = _coordinate_iteration_context(
        model["OutputShape"],
        model["OutputStrides"],
        model["OutputVectorLaneShape"],
    )
    for prefix in ("Lhs", "Rhs"):
        ctx[f"{prefix.lower()}_access"] = _broadcast_physical_access(
            model[f"{prefix}Shape"],
            model[f"{prefix}Strides"],
            model[f"{prefix}VectorLaneShape"],
            model["OutputShape"],
            ctx["lane_shape"],
            ctx["tensor_coordinates"],
            ctx["lane_coordinates"],
            ctx["major_axis"],
        )
    ctx["output_access"] = _with_major_boundary_mask(
        _tensor_access(
            ctx["tensor_coordinates"],
            model["OutputStrides"],
            ctx["lane_coordinates"],
            ctx["lane_shape"],
        ),
        model["OutputShape"],
        model["OutputStrides"],
        ctx["major_axis"],
    )
    return ctx


def _elementwise_cast_template_context(model: dict[str, Any]) -> dict[str, Any]:
    input_lanes = _validate_coordinate_lane_shape(
        model["InputVectorLaneShape"], "PyNTT Cast input"
    )
    output_lanes = _validate_coordinate_lane_shape(
        model["OutputVectorLaneShape"], "PyNTT Cast output"
    )
    input_lane_count = _product_int(list(input_lanes)) if input_lanes else 1
    output_lane_count = _product_int(list(output_lanes)) if output_lanes else 1
    common_lane_count = max(input_lane_count, output_lane_count)
    smaller_lane_count = min(input_lane_count, output_lane_count)
    if common_lane_count % smaller_lane_count != 0:
        raise ValueError(
            "PyNTT Cast vector lane counts must divide one another: "
            f"input={input_lane_count}, output={output_lane_count}"
        )

    vectorized_axes = tuple(int(value) for value in model["VectorizedAxes"])
    if common_lane_count != 1 and len(vectorized_axes) != 1:
        raise ValueError("PyNTT vector Cast requires exactly one vectorized axis")
    vectorized_axis = vectorized_axes[0] if vectorized_axes else -1
    if input_lane_count == common_lane_count:
        domain_shape = model["InputShape"]
        domain_strides = model["InputStrides"]
    else:
        domain_shape = model["OutputShape"]
        domain_strides = model["OutputStrides"]

    lane_ratio = common_lane_count // smaller_lane_count
    if common_lane_count == 1:
        domain_lane_shape: list[int] = []
    elif lane_ratio == 1:
        domain_lane_shape = [common_lane_count]
    else:
        domain_lane_shape = [lane_ratio]
        if smaller_lane_count != 1:
            domain_lane_shape.append(smaller_lane_count)
    ctx = _coordinate_iteration_context(
        domain_shape,
        domain_strides,
        domain_lane_shape,
    )
    domain_lane_coordinates = ctx["lane_coordinates"]
    common_lane_index = _flatten_coordinates(domain_lane_coordinates, ctx["lane_shape"])
    prefix_index = domain_lane_coordinates[0] if lane_ratio != 1 else "0"
    smaller_lane_index = (
        domain_lane_coordinates[-1] if smaller_lane_count != 1 else "0"
    )

    def operand_access(
        prefix: str, lane_count: int, lane_shape: tuple[int, ...]
    ) -> dict[str, Any]:
        tensor_coordinates = list(ctx["tensor_coordinates"])
        if lane_count != common_lane_count:
            tensor_coordinates[vectorized_axis] = (
                f"({tensor_coordinates[vectorized_axis]}) * {lane_ratio} + {prefix_index}"
            )
        lane_coordinates: tuple[str, ...]
        if lane_count == 1:
            lane_coordinates = ()
        elif lane_count == common_lane_count:
            lane_coordinates = (common_lane_index,)
        else:
            lane_coordinates = (smaller_lane_index,)
        return _with_major_boundary_mask(
            _tensor_access(
                tensor_coordinates,
                model[f"{prefix}Strides"],
                lane_coordinates,
                lane_shape,
            ),
            model[f"{prefix}Shape"],
            model[f"{prefix}Strides"],
            ctx["major_axis"],
        )

    ctx["input_access"] = operand_access("Input", input_lane_count, input_lanes)
    ctx["output_access"] = operand_access("Output", output_lane_count, output_lanes)
    return ctx


def _where_operand_access(
    model: dict[str, Any],
    prefix: str,
    ctx: dict[str, Any],
) -> dict[str, Any]:
    shape = model[f"{prefix}Shape"]
    strides = model[f"{prefix}Strides"]
    lanes = tuple(int(value) for value in model[f"{prefix}VectorLaneShape"])
    output_shape = model["OutputShape"]
    output_lanes = ctx["lane_shape"]
    if lanes and lanes != output_lanes:
        raise ValueError(
            "PyNTT Where vector operands must be scalar or match output lanes: "
            f"{prefix}={lanes}, output={output_lanes}"
        )
    if len(shape) > len(output_shape):
        raise ValueError(
            f"PyNTT Where {prefix} rank exceeds output rank: "
            f"operand={len(shape)}, output={len(output_shape)}"
        )
    axis_offset = len(output_shape) - len(shape)
    output_lane_count = _product_int(list(output_lanes)) if output_lanes else 1
    output_lane_index = _flatten_coordinates(ctx["lane_coordinates"], output_lanes)
    tensor_coordinates: list[str] = []
    for axis, extent in enumerate(shape):
        output_axis = axis_offset + axis
        coordinate = ctx["tensor_coordinates"][output_axis]
        if (
            not lanes
            and output_lane_count != 1
            and output_axis == len(output_shape) - 1
        ):
            coordinate = f"({coordinate}) * {output_lane_count} + {output_lane_index}"
        tensor_coordinates.append("0" if _is_fixed_one(extent) else coordinate)
    lane_coordinates = ctx["lane_coordinates"] if lanes else ()
    access = _tensor_access(tensor_coordinates, strides, lane_coordinates, lanes)
    return _with_major_boundary_mask(
        access, shape, strides, ctx["major_axis"] - axis_offset
    )


def _elementwise_where_template_context(model: dict[str, Any]) -> dict[str, Any]:
    ctx = _coordinate_iteration_context(
        model["OutputShape"],
        model["OutputStrides"],
        model["OutputVectorLaneShape"],
    )
    for prefix in ("Cond", "True", "False"):
        ctx[f"{prefix.lower()}_access"] = _where_operand_access(model, prefix, ctx)
    ctx["output_access"] = _with_major_boundary_mask(
        _tensor_access(
            ctx["tensor_coordinates"],
            model["OutputStrides"],
            ctx["lane_coordinates"],
            ctx["lane_shape"],
        ),
        model["OutputShape"],
        model["OutputStrides"],
        ctx["major_axis"],
    )
    return ctx


def _vector_layout_template_context(model: dict[str, Any]) -> dict[str, Any]:
    """Prepare coordinate-native Pack/Unpack access mappings."""

    input_shape = model["InputShape"]
    output_shape = model["OutputShape"]
    if len(input_shape) != len(output_shape):
        raise ValueError(
            "PyNTT VectorLayout requires equal input/output tensor ranks: "
            f"input={len(input_shape)}, output={len(output_shape)}"
        )

    input_lanes = _validate_coordinate_lane_shape(
        model["InputLanes"], "PyNTT VectorLayout input"
    )
    output_lanes = _validate_coordinate_lane_shape(
        model["OutputLanes"], "PyNTT VectorLayout output"
    )
    packed_lanes = _validate_coordinate_lane_shape(
        model["Lanes"], "PyNTT VectorLayout packed"
    )
    axes = tuple(int(value) for value in model["Axes"])
    if len(axes) != len(packed_lanes):
        raise ValueError(
            "PyNTT VectorLayout axes/lanes count mismatch: "
            f"axes={len(axes)}, lanes={len(packed_lanes)}"
        )
    if any(axis < 0 or axis >= len(input_shape) for axis in axes):
        raise ValueError(
            f"PyNTT VectorLayout axis is outside rank {len(input_shape)}: {axes}"
        )

    is_pack = bool(model["IsPack"])
    expected_lanes = packed_lanes + (input_lanes if is_pack else output_lanes)
    actual_lanes = output_lanes if is_pack else input_lanes
    if actual_lanes != expected_lanes:
        side = "output" if is_pack else "input"
        raise ValueError(
            f"PyNTT {'Pack' if is_pack else 'Unpack'} {side} lanes must be "
            f"the packed-lane prefix followed by the preserved lanes: "
            f"expected={expected_lanes}, actual={actual_lanes}"
        )

    domain_shape = output_shape if is_pack else input_shape
    domain_strides = model["OutputStrides"] if is_pack else model["InputStrides"]
    domain_lanes = output_lanes if is_pack else input_lanes
    ctx = _coordinate_iteration_context(
        domain_shape,
        domain_strides,
        list(domain_lanes),
        f"PyNTT {'Pack' if is_pack else 'Unpack'}",
    )

    expanded_tensor_coordinates = list(ctx["tensor_coordinates"])
    bounds = []
    for axis in range(len(input_shape)):
        lane_indices = [
            lane_index
            for lane_index, packed_axis in enumerate(axes)
            if packed_axis == axis
        ]
        if not lane_indices:
            continue

        lane_product = _product_int([packed_lanes[index] for index in lane_indices])
        terms = []
        for position, lane_index in enumerate(lane_indices):
            lane_stride = _product_int(
                [packed_lanes[index] for index in lane_indices[position + 1 :]]
            )
            coordinate = ctx["lane_coordinates"][lane_index]
            terms.append(
                coordinate if lane_stride == 1 else f"({coordinate}) * {lane_stride}"
            )
        base = ctx["tensor_coordinates"][axis]
        base = base if lane_product == 1 else f"({base}) * {lane_product}"
        expanded = base if not terms else f"{base} + {' + '.join(terms)}"
        expanded_tensor_coordinates[axis] = expanded
        bound_shape = input_shape if is_pack else output_shape
        bounds.append(f"({expanded}) < {_dim(bound_shape[axis])}")

    preserved_lane_coordinates = ctx["lane_coordinates"][len(packed_lanes) :]
    if is_pack:
        input_access = _tensor_access(
            expanded_tensor_coordinates,
            model["InputStrides"],
            preserved_lane_coordinates,
            input_lanes,
        )
        output_access = _tensor_access(
            ctx["tensor_coordinates"],
            model["OutputStrides"],
            ctx["lane_coordinates"],
            output_lanes,
        )
    else:
        input_access = _tensor_access(
            ctx["tensor_coordinates"],
            model["InputStrides"],
            ctx["lane_coordinates"],
            input_lanes,
        )
        output_access = _tensor_access(
            expanded_tensor_coordinates,
            model["OutputStrides"],
            preserved_lane_coordinates,
            output_lanes,
        )

    ctx.update(
        bounds=tuple(bounds),
        input_access=input_access,
        op="pack" if is_pack else "unpack",
        output_access=output_access,
        store_mask="mask" if is_pack else "valid",
        valid_expression="mask" + "".join(f" & ({bound})" for bound in bounds),
    )
    return ctx


def _transpose_template_context(model: dict[str, Any]) -> dict[str, Any]:
    """Prepare a coordinate-native tensor-axis permutation."""

    input_shape = model["InputShape"]
    output_shape = model["OutputShape"]
    permutation = tuple(int(value) for value in model["Perm"])
    rank = len(input_shape)
    if len(output_shape) != rank or sorted(permutation) != list(range(rank)):
        raise ValueError(
            "PyNTT Transpose requires equal input/output ranks and a complete "
            f"permutation: input={rank}, output={len(output_shape)}, perm={permutation}"
        )

    input_lanes = _validate_coordinate_lane_shape(
        model["InputVectorLaneShape"], "PyNTT Transpose input"
    )
    output_lanes = _validate_coordinate_lane_shape(
        model["OutputVectorLaneShape"], "PyNTT Transpose output"
    )
    if input_lanes != output_lanes:
        raise ValueError(
            "PyNTT Transpose must preserve vector lanes: "
            f"input={input_lanes}, output={output_lanes}"
        )

    ctx = _coordinate_iteration_context(
        output_shape,
        model["OutputStrides"],
        list(output_lanes),
        "PyNTT Transpose",
    )
    input_coordinates = ["0"] * rank
    for output_axis, input_axis in enumerate(permutation):
        input_coordinates[input_axis] = ctx["tensor_coordinates"][output_axis]
    ctx["input_access"] = _tensor_access(
        input_coordinates,
        model["InputStrides"],
        ctx["lane_coordinates"],
        input_lanes,
    )
    ctx["output_access"] = _tensor_access(
        ctx["tensor_coordinates"],
        model["OutputStrides"],
        ctx["lane_coordinates"],
        output_lanes,
    )
    return ctx


def _aligned_batch_coordinates(
    operand_shape: list[Any], trailing_rank: int, output_batch_rank: int
) -> tuple[str, ...]:
    """Map an operand's broadcast batch axes to the surrounding output loops."""

    operand_batch_rank = len(operand_shape) - trailing_rank
    if operand_batch_rank < 0 or operand_batch_rank > output_batch_rank:
        raise ValueError(
            "PyNTT operand batch rank cannot be aligned to the output: "
            f"operand_rank={len(operand_shape)}, trailing_rank={trailing_rank}, "
            f"output_batch_rank={output_batch_rank}"
        )
    axis_offset = output_batch_rank - operand_batch_rank
    return tuple(
        "0" if _is_fixed_one(operand_shape[axis]) else f"idx{axis_offset + axis}"
        for axis in range(operand_batch_rank)
    )


def _structured_axis_tile(
    name: str,
    lane_shape: tuple[int, ...] | list[int],
    scalar_block_extent: int | str,
    logical_extent: Any,
    *,
    leading_rank: int = 0,
    trailing_rank: int = 0,
    physical_base: str = "0",
) -> dict[str, Any]:
    """Describe one rectangular physical/vector axis tile for Jinja."""

    lanes = _validate_coordinate_lane_shape(list(lane_shape), f"PyNTT {name}")
    lane_count = _product_int(list(lanes)) if lanes else 1
    if isinstance(scalar_block_extent, int):
        if scalar_block_extent <= 0 or scalar_block_extent % lane_count != 0:
            raise ValueError(
                f"PyNTT {name} scalar tile must be a positive multiple of its "
                f"vector lanes: block={scalar_block_extent}, lanes={lanes}."
            )
        physical_block_extent: int | str = scalar_block_extent // lane_count
    else:
        if not scalar_block_extent:
            raise ValueError(f"PyNTT {name} scalar tile expression is empty.")
        physical_block_extent = (
            scalar_block_extent
            if lane_count == 1
            else f"(({scalar_block_extent}) // {lane_count})"
        )
    if leading_rank < 0 or trailing_rank < 0:
        raise ValueError(
            f"PyNTT {name} structured-axis ranks must be non-negative: "
            f"leading={leading_rank}, trailing={trailing_rank}."
        )

    physical_position = leading_rank
    rank = leading_rank + 1 + len(lanes) + trailing_rank
    physical_coordinate = f"{name}_physical"
    lane_coordinates = tuple(f"{name}_lane{axis}" for axis in range(len(lanes)))
    lane_terms: list[str] = []
    lane_stride = lane_count
    for coordinate, extent in zip(lane_coordinates, lanes):
        lane_stride //= extent
        lane_terms.append(
            coordinate if lane_stride == 1 else f"({coordinate}) * {lane_stride}"
        )
    logical_terms = [
        physical_coordinate
        if lane_count == 1
        else f"({physical_coordinate}) * {lane_count}"
    ]
    logical_terms.extend(lane_terms)
    structured_shape = (physical_block_extent,) + lanes
    return {
        "lane_coordinates": lane_coordinates,
        "lane_count": lane_count,
        "lane_shape": lanes,
        "logical_coordinate": f"{name}_logical",
        "logical_expression": " + ".join(logical_terms),
        "logical_extent": logical_extent,
        "name": name,
        "physical_base": physical_base,
        "physical_block_extent": physical_block_extent,
        "physical_coordinate": physical_coordinate,
        "physical_position": physical_position,
        "rank": rank,
        "scalar_block_extent": scalar_block_extent,
        "structured_shape": structured_shape,
    }


def _broadcast_axis_coordinate(expression: str, rank: int, axis: int) -> str:
    if rank <= 0 or axis < 0 or axis >= rank:
        raise ValueError(
            "PyNTT broadcast-axis coordinate is outside its tensor rank: "
            f"rank={rank}, axis={axis}."
        )
    if rank == 1:
        return expression
    indices = ["None"] * rank
    indices[axis] = ":"
    return f"{expression}[{', '.join(indices)}]"


def _structured_value_shape(
    axis: dict[str, Any],
    *,
    leading_extents: tuple[int, ...] = (),
    trailing_extents: tuple[int, ...] = (),
) -> tuple[int, ...]:
    if len(leading_extents) != axis["physical_position"]:
        raise ValueError(
            f"PyNTT {axis['name']} leading value rank mismatch: "
            f"expected={axis['physical_position']}, got={len(leading_extents)}."
        )
    expected_trailing = (
        axis["rank"] - axis["physical_position"] - 1 - len(axis["lane_shape"])
    )
    if len(trailing_extents) != expected_trailing:
        raise ValueError(
            f"PyNTT {axis['name']} trailing value rank mismatch: "
            f"expected={expected_trailing}, got={len(trailing_extents)}."
        )
    return leading_extents + axis["structured_shape"] + trailing_extents


def _qkv_packed_lane_shape(model: dict[str, Any], *, packed: bool) -> tuple[int, ...]:
    return (
        (int(model["NPackedLaneCount"]), int(model["NVectorLaneCount"]))
        if packed
        else ()
    )


def _qkv_weight_lane_shape(
    model: dict[str, Any], *, packed: bool
) -> tuple[int, ...]:
    if not packed:
        return ()
    n_lane_shape = _qkv_packed_lane_shape(model, packed=True)
    if model.get("RhsLayout", "n_major") == "n_major":
        return n_lane_shape
    return (
        *n_lane_shape,
        int(model["KPackLaneCount"]),
        int(model["KVectorLaneCount"]),
    )


def _qkv_input_access(
    model: dict[str, Any],
    output_batch_rank: int,
    m_expr: str,
    k_expr: str,
    coordinate_shape: str,
) -> dict[str, Any]:
    coordinates = _aligned_batch_coordinates(
        model["InputShape"], 2, output_batch_rank
    ) + (m_expr, k_expr)
    return _tensor_access(
        coordinates, model["InputStrides"], coordinate_shape=coordinate_shape
    )


def _qkv_weight_access(
    model: dict[str, Any],
    prefix: str,
    *,
    packed: bool,
    output_batch_rank: int,
    n_axis: dict[str, Any],
    k_expr: str,
    coordinate_shape: str,
    physical_n_base: int = 0,
) -> dict[str, Any]:
    n_lane_shape = _qkv_packed_lane_shape(model, packed=packed)
    if tuple(n_lane_shape) != tuple(n_axis["lane_shape"]):
        raise ValueError(
            f"PyNTT {prefix} QKV weight lane shape does not match its N tile: "
            f"weight N={n_lane_shape}, tile={n_axis['lane_shape']}."
        )
    weight_lane_shape = _qkv_weight_lane_shape(model, packed=packed)
    weight_shape_key = "WeightShape" if packed else f"{prefix}WeightShape"
    weight_strides_key = "WeightStrides" if packed else f"{prefix}WeightStrides"
    batch_coordinates = _aligned_batch_coordinates(
        model[weight_shape_key], 2, output_batch_rank
    )
    physical_n_coordinate = n_axis["physical_coordinate"]
    if physical_n_base:
        physical_n_coordinate = f"({physical_n_coordinate}) + {physical_n_base}"
    if not packed:
        matrix_coordinates = (k_expr, physical_n_coordinate)
        lane_coordinates = ()
    elif model.get("RhsLayout", "n_major") == "n_major":
        matrix_coordinates = (physical_n_coordinate, k_expr)
        lane_coordinates = n_axis["lane_coordinates"]
    else:
        k_pack = int(model["KPackLaneCount"])
        k_lane = int(model["KVectorLaneCount"])
        k_atom = k_pack * k_lane
        matrix_coordinates = (
            f"({k_expr}) // {k_atom}",
            physical_n_coordinate,
        )
        lane_coordinates = (
            *n_axis["lane_coordinates"],
            f"(({k_expr}) // {k_lane}) % {k_pack}",
            f"({k_expr}) % {k_lane}",
        )
    return _tensor_access(
        batch_coordinates + matrix_coordinates,
        model[weight_strides_key],
        lane_coordinates,
        weight_lane_shape,
        coordinate_shape,
    )


def _qkv_output_access(
    model: dict[str, Any],
    prefix: str,
    *,
    packed: bool,
    output_batch_rank: int,
    m_expr: str,
    n_axis: dict[str, Any],
    coordinate_shape: str,
) -> dict[str, Any]:
    lane_shape = _qkv_packed_lane_shape(model, packed=packed)
    if tuple(lane_shape) != tuple(n_axis["lane_shape"]):
        raise ValueError(
            f"PyNTT {prefix} QKV output lane shape does not match its N tile: "
            f"output={lane_shape}, tile={n_axis['lane_shape']}."
        )
    coordinates = _aligned_batch_coordinates(
        model[f"{prefix}OutputShape"], 2, output_batch_rank
    ) + (m_expr, n_axis["physical_coordinate"])
    return _tensor_access(
        coordinates,
        model[f"{prefix}OutputStrides"],
        n_axis["lane_coordinates"],
        lane_shape,
        coordinate_shape,
    )


def _qkv_bias_access(
    model: dict[str, Any],
    prefix: str,
    *,
    packed: bool,
    n_axis: dict[str, Any],
    coordinate_shape: str,
) -> dict[str, Any]:
    lane_shape = _qkv_packed_lane_shape(model, packed=packed)
    if tuple(lane_shape) != tuple(n_axis["lane_shape"]):
        raise ValueError(
            f"PyNTT {prefix} QKV bias lane shape does not match its N tile: "
            f"bias={lane_shape}, tile={n_axis['lane_shape']}."
        )
    return _tensor_access(
        (n_axis["physical_coordinate"],),
        model[f"{prefix}BiasStrides"],
        n_axis["lane_coordinates"],
        lane_shape,
        coordinate_shape,
    )


def _microkernel_context(
    model: dict[str, Any],
    expected_family: str,
    expected_variant: str | None = None,
    *,
    required_workspace_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    microkernel = model.get("MicroKernel")
    if not isinstance(microkernel, dict):
        raise ValueError(
            f"PyNTT helper requires selected microkernel {expected_family!r}."
        )
    family = _require_string(
        microkernel.get("Family"), "microkernel.Family", nonempty=True
    )
    variant = _require_string(
        microkernel.get("Variant"), "microkernel.Variant", nonempty=True
    )
    if family != expected_family:
        raise ValueError(
            f"PyNTT helper expects microkernel family {expected_family!r}, "
            f"got {family!r}."
        )
    if expected_variant is not None and variant != expected_variant:
        raise ValueError(
            f"PyNTT helper expects microkernel variant {expected_variant!r}, "
            f"got {variant!r}."
        )

    raw_parameters = microkernel.get("Parameters")
    if not isinstance(raw_parameters, dict):
        raise ValueError("microkernel.Parameters must be a JSON object.")
    parameters = {
        str(name): _require_int(
            value, f"microkernel.Parameters[{name!r}]", minimum=1
        )
        for name, value in raw_parameters.items()
    }
    for name in ("block_m", "block_n", "block_k", "num_stages"):
        if name not in parameters:
            raise ValueError(f"microkernel.Parameters is missing {name!r}.")

    raw_offsets = microkernel.get("SharedWorkspaceOffsets")
    if not isinstance(raw_offsets, dict):
        raise ValueError("microkernel.SharedWorkspaceOffsets must be a JSON object.")
    offsets = {
        str(name): _validate_python_expression(
            value, f"microkernel.SharedWorkspaceOffsets[{name!r}]"
        )
        for name, value in raw_offsets.items()
    }
    raw_shapes = microkernel.get("SharedWorkspaceShapes")
    if not isinstance(raw_shapes, dict):
        raise ValueError("microkernel.SharedWorkspaceShapes must be a JSON object.")
    shapes = {}
    for name, value in raw_shapes.items():
        dimensions = _require_list(
            value, f"microkernel.SharedWorkspaceShapes[{name!r}]"
        )
        shapes[str(name)] = tuple(
            _require_int(
                dimension,
                f"microkernel.SharedWorkspaceShapes[{name!r}][{axis}]",
                minimum=1,
            )
            for axis, dimension in enumerate(dimensions)
        )
    has_complete_consumer_lhs_stage = "lhs_stage_extent" in parameters
    if has_complete_consumer_lhs_stage and not (
        family in ("triton.matmul", "triton.matmul_glu")
        and variant == "simt_fma_smem_pipeline"
    ):
        raise ValueError(
            "microkernel.Parameters['lhs_stage_extent'] is only valid for "
            "triton.matmul or triton.matmul_glu with "
            "simt_fma_smem_pipeline."
        )
    required_offsets = required_workspace_names
    if required_offsets is None:
        required_offsets = {
            "simt_fma": (),
            "simt_fma_smem_pipeline": (
                ("lhs_stage", "rhs_stage")
                if family == "triton.qkv_parallel_linear"
                or has_complete_consumer_lhs_stage
                else ("rhs_stage",)
            ),
            "simt_fp8_fma_smem_pipeline": (
                ("lhs_stage", "rhs_stage")
                if family == "triton.qkv_parallel_linear"
                or has_complete_consumer_lhs_stage
                else ("rhs_stage",)
            ),
            "simt_block_fp8_fma_smem_pipeline": (
                ("lhs_stage", "rhs_stage")
                if family == "triton.qkv_parallel_linear"
                or has_complete_consumer_lhs_stage
                else ("rhs_stage",)
            ),
            "mma_smem_pipeline": ("lhs_stage", "rhs_stage"),
            "mma_block_fp8_smem_pipeline": (
                "rhs_stage",
                "lhs_quantized",
                "lhs_scale",
            ),
            "mma": ("lhs_stage", "rhs_stage"),
            "dot": ("lhs_stage", "rhs_stage"),
            "mma_direct": (),
            "simt_direct": (),
            "mma_tma_smem_pipeline": ("key_stage", "value_stage"),
            "simt_tma_smem_pipeline": ("key_stage", "value_stage"),
        }.get(variant)
    if required_offsets is None:
        raise ValueError(f"Unsupported PyNTT microkernel variant {variant!r}.")
    if set(offsets) != set(required_offsets):
        raise ValueError(
            "microkernel.SharedWorkspaceOffsets does not match variant "
            f"{variant!r}: expected={list(required_offsets)}, "
            f"actual={sorted(offsets)}."
        )
    if set(shapes) != set(required_offsets):
        raise ValueError(
            "microkernel.SharedWorkspaceShapes does not match variant "
            f"{variant!r}: expected={list(required_offsets)}, "
            f"actual={sorted(shapes)}."
        )

    raw_consumer_workspaces = microkernel.get("ConsumerSharedWorkspaceNames")
    if not isinstance(raw_consumer_workspaces, list) or any(
        not isinstance(name, str) or not name
        for name in raw_consumer_workspaces
    ):
        raise ValueError(
            "microkernel.ConsumerSharedWorkspaceNames must be an array of "
            "non-empty strings."
        )
    consumer_workspaces = tuple(raw_consumer_workspaces)
    if len(set(consumer_workspaces)) != len(consumer_workspaces):
        raise ValueError(
            "microkernel.ConsumerSharedWorkspaceNames must be unique."
        )
    expected_consumer_workspaces = (
        ("lhs_stage",)
        if (
            variant == "mma_smem_pipeline"
            or (
                variant == "simt_fma_smem_pipeline"
                and (
                    family == "triton.qkv_parallel_linear"
                    or has_complete_consumer_lhs_stage
                )
            )
        )
        else ()
    )
    if consumer_workspaces != expected_consumer_workspaces:
        raise ValueError(
            "microkernel.ConsumerSharedWorkspaceNames does not match the "
            f"selected algorithm: expected={list(expected_consumer_workspaces)}, "
            f"actual={list(consumer_workspaces)}."
        )

    return {
        "family": family,
        "variant": variant,
        "parameters": parameters,
        "shared_workspace_offsets": offsets,
        "shared_workspace_shapes": shapes,
        "consumer_shared_workspace_names": consumer_workspaces,
    }


def _nvfp4_projection_template_context(
    model: dict[str, Any],
    *,
    expected_family: str,
    n_tiles_per_activation_batch: int,
    input_shape_key: str,
    input_strides_key: str,
    packed_weight_shape_key: str,
    weight_scale_shape_key: str,
    packed_weight_pointer_keys: tuple[str, ...],
) -> dict[str, Any]:
    """Validate the selected decode NVFP4 projection's physical storage ABI."""

    if n_tiles_per_activation_batch < 1:
        raise ValueError(
            "PyNTT NVFP4 activation batch must contain at least one N tile."
        )

    microkernel = _microkernel_context(
        model,
        expected_family,
        "mma_tma_smem_pipeline",
        required_workspace_names=("packed_weight_stage",),
    )
    group_size = _require_int(model["GroupSize"], "NVFP4 GroupSize", minimum=1)
    parameter_group_size = microkernel["parameters"].get("group_size")
    if group_size != 16 or parameter_group_size != group_size:
        raise ValueError(
            "PyNTT NVFP4 projection requires group size 16 in both the op and "
            f"selected microkernel, got op={group_size}, microkernel={parameter_group_size}."
        )

    input_shape = _require_list(model[input_shape_key], input_shape_key)
    weight_shape = _require_list(
        model[packed_weight_shape_key], packed_weight_shape_key
    )
    scale_shape = _require_list(model[weight_scale_shape_key], weight_scale_shape_key)
    output_shape = _require_list(model["OutputShape"], "OutputShape")
    for name, shape in (
        (input_shape_key, input_shape),
        (packed_weight_shape_key, weight_shape),
        (weight_scale_shape_key, scale_shape),
        ("OutputShape", output_shape),
    ):
        if len(shape) != 2:
            raise ValueError(f"PyNTT NVFP4 {name} must have rank 2, got {len(shape)}.")

    input_lanes = tuple(
        int(value) for value in _require_list(model["LhsVectorLanes"] if input_shape_key == "LhsShape" else model["InputVectorLanes"], "NVFP4 input vector lanes")
    )
    weight_lanes = tuple(
        int(value) for value in _require_list(
            model["RhsPackedVectorLanes"] if packed_weight_shape_key == "RhsPackedShape" else model["WeightPackedVectorLanes"],
            "NVFP4 packed-weight vector lanes",
        )
    )
    output_lanes = tuple(
        int(value) for value in _require_list(model["OutputVectorLanes"], "NVFP4 output vector lanes")
    )
    if input_lanes != (8,) or weight_lanes != (2, 16) or output_lanes != (8,):
        raise ValueError(
            "PyNTT NVFP4 requires input/weight/output vector lanes "
            f"(8)/(2,16)/(8), got {input_lanes}/{weight_lanes}/{output_lanes}."
        )

    m = _require_fixed_positive_dim(input_shape[0], f"{input_shape_key}[0]")
    input_k_outer = _require_fixed_positive_dim(
        input_shape[1], f"{input_shape_key}[1]"
    )
    k = input_k_outer * _product_int(list(input_lanes))
    logical_n = _multiply_dim(output_shape[1], _product_int(list(output_lanes)))
    n_min = _min_value(logical_n)
    n_max = _max_value(logical_n)
    if n_min is None or n_max is None or n_min <= 0 or n_max < n_min:
        raise ValueError(
            "PyNTT NVFP4 output N must have a bounded positive local domain, "
            f"got {_dim(output_shape[1])}."
        )
    output_m = _require_fixed_positive_dim(output_shape[0], "OutputShape[0]")
    packed_k = _require_fixed_positive_dim(
        weight_shape[1], f"{packed_weight_shape_key}[1]"
    )
    scale_k = _require_fixed_positive_dim(
        scale_shape[1], f"{weight_scale_shape_key}[1]"
    )
    if m != 1 or output_m != 1:
        raise ValueError(
            f"PyNTT NVFP4 decode projection requires local M=1, got input={m}, output={output_m}."
        )
    if k % group_size != 0:
        raise ValueError(f"PyNTT NVFP4 local K must be divisible by {group_size}, got {k}.")
    if (
        not _dimensions_equivalent(weight_shape[0], logical_n)
        or not _dimensions_equivalent(scale_shape[0], logical_n)
        or packed_k * _product_int(list(weight_lanes)) * 2 != k
        or scale_k * group_size != k
    ):
        raise ValueError(
            "PyNTT NVFP4 physical storage must be input=bf16<8>[1,K/8], "
            "packed_weight=u8<2,16>[N,K/64], "
            f"weight_scale=[N,K/{group_size}], output=bf16<8>[1,N/8]; got "
            f"input={[m, input_k_outer]}, weight=[{_dim(weight_shape[0])}, {packed_k}], "
            f"scale=[{_dim(scale_shape[0])}, {scale_k}], "
            f"output=[{output_m}, {_dim(output_shape[1])}]."
        )

    for key in (input_strides_key, "OutputStrides"):
        if len(_require_list(model[key], key)) != 2:
            raise ValueError(f"PyNTT NVFP4 {key} must have rank 2.")

    block_n = microkernel["parameters"]["block_n"]
    block_k = microkernel["parameters"]["block_k"]
    if not _is_positive_power_of_two(block_n) or not _is_positive_power_of_two(block_k):
        raise ValueError(
            f"PyNTT NVFP4 block sizes must be powers of two, got N={block_n}, K={block_k}."
        )
    if block_k % group_size != 0:
        raise ValueError(
            f"PyNTT NVFP4 block K must be divisible by group size {group_size}, got {block_k}."
        )
    num_stages = microkernel["parameters"]["num_stages"]
    if num_stages < 2:
        raise ValueError(
            f"PyNTT NVFP4 TMA pipeline requires at least two stages, got {num_stages}."
        )
    expected_workspace_shapes = {
        "packed_weight_stage": (num_stages, block_n, block_k // 2),
    }
    if microkernel["shared_workspace_shapes"] != expected_workspace_shapes:
        raise ValueError(
            "PyNTT NVFP4 Shared workspace shapes disagree with the selected "
            f"pipeline: expected={expected_workspace_shapes}, "
            f"actual={microkernel['shared_workspace_shapes']}."
        )

    packed_k_atom = _product_int(list(weight_lanes))
    packed_block_k = block_k // 2
    if packed_block_k % packed_k_atom != 0:
        raise ValueError(
            "PyNTT NVFP4 packed block K must contain complete vector atoms, "
            f"got packed_block_k={packed_block_k}, atom={packed_k_atom}."
        )
    packed_block_k_outer = packed_block_k // packed_k_atom
    packed_plans = []
    for pointer_key in packed_weight_pointer_keys:
        pointer = model.get(pointer_key)
        if not isinstance(pointer, dict):
            raise ValueError(
                f"PyNTT NVFP4 {pointer_key} pointer metadata must be an object."
            )
        n_plan = _n_major_k_packed_gemv_descriptor_n_plan(pointer, block_n)
        k_payload_plan = _tma_packed_atom_axis_plan(
            pointer,
            1,
            tile_extent=packed_block_k_outer,
            atom_extent=packed_k_atom,
            logical_axis_stride=1,
            context=f"NVFP4 {pointer_key} TMA descriptor K",
        )
        k_plan = k_payload_plan["axis_plan"]
        tma_block_shape = (
            tuple(n_plan["block_shape"])
            + tuple(k_payload_plan["block_shape"])
        )
        if len(tma_block_shape) > 5:
            raise ValueError(
                "PyNTT NVFP4 TMA descriptor exceeds the hardware rank-5 limit: "
                f"pointer={pointer_key}, block_shape={tma_block_shape}."
            )
        if _product_int(list(tma_block_shape)) != block_n * packed_block_k:
            raise ValueError(
                "PyNTT NVFP4 TMA block does not cover the selected Shared tile: "
                f"pointer={pointer_key}, block_shape={tma_block_shape}, "
                f"expected_elements={block_n * packed_block_k}."
            )
        tma_offsets = (
            tuple("0" for _ in n_plan["block_shape"])
            + _tma_packed_atom_coordinates(
                f"k_tile * {packed_block_k_outer}", k_payload_plan
            )
        )
        packed_plans.append(
            (
                pointer_key,
                n_plan,
                k_plan,
                k_payload_plan,
                tma_block_shape,
                tma_offsets,
            )
        )

    if not packed_plans:
        raise ValueError("PyNTT NVFP4 projection requires a packed-weight pointer.")
    (
        _,
        packed_n_plan,
        packed_k_plan,
        packed_k_payload_plan,
        packed_tma_block_shape,
        packed_tma_offsets,
    ) = packed_plans[0]
    for (
        pointer_key,
        n_plan,
        k_plan,
        k_payload_plan,
        tma_block_shape,
        tma_offsets,
    ) in packed_plans[1:]:
        if (
            tuple(n_plan["block_shape"]) != tuple(packed_n_plan["block_shape"])
            or tuple(k_plan["block_shape"]) != tuple(packed_k_plan["block_shape"])
            or tuple(k_payload_plan["block_shape"])
            != tuple(packed_k_payload_plan["block_shape"])
            or tma_block_shape != packed_tma_block_shape
            or tma_offsets != packed_tma_offsets
        ):
            raise ValueError(
                "PyNTT fused NVFP4 projections require one shared TMA tile ABI, "
                f"but {pointer_key} uses {tma_block_shape} and the first "
                f"projection uses {packed_tma_block_shape}."
            )

    num_n_tiles = (n_max + block_n - 1) // block_n
    num_k_tiles = (k + block_k - 1) // block_k
    if num_n_tiles * num_k_tiles > 2**31 - 1:
        raise ValueError("PyNTT NVFP4 MMA pipe sequence exceeds signed int32.")

    return {
        "microkernel": microkernel,
        "m": input_shape[0],
        "n": logical_n,
        "k": _multiply_dim(input_shape[1], _product_int(list(input_lanes))),
        "max_n": n_max,
        "fixed_k": k,
        "block_n": block_n,
        "block_k": block_k,
        "num_stages": num_stages,
        "num_n_tiles": num_n_tiles,
        "num_k_tiles": num_k_tiles,
        "n_tiles_per_activation_batch": n_tiles_per_activation_batch,
        "packed_block_k": packed_block_k,
        "packed_block_k_outer": packed_block_k_outer,
        "packed_k_atom": packed_k_atom,
        "packed_n_plan": packed_n_plan,
        "packed_k_plan": packed_k_plan,
        "packed_k_payload_plan": packed_k_payload_plan,
        "packed_tma_block_shape": packed_tma_block_shape,
        "packed_tma_offsets": packed_tma_offsets,
        "scale_block_k": block_k // group_size,
        "group_size": group_size,
        "input_access": _tensor_access(
            ("0", f"offs_k // {input_lanes[0]}"),
            model[input_strides_key],
            (f"offs_k % {input_lanes[0]}",),
            input_lanes,
            coordinate_shape=_coordinate_shape((block_k,)),
        ),
        "output_access": _tensor_access(
            ("0", f"offs_n // {output_lanes[0]}"),
            model["OutputStrides"],
            (f"offs_n % {output_lanes[0]}",),
            output_lanes,
            coordinate_shape=_coordinate_shape((block_n,)),
        ),
    }


def _nvfp4_matmul_template_context(model: dict[str, Any]) -> dict[str, Any]:
    context = _nvfp4_projection_template_context(
        model,
        expected_family="triton.nvfp4_matmul",
        n_tiles_per_activation_batch=2,
        input_shape_key="LhsShape",
        input_strides_key="LhsStrides",
        packed_weight_shape_key="RhsPackedShape",
        weight_scale_shape_key="RhsScaleShape",
        packed_weight_pointer_keys=("RhsPacked",),
    )
    context.update(
        packed_descriptor_name=_require_string(
            model.get("RhsPackedDescriptorName"),
            "RhsPackedDescriptorName",
            nonempty=True,
        ),
        packed_weight_access=_tensor_access(
            (
                "offs_n[:, None]",
                "(k_start // 2 + packed_k[None, :]) // 32",
            ),
            model["RhsPackedStrides"],
            (
                "((k_start // 2 + packed_k[None, :]) % 32) // 16",
                "(k_start // 2 + packed_k[None, :]) % 16",
            ),
            (2, 16),
            coordinate_shape=_coordinate_shape(
                (context["block_n"], context["packed_block_k"])
            ),
        ),
        weight_scale_access=_tensor_access(
            (
                "offs_n[:, None]",
                f"k_start // {context['group_size']} + scale_k[None, :]",
            ),
            model["RhsScaleStrides"],
            coordinate_shape=_coordinate_shape(
                (context["block_n"], context["scale_block_k"])
            ),
        ),
    )
    return context


def _nvfp4_matmul_glu_template_context(model: dict[str, Any]) -> dict[str, Any]:
    context = _nvfp4_projection_template_context(
        model,
        expected_family="triton.nvfp4_matmul_glu",
        n_tiles_per_activation_batch=1,
        input_shape_key="InputShape",
        input_strides_key="InputStrides",
        packed_weight_shape_key="WeightPackedShape",
        weight_scale_shape_key="WeightScaleShape",
        packed_weight_pointer_keys=("GateWeightPacked", "UpWeightPacked"),
    )
    if model["GluType"] != "swiglu":
        raise ValueError(
            f"PyNTT NVFP4MatMulGlu supports SwiGLU, got {model['GluType']!r}."
        )
    for key in (
        "GateWeightPackedStrides",
        "UpWeightPackedStrides",
        "GateWeightScaleStrides",
        "UpWeightScaleStrides",
    ):
        if len(_require_list(model[key], key)) != 2:
            raise ValueError(f"PyNTT NVFP4MatMulGlu {key} must have rank 2.")
    context.update(
        projections=tuple(
            {
                "prefix": prefix,
                "lower": prefix.lower(),
                "sequence_offset": index,
                "packed_descriptor_name": _require_string(
                    model.get(f"{prefix}WeightPackedDescriptorName"),
                    f"{prefix}WeightPackedDescriptorName",
                    nonempty=True,
                ),
                "packed_weight_access": _tensor_access(
                    (
                        "offs_n[:, None]",
                        "(k_start // 2 + packed_k[None, :]) // 32",
                    ),
                    model[f"{prefix}WeightPackedStrides"],
                    (
                        "((k_start // 2 + packed_k[None, :]) % 32) // 16",
                        "(k_start // 2 + packed_k[None, :]) % 16",
                    ),
                    (2, 16),
                    coordinate_shape=_coordinate_shape(
                        (context["block_n"], context["packed_block_k"])
                    ),
                ),
                "weight_scale_access": _tensor_access(
                    (
                        "offs_n[:, None]",
                        f"k_start // {context['group_size']} + scale_k[None, :]",
                    ),
                    model[f"{prefix}WeightScaleStrides"],
                    coordinate_shape=_coordinate_shape(
                        (context["block_n"], context["scale_block_k"])
                    ),
                ),
            }
            for index, prefix in enumerate(("Gate", "Up"))
        )
    )
    context["projection_count"] = len(context["projections"])
    if (
        context["num_n_tiles"]
        * context["num_k_tiles"]
        * context["projection_count"]
        > 2**31 - 1
    ):
        raise ValueError(
            "PyNTT NVFP4MatMulGlu MMA pipe sequence exceeds signed int32."
        )
    return context


def _qkv_parallel_linear_template_context(
    model: dict[str, Any], *, packed: bool, variant: str | None = None
) -> dict[str, Any]:
    """Prepare QKV projection tiles and addresses for its Jinja template."""

    template_name = "PackedQKVParallelLinear" if packed else "QKVParallelLinear"
    logical_output_shapes = (
        {
            prefix: _packed_qkv_logical_output_shape(model, prefix)
            for prefix in ("Q", "K", "V")
        }
        if packed
        else {prefix: model[f"{prefix}OutputShape"] for prefix in ("Q", "K", "V")}
    )
    m = logical_output_shapes["Q"][-2]
    k = model["InputShape"][-1]
    output_batch_rank = len(model["QOutputShape"]) - 2
    use_gemv = (_max_value(m) == 1) or (_fixed(m) == 1)
    microkernel = _microkernel_context(
        model,
        "triton.qkv_parallel_linear",
        variant or ("simt_fma" if use_gemv else "mma"),
    )
    context: dict[str, Any] = {
        "microkernel": microkernel,
        "packed": packed,
        "template_name": template_name,
    }
    block_m = microkernel["parameters"]["block_m"]
    block_n = microkernel["parameters"]["block_n"]
    block_k = microkernel["parameters"]["block_k"]
    if use_gemv != (block_m == 1):
        raise ValueError(
            "PyNTT QKV microkernel block_m does not match the selected "
            f"projection shape: M={_dim(m)}, block_m={block_m}."
        )
    projections = []
    if packed:
        for key in ("Weight", "WeightShape", "WeightStrides"):
            if key not in model:
                raise ValueError(
                    "PyNTT packed QKV requires one canonical fused RHS; "
                    f"missing model field {key!r}."
                )
        projection_capacities = _packed_qkv_fixed_projection_ns(model)
    else:
        projection_capacities = None
    projection_physical_n_base = 0
    for prefix in ("Q", "K", "V"):
        lower = prefix.lower()
        has_bias = model[f"Has{prefix}Bias"]
        logical_output_shape = logical_output_shapes[prefix]
        n = logical_output_shape[-1]
        lane_shape = _qkv_packed_lane_shape(model, packed=packed)
        weight_n_axis = _structured_axis_tile(
            f"{lower}_weight_n",
            lane_shape,
            block_n,
            n,
            leading_rank=0 if use_gemv else 1,
            trailing_rank=1 if use_gemv else 0,
            physical_base=f"{lower}_n_start",
        )
        output_n_axis = _structured_axis_tile(
            f"{lower}_output_n",
            lane_shape,
            block_n,
            n,
            leading_rank=0 if use_gemv else 1,
            physical_base=f"{lower}_n_start",
        )
        weight_k_coordinate = _broadcast_axis_coordinate(
            "offs_k",
            weight_n_axis["rank"],
            weight_n_axis["rank"] - 1 if use_gemv else 0,
        )
        bias_structured_shape = _structured_value_shape(
            output_n_axis, leading_extents=() if use_gemv else (1,)
        )
        if use_gemv:
            input_coordinate_shape = _coordinate_shape((block_k,))
            weight_structured_shape = _structured_value_shape(
                weight_n_axis, trailing_extents=(block_k,)
            )
            output_structured_shape = bias_structured_shape
            input_access = _qkv_input_access(
                model,
                output_batch_rank,
                "m_idx",
                "offs_k",
                input_coordinate_shape,
            )
            weight_access = _qkv_weight_access(
                model,
                prefix,
                packed=packed,
                output_batch_rank=output_batch_rank,
                n_axis=weight_n_axis,
                k_expr=weight_k_coordinate,
                coordinate_shape=_coordinate_shape(weight_structured_shape),
                physical_n_base=projection_physical_n_base,
            )
            output_access = _qkv_output_access(
                model,
                prefix,
                packed=packed,
                output_batch_rank=output_batch_rank,
                m_expr="m_idx",
                n_axis=output_n_axis,
                coordinate_shape=_coordinate_shape(output_structured_shape),
            )
            bias_access = (
                _qkv_bias_access(
                    model,
                    prefix,
                    packed=packed,
                    n_axis=output_n_axis,
                    coordinate_shape=_coordinate_shape(bias_structured_shape),
                )
                if has_bias
                else None
            )
            input_mask = f"offs_k < {_dim(k)}"
            weight_capacity = (
                projection_capacities[prefix] if projection_capacities else n
            )
            weight_mask = (
                f"({weight_n_axis['logical_coordinate']} < {_dim(weight_capacity)}) & "
                f"({weight_k_coordinate} < {_dim(k)})"
            )
            output_mask = f"{output_n_axis['logical_coordinate']} < {_dim(n)}"
            bias_mask = output_mask
            weight_matrix_shape = (block_n, block_k)
        else:
            input_coordinate_shape = _coordinate_shape((block_m, block_k))
            weight_structured_shape = _structured_value_shape(
                weight_n_axis, leading_extents=(block_k,)
            )
            output_structured_shape = _structured_value_shape(
                output_n_axis, leading_extents=(block_m,)
            )
            input_access = _qkv_input_access(
                model,
                output_batch_rank,
                "offs_m[:, None]",
                "offs_k[None, :]",
                input_coordinate_shape,
            )
            weight_access = _qkv_weight_access(
                model,
                prefix,
                packed=packed,
                output_batch_rank=output_batch_rank,
                n_axis=weight_n_axis,
                k_expr=weight_k_coordinate,
                coordinate_shape=_coordinate_shape(weight_structured_shape),
                physical_n_base=projection_physical_n_base,
            )
            output_m_coordinate = _broadcast_axis_coordinate(
                "offs_m", output_n_axis["rank"], 0
            )
            output_access = _qkv_output_access(
                model,
                prefix,
                packed=packed,
                output_batch_rank=output_batch_rank,
                m_expr=output_m_coordinate,
                n_axis=output_n_axis,
                coordinate_shape=_coordinate_shape(output_structured_shape),
            )
            bias_access = (
                _qkv_bias_access(
                    model,
                    prefix,
                    packed=packed,
                    n_axis=output_n_axis,
                    coordinate_shape=_coordinate_shape(bias_structured_shape),
                )
                if has_bias
                else None
            )
            input_mask = (
                f"(offs_m[:, None] < {_dim(logical_output_shape[-2])}) & "
                f"(offs_k[None, :] < {_dim(k)})"
            )
            weight_capacity = (
                projection_capacities[prefix] if projection_capacities else n
            )
            weight_mask = (
                f"({weight_k_coordinate} < {_dim(k)}) & "
                f"({weight_n_axis['logical_coordinate']} < {_dim(weight_capacity)})"
            )
            output_mask = (
                f"({output_m_coordinate} < {_dim(logical_output_shape[-2])}) & "
                f"({output_n_axis['logical_coordinate']} < {_dim(n)})"
            )
            bias_mask = f"{output_n_axis['logical_coordinate']} < {_dim(n)}"
            weight_matrix_shape = (block_k, block_n)
        projections.append(
            {
                "bias_mask": bias_mask,
                "bias_access": bias_access,
                "has_bias": has_bias,
                "input_mask": input_mask,
                "input_access": input_access,
                "lower": lower,
                "n": n,
                "output_n_axis": output_n_axis,
                "output_mask": output_mask,
                "output_access": output_access,
                "output_structured_shape": output_structured_shape,
                "physical_n": model[f"{prefix}OutputShape"][-1],
                "physical_block_n": output_n_axis["physical_block_extent"],
                "prefix": prefix,
                "weight_matrix_shape": weight_matrix_shape,
                "weight_n_axis": weight_n_axis,
                "weight_mask": weight_mask,
                "weight_access": weight_access,
                "weight_key": "Weight" if packed else f"{prefix}Weight",
                "weight_variable": "weight" if packed else f"{lower}_weight",
                "weight_structured_shape": weight_structured_shape,
            }
        )
        if projection_capacities is not None:
            scalar_lanes = int(model["NPackedLaneCount"]) * int(
                model["NVectorLaneCount"]
            )
            projection_physical_n_base += (
                projection_capacities[prefix] // scalar_lanes
            )
    context.update(
        batch_axes=tuple(range(output_batch_rank)),
        block_k=block_k,
        block_m=block_m,
        block_n=block_n,
        dot_precision=(
            ', input_precision="ieee"'
            if model["InputDType"] == "float32" and model["WeightDType"] == "float32"
            else ""
        ),
        k=k,
        logical_output_shapes=logical_output_shapes,
        m=m,
        projections=tuple(projections),
        use_gemv=use_gemv,
    )
    return context


def _packed_qkv_logical_output_shape(model: dict[str, Any], prefix: str) -> list[Any]:
    scalar_lane_count = model["NPackedLaneCount"] * model["NVectorLaneCount"]
    shape = [
        dict(dim) if isinstance(dim, dict) else dim
        for dim in model[f"{prefix}OutputShape"]
    ]
    shape[-1] = _multiply_dim(shape[-1], scalar_lane_count)
    return shape


def _matmul_glu_lane_shape(model: dict[str, Any], *, packed: bool) -> tuple[int, ...]:
    if not packed:
        return ()
    n_pack = int(model["NPackedLaneCount"])
    n_vector = int(model["NVectorLaneCount"])
    return (n_pack, n_vector) if n_pack > 1 else (n_vector,)


def _matmul_glu_weight_lane_shape(
    model: dict[str, Any], *, packed: bool
) -> tuple[int, ...]:
    n_lane_shape = _matmul_glu_lane_shape(model, packed=packed)
    if not packed or model.get("RhsLayout", "n_major") == "n_major":
        return n_lane_shape
    return (
        *n_lane_shape,
        int(model["KPackLaneCount"]),
        int(model["KVectorLaneCount"]),
    )


def _matmul_glu_input_access(
    model: dict[str, Any],
    output_batch_rank: int,
    m_expr: str,
    k_expr: str,
    coordinate_shape: str,
) -> dict[str, Any]:
    coordinates = _aligned_batch_coordinates(
        model["InputShape"], 2, output_batch_rank
    ) + (m_expr, k_expr)
    return _tensor_access(
        coordinates, model["InputStrides"], coordinate_shape=coordinate_shape
    )


def _matmul_glu_weight_access(
    model: dict[str, Any],
    prefix: str,
    *,
    packed: bool,
    output_batch_rank: int,
    n_axis: dict[str, Any],
    k_expr: str,
    coordinate_shape: str,
) -> dict[str, Any]:
    n_lane_shape = _matmul_glu_lane_shape(model, packed=packed)
    if tuple(n_lane_shape) != tuple(n_axis["lane_shape"]):
        raise ValueError(
            f"PyNTT {prefix} MatMulGlu weight lane shape does not match its N tile: "
            f"weight N={n_lane_shape}, tile={n_axis['lane_shape']}."
        )
    weight_lane_shape = _matmul_glu_weight_lane_shape(model, packed=packed)
    batch_coordinates = _aligned_batch_coordinates(
        model[f"{prefix}WeightShape"], 2, output_batch_rank
    )
    if not packed:
        matrix_coordinates = (k_expr, n_axis["physical_coordinate"])
        lane_coordinates = ()
    elif model.get("RhsLayout", "n_major") == "n_major":
        matrix_coordinates = (n_axis["physical_coordinate"], k_expr)
        lane_coordinates = n_axis["lane_coordinates"]
    else:
        k_pack = int(model["KPackLaneCount"])
        k_lane = int(model["KVectorLaneCount"])
        k_atom = k_pack * k_lane
        matrix_coordinates = (
            f"({k_expr}) // {k_atom}",
            n_axis["physical_coordinate"],
        )
        lane_coordinates = (
            *n_axis["lane_coordinates"],
            f"(({k_expr}) // {k_lane}) % {k_pack}",
            f"({k_expr}) % {k_lane}",
        )
    return _tensor_access(
        batch_coordinates + matrix_coordinates,
        model[f"{prefix}WeightStrides"],
        lane_coordinates,
        weight_lane_shape,
        coordinate_shape,
    )


def _matmul_glu_output_access(
    model: dict[str, Any],
    *,
    packed: bool,
    output_batch_rank: int,
    m_expr: str,
    n_axis: dict[str, Any],
    coordinate_shape: str,
) -> dict[str, Any]:
    lane_shape = _matmul_glu_lane_shape(model, packed=packed)
    if tuple(lane_shape) != tuple(n_axis["lane_shape"]):
        raise ValueError(
            "PyNTT MatMulGlu output lane shape does not match its N tile: "
            f"output={lane_shape}, tile={n_axis['lane_shape']}."
        )
    coordinates = _aligned_batch_coordinates(
        model["OutputShape"], 2, output_batch_rank
    ) + (m_expr, n_axis["physical_coordinate"])
    return _tensor_access(
        coordinates,
        model["OutputStrides"],
        n_axis["lane_coordinates"],
        lane_shape,
        coordinate_shape,
    )


def _matmul_glu_bias_access(
    model: dict[str, Any],
    prefix: str,
    *,
    packed: bool,
    n_axis: dict[str, Any],
    coordinate_shape: str,
) -> dict[str, Any]:
    lane_shape = _matmul_glu_lane_shape(model, packed=packed)
    if tuple(lane_shape) != tuple(n_axis["lane_shape"]):
        raise ValueError(
            f"PyNTT {prefix} MatMulGlu bias lane shape does not match its N tile: "
            f"bias={lane_shape}, tile={n_axis['lane_shape']}."
        )
    return _tensor_access(
        (n_axis["physical_coordinate"],),
        model[f"{prefix}BiasStrides"],
        n_axis["lane_coordinates"],
        lane_shape,
        coordinate_shape,
    )


def _matmul_glu_template_context(
    model: dict[str, Any], *, packed: bool, variant: str | None = None
) -> dict[str, Any]:
    """Prepare MatMulGlu tiles and layout-explicit packed addresses."""

    logical_output_shape = _matmul_glu_logical_output_shape(model)
    template_name = "PackedMatMulGlu" if packed else "MatMulGlu"
    m = logical_output_shape[-2]
    n = logical_output_shape[-1]
    k = model["InputShape"][-1]
    output_batch_rank = len(model["OutputShape"]) - 2
    use_gemv = (_max_value(m) == 1) or (_fixed(m) == 1)
    microkernel = _microkernel_context(
        model,
        "triton.matmul_glu",
        variant or ("simt_fma" if use_gemv else "mma"),
    )
    context: dict[str, Any] = {
        "logical_output_shape": logical_output_shape,
        "microkernel": microkernel,
        "packed": packed,
        "template_name": template_name,
    }
    block_m = microkernel["parameters"]["block_m"]
    block_n = microkernel["parameters"]["block_n"]
    block_k = microkernel["parameters"]["block_k"]
    if use_gemv != (block_m == 1):
        raise ValueError(
            "PyNTT MatMulGlu microkernel block_m does not match the selected "
            f"projection shape: M={_dim(m)}, block_m={block_m}."
        )
    if use_gemv:
        input_m, input_m_limit = _matmul_glu_input_m_index(model, "m_idx")
        input_access = _matmul_glu_input_access(
            model,
            output_batch_rank,
            input_m,
            "offs_k",
            _coordinate_shape((block_k,)),
        )
        input_mask = f"(m_idx < {_dim(input_m_limit)}) & (offs_k < {_dim(k)})"
    else:
        input_m, input_m_limit = _matmul_glu_input_m_index(model, "offs_m[:, None]")
        input_access = _matmul_glu_input_access(
            model,
            output_batch_rank,
            input_m,
            "offs_k[None, :]",
            _coordinate_shape((block_m, block_k)),
        )
        input_mask = (
            f"(offs_m[:, None] < {_dim(m)}) & "
            f"({input_m} < {_dim(input_m_limit)}) & "
            f"(offs_k[None, :] < {_dim(k)})"
        )
    lane_shape = _matmul_glu_lane_shape(model, packed=packed)
    weight_n_axis = _structured_axis_tile(
        "weight_n",
        lane_shape,
        block_n,
        n,
        leading_rank=0 if use_gemv else 1,
        trailing_rank=1 if use_gemv else 0,
        physical_base="n_start",
    )
    output_n_axis = _structured_axis_tile(
        "output_n",
        lane_shape,
        block_n,
        n,
        leading_rank=0 if use_gemv else 1,
        physical_base="n_start",
    )
    weight_k_coordinate = _broadcast_axis_coordinate(
        "offs_k",
        weight_n_axis["rank"],
        weight_n_axis["rank"] - 1 if use_gemv else 0,
    )
    output_m_coordinate = (
        "m_idx"
        if use_gemv
        else _broadcast_axis_coordinate("offs_m", output_n_axis["rank"], 0)
    )
    weight_structured_shape = _structured_value_shape(
        weight_n_axis,
        trailing_extents=(block_k,) if use_gemv else (),
        leading_extents=() if use_gemv else (block_k,),
    )
    bias_structured_shape = _structured_value_shape(
        output_n_axis, leading_extents=() if use_gemv else (1,)
    )
    output_structured_shape = _structured_value_shape(
        output_n_axis,
        leading_extents=() if use_gemv else (block_m,),
    )
    output_access = _matmul_glu_output_access(
        model,
        packed=packed,
        output_batch_rank=output_batch_rank,
        n_axis=output_n_axis,
        m_expr=output_m_coordinate,
        coordinate_shape=_coordinate_shape(output_structured_shape),
    )
    projections = []
    for prefix, accumulator in (("Gate", "gate_acc"), ("Up", "up_acc")):
        _, weight_n_limit = _matmul_glu_weight_n_index(
            model,
            prefix,
            weight_n_axis["logical_coordinate"],
            packed=packed,
        )
        _, weight_k_limit = _matmul_glu_weight_k_index(
            model,
            prefix,
            weight_k_coordinate,
            packed=packed,
        )
        weight_access = _matmul_glu_weight_access(
            model,
            prefix,
            packed=packed,
            output_batch_rank=output_batch_rank,
            n_axis=weight_n_axis,
            k_expr=weight_k_coordinate,
            coordinate_shape=_coordinate_shape(weight_structured_shape),
        )
        if use_gemv:
            weight_mask = (
                f"({weight_n_axis['logical_coordinate']} < {_dim(n)}) & "
                f"({weight_n_axis['logical_coordinate']} < {_dim(weight_n_limit)}) & "
                f"({weight_k_coordinate} < {_dim(weight_k_limit)})"
            )
        else:
            weight_mask = (
                f"({weight_k_coordinate} < {_dim(weight_k_limit)}) & "
                f"({weight_n_axis['logical_coordinate']} < {_dim(n)}) & "
                f"({weight_n_axis['logical_coordinate']} < {_dim(weight_n_limit)})"
            )
        bias_access = None
        bias_mask = None
        if model[f"Has{prefix}Bias"]:
            bias_access = _matmul_glu_bias_access(
                model,
                prefix,
                packed=packed,
                n_axis=output_n_axis,
                coordinate_shape=_coordinate_shape(bias_structured_shape),
            )
            _, bias_n_limit = _matmul_glu_bias_n_index(
                model, prefix, output_n_axis["logical_coordinate"], packed=packed
            )
            bias_mask = (
                f"({output_n_axis['logical_coordinate']} < {_dim(n)}) & "
                f"({output_n_axis['logical_coordinate']} < {_dim(bias_n_limit)})"
            )
        projections.append(
            {
                "accumulator": accumulator,
                "bias_mask": bias_mask,
                "bias_access": bias_access,
                "has_bias": model[f"Has{prefix}Bias"],
                "lower": prefix.lower(),
                "prefix": prefix,
                "weight_mask": weight_mask,
                "weight_access": weight_access,
            }
        )
    context.update(
        batch_axes=tuple(range(output_batch_rank)),
        block_k=block_k,
        block_m=block_m,
        block_n=block_n,
        dot_precision=(
            ', input_precision="ieee"'
            if model["InputDType"] == "float32" and model["WeightDType"] == "float32"
            else ""
        ),
        input_mask=input_mask,
        input_access=input_access,
        k=k,
        m=m,
        n=n,
        output_n_axis=output_n_axis,
        output_mask=(
            f"{output_n_axis['logical_coordinate']} < {_dim(n)}"
            if use_gemv
            else (
                f"({output_m_coordinate} < {_dim(m)}) & "
                f"({output_n_axis['logical_coordinate']} < {_dim(n)})"
            )
        ),
        output_access=output_access,
        output_structured_shape=output_structured_shape,
        physical_n=model["OutputShape"][-1],
        physical_block_n=output_n_axis["physical_block_extent"],
        projections=tuple(projections),
        result_expression=_matmul_glu_expr(model, "gate_acc", "up_acc"),
        use_gemv=use_gemv,
        weight_matrix_shape=(block_n, block_k) if use_gemv else (block_k, block_n),
        weight_n_axis=weight_n_axis,
    )
    return context


def _matmul_glu_logical_output_shape(model: dict[str, Any]) -> list[Any]:
    shape = [
        dict(dim) if isinstance(dim, dict) else dim for dim in model["OutputShape"]
    ]
    if model.get("PackedN"):
        shape[-1] = _multiply_dim(
            shape[-1], model["NPackedLaneCount"] * model["NVectorLaneCount"]
        )
    return shape


def _matmul_glu_input_m_index(model: dict[str, Any], m_expr: str) -> tuple[str, Any]:
    return m_expr, model["InputShape"][-2]


def _matmul_glu_weight_k_index(
    model: dict[str, Any], prefix: str, k_expr: str, *, packed: bool
) -> tuple[str, Any]:
    weight_shape = model[f"{prefix}WeightShape"]
    if not packed:
        return k_expr, weight_shape[-2]
    if model.get("RhsLayout", "n_major") == "n_major":
        return k_expr, weight_shape[-1]
    k_atom = int(model["KPackLaneCount"]) * int(model["KVectorLaneCount"])
    return k_expr, _multiply_dim(weight_shape[-2], k_atom)


def _matmul_glu_weight_n_index(
    model: dict[str, Any],
    prefix: str,
    n_expr: str,
    *,
    packed: bool,
) -> tuple[str, Any]:
    weight_shape = model[f"{prefix}WeightShape"]
    if not packed:
        return n_expr, weight_shape[-1]
    weight_axis = (
        -2 if model.get("RhsLayout", "n_major") == "n_major" else -1
    )
    lane_scale = model["NPackedLaneCount"] * model["NVectorLaneCount"]
    return n_expr, _multiply_dim(weight_shape[weight_axis], lane_scale)


def _matmul_glu_bias_n_index(
    model: dict[str, Any],
    prefix: str,
    n_expr: str,
    *,
    packed: bool,
) -> tuple[str, Any]:
    bias_shape = model[f"{prefix}BiasShape"]
    lane_scale = model["NPackedLaneCount"] * model["NVectorLaneCount"] if packed else 1
    return n_expr, _multiply_dim(bias_shape[-1], lane_scale)


def _matmul_glu_expr(model: dict[str, Any], gate: str, up: str) -> str:
    glu_type = str(model.get("GluType", "swiglu")).lower()
    if glu_type == "swiglu":
        return f"(({gate}) / (1.0 + tl.exp(-({gate}))) * ({up}))"
    raise NotImplementedError(f"Unsupported MatMulGlu type: {model.get('GluType')}.")


def _matmul_n_lane_shape(model: dict[str, Any], prefix: str) -> tuple[int, ...]:
    packed_lane_count = int(model.get(f"{prefix}NPackedLaneCount", 1))
    vector_lane_count = int(model[f"{prefix}NVectorLaneCount"])
    if packed_lane_count > 1:
        return packed_lane_count, vector_lane_count
    if vector_lane_count > 1:
        return (vector_lane_count,)
    return ()


def _matmul_rhs_layout(model: dict[str, Any]) -> str:
    layout = str(model.get("RhsLayout", "n_major"))
    if layout not in ("n_major", "k_major", "n_major_k_packed"):
        raise ValueError(f"Unsupported PyNTT Matmul RHS layout {layout!r}.")
    return layout


def _matmul_lhs_access(
    model: dict[str, Any],
    output_batch_rank: int,
    m_expr: str,
    k_expr: str,
    coordinate_shape: str,
) -> dict[str, Any]:
    batch_coordinates = _aligned_batch_coordinates(
        model["LhsShape"], 2, output_batch_rank
    )
    matrix_coordinates = (k_expr, m_expr) if model["TransposeA"] else (m_expr, k_expr)
    return _tensor_access(
        batch_coordinates + matrix_coordinates,
        model["LhsStrides"],
        coordinate_shape=coordinate_shape,
    )


def _matmul_rhs_access(
    model: dict[str, Any],
    output_batch_rank: int,
    n_axis: dict[str, Any],
    k_expr: str,
    coordinate_shape: str,
) -> dict[str, Any]:
    lane_shape = _matmul_n_lane_shape(model, "Rhs")
    layout = _matmul_rhs_layout(model)
    if layout != "n_major_k_packed" and tuple(lane_shape) != tuple(
        n_axis["lane_shape"]
    ):
        raise ValueError(
            "PyNTT Matmul RHS lane shape does not match its N tile: "
            f"rhs={lane_shape}, tile={n_axis['lane_shape']}."
        )
    batch_coordinates = _aligned_batch_coordinates(
        model["RhsShape"], 2, output_batch_rank
    )
    k_pack = int(model["RhsKPackLaneCount"])
    k_lane = int(model["RhsKVectorLaneCount"])
    has_packed_k = k_pack != 1 or k_lane != 1
    if not has_packed_k:
        matrix_coordinates = (
            (n_axis["physical_coordinate"], k_expr)
            if model["TransposeB"]
            else (k_expr, n_axis["physical_coordinate"])
        )
        return _tensor_access(
            batch_coordinates + matrix_coordinates,
            model["RhsStrides"],
            n_axis["lane_coordinates"],
            lane_shape,
            coordinate_shape=coordinate_shape,
        )

    if layout == "n_major":
        if not model["TransposeB"]:
            raise ValueError("PyNTT N-major packed RHS must be logically transposed.")
        matrix_coordinates = (n_axis["physical_coordinate"], k_expr)
        lane_coordinates = n_axis["lane_coordinates"]
        physical_lane_shape = lane_shape
    elif layout == "k_major":
        if model["TransposeB"]:
            raise ValueError("PyNTT K-major packed RHS cannot be logically transposed.")
        if k_pack <= 0 or k_lane <= 0:
            raise ValueError(
                "PyNTT K-major packed RHS requires positive K pack/vector lanes."
            )
        k_atom = k_pack * k_lane
        matrix_coordinates = (
            f"({k_expr}) // {k_atom}",
            n_axis["physical_coordinate"],
        )
        lane_coordinates = (
            *n_axis["lane_coordinates"],
            f"(({k_expr}) // {k_lane}) % {k_pack}",
            f"({k_expr}) % {k_lane}",
        )
        physical_lane_shape = (*lane_shape, k_pack, k_lane)
    else:
        if not model["TransposeB"]:
            raise ValueError(
                "PyNTT N-major K-packed RHS must be logically transposed."
            )
        if lane_shape:
            raise ValueError(
                "PyNTT N-major K-packed RHS cannot carry N vector lanes."
            )
        if k_pack <= 0 or k_lane <= 0:
            raise ValueError(
                "PyNTT N-major K-packed RHS requires positive K pack/vector lanes."
            )
        k_atom = k_pack * k_lane
        matrix_coordinates = (
            n_axis["logical_coordinate"],
            f"({k_expr}) // {k_atom}",
        )
        lane_coordinates = (
            f"(({k_expr}) // {k_lane}) % {k_pack}",
            f"({k_expr}) % {k_lane}",
        )
        physical_lane_shape = (k_pack, k_lane)
    return _tensor_access(
        batch_coordinates + matrix_coordinates,
        model["RhsStrides"],
        lane_coordinates,
        physical_lane_shape,
        coordinate_shape,
    )


def _matmul_output_access(
    model: dict[str, Any],
    output_batch_rank: int,
    m_expr: str,
    n_axis: dict[str, Any],
    coordinate_shape: str,
    *,
    lane_shape_override: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    logical_lane_shape = _matmul_n_lane_shape(model, "Output")
    lane_shape = (
        tuple(logical_lane_shape)
        if lane_shape_override is None
        else tuple(lane_shape_override)
    )
    if _product_int(list(lane_shape)) != _product_int(list(logical_lane_shape)):
        raise ValueError(
            "PyNTT Matmul output access lane override must preserve the "
            f"scalar lane count: output={logical_lane_shape}, override={lane_shape}."
        )
    if tuple(lane_shape) != tuple(n_axis["lane_shape"]):
        raise ValueError(
            "PyNTT Matmul output lane shape does not match its N tile: "
            f"output={lane_shape}, tile={n_axis['lane_shape']}."
        )
    batch_coordinates = _aligned_batch_coordinates(
        model["OutputShape"], 2, output_batch_rank
    )
    return _tensor_access(
        batch_coordinates + (m_expr, n_axis["physical_coordinate"]),
        model["OutputStrides"],
        n_axis["lane_coordinates"],
        lane_shape,
        coordinate_shape,
    )


def _matmul_addend_access(
    model: dict[str, Any],
    output_batch_rank: int,
    m_expr: str,
    n_axis: dict[str, Any],
    coordinate_shape: str,
) -> dict[str, Any]:
    if not model["HasAddend"]:
        raise ValueError("PyNTT Matmul addend access requires an addend.")
    if not isinstance(model["Addend"], dict):
        raise ValueError("PyNTT Matmul addend must contain pointer metadata.")
    if len(model["AddendShape"]) != len(model["OutputShape"]):
        raise ValueError(
            "PyNTT Matmul addend/output ranks must match: "
            f"addend={len(model['AddendShape'])}, output={len(model['OutputShape'])}."
        )
    lane_shape = _matmul_n_lane_shape(model, "Output")
    if tuple(lane_shape) != tuple(n_axis["lane_shape"]):
        raise ValueError(
            "PyNTT Matmul addend lane shape does not match its N tile: "
            f"addend={lane_shape}, tile={n_axis['lane_shape']}."
        )
    batch_coordinates = _aligned_batch_coordinates(
        model["AddendShape"], 2, output_batch_rank
    )
    return _tensor_access(
        batch_coordinates + (m_expr, n_axis["physical_coordinate"]),
        model["AddendStrides"],
        n_axis["lane_coordinates"],
        lane_shape,
        coordinate_shape,
    )


def _matmul_stats_access(
    model: dict[str, Any],
    output_batch_rank: int,
    m_expr: str,
    component: int,
    coordinate_shape: str | None,
) -> dict[str, Any]:
    if not model["HasNormStats"]:
        raise ValueError("PyNTT Matmul statistics access requires a statistics output.")
    coordinates = (
        (str(component),)
        + tuple(f"idx{axis}" for axis in range(output_batch_rank))
        + (m_expr, "0")
    )
    return _tensor_access(
        coordinates,
        model["StatsStrides"],
        coordinate_shape=coordinate_shape,
    )


def _is_positive_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _power_of_two_segments(value: int) -> tuple[tuple[int, int], ...]:
    """Partition a positive extent into aligned descending power-of-two spans."""

    if value <= 0:
        raise ValueError(f"PyNTT segment extent must be positive, got {value}.")
    segments: list[tuple[int, int]] = []
    offset = 0
    remaining = value
    while remaining:
        extent = 1 << (remaining.bit_length() - 1)
        segments.append((offset, extent))
        offset += extent
        remaining -= extent
    return tuple(segments)


def _matmul_template_context(
    model: dict[str, Any],
    *,
    gemv: bool,
    variant: str | None = None,
    expected_family: str = "triton.matmul",
    required_workspace_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Prepare Matmul/Gemv dimensions and addresses for Jinja-owned kernels."""

    if not isinstance(model["HasAddend"], bool):
        raise ValueError("PyNTT Matmul HasAddend must be a boolean.")
    if not isinstance(model["HasNormStats"], bool):
        raise ValueError("PyNTT Matmul HasNormStats must be a boolean.")
    output_lane_count = (
        model.get("OutputNPackedLaneCount", 1) * model["OutputNVectorLaneCount"]
    )
    logical_output_shape = [
        dict(value) if isinstance(value, dict) else value
        for value in model["OutputShape"]
    ]
    logical_output_shape[-1] = _multiply_dim(
        logical_output_shape[-1], output_lane_count
    )
    microkernel = _microkernel_context(
        model,
        expected_family,
        variant or ("simt_fma" if gemv else "mma"),
        required_workspace_names=required_workspace_names,
    )
    context: dict[str, Any] = {
        "gemv": gemv,
        "logical_output_shape": logical_output_shape,
        "microkernel": microkernel,
        "template_name": "Gemv" if gemv else "Matmul",
    }

    rhs_layout = _matmul_rhs_layout(model)
    rhs_lane_count = model.get("RhsNPackedLaneCount", 1) * model["RhsNVectorLaneCount"]
    m = logical_output_shape[-2]
    n = logical_output_shape[-1]
    lhs_m = model["LhsShape"][-1] if model["TransposeA"] else model["LhsShape"][-2]
    lhs_k = model["LhsShape"][-2] if model["TransposeA"] else model["LhsShape"][-1]
    rhs_k_lane_count = int(model["RhsKPackLaneCount"]) * int(
        model["RhsKVectorLaneCount"]
    )
    if rhs_k_lane_count > 1:
        if rhs_layout == "k_major" and not model["TransposeB"]:
            rhs_k = _multiply_dim(model["RhsShape"][-2], rhs_k_lane_count)
            rhs_n = _multiply_dim(model["RhsShape"][-1], rhs_lane_count)
        elif rhs_layout == "n_major_k_packed" and model["TransposeB"]:
            rhs_k = _multiply_dim(model["RhsShape"][-1], rhs_k_lane_count)
            rhs_n = _multiply_dim(model["RhsShape"][-2], rhs_lane_count)
        else:
            raise ValueError(
                "PyNTT K-packed Matmul RHS requires either a non-transposed "
                "K-major layout or a transposed N-major K-packed layout."
            )
    else:
        rhs_k = (
            model["RhsShape"][-1]
            if model["TransposeB"]
            else model["RhsShape"][-2]
        )
        rhs_n = _multiply_dim(
            model["RhsShape"][-2]
            if model["TransposeB"]
            else model["RhsShape"][-1],
            rhs_lane_count,
        )
    context.update(
        m=m,
        n=n,
        lhs_m=lhs_m,
        lhs_k=lhs_k,
        rhs_k=rhs_k,
        rhs_n=rhs_n,
        rhs_layout=rhs_layout,
    )

    k = lhs_k
    output_batch_rank = len(logical_output_shape) - 2
    if model["HasNormStats"]:
        if model["StatsDType"] != "float32":
            raise ValueError(
                "PyNTT Matmul normalization statistics must use float32 storage."
            )
        if int(model["NormAxis"]) != len(logical_output_shape) - 1:
            raise ValueError(
                "PyNTT Matmul normalization statistics must reduce the final "
                "logical output axis."
            )
        if len(model["StatsShape"]) != len(model["OutputShape"]) + 1:
            raise ValueError(
                "PyNTT Matmul statistics rank must be output rank plus one."
            )
        if not isinstance(model["UseMean"], bool):
            raise ValueError("PyNTT Matmul UseMean must be a boolean.")
    load_c_expression = str(model.get("LoadCExpression", "False")).strip() or "False"
    load_c = load_c_expression not in ("False", "false", "0")
    load_c_predicate = (
        "True"
        if load_c_expression in ("True", "true", "1")
        else f"({load_c_expression})"
    )
    context.update(
        batch_axes=tuple(range(output_batch_rank)),
        k=k,
        has_addend=model["HasAddend"],
        load_c=load_c,
        load_c_expression=load_c_expression,
        load_c_predicate=load_c_predicate,
    )
    rhs_lane_shape = _matmul_n_lane_shape(model, "Rhs")
    output_lane_shape = _matmul_n_lane_shape(model, "Output")
    if (
        rhs_lane_shape != output_lane_shape
        and rhs_layout != "n_major_k_packed"
    ):
        raise ValueError(
            "PyNTT Matmul requires one structured N-axis layout for RHS and "
            f"output, got rhs={rhs_lane_shape}, output={output_lane_shape}."
        )
    block_m = microkernel["parameters"]["block_m"]
    block_n = microkernel["parameters"]["block_n"]
    block_k = microkernel["parameters"]["block_k"]
    if gemv != (block_m == 1):
        raise ValueError(
            "PyNTT Matmul microkernel block_m does not match the selected "
            f"matrix shape: M={_dim(m)}, block_m={block_m}."
        )
    structured_n_lane_shape = (
        output_lane_shape
        if rhs_layout == "n_major_k_packed"
        else rhs_lane_shape
    )
    if gemv:
        rhs_n_axis = _structured_axis_tile(
            "rhs_n",
            structured_n_lane_shape,
            block_n,
            n,
            trailing_rank=1,
            physical_base="n_start",
        )
        output_n_axis = _structured_axis_tile(
            "output_n",
            output_lane_shape,
            block_n,
            n,
            physical_base="n_start",
        )
        rhs_k_coordinate = _broadcast_axis_coordinate(
            "offs_k", rhs_n_axis["rank"], rhs_n_axis["rank"] - 1
        )
        rhs_structured_shape = _structured_value_shape(
            rhs_n_axis, trailing_extents=(block_k,)
        )
        output_structured_shape = _structured_value_shape(output_n_axis)
        context.update(
            block_k=block_k,
            block_n=block_n,
            lhs_mask=f"(m_idx < {_dim(lhs_m)}) & (offs_k < {_dim(k)})",
            lhs_access=_matmul_lhs_access(
                model,
                output_batch_rank,
                "m_idx",
                "offs_k",
                _coordinate_shape((block_k,)),
            ),
            output_mask=f"{output_n_axis['logical_coordinate']} < {_dim(n)}",
            output_n_axis=output_n_axis,
            output_access=_matmul_output_access(
                model,
                output_batch_rank,
                "m_idx",
                output_n_axis,
                _coordinate_shape(output_structured_shape),
            ),
            addend_access=(
                _matmul_addend_access(
                    model,
                    output_batch_rank,
                    "m_idx",
                    output_n_axis,
                    _coordinate_shape(output_structured_shape),
                )
                if model["HasAddend"]
                else None
            ),
            output_structured_shape=output_structured_shape,
            physical_n=model["OutputShape"][-1],
            physical_block_n=output_n_axis["physical_block_extent"],
            rhs_mask=(
                f"({rhs_n_axis['logical_coordinate']} < {_dim(n)}) & "
                f"({rhs_n_axis['logical_coordinate']} < {_dim(rhs_n)}) & "
                f"({rhs_k_coordinate} < {_dim(k)}) & "
                f"({rhs_k_coordinate} < {_dim(rhs_k)})"
            ),
            rhs_access=_matmul_rhs_access(
                model,
                output_batch_rank,
                rhs_n_axis,
                rhs_k_coordinate,
                _coordinate_shape(rhs_structured_shape),
            ),
            rhs_matrix_shape=(block_n, block_k),
            rhs_n_axis=rhs_n_axis,
            rhs_structured_shape=rhs_structured_shape,
        )
    else:
        rhs_n_axis = _structured_axis_tile(
            "rhs_n",
            structured_n_lane_shape,
            block_n,
            n,
            leading_rank=1,
            physical_base="n_start",
        )
        output_n_axis = _structured_axis_tile(
            "output_n",
            output_lane_shape,
            block_n,
            n,
            leading_rank=1,
            physical_base="n_start",
        )
        rhs_k_coordinate = _broadcast_axis_coordinate("offs_k", rhs_n_axis["rank"], 0)
        output_m_coordinate = _broadcast_axis_coordinate(
            "offs_m", output_n_axis["rank"], 0
        )
        rhs_structured_shape = _structured_value_shape(
            rhs_n_axis, leading_extents=(block_k,)
        )
        output_structured_shape = _structured_value_shape(
            output_n_axis, leading_extents=(block_m,)
        )
        context.update(
            block_k=block_k,
            block_m=block_m,
            block_n=block_n,
            dot_precision=(
                ', input_precision="ieee"'
                if model["LhsDType"] == "float32" and model["RhsDType"] == "float32"
                else ""
            ),
            lhs_mask=(
                f"(offs_m[:, None] < {_dim(m)}) & "
                f"(offs_m[:, None] < {_dim(lhs_m)}) & "
                f"(offs_k[None, :] < {_dim(k)})"
            ),
            lhs_access=_matmul_lhs_access(
                model,
                output_batch_rank,
                "offs_m[:, None]",
                "offs_k[None, :]",
                _coordinate_shape((block_m, block_k)),
            ),
            output_mask=(
                f"({output_m_coordinate} < {_dim(m)}) & "
                f"({output_n_axis['logical_coordinate']} < {_dim(n)})"
            ),
            output_n_axis=output_n_axis,
            output_access=_matmul_output_access(
                model,
                output_batch_rank,
                output_m_coordinate,
                output_n_axis,
                _coordinate_shape(output_structured_shape),
            ),
            addend_access=(
                _matmul_addend_access(
                    model,
                    output_batch_rank,
                    output_m_coordinate,
                    output_n_axis,
                    _coordinate_shape(output_structured_shape),
                )
                if model["HasAddend"]
                else None
            ),
            output_structured_shape=output_structured_shape,
            physical_n=model["OutputShape"][-1],
            physical_block_n=output_n_axis["physical_block_extent"],
            rhs_mask=(
                f"({rhs_k_coordinate} < {_dim(k)}) & "
                f"({rhs_k_coordinate} < {_dim(rhs_k)}) & "
                f"({rhs_n_axis['logical_coordinate']} < {_dim(n)}) & "
                f"({rhs_n_axis['logical_coordinate']} < {_dim(rhs_n)})"
            ),
            rhs_access=_matmul_rhs_access(
                model,
                output_batch_rank,
                rhs_n_axis,
                rhs_k_coordinate,
                _coordinate_shape(rhs_structured_shape),
            ),
            rhs_matrix_shape=(block_k, block_n),
            rhs_n_axis=rhs_n_axis,
            rhs_structured_shape=rhs_structured_shape,
        )
    if model["HasNormStats"]:
        stats_m_expr = "m_idx" if gemv else "offs_m"
        stats_coordinate_shape = (
            None if gemv else _coordinate_shape((block_m,))
        )
        stats_components = 2 if model["UseMean"] else 1
        context.update(
            stats_accesses=tuple(
                _matmul_stats_access(
                    model,
                    output_batch_rank,
                    stats_m_expr,
                    component,
                    stats_coordinate_shape,
                )
                for component in range(stats_components)
            ),
            stats_mask=("True" if gemv else f"offs_m < {_dim(m)}"),
        )
    return context


def _packed_gemv_consumer_layout(
    *,
    block_n: int,
    reduction_group: int,
    consumer_warps: int,
    target_worker_width: int,
) -> dict[str, tuple[int, int]]:
    """Derive a warp-local N/K partition that covers one GEMV tile exactly."""

    if (
        not _is_positive_power_of_two(block_n)
        or consumer_warps <= 0
        or block_n % consumer_warps != 0
    ):
        raise ValueError(
            "PyNTT K-major GEMV requires a power-of-two block_n divisible "
            f"by its consumer warps, got block_n={block_n}, "
            f"consumer_warps={consumer_warps}."
        )
    threads_per_warp_n = block_n // consumer_warps
    if (
        threads_per_warp_n <= 0
        or target_worker_width % threads_per_warp_n != 0
    ):
        raise ValueError(
            "PyNTT K-major GEMV cannot partition a worker across N: "
            f"worker_width={target_worker_width}, "
            f"threads_per_warp_n={threads_per_warp_n}."
        )
    threads_per_warp_k = target_worker_width // threads_per_warp_n
    if reduction_group % threads_per_warp_k != 0:
        raise ValueError(
            "PyNTT K-major GEMV cannot cover its reduction group exactly: "
            f"reduction_group={reduction_group}, "
            f"threads_per_warp_k={threads_per_warp_k}."
        )
    k_values_per_thread = reduction_group // threads_per_warp_k
    return {
        "size_per_thread": (1, k_values_per_thread),
        "threads_per_warp": (threads_per_warp_n, threads_per_warp_k),
        "warps_per_cta": (consumer_warps, 1),
    }


def _validate_packed_gemv_pipeline_resource_contract(
    *,
    algorithm: str,
    block_n: int,
    block_k: int,
    num_stages: int,
    rhs_tiles_per_group: int = 1,
    maximum_block_n: int = 64,
) -> None:
    """Validate the compiler-selected packed GEMV staging contract."""

    minimum_physical_stages = 2 * rhs_tiles_per_group
    if (
        not _is_positive_power_of_two(block_n)
        or not _is_positive_power_of_two(block_k)
        or not 8 <= block_n <= maximum_block_n
        or block_n % 8 != 0
        or not 128 <= block_k <= 1024
        or num_stages < minimum_physical_stages
        or num_stages % rhs_tiles_per_group != 0
    ):
        raise ValueError(
            f"PyNTT {algorithm} resource contract requires a power-of-two "
            f"block_n in [8, {maximum_block_n}], a power-of-two block_k in "
            "[128, 1024], "
            "and at least two complete buffered RHS groups; got "
            f"block_n={block_n}, block_k={block_k}, "
            f"num_stages={num_stages}, rhs_tiles_per_group={rhs_tiles_per_group}."
        )


def _should_outline_packed_gemv_consumer_stage(
    *, block_k: int, reduction_group: int
) -> bool:
    """Keep small stages inline and share large static reduction bodies."""

    if block_k <= 0 or reduction_group <= 0 or block_k % reduction_group != 0:
        raise ValueError(
            "PyNTT packed GEMV consumer outlining requires a positive block_k "
            "divisible by its positive reduction_group; got "
            f"block_k={block_k}, reduction_group={reduction_group}."
        )
    return (
        block_k // reduction_group
        > PACKED_GEMV_MAXIMUM_INLINE_REDUCTION_GROUPS
    )


def _gated_delta_net_convolution_template_context(
    model: dict[str, Any],
) -> dict[str, Any]:
    microkernel = _microkernel_context(
        model,
        "triton.gated_delta_net",
        "convolution",
        required_workspace_names=(),
    )
    parameters = microkernel["parameters"]
    block_n = parameters["block_n"]
    local_n = _require_int(model.get("LocalConvDim"), "LocalConvDim", minimum=1)
    lane_count = _require_int(
        model.get("ActivationLaneCount"), "ActivationLaneCount", minimum=1
    )
    if (
        parameters["block_m"] != 1
        or parameters["block_k"] != 1
        or parameters["num_stages"] != 1
        or not _is_positive_power_of_two(block_n)
        or block_n > local_n
        or local_n % lane_count != 0
    ):
        raise ValueError(
            "PyNTT GatedDeltaNet convolution requires block_m=block_k=num_stages=1, "
            "a power-of-two block_n no larger than LocalConvDim, and a local "
            f"convolution extent divisible by its lanes; got parameters={parameters}, "
            f"local_n={local_n}, lane_count={lane_count}."
        )
    return {
        "microkernel": microkernel,
        "block_n": block_n,
        "local_n": local_n,
    }


def _gated_delta_net_recurrent_core_template_context(
    model: dict[str, Any],
) -> dict[str, Any]:
    microkernel = _microkernel_context(
        model,
        "triton.gated_delta_net",
        "recurrent_core",
        required_workspace_names=(
            "b_projection_stage",
            "a_projection_stage",
            "projection_stage",
        ),
    )
    parameters = microkernel["parameters"]
    block_n = parameters["block_n"]
    block_k = parameters["block_k"]
    num_stages = parameters["num_stages"]
    state_value_tile = _require_int(
        parameters.get("state_value_tile"), "state_value_tile", minimum=1
    )
    projection_head_capacity = _require_int(
        parameters.get("projection_head_capacity"),
        "projection_head_capacity",
        minimum=1,
    )
    projection_tma_k_atom = _require_int(
        parameters.get("projection_tma_k_atom"),
        "projection_tma_k_atom",
        minimum=1,
    )
    hidden_size = _require_int(model.get("HiddenSize"), "HiddenSize", minimum=1)
    local_value_capacity = _require_int(
        model.get("LocalValueCapacity"),
        "LocalValueCapacity",
        minimum=1,
    )
    num_key_heads = _require_int(model.get("NumKeyHeads"), "NumKeyHeads", minimum=1)
    num_value_heads = _require_int(
        model.get("NumValueHeads"), "NumValueHeads", minimum=1
    )
    key_dim = _require_int(model.get("KeyHeadDim"), "KeyHeadDim", minimum=1)
    value_dim = _require_int(model.get("ValueHeadDim"), "ValueHeadDim", minimum=1)
    if num_value_heads % num_key_heads != 0:
        raise ValueError(
            "PyNTT GatedDeltaNet recurrent core requires an integral value-head "
            f"ratio, got {num_value_heads}/{num_key_heads}."
        )
    value_heads_per_key_head = num_value_heads // num_key_heads
    block_axes = [axis for axis in model["MeshAxes"] if axis["level"] == "b"]
    block_count = prod(_require_int(axis["size"], "mesh axis size", minimum=1) for axis in block_axes)
    if block_count % num_key_heads != 0:
        raise ValueError(
            "PyNTT GatedDeltaNet recurrent core requires an integral number of "
            f"CTAs per key head, got blocks={block_count}, key_heads={num_key_heads}."
        )
    ctas_per_key_head = block_count // num_key_heads
    value_tiles_per_head = value_dim // state_value_tile
    value_tiles_per_key_head = value_heads_per_key_head * value_tiles_per_head
    tasks_per_cta = (value_tiles_per_key_head + ctas_per_key_head - 1) // ctas_per_key_head
    state_waves = (tasks_per_cta + 7) // 8
    projection_head_counts = []
    for owner in range(ctas_per_key_head):
        task_begin = owner * tasks_per_cta
        task_end = min(task_begin + tasks_per_cta, value_tiles_per_key_head)
        if task_begin >= task_end:
            projection_head_counts.append(0)
        else:
            projection_head_counts.append(
                ((task_end - 1) // value_tiles_per_head)
                - (task_begin // value_tiles_per_head)
                + 1
            )
    maximum_projection_heads_per_cta = max(projection_head_counts)
    b_projection_shape = microkernel["shared_workspace_shapes"][
        "b_projection_stage"
    ]
    a_projection_shape = microkernel["shared_workspace_shapes"][
        "a_projection_stage"
    ]
    projection_shape = microkernel["shared_workspace_shapes"]["projection_stage"]
    projection_capacity = projection_shape[1] if len(projection_shape) == 2 else 0
    expected_weight_stage_shape = (
        num_stages,
        projection_head_capacity,
        block_k // projection_tma_k_atom,
        projection_tma_k_atom,
    )
    raw_channels = model["MicroKernel"].get("TransferPipelineChannels")
    expected_channels = [
        {
            "Name": "projection",
            "SharedWorkspaceNames": [
                "b_projection_stage",
                "a_projection_stage",
            ],
        }
    ]
    b_descriptor_name = model.get("BWeightDescriptorName")
    a_descriptor_name = model.get("AWeightDescriptorName")
    b_descriptor_origin = model.get("BWeightDescriptorOriginElements")
    a_descriptor_origin = model.get("AWeightDescriptorOriginElements")
    b_descriptor_owner_indexed = model.get("BWeightDescriptorOwnerIndexed")
    a_descriptor_owner_indexed = model.get("AWeightDescriptorOwnerIndexed")
    if (
        parameters["block_m"] != 1
        or num_stages < 2
        or not _is_positive_power_of_two(block_n)
        or not _is_positive_power_of_two(block_k)
        or not _is_positive_power_of_two(state_value_tile)
        or not _is_positive_power_of_two(projection_tma_k_atom)
        or hidden_size % projection_tma_k_atom != 0
        or block_k % projection_tma_k_atom != 0
        or state_value_tile > 32
        or value_dim % state_value_tile != 0
        or ctas_per_key_head <= 0
        or maximum_projection_heads_per_cta <= 0
        or projection_head_capacity != maximum_projection_heads_per_cta
        or local_value_capacity != tasks_per_cta * state_value_tile
        or b_projection_shape != expected_weight_stage_shape
        or a_projection_shape != expected_weight_stage_shape
        or len(projection_shape) != 2
        or projection_shape[0] != 2
        or not _is_positive_power_of_two(projection_capacity)
        or projection_capacity < value_heads_per_key_head
        or raw_channels != expected_channels
        or not isinstance(b_descriptor_name, str)
        or not b_descriptor_name
        or not isinstance(a_descriptor_name, str)
        or not a_descriptor_name
        or not isinstance(b_descriptor_origin, str)
        or not b_descriptor_origin
        or not isinstance(a_descriptor_origin, str)
        or not a_descriptor_origin
        or not isinstance(b_descriptor_owner_indexed, bool)
        or not isinstance(a_descriptor_owner_indexed, bool)
    ):
        raise ValueError(
            "PyNTT GatedDeltaNet recurrent core has an invalid selected schedule "
            f"or Shared contract: parameters={parameters}, hidden_size={hidden_size}, "
            f"blocks={block_count}, key_heads={num_key_heads}, "
            f"value_heads={num_value_heads}, local_value_capacity={local_value_capacity}, "
            f"projection_shapes={b_projection_shape}/{a_projection_shape}/{projection_shape}, "
            f"channels={raw_channels}."
        )
    state_value_threads = min(state_value_tile, 32)
    state_key_threads = 32 // state_value_threads
    projection_heads_per_group = min(maximum_projection_heads_per_cta, 4)
    return {
        "microkernel": microkernel,
        "block_n": block_n,
        "block_k": block_k,
        "num_stages": num_stages,
        "num_k_tiles": (hidden_size + block_k - 1) // block_k,
        "hidden_size": hidden_size,
        "local_value_capacity": local_value_capacity,
        "state_value_tile": state_value_tile,
        "state_key_threads": state_key_threads,
        "state_value_threads": state_value_threads,
        "state_waves": state_waves,
        "state_tasks_per_cta": tasks_per_cta,
        "value_tiles_per_head": value_tiles_per_head,
        "value_tiles_per_key_head": value_tiles_per_key_head,
        "has_state_tail": state_waves * 8 != tasks_per_cta
        or tasks_per_cta * ctas_per_key_head != value_tiles_per_key_head,
        "ctas_per_key_head": ctas_per_key_head,
        "value_heads_per_key_head": value_heads_per_key_head,
        "projection_capacity": projection_capacity,
        "projection_head_capacity": projection_head_capacity,
        "projection_tma_k_atom": projection_tma_k_atom,
        "projection_weight_stage_shape": expected_weight_stage_shape[1:],
        "projection_tma_block_shape": expected_weight_stage_shape[1:],
        "b_weight_descriptor_name": b_descriptor_name,
        "a_weight_descriptor_name": a_descriptor_name,
        "b_weight_descriptor_origin": b_descriptor_origin,
        "a_weight_descriptor_origin": a_descriptor_origin,
        "b_weight_descriptor_owner_indexed": b_descriptor_owner_indexed,
        "a_weight_descriptor_owner_indexed": a_descriptor_owner_indexed,
        "maximum_projection_heads_per_cta": maximum_projection_heads_per_cta,
        "projection_heads_per_group": projection_heads_per_group,
        "projection_k_warps": 8 // projection_heads_per_group,
        "projection_groups": (
            maximum_projection_heads_per_cta + projection_heads_per_group - 1
        )
        // projection_heads_per_group,
        "block_count": block_count,
    }


def _sparse_experts_down_pipeline_template_context(
    model: dict[str, Any],
) -> dict[str, Any]:
    """Validate the selected-expert concatenated GEMV resource contract."""

    microkernel = _microkernel_context(
        model,
        "triton.sparse_experts_down",
        "concatenated_mma_smem_pipeline",
        required_workspace_names=("weight_stage",),
    )
    parameters = microkernel["parameters"]
    block_m = parameters["block_m"]
    block_n = parameters["block_n"]
    stage_k = parameters["block_k"]
    num_stages = parameters["num_stages"]
    routes_per_stage = parameters.get("routes_per_stage")
    expert_block_k = parameters.get("expert_block_k")
    if not isinstance(routes_per_stage, int) or routes_per_stage <= 0:
        raise ValueError(
            "PyNTT SparseExpertsDown requires a positive routes_per_stage parameter."
        )
    if not isinstance(expert_block_k, int) or expert_block_k <= 0:
        raise ValueError(
            "PyNTT SparseExpertsDown requires a positive expert_block_k parameter."
        )
    if block_m != 16 or stage_k != routes_per_stage * expert_block_k:
        raise ValueError(
            "PyNTT SparseExpertsDown requires block_m=16 and "
            "block_k=routes_per_stage*expert_block_k."
        )

    if model.get("ActivationTritonDType") != "tl.bfloat16" or model.get(
        "OutputTritonDType"
    ) != "tl.bfloat16":
        raise ValueError(
            "PyNTT SparseExpertsDown concatenated MMA requires BF16 activation "
            "and output dtypes."
        )
    num_top_k = _require_int(model.get("NumTopK"), "NumTopK", minimum=1)
    local_k = _require_int(
        model.get("LocalIntermediateSize"), "LocalIntermediateSize", minimum=1
    )
    local_n = _require_int(
        model.get("LocalOutputSize"), "LocalOutputSize", minimum=1
    )
    token_count = _require_fixed_positive_dim(
        model["ActivationShape"][0], "SparseExpertsDown token count"
    )
    if num_top_k % routes_per_stage != 0:
        raise ValueError(
            "PyNTT SparseExpertsDown requires top-k divisible by "
            f"routes_per_stage, got {num_top_k}/{routes_per_stage}."
        )
    if local_k % expert_block_k != 0:
        raise ValueError(
            "PyNTT SparseExpertsDown requires local K divisible by "
            f"expert_block_k, got {local_k}/{expert_block_k}."
        )
    if local_n % block_n != 0:
        raise ValueError(
            "PyNTT SparseExpertsDown requires local N divisible by block_n, "
            f"got {local_n}/{block_n}."
        )

    workspace_shape = microkernel["shared_workspace_shapes"]["weight_stage"]
    if workspace_shape != (num_stages, block_n, stage_k):
        raise ValueError(
            "PyNTT SparseExpertsDown weight workspace must be "
            f"[stages,block_n,block_k], got {workspace_shape}."
        )
    if len(model["DownWeightShape"]) != 3:
        raise ValueError("PyNTT SparseExpertsDown weight must have rank 3.")
    if _fixed(model["DownWeightShape"][1]) != local_n or _fixed(
        model["DownWeightShape"][2]
    ) != local_k:
        raise ValueError(
            "PyNTT SparseExpertsDown weight local shape does not match its "
            f"selected N/K extents: {model['DownWeightShape']} vs "
            f"{local_n}/{local_k}."
        )

    return {
        "microkernel": microkernel,
        "block_m": block_m,
        "block_n": block_n,
        "stage_k": stage_k,
        "num_stages": num_stages,
        "routes_per_stage": routes_per_stage,
        "expert_block_k": expert_block_k,
        "route_groups": num_top_k // routes_per_stage,
        "expert_k_tiles": local_k // expert_block_k,
        "n_tiles": local_n // block_n,
        "token_count": token_count,
        "sequence_count": (
            token_count
            * (local_n // block_n)
            * (num_top_k // routes_per_stage)
            * (local_k // expert_block_k)
        ),
    }


def _packed_gemv_rhs_physical_view(
    model: dict[str, Any],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the fixed backing view addressed by a packed GEMV descriptor."""

    rhs_pointer = model.get("Rhs")
    if not isinstance(rhs_pointer, dict):
        raise ValueError("PyNTT packed GEMV requires RHS pointer metadata.")
    if rhs_pointer.get("DistributedStorageKind") == "CanonicalGlobal":
        rhs_shape = rhs_pointer.get("GlobalShape")
        rhs_strides = rhs_pointer.get("Strides")
    else:
        rhs_shape = model["RhsShape"]
        rhs_strides = model["RhsStrides"]
    if not isinstance(rhs_shape, list) or not isinstance(rhs_strides, list):
        raise ValueError(
            "PyNTT packed GEMV RHS physical shape/stride metadata is incomplete."
        )
    fixed_shape = tuple(_fixed(value) for value in rhs_shape)
    fixed_strides = tuple(_fixed(value) for value in rhs_strides)
    if any(value is None for value in fixed_shape + fixed_strides):
        raise ValueError(
            "PyNTT packed GEMV TMA requires fixed physical RHS shape and strides."
        )
    return (
        tuple(int(value) for value in fixed_shape),
        tuple(int(value) for value in fixed_strides),
    )


def _packed_gemv_pipeline_template_context(
    model: dict[str, Any],
    *,
    expected_family: str = "triton.matmul",
    expected_variant: str = "simt_fma_smem_pipeline",
    expected_rhs_dtype: str = "bfloat16",
    expected_k_vector_lanes: int = 8,
    require_operand_scales: bool = False,
    required_workspace_names: tuple[str, ...] | None = None,
    reduction_group: int = 32,
) -> dict[str, Any]:
    """Prepare a selected packed, shared-staged SIMT GEMV algorithm."""

    context = _matmul_template_context(
        model,
        gemv=True,
        variant=expected_variant,
        expected_family=expected_family,
        required_workspace_names=required_workspace_names,
    )
    if (
        model["LhsDType"] != "bfloat16"
        or model["RhsDType"] != expected_rhs_dtype
        or model["OutputDType"] != "bfloat16"
    ):
        raise ValueError(
            "PyNTT packed GEMV pipeline has an incompatible dtype contract: "
            f"expected=bfloat16/{expected_rhs_dtype}/bfloat16, "
            f"actual={model['LhsDType']}/{model['RhsDType']}/{model['OutputDType']}."
        )
    if bool(model.get("HasOperandScales")) != require_operand_scales:
        raise ValueError(
            "PyNTT packed GEMV operand-scale contract does not match its selected "
            f"variant {expected_variant!r}."
        )
    rhs_layout = _matmul_rhs_layout(model)
    if model["TransposeA"]:
        raise ValueError("PyNTT packed GEMV pipeline requires lhs=[M,K].")
    rhs_lane_shape = _matmul_n_lane_shape(model, "Rhs")
    output_lane_shape = _matmul_n_lane_shape(model, "Output")
    if rhs_layout != "k_major":
        raise ValueError(
            "PyNTT shared-staged GEMV requires a K-major packed RHS."
        )
    if (
        model["TransposeB"]
        or rhs_lane_shape != (8,)
        or output_lane_shape != (8,)
        or int(model["RhsKPackLaneCount"]) != 2
        or int(model["RhsKVectorLaneCount"]) != expected_k_vector_lanes
    ):
        raise ValueError(
            "PyNTT K-major GEMV pipeline requires "
            f"rhs=[K/{2 * expected_k_vector_lanes},N/8]"
            f"<8,2,{expected_k_vector_lanes}> and output=[M,N/8]<8>."
        )
    if len(model["LhsShape"]) != 2 or len(model["RhsShape"]) != 2 or len(
        model["OutputShape"]
    ) != 2:
        raise ValueError(
            "PyNTT packed GEMV pipeline currently requires rank-2 operands."
        )
    descriptor_name = model.get("RhsDescriptorName")
    if not isinstance(descriptor_name, str) or not descriptor_name:
        raise ValueError(
            "PyNTT packed GEMV pipeline requires a host RHS descriptor."
        )
    descriptor_origin_elements = model.get("RhsDescriptorOriginElements")
    if not isinstance(descriptor_origin_elements, str) or not descriptor_origin_elements:
        raise ValueError(
            "PyNTT packed GEMV pipeline requires an RHS descriptor origin."
        )
    rhs_global_offsets = model.get("RhsGlobalOffsets")
    if not isinstance(rhs_global_offsets, list) or len(rhs_global_offsets) != 2:
        raise ValueError(
            "PyNTT packed GEMV pipeline requires two RHS global offsets."
        )

    block_n = context["block_n"]
    block_k = context["block_k"]
    num_stages = context["microkernel"]["parameters"]["num_stages"]
    _validate_packed_gemv_pipeline_resource_contract(
        algorithm="packed GEMV pipeline",
        block_n=block_n,
        block_k=block_k,
        num_stages=num_stages,
    )
    n_lane = int(model["RhsNVectorLaneCount"])
    k_pack = int(model["RhsKPackLaneCount"])
    k_lane = int(model["RhsKVectorLaneCount"])
    k_atom = k_pack * k_lane
    k = _fixed(context["k"])
    if k is None or k <= 0 or k % block_k != 0:
        raise ValueError(
            "PyNTT packed GEMV pipeline requires a fixed K divisible by block_k."
        )
    rhs_k = _fixed(context["rhs_k"])
    if rhs_k != k:
        raise ValueError(
            "PyNTT packed GEMV pipeline requires equal fixed lhs/rhs K "
            f"dimensions, got lhs K={k} and rhs K={rhs_k}."
        )
    rhs_physical_shape, rhs_physical_strides = _packed_gemv_rhs_physical_view(model)
    normalized_rhs_strides = _normalize_singleton_strides(
        rhs_physical_shape,
        rhs_physical_strides,
    )
    lhs_stage_extent = context["microkernel"]["parameters"].get(
        "lhs_stage_extent"
    )
    use_complete_consumer_lhs_stage = lhs_stage_extent is not None
    lhs_copy_segments: tuple[dict[str, Any], ...] = ()
    if use_complete_consumer_lhs_stage:
        if (
            _fixed(model["LhsStrides"][-1]) != 1
            or not _is_positive_power_of_two(lhs_stage_extent)
            or lhs_stage_extent < k
            or k < 4096
            or k % 1024 != 0
        ):
            raise ValueError(
                "PyNTT complete consumer LHS staging requires a contiguous "
                "fixed K >= 4096 divisible by 1024 and a power-of-two Shared "
                f"extent covering K; got K={k}, lhs_stage_extent="
                f"{lhs_stage_extent}, lhs_stride={_fixed(model['LhsStrides'][-1])}."
            )
        raw_segments = _power_of_two_segments(k)
        if any(extent < 1024 for _, extent in raw_segments):
            raise ValueError(
                "PyNTT complete consumer LHS staging requires every copy "
                f"segment to cover at least 1024 elements, got {raw_segments}."
            )
        lhs_copy_segments = tuple(
            {
                "offset": offset,
                "extent": extent,
                "access": _matmul_lhs_access(
                    model,
                    output_batch_rank=0,
                    m_expr="0",
                    k_expr="pipeline_input_copy_k",
                    coordinate_shape=_coordinate_shape((1, extent)),
                ),
            }
            for offset, extent in raw_segments
        )
    if (
        rhs_physical_shape[-2] != k // k_atom
        or normalized_rhs_strides[-1] != 1
        or normalized_rhs_strides[-2] <= 0
    ):
        raise ValueError(
            "PyNTT packed GEMV TMA requires a positive-stride "
            f"physical RHS [K/{k_atom},N/{n_lane}]"
            f"<{n_lane},{k_pack},{k_lane}> view."
        )
    max_n = _max_value(context["n"])
    if max_n is None:
        raise ValueError(
            "PyNTT packed GEMV pipeline requires a bounded N dimension."
        )
    num_k_tiles = k // block_k
    num_n_tiles = (max_n + block_n - 1) // block_n
    max_sequence_count = num_n_tiles * num_k_tiles
    if max_sequence_count > (2**31 - 1):
        raise ValueError(
            "PyNTT packed GEMV pipeline sequence exceeds the signed int32 "
            f"pipe ABI: {max_sequence_count}."
        )

    if (
        block_k % k_atom != 0
        or block_n % n_lane != 0
        or 32 % n_lane != 0
    ):
        raise ValueError(
            "PyNTT K-major GEMV staging requires block_k divisible by "
            "KPack*KVector, and both block_n and the 32-element reduction "
            "group divisible by NVector."
        )
    packed_k_outer = block_k // k_atom
    packed_n_outer = block_n // n_lane
    rhs_pointer = model["Rhs"]
    rhs_k_transfer = _tma_local_axis_transfer(
        rhs_pointer,
        -2,
        rhs_global_offsets[-2],
        local_offset=0,
        tile_index="k_tile",
        tile_stride=packed_k_outer,
        tile_extent=packed_k_outer,
        context="packed GEMV RHS K",
    )
    rhs_n_transfer = _tma_local_axis_transfer(
        rhs_pointer,
        -1,
        rhs_global_offsets[-1],
        local_offset=0,
        tile_index="n_tile",
        tile_stride=packed_n_outer,
        tile_extent=packed_n_outer,
        context="packed GEMV RHS N",
    )
    if not _is_positive_power_of_two(reduction_group):
        raise ValueError(
            "PyNTT K-major GEMV reduction group must be a positive power of two, "
            f"got {reduction_group}."
        )
    if block_k % reduction_group != 0:
        raise ValueError(
            "PyNTT K-major GEMV staging requires block_k divisible by "
            f"the warp reduction group {reduction_group}."
        )
    target_worker_width = int(model["TargetWorkerWidth"])
    consumer_warps = int(model["NumWarps"])
    consumer_layout = _packed_gemv_consumer_layout(
        block_n=block_n,
        reduction_group=reduction_group,
        consumer_warps=consumer_warps,
        target_worker_width=target_worker_width,
    )
    tma_contiguous_extent = n_lane * k_lane
    tma_block_shape = (
        tuple(rhs_k_transfer["block_shape"])
        + tuple(rhs_n_transfer["block_shape"])
        + (k_pack, tma_contiguous_extent)
    )
    if len(tma_block_shape) > 5:
        raise ValueError(
            "PyNTT packed GEMV TMA transfer exceeds the hardware rank-5 limit: "
            f"block_shape={tma_block_shape}."
        )
    shared_weight_indices = (
        _tma_shared_axis_coordinates(
            f"shared_k // {k_atom}", rhs_k_transfer
        )
        + _tma_shared_axis_coordinates(
            f"local_n // {n_lane}", rhs_n_transfer
        )
        + (
            f"shared_payload // {tma_contiguous_extent}",
            f"shared_payload % {tma_contiguous_extent}",
        )
    )
    pipeline_context = {
        "num_k_tiles": num_k_tiles,
        "num_n_tiles": num_n_tiles,
        "runtime_num_n_tiles": f"tl.cdiv(active_n, {block_n})",
        "num_stages": num_stages,
        "k_atom": k_atom,
        "packed_k_outer": packed_k_outer,
        "packed_n_outer": packed_n_outer,
        "reduction_group": reduction_group,
        "reduction_groups_per_stage": block_k // reduction_group,
        "outline_consumer_stage": _should_outline_packed_gemv_consumer_stage(
            block_k=block_k,
            reduction_group=reduction_group,
        ),
        "shared_stage_shape": tma_block_shape,
        "tma_block_shape": tma_block_shape,
        "tma_contiguous_extent": tma_contiguous_extent,
        "rhs_descriptor_name": descriptor_name,
        "rhs_descriptor_origin_elements": descriptor_origin_elements,
        "rhs_descriptor_offsets": (
            tuple(rhs_k_transfer["coordinates"])
            + tuple(rhs_n_transfer["coordinates"])
            + ("0", descriptor_origin_elements)
        ),
        "shared_weight_indices": shared_weight_indices,
        "consumer_size_per_thread": consumer_layout["size_per_thread"],
        "consumer_threads_per_warp": consumer_layout["threads_per_warp"],
        "consumer_warps_per_cta": consumer_layout["warps_per_cta"],
        "consumer_weight_layout_name": (
            f"{model['FunctionName']}__weight_layout"
        ),
        "consumer_lhs_layout_name": f"{model['FunctionName']}__lhs_layout",
        "consumer_output_layout_name": (
            f"{model['FunctionName']}__output_layout"
        ),
        "use_complete_consumer_lhs_stage": use_complete_consumer_lhs_stage,
        "lhs_stage_extent": lhs_stage_extent,
        "lhs_copy_segments": lhs_copy_segments,
        "consumer_input_copy_layout_name": (
            f"{model['FunctionName']}__input_copy_layout"
        ),
        "input_copy_size_per_thread": 4,
        "pipeline_lhs_access": _matmul_lhs_access(
            model,
            output_batch_rank=0,
            m_expr="0",
            k_expr="pipeline_offs_k",
            coordinate_shape=_coordinate_shape((reduction_group,)),
        ),
        "pipeline_output_access": _contiguous_vector_axis_access(
            ("0", "0"),
            model["OutputStrides"],
            tensor_shape=model["OutputShape"],
            packed_axis=1,
            logical_index="pipeline_output_n",
            lane_count=n_lane,
            coordinate_shape=_coordinate_shape((block_n,)),
        ),
        "pipeline_addend_access": (
            _contiguous_vector_axis_access(
                ("0", "0"),
                model["AddendStrides"],
                tensor_shape=model.get("AddendShape"),
                packed_axis=1,
                logical_index="pipeline_output_n",
                lane_count=n_lane,
                coordinate_shape=_coordinate_shape((block_n,)),
            )
            if model["HasAddend"]
            else None
        ),
        "pipeline_output_mask": "(pipeline_output_n < active_n)",
        "pipeline_stats_accesses": (
            tuple(
                _matmul_stats_access(
                    model,
                    output_batch_rank=0,
                    m_expr="0",
                    component=component,
                    coordinate_shape=None,
                )
                for component in range(2 if model["UseMean"] else 1)
            )
            if model["HasNormStats"]
            else ()
        ),
    }
    context.update(pipeline_context)
    return context


def _paged_attention_merge_matmul_template_context(
    model: dict[str, Any],
) -> dict[str, Any]:
    """Prepare direct PagedAttentionMerge-to-packed-GEMV coordinates."""

    matmul = model.get("Matmul")
    merge = model.get("Merge")
    if not isinstance(matmul, dict) or not isinstance(merge, dict):
        raise ValueError(
            "PyNTT fused paged-attention merge/matmul requires Merge and Matmul metadata."
        )
    matmul = dict(matmul)
    for name in (
        "KernelConfig",
        "MeshAxes",
        "NoInline",
        "NumWarps",
        "ProducerRegisters",
        "ProducerWarps",
        "RegisterGranularity",
        "RegistersPerThreadLimit",
        "TargetWorkerWidth",
    ):
        if name in model:
            matmul[name] = model[name]
    context = _packed_gemv_pipeline_template_context(
        matmul,
        expected_family="triton.paged_attention_merge_matmul",
    )
    merge_context = _paged_attention_merge_template_context(merge)
    local_query_tokens = _fixed(merge_context["local_query_tokens"])
    local_q_heads = _fixed(merge_context["local_q_heads"])
    local_head_dimension = int(merge_context["local_head_dimension"])
    lhs_k = _fixed(context["k"])
    if local_query_tokens != 1 or local_q_heads != 1:
        raise ValueError(
            "PyNTT fused paged-attention merge/matmul currently requires one "
            f"local query and head, got query={local_query_tokens}, heads={local_q_heads}."
        )
    if lhs_k != local_head_dimension:
        raise ValueError(
            "PyNTT fused paged-attention merge/matmul requires the merged head "
            f"dimension to equal GEMV K, got head_dim={local_head_dimension}, K={lhs_k}."
        )
    if context["num_k_tiles"] != 1:
        raise ValueError(
            "PyNTT fused paged-attention merge/matmul requires one K tile so the "
            "merged register value is consumed exactly once per output tile."
        )
    reduction_groups = int(context["reduction_groups_per_stage"])
    if not _is_positive_power_of_two(reduction_groups):
        raise ValueError(
            "PyNTT fused paged-attention merge/matmul requires a power-of-two "
            f"number of 32-element K groups, got {reduction_groups}."
        )

    split_steps: list[dict[str, Any]] = []
    group_variables: list[str | None] = [None] * reduction_groups
    pending = [("merged_lhs_groups", tuple(range(reduction_groups)))]
    split_id = 0
    while pending:
        source, group_indices = pending.pop(0)
        if len(group_indices) == 1:
            group_variables[group_indices[0]] = source
            continue
        even = f"merged_lhs_split_{split_id}_even"
        odd = f"merged_lhs_split_{split_id}_odd"
        split_steps.append(
            {
                "source": source,
                "even": even,
                "odd": odd,
                "reshape_extent": len(group_indices) // 2,
                "needs_reshape": len(group_indices) > 2,
            }
        )
        pending.append((even, group_indices[::2]))
        pending.append((odd, group_indices[1::2]))
        split_id += 1
    if any(variable is None for variable in group_variables):
        raise AssertionError("incomplete fused GEMV register split plan")

    context.update(
        matmul=matmul,
        merge=merge,
        merge_context=merge_context,
        merged_lhs_group_variables=tuple(group_variables),
        merged_lhs_split_steps=tuple(split_steps),
        reduction_group_indices=tuple(range(reduction_groups)),
    )
    return context


def _packed_fp8_gemv_pipeline_template_context(
    model: dict[str, Any],
) -> dict[str, Any]:
    """Prepare the E4M3-weight, statically scaled SIMT GEMV algorithm."""

    return _packed_gemv_pipeline_template_context(
        model,
        expected_variant="simt_fp8_fma_smem_pipeline",
        expected_rhs_dtype="float8e4m3fn",
        expected_k_vector_lanes=16,
        require_operand_scales=True,
    )


def _packed_block_fp8_gemv_pipeline_template_context(
    model: dict[str, Any],
) -> dict[str, Any]:
    """Prepare dynamic K-group activation and 2-D block-scaled FP8 GEMV."""

    context = _packed_gemv_pipeline_template_context(
        model,
        expected_variant="simt_block_fp8_fma_smem_pipeline",
        expected_rhs_dtype="float8e4m3fn",
        expected_k_vector_lanes=16,
        require_operand_scales=False,
        reduction_group=_require_int(
            model.get("WeightBlockK"), "WeightBlockK", minimum=1
        ),
    )
    if not bool(model.get("HasRhsBlockScale")):
        raise ValueError(
            "PyNTT block-scaled FP8 GEMV requires an rhs block scale and no "
            "precomputed lhs scale."
        )
    block_n = _require_int(model.get("WeightBlockN"), "WeightBlockN", minimum=1)
    block_k = _require_int(model.get("WeightBlockK"), "WeightBlockK", minimum=1)
    if block_n != 128 or block_k != 128:
        raise ValueError(
            "PyNTT block-scaled FP8 GEMV currently implements the official "
            f"128x128 scale ABI, got {block_n}x{block_k}."
        )

    scale_shape = _require_list(model.get("RhsScaleShape"), "RhsScaleShape")
    scale_strides = _require_list(
        model.get("RhsScaleStrides"), "RhsScaleStrides"
    )
    if len(scale_shape) != 2 or len(scale_strides) != 2:
        raise ValueError(
            "PyNTT block-scaled FP8 GEMV requires a rank-2 rhs scale buffer."
        )
    n_lane = int(model["OutputNVectorLaneCount"])
    global_output_n = _pointer_local_vector_to_global_scalar_coordinate(
        model["Output"], 1, "output_n", n_lane
    )
    global_k_start = _pointer_local_to_global_coordinate(
        model["Lhs"],
        1,
        f"k_tile * {context['block_k']} + "
        f"k_group * {context['reduction_group']}",
    )
    context.update(
        weight_block_n=block_n,
        weight_block_k=block_k,
        global_output_n=global_output_n,
        global_k_start=global_k_start,
        rhs_scale_access=_tensor_access(
            ("weight_scale_n", "weight_scale_k"),
            scale_strides,
            coordinate_shape=_coordinate_shape((context["block_n"],)),
            global_coordinate_axes=(0, 1),
        ),
    )
    return context


def _packed_block_fp8_mma_gemv_pipeline_template_context(
    model: dict[str, Any],
) -> dict[str, Any]:
    """Prepare direct TMA-to-MMA block-scaled FP8 GEMV."""

    context = _matmul_template_context(
        model,
        gemv=True,
        variant="mma_block_fp8_smem_pipeline",
        required_workspace_names=("rhs_stage", "lhs_quantized", "lhs_scale"),
    )
    if (
        model["LhsDType"] != "bfloat16"
        or model["RhsDType"] != "float8e4m3fn"
        or model["OutputDType"] != "bfloat16"
    ):
        raise ValueError(
            "PyNTT block-FP8 MMA GEMV requires BF16/E4M3/BF16 operands."
        )
    if not bool(model.get("HasRhsBlockScale")) or bool(
        model.get("HasOperandScales")
    ):
        raise ValueError(
            "PyNTT block-FP8 MMA GEMV requires a block-scaled RHS and dynamic lhs scaling."
        )
    if (
        model.get("RhsLayout") != "n_major_k_packed"
        or not bool(model.get("TransposeB"))
        or _matmul_n_lane_shape(model, "Rhs")
        or _matmul_n_lane_shape(model, "Output") != (8,)
        or int(model["RhsKPackLaneCount"]) != 2
        or int(model["RhsKVectorLaneCount"]) != 16
    ):
        raise ValueError(
            "PyNTT block-FP8 MMA GEMV requires "
            "rhs=[N,K/32]<2,16>, output=[M,N/8]<8>, and logical transpose_b."
        )
    if (
        len(model["LhsShape"]) != 2
        or len(model["RhsShape"]) != 2
        or len(model["OutputShape"]) != 2
    ):
        raise ValueError(
            "PyNTT block-FP8 MMA GEMV currently requires rank-2 operands."
        )

    microkernel = context["microkernel"]
    block_n = microkernel["parameters"]["block_n"]
    block_k = microkernel["parameters"]["block_k"]
    num_stages = microkernel["parameters"]["num_stages"]
    transfer_block_k = microkernel["parameters"].get("transfer_block_k")
    if not isinstance(transfer_block_k, int) or transfer_block_k <= 0:
        raise ValueError(
            "PyNTT block-FP8 MMA GEMV requires a positive selected "
            f"transfer_block_k, got {transfer_block_k!r}."
        )
    weight_block_n = _require_int(
        model.get("WeightBlockN"), "WeightBlockN", minimum=1
    )
    reduction_group = _require_int(
        model.get("WeightBlockK"), "WeightBlockK", minimum=1
    )
    if (
        block_n % 16 != 0
        or block_k % reduction_group != 0
        or reduction_group % transfer_block_k != 0
    ):
        raise ValueError(
            "PyNTT block-FP8 MMA GEMV requires block_n divisible by 16 and "
            "a nested block_k/scale-group/transfer-K hierarchy, got "
            f"block_n={block_n}, block_k={block_k}, "
            f"scale_group_k={reduction_group}, transfer_k={transfer_block_k}."
        )
    _validate_packed_gemv_pipeline_resource_contract(
        algorithm="block-FP8 MMA GEMV pipeline",
        block_n=block_n,
        block_k=block_k,
        num_stages=num_stages,
        maximum_block_n=128,
    )
    reduction_groups_per_stage = block_k // reduction_group
    transfer_chunks_per_group = reduction_group // transfer_block_k
    expected_workspace_shape = (
        num_stages,
        reduction_groups_per_stage,
        transfer_chunks_per_group,
        block_n,
        transfer_block_k,
    )
    if microkernel["shared_workspace_shapes"]["rhs_stage"] != expected_workspace_shape:
        raise ValueError(
            "PyNTT block-FP8 MMA GEMV Shared workspace must be "
            f"{expected_workspace_shape}, got "
            f"{microkernel['shared_workspace_shapes']['rhs_stage']}."
        )
    expected_lhs_quantized_shape = (
        reduction_groups_per_stage,
        reduction_group,
    )
    if (
        microkernel["shared_workspace_shapes"]["lhs_quantized"]
        != expected_lhs_quantized_shape
    ):
        raise ValueError(
            "PyNTT block-FP8 MMA GEMV quantized activation workspace must be "
            f"{expected_lhs_quantized_shape}, got "
            f"{microkernel['shared_workspace_shapes']['lhs_quantized']}."
        )
    expected_lhs_scale_shape = (reduction_groups_per_stage, 1)
    if (
        microkernel["shared_workspace_shapes"]["lhs_scale"]
        != expected_lhs_scale_shape
    ):
        raise ValueError(
            "PyNTT block-FP8 MMA GEMV activation scale workspace must be "
            f"{expected_lhs_scale_shape}, got "
            f"{microkernel['shared_workspace_shapes']['lhs_scale']}."
        )

    k = _fixed(context["k"])
    rhs_k = _fixed(context["rhs_k"])
    if k is None or k <= 0 or rhs_k != k or k % block_k != 0:
        raise ValueError(
            "PyNTT block-FP8 MMA GEMV requires equal fixed lhs/rhs K divisible "
            f"by block_k, got lhs={k}, rhs={rhs_k}, block_k={block_k}."
        )
    max_n = _max_value(context["n"])
    if max_n is None:
        raise ValueError(
            "PyNTT block-FP8 MMA GEMV requires a bounded N dimension."
        )
    descriptor_name = model.get("RhsDescriptorName")
    descriptor_origin_elements = model.get("RhsDescriptorOriginElements")
    rhs_global_offsets = model.get("RhsGlobalOffsets")
    if not isinstance(descriptor_name, str) or not descriptor_name:
        raise ValueError(
            "PyNTT block-FP8 MMA GEMV requires a host RHS descriptor."
        )
    if not isinstance(descriptor_origin_elements, str) or not descriptor_origin_elements:
        raise ValueError(
            "PyNTT block-FP8 MMA GEMV requires an RHS descriptor origin."
        )
    if not isinstance(rhs_global_offsets, list) or len(rhs_global_offsets) != 2:
        raise ValueError(
            "PyNTT block-FP8 MMA GEMV requires two RHS global offsets."
        )

    k_atom = int(model["RhsKPackLaneCount"]) * int(
        model["RhsKVectorLaneCount"]
    )
    transfer_outer = transfer_block_k // k_atom
    group_chunk_transfers = tuple(
        tuple(
            _tma_local_axis_transfer(
                model["Rhs"],
                1,
                rhs_global_offsets[1],
                local_offset=chunk * transfer_outer,
                tile_index=(
                    f"k_tile * {reduction_groups_per_stage} + {group}"
                ),
                tile_stride=reduction_group // k_atom,
                tile_extent=transfer_outer,
                context=(
                    "N-major K-packed GEMV RHS "
                    f"K group {group} transfer chunk {chunk}"
                ),
            )
            for chunk in range(transfer_chunks_per_group)
        )
        for group in range(reduction_groups_per_stage)
    )
    if any(
        transfer["is_block_cyclic"]
        or tuple(transfer["block_shape"]) != (transfer_outer,)
        or len(transfer["coordinates"]) != 1
        for group in group_chunk_transfers
        for transfer in group
    ):
        raise ValueError(
            "PyNTT block-FP8 MMA GEMV requires a contiguous packed K axis."
        )
    n_descriptor_plan = _n_major_k_packed_gemv_descriptor_n_plan(
        model["Rhs"], block_n
    )
    rhs_descriptor_chunks_per_tile = 1
    tma_block_shape = tuple(n_descriptor_plan["block_shape"]) + (
        transfer_block_k,
    )
    rhs_scalar_bytes = TMA_DTYPE_ITEM_SIZES.get(model["RhsDType"])
    if rhs_scalar_bytes is None:
        raise ValueError(
            "PyNTT block-FP8 MMA GEMV cannot size its RHS Shared aliases for "
            f"dtype {model['RhsDType']!r}."
        )
    rhs_descriptor_offsets = tuple(
        tuple(
            tuple("0" for _ in n_descriptor_plan["block_shape"])
            + (
                f"(({transfer['coordinates'][0]}) * {k_atom}) + "
                f"({descriptor_origin_elements})",
            )
            for transfer in group
        )
        for group in group_chunk_transfers
    )

    num_k_tiles = k // block_k
    num_n_tiles = (max_n + block_n - 1) // block_n
    n_tiles_per_activation_batch = 2
    if num_n_tiles * num_k_tiles > 2**31 - 1:
        raise ValueError(
            "PyNTT block-FP8 MMA GEMV pipe sequence exceeds signed int32."
        )
    scale_shape = _require_list(model.get("RhsScaleShape"), "RhsScaleShape")
    scale_strides = _require_list(
        model.get("RhsScaleStrides"), "RhsScaleStrides"
    )
    if len(scale_shape) != 2 or len(scale_strides) != 2:
        raise ValueError(
            "PyNTT block-FP8 MMA GEMV requires a rank-2 RHS scale buffer."
        )

    n_lane = int(model["OutputNVectorLaneCount"])
    global_output_n = _pointer_local_vector_to_global_scalar_coordinate(
        model["Output"], 1, "output_n", n_lane
    )
    context.update(
        block_n=block_n,
        block_k=block_k,
        num_stages=num_stages,
        num_k_tiles=num_k_tiles,
        num_n_tiles=num_n_tiles,
        n_tiles_per_activation_batch=n_tiles_per_activation_batch,
        runtime_num_n_tiles=f"tl.cdiv(active_n, {block_n})",
        reduction_group=reduction_group,
        reduction_groups_per_stage=reduction_groups_per_stage,
        transfer_block_k=transfer_block_k,
        transfer_chunks_per_group=transfer_chunks_per_group,
        weight_block_n=weight_block_n,
        weight_block_k=reduction_group,
        rhs_descriptor_name=descriptor_name,
        tma_block_shape=tma_block_shape,
        rhs_descriptor_entries_per_owner=(
            num_n_tiles * rhs_descriptor_chunks_per_tile
        ),
        rhs_descriptor_chunks_per_tile=rhs_descriptor_chunks_per_tile,
        rhs_descriptor_offsets=rhs_descriptor_offsets,
        rhs_scalar_bytes=rhs_scalar_bytes,
        transfer_stage_bytes=(
            num_stages * block_n * transfer_block_k * rhs_scalar_bytes
        ),
        tma_n_elements=prod(tma_block_shape[:-1]),
        pipeline_lhs_access=_matmul_lhs_access(
            model,
            output_batch_rank=0,
            m_expr="0",
            k_expr="pipeline_offs_k",
            coordinate_shape=_coordinate_shape((reduction_group,)),
        ),
        pipeline_output_access=_contiguous_vector_axis_access(
            ("0", "0"),
            model["OutputStrides"],
            tensor_shape=model["OutputShape"],
            packed_axis=1,
            logical_index="pipeline_output_n",
            lane_count=n_lane,
            coordinate_shape=_coordinate_shape((block_n,)),
        ),
        pipeline_addend_access=(
            _contiguous_vector_axis_access(
                ("0", "0"),
                model["AddendStrides"],
                tensor_shape=model.get("AddendShape"),
                packed_axis=1,
                logical_index="pipeline_output_n",
                lane_count=n_lane,
                coordinate_shape=_coordinate_shape((block_n,)),
            )
            if model["HasAddend"]
            else None
        ),
        pipeline_output_mask="(pipeline_output_n < active_n)",
        pipeline_stats_accesses=(
            tuple(
                _matmul_stats_access(
                    model,
                    output_batch_rank=0,
                    m_expr="0",
                    component=component,
                    coordinate_shape=None,
                )
                for component in range(2 if model["UseMean"] else 1)
            )
            if model["HasNormStats"]
            else ()
        ),
        global_output_n=global_output_n,
        rhs_scale_access=_tensor_access(
            ("weight_scale_n", "weight_scale_k"),
            scale_strides,
            coordinate_shape=_coordinate_shape((block_n,)),
            global_coordinate_axes=(0, 1),
        ),
    )
    return context


def _packed_qkv_gemv_pipeline_template_context(
    model: dict[str, Any],
    *,
    expected_variant: str = "simt_fma_smem_pipeline",
    expected_weight_dtype: str = "bfloat16",
    expected_k_vector_lanes: int = 8,
    require_operand_scales: bool = False,
) -> dict[str, Any]:
    """Prepare one selected K-major pipeline over fused local QKV weights."""

    logical_output_shapes = {
        prefix: _packed_qkv_logical_output_shape(model, prefix)
        for prefix in ("Q", "K", "V")
    }
    microkernel = _microkernel_context(
        model,
        "triton.qkv_parallel_linear",
        expected_variant,
    )
    if (
        model["InputDType"] != "bfloat16"
        or model["WeightDType"] != expected_weight_dtype
        or model["OutputDType"] != "bfloat16"
    ):
        raise ValueError(
            "PyNTT packed QKV GEMV pipeline has an incompatible dtype contract: "
            f"expected=bfloat16/{expected_weight_dtype}/bfloat16, "
            f"actual={model['InputDType']}/{model['WeightDType']}/{model['OutputDType']}."
        )
    if bool(model.get("HasOperandScales")) != require_operand_scales:
        raise ValueError(
            "PyNTT packed QKV operand-scale contract does not match its selected "
            f"variant {expected_variant!r}."
        )
    if model.get("RhsLayout") != "k_major":
        raise ValueError(
            "PyNTT packed QKV GEMV pipeline requires K-major packed weights."
        )
    if (
        int(model["NPackedLaneCount"]) != 1
        or int(model["NVectorLaneCount"]) != 8
        or int(model["KPackLaneCount"]) != 2
        or int(model["KVectorLaneCount"]) != expected_k_vector_lanes
    ):
        raise ValueError(
            "PyNTT packed QKV GEMV pipeline requires "
            f"weight=[K/{2 * expected_k_vector_lanes},N/8]"
            f"<8,2,{expected_k_vector_lanes}> and output=[M,N/8]<8>."
        )
    if (
        len(model["InputShape"]) != 2
        or len(model["WeightShape"]) != 2
        or any(len(model[f"{prefix}OutputShape"]) != 2 for prefix in ("Q", "K", "V"))
    ):
        raise ValueError(
            "PyNTT packed QKV GEMV pipeline requires rank-2 operands."
        )
    if _fixed(model["InputStrides"][-1]) != 1:
        raise ValueError(
            "PyNTT packed QKV GEMV consumer cp.async requires a contiguous K axis."
        )

    block_n = microkernel["parameters"]["block_n"]
    block_k = microkernel["parameters"]["block_k"]
    num_stages = microkernel["parameters"]["num_stages"]
    if expected_variant == "simt_fma_smem_pipeline":
        _validate_packed_gemv_pipeline_resource_contract(
            algorithm="packed QKV GEMV pipeline",
            block_n=block_n,
            block_k=block_k,
            num_stages=num_stages,
        )

    n_lane = int(model["NVectorLaneCount"])
    k_pack = int(model["KPackLaneCount"])
    k_lane = int(model["KVectorLaneCount"])
    k_atom = k_pack * k_lane
    if block_n % n_lane != 0 or block_k % k_atom != 0:
        raise ValueError(
            "PyNTT packed QKV GEMV tile is incompatible with its packed lanes: "
            f"block_n={block_n}, block_k={block_k}, n_lane={n_lane}, "
            f"k_atom={k_atom}."
        )

    k = _fixed(model["InputShape"][-1])
    if k is None or k <= 0 or k % block_k != 0:
        raise ValueError(
            "PyNTT packed QKV GEMV requires a fixed K divisible by block_k."
        )
    projection_ns = _packed_qkv_fixed_projection_ns(model)
    if any(value % n_lane != 0 for value in projection_ns.values()):
        raise ValueError(
            "PyNTT packed QKV projection capacities must be N-vector aligned."
        )
    projection_starts: dict[str, int] = {}
    total_n = 0
    for prefix in ("Q", "K", "V"):
        projection_starts[prefix] = total_n
        total_n += projection_ns[prefix]

    if total_n % block_n != 0:
        raise ValueError(
            "PyNTT fused packed QKV capacity must be divisible by block_n; "
            f"capacity={total_n}, block_n={block_n}."
        )
    packed_k_outer = block_k // k_atom
    packed_n_outer = block_n // n_lane
    expected_weight_shape = (k // k_atom, total_n // n_lane)
    if tuple(_fixed(value) for value in model["WeightShape"]) != expected_weight_shape:
        raise ValueError(
            "PyNTT fused packed QKV weight shape does not match its capacities: "
            f"expected={expected_weight_shape}, actual={model['WeightShape']}."
        )
    if (
        _fixed(model["WeightStrides"][-1]) != 1
        or (_min_value(model["WeightStrides"][-2]) or 0) <= 0
    ):
        raise ValueError(
            "PyNTT fused packed QKV TMA requires positive-stride "
            f"[K/{k_atom},N/{n_lane}]<{n_lane},{k_pack},{k_lane}> storage."
        )

    descriptor_name = model.get("WeightDescriptorName")
    descriptor_origin_elements = model.get("WeightDescriptorOriginElements")
    global_offsets = model.get("WeightGlobalOffsets")
    if not isinstance(descriptor_name, str) or not descriptor_name:
        raise ValueError(
            "PyNTT fused packed QKV GEMV requires one host weight descriptor."
        )
    if not isinstance(descriptor_origin_elements, str) or not descriptor_origin_elements:
        raise ValueError(
            "PyNTT fused packed QKV GEMV requires a weight descriptor origin."
        )
    if not isinstance(global_offsets, list) or len(global_offsets) != 2:
        raise ValueError(
            "PyNTT fused packed QKV GEMV requires two weight global offsets."
        )

    weight_pointer = model["Weight"]
    k_plan = _tma_canonical_axis_plan(
        weight_pointer,
        -2,
        tile_extent=packed_k_outer,
        context="fused packed QKV shared K",
    )
    n_plan = _tma_canonical_axis_plan(
        weight_pointer,
        -1,
        tile_extent=packed_n_outer,
        context="fused packed QKV shared N",
    )
    k_transfer = _tma_local_axis_transfer(
        weight_pointer,
        -2,
        global_offsets[-2],
        local_offset=0,
        tile_index="k_tile",
        tile_stride=packed_k_outer,
        tile_extent=packed_k_outer,
        context="fused packed QKV weight K",
    )
    n_transfer = _tma_local_axis_transfer(
        weight_pointer,
        -1,
        global_offsets[-1],
        local_offset=0,
        tile_index="n_tile",
        tile_stride=packed_n_outer,
        tile_extent=packed_n_outer,
        context="fused packed QKV weight N",
    )
    shared_stage_shape = (
        tuple(n_plan["block_shape"])
        + tuple(k_plan["block_shape"])
        + (k_pack, n_lane * k_lane)
    )
    if len(shared_stage_shape) > 5:
        raise ValueError(
            "PyNTT fused packed QKV shared stage exceeds the TMA rank-5 limit: "
            f"shape={shared_stage_shape}."
        )
    descriptor_offsets = (
        tuple(n_transfer["coordinates"])
        + tuple(k_transfer["coordinates"])
        + ("0", descriptor_origin_elements)
    )

    projections: list[dict[str, Any]] = []
    for prefix in ("Q", "K", "V"):
        lower = prefix.lower()
        projection_n_expr = f"{lower}_projection_n"
        active_n = logical_output_shapes[prefix][-1]
        projection_mask = (
            f"({projection_n_expr} >= 0) & "
            f"({projection_n_expr} < {_dim(active_n)})"
        )
        output_access = _contiguous_vector_axis_access(
            ("0", "0"),
            model[f"{prefix}OutputStrides"],
            tensor_shape=model[f"{prefix}OutputShape"],
            packed_axis=1,
            logical_index=projection_n_expr,
            lane_count=n_lane,
            coordinate_shape=_coordinate_shape((block_n,)),
        )
        has_bias = bool(model[f"Has{prefix}Bias"])
        bias_access = (
            _contiguous_vector_axis_access(
                ("0",),
                model[f"{prefix}BiasStrides"],
                tensor_shape=model.get(f"{prefix}BiasShape"),
                packed_axis=0,
                logical_index=projection_n_expr,
                lane_count=n_lane,
                coordinate_shape=_coordinate_shape((block_n,)),
            )
            if has_bias
            else None
        )
        projections.append(
            {
                "prefix": prefix,
                "lower": lower,
                "projection_start": projection_starts[prefix],
                "projection_n_expr": projection_n_expr,
                "projection_mask": projection_mask,
                "has_bias": has_bias,
                "bias_access": bias_access,
                "output_access": output_access,
            }
        )

    num_n_tiles = total_n // block_n
    num_k_tiles = k // block_k
    sequence_count = num_n_tiles * num_k_tiles
    if sequence_count > 2**31 - 1:
        raise ValueError(
            "PyNTT fused packed QKV GEMV sequence exceeds signed int32."
        )

    reduction_group = 32
    consumer_layout = _packed_gemv_consumer_layout(
        block_n=block_n,
        reduction_group=reduction_group,
        consumer_warps=int(model["NumWarps"]),
        target_worker_width=int(model["TargetWorkerWidth"]),
    )
    input_copy_threads = int(model["NumWarps"]) * int(model["TargetWorkerWidth"])
    input_copy_size_per_thread = max(
        1,
        (block_k + input_copy_threads - 1) // input_copy_threads,
    )
    return {
        "microkernel": microkernel,
        "block_n": block_n,
        "block_k": block_k,
        "num_stages": num_stages,
        "num_k_tiles": num_k_tiles,
        "num_n_tiles": num_n_tiles,
        "projections": tuple(projections),
        "k": model["InputShape"][-1],
        "k_atom": k_atom,
        "packed_k_outer": packed_k_outer,
        "reduction_group": reduction_group,
        "reduction_groups_per_stage": block_k // reduction_group,
        "shared_stage_shape": shared_stage_shape,
        "descriptor_name": descriptor_name,
        "descriptor_offsets": descriptor_offsets,
        "shared_weight_indices": (
            _tma_shared_axis_coordinates(f"local_n // {n_lane}", n_plan)
            + _tma_shared_axis_coordinates(f"shared_k // {k_atom}", k_plan)
            + (
                f"shared_payload // {n_lane * k_lane}",
                f"shared_payload % {n_lane * k_lane}",
            )
        ),
        "consumer_size_per_thread": consumer_layout["size_per_thread"],
        "consumer_threads_per_warp": consumer_layout["threads_per_warp"],
        "consumer_warps_per_cta": consumer_layout["warps_per_cta"],
        "consumer_weight_layout_name": f"{model['FunctionName']}__weight_layout",
        "consumer_lhs_layout_name": f"{model['FunctionName']}__lhs_layout",
        "consumer_output_layout_name": f"{model['FunctionName']}__output_layout",
        "consumer_input_copy_layout_name": (
            f"{model['FunctionName']}__input_copy_layout"
        ),
        "input_copy_size_per_thread": input_copy_size_per_thread,
        "pipeline_input_copy_access": _qkv_input_access(
            model,
            output_batch_rank=0,
            m_expr="0",
            k_expr="pipeline_input_copy_k",
            coordinate_shape=_coordinate_shape((1, block_k)),
        ),
    }


def _packed_qkv_mma_pipeline_template_context(
    model: dict[str, Any],
) -> dict[str, Any]:
    """Prepare a selected BF16 QKV tensor-core pipeline."""

    context = _packed_qkv_gemv_pipeline_template_context(
        model,
        expected_variant="mma_smem_pipeline",
    )
    if any(bool(model[f"Has{prefix}Bias"]) for prefix in ("Q", "K", "V")):
        raise ValueError(
            "PyNTT packed QKV MMA pipeline does not support bias."
        )
    profile = (
        context["block_n"],
        context["block_k"],
        context["num_stages"],
        _fixed(context["k"]),
        context["num_n_tiles"],
        context["num_k_tiles"],
    )
    supported_profiles = {
        (256, 64, 2, 256, 1, 4),
        (32, 1024, 2, 2048, 1, 2),
    }
    if (
        profile not in supported_profiles
        or int(model["NumWarps"]) != 8
        or int(model["TargetWorkerWidth"]) != 32
    ):
        raise ValueError(
            "PyNTT packed QKV MMA pipeline requires a selected local profile "
            "of (N=256, K=256, block_k=64, stages=2) or "
            "(N=32, K=2048, block_k=1024, stages=2), with eight consumer "
            "warps and 32 threads per warp."
        )
    k = _fixed(context["k"])
    assert k is not None
    expected_workspaces = {
        "rhs_stage": (
            context["num_stages"],
            (context["block_k"] // 16) * (context["block_n"] // 8),
            128,
        ),
        "lhs_stage": (1, k),
    }
    if context["microkernel"]["shared_workspace_shapes"] != expected_workspaces:
        raise ValueError(
            "PyNTT packed QKV MMA workspace contract mismatch: "
            f"expected={expected_workspaces}, "
            f"actual={context['microkernel']['shared_workspace_shapes']}."
        )

    function_name = model["FunctionName"]
    context.update(
        {
            "mma_layout_name": f"{function_name}__mma_layout",
            "mma_a_layout_name": f"{function_name}__mma_a_layout",
            "mma_b_layout_name": f"{function_name}__mma_b_layout",
            "mma_b_k_layout_name": f"{function_name}__mma_b_k_layout",
            "mma_c_m_layout_name": f"{function_name}__mma_c_m_layout",
            "mma_c_n_layout_name": f"{function_name}__mma_c_n_layout",
            "mma_input_copy_layout_name": (
                f"{function_name}__mma_input_copy_layout"
            ),
            "complete_input_copy_access": _qkv_input_access(
                model,
                output_batch_rank=0,
                m_expr="0",
                k_expr="pipeline_input_copy_k",
                coordinate_shape=_coordinate_shape((1, k)),
            ),
        }
    )
    return context


def _packed_fp8_qkv_gemv_pipeline_template_context(
    model: dict[str, Any],
) -> dict[str, Any]:
    """Prepare the E4M3-weight, statically scaled fused QKV algorithm."""

    return _packed_qkv_gemv_pipeline_template_context(
        model,
        expected_variant="simt_fp8_fma_smem_pipeline",
        expected_weight_dtype="float8e4m3fn",
        expected_k_vector_lanes=16,
        require_operand_scales=True,
    )


def _packed_matmul_glu_gemv_pipeline_template_context(
    model: dict[str, Any],
    *,
    expected_variant: str = "simt_fma_smem_pipeline",
    expected_weight_dtype: str = "bfloat16",
    expected_k_vector_lanes: int = 8,
    require_operand_scales: bool = False,
    reduction_group: int = 32,
) -> dict[str, Any]:
    """Prepare one selected K-major pipeline for paired gate/up projections."""

    context = _matmul_glu_template_context(
        model,
        packed=True,
        variant=expected_variant,
    )
    if (
        model["InputDType"] != "bfloat16"
        or model["WeightDType"] != expected_weight_dtype
        or model["OutputDType"] != "bfloat16"
    ):
        raise ValueError(
            "PyNTT packed MatMulGlu GEMV pipeline has an incompatible dtype contract: "
            f"expected=bfloat16/{expected_weight_dtype}/bfloat16, "
            f"actual={model['InputDType']}/{model['WeightDType']}/{model['OutputDType']}."
        )
    if bool(model.get("HasOperandScales")) != require_operand_scales:
        raise ValueError(
            "PyNTT packed MatMulGlu operand-scale contract does not match its "
            f"selected variant {expected_variant!r}."
        )
    if model.get("RhsLayout") != "k_major":
        raise ValueError(
            "PyNTT packed MatMulGlu GEMV pipeline requires K-major packed weights."
        )
    if (
        int(model["NPackedLaneCount"]) != 1
        or int(model["NVectorLaneCount"]) != 8
        or int(model["KPackLaneCount"]) != 2
        or int(model["KVectorLaneCount"]) != expected_k_vector_lanes
    ):
        raise ValueError(
            "PyNTT packed MatMulGlu GEMV pipeline requires "
            f"weight=[K/{2 * expected_k_vector_lanes},N/8]"
            f"<8,2,{expected_k_vector_lanes}> and output=[M,N/8]<8>."
        )
    if (
        len(model["InputShape"]) != 2
        or len(model["GateWeightShape"]) != 2
        or len(model["UpWeightShape"]) != 2
        or len(model["OutputShape"]) != 2
    ):
        raise ValueError(
            "PyNTT packed MatMulGlu GEMV pipeline currently requires rank-2 operands."
        )

    microkernel = context["microkernel"]
    block_n = microkernel["parameters"]["block_n"]
    block_k = microkernel["parameters"]["block_k"]
    num_stages = microkernel["parameters"]["num_stages"]
    projection_count = 2
    _validate_packed_gemv_pipeline_resource_contract(
        algorithm="packed MatMulGlu GEMV pipeline",
        block_n=block_n,
        block_k=block_k,
        num_stages=num_stages,
        rhs_tiles_per_group=projection_count,
    )

    n_lane = int(model["NVectorLaneCount"])
    k_pack = int(model["KPackLaneCount"])
    k_lane = int(model["KVectorLaneCount"])
    k_atom = k_pack * k_lane
    if block_n % n_lane != 0 or block_k % k_atom != 0:
        raise ValueError(
            "PyNTT packed MatMulGlu GEMV pipeline tile is incompatible with "
            f"its packed lanes: block_n={block_n}, block_k={block_k}, "
            f"n_lane={n_lane}, k_atom={k_atom}."
        )
    k = _fixed(context["k"])
    if k is None or k <= 0 or k % block_k != 0:
        raise ValueError(
            "PyNTT packed MatMulGlu GEMV pipeline requires a fixed positive K "
            f"divisible by block_k={block_k}, got K={k}."
        )
    lhs_stage_extent = microkernel["parameters"].get("lhs_stage_extent")
    use_complete_consumer_lhs_stage = lhs_stage_extent is not None
    lhs_copy_segments: tuple[dict[str, Any], ...] = ()
    if use_complete_consumer_lhs_stage:
        if (
            _fixed(model["InputStrides"][-1]) != 1
            or not _is_positive_power_of_two(lhs_stage_extent)
            or lhs_stage_extent < k
            or k % 1024 != 0
        ):
            raise ValueError(
                "PyNTT packed MatMulGlu complete consumer LHS staging requires "
                "a contiguous fixed K divisible by 1024 and a power-of-two "
                f"Shared extent covering K; got K={k}, lhs_stage_extent="
                f"{lhs_stage_extent}, input_stride="
                f"{_fixed(model['InputStrides'][-1])}."
            )
        raw_segments = _power_of_two_segments(k)
        if any(extent < 1024 for _, extent in raw_segments):
            raise ValueError(
                "PyNTT packed MatMulGlu complete consumer LHS staging requires "
                "every copy segment to cover at least 1024 elements, got "
                f"{raw_segments}."
            )
        lhs_copy_segments = tuple(
            {
                "offset": offset,
                "extent": extent,
                "access": _matmul_glu_input_access(
                    model,
                    output_batch_rank=0,
                    m_expr="0",
                    k_expr="pipeline_input_copy_k",
                    coordinate_shape=_coordinate_shape((1, extent)),
                ),
            }
            for offset, extent in raw_segments
        )
    packed_k_outer = block_k // k_atom
    packed_n_outer = block_n // n_lane
    for prefix in ("Gate", "Up"):
        descriptor_name = model.get(f"{prefix}WeightDescriptorName")
        descriptor_origin_elements = model.get(
            f"{prefix}WeightDescriptorOriginElements"
        )
        global_offsets = model.get(f"{prefix}WeightGlobalOffsets")
        weight_shape = model[f"{prefix}WeightShape"]
        if not isinstance(descriptor_name, str) or not descriptor_name:
            raise ValueError(
                "PyNTT packed MatMulGlu GEMV pipeline requires a host "
                f"{prefix.lower()} weight descriptor."
            )
        if not isinstance(descriptor_origin_elements, str) or not descriptor_origin_elements:
            raise ValueError(
                "PyNTT packed MatMulGlu GEMV pipeline requires a host "
                f"{prefix.lower()} weight descriptor origin."
            )
        if not isinstance(global_offsets, list) or len(global_offsets) != 2:
            raise ValueError(
                "PyNTT packed MatMulGlu GEMV pipeline requires two "
                f"{prefix.lower()} weight global offsets."
            )
        if (
            _fixed(weight_shape[-2]) != k // k_atom
            or _fixed(model[f"{prefix}WeightStrides"][-1]) != 1
            or (_min_value(model[f"{prefix}WeightStrides"][-2]) or 0) <= 0
        ):
            raise ValueError(
                "PyNTT packed MatMulGlu GEMV TMA requires a positive-stride "
                f"{prefix.lower()} weight [K/{k_atom},N/{n_lane}]"
                f"<{n_lane},{k_pack},{k_lane}> view."
            )

    max_n = _max_value(context["n"])
    if max_n is None or max_n <= 0:
        raise ValueError(
            "PyNTT packed MatMulGlu GEMV pipeline requires a bounded positive N."
        )
    num_k_tiles = k // block_k
    num_n_tiles = (max_n + block_n - 1) // block_n
    max_sequence_count = num_n_tiles * num_k_tiles * projection_count
    if max_sequence_count > (2**31 - 1):
        raise ValueError(
            "PyNTT packed MatMulGlu GEMV sequence exceeds the signed int32 "
            f"pipe ABI: {max_sequence_count}."
        )

    target_worker_width = int(model["TargetWorkerWidth"])
    consumer_warps = int(model["NumWarps"])
    consumer_layout = _packed_gemv_consumer_layout(
        block_n=block_n,
        reduction_group=reduction_group,
        consumer_warps=consumer_warps,
        target_worker_width=target_worker_width,
    )
    tma_contiguous_extent = n_lane * k_lane
    projections = []
    tma_block_shape: tuple[int, ...] | None = None
    shared_k_plan: dict[str, Any] | None = None
    shared_n_plan: dict[str, Any] | None = None
    for projection_index, prefix in enumerate(("Gate", "Up")):
        lower = prefix.lower()
        output_n = "pipeline_output_n"
        output_mask = f"{output_n} < active_n"
        weight_pointer = model[f"{prefix}Weight"]
        weight_global_offsets = model[f"{prefix}WeightGlobalOffsets"]
        k_transfer = _tma_local_axis_transfer(
            weight_pointer,
            -2,
            weight_global_offsets[-2],
            local_offset=0,
            tile_index="k_tile",
            tile_stride=packed_k_outer,
            tile_extent=packed_k_outer,
            context=f"packed MatMulGlu {prefix} weight K",
        )
        n_transfer = _tma_local_axis_transfer(
            weight_pointer,
            -1,
            weight_global_offsets[-1],
            local_offset=0,
            tile_index="n_tile",
            tile_stride=packed_n_outer,
            tile_extent=packed_n_outer,
            context=f"packed MatMulGlu {prefix} weight N",
        )
        projection_block_shape = (
            tuple(k_transfer["block_shape"])
            + tuple(n_transfer["block_shape"])
            + (k_pack, tma_contiguous_extent)
        )
        if len(projection_block_shape) > 5:
            raise ValueError(
                "PyNTT packed MatMulGlu GEMV TMA transfer exceeds the "
                f"hardware rank-5 limit: block_shape={projection_block_shape}."
            )
        if tma_block_shape is None:
            tma_block_shape = projection_block_shape
            shared_k_plan = k_transfer
            shared_n_plan = n_transfer
        elif projection_block_shape != tma_block_shape:
            raise ValueError(
                "PyNTT packed MatMulGlu GEMV requires gate/up weights to use "
                "one common staged split layout."
            )
        projections.append(
            {
                "prefix": prefix,
                "lower": lower,
                "descriptor_name": model[f"{prefix}WeightDescriptorName"],
                "descriptor_origin_elements": model[
                    f"{prefix}WeightDescriptorOriginElements"
                ],
                "descriptor_offsets": (
                    tuple(k_transfer["coordinates"])
                    + tuple(n_transfer["coordinates"])
                    + ("0", model[f"{prefix}WeightDescriptorOriginElements"])
                ),
                "sequence_offset": projection_index,
                "has_bias": bool(model[f"Has{prefix}Bias"]),
                "bias_access": (
                    _contiguous_vector_axis_access(
                    ("0",),
                    model[f"{prefix}BiasStrides"],
                    tensor_shape=model.get(f"{prefix}BiasShape"),
                        packed_axis=0,
                        logical_index=output_n,
                        lane_count=n_lane,
                        coordinate_shape=_coordinate_shape((block_n,)),
                    )
                    if model[f"Has{prefix}Bias"]
                    else None
                ),
                "output_mask": output_mask,
            }
        )

    if tma_block_shape is None or shared_k_plan is None or shared_n_plan is None:
        raise ValueError("PyNTT packed MatMulGlu GEMV requires weight projections.")

    logical_num_stages = num_stages // projection_count
    if _product_int(list(tma_block_shape)) != block_n * block_k:
        raise ValueError(
            "PyNTT packed MatMulGlu GEMV staged RHS shape does not match its "
            f"logical tile: shape={tma_block_shape}, block_n={block_n}, "
            f"block_k={block_k}."
        )
    actual_rhs_workspace_shape = microkernel["shared_workspace_shapes"].get(
        "rhs_stage"
    )
    expected_rhs_workspace_elements = num_stages * block_n * block_k
    if (
        actual_rhs_workspace_shape is None
        or _product_int(list(actual_rhs_workspace_shape))
        != expected_rhs_workspace_elements
    ):
        raise ValueError(
            "PyNTT packed MatMulGlu GEMV RHS workspace capacity mismatch: "
            f"expected_elements={expected_rhs_workspace_elements}, "
            f"actual_shape={actual_rhs_workspace_shape}."
        )
    projection_shared_bytes = logical_num_stages * block_n * block_k * 2

    context.update(
        block_n=block_n,
        block_k=block_k,
        num_stages=num_stages,
        logical_num_stages=logical_num_stages,
        projection_shared_bytes=projection_shared_bytes,
        num_k_tiles=num_k_tiles,
        num_n_tiles=num_n_tiles,
        runtime_num_n_tiles=f"tl.cdiv(active_n, {block_n})",
        projections=tuple(projections),
        projection_count=projection_count,
        k_atom=k_atom,
        packed_k_outer=packed_k_outer,
        packed_n_outer=packed_n_outer,
        reduction_group=reduction_group,
        reduction_groups_per_stage=block_k // reduction_group,
        outline_consumer_stage=_should_outline_packed_gemv_consumer_stage(
            block_k=block_k,
            reduction_group=reduction_group,
        ),
        shared_stage_shape=tma_block_shape,
        tma_block_shape=tma_block_shape,
        tma_contiguous_extent=tma_contiguous_extent,
        shared_weight_indices=(
            _tma_shared_axis_coordinates(
                f"shared_k // {k_atom}", shared_k_plan
            )
            + _tma_shared_axis_coordinates(
                f"local_n // {n_lane}", shared_n_plan
            )
            + (
                f"shared_payload // {tma_contiguous_extent}",
                f"shared_payload % {tma_contiguous_extent}",
            )
        ),
        consumer_size_per_thread=consumer_layout["size_per_thread"],
        consumer_threads_per_warp=consumer_layout["threads_per_warp"],
        consumer_warps_per_cta=consumer_layout["warps_per_cta"],
        consumer_weight_layout_name=f"{model['FunctionName']}__weight_layout",
        consumer_lhs_layout_name=f"{model['FunctionName']}__lhs_layout",
        consumer_output_layout_name=f"{model['FunctionName']}__output_layout",
        use_complete_consumer_lhs_stage=use_complete_consumer_lhs_stage,
        lhs_stage_extent=lhs_stage_extent,
        lhs_copy_segments=lhs_copy_segments,
        consumer_input_copy_layout_name=(
            f"{model['FunctionName']}__input_copy_layout"
        ),
        input_copy_size_per_thread=4,
        pipeline_input_access=_matmul_glu_input_access(
            model,
            output_batch_rank=0,
            m_expr="0",
            k_expr="pipeline_offs_k",
            coordinate_shape=_coordinate_shape((reduction_group,)),
        ),
        pipeline_output_access=_contiguous_vector_axis_access(
            ("0", "0"),
            model["OutputStrides"],
            tensor_shape=model["OutputShape"],
            packed_axis=1,
            logical_index="pipeline_output_n",
            lane_count=n_lane,
            coordinate_shape=_coordinate_shape((block_n,)),
        ),
        pipeline_output_mask="pipeline_output_n < active_n",
    )
    return context


def _packed_fp8_matmul_glu_gemv_pipeline_template_context(
    model: dict[str, Any],
) -> dict[str, Any]:
    """Prepare the E4M3-weight, statically scaled paired GLU algorithm."""

    return _packed_matmul_glu_gemv_pipeline_template_context(
        model,
        expected_variant="simt_fp8_fma_smem_pipeline",
        expected_weight_dtype="float8e4m3fn",
        expected_k_vector_lanes=16,
        require_operand_scales=True,
    )


def _packed_block_fp8_matmul_glu_gemv_pipeline_template_context(
    model: dict[str, Any],
) -> dict[str, Any]:
    """Prepare dynamic K-group activation and block-scaled paired GLU."""

    block_n = _require_int(model.get("WeightBlockN"), "WeightBlockN", minimum=1)
    block_k = _require_int(model.get("WeightBlockK"), "WeightBlockK", minimum=1)
    if block_n != 128 or block_k != 128:
        raise ValueError(
            "PyNTT block-scaled FP8 MatMulGlu currently implements the official "
            f"128x128 scale ABI, got {block_n}x{block_k}."
        )
    if model.get("QuantizationMode") != "dynamic_block":
        raise ValueError(
            "PyNTT block-scaled FP8 MatMulGlu requires dynamic_block quantization."
        )
    if bool(model.get("HasOperandScales")) or not bool(model.get("HasWeightScales")):
        raise ValueError(
            "PyNTT block-scaled FP8 MatMulGlu requires two weight block scales "
            "and no precomputed input scales."
        )

    context = _packed_matmul_glu_gemv_pipeline_template_context(
        model,
        expected_variant="simt_block_fp8_fma_smem_pipeline",
        expected_weight_dtype="float8e4m3fn",
        expected_k_vector_lanes=16,
        require_operand_scales=False,
        reduction_group=block_k,
    )
    n_lane = int(model["NVectorLaneCount"])
    global_output_n = _pointer_local_vector_to_global_scalar_coordinate(
        model["Output"], 1, "pipeline_output_n", n_lane
    )
    global_k_start = _pointer_local_to_global_coordinate(
        model["Input"],
        1,
        f"k_tile * {context['block_k']} + "
        f"k_group * {context['reduction_group']}",
    )
    projections = []
    for projection in context["projections"]:
        prefix = projection["prefix"]
        scale_shape = _require_list(
            model.get(f"{prefix}WeightScaleShape"),
            f"{prefix}WeightScaleShape",
        )
        scale_strides = _require_list(
            model.get(f"{prefix}WeightScaleStrides"),
            f"{prefix}WeightScaleStrides",
        )
        if len(scale_shape) != 2 or len(scale_strides) != 2:
            raise ValueError(
                f"PyNTT block-scaled FP8 MatMulGlu {prefix.lower()} scale must be rank 2."
            )
        projections.append(
            {
                **projection,
                "weight_scale_access": _tensor_access(
                    ("weight_scale_n", "weight_scale_k"),
                    scale_strides,
                    coordinate_shape=_coordinate_shape((context["block_n"],)),
                    global_coordinate_axes=(0, 1),
                ),
            }
        )

    context.update(
        weight_block_n=block_n,
        weight_block_k=block_k,
        global_output_n=global_output_n,
        global_k_start=global_k_start,
        projections=tuple(projections),
    )
    return context


def _reduce_template_context(model: dict[str, Any]) -> dict[str, Any]:
    """Prepare a complete reduction over global-memory buffers."""

    output_shape = model["OutputShape"]
    rank = len(output_shape)
    block_axis = _select_block_axis(output_shape, model["OutputStrides"])
    block_extent = _one() if rank == 0 else output_shape[block_axis]
    axis_set = set(model["Axes"])
    output_index = 0
    input_coordinates: list[str] = []
    for input_index in range(len(model["InputShape"])):
        if input_index in axis_set:
            input_coordinates.append(f"reduce_idx{input_index}")
            if model["KeepDims"]:
                output_index += 1
            continue
        input_coordinates.append(
            "lane" if output_index == block_axis else f"out_idx{output_index}"
        )
        output_index += 1

    def axis_index(axis: int) -> str:
        return "lane" if axis == block_axis else f"out_idx{axis}"

    return {
        "block_axis": block_axis,
        "block_extent": block_extent,
        "input_access": _tensor_access(
            input_coordinates,
            model["InputStrides"],
            coordinate_shape="(block_size,)",
        ),
        "loop_axes": tuple(axis for axis in range(rank) if axis != block_axis),
        "output_access": _tensor_access(
            tuple(axis_index(axis) for axis in range(rank)),
            model["OutputStrides"],
            coordinate_shape="(block_size,)",
        ),
    }


def _sampling_template_context(model: dict[str, Any]) -> dict[str, Any]:
    """Prepare exact local-to-global vocabulary coordinates for sampling."""
    local_shape = model.get("LogitsShape")
    global_shape = model.get("LogitsGlobalShape")
    strides = model.get("LogitsStrides")
    shard_axes = model.get("LogitsShardAxes")
    hierarchy = model.get("Hierarchy")
    if not all(
        isinstance(value, list)
        for value in (local_shape, global_shape, strides, shard_axes, hierarchy)
    ):
        raise ValueError("PyNTT sampling has incomplete logits layout metadata.")
    if not (
        len(local_shape)
        == len(global_shape)
        == len(strides)
        == len(shard_axes)
        == 2
    ):
        raise ValueError("PyNTT sampling requires rank-2 logits layout metadata.")
    if _shard_axis_stages(shard_axes[0]):
        raise ValueError("PyNTT sampling does not support a sharded batch axis.")

    local_vocab_extent = _fixed(local_shape[1])
    global_token = _local_to_global_coordinate(
        "local_token",
        global_shape[1],
        shard_axes[1],
        hierarchy,
        local_extent=local_vocab_extent,
    )
    context = {
        "local_batch": _dim(local_shape[0]),
        "local_vocab": _dim(local_shape[1]),
        "global_batch": _dim(global_shape[0]),
        "global_vocab": _dim(global_shape[1]),
        "global_token": global_token,
        "logits_offset": (
            f"batch * {_dim(strides[0])} + local_token * {_dim(strides[1])}"
        ),
        "processed_offset": (
            f"batch * {_dim(model['ProcessedLogitsStrides'][0])} + "
            f"local_token * {_dim(model['ProcessedLogitsStrides'][1])}"
        ),
    }
    if "RadixBits" in model:
        block_count = int(model["BlockCount"])
        if block_count <= 0:
            raise ValueError("PyNTT sampling block count must be positive.")
        address = model.get("ArgMaxStateAddress")
        argmax_state = model.get("ArgMaxState")
        if not isinstance(address, dict) or not isinstance(argmax_state, dict):
            raise ValueError(
                "PyNTT SamplingCombine requires pooled argmax owner metadata."
            )
        if argmax_state.get("DistributedStorageKind") != "CompactPerOwner":
            raise ValueError(
                "PyNTT SamplingCombine argmax state must use CompactPerOwner storage."
            )
        owner_pool_index = _pool_index_expression(
            "block_offsets", address["PoolScopeSize"]
        )
        owner_byte_offset = (
            f"({owner_pool_index}) * ({address['PoolStrideBytes']})"
            f" + ({address['OffsetBytes']})"
        )
        context.update(
            argmax_owner_pointer=(
                f"({address['BaseName']} + ({owner_byte_offset})).to("
                f"{_pointer_type('tl.uint64', address['AddressSpace'])})"
            ),
            summary_width=1 << (block_count - 1).bit_length(),
            summary_slots=(1 << int(model["RadixBits"])) + 16,
        )
    return context


def _packed_matmul_sampling_partial_workspace_names() -> tuple[str, ...]:
    return ("rhs_stage",)


def _packed_matmul_sampling_partial_template_context(
    model: dict[str, Any],
) -> dict[str, Any]:
    """Prepare one shared-staged LM-head with token-local sampling."""

    matmul = model.get("Matmul")
    sampling = model.get("Sampling")
    if not isinstance(matmul, dict) or not isinstance(sampling, dict):
        raise ValueError(
            "PyNTT packed matmul sampling partial requires Matmul and Sampling metadata."
        )
    prepared_matmul = {
        **matmul,
        **{
            name: model[name]
            for name in (
                "NoInline",
                "MeshAxes",
                "NumWarps",
                "TargetWorkerWidth",
            )
            if name in model
        },
    }
    context = _packed_gemv_pipeline_template_context(
        prepared_matmul,
        expected_family="triton.matmul_sampling_partial",
        required_workspace_names=_packed_matmul_sampling_partial_workspace_names(),
    )
    sampling_context = _sampling_template_context(sampling)
    context.update(
        matmul=prepared_matmul,
        sampling=sampling_context,
    )
    return context


def _softmax_template_context(model: dict[str, Any]) -> dict[str, Any]:
    """Prepare one vector reduction over each independent Softmax slice."""

    shape = model["Shape"]
    rank = len(shape)
    axis = _require_int(model["Axis"], "PyNTT Softmax axis", minimum=0)
    if axis >= rank:
        raise ValueError(f"PyNTT Softmax axis {axis} is outside rank {rank}.")

    axis_max_extent = _require_int(
        shape[axis].get("MaxValue"),
        "PyNTT Softmax reduction-axis maximum extent",
        minimum=1,
    )
    axis_block_size = 1 << (axis_max_extent - 1).bit_length()
    indices = tuple("lane" if index == axis else f"idx{index}" for index in range(rank))
    coordinate_shape = f"({axis_block_size},)"
    return {
        "axis_block_size": axis_block_size,
        "axis_extent": shape[axis],
        "input_access": _tensor_access(
            indices,
            model["InputStrides"],
            coordinate_shape=coordinate_shape,
        ),
        "loop_axes": tuple(index for index in range(rank) if index != axis),
        "output_access": _tensor_access(
            indices,
            model["OutputStrides"],
            coordinate_shape=coordinate_shape,
        ),
    }


def _layer_norm_template_context(model: dict[str, Any]) -> dict[str, Any]:
    """Prepare direct physical/lane coordinates for TIR LayerNorm."""
    logical_output_shape = _logical_shape(
        model["OutputShape"], model["OutputVectorLaneCount"]
    )
    rank = len(logical_output_shape)
    axis = int(model["Axis"])
    if axis < 0 or axis >= rank:
        raise ValueError(f"PyNTT LayerNorm axis {axis} is outside rank {rank}.")

    lane_shapes: dict[str, tuple[int, ...]] = {}
    for prefix in ("Input", "Scale", "Bias", "Output"):
        lanes = _validate_coordinate_lane_shape(
            model[f"{prefix}VectorLaneShape"], f"PyNTT LayerNorm {prefix}"
        )
        lane_count = _product_int(list(lanes)) if lanes else 1
        if lane_count != int(model[f"{prefix}VectorLaneCount"]):
            raise ValueError(
                f"PyNTT LayerNorm {prefix} lane shape/count mismatch: "
                f"shape={lanes}, count={model[f'{prefix}VectorLaneCount']}."
            )
        lane_shapes[prefix] = lanes
    vector_lane_shapes = {lanes for lanes in lane_shapes.values() if lanes}
    if len(vector_lane_shapes) > 1:
        raise ValueError(
            "PyNTT LayerNorm vector operands must use one lane shape, got "
            f"{sorted(vector_lane_shapes)}."
        )
    common_lanes = next(iter(vector_lane_shapes), ())
    common_lane_count = _product_int(list(common_lanes)) if common_lanes else 1
    if common_lanes and not (lane_shapes["Input"] or lane_shapes["Output"]):
        raise ValueError(
            "PyNTT LayerNorm requires a vectorized input or output when its "
            "parameters are vectorized."
        )

    if lane_shapes["Input"]:
        physical_domain_shape = model["InputShape"]
    elif lane_shapes["Output"]:
        physical_domain_shape = model["OutputShape"]
    else:
        physical_domain_shape = model["OutputShape"]
    if len(physical_domain_shape) != rank:
        raise ValueError(
            "PyNTT LayerNorm input/output rank does not match its logical rank."
        )

    inner_axis = _structured_axis_tile(
        "norm_inner",
        common_lanes,
        "block_size",
        logical_output_shape[-1],
        physical_base="inner_start",
    )
    inner_coordinate_shape = _coordinate_shape(inner_axis["structured_shape"])

    def operand_access(prefix: str, parameter: bool) -> dict[str, Any]:
        shape = model[f"{prefix}Shape"]
        strides = model[f"{prefix}Strides"]
        lanes = lane_shapes[prefix]
        coordinates: list[str] = []
        for operand_axis in range(len(shape)):
            output_axis = axis + operand_axis if parameter else operand_axis
            if output_axis == rank - 1:
                coordinate = (
                    inner_axis["physical_coordinate"]
                    if lanes
                    else inner_axis["logical_coordinate"]
                )
            elif output_axis < axis:
                coordinate = f"outer_idx{output_axis}"
            else:
                coordinate = f"inner_idx{output_axis}"
            coordinates.append(coordinate)
        return _tensor_access(
            coordinates,
            strides,
            inner_axis["lane_coordinates"] if lanes else (),
            lanes,
            inner_coordinate_shape,
        )

    return {
        "bias_access": operand_access("Bias", True),
        "common_lane_count": common_lane_count,
        "inner_axis": inner_axis,
        "inner_loop_axes": tuple(range(axis, rank - 1)),
        "inner_size": _product(logical_output_shape[axis:]),
        "input_access": operand_access("Input", False),
        "logical_output_shape": logical_output_shape,
        "outer_axes": tuple(range(axis)),
        "physical_domain_shape": physical_domain_shape,
        "output_access": operand_access("Output", False),
        "scale_access": operand_access("Scale", True),
    }


def _reduce_all_axes(expression: str, rank: int) -> str:
    if rank < 1:
        raise ValueError(f"PyNTT all-axis reduction requires positive rank, got {rank}")
    for axis in reversed(range(rank)):
        expression = f"tl.sum({expression}, axis={axis})"
    return expression


def _is_contiguous_reduction_suffix(
    shape: list[Any], strides: list[Any], axis: int
) -> bool:
    expected_stride = 1
    for suffix_axis in range(len(shape) - 1, axis - 1, -1):
        extent = _fixed(shape[suffix_axis])
        stride = _fixed(strides[suffix_axis])
        if extent is None or stride != expected_stride:
            return False
        expected_stride *= extent
    return True


def _has_block_cyclic_reduction_sharding(
    pointer: Any, axis: int
) -> bool:
    if not isinstance(pointer, dict):
        return False
    shard_axes = pointer.get("ShardAxes")
    if not isinstance(shard_axes, list):
        return False
    return any(
        stage.get("Distribution") == "BlockCyclic"
        for axis_mapping in shard_axes[axis:]
        for stage in _shard_axis_stages(axis_mapping)
    )


def _norm_stats_template_context(model: dict[str, Any]) -> dict[str, Any]:
    rank = len(model["InputShape"])
    axis = int(model["Axis"])
    if axis < 0 or axis >= rank:
        raise ValueError(f"PyNTT NormStats axis {axis} is outside rank {rank}")
    outer_axes = tuple(range(model["Axis"]))
    context = _coordinate_iteration_context(
        model["InputShape"][axis:],
        model["InputStrides"][axis:],
        model["InputVectorLaneShape"],
        "PyNTT NormStats",
    )
    if context["lane_count"] != model["InputVectorLaneCount"]:
        raise ValueError(
            "PyNTT NormStats vector lane metadata is inconsistent: "
            f"shape={context['lane_shape']}, count={model['InputVectorLaneCount']}"
        )
    if model["OutputVectorLaneShape"]:
        raise ValueError("PyNTT NormStats output must have a scalar element type")
    tensor_coordinates = tuple(f"outer_idx{index}" for index in outer_axes) + tuple(
        context["tensor_coordinates"]
    )
    context["input_access"] = _tensor_access(
        tensor_coordinates,
        model["InputStrides"],
        context["lane_coordinates"],
        context["lane_shape"],
    )
    flat_reduction = _is_contiguous_reduction_suffix(
        model["InputShape"],
        model["InputStrides"],
        axis,
    ) and not _has_block_cyclic_reduction_sharding(model.get("Input"), axis)
    flat_base_coordinates = tuple(
        f"outer_idx{index}" if index < axis else "0" for index in range(rank)
    )
    context["flat_input_base_access"] = _tensor_access(
        flat_base_coordinates,
        model["InputStrides"],
        tuple("0" for _ in context["lane_shape"]),
        context["lane_shape"],
    )

    def stats_access(component: int) -> dict[str, Any]:
        coordinates = (str(component),) + tuple(
            f"outer_idx{index}" if index < axis else "0" for index in range(rank)
        )
        return _tensor_access(coordinates, model["OutputStrides"])

    reduction_rank = 1 if flat_reduction else 1 + len(context["lane_shape"])
    reduction = _reduce_all_axes("mean_partial", reduction_rank)
    square_reduction = _reduce_all_axes("square_partial", reduction_rank)
    context.update(
        {
            "logical_input_shape": _logical_shape(
                model["InputShape"], model["InputVectorLaneCount"]
            ),
            "flat_reduction": flat_reduction,
            "outer_axes": outer_axes,
            "prefix_depth": len(outer_axes),
            "reduction": reduction,
            "reduction_extent": _multiply_expr(
                _product(model["InputShape"][axis:]),
                model["InputVectorLaneCount"],
            ),
            "square_reduction": square_reduction,
            "stats_accesses": (stats_access(0), stats_access(1)),
            "tile_shape": "(block_size,)" if flat_reduction else context["tile_shape"],
        }
    )
    return context


def _norm_apply_template_context(model: dict[str, Any]) -> dict[str, Any]:
    logical_input_global_shape = _logical_shape(
        model["InputGlobalShape"], model["InputVectorLaneCount"]
    )
    logical_output_shape = _logical_shape(
        model["OutputShape"], model["OutputVectorLaneCount"]
    )
    rank = len(model["OutputShape"])
    axis = int(model["Axis"])
    if axis < 0 or axis >= rank:
        raise ValueError(f"PyNTT NormApply axis {axis} is outside rank {rank}")
    outer_axes = tuple(range(axis))
    context = _coordinate_iteration_context(
        model["OutputShape"][axis:],
        model["OutputStrides"][axis:],
        model["OutputVectorLaneShape"],
        "PyNTT NormApply",
    )
    vector_lanes = {
        tuple(model[f"{name}VectorLaneShape"])
        for name in ("Input", "Scale", "Bias", "Output")
    }
    if (
        len(vector_lanes) != 1
        or context["lane_count"] != model["OutputVectorLaneCount"]
    ):
        raise ValueError(
            "PyNTT NormApply input/scale/bias/output vector lane shapes must match: "
            f"{sorted(vector_lanes)}"
        )
    if model["StatsVectorLaneShape"]:
        raise ValueError("PyNTT NormApply stats must have a scalar element type")

    inner_coordinates = tuple(context["tensor_coordinates"])
    tensor_coordinates = (
        tuple(f"outer_idx{index}" for index in outer_axes) + inner_coordinates
    )
    context["input_access"] = _tensor_access(
        tensor_coordinates,
        model["InputStrides"],
        context["lane_coordinates"],
        context["lane_shape"],
    )
    context["output_access"] = _tensor_access(
        tensor_coordinates,
        model["OutputStrides"],
        context["lane_coordinates"],
        context["lane_shape"],
    )
    context["scale_access"] = _tensor_access(
        inner_coordinates,
        model["ScaleStrides"],
        context["lane_coordinates"],
        context["lane_shape"],
    )
    context["bias_access"] = _tensor_access(
        inner_coordinates,
        model["BiasStrides"],
        context["lane_coordinates"],
        context["lane_shape"],
    )

    def stats_access(component: int) -> dict[str, Any]:
        coordinates = (str(component),) + tuple(
            f"outer_idx{index}" if index < axis else "0" for index in range(rank)
        )
        return _tensor_access(coordinates, model["StatsStrides"])

    context.update(
        {
            "logical_output_shape": logical_output_shape,
            "local_normalization_size": _product(logical_output_shape[axis:]),
            "normalization_size": _product(logical_input_global_shape[axis:]),
            "outer_axes": outer_axes,
            "prefix_depth": len(outer_axes),
            "stats_accesses": (stats_access(0), stats_access(1)),
        }
    )
    return context


def _gather_reduce_norm_apply_template_context(
    model: dict[str, Any],
) -> dict[str, Any]:
    """Prepare a repeated partial reduction followed by local NormApply."""

    norm_apply = model["NormApply"]
    context = _norm_apply_template_context(norm_apply)
    hierarchy = tuple(int(extent) for extent in model["Hierarchy"])
    partial_axes = tuple(sorted(int(axis) for axis in model["PartialAxes"]))
    if not partial_axes or len(set(partial_axes)) != len(partial_axes):
        raise ValueError(
            "PyNTT GatherReduceNormApply requires non-empty unique partial axes"
        )
    if any(axis < 0 or axis >= len(hierarchy) for axis in partial_axes):
        raise ValueError(
            "PyNTT GatherReduceNormApply partial axes are outside the hierarchy"
        )

    dtype = norm_apply["StatsDType"]
    if dtype in ("float16", "bfloat16", "float32"):
        accumulator_dtype, zero = "tl.float32", "0.0"
    elif dtype == "float64":
        accumulator_dtype, zero = "tl.float64", "0.0"
    else:
        raise ValueError(
            f"PyNTT GatherReduceNormApply does not support stats dtype {dtype}"
        )

    reduction_extent = _product_int([hierarchy[axis] for axis in partial_axes])
    reduction_width_cap = 1 << (reduction_extent - 1).bit_length()
    axis_strides: dict[int, int] = {}
    axis_stride = 1
    for axis in reversed(partial_axes):
        axis_strides[axis] = axis_stride
        axis_stride *= hierarchy[axis]

    address = model["PartialStatsAddress"]
    source_shard_index = _split_linear_expression(
        list(range(len(hierarchy))),
        hierarchy,
        "source_shard_coord",
    )
    context["partial"] = {
        "accumulator_dtype": accumulator_dtype,
        "address": address,
        "axes": partial_axes,
        "axis_strides": axis_strides,
        "pointer_type": _pointer_type(
            norm_apply["StatsTritonDType"], address["AddressSpace"]
        ),
        "reduction_extent": reduction_extent,
        "reduction_width_cap": reduction_width_cap,
        "source_pool_index": _pool_index_expression(
            "source_shard_index", address["PoolScopeSize"]
        ),
        "source_shard_index": source_shard_index,
        "zero": zero,
    }
    return context


def _gather_reduce_add_norm_template_context(
    model: dict[str, Any],
) -> dict[str, Any]:
    """Prepare a fused partial materialization, residual add, and normalization."""

    reshard = _reshard_template_context(model["Reshard"])
    stats = _norm_stats_template_context(model["NormStats"])
    if reshard["partial"] is None:
        raise ValueError("PyNTT GatherReduceAddNorm requires a partial input")
    if not stats["flat_reduction"]:
        raise ValueError(
            "PyNTT GatherReduceAddNorm currently requires a contiguous normalization suffix"
        )
    axis = int(model["NormStats"]["Axis"])
    if any(_fixed(dim) != 1 for dim in model["NormStats"]["InputShape"][:axis]):
        raise ValueError(
            "PyNTT GatherReduceAddNorm grid-cooperative statistics require "
            "singleton dimensions before the normalization axis"
        )
    normalization_split_mesh_axes: set[int] = set()
    for tensor_axis, shard_axis in enumerate(model["Reshard"]["InputShardAxes"]):
        mesh_axes = set(_shard_axis_hierarchy_axes(shard_axis))
        if tensor_axis < axis and mesh_axes:
            raise ValueError(
                "PyNTT GatherReduceAddNorm cannot form grid-cooperative statistics "
                "when an outer tensor axis is split"
            )
        if tensor_axis >= axis:
            normalization_split_mesh_axes.update(mesh_axes)
    input_split_mesh_axes = set(reshard["input_split_mesh_axes"])
    if normalization_split_mesh_axes != input_split_mesh_axes:
        raise ValueError(
            "PyNTT GatherReduceAddNorm requires every input split mesh axis to "
            "partition the normalized suffix"
        )

    norm_model = model.get("NormApply")
    norm = None if norm_model is None else _norm_apply_template_context(norm_model)
    return {
        "reshard": reshard,
        "stats": stats,
        "norm": norm,
    }


def _rope_template_context(model: dict[str, Any]) -> dict[str, Any]:
    """Prepare RoPE's physical tensor and vector-lane coordinates."""

    rank = len(model["OutputShape"])
    rotary_axis = model["RotaryAxis"]
    output_lane_shape = _validate_coordinate_lane_shape(
        model["OutputVectorLaneShape"], "PyNTT RoPE output"
    )
    output_lane_count = (
        _product_int(list(output_lane_shape)) if output_lane_shape else 1
    )
    if output_lane_count != int(model["OutputVectorLaneCount"]):
        raise ValueError(
            "PyNTT RoPE output lane shape/count mismatch: "
            f"shape={output_lane_shape}, count={model['OutputVectorLaneCount']}"
        )
    input_lane_shape = _validate_coordinate_lane_shape(
        model["InputVectorLaneShape"], "PyNTT RoPE input"
    )
    if input_lane_shape != output_lane_shape:
        raise ValueError(
            "PyNTT RoPE input/output lane shapes must match: "
            f"input={input_lane_shape}, output={output_lane_shape}."
        )
    sincos_pack_factor = int(model.get("SinCosVectorPackFactor", 1))
    if sincos_pack_factor not in (1, 2):
        raise ValueError(
            "PyNTT RoPE direct coordinate lowering supports aligned sin/cos "
            f"lanes or canonical two-half packing, got {sincos_pack_factor}."
        )
    sincos_lane_shape = (
        output_lane_shape
        if sincos_pack_factor == 1
        else (sincos_pack_factor,) + output_lane_shape
    )
    for name in ("Cos", "Sin"):
        actual_shape = _validate_coordinate_lane_shape(
            model[f"{name}VectorLaneShape"], f"PyNTT RoPE {name.lower()}"
        )
        actual_count = int(model[f"{name}VectorLaneCount"])
        if actual_shape != sincos_lane_shape or actual_count != _product_int(
            list(sincos_lane_shape)
        ):
            raise ValueError(
                f"PyNTT RoPE {name.lower()} lane layout must be "
                f"{sincos_lane_shape}, got shape={actual_shape}, "
                f"count={actual_count}."
            )

    cos_shape = model["CosShape"]
    cos_strides = model["CosStrides"]
    if len(cos_shape) != rank or len(cos_strides) != rank:
        raise ValueError(
            "PyNTT RoPE sin/cos tensors must retain the output rank for "
            f"coordinate-native lowering: cos_rank={len(cos_shape)}, output_rank={rank}."
        )

    def operand_access(
        context: dict[str, Any],
        name: str,
        physical_rotary: str,
        lane_coordinates: tuple[str, ...],
        lane_shape: tuple[int, ...],
    ) -> dict[str, Any]:
        shape = model[f"{name}Shape"]
        strides = model[f"{name}Strides"]
        axis_offset = rank - len(shape)
        tensor_coordinates = []
        for axis, dimension in enumerate(shape):
            output_axis = axis_offset + axis
            if _is_fixed_one(dimension):
                coordinate = "0"
            elif output_axis == rotary_axis:
                coordinate = physical_rotary
            else:
                coordinate = context["tensor_coordinates"][output_axis]
            tensor_coordinates.append(coordinate)
        return _tensor_access(
            tensor_coordinates,
            strides,
            lane_coordinates,
            lane_shape,
        )

    if sincos_pack_factor == 1:
        context = _coordinate_iteration_context(
            model["OutputShape"],
            model["OutputStrides"],
            list(output_lane_shape),
            "PyNTT RoPE",
        )
        output_rotary_extent = _constant_dim_value(model["OutputShape"][rotary_axis])
        if output_rotary_extent is None or output_rotary_extent % 2 != 0:
            raise ValueError(
                "PyNTT RoPE aligned sin/cos lowering requires a static even "
                "physical rotary extent."
            )
        half_physical_extent = output_rotary_extent // 2
        output_physical_rotary = context["tensor_coordinates"][rotary_axis]
        first_half = f"{output_physical_rotary} < {half_physical_extent}"
        paired_physical_rotary = (
            f"tl.where({first_half}, {output_physical_rotary} + "
            f"{half_physical_extent}, {output_physical_rotary} - "
            f"{half_physical_extent})"
        )
        lane_flat = _flatten_coordinates(context["lane_coordinates"], output_lane_shape)
        logical_rotary = (
            f"({output_physical_rotary}) * {output_lane_count} + ({lane_flat})"
            if output_lane_count != 1
            else output_physical_rotary
        )
        context.update(
            cos_access=operand_access(
                context,
                "Cos",
                output_physical_rotary,
                context["lane_coordinates"],
                sincos_lane_shape,
            ),
            first_half=first_half,
            input_access=operand_access(
                context,
                "Input",
                output_physical_rotary,
                context["lane_coordinates"],
                input_lane_shape,
            ),
            lane_flat=lane_flat,
            logical_rotary=logical_rotary,
            output_access=operand_access(
                context,
                "Output",
                output_physical_rotary,
                context["lane_coordinates"],
                output_lane_shape,
            ),
            output_physical_rotary=output_physical_rotary,
            output_rotary_extent=model["OutputShape"][rotary_axis],
            paired_input_access=operand_access(
                context,
                "Input",
                paired_physical_rotary,
                context["lane_coordinates"],
                input_lane_shape,
            ),
            rotary_axis=rotary_axis,
            sin_access=operand_access(
                context,
                "Sin",
                output_physical_rotary,
                context["lane_coordinates"],
                sincos_lane_shape,
            ),
        )
        return context

    domain_shape = [
        dict(value) if isinstance(value, dict) else value
        for value in model["OutputShape"]
    ]
    domain_strides = [
        dict(value) if isinstance(value, dict) else value
        for value in model["OutputStrides"]
    ]
    domain_shape[rotary_axis] = cos_shape[rotary_axis]
    domain_strides[rotary_axis] = cos_strides[rotary_axis]
    context = _coordinate_iteration_context(
        domain_shape,
        domain_strides,
        list(sincos_lane_shape),
        "PyNTT RoPE",
    )
    pack_coordinate = context["lane_coordinates"][0]
    vector_lane_coordinates = context["lane_coordinates"][1:]
    lane_flat = _flatten_coordinates(vector_lane_coordinates, output_lane_shape)
    output_physical_extent = _constant_dim_value(model["OutputShape"][rotary_axis])
    sincos_physical_extent = _constant_dim_value(cos_shape[rotary_axis])
    if (
        output_physical_extent is None
        or output_physical_extent % 2 != 0
        or sincos_physical_extent is None
        or sincos_physical_extent * sincos_pack_factor != output_physical_extent
    ):
        raise ValueError(
            "PyNTT RoPE packed sin/cos lowering requires a static even "
            "output rotary extent and an interleaved physical extent matching "
            "the dtype-width pack factor: "
            f"output={output_physical_extent}, sincos={sincos_physical_extent}, "
            f"pack={sincos_pack_factor}."
        )
    half_output_physical_extent = output_physical_extent // 2
    output_physical_rotary = (
        f"({context['tensor_coordinates'][rotary_axis]}) * "
        f"{sincos_pack_factor} + ({pack_coordinate})"
    )
    paired_physical_rotary = (
        f"tl.where({output_physical_rotary} < {half_output_physical_extent}, "
        f"{output_physical_rotary} + {half_output_physical_extent}, "
        f"{output_physical_rotary} - {half_output_physical_extent})"
    )
    logical_rotary = (
        f"({output_physical_rotary}) * {output_lane_count} + ({lane_flat})"
        if output_lane_count != 1
        else output_physical_rotary
    )

    context.update(
        cos_access=operand_access(
            context,
            "Cos",
            context["tensor_coordinates"][rotary_axis],
            context["lane_coordinates"],
            sincos_lane_shape,
        ),
        first_half=(f"{output_physical_rotary} < {half_output_physical_extent}"),
        input_access=operand_access(
            context,
            "Input",
            output_physical_rotary,
            vector_lane_coordinates,
            input_lane_shape,
        ),
        lane_flat=lane_flat,
        logical_rotary=logical_rotary,
        output_access=operand_access(
            context,
            "Output",
            output_physical_rotary,
            vector_lane_coordinates,
            output_lane_shape,
        ),
        output_physical_rotary=output_physical_rotary,
        output_rotary_extent=model["OutputShape"][rotary_axis],
        paired_input_access=operand_access(
            context,
            "Input",
            paired_physical_rotary,
            vector_lane_coordinates,
            input_lane_shape,
        ),
        rotary_axis=rotary_axis,
        sin_access=operand_access(
            context,
            "Sin",
            context["tensor_coordinates"][rotary_axis],
            context["lane_coordinates"],
            sincos_lane_shape,
        ),
    )
    return context


def _norm_rope_template_context(
    norm: dict[str, Any], rope: dict[str, Any], context_name: str, scope: str
) -> dict[str, Any]:
    """Compose an inline normalization reduction into RoPE's coordinate domain."""

    context = _rope_template_context(rope)
    rank = len(norm["InputShape"])
    axis = int(norm["Axis"])
    if axis < 0 or axis >= rank:
        raise ValueError(f"{context_name} normalization axis {axis} is outside rank {rank}")
    if (
        norm["InputShape"] != rope["InputShape"]
        or norm["InputShape"] != rope["OutputShape"]
        or norm["InputStrides"] != rope["InputStrides"]
        or norm["InputStrides"] != rope["OutputStrides"]
        or norm["InputVectorLaneShape"] != rope["InputVectorLaneShape"]
        or norm["InputVectorLaneShape"] != rope["OutputVectorLaneShape"]
    ):
        raise ValueError(
            f"{context_name} inline norm and RoPE must describe the same input/output layout"
        )
    if context["major_axis"] < axis:
        raise ValueError(
            f"{context_name} RoPE block axis {context['major_axis']} must be inside "
            f"the normalized suffix starting at axis {axis}."
        )

    current = context["input_access"]
    paired = context["paired_input_access"]
    lane_shape = tuple(int(value) for value in norm["InputVectorLaneShape"])
    reduction_context = _coordinate_iteration_context(
        norm["InputShape"][axis:],
        norm["InputStrides"][axis:],
        list(lane_shape),
        context_name,
        variable_prefix=f"{scope}_reduce_",
    )
    reduction_major_extent = _fixed(reduction_context["major_extent"])
    if reduction_major_extent is not None:
        reduction_context["physical_tile_width_cap"] = 1 << max(
            reduction_major_extent - 1, 0
        ).bit_length()
    if reduction_context["lane_count"] != int(norm["InputVectorLaneCount"]):
        raise ValueError(
            f"{context_name} inline reduction lane metadata is inconsistent: "
            f"shape={lane_shape}, count={norm['InputVectorLaneCount']}."
        )
    reduction_context["input_access"] = _tensor_access(
        tuple(f"index{index}" for index in range(axis))
        + tuple(reduction_context["tensor_coordinates"]),
        norm["InputStrides"],
        reduction_context["lane_coordinates"],
        reduction_context["lane_shape"],
    )
    reduction_rank = 1 + len(reduction_context["lane_shape"])
    reduction = _reduce_all_axes("norm_value", reduction_rank)
    square_reduction = _reduce_all_axes(
        "norm_value * norm_value", reduction_rank
    )

    def parameter_access(name: str, source: dict[str, Any]) -> dict[str, Any]:
        coordinates = tuple(source["TensorIndices"])[axis:]
        strides = norm[f"{name}Strides"]
        if len(coordinates) != len(strides):
            raise ValueError(
                f"{context_name} {name.lower()} rank does not match the normalized suffix: "
                f"coordinates={len(coordinates)}, strides={len(strides)}"
            )
        return _tensor_access(
            coordinates,
            strides,
            source["LaneIndices"],
            lane_shape,
        )

    context.update(
        {
            "bias_access": parameter_access("Bias", current),
            "loop_axes": tuple(
                loop_axis
                for loop_axis in context["loop_axes"]
                if loop_axis >= axis
            ),
            "normalization_size": _product(
                _logical_shape(
                    norm["InputGlobalShape"], norm["InputVectorLaneCount"]
                )[axis:]
            ),
            "paired_bias_access": parameter_access("Bias", paired),
            "paired_scale_access": parameter_access("Scale", paired),
            "outer_axes": tuple(range(axis)),
            "prefix_depth": axis,
            "reduction": reduction,
            "reduction_context": reduction_context,
            "scale_access": parameter_access("Scale", current),
            "square_reduction": square_reduction,
        }
    )
    return context


def _qkv_rope_with_cache_template_context(
    model: dict[str, Any],
) -> dict[str, Any]:
    q_context = _norm_rope_template_context(
        model["QNorm"], model["QRoPE"], "PyNTT QKVRoPEWithCache Q", "q"
    )
    k_context = _norm_rope_template_context(
        model["KNorm"], model["KRoPE"], "PyNTT QKVRoPEWithCache K", "k"
    )
    k_cache_context = _update_paged_attention_kv_cache_template_context(
        model["KUpdate"]
    )
    v_cache_context = _update_paged_attention_kv_cache_template_context(
        model["VUpdate"]
    )
    if model["KUpdate"]["Cache"] != model["VUpdate"]["Cache"]:
        raise ValueError("PyNTT QKVRoPEWithCache K/V updates must use one cache")
    if model["KUpdate"]["LayerIdExpression"] != model["VUpdate"]["LayerIdExpression"]:
        raise ValueError("PyNTT QKVRoPEWithCache K/V updates must use one layer id")
    if model["KUpdate"]["Hierarchy"] != model["VUpdate"]["Hierarchy"]:
        raise ValueError("PyNTT QKVRoPEWithCache K/V updates must use one hierarchy")

    qkv_layout = tuple(int(axis) for axis in model["QKVLayout"])
    attention_layout = tuple(int(axis) for axis in model["AttentionLayout"])
    if sorted(qkv_layout) != [0, 1, 2] or sorted(attention_layout) != [0, 1, 2]:
        raise ValueError(
            "PyNTT QKVRoPEWithCache layouts must each contain Seq, Head, and Dim"
        )
    permutation = tuple(qkv_layout.index(axis) for axis in attention_layout)
    q_output_shape = model["QOutputShape"]
    source_shape = model["QNorm"]["InputShape"]
    if len(q_output_shape) != len(permutation) or len(source_shape) != len(permutation):
        raise ValueError(
            "PyNTT QKVRoPEWithCache Q input/output rank must match its semantic layouts"
        )
    for output_axis, source_axis in enumerate(permutation):
        output_extent = _fixed(q_output_shape[output_axis])
        source_extent = _fixed(source_shape[source_axis])
        if (
            output_extent is not None
            and source_extent is not None
            and output_extent != source_extent
        ):
            raise ValueError(
                "PyNTT QKVRoPEWithCache Q layout permutation changes a physical extent: "
                f"output axis {output_axis} has {output_extent}, source axis "
                f"{source_axis} has {source_extent}"
            )
    output_lane_shape = tuple(int(value) for value in model["QOutputVectorLaneShape"])
    source_access = q_context["input_access"]
    if (
        output_lane_shape != tuple(source_access["LaneShape"])
        or int(model["QOutputVectorLaneCount"])
        != (_product_int(list(output_lane_shape)) if output_lane_shape else 1)
    ):
        raise ValueError(
            "PyNTT QKVRoPEWithCache Q input/output vector lane layouts must match"
        )
    q_context["final_output_access"] = _tensor_access(
        tuple(
            tuple(source_access["TensorIndices"])[source_axis]
            for source_axis in permutation
        ),
        model["QOutputStrides"],
        source_access["LaneIndices"],
        output_lane_shape,
    )

    k_cache_context.update(
        {
            "source_dim_block": k_context["output_physical_rotary"],
            "source_lane_id": _flatten_coordinates(
                k_context["input_access"]["LaneIndices"],
                k_context["input_access"]["LaneShape"],
            ),
            "source_tensor_coordinates": k_context["input_access"][
                "TensorIndices"
            ],
        }
    )
    k_cache_context["global_source_tensor_coordinates"] = (
        _distributed_local_to_global_coordinates(
            tuple(k_cache_context["source_tensor_coordinates"]),
            model["KUpdate"]["SlotsGlobalShape"],
            model["KUpdate"]["SlotsGlobalOffsets"],
            model["KUpdate"]["SlotsShardAxes"],
            model["KUpdate"]["Hierarchy"],
        )
    )
    k_context["cache"] = k_cache_context
    partials = (model.get("QPartial"), model.get("KPartial"), model.get("VPartial"))
    headwise_partial = False
    if any(partial is not None for partial in partials):
        if any(partial is None for partial in partials):
            raise ValueError(
                "PyNTT fused QKV gather-reduce requires Q, K, and V partial metadata"
            )
        q_partial, k_partial, v_partial = partials
        q_context["partial_input"] = _qkv_partial_input_context(
            q_partial,
            model["QNorm"]["Input"],
            q_context["input_access"],
            1 + len(q_context["lane_shape"]),
            qkv_layout,
            "Q",
        )
        q_context["paired_partial_input"] = _qkv_partial_input_context(
            q_partial,
            model["QNorm"]["Input"],
            q_context["paired_input_access"],
            1 + len(q_context["lane_shape"]),
            qkv_layout,
            "Q paired",
        )
        q_context["reduction_context"]["partial_input"] = (
            _qkv_partial_input_context(
                q_partial,
                model["QNorm"]["Input"],
                q_context["reduction_context"]["input_access"],
                1 + len(q_context["reduction_context"]["lane_shape"]),
                qkv_layout,
                "Q reduction",
            )
        )
        k_context["partial_input"] = _qkv_partial_input_context(
            k_partial,
            model["KNorm"]["Input"],
            k_context["input_access"],
            1 + len(k_context["lane_shape"]),
            qkv_layout,
            "K",
        )
        k_context["paired_partial_input"] = _qkv_partial_input_context(
            k_partial,
            model["KNorm"]["Input"],
            k_context["paired_input_access"],
            1 + len(k_context["lane_shape"]),
            qkv_layout,
            "K paired",
        )
        k_context["reduction_context"]["partial_input"] = (
            _qkv_partial_input_context(
                k_partial,
                model["KNorm"]["Input"],
                k_context["reduction_context"]["input_access"],
                1 + len(k_context["reduction_context"]["lane_shape"]),
                qkv_layout,
                "K reduction",
            )
        )
        v_cache_context["partial_input"] = _qkv_partial_input_context(
            v_partial,
            model["VUpdate"]["Slots"],
            v_cache_context["slots_access"],
            1 + len(v_cache_context["lane_shape"]),
            qkv_layout,
            "V",
        )
        headwise_partial = _configure_qkv_partial_headwise_context(
            model, q_context, k_context, v_cache_context
        )
        if not headwise_partial:
            # The generic schedule writes only the current owner's Q buffer.
            # Every replicated owner must therefore execute it; the headwise
            # schedule instead computes Q once and explicitly fans the result
            # out to all destination pools.
            q_context["partial_input"]["active"] = "True"
    return {
        "q": q_context,
        "k": k_context,
        "v": v_cache_context,
        "headwise_partial": headwise_partial,
    }


def _qkv_partial_input_context(
    partial: dict[str, Any],
    target: dict[str, Any],
    target_access: dict[str, Any],
    target_tile_rank: int,
    qkv_layout: tuple[int, ...],
    field: str,
) -> dict[str, Any]:
    """Map one logical Q/K/V access to compact per-owner partial storage."""

    if len(qkv_layout) != 3 or len(target_access["TensorIndices"]) != 3:
        raise ValueError(f"PyNTT fused {field} partial input requires rank-3 QKV")
    if len(partial["GlobalShape"]) != 2 or len(partial["Strides"]) != 2:
        raise ValueError(
            f"PyNTT fused {field} partial projection must be a packed rank-2 tensor"
        )
    hierarchy = [int(extent) for extent in partial["Hierarchy"]]
    if target.get("Hierarchy") != partial["Hierarchy"]:
        raise ValueError(
            f"PyNTT fused {field} partial and logical views must use one hierarchy"
        )
    partial_axes = tuple(sorted(int(axis) for axis in partial["PartialAxes"]))
    if not partial_axes or any(axis < 0 or axis >= len(hierarchy) for axis in partial_axes):
        raise ValueError(
            f"PyNTT fused {field} partial axes are invalid: {partial_axes}"
        )
    split_axes = set(_shard_axes_hierarchy_axes(partial["ShardAxes"]))
    if split_axes & set(partial_axes):
        raise ValueError(
            f"PyNTT fused {field} mesh axes cannot be both split and partial"
        )

    target_lane_shape = tuple(int(value) for value in target_access["LaneShape"])
    target_lane_count = _product_int(list(target_lane_shape)) if target_lane_shape else 1
    source_lane_shape = _validate_coordinate_lane_shape(
        partial["VectorLaneShape"], f"PyNTT fused {field} partial"
    )
    source_lane_count = _product_int(list(source_lane_shape)) if source_lane_shape else 1
    if source_lane_count != int(partial["VectorLaneCount"]):
        raise ValueError(
            f"PyNTT fused {field} partial lane metadata is inconsistent"
        )

    target_global = _distributed_local_to_global_coordinates(
        tuple(target_access["TensorIndices"]),
        target["GlobalShape"],
        target["GlobalOffsets"],
        target["ShardAxes"],
        target["Hierarchy"],
    )
    seq_axis = qkv_layout.index(0)
    head_axis = qkv_layout.index(1)
    dim_axis = qkv_layout.index(2)
    target_lane_flat = _flatten_coordinates(
        tuple(target_access["LaneIndices"]), target_lane_shape
    )
    logical_dim = (
        f"({target_global[dim_axis]}) * {target_lane_count} + ({target_lane_flat})"
    )
    logical_dim_extent = _multiply_dim(
        target["GlobalShape"][dim_axis], target_lane_count
    )
    flat_projection = (
        f"({target_global[head_axis]}) * ({_dim(logical_dim_extent)}) + "
        f"({logical_dim})"
    )
    source_global = (
        target_global[seq_axis],
        f"({flat_projection}) // {source_lane_count}",
    )
    source_lane_flat = f"({flat_projection}) % {source_lane_count}"
    source_lane_coordinates: list[str] = []
    lane_stride = source_lane_count
    for extent in source_lane_shape:
        lane_stride //= extent
        source_lane_coordinates.append(
            f"(({source_lane_flat}) // {lane_stride}) % {extent}"
            if lane_stride != 1
            else f"({source_lane_flat}) % {extent}"
        )

    axis_plans = tuple(
        _global_to_local_coordinate(
            source_global[axis],
            partial["GlobalShape"][axis],
            partial["ShardAxes"][axis],
            hierarchy,
        )
        for axis in range(2)
    )
    source_access = _tensor_access(
        tuple(plan["local_coordinate"] for plan in axis_plans),
        partial["Strides"],
        tuple(source_lane_coordinates),
        source_lane_shape,
    )
    owner_expressions: dict[int, str] = {}
    for plan in axis_plans:
        for axis, owner in plan["owners"].items():
            previous = owner_expressions.setdefault(int(axis), owner)
            if previous != owner:
                raise ValueError(
                    f"PyNTT fused {field} source mesh axis {axis} has conflicting owners"
                )

    reduction_extent = _product_int([hierarchy[axis] for axis in partial_axes])
    axis_strides: dict[int, int] = {}
    axis_stride = 1
    for axis in reversed(partial_axes):
        axis_strides[axis] = axis_stride
        axis_stride *= hierarchy[axis]
    target_split_axes = set(_shard_axes_hierarchy_axes(target["ShardAxes"]))
    active = " & ".join(
        f"(shard_coord{axis} == 0)"
        for axis in range(len(hierarchy))
        if axis not in target_split_axes
    ) or "True"
    dtype = partial["DType"]
    if dtype in ("float16", "bfloat16", "float32"):
        accumulator_dtype, zero = "tl.float32", "0.0"
    elif dtype == "float64":
        accumulator_dtype, zero = "tl.float64", "0.0"
    else:
        raise ValueError(
            f"PyNTT fused {field} partial Sum does not support dtype {dtype}"
        )

    return {
        "accumulator_dtype": accumulator_dtype,
        "active": active,
        "address": partial["Address"],
        "axis_strides": axis_strides,
        "hierarchy": hierarchy,
        "owner_expressions": owner_expressions,
        "partial_axes": partial_axes,
        "pointer_type": _pointer_type(
            partial["TritonDType"], partial["Address"]["AddressSpace"]
        ),
        "reduction_axis": target_tile_rank,
        "reduction_extent": reduction_extent,
        "reduction_lane_reshape": "["
        + ", ".join(["None"] * target_tile_rank + [":"])
        + "]",
        "target_reduction_reshape": "["
        + ", ".join([":"] * target_tile_rank + ["None"])
        + "]",
        "reduction_width": min(
            1 << (reduction_extent - 1).bit_length(), reduction_extent
        ),
        "scalar_element_size_bytes": int(partial["ScalarElementSizeBytes"]),
        "vector_lane_count": int(partial["VectorLaneCount"]),
        "triton_dtype": partial["TritonDType"],
        "source_access": source_access,
        "source_pool_index": _pool_index_expression(
            "source_shard_index", partial["Address"]["PoolScopeSize"]
        ),
        "source_pool_scope_size": partial["Address"]["PoolScopeSize"],
        "source_shard_index": _split_linear_expression(
            list(range(len(hierarchy))), hierarchy, "source_shard_coord"
        ),
        "zero": zero,
    }


def _qkv_headwise_partial_layout(
    function_name: str,
    field: str,
    elements: int,
    partial: dict[str, Any],
    num_warps: int,
    worker_width: int,
) -> dict[str, Any] | None:
    reduction_extent = int(partial["reduction_extent"])
    vector_lane_count = int(partial["vector_lane_count"])
    if (
        num_warps <= 0
        or num_warps & (num_warps - 1)
        or worker_width <= 0
        or worker_width & (worker_width - 1)
        or reduction_extent <= 0
        or reduction_extent & (reduction_extent - 1)
        or vector_lane_count <= 0
        or vector_lane_count & (vector_lane_count - 1)
    ):
        return None
    vector_lane_count = min(elements, vector_lane_count)
    hidden_threads = min(elements, worker_width)
    active_warps = min(num_warps, max(1, elements // hidden_threads))
    if (
        hidden_threads <= 0
        or hidden_threads & (hidden_threads - 1)
        or active_warps <= 0
        or active_warps & (active_warps - 1)
        or elements % (hidden_threads * active_warps)
    ):
        return None
    contiguous_elements = elements // (hidden_threads * active_warps)
    alignment_bytes = min(
        16, vector_lane_count * int(partial["scalar_element_size_bytes"])
    )
    try:
        pool_stride_bytes = int(partial["address"]["PoolStrideBytes"])
        offset_bytes = int(partial["address"]["OffsetBytes"])
    except (KeyError, TypeError, ValueError):
        return None
    if (
        alignment_bytes <= 0
        or pool_stride_bytes % alignment_bytes
        or offset_bytes % alignment_bytes
    ):
        return None
    return {
        "alignment_bytes": alignment_bytes,
        "contiguous_elements": contiguous_elements,
        "load_contiguous_elements": vector_lane_count,
        "name": f"{function_name}__{field}_partial_layout",
        "shape": (1, reduction_extent, elements),
        # Keep the short split-K reduction thread-local. Spread the head over
        # only the warps needed to cover it; surplus lanes and warps map to the
        # unit row dimension and are naturally out of bounds.
        "size_per_thread": (1, reduction_extent, contiguous_elements),
        "threads_per_warp": (worker_width // hidden_threads, 1, hidden_threads),
        "warps_per_cta": (num_warps // active_warps, 1, active_warps),
        "order": (2, 1, 0),
    }


def _configure_qkv_partial_headwise_context(
    model: dict[str, Any],
    q_context: dict[str, Any],
    k_context: dict[str, Any],
    v_context: dict[str, Any],
) -> bool:
    """Configure the full-head register-reuse lowering when it is legal."""

    def fixed_iteration_shape(context: dict[str, Any]) -> tuple[int, ...] | None:
        major_extent = _fixed(context["major_extent"])
        if major_extent is None:
            return None
        shape = (major_extent,) + tuple(int(value) for value in context["lane_shape"])
        if any(value <= 0 or value & (value - 1) for value in shape):
            return None
        return shape

    def has_unit_outer_domain(norm: dict[str, Any], context: dict[str, Any]) -> bool:
        return all(
            _fixed(norm["InputShape"][axis]) == 1
            for axis in context["outer_axes"]
        )

    def coordinate_plan(context: dict[str, Any]) -> dict[str, Any]:
        lane_shape = tuple(int(value) for value in context["lane_shape"])
        lane_strides: list[int] = []
        stride = _product_int(list(lane_shape)) if lane_shape else 1
        major_stride = stride
        for extent in lane_shape:
            stride //= extent
            lane_strides.append(stride)
        return {
            "lane_strides": tuple(lane_strides),
            "major_stride": major_stride,
        }

    q_reduction_shape = fixed_iteration_shape(q_context["reduction_context"])
    k_reduction_shape = fixed_iteration_shape(k_context["reduction_context"])
    q_compute_shape = fixed_iteration_shape(q_context)
    k_compute_shape = fixed_iteration_shape(k_context)
    v_compute_shape = fixed_iteration_shape(v_context)
    shapes = (
        q_reduction_shape,
        k_reduction_shape,
        q_compute_shape,
        k_compute_shape,
        v_compute_shape,
    )
    if any(shape is None for shape in shapes):
        return False
    assert q_reduction_shape is not None
    assert k_reduction_shape is not None
    assert q_compute_shape is not None
    assert k_compute_shape is not None
    assert v_compute_shape is not None

    q_elements = _product_int(list(q_reduction_shape))
    k_elements = _product_int(list(k_reduction_shape))
    if (
        q_elements != _product_int(list(q_compute_shape))
        or k_elements != _product_int(list(k_compute_shape))
        or k_elements != _product_int(list(v_compute_shape))
        or q_elements < 2
        or k_elements < 2
        or q_elements % 2 != 0
        or k_elements % 2 != 0
        or q_elements & (q_elements - 1)
        or k_elements & (k_elements - 1)
        or not has_unit_outer_domain(model["QNorm"], q_context)
        or not has_unit_outer_domain(model["KNorm"], k_context)
        or any(_fixed(v_context["tensor_shape"][axis]) != 1 for axis in v_context["loop_axes"])
    ):
        return False

    q_partial_layout = _qkv_headwise_partial_layout(
        model["FunctionName"],
        "q",
        q_elements,
        q_context["reduction_context"]["partial_input"],
        int(model["NumWarps"]),
        int(model["TargetWorkerWidth"]),
    )
    k_partial_layout = _qkv_headwise_partial_layout(
        model["FunctionName"],
        "k",
        k_elements,
        k_context["reduction_context"]["partial_input"],
        int(model["NumWarps"]),
        int(model["TargetWorkerWidth"]),
    )
    v_partial_layout = _qkv_headwise_partial_layout(
        model["FunctionName"],
        "v",
        k_elements,
        v_context["partial_input"],
        int(model["NumWarps"]),
        int(model["TargetWorkerWidth"]),
    )
    if (
        q_partial_layout is None
        or k_partial_layout is None
        or v_partial_layout is None
    ):
        return False

    def sliced_head_layout(field: str, parent: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": f"{model['FunctionName']}__{field}_head_layout",
            "parent_name": parent["name"],
            "sliced_dimension": 1,
        }

    q_head_layout = sliced_head_layout("q", q_partial_layout)
    k_head_layout = sliced_head_layout("k", k_partial_layout)
    v_head_layout = sliced_head_layout("v", v_partial_layout)

    q_output = model.get("QOutput")
    q_output_address = model.get("QOutputAddress")
    if not isinstance(q_output, dict) or not isinstance(q_output_address, dict):
        return False
    hierarchy = [int(extent) for extent in q_context["partial_input"]["hierarchy"]]
    if q_output.get("Hierarchy") != hierarchy:
        return False
    q_input_split_axes = set(
        _shard_axes_hierarchy_axes(model["QNorm"]["Input"]["ShardAxes"])
    )
    q_output_split_axes = set(_shard_axes_hierarchy_axes(q_output["ShardAxes"]))
    if q_input_split_axes != q_output_split_axes:
        return False
    q_broadcast_axes = tuple(
        axis for axis in range(len(hierarchy)) if axis not in q_output_split_axes
    )

    q_context.update(
        {
            "headwise_compute_coordinates": coordinate_plan(q_context),
            "headwise_compute_shape": (1, q_elements),
            "headwise_elements": q_elements,
            "headwise_layout": q_head_layout,
            "headwise_pair_shape": (2, q_elements // 2),
            "headwise_partial_layout": q_partial_layout,
            "headwise_reduction_coordinates": coordinate_plan(
                q_context["reduction_context"]
            ),
            "headwise_output": {
                "address": q_output_address,
                "broadcast_axes": q_broadcast_axes,
                "destination_pool_index": _pool_index_expression(
                    "destination_shard_index", q_output_address["PoolScopeSize"]
                ),
                "destination_shard_index": _split_linear_expression(
                    list(range(len(hierarchy))),
                    hierarchy,
                    "destination_shard_coord",
                ),
                "hierarchy": hierarchy,
                "pointer_type": _pointer_type(
                    model["QOutputTritonDType"], q_output_address["AddressSpace"]
                ),
            },
        }
    )
    k_context.update(
        {
            "headwise_compute_coordinates": coordinate_plan(k_context),
            "headwise_compute_shape": (1, k_elements),
            "headwise_elements": k_elements,
            "headwise_layout": k_head_layout,
            "headwise_pair_shape": (2, k_elements // 2),
            "headwise_partial_layout": k_partial_layout,
            "headwise_reduction_coordinates": coordinate_plan(
                k_context["reduction_context"]
            ),
        }
    )
    v_context.update(
        {
            "headwise_compute_coordinates": coordinate_plan(v_context),
            "headwise_compute_shape": (1, k_elements),
            "headwise_elements": k_elements,
            "headwise_layout": v_head_layout,
            "headwise_partial_layout": v_partial_layout,
        }
    )

    canonical_writer_axes = tuple(int(axis) for axis in v_context["canonical_writer_axes"])
    writer_coordinates = {axis: 0 for axis in canonical_writer_axes}
    if canonical_writer_axes:
        split_axis = canonical_writer_axes[0]
        if hierarchy[split_axis] > 1:
            writer_coordinates[split_axis] = 1
            v_context["partial_input"]["active"] = " & ".join(
                f"(shard_coord{axis} == {writer_coordinates.get(axis, 0)})"
                for axis in canonical_writer_axes
            )
    v_context["canonical_writer_coordinates"] = writer_coordinates
    return True


def _gather_template_context(model: dict[str, Any]) -> dict[str, Any]:
    """Prepare coordinate-native Gather input/index/output accesses."""

    lane_shape = _validate_coordinate_lane_shape(
        model["ValueVectorLaneShape"], "PyNTT Gather value"
    )
    lane_count = _product_int(list(lane_shape)) if lane_shape else 1
    if lane_count != int(model["ValueVectorLaneCount"]):
        raise ValueError(
            "PyNTT Gather vector lane shape/count mismatch: "
            f"shape={lane_shape}, count={model['ValueVectorLaneCount']}."
        )
    index_rank = len(model["IndexShape"])
    context = _coordinate_iteration_context(
        model["OutputShape"],
        model["OutputStrides"],
        list(lane_shape),
        "PyNTT Gather",
    )
    index_coordinates = []
    index_varies_on_major = False
    for index_axis, extent in enumerate(model["IndexShape"]):
        output_axis = model["Axis"] + index_axis
        index_coordinates.append(
            "0"
            if _is_fixed_one(extent)
            else context["tensor_coordinates"][output_axis]
        )
        index_varies_on_major = index_varies_on_major or (
            output_axis == context["major_axis"]
            and not _is_fixed_one(extent)
            and _fixed(model["IndexStrides"][index_axis]) != 0
        )
    index_access = _with_access_boundary_mask(
        _tensor_access(index_coordinates, model["IndexStrides"]),
        "mask" if index_varies_on_major else "True",
    )

    input_is_canonical = (
        model["Input"].get("DistributedStorageKind") == "CanonicalGlobal"
    )
    input_coordinates = []
    for input_axis in range(len(model["InputShape"])):
        if input_axis < model["Axis"]:
            coordinate = context["tensor_coordinates"][input_axis]
        elif input_axis == model["Axis"]:
            coordinate = "local_gather_index"
        else:
            coordinate = context["tensor_coordinates"][input_axis + index_rank - 1]
        input_coordinates.append(coordinate)
    input_access = _tensor_access(
        input_coordinates,
        model["InputStrides"],
        context["lane_coordinates"],
        lane_shape,
        global_coordinate_axes=(model["Axis"],) if input_is_canonical else (),
    )
    output_access = _tensor_access(
        context["tensor_coordinates"],
        model["OutputStrides"],
        context["lane_coordinates"],
        lane_shape,
    )

    gather_shard_axis = model["InputShardAxes"][model["Axis"]]
    gather_plan = _global_to_local_coordinate(
        "gather_index",
        model["InputGlobalShape"][model["Axis"]],
        gather_shard_axis,
        model["Hierarchy"],
    )
    if input_is_canonical:
        local_gather_index = "gather_index"
        input_owner_active = "True"
    else:
        local_gather_index = gather_plan["local_coordinate"]
        input_owner_active = " & ".join(
            f"(shard_coord{axis} == ({owner}))"
            for axis, owner in sorted(gather_plan["owners"].items())
        ) or "True"
    context.update(
        gather_is_split=bool(_shard_axis_stages(gather_shard_axis)),
        index_access=index_access,
        input_access=input_access,
        input_owner_active=input_owner_active,
        local_gather_index=local_gather_index,
        output_access=output_access,
        signed_index=not str(model["IndexDType"]).startswith("uint"),
    )
    if model["Axis"] < 0 or model["Axis"] >= len(model["InputShape"]):
        raise ValueError(
            f"PyNTT Gather axis {model['Axis']} is outside input rank "
            f"{len(model['InputShape'])}."
        )
    return context


def _concat_template_context(model: dict[str, Any]) -> dict[str, Any]:
    """Prepare each Concat input's iteration domain and output placement."""

    rank = len(model["OutputShape"])

    def axis_index(axis: int, block_axis: int) -> str:
        return "lane" if axis == block_axis else f"idx{axis}"

    entries = []
    axis_offset = _zero()
    for input_index, input_shape in enumerate(model["InputShapes"]):
        input_strides = model["InputStrides"][input_index]
        block_axis = _select_block_axis(input_shape, input_strides)
        input_terms = [
            f"{axis_index(axis, block_axis)} * {_dim(input_strides[axis])}"
            for axis in range(len(input_shape))
        ]
        output_terms = []
        for axis in range(rank):
            index = axis_index(axis, block_axis)
            if axis == model["Axis"] and _fixed(axis_offset) != 0:
                index = f"({index} + {_dim(axis_offset)})"
            output_terms.append(f"{index} * {_dim(model['OutputStrides'][axis])}")
        entries.append(
            {
                "block_extent": _one() if not input_shape else input_shape[block_axis],
                "index": input_index,
                "input_expression": model["Inputs"][input_index].get(
                    "Expression", model["Inputs"][input_index].get("expression")
                ),
                "input_offset": (
                    "lane * 0"
                    if not input_terms
                    else "lane * 0 + " + " + ".join(input_terms)
                ),
                "loop_axes": tuple(
                    axis for axis in range(len(input_shape)) if axis != block_axis
                ),
                "output_offset": (
                    "lane * 0"
                    if not output_terms
                    else "lane * 0 + " + " + ".join(output_terms)
                ),
                "shape": input_shape,
            }
        )
        axis_offset = _add_dims(axis_offset, input_shape[model["Axis"]])
    return {
        "entries": tuple(entries),
        "pointer_values": tuple(model["Inputs"]) + (model["Output"],),
    }


def _scatter_nd_template_context(model: dict[str, Any]) -> dict[str, Any]:
    """Prepare ScatterND copy/scatter domains and index expressions."""

    input_rank = len(model["InputShape"])
    updates_rank = len(model["UpdatesShape"])
    indices_rank = len(model["IndicesShape"])
    prefix_rank = indices_rank - 1
    index_depth = _fixed(model["IndicesShape"][-1])
    if index_depth is None:
        raise RuntimeError("ScatterND index depth must be fixed in PyNTT renderer.")
    slice_rank = input_rank - index_depth
    copy_block_axis = _select_block_axis(model["OutputShape"], model["OutputStrides"])
    updates_block_axis = _select_block_axis(
        model["UpdatesShape"], model["UpdatesStrides"]
    )

    def axis_index(prefix: str, axis: int, block_axis: int) -> str:
        return f"{prefix}_lane" if axis == block_axis else f"{prefix}_idx{axis}"

    def offset(
        prefix: str, shape: list[Any], strides: list[Any], block_axis: int
    ) -> str:
        terms = [
            f"{axis_index(prefix, axis, block_axis)} * {_dim(strides[axis])}"
            for axis in range(len(shape))
        ]
        return (
            f"{prefix}_lane * 0"
            if not terms
            else f"{prefix}_lane * 0 + " + " + ".join(terms)
        )

    indices_prefix_terms = [
        f"{axis_index('upd', axis, updates_block_axis)} * "
        f"{_dim(model['IndicesStrides'][axis])}"
        for axis in range(prefix_rank)
    ]
    updates_terms = [
        f"{axis_index('upd', axis, updates_block_axis)} * "
        f"{_dim(model['UpdatesStrides'][axis])}"
        for axis in range(updates_rank)
    ]
    scatter_terms = [
        f"scatter_idx{axis} * {_dim(model['OutputStrides'][axis])}"
        for axis in range(index_depth)
    ]
    for axis in range(slice_rank):
        updates_axis = prefix_rank + axis
        output_axis = index_depth + axis
        scatter_terms.append(
            f"{axis_index('upd', updates_axis, updates_block_axis)} * "
            f"{_dim(model['OutputStrides'][output_axis])}"
        )

    return {
        "copy_block_extent": (
            _one()
            if not model["OutputShape"]
            else model["OutputShape"][copy_block_axis]
        ),
        "copy_input_offset": offset(
            "copy", model["InputShape"], model["InputStrides"], copy_block_axis
        ),
        "copy_loop_axes": tuple(
            axis for axis in range(len(model["OutputShape"])) if axis != copy_block_axis
        ),
        "copy_output_offset": offset(
            "copy", model["OutputShape"], model["OutputStrides"], copy_block_axis
        ),
        "index_depth": index_depth,
        "indices_prefix_offset": (
            "upd_lane * 0"
            if not indices_prefix_terms
            else "upd_lane * 0 + " + " + ".join(indices_prefix_terms)
        ),
        "scatter_output_offset": (
            "upd_lane * 0"
            if not scatter_terms
            else "upd_lane * 0 + " + " + ".join(scatter_terms)
        ),
        "signed_indices": not str(model["IndicesDType"]).startswith("uint"),
        "updates_block_extent": (
            _one()
            if not model["UpdatesShape"]
            else model["UpdatesShape"][updates_block_axis]
        ),
        "updates_loop_axes": tuple(
            axis
            for axis in range(len(model["UpdatesShape"]))
            if axis != updates_block_axis
        ),
        "updates_offset": (
            "upd_lane * 0"
            if not updates_terms
            else "upd_lane * 0 + " + " + ".join(updates_terms)
        ),
    }


def _conv2d_template_context(model: dict[str, Any]) -> dict[str, Any]:
    """Validate Conv2D's fixed microkernel axes and prepare offsets."""

    stride_h, stride_w = model["Stride"][0], model["Stride"][1]
    pad_top, pad_left = model["Padding"][0], model["Padding"][2]
    dilation_h, dilation_w = model["Dilation"][0], model["Dilation"][1]
    input_channels_per_group = _fixed(model["WeightsShape"][1])
    output_channels = _fixed(model["OutputShape"][1])
    kernel_h = _fixed(model["WeightsShape"][2])
    kernel_w = _fixed(model["WeightsShape"][3])
    if None in (input_channels_per_group, output_channels, kernel_h, kernel_w):
        raise RuntimeError(
            "Conv2D PyNTT renderer requires fixed channel/kernel dimensions."
        )
    output_channels_per_group = output_channels // model["Groups"]
    block_axis = _select_block_axis(model["OutputShape"], model["OutputStrides"])

    def axis_index(axis: int) -> str:
        return "lane" if axis == block_axis else f"idx{axis}"

    n, oc, oh, ow = (axis_index(axis) for axis in range(4))
    group = "0" if model["Groups"] == 1 else f"{oc} // {output_channels_per_group}"
    input_channel = (
        "ic" if model["Groups"] == 1 else f"({group}) * {input_channels_per_group} + ic"
    )
    ih = f"{oh} * {stride_h} + kh * {dilation_h} - {pad_top}"
    iw = f"{ow} * {stride_w} + kw * {dilation_w} - {pad_left}"
    return {
        "bias_offset": f"lane * 0 + {oc} * {_dim(model['BiasStrides'][0])}",
        "block_extent": model["OutputShape"][block_axis],
        "ih": ih,
        "input_channels_per_group": input_channels_per_group,
        "input_offset": (
            f"lane * 0 + {n} * {_dim(model['InputStrides'][0])} + "
            f"({input_channel}) * {_dim(model['InputStrides'][1])} + "
            f"({ih}) * {_dim(model['InputStrides'][2])} + "
            f"({iw}) * {_dim(model['InputStrides'][3])}"
        ),
        "iw": iw,
        "kernel_h": kernel_h,
        "kernel_w": kernel_w,
        "loop_axes": tuple(
            axis for axis in range(len(model["OutputShape"])) if axis != block_axis
        ),
        "output_offset": (
            f"lane * 0 + {n} * {_dim(model['OutputStrides'][0])} + "
            f"{oc} * {_dim(model['OutputStrides'][1])} + "
            f"{oh} * {_dim(model['OutputStrides'][2])} + "
            f"{ow} * {_dim(model['OutputStrides'][3])}"
        ),
        "weights_offset": (
            f"lane * 0 + {oc} * {_dim(model['WeightsStrides'][0])} + "
            f"ic * {_dim(model['WeightsStrides'][1])} + "
            f"kh * {_dim(model['WeightsStrides'][2])} + "
            f"kw * {_dim(model['WeightsStrides'][3])}"
        ),
    }


def _reshard_template_context(model: dict[str, Any]) -> dict[str, Any]:
    """Validate Reshard placement and prepare its address expressions."""

    if model.get("Stage") != "tile_scatter":
        raise ValueError(
            f"Unsupported PyNTT direct reshard stage: {model.get('Stage')}"
        )
    input_split_mesh_axes = set(
        _shard_axes_hierarchy_axes(model["InputShardAxes"])
    )
    input_partial_mesh_axes = set(model["InputPartialAxes"])
    if input_split_mesh_axes & input_partial_mesh_axes:
        raise ValueError(
            "A PyNTT reshard mesh axis cannot be both split and partial: "
            f"{sorted(input_split_mesh_axes & input_partial_mesh_axes)}"
        )
    output_split_mesh_axes = set(
        _shard_axes_hierarchy_axes(model["OutputShardAxes"])
    )
    output_is_canonical = (
        model["Output"].get("DistributedStorageKind") == "CanonicalGlobal"
    )
    output_broadcast_mesh_axes = tuple(
        axis
        for axis in range(len(model["Hierarchy"]))
        if axis not in output_split_mesh_axes
    ) if not output_is_canonical else ()
    destination_local_mesh_axes = (
        tuple(
            axis
            for axis in output_broadcast_mesh_axes
            if axis not in input_split_mesh_axes
        )
        if input_partial_mesh_axes
        else ()
    )
    destination_loop_mesh_axes = tuple(
        axis
        for axis in output_broadcast_mesh_axes
        if axis not in destination_local_mesh_axes
    )
    writer_active = "True"
    for axis in sorted(input_partial_mesh_axes):
        if axis in destination_local_mesh_axes:
            continue
        owner = (
            f"destination_shard_coord{axis}"
            if axis in output_split_mesh_axes
            else "0"
        )
        writer_active = f"({writer_active}) & (shard_coord{axis} == {owner})"
    context = _coordinate_iteration_context(
        model["InputActiveShape"],
        model["InputStrides"],
        model["VectorLaneShape"],
        "PyNTT Reshard",
    )
    major_extent_max = _max_value(context["major_extent"])
    if major_extent_max is not None and major_extent_max > 0:
        context["physical_tile_width_cap"] = 1 << (major_extent_max - 1).bit_length()
    if context["lane_count"] != model["VectorLaneCount"]:
        raise ValueError(
            "PyNTT Reshard vector lane metadata is inconsistent: "
            f"shape={context['lane_shape']}, count={model['VectorLaneCount']}"
        )
    context["tile_shape"] = _coordinate_tile_shape(
        "elementwise_physical_tile_width", context["lane_shape"]
    )
    context["global_coordinates"] = tuple(
        _local_to_global_coordinate(
            context["tensor_coordinates"][axis],
            model["GlobalShape"][axis],
            model["InputShardAxes"][axis],
            model["Hierarchy"],
        )
        for axis in range(len(model["GlobalShape"]))
    )
    context["input_access"] = _tensor_access(
        context["tensor_coordinates"],
        model["InputStrides"],
        context["lane_coordinates"],
        context["lane_shape"],
        context["tile_shape"],
    )
    output_axis_plans = tuple(
        _global_to_local_coordinate(
            f"global_idx{axis}",
            model["GlobalShape"][axis],
            model["OutputShardAxes"][axis],
            model["Hierarchy"],
        )
        for axis in range(len(model["GlobalShape"]))
    )
    output_indices = (
        tuple(f"global_idx{axis}" for axis in range(len(model["OutputStrides"])))
        if output_is_canonical
        else tuple(f"output_idx{axis}" for axis in range(len(model["OutputStrides"])))
    )
    context["output_access"] = _tensor_access(
        output_indices,
        model["OutputStrides"],
        context["lane_coordinates"],
        context["lane_shape"],
        context["tile_shape"],
    )
    destination_shard_index = _split_linear_expression(
        list(range(len(model["Hierarchy"]))),
        model["Hierarchy"],
        "destination_shard_coord",
    )
    destination_pool_index = _pool_index_expression(
        "destination_shard_index", model["OutputAddress"]["PoolScopeSize"]
    )
    partial: dict[str, Any] | None = None
    if input_partial_mesh_axes:
        partial_input_address = model.get("PartialInputAddress")
        if partial_input_address is None:
            raise ValueError("PyNTT partial reshard requires PartialInputAddress")
        dtype = model["DType"]
        if dtype in ("float16", "bfloat16", "float32"):
            accumulator_dtype, zero = "tl.float32", "0.0"
        elif dtype == "float64":
            accumulator_dtype, zero = "tl.float64", "0.0"
        elif dtype.startswith("uint"):
            accumulator_dtype, zero = "tl.uint64", "0"
        elif dtype.startswith("int"):
            accumulator_dtype, zero = "tl.int64", "0"
        else:
            raise ValueError(f"PyNTT partial Sum does not support dtype {dtype}")
        source_shard_index = _split_linear_expression(
            list(range(len(model["Hierarchy"]))),
            model["Hierarchy"],
            "source_shard_coord",
        )
        reduction_axes = tuple(sorted(input_partial_mesh_axes))
        reduction_extent = _product_int(
            [model["Hierarchy"][axis] for axis in reduction_axes]
        )
        reduction_width_cap = 1 << (reduction_extent - 1).bit_length()
        axis_strides: dict[int, int] = {}
        axis_stride = 1
        for axis in reversed(reduction_axes):
            axis_strides[axis] = axis_stride
            axis_stride *= model["Hierarchy"][axis]
        tile_rank = 1 + len(context["lane_shape"])
        partial = {
            "accumulator_dtype": accumulator_dtype,
            "address": partial_input_address,
            "axes": reduction_axes,
            "axis_strides": axis_strides,
            "pointer_type": _pointer_type(
                model["TritonDType"], partial_input_address["AddressSpace"]
            ),
            "reduction_axis": tile_rank,
            "reduction_extent": reduction_extent,
            "append_reduction_axis": "["
            + ", ".join([":"] * tile_rank + ["None"])
            + "]",
            "reduction_lane_reshape": "["
            + ", ".join(["None"] * tile_rank + [":"])
            + "]",
            "reduction_width_cap": reduction_width_cap,
            "source_pool_index": _pool_index_expression(
                "source_shard_index", partial_input_address["PoolScopeSize"]
            ),
            "source_shard_index": source_shard_index,
            "zero": zero,
        }
    context.update(
        {
            "destination_local_mesh_axes": destination_local_mesh_axes,
            "destination_loop_mesh_axes": destination_loop_mesh_axes,
            "destination_pool_index": destination_pool_index,
            "destination_shard_index": destination_shard_index,
            "input_partial_mesh_axes": tuple(sorted(input_partial_mesh_axes)),
            "input_split_mesh_axes": tuple(sorted(input_split_mesh_axes)),
            "output_broadcast_mesh_axes": output_broadcast_mesh_axes,
            "output_axis_plans": output_axis_plans,
            "output_is_canonical": output_is_canonical,
            "output_pointer_type": _pointer_type(
                model["TritonDType"], model["OutputAddress"]["AddressSpace"]
            ),
            "partial": partial,
            "prefix_depth": len(destination_loop_mesh_axes),
            "writer_active": writer_active,
        }
    )
    return context


def _pool_index_expression(linear_index: str, pool_scope_size: Any) -> str:
    expression = str(pool_scope_size).strip()
    if not expression:
        raise ValueError("Pool scope size expression must not be empty")
    try:
        scope_size = int(expression)
    except ValueError:
        return f"(({linear_index}) // ({expression}))"
    if scope_size <= 0:
        raise ValueError(f"Pool scope size must be positive, got {scope_size}")
    return linear_index if scope_size == 1 else f"(({linear_index}) // {scope_size})"


def _summa_template_context(model: dict[str, Any]) -> dict[str, Any]:
    """Prepare SUMMA sharding and direct physical/vector N coordinates."""

    def output_axis_range(global_extent: Any, shard_axis: Any) -> dict[str, Any]:
        return {
            "active_extent": _local_axis_active_extent(
                global_extent, shard_axis, model["Hierarchy"]
            ),
            "global_extent": global_extent,
            "is_split": bool(_shard_axis_stages(shard_axis)),
            "shard_axis": shard_axis,
        }

    def local_index(
        prefix: str,
        global_index: str,
        global_extent: Any,
        shard_axis: Any,
    ) -> dict[str, Any]:
        plan = _global_to_local_coordinate(
            global_index, global_extent, shard_axis, model["Hierarchy"]
        )
        return {
            "global_extent": global_extent,
            "global_index": global_index,
            "is_split": bool(_shard_axis_stages(shard_axis)),
            "owners": plan["owners"],
            "prefix": prefix,
            "result": plan["local_coordinate"],
        }

    rhs_lanes = _validate_coordinate_lane_shape(
        model["RhsNVectorLaneShape"], "PyNTT SUMMA RHS N"
    )
    output_lanes = _validate_coordinate_lane_shape(
        model["OutputNVectorLaneShape"], "PyNTT SUMMA output N"
    )
    if rhs_lanes != output_lanes:
        raise ValueError(
            "PyNTT SUMMA RHS/output N lane shapes must match: "
            f"rhs={rhs_lanes}, output={output_lanes}."
        )
    lane_count = _product_int(list(rhs_lanes)) if rhs_lanes else 1
    if lane_count != int(model["RhsNVectorLaneCount"]) or lane_count != int(
        model["OutputNVectorLaneCount"]
    ):
        raise ValueError(
            "PyNTT SUMMA N lane shape/count metadata is inconsistent: "
            f"shape={rhs_lanes}, rhs={model['RhsNVectorLaneCount']}, "
            f"output={model['OutputNVectorLaneCount']}."
        )

    microkernel = _microkernel_context(model, "triton.summa")
    block_k = microkernel["parameters"]["block_k"]
    block_m = microkernel["parameters"]["block_m"]
    block_n = microkernel["parameters"]["block_n"]
    n_axis = _structured_axis_tile(
        "summa_n",
        rhs_lanes,
        block_n,
        _multiply_dim(model["OutputGlobalShape"][1], lane_count),
        leading_rank=1,
        physical_base="n_start",
    )
    lane_index = _flatten_coordinates(n_axis["lane_coordinates"], n_axis["lane_shape"])
    output_global_physical_n = model["OutputGlobalShape"][1]
    output_global_logical_n = _multiply_dim(output_global_physical_n, lane_count)
    rhs_global_logical_n = _multiply_dim(model["RhsGlobalShape"][1], lane_count)
    rhs_offset = "rhs_physical_offsets"
    output_offset = "output_physical_offsets"
    if lane_count != 1:
        rhs_offset = f"((rhs_physical_offsets) * {lane_count} + ({lane_index}))"
        output_offset = f"((output_physical_offsets) * {lane_count} + ({lane_index}))"
    broadcast_global_k = _broadcast_axis_coordinate("global_k", n_axis["rank"], 0)
    broadcast_offs_m = _broadcast_axis_coordinate("offs_m", n_axis["rank"], 0)
    global_n_physical = _local_to_global_coordinate(
        n_axis["physical_coordinate"],
        output_global_physical_n,
        model["OutputShardAxes"][1],
        model["Hierarchy"],
    )
    global_n_logical = f"({global_n_physical}) * {lane_count} + ({lane_index})"
    global_m = _local_to_global_coordinate(
        "offs_m",
        model["OutputGlobalShape"][0],
        model["OutputShardAxes"][0],
        model["Hierarchy"],
    )
    return {
        "microkernel": microkernel,
        "block_k": block_k,
        "block_m": block_m,
        "block_n": block_n,
        "dot_precision": (
            ', input_precision="ieee"'
            if model["LhsDType"] == "float32" and model["RhsDType"] == "float32"
            else ""
        ),
        "full_source_shard_index": _split_linear_expression(
            list(range(len(model["Hierarchy"]))),
            model["Hierarchy"],
            "source_shard_coord",
        ),
        "lhs_k": local_index(
            "lhs_k",
            "global_k[None, :]",
            model["LhsGlobalShape"][1],
            model["LhsShardAxes"][1],
        ),
        "lhs_m": local_index(
            "lhs_m",
            "global_m[:, None]",
            model["LhsGlobalShape"][0],
            model["LhsShardAxes"][0],
        ),
        "lhs_pointer_type": _pointer_type(
            model["LhsTritonDType"], model["LhsAddressSpace"]
        ),
        "out_m": output_axis_range(
            model["OutputGlobalShape"][0], model["OutputShardAxes"][0]
        ),
        "out_n": output_axis_range(
            output_global_physical_n, model["OutputShardAxes"][1]
        ),
        "broadcast_global_k": broadcast_global_k,
        "broadcast_offs_m": broadcast_offs_m,
        "global_n_logical": global_n_logical,
        "global_n_physical": global_n_physical,
        "global_m": global_m,
        "n_axis": n_axis,
        "output_global_logical_n": output_global_logical_n,
        "output_offset": output_offset,
        "output_pointer_type": _pointer_type(
            model["OutputTritonDType"], model["OutputAddressSpace"]
        ),
        "output_structured_shape": _structured_value_shape(
            n_axis, leading_extents=(block_m,)
        ),
        "physical_block_n": n_axis["physical_block_extent"],
        "rhs_global_logical_n": rhs_global_logical_n,
        "rhs_mask": (
            f"({broadcast_global_k} < {_dim(model['LhsGlobalShape'][1])}) & "
            f"({n_axis['physical_coordinate']} < out_n_iter_dim) & "
            f"({global_n_logical} < {_dim(rhs_global_logical_n)})"
        ),
        "rhs_k": local_index(
            "rhs_k",
            _broadcast_axis_coordinate("global_k", n_axis["rank"], 0),
            model["RhsGlobalShape"][0],
            model["RhsShardAxes"][0],
        ),
        "rhs_n": local_index(
            "rhs_n",
            "global_n_physical",
            model["RhsGlobalShape"][1],
            model["RhsShardAxes"][1],
        ),
        "rhs_offset": rhs_offset,
        "rhs_pointer_type": _pointer_type(
            model["RhsTritonDType"], model["RhsAddressSpace"]
        ),
        "rhs_structured_shape": _structured_value_shape(
            n_axis, leading_extents=(block_k,)
        ),
        "output_mask": (
            f"({broadcast_offs_m} < out_m_iter_dim) & "
            f"({n_axis['physical_coordinate']} < out_n_iter_dim) & "
            f"({global_n_logical} < {_dim(output_global_logical_n)})"
        ),
    }


def _paged_attention_tile_geometry(
    block_n: int,
    page_size: int,
    *,
    allow_cross_page: bool,
) -> tuple[int, int]:
    if block_n <= 0 or block_n & (block_n - 1):
        raise ValueError(
            "PyNTT PagedAttention block_n must be a positive power of two, "
            f"got {block_n}."
        )
    if page_size <= 0:
        raise ValueError(
            f"PyNTT PagedAttention cache page size must be positive, got {page_size}."
        )
    if block_n <= page_size:
        if page_size % block_n != 0:
            raise ValueError(
                "PyNTT PagedAttention page-local block_n must divide the cache "
                f"page size, got block_n={block_n}, page_size={page_size}."
            )
        return block_n, 1
    if not allow_cross_page or block_n % page_size != 0:
        raise ValueError(
            "PyNTT PagedAttention cross-page block_n requires a transfer "
            "pipeline and an integral number of cache pages, got "
            f"block_n={block_n}, page_size={page_size}."
        )
    return page_size, block_n // page_size


def _query_heads_form_contiguous_gqa_groups(
    shard_axis: dict[str, Any],
    hierarchy: list[int],
    *,
    global_query_heads: int,
    local_query_heads: int,
    query_heads_per_kv_head: int,
) -> bool:
    """Return whether the local query-head interval preserves whole GQA groups."""

    stages = _shard_axis_stages(shard_axis)
    if not stages:
        return True
    if len(stages) != 1:
        return False

    stage = stages[0]
    distribution = stage.get("Distribution")
    if distribution == "BlockCyclic":
        block_size = stage.get("BlockSize")
        return (
            isinstance(block_size, int)
            and block_size > 0
            and block_size % query_heads_per_kv_head == 0
            and local_query_heads <= block_size
        )
    if distribution == "Contiguous":
        granularity = stage.get("Granularity")
        capacity = _fixed(granularity) if granularity is not None else None
        if capacity is None:
            shard_count = _stage_shard_count(stage, hierarchy)
            capacity = (global_query_heads + shard_count - 1) // shard_count
        return (
            capacity > 0
            and capacity % query_heads_per_kv_head == 0
            and local_query_heads <= capacity
        )
    return False


def _paged_attention_template_context(model: dict[str, Any]) -> dict[str, Any]:
    """Validate PagedAttention layouts and prepare coordinate-native accesses."""

    cache = model["Cache"]
    microkernel_variant = None
    if model.get("MicroKernel") is not None:
        microkernel = _microkernel_context(
            model,
            "triton.paged_attention_partial",
            str(model["MicroKernel"]["Variant"]),
        )
        microkernel_variant = microkernel["variant"]
        attention_block_size = microkernel["parameters"]["block_n"]
        attention_num_stages = microkernel["parameters"]["num_stages"]
    else:
        kernel_config = model.get("KernelConfig")
        if not isinstance(kernel_config, dict):
            raise ValueError("PyNTT PagedAttention requires a renderer kernel config.")
        attention_config = kernel_config.get("paged_attention")
        if not isinstance(attention_config, dict):
            raise ValueError(
                "PyNTT PagedAttention requires a paged_attention backend config."
            )
        attention_block_size = _require_int(
            attention_config.get("block_n"),
            "PyNTT PagedAttention backend block_n",
            minimum=1,
        )
        attention_num_stages = _require_int(
            attention_config.get("num_stages"),
            "PyNTT PagedAttention backend num_stages",
            minimum=1,
        )
    cache_block_size = int(cache["BlockSize"])
    allow_cross_page = microkernel_variant in (
        "mma_tma_smem_pipeline",
        "simt_tma_smem_pipeline",
    )
    attention_copy_block_size, attention_copies_per_tile = (
        _paged_attention_tile_geometry(
            attention_block_size,
            cache_block_size,
            allow_cross_page=allow_cross_page,
        )
    )
    attention_reduction_block_size = (
        attention_block_size
        if microkernel_variant
        in ("mma_tma_smem_pipeline", "simt_tma_smem_pipeline")
        else attention_copy_block_size
    )
    attention_reductions_per_tile = (
        attention_block_size // attention_reduction_block_size
    )

    def global_index_expression(
        axis: int,
        local_index: str,
        global_extent: Any,
        *,
        local_extent: int | None = None,
    ) -> str:
        return _local_to_global_coordinate(
            local_index,
            global_extent,
            model["OutputShardAxes"][axis],
            model["Hierarchy"],
            local_extent=local_extent,
        )

    if cache["KeyVectorizedDim"] != 5:
        raise ValueError(
            "PyNTT PagedAttention requires the key cache to be HeadDim-vectorized."
        )
    if cache["ValueVectorizedDim"] not in (3, 5):
        raise ValueError(
            "PyNTT PagedAttention requires the value cache to be vectorized "
            "over BlockOffset or HeadDim."
        )

    query_lanes = _validate_coordinate_lane_shape(
        model["QueryVectorLaneShape"], "PyNTT PagedAttention query"
    )
    output_lanes = _validate_coordinate_lane_shape(
        model["OutputVectorLaneShape"], "PyNTT PagedAttention output"
    )
    if query_lanes != output_lanes:
        raise ValueError(
            "PyNTT PagedAttention query/output vector lanes must match: "
            f"query={query_lanes}, output={output_lanes}."
        )
    query_lane_count = _product_int(list(query_lanes)) if query_lanes else 1
    if query_lane_count != int(cache["KeyLaneCount"]):
        raise ValueError(
            "PyNTT PagedAttention query lanes must match key-cache HeadDim "
            f"lanes: query={query_lane_count}, cache={cache['KeyLaneCount']}."
        )

    dim_axis = int(model["DimAxis"])
    query_physical_dim = _constant_dim_value(model["QueryShape"][dim_axis])
    output_physical_dim = _constant_dim_value(model["OutputShape"][dim_axis])
    expected_physical_dim = int(cache["KeyHeadDimBlocks"])
    if (
        query_physical_dim != expected_physical_dim
        or output_physical_dim != expected_physical_dim
        or expected_physical_dim * query_lane_count != int(cache["HeadDim"])
    ):
        raise ValueError(
            "PyNTT PagedAttention query/output physical HeadDim does not "
            "match the cache layout: "
            f"query={query_physical_dim}, output={output_physical_dim}, "
            f"cache_blocks={expected_physical_dim}, lanes={query_lane_count}, "
            f"head_dim={cache['HeadDim']}."
        )

    global_num_query_heads = int(model["GlobalNumQueryHeads"])
    num_kv_heads = int(cache["NumKVHeads"])
    if (
        global_num_query_heads <= 0
        or num_kv_heads <= 0
        or global_num_query_heads % num_kv_heads != 0
    ):
        raise ValueError(
            "PyNTT PagedAttention requires a positive integral GQA group: "
            f"query_heads={global_num_query_heads}, kv_heads={num_kv_heads}."
        )
    local_q_heads = _constant_dim_value(model["OutputShape"][model["HeadAxis"]])
    if local_q_heads <= 0:
        raise ValueError(
            f"PyNTT PagedAttention local query-head extent must be positive, got {local_q_heads}."
        )
    q_head_group_size = global_num_query_heads // num_kv_heads
    q_head_group_tile = 1 << (q_head_group_size - 1).bit_length()
    contiguous_q_head_groups = _query_heads_form_contiguous_gqa_groups(
        model["OutputShardAxes"][model["HeadAxis"]],
        model["Hierarchy"],
        global_query_heads=global_num_query_heads,
        local_query_heads=local_q_heads,
        query_heads_per_kv_head=q_head_group_size,
    )
    global_q_head_begin = global_index_expression(
        model["HeadAxis"],
        "0",
        global_num_query_heads,
        local_extent=local_q_heads,
    )
    global_q_head_from_local = global_index_expression(
        model["HeadAxis"],
        "local_kv_group",
        global_num_query_heads,
        local_extent=local_q_heads,
    )

    query_dim_axis = _structured_axis_tile(
        "query_dim",
        query_lanes,
        int(cache["HeadDim"]),
        cache["HeadDim"],
        leading_rank=1,
    )
    key_dim_axis = _structured_axis_tile(
        "key_dim",
        query_lanes,
        int(cache["HeadDim"]),
        cache["HeadDim"],
        trailing_rank=1,
    )

    query_indices = ["0"] * len(model["QueryShape"])
    query_indices[model["SeqAxis"]] = "local_query_id"
    query_indices[model["HeadAxis"]] = _broadcast_axis_coordinate(
        "safe_local_q_heads", query_dim_axis["rank"], 0
    )
    query_indices[dim_axis] = query_dim_axis["physical_coordinate"]
    output_indices = ["0"] * len(model["OutputShape"])
    output_indices[model["SeqAxis"]] = "local_query_id"
    output_indices[model["HeadAxis"]] = _broadcast_axis_coordinate(
        "safe_local_q_heads", query_dim_axis["rank"], 0
    )
    output_indices[dim_axis] = query_dim_axis["physical_coordinate"]
    query_structured_shape = _structured_value_shape(
        query_dim_axis, leading_extents=(q_head_group_tile,)
    )

    key_lane = _flatten_coordinates(
        key_dim_axis["lane_coordinates"], key_dim_axis["lane_shape"]
    )
    key_block_offset = _broadcast_axis_coordinate(
        "block_offsets", key_dim_axis["rank"], key_dim_axis["rank"] - 1
    )
    key_vector_offset = (
        f"({cache['KeySectionOffset']} + ((layer_id_value) * "
        f"{cache['KeyLayerStride']} + kv_head * {cache['KeyHeadStride']} + "
        f"({key_dim_axis['physical_coordinate']}) * "
        f"{cache['KeyDimBlockStride']} + ({key_block_offset}) * "
        f"{cache['KeyBlockOffsetStride']}) * {cache['KeyLaneCount']} + "
        f"({key_lane}))"
    )

    value_lane_count = int(cache["ValueLaneCount"])
    if value_lane_count <= 0 or value_lane_count & (value_lane_count - 1):
        raise ValueError(
            "PyNTT PagedAttention value-cache lane count must be a positive "
            f"power of two, got {value_lane_count}."
        )
    if cache["ValueVectorizedDim"] == 3:
        if (
            attention_block_size % value_lane_count != 0
            or cache_block_size % value_lane_count != 0
            or int(cache["ValueHeadDimBlocks"]) != int(cache["HeadDim"])
        ):
            raise ValueError(
                "PyNTT PagedAttention BlockOffset-vectorized value cache has "
                "an incompatible block or HeadDim layout."
            )
        value_axis = _structured_axis_tile(
            "value_context",
            (value_lane_count,),
            attention_block_size,
            attention_block_size,
            trailing_rank=1,
            physical_base=(
                f"((context_start_i32 % {cache_block_size}) // {value_lane_count})"
            ),
        )
        value_lane = value_axis["lane_coordinates"][0]
        value_dim_index = _broadcast_axis_coordinate(
            "dim_offsets", value_axis["rank"], value_axis["rank"] - 1
        )
        value_vector_offset = (
            f"({cache['ValueSectionOffset']} + ((layer_id_value) * "
            f"{cache['ValueLayerStride']} + kv_head * "
            f"{cache['ValueHeadStride']} + ({value_dim_index}) * "
            f"{cache['ValueDimBlockStride']} + "
            f"({value_axis['physical_coordinate']}) * "
            f"{cache['ValueBlockOffsetStride']}) * {value_lane_count} + "
            f"({value_lane}))"
        )
        value_mask = (
            f"tl.reshape(context_mask, {value_axis['structured_shape']})[:, :, None]"
        )
        value_structured_shape = _structured_value_shape(
            value_axis, trailing_extents=(int(cache["HeadDim"]),)
        )
        value_axis_kind = "context"
    else:
        if (
            int(cache["HeadDim"]) % value_lane_count != 0
            or int(cache["ValueHeadDimBlocks"])
            != int(cache["HeadDim"]) // value_lane_count
        ):
            raise ValueError(
                "PyNTT PagedAttention HeadDim-vectorized value cache has an "
                "incompatible HeadDim layout."
            )
        value_axis = _structured_axis_tile(
            "value_dim",
            (value_lane_count,),
            int(cache["HeadDim"]),
            cache["HeadDim"],
            leading_rank=1,
        )
        value_lane = value_axis["lane_coordinates"][0]
        value_block_offset = _broadcast_axis_coordinate(
            "block_offsets", value_axis["rank"], 0
        )
        value_vector_offset = (
            f"({cache['ValueSectionOffset']} + ((layer_id_value) * "
            f"{cache['ValueLayerStride']} + kv_head * "
            f"{cache['ValueHeadStride']} + "
            f"({value_axis['physical_coordinate']}) * "
            f"{cache['ValueDimBlockStride']} + ({value_block_offset}) * "
            f"{cache['ValueBlockOffsetStride']}) * {value_lane_count} + "
            f"({value_lane}))"
        )
        value_mask = "context_mask[:, None, None]"
        value_structured_shape = _structured_value_shape(
            value_axis, leading_extents=(attention_block_size,)
        )
        value_axis_kind = "dim"

    local_query_tokens = model["OutputShape"][model["SeqAxis"]]
    global_query_tokens = model["OutputGlobalShape"][model["SeqAxis"]]
    context = {
        "attention_block_size": attention_block_size,
        "attention_copy_block_size": attention_copy_block_size,
        "attention_copies_per_tile": attention_copies_per_tile,
        "attention_reduction_block_size": attention_reduction_block_size,
        "attention_reductions_per_tile": attention_reductions_per_tile,
        "attention_num_stages": attention_num_stages,
        "key_cache_block_id": (
            "(key_topology_id * num_blocks_per_shard + key_block_id)"
            if cache["IdLength"] > 1
            else "key_block_id"
        ),
        "value_cache_block_id": (
            "(key_topology_id * num_blocks_per_shard + key_block_id)"
            if cache["IdLength"] > 1
            else "key_block_id"
        ),
        "global_q_head_begin": global_q_head_begin,
        "global_q_head_from_local": global_q_head_from_local,
        "global_query_id": global_index_expression(
            model["SeqAxis"], "local_query_id", global_query_tokens
        ),
        "global_query_tokens": global_query_tokens,
        "key_dim_axis": key_dim_axis,
        "key_mask": _broadcast_axis_coordinate(
            "context_mask", key_dim_axis["rank"], key_dim_axis["rank"] - 1
        ),
        "key_structured_shape": _structured_value_shape(
            key_dim_axis, trailing_extents=(attention_block_size,)
        ),
        "key_vector_offset": key_vector_offset,
        "local_q_heads": local_q_heads,
        "contiguous_q_head_groups": contiguous_q_head_groups,
        "local_query_tokens": local_query_tokens,
        "output_mask": _broadcast_axis_coordinate(
            "q_head_mask", query_dim_axis["rank"], 0
        ),
        "output_access": _tensor_access(
            output_indices,
            model["OutputStrides"],
            query_dim_axis["lane_coordinates"],
            output_lanes,
            _coordinate_shape(query_structured_shape),
        ),
        "q_head_group_size": q_head_group_size,
        "q_head_group_tile": q_head_group_tile,
        "query_mask": _broadcast_axis_coordinate(
            "q_head_mask", query_dim_axis["rank"], 0
        ),
        "query_access": _tensor_access(
            query_indices,
            model["QueryStrides"],
            query_dim_axis["lane_coordinates"],
            query_lanes,
            _coordinate_shape(query_structured_shape),
        ),
        "query_dim_axis": query_dim_axis,
        "query_structured_shape": query_structured_shape,
        "value_axis": value_axis,
        "value_axis_kind": value_axis_kind,
        "value_mask": value_mask,
        "value_structured_shape": value_structured_shape,
        "value_vector_offset": value_vector_offset,
    }
    if microkernel_variant in ("simt_direct", "simt_tma_smem_pipeline"):
        target_worker_width = int(model["TargetWorkerWidth"])
        consumer_warps = int(model["NumWarps"])
        vector_elements = 8
        if (
            model["QueryDType"] != "bfloat16"
            or int(cache["HeadDim"]) % vector_elements != 0
        ):
            raise ValueError(
                "PyNTT SIMT paged attention requires a BF16 query and a "
                "HeadDim divisible by eight elements."
            )
        dim_threads = int(cache["HeadDim"]) // vector_elements
        if (
            dim_threads <= 0
            or target_worker_width % dim_threads != 0
            or q_head_group_tile > consumer_warps
            or consumer_warps % q_head_group_tile != 0
        ):
            raise ValueError(
                "PyNTT SIMT paged attention cannot partition its GQA and "
                "HeadDim axes over the configured consumer workers: "
                f"group_tile={q_head_group_tile}, head_dim={cache['HeadDim']}, "
                f"worker_width={target_worker_width}, warps={consumer_warps}."
            )
        token_threads = target_worker_width // dim_threads
        token_warps = consumer_warps // q_head_group_tile
        if attention_reduction_block_size % (token_threads * token_warps) != 0:
            raise ValueError(
                "PyNTT SIMT paged attention requires exact token coverage by "
                "its thread/warp partition: "
                f"reduction_block_n={attention_reduction_block_size}, "
                f"token_threads={token_threads}, "
                f"token_warps={token_warps}."
            )
        function_name = model["FunctionName"]
        context.update(
            simt_product_size_per_thread=(1, 1, vector_elements),
            simt_product_threads_per_warp=(1, token_threads, dim_threads),
            simt_product_warps_per_cta=(
                q_head_group_tile,
                token_warps,
                1,
            ),
            simt_product_layout_name=f"{function_name}__product_layout",
            simt_query_layout_name=f"{function_name}__query_layout",
            simt_score_layout_name=f"{function_name}__score_layout",
            simt_acc_layout_name=f"{function_name}__acc_layout",
        )
    return context


def _paged_attention_partial_template_context(
    model: dict[str, Any],
) -> dict[str, Any]:
    """Extend the common attention coordinates with FP32 partial-state stores."""

    has_direct_output = bool(model.get("HasDirectOutput"))
    if has_direct_output != isinstance(model.get("Output"), dict):
        raise ValueError(
            "PyNTT PagedAttentionPartial direct-output metadata and pointer disagree"
        )
    if not has_direct_output and int(model["DirectContextThreshold"]) != 0:
        raise ValueError(
            "PyNTT state-only PagedAttentionPartial requires a zero direct-context threshold"
        )

    ctx = _paged_attention_template_context(model)
    query_rank = len(model["QueryShape"])
    state_rank = query_rank
    for name in ("MaxState", "SumState", "AccState"):
        if len(model[f"{name}Shape"]) != state_rank or len(
            model[f"{name}Strides"]
        ) != state_rank:
            raise ValueError(
                f"PyNTT PagedAttentionPartial {name} must have rank {state_rank}"
            )

    split_count = int(model["SplitCount"])
    split_axis = int(model["SplitHierarchyAxis"])
    if (
        split_axis < 0
        or split_axis >= len(model["Hierarchy"])
        or split_count <= 1
        or split_count > model["Hierarchy"][split_axis]
    ):
        raise ValueError(
            "PyNTT PagedAttentionPartial split count must fit its hierarchy axis"
        )

    def state_indices(head_index: str, dim_index: str) -> list[str]:
        indices = ["0"] * state_rank
        indices[model["SeqAxis"]] = "local_query_id"
        indices[model["HeadAxis"]] = head_index
        indices[model["DimAxis"]] = dim_index
        return indices

    q_head_group_tile = int(ctx["q_head_group_tile"])
    head_dimension = int(model["Cache"]["HeadDim"])
    ctx.update(
        acc_state_access=_tensor_access(
            state_indices("safe_local_q_heads[:, None]", "dim_offsets[None, :]"),
            model["AccStateStrides"],
            coordinate_shape=f"({q_head_group_tile}, {head_dimension})",
        ),
        max_state_access=_tensor_access(
            state_indices("safe_local_q_heads", "0"),
            model["MaxStateStrides"],
            coordinate_shape=f"({q_head_group_tile},)",
        ),
        split_coord=f"shard_coord{split_axis}",
        split_count=split_count,
        sum_state_access=_tensor_access(
            state_indices("safe_local_q_heads", "0"),
            model["SumStateStrides"],
            coordinate_shape=f"({q_head_group_tile},)",
        ),
        simt_acc_state_access=_tensor_access(
            state_indices("state_local_q_heads", "state_dim_offsets"),
            model["AccStateStrides"],
            coordinate_shape=f"({q_head_group_tile}, {head_dimension})",
        ),
        simt_max_state_access=_tensor_access(
            state_indices("state_scalar_q_heads", "0"),
            model["MaxStateStrides"],
            coordinate_shape=f"({q_head_group_tile},)",
        ),
        simt_sum_state_access=_tensor_access(
            state_indices("state_scalar_q_heads", "0"),
            model["SumStateStrides"],
            coordinate_shape=f"({q_head_group_tile},)",
        ),
    )
    return ctx


def _paged_attention_merge_template_context(
    model: dict[str, Any],
) -> dict[str, Any]:
    """Prepare output and split-state coordinates for online-softmax merging."""

    output_rank = len(model["OutputShape"])
    state_rank = output_rank
    for name in ("MaxState", "SumState", "AccState"):
        if len(model[f"{name}Shape"]) != state_rank or len(
            model[f"{name}Strides"]
        ) != state_rank:
            raise ValueError(
                f"PyNTT PagedAttentionMerge {name} must have rank {state_rank}"
            )

    split_count = int(model["SplitCount"])
    split_axis = int(model["SplitHierarchyAxis"])
    if (
        split_axis < 0
        or split_axis >= len(model["Hierarchy"])
        or split_count <= 1
        or split_count > model["Hierarchy"][split_axis]
    ):
        raise ValueError(
            "PyNTT PagedAttentionMerge split count must fit its hierarchy axis"
        )
    split_tile = 1 << (split_count - 1).bit_length()
    head_dimension = int(model["HeadDimension"])
    output_lanes = _validate_coordinate_lane_shape(
        model["OutputVectorLaneShape"], "PyNTT PagedAttentionMerge output"
    )
    output_lane_count = _product_int(list(output_lanes)) if output_lanes else 1
    dim_axis = int(model["DimAxis"])
    output_physical_dim = _constant_dim_value(model["OutputShape"][dim_axis])
    if output_physical_dim is None:
        raise ValueError(
            "PyNTT PagedAttentionMerge requires a fixed local output HeadDim"
        )
    local_head_dimension = output_physical_dim * output_lane_count

    output_dim_axis = _structured_axis_tile(
        "output_dim",
        output_lanes,
        local_head_dimension,
        head_dimension,
    )

    def global_index_expression(
        axis: int,
        local_index: str,
        global_extent: Any,
        local_extent: int | None = None,
    ) -> str:
        return _local_to_global_coordinate(
            local_index,
            global_extent,
            model["OutputShardAxes"][axis],
            model["Hierarchy"],
            local_extent=local_extent,
        )

    global_query_tokens = model["OutputGlobalShape"][model["SeqAxis"]]
    global_query_id = global_index_expression(
        model["SeqAxis"],
        "local_query_id",
        global_query_tokens,
        _constant_dim_value(model["OutputShape"][model["SeqAxis"]]),
    )
    global_q_head = global_index_expression(
        model["HeadAxis"],
        "q_head",
        model["GlobalNumQueryHeads"],
        _constant_dim_value(model["OutputShape"][model["HeadAxis"]]),
    )
    global_output_physical_dim = global_index_expression(
        dim_axis,
        output_dim_axis["physical_coordinate"],
        model["OutputGlobalShape"][dim_axis],
        output_physical_dim,
    )
    output_scalar_global_extent = _multiply_dim(
        model["OutputGlobalShape"][dim_axis], output_lane_count
    )
    output_scalar_shard_axis = _scale_shard_axis_mapping(
        model["OutputShardAxes"][dim_axis], output_lane_count
    )
    global_output_scalar_dim = _local_to_global_coordinate(
        output_dim_axis["logical_expression"],
        output_scalar_global_extent,
        output_scalar_shard_axis,
        model["Hierarchy"],
        local_extent=local_head_dimension,
    )
    state_dim_global_extent = model["StateGlobalShape"][dim_axis]
    if not _dimensions_equivalent(
        output_scalar_global_extent, state_dim_global_extent
    ):
        raise ValueError(
            "PyNTT PagedAttentionMerge output scalar HeadDim and partial-state "
            "HeadDim disagree"
        )
    flat_global_output_scalar_dim = _local_to_global_coordinate(
        "dim_grid",
        output_scalar_global_extent,
        output_scalar_shard_axis,
        model["Hierarchy"],
        local_extent=local_head_dimension,
    )

    state_query_id = _remap_local_coordinate(
        "local_query_id",
        global_query_tokens,
        model["OutputShardAxes"][model["SeqAxis"]],
        model["StateGlobalShape"][model["SeqAxis"]],
        model["StateShardAxes"][model["SeqAxis"]],
        model["Hierarchy"],
        local_extent=_constant_dim_value(
            model["OutputShape"][model["SeqAxis"]]
        ),
    )
    state_q_head = _remap_local_coordinate(
        "q_head",
        model["OutputGlobalShape"][model["HeadAxis"]],
        model["OutputShardAxes"][model["HeadAxis"]],
        model["StateGlobalShape"][model["HeadAxis"]],
        model["StateShardAxes"][model["HeadAxis"]],
        model["Hierarchy"],
        local_extent=_constant_dim_value(
            model["OutputShape"][model["HeadAxis"]]
        ),
    )
    state_dim = _remap_local_coordinate(
        "dim_grid",
        output_scalar_global_extent,
        output_scalar_shard_axis,
        state_dim_global_extent,
        model["StateShardAxes"][dim_axis],
        model["Hierarchy"],
        local_extent=local_head_dimension,
    )

    output_is_canonical = (
        model["Output"].get("DistributedStorageKind") == "CanonicalGlobal"
    )
    output_indices = ["0"] * output_rank
    output_indices[model["SeqAxis"]] = (
        global_query_id if output_is_canonical else "local_query_id"
    )
    output_indices[model["HeadAxis"]] = (
        global_q_head if output_is_canonical else "q_head"
    )
    output_indices[dim_axis] = (
        global_output_physical_dim
        if output_is_canonical
        else output_dim_axis["physical_coordinate"]
    )

    def state_indices(dim_index: str) -> list[str]:
        indices = ["0"] * state_rank
        indices[model["SeqAxis"]] = state_query_id
        indices[model["HeadAxis"]] = state_q_head
        indices[dim_axis] = dim_index
        return indices

    return {
        "acc_state_access": _tensor_access(
            state_indices(state_dim),
            model["AccStateStrides"],
            coordinate_shape=f"({split_tile}, {local_head_dimension})",
        ),
        "acc_state_source_pointer": _partial_state_source_pointer(
            model, "AccState", "split_offsets"
        ),
        "global_q_head": global_q_head,
        "global_query_id": global_query_id,
        "local_q_heads": model["OutputShape"][model["HeadAxis"]],
        "local_head_dimension": local_head_dimension,
        "local_query_tokens": model["OutputShape"][model["SeqAxis"]],
        "max_state_access": _tensor_access(
            state_indices("0"),
            model["MaxStateStrides"],
            coordinate_shape=f"({split_tile},)",
        ),
        "max_state_source_pointer": _partial_state_source_pointer(
            model, "MaxState", "split_offsets"
        ),
        "output_access": _tensor_access(
            output_indices,
            model["OutputStrides"],
            output_dim_axis["lane_coordinates"],
            output_lanes,
            _coordinate_shape(output_dim_axis["structured_shape"]),
            global_coordinate_axes=(
                (model["SeqAxis"], model["HeadAxis"], dim_axis)
                if output_is_canonical
                else ()
            ),
        ),
        "output_active": (
            f"({global_query_id} < {_dim(global_query_tokens)}) & "
            f"({global_q_head} < {int(model['GlobalNumQueryHeads'])}) & "
            f"({global_output_scalar_dim} < {head_dimension})"
        ),
        "output_dim_axis": output_dim_axis,
        "output_structured_shape": output_dim_axis["structured_shape"],
        "split_count": split_count,
        "split_tile": split_tile,
        "sum_state_access": _tensor_access(
            state_indices("0"),
            model["SumStateStrides"],
            coordinate_shape=f"({split_tile},)",
        ),
        "sum_state_source_pointer": _partial_state_source_pointer(
            model, "SumState", "split_offsets"
        ),
    }


def _partial_state_source_pointer(
    model: dict[str, Any],
    name: str,
    source_axis_coordinate: str,
    source_owner_coordinates: dict[int, str] | None = None,
) -> str:
    """Address one contributor's compact-local partial state in its pool."""

    address = model.get(f"{name}Address")
    if not isinstance(address, dict):
        raise ValueError(f"PyNTT partial state {name} requires pooled address metadata")
    split_axis = int(model["SplitHierarchyAxis"])
    hierarchy = tuple(int(extent) for extent in model["Hierarchy"])
    if split_axis < 0 or split_axis >= len(hierarchy):
        raise ValueError(
            f"PyNTT partial state split axis {split_axis} is outside rank {len(hierarchy)}"
        )
    source_coordinates = [f"shard_coord{axis}" for axis in range(len(hierarchy))]
    source_coordinates[split_axis] = source_axis_coordinate
    for axis, coordinate in (source_owner_coordinates or {}).items():
        if axis < 0 or axis >= len(hierarchy):
            raise ValueError(
                f"PyNTT partial state owner axis {axis} is outside rank {len(hierarchy)}"
            )
        if axis == split_axis:
            raise ValueError(
                "PyNTT partial state owner coordinates must not replace its partial axis"
            )
        source_coordinates[axis] = coordinate
    source_shard_index = source_coordinates[0]
    for axis in range(1, len(hierarchy)):
        source_shard_index = (
            f"(({source_shard_index}) * {hierarchy[axis]} + "
            f"({source_coordinates[axis]}))"
        )
    source_pool_index = _pool_index_expression(
        source_shard_index, address["PoolScopeSize"]
    )
    byte_offset = (
        f"({source_pool_index}) * ({address['PoolStrideBytes']})"
        f" + ({address['OffsetBytes']})"
    )
    pointer_type = _pointer_type("tl.float32", address["AddressSpace"])
    return f"({address['BaseName']} + ({byte_offset})).to({pointer_type})"


def _update_paged_attention_kv_cache_template_context(
    model: dict[str, Any],
) -> dict[str, Any]:
    """Prepare coordinate-native cache and slot addresses."""

    cache = model["Cache"]
    kind_prefix = "Key" if model["CacheKind"] == 0 else "Value"
    lane_count = cache[f"{kind_prefix}LaneCount"]
    vectorized_dim = cache[f"{kind_prefix}VectorizedDim"]
    slots_lane_shape = _validate_coordinate_lane_shape(
        model["SlotsVectorLaneShape"], "PyNTT UpdatePagedAttentionKVCache slots"
    )
    slots_lane_count = _product_int(list(slots_lane_shape)) if slots_lane_shape else 1
    if slots_lane_count != int(model["SlotsVectorLaneCount"]):
        raise ValueError(
            "PyNTT UpdatePagedAttentionKVCache slot lane shape/count mismatch: "
            f"shape={slots_lane_shape}, count={model['SlotsVectorLaneCount']}"
        )
    source_split_axes = sorted(
        _shard_axes_hierarchy_axes(model["SlotsSourceShardAxes"])
    )
    topology_match_axes = tuple(
        axis for axis in cache["NumBlocksHierarchyAxes"] if axis not in source_split_axes
    )
    canonical_writer_axes = tuple(
        axis
        for axis in range(len(model["Hierarchy"]))
        if axis not in source_split_axes and axis not in topology_match_axes
    )
    block_index = (
        "(topology_id * num_blocks_per_shard + block_id)"
        if cache["IdLength"] > 1
        else "block_id"
    )
    cache_offset = (
        f"({block_index} * {cache['BlockElements']} + "
        f"{cache[f'{kind_prefix}SectionOffset']} + (layer_id_value * "
        f"{cache[f'{kind_prefix}LayerStride']} + cache_head_id * "
        f"{cache[f'{kind_prefix}HeadStride']} + cache_dim_block * "
        f"{cache[f'{kind_prefix}DimBlockStride']} + cache_block_offset * "
        f"{cache[f'{kind_prefix}BlockOffsetStride']}) * {lane_count} + "
        "cache_lane_id)"
    )

    context = _coordinate_iteration_context(
        model["SlotsShape"],
        model["SlotsStrides"],
        model["SlotsVectorLaneShape"],
        "PyNTT UpdatePagedAttentionKVCache",
    )
    source_lane_id = _flatten_coordinates(
        context["lane_coordinates"], context["lane_shape"]
    )
    global_source_tensor_coordinates = _distributed_local_to_global_coordinates(
        tuple(context["tensor_coordinates"]),
        model["SlotsGlobalShape"],
        model["SlotsGlobalOffsets"],
        model["SlotsShardAxes"],
        model["Hierarchy"],
    )
    context.update(
        {
            "cache_offset": cache_offset,
            "canonical_writer_axes": canonical_writer_axes,
            "kind_prefix": kind_prefix,
            "lane_count": lane_count,
            "non_data_axes": tuple(
                axis
                for axis in range(len(model["SlotsGlobalShape"]))
                if axis not in (model["SeqAxis"], model["HeadAxis"], model["DimAxis"])
            ),
            "slots_access": _tensor_access(
                context["tensor_coordinates"],
                model["SlotsStrides"],
                context["lane_coordinates"],
                context["lane_shape"],
            ),
            "slots_lane_count": slots_lane_count,
            "source_dim_block": context["tensor_coordinates"][model["DimAxis"]],
            "source_lane_id": source_lane_id,
            "source_tensor_coordinates": context["tensor_coordinates"],
            "global_source_tensor_coordinates": global_source_tensor_coordinates,
            "topology_match_axes": topology_match_axes,
            "vectorized_dim": vectorized_dim,
        }
    )
    return context
