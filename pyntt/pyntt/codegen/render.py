"""Render generated PyNTT Triton kernels from a nncase codegen manifest."""

from __future__ import annotations

import ast
import importlib
import json
import re
import sys
from math import gcd
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
TMA_DTYPE_ITEM_SIZES = {
    "uint8": 1,
    "int8": 1,
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
            "preserve_helper_call_boundaries",
            "helpers",
            "body_source",
            "parameter_overrides",
            "extra_parameters",
            "extra_parameter_arguments",
        },
    )
    _require_string(device_function["name"], f"{path}.name", nonempty=True)
    for field in ("noinline", "preserve_helper_call_boundaries"):
        if not isinstance(device_function[field], bool):
            raise ValueError(f"{path}.{field} must be a boolean.")
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
        _require_string(
            descriptor["scalar_dtype"],
            f"{descriptor_path}.scalar_dtype",
            nonempty=True,
        )
        logical_shape = _require_positive_int_list(
            descriptor["logical_shape"], f"{descriptor_path}.logical_shape"
        )
        logical_strides = _require_positive_int_list(
            descriptor["logical_strides"], f"{descriptor_path}.logical_strides"
        )
        if len(logical_shape) != len(logical_strides):
            raise ValueError(
                f"{descriptor_path} logical shape/stride ranks differ: "
                f"{len(logical_shape)} and {len(logical_strides)}."
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
    if template == "triton/kernels/matmul/simt_fma_smem_pipeline.py.jinja":
        descriptor_names = (model.get("RhsDescriptorName"),)
        descriptor_specs = (_packed_gemv_host_descriptor_spec,)
    elif (
        template
        == "triton/kernels/qkv_parallel_linear/simt_fma_smem_pipeline.py.jinja"
    ):
        descriptor_names = tuple(
            model.get(f"{prefix}WeightDescriptorName")
            for prefix in ("Q", "K", "V")
        )
        descriptor_specs = tuple(
            (
                lambda current_model, backing, current_prefix=prefix:
                _packed_qkv_gemv_host_descriptor_spec(
                    current_model, backing, current_prefix
                )
            )
            for prefix in ("Q", "K", "V")
        )
    elif (
        template
        == "triton/kernels/matmul_glu/simt_fma_smem_pipeline.py.jinja"
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
    strides = tuple(int(value) for value in backing["logical_strides"])
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
        model, "triton.matmul", "simt_fma_smem_pipeline"
    )
    return _k_major_gemv_host_descriptor_spec(
        model,
        backing,
        block_n=microkernel["parameters"]["block_n"],
        block_k=microkernel["parameters"]["block_k"],
        pointer=model["Rhs"],
    )


def _packed_qkv_gemv_host_descriptor_spec(
    model: dict[str, Any],
    backing: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    microkernel = _microkernel_context(
        model, "triton.qkv_parallel_linear", "simt_fma_smem_pipeline"
    )
    block_n = microkernel["parameters"]["block_n"]
    projection_ns = _packed_qkv_fixed_projection_ns(model)
    if prefix not in projection_ns:
        raise ValueError(f"Unknown PyNTT QKV projection prefix {prefix!r}.")
    descriptor_block_ns, _ = _packed_qkv_transfer_plan(block_n, projection_ns)
    descriptor_block_n = descriptor_block_ns[prefix]
    spec = _k_major_gemv_host_descriptor_spec(
        model,
        backing,
        block_n=descriptor_block_n,
        block_k=microkernel["parameters"]["block_k"],
        pointer=model[f"{prefix}Weight"],
        transpose_kn=True,
    )
    return spec


def _packed_matmul_glu_gemv_host_descriptor_spec(
    model: dict[str, Any],
    backing: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    microkernel = _microkernel_context(
        model, "triton.matmul_glu", "simt_fma_smem_pipeline"
    )
    return _k_major_gemv_host_descriptor_spec(
        model,
        backing,
        block_n=microkernel["parameters"]["block_n"],
        block_k=microkernel["parameters"]["block_k"],
        pointer=model[f"{prefix}Weight"],
    )


def _packed_qkv_fixed_projection_ns(model: dict[str, Any]) -> dict[str, int]:
    projection_ns: dict[str, int] = {}
    for prefix in ("Q", "K", "V"):
        projection_n = _fixed(_packed_qkv_logical_output_shape(model, prefix)[-1])
        if projection_n is None or projection_n <= 0:
            raise ValueError(
                "PyNTT packed QKV GEMV requires fixed positive "
                f"{prefix} projection N."
            )
        projection_ns[prefix] = projection_n
    return projection_ns


def _packed_qkv_copy_n(block_n: int, projection_ns: dict[str, int]) -> int:
    copy_n = block_n
    for projection_n in projection_ns.values():
        copy_n = gcd(copy_n, projection_n)
    if copy_n <= 0:
        raise ValueError(
            f"PyNTT packed QKV GEMV derived invalid TMA copy N extent {copy_n}."
        )
    return copy_n


def _packed_qkv_transfer_plan(
    block_n: int,
    projection_ns: dict[str, int],
) -> tuple[dict[str, int], tuple[tuple[dict[str, int | str], ...], ...]]:
    """Plan exact per-projection TMA copies over the concatenated Q/K/V N stream."""

    prefixes = ("Q", "K", "V")
    projection_starts: dict[str, int] = {}
    total_n = 0
    for prefix in prefixes:
        projection_starts[prefix] = total_n
        total_n += projection_ns[prefix]

    candidate_block_ns = {
        prefix: gcd(block_n, projection_ns[prefix]) for prefix in prefixes
    }
    if total_n % block_n == 0:
        candidate_tiles: list[tuple[dict[str, int | str], ...]] = []
        candidate_is_exact = True
        for tile_start in range(0, total_n, block_n):
            tile_end = tile_start + block_n
            position = tile_start
            copies: list[dict[str, int | str]] = []
            while position < tile_end:
                prefix = next(
                    (
                        value
                        for value in prefixes
                        if projection_starts[value]
                        <= position
                        < projection_starts[value] + projection_ns[value]
                    ),
                    None,
                )
                if prefix is None:
                    candidate_is_exact = False
                    break
                copy_n = candidate_block_ns[prefix]
                projection_end = projection_starts[prefix] + projection_ns[prefix]
                if position + copy_n > min(tile_end, projection_end):
                    candidate_is_exact = False
                    break
                copies.append(
                    {
                        "prefix": prefix,
                        "tile_offset": position - tile_start,
                        "projection_offset": position - projection_starts[prefix],
                        "copy_n": copy_n,
                    }
                )
                position += copy_n
            if not candidate_is_exact:
                break
            candidate_tiles.append(tuple(copies))
        if candidate_is_exact:
            return candidate_block_ns, tuple(candidate_tiles)

    common_copy_n = _packed_qkv_copy_n(block_n, projection_ns)
    common_block_ns = {prefix: common_copy_n for prefix in prefixes}
    padded_total_n = ((total_n + block_n - 1) // block_n) * block_n
    fallback_tiles: list[tuple[dict[str, int | str], ...]] = []
    for tile_start in range(0, padded_total_n, block_n):
        copies = []
        for tile_offset in range(0, block_n, common_copy_n):
            position = tile_start + tile_offset
            prefix = next(
                (
                    value
                    for value in prefixes
                    if projection_starts[value]
                    <= position
                    < projection_starts[value] + projection_ns[value]
                ),
                "V",
            )
            projection_offset = position - projection_starts[prefix]
            if position < total_n and (
                projection_offset < 0
                or projection_offset + common_copy_n > projection_ns[prefix]
            ):
                raise ValueError(
                    "PyNTT packed QKV GEMV common TMA copy crosses a projection boundary."
                )
            copies.append(
                {
                    "prefix": prefix,
                    "tile_offset": tile_offset,
                    "projection_offset": projection_offset,
                    "copy_n": common_copy_n,
                }
            )
        fallback_tiles.append(tuple(copies))
    return common_block_ns, tuple(fallback_tiles)


def _k_major_gemv_host_descriptor_spec(
    model: dict[str, Any],
    backing: dict[str, Any],
    *,
    block_n: int,
    block_k: int,
    pointer: dict[str, Any],
    transpose_kn: bool = False,
) -> dict[str, Any]:
    logical_shape = tuple(int(value) for value in backing["logical_shape"])
    logical_strides = tuple(int(value) for value in backing["logical_strides"])
    vector_lane_shape = tuple(int(value) for value in backing["vector_lane_shape"])
    if len(logical_shape) != 2 or len(logical_strides) != 2:
        raise ValueError(
            "PyNTT packed GEMV host descriptor requires a rank-2 logical RHS "
            f"backing, got ranks {len(logical_shape)}/{len(logical_strides)}."
        )
    if vector_lane_shape != (8, 2, 8):
        raise ValueError(
            "PyNTT packed GEMV host descriptor requires vector lane shape "
            f"(8, 2, 8), got {vector_lane_shape}."
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
    k_plan = _tma_local_axis_plan(
        pointer,
        0,
        tile_extent=packed_k_outer,
        context="packed GEMV descriptor K",
    )
    n_plan = _tma_local_axis_plan(
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
            axis: int, entry: dict[str, Any]
        ) -> tuple[tuple[int, ...], tuple[int, ...]]:
            return (
                tuple(int(value) for value in entry["descriptor_shape"]),
                tuple(
                    logical_strides[axis]
                    * scalar_lanes_per_logical_element
                    * int(value)
                    for value in entry["stride_multipliers"]
                ),
            )

        k_group = axis_group(0, k_entry)
        n_group = axis_group(1, n_entry)
        ordered_groups = (n_group, k_group) if transpose_kn else (k_group, n_group)
        descriptor_shape = tuple(
            value for group in ordered_groups for value in group[0]
        ) + (k_pack, descriptor_contiguous_extent)
        descriptor_strides = tuple(
            value for group in ordered_groups for value in group[1]
        ) + (contiguous_extent, 1)
        base_scalar_elements = (
            k_entry["base"] * logical_strides[0]
            + n_entry["base"] * logical_strides[1]
        ) * scalar_lanes_per_logical_element
        entries.append(
            {
                "offset_bytes": int(backing["offset_bytes"])
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
        noinline=True,
        rematerialize_entry_indices=True,
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
            rematerialize_entry_indices=True,
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
    *,
    rematerialize_entry_indices: bool,
) -> str:
    helper_sources = _render_helper_sources(
        env,
        device_function.get("helpers", ()),
        noinline=bool(device_function["preserve_helper_call_boundaries"]),
        rematerialize_entry_indices=rematerialize_entry_indices,
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
    body_source = _replace_device_function_calls(
        device_function["body_source"],
        device_functions_by_name,
    )
    parts.append(
        env.get_template("triton/top_kernel.py.jinja")
        .render(
            name=device_function["name"],
            parameters=", ".join(device_parameters),
            do_not_specialize="",
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
    rematerialize_entry_indices: bool = False,
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
        model["RematerializeEntryIndices"] = bool(
            rematerialize_entry_indices
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
    model["RematerializeEntryIndices"] = False
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
        matmul_context=_matmul_template_context,
        multiply_expr=_multiply_expr,
        norm_apply_context=_norm_apply_template_context,
        norm_stats_context=_norm_stats_template_context,
        paged_attention_context=_paged_attention_template_context,
        paged_attention_merge_context=_paged_attention_merge_template_context,
        paged_attention_partial_context=_paged_attention_partial_template_context,
        packed_gemv_pipeline_context=_packed_gemv_pipeline_template_context,
        packed_qkv_gemv_pipeline_context=_packed_qkv_gemv_pipeline_template_context,
        product=_product,
        qkv_rope_with_cache_context=_qkv_rope_with_cache_template_context,
        qkv_parallel_linear_context=_qkv_parallel_linear_template_context,
        reduce_context=_reduce_template_context,
        ptr=_ptr,
        pyrepr=repr,
        reshard_context=_reshard_template_context,
        rope_context=_rope_template_context,
        scatter_nd_context=_scatter_nd_template_context,
        select_block_axis=_select_block_axis,
        shape_tuple=_shape_tuple,
        softmax_context=_softmax_template_context,
        summa_context=_summa_template_context,
        tensor_copy_context=_tensor_copy_template_context,
        tensor_access=_tensor_access,
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
    physical_axes = {
        int(axis["placement_axis"])
        for axis in topology
        if axis["level"] == "b"
    }
    axis_groups: set[tuple[int, ...]] = set()
    for function in manifest.get("functions", ()):
        for kernel in function.get("render_kernels", ()):
            metadata = kernel.get("metadata", {})
            raw_groups = _attrs(metadata).get("grid_barrier_axis_groups", ())
            if not isinstance(raw_groups, (list, tuple)):
                raise ValueError("attrs.grid_barrier_axis_groups must be an array.")
            for group_index, raw_group in enumerate(raw_groups):
                if not isinstance(raw_group, (list, tuple)):
                    raise ValueError(
                        f"attrs.grid_barrier_axis_groups[{group_index}] must be an array."
                    )
                axes = tuple(
                    sorted(
                        {
                            _require_int(
                                axis,
                                f"attrs.grid_barrier_axis_groups[{group_index}]",
                                minimum=0,
                            )
                            for axis in raw_group
                        }
                    )
                )
                if not axes:
                    raise ValueError("A grid barrier axis group cannot be empty.")
                unknown = set(axes) - physical_axes
                if unknown:
                    raise ValueError(
                        f"Grid barrier axis-group axes {sorted(unknown)} are not block axes in {topology}."
                    )
                if set(axes) == physical_axes:
                    raise ValueError(
                        f"Full-mesh barrier axes {axes} must use the canonical full grid barrier."
                    )
                axis_groups.add(axes)

    groups: list[dict[str, Any]] = []
    for axes in sorted(axis_groups):
        key = "_".join(str(axis) for axis in axes)
        axis_names = tuple(
            axis["name"]
            for axis in topology
            if int(axis["placement_axis"]) in axes
        )
        groups.append(
            {
                "key": key,
                "axis_names": axis_names,
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
    packed_axis: int,
    logical_index: str,
    lane_count: int,
    coordinate_shape: str | None = None,
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
    if _fixed(strides[packed_axis]) != 1:
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
    return {
        "CoordinateShape": coordinate_shape,
        "RawScalarOffset": raw_scalar_offset,
        "ScalarOffset": scalar_offset,
        "TensorIndices": tensor_indices,
        "TensorStrides": tuple(strides),
        "ContiguousVectorAxis": packed_axis,
        "LogicalIndex": str(logical_index),
        "LaneCount": lane_count,
    }


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
    varies_on_major = (
        0 <= major_axis < len(shape)
        and not _is_fixed_one(shape[major_axis])
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
) -> str:
    """Compose ordered SplitStage mappings from a dense local coordinate."""

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
    for stage_info in reversed(stage_infos):
        shard_index = stage_info["shard_index"]
        if stage_info["distribution"] == "BlockCyclic":
            block_size = stage_info["block_size"]
            period = stage_info["period"]
            coordinate = (
                f"((({coordinate}) // {block_size}) * {period} + "
                f"({shard_index}) * {block_size} + (({coordinate}) % {block_size}))"
            )
        else:
            capacity = stage_info["capacity"]
            coordinate = f"(({shard_index}) * ({capacity}) + ({coordinate}))"
    return coordinate


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
    plan = _tma_local_axis_plan(
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
        coordinates = (
            f"(({local_coordinate}) // {block_size})",
            f"(({local_coordinate}) % {block_size})",
        )
    else:
        coordinates = (local_coordinate,)
    return {**plan, "coordinates": coordinates}


def _tma_shared_axis_coordinates(
    local_coordinate: str,
    plan: dict[str, Any],
) -> tuple[str, ...]:
    if not plan["is_block_cyclic"] or plan["block_size"] == 1:
        return (local_coordinate,)
    block_size = plan["block_size"]
    try:
        fixed_coordinate = int(local_coordinate)
    except ValueError:
        pass
    else:
        return (
            str(fixed_coordinate // block_size),
            str(fixed_coordinate % block_size),
        )
    return (
        f"(({local_coordinate}) // {block_size})",
        f"(({local_coordinate}) % {block_size})",
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
            base += shard_owner * block_size
            parent_extent = active_extent
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
            "descriptor_shape": tuple(1 for _ in descriptor_shape),
            "stride_multipliers": stride_multipliers,
        }
    return {
        **plan,
        "active": True,
        "base": base,
        "descriptor_shape": descriptor_shape,
        "stride_multipliers": stride_multipliers,
    }


def _nv_tma_swizzle_mode(block_shape: tuple[int, ...], dtype: str) -> int:
    try:
        item_size = TMA_DTYPE_ITEM_SIZES[dtype]
    except KeyError as ex:
        raise ValueError(f"PyNTT TMA does not support descriptor dtype {dtype!r}") from ex
    if len(block_shape) < 2 or _product_int(list(block_shape[:-1])) < 8:
        return 0
    contiguous_bytes = block_shape[-1] * item_size
    for width, mode in ((128, 3), (64, 2), (32, 1)):
        if contiguous_bytes >= width and contiguous_bytes % width == 0:
            return mode
    return 0


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
    shard_axes = pointer.get("ShardAxes")
    hierarchy = pointer.get("Hierarchy")
    strides = pointer.get("Strides")
    if not all(isinstance(value, list) for value in (global_shape, shard_axes, hierarchy, strides)):
        raise ValueError("PyNTT canonical-global pointer has incomplete shard metadata")
    if not (len(global_shape) == len(shard_axes) == len(strides)):
        raise ValueError(
            "PyNTT canonical-global pointer shape/mapping/stride rank mismatch: "
            f"{len(global_shape)}/{len(shard_axes)}/{len(strides)}"
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
            coordinate, global_shape[axis], shard_axes[axis], hierarchy
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
    tile_extents = (physical_tile_extent,) + tuple(str(value) for value in lanes)
    tile_shape = f"({', '.join(tile_extents)}{',' if len(tile_extents) == 1 else ''})"
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
        "tile_shape": tile_shape,
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
) -> dict[str, Any]:
    n_lane_shape = _qkv_packed_lane_shape(model, packed=packed)
    if tuple(n_lane_shape) != tuple(n_axis["lane_shape"]):
        raise ValueError(
            f"PyNTT {prefix} QKV weight lane shape does not match its N tile: "
            f"weight N={n_lane_shape}, tile={n_axis['lane_shape']}."
        )
    weight_lane_shape = _qkv_weight_lane_shape(model, packed=packed)
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
    required_offsets = {
        "simt_fma": (),
        "simt_fma_smem_pipeline": ("rhs_stage",),
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

    return {
        "family": family,
        "variant": variant,
        "parameters": parameters,
        "shared_workspace_offsets": offsets,
    }


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
            weight_mask = (
                f"({weight_n_axis['logical_coordinate']} < {_dim(n)}) & "
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
            weight_mask = (
                f"({weight_k_coordinate} < {_dim(k)}) & "
                f"({weight_n_axis['logical_coordinate']} < {_dim(n)})"
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
                "weight_structured_shape": weight_structured_shape,
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
    if layout not in ("n_major", "k_major"):
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
    if tuple(lane_shape) != tuple(n_axis["lane_shape"]):
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

    layout = _matmul_rhs_layout(model)
    if layout == "n_major":
        if not model["TransposeB"]:
            raise ValueError("PyNTT N-major packed RHS must be logically transposed.")
        matrix_coordinates = (n_axis["physical_coordinate"], k_expr)
        lane_coordinates = n_axis["lane_coordinates"]
        physical_lane_shape = lane_shape
    else:
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


def _is_positive_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _matmul_template_context(
    model: dict[str, Any],
    *,
    gemv: bool,
    variant: str | None = None,
) -> dict[str, Any]:
    """Prepare Matmul/Gemv dimensions and addresses for Jinja-owned kernels."""

    if not isinstance(model["HasAddend"], bool):
        raise ValueError("PyNTT Matmul HasAddend must be a boolean.")
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
        "triton.matmul",
        variant or ("simt_fma" if gemv else "mma"),
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
        if rhs_layout != "k_major" or model["TransposeB"]:
            raise ValueError(
                "PyNTT K-packed Matmul RHS requires a non-transposed K-major layout."
            )
        rhs_k = _multiply_dim(model["RhsShape"][-2], rhs_k_lane_count)
        rhs_n = _multiply_dim(model["RhsShape"][-1], rhs_lane_count)
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
    if rhs_lane_shape != output_lane_shape:
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
    if gemv:
        rhs_n_axis = _structured_axis_tile(
            "rhs_n",
            rhs_lane_shape,
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
            rhs_lane_shape,
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
) -> None:
    """Validate the compiler-selected packed GEMV staging contract."""

    minimum_physical_stages = 2 * rhs_tiles_per_group
    if (
        not _is_positive_power_of_two(block_n)
        or not _is_positive_power_of_two(block_k)
        or not 8 <= block_n <= 64
        or block_n % 8 != 0
        or not 128 <= block_k <= 1024
        or num_stages < minimum_physical_stages
        or num_stages % rhs_tiles_per_group != 0
    ):
        raise ValueError(
            f"PyNTT {algorithm} resource contract requires a power-of-two "
            "block_n in [8, 64], a power-of-two block_k in [128, 1024], "
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


def _packed_gemv_pipeline_template_context(
    model: dict[str, Any],
) -> dict[str, Any]:
    """Prepare the packed BF16, shared-staged SIMT GEMV algorithm."""

    context = _matmul_template_context(
        model,
        gemv=True,
        variant="simt_fma_smem_pipeline",
    )
    if model["LhsDType"] != "bfloat16" or model["RhsDType"] != "bfloat16":
        raise ValueError(
            "PyNTT packed GEMV pipeline requires BF16 lhs and rhs operands."
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
        or int(model["RhsKVectorLaneCount"]) != 8
    ):
        raise ValueError(
            "PyNTT K-major BF16 GEMV pipeline requires "
            "rhs=[K/16,N/8]<8,2,8> and output=[M,N/8]<8>."
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
    if (
        _fixed(model["RhsShape"][-2]) != k // k_atom
        or _fixed(model["RhsStrides"][-1]) != 1
        or (_min_value(model["RhsStrides"][-2]) or 0) <= 0
    ):
        raise ValueError(
            "PyNTT packed GEMV TMA requires a positive-stride "
            "physical RHS [K/16,N/8]<8,2,8> view."
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
    rhs_pointer = model.get("Rhs")
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
    reduction_group = 32
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
            packed_axis=1,
            logical_index="pipeline_output_n",
            lane_count=n_lane,
            coordinate_shape=_coordinate_shape((block_n,)),
        ),
        "pipeline_addend_access": (
            _contiguous_vector_axis_access(
                ("0", "0"),
                model["AddendStrides"],
                packed_axis=1,
                logical_index="pipeline_output_n",
                lane_count=n_lane,
                coordinate_shape=_coordinate_shape((block_n,)),
            )
            if model["HasAddend"]
            else None
        ),
        "pipeline_output_mask": "(pipeline_output_n < active_n)",
    }
    context.update(pipeline_context)
    return context


def _packed_qkv_gemv_pipeline_template_context(
    model: dict[str, Any],
) -> dict[str, Any]:
    """Prepare one K-major BF16 pipeline over the concatenated Q/K/V N axis."""

    context = _qkv_parallel_linear_template_context(
        model,
        packed=True,
        variant="simt_fma_smem_pipeline",
    )
    if model["InputDType"] != "bfloat16" or model["WeightDType"] != "bfloat16":
        raise ValueError(
            "PyNTT packed QKV GEMV pipeline requires BF16 input and weights."
        )
    if model.get("RhsLayout") != "k_major":
        raise ValueError(
            "PyNTT packed QKV GEMV pipeline requires K-major packed weights."
        )
    if (
        int(model["NPackedLaneCount"]) != 1
        or int(model["NVectorLaneCount"]) != 8
        or int(model["KPackLaneCount"]) != 2
        or int(model["KVectorLaneCount"]) != 8
    ):
        raise ValueError(
            "PyNTT packed QKV GEMV pipeline requires "
            "weight=[K/16,N/8]<8,2,8> and output=[M,N/8]<8>."
        )
    if len(model["InputShape"]) != 2 or any(
        len(model[f"{prefix}{suffix}Shape"]) != 2
        for prefix in ("Q", "K", "V")
        for suffix in ("Weight", "Output")
    ):
        raise ValueError(
            "PyNTT packed QKV GEMV pipeline currently requires rank-2 operands."
        )

    microkernel = context["microkernel"]
    block_n = microkernel["parameters"]["block_n"]
    block_k = microkernel["parameters"]["block_k"]
    num_stages = microkernel["parameters"]["num_stages"]
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
            "PyNTT packed QKV GEMV pipeline tile is incompatible with its "
            f"packed lanes: block_n={block_n}, block_k={block_k}, "
            f"n_lane={n_lane}, k_atom={k_atom}."
        )
    k = _fixed(context["k"])
    if k is None or k <= 0 or k % block_k != 0:
        raise ValueError(
            "PyNTT packed QKV GEMV pipeline requires a fixed K divisible by block_k."
        )
    packed_k_outer = block_k // k_atom
    packed_n_outer = block_n // n_lane
    for prefix in ("Q", "K", "V"):
        descriptor_name = model.get(f"{prefix}WeightDescriptorName")
        descriptor_origin_elements = model.get(
            f"{prefix}WeightDescriptorOriginElements"
        )
        global_offsets = model.get(f"{prefix}WeightGlobalOffsets")
        weight_shape = model[f"{prefix}WeightShape"]
        if not isinstance(descriptor_name, str) or not descriptor_name:
            raise ValueError(
                f"PyNTT packed QKV GEMV pipeline requires a host {prefix} weight descriptor."
            )
        if not isinstance(descriptor_origin_elements, str) or not descriptor_origin_elements:
            raise ValueError(
                f"PyNTT packed QKV GEMV pipeline requires a {prefix} "
                "weight descriptor origin."
            )
        if not isinstance(global_offsets, list) or len(global_offsets) != 2:
            raise ValueError(
                f"PyNTT packed QKV GEMV pipeline requires two {prefix} weight global offsets."
            )
        if (
            _fixed(weight_shape[-2]) != k // k_atom
            or _fixed(model[f"{prefix}WeightStrides"][-1]) != 1
            or (_min_value(model[f"{prefix}WeightStrides"][-2]) or 0) <= 0
        ):
            raise ValueError(
                f"PyNTT packed QKV GEMV TMA requires a positive-stride "
                f"{prefix} weight [K/16,N/8]<8,2,8> view."
            )

    projection_ns = _packed_qkv_fixed_projection_ns(model)
    if any(value % n_lane != 0 for value in projection_ns.values()):
        raise ValueError(
            "PyNTT packed QKV GEMV projection N extents must be divisible by NVector."
        )

    projection_starts: dict[str, int] = {}
    total_n = 0
    for prefix in ("Q", "K", "V"):
        projection_starts[prefix] = total_n
        total_n += projection_ns[prefix]
    num_n_tiles = (total_n + block_n - 1) // block_n
    num_k_tiles = k // block_k
    max_sequence_count = num_n_tiles * num_k_tiles
    if max_sequence_count > (2**31 - 1):
        raise ValueError(
            "PyNTT packed QKV GEMV sequence exceeds the signed int32 pipe ABI: "
            f"{max_sequence_count}."
        )

    descriptor_block_ns, transfer_plan = _packed_qkv_transfer_plan(
        block_n, projection_ns
    )
    if any(copy_n % n_lane != 0 for copy_n in descriptor_block_ns.values()):
        raise ValueError(
            "PyNTT packed QKV GEMV cannot form NVector-aligned per-projection "
            f"TMA copy extents from block_n={block_n} and projections={projection_ns}."
        )
    projection_by_prefix = {
        projection["prefix"]: projection for projection in context["projections"]
    }
    projections: list[dict[str, Any]] = []
    for prefix in ("Q", "K", "V"):
        lower = prefix.lower()
        projection_start = projection_starts[prefix]
        projection_end = projection_start + projection_ns[prefix]
        projection_n_expr = f"{lower}_projection_n"
        projection_mask = (
            f"({projection_n_expr} >= 0) & "
            f"({projection_n_expr} < {projection_ns[prefix]})"
        )
        output_access = _contiguous_vector_axis_access(
            ("0", "0"),
            model[f"{prefix}OutputStrides"],
            packed_axis=1,
            logical_index=projection_n_expr,
            lane_count=n_lane,
            coordinate_shape=_coordinate_shape((block_n,)),
        )
        has_bias = projection_by_prefix[prefix]["has_bias"]
        bias_access = (
            _contiguous_vector_axis_access(
                ("0",),
                model[f"{prefix}BiasStrides"],
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
                "descriptor_name": model[f"{prefix}WeightDescriptorName"],
                "descriptor_origin_elements": model[
                    f"{prefix}WeightDescriptorOriginElements"
                ],
                "weight_pointer": model[f"{prefix}Weight"],
                "weight_global_offsets": model[f"{prefix}WeightGlobalOffsets"],
                "projection_start": projection_start,
                "projection_end": projection_end,
                "projection_n_expr": projection_n_expr,
                "projection_mask": projection_mask,
                "has_bias": has_bias,
                "bias_access": bias_access,
                "output_access": output_access,
            }
        )

    projection_metadata = {
        projection["prefix"]: projection for projection in projections
    }
    shared_k_plans = [
        _tma_local_axis_plan(
            projection["weight_pointer"],
            -2,
            tile_extent=packed_k_outer,
            context=f"packed QKV {projection['prefix']} shared K",
        )
        for projection in projections
    ]
    shared_n_plans = [
        _tma_local_axis_plan(
            projection["weight_pointer"],
            -1,
            tile_extent=packed_n_outer,
            context=f"packed QKV {projection['prefix']} shared N",
        )
        for projection in projections
    ]

    def shared_axis_signature(plan: dict[str, Any]) -> tuple[Any, ...]:
        return (
            plan["is_block_cyclic"],
            plan["block_size"],
            tuple(plan["block_shape"]),
        )

    if len({shared_axis_signature(plan) for plan in shared_k_plans}) != 1 or len(
        {shared_axis_signature(plan) for plan in shared_n_plans}
    ) != 1:
        raise ValueError(
            "PyNTT packed QKV GEMV requires Q/K/V weights to use one common "
            "staged split layout for each K/N axis."
        )
    shared_k_plan = shared_k_plans[0]
    shared_n_plan = shared_n_plans[0]
    shared_stage_shape = (
        tuple(shared_n_plan["block_shape"])
        + tuple(shared_k_plan["block_shape"])
        + (k_pack, n_lane * k_lane)
    )
    if len(shared_stage_shape) > 5:
        raise ValueError(
            "PyNTT packed QKV GEMV shared stage exceeds the hardware rank-5 "
            f"TMA limit: shape={shared_stage_shape}."
        )
    transfer_tiles = []
    for tile_index, copies in enumerate(transfer_plan):
        transfer_copies = []
        for copy in copies:
            prefix = str(copy["prefix"])
            copy_n = int(copy["copy_n"])
            copy_n_outer = copy_n // n_lane
            projection = projection_metadata[prefix]
            projection_n_outer_offset = int(copy["projection_offset"]) // n_lane
            k_transfer = _tma_local_axis_transfer(
                projection["weight_pointer"],
                -2,
                projection["weight_global_offsets"][-2],
                local_offset=0,
                tile_index="k_tile",
                tile_stride=packed_k_outer,
                tile_extent=packed_k_outer,
                context=f"packed QKV {prefix} weight K",
            )
            n_transfer = _tma_local_axis_transfer(
                projection["weight_pointer"],
                -1,
                projection["weight_global_offsets"][-1],
                local_offset=projection_n_outer_offset,
                tile_index=0,
                tile_stride=copy_n_outer,
                tile_extent=copy_n_outer,
                context=f"packed QKV {prefix} weight N",
            )
            copy_shape = (
                tuple(n_transfer["block_shape"])
                + tuple(k_transfer["block_shape"])
                + (k_pack, n_lane * k_lane)
            )
            destination_offsets = (
                _tma_shared_axis_coordinates(
                    str(int(copy["tile_offset"]) // n_lane), shared_n_plan
                )
                + _tma_shared_axis_coordinates("0", shared_k_plan)
                + ("0", "0")
            )
            if len(copy_shape) != len(shared_stage_shape):
                raise ValueError(
                    "PyNTT packed QKV GEMV descriptor/shared rank mismatch: "
                    f"copy={copy_shape}, shared={shared_stage_shape}."
                )
            transfer_copies.append(
                {
                    "prefix": prefix,
                    "lower": projection["lower"],
                    "descriptor_name": projection["descriptor_name"],
                    "descriptor_origin_elements": projection[
                        "descriptor_origin_elements"
                    ],
                    "descriptor_offsets": (
                        tuple(n_transfer["coordinates"])
                        + tuple(k_transfer["coordinates"])
                        + (
                            "0",
                            projection["descriptor_origin_elements"],
                        )
                    ),
                    "destination_offsets": destination_offsets,
                    "copy_shape": copy_shape,
                    "copy_is_full_stage": (
                        copy_shape == shared_stage_shape
                        and all(offset == "0" for offset in destination_offsets)
                    ),
                }
            )
        transfer_tiles.append(
            {"tile_index": tile_index, "copies": tuple(transfer_copies)}
        )

    reduction_group = 32
    target_worker_width = int(model["TargetWorkerWidth"])
    consumer_warps = int(model["NumWarps"])
    consumer_layout = _packed_gemv_consumer_layout(
        block_n=block_n,
        reduction_group=reduction_group,
        consumer_warps=consumer_warps,
        target_worker_width=target_worker_width,
    )
    context.update(
        block_n=block_n,
        block_k=block_k,
        num_stages=num_stages,
        num_k_tiles=num_k_tiles,
        num_n_tiles=num_n_tiles,
        projections=tuple(projections),
        transfer_tiles=tuple(transfer_tiles),
        k_atom=k_atom,
        packed_k_outer=packed_k_outer,
        reduction_group=reduction_group,
        reduction_groups_per_stage=block_k // reduction_group,
        shared_stage_shape=shared_stage_shape,
        shared_weight_indices=(
            _tma_shared_axis_coordinates(
                f"local_n // {n_lane}", shared_n_plan
            )
            + _tma_shared_axis_coordinates(
                f"shared_k // {k_atom}", shared_k_plan
            )
            + (
                f"shared_payload // {n_lane * k_lane}",
                f"shared_payload % {n_lane * k_lane}",
            )
        ),
        tma_contiguous_extent=n_lane * k_lane,
        consumer_size_per_thread=consumer_layout["size_per_thread"],
        consumer_threads_per_warp=consumer_layout["threads_per_warp"],
        consumer_warps_per_cta=consumer_layout["warps_per_cta"],
        consumer_weight_layout_name=f"{model['FunctionName']}__weight_layout",
        consumer_lhs_layout_name=f"{model['FunctionName']}__lhs_layout",
        consumer_output_layout_name=f"{model['FunctionName']}__output_layout",
        pipeline_input_access=_qkv_input_access(
            model,
            output_batch_rank=0,
            m_expr="0",
            k_expr="pipeline_offs_k",
            coordinate_shape=_coordinate_shape((reduction_group,)),
        ),
    )
    return context


def _packed_matmul_glu_gemv_pipeline_template_context(
    model: dict[str, Any],
) -> dict[str, Any]:
    """Prepare one K-major BF16 pipeline for paired gate/up projections."""

    context = _matmul_glu_template_context(
        model,
        packed=True,
        variant="simt_fma_smem_pipeline",
    )
    if model["InputDType"] != "bfloat16" or model["WeightDType"] != "bfloat16":
        raise ValueError(
            "PyNTT packed MatMulGlu GEMV pipeline requires BF16 input and weights."
        )
    if model.get("RhsLayout") != "k_major":
        raise ValueError(
            "PyNTT packed MatMulGlu GEMV pipeline requires K-major packed weights."
        )
    if (
        int(model["NPackedLaneCount"]) != 1
        or int(model["NVectorLaneCount"]) != 8
        or int(model["KPackLaneCount"]) != 2
        or int(model["KVectorLaneCount"]) != 8
    ):
        raise ValueError(
            "PyNTT packed MatMulGlu GEMV pipeline requires "
            "weight=[K/16,N/8]<8,2,8> and output=[M,N/8]<8>."
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
                f"{prefix.lower()} weight [K/16,N/8]<8,2,8> view."
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

    reduction_group = 32
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

    context.update(
        block_n=block_n,
        block_k=block_k,
        num_stages=num_stages,
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
            packed_axis=1,
            logical_index="pipeline_output_n",
            lane_count=n_lane,
            coordinate_shape=_coordinate_shape((block_n,)),
        ),
        pipeline_output_mask="pipeline_output_n < active_n",
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


def _softmax_template_context(model: dict[str, Any]) -> dict[str, Any]:
    """Prepare Softmax's independent-slice and storage index expressions."""

    rank = len(model["Shape"])
    block_axis = _select_block_axis(model["Shape"], model["OutputStrides"])

    def axis_index(axis: int) -> str:
        return "lane" if axis == block_axis else f"idx{axis}"

    def offset(strides: list[Any]) -> str:
        terms = [f"{axis_index(axis)} * {_dim(strides[axis])}" for axis in range(rank)]
        return "lane * 0" if not terms else "lane * 0 + " + " + ".join(terms)

    slice_terms = [
        f"{axis_index(axis)} * {_dim(model['InputStrides'][axis])}"
        for axis in range(rank)
        if axis != model["Axis"]
    ]
    return {
        "axis_extent": model["Shape"][model["Axis"]],
        "block_extent": model["Shape"][block_axis],
        "input_offset": offset(model["InputStrides"]),
        "loop_axes": tuple(axis for axis in range(rank) if axis != block_axis),
        "output_offset": offset(model["OutputStrides"]),
        "slice_base": (
            "lane * 0" if not slice_terms else "lane * 0 + " + " + ".join(slice_terms)
        ),
        "slice_offset": (
            f"slice_base + axis_pos * {_dim(model['InputStrides'][model['Axis']])}"
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

    def stats_access(component: int) -> dict[str, Any]:
        coordinates = (str(component),) + tuple(
            f"outer_idx{index}" if index < axis else "0" for index in range(rank)
        )
        return _tensor_access(coordinates, model["OutputStrides"])

    reduction = "value0"
    square_reduction = "value0 * value0"
    for _ in range(1 + len(context["lane_shape"])):
        reduction = f"tl.sum({reduction}, axis=0)"
        square_reduction = f"tl.sum({square_reduction}, axis=0)"
    context.update(
        {
            "logical_input_shape": _logical_shape(
                model["InputShape"], model["InputVectorLaneCount"]
            ),
            "outer_axes": outer_axes,
            "prefix_depth": len(outer_axes),
            "reduction": reduction,
            "square_reduction": square_reduction,
            "stats_accesses": (stats_access(0), stats_access(1)),
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
            "normalization_size": _product(logical_input_global_shape[axis:]),
            "outer_axes": outer_axes,
            "prefix_depth": len(outer_axes),
            "stats_accesses": (stats_access(0), stats_access(1)),
        }
    )
    return context


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
    reduction = "norm_value"
    square_reduction = "norm_value * norm_value"
    for _ in range(1 + len(reduction_context["lane_shape"])):
        reduction = f"tl.sum({reduction}, axis=0)"
        square_reduction = f"tl.sum({square_reduction}, axis=0)"

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
    return {"q": q_context, "k": k_context, "v": v_cache_context}


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
    writer_active = "True"
    for axis in sorted(input_partial_mesh_axes):
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
    if context["lane_count"] != model["VectorLaneCount"]:
        raise ValueError(
            "PyNTT Reshard vector lane metadata is inconsistent: "
            f"shape={context['lane_shape']}, count={model['VectorLaneCount']}"
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
        partial = {
            "accumulator_dtype": accumulator_dtype,
            "address": partial_input_address,
            "axes": tuple(sorted(input_partial_mesh_axes)),
            "pointer_type": _pointer_type(
                model["TritonDType"], partial_input_address["AddressSpace"]
            ),
            "source_pool_index": _pool_index_expression(
                "source_shard_index", partial_input_address["PoolScopeSize"]
            ),
            "source_shard_index": source_shard_index,
            "zero": zero,
        }
    context.update(
        {
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
            "prefix_depth": len(output_broadcast_mesh_axes),
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

    def global_index_expression(axis: int, local_index: str, global_extent: Any) -> str:
        return _local_to_global_coordinate(
            local_index,
            global_extent,
            model["OutputShardAxes"][axis],
            model["Hierarchy"],
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
    global_q_head_begin = global_index_expression(
        model["HeadAxis"], "0", global_num_query_heads
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

    ctx = _paged_attention_template_context(model)
    query_rank = len(model["QueryShape"])
    state_rank = query_rank + 1
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
        indices[-1] = "0"
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
    state_rank = output_rank + 1
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
    if output_physical_dim is None or output_physical_dim * output_lane_count != head_dimension:
        raise ValueError(
            "PyNTT PagedAttentionMerge output HeadDim does not match its vector lanes"
        )

    output_dim_axis = _structured_axis_tile(
        "output_dim",
        output_lanes,
        head_dimension,
        head_dimension,
    )

    def global_index_expression(axis: int, local_index: str, global_extent: Any) -> str:
        return _local_to_global_coordinate(
            local_index,
            global_extent,
            model["OutputShardAxes"][axis],
            model["Hierarchy"],
        )

    output_indices = ["0"] * output_rank
    output_indices[model["SeqAxis"]] = "local_query_id"
    output_indices[model["HeadAxis"]] = "q_head"
    output_indices[dim_axis] = output_dim_axis["physical_coordinate"]

    def state_indices(dim_index: str, split_index: str) -> list[str]:
        indices = ["0"] * state_rank
        indices[model["SeqAxis"]] = "local_query_id"
        indices[model["HeadAxis"]] = "q_head"
        indices[dim_axis] = dim_index
        indices[-1] = split_index
        return indices

    global_query_tokens = model["OutputGlobalShape"][model["SeqAxis"]]
    global_query_id = global_index_expression(
        model["SeqAxis"], "local_query_id", global_query_tokens
    )
    global_q_head = global_index_expression(
        model["HeadAxis"], "q_head", model["GlobalNumQueryHeads"]
    )
    return {
        "acc_state_access": _tensor_access(
            state_indices("dim_grid", "split_grid"),
            model["AccStateStrides"],
            coordinate_shape=f"({split_tile}, {head_dimension})",
        ),
        "global_q_head": global_q_head,
        "global_query_id": global_query_id,
        "local_q_heads": model["OutputShape"][model["HeadAxis"]],
        "local_query_tokens": model["OutputShape"][model["SeqAxis"]],
        "max_state_access": _tensor_access(
            state_indices("0", "split_offsets"),
            model["MaxStateStrides"],
            coordinate_shape=f"({split_tile},)",
        ),
        "output_access": _tensor_access(
            output_indices,
            model["OutputStrides"],
            output_dim_axis["lane_coordinates"],
            output_lanes,
            _coordinate_shape(output_dim_axis["structured_shape"]),
        ),
        "output_active": (
            f"({global_query_id} < {_dim(global_query_tokens)}) & "
            f"({global_q_head} < {int(model['GlobalNumQueryHeads'])})"
        ),
        "output_dim_axis": output_dim_axis,
        "output_structured_shape": output_dim_axis["structured_shape"],
        "owner": f"shard_coord{split_axis} == 0",
        "split_count": split_count,
        "split_tile": split_tile,
        "sum_state_access": _tensor_access(
            state_indices("0", "split_offsets"),
            model["SumStateStrides"],
            coordinate_shape=f"({split_tile},)",
        ),
    }


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
