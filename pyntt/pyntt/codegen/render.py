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

SHARD_INDEX_PARAMETER = "shard_index"
PYNTT_CODEGEN_MANIFEST_VERSION = 8
THROUGHPUT_BLOCK_SIZE_CANDIDATES = (128, 256, 512, 1024)
WARP_SPECIALIZATION_PRODUCER_WARPS = 1
WARP_SPECIALIZATION_WORKER_ALLOCATION_WARPS = 4

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
        _require_positive_int_list(
            descriptor["vector_lane_shape"],
            f"{descriptor_path}.vector_lane_shape",
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


def _validate_codegen_manifest_v8(manifest: dict[str, Any]) -> None:
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
    _validate_codegen_manifest_v8(manifest)


def render_manifest(manifest: dict[str, Any]) -> str:
    validate_manifest(manifest)

    render_kernels = [
        kernel
        for function in manifest.get("functions", ())
        for kernel in function.get("render_kernels", ())
    ]
    kernels = [_render_kernel(kernel) for kernel in render_kernels]
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
    needs_grid_barrier = any(
        _attrs(kernel.get("metadata", {})).get("requires_grid_barrier")
        for function in manifest.get("functions", ())
        for kernel in function.get("render_kernels", ())
    )
    env = _make_env()
    return env.get_template("triton/module.py.jinja").render(
        kernels=kernels,
        kernel_configs=kernel_configs,
        host_tensor_descriptor_specs=host_tensor_descriptor_specs,
        needs_grid_barrier=needs_grid_barrier,
        grid_mesh_size=_grid_mesh_size(manifest),
    )


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
    worker_registers = (
        uniform_register_limit // register_granularity
    ) * register_granularity
    if worker_registers <= 0:
        raise ValueError(
            f"PyNTT kernel {metadata['name']} has insufficient register capacity "
            "for its fixed producer/consumer execution geometry."
        )
    registers_per_thread_limit = _require_int(
        attrs.get("registers_per_thread_limit"),
        f"PyNTT kernel {metadata['name']} attrs.registers_per_thread_limit",
        minimum=1,
    )
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
        "producer_warps": WARP_SPECIALIZATION_PRODUCER_WARPS,
        "worker_registers": worker_registers,
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
        descriptor_specs = (
            _packed_matmul_glu_gemv_host_descriptor_spec,
            _packed_matmul_glu_gemv_host_descriptor_spec,
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
    descriptor_block_n = _packed_qkv_copy_n(block_n, projection_ns)
    spec = _k_major_gemv_host_descriptor_spec(
        model,
        backing,
        block_n=descriptor_block_n,
        block_k=microkernel["parameters"]["block_k"],
    )
    k_outer, n_outer, k_pack, contiguous_extent = spec["shape"]
    k_stride, n_stride, k_pack_stride, contiguous_stride = spec["strides"]
    block_k_outer, block_n_outer, block_k_pack, block_contiguous = spec[
        "block_shape"
    ]
    return {
        **spec,
        "shape": (n_outer, k_outer, k_pack, contiguous_extent),
        "strides": (
            n_stride,
            k_stride,
            k_pack_stride,
            contiguous_stride,
        ),
        "block_shape": (
            block_n_outer,
            block_k_outer,
            block_k_pack,
            block_contiguous,
        ),
    }


def _packed_matmul_glu_gemv_host_descriptor_spec(
    model: dict[str, Any],
    backing: dict[str, Any],
) -> dict[str, Any]:
    microkernel = _microkernel_context(
        model, "triton.matmul_glu", "simt_fma_smem_pipeline"
    )
    return _k_major_gemv_host_descriptor_spec(
        model,
        backing,
        block_n=microkernel["parameters"]["block_n"],
        block_k=microkernel["parameters"]["block_k"],
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


def _k_major_gemv_host_descriptor_spec(
    model: dict[str, Any],
    backing: dict[str, Any],
    *,
    block_n: int,
    block_k: int,
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
    return {
        "name": backing["name"],
        "source": backing["source"],
        "offset_bytes": int(backing["offset_bytes"]),
        "dtype": backing["scalar_dtype"],
        "shape": (
            logical_shape[0],
            logical_shape[1],
            k_pack,
            contiguous_extent,
        ),
        "strides": (
            logical_strides[0] * scalar_lanes_per_logical_element,
            logical_strides[1] * scalar_lanes_per_logical_element,
            contiguous_extent,
            1,
        ),
        "block_shape": (
            block_k // k_atom,
            block_n // n_lane,
            k_pack,
            contiguous_extent,
        ),
        "padding": "zero",
    }


def _render_kernel(kernel: dict[str, Any]) -> str:
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
        num_warps=num_warps,
        target_worker_width=target_worker_width,
        producer_warps=backend_config["producer_warps"],
        worker_registers=backend_config["worker_registers"],
        register_granularity=backend_config["register_granularity"],
        registers_per_thread_limit=backend_config["registers_per_thread_limit"],
        device_functions_by_name=device_functions_by_name,
    )
    device_function_sources = [
        _render_device_function(
            env,
            device_function,
            device_functions_by_name,
            num_warps,
            target_worker_width,
            backend_config["producer_warps"],
            backend_config["worker_registers"],
            backend_config["register_granularity"],
            backend_config["registers_per_thread_limit"],
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
            body_source=body_source.rstrip(),
            materialize_shard_index=_needs_shard_index_prelude(
                body_source,
                parameters,
            ),
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
    worker_registers: int,
    register_granularity: int,
    registers_per_thread_limit: int,
) -> str:
    helper_sources = _render_helper_sources(
        env,
        device_function.get("helpers", ()),
        noinline=bool(device_function["preserve_helper_call_boundaries"]),
        num_warps=num_warps,
        target_worker_width=target_worker_width,
        producer_warps=producer_warps,
        worker_registers=worker_registers,
        register_granularity=register_granularity,
        registers_per_thread_limit=registers_per_thread_limit,
        device_functions_by_name=device_functions_by_name,
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
            body_source=body_source.rstrip(),
            materialize_shard_index=_needs_shard_index_prelude(
                body_source,
                device_parameters,
            ),
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


def _needs_shard_index_prelude(
    source: str,
    parameters: tuple[str, ...],
) -> bool:
    return SHARD_INDEX_PARAMETER not in _parameter_call_arguments(
        parameters
    ) and SHARD_INDEX_PARAMETER in _referenced_parameter_names(
        source, (SHARD_INDEX_PARAMETER,)
    )


def _render_helper_sources(
    env: Environment,
    helpers: Any,
    *,
    noinline: bool = False,
    num_warps: int | None = None,
    target_worker_width: int | None = None,
    producer_warps: int | None = None,
    worker_registers: int | None = None,
    register_granularity: int | None = None,
    registers_per_thread_limit: int | None = None,
    device_functions_by_name: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    helper_sources = []
    for helper in helpers:
        model = _prepare_helper_model(
            helper["model"],
            noinline=noinline,
            num_warps=num_warps,
            target_worker_width=target_worker_width,
            producer_warps=producer_warps,
            worker_registers=worker_registers,
            register_granularity=register_granularity,
            registers_per_thread_limit=registers_per_thread_limit,
            device_functions_by_name=device_functions_by_name,
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
    worker_registers: int | None,
    register_granularity: int | None,
    registers_per_thread_limit: int | None,
    device_functions_by_name: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    model = dict(raw_model)
    model["NoInline"] = bool(noinline)
    if num_warps is not None:
        model["NumWarps"] = num_warps
    if target_worker_width is not None:
        model["TargetWorkerWidth"] = target_worker_width
    if producer_warps is not None:
        model["ProducerWarps"] = producer_warps
    if worker_registers is not None:
        model["WorkerRegisters"] = worker_registers
    if register_granularity is not None:
        model["RegisterGranularity"] = register_granularity
    if registers_per_thread_limit is not None:
        model["RegistersPerThreadLimit"] = registers_per_thread_limit

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
                worker_registers=worker_registers,
                register_granularity=register_granularity,
                registers_per_thread_limit=registers_per_thread_limit,
                device_functions_by_name=device_functions_by_name,
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
    for role in ("Consumer", "Producer"):
        body_source = model.get(f"{role}BodySource")
        if isinstance(body_source, str):
            model[f"{role}NeedsShardIndex"] = _needs_shard_index_prelude(
                body_source,
                (),
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
        packed_gemv_pipeline_context=_packed_gemv_pipeline_template_context,
        packed_qkv_gemv_pipeline_context=_packed_qkv_gemv_pipeline_template_context,
        pointer_shard_hierarchy=_pointer_shard_hierarchy,
        product=_product,
        qkv_parallel_linear_context=_qkv_parallel_linear_template_context,
        reduce_context=_reduce_template_context,
        ptr=_ptr,
        pyrepr=repr,
        reverse_indices=lambda values: range(len(values) - 1, -1, -1),
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
        transpose_context=_transpose_template_context,
        update_paged_attention_kv_cache_context=(
            _update_paged_attention_kv_cache_template_context
        ),
        vector_layout_context=_vector_layout_template_context,
    )
    return env


def _grid_mesh_size(manifest: dict[str, Any]) -> int:
    sizes = set()
    for function in manifest.get("functions", ()):
        for kernel in function.get("render_kernels", ()):
            metadata = kernel.get("metadata", {})
            if not _attrs(metadata).get("requires_grid_barrier"):
                continue
            hierarchy = (
                metadata.get("launch", {}).get("sharding", {}).get("hierarchy", (1,))
            )
            product = 1
            for value in hierarchy:
                product *= int(value)
            sizes.add(max(product, 1))
    if not sizes:
        return 1
    if len(sizes) != 1:
        raise RuntimeError(
            "PyNTT generated kernels with grid barriers must use one grid mesh "
            f"size, got {sorted(sizes)}."
        )
    return next(iter(sizes))


def _attrs(metadata: dict[str, Any]) -> dict[str, Any]:
    return metadata.get("attrs") or metadata.get("Attrs") or {}


def _runtime_shape_args(metadata: dict[str, Any]) -> tuple[str, ...]:
    value = _attrs(metadata).get("runtime_shape_args", ())
    return tuple(value or ())


def _abi_view_stride_args(metadata: dict[str, Any]) -> tuple[str, ...]:
    value = _attrs(metadata).get("abi_view_stride_args", ())
    return tuple(value or ())


def _dim(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("TritonExpression", value.get("triton_expression", "0")))
    return str(value)


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


def _tensor_access(
    tensor_indices: tuple[str, ...] | list[str],
    strides: list[Any],
    lane_indices: tuple[str, ...] | list[str] = (),
    lane_shape: tuple[int, ...] | list[int] = (),
    coordinate_shape: str | None = None,
) -> dict[str, Any]:
    """Build one coordinate-preserving tensor access at render time."""

    tensor_indices = tuple(str(value) for value in tensor_indices)
    lane_indices = tuple(str(value) for value in lane_indices)
    lane_shape = tuple(int(value) for value in lane_shape)
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

    return {
        "CoordinateShape": coordinate_shape,
        "RawScalarOffset": raw_scalar_offset,
        "ScalarOffset": scalar_offset,
        "TensorIndices": tensor_indices,
        "TensorStrides": tuple(strides),
        "LaneIndices": lane_indices,
        "LaneShape": lane_shape,
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


def _add_coordinate(base: Any, index: str) -> str:
    if _fixed(base) == 0:
        return index
    if index == "0":
        return _dim(base)
    return f"({_dim(base)}) + ({index})"


def _access_pointer(
    model: dict[str, Any],
    name: str,
    local_name: str,
    access: Any = None,
) -> str:
    scalar_offset = _access_scalar_offset(access)
    return local_name if scalar_offset == "0" else f"{local_name} + {scalar_offset}"


def _pointer_shard_coord_hierarchy(pointer: Any) -> tuple[int, ...] | None:
    if not isinstance(pointer, dict):
        return None
    value = pointer.get("ShardCoordHierarchy", pointer.get("shard_coord_hierarchy"))
    if not value:
        return None
    return tuple(int(axis) for axis in value)


def _pointer_shard_hierarchy(pointers: list[Any]) -> tuple[int, ...] | None:
    hierarchies = {
        hierarchy
        for hierarchy in (
            _pointer_shard_coord_hierarchy(pointer) for pointer in pointers
        )
        if hierarchy is not None
    }
    if not hierarchies:
        return None
    if len(hierarchies) != 1:
        raise RuntimeError(
            "PyNTT generated helper has pointer offsets from multiple shard "
            f"hierarchies: {sorted(hierarchies)}."
        )
    return next(iter(hierarchies))


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
    ctx = _coordinate_iteration_context(
        local_shape,
        local_strides,
        lane_shape,
        "PyNTT TensorLoad" if is_load else "PyNTT TensorStore",
    )
    local_access = _tensor_access(
        ctx["tensor_coordinates"],
        local_strides,
        ctx["lane_coordinates"],
        ctx["lane_shape"],
        ctx["tile_shape"],
    )
    global_coordinates = tuple(
        _add_coordinate(offset, coordinate)
        for offset, coordinate in zip(model["GlobalOffsets"], ctx["tensor_coordinates"])
    )
    global_access = _tensor_access(
        global_coordinates,
        global_strides,
        ctx["lane_coordinates"],
        ctx["lane_shape"],
        ctx["tile_shape"],
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
        if index != "zero_coord":
            terms.append(index if stride == 1 else f"({index}) * {stride}")
        stride *= extent
    return _join_index_terms(list(reversed(terms)))


def _coordinate_iteration_context(
    tensor_shape: list[Any],
    tensor_strides: list[Any],
    lane_shape: list[int],
    context: str = "PyNTT elementwise",
) -> dict[str, Any]:
    """Build a coordinate-native block tile without scalar unflattening."""

    if len(tensor_shape) != len(tensor_strides):
        raise ValueError(
            "PyNTT coordinate iteration shape/stride rank mismatch: "
            f"shape={len(tensor_shape)}, strides={len(tensor_strides)}"
        )
    lanes = _validate_coordinate_lane_shape(lane_shape, context)
    lane_count = _product_int(list(lanes)) if lanes else 1
    lane_shift = lane_count.bit_length() - 1

    if tensor_shape:
        major_axis = _select_block_axis(tensor_shape, tensor_strides)
        major_extent = tensor_shape[major_axis]
        loop_axes = tuple(
            axis for axis in range(len(tensor_shape)) if axis != major_axis
        )
        tensor_coordinates = tuple(
            f"raw_coord{axis}" for axis in range(len(tensor_shape))
        )
        block_tensor_coordinates = tuple(
            f"coord{axis}" for axis in range(len(tensor_shape))
        )
    else:
        major_axis = -1
        major_extent = _one()
        loop_axes = ()
        tensor_coordinates = ()
        block_tensor_coordinates = ()

    major_width = (
        "block_size" if lane_shift == 0 else f"(block_size >> {lane_shift})"
    )

    lane_coordinates = tuple(f"raw_lane_coord{axis}" for axis in range(len(lanes)))
    block_lane_coordinates = tuple(f"lane_coord{axis}" for axis in range(len(lanes)))
    tile_extents = (major_width,) + tuple(str(value) for value in lanes)
    tile_shape = f"({', '.join(tile_extents)}{',' if len(tile_extents) == 1 else ''})"
    major_reshape = "" if not lanes else "[:, " + ", ".join("None" for _ in lanes) + "]"
    lane_reshapes = []
    for axis in range(len(lanes)):
        dimensions = ["None"] * (len(lanes) + 1)
        dimensions[axis + 1] = ":"
        lane_reshapes.append("[" + ", ".join(dimensions) + "]")

    return {
        "block_lane_coordinates": block_lane_coordinates,
        "block_tensor_coordinates": block_tensor_coordinates,
        "lane_count": lane_count,
        "lane_coordinates": lane_coordinates,
        "lane_reshapes": tuple(lane_reshapes),
        "lane_shape": lanes,
        "loop_axes": loop_axes,
        "major_axis": major_axis,
        "major_extent": major_extent,
        "major_reshape": major_reshape,
        "major_width": major_width,
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
    coordinate_shape: str,
) -> dict[str, Any]:
    lanes = tuple(int(value) for value in lane_shape)
    if len(shape) > len(output_shape):
        raise ValueError(
            "PyNTT elementwise operand rank exceeds output rank: "
            f"operand={len(shape)}, output={len(output_shape)}"
        )
    axis_offset = len(output_shape) - len(shape)
    tensor_coordinates = tuple(
        "zero_coord"
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
    return _tensor_access(
        tensor_coordinates,
        strides,
        lane_coordinates,
        lanes,
        coordinate_shape,
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
    ctx["source_access"] = _tensor_access(
        ctx["tensor_coordinates"],
        model["SourceStrides"],
        ctx["lane_coordinates"],
        lanes,
        ctx["tile_shape"],
    )
    ctx["destination_access"] = _tensor_access(
        ctx["tensor_coordinates"],
        model["DestinationStrides"],
        ctx["lane_coordinates"],
        lanes,
        ctx["tile_shape"],
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
        ctx["tile_shape"],
    )
    ctx["output_access"] = _tensor_access(
        ctx["tensor_coordinates"],
        model["OutputStrides"],
        ctx["lane_coordinates"],
        ctx["lane_shape"],
        ctx["tile_shape"],
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
            ctx["tile_shape"],
        )
    ctx["output_access"] = _tensor_access(
        ctx["tensor_coordinates"],
        model["OutputStrides"],
        ctx["lane_coordinates"],
        ctx["lane_shape"],
        ctx["tile_shape"],
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
    prefix_index = domain_lane_coordinates[0] if lane_ratio != 1 else "zero_coord"
    smaller_lane_index = (
        domain_lane_coordinates[-1] if smaller_lane_count != 1 else "zero_coord"
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
        return _tensor_access(
            tensor_coordinates,
            model[f"{prefix}Strides"],
            lane_coordinates,
            lane_shape,
            ctx["tile_shape"],
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
        tensor_coordinates.append("zero_coord" if _is_fixed_one(extent) else coordinate)
    lane_coordinates = ctx["lane_coordinates"] if lanes else ()
    return _tensor_access(
        tensor_coordinates,
        strides,
        lane_coordinates,
        lanes,
        ctx["tile_shape"],
    )


def _elementwise_where_template_context(model: dict[str, Any]) -> dict[str, Any]:
    ctx = _coordinate_iteration_context(
        model["OutputShape"],
        model["OutputStrides"],
        model["OutputVectorLaneShape"],
    )
    for prefix in ("Cond", "True", "False"):
        ctx[f"{prefix.lower()}_access"] = _where_operand_access(model, prefix, ctx)
    ctx["output_access"] = _tensor_access(
        ctx["tensor_coordinates"],
        model["OutputStrides"],
        ctx["lane_coordinates"],
        ctx["lane_shape"],
        ctx["tile_shape"],
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
            ctx["tile_shape"],
        )
        output_access = _tensor_access(
            ctx["tensor_coordinates"],
            model["OutputStrides"],
            ctx["lane_coordinates"],
            output_lanes,
            ctx["tile_shape"],
        )
    else:
        input_access = _tensor_access(
            ctx["tensor_coordinates"],
            model["InputStrides"],
            ctx["lane_coordinates"],
            input_lanes,
            ctx["tile_shape"],
        )
        output_access = _tensor_access(
            expanded_tensor_coordinates,
            model["OutputStrides"],
            preserved_lane_coordinates,
            output_lanes,
            ctx["tile_shape"],
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
    input_coordinates = ["zero_coord"] * rank
    for output_axis, input_axis in enumerate(permutation):
        input_coordinates[input_axis] = ctx["tensor_coordinates"][output_axis]
    ctx["input_access"] = _tensor_access(
        input_coordinates,
        model["InputStrides"],
        ctx["lane_coordinates"],
        input_lanes,
        ctx["tile_shape"],
    )
    ctx["output_access"] = _tensor_access(
        ctx["tensor_coordinates"],
        model["OutputStrides"],
        ctx["lane_coordinates"],
        output_lanes,
        ctx["tile_shape"],
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
        k_pack = int(model["RhsKPackLaneCount"])
        k_lane = int(model["RhsKVectorLaneCount"])
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


def _is_positive_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _matmul_template_context(
    model: dict[str, Any],
    *,
    gemv: bool,
    variant: str | None = None,
) -> dict[str, Any]:
    """Prepare Matmul/Gemv dimensions and addresses for Jinja-owned kernels."""

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
    if rhs_layout == "n_major":
        rhs_k = model["RhsShape"][-1]
        rhs_n = _multiply_dim(model["RhsShape"][-2], rhs_lane_count)
    else:
        rhs_k_lane_count = int(model["RhsKPackLaneCount"]) * int(
            model["RhsKVectorLaneCount"]
        )
        rhs_k = _multiply_dim(model["RhsShape"][-2], rhs_k_lane_count)
        rhs_n = _multiply_dim(model["RhsShape"][-1], rhs_lane_count)
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
    rhs_global_offsets = model.get("RhsGlobalOffsets")
    if not isinstance(rhs_global_offsets, list) or len(rhs_global_offsets) != 2:
        raise ValueError(
            "PyNTT packed GEMV pipeline requires two RHS global offsets."
        )

    block_n = context["block_n"]
    block_k = context["block_k"]
    num_stages = context["microkernel"]["parameters"]["num_stages"]
    if block_n != 64 or block_k != 128 or num_stages != 4:
        raise ValueError(
            "PyNTT packed GEMV pipeline resource contract must be "
            f"block_n=64, block_k=128, num_stages=4; got "
            f"{block_n}, {block_k}, {num_stages}."
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
            "KPack*KVector and block_n/32 divisible by NVector."
        )
    packed_k_outer = block_k // k_atom
    packed_n_outer = block_n // n_lane
    reduction_group = 32
    if block_k % reduction_group != 0:
        raise ValueError(
            "PyNTT K-major GEMV staging requires block_k divisible by "
            f"the warp reduction group {reduction_group}."
        )
    target_worker_width = int(model["TargetWorkerWidth"])
    consumer_warps = int(model["NumWarps"])
    consumer_k_values_per_thread = 8
    consumer_threads_per_warp_k = reduction_group // consumer_k_values_per_thread
    if (
        reduction_group % consumer_k_values_per_thread != 0
        or target_worker_width % consumer_threads_per_warp_k != 0
    ):
        raise ValueError(
            "PyNTT K-major GEMV consumer encoding cannot cover its reduction "
            f"group: reduction_group={reduction_group}, "
            f"values_per_thread={consumer_k_values_per_thread}, "
            f"worker_width={target_worker_width}."
        )
    consumer_threads_per_warp_n = (
        target_worker_width // consumer_threads_per_warp_k
    )
    if block_n % consumer_threads_per_warp_n != 0:
        raise ValueError(
            "PyNTT K-major GEMV consumer encoding cannot cover block_n: "
            f"block_n={block_n}, "
            f"threads_per_warp_n={consumer_threads_per_warp_n}."
        )
    consumer_warps_n = block_n // consumer_threads_per_warp_n
    if consumer_warps != consumer_warps_n:
        raise ValueError(
            "PyNTT K-major GEMV requires one consumer warp partition per N "
            f"slice: expected {consumer_warps_n} warps for block_n={block_n}, "
            f"got {consumer_warps}."
        )
    consumer_size_per_thread = (1, consumer_k_values_per_thread)
    consumer_threads_per_warp = (
        consumer_threads_per_warp_n,
        consumer_threads_per_warp_k,
    )
    consumer_warps_per_cta = (consumer_warps_n, 1)
    tma_contiguous_extent = n_lane * k_lane
    tma_block_shape = (
        packed_k_outer,
        packed_n_outer,
        k_pack,
        tma_contiguous_extent,
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
        "rhs_global_k_outer_offset": rhs_global_offsets[-2],
        "rhs_global_n_outer_offset": rhs_global_offsets[-1],
        "consumer_size_per_thread": consumer_size_per_thread,
        "consumer_threads_per_warp": consumer_threads_per_warp,
        "consumer_warps_per_cta": consumer_warps_per_cta,
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
        "pipeline_output_access": _tensor_access(
            ("0", f"(pipeline_output_n) // {n_lane}"),
            model["OutputStrides"],
            (f"(pipeline_output_n) % {n_lane}",),
            (n_lane,),
            _coordinate_shape((block_n,)),
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
    if block_n != 64 or block_k != 128 or num_stages != 4:
        raise ValueError(
            "PyNTT packed QKV GEMV pipeline resource contract must be "
            f"block_n=64, block_k=128, num_stages=4; got "
            f"{block_n}, {block_k}, {num_stages}."
        )

    n_lane = int(model["NVectorLaneCount"])
    k_pack = int(model["KPackLaneCount"])
    k_lane = int(model["KVectorLaneCount"])
    k_atom = k_pack * k_lane
    k = _fixed(context["k"])
    if k is None or k <= 0 or k % block_k != 0:
        raise ValueError(
            "PyNTT packed QKV GEMV pipeline requires a fixed K divisible by block_k."
        )
    packed_k_outer = block_k // k_atom
    packed_n_outer = block_n // n_lane
    for prefix in ("Q", "K", "V"):
        descriptor_name = model.get(f"{prefix}WeightDescriptorName")
        global_offsets = model.get(f"{prefix}WeightGlobalOffsets")
        weight_shape = model[f"{prefix}WeightShape"]
        if not isinstance(descriptor_name, str) or not descriptor_name:
            raise ValueError(
                f"PyNTT packed QKV GEMV pipeline requires a host {prefix} weight descriptor."
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
    padded_total_n = num_n_tiles * block_n
    max_sequence_count = num_n_tiles * num_k_tiles
    if max_sequence_count > (2**31 - 1):
        raise ValueError(
            "PyNTT packed QKV GEMV sequence exceeds the signed int32 pipe ABI: "
            f"{max_sequence_count}."
        )

    copy_n = _packed_qkv_copy_n(block_n, projection_ns)
    if copy_n % n_lane != 0:
        raise ValueError(
            "PyNTT packed QKV GEMV cannot form an NVector-aligned common "
            f"TMA copy extent from block_n={block_n} and projections={projection_ns}."
        )
    copy_n_outer = copy_n // n_lane
    copies_per_tile = block_n // copy_n
    copy_shape = (
        copy_n_outer,
        packed_k_outer,
        k_pack,
        n_lane * k_lane,
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
        output_access = _tensor_access(
            ("0", f"({projection_n_expr}) // {n_lane}"),
            model[f"{prefix}OutputStrides"],
            (f"({projection_n_expr}) % {n_lane}",),
            (n_lane,),
            _coordinate_shape((block_n,)),
        )
        has_bias = projection_by_prefix[prefix]["has_bias"]
        bias_access = (
            _tensor_access(
                (f"({projection_n_expr}) // {n_lane}",),
                model[f"{prefix}BiasStrides"],
                (f"({projection_n_expr}) % {n_lane}",),
                (n_lane,),
                _coordinate_shape((block_n,)),
            )
            if has_bias
            else None
        )
        projections.append(
            {
                "prefix": prefix,
                "lower": lower,
                "descriptor_name": model[f"{prefix}WeightDescriptorName"],
                "descriptor_global_k_outer_offset": model[
                    f"{prefix}WeightGlobalOffsets"
                ][-2],
                "descriptor_global_n_outer_offset": model[
                    f"{prefix}WeightGlobalOffsets"
                ][-1],
                "projection_start": projection_start,
                "projection_end": projection_end,
                "copy_end": (
                    padded_total_n if prefix == "V" else projection_end
                ),
                "projection_n_expr": projection_n_expr,
                "projection_mask": projection_mask,
                "has_bias": has_bias,
                "bias_access": bias_access,
                "output_access": output_access,
            }
        )

    reduction_group = 32
    consumer_k_values_per_thread = 8
    target_worker_width = int(model["TargetWorkerWidth"])
    consumer_warps = int(model["NumWarps"])
    consumer_threads_per_warp_k = reduction_group // consumer_k_values_per_thread
    consumer_threads_per_warp_n = (
        target_worker_width // consumer_threads_per_warp_k
    )
    consumer_warps_n = block_n // consumer_threads_per_warp_n
    if consumer_warps != consumer_warps_n:
        raise ValueError(
            "PyNTT packed QKV GEMV requires one consumer warp partition per N "
            f"slice: expected {consumer_warps_n}, got {consumer_warps}."
        )
    context.update(
        block_n=block_n,
        block_k=block_k,
        num_stages=num_stages,
        num_k_tiles=num_k_tiles,
        num_n_tiles=num_n_tiles,
        projections=tuple(projections),
        copy_n=copy_n,
        copy_n_outer=copy_n_outer,
        copies_per_tile=copies_per_tile,
        copy_shape=copy_shape,
        copy_is_full_stage=copy_n == block_n,
        k_atom=k_atom,
        packed_k_outer=packed_k_outer,
        reduction_group=reduction_group,
        reduction_groups_per_stage=block_k // reduction_group,
        shared_stage_shape=(
            packed_n_outer,
            packed_k_outer,
            k_pack,
            n_lane * k_lane,
        ),
        tma_contiguous_extent=n_lane * k_lane,
        consumer_size_per_thread=(1, consumer_k_values_per_thread),
        consumer_threads_per_warp=(
            consumer_threads_per_warp_n,
            consumer_threads_per_warp_k,
        ),
        consumer_warps_per_cta=(consumer_warps_n, 1),
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
    if (
        block_n != 64
        or block_k != 128
        or not 2 <= num_stages <= 4
        or num_stages % projection_count != 0
        or num_stages // projection_count < 2
    ):
        raise ValueError(
            "PyNTT packed MatMulGlu GEMV pipeline resource contract requires "
            "block_n=64, block_k=128, and at least two complete gate/up "
            "transfer groups in 2-4 physical stages; got "
            f"{block_n}, {block_k}, {num_stages}."
        )

    n_lane = int(model["NVectorLaneCount"])
    k_pack = int(model["KPackLaneCount"])
    k_lane = int(model["KVectorLaneCount"])
    k_atom = k_pack * k_lane
    k = _fixed(context["k"])
    if k is None or k <= 0 or k % block_k != 0:
        raise ValueError(
            "PyNTT packed MatMulGlu GEMV pipeline requires a fixed K "
            "divisible by block_k."
        )
    packed_k_outer = block_k // k_atom
    packed_n_outer = block_n // n_lane
    for prefix in ("Gate", "Up"):
        descriptor_name = model.get(f"{prefix}WeightDescriptorName")
        global_offsets = model.get(f"{prefix}WeightGlobalOffsets")
        weight_shape = model[f"{prefix}WeightShape"]
        if not isinstance(descriptor_name, str) or not descriptor_name:
            raise ValueError(
                "PyNTT packed MatMulGlu GEMV pipeline requires a host "
                f"{prefix.lower()} weight descriptor."
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
    consumer_k_values_per_thread = 8
    target_worker_width = int(model["TargetWorkerWidth"])
    consumer_warps = int(model["NumWarps"])
    consumer_threads_per_warp_k = reduction_group // consumer_k_values_per_thread
    consumer_threads_per_warp_n = (
        target_worker_width // consumer_threads_per_warp_k
    )
    consumer_warps_n = block_n // consumer_threads_per_warp_n
    if consumer_warps != consumer_warps_n:
        raise ValueError(
            "PyNTT packed MatMulGlu GEMV requires one consumer warp partition "
            f"per N slice: expected {consumer_warps_n}, got {consumer_warps}."
        )
    tma_contiguous_extent = n_lane * k_lane
    tma_block_shape = (
        packed_k_outer,
        packed_n_outer,
        k_pack,
        tma_contiguous_extent,
    )
    projections = []
    for projection_index, prefix in enumerate(("Gate", "Up")):
        lower = prefix.lower()
        output_n = "pipeline_output_n"
        output_mask = f"{output_n} < active_n"
        projections.append(
            {
                "prefix": prefix,
                "lower": lower,
                "descriptor_name": model[f"{prefix}WeightDescriptorName"],
                "descriptor_global_k_outer_offset": model[
                    f"{prefix}WeightGlobalOffsets"
                ][-2],
                "descriptor_global_n_outer_offset": model[
                    f"{prefix}WeightGlobalOffsets"
                ][-1],
                "sequence_offset": projection_index,
                "has_bias": bool(model[f"Has{prefix}Bias"]),
                "bias_access": (
                    _tensor_access(
                        (f"({output_n}) // {n_lane}",),
                        model[f"{prefix}BiasStrides"],
                        (f"({output_n}) % {n_lane}",),
                        (n_lane,),
                        _coordinate_shape((block_n,)),
                    )
                    if model[f"Has{prefix}Bias"]
                    else None
                ),
                "output_mask": output_mask,
            }
        )

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
        shared_stage_shape=(
            packed_k_outer,
            packed_n_outer,
            k_pack,
            tma_contiguous_extent,
        ),
        tma_block_shape=tma_block_shape,
        tma_contiguous_extent=tma_contiguous_extent,
        consumer_size_per_thread=(1, consumer_k_values_per_thread),
        consumer_threads_per_warp=(
            consumer_threads_per_warp_n,
            consumer_threads_per_warp_k,
        ),
        consumer_warps_per_cta=(consumer_warps_n, 1),
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
        pipeline_output_access=_tensor_access(
            ("0", f"(pipeline_output_n) // {n_lane}"),
            model["OutputStrides"],
            (f"(pipeline_output_n) % {n_lane}",),
            (n_lane,),
            _coordinate_shape((block_n,)),
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
        context["tile_shape"],
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
        context["tile_shape"],
    )
    context["output_access"] = _tensor_access(
        tensor_coordinates,
        model["OutputStrides"],
        context["lane_coordinates"],
        context["lane_shape"],
        context["tile_shape"],
    )
    context["scale_access"] = _tensor_access(
        inner_coordinates,
        model["ScaleStrides"],
        context["lane_coordinates"],
        context["lane_shape"],
        context["tile_shape"],
    )
    context["bias_access"] = _tensor_access(
        inner_coordinates,
        model["BiasStrides"],
        context["lane_coordinates"],
        context["lane_shape"],
        context["tile_shape"],
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
                coordinate = "zero_coord"
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
            context["tile_shape"],
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
        output_physical_rotary = f"coord{rotary_axis}"
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
        f"(coord{rotary_axis}) * {sincos_pack_factor} + ({pack_coordinate})"
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
            f"coord{rotary_axis}",
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
            f"coord{rotary_axis}",
            context["lane_coordinates"],
            sincos_lane_shape,
        ),
    )
    return context


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
    for index_axis, extent in enumerate(model["IndexShape"]):
        output_axis = model["Axis"] + index_axis
        index_coordinates.append(
            "zero_coord"
            if _is_fixed_one(extent)
            else context["tensor_coordinates"][output_axis]
        )
    index_access = _tensor_access(
        index_coordinates,
        model["IndexStrides"],
        coordinate_shape=context["tile_shape"],
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
        context["tile_shape"],
    )
    output_access = _tensor_access(
        context["tensor_coordinates"],
        model["OutputStrides"],
        context["lane_coordinates"],
        lane_shape,
        context["tile_shape"],
    )

    gather_split_axes = model["InputSplitAxes"][model["Axis"]]
    context.update(
        gather_split_axes=gather_split_axes,
        index_access=index_access,
        input_access=input_access,
        input_split_linear=_split_linear_expression(
            gather_split_axes, model["Hierarchy"]
        ),
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
    input_split_mesh_axes = {
        axis for split_axes in model["InputSplitAxes"] for axis in split_axes
    }
    input_partial_mesh_axes = set(model["InputPartialAxes"])
    if input_split_mesh_axes & input_partial_mesh_axes:
        raise ValueError(
            "A PyNTT reshard mesh axis cannot be both split and partial: "
            f"{sorted(input_split_mesh_axes & input_partial_mesh_axes)}"
        )
    output_split_mesh_axes = {
        axis for split_axes in model["OutputSplitAxes"] for axis in split_axes
    }
    output_broadcast_mesh_axes = tuple(
        axis
        for axis in range(len(model["Hierarchy"]))
        if axis not in output_split_mesh_axes
    )
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
        _add_coordinate(offset, coordinate)
        for offset, coordinate in zip(
            model["InputGlobalOffsets"], context["tensor_coordinates"]
        )
    )
    context["input_access"] = _tensor_access(
        context["tensor_coordinates"],
        model["InputStrides"],
        context["lane_coordinates"],
        context["lane_shape"],
        context["tile_shape"],
    )
    context["output_access"] = _tensor_access(
        tuple(f"output_idx{axis}" for axis in range(len(model["OutputStrides"]))),
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

    def output_axis_range(global_extent: Any, split_axes: list[int]) -> dict[str, Any]:
        return {
            "divisor": _split_divisor(split_axes, model["Hierarchy"])
            if split_axes
            else 1,
            "global_extent": global_extent,
            "split_axes": tuple(split_axes),
            "split_linear": _split_linear_expression(split_axes, model["Hierarchy"])
            if split_axes
            else "0",
        }

    def local_index(
        prefix: str,
        global_index: str,
        global_extent: Any,
        split_axes: list[int],
    ) -> dict[str, Any]:
        return {
            "divisor": _split_divisor(split_axes, model["Hierarchy"])
            if split_axes
            else 1,
            "global_extent": global_extent,
            "global_index": global_index,
            "prefix": prefix,
            "result": f"{prefix}_local" if split_axes else global_index,
            "split_axes": tuple(split_axes),
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
    global_n_logical = (
        f"({n_axis['logical_coordinate']}) + out_n_global_base * {lane_count}"
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
            model["LhsSplitAxes"][1],
        ),
        "lhs_m": local_index(
            "lhs_m",
            "global_m[:, None]",
            model["LhsGlobalShape"][0],
            model["LhsSplitAxes"][0],
        ),
        "lhs_pointer_type": _pointer_type(
            model["LhsTritonDType"], model["LhsAddressSpace"]
        ),
        "out_m": output_axis_range(
            model["OutputGlobalShape"][0], model["OutputSplitAxes"][0]
        ),
        "out_n": output_axis_range(
            output_global_physical_n, model["OutputSplitAxes"][1]
        ),
        "broadcast_global_k": broadcast_global_k,
        "broadcast_offs_m": broadcast_offs_m,
        "global_n_logical": global_n_logical,
        "global_n_physical": (f"out_n_global_base + {n_axis['physical_coordinate']}"),
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
            model["RhsSplitAxes"][0],
        ),
        "rhs_n": local_index(
            "rhs_n",
            "global_n_physical",
            model["RhsGlobalShape"][1],
            model["RhsSplitAxes"][1],
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


def _paged_attention_template_context(model: dict[str, Any]) -> dict[str, Any]:
    """Validate PagedAttention layouts and prepare coordinate-native accesses."""

    cache = model["Cache"]
    attention_block_size = int(model["AttentionBlockSize"])
    cache_block_size = int(cache["BlockSize"])
    if (
        attention_block_size <= 0
        or attention_block_size & (attention_block_size - 1)
        or attention_block_size > cache_block_size
        or cache_block_size % attention_block_size != 0
    ):
        raise ValueError(
            "PyNTT PagedAttention AttentionBlockSize must be a positive power "
            "of two that divides the cache block size, got "
            f"attention={attention_block_size}, cache={cache_block_size}."
        )

    def global_index_expression(axis: int, local_index: str, global_extent: Any) -> str:
        split_axes = model["OutputSplitAxes"][axis]
        if not split_axes:
            return local_index
        divisor = _split_divisor(split_axes, model["Hierarchy"])
        return (
            f"{local_index} + "
            f"({_split_linear_expression(split_axes, model['Hierarchy'])}) * "
            f"tl.cdiv({_dim(global_extent)}, {divisor})"
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

    query_dim_axis = _structured_axis_tile(
        "query_dim",
        query_lanes,
        int(cache["HeadDim"]),
        cache["HeadDim"],
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
    query_indices[model["HeadAxis"]] = "q_head"
    query_indices[dim_axis] = query_dim_axis["physical_coordinate"]
    output_indices = ["0"] * len(model["OutputShape"])
    output_indices[model["SeqAxis"]] = "local_query_id"
    output_indices[model["HeadAxis"]] = "q_head"
    output_indices[dim_axis] = query_dim_axis["physical_coordinate"]

    key_lane = _flatten_coordinates(
        key_dim_axis["lane_coordinates"], key_dim_axis["lane_shape"]
    )
    key_block_offset = _broadcast_axis_coordinate(
        "block_offsets", key_dim_axis["rank"], key_dim_axis["rank"] - 1
    )
    key_vector_offset = (
        f"(cache_block_id * {cache['BlockElements']} + "
        f"{cache['KeySectionOffset']} + ((layer_id_value) * "
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
                f"((context_start % {cache_block_size}) // {value_lane_count})"
            ),
        )
        value_lane = value_axis["lane_coordinates"][0]
        value_dim_index = _broadcast_axis_coordinate(
            "dim_offsets", value_axis["rank"], value_axis["rank"] - 1
        )
        value_vector_offset = (
            f"(cache_block_id * {cache['BlockElements']} + "
            f"{cache['ValueSectionOffset']} + ((layer_id_value) * "
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
            f"(cache_block_id * {cache['BlockElements']} + "
            f"{cache['ValueSectionOffset']} + ((layer_id_value) * "
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
    return {
        "attention_block_size": attention_block_size,
        "cache_block_id": (
            "(topology_id * num_blocks_per_shard + block_id)"
            if cache["IdLength"] > 1
            else "block_id"
        ),
        "global_q_head": global_index_expression(
            model["HeadAxis"], "q_head", model["GlobalNumQueryHeads"]
        ),
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
        "local_q_heads": model["OutputShape"][model["HeadAxis"]],
        "local_query_tokens": local_query_tokens,
        "output_access": _tensor_access(
            output_indices,
            model["OutputStrides"],
            query_dim_axis["lane_coordinates"],
            output_lanes,
            _coordinate_shape(query_dim_axis["structured_shape"]),
        ),
        "query_access": _tensor_access(
            query_indices,
            model["QueryStrides"],
            query_dim_axis["lane_coordinates"],
            query_lanes,
            _coordinate_shape(query_dim_axis["structured_shape"]),
        ),
        "query_dim_axis": query_dim_axis,
        "query_structured_shape": query_dim_axis["structured_shape"],
        "value_axis": value_axis,
        "value_axis_kind": value_axis_kind,
        "value_mask": value_mask,
        "value_structured_shape": value_structured_shape,
        "value_vector_offset": value_vector_offset,
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
    if vectorized_dim == 5 and lane_count != slots_lane_count:
        raise ValueError(
            "PyNTT key-cache HeadDim lanes must match the slot tensor lanes: "
            f"cache={lane_count}, slots={slots_lane_count}."
        )
    source_split_axes = sorted(
        {axis for split_axes in model["SlotsSourceSplitAxes"] for axis in split_axes}
    )
    topology_match_axes = tuple(
        axis for axis in cache["NumBlocksSplitAxes"] if axis not in source_split_axes
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
        context["block_lane_coordinates"], context["lane_shape"]
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
                context["tile_shape"],
            ),
            "slots_lane_count": slots_lane_count,
            "source_lane_id": source_lane_id,
            "topology_match_axes": topology_match_axes,
            "vectorized_dim": vectorized_dim,
        }
    )
    return context
