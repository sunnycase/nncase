"""Render generated PyNTT Triton kernels from a nncase codegen manifest."""

from __future__ import annotations

import ast
import importlib
import json
import math
import re
import sys
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

DEVICE_CALL_RE = re.compile(
    r"(?m)^(?P<indent>[ \t]*)__pyntt_device_call__(?P<name>[A-Za-z_]\w*)\((?P<args>.*)\)$"
)
DEVICE_CALL_NAME_RE = re.compile(r"__pyntt_device_call__(?P<name>[A-Za-z_]\w*)\(")


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


def render_manifest(manifest: dict[str, Any]) -> str:
    kernels = [
        _render_kernel(kernel)
        for function in manifest.get("functions", ())
        for kernel in function.get("render_kernels", ())
    ]
    needs_grid_barrier = any(
        _attrs(kernel.get("metadata", {})).get("requires_grid_barrier")
        for function in manifest.get("functions", ())
        for kernel in function.get("render_kernels", ())
    )
    needs_shared_memory = any(
        int(_attrs(kernel.get("metadata", {})).get("shared_memory_bytes", 0)) > 0
        for function in manifest.get("functions", ())
        for kernel in function.get("render_kernels", ())
    )
    env = _make_env()
    return env.get_template("triton/module.py.jinja").render(
        kernels=kernels,
        needs_tle=needs_grid_barrier or needs_shared_memory,
        needs_grid_barrier=needs_grid_barrier,
        grid_mesh_size=_grid_mesh_size(manifest),
    )


def _kernel_parameters(metadata: dict[str, Any]) -> tuple[str, ...]:
    grid_barrier_parameters = (
        ("pyntt_grid_mesh: tl.constexpr",)
        if _attrs(metadata).get("requires_grid_barrier")
        else ()
    )
    return (
        tuple(
            f"input{index}"
            for index, _ in enumerate(metadata.get("inputs", ()))
        )
        + tuple(
            f"output{index}"
            for index, _ in enumerate(metadata.get("outputs", ()))
        )
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
        + grid_barrier_parameters
        + ("numel", "block_size: tl.constexpr")
    )


def _render_kernel(kernel: dict[str, Any]) -> str:
    env = _make_env()
    metadata = kernel["metadata"]
    kernel_attrs = _attrs(metadata)
    parameters = _kernel_parameters(metadata)
    shared_memory_bytes = int(kernel_attrs.get("shared_memory_bytes", 0))
    if shared_memory_bytes < 0:
        raise ValueError(
            f"PyNTT kernel {metadata['name']} has invalid shared_memory_bytes={shared_memory_bytes}."
    )
    raw_device_functions = tuple(kernel["device_functions"])
    shared_allocation_bytes = _round_memory_arena_size(
        shared_memory_bytes,
        str(kernel_attrs.get("shared_memory_allocation_size_policy", "")),
        int(kernel_attrs.get("shared_memory_allocation_granularity_bytes", 0)),
    )
    shared_memory_capacity_bytes = int(
        kernel_attrs.get("shared_memory_capacity_bytes", 0)
    )
    if (
        shared_memory_capacity_bytes > 0
        and shared_allocation_bytes > shared_memory_capacity_bytes
    ):
        raise ValueError(
            f"PyNTT kernel {metadata['name']} requires "
            f"{shared_allocation_bytes} shared-memory bytes after allocation "
            f"rounding (AutoTiling arena {shared_memory_bytes}), "
            f"exceeding target capacity {shared_memory_capacity_bytes}."
        )
    hidden_device_parameters = (
        ("pyntt_shared_arena",) if shared_allocation_bytes > 0 else ()
    )
    device_functions = _prepare_device_functions(
        raw_device_functions,
        parameters,
        hidden_device_parameters,
    )
    device_functions_by_name = {
        device_function["name"]: device_function
        for device_function in device_functions
    }
    helper_sources = _render_helper_sources(env, kernel.get("helpers", ()))
    device_function_sources = [
        _render_device_function(
            env,
            device_function,
            hidden_device_parameters,
            device_functions_by_name,
        )
        for device_function in device_functions
    ]
    body_source = _replace_device_function_calls(
        kernel.get("body_source", ""),
        device_functions_by_name,
    )
    top_kernel = env.get_template("triton/top_kernel.py.jinja").render(
        name=metadata["name"],
        parameters=", ".join(parameters),
        body_source=body_source.rstrip(),
        materialize_shard_index=_needs_shard_index_prelude(
            body_source,
            parameters,
        ),
        shared_allocation_bytes=shared_allocation_bytes,
        noinline=False,
    ).strip()
    parts = [source for source in helper_sources if source]
    parts.extend(source for source in device_function_sources if source)
    parts.append(top_kernel)
    return "\n\n".join(parts)


def _render_device_function(
    env: Environment,
    device_function: dict[str, Any],
    hidden_parameters: tuple[str, ...],
    device_functions_by_name: dict[str, dict[str, Any]],
) -> str:
    helper_sources = _render_helper_sources(
        env,
        device_function.get("helpers", ()),
        noinline=bool(device_function["preserve_helper_call_boundaries"]),
    )
    parts = [source for source in helper_sources if source]
    device_parameters = (
        tuple(device_function["direct_parameters"])
        + hidden_parameters
        + tuple(device_function["direct_extra_parameters"])
    )
    for stage in device_function["stages"]:
        body_source = _replace_device_function_calls(
            stage["body_source"],
            device_functions_by_name,
        )
        parts.append(
            env.get_template("triton/top_kernel.py.jinja").render(
                name=stage["name"],
                parameters=", ".join(device_parameters),
                body_source=body_source.rstrip(),
                materialize_shard_index=_needs_shard_index_prelude(
                    body_source,
                    device_parameters,
                ),
                shared_allocation_bytes=0,
                noinline=device_function["noinline"],
            ).strip()
        )
    return "\n\n".join(parts)


def _prepare_device_functions(
    device_functions: tuple[dict[str, Any], ...],
    parameters: tuple[str, ...],
    hidden_parameters: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    parameter_names = _parameter_call_arguments(parameters)
    parameter_by_name = dict(zip(parameter_names, parameters))
    prepared_functions = []
    for device_function in device_functions:
        prepared = dict(device_function)
        extra_parameters = tuple(device_function["extra_parameters"])
        prepared["direct_extra_parameters"] = extra_parameters
        prepared["hidden_parameters"] = hidden_parameters
        prepared["stages"] = (
            {
                "name": device_function["name"],
                "body_source": device_function.get("body_source", "").rstrip()
                or "pass",
            },
        )
        prepared_functions.append(prepared)

    functions_by_name = {
        device_function["name"]: device_function
        for device_function in prepared_functions
    }
    required_parameters = {
        name: _referenced_parameter_names(
            device_function.get("body_source", ""), parameter_names
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
                device_function.get("body_source", "")
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
    return (
        SHARD_INDEX_PARAMETER not in _parameter_call_arguments(parameters)
        and SHARD_INDEX_PARAMETER
        in _referenced_parameter_names(source, (SHARD_INDEX_PARAMETER,))
    )


def _render_helper_sources(
    env: Environment, helpers: Any, *, noinline: bool = False
) -> list[str]:
    helper_sources = []
    for helper in helpers:
        model = dict(helper["model"])
        model["NoInline"] = bool(noinline) and not bool(helper["requires_inline"])
        arguments = tuple(helper.get("arguments", ()) or ())
        if arguments:
            model["Arguments"] = arguments
        helper_sources.append(
            env.get_template(helper["template"]).render(model=model).strip()
        )
    return helper_sources


def _parameter_call_arguments(parameters: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(parameter.split(":", 1)[0].strip() for parameter in parameters)


def _split_expression_arguments(source: str) -> tuple[str, ...]:
    if not source.strip():
        return ()
    wrapped = f"_pyntt_call({source})"
    try:
        expression = ast.parse(wrapped, mode="eval").body
    except SyntaxError as ex:
        raise RuntimeError(
            f"Invalid PyNTT device-function arguments: {source}"
        ) from ex
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
    extra_parameters = tuple(device_function["extra_parameters"])
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
    extra_arguments = _bind_device_function_extra_arguments(
        device_function, explicit_extra_arguments
    )

    parameter_overrides = dict(device_function["parameter_overrides"])
    call_arguments = tuple(
        parameter_overrides.get(argument, argument)
        for argument in _parameter_call_arguments(
            tuple(device_function["direct_parameters"])
        )
    ) + tuple(
        parameter_overrides.get(argument, argument)
        for argument in device_function["hidden_parameters"]
    ) + tuple(
        extra_arguments[parameter]
        for parameter in device_function["direct_extra_parameters"]
    )
    return f"{device_function['name']}({', '.join(call_arguments)})"


def _replace_device_function_calls(
    source: str,
    device_functions: dict[str, dict[str, Any]],
) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        if name not in device_functions:
            raise RuntimeError(f"PyNTT kernel references unknown device function {name}.")
        indent = match.group("indent")
        extra_arguments = _split_expression_arguments(match.group("args"))
        call_source = _build_device_function_call(
            device_functions[name],
            extra_arguments,
        )

        return "\n".join(
            f"{indent}{line}" if line else line
            for line in call_source.splitlines()
        )

    return DEVICE_CALL_RE.sub(replace, source)


def _round_memory_arena_size(
    requested_bytes: int, policy: str, granularity_bytes: int
) -> int:
    if requested_bytes < 0:
        raise ValueError(
            f"Shared-memory allocation size must not be negative, got {requested_bytes}."
        )
    if requested_bytes == 0:
        return 0
    if granularity_bytes <= 0 or granularity_bytes & (granularity_bytes - 1):
        raise ValueError(
            "Shared-memory allocation granularity must be a positive power of two, "
            f"got {granularity_bytes}."
        )
    if policy == "granularity_aligned":
        return (
            (requested_bytes + granularity_bytes - 1) // granularity_bytes
        ) * granularity_bytes
    if policy == "power_of_two":
        return 1 << (max(requested_bytes, granularity_bytes) - 1).bit_length()
    raise ValueError(f"Unsupported shared-memory allocation size policy: {policy!r}.")


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
        access_pointer_value=_access_pointer_value,
        axes_except=lambda rank, excluded: tuple(
            axis for axis in range(rank) if axis != excluded
        ),
        concat_context=_concat_template_context,
        conv2d_context=_conv2d_template_context,
        dim=_dim,
        fixed=_fixed,
        gather_context=_gather_template_context,
        helper_parameters=_helper_parameters,
        is_bool_dtype=_is_bool_dtype,
        is_fixed_one=_is_fixed_one,
        layer_norm_context=_layer_norm_template_context,
        local_buffer=_local_buffer,
        logical_shape=_logical_shape,
        logical_strides=_logical_strides,
        matmul_glu_context=_matmul_glu_template_context,
        matmul_context=_matmul_template_context,
        multiply_expr=_multiply_expr,
        norm_apply_context=_norm_apply_template_context,
        norm_stats_context=_norm_stats_template_context,
        paged_attention_context=_paged_attention_template_context,
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
        tensor_region_copy_context=_tensor_region_copy_template_context,
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
                metadata.get("launch", {})
                .get("sharding", {})
                .get("hierarchy", (1,))
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
        return str(value.get("PythonExpression", value.get("python_expression", _dim(value))))
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
        return dict(dim) if isinstance(dim, dict) else {
            "PythonExpression": str(dim),
            "TritonExpression": str(dim),
            "FixedValue": dim if isinstance(dim, int) else None,
        }
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


def _ptr(model: dict[str, Any], name: str) -> str:
    value = model[name]
    if isinstance(value, dict):
        return value.get("Expression", value.get("expression"))
    return str(value)






def _local_buffer(pointer: Any) -> dict[str, Any] | None:
    if not isinstance(pointer, dict):
        return None
    value = pointer.get("LocalBuffer", pointer.get("local_buffer"))
    return value if isinstance(value, dict) else None


def _local_buffer_value(buffer: dict[str, Any], name: str) -> Any:
    snake_name = "".join(
        f"_{character.lower()}" if character.isupper() else character
        for character in name
    ).lstrip("_")
    return buffer.get(name, buffer.get(snake_name))


def _local_pointer(pointer: Any, scalar_offset: str = "0") -> str | None:
    buffer = _local_buffer(pointer)
    if buffer is None:
        return None
    descriptor = _local_buffer_value(buffer, "DescriptorExpression")
    descriptor_shape = tuple(
        int(value) for value in (_local_buffer_value(buffer, "DescriptorShape") or ())
    )
    base_offset = str(_local_buffer_value(buffer, "BaseScalarOffset") or "0")
    if not descriptor or not descriptor_shape or any(value <= 0 for value in descriptor_shape):
        raise ValueError("PyNTT local buffer requires a descriptor and a positive physical shape")

    if scalar_offset == "0":
        total_offset = base_offset
    elif base_offset == "0":
        total_offset = scalar_offset
    else:
        total_offset = f"({base_offset}) + ({scalar_offset})"

    indices = []
    inner_stride = math.prod(descriptor_shape)
    for axis, extent in enumerate(descriptor_shape):
        inner_stride //= extent
        index = total_offset if inner_stride == 1 else f"(({total_offset}) // {inner_stride})"
        if axis != 0:
            index = f"({index}) % {extent}"
        indices.append(index)
    suffix = "," if len(indices) == 1 else ""
    return f"tle.gpu.local_ptr({descriptor}, ({', '.join(indices)}{suffix}))"


def _access_pointer(
    model: dict[str, Any],
    name: str,
    local_name: str,
    scalar_offset: str = "0",
) -> str:
    return _access_pointer_value(model[name], local_name, scalar_offset)


def _access_pointer_value(
    pointer: Any,
    local_name: str,
    scalar_offset: str = "0",
) -> str:
    local_pointer = _local_pointer(pointer, scalar_offset)
    if local_pointer is not None:
        return local_pointer
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
        for hierarchy in (_pointer_shard_coord_hierarchy(pointer) for pointer in pointers)
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


def _split_linear_expression(split_axes: list[int], hierarchy: list[int], coord_prefix: str = "shard_coord") -> str:
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


def _helper_parameters(
    model: dict[str, Any], args: tuple[str, ...] | list[str] = ()
) -> str:
    """Build the stable helper ABI; kernel control flow belongs to Jinja."""

    abi_args = tuple(model.get("Arguments", ()) or ())
    return ", ".join(
        tuple(args)
        + abi_args
        + WORKSPACE_PARAMETERS
        + WORKSPACE_STRIDE_PARAMETERS
        + tuple(model.get("RuntimeShapeArgs", ()) or ())
        + ("block_size: tl.constexpr",)
    )








def _is_bool_dtype(dtype: Any) -> bool:
    return str(dtype) == "bool"






def _constant_dim_value(value: Any) -> int | None:
    fixed = _fixed(value)
    if fixed is not None:
        return fixed
    minimum = _min_value(value)
    maximum = _max_value(value)
    return minimum if minimum is not None and minimum == maximum else None


def _is_compact_region(shape: list[int], strides: list[int]) -> bool:
    if len(shape) != len(strides):
        return False
    expected_stride = 1
    for extent, stride in zip(reversed(shape), reversed(strides)):
        if extent <= 0 or (extent > 1 and stride != expected_stride):
            return False
        expected_stride *= extent
    return True


def _region_copy_tle_plan(model: dict[str, Any]) -> dict[str, Any] | None:
    operation = model.get("OperationKind", model.get("operation_kind"))
    if operation == "TileLoad":
        local_name = "destination"
        local_model_name = "Destination"
        global_name = "source"
        global_model_name = "Source"
    elif operation == "TileStore":
        local_name = "source"
        local_model_name = "Source"
        global_name = "destination"
        global_model_name = "Destination"
    else:
        return None
    if model.get("RegionsCoincident", model.get("regions_coincident")) is not True:
        return None

    local_pointer = model[local_model_name]
    global_pointer = model[global_model_name]
    local_buffer = _local_buffer(local_pointer)
    if local_buffer is None:
        return None
    if int(local_pointer.get("AddressSpace", 1)) != 3 or int(global_pointer.get("AddressSpace", 1)) != 1:
        return None

    local_shape = model[f"{local_model_name}Shape"]
    local_shape_values = [_constant_dim_value(dimension) for dimension in local_shape]
    if any(dimension is None or dimension <= 0 for dimension in local_shape_values):
        return None
    static_shape = [int(dimension) for dimension in local_shape_values]
    local_strides = [_constant_dim_value(value) for value in model[f"{local_model_name}Strides"]]
    global_strides = [_constant_dim_value(value) for value in model[f"{global_model_name}Strides"]]
    if any(stride is None for stride in local_strides + global_strides):
        return None
    if not _is_compact_region(static_shape, [int(stride) for stride in local_strides]):
        return None
    if not _is_compact_region(static_shape, [int(stride) for stride in global_strides]):
        return None

    descriptor_shape = tuple(
        int(value)
        for value in (_local_buffer_value(local_buffer, "DescriptorShape") or ())
    )
    base_scalar_offset = str(
        _local_buffer_value(local_buffer, "BaseScalarOffset") or "0"
    )
    scalar_capacity = math.prod(static_shape) * int(model["VectorLaneCount"])
    scalar_element_size = int(
        _local_buffer_value(local_buffer, "ScalarElementSizeBytes") or 0
    )
    available_bytes = int(_local_buffer_value(local_buffer, "AvailableBytes") or 0)
    required_bytes = scalar_capacity * scalar_element_size
    if (
        scalar_capacity <= 0
        or not descriptor_shape
        or any(
            extent <= 0 or extent & (extent - 1) != 0
            for extent in descriptor_shape
        )
        or math.prod(descriptor_shape) != scalar_capacity
        or base_scalar_offset != "0"
        or scalar_element_size <= 0
        or required_bytes > available_bytes
    ):
        return None

    return {
        "local_name": local_name,
        "local_model_name": local_model_name,
        "global_name": global_name,
        "global_model_name": global_model_name,
        "local_buffer": local_buffer,
        "static_shape": static_shape,
        "descriptor_shape": descriptor_shape,
        "scalar_capacity": scalar_capacity,
    }


def _tensor_region_copy_template_context(model: dict[str, Any]) -> dict[str, Any]:
    """Validate a region copy and prepare address expressions for its template."""

    rank = len(model["SourceShape"])
    if rank != len(model["DestinationShape"]):
        raise ValueError("TensorRegionCopy source and destination ranks must match")
    plan = _region_copy_tle_plan(model)
    context: dict[str, Any] = {
        "plan": plan,
        "rank": rank,
        "pointers": (
            ((plan["global_name"], plan["global_model_name"]),)
            if plan is not None
            else (("source", "Source"), ("destination", "Destination"))
        ),
    }
    context["pointer_values"] = tuple(
        model[model_name] for _, model_name in context["pointers"]
    )
    if plan is None:
        dimensions = [f"(copy_dim{axis})" for axis in range(rank)]
        dimensions.append(f"({model['VectorLaneCount']})")
        source_terms = [
            f"(source_base{axis} + idx{axis}) * "
            f"{_dim(model['SourceStrides'][axis])}"
            for axis in range(rank)
        ]
        destination_terms = [
            f"(destination_base{axis} + idx{axis}) * "
            f"{_dim(model['DestinationStrides'][axis])}"
            for axis in range(rank)
        ]
        source_offset = (
            "tensor_linear * 0" if not source_terms else " + ".join(source_terms)
        )
        destination_offset = (
            "tensor_linear * 0"
            if not destination_terms
            else " + ".join(destination_terms)
        )
        if model["VectorLaneCount"] != 1:
            source_offset = (
                f"(({source_offset}) * {model['VectorLaneCount']} + vector_lane)"
            )
            destination_offset = (
                f"(({destination_offset}) * {model['VectorLaneCount']} + vector_lane)"
            )
        context.update(
            destination_offset=destination_offset,
            source_offset=source_offset,
            total=" * ".join(dimensions),
        )
        return context

    descriptor_shape = plan["descriptor_shape"]
    lane_count = int(model["VectorLaneCount"])
    local_descriptor = _local_buffer_value(
        plan["local_buffer"], "DescriptorExpression"
    )
    if not local_descriptor:
        raise ValueError("TensorRegionCopy local buffer is missing DescriptorExpression")
    expanded_indices = []
    for axis, extent in enumerate(descriptor_shape):
        if len(descriptor_shape) == 1:
            expanded_indices.append(f"tl.arange(0, {extent})")
        else:
            suffix = (
                "["
                + ", ".join(
                    ":" if position == axis else "None"
                    for position in range(len(descriptor_shape))
                )
                + "]"
            )
            expanded_indices.append(f"tl.arange(0, {extent}){suffix}")
    linear_terms = []
    inner_stride = math.prod(descriptor_shape)
    for axis, extent in enumerate(descriptor_shape):
        inner_stride //= extent
        term = f"copy_desc_idx{axis}"
        if inner_stride != 1:
            term = f"{term} * {inner_stride}"
        linear_terms.append(term)
    global_base = f"{plan['global_name']}_base"
    global_strides = model[f"{plan['global_model_name']}Strides"]
    global_terms = [
        f"({global_base}{axis} + copy_idx{axis}) * "
        f"{_dim(global_strides[axis])}"
        for axis in range(len(plan["static_shape"]))
    ]
    global_offset = (
        "copy_tensor_linear * 0" if not global_terms else " + ".join(global_terms)
    )
    if lane_count != 1:
        global_offset = f"(({global_offset}) * {lane_count} + copy_vector_lane)"
    context.update(
        copy_shape=f"[{', '.join(str(extent) for extent in descriptor_shape)}]",
        expanded_indices=tuple(expanded_indices),
        global_offset=global_offset,
        lane_count=lane_count,
        linear_expression=" + ".join(linear_terms),
        local_descriptor=local_descriptor,
    )
    return context


def _tensor_copy_template_context(
    model: dict[str, Any], *, is_load: bool
) -> dict[str, Any]:
    """Prepare TensorLoad/TensorStore address expressions for Jinja."""

    local_shape = model["LocalShape"]
    global_shape = model["GlobalShape"]
    local_strides = model["DestinationStrides" if is_load else "SourceStrides"]
    explicit_global_strides = model.get(
        "SourceStrides" if is_load else "DestinationStrides"
    )
    global_strides = explicit_global_strides or _contiguous_strides(global_shape)
    block_axis = _select_block_axis(local_shape, local_strides)
    vector_lane_count = int(model.get("VectorLaneCount", 1))

    def axis_index(axis: int) -> str:
        return "lane" if axis == block_axis else f"idx{axis}"

    local_terms = [
        f"{axis_index(axis)} * {_dim(local_strides[axis])}"
        for axis in range(len(local_shape))
    ]
    local_offset = "0" if not local_terms else " + ".join(local_terms)
    global_terms = [
        f"global_idx{axis} * {_dim(global_strides[axis])}"
        for axis in range(len(global_shape))
    ]
    global_offset = "0" if not global_terms else " + ".join(global_terms)

    def scalar_offset(value: str) -> str:
        return (
            value
            if vector_lane_count == 1
            else f"(({value}) * {vector_lane_count} + vector_lane)"
        )

    internal_source = model.get("Source") if is_load else None
    internal_destination = model.get("Destination") if not is_load else None
    if is_load:
        source_pool_offset = (
            "0"
            if internal_source is not None
            else "source_pool_stride_elements * shard_index"
        )
        source_offsets = (
            f"{source_pool_offset} + {model['SourceOffset']} + "
            f"{scalar_offset(global_offset)}"
        )
        destination_offsets = scalar_offset(local_offset)
    else:
        destination_pool_offset = (
            "0"
            if internal_destination is not None
            else "destination_pool_stride_elements * shard_index"
        )
        source_offsets = scalar_offset(local_offset)
        destination_offsets = (
            f"{destination_pool_offset} + {model['DestinationOffset']} + "
            f"{scalar_offset(global_offset)}"
        )

    return {
        "block_axis": block_axis,
        "block_extent": _one() if not local_shape else local_shape[block_axis],
        "destination_offsets": destination_offsets,
        "global_shape": global_shape,
        "internal_destination": internal_destination,
        "internal_source": internal_source,
        "is_load": is_load,
        "local_shape": local_shape,
        "loop_axes": tuple(
            axis for axis in range(len(local_shape)) if axis != block_axis
        ),
        "source_offsets": source_offsets,
        "vector_lane_count": vector_lane_count,
    }


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


def _vector_layout_template_context(model: dict[str, Any]) -> dict[str, Any]:
    """Prepare pack/unpack lane decomposition without owning kernel flow."""

    input_lane_count = _product_int(model["InputLanes"])
    output_lane_count = _product_int(model["OutputLanes"])
    domain_shape = model["OutputShape"] if model["IsPack"] else model["InputShape"]
    domain_lane_count = output_lane_count if model["IsPack"] else input_lane_count

    def axis_lane_indices(axis: int) -> list[int]:
        return [
            lane_index
            for lane_index, packed_axis in enumerate(model["Axes"])
            if packed_axis == axis
        ]

    def axis_lane_product(indices: list[int]) -> int:
        return _product_int([model["Lanes"][lane_index] for lane_index in indices])

    axis_infos = []
    rank = len(model["InputShape"] if model["IsPack"] else model["OutputShape"])
    for axis in range(rank):
        indices = axis_lane_indices(axis)
        assignments = []
        terms = []
        for index, lane_index in enumerate(indices):
            suffix_stride = _product_int(model["Lanes"][lane_index + 1 :])
            lane_expression = (
                f"(new_lane_group) % {model['Lanes'][lane_index]}"
                if suffix_stride == 1
                else f"((new_lane_group) // {suffix_stride}) % {model['Lanes'][lane_index]}"
            )
            lane_name = f"lane{lane_index}"
            lane_stride = axis_lane_product(indices[index + 1 :])
            assignments.append((lane_name, lane_expression))
            terms.append(
                lane_name if lane_stride == 1 else f"{lane_name} * {lane_stride}"
            )
        axis_infos.append(
            {
                "assignments": tuple(assignments),
                "lane_offset": "0" if not terms else " + ".join(terms),
                "lane_product": axis_lane_product(indices),
                "packed": bool(indices),
            }
        )

    def tensor_offset(
        prefix: str, strides: list[Any], lane_count: int, lane_flat: str
    ) -> str:
        terms = [
            f"{prefix}{axis} * {_dim(stride)}"
            for axis, stride in enumerate(strides)
        ]
        tensor = "lane_flat * 0" if not terms else " + ".join(terms)
        return (
            tensor
            if lane_count == 1
            else f"(({tensor}) * {lane_count} + {lane_flat})"
        )

    return {
        "axis_infos": tuple(axis_infos),
        "domain_lane_count": domain_lane_count,
        "domain_shape": domain_shape,
        "input_lane_count": input_lane_count,
        "input_lane_flat": (
            "lane_flat * 0"
            if input_lane_count == 1
            else f"lane_flat % {input_lane_count}"
        ),
        "input_offset": tensor_offset(
            "in_idx", model["InputStrides"], input_lane_count,
            "input_lane_flat" if model["IsPack"] else "lane_flat",
        ),
        "new_lane_group": (
            "lane_flat"
            if (input_lane_count if model["IsPack"] else output_lane_count) == 1
            else f"lane_flat // {input_lane_count if model['IsPack'] else output_lane_count}"
        ),
        "op": "pack" if model["IsPack"] else "unpack",
        "output_lane_count": output_lane_count,
        "output_lane_flat": (
            "lane_flat * 0"
            if output_lane_count == 1
            else f"lane_flat % {output_lane_count}"
        ),
        "output_offset": tensor_offset(
            "out_idx", model["OutputStrides"], output_lane_count,
            "lane_flat" if model["IsPack"] else "output_lane_flat",
        ),
        "total": _multiply_expr(_product(domain_shape), domain_lane_count),
    }




def _logical_n_index(expression: str, packed_lane_count: int, vector_lane_count: int) -> str:
    scalar_lane_count = packed_lane_count * vector_lane_count
    return expression if scalar_lane_count == 1 else f"(({expression}) // {scalar_lane_count})"


def _logical_packed_lane_index(expression: str, packed_lane_count: int, vector_lane_count: int) -> str:
    return "0" if packed_lane_count == 1 else f"((({expression}) // {vector_lane_count}) % {packed_lane_count})"


def _logical_vector_lane_index(expression: str, vector_lane_count: int) -> str:
    return "0" if vector_lane_count == 1 else f"(({expression}) % {vector_lane_count})"


def _maybe_vectorized_offset(physical_offset: str, packed_lane_index: str, vector_lane_index: str, packed_lane_count: int, vector_lane_count: int) -> str:
    scalar_lane_count = packed_lane_count * vector_lane_count
    return physical_offset if scalar_lane_count == 1 else f"((({physical_offset}) * {packed_lane_count} + {packed_lane_index}) * {vector_lane_count} + {vector_lane_index})"


def _batch_offset_expression(operand_shape: list[Any], operand_strides: list[Any], output_batch_rank: int) -> str:
    operand_batch_rank = len(operand_shape) - 2
    axis_offset = output_batch_rank - operand_batch_rank
    terms = []
    for operand_axis in range(operand_batch_rank):
        if _is_fixed_one(operand_shape[operand_axis]):
            continue
        output_axis = axis_offset + operand_axis
        terms.append(f"idx{output_axis} * {_dim(operand_strides[operand_axis])}")
    return "0" if not terms else " + ".join(terms)


def _qkv_parallel_linear_template_context(
    model: dict[str, Any], *, packed: bool
) -> dict[str, Any]:
    """Prepare QKV projection tiles and addresses for its Jinja template."""

    phase = str(model.get("ReductionPhase", "complete")).lower()
    if phase not in {"complete", "accumulate", "finalize"}:
        prefix = "Packed" if packed else ""
        raise ValueError(
            f"Unsupported {prefix}QKVParallelLinear reduction phase: {phase!r}."
        )
    template_name = "PackedQKVParallelLinear" if packed else "QKVParallelLinear"
    output_shapes = {
        prefix: _qkv_reduction_logical_output_shape(model, prefix, packed=packed)
        for prefix in ("Q", "K", "V")
    }
    context: dict[str, Any] = {
        "packed": packed,
        "phase": phase,
        "template_name": template_name,
    }

    if phase == "accumulate":
        if any(len(shape) != 2 for shape in output_shapes.values()):
            raise ValueError(
                "QKVParallelLinear backend-private reduction requires rank-2 "
                "local output tiles."
            )
        block_m = int(model["ReductionBlockM"])
        block_k = int(model["ReductionBlockK"])
        k = model["InputShape"][-1]
        m = output_shapes["Q"][-2]
        if block_m == 1:
            input_offset = f"offs_k * {_dim(model['InputStrides'][-1])}"
            input_mask = (
                f"(0 < {_dim(m)}) & (0 < {_dim(model['InputShape'][-2])}) & "
                f"(offs_k < {_dim(k)})"
            )
        else:
            input_offset = (
                f"(offs_m[:, None] * {_dim(model['InputStrides'][-2])} + "
                f"offs_k[None, :] * {_dim(model['InputStrides'][-1])})"
            )
            input_mask = (
                f"(offs_m[:, None] < {_dim(m)}) & "
                f"(offs_m[:, None] < {_dim(model['InputShape'][-2])}) & "
                f"(offs_k[None, :] < {_dim(k)})"
            )
        projections = []
        for prefix, accumulator in (
            ("Q", "q_acc"),
            ("K", "k_acc"),
            ("V", "v_acc"),
        ):
            lower = prefix.lower()
            block_n = int(model[f"Reduction{prefix}BlockN"])
            n = output_shapes[prefix][-1]
            weight_shape = model[f"{prefix}WeightShape"]
            weight_strides = model[f"{prefix}WeightStrides"]
            weight_k = weight_shape[-1] if packed else weight_shape[-2]
            offs_n = f"{lower}_offs_n"
            if packed:
                weight_offset = _packed_qkv_weight_offsets(
                    model,
                    prefix,
                    "0",
                    f"{offs_n}[:, None]"
                    if block_m == 1
                    else f"{offs_n}[None, :]",
                    "offs_k[None, :]" if block_m == 1 else "offs_k[:, None]",
                )
            elif block_m == 1:
                weight_offset = (
                    f"(offs_k[None, :] * {_dim(weight_strides[-2])} + "
                    f"{offs_n}[:, None] * {_dim(weight_strides[-1])})"
                )
            else:
                weight_offset = (
                    f"(offs_k[:, None] * {_dim(weight_strides[-2])} + "
                    f"{offs_n}[None, :] * {_dim(weight_strides[-1])})"
                )
            if block_m == 1:
                weight_mask = (
                    f"({offs_n}[:, None] < {_dim(n)}) & "
                    f"(offs_k[None, :] < {_dim(k)}) & "
                    f"(offs_k[None, :] < {_dim(weight_k)})"
                )
            else:
                weight_mask = (
                    f"(offs_k[:, None] < {_dim(k)}) & "
                    f"(offs_k[:, None] < {_dim(weight_k)}) & "
                    f"({offs_n}[None, :] < {_dim(n)})"
                )
            projections.append(
                {
                    "accumulator": accumulator,
                    "block_n": block_n,
                    "lower": lower,
                    "n": n,
                    "offs_n": offs_n,
                    "prefix": prefix,
                    "weight_mask": weight_mask,
                    "weight_offset": weight_offset,
                }
            )
        context.update(
            block_k=block_k,
            block_m=block_m,
            dot_precision=(
                ', input_precision="ieee"'
                if model["InputDType"] == "float32"
                and model["WeightDType"] == "float32"
                else ""
            ),
            input_mask=input_mask,
            input_offset=input_offset,
            projections=tuple(projections),
        )
        return context

    if phase == "finalize":
        block_m = int(model["ReductionBlockM"])
        projections = []
        pointer_values = [model[f"{prefix}Output"] for prefix in ("Q", "K", "V")]
        pointer_values.extend(
            model[f"{prefix}Bias"]
            for prefix in ("Q", "K", "V")
            if model[f"Has{prefix}Bias"]
        )
        for prefix, accumulator in (
            ("Q", "q_acc"),
            ("K", "k_acc"),
            ("V", "v_acc"),
        ):
            lower = prefix.lower()
            block_n = int(model[f"Reduction{prefix}BlockN"])
            n = output_shapes[prefix][-1]
            offs_n = f"{lower}_offs_n"
            bias_offset = None
            if model[f"Has{prefix}Bias"]:
                bias_offset = (
                    _packed_qkv_bias_offsets(model, prefix, offs_n)
                    if packed
                    else f"{offs_n} * {_dim(model[f'{prefix}BiasStrides'][-1])}"
                )
            if block_m == 1:
                output_offset = (
                    _packed_qkv_output_offsets(model, prefix, "0", offs_n, "0")
                    if packed
                    else (
                        f"(0 * {_dim(model[f'{prefix}OutputStrides'][-2])} + "
                        f"{offs_n} * {_dim(model[f'{prefix}OutputStrides'][-1])})"
                    )
                )
                output_mask = f"{offs_n} < {_dim(n)}"
            else:
                output_offset = (
                    _packed_qkv_output_offsets(
                        model,
                        prefix,
                        "0",
                        f"{offs_n}[None, :]",
                        "offs_m[:, None]",
                    )
                    if packed
                    else (
                        f"(offs_m[:, None] * "
                        f"{_dim(model[f'{prefix}OutputStrides'][-2])} + "
                        f"{offs_n}[None, :] * "
                        f"{_dim(model[f'{prefix}OutputStrides'][-1])})"
                    )
                )
                output_mask = (
                    f"(offs_m[:, None] < {_dim(output_shapes[prefix][-2])}) & "
                    f"({offs_n}[None, :] < {_dim(n)})"
                )
            projections.append(
                {
                    "accumulator": accumulator,
                    "bias_offset": bias_offset,
                    "block_n": block_n,
                    "has_bias": model[f"Has{prefix}Bias"],
                    "lower": lower,
                    "n": n,
                    "offs_n": offs_n,
                    "output_mask": output_mask,
                    "output_offset": output_offset,
                    "prefix": prefix,
                }
            )
        context.update(
            block_m=block_m,
            pointer_values=tuple(pointer_values),
            projections=tuple(projections),
        )
        return context

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
    input_batch_offset = _batch_offset_expression(
        model["InputShape"], model["InputStrides"], output_batch_rank
    )
    use_gemv = (_max_value(m) == 1) or (_fixed(m) == 1)
    if use_gemv:
        block_m = 1
        block_k = 256
        k_max = _max_value(k)
        n_max = max(
            _max_value(logical_output_shapes[prefix][-1]) or 0
            for prefix in ("Q", "K", "V")
        )
        use_large_n = k_max is not None and k_max <= block_k and n_max >= 4096
        block_n = 128 if use_large_n else 32
        n_stages = 2 if use_large_n else 1
    else:
        block_m = 16
        block_n = (
            model["NPackedLaneCount"] * model["NVectorLaneCount"]
            if packed
            else 64
        )
        block_k = 64
        n_stages = 5
    projections = []
    for prefix in ("Q", "K", "V"):
        lower = prefix.lower()
        weight_shape = model[f"{prefix}WeightShape"]
        output_shape = model[f"{prefix}OutputShape"]
        weight_strides = model[f"{prefix}WeightStrides"]
        bias_strides = model[f"{prefix}BiasStrides"]
        output_strides = model[f"{prefix}OutputStrides"]
        has_bias = model[f"Has{prefix}Bias"]
        logical_output_shape = logical_output_shapes[prefix]
        weight_batch_offset = _batch_offset_expression(
            weight_shape, weight_strides, len(output_shape) - 2
        )
        output_batch_offset = _batch_offset_expression(
            output_shape, output_strides, len(output_shape) - 2
        )
        n = logical_output_shape[-1]
        if use_gemv:
            input_terms = [] if input_batch_offset == "0" else [input_batch_offset]
            input_terms += [
                f"m_idx * {_dim(model['InputStrides'][-2])}",
                f"offs_k * {_dim(model['InputStrides'][-1])}",
            ]
            input_offset = f"({' + '.join(input_terms)})"
            if packed:
                weight_offset = _packed_qkv_weight_offsets(
                    model,
                    prefix,
                    weight_batch_offset,
                    "offs_n[:, None]",
                    "offs_k[None, :]",
                )
                output_offset = _packed_qkv_output_offsets(
                    model, prefix, output_batch_offset, "offs_n", "m_idx"
                )
                bias_offset = (
                    _packed_qkv_bias_offsets(model, prefix, "offs_n")
                    if has_bias
                    else None
                )
            else:
                weight_terms = (
                    [] if weight_batch_offset == "0" else [weight_batch_offset]
                )
                weight_terms += [
                    f"offs_k[None, :] * {_dim(weight_strides[-2])}",
                    f"offs_n[:, None] * {_dim(weight_strides[-1])}",
                ]
                weight_offset = f"({' + '.join(weight_terms)})"
                output_terms = (
                    [] if output_batch_offset == "0" else [output_batch_offset]
                )
                output_terms += [
                    f"m_idx * {_dim(output_strides[-2])}",
                    f"offs_n * {_dim(output_strides[-1])}",
                ]
                output_offset = f"({' + '.join(output_terms)})"
                bias_offset = (
                    f"offs_n * {_dim(bias_strides[-1])}" if has_bias else None
                )
            input_mask = f"offs_k < {_dim(k)}"
            weight_mask = (
                f"(offs_n[:, None] < {_dim(n)}) & "
                f"(offs_k[None, :] < {_dim(k)})"
            )
            output_mask = f"offs_n < {_dim(n)}"
            bias_mask = output_mask
        else:
            input_terms = [] if input_batch_offset == "0" else [input_batch_offset]
            input_terms += [
                f"offs_m[:, None] * {_dim(model['InputStrides'][-2])}",
                f"offs_k[None, :] * {_dim(model['InputStrides'][-1])}",
            ]
            input_offset = f"({' + '.join(input_terms)})"
            if packed:
                weight_offset = _packed_qkv_weight_offsets(
                    model,
                    prefix,
                    weight_batch_offset,
                    "offs_n[None, :]",
                    "offs_k[:, None]",
                )
                output_offset = _packed_qkv_output_offsets(
                    model,
                    prefix,
                    output_batch_offset,
                    "offs_n[None, :]",
                    "offs_m[:, None]",
                )
                bias_offset = (
                    _packed_qkv_bias_offsets(model, prefix, "offs_n")
                    if has_bias
                    else None
                )
            else:
                weight_terms = (
                    [] if weight_batch_offset == "0" else [weight_batch_offset]
                )
                weight_terms += [
                    f"offs_k[:, None] * {_dim(weight_strides[-2])}",
                    f"offs_n[None, :] * {_dim(weight_strides[-1])}",
                ]
                weight_offset = f"({' + '.join(weight_terms)})"
                output_terms = (
                    [] if output_batch_offset == "0" else [output_batch_offset]
                )
                output_terms += [
                    f"offs_m[:, None] * {_dim(output_strides[-2])}",
                    f"offs_n[None, :] * {_dim(output_strides[-1])}",
                ]
                output_offset = f"({' + '.join(output_terms)})"
                bias_offset = (
                    f"offs_n * {_dim(bias_strides[-1])}" if has_bias else None
                )
            input_mask = (
                f"(offs_m[:, None] < {_dim(logical_output_shape[-2])}) & "
                f"(offs_k[None, :] < {_dim(k)})"
            )
            weight_mask = (
                f"(offs_k[:, None] < {_dim(k)}) & "
                f"(offs_n[None, :] < {_dim(n)})"
            )
            output_mask = (
                f"(offs_m[:, None] < {_dim(logical_output_shape[-2])}) & "
                f"(offs_n[None, :] < {_dim(n)})"
            )
            bias_mask = f"offs_n < {_dim(n)}"
        projections.append(
            {
                "bias_mask": bias_mask,
                "bias_offset": bias_offset,
                "has_bias": has_bias,
                "input_mask": input_mask,
                "input_offset": input_offset,
                "lower": lower,
                "n": n,
                "output_mask": output_mask,
                "output_offset": output_offset,
                "prefix": prefix,
                "weight_mask": weight_mask,
                "weight_offset": weight_offset,
            }
        )
    context.update(
        batch_axes=tuple(range(output_batch_rank)),
        block_k=block_k,
        block_m=block_m,
        block_n=block_n,
        dot_precision=(
            ', input_precision="ieee"'
            if model["InputDType"] == "float32"
            and model["WeightDType"] == "float32"
            else ""
        ),
        k=k,
        logical_output_shapes=logical_output_shapes,
        m=m,
        n_stages=n_stages,
        projections=tuple(projections),
        use_gemv=use_gemv,
    )
    return context






def _qkv_reduction_logical_output_shape(
    model: dict[str, Any], prefix: str, *, packed: bool
) -> list[Any]:
    shape = [
        dict(dim) if isinstance(dim, dict) else dim
        for dim in model[f"{prefix}OutputShape"]
    ]
    if packed:
        shape[-1] = _multiply_dim(
            shape[-1], model["NPackedLaneCount"] * model["NVectorLaneCount"]
        )
    return shape






def _packed_qkv_logical_output_shape(model: dict[str, Any], prefix: str) -> list[Any]:
    scalar_lane_count = model["NPackedLaneCount"] * model["NVectorLaneCount"]
    shape = [dict(dim) if isinstance(dim, dict) else dim for dim in model[f"{prefix}OutputShape"]]
    shape[-1] = _multiply_dim(shape[-1], scalar_lane_count)
    return shape


def _packed_qkv_weight_offsets(model: dict[str, Any], prefix: str, weight_batch_offset: str, n_expr: str, k_expr: str) -> str:
    weight_strides = model[f"{prefix}WeightStrides"]
    terms = [] if weight_batch_offset == "0" else [weight_batch_offset]
    terms += [
        f"{_logical_n_index(n_expr, model['NPackedLaneCount'], model['NVectorLaneCount'])} * {_dim(weight_strides[-2])}",
        f"{k_expr} * {_dim(weight_strides[-1])}",
    ]
    return _maybe_vectorized_offset(
        f"({' + '.join(terms)})",
        _logical_packed_lane_index(n_expr, model["NPackedLaneCount"], model["NVectorLaneCount"]),
        _logical_vector_lane_index(n_expr, model["NVectorLaneCount"]),
        model["NPackedLaneCount"],
        model["NVectorLaneCount"],
    )


def _packed_qkv_output_offsets(model: dict[str, Any], prefix: str, output_batch_offset: str, n_expr: str, m_expr: str) -> str:
    output_strides = model[f"{prefix}OutputStrides"]
    terms = [] if output_batch_offset == "0" else [output_batch_offset]
    terms += [
        f"{m_expr} * {_dim(output_strides[-2])}",
        f"{_logical_n_index(n_expr, model['NPackedLaneCount'], model['NVectorLaneCount'])} * {_dim(output_strides[-1])}",
    ]
    return _maybe_vectorized_offset(
        f"({' + '.join(terms)})",
        _logical_packed_lane_index(n_expr, model["NPackedLaneCount"], model["NVectorLaneCount"]),
        _logical_vector_lane_index(n_expr, model["NVectorLaneCount"]),
        model["NPackedLaneCount"],
        model["NVectorLaneCount"],
    )


def _packed_qkv_bias_offsets(model: dict[str, Any], prefix: str, n_expr: str) -> str:
    bias_strides = model[f"{prefix}BiasStrides"]
    physical = f"({_logical_n_index(n_expr, model['NPackedLaneCount'], model['NVectorLaneCount'])} * {_dim(bias_strides[-1])})"
    return _maybe_vectorized_offset(
        physical,
        _logical_packed_lane_index(n_expr, model["NPackedLaneCount"], model["NVectorLaneCount"]),
        _logical_vector_lane_index(n_expr, model["NVectorLaneCount"]),
        model["NPackedLaneCount"],
        model["NVectorLaneCount"],
    )






















def _matmul_glu_template_context(
    model: dict[str, Any], *, packed: bool
) -> dict[str, Any]:
    """Prepare MatMulGlu tiles, reductions, and packed-N addresses."""

    phase = str(model.get("ReductionPhase", "complete")).lower()
    if phase not in {"complete", "accumulate", "finalize"}:
        raise ValueError(f"Unsupported MatMulGlu reduction phase: {phase!r}.")
    logical_output_shape = _matmul_glu_logical_output_shape(model)
    template_name = "PackedMatMulGlu" if packed else "MatMulGlu"
    context: dict[str, Any] = {
        "logical_output_shape": logical_output_shape,
        "packed": packed,
        "phase": phase,
        "template_name": template_name,
    }

    if phase == "accumulate":
        if len(logical_output_shape) != 2:
            raise ValueError(
                "MatMulGlu backend-private reduction requires a rank-2 local "
                "output tile."
            )
        block_m = int(model["ReductionBlockM"])
        block_n = int(model["ReductionBlockN"])
        block_k = int(model["ReductionBlockK"])
        k = model["InputShape"][-1]
        if block_m == 1:
            input_m, input_m_limit = _matmul_glu_input_m_index(model, "0")
            input_offset = (
                f"({input_m} * {_dim(model['InputStrides'][-2])} + "
                f"offs_k * {_dim(model['InputStrides'][-1])})"
            )
            input_mask = (
                f"({input_m} < {_dim(input_m_limit)}) & "
                f"(offs_k < {_dim(k)})"
            )
        else:
            input_m, input_m_limit = _matmul_glu_input_m_index(
                model, "offs_m[:, None]"
            )
            input_offset = (
                f"({input_m} * {_dim(model['InputStrides'][-2])} + "
                f"offs_k[None, :] * {_dim(model['InputStrides'][-1])})"
            )
            input_mask = (
                f"(offs_m[:, None] < {_dim(logical_output_shape[-2])}) & "
                f"({input_m} < {_dim(input_m_limit)}) & "
                f"(offs_k[None, :] < {_dim(k)})"
            )
        projections = []
        n = logical_output_shape[-1]
        for prefix, accumulator in (("Gate", "gate_acc"), ("Up", "up_acc")):
            weight_shape = model[f"{prefix}WeightShape"]
            weight_strides = model[f"{prefix}WeightStrides"]
            weight_batch_offset = _batch_offset_expression(
                weight_shape, weight_strides, len(model["OutputShape"]) - 2
            )
            weight_offset, weight_n_limit, weight_k_limit = (
                _matmul_glu_weight_offsets(
                    model,
                    prefix,
                    weight_batch_offset,
                    "offs_n[:, None]"
                    if block_m == 1
                    else "offs_n[None, :]",
                    "offs_k[None, :]" if block_m == 1 else "offs_k[:, None]",
                    packed=packed,
                )
            )
            if block_m == 1:
                weight_mask = (
                    f"(offs_n[:, None] < {_dim(n)}) & "
                    f"(offs_n[:, None] < {_dim(weight_n_limit)}) & "
                    f"(offs_k[None, :] < {_dim(weight_k_limit)})"
                )
            else:
                weight_mask = (
                    f"(offs_k[:, None] < {_dim(weight_k_limit)}) & "
                    f"(offs_n[None, :] < {_dim(n)}) & "
                    f"(offs_n[None, :] < {_dim(weight_n_limit)})"
                )
            projections.append(
                {
                    "accumulator": accumulator,
                    "lower": prefix.lower(),
                    "prefix": prefix,
                    "weight_mask": weight_mask,
                    "weight_offset": weight_offset,
                }
            )
        context.update(
            block_k=block_k,
            block_m=block_m,
            block_n=block_n,
            dot_precision=(
                ', input_precision="ieee"'
                if model["InputDType"] == "float32"
                and model["WeightDType"] == "float32"
                else ""
            ),
            input_mask=input_mask,
            input_offset=input_offset,
            projections=tuple(projections),
        )
        return context

    if phase == "finalize":
        block_m = int(model["ReductionBlockM"])
        block_n = int(model["ReductionBlockN"])
        n = logical_output_shape[-1]
        biases = []
        pointer_values = [model["GateBias"], model["UpBias"], model["Output"]]
        for prefix, accumulator in (("Gate", "gate_acc"), ("Up", "up_acc")):
            if not model[f"Has{prefix}Bias"]:
                continue
            bias_offset = _matmul_glu_bias_offsets(
                model, prefix, "offs_n", packed=packed
            )
            _, bias_n_limit = _matmul_glu_bias_n_index(
                model, prefix, "offs_n", packed=packed
            )
            biases.append(
                {
                    "accumulator": accumulator,
                    "lower": prefix.lower(),
                    "mask": (
                        f"(offs_n < {_dim(n)}) & "
                        f"(offs_n < {_dim(bias_n_limit)})"
                    ),
                    "offset": bias_offset,
                    "prefix": prefix,
                }
            )
        if block_m == 1:
            output_offset = _matmul_glu_output_offsets(
                model, "0", "offs_n", "0", packed=packed
            )
            output_mask = f"offs_n < {_dim(n)}"
        else:
            output_offset = _matmul_glu_output_offsets(
                model,
                "0",
                "offs_n[None, :]",
                "offs_m[:, None]",
                packed=packed,
            )
            output_mask = (
                f"(offs_m[:, None] < {_dim(logical_output_shape[-2])}) & "
                f"(offs_n[None, :] < {_dim(n)})"
            )
        context.update(
            biases=tuple(biases),
            block_m=block_m,
            block_n=block_n,
            output_mask=output_mask,
            output_offset=output_offset,
            pointer_values=tuple(pointer_values),
            result_expression=_matmul_glu_expr(model, "gate_acc", "up_acc"),
        )
        return context

    m = logical_output_shape[-2]
    n = logical_output_shape[-1]
    k = model["InputShape"][-1]
    output_batch_rank = len(model["OutputShape"]) - 2
    input_batch_offset = _batch_offset_expression(
        model["InputShape"], model["InputStrides"], output_batch_rank
    )
    output_batch_offset = _batch_offset_expression(
        model["OutputShape"], model["OutputStrides"], output_batch_rank
    )
    use_gemv = (_max_value(m) == 1) or (_fixed(m) == 1)
    if use_gemv:
        block_m = 1
        block_k = 256
        k_max = _max_value(k)
        n_max = _max_value(n) or 0
        use_large_n = k_max is not None and k_max <= block_k and n_max >= 4096
        block_n = 128 if use_large_n else 32
        n_stages = 2 if use_large_n else 1
        input_m, input_m_limit = _matmul_glu_input_m_index(model, "m_idx")
        input_terms = [] if input_batch_offset == "0" else [input_batch_offset]
        input_terms += [
            f"{input_m} * {_dim(model['InputStrides'][-2])}",
            f"offs_k * {_dim(model['InputStrides'][-1])}",
        ]
        input_offset = f"({' + '.join(input_terms)})"
        input_mask = (
            f"(m_idx < {_dim(input_m_limit)}) & (offs_k < {_dim(k)})"
        )
        output_offset = _matmul_glu_output_offsets(
            model, output_batch_offset, "offs_n", "m_idx", packed=packed
        )
    else:
        block_m, block_n, block_k, n_stages = 16, 64, 64, 5
        input_m, input_m_limit = _matmul_glu_input_m_index(
            model, "offs_m[:, None]"
        )
        input_terms = [] if input_batch_offset == "0" else [input_batch_offset]
        input_terms += [
            f"{input_m} * {_dim(model['InputStrides'][-2])}",
            f"offs_k[None, :] * {_dim(model['InputStrides'][-1])}",
        ]
        input_offset = f"({' + '.join(input_terms)})"
        input_mask = (
            f"(offs_m[:, None] < {_dim(m)}) & "
            f"({input_m} < {_dim(input_m_limit)}) & "
            f"(offs_k[None, :] < {_dim(k)})"
        )
        output_offset = _matmul_glu_output_offsets(
            model,
            output_batch_offset,
            "offs_n[None, :]",
            "offs_m[:, None]",
            packed=packed,
        )
    projections = []
    for prefix, accumulator in (("Gate", "gate_acc"), ("Up", "up_acc")):
        weight_shape = model[f"{prefix}WeightShape"]
        weight_strides = model[f"{prefix}WeightStrides"]
        weight_batch_offset = _batch_offset_expression(
            weight_shape, weight_strides, output_batch_rank
        )
        weight_offset, weight_n_limit, weight_k_limit = _matmul_glu_weight_offsets(
            model,
            prefix,
            weight_batch_offset,
            "offs_n[:, None]" if use_gemv else "offs_n[None, :]",
            "offs_k[None, :]" if use_gemv else "offs_k[:, None]",
            packed=packed,
        )
        if use_gemv:
            weight_mask = (
                f"(offs_n[:, None] < {_dim(n)}) & "
                f"(offs_n[:, None] < {_dim(weight_n_limit)}) & "
                f"(offs_k[None, :] < {_dim(weight_k_limit)})"
            )
        else:
            weight_mask = (
                f"(offs_k[:, None] < {_dim(weight_k_limit)}) & "
                f"(offs_n[None, :] < {_dim(n)}) & "
                f"(offs_n[None, :] < {_dim(weight_n_limit)})"
            )
        bias_offset = None
        bias_mask = None
        if model[f"Has{prefix}Bias"]:
            bias_offset = _matmul_glu_bias_offsets(
                model, prefix, "offs_n", packed=packed
            )
            _, bias_n_limit = _matmul_glu_bias_n_index(
                model, prefix, "offs_n", packed=packed
            )
            bias_mask = (
                f"(offs_n < {_dim(n)}) & "
                f"(offs_n < {_dim(bias_n_limit)})"
            )
        projections.append(
            {
                "accumulator": accumulator,
                "bias_mask": bias_mask,
                "bias_offset": bias_offset,
                "has_bias": model[f"Has{prefix}Bias"],
                "lower": prefix.lower(),
                "prefix": prefix,
                "weight_mask": weight_mask,
                "weight_offset": weight_offset,
            }
        )
    context.update(
        batch_axes=tuple(range(output_batch_rank)),
        block_k=block_k,
        block_m=block_m,
        block_n=block_n,
        dot_precision=(
            ', input_precision="ieee"'
            if model["InputDType"] == "float32"
            and model["WeightDType"] == "float32"
            else ""
        ),
        input_mask=input_mask,
        input_offset=input_offset,
        k=k,
        m=m,
        n=n,
        n_stages=n_stages,
        output_mask=(
            f"offs_n < {_dim(n)}"
            if use_gemv
            else (
                f"(offs_m[:, None] < {_dim(m)}) & "
                f"(offs_n[None, :] < {_dim(n)})"
            )
        ),
        output_offset=output_offset,
        projections=tuple(projections),
        result_expression=_matmul_glu_expr(model, "gate_acc", "up_acc"),
        use_gemv=use_gemv,
    )
    return context












def _matmul_glu_logical_output_shape(model: dict[str, Any]) -> list[Any]:
    shape = [dict(dim) if isinstance(dim, dict) else dim for dim in model["OutputShape"]]
    if model.get("PackedN"):
        shape[-1] = _multiply_dim(shape[-1], model["NPackedLaneCount"] * model["NVectorLaneCount"])
    return shape


def _matmul_glu_input_m_index(model: dict[str, Any], m_expr: str) -> tuple[str, Any]:
    return m_expr, model["InputShape"][-2]


def _matmul_glu_weight_k_index(model: dict[str, Any], prefix: str, k_expr: str, *, packed: bool) -> tuple[str, Any]:
    weight_shape = model[f"{prefix}WeightShape"]
    weight_axis = -1 if packed else -2
    return k_expr, weight_shape[weight_axis]


def _matmul_glu_weight_n_index(
    model: dict[str, Any],
    prefix: str,
    n_expr: str,
    *,
    packed: bool,
) -> tuple[str, Any]:
    weight_shape = model[f"{prefix}WeightShape"]
    weight_axis = -2 if packed else -1
    lane_scale = model["NPackedLaneCount"] * model["NVectorLaneCount"] if packed else 1
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


def _matmul_glu_weight_offsets(
    model: dict[str, Any],
    prefix: str,
    weight_batch_offset: str,
    n_expr: str,
    k_expr: str,
    *,
    packed: bool,
) -> tuple[str, str, str]:
    weight_strides = model[f"{prefix}WeightStrides"]
    weight_n, weight_n_limit = _matmul_glu_weight_n_index(model, prefix, n_expr, packed=packed)
    weight_k, weight_k_limit = _matmul_glu_weight_k_index(model, prefix, k_expr, packed=packed)
    terms = [] if weight_batch_offset == "0" else [weight_batch_offset]
    if not packed:
        terms += [
            f"{weight_k} * {_dim(weight_strides[-2])}",
            f"{weight_n} * {_dim(weight_strides[-1])}",
        ]
        return f"({' + '.join(terms)})", weight_n_limit, weight_k_limit

    terms += [
        f"{_logical_n_index(weight_n, model['NPackedLaneCount'], model['NVectorLaneCount'])} * {_dim(weight_strides[-2])}",
        f"{weight_k} * {_dim(weight_strides[-1])}",
    ]
    return _maybe_vectorized_offset(
        f"({' + '.join(terms)})",
        _logical_packed_lane_index(weight_n, model["NPackedLaneCount"], model["NVectorLaneCount"]),
        _logical_vector_lane_index(weight_n, model["NVectorLaneCount"]),
        model["NPackedLaneCount"],
        model["NVectorLaneCount"],
    ), weight_n_limit, weight_k_limit


def _matmul_glu_output_offsets(model: dict[str, Any], output_batch_offset: str, n_expr: str, m_expr: str, *, packed: bool) -> str:
    output_strides = model["OutputStrides"]
    terms = [] if output_batch_offset == "0" else [output_batch_offset]
    n_index = _logical_n_index(n_expr, model["NPackedLaneCount"], model["NVectorLaneCount"]) if packed else n_expr
    terms += [
        f"{m_expr} * {_dim(output_strides[-2])}",
        f"{n_index} * {_dim(output_strides[-1])}",
    ]
    physical = f"({' + '.join(terms)})"
    if not packed:
        return physical
    return _maybe_vectorized_offset(
        physical,
        _logical_packed_lane_index(n_expr, model["NPackedLaneCount"], model["NVectorLaneCount"]),
        _logical_vector_lane_index(n_expr, model["NVectorLaneCount"]),
        model["NPackedLaneCount"],
        model["NVectorLaneCount"],
    )


def _matmul_glu_bias_offsets(model: dict[str, Any], prefix: str, n_expr: str, *, packed: bool) -> str:
    bias_strides = model[f"{prefix}BiasStrides"]
    bias_n, _ = _matmul_glu_bias_n_index(model, prefix, n_expr, packed=packed)
    if not packed:
        return f"{bias_n} * {_dim(bias_strides[-1])}"
    physical = f"({_logical_n_index(bias_n, model['NPackedLaneCount'], model['NVectorLaneCount'])} * {_dim(bias_strides[-1])})"
    return _maybe_vectorized_offset(
        physical,
        _logical_packed_lane_index(bias_n, model["NPackedLaneCount"], model["NVectorLaneCount"]),
        _logical_vector_lane_index(bias_n, model["NVectorLaneCount"]),
        model["NPackedLaneCount"],
        model["NVectorLaneCount"],
    )






def _matmul_template_context(
    model: dict[str, Any], *, gemv: bool
) -> dict[str, Any]:
    """Prepare Matmul/Gemv dimensions and addresses for Jinja-owned kernels."""

    reduction_phase = str(model.get("ReductionPhase", "complete")).lower()
    if reduction_phase not in ("complete", "accumulate", "finalize"):
        raise ValueError(f"Unsupported Matmul reduction phase: {reduction_phase!r}.")
    if reduction_phase == "finalize":
        gemv = bool(model.get("Gemv", gemv))

    output_lane_count = (
        model.get("OutputNPackedLaneCount", 1)
        * model["OutputNVectorLaneCount"]
    )
    logical_output_shape = [
        dict(value) if isinstance(value, dict) else value
        for value in model["OutputShape"]
    ]
    logical_output_shape[-1] = _multiply_dim(
        logical_output_shape[-1], output_lane_count
    )
    context: dict[str, Any] = {
        "gemv": gemv,
        "logical_output_shape": logical_output_shape,
        "phase": reduction_phase,
        "template_name": "Gemv" if gemv else "Matmul",
    }

    if reduction_phase == "finalize":
        block_m = int(model["ReductionBlockM"])
        block_n = int(model["ReductionBlockN"])
        n = logical_output_shape[-1]
        n_expression = "offs_n" if gemv else "offs_n[None, :]"
        physical_n = _logical_n_index(
            n_expression,
            model.get("OutputNPackedLaneCount", 1),
            model["OutputNVectorLaneCount"],
        )
        m_expression = "0" if gemv else "offs_m[:, None]"
        physical = (
            f"({m_expression} * {_dim(model['OutputStrides'][-2])} + "
            f"{physical_n} * {_dim(model['OutputStrides'][-1])})"
        )
        output_offsets = _maybe_vectorized_offset(
            physical,
            _logical_packed_lane_index(
                n_expression,
                model.get("OutputNPackedLaneCount", 1),
                model["OutputNVectorLaneCount"],
            ),
            _logical_vector_lane_index(
                n_expression, model["OutputNVectorLaneCount"]
            ),
            model.get("OutputNPackedLaneCount", 1),
            model["OutputNVectorLaneCount"],
        )
        context.update(
            block_m=block_m,
            block_n=block_n,
            n=n,
            output_mask=(
                f"offs_n < {_dim(n)}"
                if gemv
                else f"(offs_m[:, None] < {_dim(logical_output_shape[-2])}) & "
                f"(offs_n[None, :] < {_dim(n)})"
            ),
            output_offsets=output_offsets,
        )
        return context

    rhs_lane_count = (
        model.get("RhsNPackedLaneCount", 1) * model["RhsNVectorLaneCount"]
    )
    m = logical_output_shape[-2]
    n = logical_output_shape[-1]
    lhs_m = (
        model["LhsShape"][-1] if model["TransposeA"] else model["LhsShape"][-2]
    )
    lhs_k = (
        model["LhsShape"][-2] if model["TransposeA"] else model["LhsShape"][-1]
    )
    rhs_k = (
        model["RhsShape"][-1] if model["TransposeB"] else model["RhsShape"][-2]
    )
    rhs_n = _multiply_dim(
        model["RhsShape"][-2]
        if model["TransposeB"]
        else model["RhsShape"][-1],
        rhs_lane_count,
    )
    context.update(m=m, n=n, lhs_m=lhs_m, lhs_k=lhs_k, rhs_k=rhs_k, rhs_n=rhs_n)

    if reduction_phase == "accumulate":
        if len(logical_output_shape) != 2:
            raise ValueError(
                "Matmul backend-private reduction requires a rank-2 local output tile."
            )
        block_m = int(model["ReductionBlockM"])
        block_n = int(model["ReductionBlockN"])
        block_k = int(model["ReductionBlockK"])
        if gemv:
            lhs_offsets = _gemv_lhs_offsets(model, "0")
            rhs_offsets = _gemv_rhs_offsets(model, "0")
            rhs_mask = (
                f"(offs_n[:, None] < {_dim(n)}) & "
                f"(offs_n[:, None] < {_dim(rhs_n)}) & "
                f"(offs_k[None, :] < {_dim(lhs_k)}) & "
                f"(offs_k[None, :] < {_dim(rhs_k)})"
            )
            lhs_mask = (
                f"(0 < {_dim(m)}) & (0 < {_dim(lhs_m)}) & "
                f"(offs_k < {_dim(lhs_k)})"
            )
        else:
            lhs_offsets, rhs_offsets, _ = _matmul_offsets(model, "0", "0", "0")
            rhs_mask = (
                f"(offs_k[:, None] < {_dim(lhs_k)}) & "
                f"(offs_k[:, None] < {_dim(rhs_k)}) & "
                f"(offs_n[None, :] < {_dim(n)}) & "
                f"(offs_n[None, :] < {_dim(rhs_n)})"
            )
            lhs_mask = (
                f"(offs_m[:, None] < {_dim(m)}) & "
                f"(offs_m[:, None] < {_dim(lhs_m)}) & "
                f"(offs_k[None, :] < {_dim(lhs_k)})"
            )
        context.update(
            block_k=block_k,
            block_m=block_m,
            block_n=block_n,
            dot_precision=(
                ', input_precision="ieee"'
                if model["LhsDType"] == "float32"
                and model["RhsDType"] == "float32"
                else ""
            ),
            lhs_mask=lhs_mask,
            lhs_offsets=lhs_offsets,
            rhs_mask=rhs_mask,
            rhs_offsets=rhs_offsets,
        )
        return context

    k = lhs_k
    output_batch_rank = len(logical_output_shape) - 2
    lhs_batch_offset = _batch_offset_expression(
        model["LhsShape"], model["LhsStrides"], output_batch_rank
    )
    rhs_batch_offset = _batch_offset_expression(
        model["RhsShape"], model["RhsStrides"], output_batch_rank
    )
    output_batch_offset = _batch_offset_expression(
        model["OutputShape"], model["OutputStrides"], output_batch_rank
    )
    load_c_expression = str(model.get("LoadCExpression", "False")).strip() or "False"
    load_c = load_c_expression not in ("False", "false", "0")
    load_c_predicate = (
        "True" if load_c_expression in ("True", "true", "1") else f"({load_c_expression})"
    )
    context.update(
        batch_axes=tuple(range(output_batch_rank)),
        k=k,
        load_c=load_c,
        load_c_expression=load_c_expression,
        load_c_predicate=load_c_predicate,
    )
    if gemv:
        block_k = 256
        k_max = _max_value(k)
        n_min = _min_value(n) or 0
        n_max = _max_value(n) or 0
        use_large_n = k_max is not None and k_max <= block_k and n_max >= 4096
        block_n = 128 if use_large_n else 32
        n_stages = (
            2
            if use_large_n
            else (
                3
                if k_max is not None
                and k_max <= block_k
                and n_min >= block_n * 4
                else 1
            )
        )
        context.update(
            block_k=block_k,
            block_n=block_n,
            lhs_mask=f"(m_idx < {_dim(lhs_m)}) & (offs_k < {_dim(k)})",
            lhs_offsets=_gemv_lhs_offsets(model, lhs_batch_offset),
            n_stages=n_stages,
            output_mask=f"offs_n < {_dim(n)}",
            output_offsets=_gemv_output_offsets(model, output_batch_offset),
            rhs_mask=(
                f"(offs_n[:, None] < {_dim(n)}) & "
                f"(offs_n[:, None] < {_dim(rhs_n)}) & "
                f"(offs_k[None, :] < {_dim(k)}) & "
                f"(offs_k[None, :] < {_dim(rhs_k)})"
            ),
            rhs_offsets=_gemv_rhs_offsets(model, rhs_batch_offset),
        )
    else:
        block_m, block_n, block_k = 16, 64, 64
        lhs_offsets, rhs_offsets, output_offsets = _matmul_offsets(
            model, lhs_batch_offset, rhs_batch_offset, output_batch_offset
        )
        context.update(
            block_k=block_k,
            block_m=block_m,
            block_n=block_n,
            dot_precision=(
                ', input_precision="ieee"'
                if model["LhsDType"] == "float32"
                and model["RhsDType"] == "float32"
                else ""
            ),
            lhs_mask=(
                f"(offs_m[:, None] < {_dim(m)}) & "
                f"(offs_m[:, None] < {_dim(lhs_m)}) & "
                f"(offs_k[None, :] < {_dim(k)})"
            ),
            lhs_offsets=lhs_offsets,
            output_mask=(
                f"(offs_m[:, None] < {_dim(m)}) & "
                f"(offs_n[None, :] < {_dim(n)})"
            ),
            output_offsets=output_offsets,
            rhs_mask=(
                f"(offs_k[:, None] < {_dim(k)}) & "
                f"(offs_k[:, None] < {_dim(rhs_k)}) & "
                f"(offs_n[None, :] < {_dim(n)}) & "
                f"(offs_n[None, :] < {_dim(rhs_n)})"
            ),
            rhs_offsets=rhs_offsets,
        )
    return context


def _matmul_offsets(model: dict[str, Any], lhs_batch_offset: str, rhs_batch_offset: str, output_batch_offset: str) -> tuple[str, str, str]:
    lhs_terms = [] if lhs_batch_offset == "0" else [lhs_batch_offset]
    if model["TransposeA"]:
        lhs_terms += [f"offs_k[None, :] * {_dim(model['LhsStrides'][-2])}", f"offs_m[:, None] * {_dim(model['LhsStrides'][-1])}"]
    else:
        lhs_terms += [f"offs_m[:, None] * {_dim(model['LhsStrides'][-2])}", f"offs_k[None, :] * {_dim(model['LhsStrides'][-1])}"]
    rhs_terms = [] if rhs_batch_offset == "0" else [rhs_batch_offset]
    if model["TransposeB"]:
        rhs_terms += [
            f"{_logical_n_index('offs_n[None, :]', model.get('RhsNPackedLaneCount', 1), model['RhsNVectorLaneCount'])} * {_dim(model['RhsStrides'][-2])}",
            f"offs_k[:, None] * {_dim(model['RhsStrides'][-1])}",
        ]
    else:
        rhs_terms += [
            f"offs_k[:, None] * {_dim(model['RhsStrides'][-2])}",
            f"{_logical_n_index('offs_n[None, :]', model.get('RhsNPackedLaneCount', 1), model['RhsNVectorLaneCount'])} * {_dim(model['RhsStrides'][-1])}",
        ]
    lhs_offsets = f"({' + '.join(lhs_terms)})"
    rhs_physical = f"({' + '.join(rhs_terms)})"
    rhs_offsets = _maybe_vectorized_offset(
        rhs_physical,
        _logical_packed_lane_index("offs_n[None, :]", model.get("RhsNPackedLaneCount", 1), model["RhsNVectorLaneCount"]),
        _logical_vector_lane_index("offs_n[None, :]", model["RhsNVectorLaneCount"]),
        model.get("RhsNPackedLaneCount", 1),
        model["RhsNVectorLaneCount"],
    )
    output_terms = [] if output_batch_offset == "0" else [output_batch_offset]
    output_terms += [
        f"offs_m[:, None] * {_dim(model['OutputStrides'][-2])}",
        f"{_logical_n_index('offs_n[None, :]', model.get('OutputNPackedLaneCount', 1), model['OutputNVectorLaneCount'])} * {_dim(model['OutputStrides'][-1])}",
    ]
    output_offsets = _maybe_vectorized_offset(
        f"({' + '.join(output_terms)})",
        _logical_packed_lane_index("offs_n[None, :]", model.get("OutputNPackedLaneCount", 1), model["OutputNVectorLaneCount"]),
        _logical_vector_lane_index("offs_n[None, :]", model["OutputNVectorLaneCount"]),
        model.get("OutputNPackedLaneCount", 1),
        model["OutputNVectorLaneCount"],
    )
    return lhs_offsets, rhs_offsets, output_offsets


def _gemv_lhs_offsets(model: dict[str, Any], lhs_batch_offset: str) -> str:
    terms = [] if lhs_batch_offset == "0" else [lhs_batch_offset]
    if model["TransposeA"]:
        terms += [f"offs_k * {_dim(model['LhsStrides'][-2])}", f"m_idx * {_dim(model['LhsStrides'][-1])}"]
    else:
        terms += [f"m_idx * {_dim(model['LhsStrides'][-2])}", f"offs_k * {_dim(model['LhsStrides'][-1])}"]
    return f"({' + '.join(terms)})"


def _gemv_rhs_offsets(model: dict[str, Any], rhs_batch_offset: str) -> str:
    terms = [] if rhs_batch_offset == "0" else [rhs_batch_offset]
    n_expr = "offs_n[:, None]"
    k_expr = "offs_k[None, :]"
    if model["TransposeB"]:
        terms += [
            f"{_logical_n_index(n_expr, model.get('RhsNPackedLaneCount', 1), model['RhsNVectorLaneCount'])} * {_dim(model['RhsStrides'][-2])}",
            f"{k_expr} * {_dim(model['RhsStrides'][-1])}",
        ]
    else:
        terms += [
            f"{k_expr} * {_dim(model['RhsStrides'][-2])}",
            f"{_logical_n_index(n_expr, model.get('RhsNPackedLaneCount', 1), model['RhsNVectorLaneCount'])} * {_dim(model['RhsStrides'][-1])}",
        ]
    return _maybe_vectorized_offset(
        f"({' + '.join(terms)})",
        _logical_packed_lane_index(n_expr, model.get("RhsNPackedLaneCount", 1), model["RhsNVectorLaneCount"]),
        _logical_vector_lane_index(n_expr, model["RhsNVectorLaneCount"]),
        model.get("RhsNPackedLaneCount", 1),
        model["RhsNVectorLaneCount"],
    )


def _gemv_output_offsets(model: dict[str, Any], output_batch_offset: str) -> str:
    terms = [] if output_batch_offset == "0" else [output_batch_offset]
    terms += [
        f"m_idx * {_dim(model['OutputStrides'][-2])}",
        f"{_logical_n_index('offs_n', model.get('OutputNPackedLaneCount', 1), model['OutputNVectorLaneCount'])} * {_dim(model['OutputStrides'][-1])}",
    ]
    return _maybe_vectorized_offset(
        f"({' + '.join(terms)})",
        _logical_packed_lane_index("offs_n", model.get("OutputNPackedLaneCount", 1), model["OutputNVectorLaneCount"]),
        _logical_vector_lane_index("offs_n", model["OutputNVectorLaneCount"]),
        model.get("OutputNPackedLaneCount", 1),
        model["OutputNVectorLaneCount"],
    )


def _reduce_template_context(model: dict[str, Any]) -> dict[str, Any]:
    """Validate a reduction phase and prepare logical index expressions."""

    phase = str(model.get("ReductionPhase", "complete")).lower()
    if phase not in {"complete", "accumulate", "finalize"}:
        raise ValueError(f"Unsupported Reduce reduction phase: {phase!r}.")

    output_shape = model["OutputShape"]
    output_elements = _product(output_shape)
    context: dict[str, Any] = {
        "phase": phase,
        "output_elements": output_elements,
    }
    if phase == "finalize":
        track_element_count = bool(model.get("TrackReductionElementCount", False))
        output_terms = [
            f"out_idx{axis} * {_dim(model['OutputStrides'][axis])}"
            for axis in range(len(output_shape))
        ]
        context.update(
            block_size=int(model["ReductionBlockSize"]),
            output_offset=" + ".join(output_terms) if output_terms else "lane * 0",
            state_parameters=(
                ("acc", "reduced_element_count")
                if track_element_count
                else ("acc",)
            ),
            track_element_count=track_element_count,
        )
        return context

    axis_set = set(model["Axes"])
    output_index = 0
    input_terms = []
    for input_index in range(len(model["InputShape"])):
        if input_index in axis_set:
            if model["KeepDims"]:
                output_index += 1
            continue
        index = "lane" if phase == "complete" and output_index == _select_block_axis(
            output_shape, model["OutputStrides"]
        ) else f"out_idx{output_index}"
        input_terms.append(
            f"{index} * {_dim(model['InputStrides'][input_index])}"
        )
        output_index += 1
    input_base = "lane * 0" if not input_terms else " + ".join(input_terms)
    if phase == "complete" and input_terms:
        input_base = "lane * 0 + " + input_base
    reduce_terms = [
        f"reduce_idx{axis} * {_dim(model['InputStrides'][axis])}"
        for axis in model["Axes"]
    ]
    reduce_offset = "lane * 0" if not reduce_terms else " + ".join(reduce_terms)

    if phase == "accumulate":
        track_element_count = bool(model.get("TrackReductionElementCount", False))
        context.update(
            block_size=int(model["ReductionBlockSize"]),
            input_base=input_base,
            reduce_offset=reduce_offset,
            state_parameters=(
                ("acc", "reduced_element_count")
                if track_element_count
                else ("acc",)
            ),
            tile_element_count=_product(
                [model["InputShape"][axis] for axis in model["Axes"]]
            ),
            track_element_count=track_element_count,
        )
        return context

    rank = len(output_shape)
    block_axis = _select_block_axis(output_shape, model["OutputStrides"])
    block_extent = _one() if rank == 0 else output_shape[block_axis]

    def axis_index(axis: int) -> str:
        return "lane" if axis == block_axis else f"out_idx{axis}"

    output_terms = [
        f"{axis_index(axis)} * {_dim(model['OutputStrides'][axis])}"
        for axis in range(rank)
    ]
    context.update(
        block_axis=block_axis,
        block_extent=block_extent,
        input_base=input_base,
        loop_axes=tuple(axis for axis in range(rank) if axis != block_axis),
        output_offset=(
            "lane * 0"
            if not output_terms
            else "lane * 0 + " + " + ".join(output_terms)
        ),
        reduce_offset=reduce_offset,
    )
    return context








def _softmax_template_context(model: dict[str, Any]) -> dict[str, Any]:
    """Prepare Softmax's independent-slice and storage index expressions."""

    rank = len(model["Shape"])
    block_axis = _select_block_axis(model["Shape"], model["OutputStrides"])

    def axis_index(axis: int) -> str:
        return "lane" if axis == block_axis else f"idx{axis}"

    def offset(strides: list[Any]) -> str:
        terms = [
            f"{axis_index(axis)} * {_dim(strides[axis])}"
            for axis in range(rank)
        ]
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
            "lane * 0"
            if not slice_terms
            else "lane * 0 + " + " + ".join(slice_terms)
        ),
        "slice_offset": (
            "slice_base + axis_pos * "
            f"{_dim(model['InputStrides'][model['Axis']])}"
        ),
    }




def _layer_norm_template_context(model: dict[str, Any]) -> dict[str, Any]:
    """Prepare LayerNorm logical-lane and parameter address expressions."""

    logical_input_shape = _logical_shape(
        model["InputShape"], model["InputVectorLaneCount"]
    )
    logical_scale_shape = _logical_shape(
        model["ScaleShape"], model["ScaleVectorLaneCount"]
    )
    logical_bias_shape = _logical_shape(
        model["BiasShape"], model["BiasVectorLaneCount"]
    )
    logical_output_shape = _logical_shape(
        model["OutputShape"], model["OutputVectorLaneCount"]
    )
    rank = len(logical_output_shape)

    def axis_index(axis: int) -> str:
        return f"outer_idx{axis}" if axis < model["Axis"] else f"inner_idx{axis}"

    def tensor_offset(
        physical_shape: list[Any],
        logical_shape: list[Any],
        strides: list[Any],
        lane_count: int,
    ) -> str:
        terms = []
        for axis in range(rank):
            if _is_fixed_one(logical_shape[axis]):
                continue
            index = axis_index(axis)
            if lane_count > 1 and axis == len(physical_shape) - 1:
                index = f"(({index}) // {lane_count})"
            terms.append(f"{index} * {_dim(strides[axis])}")
        physical_index = (
            "inner_lane * 0"
            if not terms
            else "inner_lane * 0 + " + " + ".join(terms)
        )
        if lane_count == 1:
            return physical_index
        lane_index = f"(({axis_index(len(physical_shape) - 1)}) % {lane_count})"
        return f"(({physical_index}) * {lane_count} + {lane_index})"

    def parameter_offset(
        physical_shape: list[Any],
        logical_shape: list[Any],
        strides: list[Any],
        lane_count: int,
    ) -> str:
        terms = []
        for axis in range(len(logical_shape)):
            if _is_fixed_one(logical_shape[axis]):
                continue
            output_axis = model["Axis"] + axis
            index = axis_index(output_axis)
            if lane_count > 1 and axis == len(physical_shape) - 1:
                index = f"(({index}) // {lane_count})"
            terms.append(f"{index} * {_dim(strides[axis])}")
        physical_index = (
            "inner_lane * 0"
            if not terms
            else "inner_lane * 0 + " + " + ".join(terms)
        )
        if lane_count == 1:
            return physical_index
        lane_index = (
            f"(({axis_index(model['Axis'] + len(physical_shape) - 1)}) % "
            f"{lane_count})"
        )
        return f"(({physical_index}) * {lane_count} + {lane_index})"

    return {
        "bias_offset": parameter_offset(
            model["BiasShape"],
            logical_bias_shape,
            model["BiasStrides"],
            model["BiasVectorLaneCount"],
        ),
        "inner_axes": tuple(range(model["Axis"], rank)),
        "inner_size": _product(logical_output_shape[model["Axis"] :]),
        "input_offset": tensor_offset(
            model["InputShape"],
            logical_input_shape,
            model["InputStrides"],
            model["InputVectorLaneCount"],
        ),
        "logical_output_shape": logical_output_shape,
        "outer_axes": tuple(range(model["Axis"])),
        "output_offset": tensor_offset(
            model["OutputShape"],
            logical_output_shape,
            model["OutputStrides"],
            model["OutputVectorLaneCount"],
        ),
        "scale_offset": parameter_offset(
            model["ScaleShape"],
            logical_scale_shape,
            model["ScaleStrides"],
            model["ScaleVectorLaneCount"],
        ),
    }




def _norm_stats_template_context(model: dict[str, Any]) -> dict[str, Any]:
    logical_input_shape = _logical_shape(
        model["InputShape"], model["InputVectorLaneCount"]
    )
    rank = len(logical_input_shape)
    inner_axes = tuple(range(model["Axis"], rank))
    outer_axes = tuple(range(model["Axis"]))

    def axis_index(axis: int) -> str:
        return f"outer_idx{axis}" if axis < model["Axis"] else f"inner_idx{axis}"

    input_terms = []
    for axis in range(rank):
        if _is_fixed_one(logical_input_shape[axis]):
            continue
        index = axis_index(axis)
        if (
            model["InputVectorLaneCount"] > 1
            and axis == len(model["InputShape"]) - 1
        ):
            index = f"(({index}) // {model['InputVectorLaneCount']})"
        input_terms.append(f"{index} * {_dim(model['InputStrides'][axis])}")
    physical_input = (
        "inner_lane * 0"
        if not input_terms
        else "inner_lane * 0 + " + " + ".join(input_terms)
    )
    if model["InputVectorLaneCount"] == 1:
        input_offset = physical_input
    else:
        lane_index = (
            f"(({axis_index(len(model['InputShape']) - 1)}) % "
            f"{model['InputVectorLaneCount']})"
        )
        input_offset = (
            f"(({physical_input}) * {model['InputVectorLaneCount']} + {lane_index})"
        )

    def stats_offset(component: int) -> str:
        terms = []
        if component:
            terms.append(f"{component} * {_dim(model['OutputStrides'][0])}")
        for axis in outer_axes:
            if not _is_fixed_one(logical_input_shape[axis]):
                terms.append(
                    f"outer_idx{axis} * {_dim(model['OutputStrides'][axis + 1])}"
                )
        return "0" if not terms else " + ".join(terms)

    return {
        "inner_axes": inner_axes,
        "inner_size": _product(logical_input_shape[model["Axis"] :]),
        "input_offset": input_offset,
        "logical_input_shape": logical_input_shape,
        "outer_axes": outer_axes,
        "stats_offsets": (stats_offset(0), stats_offset(1)),
    }


def _norm_apply_template_context(model: dict[str, Any]) -> dict[str, Any]:
    logical_input_shape = _logical_shape(
        model["InputShape"], model["InputVectorLaneCount"]
    )
    logical_input_global_shape = _logical_shape(
        model["InputGlobalShape"], model["InputVectorLaneCount"]
    )
    logical_scale_shape = _logical_shape(
        model["ScaleShape"], model["ScaleVectorLaneCount"]
    )
    logical_bias_shape = _logical_shape(
        model["BiasShape"], model["BiasVectorLaneCount"]
    )
    logical_output_shape = _logical_shape(
        model["OutputShape"], model["OutputVectorLaneCount"]
    )
    rank = len(logical_output_shape)
    inner_axes = tuple(range(model["Axis"], rank))
    outer_axes = tuple(range(model["Axis"]))

    def axis_index(axis: int) -> str:
        return f"outer_idx{axis}" if axis < model["Axis"] else f"inner_idx{axis}"

    def tensor_offset(
        physical_shape: list[Any],
        logical_shape_value: list[Any],
        strides: list[Any],
        lane_count: int,
    ) -> str:
        terms = []
        for axis in range(rank):
            if _is_fixed_one(logical_shape_value[axis]):
                continue
            index = axis_index(axis)
            if lane_count > 1 and axis == len(physical_shape) - 1:
                index = f"(({index}) // {lane_count})"
            terms.append(f"{index} * {_dim(strides[axis])}")
        physical_index = (
            "inner_lane * 0"
            if not terms
            else "inner_lane * 0 + " + " + ".join(terms)
        )
        if lane_count == 1:
            return physical_index
        lane_index = f"(({axis_index(len(physical_shape) - 1)}) % {lane_count})"
        return f"(({physical_index}) * {lane_count} + {lane_index})"

    def parameter_offset(
        physical_shape: list[Any],
        logical_shape_value: list[Any],
        strides: list[Any],
        lane_count: int,
    ) -> str:
        terms = []
        for axis in range(len(logical_shape_value)):
            if _is_fixed_one(logical_shape_value[axis]):
                continue
            index = axis_index(model["Axis"] + axis)
            if lane_count > 1 and axis == len(physical_shape) - 1:
                index = f"(({index}) // {lane_count})"
            terms.append(f"{index} * {_dim(strides[axis])}")
        physical_index = (
            "inner_lane * 0"
            if not terms
            else "inner_lane * 0 + " + " + ".join(terms)
        )
        if lane_count == 1:
            return physical_index
        lane_index = (
            f"(({axis_index(model['Axis'] + len(physical_shape) - 1)}) % "
            f"{lane_count})"
        )
        return f"(({physical_index}) * {lane_count} + {lane_index})"

    def stats_offset(component: int) -> str:
        terms = []
        if component:
            terms.append(f"{component} * {_dim(model['StatsStrides'][0])}")
        for axis in outer_axes:
            if not _is_fixed_one(logical_output_shape[axis]):
                terms.append(
                    f"outer_idx{axis} * {_dim(model['StatsStrides'][axis + 1])}"
                )
        return "0" if not terms else " + ".join(terms)

    return {
        "bias_offset": parameter_offset(
            model["BiasShape"],
            logical_bias_shape,
            model["BiasStrides"],
            model["BiasVectorLaneCount"],
        ),
        "inner_axes": inner_axes,
        "inner_size": _product(logical_output_shape[model["Axis"] :]),
        "input_offset": tensor_offset(
            model["InputShape"],
            logical_input_shape,
            model["InputStrides"],
            model["InputVectorLaneCount"],
        ),
        "logical_output_shape": logical_output_shape,
        "normalization_size": _product(
            logical_input_global_shape[model["Axis"] :]
        ),
        "outer_axes": outer_axes,
        "output_offset": tensor_offset(
            model["OutputShape"],
            logical_output_shape,
            model["OutputStrides"],
            model["OutputVectorLaneCount"],
        ),
        "scale_offset": parameter_offset(
            model["ScaleShape"],
            logical_scale_shape,
            model["ScaleStrides"],
            model["ScaleVectorLaneCount"],
        ),
        "stats_offsets": (stats_offset(0), stats_offset(1)),
    }


def _rope_template_context(model: dict[str, Any]) -> dict[str, Any]:
    """Prepare RoPE's scalar-lane pairing and operand offsets."""

    rank = len(model["OutputShape"])
    rotary_axis = model["RotaryAxis"]
    output_lane_count = model["OutputVectorLaneCount"]
    sincos_pack_factor = int(model.get("SinCosVectorPackFactor", 1))
    total = _multiply_expr(_product(model["OutputShape"]), output_lane_count)
    rotary_extent = _multiply_expr(
        f"({_dim(model['OutputShape'][rotary_axis])})", output_lane_count
    )
    half_dim = f"(({rotary_extent}) // 2)"

    def offset(
        operand_shape: list[Any],
        strides: list[Any],
        lane_count: int,
        rotary_index: str,
        lane_index: str,
    ) -> str:
        axis_offset = rank - len(operand_shape)
        terms = []
        for axis, dimension in enumerate(operand_shape):
            if _is_fixed_one(dimension):
                continue
            output_axis = axis_offset + axis
            index = rotary_index if output_axis == rotary_axis else f"idx{output_axis}"
            terms.append(f"{index} * {_dim(strides[axis])}")
        tensor = "lane_flat * 0" if not terms else " + ".join(terms)
        return (
            tensor
            if lane_count == 1
            else f"(({tensor}) * {lane_count} + {lane_index})"
        )

    if sincos_pack_factor == 1:
        cos_rotary_index = f"idx{rotary_axis}"
        cos_lane_index = "lane_flat"
    else:
        cos_rotary_index = "sincos_rotary"
        cos_lane_index = "sincos_lane"

    output_terms = [
        f"idx{axis} * {_dim(stride)}"
        for axis, stride in enumerate(model["OutputStrides"])
    ]
    output_physical = "lane_flat * 0" if not output_terms else " + ".join(
        output_terms
    )
    output_offset = (
        output_physical
        if output_lane_count == 1
        else f"(({output_physical}) * {output_lane_count} + lane_flat)"
    )
    return {
        "cos_offset": offset(
            model["CosShape"],
            model["CosStrides"],
            model["CosVectorLaneCount"],
            cos_rotary_index,
            cos_lane_index,
        ),
        "half_dim": half_dim,
        "input_offset": offset(
            model["InputShape"],
            model["InputStrides"],
            model["InputVectorLaneCount"],
            f"idx{rotary_axis}",
            "lane_flat",
        ),
        "output_offset": output_offset,
        "paired_input_offset": offset(
            model["InputShape"],
            model["InputStrides"],
            model["InputVectorLaneCount"],
            f"paired_idx{rotary_axis}",
            "paired_lane",
        ),
        "rotary_axis": rotary_axis,
        "sin_offset": offset(
            model["SinShape"],
            model["SinStrides"],
            model["SinVectorLaneCount"],
            cos_rotary_index,
            cos_lane_index,
        ),
        "sincos_pack_factor": sincos_pack_factor,
        "total": total,
    }




def _gather_template_context(model: dict[str, Any]) -> dict[str, Any]:
    """Prepare Gather index expressions without owning its Triton control flow."""

    logical_output_shape = [
        dict(value) if isinstance(value, dict) else value
        for value in model["OutputShape"]
    ]
    logical_output_shape[-1] = _multiply_dim(
        logical_output_shape[-1], model["ValueVectorLaneCount"]
    )
    logical_output_strides = [
        dict(value) if isinstance(value, dict) else value
        for value in model["OutputStrides"]
    ]
    if model["ValueVectorLaneCount"] > 1:
        logical_output_strides[-1] = _one()
    output_rank = len(logical_output_shape)
    index_rank = len(model["IndexShape"])
    block_axis = _select_block_axis(logical_output_shape, logical_output_strides)

    def axis_index(axis: int) -> str:
        return "lane" if axis == block_axis else f"idx{axis}"

    output_terms = []
    for output_axis in range(len(model["OutputShape"])):
        index = (
            f"(({axis_index(output_axis)}) // {model['ValueVectorLaneCount']})"
            if output_axis == len(model["OutputShape"]) - 1
            and model["ValueVectorLaneCount"] > 1
            else axis_index(output_axis)
        )
        output_terms.append(
            f"{index} * {_dim(model['OutputStrides'][output_axis])}"
        )
    output_physical = (
        "lane * 0"
        if not output_terms
        else "lane * 0 + " + " + ".join(output_terms)
    )
    if model["ValueVectorLaneCount"] == 1:
        output_offset = output_physical
    else:
        lane_index = (
            f"(({axis_index(len(model['OutputShape']) - 1)}) % "
            f"{model['ValueVectorLaneCount']})"
        )
        output_offset = (
            f"(({output_physical}) * {model['ValueVectorLaneCount']} + {lane_index})"
        )

    index_terms = [
        f"{axis_index(model['Axis'] + index_axis)} * "
        f"{_dim(model['IndexStrides'][index_axis])}"
        for index_axis in range(index_rank)
    ]
    index_offset = (
        "lane * 0" if not index_terms else "lane * 0 + " + " + ".join(index_terms)
    )

    input_terms = []
    for input_axis in range(len(model["InputShape"])):
        if input_axis < model["Axis"]:
            index = axis_index(input_axis)
        elif input_axis == model["Axis"]:
            index = "local_gather_index"
        else:
            index = axis_index(input_axis + index_rank - 1)
        if (
            input_axis == len(model["InputShape"]) - 1
            and model["ValueVectorLaneCount"] > 1
        ):
            index = f"(({index}) // {model['ValueVectorLaneCount']})"
        input_terms.append(f"{index} * {_dim(model['InputStrides'][input_axis])}")
    input_physical = (
        "lane * 0" if not input_terms else "lane * 0 + " + " + ".join(input_terms)
    )
    if model["ValueVectorLaneCount"] == 1:
        input_offset = input_physical
    else:
        value_output_axis = len(model["InputShape"]) + index_rank - 2
        lane_index = (
            f"(({axis_index(value_output_axis)}) % {model['ValueVectorLaneCount']})"
        )
        input_offset = (
            f"(({input_physical}) * {model['ValueVectorLaneCount']} + {lane_index})"
        )

    gather_split_axes = model["InputSplitAxes"][model["Axis"]]
    return {
        "block_axis": block_axis,
        "block_extent": _one()
        if output_rank == 0
        else logical_output_shape[block_axis],
        "gather_split_axes": gather_split_axes,
        "index_offset": index_offset,
        "input_offset": input_offset,
        "input_split_linear": _split_linear_expression(
            gather_split_axes, model["Hierarchy"]
        ),
        "logical_output_shape": logical_output_shape,
        "loop_axes": tuple(
            axis for axis in range(output_rank) if axis != block_axis
        ),
        "output_offset": output_offset,
        "signed_index": not str(model["IndexDType"]).startswith("uint"),
    }


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
            output_terms.append(
                f"{index} * {_dim(model['OutputStrides'][axis])}"
            )
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
    copy_block_axis = _select_block_axis(
        model["OutputShape"], model["OutputStrides"]
    )
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
            axis
            for axis in range(len(model["OutputShape"]))
            if axis != copy_block_axis
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
    group = (
        "0"
        if model["Groups"] == 1
        else f"{oc} // {output_channels_per_group}"
    )
    input_channel = (
        "ic"
        if model["Groups"] == 1
        else f"({group}) * {input_channels_per_group} + ic"
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
        raise ValueError(f"Unsupported PyNTT direct reshard stage: {model.get('Stage')}")
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
        writer_active = f"({writer_active}) & (shard_coord{axis} == 0)"
    total = " * ".join(
        [f"({_dim(value)})" for value in model["InputActiveShape"]]
        + [f"({model['VectorLaneCount']})"]
    )
    destination_shard_index = _split_linear_expression(
        list(range(len(model["Hierarchy"]))),
        model["Hierarchy"],
        "destination_shard_coord",
    )
    input_offsets = _scalar_offset(
        _tensor_offset("input_idx", model["InputStrides"]),
        model["VectorLaneCount"],
    )
    output_offsets = _scalar_offset(
        _tensor_offset("output_idx", model["OutputStrides"]),
        model["VectorLaneCount"],
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
    return {
        "destination_pool_index": destination_pool_index,
        "destination_shard_index": destination_shard_index,
        "input_offsets": input_offsets,
        "input_partial_mesh_axes": tuple(sorted(input_partial_mesh_axes)),
        "input_split_mesh_axes": tuple(sorted(input_split_mesh_axes)),
        "output_broadcast_mesh_axes": output_broadcast_mesh_axes,
        "output_offsets": output_offsets,
        "output_pointer_type": _pointer_type(
            model["TritonDType"], model["OutputAddress"]["AddressSpace"]
        ),
        "partial": partial,
        "total": total,
        "writer_active": writer_active,
    }


def _tensor_offset(prefix: str, strides: list[Any]) -> str:
    terms = [f"{prefix}{axis} * {_dim(strides[axis])}" for axis in range(len(strides))]
    return "lane * 0" if not terms else "lane * 0 + " + " + ".join(terms)


def _scalar_offset(element_offset: str, vector_lane_count: int) -> str:
    return element_offset if vector_lane_count == 1 else f"(({element_offset}) * {vector_lane_count} + lane_value)"


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
    """Prepare SUMMA shard-index reconstruction and vector addresses."""

    def output_axis_range(global_extent: Any, split_axes: list[int]) -> dict[str, Any]:
        return {
            "divisor": _split_divisor(split_axes, model["Hierarchy"])
            if split_axes
            else 1,
            "global_extent": global_extent,
            "split_axes": tuple(split_axes),
            "split_linear": _split_linear_expression(
                split_axes, model["Hierarchy"]
            )
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

    output_global_physical_n = model["OutputGlobalShape"][1]
    output_global_logical_n = _multiply_dim(
        output_global_physical_n, model["OutputNVectorLaneCount"]
    )
    rhs_global_logical_n = _multiply_dim(
        model["RhsGlobalShape"][1], model["RhsNVectorLaneCount"]
    )
    rhs_offset = (
        "rhs_physical_offsets"
        if model["RhsNVectorLaneCount"] == 1
        else (
            f"((rhs_physical_offsets) * {model['RhsNVectorLaneCount']} + "
            "rhs_lane[None, :])"
        )
    )
    output_offset = (
        "output_physical_offsets"
        if model["OutputNVectorLaneCount"] == 1
        else (
            f"((output_physical_offsets) * {model['OutputNVectorLaneCount']} + "
            "out_lane[None, :])"
        )
    )
    return {
        "block_k": 32,
        "block_m": 16,
        "block_n": 16,
        "dot_precision": (
            ', input_precision="ieee"'
            if model["LhsDType"] == "float32"
            and model["RhsDType"] == "float32"
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
        "output_global_logical_n": output_global_logical_n,
        "output_offset": output_offset,
        "output_pointer_type": _pointer_type(
            model["OutputTritonDType"], model["OutputAddressSpace"]
        ),
        "rhs_global_logical_n": rhs_global_logical_n,
        "rhs_k": local_index(
            "rhs_k",
            "global_k[:, None]",
            model["RhsGlobalShape"][0],
            model["RhsSplitAxes"][0],
        ),
        "rhs_n": local_index(
            "rhs_n",
            "rhs_global_n_physical[None, :]",
            model["RhsGlobalShape"][1],
            model["RhsSplitAxes"][1],
        ),
        "rhs_offset": rhs_offset,
        "rhs_pointer_type": _pointer_type(
            model["RhsTritonDType"], model["RhsAddressSpace"]
        ),
    }




def _paged_attention_template_context(model: dict[str, Any]) -> dict[str, Any]:
    """Validate PagedAttention cache layout and prepare vector addresses."""

    cache = model["Cache"]
    attention_block_size = int(model["AttentionBlockSize"])
    if (
        attention_block_size <= 0
        or attention_block_size & (attention_block_size - 1)
        or attention_block_size > int(cache["BlockSize"])
    ):
        raise ValueError(
            "PyNTT PagedAttention AttentionBlockSize must be a positive power "
            "of two no larger than the cache block size, got "
            f"{attention_block_size}."
        )

    def tensor_offset(
        strides: list[Any],
        seq_axis: int,
        head_axis: int,
        dim_axis: int,
        head: str,
        dim_block: str,
        token: str,
        lane: str,
        lane_count: int,
    ) -> str:
        indices = ["0"] * len(strides)
        indices[seq_axis] = token
        indices[head_axis] = head
        indices[dim_axis] = dim_block
        terms = [
            f"{indices[axis]} * {_dim(strides[axis])}"
            for axis in range(len(strides))
        ]
        return f"(({' + '.join(terms)}) * {lane_count} + {lane})"

    def global_index_expression(
        axis: int, local_index: str, global_extent: Any
    ) -> str:
        split_axes = model["OutputSplitAxes"][axis]
        if not split_axes:
            return local_index
        divisor = _split_divisor(split_axes, model["Hierarchy"])
        return (
            f"{local_index} + "
            f"({_split_linear_expression(split_axes, model['Hierarchy'])}) * "
            f"tl.cdiv({_dim(global_extent)}, {divisor})"
        )

    def cache_dim_index(prefix: str, dim_name: str) -> str:
        if cache[f"{prefix}VectorizedDim"] == 5:
            return f"{dim_name} // {cache[f'{prefix}LaneCount']}"
        return dim_name

    def cache_block_index(prefix: str, block_name: str) -> str:
        if cache[f"{prefix}VectorizedDim"] == 3:
            return f"{block_name} // {cache[f'{prefix}LaneCount']}"
        return block_name

    def cache_lane(prefix: str, dim_name: str, block_name: str) -> str:
        if cache[f"{prefix}VectorizedDim"] == 5:
            return f"{dim_name} % {cache[f'{prefix}LaneCount']}"
        if cache[f"{prefix}VectorizedDim"] == 3:
            return f"{block_name} % {cache[f'{prefix}LaneCount']}"
        return "0"

    if cache["KeyVectorizedDim"] != 5:
        raise ValueError(
            "PyNTT PagedAttention currently requires key cache to be "
            "HeadDim-vectorized."
        )

    local_query_tokens = model["OutputShape"][model["SeqAxis"]]
    global_query_tokens = model["OutputGlobalShape"][model["SeqAxis"]]
    key_block_index = (
        "(topology_id[None, :] * num_blocks_per_shard + block_id[None, :])"
        if cache["IdLength"] > 1
        else "block_id[None, :]"
    )
    value_block_index = (
        "(topology_id[:, None] * num_blocks_per_shard + block_id[:, None])"
        if cache["IdLength"] > 1
        else "block_id[:, None]"
    )
    key_lane = (
        "key_lane[:, None]"
        if cache["KeyVectorizedDim"] == 5
        else "key_lane[None, :]"
        if cache["KeyVectorizedDim"] == 3
        else "0"
    )
    value_lane = (
        "value_lane[None, :]"
        if cache["ValueVectorizedDim"] == 5
        else "value_lane[:, None]"
        if cache["ValueVectorizedDim"] == 3
        else "0"
    )
    key_vector_offset = (
        f"({key_block_index} * {cache['BlockElements']} + "
        f"{cache['KeySectionOffset']} + ((layer_id_value) * "
        f"{cache['KeyLayerStride']} + kv_head * {cache['KeyHeadStride']} + "
        f"key_dim_index[:, None] * {cache['KeyDimBlockStride']} + "
        f"key_block_index[None, :] * {cache['KeyBlockOffsetStride']}) * "
        f"{cache['KeyLaneCount']} + {key_lane})"
    )
    value_vector_offset = (
        f"({value_block_index} * {cache['BlockElements']} + "
        f"{cache['ValueSectionOffset']} + ((layer_id_value) * "
        f"{cache['ValueLayerStride']} + kv_head * {cache['ValueHeadStride']} + "
        f"value_dim_index[None, :] * {cache['ValueDimBlockStride']} + "
        f"value_block_index[:, None] * {cache['ValueBlockOffsetStride']}) * "
        f"{cache['ValueLaneCount']} + {value_lane})"
    )
    return {
        "attention_block_size": attention_block_size,
        "global_q_head": global_index_expression(
            model["HeadAxis"], "q_head", model["GlobalNumQueryHeads"]
        ),
        "global_query_id": global_index_expression(
            model["SeqAxis"], "local_query_id", global_query_tokens
        ),
        "global_query_tokens": global_query_tokens,
        "key_block_index": cache_block_index("Key", "block_offsets"),
        "key_dim_index": cache_dim_index("Key", "dim_offsets"),
        "key_lane": cache_lane("Key", "dim_offsets", "block_offsets"),
        "key_vector_offset": key_vector_offset,
        "local_q_heads": model["OutputShape"][model["HeadAxis"]],
        "local_query_tokens": local_query_tokens,
        "output_vector_offset": tensor_offset(
            model["OutputStrides"],
            model["SeqAxis"],
            model["HeadAxis"],
            model["DimAxis"],
            "q_head",
            "query_dim_blocks",
            "local_query_id",
            "query_dim_lanes",
            cache["KeyLaneCount"],
        ),
        "query_vector_offset": tensor_offset(
            model["QueryStrides"],
            model["SeqAxis"],
            model["HeadAxis"],
            model["DimAxis"],
            "q_head",
            "query_dim_blocks",
            "local_query_id",
            "query_dim_lanes",
            cache["KeyLaneCount"],
        ),
        "value_block_index": cache_block_index("Value", "block_offsets"),
        "value_dim_index": cache_dim_index("Value", "dim_offsets"),
        "value_lane": cache_lane("Value", "dim_offsets", "block_offsets"),
        "value_vector_offset": value_vector_offset,
    }




def _update_paged_attention_kv_cache_template_context(
    model: dict[str, Any],
) -> dict[str, Any]:
    """Prepare UpdatePagedAttentionKVCache's cache and slot addresses."""

    cache = model["Cache"]
    kind_prefix = "Key" if model["CacheKind"] == 0 else "Value"
    lane_count = cache[f"{kind_prefix}LaneCount"]
    vectorized_dim = cache[f"{kind_prefix}VectorizedDim"]
    slots_lane_count = model["SlotsVectorLaneCount"]
    source_split_axes = sorted(
        {
            axis
            for split_axes in model["SlotsSourceSplitAxes"]
            for axis in split_axes
        }
    )
    topology_match_axes = tuple(
        axis
        for axis in cache["NumBlocksSplitAxes"]
        if axis not in source_split_axes
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

    def slot_offset(lane_expr: str | None = "source_lane_id") -> str:
        terms = [
            f"source_idx{axis} * {_dim(model['SlotsStrides'][axis])}"
            for axis in range(len(model["SlotsStrides"]))
        ]
        element_offset = "linear * 0" if not terms else " + ".join(terms)
        if slots_lane_count == 1:
            return element_offset
        if lane_expr is None:
            return f"(({element_offset}) * {slots_lane_count})"
        return f"(({element_offset}) * {slots_lane_count} + {lane_expr})"

    def local_index_name(axis: int) -> str:
        if axis == model["SeqAxis"]:
            return "token_id"
        if axis == model["HeadAxis"]:
            return "head_id"
        if axis == model["DimAxis"]:
            return "source_dim_block"
        return f"local_idx{axis}"

    use_key_vector_copy = (
        model["CacheKind"] == 0
        and vectorized_dim == 5
        and slots_lane_count == lane_count
        and slots_lane_count > 1
        and slots_lane_count & (slots_lane_count - 1) == 0
        and cache["HeadDim"] % lane_count == 0
        and cache.get("TritonDType") == model["SlotsTritonDType"]
    )
    total_factors = [
        _dim(model["SlotsShape"][model["SeqAxis"]]),
        _dim(model["SlotsShape"][model["HeadAxis"]]),
        _dim(model["SlotsShape"][model["DimAxis"]]),
    ]
    if not use_key_vector_copy:
        total_factors.append(str(slots_lane_count))
    total_elements = " * ".join(f"({value})" for value in total_factors)
    return {
        "cache_offset": cache_offset,
        "kind_prefix": kind_prefix,
        "lane_count": lane_count,
        "local_indices": tuple(
            local_index_name(axis) for axis in range(len(model["SlotsShape"]))
        ),
        "non_data_axes": tuple(
            axis
            for axis in range(len(model["SlotsGlobalShape"]))
            if axis not in (model["SeqAxis"], model["HeadAxis"], model["DimAxis"])
        ),
        "slot_offset": slot_offset(None if use_key_vector_copy else "source_lane_id"),
        "slots_lane_count": slots_lane_count,
        "topology_match_axes": topology_match_axes,
        "total_elements": total_elements,
        "use_key_vector_copy": use_key_vector_copy,
        "vectorized_dim": vectorized_dim,
    }
