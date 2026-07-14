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
    body_source = _with_shard_index_prelude(body_source)
    body_source = _with_shared_memory_prelude(
        body_source,
        shared_allocation_bytes,
    )
    top_kernel = env.get_template("triton/top_kernel.py.jinja").render(
        name=metadata["name"],
        parameters=", ".join(parameters),
        body_source=_indent_block(body_source, 4),
        decorator="@triton.jit",
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
        if SHARD_INDEX_PARAMETER not in _parameter_call_arguments(device_parameters):
            body_source = _with_shard_index_prelude(body_source)
        parts.append(
            env.get_template("triton/top_kernel.py.jinja").render(
                name=stage["name"],
                parameters=", ".join(device_parameters),
                body_source=_indent_block(body_source, 4),
                decorator=(
                    "@triton.jit(noinline=True)"
                    if device_function["noinline"]
                    else "@triton.jit"
                ),
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


def _with_shard_index_prelude(source: str) -> str:
    source = source.rstrip()
    if SHARD_INDEX_PARAMETER not in _referenced_parameter_names(
        source, (SHARD_INDEX_PARAMETER,)
    ):
        return source
    prelude = "shard_index = tl.program_id(0).to(tl.int64)"
    if not source:
        return prelude
    return f"{prelude}\n{source}"


def _with_shared_memory_prelude(
    source: str,
    shared_size_bytes: int,
) -> str:
    if shared_size_bytes == 0:
        return source
    prelude = "\n".join(
        (
            f"pyntt_shared_arena = tle.gpu.alloc([{shared_size_bytes}], dtype=tl.uint8, "
            "layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)",
        )
    )
    source = source.rstrip()
    return prelude if not source else f"{prelude}\n{source}"


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


def emit(template_name: str, model: dict[str, Any]) -> str:
    name = Path(template_name).name
    if name.endswith(".py.jinja"):
        name = name[: -len(".py.jinja")]
    func = _EMITTERS.get(name)
    if func is None:
        raise NotImplementedError(f"PyNTT Jinja renderer has no emitter for {name}.")
    return func(model)


def _make_env() -> Environment:
    env = Environment(
        loader=PackageLoader("pyntt", "codegen/templates"),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.globals["emit"] = emit
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


def _indent_block(text: str, spaces: int) -> str:
    prefix = " " * spaces
    if not text.strip():
        return prefix + "pass"
    return "\n".join(prefix + line if line else line for line in text.splitlines())


def _i(level: int) -> str:
    return " " * (level * 4)


def _line(lines: list[str], level: int, text: str = "") -> None:
    lines.append(f"{_i(level)}{text}" if text else "")


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


def _append_pointer_binding(
    lines: list[str],
    level: int,
    model: dict[str, Any],
    local_name: str,
    model_name: str,
) -> None:
    if _local_buffer(model[model_name]) is None:
        _line(lines, level, f"{local_name} = {_ptr(model, model_name)}")


def _append_pointer_value_binding(
    lines: list[str],
    level: int,
    pointer: Any,
    local_name: str,
) -> None:
    if _local_buffer(pointer) is None:
        expression = pointer.get("Expression", pointer.get("expression"))
        _line(lines, level, f"{local_name} = {expression}")


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


def _load_operand(
    model: dict[str, Any],
    name: str,
    local_name: str,
    offsets: str,
    mask: str,
    *,
    other: str = "0.0",
    suffix: str = "",
) -> str:
    pointer = _access_pointer(model, name, local_name, offsets)
    return f"tl.load({pointer}, mask={mask}, other={other}{suffix})"


def _load_operand_nd(
    model: dict[str, Any],
    name: str,
    local_name: str,
    offsets: str,
    mask: str,
    shape: tuple[int, ...],
    *,
    other: str = "0.0",
    suffix: str = "",
) -> str:
    pointer = _access_pointer(model, name, local_name, offsets)
    return f"tl.load({pointer}, mask={mask}, other={other}{suffix})"


def _append_store(
    lines: list[str],
    level: int,
    model: dict[str, Any],
    name: str,
    local_name: str,
    offsets: str,
    value: str,
    mask: str,
) -> None:
    pointer = _access_pointer(model, name, local_name, offsets)
    _line(lines, level, f"tl.store({pointer}, {value}, mask={mask})")


def _pointer_shard_coord_hierarchy(pointer: Any) -> tuple[int, ...] | None:
    if not isinstance(pointer, dict):
        return None
    value = pointer.get("ShardCoordHierarchy", pointer.get("shard_coord_hierarchy"))
    if not value:
        return None
    return tuple(int(axis) for axis in value)


def _append_pointer_shard_coords(lines: list[str], level: int, pointers: list[Any]) -> None:
    hierarchies = {
        hierarchy
        for hierarchy in (_pointer_shard_coord_hierarchy(pointer) for pointer in pointers)
        if hierarchy is not None
    }
    if not hierarchies:
        return
    if len(hierarchies) != 1:
        raise RuntimeError(
            "PyNTT generated helper has pointer offsets from multiple shard "
            f"hierarchies: {sorted(hierarchies)}."
        )
    _append_shard_coords(lines, level, list(next(iter(hierarchies))))


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


def _shape_extent_for_static_split(value: Any) -> int:
    extent = _max_value(value)
    if extent is None:
        raise RuntimeError(
            "PyNTT staged reshard requires bounded non-split dimensions "
            f"for static mesh partitioning, got {value}."
        )
    return max(extent, 1)


def _non_split_tensor_axis_split_counts(
    shape: list[Any],
    non_split_tensor_axes: list[int],
    expected_split_count: int,
) -> list[int]:
    if not non_split_tensor_axes:
        return []
    if expected_split_count <= 1:
        return [1 for _ in non_split_tensor_axes]

    dims = [_shape_extent_for_static_split(shape[axis]) for axis in non_split_tensor_axes]
    log_dims = [math.log(float(dim)) for dim in dims]
    total_log_dim = sum(log_dims)
    split_counts = []
    for dim, log_dim in zip(dims, log_dims):
        split_factor = (
            0.0
            if total_log_dim == 0.0
            else log_dim / total_log_dim * math.log(float(expected_split_count))
        )
        split_counts.append(max(1, min(dim, int(math.exp(split_factor)))))

    total_splits = math.prod(split_counts)
    total_diff = expected_split_count - total_splits
    improved = True
    while improved:
        adjust_axis = 0
        adjust_delta = 0
        adjust_total_splits = total_splits
        adjust_diff = total_diff

        if total_diff < 0:
            for axis, split_count in enumerate(split_counts):
                if split_count <= 1:
                    continue
                new_total_splits = total_splits // split_count * (split_count - 1)
                new_total_diff = expected_split_count - new_total_splits
                if (
                    (adjust_diff < 0 and new_total_diff > adjust_diff)
                    or (adjust_diff > 0 and new_total_diff >= 0 and new_total_diff < adjust_diff)
                ):
                    adjust_axis = axis
                    adjust_delta = -1
                    adjust_total_splits = new_total_splits
                    adjust_diff = new_total_diff
        elif total_diff > 0:
            for axis, split_count in enumerate(split_counts):
                if split_count >= dims[axis]:
                    continue
                new_total_splits = total_splits // split_count * (split_count + 1)
                new_total_diff = expected_split_count - new_total_splits
                if new_total_diff >= 0 and new_total_diff < adjust_diff:
                    adjust_axis = axis
                    adjust_delta = 1
                    adjust_total_splits = new_total_splits
                    adjust_diff = new_total_diff

        if adjust_delta:
            split_counts[adjust_axis] += adjust_delta
            total_splits = adjust_total_splits
            total_diff = adjust_diff
        else:
            improved = False

    return split_counts


def _append_shard_coords(lines: list[str], level: int, hierarchy: list[int], *, shard: str = "shard_index", prefix: str = "shard_coord") -> None:
    _line(lines, level, f"tmp_{prefix} = {shard}")
    for axis in range(len(hierarchy) - 1, -1, -1):
        _line(lines, level, f"{prefix}{axis} = tmp_{prefix} % {hierarchy[axis]}")
        _line(lines, level, f"tmp_{prefix} = tmp_{prefix} // {hierarchy[axis]}")


def _append_tensor_index_decompose(
    lines: list[str],
    level: int,
    linear_name: str,
    prefix: str,
    shape: list[Any],
) -> None:
    _line(lines, level, f"tmp = {linear_name}")
    for axis in range(len(shape) - 1, -1, -1):
        _line(lines, level, f"{prefix}{axis} = tmp % {_dim(shape[axis])}")
        _line(lines, level, f"tmp = tmp // {_dim(shape[axis])}")


def _runtime_suffix(model: dict[str, Any]) -> str:
    args = tuple(model.get("RuntimeShapeArgs", ()) or ())
    if not args:
        return ""
    return ", " + ", ".join(args)


def _helper_header(
    model: dict[str, Any],
    args: tuple[str, ...] = (),
    *,
    comment: str | None = None,
) -> list[str]:
    runtime = _runtime_suffix(model)
    abi_args = tuple(model.get("Arguments", ()) or ())
    parameters = ", ".join(
        args
        + abi_args
        + WORKSPACE_PARAMETERS
        + WORKSPACE_STRIDE_PARAMETERS
        + tuple(model.get("RuntimeShapeArgs", ()) or ())
        + ("block_size: tl.constexpr",))
    lines = []
    if comment:
        lines.append(comment)
    lines.append(
        "@triton.jit(noinline=True)" if model["NoInline"] else "@triton.jit"
    )
    lines.append(f"def {model['FunctionName']}({parameters}):")
    return lines


def _standard_header(model: dict[str, Any], comment: str, pointers: list[tuple[str, str]]) -> list[str]:
    lines = _helper_header(model, comment=comment)
    _line(lines, 1, "shard_index = tl.program_id(0).to(tl.int64)")
    _append_pointer_shard_coords(
        lines,
        1,
        [model[model_name] for _, model_name in pointers],
    )
    for local_name, model_name in pointers:
        _append_pointer_binding(lines, 1, model, local_name, model_name)
    return lines


def _finish(lines: list[str]) -> str:
    return "\n".join(lines).rstrip()


def _is_bool_dtype(dtype: Any) -> bool:
    return str(dtype) == "bool"


def _as_predicate(expression: str, dtype: Any) -> str:
    return f"({expression} != 0)" if _is_bool_dtype(dtype) else expression


def _store_value(expression: str, dtype: Any) -> str:
    return f"({expression}).to(tl.uint8)" if _is_bool_dtype(dtype) else expression


def _emit_elementwise_unary(model: dict[str, Any]) -> str:
    shape = model["Shape"]
    total = _multiply_expr(_product(model["OutputShape"]), model["OutputVectorLaneCount"])

    def input_offset() -> str:
        axis_offset = len(model["OutputShape"]) - len(model["InputShape"])
        terms = []
        for axis, dim in enumerate(model["InputShape"]):
            if _is_fixed_one(dim):
                continue
            terms.append(f"idx{axis_offset + axis} * {_dim(model['InputStrides'][axis])}")
        tensor_offset = "lane_flat * 0" if not terms else " + ".join(terms)
        lanes = model["InputVectorLaneCount"]
        return tensor_offset if lanes == 1 else f"(({tensor_offset}) * {lanes} + lane_flat)"

    def output_offset() -> str:
        terms = [
            f"idx{axis} * {_dim(stride)}"
            for axis, stride in enumerate(model["OutputStrides"])
        ]
        tensor_offset = "lane_flat * 0" if not terms else " + ".join(terms)
        lanes = model["OutputVectorLaneCount"]
        return tensor_offset if lanes == 1 else f"(({tensor_offset}) * {lanes} + lane_flat)"

    lines = _standard_header(
        model,
        f"# generated from PyNTT Jinja ElementwiseUnary.py.jinja\n# {model['Comment']}; op={model['Op']}, input_dtype={model['InputDType']}, output_dtype={model['OutputDType']}, shape={_shape_tuple(shape)}",
        [("input0", "Input"), ("output", "Output")],
    )
    indent = 2
    _line(lines, 1, f"for linear_start in tl.range(0, {total}, block_size):")
    _line(lines, indent, "linear = linear_start + tl.arange(0, block_size)")
    _line(lines, indent, f"mask = linear < {total}")
    _line(lines, indent, f"lane_flat = linear % {model['OutputVectorLaneCount']}")
    _line(lines, indent, f"output_tensor_linear = linear // {model['OutputVectorLaneCount']}")
    _append_tensor_index_decompose(lines, indent, "output_tensor_linear", "idx", model["OutputShape"])
    _line(lines, indent, f"input_offsets = {input_offset()}")
    _line(lines, indent, f"output_offsets = {output_offset()}")
    _line(lines, indent, f"value0 = {_load_operand(model, 'Input', 'input0', 'input_offsets', 'mask')}")
    _line(lines, indent, "value0_f32 = value0.to(tl.float32)")
    _line(lines, indent, f"result = {model['UnaryExpression']}")
    _append_store(
        lines,
        indent,
        model,
        "Output",
        "output",
        "output_offsets",
        _store_value("result", model["OutputDType"]),
        "mask",
    )
    return _finish(lines)


def _emit_elementwise_binary(model: dict[str, Any]) -> str:
    rank = len(model["OutputShape"])
    total = _multiply_expr(_product(model["OutputShape"]), model["OutputVectorLaneCount"])

    def offset(operand_shape: list[Any], strides: list[Any], lanes: int) -> str:
        axis_offset = rank - len(operand_shape)
        terms = []
        for axis, dim in enumerate(operand_shape):
            if _is_fixed_one(dim):
                continue
            terms.append(f"idx{axis_offset + axis} * {_dim(strides[axis])}")
        tensor_offset = "lane_flat * 0" if not terms else " + ".join(terms)
        return tensor_offset if lanes == 1 else f"(({tensor_offset}) * {lanes} + lane_flat)"

    def output_offset() -> str:
        terms = [
            f"idx{axis} * {_dim(stride)}"
            for axis, stride in enumerate(model["OutputStrides"])
        ]
        tensor_offset = "lane_flat * 0" if not terms else " + ".join(terms)
        lanes = model["OutputVectorLaneCount"]
        return tensor_offset if lanes == 1 else f"(({tensor_offset}) * {lanes} + lane_flat)"

    lines = _standard_header(
        model,
        f"# generated from PyNTT Jinja ElementwiseBinary.py.jinja\n# {model['Comment']}; op={model['Op']}, lhs_dtype={model['LhsDType']}, rhs_dtype={model['RhsDType']}, output_dtype={model['OutputDType']}, shape={_shape_tuple(model['Shape'])}",
        [("lhs", "Lhs"), ("rhs", "Rhs"), ("output", "Output")],
    )
    indent = 2
    _line(lines, 1, f"for linear_start in tl.range(0, {total}, block_size):")
    _line(lines, indent, "linear = linear_start + tl.arange(0, block_size)")
    _line(lines, indent, f"mask = linear < {total}")
    _line(lines, indent, f"lane_flat = linear % {model['OutputVectorLaneCount']}")
    _line(lines, indent, f"output_tensor_linear = linear // {model['OutputVectorLaneCount']}")
    _append_tensor_index_decompose(lines, indent, "output_tensor_linear", "idx", model["OutputShape"])
    _line(lines, indent, f"lhs_offsets = {offset(model['LhsShape'], model['LhsStrides'], model['LhsVectorLaneCount'])}")
    _line(lines, indent, f"rhs_offsets = {offset(model['RhsShape'], model['RhsStrides'], model['RhsVectorLaneCount'])}")
    _line(lines, indent, f"output_offsets = {output_offset()}")
    _line(lines, indent, f"value0 = {_load_operand(model, 'Lhs', 'lhs', 'lhs_offsets', 'mask')}")
    _line(lines, indent, f"value1 = {_load_operand(model, 'Rhs', 'rhs', 'rhs_offsets', 'mask')}")
    _line(lines, indent, f"result = {model['BinaryExpression']}")
    _append_store(
        lines,
        indent,
        model,
        "Output",
        "output",
        "output_offsets",
        _store_value("result", model["OutputDType"]),
        "mask",
    )
    return _finish(lines)


def _emit_elementwise_cast(model: dict[str, Any]) -> str:
    total = _product(model["Shape"])
    vectorized_axis = model["VectorizedAxes"][0] if model.get("VectorizedAxes") else -1

    def tensor_offset(prefix: str, strides: list[Any], lane_count: int) -> str:
        terms = []
        lane_flat = "0"
        for axis, stride in enumerate(strides):
            index = f"{prefix}{axis}"
            if axis == vectorized_axis and lane_count != 1:
                lane_flat = f"{index} % {lane_count}"
                index = f"{index} // {lane_count}"
            terms.append(f"{index} * {_dim(stride)}")
        tensor = "linear * 0" if not terms else " + ".join(terms)
        return tensor if lane_count == 1 else f"(({tensor}) * {lane_count} + {lane_flat})"

    lines = _standard_header(
        model,
        f"# generated from PyNTT Jinja ElementwiseCast.py.jinja\n# {model['Comment']}; cast_mode={model['CastMode']}, input_dtype={model['InputDType']}, output_dtype={model['OutputDType']}, shape={_shape_tuple(model['Shape'])}",
        [("input0", "Input"), ("output", "Output")],
    )
    indent = 2
    _line(lines, 1, f"for linear_start in tl.range(0, {total}, block_size):")
    _line(lines, indent, "linear = linear_start + tl.arange(0, block_size)")
    _line(lines, indent, f"mask = linear < {total}")
    _append_tensor_index_decompose(lines, indent, "linear", "idx", model["Shape"])
    _line(lines, indent, f"input_offsets = {tensor_offset('idx', model['InputStrides'], model['InputVectorLaneCount'])}")
    _line(lines, indent, f"output_offsets = {tensor_offset('idx', model['OutputStrides'], model['OutputVectorLaneCount'])}")
    _line(lines, indent, f"value0 = {_load_operand(model, 'Input', 'input0', 'input_offsets', 'mask')}")
    _line(lines, indent, f"result = {model['CastExpression']}")
    _append_store(
        lines,
        indent,
        model,
        "Output",
        "output",
        "output_offsets",
        _store_value("result", model["OutputDType"]),
        "mask",
    )
    return _finish(lines)


def _emit_tensor_load(model: dict[str, Any]) -> str:
    return _emit_tensor_copy(model, is_load=True)


def _emit_tensor_store(model: dict[str, Any]) -> str:
    return _emit_tensor_copy(model, is_load=False)


def _emit_memcopy(model: dict[str, Any]) -> str:
    total = _multiply_expr(_product(model["Shape"]), model["VectorLaneCount"])

    def tensor_offset(prefix: str, strides: list[Any], lane_flat: str) -> str:
        terms = [f"{prefix}{axis} * {_dim(stride)}" for axis, stride in enumerate(strides)]
        tensor = "lane_flat * 0" if not terms else " + ".join(terms)
        lane_count = model["VectorLaneCount"]
        return tensor if lane_count == 1 else f"(({tensor}) * {lane_count} + {lane_flat})"

    lines = _standard_header(
        model,
        f"# generated from PyNTT Jinja Memcopy.py.jinja\n# {model['Comment']}; dtype={model['DType']}, shape={_shape_tuple(model['Shape'])}",
        [("source", "Source"), ("destination", "Destination")],
    )
    indent = 2
    _line(lines, 1, f"for linear_start in tl.range(0, {total}, block_size):")
    _line(lines, indent, "linear = linear_start + tl.arange(0, block_size)")
    _line(lines, indent, f"mask = linear < {total}")
    _line(lines, indent, f"lane_flat = linear % {model['VectorLaneCount']}")
    _line(lines, indent, f"tensor_linear = linear // {model['VectorLaneCount']}")
    _append_tensor_index_decompose(lines, indent, "tensor_linear", "idx", model["Shape"])
    _line(lines, indent, f"source_offsets = {tensor_offset('idx', model['SourceStrides'], 'lane_flat')}")
    _line(lines, indent, f"destination_offsets = {tensor_offset('idx', model['DestinationStrides'], 'lane_flat')}")
    _line(lines, indent, f"value = {_load_operand(model, 'Source', 'source', 'source_offsets', 'mask')}")
    _append_store(
        lines,
        indent,
        model,
        "Destination",
        "destination",
        "destination_offsets",
        "value",
        "mask",
    )
    return _finish(lines)


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


def _append_tle_region_copy(
    lines: list[str],
    level: int,
    model: dict[str, Any],
    plan: dict[str, Any],
) -> None:
    descriptor_shape = plan["descriptor_shape"]
    lane_count = int(model["VectorLaneCount"])
    global_model_name = plan["global_model_name"]
    global_name = plan["global_name"]
    local_buffer = plan["local_buffer"]
    local_descriptor = _local_buffer_value(local_buffer, "DescriptorExpression")
    if not local_descriptor:
        raise ValueError("TensorRegionCopy local buffer is missing DescriptorExpression")

    for axis, extent in enumerate(descriptor_shape):
        if len(descriptor_shape) == 1:
            expanded_index = f"tl.arange(0, {extent})"
        else:
            suffix = f"[{', '.join(':' if position == axis else 'None' for position in range(len(descriptor_shape)))}]"
            expanded_index = f"tl.arange(0, {extent}){suffix}"
        _line(lines, level, f"copy_desc_idx{axis} = {expanded_index}")

    linear_terms = []
    inner_stride = math.prod(descriptor_shape)
    for axis, extent in enumerate(descriptor_shape):
        inner_stride //= extent
        term = f"copy_desc_idx{axis}"
        if inner_stride != 1:
            term = f"{term} * {inner_stride}"
        linear_terms.append(term)
    _line(lines, level, f"copy_linear = {' + '.join(linear_terms)}")
    if lane_count == 1:
        _line(lines, level, "copy_tensor_linear = copy_linear")
        _line(lines, level, "copy_vector_lane = copy_linear * 0")
    else:
        _line(lines, level, f"copy_tensor_linear = copy_linear // {lane_count}")
        _line(lines, level, f"copy_vector_lane = copy_linear % {lane_count}")
    _line(lines, level, "copy_remaining = copy_tensor_linear")
    for axis in range(len(plan["static_shape"]) - 1, -1, -1):
        _line(lines, level, f"copy_idx{axis} = copy_remaining % {plan['static_shape'][axis]}")
        _line(lines, level, f"copy_remaining = copy_remaining // {plan['static_shape'][axis]}")

    global_base = f"{global_name}_base"
    global_terms = [
        f"({global_base}{axis} + copy_idx{axis}) * {_dim(model[f'{global_model_name}Strides'][axis])}"
        for axis in range(len(plan["static_shape"]))
    ]
    global_offset = "copy_tensor_linear * 0" if not global_terms else " + ".join(global_terms)
    if lane_count != 1:
        global_offset = f"(({global_offset}) * {lane_count} + copy_vector_lane)"
    _line(lines, level, f"copy_global_offset = {global_offset}")
    copy_shape = f"[{', '.join(str(extent) for extent in descriptor_shape)}]"
    if plan["local_model_name"] == "Destination":
        _line(lines, level, f"tle.gpu.copy({global_name} + copy_global_offset, {local_descriptor}, {copy_shape})")
    else:
        _line(lines, level, f"tle.gpu.copy({local_descriptor}, {global_name} + copy_global_offset, {copy_shape})")


def _append_region_copy_fallback(
    lines: list[str],
    level: int,
    model: dict[str, Any],
) -> None:
    rank = len(model["SourceShape"])
    dimensions = [f"(copy_dim{axis})" for axis in range(rank)]
    dimensions.append(f"({model['VectorLaneCount']})")
    total = " * ".join(dimensions)
    _line(lines, level, f"for linear_start in tl.range(0, {total}, block_size):")
    body_level = level + 1
    _line(lines, body_level, "linear = linear_start + tl.arange(0, block_size)")
    _line(lines, body_level, f"mask = linear < {total}")
    if model["VectorLaneCount"] == 1:
        _line(lines, body_level, "tensor_linear = linear")
        _line(lines, body_level, "vector_lane = linear * 0")
    else:
        _line(lines, body_level, f"tensor_linear = linear // {model['VectorLaneCount']}")
        _line(lines, body_level, f"vector_lane = linear % {model['VectorLaneCount']}")
    _line(lines, body_level, "remaining = tensor_linear")
    for axis in range(rank - 1, -1, -1):
        _line(lines, body_level, f"idx{axis} = remaining % copy_dim{axis}")
        _line(lines, body_level, f"remaining = remaining // copy_dim{axis}")

    source_terms = [
        f"(source_base{axis} + idx{axis}) * {_dim(model['SourceStrides'][axis])}"
        for axis in range(rank)
    ]
    destination_terms = [
        f"(destination_base{axis} + idx{axis}) * {_dim(model['DestinationStrides'][axis])}"
        for axis in range(rank)
    ]
    source_offset = "tensor_linear * 0" if not source_terms else " + ".join(source_terms)
    destination_offset = "tensor_linear * 0" if not destination_terms else " + ".join(destination_terms)
    if model["VectorLaneCount"] != 1:
        source_offset = f"(({source_offset}) * {model['VectorLaneCount']} + vector_lane)"
        destination_offset = f"(({destination_offset}) * {model['VectorLaneCount']} + vector_lane)"
    _line(lines, body_level, f"source_offset = {source_offset}")
    _line(lines, body_level, f"destination_offset = {destination_offset}")
    source_pointer = _access_pointer(model, "Source", "source", "source_offset")
    destination_pointer = _access_pointer(
        model, "Destination", "destination", "destination_offset"
    )
    _line(lines, body_level, f"value = tl.load({source_pointer}, mask=mask)")
    _line(lines, body_level, f"tl.store({destination_pointer}, value, mask=mask)")


def _emit_tensor_region_copy(model: dict[str, Any]) -> str:
    rank = len(model["SourceShape"])
    if rank != len(model["DestinationShape"]):
        raise ValueError("TensorRegionCopy source and destination ranks must match")

    plan = _region_copy_tle_plan(model)
    pointers = [("source", "Source"), ("destination", "Destination")]
    if plan is not None:
        pointers = [(plan["global_name"], plan["global_model_name"])]
    lines = _standard_header(
        model,
        f"# generated from PyNTT Jinja TensorRegionCopy.py.jinja\n# {model['Comment']}; dtype={model['DType']}, source_shape={_shape_tuple(model['SourceShape'])}, destination_shape={_shape_tuple(model['DestinationShape'])}",
        pointers,
    )
    for axis in range(rank):
        source_offset = _dim(model["SourceGlobalOffsets"][axis])
        destination_offset = _dim(model["DestinationGlobalOffsets"][axis])
        _line(lines, 1, f"copy_start{axis} = tl.maximum({source_offset}, {destination_offset})")
        _line(lines, 1, f"copy_end{axis} = tl.minimum({source_offset} + {_dim(model['SourceShape'][axis])}, {destination_offset} + {_dim(model['DestinationShape'][axis])})")
        _line(lines, 1, f"copy_dim{axis} = tl.maximum(copy_end{axis} - copy_start{axis}, 0)")
        _line(lines, 1, f"source_base{axis} = copy_start{axis} - {source_offset}")
        _line(lines, 1, f"destination_base{axis} = copy_start{axis} - {destination_offset}")

    if plan is not None:
        _append_tle_region_copy(lines, 1, model, plan)
    else:
        _append_region_copy_fallback(lines, 1, model)
    return _finish(lines)


def _emit_tensor_copy(model: dict[str, Any], *, is_load: bool) -> str:
    local_shape = model["LocalShape"]
    global_shape = model["GlobalShape"]
    local_strides = model["DestinationStrides" if is_load else "SourceStrides"]
    explicit_global_strides = model.get("SourceStrides" if is_load else "DestinationStrides")
    global_strides = explicit_global_strides or _contiguous_strides(global_shape)
    block_axis = _select_block_axis(local_shape, local_strides)
    block_extent = _one() if not local_shape else local_shape[block_axis]
    loop_axes = [axis for axis in range(len(local_shape)) if axis != block_axis]

    def axis_index(axis: int) -> str:
        return "lane" if axis == block_axis else f"idx{axis}"

    def split_linear(split_axes: list[int]) -> str:
        return _split_linear_expression(split_axes, model["Hierarchy"])

    def local_offset(strides: list[Any]) -> str:
        terms = [f"{axis_index(axis)} * {_dim(strides[axis])}" for axis in range(len(local_shape))]
        return "0" if not terms else " + ".join(terms)

    def global_offset() -> str:
        terms = [
            f"global_idx{axis} * {_dim(global_strides[axis])}"
            for axis in range(len(global_shape))
        ]
        return "0" if not terms else " + ".join(terms)

    vector_lane_count = int(model.get("VectorLaneCount", 1))

    def scalar_offset(tensor_offset: str) -> str:
        if vector_lane_count == 1:
            return tensor_offset
        return f"(({tensor_offset}) * {vector_lane_count} + vector_lane)"

    if is_load:
        internal_source = model.get("Source")
        source_args = () if internal_source is not None else ("source", "source_pool_stride_elements: tl.constexpr")
        lines = _helper_header(
            model,
            source_args,
            comment=f"# generated from PyNTT Jinja TensorLoad.py.jinja\n# {model['Comment']}; dtype={model['DType']}, local_shape={_shape_tuple(local_shape)}, global_shape={_shape_tuple(global_shape)}",
        )
        _line(lines, 1, "shard_index = tl.program_id(0).to(tl.int64)")
        pointer_models = [model["Destination"]]
        if internal_source is not None:
            pointer_models.append(internal_source)
        _append_pointer_shard_coords(
            lines,
            1,
            pointer_models,
        )
        if internal_source is not None:
            _append_pointer_binding(lines, 1, model, "source", "Source")
        _append_pointer_binding(lines, 1, model, "destination", "Destination")
    else:
        internal_destination = model.get("Destination")
        destination_args = () if internal_destination is not None else ("destination", "destination_pool_stride_elements: tl.constexpr")
        lines = _helper_header(
            model,
            destination_args,
            comment=f"# generated from PyNTT Jinja TensorStore.py.jinja\n# {model['Comment']}; dtype={model['DType']}, local_shape={_shape_tuple(local_shape)}, global_shape={_shape_tuple(global_shape)}",
        )
        _line(lines, 1, "shard_index = tl.program_id(0).to(tl.int64)")
        pointer_models = [model["Source"]]
        if internal_destination is not None:
            pointer_models.append(internal_destination)
        _append_pointer_shard_coords(
            lines,
            1,
            pointer_models,
        )
        _append_pointer_binding(lines, 1, model, "source", "Source")
        if internal_destination is not None:
            _append_pointer_binding(lines, 1, model, "destination", "Destination")

    _line(lines, 1, "tmp_shard = shard_index")
    for axis in range(len(model["Hierarchy"]) - 1, -1, -1):
        _line(lines, 1, f"shard_coord{axis} = tmp_shard % {model['Hierarchy'][axis]}")
        _line(lines, 1, f"tmp_shard = tmp_shard // {model['Hierarchy'][axis]}")
    indent = 1
    for axis in loop_axes:
        _line(lines, indent, f"for idx{axis} in tl.range(0, {_dim(local_shape[axis])}):")
        indent += 1
    _line(lines, indent, f"for axis_start in tl.range(0, {_dim(block_extent)}, block_size):")
    _line(lines, indent + 1, "lane = axis_start + tl.arange(0, block_size)")
    _line(lines, indent + 1, f"mask = lane < {_dim(block_extent)}")
    for axis in range(len(global_shape)):
        _line(lines, indent + 1, f"global_idx{axis} = {axis_index(axis)} + {_dim(model['GlobalOffsets'][axis])}")
        _line(lines, indent + 1, f"mask = mask & (global_idx{axis} < {_dim(global_shape[axis])})")
    copy_indent = indent + 1
    if vector_lane_count != 1:
        _line(lines, copy_indent, f"for vector_lane in tl.range(0, {vector_lane_count}):")
        copy_indent += 1
    if is_load:
        source_pool_offset = "0" if internal_source is not None else "source_pool_stride_elements * shard_index"
        _line(lines, copy_indent, f"source_offsets = {source_pool_offset} + {model['SourceOffset']} + {scalar_offset(global_offset())}")
        _line(lines, copy_indent, f"destination_offsets = {scalar_offset(local_offset(local_strides))}")
        _line(lines, copy_indent, f"value = {_load_operand(model, 'Source', 'source', 'source_offsets', 'mask')}")
        _append_store(
            lines,
            copy_indent,
            model,
            "Destination",
            "destination",
            "destination_offsets",
            "value",
            "mask",
        )
    else:
        _line(lines, copy_indent, f"source_offsets = {scalar_offset(local_offset(local_strides))}")
        destination_pool_offset = "0" if internal_destination is not None else "destination_pool_stride_elements * shard_index"
        _line(lines, copy_indent, f"destination_offsets = {destination_pool_offset} + {model['DestinationOffset']} + {scalar_offset(global_offset())}")
        _line(lines, copy_indent, f"value = {_load_operand(model, 'Source', 'source', 'source_offsets', 'mask')}")
        _append_store(
            lines,
            copy_indent,
            model,
            "Destination",
            "destination",
            "destination_offsets",
            "value",
            "mask",
        )
    return _finish(lines)


# Larger emitters are intentionally kept in Python code rather than C# so the
# generated manifest stays stable while PyNTT-side kernel templates evolve.


def _emit_pad(model: dict[str, Any]) -> str:
    rank = len(model["OutputShape"])
    block_axis = _select_block_axis(model["OutputShape"], model["OutputStrides"])
    block_extent = _one() if rank == 0 else model["OutputShape"][block_axis]
    loop_axes = [axis for axis in range(rank) if axis != block_axis]

    def axis_index(axis: int) -> str:
        return "lane" if axis == block_axis else f"idx{axis}"

    def output_offset() -> str:
        terms = [f"{axis_index(axis)} * {_dim(model['OutputStrides'][axis])}" for axis in range(rank)]
        return "lane * 0" if not terms else "lane * 0 + " + " + ".join(terms)

    def input_offset() -> str:
        terms = [f"safe_in_idx{axis} * {_dim(model['InputStrides'][axis])}" for axis in range(rank)]
        return "lane * 0" if not terms else "lane * 0 + " + " + ".join(terms)

    lines = _standard_header(
        model,
        f"# generated from PyNTT Jinja Pad.py.jinja\n# {model['Comment']}; input_dtype={model['InputDType']}, output_dtype={model['OutputDType']}, input_shape={_shape_tuple(model['InputShape'])}, output_shape={_shape_tuple(model['OutputShape'])}",
        [("input", "Input"), ("output", "Output")],
    )
    indent = 1
    for axis in loop_axes:
        _line(lines, indent, f"for idx{axis} in tl.range(0, {_dim(model['OutputShape'][axis])}):")
        indent += 1
    _line(lines, indent, f"for axis_start in tl.range(0, {_dim(block_extent)}, block_size):")
    _line(lines, indent + 1, "lane = axis_start + tl.arange(0, block_size)")
    _line(lines, indent + 1, f"mask = lane < {_dim(block_extent)}")
    _line(lines, indent + 1, "in_bounds = mask")
    for axis in range(rank):
        pad_before = model["Pads"][axis][0]
        _line(lines, indent + 1, f"in_idx{axis} = {axis_index(axis)} - {pad_before}")
        _line(lines, indent + 1, f"axis{axis}_in_bounds = (in_idx{axis} >= 0) & (in_idx{axis} < {_dim(model['InputShape'][axis])})")
        _line(lines, indent + 1, f"in_bounds = in_bounds & axis{axis}_in_bounds")
        _line(lines, indent + 1, f"safe_in_idx{axis} = tl.where(axis{axis}_in_bounds, in_idx{axis}, 0)")
    _line(lines, indent + 1, f"input_offsets = {input_offset()}")
    _line(lines, indent + 1, f"output_offsets = {output_offset()}")
    _line(
        lines,
        indent + 1,
        f"input_value = {_load_operand(model, 'Input', 'input', 'input_offsets', 'in_bounds', other=str(model['PadValue']))}",
    )
    _line(lines, indent + 1, f"result = tl.where(in_bounds, input_value, {model['PadValue']})")
    _append_store(
        lines, indent + 1, model, "Output", "output", "output_offsets", "result", "mask"
    )
    return _finish(lines)


def _emit_slice(model: dict[str, Any]) -> str:
    rank = len(model["OutputShape"])
    block_axis = _select_block_axis(model["OutputShape"], model["OutputStrides"])
    block_extent = _one() if rank == 0 else model["OutputShape"][block_axis]
    loop_axes = [axis for axis in range(rank) if axis != block_axis]

    def axis_index(axis: int) -> str:
        return "lane" if axis == block_axis else f"idx{axis}"

    def output_offset() -> str:
        terms = [f"{axis_index(axis)} * {_dim(model['OutputStrides'][axis])}" for axis in range(rank)]
        return "lane * 0" if not terms else "lane * 0 + " + " + ".join(terms)

    def input_offset() -> str:
        terms = []
        for axis in range(rank):
            start = model["Starts"][axis]
            stride = model["Strides"][axis]
            index = f"{axis_index(axis)} + {start}" if stride == 1 else f"{start} + {axis_index(axis)} * {stride}"
            terms.append(f"({index}) * {_dim(model['InputStrides'][axis])}")
        return "lane * 0" if not terms else "lane * 0 + " + " + ".join(terms)

    lines = _standard_header(
        model,
        f"# generated from PyNTT Jinja Slice.py.jinja\n# {model['Comment']}; input_dtype={model['InputDType']}, output_dtype={model['OutputDType']}, input_shape={_shape_tuple(model['InputShape'])}, output_shape={_shape_tuple(model['OutputShape'])}",
        [("input", "Input"), ("output", "Output")],
    )
    indent = 1
    for axis in loop_axes:
        _line(lines, indent, f"for idx{axis} in tl.range(0, {_dim(model['OutputShape'][axis])}):")
        indent += 1
    _line(lines, indent, f"for axis_start in tl.range(0, {_dim(block_extent)}, block_size):")
    _line(lines, indent + 1, "lane = axis_start + tl.arange(0, block_size)")
    _line(lines, indent + 1, f"mask = lane < {_dim(block_extent)}")
    _line(lines, indent + 1, f"input_offsets = {input_offset()}")
    _line(lines, indent + 1, f"output_offsets = {output_offset()}")
    _line(
        lines,
        indent + 1,
        f"value = {_load_operand(model, 'Input', 'input', 'input_offsets', 'mask')}",
    )
    _append_store(
        lines, indent + 1, model, "Output", "output", "output_offsets", "value", "mask"
    )
    return _finish(lines)


def _emit_transpose(model: dict[str, Any]) -> str:
    total = _multiply_expr(_product(model["OutputShape"]), model["OutputVectorLaneCount"])

    def tensor_offset(prefix: str, strides: list[Any], lane_count: int, lane_flat: str) -> str:
        terms = [f"{prefix}{axis} * {_dim(stride)}" for axis, stride in enumerate(strides)]
        tensor = "lane_flat * 0" if not terms else " + ".join(terms)
        return tensor if lane_count == 1 else f"(({tensor}) * {lane_count} + {lane_flat})"

    lines = _standard_header(
        model,
        f"# generated from PyNTT Jinja Transpose.py.jinja\n# {model['Comment']}; input_dtype={model['InputDType']}, output_dtype={model['OutputDType']}, input_shape={_shape_tuple(model['InputShape'])}, output_shape={_shape_tuple(model['OutputShape'])}, perm={tuple(model['Perm'])}",
        [("input", "Input"), ("output", "Output")],
    )
    indent = 2
    _line(lines, 1, f"for linear_start in tl.range(0, {total}, block_size):")
    _line(lines, indent, "linear = linear_start + tl.arange(0, block_size)")
    _line(lines, indent, f"mask = linear < {total}")
    _line(lines, indent, f"lane_flat = linear % {model['OutputVectorLaneCount']}")
    _line(lines, indent, f"output_tensor_linear = linear // {model['OutputVectorLaneCount']}")
    _append_tensor_index_decompose(lines, indent, "output_tensor_linear", "out_idx", model["OutputShape"])
    for input_axis in range(len(model["InputShape"])):
        output_axis = model["Perm"].index(input_axis)
        _line(lines, indent, f"in_idx{input_axis} = out_idx{output_axis}")
    _line(lines, indent, f"input_offsets = {tensor_offset('in_idx', model['InputStrides'], model['InputVectorLaneCount'], 'lane_flat')}")
    _line(lines, indent, f"output_offsets = {tensor_offset('out_idx', model['OutputStrides'], model['OutputVectorLaneCount'], 'lane_flat')}")
    _line(lines, indent, f"value = {_load_operand(model, 'Input', 'input', 'input_offsets', 'mask')}")
    _append_store(
        lines, indent, model, "Output", "output", "output_offsets", "value", "mask"
    )
    return _finish(lines)


def _emit_elementwise_where(model: dict[str, Any]) -> str:
    logical_cond_shape = _logical_shape(model["CondShape"], model["CondVectorLaneCount"])
    logical_true_shape = _logical_shape(model["TrueShape"], model["TrueVectorLaneCount"])
    logical_false_shape = _logical_shape(model["FalseShape"], model["FalseVectorLaneCount"])
    logical_output_shape = _logical_shape(model["OutputShape"], model["OutputVectorLaneCount"])
    logical_output_strides = _logical_strides(model["OutputStrides"], model["OutputVectorLaneCount"])
    rank = len(logical_output_shape)
    block_axis = _select_block_axis(logical_output_shape, logical_output_strides)
    block_extent = _one() if rank == 0 else logical_output_shape[block_axis]
    loop_axes = [axis for axis in range(rank) if axis != block_axis]

    def axis_index(axis: int) -> str:
        return "lane" if axis == block_axis else f"idx{axis}"

    def offset(physical_shape: list[Any], logical_shape: list[Any], strides: list[Any], lane_count: int) -> str:
        axis_offset = rank - len(logical_shape)
        terms = []
        for axis, dim in enumerate(logical_shape):
            if _is_fixed_one(dim):
                continue
            output_axis = axis_offset + axis
            index = axis_index(output_axis)
            if lane_count > 1 and axis == len(physical_shape) - 1:
                index = f"(({index}) // {lane_count})"
            terms.append(f"{index} * {_dim(strides[axis])}")
        physical_index = "lane * 0" if not terms else "lane * 0 + " + " + ".join(terms)
        if lane_count == 1:
            return physical_index
        logical_last = axis_offset + len(logical_shape) - 1
        lane_index = f"(({axis_index(logical_last)}) % {lane_count})"
        return f"(({physical_index}) * {lane_count} + {lane_index})"

    lines = _standard_header(
        model,
        f"# generated from PyNTT Jinja ElementwiseWhere.py.jinja\n# {model['Comment']}; cond_dtype={model['CondDType']}, value_dtype={model['ValueDType']}, output_dtype={model['OutputDType']}, shape={_shape_tuple(model['Shape'])}",
        [("cond", "Cond"), ("true_value", "TrueValue"), ("false_value", "FalseValue"), ("output", "Output")],
    )
    indent = 1
    for axis in loop_axes:
        _line(lines, indent, f"for idx{axis} in tl.range(0, {_dim(logical_output_shape[axis])}):")
        indent += 1
    _line(lines, indent, f"for axis_start in tl.range(0, {_dim(block_extent)}, block_size):")
    _line(lines, indent + 1, "lane = axis_start + tl.arange(0, block_size)")
    _line(lines, indent + 1, f"mask = lane < {_dim(block_extent)}")
    _line(lines, indent + 1, f"cond_offsets = {offset(model['CondShape'], logical_cond_shape, model['CondStrides'], model['CondVectorLaneCount'])}")
    _line(lines, indent + 1, f"true_offsets = {offset(model['TrueShape'], logical_true_shape, model['TrueStrides'], model['TrueVectorLaneCount'])}")
    _line(lines, indent + 1, f"false_offsets = {offset(model['FalseShape'], logical_false_shape, model['FalseStrides'], model['FalseVectorLaneCount'])}")
    _line(lines, indent + 1, f"output_offsets = {offset(model['OutputShape'], logical_output_shape, model['OutputStrides'], model['OutputVectorLaneCount'])}")
    _line(
        lines,
        indent + 1,
        f"predicate_value = {_load_operand(model, 'Cond', 'cond', 'cond_offsets', 'mask', other='0')}",
    )
    _line(lines, indent + 1, f"predicate = {_as_predicate('predicate_value', model['CondDType'])}")
    _line(
        lines,
        indent + 1,
        f"value0 = {_load_operand(model, 'TrueValue', 'true_value', 'true_offsets', 'mask')}",
    )
    _line(
        lines,
        indent + 1,
        f"value1 = {_load_operand(model, 'FalseValue', 'false_value', 'false_offsets', 'mask')}",
    )
    _line(lines, indent + 1, "result = tl.where(predicate, value0, value1)")
    _append_store(
        lines,
        indent + 1,
        model,
        "Output",
        "output",
        "output_offsets",
        _store_value("result", model["OutputDType"]),
        "mask",
    )
    return _finish(lines)


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


def _emit_vector_layout(model: dict[str, Any]) -> str:
    op = "pack" if model["IsPack"] else "unpack"
    input_lane_count = _product_int(model["InputLanes"])
    output_lane_count = _product_int(model["OutputLanes"])
    domain_shape = model["OutputShape"] if model["IsPack"] else model["InputShape"]
    domain_lane_count = output_lane_count if model["IsPack"] else input_lane_count
    total = _multiply_expr(_product(domain_shape), domain_lane_count)

    def lane_at(lane_group: str, lanes: list[int], index: int) -> str:
        stride = _product_int(lanes[index + 1 :])
        return f"({lane_group}) % {lanes[index]}" if stride == 1 else f"(({lane_group}) // {stride}) % {lanes[index]}"

    def axis_lane_indices(axis: int) -> list[int]:
        return [
            lane_index
            for lane_index, packed_axis in enumerate(model["Axes"])
            if packed_axis == axis
        ]

    def axis_lane_product(indices: list[int]) -> int:
        return _product_int([model["Lanes"][lane_index] for lane_index in indices])

    def axis_lane_offset(lines: list[str], level: int, lane_group: str, indices: list[int]) -> str:
        terms = []
        for index, lane_index in enumerate(indices):
            lane_name = f"lane{lane_index}"
            _line(lines, level, f"{lane_name} = {lane_at(lane_group, model['Lanes'], lane_index)}")
            lane_stride = axis_lane_product(indices[index + 1 :])
            terms.append(lane_name if lane_stride == 1 else f"{lane_name} * {lane_stride}")
        return "0" if not terms else " + ".join(terms)

    def tensor_offset(prefix: str, strides: list[Any], lane_count: int, lane_flat: str) -> str:
        terms = [f"{prefix}{axis} * {_dim(stride)}" for axis, stride in enumerate(strides)]
        tensor = "lane_flat * 0" if not terms else " + ".join(terms)
        return tensor if lane_count == 1 else f"(({tensor}) * {lane_count} + {lane_flat})"

    lines = _standard_header(
        model,
        f"# generated from PyNTT Jinja VectorLayout.py.jinja\n# {model['Comment']}; op={op}, input_dtype={model['InputDType']}, output_dtype={model['OutputDType']}, input_shape={_shape_tuple(model['InputShape'])}, output_shape={_shape_tuple(model['OutputShape'])}",
        [("input_ptr", "Input"), ("output_ptr", "Output")],
    )
    value_indent = 2
    _line(lines, 1, f"for linear_start in tl.range(0, {total}, block_size):")
    _line(lines, value_indent, "linear = linear_start + tl.arange(0, block_size)")
    _line(lines, value_indent, f"mask = linear < {total}")
    _line(lines, value_indent, f"lane_flat = linear % {domain_lane_count}")
    _line(lines, value_indent, f"tensor_linear = linear // {domain_lane_count}")
    if model["IsPack"]:
        _append_tensor_index_decompose(lines, value_indent, "tensor_linear", "out_idx", model["OutputShape"])
        input_lane_flat = "lane_flat * 0" if input_lane_count == 1 else f"lane_flat % {input_lane_count}"
        pack_lane_group = "lane_flat" if input_lane_count == 1 else f"lane_flat // {input_lane_count}"
        _line(lines, value_indent, f"input_lane_flat = {input_lane_flat}")
        _line(lines, value_indent, f"new_lane_group = {pack_lane_group}")
        _line(lines, value_indent, "valid = mask")
        for axis in range(len(model["InputShape"])):
            packed_axes = axis_lane_indices(axis)
            if packed_axes:
                lane_product = axis_lane_product(packed_axes)
                lane_offset = axis_lane_offset(lines, value_indent, "new_lane_group", packed_axes)
                _line(lines, value_indent, f"in_idx{axis} = out_idx{axis} * {lane_product} + {lane_offset}")
                _line(lines, value_indent, f"valid = valid & (in_idx{axis} < {_dim(model['InputShape'][axis])})")
            else:
                _line(lines, value_indent, f"in_idx{axis} = out_idx{axis}")
        _line(lines, value_indent, f"input_offsets = {tensor_offset('in_idx', model['InputStrides'], input_lane_count, 'input_lane_flat')}")
        _line(lines, value_indent, f"output_offsets = {tensor_offset('out_idx', model['OutputStrides'], output_lane_count, 'lane_flat')}")
        _line(lines, value_indent, f"value = {_load_operand(model, 'Input', 'input_ptr', 'input_offsets', 'valid', other='0')}")
        _append_store(
            lines,
            value_indent,
            model,
            "Output",
            "output_ptr",
            "output_offsets",
            "value",
            "mask",
        )
    else:
        _append_tensor_index_decompose(lines, value_indent, "tensor_linear", "in_idx", model["InputShape"])
        output_lane_flat = "lane_flat * 0" if output_lane_count == 1 else f"lane_flat % {output_lane_count}"
        unpack_lane_group = "lane_flat" if output_lane_count == 1 else f"lane_flat // {output_lane_count}"
        _line(lines, value_indent, f"output_lane_flat = {output_lane_flat}")
        _line(lines, value_indent, f"new_lane_group = {unpack_lane_group}")
        _line(lines, value_indent, "valid = mask")
        for axis in range(len(model["OutputShape"])):
            unpacked_axes = axis_lane_indices(axis)
            if unpacked_axes:
                lane_product = axis_lane_product(unpacked_axes)
                lane_offset = axis_lane_offset(lines, value_indent, "new_lane_group", unpacked_axes)
                _line(lines, value_indent, f"out_idx{axis} = in_idx{axis} * {lane_product} + {lane_offset}")
                _line(lines, value_indent, f"valid = valid & (out_idx{axis} < {_dim(model['OutputShape'][axis])})")
            else:
                _line(lines, value_indent, f"out_idx{axis} = in_idx{axis}")
        _line(lines, value_indent, f"input_offsets = {tensor_offset('in_idx', model['InputStrides'], input_lane_count, 'lane_flat')}")
        _line(lines, value_indent, f"output_offsets = {tensor_offset('out_idx', model['OutputStrides'], output_lane_count, 'output_lane_flat')}")
        _line(lines, value_indent, f"value = {_load_operand(model, 'Input', 'input_ptr', 'input_offsets', 'valid', other='0')}")
        _append_store(
            lines,
            value_indent,
            model,
            "Output",
            "output_ptr",
            "output_offsets",
            "value",
            "valid",
        )
    return _finish(lines)


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


def _emit_matmul(model: dict[str, Any]) -> str:
    return _emit_matmul_like(model, gemv=False)


def _emit_gemv(model: dict[str, Any]) -> str:
    return _emit_matmul_like(model, gemv=True)


def _emit_qkv_parallel_linear(model: dict[str, Any]) -> str:
    reduction_phase = str(model.get("ReductionPhase", "complete")).lower()
    if reduction_phase == "accumulate":
        return _emit_qkv_reduction_accumulate(model, packed=False)
    if reduction_phase == "finalize":
        return _emit_qkv_reduction_finalize(model, packed=False)
    if reduction_phase != "complete":
        raise ValueError(
            f"Unsupported QKVParallelLinear reduction phase: {reduction_phase!r}."
        )

    m = model["QOutputShape"][-2]
    k = model["InputShape"][-1]
    output_batch_rank = len(model["QOutputShape"]) - 2
    batch_axes = list(range(output_batch_rank))
    input_batch_offset = _batch_offset_expression(model["InputShape"], model["InputStrides"], output_batch_rank)

    lines = _standard_header(
        model,
        f"# generated from PyNTT Jinja QKVParallelLinear.py.jinja\n# {model['Comment']}; input_dtype={model['InputDType']}, weight_dtype={model['WeightDType']}, output_dtype={model['OutputDType']}, input_shape={_shape_tuple(model['InputShape'])}, q_weight_shape={_shape_tuple(model['QWeightShape'])}, k_weight_shape={_shape_tuple(model['KWeightShape'])}, v_weight_shape={_shape_tuple(model['VWeightShape'])}, q_output_shape={_shape_tuple(model['QOutputShape'])}, k_output_shape={_shape_tuple(model['KOutputShape'])}, v_output_shape={_shape_tuple(model['VOutputShape'])}",
        [
            ("input0", "Input"),
            ("q_weight", "QWeight"),
            ("k_weight", "KWeight"),
            ("v_weight", "VWeight"),
            ("q_bias", "QBias"),
            ("k_bias", "KBias"),
            ("v_bias", "VBias"),
            ("q_output", "QOutput"),
            ("k_output", "KOutput"),
            ("v_output", "VOutput"),
        ],
    )
    indent = 1
    for axis in batch_axes:
        _line(lines, indent, f"for idx{axis} in tl.range(0, {_dim(model['QOutputShape'][axis])}):")
        indent += 1

    use_gemv = (_max_value(m) == 1) or (_fixed(m) == 1)
    if use_gemv:
        block_k = 256
        k_max = _max_value(k)
        n_max = max(_max_value(model[name][-1]) or 0 for name in ("QOutputShape", "KOutputShape", "VOutputShape"))
        use_large_n = k_max is not None and k_max <= block_k and n_max >= 4096
        block_n = 128 if use_large_n else 32
        n_stages = 2 if use_large_n else 1
        _line(lines, indent, f"for m_idx in tl.range(0, {_dim(m)}):")
        indent += 1
        for prefix in ("Q", "K", "V"):
            _emit_qkv_gemv_projection(lines, indent, model, prefix, input_batch_offset, block_n, block_k, n_stages)
    else:
        block_m = 16
        block_n = 64
        block_k = 64
        _line(lines, indent, f"for m_start in tl.range(0, {_dim(m)}, {block_m}):")
        _line(lines, indent + 1, f"offs_m = m_start + tl.arange(0, {block_m})")
        for prefix in ("Q", "K", "V"):
            _emit_qkv_matmul_projection(lines, indent + 1, model, prefix, input_batch_offset, block_m, block_n, block_k)
    return _finish(lines)


def _emit_packed_qkv_parallel_linear(model: dict[str, Any]) -> str:
    reduction_phase = str(model.get("ReductionPhase", "complete")).lower()
    if reduction_phase == "accumulate":
        return _emit_qkv_reduction_accumulate(model, packed=True)
    if reduction_phase == "finalize":
        return _emit_qkv_reduction_finalize(model, packed=True)
    if reduction_phase != "complete":
        raise ValueError(
            f"Unsupported PackedQKVParallelLinear reduction phase: {reduction_phase!r}."
        )

    q_logical_output_shape = _packed_qkv_logical_output_shape(model, "Q")
    m = q_logical_output_shape[-2]
    k = model["InputShape"][-1]
    output_batch_rank = len(model["QOutputShape"]) - 2
    batch_axes = list(range(output_batch_rank))
    input_batch_offset = _batch_offset_expression(model["InputShape"], model["InputStrides"], output_batch_rank)
    lane_comment = (
        f", n_packed_lane={model['NPackedLaneCount']}, n_lane={model['NVectorLaneCount']}, "
        f"n_scalar_lane={model['NPackedLaneCount'] * model['NVectorLaneCount']}"
    )

    lines = _standard_header(
        model,
        f"# generated from PyNTT Jinja PackedQKVParallelLinear.py.jinja\n# {model['Comment']}; input_dtype={model['InputDType']}, weight_dtype={model['WeightDType']}, output_dtype={model['OutputDType']}, input_shape={_shape_tuple(model['InputShape'])}, q_weight_shape={_shape_tuple(model['QWeightShape'])}, k_weight_shape={_shape_tuple(model['KWeightShape'])}, v_weight_shape={_shape_tuple(model['VWeightShape'])}, q_output_shape={_shape_tuple(model['QOutputShape'])}, k_output_shape={_shape_tuple(model['KOutputShape'])}, v_output_shape={_shape_tuple(model['VOutputShape'])}{lane_comment}",
        [
            ("input0", "Input"),
            ("q_weight", "QWeight"),
            ("k_weight", "KWeight"),
            ("v_weight", "VWeight"),
            ("q_bias", "QBias"),
            ("k_bias", "KBias"),
            ("v_bias", "VBias"),
            ("q_output", "QOutput"),
            ("k_output", "KOutput"),
            ("v_output", "VOutput"),
        ],
    )
    _line(lines, 1, "tmp_shard = shard_index")
    for axis in range(len(model["Hierarchy"]) - 1, -1, -1):
        _line(lines, 1, f"shard_coord{axis} = tmp_shard % {model['Hierarchy'][axis]}")
        _line(lines, 1, f"tmp_shard = tmp_shard // {model['Hierarchy'][axis]}")

    indent = 1
    for axis in batch_axes:
        _line(lines, indent, f"for idx{axis} in tl.range(0, {_dim(q_logical_output_shape[axis])}):")
        indent += 1

    use_gemv = (_max_value(m) == 1) or (_fixed(m) == 1)
    if use_gemv:
        block_k = 256
        k_max = _max_value(k)
        n_max = max(_max_value(_packed_qkv_logical_output_shape(model, name)[-1]) or 0 for name in ("Q", "K", "V"))
        use_large_n = k_max is not None and k_max <= block_k and n_max >= 4096
        block_n = 128 if use_large_n else 32
        n_stages = 2 if use_large_n else 1
        _line(lines, indent, f"for m_idx in tl.range(0, {_dim(m)}):")
        indent += 1
        for prefix in ("Q", "K", "V"):
            _emit_packed_qkv_gemv_projection(lines, indent, model, prefix, input_batch_offset, block_n, block_k, n_stages)
    else:
        block_m = 16
        block_n = model["NPackedLaneCount"] * model["NVectorLaneCount"]
        block_k = 64
        _line(lines, indent, f"for m_start in tl.range(0, {_dim(m)}, {block_m}):")
        _line(lines, indent + 1, f"offs_m = m_start + tl.arange(0, {block_m})")
        for prefix in ("Q", "K", "V"):
            _emit_packed_qkv_matmul_projection(lines, indent + 1, model, prefix, input_batch_offset, block_m, block_n, block_k)
    return _finish(lines)


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


def _emit_qkv_reduction_accumulate(
    model: dict[str, Any], *, packed: bool
) -> str:
    template_name = (
        "PackedQKVParallelLinear.py.jinja"
        if packed
        else "QKVParallelLinear.py.jinja"
    )
    output_shapes = {
        prefix: _qkv_reduction_logical_output_shape(model, prefix, packed=packed)
        for prefix in ("Q", "K", "V")
    }
    if any(len(shape) != 2 for shape in output_shapes.values()):
        raise ValueError(
            "QKVParallelLinear backend-private reduction requires rank-2 local output tiles."
        )

    block_m = int(model["ReductionBlockM"])
    block_k = int(model["ReductionBlockK"])
    block_ns = {
        prefix: int(model[f"Reduction{prefix}BlockN"])
        for prefix in ("Q", "K", "V")
    }
    lines = _helper_header(
        model,
        ("q_acc", "k_acc", "v_acc"),
        comment=(
            f"# generated from PyNTT Jinja {template_name} reduction accumulate\n"
            f"# {model['Comment']}; accumulator=backend-private-fp32"
        ),
    )
    _line(lines, 1, "shard_index = tl.program_id(0).to(tl.int64)")
    _append_pointer_shard_coords(
        lines,
        1,
        [model["Input"], model["QWeight"], model["KWeight"], model["VWeight"]],
    )
    _append_pointer_binding(lines, 1, model, "input0", "Input")
    for prefix in ("Q", "K", "V"):
        _append_pointer_binding(
            lines, 1, model, f"{prefix.lower()}_weight", f"{prefix}Weight"
        )
    _line(lines, 1, f"offs_k = tl.arange(0, {block_k})")
    k = model["InputShape"][-1]
    m = output_shapes["Q"][-2]

    if block_m == 1:
        input_offsets = f"offs_k * {_dim(model['InputStrides'][-1])}"
        input_mask = (
            f"(0 < {_dim(m)}) & (0 < {_dim(model['InputShape'][-2])}) & "
            f"(offs_k < {_dim(k)})"
        )
        _line(
            lines,
            1,
            f"input_values = {_load_operand(model, 'Input', 'input0', input_offsets, input_mask)}.to(tl.float32)",
        )
    else:
        _line(lines, 1, f"offs_m = tl.arange(0, {block_m})")
        input_offsets = (
            f"(offs_m[:, None] * {_dim(model['InputStrides'][-2])} + "
            f"offs_k[None, :] * {_dim(model['InputStrides'][-1])})"
        )
        input_mask = (
            f"(offs_m[:, None] < {_dim(m)}) & "
            f"(offs_m[:, None] < {_dim(model['InputShape'][-2])}) & "
            f"(offs_k[None, :] < {_dim(k)})"
        )
        _line(
            lines,
            1,
            f"input_values = {_load_operand_nd(model, 'Input', 'input0', input_offsets, input_mask, (block_m, block_k))}",
        )

    dot_precision = (
        ', input_precision="ieee"'
        if model["InputDType"] == "float32" and model["WeightDType"] == "float32"
        else ""
    )
    for prefix, acc_name in (("Q", "q_acc"), ("K", "k_acc"), ("V", "v_acc")):
        block_n = block_ns[prefix]
        n = output_shapes[prefix][-1]
        weight_shape = model[f"{prefix}WeightShape"]
        weight_strides = model[f"{prefix}WeightStrides"]
        weight_k = weight_shape[-1] if packed else weight_shape[-2]
        offs_n = f"{prefix.lower()}_offs_n"
        _line(lines, 1, f"{offs_n} = tl.arange(0, {block_n})")
        if packed:
            weight_offsets = _packed_qkv_weight_offsets(
                model,
                prefix,
                "0",
                f"{offs_n}[:, None]" if block_m == 1 else f"{offs_n}[None, :]",
                "offs_k[None, :]" if block_m == 1 else "offs_k[:, None]",
            )
        elif block_m == 1:
            weight_offsets = (
                f"(offs_k[None, :] * {_dim(weight_strides[-2])} + "
                f"{offs_n}[:, None] * {_dim(weight_strides[-1])})"
            )
        else:
            weight_offsets = (
                f"(offs_k[:, None] * {_dim(weight_strides[-2])} + "
                f"{offs_n}[None, :] * {_dim(weight_strides[-1])})"
            )
        if block_m == 1:
            weight_mask = (
                f"({offs_n}[:, None] < {_dim(n)}) & "
                f"(offs_k[None, :] < {_dim(k)}) & "
                f"(offs_k[None, :] < {_dim(weight_k)})"
            )
            weight_value = _load_operand_nd(
                model,
                f"{prefix}Weight",
                f"{prefix.lower()}_weight",
                weight_offsets,
                weight_mask,
                (block_n, block_k),
                suffix=', eviction_policy="evict_first"',
            )
            _line(
                lines,
                1,
                f"{prefix.lower()}_weight_values = {weight_value}.to(tl.float32)",
            )
            _line(
                lines,
                1,
                f"{acc_name} += tl.sum({prefix.lower()}_weight_values * input_values[None, :], axis=1)",
            )
        else:
            weight_mask = (
                f"(offs_k[:, None] < {_dim(k)}) & "
                f"(offs_k[:, None] < {_dim(weight_k)}) & "
                f"({offs_n}[None, :] < {_dim(n)})"
            )
            weight_value = _load_operand_nd(
                model,
                f"{prefix}Weight",
                f"{prefix.lower()}_weight",
                weight_offsets,
                weight_mask,
                (block_k, block_n),
            )
            _line(
                lines,
                1,
                f"{prefix.lower()}_weight_values = {weight_value}",
            )
            _line(
                lines,
                1,
                f"{acc_name} += tl.dot(input_values, {prefix.lower()}_weight_values{dot_precision})",
            )
    _line(lines, 1, "return q_acc, k_acc, v_acc")
    return _finish(lines)


def _emit_qkv_reduction_finalize(
    model: dict[str, Any], *, packed: bool
) -> str:
    template_name = (
        "PackedQKVParallelLinear.py.jinja"
        if packed
        else "QKVParallelLinear.py.jinja"
    )
    output_shapes = {
        prefix: _qkv_reduction_logical_output_shape(model, prefix, packed=packed)
        for prefix in ("Q", "K", "V")
    }
    block_m = int(model["ReductionBlockM"])
    block_ns = {
        prefix: int(model[f"Reduction{prefix}BlockN"])
        for prefix in ("Q", "K", "V")
    }
    lines = _helper_header(
        model,
        ("q_acc", "k_acc", "v_acc"),
        comment=(
            f"# generated from PyNTT Jinja {template_name} reduction finalize\n"
            f"# {model['Comment']}; accumulator=backend-private-fp32"
        ),
    )
    _line(lines, 1, "shard_index = tl.program_id(0).to(tl.int64)")
    pointers = [model[f"{prefix}Output"] for prefix in ("Q", "K", "V")]
    pointers += [
        model[f"{prefix}Bias"]
        for prefix in ("Q", "K", "V")
        if model[f"Has{prefix}Bias"]
    ]
    _append_pointer_shard_coords(lines, 1, pointers)
    for prefix in ("Q", "K", "V"):
        _append_pointer_binding(
            lines, 1, model, f"{prefix.lower()}_output", f"{prefix}Output"
        )
        if model[f"Has{prefix}Bias"]:
            _append_pointer_binding(
                lines, 1, model, f"{prefix.lower()}_bias", f"{prefix}Bias"
            )
    if block_m != 1:
        _line(lines, 1, f"offs_m = tl.arange(0, {block_m})")

    for prefix, acc_name in (("Q", "q_acc"), ("K", "k_acc"), ("V", "v_acc")):
        block_n = block_ns[prefix]
        n = output_shapes[prefix][-1]
        offs_n = f"{prefix.lower()}_offs_n"
        _line(lines, 1, f"{offs_n} = tl.arange(0, {block_n})")
        if model[f"Has{prefix}Bias"]:
            if packed:
                bias_offsets = _packed_qkv_bias_offsets(model, prefix, offs_n)
            else:
                bias_offsets = f"{offs_n} * {_dim(model[f'{prefix}BiasStrides'][-1])}"
            suffix = "" if block_m == 1 else "[None, :]"
            bias_value = _load_operand(
                model,
                f"{prefix}Bias",
                f"{prefix.lower()}_bias",
                bias_offsets,
                f"{offs_n} < {_dim(n)}",
            )
            _line(
                lines,
                1,
                f"{acc_name} += {bias_value}.to(tl.float32){suffix}",
            )
        if block_m == 1:
            if packed:
                output_offsets = _packed_qkv_output_offsets(
                    model, prefix, "0", offs_n, "0"
                )
            else:
                output_offsets = (
                    f"(0 * {_dim(model[f'{prefix}OutputStrides'][-2])} + "
                    f"{offs_n} * {_dim(model[f'{prefix}OutputStrides'][-1])})"
                )
            mask = f"{offs_n} < {_dim(n)}"
        else:
            if packed:
                output_offsets = _packed_qkv_output_offsets(
                    model, prefix, "0", f"{offs_n}[None, :]", "offs_m[:, None]"
                )
            else:
                output_offsets = (
                    f"(offs_m[:, None] * {_dim(model[f'{prefix}OutputStrides'][-2])} + "
                    f"{offs_n}[None, :] * {_dim(model[f'{prefix}OutputStrides'][-1])})"
                )
            mask = (
                f"(offs_m[:, None] < {_dim(output_shapes[prefix][-2])}) & "
                f"({offs_n}[None, :] < {_dim(n)})"
            )
        _append_store(
            lines,
            1,
            model,
            f"{prefix}Output",
            f"{prefix.lower()}_output",
            output_offsets,
            acc_name,
            mask,
        )
    return _finish(lines)


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


def _packed_qkv_logical_n(model: dict[str, Any], prefix: str) -> Any:
    return _packed_qkv_logical_output_shape(model, prefix)[-1]


def _packed_qkv_total_logical_n(model: dict[str, Any]) -> Any:
    return _add_dims(_add_dims(_packed_qkv_logical_n(model, "Q"), _packed_qkv_logical_n(model, "K")), _packed_qkv_logical_n(model, "V"))


def _packed_qkv_qk_logical_n(model: dict[str, Any]) -> Any:
    return _add_dims(_packed_qkv_logical_n(model, "Q"), _packed_qkv_logical_n(model, "K"))


def _append_packed_qkv_concat_n_indices(lines: list[str], indent: int, model: dict[str, Any]) -> None:
    q_n = _packed_qkv_logical_n(model, "Q")
    qk_n = _packed_qkv_qk_logical_n(model)
    total_n = _packed_qkv_total_logical_n(model)
    _line(lines, indent, f"q_active = offs_n < {_dim(q_n)}")
    _line(lines, indent, f"k_active = (offs_n >= {_dim(q_n)}) & (offs_n < {_dim(qk_n)})")
    _line(lines, indent, f"v_active = (offs_n >= {_dim(qk_n)}) & (offs_n < {_dim(total_n)})")
    _line(lines, indent, "q_local_n = offs_n")
    _line(lines, indent, f"k_local_n = offs_n - {_dim(q_n)}")
    _line(lines, indent, f"v_local_n = offs_n - {_dim(qk_n)}")


def _emit_packed_qkv_gemv_concat_n(
    lines: list[str],
    indent: int,
    model: dict[str, Any],
    input_batch_offset: str,
    block_n: int,
    block_k: int,
    n_stages: int,
) -> None:
    total_n = _packed_qkv_total_logical_n(model)
    k = model["InputShape"][-1]
    input_terms = [] if input_batch_offset == "0" else [input_batch_offset]
    input_terms += [f"m_idx * {_dim(model['InputStrides'][-2])}", f"offs_k * {_dim(model['InputStrides'][-1])}"]
    input_offsets = f"({' + '.join(input_terms)})"

    _line(lines, indent, f"for n_start in tl.range(0, {_dim(total_n)}, {block_n}, num_stages={n_stages}):")
    _line(lines, indent + 1, f"offs_n = n_start + tl.arange(0, {block_n})")
    _append_packed_qkv_concat_n_indices(lines, indent + 1, model)
    _line(lines, indent + 1, f"acc = tl.zeros(({block_n},), tl.float32)")
    _line(lines, indent + 1, f"for k_start in tl.range(0, {_dim(k)}, {block_k}):")
    _line(lines, indent + 2, f"offs_k = k_start + tl.arange(0, {block_k})")

    for prefix in ("Q", "K", "V"):
        weight_shape = model[f"{prefix}WeightShape"]
        output_shape = model[f"{prefix}OutputShape"]
        weight_strides = model[f"{prefix}WeightStrides"]
        weight_batch_offset = _batch_offset_expression(weight_shape, weight_strides, len(output_shape) - 2)
        local_n = f"{prefix.lower()}_local_n"
        weight_offsets = _packed_qkv_weight_offsets(model, prefix, weight_batch_offset, f"{local_n}[:, None]", "offs_k[None, :]")
        weight_mask = f"{prefix.lower()}_active[:, None] & (offs_n[:, None] < {_dim(total_n)}) & (offs_k[None, :] < {_dim(k)})"
        weight_value = _load_operand_nd(
            model,
            f"{prefix}Weight",
            f"{prefix.lower()}_weight",
            weight_offsets,
            weight_mask,
            (block_n, block_k),
            suffix=', eviction_policy="evict_first"',
        )
        _line(lines, indent + 2, f"{prefix.lower()}_weight_values = {weight_value}.to(tl.float32)")

    _line(lines, indent + 2, "weight_values = tl.where(q_active[:, None], q_weight_values, tl.where(k_active[:, None], k_weight_values, v_weight_values))")
    input_mask = f"offs_k < {_dim(k)}"
    _line(lines, indent + 2, f"input_values = {_load_operand(model, 'Input', 'input0', input_offsets, input_mask)}.to(tl.float32)")
    _line(lines, indent + 2, "acc += tl.sum(weight_values * input_values[None, :], axis=1)")

    for prefix, active in (("Q", "q_active"), ("K", "k_active"), ("V", "v_active")):
        if model[f"Has{prefix}Bias"]:
            bias_offsets = _packed_qkv_bias_offsets(model, prefix, f"{prefix.lower()}_local_n")
            bias_mask = f"{active} & (offs_n < {_dim(total_n)})"
            bias_value = _load_operand(model, f"{prefix}Bias", f"{prefix.lower()}_bias", bias_offsets, bias_mask)
            _line(lines, indent + 1, f"acc += {bias_value}.to(tl.float32)")

    output_offsets_by_prefix = {}
    for prefix in ("Q", "K", "V"):
        output_shape = model[f"{prefix}OutputShape"]
        output_strides = model[f"{prefix}OutputStrides"]
        output_batch_offset = _batch_offset_expression(output_shape, output_strides, len(output_shape) - 2)
        output_offsets = _packed_qkv_output_offsets(model, prefix, output_batch_offset, f"{prefix.lower()}_local_n", "m_idx")
        output_offsets_by_prefix[prefix] = output_offsets

    for prefix, active in (("Q", "q_active"), ("K", "k_active"), ("V", "v_active")):
        _append_store(
            lines,
            indent + 1,
            model,
            f"{prefix}Output",
            f"{prefix.lower()}_output",
            output_offsets_by_prefix[prefix],
            "acc",
            f"{active} & (offs_n < {_dim(total_n)})",
        )


def _emit_packed_qkv_matmul_concat_n(
    lines: list[str],
    indent: int,
    model: dict[str, Any],
    input_batch_offset: str,
    block_m: int,
    block_n: int,
    block_k: int,
) -> None:
    q_output_logical_shape = _packed_qkv_logical_output_shape(model, "Q")
    total_n = _packed_qkv_total_logical_n(model)
    m = q_output_logical_shape[-2]
    k = model["InputShape"][-1]

    input_terms = [] if input_batch_offset == "0" else [input_batch_offset]
    input_terms += [f"offs_m[:, None] * {_dim(model['InputStrides'][-2])}", f"offs_k[None, :] * {_dim(model['InputStrides'][-1])}"]
    input_offsets = f"({' + '.join(input_terms)})"

    _line(lines, indent, f"for n_start in tl.range(0, {_dim(total_n)}, {block_n}):")
    _line(lines, indent + 1, f"offs_n = n_start + tl.arange(0, {block_n})")
    _append_packed_qkv_concat_n_indices(lines, indent + 1, model)
    _line(lines, indent + 1, f"acc = tl.zeros(({block_m}, {block_n}), tl.float32)")
    _line(lines, indent + 1, f"for k_start in tl.range(0, {_dim(k)}, {block_k}, num_stages=5):")
    _line(lines, indent + 2, f"offs_k = k_start + tl.arange(0, {block_k})")

    for prefix in ("Q", "K", "V"):
        weight_shape = model[f"{prefix}WeightShape"]
        output_shape = model[f"{prefix}OutputShape"]
        weight_strides = model[f"{prefix}WeightStrides"]
        weight_batch_offset = _batch_offset_expression(weight_shape, weight_strides, len(output_shape) - 2)
        local_n = f"{prefix.lower()}_local_n"
        weight_offsets = _packed_qkv_weight_offsets(model, prefix, weight_batch_offset, f"{local_n}[None, :]", "offs_k[:, None]")
        weight_mask = f"{prefix.lower()}_active[None, :] & (offs_k[:, None] < {_dim(k)}) & (offs_n[None, :] < {_dim(total_n)})"
        weight_value = _load_operand_nd(
            model,
            f"{prefix}Weight",
            f"{prefix.lower()}_weight",
            weight_offsets,
            weight_mask,
            (block_k, block_n),
        )
        _line(lines, indent + 2, f"{prefix.lower()}_weight_values = {weight_value}")

    _line(lines, indent + 2, "weight_values = tl.where(q_active[None, :], q_weight_values, tl.where(k_active[None, :], k_weight_values, v_weight_values))")
    input_mask = f"(offs_m[:, None] < {_dim(m)}) & (offs_k[None, :] < {_dim(k)})"
    _line(lines, indent + 2, f"input_values = {_load_operand_nd(model, 'Input', 'input0', input_offsets, input_mask, (block_m, block_k))}")
    dot_precision = ', input_precision="ieee"' if model["InputDType"] == "float32" and model["WeightDType"] == "float32" else ""
    _line(lines, indent + 2, f"acc += tl.dot(input_values, weight_values{dot_precision})")

    for prefix, active in (("Q", "q_active"), ("K", "k_active"), ("V", "v_active")):
        if model[f"Has{prefix}Bias"]:
            bias_offsets = _packed_qkv_bias_offsets(model, prefix, f"{prefix.lower()}_local_n")
            bias_mask = f"{active} & (offs_n < {_dim(total_n)})"
            bias_value = _load_operand(model, f"{prefix}Bias", f"{prefix.lower()}_bias", bias_offsets, bias_mask)
            _line(lines, indent + 1, f"acc += {bias_value}.to(tl.float32)[None, :]")

    output_offsets_by_prefix = {}
    for prefix in ("Q", "K", "V"):
        output_shape = model[f"{prefix}OutputShape"]
        output_strides = model[f"{prefix}OutputStrides"]
        output_batch_offset = _batch_offset_expression(output_shape, output_strides, len(output_shape) - 2)
        output_offsets = _packed_qkv_output_offsets(model, prefix, output_batch_offset, f"{prefix.lower()}_local_n[None, :]", "offs_m[:, None]")
        output_offsets_by_prefix[prefix] = output_offsets

    for prefix, active in (("Q", "q_active"), ("K", "k_active"), ("V", "v_active")):
        _append_store(
            lines,
            indent + 1,
            model,
            f"{prefix}Output",
            f"{prefix.lower()}_output",
            output_offsets_by_prefix[prefix],
            "acc",
            f"(offs_m[:, None] < {_dim(m)}) & {active}[None, :] & (offs_n[None, :] < {_dim(total_n)})",
        )


def _emit_packed_qkv_gemv_projection(
    lines: list[str],
    indent: int,
    model: dict[str, Any],
    prefix: str,
    input_batch_offset: str,
    block_n: int,
    block_k: int,
    n_stages: int,
) -> None:
    weight_shape = model[f"{prefix}WeightShape"]
    output_shape = model[f"{prefix}OutputShape"]
    weight_strides = model[f"{prefix}WeightStrides"]
    output_strides = model[f"{prefix}OutputStrides"]
    output_logical_shape = _packed_qkv_logical_output_shape(model, prefix)
    weight_batch_offset = _batch_offset_expression(weight_shape, weight_strides, len(output_shape) - 2)
    output_batch_offset = _batch_offset_expression(output_shape, output_strides, len(output_shape) - 2)
    n = output_logical_shape[-1]
    k = model["InputShape"][-1]
    ptr_weight = f"{prefix.lower()}_weight"
    ptr_bias = f"{prefix.lower()}_bias"
    ptr_output = f"{prefix.lower()}_output"

    input_terms = [] if input_batch_offset == "0" else [input_batch_offset]
    input_terms += [f"m_idx * {_dim(model['InputStrides'][-2])}", f"offs_k * {_dim(model['InputStrides'][-1])}"]
    input_offsets = f"({' + '.join(input_terms)})"
    weight_offsets = _packed_qkv_weight_offsets(model, prefix, weight_batch_offset, "offs_n[:, None]", "offs_k[None, :]")
    output_offsets = _packed_qkv_output_offsets(model, prefix, output_batch_offset, "offs_n", "m_idx")

    _line(lines, indent, f"for {prefix.lower()}_n_start in tl.range(0, {_dim(n)}, {block_n}, num_stages={n_stages}):")
    _line(lines, indent + 1, f"offs_n = {prefix.lower()}_n_start + tl.arange(0, {block_n})")
    _line(lines, indent + 1, f"acc = tl.zeros(({block_n},), tl.float32)")
    _line(lines, indent + 1, f"for k_start in tl.range(0, {_dim(k)}, {block_k}):")
    _line(lines, indent + 2, f"offs_k = k_start + tl.arange(0, {block_k})")
    weight_mask = f"(offs_n[:, None] < {_dim(n)}) & (offs_k[None, :] < {_dim(k)})"
    weight_value = _load_operand_nd(
        model,
        f"{prefix}Weight",
        ptr_weight,
        weight_offsets,
        weight_mask,
        (block_n, block_k),
        suffix=', eviction_policy="evict_first"',
    )
    _line(lines, indent + 2, f"weight_values = {weight_value}.to(tl.float32)")
    input_mask = f"offs_k < {_dim(k)}"
    _line(lines, indent + 2, f"input_values = {_load_operand(model, 'Input', 'input0', input_offsets, input_mask)}.to(tl.float32)")
    _line(lines, indent + 2, "acc += tl.sum(weight_values * input_values[None, :], axis=1)")
    if model[f"Has{prefix}Bias"]:
        bias_offsets = _packed_qkv_bias_offsets(model, prefix, "offs_n")
        bias_value = _load_operand(model, f"{prefix}Bias", ptr_bias, bias_offsets, f"offs_n < {_dim(n)}")
        _line(lines, indent + 1, f"bias_values = {bias_value}.to(tl.float32)")
        _line(lines, indent + 1, "acc += bias_values")
    _append_store(
        lines,
        indent + 1,
        model,
        f"{prefix}Output",
        ptr_output,
        output_offsets,
        "acc",
        f"offs_n < {_dim(n)}",
    )


def _emit_packed_qkv_matmul_projection(
    lines: list[str],
    indent: int,
    model: dict[str, Any],
    prefix: str,
    input_batch_offset: str,
    block_m: int,
    block_n: int,
    block_k: int,
) -> None:
    weight_shape = model[f"{prefix}WeightShape"]
    output_shape = model[f"{prefix}OutputShape"]
    weight_strides = model[f"{prefix}WeightStrides"]
    output_strides = model[f"{prefix}OutputStrides"]
    output_logical_shape = _packed_qkv_logical_output_shape(model, prefix)
    weight_batch_offset = _batch_offset_expression(weight_shape, weight_strides, len(output_shape) - 2)
    output_batch_offset = _batch_offset_expression(output_shape, output_strides, len(output_shape) - 2)
    n = output_logical_shape[-1]
    k = model["InputShape"][-1]
    ptr_weight = f"{prefix.lower()}_weight"
    ptr_bias = f"{prefix.lower()}_bias"
    ptr_output = f"{prefix.lower()}_output"

    input_terms = [] if input_batch_offset == "0" else [input_batch_offset]
    input_terms += [f"offs_m[:, None] * {_dim(model['InputStrides'][-2])}", f"offs_k[None, :] * {_dim(model['InputStrides'][-1])}"]
    input_offsets = f"({' + '.join(input_terms)})"
    weight_offsets = _packed_qkv_weight_offsets(model, prefix, weight_batch_offset, "offs_n[None, :]", "offs_k[:, None]")
    output_offsets = _packed_qkv_output_offsets(model, prefix, output_batch_offset, "offs_n[None, :]", "offs_m[:, None]")

    _line(lines, indent, f"for {prefix.lower()}_n_start in tl.range(0, {_dim(n)}, {block_n}):")
    _line(lines, indent + 1, f"offs_n = {prefix.lower()}_n_start + tl.arange(0, {block_n})")
    _line(lines, indent + 1, f"acc = tl.zeros(({block_m}, {block_n}), tl.float32)")
    _line(lines, indent + 1, f"for k_start in tl.range(0, {_dim(k)}, {block_k}, num_stages=5):")
    _line(lines, indent + 2, f"offs_k = k_start + tl.arange(0, {block_k})")
    weight_mask = f"(offs_k[:, None] < {_dim(k)}) & (offs_n[None, :] < {_dim(n)})"
    weight_value = _load_operand_nd(
        model,
        f"{prefix}Weight",
        ptr_weight,
        weight_offsets,
        weight_mask,
        (block_k, block_n),
    )
    _line(lines, indent + 2, f"weight_values = {weight_value}")
    input_mask = f"(offs_m[:, None] < {_dim(output_logical_shape[-2])}) & (offs_k[None, :] < {_dim(k)})"
    _line(lines, indent + 2, f"input_values = {_load_operand_nd(model, 'Input', 'input0', input_offsets, input_mask, (block_m, block_k))}")
    dot_precision = ', input_precision="ieee"' if model["InputDType"] == "float32" and model["WeightDType"] == "float32" else ""
    _line(lines, indent + 2, f"acc += tl.dot(input_values, weight_values{dot_precision})")
    if model[f"Has{prefix}Bias"]:
        bias_offsets = _packed_qkv_bias_offsets(model, prefix, "offs_n")
        bias_value = _load_operand(model, f"{prefix}Bias", ptr_bias, bias_offsets, f"offs_n < {_dim(n)}")
        _line(lines, indent + 1, f"bias_values = {bias_value}.to(tl.float32)")
        _line(lines, indent + 1, "acc += bias_values[None, :]")
    _append_store(
        lines,
        indent + 1,
        model,
        f"{prefix}Output",
        ptr_output,
        output_offsets,
        "acc",
        f"(offs_m[:, None] < {_dim(output_logical_shape[-2])}) & (offs_n[None, :] < {_dim(n)})",
    )


def _emit_qkv_gemv_projection(
    lines: list[str],
    indent: int,
    model: dict[str, Any],
    prefix: str,
    input_batch_offset: str,
    block_n: int,
    block_k: int,
    n_stages: int,
) -> None:
    weight_shape = model[f"{prefix}WeightShape"]
    output_shape = model[f"{prefix}OutputShape"]
    weight_strides = model[f"{prefix}WeightStrides"]
    bias_strides = model[f"{prefix}BiasStrides"]
    output_strides = model[f"{prefix}OutputStrides"]
    weight_batch_offset = _batch_offset_expression(weight_shape, weight_strides, len(output_shape) - 2)
    output_batch_offset = _batch_offset_expression(output_shape, output_strides, len(output_shape) - 2)
    n = output_shape[-1]
    k = model["InputShape"][-1]
    ptr_weight = f"{prefix.lower()}_weight"
    ptr_bias = f"{prefix.lower()}_bias"
    ptr_output = f"{prefix.lower()}_output"

    input_terms = [] if input_batch_offset == "0" else [input_batch_offset]
    input_terms += [f"m_idx * {_dim(model['InputStrides'][-2])}", f"offs_k * {_dim(model['InputStrides'][-1])}"]
    input_offsets = f"({' + '.join(input_terms)})"

    weight_terms = [] if weight_batch_offset == "0" else [weight_batch_offset]
    weight_terms += [f"offs_k[None, :] * {_dim(weight_strides[-2])}", f"offs_n[:, None] * {_dim(weight_strides[-1])}"]
    weight_offsets = f"({' + '.join(weight_terms)})"

    output_terms = [] if output_batch_offset == "0" else [output_batch_offset]
    output_terms += [f"m_idx * {_dim(output_strides[-2])}", f"offs_n * {_dim(output_strides[-1])}"]
    output_offsets = f"({' + '.join(output_terms)})"

    _line(lines, indent, f"for {prefix.lower()}_n_start in tl.range(0, {_dim(n)}, {block_n}, num_stages={n_stages}):")
    _line(lines, indent + 1, f"offs_n = {prefix.lower()}_n_start + tl.arange(0, {block_n})")
    _line(lines, indent + 1, f"acc = tl.zeros(({block_n},), tl.float32)")
    _line(lines, indent + 1, f"for k_start in tl.range(0, {_dim(k)}, {block_k}):")
    _line(lines, indent + 2, f"offs_k = k_start + tl.arange(0, {block_k})")
    weight_mask = f"(offs_n[:, None] < {_dim(n)}) & (offs_k[None, :] < {_dim(k)})"
    weight_value = _load_operand_nd(
        model,
        f"{prefix}Weight",
        ptr_weight,
        weight_offsets,
        weight_mask,
        (block_n, block_k),
        suffix=', eviction_policy="evict_first"',
    )
    _line(lines, indent + 2, f"weight_values = {weight_value}.to(tl.float32)")
    input_mask = f"offs_k < {_dim(k)}"
    _line(lines, indent + 2, f"input_values = {_load_operand(model, 'Input', 'input0', input_offsets, input_mask)}.to(tl.float32)")
    _line(lines, indent + 2, "acc += tl.sum(weight_values * input_values[None, :], axis=1)")
    if model[f"Has{prefix}Bias"]:
        bias_offsets = f"offs_n * {_dim(bias_strides[-1])}"
        bias_value = _load_operand(model, f"{prefix}Bias", ptr_bias, bias_offsets, f"offs_n < {_dim(n)}")
        _line(lines, indent + 1, f"bias_values = {bias_value}.to(tl.float32)")
        _line(lines, indent + 1, "acc += bias_values")
    _append_store(
        lines,
        indent + 1,
        model,
        f"{prefix}Output",
        ptr_output,
        output_offsets,
        "acc",
        f"offs_n < {_dim(n)}",
    )


def _emit_qkv_matmul_projection(
    lines: list[str],
    indent: int,
    model: dict[str, Any],
    prefix: str,
    input_batch_offset: str,
    block_m: int,
    block_n: int,
    block_k: int,
) -> None:
    weight_shape = model[f"{prefix}WeightShape"]
    output_shape = model[f"{prefix}OutputShape"]
    weight_strides = model[f"{prefix}WeightStrides"]
    bias_strides = model[f"{prefix}BiasStrides"]
    output_strides = model[f"{prefix}OutputStrides"]
    weight_batch_offset = _batch_offset_expression(weight_shape, weight_strides, len(output_shape) - 2)
    output_batch_offset = _batch_offset_expression(output_shape, output_strides, len(output_shape) - 2)
    n = output_shape[-1]
    k = model["InputShape"][-1]
    ptr_weight = f"{prefix.lower()}_weight"
    ptr_bias = f"{prefix.lower()}_bias"
    ptr_output = f"{prefix.lower()}_output"

    input_terms = [] if input_batch_offset == "0" else [input_batch_offset]
    input_terms += [f"offs_m[:, None] * {_dim(model['InputStrides'][-2])}", f"offs_k[None, :] * {_dim(model['InputStrides'][-1])}"]
    input_offsets = f"({' + '.join(input_terms)})"

    weight_terms = [] if weight_batch_offset == "0" else [weight_batch_offset]
    weight_terms += [f"offs_k[:, None] * {_dim(weight_strides[-2])}", f"offs_n[None, :] * {_dim(weight_strides[-1])}"]
    weight_offsets = f"({' + '.join(weight_terms)})"

    output_terms = [] if output_batch_offset == "0" else [output_batch_offset]
    output_terms += [f"offs_m[:, None] * {_dim(output_strides[-2])}", f"offs_n[None, :] * {_dim(output_strides[-1])}"]
    output_offsets = f"({' + '.join(output_terms)})"

    _line(lines, indent, f"for {prefix.lower()}_n_start in tl.range(0, {_dim(n)}, {block_n}):")
    _line(lines, indent + 1, f"offs_n = {prefix.lower()}_n_start + tl.arange(0, {block_n})")
    _line(lines, indent + 1, f"acc = tl.zeros(({block_m}, {block_n}), tl.float32)")
    _line(lines, indent + 1, f"for k_start in tl.range(0, {_dim(k)}, {block_k}, num_stages=5):")
    _line(lines, indent + 2, f"offs_k = k_start + tl.arange(0, {block_k})")
    weight_mask = f"(offs_k[:, None] < {_dim(k)}) & (offs_n[None, :] < {_dim(n)})"
    weight_value = _load_operand_nd(
        model,
        f"{prefix}Weight",
        ptr_weight,
        weight_offsets,
        weight_mask,
        (block_k, block_n),
    )
    _line(lines, indent + 2, f"weight_values = {weight_value}")
    input_mask = f"(offs_m[:, None] < {_dim(model['QOutputShape'][-2])}) & (offs_k[None, :] < {_dim(k)})"
    _line(lines, indent + 2, f"input_values = {_load_operand_nd(model, 'Input', 'input0', input_offsets, input_mask, (block_m, block_k))}")
    dot_precision = ', input_precision="ieee"' if model["InputDType"] == "float32" and model["WeightDType"] == "float32" else ""
    _line(lines, indent + 2, f"acc += tl.dot(input_values, weight_values{dot_precision})")
    if model[f"Has{prefix}Bias"]:
        bias_offsets = f"offs_n * {_dim(bias_strides[-1])}"
        bias_value = _load_operand(model, f"{prefix}Bias", ptr_bias, bias_offsets, f"offs_n < {_dim(n)}")
        _line(lines, indent + 1, f"bias_values = {bias_value}.to(tl.float32)")
        _line(lines, indent + 1, "acc += bias_values[None, :]")
    _append_store(
        lines,
        indent + 1,
        model,
        f"{prefix}Output",
        ptr_output,
        output_offsets,
        "acc",
        f"(offs_m[:, None] < {_dim(output_shape[-2])}) & (offs_n[None, :] < {_dim(n)})",
    )


def _emit_matmul_glu(model: dict[str, Any]) -> str:
    return _emit_matmul_glu_like(model, packed=False)


def _emit_packed_matmul_glu(model: dict[str, Any]) -> str:
    return _emit_matmul_glu_like(model, packed=True)


def _emit_matmul_glu_like(model: dict[str, Any], *, packed: bool) -> str:
    reduction_phase = str(model.get("ReductionPhase", "complete")).lower()
    if reduction_phase == "accumulate":
        return _emit_matmul_glu_reduction_accumulate(model, packed=packed)
    if reduction_phase == "finalize":
        return _emit_matmul_glu_reduction_finalize(model, packed=packed)
    if reduction_phase != "complete":
        raise ValueError(f"Unsupported MatMulGlu reduction phase: {reduction_phase!r}.")

    logical_output_shape = _matmul_glu_logical_output_shape(model)
    m = logical_output_shape[-2]
    k = model["InputShape"][-1]
    output_batch_rank = len(model["OutputShape"]) - 2
    batch_axes = list(range(output_batch_rank))
    input_batch_offset = _batch_offset_expression(model["InputShape"], model["InputStrides"], output_batch_rank)
    lane_comment = (
        f", n_packed_lane={model['NPackedLaneCount']}, n_lane={model['NVectorLaneCount']}, "
        f"n_scalar_lane={model['NPackedLaneCount'] * model['NVectorLaneCount']}"
        if packed
        else ""
    )
    template_name = "PackedMatMulGlu" if packed else "MatMulGlu"

    lines = _standard_header(
        model,
        f"# generated from PyNTT Jinja {template_name}.py.jinja\n# {model['Comment']}; glu={model['GluType']}, input_dtype={model['InputDType']}, weight_dtype={model['WeightDType']}, output_dtype={model['OutputDType']}, input_shape={_shape_tuple(model['InputShape'])}, gate_weight_shape={_shape_tuple(model['GateWeightShape'])}, up_weight_shape={_shape_tuple(model['UpWeightShape'])}, output_shape={_shape_tuple(model['OutputShape'])}{lane_comment}",
        [
            ("input0", "Input"),
            ("gate_weight", "GateWeight"),
            ("up_weight", "UpWeight"),
            ("gate_bias", "GateBias"),
            ("up_bias", "UpBias"),
            ("output", "Output"),
        ],
    )
    indent = 1
    for axis in batch_axes:
        _line(lines, indent, f"for idx{axis} in tl.range(0, {_dim(logical_output_shape[axis])}):")
        indent += 1

    use_gemv = (_max_value(m) == 1) or (_fixed(m) == 1)
    if use_gemv:
        block_k = 256
        k_max = _max_value(k)
        n_max = _max_value(logical_output_shape[-1]) or 0
        use_large_n = k_max is not None and k_max <= block_k and n_max >= 4096
        block_n = 128 if use_large_n else 32
        n_stages = 2 if use_large_n else 1
        _line(lines, indent, f"for m_idx in tl.range(0, {_dim(m)}):")
        _emit_matmul_glu_gemv(lines, indent + 1, model, input_batch_offset, block_n, block_k, n_stages, packed=packed)
    else:
        block_m = 16
        block_n = 64
        block_k = 64
        _line(lines, indent, f"for m_start in tl.range(0, {_dim(m)}, {block_m}):")
        _line(lines, indent + 1, f"offs_m = m_start + tl.arange(0, {block_m})")
        _emit_matmul_glu_matmul(lines, indent + 1, model, input_batch_offset, block_m, block_n, block_k, packed=packed)
    return _finish(lines)


def _emit_matmul_glu_reduction_accumulate(model: dict[str, Any], *, packed: bool) -> str:
    logical_output_shape = _matmul_glu_logical_output_shape(model)
    if len(logical_output_shape) != 2:
        raise ValueError(
            "MatMulGlu backend-private reduction requires a rank-2 local output tile."
        )
    block_m = int(model["ReductionBlockM"])
    block_n = int(model["ReductionBlockN"])
    block_k = int(model["ReductionBlockK"])
    template_name = "PackedMatMulGlu" if packed else "MatMulGlu"
    lines = _helper_header(
        model,
        ("gate_acc", "up_acc"),
        comment=(
            f"# generated from PyNTT Jinja {template_name}.py.jinja reduction accumulate\n"
            f"# {model['Comment']}; accumulator=backend-private-fp32"
        ),
    )
    _line(lines, 1, "shard_index = tl.program_id(0).to(tl.int64)")
    _append_pointer_shard_coords(
        lines,
        1,
        [model["Input"], model["GateWeight"], model["UpWeight"]],
    )
    _append_pointer_binding(lines, 1, model, "input0", "Input")
    _append_pointer_binding(lines, 1, model, "gate_weight", "GateWeight")
    _append_pointer_binding(lines, 1, model, "up_weight", "UpWeight")
    _line(lines, 1, f"offs_k = tl.arange(0, {block_k})")
    _line(lines, 1, f"offs_n = tl.arange(0, {block_n})")
    k = model["InputShape"][-1]
    n = logical_output_shape[-1]
    input_batch_offset = "0"

    if block_m == 1:
        input_m, input_m_limit = _matmul_glu_input_m_index(model, "0")
        input_offsets = (
            f"({input_m} * {_dim(model['InputStrides'][-2])} + "
            f"offs_k * {_dim(model['InputStrides'][-1])})"
        )
        input_mask = f"({input_m} < {_dim(input_m_limit)}) & (offs_k < {_dim(k)})"
        _line(
            lines,
            1,
            f"input_values = {_load_operand(model, 'Input', 'input0', input_offsets, input_mask)}.to(tl.float32)",
        )
        for prefix, acc_name in (("Gate", "gate_acc"), ("Up", "up_acc")):
            weight_shape = model[f"{prefix}WeightShape"]
            weight_strides = model[f"{prefix}WeightStrides"]
            weight_batch_offset = _batch_offset_expression(
                weight_shape, weight_strides, len(model["OutputShape"]) - 2
            )
            weight_offsets, weight_n_limit, weight_k_limit = _matmul_glu_weight_offsets(
                model,
                prefix,
                weight_batch_offset,
                "offs_n[:, None]",
                "offs_k[None, :]",
                packed=packed,
            )
            weight_mask = (
                f"(offs_n[:, None] < {_dim(n)}) & "
                f"(offs_n[:, None] < {_dim(weight_n_limit)}) & "
                f"(offs_k[None, :] < {_dim(weight_k_limit)})"
            )
            weight_value = _load_operand_nd(
                model,
                f"{prefix}Weight",
                f"{prefix.lower()}_weight",
                weight_offsets,
                weight_mask,
                (block_n, block_k),
                suffix=', eviction_policy="evict_first"',
            )
            _line(
                lines,
                1,
                f"{prefix.lower()}_weight_values = {weight_value}.to(tl.float32)",
            )
            _line(
                lines,
                1,
                f"{acc_name} += tl.sum({prefix.lower()}_weight_values * input_values[None, :], axis=1)",
            )
    else:
        _line(lines, 1, f"offs_m = tl.arange(0, {block_m})")
        input_m, input_m_limit = _matmul_glu_input_m_index(model, "offs_m[:, None]")
        input_offsets = (
            f"({input_m} * {_dim(model['InputStrides'][-2])} + "
            f"offs_k[None, :] * {_dim(model['InputStrides'][-1])})"
        )
        input_mask = (
            f"(offs_m[:, None] < {_dim(logical_output_shape[-2])}) & "
            f"({input_m} < {_dim(input_m_limit)}) & (offs_k[None, :] < {_dim(k)})"
        )
        _line(
            lines,
            1,
            f"input_values = {_load_operand_nd(model, 'Input', 'input0', input_offsets, input_mask, (block_m, block_k))}",
        )
        dot_precision = (
            ', input_precision="ieee"'
            if model["InputDType"] == "float32" and model["WeightDType"] == "float32"
            else ""
        )
        for prefix, acc_name in (("Gate", "gate_acc"), ("Up", "up_acc")):
            weight_shape = model[f"{prefix}WeightShape"]
            weight_strides = model[f"{prefix}WeightStrides"]
            weight_batch_offset = _batch_offset_expression(
                weight_shape, weight_strides, len(model["OutputShape"]) - 2
            )
            weight_offsets, weight_n_limit, weight_k_limit = _matmul_glu_weight_offsets(
                model,
                prefix,
                weight_batch_offset,
                "offs_n[None, :]",
                "offs_k[:, None]",
                packed=packed,
            )
            weight_mask = (
                f"(offs_k[:, None] < {_dim(weight_k_limit)}) & "
                f"(offs_n[None, :] < {_dim(n)}) & "
                f"(offs_n[None, :] < {_dim(weight_n_limit)})"
            )
            weight_value = _load_operand_nd(
                model,
                f"{prefix}Weight",
                f"{prefix.lower()}_weight",
                weight_offsets,
                weight_mask,
                (block_k, block_n),
            )
            _line(
                lines,
                1,
                f"{prefix.lower()}_weight_values = {weight_value}",
            )
            _line(
                lines,
                1,
                f"{acc_name} += tl.dot(input_values, {prefix.lower()}_weight_values{dot_precision})",
            )
    _line(lines, 1, "return gate_acc, up_acc")
    return _finish(lines)


def _emit_matmul_glu_reduction_finalize(model: dict[str, Any], *, packed: bool) -> str:
    logical_output_shape = _matmul_glu_logical_output_shape(model)
    block_m = int(model["ReductionBlockM"])
    block_n = int(model["ReductionBlockN"])
    template_name = "PackedMatMulGlu" if packed else "MatMulGlu"
    lines = _helper_header(
        model,
        ("gate_acc", "up_acc"),
        comment=(
            f"# generated from PyNTT Jinja {template_name}.py.jinja reduction finalize\n"
            f"# {model['Comment']}; accumulator=backend-private-fp32"
        ),
    )
    _line(lines, 1, "shard_index = tl.program_id(0).to(tl.int64)")
    pointer_models = [model["GateBias"], model["UpBias"], model["Output"]]
    _append_pointer_shard_coords(lines, 1, pointer_models)
    if model["HasGateBias"]:
        _append_pointer_binding(lines, 1, model, "gate_bias", "GateBias")
    if model["HasUpBias"]:
        _append_pointer_binding(lines, 1, model, "up_bias", "UpBias")
    _append_pointer_binding(lines, 1, model, "output", "Output")
    _line(lines, 1, f"offs_n = tl.arange(0, {block_n})")
    n = logical_output_shape[-1]
    if block_m == 1:
        if model["HasGateBias"]:
            gate_bias_offsets = _matmul_glu_bias_offsets(model, "Gate", "offs_n", packed=packed)
            _, gate_bias_n_limit = _matmul_glu_bias_n_index(model, "Gate", "offs_n", packed=packed)
            gate_bias_mask = (
                f"(offs_n < {_dim(n)}) & (offs_n < {_dim(gate_bias_n_limit)})"
            )
            gate_bias_value = _load_operand(
                model,
                "GateBias",
                "gate_bias",
                gate_bias_offsets,
                gate_bias_mask,
            )
            _line(lines, 1, f"gate_acc += {gate_bias_value}.to(tl.float32)")
        if model["HasUpBias"]:
            up_bias_offsets = _matmul_glu_bias_offsets(model, "Up", "offs_n", packed=packed)
            _, up_bias_n_limit = _matmul_glu_bias_n_index(model, "Up", "offs_n", packed=packed)
            up_bias_mask = (
                f"(offs_n < {_dim(n)}) & (offs_n < {_dim(up_bias_n_limit)})"
            )
            up_bias_value = _load_operand(
                model,
                "UpBias",
                "up_bias",
                up_bias_offsets,
                up_bias_mask,
            )
            _line(lines, 1, f"up_acc += {up_bias_value}.to(tl.float32)")
        output_offsets = _matmul_glu_output_offsets(model, "0", "offs_n", "0", packed=packed)
        _line(lines, 1, f"result = {_matmul_glu_expr(model, 'gate_acc', 'up_acc')}")
        _append_store(
            lines,
            1,
            model,
            "Output",
            "output",
            output_offsets,
            "result",
            f"offs_n < {_dim(n)}",
        )
    else:
        _line(lines, 1, f"offs_m = tl.arange(0, {block_m})")
        if model["HasGateBias"]:
            gate_bias_offsets = _matmul_glu_bias_offsets(model, "Gate", "offs_n", packed=packed)
            _, gate_bias_n_limit = _matmul_glu_bias_n_index(model, "Gate", "offs_n", packed=packed)
            gate_bias_mask = (
                f"(offs_n < {_dim(n)}) & (offs_n < {_dim(gate_bias_n_limit)})"
            )
            gate_bias_value = _load_operand(
                model,
                "GateBias",
                "gate_bias",
                gate_bias_offsets,
                gate_bias_mask,
            )
            _line(lines, 1, f"gate_acc += {gate_bias_value}.to(tl.float32)[None, :]")
        if model["HasUpBias"]:
            up_bias_offsets = _matmul_glu_bias_offsets(model, "Up", "offs_n", packed=packed)
            _, up_bias_n_limit = _matmul_glu_bias_n_index(model, "Up", "offs_n", packed=packed)
            up_bias_mask = (
                f"(offs_n < {_dim(n)}) & (offs_n < {_dim(up_bias_n_limit)})"
            )
            up_bias_value = _load_operand(
                model,
                "UpBias",
                "up_bias",
                up_bias_offsets,
                up_bias_mask,
            )
            _line(lines, 1, f"up_acc += {up_bias_value}.to(tl.float32)[None, :]")
        output_offsets = _matmul_glu_output_offsets(
            model, "0", "offs_n[None, :]", "offs_m[:, None]", packed=packed
        )
        _line(lines, 1, f"result = {_matmul_glu_expr(model, 'gate_acc', 'up_acc')}")
        _append_store(
            lines,
            1,
            model,
            "Output",
            "output",
            output_offsets,
            "result",
            f"(offs_m[:, None] < {_dim(logical_output_shape[-2])}) & "
            f"(offs_n[None, :] < {_dim(n)})",
        )
    return _finish(lines)


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


def _emit_matmul_glu_gemv(
    lines: list[str],
    indent: int,
    model: dict[str, Any],
    input_batch_offset: str,
    block_n: int,
    block_k: int,
    n_stages: int,
    *,
    packed: bool,
) -> None:
    output_shape = _matmul_glu_logical_output_shape(model)
    output_batch_offset = _batch_offset_expression(model["OutputShape"], model["OutputStrides"], len(model["OutputShape"]) - 2)
    n = output_shape[-1]
    k = model["InputShape"][-1]

    input_m, input_m_limit = _matmul_glu_input_m_index(model, "m_idx")
    input_terms = [] if input_batch_offset == "0" else [input_batch_offset]
    input_terms += [f"{input_m} * {_dim(model['InputStrides'][-2])}", f"offs_k * {_dim(model['InputStrides'][-1])}"]
    input_offsets = f"({' + '.join(input_terms)})"
    output_offsets = _matmul_glu_output_offsets(model, output_batch_offset, "offs_n", "m_idx", packed=packed)

    _line(lines, indent, f"for n_start in tl.range(0, {_dim(n)}, {block_n}, num_stages={n_stages}):")
    _line(lines, indent + 1, f"offs_n = n_start + tl.arange(0, {block_n})")
    _line(lines, indent + 1, f"gate_acc = tl.zeros(({block_n},), tl.float32)")
    _line(lines, indent + 1, f"up_acc = tl.zeros(({block_n},), tl.float32)")
    _line(lines, indent + 1, f"for k_start in tl.range(0, {_dim(k)}, {block_k}):")
    _line(lines, indent + 2, f"offs_k = k_start + tl.arange(0, {block_k})")
    input_mask = f"(m_idx < {_dim(input_m_limit)}) & (offs_k < {_dim(k)})"
    _line(lines, indent + 2, f"input_values = {_load_operand(model, 'Input', 'input0', input_offsets, input_mask)}.to(tl.float32)")
    for prefix, acc_name in (("Gate", "gate_acc"), ("Up", "up_acc")):
        weight_shape = model[f"{prefix}WeightShape"]
        weight_strides = model[f"{prefix}WeightStrides"]
        weight_batch_offset = _batch_offset_expression(weight_shape, weight_strides, len(model["OutputShape"]) - 2)
        weight_offsets, weight_n_limit, weight_k_limit = _matmul_glu_weight_offsets(model, prefix, weight_batch_offset, "offs_n[:, None]", "offs_k[None, :]", packed=packed)
        ptr_weight = f"{prefix.lower()}_weight"
        weight_mask = f"(offs_n[:, None] < {_dim(n)}) & (offs_n[:, None] < {_dim(weight_n_limit)}) & (offs_k[None, :] < {_dim(weight_k_limit)})"
        weight_value = _load_operand_nd(
            model,
            f"{prefix}Weight",
            ptr_weight,
            weight_offsets,
            weight_mask,
            (block_n, block_k),
            suffix=', eviction_policy="evict_first"',
        )
        _line(lines, indent + 2, f"{prefix.lower()}_weight_values = {weight_value}.to(tl.float32)")
        _line(lines, indent + 2, f"{acc_name} += tl.sum({prefix.lower()}_weight_values * input_values[None, :], axis=1)")
    if model["HasGateBias"]:
        gate_bias_offsets = _matmul_glu_bias_offsets(model, "Gate", "offs_n", packed=packed)
        _, gate_bias_n_limit = _matmul_glu_bias_n_index(model, "Gate", "offs_n", packed=packed)
        gate_bias_mask = f"(offs_n < {_dim(n)}) & (offs_n < {_dim(gate_bias_n_limit)})"
        gate_bias_value = _load_operand(
            model, "GateBias", "gate_bias", gate_bias_offsets, gate_bias_mask
        )
        _line(lines, indent + 1, f"gate_acc += {gate_bias_value}.to(tl.float32)")
    if model["HasUpBias"]:
        up_bias_offsets = _matmul_glu_bias_offsets(model, "Up", "offs_n", packed=packed)
        _, up_bias_n_limit = _matmul_glu_bias_n_index(model, "Up", "offs_n", packed=packed)
        up_bias_mask = f"(offs_n < {_dim(n)}) & (offs_n < {_dim(up_bias_n_limit)})"
        up_bias_value = _load_operand(
            model, "UpBias", "up_bias", up_bias_offsets, up_bias_mask
        )
        _line(lines, indent + 1, f"up_acc += {up_bias_value}.to(tl.float32)")
    _line(lines, indent + 1, f"result = {_matmul_glu_expr(model, 'gate_acc', 'up_acc')}")
    _append_store(
        lines,
        indent + 1,
        model,
        "Output",
        "output",
        output_offsets,
        "result",
        f"offs_n < {_dim(n)}",
    )


def _emit_matmul_glu_matmul(
    lines: list[str],
    indent: int,
    model: dict[str, Any],
    input_batch_offset: str,
    block_m: int,
    block_n: int,
    block_k: int,
    *,
    packed: bool,
) -> None:
    output_shape = _matmul_glu_logical_output_shape(model)
    output_batch_offset = _batch_offset_expression(model["OutputShape"], model["OutputStrides"], len(model["OutputShape"]) - 2)
    n = output_shape[-1]
    m = output_shape[-2]
    k = model["InputShape"][-1]
    output_offsets = _matmul_glu_output_offsets(model, output_batch_offset, "offs_n[None, :]", "offs_m[:, None]", packed=packed)

    input_m, input_m_limit = _matmul_glu_input_m_index(model, "offs_m[:, None]")
    input_terms = [] if input_batch_offset == "0" else [input_batch_offset]
    input_terms += [f"{input_m} * {_dim(model['InputStrides'][-2])}", f"offs_k[None, :] * {_dim(model['InputStrides'][-1])}"]
    input_offsets = f"({' + '.join(input_terms)})"

    _line(lines, indent, f"for n_start in tl.range(0, {_dim(n)}, {block_n}):")
    _line(lines, indent + 1, f"offs_n = n_start + tl.arange(0, {block_n})")
    _line(lines, indent + 1, f"gate_acc = tl.zeros(({block_m}, {block_n}), tl.float32)")
    _line(lines, indent + 1, f"up_acc = tl.zeros(({block_m}, {block_n}), tl.float32)")
    _line(lines, indent + 1, f"for k_start in tl.range(0, {_dim(k)}, {block_k}, num_stages=5):")
    _line(lines, indent + 2, f"offs_k = k_start + tl.arange(0, {block_k})")
    input_mask = f"(offs_m[:, None] < {_dim(m)}) & ({input_m} < {_dim(input_m_limit)}) & (offs_k[None, :] < {_dim(k)})"
    _line(lines, indent + 2, f"input_values = {_load_operand_nd(model, 'Input', 'input0', input_offsets, input_mask, (block_m, block_k))}")
    dot_precision = ', input_precision="ieee"' if model["InputDType"] == "float32" and model["WeightDType"] == "float32" else ""
    for prefix, acc_name in (("Gate", "gate_acc"), ("Up", "up_acc")):
        weight_shape = model[f"{prefix}WeightShape"]
        weight_strides = model[f"{prefix}WeightStrides"]
        weight_batch_offset = _batch_offset_expression(weight_shape, weight_strides, len(model["OutputShape"]) - 2)
        weight_offsets, weight_n_limit, weight_k_limit = _matmul_glu_weight_offsets(model, prefix, weight_batch_offset, "offs_n[None, :]", "offs_k[:, None]", packed=packed)
        ptr_weight = f"{prefix.lower()}_weight"
        weight_mask = f"(offs_k[:, None] < {_dim(weight_k_limit)}) & (offs_n[None, :] < {_dim(n)}) & (offs_n[None, :] < {_dim(weight_n_limit)})"
        weight_value = _load_operand_nd(
            model,
            f"{prefix}Weight",
            ptr_weight,
            weight_offsets,
            weight_mask,
            (block_k, block_n),
        )
        _line(lines, indent + 2, f"{prefix.lower()}_weight_values = {weight_value}")
        _line(lines, indent + 2, f"{acc_name} += tl.dot(input_values, {prefix.lower()}_weight_values{dot_precision})")
    if model["HasGateBias"]:
        gate_bias_offsets = _matmul_glu_bias_offsets(model, "Gate", "offs_n", packed=packed)
        _, gate_bias_n_limit = _matmul_glu_bias_n_index(model, "Gate", "offs_n", packed=packed)
        gate_bias_mask = f"(offs_n < {_dim(n)}) & (offs_n < {_dim(gate_bias_n_limit)})"
        gate_bias_value = _load_operand(
            model, "GateBias", "gate_bias", gate_bias_offsets, gate_bias_mask
        )
        _line(lines, indent + 1, f"gate_acc += {gate_bias_value}.to(tl.float32)[None, :]")
    if model["HasUpBias"]:
        up_bias_offsets = _matmul_glu_bias_offsets(model, "Up", "offs_n", packed=packed)
        _, up_bias_n_limit = _matmul_glu_bias_n_index(model, "Up", "offs_n", packed=packed)
        up_bias_mask = f"(offs_n < {_dim(n)}) & (offs_n < {_dim(up_bias_n_limit)})"
        up_bias_value = _load_operand(
            model, "UpBias", "up_bias", up_bias_offsets, up_bias_mask
        )
        _line(lines, indent + 1, f"up_acc += {up_bias_value}.to(tl.float32)[None, :]")
    _line(lines, indent + 1, f"result = {_matmul_glu_expr(model, 'gate_acc', 'up_acc')}")
    _append_store(
        lines,
        indent + 1,
        model,
        "Output",
        "output",
        output_offsets,
        "result",
        f"(offs_m[:, None] < {_dim(m)}) & (offs_n[None, :] < {_dim(n)})",
    )


def _emit_matmul_like(model: dict[str, Any], *, gemv: bool) -> str:
    reduction_phase = str(model.get("ReductionPhase", "complete")).lower()
    if reduction_phase == "accumulate":
        return _emit_matmul_reduction_accumulate(model, gemv=gemv)
    if reduction_phase == "finalize":
        return _emit_matmul_reduction_finalize(model, gemv=gemv)
    if reduction_phase != "complete":
        raise ValueError(f"Unsupported Matmul reduction phase: {reduction_phase!r}.")

    output_n_scalar_lane_count = model.get("OutputNPackedLaneCount", 1) * model["OutputNVectorLaneCount"]
    rhs_n_scalar_lane_count = model.get("RhsNPackedLaneCount", 1) * model["RhsNVectorLaneCount"]
    logical_output_shape = [dict(dim) if isinstance(dim, dict) else dim for dim in model["OutputShape"]]
    logical_output_shape[-1] = _multiply_dim(logical_output_shape[-1], output_n_scalar_lane_count)
    m = logical_output_shape[-2]
    n = logical_output_shape[-1]
    lhs_m = model["LhsShape"][-1] if model["TransposeA"] else model["LhsShape"][-2]
    k = model["LhsShape"][-2] if model["TransposeA"] else model["LhsShape"][-1]
    rhs_k = model["RhsShape"][-1] if model["TransposeB"] else model["RhsShape"][-2]
    rhs_n = _multiply_dim(
        model["RhsShape"][-2] if model["TransposeB"] else model["RhsShape"][-1],
        rhs_n_scalar_lane_count,
    )
    output_batch_rank = len(logical_output_shape) - 2
    batch_axes = list(range(output_batch_rank))
    lhs_batch_offset = _batch_offset_expression(model["LhsShape"], model["LhsStrides"], output_batch_rank)
    rhs_batch_offset = _batch_offset_expression(model["RhsShape"], model["RhsStrides"], output_batch_rank)
    output_batch_offset = _batch_offset_expression(model["OutputShape"], model["OutputStrides"], output_batch_rank)
    lane_comment = (
        f", rhs_n_packed_lane={model.get('RhsNPackedLaneCount', 1)}, rhs_n_lane={model['RhsNVectorLaneCount']}, "
        f"output_n_packed_lane={model.get('OutputNPackedLaneCount', 1)}, output_n_lane={model['OutputNVectorLaneCount']}"
    )
    load_c_expression = str(model.get("LoadCExpression", "False")).strip() or "False"

    def append_load_c(lines: list[str], indent: int, output_offsets: str, output_mask: str) -> None:
        if load_c_expression in ("False", "false", "0"):
            return

        predicate = "True" if load_c_expression in ("True", "true", "1") else f"({load_c_expression})"
        output_pointer = _access_pointer(model, "Output", "output", output_offsets)
        _line(lines, indent, f"previous = tl.load({output_pointer}, mask=({output_mask}) & ({predicate}), other=0.0).to(tl.float32)")
        if predicate == "True":
            _line(lines, indent, "result = previous + result")
        else:
            _line(lines, indent, f"result = tl.where({predicate}, previous + result, result)")

    lines = _standard_header(
        model,
        f"# generated from PyNTT Jinja {'Gemv' if gemv else 'Matmul'}.py.jinja\n# {model['Comment']}; lhs_dtype={model['LhsDType']}, rhs_dtype={model['RhsDType']}, output_dtype={model['OutputDType']}, lhs_shape={_shape_tuple(model['LhsShape'])}, rhs_shape={_shape_tuple(model['RhsShape'])}, output_shape={_shape_tuple(model['OutputShape'])}{lane_comment}, load_c={load_c_expression}",
        [("lhs", "Lhs"), ("rhs", "Rhs"), ("output", "Output")],
    )
    _line(lines, 1, "tmp_shard = shard_index")
    for axis in range(len(model["Hierarchy"]) - 1, -1, -1):
        _line(lines, 1, f"shard_coord{axis} = tmp_shard % {model['Hierarchy'][axis]}")
        _line(lines, 1, f"tmp_shard = tmp_shard // {model['Hierarchy'][axis]}")
    indent = 1
    for axis in batch_axes:
        _line(lines, indent, f"for idx{axis} in tl.range(0, {_dim(logical_output_shape[axis])}):")
        indent += 1

    if gemv:
        block_k = 256
        k_max = _max_value(k)
        n_min = _min_value(n) or 0
        n_max = _max_value(n) or 0
        use_large_n = k_max is not None and k_max <= block_k and n_max >= 4096
        block_n = 128 if use_large_n else 32
        n_stages = 2 if use_large_n else (3 if k_max is not None and k_max <= block_k and n_min >= block_n * 4 else 1)
        lhs_offsets = _gemv_lhs_offsets(model, lhs_batch_offset)
        rhs_offsets = _gemv_rhs_offsets(model, rhs_batch_offset)
        output_offsets = _gemv_output_offsets(model, output_batch_offset)
        _line(lines, indent, f"for m_idx in tl.range(0, {_dim(m)}):")
        indent += 1
        _line(lines, indent, f"for n_start in tl.range(0, {_dim(n)}, {block_n}, num_stages={n_stages}):")
        indent += 1
        _line(lines, indent, f"offs_n = n_start + tl.arange(0, {block_n})")
        _line(lines, indent, f"acc = tl.zeros(({block_n},), tl.float32)")
        _line(lines, indent, f"for k_start in tl.range(0, {_dim(k)}, {block_k}):")
        _line(lines, indent + 1, f"offs_k = k_start + tl.arange(0, {block_k})")
        rhs_mask = f"(offs_n[:, None] < {_dim(n)}) & (offs_n[:, None] < {_dim(rhs_n)}) & (offs_k[None, :] < {_dim(k)}) & (offs_k[None, :] < {_dim(rhs_k)})"
        lhs_mask = f"(m_idx < {_dim(lhs_m)}) & (offs_k < {_dim(k)})"
        rhs_value = _load_operand_nd(
            model,
            "Rhs",
            "rhs",
            rhs_offsets,
            rhs_mask,
            (block_n, block_k),
            suffix=', eviction_policy="evict_first"',
        )
        _line(lines, indent + 1, f"rhs_values = {rhs_value}.to(tl.float32)")
        _line(lines, indent + 1, f"lhs_values = {_load_operand(model, 'Lhs', 'lhs', lhs_offsets, lhs_mask)}.to(tl.float32)")
        _line(lines, indent + 1, "acc += tl.sum(rhs_values * lhs_values[None, :], axis=1)")
        _line(lines, indent, f"result = acc * {model['Scale']}")
        output_mask = f"offs_n < {_dim(n)}"
        append_load_c(lines, indent, output_offsets, output_mask)
        _append_store(
            lines,
            indent,
            model,
            "Output",
            "output",
            output_offsets,
            "result",
            output_mask,
        )
    else:
        block_m = 16
        block_n = 64
        block_k = 64
        lhs_offsets, rhs_offsets, output_offsets = _matmul_offsets(model, lhs_batch_offset, rhs_batch_offset, output_batch_offset)
        compute_indent = indent + 2
        _line(lines, indent, f"for m_start in tl.range(0, {_dim(m)}, {block_m}):")
        _line(lines, indent + 1, f"offs_m = m_start + tl.arange(0, {block_m})")
        _line(lines, indent + 1, f"for n_start in tl.range(0, {_dim(n)}, {block_n}):")
        _line(lines, compute_indent, f"offs_n = n_start + tl.arange(0, {block_n})")
        _line(lines, compute_indent, f"acc = tl.zeros(({block_m}, {block_n}), tl.float32)")
        _line(lines, compute_indent, f"for k_start in tl.range(0, {_dim(k)}, {block_k}, num_stages=5):")
        _line(lines, compute_indent + 1, f"offs_k = k_start + tl.arange(0, {block_k})")
        rhs_mask = f"(offs_k[:, None] < {_dim(k)}) & (offs_k[:, None] < {_dim(rhs_k)}) & (offs_n[None, :] < {_dim(n)}) & (offs_n[None, :] < {_dim(rhs_n)})"
        lhs_mask = f"(offs_m[:, None] < {_dim(m)}) & (offs_m[:, None] < {_dim(lhs_m)}) & (offs_k[None, :] < {_dim(k)})"
        _line(
            lines,
            compute_indent + 1,
            f"rhs_values = {_load_operand_nd(model, 'Rhs', 'rhs', rhs_offsets, rhs_mask, (block_k, block_n))}",
        )
        _line(lines, compute_indent + 1, f"lhs_values = {_load_operand_nd(model, 'Lhs', 'lhs', lhs_offsets, lhs_mask, (block_m, block_k))}")
        dot_precision = ', input_precision="ieee"' if model["LhsDType"] == "float32" and model["RhsDType"] == "float32" else ""
        _line(lines, compute_indent + 1, f"acc += tl.dot(lhs_values, rhs_values{dot_precision})")
        _line(lines, compute_indent, f"result = acc * {model['Scale']}")
        output_mask = f"(offs_m[:, None] < {_dim(m)}) & (offs_n[None, :] < {_dim(n)})"
        append_load_c(lines, compute_indent, output_offsets, output_mask)
        _append_store(
            lines,
            compute_indent,
            model,
            "Output",
            "output",
            output_offsets,
            "result",
            output_mask,
        )
    return _finish(lines)


def _emit_matmul_reduction_accumulate(
    model: dict[str, Any], *, gemv: bool
) -> str:
    output_lane_count = (
        model.get("OutputNPackedLaneCount", 1)
        * model["OutputNVectorLaneCount"]
    )
    rhs_lane_count = (
        model.get("RhsNPackedLaneCount", 1) * model["RhsNVectorLaneCount"]
    )
    logical_output_shape = [
        dict(dim) if isinstance(dim, dict) else dim for dim in model["OutputShape"]
    ]
    logical_output_shape[-1] = _multiply_dim(
        logical_output_shape[-1], output_lane_count
    )
    if len(logical_output_shape) != 2:
        raise ValueError(
            "Matmul backend-private reduction requires a rank-2 local output tile."
        )

    block_m = int(model["ReductionBlockM"])
    block_n = int(model["ReductionBlockN"])
    block_k = int(model["ReductionBlockK"])
    template_name = "Gemv" if gemv else "Matmul"
    lines = _helper_header(
        model,
        ("acc",),
        comment=(
            f"# generated from PyNTT Jinja {template_name}.py.jinja reduction accumulate\n"
            f"# {model['Comment']}; accumulator=backend-private-fp32"
        ),
    )
    _line(lines, 1, "shard_index = tl.program_id(0).to(tl.int64)")
    _append_pointer_shard_coords(lines, 1, [model["Lhs"], model["Rhs"]])
    _append_pointer_binding(lines, 1, model, "lhs", "Lhs")
    _append_pointer_binding(lines, 1, model, "rhs", "Rhs")
    _line(lines, 1, f"offs_k = tl.arange(0, {block_k})")
    _line(lines, 1, f"offs_n = tl.arange(0, {block_n})")

    m = logical_output_shape[-2]
    n = logical_output_shape[-1]
    lhs_m = model["LhsShape"][-1] if model["TransposeA"] else model["LhsShape"][-2]
    lhs_k = model["LhsShape"][-2] if model["TransposeA"] else model["LhsShape"][-1]
    rhs_k = model["RhsShape"][-1] if model["TransposeB"] else model["RhsShape"][-2]
    rhs_n = _multiply_dim(
        model["RhsShape"][-2] if model["TransposeB"] else model["RhsShape"][-1],
        rhs_lane_count,
    )

    if gemv:
        _line(lines, 1, "m_idx = 0")
        lhs_offsets = _gemv_lhs_offsets(model, "0")
        rhs_offsets = _gemv_rhs_offsets(model, "0")
        rhs_mask = (
            f"(offs_n[:, None] < {_dim(n)}) & "
            f"(offs_n[:, None] < {_dim(rhs_n)}) & "
            f"(offs_k[None, :] < {_dim(lhs_k)}) & "
            f"(offs_k[None, :] < {_dim(rhs_k)})"
        )
        lhs_mask = f"(0 < {_dim(m)}) & (0 < {_dim(lhs_m)}) & (offs_k < {_dim(lhs_k)})"
        rhs_value = _load_operand_nd(
            model,
            "Rhs",
            "rhs",
            rhs_offsets,
            rhs_mask,
            (block_n, block_k),
            suffix=', eviction_policy="evict_first"',
        )
        _line(lines, 1, f"rhs_values = {rhs_value}.to(tl.float32)")
        _line(
            lines,
            1,
            f"lhs_values = {_load_operand(model, 'Lhs', 'lhs', lhs_offsets, lhs_mask)}.to(tl.float32)",
        )
        _line(lines, 1, "acc += tl.sum(rhs_values * lhs_values[None, :], axis=1)")
    else:
        _line(lines, 1, f"offs_m = tl.arange(0, {block_m})")
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
        _line(
            lines,
            1,
            f"rhs_values = {_load_operand_nd(model, 'Rhs', 'rhs', rhs_offsets, rhs_mask, (block_k, block_n))}",
        )
        _line(
            lines,
            1,
            f"lhs_values = {_load_operand_nd(model, 'Lhs', 'lhs', lhs_offsets, lhs_mask, (block_m, block_k))}",
        )
        dot_precision = (
            ', input_precision="ieee"'
            if model["LhsDType"] == "float32" and model["RhsDType"] == "float32"
            else ""
        )
        _line(lines, 1, f"acc += tl.dot(lhs_values, rhs_values{dot_precision})")
    _line(lines, 1, "return acc")
    return _finish(lines)


def _emit_matmul_reduction_finalize(
    model: dict[str, Any], *, gemv: bool
) -> str:
    gemv = bool(model.get("Gemv", gemv))
    output_lane_count = (
        model.get("OutputNPackedLaneCount", 1)
        * model["OutputNVectorLaneCount"]
    )
    logical_output_shape = [
        dict(dim) if isinstance(dim, dict) else dim for dim in model["OutputShape"]
    ]
    logical_output_shape[-1] = _multiply_dim(
        logical_output_shape[-1], output_lane_count
    )
    block_m = int(model["ReductionBlockM"])
    block_n = int(model["ReductionBlockN"])
    template_name = "Gemv" if gemv else "Matmul"
    lines = _helper_header(
        model,
        ("acc",),
        comment=(
            f"# generated from PyNTT Jinja {template_name}.py.jinja reduction finalize\n"
            f"# {model['Comment']}; accumulator=backend-private-fp32"
        ),
    )
    _line(lines, 1, "shard_index = tl.program_id(0).to(tl.int64)")
    _append_pointer_shard_coords(lines, 1, [model["Output"]])
    _append_pointer_binding(lines, 1, model, "output", "Output")
    _line(lines, 1, f"offs_n = tl.arange(0, {block_n})")
    n = logical_output_shape[-1]
    result = f"acc * {model['Scale']}"
    if gemv:
        physical_n = _logical_n_index(
            "offs_n",
            model.get("OutputNPackedLaneCount", 1),
            model["OutputNVectorLaneCount"],
        )
        physical = (
            f"(0 * {_dim(model['OutputStrides'][-2])} + "
            f"{physical_n} * {_dim(model['OutputStrides'][-1])})"
        )
        output_offsets = _maybe_vectorized_offset(
            physical,
            _logical_packed_lane_index(
                "offs_n",
                model.get("OutputNPackedLaneCount", 1),
                model["OutputNVectorLaneCount"],
            ),
            _logical_vector_lane_index(
                "offs_n", model["OutputNVectorLaneCount"]
            ),
            model.get("OutputNPackedLaneCount", 1),
            model["OutputNVectorLaneCount"],
        )
        _append_store(
            lines,
            1,
            model,
            "Output",
            "output",
            output_offsets,
            result,
            f"offs_n < {_dim(n)}",
        )
    else:
        _line(lines, 1, f"offs_m = tl.arange(0, {block_m})")
        physical_n = _logical_n_index(
            "offs_n[None, :]",
            model.get("OutputNPackedLaneCount", 1),
            model["OutputNVectorLaneCount"],
        )
        physical = (
            f"(offs_m[:, None] * {_dim(model['OutputStrides'][-2])} + "
            f"{physical_n} * {_dim(model['OutputStrides'][-1])})"
        )
        output_offsets = _maybe_vectorized_offset(
            physical,
            _logical_packed_lane_index(
                "offs_n[None, :]",
                model.get("OutputNPackedLaneCount", 1),
                model["OutputNVectorLaneCount"],
            ),
            _logical_vector_lane_index(
                "offs_n[None, :]", model["OutputNVectorLaneCount"]
            ),
            model.get("OutputNPackedLaneCount", 1),
            model["OutputNVectorLaneCount"],
        )
        _append_store(
            lines,
            1,
            model,
            "Output",
            "output",
            output_offsets,
            result,
            f"(offs_m[:, None] < {_dim(logical_output_shape[-2])}) & "
            f"(offs_n[None, :] < {_dim(n)})",
        )
    return _finish(lines)


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


def _emit_reduce(model: dict[str, Any]) -> str:
    reduction_phase = str(model.get("ReductionPhase", "complete")).lower()
    if reduction_phase == "accumulate":
        return _emit_reduce_reduction_accumulate(model)
    if reduction_phase == "finalize":
        return _emit_reduce_reduction_finalize(model)
    if reduction_phase != "complete":
        raise ValueError(f"Unsupported Reduce reduction phase: {reduction_phase!r}.")

    rank = len(model["OutputShape"])
    axis_set = set(model["Axes"])
    block_axis = _select_block_axis(model["OutputShape"], model["OutputStrides"])
    block_extent = _one() if rank == 0 else model["OutputShape"][block_axis]
    loop_axes = [axis for axis in range(rank) if axis != block_axis]

    def axis_index(axis: int) -> str:
        return "lane" if axis == block_axis else f"out_idx{axis}"

    def output_offset() -> str:
        terms = [f"{axis_index(axis)} * {_dim(model['OutputStrides'][axis])}" for axis in range(len(model["OutputShape"]))]
        return "lane * 0" if not terms else "lane * 0 + " + " + ".join(terms)

    def input_base() -> str:
        terms = []
        output_index = 0
        for input_index in range(len(model["InputShape"])):
            if input_index in axis_set:
                if model["KeepDims"]:
                    output_index += 1
                continue
            terms.append(f"{axis_index(output_index)} * {_dim(model['InputStrides'][input_index])}")
            output_index += 1
        return "lane * 0" if not terms else "lane * 0 + " + " + ".join(terms)

    def reduce_offset() -> str:
        terms = [f"reduce_idx{axis} * {_dim(model['InputStrides'][axis])}" for axis in model["Axes"]]
        return "lane * 0" if not terms else " + ".join(terms)

    lines = _standard_header(
        model,
        f"# generated from PyNTT Jinja Reduce.py.jinja\n# {model['Comment']}; op={model['ReduceOp']}, input_dtype={model['InputDType']}, output_dtype={model['OutputDType']}, input_shape={_shape_tuple(model['InputShape'])}, output_shape={_shape_tuple(model['OutputShape'])}",
        [("input0", "Input"), ("output", "Output")],
    )
    indent = 1
    for axis in loop_axes:
        _line(lines, indent, f"for out_idx{axis} in tl.range(0, {_dim(model['OutputShape'][axis])}):")
        indent += 1
    _line(lines, indent, f"for axis_start in tl.range(0, {_dim(block_extent)}, block_size):")
    _line(lines, indent + 1, "lane = axis_start + tl.arange(0, block_size)")
    _line(lines, indent + 1, f"mask = lane < {_dim(block_extent)}")
    _line(lines, indent + 1, f"input_base = {input_base()}")
    _line(lines, indent + 1, f"output_offsets = {output_offset()}")
    _line(lines, indent + 1, f"acc = tl.full((block_size,), {model['InitValue']}, tl.float32)")
    reduce_indent = indent + 1
    for axis in model["Axes"]:
        _line(lines, reduce_indent, f"for reduce_idx{axis} in tl.range(0, {_dim(model['InputShape'][axis])}):")
        reduce_indent += 1
    reduce_input_offset = f"input_base + {reduce_offset()}"
    _line(
        lines,
        reduce_indent,
        f"value0 = {_load_operand(model, 'Input', 'input0', reduce_input_offset, 'mask', other=str(model['InitValue']))}",
    )
    _line(lines, reduce_indent, f"acc = {model['UpdateExpression']}")
    _line(lines, indent + 1, f"result = {model['FinalizeExpression']}")
    _append_store(
        lines, indent + 1, model, "Output", "output", "output_offsets", "result", "mask"
    )
    return _finish(lines)


def _emit_reduce_reduction_accumulate(model: dict[str, Any]) -> str:
    block_size = int(model["ReductionBlockSize"])
    axis_set = set(model["Axes"])
    track_element_count = bool(model.get("TrackReductionElementCount", False))
    state_parameters = (
        ("acc", "reduced_element_count")
        if track_element_count
        else ("acc",)
    )
    lines = _helper_header(
        model,
        state_parameters,
        comment=(
            "# generated from PyNTT Jinja Reduce.py.jinja reduction accumulate\n"
            f"# {model['Comment']}; op={model['ReduceOp']}, accumulator=backend-private"
        ),
    )
    _line(lines, 1, "shard_index = tl.program_id(0).to(tl.int64)")
    _append_pointer_shard_coords(lines, 1, [model["Input"]])
    _append_pointer_binding(lines, 1, model, "input0", "Input")
    _line(lines, 1, f"lane = tl.arange(0, {block_size})")
    output_elements = _product(model["OutputShape"])
    _line(lines, 1, f"mask = lane < {_dim(output_elements)}")
    if model["OutputShape"]:
        _append_tensor_index_decompose(
            lines, 1, "lane", "out_idx", model["OutputShape"]
        )

    input_terms: list[str] = []
    output_index = 0
    for input_index in range(len(model["InputShape"])):
        if input_index in axis_set:
            if model["KeepDims"]:
                output_index += 1
            continue
        input_terms.append(
            f"out_idx{output_index} * {_dim(model['InputStrides'][input_index])}"
        )
        output_index += 1
    input_base = "lane * 0" if not input_terms else " + ".join(input_terms)
    _line(lines, 1, f"input_base = {input_base}")

    indent = 1
    for axis in model["Axes"]:
        _line(
            lines,
            indent,
            f"for reduce_idx{axis} in tl.range(0, {_dim(model['InputShape'][axis])}):",
        )
        indent += 1
    reduce_terms = [
        f"reduce_idx{axis} * {_dim(model['InputStrides'][axis])}"
        for axis in model["Axes"]
    ]
    reduce_offset = "lane * 0" if not reduce_terms else " + ".join(reduce_terms)
    reduce_input_offset = f"input_base + {reduce_offset}"
    _line(
        lines,
        indent,
        f"value0 = {_load_operand(model, 'Input', 'input0', reduce_input_offset, 'mask', other=str(model['InitValue']))}",
    )
    _line(lines, indent, f"acc = {model['UpdateExpression']}")
    if track_element_count:
        tile_element_count = _product(
            [model["InputShape"][axis] for axis in model["Axes"]]
        )
        _line(
            lines,
            1,
            f"reduced_element_count += {_dim(tile_element_count)}",
        )
        _line(lines, 1, "return acc, reduced_element_count")
    else:
        _line(lines, 1, "return acc")
    return _finish(lines)


def _emit_reduce_reduction_finalize(model: dict[str, Any]) -> str:
    block_size = int(model["ReductionBlockSize"])
    track_element_count = bool(model.get("TrackReductionElementCount", False))
    state_parameters = (
        ("acc", "reduced_element_count")
        if track_element_count
        else ("acc",)
    )
    lines = _helper_header(
        model,
        state_parameters,
        comment=(
            "# generated from PyNTT Jinja Reduce.py.jinja reduction finalize\n"
            f"# {model['Comment']}; accumulator=backend-private"
        ),
    )
    _line(lines, 1, "shard_index = tl.program_id(0).to(tl.int64)")
    _append_pointer_shard_coords(lines, 1, [model["Output"]])
    _append_pointer_binding(lines, 1, model, "output", "Output")
    _line(lines, 1, f"lane = tl.arange(0, {block_size})")
    output_elements = _product(model["OutputShape"])
    _line(lines, 1, f"mask = lane < {_dim(output_elements)}")
    if model["OutputShape"]:
        _append_tensor_index_decompose(
            lines, 1, "lane", "out_idx", model["OutputShape"]
        )
        output_terms = [
            f"out_idx{axis} * {_dim(model['OutputStrides'][axis])}"
            for axis in range(len(model["OutputShape"]))
        ]
        output_offsets = " + ".join(output_terms)
    else:
        output_offsets = "lane * 0"
    _line(lines, 1, f"result = {model['FinalizeExpression']}")
    _append_store(
        lines, 1, model, "Output", "output", output_offsets, "result", "mask"
    )
    return _finish(lines)


def _emit_softmax(model: dict[str, Any]) -> str:
    rank = len(model["Shape"])
    axis_extent = model["Shape"][model["Axis"]]
    block_axis = _select_block_axis(model["Shape"], model["OutputStrides"])
    block_extent = model["Shape"][block_axis]
    loop_axes = [axis for axis in range(rank) if axis != block_axis]

    def axis_index(axis: int) -> str:
        return "lane" if axis == block_axis else f"idx{axis}"

    def offset(strides: list[Any]) -> str:
        terms = [f"{axis_index(axis)} * {_dim(strides[axis])}" for axis in range(len(model["Shape"]))]
        return "lane * 0" if not terms else "lane * 0 + " + " + ".join(terms)

    def slice_base() -> str:
        terms = [
            f"{axis_index(axis)} * {_dim(model['InputStrides'][axis])}"
            for axis in range(len(model["Shape"]))
            if axis != model["Axis"]
        ]
        return "lane * 0" if not terms else "lane * 0 + " + " + ".join(terms)

    lines = _standard_header(
        model,
        f"# generated from PyNTT Jinja Softmax.py.jinja\n# {model['Comment']}; input_dtype={model['InputDType']}, output_dtype={model['OutputDType']}, shape={_shape_tuple(model['Shape'])}, axis={model['Axis']}",
        [("input0", "Input"), ("output", "Output")],
    )
    indent = 1
    for axis in loop_axes:
        _line(lines, indent, f"for idx{axis} in tl.range(0, {_dim(model['Shape'][axis])}):")
        indent += 1
    _line(lines, indent, f"for axis_start in tl.range(0, {_dim(block_extent)}, block_size):")
    _line(lines, indent + 1, "lane = axis_start + tl.arange(0, block_size)")
    _line(lines, indent + 1, f"mask = lane < {_dim(block_extent)}")
    _line(lines, indent + 1, f"slice_base = {slice_base()}")
    _line(lines, indent + 1, 'max_value = tl.full((block_size,), -float("inf"), tl.float32)')
    _line(lines, indent + 1, f"for axis_pos in tl.range(0, {_dim(axis_extent)}):")
    slice_offset = f'slice_base + axis_pos * {_dim(model["InputStrides"][model["Axis"]])}'
    negative_infinity = '-float("inf")'
    _line(
        lines,
        indent + 2,
        f'values = {_load_operand(model, "Input", "input0", slice_offset, "mask", other=negative_infinity)}',
    )
    _line(lines, indent + 2, "max_value = tl.maximum(max_value, values)")
    _line(lines, indent + 1, "sum_value = tl.full((block_size,), 0.0, tl.float32)")
    _line(lines, indent + 1, f"for axis_pos in tl.range(0, {_dim(axis_extent)}):")
    _line(
        lines,
        indent + 2,
        f'values = {_load_operand(model, "Input", "input0", slice_offset, "mask", other=negative_infinity)}',
    )
    _line(lines, indent + 2, "sum_value += tl.exp(values - max_value)")
    input_offset = offset(model["InputStrides"])
    _line(
        lines,
        indent + 1,
        f'value0 = {_load_operand(model, "Input", "input0", input_offset, "mask", other=negative_infinity)}',
    )
    _line(lines, indent + 1, "result = tl.exp(value0 - max_value) / sum_value")
    _append_store(
        lines,
        indent + 1,
        model,
        "Output",
        "output",
        offset(model["OutputStrides"]),
        "result",
        "mask",
    )
    return _finish(lines)


def _emit_layer_norm(model: dict[str, Any]) -> str:
    logical_input_shape = _logical_shape(model["InputShape"], model["InputVectorLaneCount"])
    logical_scale_shape = _logical_shape(model["ScaleShape"], model["ScaleVectorLaneCount"])
    logical_bias_shape = _logical_shape(model["BiasShape"], model["BiasVectorLaneCount"])
    logical_output_shape = _logical_shape(model["OutputShape"], model["OutputVectorLaneCount"])
    rank = len(logical_output_shape)
    inner_axes = list(range(model["Axis"], rank))
    outer_axes = list(range(0, model["Axis"]))

    def axis_index(axis: int) -> str:
        return f"outer_idx{axis}" if axis < model["Axis"] else f"inner_idx{axis}"

    def tensor_offset(physical_shape: list[Any], logical_shape: list[Any], strides: list[Any], lane_count: int) -> str:
        terms = []
        for axis in range(rank):
            if _is_fixed_one(logical_shape[axis]):
                continue
            index = axis_index(axis)
            if lane_count > 1 and axis == len(physical_shape) - 1:
                index = f"(({index}) // {lane_count})"
            terms.append(f"{index} * {_dim(strides[axis])}")
        physical_index = "inner_lane * 0" if not terms else "inner_lane * 0 + " + " + ".join(terms)
        if lane_count == 1:
            return physical_index
        lane_index = f"(({axis_index(len(physical_shape) - 1)}) % {lane_count})"
        return f"(({physical_index}) * {lane_count} + {lane_index})"

    def param_offset(physical_shape: list[Any], logical_shape: list[Any], strides: list[Any], lane_count: int) -> str:
        terms = []
        for axis in range(len(logical_shape)):
            if _is_fixed_one(logical_shape[axis]):
                continue
            output_axis = model["Axis"] + axis
            index = axis_index(output_axis)
            if lane_count > 1 and axis == len(physical_shape) - 1:
                index = f"(({index}) // {lane_count})"
            terms.append(f"{index} * {_dim(strides[axis])}")
        physical_index = "inner_lane * 0" if not terms else "inner_lane * 0 + " + " + ".join(terms)
        if lane_count == 1:
            return physical_index
        lane_index = f"(({axis_index(model['Axis'] + len(physical_shape) - 1)}) % {lane_count})"
        return f"(({physical_index}) * {lane_count} + {lane_index})"

    def append_inner_decode(lines: list[str], indent: int) -> None:
        _line(lines, indent, "inner_remaining = inner_lane")
        for axis in reversed(inner_axes):
            _line(lines, indent, f"inner_idx{axis} = inner_remaining % {_dim(logical_output_shape[axis])}")
            _line(lines, indent, f"inner_remaining = inner_remaining // {_dim(logical_output_shape[axis])}")

    inner_size = _product(logical_output_shape[model["Axis"] :])
    input_offset = tensor_offset(model["InputShape"], logical_input_shape, model["InputStrides"], model["InputVectorLaneCount"])
    output_offset = tensor_offset(model["OutputShape"], logical_output_shape, model["OutputStrides"], model["OutputVectorLaneCount"])
    scale_offset = param_offset(model["ScaleShape"], logical_scale_shape, model["ScaleStrides"], model["ScaleVectorLaneCount"])
    bias_offset = param_offset(model["BiasShape"], logical_bias_shape, model["BiasStrides"], model["BiasVectorLaneCount"])
    lines = _standard_header(
        model,
        f"# generated from PyNTT Jinja LayerNorm.py.jinja\n# {model['Comment']}; input_dtype={model['InputDType']}, scale_dtype={model['ScaleDType']}, bias_dtype={model['BiasDType']}, output_dtype={model['OutputDType']}, output_shape={_shape_tuple(model['OutputShape'])}, axis={model['Axis']}",
        [("input0", "Input"), ("scale0", "Scale"), ("bias0", "Bias"), ("output", "Output")],
    )
    indent = 1
    for axis in outer_axes:
        _line(lines, indent, f"for outer_idx{axis} in tl.range(0, {_dim(logical_output_shape[axis])}):")
        indent += 1
    _line(lines, indent, f"inner_size = {inner_size}")
    _line(lines, indent, "inner_count = inner_size + 0.0")
    if model["UseMean"]:
        _line(lines, indent, "mean_sum = tl.full((), 0.0, tl.float32)")
        _line(lines, indent, "for inner_start in tl.range(0, inner_size, block_size):")
        _line(lines, indent + 1, "inner_lane = inner_start + tl.arange(0, block_size)")
        _line(lines, indent + 1, "mask = inner_lane < inner_size")
        append_inner_decode(lines, indent + 1)
        _line(lines, indent + 1, f"input_offsets = {input_offset}")
        _line(lines, indent + 1, f"value0 = {_load_operand(model, 'Input', 'input0', 'input_offsets', 'mask')}.to(tl.float32)")
        _line(lines, indent + 1, "mean_sum += tl.sum(value0, axis=0)")
        _line(lines, indent, "mean_value = mean_sum / inner_count")
    else:
        _line(lines, indent, "mean_value = 0.0")
    _line(lines, indent, "variance_sum = tl.full((), 0.0, tl.float32)")
    _line(lines, indent, "for inner_start in tl.range(0, inner_size, block_size):")
    _line(lines, indent + 1, "inner_lane = inner_start + tl.arange(0, block_size)")
    _line(lines, indent + 1, "mask = inner_lane < inner_size")
    append_inner_decode(lines, indent + 1)
    _line(lines, indent + 1, f"input_offsets = {input_offset}")
    _line(lines, indent + 1, f"value0 = {_load_operand(model, 'Input', 'input0', 'input_offsets', 'mask')}.to(tl.float32)")
    _line(lines, indent + 1, "centered = value0 - mean_value")
    _line(lines, indent + 1, "variance_sum += tl.sum(centered * centered, axis=0)")
    _line(lines, indent, f"inv_std = tl.rsqrt((variance_sum / inner_count) + {model['Epsilon']!r})")
    _line(lines, indent, "for inner_start in tl.range(0, inner_size, block_size):")
    _line(lines, indent + 1, "inner_lane = inner_start + tl.arange(0, block_size)")
    _line(lines, indent + 1, "mask = inner_lane < inner_size")
    append_inner_decode(lines, indent + 1)
    _line(lines, indent + 1, f"input_offsets = {input_offset}")
    _line(lines, indent + 1, f"scale_offsets = {scale_offset}")
    _line(lines, indent + 1, f"bias_offsets = {bias_offset}")
    _line(lines, indent + 1, f"output_offsets = {output_offset}")
    _line(lines, indent + 1, f"value0 = {_load_operand(model, 'Input', 'input0', 'input_offsets', 'mask')}.to(tl.float32)")
    _line(lines, indent + 1, f"scale_value = {_load_operand(model, 'Scale', 'scale0', 'scale_offsets', 'mask')}.to(tl.float32)")
    _line(lines, indent + 1, f"bias_value = {_load_operand(model, 'Bias', 'bias0', 'bias_offsets', 'mask')}.to(tl.float32)")
    _line(lines, indent + 1, "result = ((value0 - mean_value) * inv_std * scale_value) + bias_value")
    _append_store(
        lines, indent + 1, model, "Output", "output", "output_offsets", "result", "mask"
    )
    return _finish(lines)


def _emit_norm_stats(model: dict[str, Any]) -> str:
    logical_input_shape = _logical_shape(model["InputShape"], model["InputVectorLaneCount"])
    rank = len(logical_input_shape)
    inner_axes = list(range(model["Axis"], rank))
    outer_axes = list(range(0, model["Axis"]))

    def axis_index(axis: int) -> str:
        return f"outer_idx{axis}" if axis < model["Axis"] else f"inner_idx{axis}"

    def input_offset() -> str:
        terms = []
        for axis in range(rank):
            if _is_fixed_one(logical_input_shape[axis]):
                continue
            index = axis_index(axis)
            if model["InputVectorLaneCount"] > 1 and axis == len(model["InputShape"]) - 1:
                index = f"(({index}) // {model['InputVectorLaneCount']})"
            terms.append(f"{index} * {_dim(model['InputStrides'][axis])}")
        physical_index = "inner_lane * 0" if not terms else "inner_lane * 0 + " + " + ".join(terms)
        if model["InputVectorLaneCount"] == 1:
            return physical_index
        lane_index = f"(({axis_index(len(model['InputShape']) - 1)}) % {model['InputVectorLaneCount']})"
        return f"(({physical_index}) * {model['InputVectorLaneCount']} + {lane_index})"

    def stats_offset(component: int) -> str:
        terms = []
        if component != 0:
            terms.append(f"{component} * {_dim(model['OutputStrides'][0])}")
        for axis in outer_axes:
            if _is_fixed_one(logical_input_shape[axis]):
                continue
            terms.append(f"outer_idx{axis} * {_dim(model['OutputStrides'][axis + 1])}")
        return "0" if not terms else " + ".join(terms)

    def append_inner_decode(lines: list[str], indent: int) -> None:
        _line(lines, indent, "inner_remaining = inner_lane")
        for axis in reversed(inner_axes):
            _line(lines, indent, f"inner_idx{axis} = inner_remaining % {_dim(logical_input_shape[axis])}")
            _line(lines, indent, f"inner_remaining = inner_remaining // {_dim(logical_input_shape[axis])}")

    inner_size = _product(logical_input_shape[model["Axis"] :])
    lines = _standard_header(
        model,
        f"# generated from PyNTT Jinja NormStats.py.jinja\n# {model['Comment']}; input_dtype={model['InputDType']}, output_dtype={model['OutputDType']}, input_shape={_shape_tuple(model['InputShape'])}, output_shape={_shape_tuple(model['OutputShape'])}, axis={model['Axis']}",
        [("input0", "Input"), ("output", "Output")],
    )
    indent = 1
    for axis in outer_axes:
        _line(lines, indent, f"for outer_idx{axis} in tl.range(0, {_dim(logical_input_shape[axis])}):")
        indent += 1
    if model["UseMean"]:
        _line(lines, indent, "mean_sum = tl.full((), 0.0, tl.float32)")
    _line(lines, indent, "square_sum = tl.full((), 0.0, tl.float32)")
    _line(lines, indent, f"inner_size = {inner_size}")
    _line(lines, indent, "for inner_start in tl.range(0, inner_size, block_size):")
    _line(lines, indent + 1, "inner_lane = inner_start + tl.arange(0, block_size)")
    _line(lines, indent + 1, "mask = inner_lane < inner_size")
    append_inner_decode(lines, indent + 1)
    _line(lines, indent + 1, f"input_offsets = {input_offset()}")
    _line(
        lines,
        indent + 1,
        f"value0 = {_load_operand(model, 'Input', 'input0', 'input_offsets', 'mask')}.to(tl.float32)",
    )
    if model["UseMean"]:
        _line(lines, indent + 1, "mean_sum += tl.sum(value0, axis=0)")
    _line(lines, indent + 1, "square_sum += tl.sum(value0 * value0, axis=0)")
    if model["UseMean"]:
        _append_store(
            lines, indent, model, "Output", "output", stats_offset(0), "mean_sum", "True"
        )
        _append_store(
            lines,
            indent,
            model,
            "Output",
            "output",
            stats_offset(1),
            "square_sum",
            "True",
        )
    else:
        _append_store(
            lines,
            indent,
            model,
            "Output",
            "output",
            stats_offset(0),
            "square_sum",
            "True",
        )
    return _finish(lines)


def _emit_norm_apply(model: dict[str, Any]) -> str:
    logical_input_shape = _logical_shape(model["InputShape"], model["InputVectorLaneCount"])
    logical_input_global_shape = _logical_shape(model["InputGlobalShape"], model["InputVectorLaneCount"])
    logical_scale_shape = _logical_shape(model["ScaleShape"], model["ScaleVectorLaneCount"])
    logical_bias_shape = _logical_shape(model["BiasShape"], model["BiasVectorLaneCount"])
    logical_output_shape = _logical_shape(model["OutputShape"], model["OutputVectorLaneCount"])
    rank = len(logical_output_shape)
    inner_axes = list(range(model["Axis"], rank))
    outer_axes = list(range(0, model["Axis"]))

    def axis_index(axis: int) -> str:
        return f"outer_idx{axis}" if axis < model["Axis"] else f"inner_idx{axis}"

    def tensor_offset(physical_shape: list[Any], logical_shape: list[Any], strides: list[Any], lane_count: int) -> str:
        terms = []
        for axis in range(rank):
            if _is_fixed_one(logical_shape[axis]):
                continue
            index = axis_index(axis)
            if lane_count > 1 and axis == len(physical_shape) - 1:
                index = f"(({index}) // {lane_count})"
            terms.append(f"{index} * {_dim(strides[axis])}")
        physical_index = "inner_lane * 0" if not terms else "inner_lane * 0 + " + " + ".join(terms)
        if lane_count == 1:
            return physical_index
        lane_index = f"(({axis_index(len(physical_shape) - 1)}) % {lane_count})"
        return f"(({physical_index}) * {lane_count} + {lane_index})"

    def param_offset(physical_shape: list[Any], logical_shape: list[Any], strides: list[Any], lane_count: int) -> str:
        terms = []
        for axis in range(len(logical_shape)):
            if _is_fixed_one(logical_shape[axis]):
                continue
            output_axis = model["Axis"] + axis
            index = axis_index(output_axis)
            if lane_count > 1 and axis == len(physical_shape) - 1:
                index = f"(({index}) // {lane_count})"
            terms.append(f"{index} * {_dim(strides[axis])}")
        physical_index = "inner_lane * 0" if not terms else "inner_lane * 0 + " + " + ".join(terms)
        if lane_count == 1:
            return physical_index
        lane_index = f"(({axis_index(model['Axis'] + len(physical_shape) - 1)}) % {lane_count})"
        return f"(({physical_index}) * {lane_count} + {lane_index})"

    def stats_offset(component: int) -> str:
        terms = []
        if component != 0:
            terms.append(f"{component} * {_dim(model['StatsStrides'][0])}")
        for axis in outer_axes:
            if _is_fixed_one(logical_output_shape[axis]):
                continue
            terms.append(f"outer_idx{axis} * {_dim(model['StatsStrides'][axis + 1])}")
        return "0" if not terms else " + ".join(terms)

    def append_inner_decode(lines: list[str], indent: int) -> None:
        _line(lines, indent, "inner_remaining = inner_lane")
        for axis in reversed(inner_axes):
            _line(lines, indent, f"inner_idx{axis} = inner_remaining % {_dim(logical_output_shape[axis])}")
            _line(lines, indent, f"inner_remaining = inner_remaining // {_dim(logical_output_shape[axis])}")

    def stats_load(component: int) -> str:
        offset_value = stats_offset(component)
        return f"tl.load({_access_pointer(model, 'Stats', 'stats0', offset_value)}).to(tl.float32)"

    inner_size = _product(logical_output_shape[model["Axis"] :])
    normalization_size = _product(logical_input_global_shape[model["Axis"] :])
    input_offset = tensor_offset(model["InputShape"], logical_input_shape, model["InputStrides"], model["InputVectorLaneCount"])
    output_offset = tensor_offset(model["OutputShape"], logical_output_shape, model["OutputStrides"], model["OutputVectorLaneCount"])
    scale_offset = param_offset(model["ScaleShape"], logical_scale_shape, model["ScaleStrides"], model["ScaleVectorLaneCount"])
    bias_offset = param_offset(model["BiasShape"], logical_bias_shape, model["BiasStrides"], model["BiasVectorLaneCount"])
    lines = _standard_header(
        model,
        f"# generated from PyNTT Jinja NormApply.py.jinja\n# {model['Comment']}; input_dtype={model['InputDType']}, stats_dtype={model['StatsDType']}, scale_dtype={model['ScaleDType']}, bias_dtype={model['BiasDType']}, output_dtype={model['OutputDType']}, output_shape={_shape_tuple(model['OutputShape'])}, axis={model['Axis']}",
        [("input0", "Input"), ("stats0", "Stats"), ("scale0", "Scale"), ("bias0", "Bias"), ("output", "Output")],
    )
    indent = 1
    for axis in outer_axes:
        _line(lines, indent, f"for outer_idx{axis} in tl.range(0, {_dim(logical_output_shape[axis])}):")
        indent += 1
    _line(lines, indent, f"inner_size = {inner_size}")
    _line(lines, indent, f"inner_count = ({normalization_size}) + 0.0")
    if model["UseMean"]:
        _line(lines, indent, f"mean_sum = {stats_load(0)}")
        _line(lines, indent, f"square_sum = {stats_load(1)}")
        _line(lines, indent, "mean_value = mean_sum / inner_count")
        _line(lines, indent, "variance = (square_sum / inner_count) - (mean_value * mean_value)")
    else:
        _line(lines, indent, f"square_sum = {stats_load(0)}")
        _line(lines, indent, "mean_value = 0.0")
        _line(lines, indent, "variance = square_sum / inner_count")
    _line(lines, indent, "variance = tl.maximum(variance, 0.0)")
    _line(lines, indent, f"inv_std = tl.rsqrt(variance + {model['Epsilon']!r})")
    value_indent = indent + 1
    _line(lines, indent, "for inner_start in tl.range(0, inner_size, block_size):")
    _line(lines, value_indent, "inner_lane = inner_start + tl.arange(0, block_size)")
    _line(lines, value_indent, "mask = inner_lane < inner_size")
    append_inner_decode(lines, value_indent)
    _line(lines, value_indent, f"input_offsets = {input_offset}")
    _line(lines, value_indent, f"scale_offsets = {scale_offset}")
    _line(lines, value_indent, f"bias_offsets = {bias_offset}")
    _line(lines, value_indent, f"output_offsets = {output_offset}")
    _line(lines, value_indent, f"value0 = {_load_operand(model, 'Input', 'input0', 'input_offsets', 'mask')}.to(tl.float32)")
    _line(lines, value_indent, f"scale_value = {_load_operand(model, 'Scale', 'scale0', 'scale_offsets', 'mask')}.to(tl.float32)")
    _line(lines, value_indent, f"bias_value = {_load_operand(model, 'Bias', 'bias0', 'bias_offsets', 'mask')}.to(tl.float32)")
    _line(lines, value_indent, "result = ((value0 - mean_value) * inv_std * scale_value) + bias_value")
    _append_store(
        lines,
        value_indent,
        model,
        "Output",
        "output",
        "output_offsets",
        "result",
        "mask",
    )
    return _finish(lines)


def _emit_rope(model: dict[str, Any]) -> str:
    rank = len(model["OutputShape"])
    rotary_axis = model["RotaryAxis"]
    sincos_pack_factor = int(model.get("SinCosVectorPackFactor", 1))
    total = _multiply_expr(_product(model["OutputShape"]), model["OutputVectorLaneCount"])
    rotary_extent = _multiply_expr(f"({_dim(model['OutputShape'][rotary_axis])})", model["OutputVectorLaneCount"])
    half_dim = f"(({rotary_extent}) // 2)"

    def offset(operand_shape: list[Any], strides: list[Any], lane_count: int, rotary_index: str, lane_index: str) -> str:
        axis_offset = rank - len(operand_shape)
        terms = []
        for axis, dim in enumerate(operand_shape):
            if _is_fixed_one(dim):
                continue
            output_axis = axis_offset + axis
            index = rotary_index if output_axis == rotary_axis else f"idx{output_axis}"
            terms.append(f"{index} * {_dim(strides[axis])}")
        tensor = "lane_flat * 0" if not terms else " + ".join(terms)
        return tensor if lane_count == 1 else f"(({tensor}) * {lane_count} + {lane_index})"

    def output_offset() -> str:
        terms = [f"idx{axis} * {_dim(stride)}" for axis, stride in enumerate(model["OutputStrides"])]
        tensor = "lane_flat * 0" if not terms else " + ".join(terms)
        lanes = model["OutputVectorLaneCount"]
        return tensor if lanes == 1 else f"(({tensor}) * {lanes} + lane_flat)"

    lines = _standard_header(
        model,
        f"# generated from PyNTT Jinja RoPE.py.jinja\n# {model['Comment']}; input_dtype={model['InputDType']}, cos_dtype={model['CosDType']}, sin_dtype={model['SinDType']}, output_dtype={model['OutputDType']}, shape={_shape_tuple(model['OutputShape'])}, rotary_axis={rotary_axis}",
        [("input0", "Input"), ("cos0", "Cos"), ("sin0", "Sin"), ("output", "Output")],
    )
    indent = 2
    _line(lines, 1, f"for linear_start in tl.range(0, {total}, block_size):")
    _line(lines, indent, "linear = linear_start + tl.arange(0, block_size)")
    _line(lines, indent, f"mask = linear < {total}")
    _line(lines, indent, f"lane_flat = linear % {model['OutputVectorLaneCount']}")
    _line(lines, indent, f"output_tensor_linear = linear // {model['OutputVectorLaneCount']}")
    _append_tensor_index_decompose(lines, indent, "output_tensor_linear", "idx", model["OutputShape"])
    _line(lines, indent, f"logical_rotary = idx{rotary_axis} * {model['OutputVectorLaneCount']} + lane_flat")
    _line(lines, indent, f"paired_logical = tl.where(logical_rotary < {half_dim}, logical_rotary + {half_dim}, logical_rotary - {half_dim})")
    _line(lines, indent, f"paired_idx{rotary_axis} = paired_logical // {model['OutputVectorLaneCount']}")
    _line(lines, indent, f"paired_lane = paired_logical % {model['OutputVectorLaneCount']}")
    if sincos_pack_factor == 1:
        cos_rotary_index = f"idx{rotary_axis}"
        cos_lane_index = "lane_flat"
    else:
        _line(lines, indent, f"sincos_rotary = logical_rotary // ({sincos_pack_factor} * {model['OutputVectorLaneCount']})")
        _line(lines, indent, f"sincos_lane = ((logical_rotary // {model['OutputVectorLaneCount']}) % {sincos_pack_factor}) * {model['OutputVectorLaneCount']} + lane_flat")
        cos_rotary_index = "sincos_rotary"
        cos_lane_index = "sincos_lane"
    _line(lines, indent, f"input_offsets = {offset(model['InputShape'], model['InputStrides'], model['InputVectorLaneCount'], f'idx{rotary_axis}', 'lane_flat')}")
    _line(lines, indent, f"paired_input_offsets = {offset(model['InputShape'], model['InputStrides'], model['InputVectorLaneCount'], f'paired_idx{rotary_axis}', 'paired_lane')}")
    _line(lines, indent, f"cos_offsets = {offset(model['CosShape'], model['CosStrides'], model['CosVectorLaneCount'], cos_rotary_index, cos_lane_index)}")
    _line(lines, indent, f"sin_offsets = {offset(model['SinShape'], model['SinStrides'], model['SinVectorLaneCount'], cos_rotary_index, cos_lane_index)}")
    _line(lines, indent, f"output_offsets = {output_offset()}")
    _line(lines, indent, f"value0 = {_load_operand(model, 'Input', 'input0', 'input_offsets', 'mask')}.to(tl.float32)")
    _line(lines, indent, f"paired_value = {_load_operand(model, 'Input', 'input0', 'paired_input_offsets', 'mask')}.to(tl.float32)")
    _line(lines, indent, f"cos_value = {_load_operand(model, 'Cos', 'cos0', 'cos_offsets', 'mask')}.to(tl.float32)")
    _line(lines, indent, f"sin_value = {_load_operand(model, 'Sin', 'sin0', 'sin_offsets', 'mask')}.to(tl.float32)")
    _line(lines, indent, f"rotated = tl.where(logical_rotary < {half_dim}, -paired_value, paired_value)")
    _line(lines, indent, "result = value0 * cos_value + rotated * sin_value")
    _append_store(
        lines, indent, model, "Output", "output", "output_offsets", "result", "mask"
    )
    return _finish(lines)


def _emit_gather(model: dict[str, Any]) -> str:
    logical_output_shape = [dict(dim) if isinstance(dim, dict) else dim for dim in model["OutputShape"]]
    logical_output_shape[-1] = _multiply_dim(logical_output_shape[-1], model["ValueVectorLaneCount"])
    logical_output_strides = [dict(dim) if isinstance(dim, dict) else dim for dim in model["OutputStrides"]]
    if model["ValueVectorLaneCount"] > 1:
        logical_output_strides[-1] = _one()
    output_rank = len(logical_output_shape)
    index_rank = len(model["IndexShape"])
    block_axis = _select_block_axis(logical_output_shape, logical_output_strides)
    block_extent = _one() if output_rank == 0 else logical_output_shape[block_axis]
    loop_axes = [axis for axis in range(output_rank) if axis != block_axis]
    signed_index = not str(model["IndexDType"]).startswith("uint")

    def axis_index(axis: int) -> str:
        return "lane" if axis == block_axis else f"idx{axis}"

    def output_offset() -> str:
        terms = []
        for output_axis in range(len(model["OutputShape"])):
            index = (
                f"(({axis_index(output_axis)}) // {model['ValueVectorLaneCount']})"
                if output_axis == len(model["OutputShape"]) - 1 and model["ValueVectorLaneCount"] > 1
                else axis_index(output_axis)
            )
            terms.append(f"{index} * {_dim(model['OutputStrides'][output_axis])}")
        physical = "lane * 0" if not terms else "lane * 0 + " + " + ".join(terms)
        if model["ValueVectorLaneCount"] == 1:
            return physical
        lane_index = f"(({axis_index(len(model['OutputShape']) - 1)}) % {model['ValueVectorLaneCount']})"
        return f"(({physical}) * {model['ValueVectorLaneCount']} + {lane_index})"

    def index_offset() -> str:
        terms = [
            f"{axis_index(model['Axis'] + index_axis)} * {_dim(model['IndexStrides'][index_axis])}"
            for index_axis in range(index_rank)
        ]
        return "lane * 0" if not terms else "lane * 0 + " + " + ".join(terms)

    def input_offset() -> str:
        terms = []
        for input_axis in range(len(model["InputShape"])):
            if input_axis < model["Axis"]:
                index = axis_index(input_axis)
            elif input_axis == model["Axis"]:
                index = "local_gather_index"
            else:
                index = axis_index(input_axis + index_rank - 1)
            if input_axis == len(model["InputShape"]) - 1 and model["ValueVectorLaneCount"] > 1:
                index = f"(({index}) // {model['ValueVectorLaneCount']})"
            terms.append(f"{index} * {_dim(model['InputStrides'][input_axis])}")
        physical = "lane * 0" if not terms else "lane * 0 + " + " + ".join(terms)
        if model["ValueVectorLaneCount"] == 1:
            return physical
        value_output_axis = len(model["InputShape"]) - 1 + index_rank - 1
        lane_index = f"(({axis_index(value_output_axis)}) % {model['ValueVectorLaneCount']})"
        return f"(({physical}) * {model['ValueVectorLaneCount']} + {lane_index})"

    lines = _standard_header(
        model,
        f"# generated from PyNTT Jinja Gather.py.jinja\n# {model['Comment']}; input_dtype={model['InputDType']}, index_dtype={model['IndexDType']}, output_dtype={model['OutputDType']}, input_shape={_shape_tuple(model['InputShape'])}, output_shape={_shape_tuple(model['OutputShape'])}, axis={model['Axis']}",
        [("input", "Input"), ("index", "Index"), ("output", "Output")],
    )
    _line(lines, 1, "tmp_shard = shard_index")
    for axis in range(len(model["Hierarchy"]) - 1, -1, -1):
        _line(lines, 1, f"shard_coord{axis} = tmp_shard % {model['Hierarchy'][axis]}")
        _line(lines, 1, f"tmp_shard = tmp_shard // {model['Hierarchy'][axis]}")
    indent = 1
    for axis in loop_axes:
        _line(lines, indent, f"for idx{axis} in tl.range(0, {_dim(logical_output_shape[axis])}):")
        indent += 1
    _line(lines, indent, f"for axis_start in tl.range(0, {_dim(block_extent)}, block_size):")
    _line(lines, indent + 1, "lane = axis_start + tl.arange(0, block_size)")
    _line(lines, indent + 1, f"mask = lane < {_dim(block_extent)}")
    _line(lines, indent + 1, f"index_offsets = {index_offset()}")
    _line(
        lines,
        indent + 1,
        f"gather_index = {_load_operand(model, 'Index', 'index', 'index_offsets', 'mask', other='0')}",
    )
    if signed_index:
        _line(lines, indent + 1, f"gather_index = tl.where(gather_index < 0, gather_index + {_dim(model['InputGlobalShape'][model['Axis']])}, gather_index)")
    gather_split_axes = model["InputSplitAxes"][model["Axis"]]
    _line(lines, indent + 1, f"input_active = (gather_index >= 0) & (gather_index < {_dim(model['InputGlobalShape'][model['Axis']])})")
    if not gather_split_axes:
        _line(lines, indent + 1, "local_gather_index = gather_index")
    else:
        _line(lines, indent + 1, f"input_local_dim = {_dim(model['InputShape'][model['Axis']])}")
        _line(lines, indent + 1, f"input_split_linear = {_split_linear_expression(gather_split_axes, model['Hierarchy'])}")
        _line(lines, indent + 1, "input_global_base = input_split_linear * input_local_dim")
        _line(lines, indent + 1, "local_gather_index = gather_index - input_global_base")
        _line(lines, indent + 1, "input_active = input_active & (gather_index >= input_global_base) & (gather_index < input_global_base + input_local_dim)")
    _line(lines, indent + 1, f"input_offsets = {input_offset()}")
    _line(lines, indent + 1, f"output_offsets = {output_offset()}")
    _line(
        lines,
        indent + 1,
        f"value = {_load_operand(model, 'Input', 'input', 'input_offsets', 'mask & input_active')}",
    )
    _append_store(
        lines, indent + 1, model, "Output", "output", "output_offsets", "value", "mask"
    )
    return _finish(lines)


def _emit_concat(model: dict[str, Any]) -> str:
    rank = len(model["OutputShape"])

    def axis_index(axis: int, block_axis: int) -> str:
        return "lane" if axis == block_axis else f"idx{axis}"

    def input_offset(shape: list[Any], strides: list[Any], block_axis: int) -> str:
        terms = [f"{axis_index(axis, block_axis)} * {_dim(strides[axis])}" for axis in range(len(shape))]
        return "lane * 0" if not terms else "lane * 0 + " + " + ".join(terms)

    def output_offset(block_axis: int, axis_offset: Any) -> str:
        terms = []
        for axis in range(rank):
            index = axis_index(axis, block_axis)
            if axis == model["Axis"] and _fixed(axis_offset) != 0:
                index = f"({index} + {_dim(axis_offset)})"
            terms.append(f"{index} * {_dim(model['OutputStrides'][axis])}")
        return "lane * 0" if not terms else "lane * 0 + " + " + ".join(terms)

    lines = _standard_header(
        model,
        f"# generated from PyNTT Jinja Concat.py.jinja\n# {model['Comment']}; dtype={model['OutputDType']}, axis={model['Axis']}, output_shape={_shape_tuple(model['OutputShape'])}",
        [],
    )
    _append_pointer_shard_coords(lines, 1, list(model["Inputs"]) + [model["Output"]])
    for input_index, input_ptr in enumerate(model["Inputs"]):
        _append_pointer_value_binding(lines, 1, input_ptr, f"input{input_index}")
    _append_pointer_binding(lines, 1, model, "output", "Output")
    axis_offset = _zero()
    for input_index, input_shape in enumerate(model["InputShapes"]):
        input_strides = model["InputStrides"][input_index]
        block_axis = _select_block_axis(input_shape, input_strides)
        block_extent = _one() if not input_shape else input_shape[block_axis]
        loop_axes = [axis for axis in range(len(input_shape)) if axis != block_axis]
        if input_index > 0:
            _line(lines, 0)
        _line(lines, 1, f"# concat input{input_index}")
        indent = 1
        for axis in loop_axes:
            _line(lines, indent, f"for idx{axis} in tl.range(0, {_dim(input_shape[axis])}):")
            indent += 1
        _line(lines, indent, f"for axis_start in tl.range(0, {_dim(block_extent)}, block_size):")
        _line(lines, indent + 1, "lane = axis_start + tl.arange(0, block_size)")
        _line(lines, indent + 1, f"mask = lane < {_dim(block_extent)}")
        _line(lines, indent + 1, f"input_offsets = {input_offset(input_shape, input_strides, block_axis)}")
        _line(lines, indent + 1, f"output_offsets = {output_offset(block_axis, axis_offset)}")
        input_pointer = _access_pointer_value(
            model["Inputs"][input_index], f"input{input_index}", "input_offsets"
        )
        _line(lines, indent + 1, f"value = tl.load({input_pointer}, mask=mask)")
        _append_store(
            lines,
            indent + 1,
            model,
            "Output",
            "output",
            "output_offsets",
            "value",
            "mask",
        )
        axis_offset = _add_dims(axis_offset, input_shape[model["Axis"]])
    return _finish(lines)


def _emit_scatter_nd(model: dict[str, Any]) -> str:
    input_rank = len(model["InputShape"])
    updates_rank = len(model["UpdatesShape"])
    indices_rank = len(model["IndicesShape"])
    prefix_rank = indices_rank - 1
    index_depth = _fixed(model["IndicesShape"][-1])
    if index_depth is None:
        raise RuntimeError("ScatterND index depth must be fixed in PyNTT renderer.")
    slice_rank = input_rank - index_depth
    signed_indices = not str(model["IndicesDType"]).startswith("uint")
    copy_block_axis = _select_block_axis(model["OutputShape"], model["OutputStrides"])
    copy_block_extent = _one() if not model["OutputShape"] else model["OutputShape"][copy_block_axis]
    copy_loop_axes = [axis for axis in range(len(model["OutputShape"])) if axis != copy_block_axis]
    updates_block_axis = _select_block_axis(model["UpdatesShape"], model["UpdatesStrides"])
    updates_block_extent = _one() if not model["UpdatesShape"] else model["UpdatesShape"][updates_block_axis]
    updates_loop_axes = [axis for axis in range(len(model["UpdatesShape"])) if axis != updates_block_axis]

    def axis_index(prefix: str, axis: int, block_axis: int) -> str:
        return f"{prefix}_lane" if axis == block_axis else f"{prefix}_idx{axis}"

    def offset(prefix: str, shape: list[Any], strides: list[Any], block_axis: int) -> str:
        terms = [f"{axis_index(prefix, axis, block_axis)} * {_dim(strides[axis])}" for axis in range(len(shape))]
        return f"{prefix}_lane * 0" if not terms else f"{prefix}_lane * 0 + " + " + ".join(terms)

    def indices_prefix_offset() -> str:
        terms = [
            f"{axis_index('upd', axis, updates_block_axis)} * {_dim(model['IndicesStrides'][axis])}"
            for axis in range(prefix_rank)
        ]
        return "upd_lane * 0" if not terms else "upd_lane * 0 + " + " + ".join(terms)

    def updates_offset() -> str:
        terms = [
            f"{axis_index('upd', axis, updates_block_axis)} * {_dim(model['UpdatesStrides'][axis])}"
            for axis in range(updates_rank)
        ]
        return "upd_lane * 0" if not terms else "upd_lane * 0 + " + " + ".join(terms)

    def scatter_output_offset() -> str:
        terms = [f"scatter_idx{axis} * {_dim(model['OutputStrides'][axis])}" for axis in range(index_depth)]
        for axis in range(slice_rank):
            updates_axis = prefix_rank + axis
            output_axis = index_depth + axis
            terms.append(f"{axis_index('upd', updates_axis, updates_block_axis)} * {_dim(model['OutputStrides'][output_axis])}")
        return "upd_lane * 0" if not terms else "upd_lane * 0 + " + " + ".join(terms)

    lines = _standard_header(
        model,
        f"# generated from PyNTT Jinja ScatterND.py.jinja\n# {model['Comment']}; input_dtype={model['InputDType']}, indices_dtype={model['IndicesDType']}, updates_dtype={model['UpdatesDType']}, output_dtype={model['OutputDType']}",
        [("input", "Input"), ("indices", "Indices"), ("updates", "Updates"), ("output", "Output")],
    )
    _line(lines, 1, "# copy input to output")
    indent = 1
    for axis in copy_loop_axes:
        _line(lines, indent, f"for copy_idx{axis} in tl.range(0, {_dim(model['OutputShape'][axis])}):")
        indent += 1
    _line(lines, indent, f"for copy_axis_start in tl.range(0, {_dim(copy_block_extent)}, block_size):")
    _line(lines, indent + 1, "copy_lane = copy_axis_start + tl.arange(0, block_size)")
    _line(lines, indent + 1, f"copy_mask = copy_lane < {_dim(copy_block_extent)}")
    _line(lines, indent + 1, f"copy_input_offsets = {offset('copy', model['InputShape'], model['InputStrides'], copy_block_axis)}")
    _line(lines, indent + 1, f"copy_output_offsets = {offset('copy', model['OutputShape'], model['OutputStrides'], copy_block_axis)}")
    _line(
        lines,
        indent + 1,
        f"copy_value = {_load_operand(model, 'Input', 'input', 'copy_input_offsets', 'copy_mask')}",
    )
    _append_store(
        lines,
        indent + 1,
        model,
        "Output",
        "output",
        "copy_output_offsets",
        "copy_value",
        "copy_mask",
    )
    _line(lines, 0)
    _line(lines, 1, "# scatter updates")
    indent = 1
    for axis in updates_loop_axes:
        _line(lines, indent, f"for upd_idx{axis} in tl.range(0, {_dim(model['UpdatesShape'][axis])}):")
        indent += 1
    _line(lines, indent, f"for upd_axis_start in tl.range(0, {_dim(updates_block_extent)}, block_size):")
    _line(lines, indent + 1, "upd_lane = upd_axis_start + tl.arange(0, block_size)")
    _line(lines, indent + 1, f"upd_mask = upd_lane < {_dim(updates_block_extent)}")
    _line(lines, indent + 1, f"indices_prefix_offsets = {indices_prefix_offset()}")
    for axis in range(index_depth):
        index_offset = (
            f"indices_prefix_offsets + {axis} * {_dim(model['IndicesStrides'][-1])}"
        )
        _line(
            lines,
            indent + 1,
            f"scatter_idx{axis} = {_load_operand(model, 'Indices', 'indices', index_offset, 'upd_mask', other='0')}",
        )
        if signed_indices:
            _line(lines, indent + 1, f"scatter_idx{axis} = tl.where(scatter_idx{axis} < 0, scatter_idx{axis} + {_dim(model['OutputShape'][axis])}, scatter_idx{axis})")
    _line(lines, indent + 1, f"updates_offsets = {updates_offset()}")
    _line(lines, indent + 1, f"scatter_output_offsets = {scatter_output_offset()}")
    _line(
        lines,
        indent + 1,
        f"updates_value = {_load_operand(model, 'Updates', 'updates', 'updates_offsets', 'upd_mask')}",
    )
    _append_store(
        lines,
        indent + 1,
        model,
        "Output",
        "output",
        "scatter_output_offsets",
        "updates_value",
        "upd_mask",
    )
    return _finish(lines)


def _emit_conv2d(model: dict[str, Any]) -> str:
    stride_h, stride_w = model["Stride"][0], model["Stride"][1]
    pad_top, pad_left = model["Padding"][0], model["Padding"][2]
    dilation_h, dilation_w = model["Dilation"][0], model["Dilation"][1]
    input_channels_per_group = _fixed(model["WeightsShape"][1])
    output_channels = _fixed(model["OutputShape"][1])
    kernel_h = _fixed(model["WeightsShape"][2])
    kernel_w = _fixed(model["WeightsShape"][3])
    if None in (input_channels_per_group, output_channels, kernel_h, kernel_w):
        raise RuntimeError("Conv2D PyNTT renderer requires fixed channel/kernel dimensions.")
    output_channels_per_group = output_channels // model["Groups"]
    block_axis = _select_block_axis(model["OutputShape"], model["OutputStrides"])
    block_extent = model["OutputShape"][block_axis]
    loop_axes = [axis for axis in range(len(model["OutputShape"])) if axis != block_axis]

    def axis_index(axis: int) -> str:
        return "lane" if axis == block_axis else f"idx{axis}"

    n = axis_index(0)
    oc = axis_index(1)
    oh = axis_index(2)
    ow = axis_index(3)
    group = "0" if model["Groups"] == 1 else f"{oc} // {output_channels_per_group}"
    input_channel = "ic" if model["Groups"] == 1 else f"({group}) * {input_channels_per_group} + ic"
    ih = f"{oh} * {stride_h} + kh * {dilation_h} - {pad_top}"
    iw = f"{ow} * {stride_w} + kw * {dilation_w} - {pad_left}"
    input_offset = f"lane * 0 + {n} * {_dim(model['InputStrides'][0])} + ({input_channel}) * {_dim(model['InputStrides'][1])} + ({ih}) * {_dim(model['InputStrides'][2])} + ({iw}) * {_dim(model['InputStrides'][3])}"
    weights_offset = f"lane * 0 + {oc} * {_dim(model['WeightsStrides'][0])} + ic * {_dim(model['WeightsStrides'][1])} + kh * {_dim(model['WeightsStrides'][2])} + kw * {_dim(model['WeightsStrides'][3])}"
    bias_offset = f"lane * 0 + {oc} * {_dim(model['BiasStrides'][0])}"
    output_offset = f"lane * 0 + {n} * {_dim(model['OutputStrides'][0])} + {oc} * {_dim(model['OutputStrides'][1])} + {oh} * {_dim(model['OutputStrides'][2])} + {ow} * {_dim(model['OutputStrides'][3])}"
    lines = _standard_header(
        model,
        f"# generated from PyNTT Jinja Conv2D.py.jinja\n# {model['Comment']}; input_dtype={model['InputDType']}, weights_dtype={model['WeightsDType']}, bias_dtype={model['BiasDType']}, output_dtype={model['OutputDType']}",
        [("input", "Input"), ("weights", "Weights"), ("bias", "Bias"), ("output", "Output")],
    )
    indent = 1
    for axis in loop_axes:
        _line(lines, indent, f"for idx{axis} in tl.range(0, {_dim(model['OutputShape'][axis])}):")
        indent += 1
    _line(lines, indent, f"for axis_start in tl.range(0, {_dim(block_extent)}, block_size):")
    _line(lines, indent + 1, "lane = axis_start + tl.arange(0, block_size)")
    _line(lines, indent + 1, f"mask = lane < {_dim(block_extent)}")
    _line(
        lines,
        indent + 1,
        f"acc = {_load_operand(model, 'Bias', 'bias', bias_offset, 'mask')}.to(tl.float32)",
    )
    _line(lines, indent + 1, f"for ic in tl.range(0, {input_channels_per_group}):")
    _line(lines, indent + 2, f"for kh in tl.range(0, {kernel_h}):")
    _line(lines, indent + 3, f"for kw in tl.range(0, {kernel_w}):")
    _line(lines, indent + 4, f"ih = {ih}")
    _line(lines, indent + 4, f"iw = {iw}")
    _line(lines, indent + 4, f"input_mask = mask & (ih >= 0) & (ih < {_dim(model['InputShape'][2])}) & (iw >= 0) & (iw < {_dim(model['InputShape'][3])})")
    _line(
        lines,
        indent + 4,
        f"input_value = {_load_operand(model, 'Input', 'input', input_offset, 'input_mask')}.to(tl.float32)",
    )
    _line(
        lines,
        indent + 4,
        f"weight_value = {_load_operand(model, 'Weights', 'weights', weights_offset, 'mask')}.to(tl.float32)",
    )
    _line(lines, indent + 4, "acc += input_value * weight_value")
    _append_store(
        lines,
        indent + 1,
        model,
        "Output",
        "output",
        output_offset,
        f"acc.to({model['OutputTritonDType']})",
        "mask",
    )
    return _finish(lines)


def _emit_get_position_ids(model: dict[str, Any]) -> str:
    local_extent = model["LocalShape"][0]
    lines = _helper_header(
        model,
        ("cache_meta",),
        comment=f"# generated from PyNTT Jinja GetPositionIds.py.jinja\n# {model['Comment']}; output_dtype={model['OutputDType']}, local_shape={_shape_tuple(model['LocalShape'])}, global_shape={_shape_tuple(model['GlobalShape'])}",
    )
    _line(lines, 1, "shard_index = tl.program_id(0).to(tl.int64)")
    _append_pointer_shard_coords(lines, 1, [model["Output"]])
    _append_pointer_binding(lines, 1, model, "output", "Output")
    _line(lines, 1, f"global_start = {_dim(model['OutputGlobalOffsets'][0])}")
    indent = 2
    result_shape = "block_size"
    _line(lines, 1, f"for axis_start in tl.range(0, {_dim(local_extent)}, block_size):")
    _line(lines, indent, "lane = axis_start + tl.arange(0, block_size)")
    _line(lines, indent, f"mask = lane < {_dim(local_extent)}")
    _line(lines, indent, "global_lane = global_start + lane")
    _line(lines, indent, f"result = tl.full(({result_shape},), 0.0, tl.float32)")
    _line(lines, indent, f"active = tl.full(({result_shape},), False, tl.int1)")
    _line(lines, indent, "num_seqs = tl.load(cache_meta + 0).to(tl.int64)")
    _line(lines, indent, "query_start = tl.full((), 0, tl.int64)")
    _line(lines, indent, "for seq_id in tl.range(0, num_seqs):")
    _line(lines, indent + 1, "context_len = tl.load(cache_meta + 1 + seq_id * 2)")
    _line(lines, indent + 1, "seq_len = tl.load(cache_meta + 2 + seq_id * 2)")
    _line(lines, indent + 1, "query_len = seq_len - context_len")
    _line(lines, indent + 1, "query_end = query_start + query_len")
    _line(lines, indent + 1, "in_seq = (global_lane >= query_start) & (global_lane < query_end)")
    _line(lines, indent + 1, "result = tl.where(in_seq, (context_len + global_lane - query_start).to(tl.float32), result)")
    _line(lines, indent + 1, "active = active | in_seq")
    _line(lines, indent + 1, "query_start = query_end")
    _append_store(
        lines,
        indent,
        model,
        "Output",
        "output",
        f"lane * {_dim(model['OutputStrides'][0])}",
        "result",
        "mask & active",
    )
    return _finish(lines)


def _emit_reshard(model: dict[str, Any]) -> str:
    return _emit_reshard_direct(model)


def _emit_reshard_direct(model: dict[str, Any]) -> str:
    lines = _standard_header(
        model,
        f"# generated from PyNTT Jinja Reshard.py.jinja\n# {model['Comment']}; dtype={model['DType']}, global_shape={_shape_tuple(model['GlobalShape'])}, input_tile_shape={_shape_tuple(model['InputActiveShape'])}, output_local_shape={_shape_tuple(model['OutputLocalShape'])}, lane={model['VectorLaneCount']}, stage=tile_scatter",
        [("input0", "Input")],
    )
    if model.get("Stage") != "tile_scatter":
        raise ValueError(f"Unsupported PyNTT direct reshard stage: {model.get('Stage')}")

    _line(lines, 1, "tmp_shard = shard_index")
    for axis in range(len(model["Hierarchy"]) - 1, -1, -1):
        _line(lines, 1, f"shard_coord{axis} = tmp_shard % {model['Hierarchy'][axis]}")
        _line(lines, 1, f"tmp_shard = tmp_shard // {model['Hierarchy'][axis]}")

    input_split_mesh_axes = {
        axis
        for split_axes in model["InputSplitAxes"]
        for axis in split_axes
    }
    input_partial_mesh_axes = set(model["InputPartialAxes"])
    if input_split_mesh_axes & input_partial_mesh_axes:
        raise ValueError(
            "A PyNTT reshard mesh axis cannot be both split and partial: "
            f"{sorted(input_split_mesh_axes & input_partial_mesh_axes)}"
        )
    output_split_mesh_axes = {
        axis
        for split_axes in model["OutputSplitAxes"]
        for axis in split_axes
    }
    output_broadcast_mesh_axes = [
        axis
        for axis in range(len(model["Hierarchy"]))
        if axis not in output_split_mesh_axes
    ]

    indent = 1
    for axis in output_broadcast_mesh_axes:
        _line(lines, indent, f"for destination_shard_coord{axis} in tl.range(0, {model['Hierarchy'][axis]}):")
        indent += 1

    writer_active = "True"
    for axis in sorted(input_partial_mesh_axes):
        writer_active = f"({writer_active}) & (shard_coord{axis} == 0)"

    total = " * ".join(
        [f"({_dim(dim)})" for dim in model["InputActiveShape"]]
        + [f"({model['VectorLaneCount']})"]
    )
    _line(lines, indent, f"for linear_start in tl.range(0, {total}, block_size):")
    _line(lines, indent + 1, "linear = linear_start + tl.arange(0, block_size)")
    _line(lines, indent + 1, f"mask = (linear < {total}) & ({writer_active})")
    if model["VectorLaneCount"] == 1:
        _line(lines, indent + 1, "lane = linear")
        _line(lines, indent + 1, "lane_value = linear * 0")
    else:
        _line(lines, indent + 1, f"lane = linear // {model['VectorLaneCount']}")
        _line(lines, indent + 1, f"lane_value = linear % {model['VectorLaneCount']}")
    _line(lines, indent + 1, "remaining = lane")
    for axis in range(len(model["InputActiveShape"]) - 1, -1, -1):
        _line(lines, indent + 1, f"input_idx{axis} = remaining % {_dim(model['InputActiveShape'][axis])}")
        _line(lines, indent + 1, f"remaining = remaining // {_dim(model['InputActiveShape'][axis])}")

    for axis in range(len(model["GlobalShape"])):
        _line(lines, indent + 1, f"global_idx{axis} = input_idx{axis} + {_dim(model['InputGlobalOffsets'][axis])}")
        _line(lines, indent + 1, f"mask = mask & (global_idx{axis} < {_dim(model['GlobalShape'][axis])})")

    for axis in range(len(model["GlobalShape"])):
        split_axes = model["OutputSplitAxes"][axis]
        if split_axes:
            _line(lines, indent + 1, f"output_local_dim{axis} = {_dim(model['OutputLocalShape'][axis])}")
            _line(lines, indent + 1, f"output_split_linear{axis} = global_idx{axis} // output_local_dim{axis}")
            _line(lines, indent + 1, f"output_idx{axis} = global_idx{axis} - output_split_linear{axis} * output_local_dim{axis}")
            _line(lines, indent + 1, f"tmp_output_split{axis} = output_split_linear{axis}")
            for index in range(len(split_axes) - 1, -1, -1):
                placement_axis = split_axes[index]
                _line(lines, indent + 1, f"destination_shard_coord{placement_axis} = tmp_output_split{axis} % {model['Hierarchy'][placement_axis]}")
                _line(lines, indent + 1, f"tmp_output_split{axis} = tmp_output_split{axis} // {model['Hierarchy'][placement_axis]}")
        else:
            _line(lines, indent + 1, f"output_idx{axis} = global_idx{axis}")

    for axis in range(len(model["Hierarchy"])):
        if axis not in input_split_mesh_axes and axis not in input_partial_mesh_axes:
            _line(lines, indent + 1, f"mask = mask & (shard_coord{axis} == destination_shard_coord{axis})")

    destination_shard_index = _split_linear_expression(
        list(range(len(model["Hierarchy"]))),
        model["Hierarchy"],
        "destination_shard_coord",
    )
    input_offsets = _scalar_offset(_tensor_offset("input_idx", model["InputStrides"]), model["VectorLaneCount"])
    output_offsets = _scalar_offset(_tensor_offset("output_idx", model["OutputStrides"]), model["VectorLaneCount"])
    _line(lines, indent + 1, f"destination_shard_index = {destination_shard_index}")
    _line(lines, indent + 1, f"input_offsets = {input_offsets}")
    _line(lines, indent + 1, f"output_offsets = {output_offsets}")
    destination_pool_index = _pool_index_expression(
        "destination_shard_index", model["OutputAddress"]["PoolScopeSize"]
    )
    _line(lines, indent + 1, f"destination_pool_index = {destination_pool_index}")
    _line(lines, indent + 1, f"output_byte_offsets = destination_pool_index * {model['OutputAddress']['PoolStrideBytes']} + {model['OutputAddress']['OffsetBytes']} + output_offsets * {model['ScalarElementSizeBytes']}")
    if not input_partial_mesh_axes:
        input_pointer = _access_pointer(model, "Input", "input0", "input_offsets")
        _line(lines, indent + 1, f"value = tl.load({input_pointer}, mask=mask)")
    else:
        partial_input_address = model.get("PartialInputAddress")
        if partial_input_address is None:
            raise ValueError("PyNTT partial reshard requires PartialInputAddress")
        dtype = model["DType"]
        if dtype in ("float16", "bfloat16", "float32"):
            accumulator_dtype = "tl.float32"
            zero = "0.0"
        elif dtype == "float64":
            accumulator_dtype = "tl.float64"
            zero = "0.0"
        elif dtype.startswith("uint"):
            accumulator_dtype = "tl.uint64"
            zero = "0"
        elif dtype.startswith("int"):
            accumulator_dtype = "tl.int64"
            zero = "0"
        else:
            raise ValueError(f"PyNTT partial Sum does not support dtype {dtype}")

        _line(lines, indent + 1, f"acc = tl.full((block_size,), {zero}, {accumulator_dtype})")
        for axis in range(len(model["Hierarchy"])):
            _line(lines, indent + 1, f"source_shard_coord{axis} = shard_coord{axis}")
        reduce_indent = indent + 1
        for axis in sorted(input_partial_mesh_axes):
            _line(lines, reduce_indent, f"for reduce_coord{axis} in tl.range(0, {model['Hierarchy'][axis]}):")
            reduce_indent += 1
            _line(lines, reduce_indent, f"source_shard_coord{axis} = reduce_coord{axis}.to(tl.int64)")
        source_shard_index = _split_linear_expression(
            list(range(len(model["Hierarchy"]))),
            model["Hierarchy"],
            "source_shard_coord",
        )
        source_pool_index = _pool_index_expression(
            "source_shard_index", partial_input_address["PoolScopeSize"]
        )
        _line(lines, reduce_indent, f"source_shard_index = {source_shard_index}")
        _line(lines, reduce_indent, f"source_pool_index = {source_pool_index}")
        _line(lines, reduce_indent, f"source_byte_offset = source_pool_index * {partial_input_address['PoolStrideBytes']} + {partial_input_address['OffsetBytes']}")
        input_pointer_type = _pointer_type(model["TritonDType"], partial_input_address["AddressSpace"])
        _line(lines, reduce_indent, f"source = ({partial_input_address['BaseName']} + source_byte_offset).to({input_pointer_type})")
        _line(lines, reduce_indent, f"source_value = tl.load(source + input_offsets, mask=mask, other={zero}).to({accumulator_dtype})")
        _line(lines, reduce_indent, "acc += source_value")
        _line(lines, indent + 1, "value = acc")
    output_pointer_type = _pointer_type(model["TritonDType"], model["OutputAddress"]["AddressSpace"])
    _line(lines, indent + 1, f"tl.store(({model['OutputAddress']['BaseName']} + output_byte_offsets).to({output_pointer_type}), value, mask=mask)")
    return _finish(lines)


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


def _emit_summa(model: dict[str, Any]) -> str:
    block_m = 16
    block_n = 16
    block_k = 32
    output_global_physical_n = model["OutputGlobalShape"][1]
    output_global_logical_n = _multiply_dim(output_global_physical_n, model["OutputNVectorLaneCount"])
    rhs_global_logical_n = _multiply_dim(model["RhsGlobalShape"][1], model["RhsNVectorLaneCount"])

    def append_output_axis_range(lines: list[str], level: int, prefix: str, global_extent: Any, split_axes: list[int]) -> None:
        if not split_axes:
            _line(lines, level, f"{prefix}_local_dim = {_dim(global_extent)}")
            _line(lines, level, f"{prefix}_global_base = 0")
            _line(lines, level, f"{prefix}_iter_dim = {_dim(global_extent)}")
            return
        divisor = _split_divisor(split_axes, model["Hierarchy"])
        _line(lines, level, f"{prefix}_local_dim = tl.cdiv({_dim(global_extent)}, {divisor})")
        _line(lines, level, f"{prefix}_split_linear = {_split_linear_expression(split_axes, model['Hierarchy'])}")
        _line(lines, level, f"{prefix}_global_base = {prefix}_split_linear * {prefix}_local_dim")
        _line(lines, level, f"{prefix}_remaining = {_dim(global_extent)} - {prefix}_global_base")
        _line(lines, level, f"{prefix}_iter_dim = tl.minimum({prefix}_local_dim, tl.maximum({prefix}_remaining, 0))")

    def append_source_shard_coords(lines: list[str], level: int) -> None:
        for axis in range(len(model["Hierarchy"])):
            _line(lines, level, f"source_shard_coord{axis} = shard_coord{axis}")

    def local_index_expression(lines: list[str], level: int, prefix: str, global_index: str, global_extent: Any, split_axes: list[int]) -> str:
        if not split_axes:
            return global_index
        divisor = _split_divisor(split_axes, model["Hierarchy"])
        _line(lines, level, f"{prefix}_local_dim = tl.cdiv({_dim(global_extent)}, {divisor})")
        _line(lines, level, f"{prefix}_split_linear = ({global_index}) // {prefix}_local_dim")
        _line(lines, level, f"{prefix}_local = ({global_index}) - {prefix}_split_linear * {prefix}_local_dim")
        _line(lines, level, f"{prefix}_tmp_split = {prefix}_split_linear")
        for index in range(len(split_axes) - 1, -1, -1):
            placement_axis = split_axes[index]
            _line(lines, level, f"source_shard_coord{placement_axis} = {prefix}_tmp_split % {model['Hierarchy'][placement_axis]}")
            _line(lines, level, f"{prefix}_tmp_split = {prefix}_tmp_split // {model['Hierarchy'][placement_axis]}")
        return f"{prefix}_local"

    def maybe_vectorized_offset(physical_offset: str, lane_index: str, vector_lane_count: int) -> str:
        return physical_offset if vector_lane_count == 1 else f"(({physical_offset}) * {vector_lane_count} + {lane_index})"

    full_source_shard_index = _split_linear_expression(list(range(len(model["Hierarchy"]))), model["Hierarchy"], "source_shard_coord")
    dot_precision = ', input_precision="ieee"' if model["LhsDType"] == "float32" and model["RhsDType"] == "float32" else ""
    lines = _helper_header(
        model,
        comment=(
            f"# generated from PyNTT Jinja Summa.py.jinja\n# {model['Comment']}; lhs_dtype={model['LhsDType']}, "
            f"rhs_dtype={model['RhsDType']}, output_dtype={model['OutputDType']}, lhs_shape={_shape_tuple(model['LhsShape'])}, "
            f"rhs_shape={_shape_tuple(model['RhsShape'])}, output_shape={_shape_tuple(model['OutputShape'])}, "
            f"lhs_global_shape={_shape_tuple(model['LhsGlobalShape'])}, rhs_global_shape={_shape_tuple(model['RhsGlobalShape'])}, "
            f"output_global_shape={_shape_tuple(model['OutputGlobalShape'])}, rhs_n_lane={model['RhsNVectorLaneCount']}, "
            f"output_n_lane={model['OutputNVectorLaneCount']}"
        ),
    )
    _line(lines, 1, "shard_index = tl.program_id(0).to(tl.int64)")
    _append_shard_coords(lines, 1, model["Hierarchy"])
    append_output_axis_range(lines, 1, "out_m", model["OutputGlobalShape"][0], model["OutputSplitAxes"][0])
    append_output_axis_range(lines, 1, "out_n", output_global_physical_n, model["OutputSplitAxes"][1])
    _line(lines, 1, f"out_n_logical_iter_dim = out_n_iter_dim * {model['OutputNVectorLaneCount']}")
    _line(lines, 1, f"out_n_logical_global_base = out_n_global_base * {model['OutputNVectorLaneCount']}")
    _line(lines, 1, f"for m_start in tl.range(0, out_m_iter_dim, {block_m}):")
    _line(lines, 2, f"offs_m = m_start + tl.arange(0, {block_m})")
    _line(lines, 2, "global_m = out_m_global_base + offs_m")
    _line(lines, 2, f"for n_start in tl.range(0, out_n_logical_iter_dim, {block_n}):")
    _line(lines, 3, f"offs_n = n_start + tl.arange(0, {block_n})")
    _line(lines, 3, "global_n = out_n_logical_global_base + offs_n")
    _line(lines, 3, f"acc = tl.zeros(({block_m}, {block_n}), tl.float32)")
    _line(lines, 3, f"for k_start in tl.range(0, {_dim(model['LhsGlobalShape'][1])}, {block_k}):")
    _line(lines, 4, f"global_k = k_start + tl.arange(0, {block_k})")
    append_source_shard_coords(lines, 4)
    lhs_m = local_index_expression(lines, 4, "lhs_m", "global_m[:, None]", model["LhsGlobalShape"][0], model["LhsSplitAxes"][0])
    lhs_k = local_index_expression(lines, 4, "lhs_k", "global_k[None, :]", model["LhsGlobalShape"][1], model["LhsSplitAxes"][1])
    _line(lines, 4, f"lhs_shard_index = {full_source_shard_index}")
    _line(lines, 4, f"lhs_offsets = {lhs_m} * {_dim(model['LhsStrides'][0])} + {lhs_k} * {_dim(model['LhsStrides'][1])}")
    lhs_pointer_type = _pointer_type(model["LhsTritonDType"], model["LhsAddressSpace"])
    _line(lines, 4, f"lhs_ptrs = ({model['LhsBaseName']} + lhs_shard_index * {model['LhsPoolBytes']} + {model['LhsOffsetBytes']}).to({lhs_pointer_type})")
    _line(lines, 4, f"lhs_mask = (offs_m[:, None] < out_m_iter_dim) & (global_m[:, None] < {_dim(model['OutputGlobalShape'][0])}) & (global_k[None, :] < {_dim(model['LhsGlobalShape'][1])})")
    _line(lines, 4, "lhs_values = tl.load(lhs_ptrs + lhs_offsets, mask=lhs_mask, other=0.0)")
    append_source_shard_coords(lines, 4)
    _line(lines, 4, f"rhs_global_n_physical = global_n // {model['RhsNVectorLaneCount']}")
    _line(lines, 4, "rhs_lane = global_n * 0")
    if model["RhsNVectorLaneCount"] != 1:
        _line(lines, 4, f"rhs_lane = global_n % {model['RhsNVectorLaneCount']}")
    rhs_k = local_index_expression(lines, 4, "rhs_k", "global_k[:, None]", model["RhsGlobalShape"][0], model["RhsSplitAxes"][0])
    rhs_n = local_index_expression(lines, 4, "rhs_n", "rhs_global_n_physical[None, :]", model["RhsGlobalShape"][1], model["RhsSplitAxes"][1])
    _line(lines, 4, f"rhs_shard_index = {full_source_shard_index}")
    _line(lines, 4, f"rhs_physical_offsets = {rhs_k} * {_dim(model['RhsStrides'][0])} + {rhs_n} * {_dim(model['RhsStrides'][1])}")
    _line(lines, 4, f"rhs_offsets = {maybe_vectorized_offset('rhs_physical_offsets', 'rhs_lane[None, :]', model['RhsNVectorLaneCount'])}")
    rhs_pointer_type = _pointer_type(model["RhsTritonDType"], model["RhsAddressSpace"])
    _line(lines, 4, f"rhs_ptrs = ({model['RhsBaseName']} + rhs_shard_index * {model['RhsPoolBytes']} + {model['RhsOffsetBytes']}).to({rhs_pointer_type})")
    _line(lines, 4, f"rhs_mask = (global_k[:, None] < {_dim(model['LhsGlobalShape'][1])}) & (offs_n[None, :] < out_n_logical_iter_dim) & (global_n[None, :] < {_dim(rhs_global_logical_n)})")
    _line(lines, 4, "rhs_values = tl.load(rhs_ptrs + rhs_offsets, mask=rhs_mask, other=0.0)")
    _line(lines, 4, f"acc += tl.dot(lhs_values, rhs_values{dot_precision})")
    _line(lines, 3, f"out_n_physical = offs_n // {model['OutputNVectorLaneCount']}")
    _line(lines, 3, "out_lane = offs_n * 0")
    if model["OutputNVectorLaneCount"] != 1:
        _line(lines, 3, f"out_lane = offs_n % {model['OutputNVectorLaneCount']}")
    _line(lines, 3, f"output_physical_offsets = offs_m[:, None] * {_dim(model['OutputStrides'][0])} + out_n_physical[None, :] * {_dim(model['OutputStrides'][1])}")
    _line(lines, 3, f"output_offsets = {maybe_vectorized_offset('output_physical_offsets', 'out_lane[None, :]', model['OutputNVectorLaneCount'])}")
    output_pointer_type = _pointer_type(model["OutputTritonDType"], model["OutputAddressSpace"])
    _line(lines, 3, f"output_ptr = ({model['OutputBaseName']} + shard_index * {model['OutputPoolBytes']} + {model['OutputOffsetBytes']}).to({output_pointer_type})")
    _line(lines, 3, f"result = acc * {model['Scale']}")
    _line(lines, 3, f"output_mask = (offs_m[:, None] < out_m_iter_dim) & (offs_n[None, :] < out_n_logical_iter_dim) & (global_n[None, :] < {_dim(output_global_logical_n)})")
    _line(lines, 3, "tl.store(output_ptr + output_offsets, result, mask=output_mask)")
    return _finish(lines)


def _emit_paged_attention(model: dict[str, Any]) -> str:
    cache = model["Cache"]
    attention_block_size = int(model["AttentionBlockSize"])
    if (
        attention_block_size <= 0
        or attention_block_size & (attention_block_size - 1)
        or attention_block_size > int(cache["BlockSize"])
    ):
        raise ValueError(
            "PyNTT PagedAttention AttentionBlockSize must be a positive power "
            f"of two no larger than the cache block size, got {attention_block_size}."
        )

    def tensor_offset(strides: list[Any], seq_axis: int, head_axis: int, dim_axis: int, head: str, dim_block: str, token: str, lane: str, lane_count: int) -> str:
        indices = ["0"] * len(strides)
        indices[seq_axis] = token
        indices[head_axis] = head
        indices[dim_axis] = dim_block
        terms = [f"{indices[axis]} * {_dim(strides[axis])}" for axis in range(len(strides))]
        return f"(({ ' + '.join(terms) }) * {lane_count} + {lane})"

    def global_index_expression(axis: int, local_index: str, global_extent: Any) -> str:
        split_axes = model["OutputSplitAxes"][axis]
        if not split_axes:
            return local_index
        divisor = _split_divisor(split_axes, model["Hierarchy"])
        return f"{local_index} + ({_split_linear_expression(split_axes, model['Hierarchy'])}) * tl.cdiv({_dim(global_extent)}, {divisor})"

    def cache_dim_index(prefix: str, dim_name: str, block_name: str) -> str:
        if cache[f"{prefix}VectorizedDim"] == 5:  # HeadDim
            return f"{dim_name} // {cache[f'{prefix}LaneCount']}"
        return dim_name

    def cache_block_index(prefix: str, block_name: str) -> str:
        if cache[f"{prefix}VectorizedDim"] == 3:  # BlockSize
            return f"{block_name} // {cache[f'{prefix}LaneCount']}"
        return block_name

    def cache_lane(prefix: str, dim_name: str, block_name: str) -> str:
        if cache[f"{prefix}VectorizedDim"] == 5:  # HeadDim
            return f"{dim_name} % {cache[f'{prefix}LaneCount']}"
        if cache[f"{prefix}VectorizedDim"] == 3:  # BlockSize
            return f"{block_name} % {cache[f'{prefix}LaneCount']}"
        return "0"

    if cache["KeyVectorizedDim"] != 5:
        raise ValueError("PyNTT PagedAttention currently requires key cache to be HeadDim-vectorized.")

    query_vector_offset = tensor_offset(model["QueryStrides"], model["SeqAxis"], model["HeadAxis"], model["DimAxis"], "q_head", "query_dim_blocks", "local_query_id", "query_dim_lanes", cache["KeyLaneCount"])
    output_vector_offset = tensor_offset(model["OutputStrides"], model["SeqAxis"], model["HeadAxis"], model["DimAxis"], "q_head", "query_dim_blocks", "local_query_id", "query_dim_lanes", cache["KeyLaneCount"])
    local_q_heads = model["OutputShape"][model["HeadAxis"]]
    local_query_tokens = model["OutputShape"][model["SeqAxis"]]
    global_query_tokens = model["OutputGlobalShape"][model["SeqAxis"]]
    global_query_id = global_index_expression(model["SeqAxis"], "local_query_id", global_query_tokens)
    global_q_head = global_index_expression(model["HeadAxis"], "q_head", model["GlobalNumQueryHeads"])
    block_index_key_matrix = "(topology_id[None, :] * num_blocks_per_shard + block_id[None, :])" if cache["IdLength"] > 1 else "block_id[None, :]"
    block_index_value_matrix = "(topology_id[:, None] * num_blocks_per_shard + block_id[:, None])" if cache["IdLength"] > 1 else "block_id[:, None]"
    key_lane_broadcast = "key_lane[:, None]" if cache["KeyVectorizedDim"] == 5 else "key_lane[None, :]" if cache["KeyVectorizedDim"] == 3 else "0"
    value_lane_broadcast = "value_lane[None, :]" if cache["ValueVectorizedDim"] == 5 else "value_lane[:, None]" if cache["ValueVectorizedDim"] == 3 else "0"
    layer_id = "layer_id_value"
    key_vector_offset = f"({block_index_key_matrix} * {cache['BlockElements']} + {cache['KeySectionOffset']} + (({layer_id}) * {cache['KeyLayerStride']} + kv_head * {cache['KeyHeadStride']} + key_dim_index[:, None] * {cache['KeyDimBlockStride']} + key_block_index[None, :] * {cache['KeyBlockOffsetStride']}) * {cache['KeyLaneCount']} + {key_lane_broadcast})"
    value_vector_offset = f"({block_index_value_matrix} * {cache['BlockElements']} + {cache['ValueSectionOffset']} + (({layer_id}) * {cache['ValueLayerStride']} + kv_head * {cache['ValueHeadStride']} + value_dim_index[None, :] * {cache['ValueDimBlockStride']} + value_block_index[:, None] * {cache['ValueBlockOffsetStride']}) * {cache['ValueLaneCount']} + {value_lane_broadcast})"

    lines = _helper_header(
        model,
        ("block_tables", "kv_cache", "num_blocks_per_shard", "cache_meta"),
        comment=f"# generated from PyNTT Jinja PagedAttention.py.jinja\n# {model['Comment']}; query_dtype={model['QueryDType']}, output_dtype={model['OutputDType']}, query_shape={_shape_tuple(model['QueryShape'])}, output_shape={_shape_tuple(model['OutputShape'])}, layer={model['LayerIdExpression']}, block_n={attention_block_size}",
    )
    _line(lines, 1, "shard_index = tl.program_id(0).to(tl.int64)")
    _line(lines, 1, f"layer_id_value = (tl.full((), 0, tl.int64) + ({model['LayerIdExpression']})).to(tl.int64)")
    _append_pointer_shard_coords(lines, 1, [model["Query"], model["Scale"], model["Output"]])
    _append_pointer_binding(lines, 1, model, "query", "Query")
    _append_pointer_binding(lines, 1, model, "scale_ptr", "Scale")
    _append_pointer_binding(lines, 1, model, "output", "Output")
    scale_pointer = _access_pointer(model, "Scale", "scale_ptr")
    _line(lines, 1, f"scale_value = tl.load({scale_pointer}).to(tl.float32)")
    _line(lines, 1, "num_seqs = tl.load(cache_meta + 0).to(tl.int64)")
    _append_shard_coords(lines, 1, model["Hierarchy"])
    _line(lines, 1, "max_seq_len = tl.full((), 0, tl.int64)")
    _line(lines, 1, "num_tokens = tl.full((), 0, tl.int64)")
    _line(lines, 1, "for seq_scan in tl.range(0, num_seqs):")
    _line(lines, 2, "context_len_scan = tl.load(cache_meta + 1 + seq_scan * 2)")
    _line(lines, 2, "seq_len_scan = tl.load(cache_meta + 2 + seq_scan * 2)")
    _line(lines, 2, "max_seq_len = tl.maximum(max_seq_len, seq_len_scan)")
    _line(lines, 2, "num_tokens += seq_len_scan - context_len_scan")
    _line(lines, 1, f"block_table_blocks = tl.cdiv(max_seq_len, {cache['BlockSize']})")
    _line(lines, 1, f"dim_offsets = tl.arange(0, {cache['HeadDim']})")
    _line(lines, 1, f"query_dim_blocks = dim_offsets // {cache['KeyLaneCount']}")
    _line(lines, 1, f"query_dim_lanes = dim_offsets % {cache['KeyLaneCount']}")
    _line(lines, 1, f"key_dim_index = {cache_dim_index('Key', 'dim_offsets', 'block_offsets')}")
    _line(lines, 1, f"key_lane = {cache_lane('Key', 'dim_offsets', 'block_offsets')}")
    _line(lines, 1, f"value_dim_index = {cache_dim_index('Value', 'dim_offsets', 'block_offsets')}")
    if cache["ValueVectorizedDim"] != 3:
        _line(lines, 1, f"value_lane = {cache_lane('Value', 'dim_offsets', 'block_offsets')}")
    _line(lines, 1, f"for local_query_id in tl.range(0, {_dim(local_query_tokens)}):")
    _line(lines, 2, f"global_query_id = {global_query_id}")
    _line(lines, 2, f"for q_head in tl.range(0, {_dim(local_q_heads)}):")
    _line(lines, 3, f"global_q_head = {global_q_head}")
    _line(lines, 3, "query_start = tl.full((), 0, tl.int64)")
    _line(lines, 3, "context_len = tl.full((), 0, tl.int64)")
    _line(lines, 3, "seq_len = tl.full((), 0, tl.int64)")
    _line(lines, 3, "query_id = tl.full((), 0, tl.int64)")
    _line(lines, 3, "matched_seq_id = tl.full((), 0, tl.int64)")
    _line(lines, 3, f"query_active = (local_query_id < {_dim(local_query_tokens)}) & (global_query_id < {_dim(global_query_tokens)}) & (global_query_id < num_tokens) & (global_q_head < {model['GlobalNumQueryHeads']})")
    _line(lines, 3, "found_seq = False")
    _line(lines, 3, "for seq_id in tl.range(0, num_seqs):")
    _line(lines, 4, "current_context_len = tl.load(cache_meta + 1 + seq_id * 2)")
    _line(lines, 4, "current_seq_len = tl.load(cache_meta + 2 + seq_id * 2)")
    _line(lines, 4, "current_query_len = current_seq_len - current_context_len")
    _line(lines, 4, "in_seq = query_active & (global_query_id >= query_start) & (global_query_id < query_start + current_query_len)")
    _line(lines, 4, "matched_seq_id = tl.where(in_seq, seq_id, matched_seq_id)")
    _line(lines, 4, "context_len = tl.where(in_seq, current_context_len, context_len)")
    _line(lines, 4, "seq_len = tl.where(in_seq, current_seq_len, seq_len)")
    _line(lines, 4, "query_id = tl.where(in_seq, global_query_id - query_start, query_id)")
    _line(lines, 4, "found_seq = found_seq | in_seq")
    _line(lines, 4, "query_start += current_query_len")
    _line(lines, 3, f"head_group = {model['GlobalNumQueryHeads']} // {cache['NumKVHeads']}")
    _line(lines, 3, "kv_head = global_q_head // head_group")
    _line(lines, 3, f"q_offsets = {query_vector_offset}")
    _line(
        lines,
        3,
        f"q_values = {_load_operand(model, 'Query', 'query', 'q_offsets', 'found_seq')}",
    )
    _line(lines, 3, 'max_score = -float("inf")')
    _line(lines, 3, "sum_exp = 0.0")
    _line(lines, 3, f"acc = tl.zeros((1, {cache['HeadDim']}), tl.float32)")
    _line(lines, 3, f"for context_start in tl.range(0, max_seq_len, {attention_block_size}):")
    _line(lines, 4, f"context_ids = context_start + tl.arange(0, {attention_block_size})")
    _line(lines, 4, "context_mask = found_seq & (context_ids < seq_len) & (context_ids <= context_len + query_id)")
    _line(lines, 4, f"context_bid = context_ids // {cache['BlockSize']}")
    _line(lines, 4, f"block_offsets = context_ids % {cache['BlockSize']}")
    _line(lines, 4, f"key_block_index = {cache_block_index('Key', 'block_offsets')}")
    _line(lines, 4, f"value_block_index = {cache_block_index('Value', 'block_offsets')}")
    if cache["ValueVectorizedDim"] == 3:
        _line(lines, 4, f"value_lane = {cache_lane('Value', 'dim_offsets', 'block_offsets')}")
    if cache["IdLength"] > 1:
        _line(lines, 4, f"topology_id = tl.load(block_tables + matched_seq_id * block_table_blocks * {cache['IdLength']} + context_bid * {cache['IdLength']}, mask=context_mask, other=0)")
    _line(lines, 4, f"block_id = tl.load(block_tables + matched_seq_id * block_table_blocks * {cache['IdLength']} + context_bid * {cache['IdLength']} + {cache['IdLength'] - 1}, mask=context_mask, other=0)")
    _line(lines, 4, f"key_offsets = {key_vector_offset}")
    _line(lines, 4, "key_values = tl.load(kv_cache + key_offsets, mask=context_mask[None, :], other=0.0)")
    _line(lines, 4, f"score = tl.reshape(tl.dot(q_values[None, :], key_values), ({attention_block_size},))")
    _line(lines, 4, 'score = tl.where(context_mask, score * scale_value, -float("inf"))')
    _line(lines, 4, "block_max = tl.max(score, axis=0)")
    _line(lines, 4, 'has_context = block_max != -float("inf")')
    _line(lines, 4, "new_max = tl.maximum(max_score, block_max)")
    _line(lines, 4, "safe_new_max = tl.where(has_context, new_max, 0.0)")
    _line(lines, 4, "safe_prev_max = tl.where(has_context, max_score, 0.0)")
    _line(lines, 4, "alpha = tl.exp(safe_prev_max - safe_new_max)")
    _line(lines, 4, 'prob = tl.exp(tl.where(context_mask, score - safe_new_max, -float("inf")))')
    _line(lines, 4, f"value_offsets = {value_vector_offset}")
    _line(lines, 4, "value_values = tl.load(kv_cache + value_offsets, mask=context_mask[:, None], other=0.0).to(tl.float32)")
    _line(lines, 4, 'acc = acc * alpha + tl.dot(prob[None, :], value_values, input_precision="ieee")')
    _line(lines, 4, "sum_exp = sum_exp * alpha + tl.sum(prob, axis=0)")
    _line(lines, 4, "max_score = tl.where(has_context, new_max, max_score)")
    _line(lines, 3, "inv_sum = tl.where(sum_exp != 0.0, 1.0 / sum_exp, 0.0)")
    _line(lines, 3, f"result = tl.reshape(acc * inv_sum, ({cache['HeadDim']},))")
    _line(lines, 3, f"output_offsets = {output_vector_offset}")
    _append_store(
        lines, 3, model, "Output", "output", "output_offsets", "result", "found_seq"
    )
    return _finish(lines)


def _emit_update_paged_attention_kv_cache(model: dict[str, Any]) -> str:
    cache = model["Cache"]
    kind_prefix = "Key" if model["CacheKind"] == 0 else "Value"
    lane_count = cache[f"{kind_prefix}LaneCount"]
    section_offset = cache[f"{kind_prefix}SectionOffset"]
    layer_stride = cache[f"{kind_prefix}LayerStride"]
    head_stride = cache[f"{kind_prefix}HeadStride"]
    dim_block_stride = cache[f"{kind_prefix}DimBlockStride"]
    block_offset_stride = cache[f"{kind_prefix}BlockOffsetStride"]
    vectorized_dim = cache[f"{kind_prefix}VectorizedDim"]
    slots_lane_count = model["SlotsVectorLaneCount"]
    source_split_axes = sorted({axis for split_axes in model["SlotsSourceSplitAxes"] for axis in split_axes})
    topology_match_axes = [axis for axis in cache["NumBlocksSplitAxes"] if axis not in source_split_axes]
    block_index = "(topology_id * num_blocks_per_shard + block_id)" if cache["IdLength"] > 1 else "block_id"
    cache_offset = f"({block_index} * {cache['BlockElements']} + {section_offset} + (layer_id_value * {layer_stride} + cache_head_id * {head_stride} + cache_dim_block * {dim_block_stride} + cache_block_offset * {block_offset_stride}) * {lane_count} + cache_lane_id)"

    def product(values: list[str]) -> str:
        return "1" if not values else " * ".join(f"({value})" for value in values)

    def slot_offset(lane_expr: str | None = "source_lane_id") -> str:
        terms = [f"source_idx{axis} * {_dim(model['SlotsStrides'][axis])}" for axis in range(len(model["SlotsStrides"]))]
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
    total_elements = product([
        _dim(model["SlotsShape"][model["SeqAxis"]]),
        _dim(model["SlotsShape"][model["HeadAxis"]]),
        _dim(model["SlotsShape"][model["DimAxis"]]),
    ] + ([] if use_key_vector_copy else [str(slots_lane_count)]))
    lines = _helper_header(
        model,
        ("slot_mapping", "kv_cache", "num_blocks_per_shard", "cache_meta"),
        comment=f"# generated from PyNTT Jinja UpdatePagedAttentionKVCache.py.jinja\n# {model['Comment']}; dtype={model['SlotsDType']}, slots_shape={_shape_tuple(model['SlotsShape'])}, slots_global_shape={_shape_tuple(model['SlotsGlobalShape'])}, layer={model['LayerIdExpression']}, cache_kind={model['CacheKind']}",
    )
    _line(lines, 1, "shard_index = tl.program_id(0).to(tl.int64)")
    _line(lines, 1, f"layer_id_value = (tl.full((), 0, tl.int64) + ({model['LayerIdExpression']})).to(tl.int64)")
    _append_shard_coords(lines, 1, model["Hierarchy"])
    _line(lines, 1, "num_seqs = tl.load(cache_meta + 0).to(tl.int64)")
    _line(lines, 1, "num_tokens = tl.full((), 0, tl.int64)")
    _line(lines, 1, "for seq_id in tl.range(0, num_seqs):")
    _line(lines, 2, "context_len = tl.load(cache_meta + 1 + seq_id * 2)")
    _line(lines, 2, "seq_len = tl.load(cache_meta + 2 + seq_id * 2)")
    _line(lines, 2, "num_tokens += seq_len - context_len")
    _line(lines, 1)
    _line(lines, 1, f"for linear_start in tl.range(0, {total_elements}, block_size):")
    _line(lines, 2, "linear = linear_start + tl.arange(0, block_size)")
    _line(lines, 2, f"mask = linear < {total_elements}")
    if not use_key_vector_copy:
        _line(lines, 2, f"source_lane_id = linear % {slots_lane_count}")
        _line(lines, 2, f"tmp = linear // {slots_lane_count}")
    else:
        _line(lines, 2, "tmp = linear")
    _line(lines, 2, f"source_dim_block = tmp % {_dim(model['SlotsShape'][model['DimAxis']])}")
    _line(lines, 2, f"tmp = tmp // {_dim(model['SlotsShape'][model['DimAxis']])}")
    _line(lines, 2, f"head_id = tmp % {_dim(model['SlotsShape'][model['HeadAxis']])}")
    _line(lines, 2, f"token_id = tmp // {_dim(model['SlotsShape'][model['HeadAxis']])}")
    for axis in range(len(model["SlotsGlobalShape"])):
        if axis in (model["SeqAxis"], model["HeadAxis"], model["DimAxis"]):
            continue
        _line(lines, 2, f"local_idx{axis} = 0")
        _line(lines, 2, f"global_idx{axis} = local_idx{axis} + {_dim(model['SlotsGlobalOffsets'][axis])}")
    _line(lines, 2, f"global_token_id = token_id + {_dim(model['SlotsGlobalOffsets'][model['SeqAxis']])}")
    _line(lines, 2, f"global_head_id = head_id + {_dim(model['SlotsGlobalOffsets'][model['HeadAxis']])}")
    _line(lines, 2, f"global_source_dim_block = source_dim_block + {_dim(model['SlotsGlobalOffsets'][model['DimAxis']])}")
    if use_key_vector_copy:
        _line(lines, 2, f"logical_dim = global_source_dim_block * {slots_lane_count}")
    elif slots_lane_count == 1:
        _line(lines, 2, "logical_dim = global_source_dim_block")
    else:
        _line(lines, 2, f"logical_dim = global_source_dim_block * {slots_lane_count} + source_lane_id")
    _line(lines, 2, "cache_head_id = global_head_id")
    _line(lines, 2, f"active = mask & (global_token_id < num_tokens) & (cache_head_id < {cache['NumKVHeads']}) & (logical_dim < {cache['HeadDim']})")
    for axis in range(len(model["SlotsShape"])):
        local_index = local_index_name(axis)
        _line(lines, 2, f"active = active & ({local_index} < {_dim(model['SlotsShape'][axis])})")
    for axis in range(len(model["SlotsShape"])):
        _line(lines, 2, f"source_idx{axis} = {local_index_name(axis)}")
    if cache["IdLength"] > 1:
        _line(lines, 2, f"topology_id = tl.load(slot_mapping + global_token_id * {cache['IdLength']}, mask=active, other=0)")
        _line(lines, 2, "tmp_topology = topology_id")
        for index in range(len(cache["NumBlocksSplitAxes"]) - 1, -1, -1):
            axis = cache["NumBlocksSplitAxes"][index]
            _line(lines, 2, f"topology_coord{axis} = tmp_topology % {model['Hierarchy'][axis]}")
            _line(lines, 2, f"tmp_topology = tmp_topology // {model['Hierarchy'][axis]}")
        for axis in topology_match_axes:
            _line(lines, 2, f"active = active & (topology_coord{axis} == shard_coord{axis})")
    _line(lines, 2, f"slot_id = tl.load(slot_mapping + global_token_id * {cache['IdLength']} + {cache['IdLength'] - 1}, mask=active, other=0)")
    _line(lines, 2, f"block_id = slot_id // {cache['BlockSize']}")
    _line(lines, 2, f"block_offset = slot_id % {cache['BlockSize']}")
    if vectorized_dim == 5:  # HeadDim
        if use_key_vector_copy:
            _line(lines, 2, "cache_dim_block = source_dim_block")
        else:
            _line(lines, 2, f"cache_dim_block = logical_dim // {lane_count}")
        _line(lines, 2, "cache_block_offset = block_offset")
        if use_key_vector_copy:
            _line(lines, 2, "cache_lane_id = 0")
        else:
            _line(lines, 2, f"cache_lane_id = logical_dim % {lane_count}")
    elif vectorized_dim == 3:  # BlockSize
        _line(lines, 2, "cache_dim_block = logical_dim")
        _line(lines, 2, f"cache_block_offset = block_offset // {lane_count}")
        _line(lines, 2, f"cache_lane_id = block_offset % {lane_count}")
    else:
        _line(lines, 2, "cache_dim_block = logical_dim")
        _line(lines, 2, "cache_block_offset = block_offset")
        _line(lines, 2, "cache_lane_id = 0")
    _line(lines, 2, f"slot_offsets = {slot_offset(None) if use_key_vector_copy else slot_offset()}")
    _line(lines, 2, f"cache_offsets = {cache_offset}")
    if use_key_vector_copy:
        _line(lines, 2, f"source_lane_offsets = tl.arange(0, {slots_lane_count})")
        source_offsets = "slot_offsets[:, None] + source_lane_offsets[None, :]"
        source_pointer = _access_pointer(model, "Slots", _ptr(model, "Slots"), source_offsets)
        _line(lines, 2, f"values = tl.load({source_pointer}, mask=active[:, None], other=0.0)")
        _line(lines, 2, "tl.store(kv_cache + cache_offsets[:, None] + source_lane_offsets[None, :], values, mask=active[:, None])")
    else:
        source_pointer = _access_pointer(model, "Slots", _ptr(model, "Slots"), "slot_offsets")
        _line(lines, 2, f"value = tl.load({source_pointer}, mask=active, other=0.0)")
        _line(lines, 2, "tl.store(kv_cache + cache_offsets, value, mask=active)")
    return _finish(lines)


_EMITTERS = {
    "Concat": _emit_concat,
    "Conv2D": _emit_conv2d,
    "ElementwiseBinary": _emit_elementwise_binary,
    "ElementwiseCast": _emit_elementwise_cast,
    "ElementwiseUnary": _emit_elementwise_unary,
    "ElementwiseWhere": _emit_elementwise_where,
    "Gather": _emit_gather,
    "Gemv": _emit_gemv,
    "GetPositionIds": _emit_get_position_ids,
    "LayerNorm": _emit_layer_norm,
    "MatMulGlu": _emit_matmul_glu,
    "Matmul": _emit_matmul,
    "Memcopy": _emit_memcopy,
    "NormApply": _emit_norm_apply,
    "NormStats": _emit_norm_stats,
    "Pad": _emit_pad,
    "PagedAttention": _emit_paged_attention,
    "PackedMatMulGlu": _emit_packed_matmul_glu,
    "PackedQKVParallelLinear": _emit_packed_qkv_parallel_linear,
    "QKVParallelLinear": _emit_qkv_parallel_linear,
    "Reduce": _emit_reduce,
    "Reshard": _emit_reshard,
    "RoPE": _emit_rope,
    "ScatterND": _emit_scatter_nd,
    "Slice": _emit_slice,
    "Softmax": _emit_softmax,
    "Summa": _emit_summa,
    "TensorLoad": _emit_tensor_load,
    "TensorRegionCopy": _emit_tensor_region_copy,
    "TensorStore": _emit_tensor_store,
    "Transpose": _emit_transpose,
    "UpdatePagedAttentionKVCache": _emit_update_paged_attention_kv_cache,
    "VectorLayout": _emit_vector_layout,
}
