"""torch.Tensor validation and allocation utilities for PyNTT."""

from __future__ import annotations

from typing import Any

from pyntt.ir import FunctionSpec, TensorSpec
from pyntt.runtime.errors import PyNTTArgumentError, PyNTTBackendError


def _import_torch():
    try:
        import torch
    except ImportError as ex:
        raise PyNTTBackendError(
            "PyNTT runtime requires torch to validate and allocate tensors."
        ) from ex
    return torch


def _torch_dtype(torch, dtype: str):
    dtype_map = {
        "bool": torch.bool,
        "i8": torch.int8,
        "int8": torch.int8,
        "u8": torch.uint8,
        "uint8": torch.uint8,
        "i16": torch.int16,
        "int16": torch.int16,
        "u16": getattr(torch, "uint16", None),
        "uint16": getattr(torch, "uint16", None),
        "i32": torch.int32,
        "int32": torch.int32,
        "u32": getattr(torch, "uint32", None),
        "uint32": getattr(torch, "uint32", None),
        "i64": torch.int64,
        "int64": torch.int64,
        "u64": getattr(torch, "uint64", None),
        "uint64": getattr(torch, "uint64", None),
        "f16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "f32": torch.float32,
        "float32": torch.float32,
        "f64": torch.float64,
        "float64": torch.float64,
    }
    try:
        torch_dtype = dtype_map[dtype]
    except KeyError as ex:
        raise PyNTTArgumentError(f"Unsupported PyNTT tensor dtype: {dtype}") from ex
    if torch_dtype is None:
        raise PyNTTArgumentError(
            f"PyNTT tensor dtype {dtype} is not supported by the installed torch."
        )
    return torch_dtype


def validate_inputs(function: FunctionSpec, inputs: tuple[Any, ...]) -> None:
    """Validate runtime inputs against a function spec."""
    torch = _import_torch()

    if len(inputs) != len(function.inputs):
        raise PyNTTArgumentError(
            f"Function {function.name} expects {len(function.inputs)} inputs, "
            f"got {len(inputs)}."
        )

    for index, (tensor, spec) in enumerate(zip(inputs, function.inputs)):
        _validate_tensor(torch, function.name, index, tensor, spec)


def allocate_outputs(function: FunctionSpec, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    """Allocate output tensors for a function spec."""
    torch = _import_torch()
    outputs = []
    for spec in function.outputs:
        outputs.append(
            torch.empty(
                tuple(spec.shape),
                dtype=_torch_dtype(torch, spec.dtype),
                device=_resolve_output_device(spec, inputs),
            )
        )
    return tuple(outputs)


def _validate_tensor(torch, function_name: str, index: int, tensor: Any, spec: TensorSpec) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise PyNTTArgumentError(
            f"Function {function_name} input {index} ({spec.name}) must be "
            f"torch.Tensor, got {type(tensor).__name__}."
        )

    expected_dtype = _torch_dtype(torch, spec.dtype)
    if tensor.dtype != expected_dtype:
        raise PyNTTArgumentError(
            f"Function {function_name} input {index} ({spec.name}) expects "
            f"dtype {expected_dtype}, got {tensor.dtype}."
        )

    expected_shape = tuple(spec.shape)
    actual_shape = tuple(tensor.shape)
    if actual_shape != expected_shape:
        raise PyNTTArgumentError(
            f"Function {function_name} input {index} ({spec.name}) expects "
            f"shape {expected_shape}, got {actual_shape}."
        )

    if spec.strides is not None and tuple(tensor.stride()) != tuple(spec.strides):
        raise PyNTTArgumentError(
            f"Function {function_name} input {index} ({spec.name}) expects "
            f"strides {tuple(spec.strides)}, got {tuple(tensor.stride())}."
        )

    if spec.layout == "contiguous" and not tensor.is_contiguous():
        raise PyNTTArgumentError(
            f"Function {function_name} input {index} ({spec.name}) must be contiguous."
        )

    _validate_device(function_name, index, tensor, spec)


def _validate_device(function_name: str, index: int, tensor: Any, spec: TensorSpec) -> None:
    if spec.device in ("any", "like_input", ""):
        return

    if spec.device == "cuda" and not tensor.is_cuda:
        raise PyNTTArgumentError(
            f"Function {function_name} input {index} ({spec.name}) expects a CUDA tensor, "
            f"got {tensor.device}."
        )

    if spec.device == "cpu" and tensor.device.type != "cpu":
        raise PyNTTArgumentError(
            f"Function {function_name} input {index} ({spec.name}) expects a CPU tensor, "
            f"got {tensor.device}."
        )


def _resolve_output_device(spec: TensorSpec, inputs: tuple[Any, ...]):
    if spec.device in ("any", "like_input", ""):
        if inputs:
            return inputs[0].device
        return "cpu"

    return spec.device
