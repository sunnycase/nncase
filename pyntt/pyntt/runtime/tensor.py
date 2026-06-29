"""torch.Tensor validation and allocation utilities for PyNTT."""

from __future__ import annotations

from typing import Any, Mapping

from pyntt.ir import FunctionSpec, TensorSpec
from pyntt.runtime.errors import PyNTTArgumentError, PyNTTBackendError


_KV_CACHE_STORAGE_CACHE: dict[tuple[Any, ...], Any] = {}


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


def is_object_spec(spec: TensorSpec) -> bool:
    """Return whether a tensor spec represents a runtime object."""
    return spec.memory == "object" or spec.layout == "object" or spec.dtype == "object"


def resolve_shape_env(function: FunctionSpec, inputs: tuple[Any, ...]) -> dict[str, int]:
    """Resolve dynamic dimension variables from runtime input tensors."""
    env: dict[str, int] = {}
    for binding in function.shape_bindings:
        if binding.input_index >= len(inputs):
            raise PyNTTArgumentError(
                f"Function {function.name} shape binding {binding.name} references "
                f"missing input {binding.input_index}."
            )

        tensor = inputs[binding.input_index]
        if binding.axis >= len(tensor.shape):
            raise PyNTTArgumentError(
                f"Function {function.name} shape binding {binding.name} references "
                f"axis {binding.axis} of rank-{len(tensor.shape)} input {binding.input_index}."
            )

        value = int(tensor.shape[binding.axis])
        if binding.min_value is not None and value < binding.min_value:
            raise PyNTTArgumentError(
                f"Dynamic dimension {binding.name}={value} is below "
                f"the allowed minimum {binding.min_value}."
            )
        if binding.max_value is not None and value > binding.max_value:
            raise PyNTTArgumentError(
                f"Dynamic dimension {binding.name}={value} is above "
                f"the allowed maximum {binding.max_value}."
            )

        previous = env.get(binding.name)
        if previous is not None and previous != value:
            raise PyNTTArgumentError(
                f"Dynamic dimension {binding.name} has inconsistent values: "
                f"{previous} and {value}."
            )
        env[binding.name] = value
    return env


def validate_inputs(
    function: FunctionSpec,
    inputs: tuple[Any, ...],
    shape_env: Mapping[str, int] | None = None,
) -> None:
    """Validate runtime inputs against a function spec."""
    torch = _import_torch()
    shape_env = shape_env or resolve_shape_env(function, inputs)

    if len(inputs) != len(function.inputs):
        raise PyNTTArgumentError(
            f"Function {function.name} expects {len(function.inputs)} inputs, "
            f"got {len(inputs)}."
        )

    for index, (tensor, spec) in enumerate(zip(inputs, function.inputs)):
        _validate_tensor(torch, function.name, index, tensor, spec, shape_env)


def allocate_outputs(
    function: FunctionSpec,
    inputs: tuple[Any, ...],
    shape_env: Mapping[str, int] | None = None,
) -> tuple[Any, ...]:
    """Allocate output tensors for a function spec."""
    torch = _import_torch()
    shape_env = shape_env or resolve_shape_env(function, inputs)
    outputs = []
    for spec in function.outputs:
        outputs.append(
            torch.empty(
                _resolve_shape(spec.shape, shape_env),
                dtype=_torch_dtype(torch, spec.dtype),
                device=_resolve_output_device(spec, inputs),
            )
        )
    return tuple(outputs)


def _validate_tensor(
    torch,
    function_name: str,
    index: int,
    tensor: Any,
    spec: TensorSpec,
    shape_env: Mapping[str, int],
) -> None:
    if is_object_spec(spec):
        if tensor is None:
            raise PyNTTArgumentError(
                f"Function {function_name} input {index} ({spec.name}) must not be None."
            )
        return

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

    expected_shape = _resolve_shape(spec.shape, shape_env)
    actual_shape = tuple(tensor.shape)
    if actual_shape != expected_shape:
        raise PyNTTArgumentError(
            f"Function {function_name} input {index} ({spec.name}) expects "
            f"shape {expected_shape}, got {actual_shape}."
        )

    expected_strides = None if spec.strides is None else _resolve_shape(spec.strides, shape_env)
    if expected_strides is not None and tuple(tensor.stride()) != expected_strides:
        raise PyNTTArgumentError(
            f"Function {function_name} input {index} ({spec.name}) expects "
            f"strides {expected_strides}, got {tuple(tensor.stride())}."
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
        for input_value in inputs:
            device = getattr(input_value, "device", None)
            if device is not None:
                return device
        return "cpu"

    return spec.device


def materialize_kv_cache_metadata(value: Any, device=None):
    """Pack a paged-attention KV cache object's lengths for Triton kernels.

    The generated Triton kernel consumes a flat int64 tensor:
    [num_seqs, context0, seq0, context1, seq1, ...].
    """
    torch = _import_torch()
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=torch.int64) if device is not None else value.to(dtype=torch.int64)
        if tensor.ndim == 1:
            return tensor.contiguous()
        if tensor.ndim == 2 and tensor.shape[-1] == 2:
            num_seqs = torch.tensor([tensor.shape[0]], dtype=torch.int64, device=tensor.device)
            return torch.cat([num_seqs, tensor.reshape(-1).to(torch.int64)]).contiguous()
        raise PyNTTArgumentError(
            "KV cache metadata tensor must be flat [1 + 2 * num_seqs] or [num_seqs, 2]."
        )

    context_lens = _extract_kv_cache_field(value, "context_lens")
    seq_lens = _extract_kv_cache_field(value, "seq_lens")
    context_lens_tensor = _to_int64_tensor(torch, context_lens, device)
    seq_lens_tensor = _to_int64_tensor(torch, seq_lens, device)
    if context_lens_tensor.numel() != seq_lens_tensor.numel():
        raise PyNTTArgumentError(
            "KV cache context_lens and seq_lens must have the same number of elements."
        )

    num_seqs = torch.tensor(
        [context_lens_tensor.numel()],
        dtype=torch.int64,
        device=context_lens_tensor.device,
    )
    interleaved = torch.stack(
        [context_lens_tensor.reshape(-1), seq_lens_tensor.reshape(-1)], dim=1
    ).reshape(-1)
    return torch.cat([num_seqs, interleaved]).contiguous()


def materialize_kv_cache_tensor_field(value: Any, name: str, device=None, dtype: str = "int64"):
    """Materialize a paged-attention KV-cache tensor field as a torch tensor."""
    torch = _import_torch()
    field = _extract_kv_cache_field(value, name)
    return _to_torch_tensor(torch, field, device=device, dtype=_torch_dtype(torch, dtype)).contiguous()


def materialize_kv_cache_storage(
    value: Any,
    device=None,
    dtype: str = "float16",
    topology_shape: tuple[int, ...] = (1,),
    tail_shape: tuple[int, ...] = (),
    block_size: int = 1,
):
    """Return persistent torch storage for PyNTT paged-attention kernels.

    Native nncase runtime KV-cache objects expose a small address table rather
    than the host storage itself. PyNTT kernels need a CUDA tensor, so this
    helper allocates and caches a torch-backed storage tensor keyed by that
    address table. If a reference/evaluator object already exposes the full
    cache tensor, it is used to initialize the storage.
    """
    torch = _import_torch()
    torch_dtype = _torch_dtype(torch, dtype)
    topology_shape = tuple(int(dim) for dim in topology_shape)
    tail_shape = tuple(int(dim) for dim in tail_shape)
    if not tail_shape:
        raise PyNTTArgumentError("KV cache storage tail_shape must not be empty.")
    if block_size <= 0:
        raise PyNTTArgumentError(f"KV cache block_size must be positive, got {block_size}.")

    required_blocks = _infer_required_kv_blocks(value, block_size, topology_shape)
    raw_storage = _extract_kv_cache_field(value, "kv_caches")
    storage_key = _kv_cache_storage_key(raw_storage, dtype, topology_shape, tail_shape)
    existing = _KV_CACHE_STORAGE_CACHE.get(storage_key)

    if existing is not None and _storage_has_enough_blocks(existing, len(topology_shape), required_blocks):
        if device is not None and existing.device != torch.device(device):
            existing = existing.to(device=device)
            _KV_CACHE_STORAGE_CACHE[storage_key] = existing
        return existing

    initial = _try_materialize_initial_kv_storage(
        torch,
        raw_storage,
        device,
        torch_dtype,
        topology_shape,
        tail_shape,
        required_blocks,
    )
    if existing is not None:
        initial = _grow_kv_storage(torch, existing, initial)

    _KV_CACHE_STORAGE_CACHE[storage_key] = initial
    return initial


def materialize_kv_cache_blocks_per_shard(
    value: Any,
    topology_shape: tuple[int, ...] = (),
    block_size: int = 1,
) -> int:
    """Return the local block capacity used by PyNTT KV-cache storage."""
    topology_shape = tuple(int(dim) for dim in topology_shape)
    if block_size <= 0:
        raise PyNTTArgumentError(f"KV cache block_size must be positive, got {block_size}.")
    return _infer_required_kv_blocks(value, block_size, topology_shape)


def _extract_kv_cache_field(value: Any, name: str):
    if not hasattr(value, name):
        raise PyNTTArgumentError(
            f"KV cache object must expose {name!r}; got {type(value).__name__}."
        )
    field = getattr(value, name)
    return field() if callable(field) else field


def _to_torch_tensor(torch, value: Any, device, dtype=None):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype) if device is not None or dtype is not None else value

    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]

    if hasattr(value, "to_runtime_tensor"):
        value = value.to_runtime_tensor()

    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()

    return torch.as_tensor(value, dtype=dtype, device=device)


def _to_int64_tensor(torch, value: Any, device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.int64).contiguous()

    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()

    return torch.as_tensor(value, dtype=torch.int64, device=device).contiguous()


def _infer_required_kv_blocks(value: Any, block_size: int, topology_shape: tuple[int, ...] = ()) -> int:
    required = 1
    topology_extent = 1
    for dim in topology_shape:
        topology_extent *= max(1, int(dim))

    num_blocks = getattr(value, "num_blocks", None)
    if callable(num_blocks):
        num_blocks = num_blocks()
    if num_blocks is not None:
        try:
            total_blocks = int(num_blocks)
            required = max(required, (total_blocks + topology_extent - 1) // topology_extent)
        except TypeError:
            pass

    try:
        slot_mapping = _runtime_value_to_numpy(_extract_kv_cache_field(value, "slot_mapping"))
        if slot_mapping is not None and slot_mapping.size > 0:
            required = max(required, int(slot_mapping[..., -1].max()) // block_size + 1)
    except PyNTTArgumentError:
        pass

    try:
        block_tables = _runtime_value_to_numpy(_extract_kv_cache_field(value, "block_tables"))
        if block_tables is not None and block_tables.size > 0:
            required = max(required, int(block_tables[..., -1].max()) + 1)
    except PyNTTArgumentError:
        pass

    return required


def _kv_cache_storage_key(raw_storage: Any, dtype: str, topology_shape: tuple[int, ...], tail_shape: tuple[int, ...]):
    raw_value = raw_storage[0] if isinstance(raw_storage, (list, tuple)) and len(raw_storage) == 1 else raw_storage
    raw_numpy = _runtime_value_to_numpy(raw_value)
    if raw_numpy is not None and raw_numpy.dtype.kind in ("i", "u") and raw_numpy.size <= 4096:
        return ("kv_addr_table", tuple(int(v) for v in raw_numpy.reshape(-1).tolist()), dtype, topology_shape, tail_shape)
    return ("kv_object", id(raw_value), dtype, topology_shape, tail_shape)


def _runtime_value_to_numpy(value: Any):
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if hasattr(value, "to_runtime_tensor"):
        value = value.to_runtime_tensor()
    if hasattr(value, "to_numpy"):
        return value.to_numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return None


def _storage_has_enough_blocks(storage: Any, topology_rank: int, required_blocks: int) -> bool:
    return len(storage.shape) > topology_rank and int(storage.shape[topology_rank]) >= required_blocks


def _try_materialize_initial_kv_storage(
    torch,
    raw_storage: Any,
    device,
    torch_dtype,
    topology_shape: tuple[int, ...],
    tail_shape: tuple[int, ...],
    required_blocks: int,
):
    raw_tensor = raw_storage[0] if isinstance(raw_storage, (list, tuple)) and len(raw_storage) == 1 else raw_storage
    raw_numpy = _runtime_value_to_numpy(raw_tensor)
    if raw_numpy is not None and raw_numpy.dtype.kind not in ("i", "u"):
        raw_torch = torch.as_tensor(raw_numpy, dtype=torch_dtype, device=device).contiguous()
        if len(raw_torch.shape) >= len(topology_shape) + 1 and tuple(raw_torch.shape[:len(topology_shape)]) == topology_shape:
            return raw_torch

    shape = topology_shape + (required_blocks,) + tail_shape
    return torch.zeros(shape, dtype=torch_dtype, device=device)


def _grow_kv_storage(torch, old_storage: Any, new_storage: Any):
    if tuple(old_storage.shape) == tuple(new_storage.shape):
        return old_storage

    slices = tuple(slice(0, min(int(old_storage.shape[i]), int(new_storage.shape[i]))) for i in range(len(new_storage.shape)))
    new_storage[slices].copy_(old_storage[slices].to(device=new_storage.device, dtype=new_storage.dtype))
    return new_storage.contiguous()


def _resolve_shape(shape, shape_env: Mapping[str, int]) -> tuple[int, ...]:
    return tuple(_resolve_dim(dim, shape_env) for dim in shape)


def _resolve_dim(dim, shape_env: Mapping[str, int]) -> int:
    if isinstance(dim, int):
        return dim
    if isinstance(dim, str):
        try:
            return int(dim)
        except ValueError:
            pass

        allowed_globals = {"__builtins__": {}}
        allowed_locals = {
            **{name: int(value) for name, value in shape_env.items()},
            "abs": abs,
            "max": max,
            "min": min,
        }
        try:
            value = eval(dim, allowed_globals, allowed_locals)  # pylint: disable=eval-used
        except NameError as ex:
            raise PyNTTArgumentError(
                f"Cannot resolve dynamic PyNTT dimension expression {dim!r}; "
                f"known symbols are {sorted(shape_env)}."
            ) from ex
        return int(value)
    raise PyNTTArgumentError(f"Invalid PyNTT dimension expression: {dim!r}.")
