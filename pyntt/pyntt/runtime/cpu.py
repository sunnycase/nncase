"""Direct CPU NTT module runtime used by heterogeneous PyNTT packages."""

from __future__ import annotations

import ctypes
import mmap
import queue
import threading
from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from pyntt.runtime.pipeline import PipelineChannel


class _TensorDesc(ctypes.Structure):
    _fields_ = (
        ("data", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("shape", ctypes.POINTER(ctypes.c_size_t)),
        ("strides", ctypes.POINTER(ctypes.c_size_t)),
        ("rank", ctypes.c_size_t),
    )


class _RunParams(ctypes.Structure):
    _fields_ = (
        ("function_id", ctypes.c_uint32),
        ("bdim", ctypes.c_size_t),
        ("cdim", ctypes.c_size_t),
        ("inputs", ctypes.POINTER(_TensorDesc)),
        ("input_count", ctypes.c_size_t),
        ("outputs", ctypes.POINTER(_TensorDesc)),
        ("output_count", ctypes.c_size_t),
        ("rdata", ctypes.c_void_p),
        ("rdata_size", ctypes.c_size_t),
        ("block_local_rdata", ctypes.c_void_p),
        ("block_local_rdata_size", ctypes.c_size_t),
        ("data", ctypes.c_void_p),
        ("data_bytes_per_block", ctypes.c_size_t),
        ("block_local_data", ctypes.c_void_p),
        ("block_local_data_bytes_per_block", ctypes.c_size_t),
        ("output", ctypes.c_void_p),
        ("output_size", ctypes.c_size_t),
    )


class _PagedAttentionKVCacheDesc(ctypes.Structure):
    _fields_ = (
        ("num_seqs", ctypes.c_size_t),
        ("num_tokens", ctypes.c_size_t),
        ("context_lens", ctypes.POINTER(ctypes.c_int64)),
        ("context_lens_size", ctypes.c_size_t),
        ("seq_lens", ctypes.POINTER(ctypes.c_int64)),
        ("seq_lens_size", ctypes.c_size_t),
        ("block_table", ctypes.POINTER(ctypes.c_int64)),
        ("block_table_shape", ctypes.c_size_t * 3),
        ("slot_mapping", ctypes.POINTER(ctypes.c_int64)),
        ("slot_mapping_shape", ctypes.c_size_t * 2),
        ("kv_cache_addrs", ctypes.c_ssize_t * 128),
    )


@dataclass(frozen=True)
class _PreparedInput:
    data: int
    size: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    keepalive: tuple[Any, ...]


@dataclass(frozen=True)
class _PreparedInvocation:
    params: _RunParams
    outputs: tuple[torch.Tensor, ...]
    keepalive: tuple[Any, ...]


@dataclass(frozen=True)
class CpuOutputSpec:
    dtype: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    offset_bytes: int


@dataclass(frozen=True)
class CpuInputSpec:
    kind: str
    dtype: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    lane_shape: tuple[int, ...]


@dataclass(frozen=True)
class CpuFunctionSpec:
    function_id: int
    output_pool_bytes: int
    data_bytes_per_block: int
    block_local_data_bytes_per_block: int


@dataclass(frozen=True)
class _RegisteredPipelineCall:
    function: CpuFunctionSpec
    input_specs: tuple[CpuInputSpec, ...]
    output_specs: tuple[CpuOutputSpec, ...]


@dataclass(frozen=True)
class _BoundPipelineCall:
    inputs: tuple[Any, ...]
    prepared_inputs: tuple[_PreparedInput, ...]
    invocation: _PreparedInvocation


class CpuInvocation:
    """One asynchronously executing CPU NTT worker invocation."""

    def __init__(self, future: Future[tuple[torch.Tensor, ...]]):
        self._future = future

    def wait(self) -> tuple[torch.Tensor, ...]:
        """Wait for completion and propagate the native worker exception."""
        return self._future.result()


class _CpuWorker:
    """One persistent submission queue for the blocking native CPU entry."""

    def __init__(self):
        self._queue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._run,
            name="pyntt-cpu-worker",
            daemon=True,
        )
        self._thread.start()

    def submit(self, function, *args) -> Future:
        future = Future()
        self._queue.put((future, function, args))
        return future

    def _run(self) -> None:
        while True:
            future, function, args = self._queue.get()
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(function(*args))
            except BaseException as ex:
                future.set_exception(ex)


_TORCH_DTYPES = {
    "bool": torch.bool,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int16": torch.int16,
    "uint16": torch.uint16,
    "int32": torch.int32,
    "uint32": torch.uint32,
    "int64": torch.int64,
    "uint64": torch.uint64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}
if hasattr(torch, "float8_e4m3fn"):
    _TORCH_DTYPES["float8e4m3fn"] = torch.float8_e4m3fn
if hasattr(torch, "float8_e5m2"):
    _TORCH_DTYPES["float8e5m2"] = torch.float8_e5m2


class _MappedAsset:
    def __init__(self, path: Path):
        self._file = path.open("r+b")
        self.mapping = None
        self.buffer = None
        self._size = path.stat().st_size
        if self._size:
            self.mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_WRITE)
            self.buffer = (ctypes.c_byte * len(self.mapping)).from_buffer(self.mapping)

    @property
    def pointer(self) -> int:
        return ctypes.addressof(self.buffer) if self.buffer is not None else 0

    @property
    def size(self) -> int:
        return self._size


class CpuNttModule:
    """Owns one generated CPU NTT shared library and persistent host storage."""

    def __init__(self, base_dir: Path, bdim: int, cdim: int):
        self._base_dir = Path(base_dir)
        self._bdim = int(bdim)
        self._cdim = int(cdim)
        if self._bdim <= 0 or self._cdim <= 0:
            raise ValueError(f"CPU NTT topology must be positive, got {bdim}x{cdim}.")

        self._library = ctypes.CDLL(str(self._base_dir / "cpu_module.so"))
        self._run = self._library.nncase_ntt_cpu_run
        self._run.argtypes = (ctypes.POINTER(_RunParams),)
        self._run.restype = ctypes.c_int32
        self._last_error = self._library.nncase_ntt_cpu_last_error
        self._last_error.argtypes = ()
        self._last_error.restype = ctypes.c_char_p
        self._rdata = _MappedAsset(self._base_dir / "cpu_rdata.bin")
        self._block_local_rdata = _MappedAsset(
            self._base_dir / "cpu_block_local_rdata.bin"
        )
        self._storage: dict[tuple[int, int, int, int], tuple[torch.Tensor, ...]] = {}
        self._pipeline_calls: dict[int, _RegisteredPipelineCall] = {}
        self._bound_pipeline_calls: dict[int, _BoundPipelineCall] = {}
        self._worker = _CpuWorker()

    @staticmethod
    def to_device(tensor: torch.Tensor, device: Any) -> torch.Tensor:
        target = torch.device(device)
        if tensor.device == target:
            return tensor
        return tensor.to(device=target, non_blocking=False).contiguous()

    def invoke(
        self,
        function: CpuFunctionSpec,
        inputs: Sequence[Any],
        input_specs: Sequence[CpuInputSpec],
        output_specs: Sequence[CpuOutputSpec],
    ) -> tuple[torch.Tensor, ...]:
        prepared_inputs = self._prepare_inputs(inputs, input_specs)
        invocation = self._bind_invocation(
            function, prepared_inputs, output_specs
        )
        return self._submit(function.function_id, invocation).wait()

    def register_pipeline_call(
        self,
        call_id: int,
        function: CpuFunctionSpec,
        input_specs: Sequence[CpuInputSpec],
        output_specs: Sequence[CpuOutputSpec],
    ) -> None:
        """Register one immutable compiler-generated pipeline call schema."""
        call_id = int(call_id)
        registered = _RegisteredPipelineCall(
            function,
            tuple(input_specs),
            tuple(output_specs),
        )
        existing = self._pipeline_calls.get(call_id)
        if existing is not None and existing != registered:
            raise ValueError(
                f"CPU NTT pipeline call {call_id} is already registered with "
                "a different ABI."
            )
        self._pipeline_calls[call_id] = registered

    def start(
        self,
        call_id: int,
        inputs: Sequence[Any],
    ) -> CpuInvocation:
        """Submit a registered pipeline call, binding its dynamic objects once."""
        call_id = int(call_id)
        try:
            registered = self._pipeline_calls[call_id]
        except KeyError as ex:
            raise KeyError(
                f"CPU NTT pipeline call {call_id} was not registered."
            ) from ex

        inputs = tuple(inputs)
        if len(inputs) != len(registered.input_specs):
            raise ValueError(
                f"CPU NTT pipeline call {call_id} received {len(inputs)} "
                f"inputs but its ABI declares {len(registered.input_specs)}."
            )

        bound = self._bound_pipeline_calls.get(call_id)
        if bound is not None and self._binding_matches(
            inputs,
            bound,
            registered.input_specs,
        ):
            return self._submit(registered.function.function_id, bound.invocation)

        prepared_inputs = self._prepare_inputs(inputs, registered.input_specs)
        invocation = self._bind_invocation(
            registered.function,
            prepared_inputs,
            registered.output_specs,
        )
        self._bound_pipeline_calls[call_id] = _BoundPipelineCall(
            inputs,
            prepared_inputs,
            invocation,
        )
        return self._submit(registered.function.function_id, invocation)

    def _bind_invocation(
        self,
        function: CpuFunctionSpec,
        cpu_inputs: tuple[_PreparedInput, ...],
        output_specs: Sequence[CpuOutputSpec],
    ) -> _PreparedInvocation:
        block_count = self._bdim * self._cdim
        key = (
            function.function_id,
            function.output_pool_bytes,
            function.data_bytes_per_block,
            function.block_local_data_bytes_per_block,
        )
        storage = self._storage.get(key)
        if storage is None:
            pin = torch.cuda.is_available()
            storage = (
                torch.empty(max(1, function.output_pool_bytes), dtype=torch.uint8, pin_memory=pin),
                torch.empty(
                    max(1, function.data_bytes_per_block * block_count),
                    dtype=torch.uint8,
                    pin_memory=pin,
                ),
                torch.empty(
                    max(1, function.block_local_data_bytes_per_block * block_count),
                    dtype=torch.uint8,
                    pin_memory=pin,
                ),
            )
            self._storage[key] = storage

        output_pool, data, block_local_data = storage
        return self._prepare_invocation(
            function,
            cpu_inputs,
            output_specs,
            output_pool,
            data,
            block_local_data,
        )

    def _submit(
        self,
        function_id: int,
        invocation: _PreparedInvocation,
    ) -> CpuInvocation:
        future = self._worker.submit(
            self._invoke_prepared,
            function_id,
            invocation.params,
            invocation.outputs,
            invocation.keepalive,
        )
        return CpuInvocation(future)

    @classmethod
    def _prepare_inputs(
        cls,
        inputs: Sequence[Any],
        input_specs: Sequence[CpuInputSpec],
    ) -> tuple[_PreparedInput, ...]:
        if len(inputs) != len(input_specs):
            raise ValueError(
                f"CPU NTT call received {len(inputs)} inputs but its ABI "
                f"declares {len(input_specs)}."
            )
        return tuple(
            cls._prepare_input(value, input_specs[index], index)
            for index, value in enumerate(inputs)
        )

    @classmethod
    def _binding_matches(
        cls,
        inputs: tuple[Any, ...],
        bound: _BoundPipelineCall,
        input_specs: tuple[CpuInputSpec, ...],
    ) -> bool:
        if len(inputs) != len(bound.inputs):
            return False
        return all(
            cls._input_binding_matches(value, old_value, prepared, spec)
            for value, old_value, prepared, spec in zip(
                inputs,
                bound.inputs,
                bound.prepared_inputs,
                input_specs,
            )
        )

    @classmethod
    def _input_binding_matches(
        cls,
        value: Any,
        old_value: Any,
        prepared: _PreparedInput,
        spec: CpuInputSpec,
    ) -> bool:
        if spec.kind == "dimension" and isinstance(value, int):
            return isinstance(old_value, int) and value == old_value
        if value is not old_value:
            return False
        if spec.kind == "tensor" or (
            spec.kind == "dimension" and isinstance(value, torch.Tensor)
        ):
            return (
                isinstance(value, torch.Tensor)
                and value.data_ptr() == prepared.data
                and value.numel() * value.element_size() == prepared.size
            )

        # Reference-valued inputs have a stable ABI for the lifetime of the
        # object. Their tensor contents may change in place between launches;
        # replacing backing storage requires passing a new reference object.
        return True

    def _prepare_invocation(
        self,
        function: CpuFunctionSpec,
        cpu_inputs: tuple[_PreparedInput, ...],
        output_specs: Sequence[CpuOutputSpec],
        output_pool: torch.Tensor,
        data: torch.Tensor,
        block_local_data: torch.Tensor,
    ) -> _PreparedInvocation:
        outputs = tuple(
            self._view_output(output_pool, spec) for spec in output_specs
        )
        input_descs, input_shapes, input_strides = self._make_prepared_descs(
            cpu_inputs
        )
        output_descs, output_shapes, output_strides = self._make_descs(
            outputs, require_contiguous=False
        )
        params = _RunParams(
            function_id=function.function_id,
            bdim=self._bdim,
            cdim=self._cdim,
            inputs=input_descs,
            input_count=len(cpu_inputs),
            outputs=output_descs,
            output_count=len(outputs),
            rdata=self._rdata.pointer,
            rdata_size=self._rdata.size,
            block_local_rdata=self._block_local_rdata.pointer,
            block_local_rdata_size=self._block_local_rdata.size,
            data=data.data_ptr(),
            data_bytes_per_block=function.data_bytes_per_block,
            block_local_data=block_local_data.data_ptr(),
            block_local_data_bytes_per_block=function.block_local_data_bytes_per_block,
            output=output_pool.data_ptr(),
            output_size=function.output_pool_bytes,
        )
        keepalive = (
            cpu_inputs,
            output_pool,
            data,
            block_local_data,
            input_descs,
            input_shapes,
            input_strides,
            output_descs,
            output_shapes,
            output_strides,
        )
        return _PreparedInvocation(params, outputs, keepalive)

    def _invoke_prepared(
        self,
        function_id: int,
        params: _RunParams,
        outputs: tuple[torch.Tensor, ...],
        keepalive: tuple[Any, ...],
    ) -> tuple[torch.Tensor, ...]:
        status = self._run(ctypes.byref(params))
        if status != 0:
            message = self._last_error()
            detail = message.decode("utf-8") if message else "unknown error"
            raise RuntimeError(
                f"CPU NTT function {function_id} failed: {detail}"
            )

        _ = keepalive
        return outputs

    @classmethod
    def _prepare_input(
        cls, value: Any, spec: CpuInputSpec, index: int
    ) -> _PreparedInput:
        if spec.kind == "tensor":
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"CPU NTT tensor input {index} must be a torch.Tensor, "
                    f"got {type(value).__name__}."
                )
            return cls._prepare_tensor_input(cls.to_device(value, "cpu"), spec)
        if spec.kind == "dimension":
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise ValueError(
                        f"CPU NTT dimension input {index} must contain one value."
                    )
                scalar = value.to(device="cpu", dtype=torch.int64).reshape(())
            elif isinstance(value, int):
                scalar = torch.tensor(value, dtype=torch.int64)
            else:
                raise TypeError(
                    f"CPU NTT dimension input {index} must be an integer, "
                    f"got {type(value).__name__}."
                )
            return cls._prepare_tensor_input(scalar, spec)
        if spec.kind == "paged_attention_kv_cache":
            if not cls._is_paged_attention_kv_cache(value):
                raise TypeError(
                    f"CPU NTT input {index} does not implement the "
                    "paged-attention KV-cache protocol."
                )
            return cls._prepare_paged_attention_kv_cache(value)
        if spec.kind == "pipeline_channel":
            if not isinstance(value, PipelineChannel):
                raise TypeError(
                    f"CPU NTT pipeline channel input {index} must be a "
                    f"PipelineChannel, got {type(value).__name__}."
                )
            return _PreparedInput(
                data=value.data_ptr,
                size=value.total_bytes,
                shape=(),
                strides=(),
                keepalive=(value,),
            )
        raise ValueError(
            f"CPU NTT input {index} has unsupported ABI kind {spec.kind!r}."
        )

    @staticmethod
    def _prepare_tensor_input(
        tensor: torch.Tensor, spec: CpuInputSpec
    ) -> _PreparedInput:
        if tensor.device.type != "cpu":
            raise ValueError(
                f"CPU NTT tensor input must reside on CPU, got {tensor.device}."
            )
        try:
            expected_dtype = _TORCH_DTYPES[spec.dtype]
        except KeyError as ex:
            raise ValueError(
                f"Unsupported CPU NTT input dtype {spec.dtype!r}."
            ) from ex
        if tensor.dtype != expected_dtype:
            raise ValueError(
                f"CPU NTT tensor input must have dtype {expected_dtype}, "
                f"got {tensor.dtype}."
            )
        if len(spec.shape) != len(spec.strides):
            raise ValueError(
                f"CPU NTT input shape rank {len(spec.shape)} does not match "
                f"stride rank {len(spec.strides)}."
            )
        if any(dim < 0 for dim in spec.shape) or any(
            stride < 0 for stride in spec.strides
        ):
            raise ValueError("CPU NTT input shape and strides must be non-negative.")
        lane_count = 1
        for lane in spec.lane_shape:
            if lane <= 0:
                raise ValueError(
                    f"CPU NTT input lane dimensions must be positive, got {spec.lane_shape}."
                )
            lane_count *= lane
        required_elements = 0
        if all(dim > 0 for dim in spec.shape):
            required_elements = 1 + sum(
                (dim - 1) * stride
                for dim, stride in zip(spec.shape, spec.strides)
            )
            required_elements *= lane_count
        required_bytes = required_elements * tensor.element_size()
        available_bytes = tensor.numel() * tensor.element_size()
        if required_bytes > available_bytes:
            raise ValueError(
                f"CPU NTT input ABI requires {required_bytes} bytes, but the "
                f"runtime tensor exposes only {available_bytes}."
            )
        return _PreparedInput(
            data=tensor.data_ptr(),
            size=available_bytes,
            shape=tuple(spec.shape),
            strides=tuple(spec.strides),
            keepalive=(tensor,),
        )

    @staticmethod
    def _get_object_field(value: Any, name: str, default: Any = None) -> Any:
        field = value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)
        return field() if callable(field) else field

    @classmethod
    def _is_paged_attention_kv_cache(cls, value: Any) -> bool:
        return all(
            cls._get_object_field(value, name) is not None
            for name in ("seq_lens", "block_table", "slot_mapping", "kv_caches")
        ) and (
            cls._get_object_field(value, "query_start_loc") is not None
            or cls._get_object_field(value, "context_lens") is not None
        )

    @staticmethod
    def _to_cpu_metadata_tensor(value: Any, dtype: torch.dtype, name: str) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value
        else:
            if hasattr(value, "to_runtime_tensor"):
                value = value.to_runtime_tensor()
            if hasattr(value, "to_numpy"):
                value = value.to_numpy()
            elif hasattr(value, "numpy"):
                value = value.numpy()
            try:
                tensor = torch.as_tensor(value)
            except (TypeError, ValueError) as ex:
                raise TypeError(
                    f"Paged-attention field {name!r} cannot be converted to a tensor."
                ) from ex
        return tensor.to(device="cpu", dtype=dtype).contiguous()

    @classmethod
    def _prepare_paged_attention_kv_cache(cls, value: Any) -> _PreparedInput:
        seq_lens = cls._to_cpu_metadata_tensor(
            cls._get_object_field(value, "seq_lens"), torch.int64, "seq_lens"
        )
        if seq_lens.ndim != 1:
            raise ValueError(
                f"Paged-attention seq_lens must be rank one, got {tuple(seq_lens.shape)}."
            )
        num_seqs = int(seq_lens.numel())

        context_value = cls._get_object_field(value, "context_lens")
        if context_value is None:
            query_start_loc = cls._to_cpu_metadata_tensor(
                cls._get_object_field(value, "query_start_loc"),
                torch.int64,
                "query_start_loc",
            )
            if query_start_loc.ndim != 1 or query_start_loc.numel() != num_seqs + 1:
                raise ValueError(
                    "Paged-attention query_start_loc must be rank one and contain "
                    "exactly num_seqs + 1 elements."
                )
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            context_lens = (seq_lens - query_lens).contiguous()
            num_tokens = int(query_start_loc[-1].item())
        else:
            context_lens = cls._to_cpu_metadata_tensor(
                context_value, torch.int64, "context_lens"
            )
            if context_lens.ndim != 1 or context_lens.numel() != num_seqs:
                raise ValueError(
                    "Paged-attention context_lens must be rank one and match seq_lens."
                )
            explicit_num_tokens = cls._get_object_field(value, "num_tokens")
            num_tokens = (
                int(explicit_num_tokens)
                if explicit_num_tokens is not None
                else int((seq_lens - context_lens).sum().item())
            )
        if torch.any(context_lens < 0) or torch.any(seq_lens < context_lens):
            raise ValueError("Paged-attention sequence metadata is inconsistent.")

        block_table = cls._to_cpu_metadata_tensor(
            cls._get_object_field(value, "block_table"),
            torch.int64,
            "block_table",
        )
        slot_mapping = cls._to_cpu_metadata_tensor(
            cls._get_object_field(value, "slot_mapping"),
            torch.int64,
            "slot_mapping",
        )
        if block_table.ndim != 3:
            raise ValueError(
                f"Paged-attention block_table must be rank three, got {tuple(block_table.shape)}."
            )
        if slot_mapping.ndim != 2:
            raise ValueError(
                f"Paged-attention slot_mapping must be rank two, got {tuple(slot_mapping.shape)}."
            )

        addresses, address_keepalive = cls._extract_kv_cache_addresses(
            cls._get_object_field(value, "kv_caches")
        )
        if not addresses:
            raise ValueError("Paged-attention KV cache exposes no backing addresses.")
        if len(addresses) > 128:
            raise ValueError(
                f"Paged-attention KV cache exposes {len(addresses)} addresses; maximum is 128."
            )

        desc = _PagedAttentionKVCacheDesc()
        desc.num_seqs = num_seqs
        desc.num_tokens = num_tokens
        desc.context_lens = ctypes.cast(
            context_lens.data_ptr(), ctypes.POINTER(ctypes.c_int64)
        )
        desc.context_lens_size = context_lens.numel()
        desc.seq_lens = ctypes.cast(
            seq_lens.data_ptr(), ctypes.POINTER(ctypes.c_int64)
        )
        desc.seq_lens_size = seq_lens.numel()
        desc.block_table = ctypes.cast(
            block_table.data_ptr(), ctypes.POINTER(ctypes.c_int64)
        )
        desc.block_table_shape[:] = tuple(block_table.shape)
        desc.slot_mapping = ctypes.cast(
            slot_mapping.data_ptr(), ctypes.POINTER(ctypes.c_int64)
        )
        desc.slot_mapping_shape[:] = tuple(slot_mapping.shape)
        for index, address in enumerate(addresses):
            desc.kv_cache_addrs[index] = address

        return _PreparedInput(
            data=ctypes.addressof(desc),
            size=ctypes.sizeof(desc),
            shape=(),
            strides=(),
            keepalive=(
                value,
                desc,
                context_lens,
                seq_lens,
                block_table,
                slot_mapping,
                *address_keepalive,
            ),
        )

    @classmethod
    def _extract_kv_cache_addresses(
        cls, storage: Any
    ) -> tuple[list[int], tuple[Any, ...]]:
        values = storage if isinstance(storage, (list, tuple)) else (storage,)
        addresses: list[int] = []
        keepalive: list[Any] = []
        for value in values:
            if isinstance(value, torch.Tensor):
                if value.device.type == "cpu" and value.dtype not in (
                    torch.int64,
                    torch.uint64,
                ):
                    tensor = value.contiguous()
                    addresses.append(tensor.data_ptr())
                    keepalive.append(tensor)
                    continue
                if value.device.type != "cpu":
                    raise ValueError(
                        "CPU paged-attention cannot dereference CUDA KV-cache storage."
                    )
                raw = value.contiguous().view(-1).tolist()
                addresses.extend(int(item) for item in raw)
                keepalive.append(value)
                continue

            if hasattr(value, "to_runtime_tensor"):
                value = value.to_runtime_tensor()
            if hasattr(value, "to_numpy"):
                array = value.to_numpy()
            elif hasattr(value, "numpy"):
                array = value.numpy()
            elif isinstance(value, int):
                addresses.append(value)
                continue
            else:
                raise TypeError(
                    "Paged-attention KV-cache storage must expose host tensors "
                    "or an integer address table."
                )
            if getattr(array.dtype, "kind", None) in ("i", "u"):
                addresses.extend(int(item) for item in array.reshape(-1).tolist())
            else:
                contiguous = array if array.flags.c_contiguous else array.copy()
                addresses.append(int(contiguous.ctypes.data))
                keepalive.append(contiguous)
            keepalive.append(value)
        return addresses, tuple(keepalive)

    @staticmethod
    def _make_prepared_descs(inputs: Sequence[_PreparedInput]):
        descs = (_TensorDesc * len(inputs))()
        shapes = []
        strides = []
        for index, value in enumerate(inputs):
            shape = (ctypes.c_size_t * len(value.shape))(*value.shape)
            stride = (ctypes.c_size_t * len(value.strides))(*value.strides)
            shapes.append(shape)
            strides.append(stride)
            descs[index] = _TensorDesc(
                data=value.data,
                size=value.size,
                shape=shape,
                strides=stride,
                rank=len(value.shape),
            )
        return descs, shapes, strides

    @staticmethod
    def _make_descs(
        tensors: Sequence[torch.Tensor], *, require_contiguous: bool = True
    ):
        descs = (_TensorDesc * len(tensors))()
        shapes = []
        strides = []
        for index, tensor in enumerate(tensors):
            if require_contiguous and not tensor.is_contiguous():
                raise ValueError(f"CPU NTT input {index} must be contiguous.")
            shape = (ctypes.c_size_t * tensor.ndim)(*tensor.shape)
            stride = (ctypes.c_size_t * tensor.ndim)(*tensor.stride())
            shapes.append(shape)
            strides.append(stride)
            descs[index] = _TensorDesc(
                data=tensor.data_ptr(),
                size=tensor.numel() * tensor.element_size(),
                shape=shape,
                strides=stride,
                rank=tensor.ndim,
            )
        return descs, shapes, strides

    @staticmethod
    def _view_output(pool: torch.Tensor, spec: CpuOutputSpec) -> torch.Tensor:
        try:
            dtype = _TORCH_DTYPES[spec.dtype]
        except KeyError as ex:
            raise ValueError(f"Unsupported CPU NTT output dtype {spec.dtype!r}.") from ex
        element_size = torch.empty((), dtype=dtype).element_size()
        if spec.offset_bytes % element_size != 0:
            raise ValueError(
                f"CPU NTT output offset {spec.offset_bytes} is not aligned to {element_size}."
            )
        typed = pool[spec.offset_bytes :].view(dtype)
        return torch.as_strided(typed, size=spec.shape, stride=spec.strides)
