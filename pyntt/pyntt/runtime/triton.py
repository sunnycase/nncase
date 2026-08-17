"""Triton runtime helpers for generated PyNTT models."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from pyntt.runtime.tensor import dtype_item_size, view_typed_buffer

_TRITON_ALLOCATOR_INSTALLED = False
_VALIDATED_KERNEL_RESOURCES: set[tuple[object, ...]] = set()


class TritonKernelResourceError(RuntimeError):
    """A compiled specialization violates the target resource contract."""


def _validate_execution_model_launch_options(
    launch_options: Mapping[str, object],
    *,
    expected_compute_num_warps: int,
    expected_resident_blocks_per_compute_unit: int,
) -> None:
    requested_num_warps = launch_options.get("num_warps")
    if (
        requested_num_warps is not None
        and int(requested_num_warps) != expected_compute_num_warps
    ):
        raise TritonKernelResourceError(
            f"Triton launch requests {int(requested_num_warps)} compute warps; "
            f"the target execution model requires {expected_compute_num_warps}."
        )

    requested_min_ctas = launch_options.get("min_ctas_per_sm")
    if requested_min_ctas is None:
        raise TritonKernelResourceError(
            "Triton launch does not specify min_ctas_per_sm; the target execution "
            "model requires an explicit resident-block contract."
        )
    if int(requested_min_ctas) != expected_resident_blocks_per_compute_unit:
        raise TritonKernelResourceError(
            f"Triton launch requests min_ctas_per_sm={int(requested_min_ctas)}; "
            "the target execution model requires "
            f"{expected_resident_blocks_per_compute_unit} resident blocks per "
            "compute unit."
        )


class PreparedTritonKernel:
    """One resource-validated Triton specialization ready for direct launch."""

    __slots__ = (
        "kernel_name",
        "parameter_name",
        "parameter_value",
        "_prepared",
        "_runtime_argument_count",
    )

    def __init__(
        self,
        kernel_name: str,
        parameter_name: str,
        parameter_value: int,
        prepared: object,
        runtime_argument_count: int,
    ) -> None:
        self.kernel_name = str(kernel_name)
        self.parameter_name = str(parameter_name)
        self.parameter_value = int(parameter_value)
        self._prepared = prepared
        self._runtime_argument_count = int(runtime_argument_count)

    @property
    def compiled_kernel(self):
        return self._prepared.compiled_kernel

    def launch(self, *kernel_args: object, grid, stream=None) -> None:
        if len(kernel_args) != self._runtime_argument_count:
            raise TypeError(
                f"Prepared PyNTT kernel {self.kernel_name} expects "
                f"{self._runtime_argument_count} runtime arguments before "
                f"{self.parameter_name}, got {len(kernel_args)}."
            )
        self._prepared.launch(*kernel_args, grid=grid, stream=stream)


class TritonTensorDescriptorCache:
    """Materialize reusable host descriptors and per-owner descriptor tables."""

    _DESCRIPTOR_ALIGNMENT_BYTES = 16
    _TENSOR_MAP_ALIGNMENT_BYTES = 128
    _MAX_BLOCK_ELEMENTS = 1_048_576
    _SINGLE_SPEC_FIELDS = frozenset(
        {
            "kind",
            "name",
            "source",
            "offset_bytes",
            "dtype",
            "shape",
            "strides",
            "block_shape",
            "source_shape_axes",
            "padding",
        }
    )
    _TABLE_SPEC_FIELDS = frozenset(
        {
            "kind",
            "name",
            "source",
            "dtype",
            "block_shape",
            "padding",
            "swizzle_mode",
            "entry_size_bytes",
            "entries",
        }
    )
    _TABLE_ENTRY_FIELDS = frozenset(
        {"offset_bytes", "shape", "strides", "source_shape_axes"}
    )
    _TMA_HOST_DTYPE = {
        "uint8": 0,
        "uint16": 1,
        "uint32": 2,
        "int32": 3,
        "uint64": 4,
        "int64": 5,
        "float16": 6,
        "float32": 7,
        "float64": 8,
        "bfloat16": 9,
    }

    def __init__(self) -> None:
        self._entries: dict[str, tuple[tuple[object, ...], object]] = {}

    def materialize_many(
        self,
        kernel_name: str,
        specs: Sequence[Mapping[str, Any]],
        sources: Mapping[str, Any],
    ) -> tuple[object, ...]:
        """Return descriptors in compiler-defined ABI order."""
        descriptors = []
        for spec in specs:
            kind = spec.get("kind")
            expected_fields = (
                self._SINGLE_SPEC_FIELDS
                if kind == "single"
                else self._TABLE_SPEC_FIELDS
                if kind == "table"
                else None
            )
            if expected_fields is None:
                raise ValueError(
                    "PyNTT host tensor descriptor spec kind must be "
                    f"'single' or 'table', got {kind!r}."
                )
            fields = frozenset(spec)
            if fields != expected_fields:
                missing = sorted(expected_fields - fields)
                unexpected = sorted(fields - expected_fields)
                raise ValueError(
                    f"PyNTT host tensor descriptor spec has missing fields "
                    f"{missing} and unexpected fields {unexpected}."
                )
            name = str(spec["name"])
            source_name = str(spec["source"])
            if not name or not source_name:
                raise ValueError(
                    "PyNTT host tensor descriptor name/source must be non-empty."
                )
            try:
                storage = sources[source_name]
            except KeyError as ex:
                raise ValueError(
                    f"PyNTT host tensor descriptor {name!r} references "
                    f"unbound source {source_name!r}."
                ) from ex
            slot = f"{kernel_name}:{name}"
            if kind == "single":
                descriptor = self._materialize(
                    slot,
                    storage,
                    offset_bytes=int(spec["offset_bytes"]),
                    dtype=str(spec["dtype"]),
                    shape=self._resolve_shape(
                        slot,
                        storage,
                        tuple(int(value) for value in spec["shape"]),
                        tuple(
                            tuple(int(axis) for axis in axes)
                            for axes in spec["source_shape_axes"]
                        ),
                    ),
                    strides=tuple(int(value) for value in spec["strides"]),
                    block_shape=tuple(int(value) for value in spec["block_shape"]),
                    padding=str(spec["padding"]),
                )
            else:
                descriptor = self._materialize_table(
                    slot,
                    storage,
                    dtype=str(spec["dtype"]),
                    block_shape=tuple(int(value) for value in spec["block_shape"]),
                    padding=str(spec["padding"]),
                    swizzle_mode=int(spec["swizzle_mode"]),
                    entry_size_bytes=int(spec["entry_size_bytes"]),
                    entries=tuple(spec["entries"]),
                )
            descriptors.append(descriptor)
        return tuple(descriptors)

    @staticmethod
    def _resolve_shape(
        slot: str,
        storage: Any,
        static_shape: tuple[int, ...],
        source_shape_axes: tuple[tuple[int, ...], ...],
    ) -> tuple[int, ...]:
        if len(source_shape_axes) != len(static_shape):
            raise ValueError(
                f"PyNTT host tensor descriptor {slot} static/source-shape "
                f"ranks differ: {len(static_shape)}/{len(source_shape_axes)}."
            )

        storage_shape = tuple(int(extent) for extent in getattr(storage, "shape", ()))
        used_axes: set[int] = set()
        resolved = []
        for descriptor_axis, (static_extent, source_axes) in enumerate(
            zip(static_shape, source_shape_axes)
        ):
            extent = static_extent
            if source_axes:
                extent = 1
                for source_axis in source_axes:
                    if source_axis < 0 or source_axis >= len(storage_shape):
                        raise ValueError(
                            f"PyNTT host tensor descriptor {slot} dimension "
                            f"{descriptor_axis} references source axis {source_axis}, "
                            f"but source rank is {len(storage_shape)}."
                        )
                    if source_axis in used_axes:
                        raise ValueError(
                            f"PyNTT host tensor descriptor {slot} reuses source "
                            f"axis {source_axis}."
                        )
                    used_axes.add(source_axis)
                    extent *= storage_shape[source_axis]
            if extent <= 0:
                raise ValueError(
                    f"PyNTT host tensor descriptor {slot} resolved non-positive "
                    f"extent {extent} for dimension {descriptor_axis}."
                )
            resolved.append(extent)
        return tuple(resolved)

    def _prepare_descriptor_base(
        self,
        slot: str,
        storage: Any,
        *,
        offset_bytes: int,
        dtype: str,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        block_shape: tuple[int, ...],
        padding: str,
    ) -> tuple[object, tuple[object, ...]]:
        if len(shape) == 0 or len(shape) > 5:
            raise ValueError(
                f"PyNTT host tensor descriptor {slot} rank must be in [1, 5], "
                f"got {len(shape)}."
            )
        if len(strides) != len(shape) or len(block_shape) != len(shape):
            raise ValueError(
                f"PyNTT host tensor descriptor {slot} shape/stride/block ranks "
                f"differ: {len(shape)}/{len(strides)}/{len(block_shape)}."
            )
        if any(value <= 0 for value in (*shape, *strides, *block_shape)):
            raise ValueError(
                f"PyNTT host tensor descriptor {slot} dimensions, strides, and "
                "block dimensions must be positive."
            )
        if strides[-1] != 1:
            raise ValueError(
                f"PyNTT host tensor descriptor {slot} requires a contiguous "
                f"last dimension, got stride {strides[-1]}."
            )
        item_size = dtype_item_size(dtype)
        for axis, stride in enumerate(strides[:-1]):
            if (stride * item_size) % self._DESCRIPTOR_ALIGNMENT_BYTES != 0:
                raise ValueError(
                    f"PyNTT host tensor descriptor {slot} stride {stride} on "
                    f"axis {axis} is not {self._DESCRIPTOR_ALIGNMENT_BYTES}-byte "
                    "aligned."
                )
        block_elements = 1
        for axis, extent in enumerate(block_shape):
            if extent & (extent - 1):
                raise ValueError(
                    f"PyNTT host tensor descriptor {slot} block extent {extent} "
                    f"on axis {axis} is not a power of two."
                )
            block_elements *= extent
        if block_elements > self._MAX_BLOCK_ELEMENTS:
            raise ValueError(
                f"PyNTT host tensor descriptor {slot} block has "
                f"{block_elements} elements, exceeding the Triton limit "
                f"{self._MAX_BLOCK_ELEMENTS}."
            )
        if offset_bytes < 0:
            raise ValueError(
                f"PyNTT host tensor descriptor {slot} has negative byte offset "
                f"{offset_bytes}."
            )
        if padding not in ("zero", "nan"):
            raise ValueError(
                f"PyNTT host tensor descriptor {slot} has invalid padding "
                f"{padding!r}."
            )
        if not hasattr(storage, "data_ptr") or not hasattr(storage, "device"):
            raise TypeError(
                f"PyNTT host tensor descriptor {slot} expects tensor storage, "
                f"got {type(storage).__name__}."
            )

        signature: tuple[object, ...] = (
            int(storage.data_ptr()),
            str(storage.device),
            str(getattr(storage, "dtype", "")),
            offset_bytes,
            dtype,
            shape,
            strides,
            block_shape,
            padding,
        )
        span_elements = 1 + sum(
            (extent - 1) * stride for extent, stride in zip(shape, strides)
        )
        size_bytes = span_elements * item_size
        base = view_typed_buffer(storage, offset_bytes, size_bytes, dtype)
        if int(base.data_ptr()) % self._DESCRIPTOR_ALIGNMENT_BYTES != 0:
            raise ValueError(
                f"PyNTT host tensor descriptor {slot} base address after byte "
                f"offset {offset_bytes} is not "
                f"{self._DESCRIPTOR_ALIGNMENT_BYTES}-byte aligned."
            )
        if padding == "nan" and not base.dtype.is_floating_point:
            raise ValueError(
                f"PyNTT host tensor descriptor {slot} cannot use NaN padding "
                f"with dtype {dtype}."
            )
        return base, signature

    def _materialize(
        self,
        slot: str,
        storage: Any,
        *,
        offset_bytes: int,
        dtype: str,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        block_shape: tuple[int, ...],
        padding: str,
    ) -> object:
        base, descriptor_signature = self._prepare_descriptor_base(
            slot,
            storage,
            offset_bytes=offset_bytes,
            dtype=dtype,
            shape=shape,
            strides=strides,
            block_shape=block_shape,
            padding=padding,
        )
        signature = ("single", *descriptor_signature)
        cached = self._entries.get(slot)
        if cached is not None and cached[0] == signature:
            return cached[1]

        from triton.tools.tensor_descriptor import TensorDescriptor

        descriptor = TensorDescriptor(
            base,
            shape=list(shape),
            strides=list(strides),
            block_shape=list(block_shape),
            padding=padding,
        )
        self._entries[slot] = (signature, descriptor)
        return descriptor

    def _materialize_table(
        self,
        slot: str,
        storage: Any,
        *,
        dtype: str,
        block_shape: tuple[int, ...],
        padding: str,
        swizzle_mode: int,
        entry_size_bytes: int,
        entries: tuple[Mapping[str, Any], ...],
    ) -> object:
        if entry_size_bytes != self._TENSOR_MAP_ALIGNMENT_BYTES:
            raise ValueError(
                f"PyNTT tensor-map table {slot} entry size must be "
                f"{self._TENSOR_MAP_ALIGNMENT_BYTES} bytes, got "
                f"{entry_size_bytes}."
            )
        if swizzle_mode < 0 or swizzle_mode > 3:
            raise ValueError(
                f"PyNTT tensor-map table {slot} has invalid swizzle mode "
                f"{swizzle_mode}."
            )
        if not entries:
            raise ValueError(f"PyNTT tensor-map table {slot} cannot be empty.")
        try:
            host_dtype = self._TMA_HOST_DTYPE[dtype]
        except KeyError as ex:
            raise ValueError(
                f"PyNTT tensor-map table {slot} does not support dtype {dtype!r}."
            ) from ex

        prepared_entries = []
        entry_signatures = []
        for index, entry in enumerate(entries):
            fields = frozenset(entry)
            if fields != self._TABLE_ENTRY_FIELDS:
                missing = sorted(self._TABLE_ENTRY_FIELDS - fields)
                unexpected = sorted(fields - self._TABLE_ENTRY_FIELDS)
                raise ValueError(
                    f"PyNTT tensor-map table {slot} entry {index} has missing "
                    f"fields {missing} and unexpected fields {unexpected}."
                )
            entry_slot = f"{slot}[{index}]"
            shape = self._resolve_shape(
                entry_slot,
                storage,
                tuple(int(value) for value in entry["shape"]),
                tuple(
                    tuple(int(axis) for axis in axes)
                    for axes in entry["source_shape_axes"]
                ),
            )
            strides = tuple(int(value) for value in entry["strides"])
            base, entry_signature = self._prepare_descriptor_base(
                entry_slot,
                storage,
                offset_bytes=int(entry["offset_bytes"]),
                dtype=dtype,
                shape=shape,
                strides=strides,
                block_shape=block_shape,
                padding=padding,
            )
            prepared_entries.append((base, shape, strides))
            entry_signatures.append(entry_signature)

        signature = (
            "table",
            dtype,
            block_shape,
            padding,
            swizzle_mode,
            entry_size_bytes,
            tuple(entry_signatures),
        )
        cached = self._entries.get(slot)
        if cached is not None and cached[0] == signature:
            return cached[1]

        device = getattr(storage, "device", None)
        if getattr(device, "type", None) != "cuda":
            raise ValueError(
                f"PyNTT tensor-map table {slot} requires CUDA storage, got "
                f"{device}."
            )

        import torch
        import triton

        encoder = getattr(
            triton.runtime.driver.active.utils, "encode_tma_descriptor", None
        )
        if encoder is None:
            raise RuntimeError(
                "PyNTT tensor-map tables require FlagTree's "
                "encode_tma_descriptor driver API."
            )
        item_size = dtype_item_size(dtype)
        padding_mode = 1 if padding == "nan" else 0
        payload = bytearray()
        for index, (base, shape, strides) in enumerate(prepared_entries):
            encoded = encoder(
                int(base.data_ptr()),
                swizzle_mode,
                item_size,
                host_dtype,
                block_shape,
                shape,
                strides,
                padding_mode,
            )
            if not isinstance(encoded, bytes) or len(encoded) != entry_size_bytes:
                raise RuntimeError(
                    f"FlagTree encoded tensor-map table {slot} entry {index} "
                    f"as {type(encoded).__name__} with length "
                    f"{len(encoded) if isinstance(encoded, bytes) else 'unknown'}; "
                    f"expected {entry_size_bytes} bytes."
                )
            payload.extend(encoded)

        host_table = torch.frombuffer(payload, dtype=torch.uint8)
        table = host_table.to(device=device, non_blocking=False)
        if int(table.data_ptr()) % self._TENSOR_MAP_ALIGNMENT_BYTES != 0:
            raise RuntimeError(
                f"PyNTT tensor-map table {slot} device address is not "
                f"{self._TENSOR_MAP_ALIGNMENT_BYTES}-byte aligned."
            )
        self._entries[slot] = (signature, table)
        return table


def ensure_triton_allocator(device: Optional[object] = None) -> None:
    """Install a default Triton scratch allocator backed by torch tensors."""
    global _TRITON_ALLOCATOR_INSTALLED
    if _TRITON_ALLOCATOR_INSTALLED:
        return

    import torch
    import triton

    allocation_device = torch.device(device) if device is not None else None

    def _alloc(size: int, _alignment: int, _stream: Optional[int]):
        target_device = allocation_device
        if target_device is None:
            if not torch.cuda.is_available():
                raise RuntimeError("Triton scratch allocation requires a CUDA device.")
            target_device = torch.device("cuda", torch.cuda.current_device())
        return torch.empty((size,), dtype=torch.uint8, device=target_device)

    triton.set_allocator(_alloc)
    _TRITON_ALLOCATOR_INSTALLED = True


def validate_triton_kernel_resources(
    kernel,
    *args,
    grid,
    expected_compute_num_warps: int,
    expected_resident_blocks_per_compute_unit: int,
    registers_per_thread_limit: int,
    shared_memory_capacity_bytes: int,
    forbid_spills: bool,
    **kwargs,
) -> None:
    """Compile and validate one specialization before its first launch."""
    _validate_execution_model_launch_options(
        kwargs,
        expected_compute_num_warps=expected_compute_num_warps,
        expected_resident_blocks_per_compute_unit=(
            expected_resident_blocks_per_compute_unit
        ),
    )

    # Compile the exact specialization that the subsequent launch will use.
    # KernelInterface.warmup() replaces tensors with MockTensor values and can
    # therefore drop pointer-alignment attributes from the specialization key.
    compiled = kernel.run(*args, grid=grid, warmup=True, **kwargs)
    key = (
        compiled.hash,
        expected_compute_num_warps,
        expected_resident_blocks_per_compute_unit,
        registers_per_thread_limit,
        shared_memory_capacity_bytes,
        forbid_spills,
    )
    if key in _VALIDATED_KERNEL_RESOURCES:
        return

    compiled._init_handles()
    _validate_compiled_triton_kernel_resources(
        compiled,
        expected_compute_num_warps=expected_compute_num_warps,
        expected_resident_blocks_per_compute_unit=(
            expected_resident_blocks_per_compute_unit
        ),
        registers_per_thread_limit=registers_per_thread_limit,
        shared_memory_capacity_bytes=shared_memory_capacity_bytes,
        forbid_spills=forbid_spills,
    )


def _validate_compiled_triton_kernel_resources(
    compiled,
    *,
    expected_compute_num_warps: int,
    expected_resident_blocks_per_compute_unit: int,
    registers_per_thread_limit: int,
    shared_memory_capacity_bytes: int,
    forbid_spills: bool,
) -> None:
    key = (
        compiled.hash,
        expected_compute_num_warps,
        expected_resident_blocks_per_compute_unit,
        registers_per_thread_limit,
        shared_memory_capacity_bytes,
        forbid_spills,
    )
    if key in _VALIDATED_KERNEL_RESOURCES:
        return

    actual_num_warps = int(compiled.metadata.num_warps)
    # FlagTree reports the physical total after adding and warp-group-aligning
    # explicit warp-specialized workers. The launch option still describes the
    # default compute partition, so additional backend workers are valid.
    if actual_num_warps < expected_compute_num_warps:
        raise TritonKernelResourceError(
            f"Triton kernel {compiled.name} compiled with {actual_num_warps} warps; "
            f"the target execution model requires at least "
            f"{expected_compute_num_warps} compute warps."
        )

    registers_per_thread = int(compiled.n_regs)
    if registers_per_thread > registers_per_thread_limit:
        raise TritonKernelResourceError(
            f"Triton kernel {compiled.name} uses {registers_per_thread} registers "
            f"per thread, exceeding the target limit {registers_per_thread_limit}."
        )

    shared_bytes = int(compiled.metadata.shared)
    if shared_bytes > shared_memory_capacity_bytes:
        raise TritonKernelResourceError(
            f"Triton kernel {compiled.name} uses {shared_bytes} shared-memory bytes, "
            f"exceeding the target limit {shared_memory_capacity_bytes}."
        )

    spill_stores = int(compiled.n_spill_stores)
    spill_loads = int(compiled.n_spill_loads)
    if forbid_spills and (spill_stores != 0 or spill_loads != 0):
        raise TritonKernelResourceError(
            f"Triton kernel {compiled.name} has {spill_stores} spill-store bytes "
            f"and {spill_loads} spill-load bytes with "
            f"n_regs={int(compiled.n_regs)}, "
            f"shared_bytes={shared_bytes}, stack_bytes={int(compiled.n_stack_bytes)}, "
            f"local_bytes={int(compiled.n_local_bytes)}; the target model forbids "
            "register spilling."
        )

    _VALIDATED_KERNEL_RESOURCES.add(key)


def prepare_and_validate_triton_kernel(
    kernel_name: str,
    parameter_name: str,
    candidates,
    *,
    source: str,
    kernel,
    kernel_args: tuple[object, ...],
    dynamic_argument_indices: tuple[int, ...] | None = None,
    grid_for_candidate,
    expected_compute_num_warps: int,
    expected_resident_blocks_per_compute_unit: int,
    registers_per_thread_limit: int,
    shared_memory_capacity_bytes: int,
    forbid_spills: bool,
    **launch_options,
) -> PreparedTritonKernel:
    """Select, validate, and prepare one specialization for repeated launch."""
    from pyntt.runtime.tuning import tuning_parameter_candidates
    from triton.runtime.errors import OutOfResources

    _validate_execution_model_launch_options(
        launch_options,
        expected_compute_num_warps=expected_compute_num_warps,
        expected_resident_blocks_per_compute_unit=(
            expected_resident_blocks_per_compute_unit
        ),
    )

    prepare = getattr(kernel, "prepare", None)
    if not callable(prepare):
        raise TritonKernelResourceError(
            "The installed Triton runtime does not provide JITFunction.prepare(); "
            "PyNTT requires the prepared C-launcher ABI."
        )

    failures = []
    if dynamic_argument_indices is None:
        dynamic_argument_indices = tuple(range(len(kernel_args)))
    else:
        dynamic_argument_indices = tuple(int(index) for index in dynamic_argument_indices)
    if len(set(dynamic_argument_indices)) != len(dynamic_argument_indices):
        raise TritonKernelResourceError(
            f"PyNTT kernel {kernel_name} dynamic argument indices must be unique."
        )
    if any(index < 0 or index >= len(kernel_args) for index in dynamic_argument_indices):
        raise TritonKernelResourceError(
            f"PyNTT kernel {kernel_name} has dynamic argument indices outside "
            f"[0, {len(kernel_args)})."
        )
    ordered_candidates = tuning_parameter_candidates(
        kernel_name, parameter_name, candidates, source=source
    )
    for candidate in ordered_candidates:
        try:
            prepared = prepare(
                *kernel_args,
                candidate,
                grid=grid_for_candidate(candidate),
                dynamic_arg_indices=dynamic_argument_indices,
                trusted_pointer_arguments=True,
                **launch_options,
            )
            _validate_compiled_triton_kernel_resources(
                prepared.compiled_kernel,
                expected_compute_num_warps=expected_compute_num_warps,
                expected_resident_blocks_per_compute_unit=(
                    expected_resident_blocks_per_compute_unit
                ),
                registers_per_thread_limit=registers_per_thread_limit,
                shared_memory_capacity_bytes=shared_memory_capacity_bytes,
                forbid_spills=forbid_spills,
            )
        except (TritonKernelResourceError, OutOfResources) as ex:
            failures.append(f"{candidate}: {ex}")
            continue
        return PreparedTritonKernel(
            kernel_name,
            parameter_name,
            candidate,
            prepared,
            len(dynamic_argument_indices),
        )

    detail = "; ".join(failures)
    raise TritonKernelResourceError(
        f"No resource-feasible candidate for {kernel_name}.{parameter_name} "
        f"from {ordered_candidates}. {detail}"
    )
