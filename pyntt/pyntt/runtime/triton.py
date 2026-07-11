"""Triton runtime helpers for generated PyNTT models."""

from __future__ import annotations

from typing import Optional

_TRITON_ALLOCATOR_INSTALLED = False
_VALIDATED_KERNEL_RESOURCES: set[tuple[object, ...]] = set()
_SELECTED_TUNING_PARAMETERS: dict[tuple[object, ...], int] = {}


class TritonKernelResourceError(RuntimeError):
    """A compiled specialization violates the target resource contract."""


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
    expected_num_warps: int,
    worker_width: int,
    register_capacity_bytes: int,
    shared_memory_capacity_bytes: int,
    forbid_spills: bool,
    **kwargs,
) -> None:
    """Compile and validate one specialization before its first launch."""
    compiled = kernel.warmup(*args, grid=grid, **kwargs)
    key = (
        compiled.hash,
        expected_num_warps,
        worker_width,
        register_capacity_bytes,
        shared_memory_capacity_bytes,
        forbid_spills,
    )
    if key in _VALIDATED_KERNEL_RESOURCES:
        return

    compiled._init_handles()
    actual_num_warps = int(compiled.metadata.num_warps)
    if actual_num_warps != expected_num_warps:
        raise TritonKernelResourceError(
            f"Triton kernel {compiled.name} compiled with {actual_num_warps} warps; "
            f"the target execution model requires {expected_num_warps}."
        )

    shared_bytes = int(compiled.metadata.shared)
    if shared_bytes > shared_memory_capacity_bytes:
        raise TritonKernelResourceError(
            f"Triton kernel {compiled.name} uses {shared_bytes} shared-memory bytes, "
            f"exceeding the target limit {shared_memory_capacity_bytes}."
        )

    register_bytes = (
        int(compiled.n_regs) * expected_num_warps * worker_width * 4
    )
    if register_bytes > register_capacity_bytes:
        raise TritonKernelResourceError(
            f"Triton kernel {compiled.name} uses {register_bytes} register bytes per block, "
            f"exceeding the target limit {register_capacity_bytes}."
        )

    spill_stores = int(compiled.n_spill_stores)
    spill_loads = int(compiled.n_spill_loads)
    if forbid_spills and (spill_stores != 0 or spill_loads != 0):
        raise TritonKernelResourceError(
            f"Triton kernel {compiled.name} has {spill_stores} spill-store bytes "
            f"and {spill_loads} spill-load bytes with "
            f"n_regs={int(compiled.n_regs)}, register_bytes={register_bytes}, "
            f"shared_bytes={shared_bytes}, stack_bytes={int(compiled.n_stack_bytes)}, "
            f"local_bytes={int(compiled.n_local_bytes)}; the target model forbids "
            "register spilling."
        )

    _VALIDATED_KERNEL_RESOURCES.add(key)


def select_and_validate_triton_tuning_parameter(
    kernel_name: str,
    parameter_name: str,
    candidates,
    *,
    source: str,
    kernel,
    kernel_args: tuple[object, ...],
    grid_for_candidate,
    expected_num_warps: int,
    worker_width: int,
    register_capacity_bytes: int,
    shared_memory_capacity_bytes: int,
    forbid_spills: bool,
    **launch_options,
) -> int:
    """Select the highest-priority specialization satisfying target resources."""
    from pyntt.runtime.tuning import tuning_parameter_candidates
    from triton.runtime.errors import OutOfResources

    failures = []
    ordered_candidates = tuning_parameter_candidates(
        kernel_name, parameter_name, candidates, source=source
    )
    selection_key = (
        kernel,
        kernel_name,
        parameter_name,
        ordered_candidates,
        source,
        tuple(_specialization_signature(arg) for arg in kernel_args),
        tuple(
            sorted(
                (name, _specialization_signature(value))
                for name, value in launch_options.items()
            )
        ),
        expected_num_warps,
        worker_width,
        register_capacity_bytes,
        shared_memory_capacity_bytes,
        forbid_spills,
    )
    selected = _SELECTED_TUNING_PARAMETERS.get(selection_key)
    if selected is not None:
        return selected

    for candidate in ordered_candidates:
        try:
            validate_triton_kernel_resources(
                kernel,
                *kernel_args,
                candidate,
                grid=grid_for_candidate(candidate),
                expected_num_warps=expected_num_warps,
                worker_width=worker_width,
                register_capacity_bytes=register_capacity_bytes,
                shared_memory_capacity_bytes=shared_memory_capacity_bytes,
                forbid_spills=forbid_spills,
                **launch_options,
            )
        except (TritonKernelResourceError, OutOfResources) as ex:
            failures.append(f"{candidate}: {ex}")
            continue
        _SELECTED_TUNING_PARAMETERS[selection_key] = candidate
        return candidate

    detail = "; ".join(failures)
    raise TritonKernelResourceError(
        f"No resource-feasible candidate for {kernel_name}.{parameter_name} "
        f"from {ordered_candidates}. {detail}"
    )


def _specialization_signature(value: object) -> tuple[object, ...]:
    """Describe compile-relevant argument properties without retaining buffers."""
    if hasattr(value, "data_ptr") and hasattr(value, "dtype"):
        pointer = int(value.data_ptr())
        device = getattr(value, "device", None)
        return (
            "tensor",
            type(value).__module__,
            type(value).__qualname__,
            str(value.dtype),
            str(device),
            pointer % 16,
        )

    if isinstance(value, tuple):
        return ("tuple", *(_specialization_signature(item) for item in value))

    if isinstance(value, list):
        return ("list", *(_specialization_signature(item) for item in value))

    if isinstance(value, dict):
        return (
            "dict",
            *(
                (str(key), _specialization_signature(item))
                for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            ),
        )

    try:
        hash(value)
    except TypeError:
        return ("object", type(value).__module__, type(value).__qualname__, repr(value))

    return ("value", type(value).__module__, type(value).__qualname__, value)
