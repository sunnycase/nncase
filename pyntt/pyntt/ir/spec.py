"""PyNTT spec objects used by generated model packages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


DimExpr = Union[int, str]


def _as_tuple(value):
    return tuple(value or ())


@dataclass(frozen=True)
class TensorSpec:
    """Tensor contract for generated PyNTT functions and kernels."""

    name: str
    dtype: str
    shape: tuple[DimExpr, ...]
    strides: tuple[DimExpr, ...] | None = None
    role: str = "input"
    device: str = "any"
    layout: str = "contiguous"
    memory: str = "global"

    def __post_init__(self):
        object.__setattr__(self, "shape", _as_tuple(self.shape))
        if self.strides is not None:
            object.__setattr__(self, "strides", _as_tuple(self.strides))


@dataclass(frozen=True)
class TensorResultSpec:
    """Logical tensor result backed by an input or caller-allocated output."""

    tensor: TensorSpec
    source: str
    source_index: int
    offset_bytes: DimExpr = 0

    def __post_init__(self):
        if self.source not in ("input", "output"):
            raise ValueError(
                f"Tensor result source must be 'input' or 'output', got {self.source!r}."
            )
        if self.source_index < 0:
            raise ValueError(
                f"Tensor result source_index must be non-negative, got {self.source_index}."
            )


@dataclass(frozen=True)
class ShapeBinding:
    """Bind one dynamic dimension variable to a runtime input shape axis."""

    name: str
    input_index: int
    axis: int
    min_value: int | None = None
    max_value: int | None = None


@dataclass(frozen=True)
class FunctionSpec:
    """Generated function metadata and runtime contract."""

    name: str
    module_kind: str
    is_entry: bool
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    results: tuple[TensorResultSpec, ...]
    parameters: tuple[str, ...] = ()
    shape_bindings: tuple[ShapeBinding, ...] = ()

    def __post_init__(self):
        inputs = _as_tuple(self.inputs)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", _as_tuple(self.outputs))
        object.__setattr__(self, "results", _as_tuple(self.results))
        object.__setattr__(self, "shape_bindings", _as_tuple(self.shape_bindings))

        parameters = _as_tuple(self.parameters)
        if not parameters and inputs:
            parameters = tuple(tensor.name for tensor in inputs)
        object.__setattr__(self, "parameters", parameters)


@dataclass(frozen=True)
class ModuleSpec:
    """Generated module metadata."""

    name: str
    backend: str
    functions: tuple[FunctionSpec, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "functions", _as_tuple(self.functions))

    @property
    def entry(self) -> FunctionSpec | None:
        """Return the entry function when one is present."""
        for function in self.functions:
            if function.is_entry:
                return function
        return None

    def get_function(self, name: str) -> FunctionSpec:
        """Return a generated function by name."""
        for function in self.functions:
            if function.name == name:
                return function
        raise KeyError(f"Function not found in PyNTT module spec: {name}")
