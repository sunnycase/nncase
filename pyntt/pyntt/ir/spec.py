"""PyNTT spec objects used by generated model packages."""

from __future__ import annotations

from dataclasses import dataclass


def _as_tuple(value):
    return tuple(value or ())


@dataclass(frozen=True)
class TensorSpec:
    """Static tensor contract for generated PyNTT functions and kernels."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    strides: tuple[int, ...] | None = None
    role: str = "input"
    device: str = "any"
    layout: str = "contiguous"
    memory: str = "global"

    def __post_init__(self):
        object.__setattr__(self, "shape", _as_tuple(self.shape))
        if self.strides is not None:
            object.__setattr__(self, "strides", _as_tuple(self.strides))


@dataclass(frozen=True)
class FunctionSpec:
    """Generated function metadata and static runtime contract."""

    name: str
    module_kind: str
    is_entry: bool
    parameters: tuple[str, ...] = ()
    inputs: tuple[TensorSpec, ...] = ()
    outputs: tuple[TensorSpec, ...] = ()

    def __post_init__(self):
        inputs = _as_tuple(self.inputs)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", _as_tuple(self.outputs))

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
