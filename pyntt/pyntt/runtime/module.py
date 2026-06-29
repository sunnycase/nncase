"""PyNTT runtime module."""

from __future__ import annotations

from pyntt.ir import ModuleSpec
from pyntt.runtime.errors import PyNTTSpecError
from pyntt.runtime.tensor import allocate_outputs, validate_inputs


class PyNTTModule:
    """Generated PyNTT model wrapper."""

    def __init__(self, spec: ModuleSpec):
        self.spec = spec

    def __call__(self, *inputs):
        entry = self.spec.entry
        if entry is None:
            raise PyNTTSpecError(
                f"PyNTT module {self.spec.name} does not declare an entry function."
            )

        validate_inputs(entry, inputs)
        outputs = allocate_outputs(entry, inputs)

        if len(outputs) == 1:
            return outputs[0]
        return outputs
