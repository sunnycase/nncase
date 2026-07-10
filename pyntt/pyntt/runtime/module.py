"""PyNTT runtime module."""

from __future__ import annotations

from pyntt.ir import ModuleSpec
from pyntt.runtime.errors import PyNTTSpecError
from pyntt.runtime.tensor import allocate_outputs, materialize_results, resolve_shape_env, validate_inputs


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

        shape_env = resolve_shape_env(entry, inputs)
        validate_inputs(entry, inputs, shape_env)
        outputs = allocate_outputs(entry, inputs, shape_env)
        results = materialize_results(entry, inputs, outputs, shape_env)

        if len(results) == 1:
            return results[0]
        return results
