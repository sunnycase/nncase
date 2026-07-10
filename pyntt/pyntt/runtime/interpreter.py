"""Stateful PyNTT interpreter for generated model packages."""

from __future__ import annotations

from typing import Any, Mapping

from pyntt.ir import ModuleSpec
from pyntt.runtime.errors import PyNTTSpecError
from pyntt.runtime.tensor import (
    allocate_outputs,
    materialize_results,
    resolve_shape_env,
    validate_inputs,
)
from pyntt.runtime.workspace import RDataCache, WorkspacePool


class PyNTTInterpreter:
    """Runtime owner for generated PyNTT state.

    Generated packages provide the top-kernel launch code by overriding
    ``_run_entry``. The interpreter owns state that should live across runs:
    readonly data, workspace buffers, and any future executable/tuning caches.
    """

    def __init__(self, spec: ModuleSpec, rdata_bundles: Mapping[str, Mapping[str, Any]] | None = None):
        self.spec = spec
        self.rdata_bundles = dict(rdata_bundles or {})
        self.workspace_pool = WorkspacePool()
        self.rdata_cache = RDataCache()
        self.loaded = False

    def load(self, device: Any | None = None):
        """Load package-level state and optionally materialize rdata on a device."""
        for bundle in self.rdata_bundles.values():
            self.rdata_cache.prepare_bundle(dict(bundle))
            if device is not None:
                self.rdata_cache.materialize_bundle((), dict(bundle), device=device)
        self.loaded = True
        return self

    def run(self, *inputs):
        """Validate inputs, allocate outputs, and execute the generated entry."""
        entry = self.spec.entry
        if entry is None:
            raise PyNTTSpecError(
                f"PyNTT module {self.spec.name} does not declare an entry function."
            )

        if not self.loaded:
            self.load()

        shape_env = resolve_shape_env(entry, inputs)
        validate_inputs(entry, inputs, shape_env)
        outputs = list(allocate_outputs(entry, inputs, shape_env))
        self._run_entry(inputs, outputs, shape_env)
        results = materialize_results(entry, inputs, outputs, shape_env)

        if len(results) == 1:
            return results[0]
        return results

    def __call__(self, *inputs):
        return self.run(*inputs)

    def allocate_workspace(self, inputs: tuple[Any, ...], key: str, elements: int, dtype: str):
        return self.workspace_pool.allocate(inputs, key, elements, dtype)

    def materialize_rdata_bundle(self, inputs: tuple[Any, ...], name: str):
        try:
            bundle = self.rdata_bundles[name]
        except KeyError as ex:
            raise PyNTTSpecError(f"PyNTT rdata bundle {name!r} was not found.") from ex
        return self.rdata_cache.materialize_bundle(inputs, dict(bundle))

    def _run_entry(self, inputs: tuple[Any, ...], outputs: list[Any], shape_env: dict[str, int]) -> None:
        # Base interpreter keeps PyNTTModule-compatible behavior for tests and
        # for metadata-only packages. Generated model.py overrides this method.
        return None
