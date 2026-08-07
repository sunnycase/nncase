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
    validate_outputs,
)
from pyntt.runtime.triton import TritonTensorDescriptorCache
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
        self.triton_tensor_descriptor_cache = TritonTensorDescriptorCache()
        self._prepared_triton_kernels: dict[tuple[str, str], Any] = {}
        self._launch_resources: dict[tuple[str, str], tuple[Any, ...]] = {}
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

    def run_into(self, outputs, *inputs) -> None:
        """Execute into caller-owned ABI output buffers.

        This API deliberately does not allocate outputs or materialize logical
        result views. The caller owns output lifetime and may reuse the buffers
        after respecting the execution stream's ordering.
        """
        entry = self.spec.entry
        if entry is None:
            raise PyNTTSpecError(
                f"PyNTT module {self.spec.name} does not declare an entry function."
            )

        if not self.loaded:
            self.load()

        inputs = tuple(inputs)
        outputs = tuple(outputs)
        shape_env = resolve_shape_env(entry, inputs)
        validate_inputs(entry, inputs, shape_env)
        validate_outputs(entry, inputs, outputs, shape_env)
        self._run_entry(inputs, list(outputs), shape_env)

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

    def materialize_triton_tensor_descriptors(
        self,
        kernel_name: str,
        specs,
        sources: Mapping[str, Any],
    ):
        return self.triton_tensor_descriptor_cache.materialize_many(
            kernel_name, specs, sources
        )

    @staticmethod
    def _launch_cache_key(kernel_name: str, device: Any) -> tuple[str, str]:
        return str(kernel_name), str(device)

    def lookup_prepared_triton_kernel(self, kernel_name: str, device: Any):
        """Return the device-bound launch plan for a generated top kernel."""
        return self._prepared_triton_kernels.get(
            self._launch_cache_key(kernel_name, device)
        )

    def store_prepared_triton_kernel(
        self, kernel_name: str, device: Any, prepared
    ) -> None:
        """Publish one immutable prepared specialization."""
        key = self._launch_cache_key(kernel_name, device)
        existing = self._prepared_triton_kernels.get(key)
        if existing is not None and existing is not prepared:
            raise RuntimeError(
                f"PyNTT kernel {key[0]} on {key[1]} already has a different "
                "prepared specialization."
            )
        self._prepared_triton_kernels[key] = prepared

    def lookup_launch_resources(self, kernel_name: str, device: Any):
        """Return persistent workspace/rdata/descriptor bindings for a launch."""
        return self._launch_resources.get(self._launch_cache_key(kernel_name, device))

    def store_launch_resources(
        self, kernel_name: str, device: Any, resources: tuple[Any, ...]
    ) -> None:
        """Retain immutable launch bindings and their backing tensor lifetimes."""
        key = self._launch_cache_key(kernel_name, device)
        existing = self._launch_resources.get(key)
        if existing is not None and existing is not resources:
            raise RuntimeError(
                f"PyNTT kernel {key[0]} on {key[1]} already has different "
                "persistent launch resources."
            )
        self._launch_resources[key] = resources

    def _run_entry(self, inputs: tuple[Any, ...], outputs: list[Any], shape_env: dict[str, int]) -> None:
        # Base interpreter keeps PyNTTModule-compatible behavior for tests and
        # for metadata-only packages. Generated model.py overrides this method.
        return None
