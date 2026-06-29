"""Triton backend marker for generated PyNTT models."""

from pyntt.runtime.errors import PyNTTBackendError


class TritonBackend:
    """Marker object for the Triton backend.

    Generated PyNTT models launch generated Triton top kernels directly from
    their `model.py`; they do not use runtime kernel dispatch.
    """

    def run(self, *args, **kwargs) -> None:
        raise PyNTTBackendError(
            "Generated PyNTT models launch Triton top kernels directly; "
            "runtime backend dispatch is not used."
        )
