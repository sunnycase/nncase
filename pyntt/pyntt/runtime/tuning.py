"""Runtime selection helpers for PyNTT tunable parameters."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence

from pyntt.runtime.errors import PyNTTSpecError


def select_tuning_parameter(
    kernel_name: str,
    parameter_name: str,
    candidates: Sequence[int],
    *,
    source: str,
) -> int:
    """Select a tunable parameter value for a generated kernel launch."""
    if not candidates:
        raise PyNTTSpecError(
            f"PyNTT tuning parameter {kernel_name}.{parameter_name} has no candidates."
        )

    normalized_candidates = tuple(int(candidate) for candidate in candidates)
    override_name = _override_env_name(kernel_name, parameter_name)
    override = os.environ.get(override_name)
    if override is not None:
        try:
            value = int(override)
        except ValueError as ex:
            raise PyNTTSpecError(
                f"{override_name} must be an integer candidate for "
                f"{kernel_name}.{parameter_name}, got {override!r}."
            ) from ex

        if value not in normalized_candidates:
            raise PyNTTSpecError(
                f"{override_name}={value} is not in the candidate set "
                f"{normalized_candidates} for {kernel_name}.{parameter_name}."
            )

        return value

    if source not in ("search_space", "autotune", "auto_tiling"):
        raise PyNTTSpecError(
            f"Unsupported PyNTT tuning source for {kernel_name}.{parameter_name}: {source}."
        )

    return normalized_candidates[-1]


def _override_env_name(kernel_name: str, parameter_name: str) -> str:
    key = re.sub(r"[^0-9A-Za-z]+", "_", f"{kernel_name}_{parameter_name}").upper()
    return f"PYNTT_TUNE_{key}"
