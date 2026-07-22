# Copyright 2019-2021 Canaan Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Benchmark a generated one-layer Qwen3 PyNTT package without instrumentation.

The generated Qwen3 test package currently accepts one ``int64`` token per
invocation.  Consequently, a P-token prompt followed by O generated tokens is
measured as P prefill calls and O - 1 decode calls over one shared paged-KV
session.  The last prefill call produces the first generated token.

Use the ``svg`` subcommand to render an existing result, or pass
``--output-svg`` to ``run``.  SVG output is byte-for-byte deterministic for the
same input JSON documents.

The ``run`` command is also the acceptance gate for explicit two-slot
GEMM/GEMV pipelines.  Use ``--allow-ordinary`` only when recording an intentional
ordinary non-pipelined baseline.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from xml.sax.saxutils import escape


SCHEMA_VERSION = 1
BENCHMARK_NAME = "nncase_pyntt_qwen3_0_6b_1layer"
DEFAULT_SCENARIOS = (
    "decode_1x3:1:3",
    "prefill_20x3:20:3",
)
SVG_METRICS = (
    "total_model_cuda",
    "time_to_first_token_cuda",
    "decode_token_cuda",
)


@dataclass(frozen=True)
class Scenario:
    """A fixed-length token generation workload."""

    name: str
    prompt_tokens: int
    output_tokens: int

    @property
    def key(self) -> str:
        return f"prompt={self.prompt_tokens};output={self.output_tokens}"

    @property
    def model_calls(self) -> int:
        return self.prompt_tokens + self.output_tokens - 1


@dataclass(frozen=True)
class StepMeasurement:
    """Timing and optional prediction from one generated-model invocation."""

    cuda_ms: float
    wall_ms: float
    predicted_token: int | None


@dataclass(frozen=True)
class PackageContract:
    """The part of the generated package ABI required by this benchmark."""

    entry_function: str
    input_name: str
    input_dtype: str
    input_shape: tuple[int, ...]
    detected_layer_ids: tuple[str, ...]
    manifest_summary: Mapping[str, Any]
    pipeline_gate_summary: Mapping[str, Any]
    source_file_sha256: Mapping[str, str]
    asset_file_bytes: Mapping[str, int]
    asset_file_sha256: Mapping[str, str]
    code_manifest_sha256: str


def percentile(values: Sequence[float], percent: float) -> float:
    """Return a linearly interpolated percentile."""

    if not values:
        raise ValueError("percentile requires at least one sample")
    if not 0.0 <= percent <= 100.0:
        raise ValueError(f"percent must be in [0, 100], got {percent}")

    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_ms(values: Sequence[float]) -> dict[str, float | int]:
    """Summarize non-empty millisecond samples for stable JSON output."""

    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("latency summary requires at least one sample")
    if any(not math.isfinite(value) or value < 0.0 for value in samples):
        raise ValueError("latency samples must be finite and non-negative")

    return {
        "count": len(samples),
        "mean_ms": statistics.mean(samples),
        "stdev_ms": statistics.pstdev(samples),
        "min_ms": min(samples),
        "p50_ms": percentile(samples, 50.0),
        "p90_ms": percentile(samples, 90.0),
        "p99_ms": percentile(samples, 99.0),
        "max_ms": max(samples),
    }


def parse_scenario(value: str) -> Scenario:
    """Parse NAME:PROMPT_TOKENS:OUTPUT_TOKENS."""

    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"invalid scenario {value!r}; expected NAME:PROMPT_TOKENS:OUTPUT_TOKENS"
        )
    name = parts[0].strip()
    if not name:
        raise ValueError("scenario name must not be empty")
    try:
        prompt_tokens = int(parts[1])
        output_tokens = int(parts[2])
    except ValueError as ex:
        raise ValueError(f"scenario token counts must be integers: {value!r}") from ex
    if prompt_tokens <= 0 or output_tokens <= 0:
        raise ValueError(f"scenario token counts must be positive: {value!r}")
    return Scenario(name, prompt_tokens, output_tokens)


def parse_scenarios(values: Sequence[str] | None) -> list[Scenario]:
    scenarios = [parse_scenario(value) for value in (values or DEFAULT_SCENARIOS)]
    names = [scenario.name for scenario in scenarios]
    if len(names) != len(set(names)):
        raise ValueError("scenario names must be unique")
    keys = [scenario.key for scenario in scenarios]
    if len(keys) != len(set(keys)):
        raise ValueError("scenario prompt/output shapes must be unique")
    return scenarios


def parse_annotations(values: Sequence[str]) -> dict[str, Any]:
    """Parse repeatable KEY=JSON annotations used to identify a candidate."""

    result: dict[str, Any] = {}
    for value in values:
        key, separator, raw = value.partition("=")
        key = key.strip()
        if not separator or not key:
            raise ValueError(f"invalid annotation {value!r}; expected KEY=VALUE")
        if key in result:
            raise ValueError(f"duplicate annotation key {key!r}")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
        result[key] = parsed
    return dict(sorted(result.items()))


def drive_token_by_token(
    prompt_token_ids: Sequence[int],
    output_tokens: int,
    schedule_one_token: Callable[[], Any],
    invoke_one_token: Callable[[int, Any, bool], StepMeasurement],
) -> dict[str, Any]:
    """Drive one request according to the static single-token model ABI.

    ``schedule_one_token`` must advance one existing paged-KV session by one
    token.  ``invoke_one_token`` receives the token, scheduled KV object, and a
    flag indicating whether its prediction is needed to continue generation.
    """

    if not prompt_token_ids:
        raise ValueError("prompt_token_ids must not be empty")
    if output_tokens <= 0:
        raise ValueError("output_tokens must be positive")

    prefill_cuda_ms: list[float] = []
    prefill_wall_ms: list[float] = []
    decode_cuda_ms: list[float] = []
    decode_wall_ms: list[float] = []

    first_generated_token: int | None = None
    for index, prompt_token in enumerate(prompt_token_ids):
        needs_prediction = index == len(prompt_token_ids) - 1
        kv_cache = schedule_one_token()
        measurement = invoke_one_token(int(prompt_token), kv_cache, needs_prediction)
        prefill_cuda_ms.append(float(measurement.cuda_ms))
        prefill_wall_ms.append(float(measurement.wall_ms))
        if needs_prediction:
            first_generated_token = measurement.predicted_token

    if first_generated_token is None:
        raise RuntimeError("the final prefill call did not produce a token")

    generated_token_ids = [int(first_generated_token)]
    while len(generated_token_ids) < output_tokens:
        kv_cache = schedule_one_token()
        measurement = invoke_one_token(generated_token_ids[-1], kv_cache, True)
        if measurement.predicted_token is None:
            raise RuntimeError("a decode call did not produce a token")
        decode_cuda_ms.append(float(measurement.cuda_ms))
        decode_wall_ms.append(float(measurement.wall_ms))
        generated_token_ids.append(int(measurement.predicted_token))

    all_cuda_ms = prefill_cuda_ms + decode_cuda_ms
    all_wall_ms = prefill_wall_ms + decode_wall_ms
    for value in all_cuda_ms + all_wall_ms:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("step latency must be finite and non-negative")

    return {
        "prefill_cuda_ms": prefill_cuda_ms,
        "prefill_wall_ms": prefill_wall_ms,
        "decode_cuda_ms": decode_cuda_ms,
        "decode_wall_ms": decode_wall_ms,
        "total_model_cuda_ms": sum(all_cuda_ms),
        "total_model_wall_ms": sum(all_wall_ms),
        "generated_token_ids": generated_token_ids,
    }


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as ex:
        raise FileNotFoundError(
            f"required generated-package file not found: {path}"
        ) from ex
    except json.JSONDecodeError as ex:
        raise ValueError(f"invalid JSON in generated-package file {path}: {ex}") from ex


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_named_values(value: Any, key: str) -> list[Any]:
    result: list[Any] = []
    if isinstance(value, Mapping):
        for item_key, item_value in value.items():
            if item_key == key:
                result.append(item_value)
            result.extend(_collect_named_values(item_value, key))
    elif isinstance(value, list):
        for item in value:
            result.extend(_collect_named_values(item, key))
    return result


_MATRIX_OPERATION_MARKERS = (
    "matmul",
    "gemm",
    "gemv",
    "qkvparallellinear",
)
_CP_ASYNC_PIPELINE_TEMPLATE = "triton.loop.cp_async.n2.v1"


def _normalized_operation_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _is_matrix_render_kernel(kernel: Mapping[str, Any]) -> bool:
    metadata = kernel.get("metadata")
    if not isinstance(metadata, Mapping):
        return False
    candidates: list[Any] = [metadata.get("name"), metadata.get("op_kind")]
    attrs = metadata.get("attrs")
    if isinstance(attrs, Mapping):
        candidates.append(attrs.get("op"))
        operations = attrs.get("ops")
        if isinstance(operations, list):
            candidates.extend(operations)
        microkernels = attrs.get("block_microkernels")
        if isinstance(microkernels, list):
            for microkernel in microkernels:
                if isinstance(microkernel, Mapping):
                    candidates.extend(
                        (microkernel.get("family"), microkernel.get("helper"))
                    )
    return any(
        marker in _normalized_operation_name(candidate)
        for candidate in candidates
        if candidate is not None
        for marker in _MATRIX_OPERATION_MARKERS
    )


def _call_name(function: ast.expr) -> str:
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        owner = _call_name(function.value)
        return f"{owner}.{function.attr}" if owner else function.attr
    return ""


def _constant_int(argument: ast.expr) -> int | None:
    if (
        isinstance(argument, ast.Constant)
        and isinstance(argument.value, int)
        and not isinstance(argument.value, bool)
    ):
        return int(argument.value)
    return None


def _pipeline_source_evidence(nodes: Iterable[ast.AST]) -> dict[str, bool]:
    """Inspect the functions statically reachable from one exact kernel."""

    evidence = {
        "async_copy": False,
        "async_commit_group": False,
        "async_wait_group_1": False,
        "async_wait_group_0": False,
        "staged_slot": False,
    }
    for node in nodes:
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            name = _call_name(child.func)
            if name == "tle.gpu.copy":
                evidence["async_copy"] |= any(
                    keyword.arg == "is_async"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                    for keyword in child.keywords
                )
            elif name == "tle.gpu.async_commit_group":
                evidence["async_commit_group"] = True
            elif name == "tle.gpu.async_wait_group":
                pending = _constant_int(child.args[0]) if child.args else None
                if pending is None:
                    for keyword in child.keywords:
                        if keyword.arg in {"max_pending", "num"}:
                            pending = _constant_int(keyword.value)
                            break
                if pending == 1:
                    evidence["async_wait_group_1"] = True
                elif pending == 0:
                    evidence["async_wait_group_0"] = True
            if isinstance(child.func, ast.Attribute) and child.func.attr == "slot":
                evidence["staged_slot"] = True
    return evidence


def _inspect_pipeline_source(
    source: str, source_path: Path, kernel_symbols: Iterable[str]
) -> dict[str, dict[str, bool]]:
    """Return evidence from each exact matrix kernel's reachable call graph."""

    try:
        tree = ast.parse(source, filename=str(source_path))
    except SyntaxError as ex:
        raise ValueError(
            f"generated source is not valid Python and cannot satisfy the pipeline "
            f"gate: {source_path}: {ex}"
        ) from ex

    forbidden: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name == "tle.pipe":
            forbidden.append("tle.pipe")
        if "warp_specialize" in name or any(
            keyword.arg == "warp_specialize" for keyword in node.keywords
        ):
            forbidden.append("warp_specialize")
        if name == "tl.range" and any(
            keyword.arg == "num_stages" for keyword in node.keywords
        ):
            forbidden.append("tl.range(..., num_stages=...)")

    if forbidden:
        raise ValueError(
            "generated source uses forbidden pipeline constructs: "
            + ", ".join(sorted(set(forbidden)))
        )

    functions: dict[str, list[ast.AST]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.setdefault(node.name, []).append(node)

    result: dict[str, dict[str, bool]] = {}
    for symbol in kernel_symbols:
        matches = functions.get(symbol, [])
        if len(matches) != 1:
            raise ValueError(
                "generated source must define exactly one top-level function for "
                f"matrix kernel symbol {symbol!r}, found {len(matches)} in {source_path}"
            )
        reachable_names: set[str] = set()
        pending = [symbol]
        while pending:
            name = pending.pop()
            if name in reachable_names:
                continue
            local_matches = functions.get(name, [])
            if len(local_matches) != 1:
                raise ValueError(
                    f"generated source call graph for matrix kernel {symbol!r} has "
                    f"{len(local_matches)} definitions of reachable function {name!r}"
                )
            reachable_names.add(name)
            for child in ast.walk(local_matches[0]):
                if not isinstance(child, ast.Call):
                    continue
                called_name = _call_name(child.func)
                if called_name in functions and called_name not in reachable_names:
                    pending.append(called_name)
        result[symbol] = _pipeline_source_evidence(
            functions[name][0] for name in sorted(reachable_names)
        )
    return result


def _validate_pyntt_codegen_manifest(manifest: Mapping[str, Any]) -> None:
    """Use the PyNTT reader as the single owner of the manifest contract."""

    package_root = Path(__file__).resolve().parents[1] / "pyntt"
    package_root_text = str(package_root)
    if package_root_text not in sys.path:
        sys.path.insert(0, package_root_text)

    from pyntt.codegen.render import validate_manifest

    validate_manifest(manifest)


def _required_string(value: Mapping[str, Any], field: str, path: str) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{path}.{field} must be a non-empty string")
    return item


def _physical_capacity_bytes(
    ranges: Iterable[tuple[str, int, int]],
) -> int:
    return sum(size for _, _, size in set(ranges))


def _pipeline_owners(
    kernel: Mapping[str, Any], kernel_path: str, kernel_name: str
) -> Iterable[tuple[str, str, str, Mapping[str, Any]]]:
    yield "render_kernel", kernel_name, kernel_path, kernel
    for index, device_function in enumerate(kernel["device_functions"]):
        device_path = f"{kernel_path}.device_functions[{index}]"
        yield (
            "device_function",
            _required_string(device_function, "name", device_path),
            device_path,
            device_function,
        )


def _validate_pipeline_gate(
    manifest: Mapping[str, Any],
    generated_source: str,
    source_path: Path,
    allow_ordinary: bool,
) -> dict[str, Any]:
    _validate_pyntt_codegen_manifest(manifest)

    functions = manifest["functions"]
    matrix_kernels: list[dict[str, Any]] = []
    matrix_kernel_symbols: set[str] = set()
    declared_execution_symbols: dict[str, str] = {}
    declared_async_execution_count = 0
    staged_buffer_count = 0
    maximum_stage_count = 1
    total_physical_capacity_bytes = 0

    for function_index, function in enumerate(functions):
        for kernel_index, kernel in enumerate(function["render_kernels"]):
            kernel_path = (
                f"manifest.functions[{function_index}].render_kernels[{kernel_index}]"
            )
            metadata = kernel["metadata"]
            kernel_name = _required_string(metadata, "name", f"{kernel_path}.metadata")
            is_matrix = _is_matrix_render_kernel(kernel)
            if is_matrix:
                if kernel_name in matrix_kernel_symbols:
                    raise ValueError(
                        "matrix render kernel symbols must be unique for compiled "
                        f"artifact correlation, duplicate {kernel_name!r}"
                    )
                matrix_kernel_symbols.add(kernel_name)

            execution_evidence: list[dict[str, Any]] = []
            kernel_physical_ranges: set[tuple[str, int, int]] = set()
            for owner_kind, owner_name, owner_path, owner in _pipeline_owners(
                kernel, kernel_path, kernel_name
            ):
                for execution_index, execution in enumerate(
                    owner["pipeline_executions"]
                ):
                    execution_path = (
                        f"{owner_path}.pipeline_executions[{execution_index}]"
                    )
                    region_id = _required_string(
                        execution, "region_id", execution_path
                    )
                    previous_symbol = declared_execution_symbols.get(region_id)
                    if previous_symbol is not None:
                        raise ValueError(
                            "pipeline execution region_id values must be globally "
                            "unique for execution-to-artifact correlation; "
                            f"{region_id!r} occurs in both {previous_symbol!r} and "
                            f"{kernel_name!r}"
                        )
                    declared_execution_symbols[region_id] = kernel_name

                    channels = execution["channels"]
                    execution_buffer_count = len(channels)
                    execution_stage_count = int(execution["stage_count"])
                    for channel in channels:
                        allocation = channel["allocation"]
                        kernel_physical_ranges.add(
                            (
                                str(allocation["arena_id"]),
                                int(allocation["arena_offset_bytes"]),
                                int(allocation["physical_bytes"]),
                            )
                        )
                    partition = _required_string(
                        execution, "partition", execution_path
                    )
                    if partition != "tail":
                        declared_async_execution_count += 1
                    staged_buffer_count += execution_buffer_count
                    maximum_stage_count = max(
                        maximum_stage_count, execution_stage_count
                    )
                    execution_evidence.append(
                        {
                            "region_id": region_id,
                            "schedule_id": _required_string(
                                execution, "schedule_id", execution_path
                            ),
                            "template_id": _required_string(
                                execution, "template_id", execution_path
                            ),
                            "partition": partition,
                            "owner_kind": owner_kind,
                            "owner_name": owner_name,
                            "staged_buffer_count": execution_buffer_count,
                            "stage_count": execution_stage_count,
                        }
                    )

            if not is_matrix:
                continue
            kernel_capacity_bytes = _physical_capacity_bytes(
                kernel_physical_ranges
            )
            total_physical_capacity_bytes += kernel_capacity_bytes
            matrix_kernels.append(
                {
                    "manifest_path": kernel_path,
                    "kernel_symbol": kernel_name,
                    "pipeline_mode": "cp_async_n2"
                    if any(item["partition"] != "tail" for item in execution_evidence)
                    else "ordinary",
                    "async_execution_ids": sorted(
                        item["region_id"]
                        for item in execution_evidence
                        if item["partition"] != "tail"
                    ),
                    "schedule_ids": sorted(
                        {item["schedule_id"] for item in execution_evidence}
                    ),
                    "pipeline_execution_count": len(execution_evidence),
                    "staged_buffer_count": sum(
                        item["staged_buffer_count"] for item in execution_evidence
                    ),
                    "unique_physical_range_count": len(kernel_physical_ranges),
                    "physical_staged_capacity_bytes": kernel_capacity_bytes,
                    "pipeline_executions": sorted(
                        execution_evidence, key=lambda item: item["region_id"]
                    ),
                }
            )

    if not matrix_kernels:
        raise ValueError(
            "Qwen3 pipeline gate found no GEMM/GEMV matrix render kernels "
            "(MatMul/PackedMatMul/MatMulGlu/QKVParallelLinear families)"
        )

    source_evidence = _inspect_pipeline_source(
        generated_source,
        source_path,
        (kernel["kernel_symbol"] for kernel in matrix_kernels),
    )
    async_matrix_kernels = 0
    async_matrix_execution_count = 0
    for kernel in matrix_kernels:
        symbol = kernel["kernel_symbol"]
        evidence = source_evidence[symbol]
        kernel["source_evidence"] = evidence
        if kernel["async_execution_ids"]:
            async_matrix_kernels += 1
            async_matrix_execution_count += len(kernel["async_execution_ids"])
            missing_evidence = [
                name for name, present in evidence.items() if not present
            ]
            if missing_evidence:
                raise ValueError(
                    "generated source function for declared cp.async matrix kernel "
                    f"{symbol!r} is missing evidence: "
                    + ", ".join(missing_evidence)
                )
        else:
            unexpected_async_evidence = [
                name
                for name in (
                    "async_copy",
                    "async_commit_group",
                    "async_wait_group_1",
                    "async_wait_group_0",
                )
                if evidence[name]
            ]
            if unexpected_async_evidence:
                raise ValueError(
                    "generated source function for ordinary matrix kernel "
                    f"{symbol!r} contains undeclared async pipeline evidence: "
                    + ", ".join(unexpected_async_evidence)
                )

    has_async_matrix_pipeline = async_matrix_execution_count > 0
    if not has_async_matrix_pipeline and not allow_ordinary:
        raise ValueError(
            "Qwen3 pipeline gate requires at least one GEMM/GEMV matrix execution "
            f"using {_CP_ASYNC_PIPELINE_TEMPLATE}"
        )

    return {
        "status": (
            "passed" if has_async_matrix_pipeline else "ordinary_baseline_allowed"
        ),
        "enforced": not allow_ordinary,
        "allow_ordinary": bool(allow_ordinary),
        "manifest_version": 6,
        "matrix_kernel_count": len(matrix_kernels),
        "async_matrix_kernel_count": async_matrix_kernels,
        "ordinary_matrix_kernel_count": len(matrix_kernels) - async_matrix_kernels,
        "async_matrix_execution_count": async_matrix_execution_count,
        "declared_async_execution_count": declared_async_execution_count,
        "staged_buffer_count": staged_buffer_count,
        "maximum_stage_count": maximum_stage_count,
        "physical_staged_capacity_bytes": total_physical_capacity_bytes,
        "matrix_kernels": matrix_kernels,
        "source_evidence_scope": "exact_matrix_kernel_transitive_static_call_graph",
    }


def _count_matches(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.MULTILINE))


def _count_lines_matching_all(text: str, patterns: Sequence[str]) -> int:
    compiled = [re.compile(pattern) for pattern in patterns]
    return sum(
        all(pattern.search(line) is not None for pattern in compiled)
        for line in text.splitlines()
    )


def _count_ptx_wait_then_thread_rendezvous(text: str, max_pending: int) -> int:
    """Count waits followed by a CTA thread rendezvous before shared use.

    PTX ``membar`` and ``fence`` instructions only order memory accesses; they
    do not wait for the other CTA threads that issued their own asynchronous
    copies.  Accept only the ``sync`` forms of the PTX ``bar``/``barrier``
    instruction families, including their canonical CTA/aligned spellings.
    """

    wait = re.compile(rf"\bcp\.async\.wait_group\s+{max_pending}\s*;", re.IGNORECASE)
    thread_rendezvous = re.compile(
        r"^(?:@!?%[A-Za-z0-9_$]+\s+)?"
        r"(?:bar|barrier)(?:\.cta)?\.sync(?:\.aligned)?\b",
        re.IGNORECASE,
    )
    boundary = re.compile(
        r"\b(?:cp\.async\.(?:commit_group|wait_group|ca|cg)|"
        r"ld(?:matrix)?[^;\n]*\.shared|mma(?:\.sync)?|call(?:\.uni)?|ret)\b",
        re.IGNORECASE,
    )
    lines = text.splitlines()
    count = 0
    for index, line in enumerate(lines):
        if wait.search(line) is None:
            continue
        for following in lines[index + 1 : index + 65]:
            instruction = following.split("//", 1)[0].strip()
            if not instruction or instruction.startswith((".", "$")):
                continue
            if thread_rendezvous.search(instruction):
                count += 1
                break
            if boundary.search(instruction):
                break
    return count


def _compiled_artifact_evidence(
    cache_root: Path,
    metadata_path: Path,
    expected_symbol: str,
    expected_async_execution_count: int,
) -> dict[str, Any]:
    if expected_async_execution_count < 1:
        raise ValueError(
            "compiled async artifact requires at least one async execution"
        )
    for path in (
        metadata_path,
        metadata_path.with_suffix(".ttgir"),
        metadata_path.with_suffix(".ptx"),
    ):
        if path.is_symlink():
            raise ValueError(
                f"compiled artifact gate rejects symlinks in the isolated cache: {path}"
            )
        try:
            path.resolve(strict=True).relative_to(cache_root)
        except (FileNotFoundError, ValueError) as ex:
            raise ValueError(
                f"compiled artifact gate requires {path.name} inside isolated cache "
                f"{cache_root}"
            ) from ex

    metadata = _read_json(metadata_path)
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Triton kernel metadata must be an object: {metadata_path}")
    actual_symbol = metadata.get("name")
    if actual_symbol != expected_symbol:
        raise ValueError(
            f"Triton artifact {metadata_path} declares kernel symbol "
            f"{actual_symbol!r}, expected {expected_symbol!r}"
        )

    ttgir_path = metadata_path.with_suffix(".ttgir")
    ptx_path = metadata_path.with_suffix(".ptx")
    try:
        ttgir = ttgir_path.read_text(encoding="utf-8")
        ptx = ptx_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as ex:
        raise ValueError(f"compiled text artifact is not UTF-8: {ex}") from ex

    ttgir_evidence = {
        "kernel_symbol_definition_count": _count_matches(
            rf"\btt\.func\s+public\s+@{re.escape(expected_symbol)}\b", ttgir
        ),
        "async_copy_global_to_local_count": _count_matches(
            r"\bttg\.async_copy_global_to_local\b", ttgir
        ),
        "required_async_copy_marker_count": _count_lines_matching_all(
            ttgir,
            (
                r"\bttg\.async_copy_global_to_local\b",
                r"\btle\.required_async_copy\b",
            ),
        ),
        "async_commit_group_count": _count_matches(
            r"\bttg\.async_commit_group\b", ttgir
        ),
        "async_wait_group_1_count": _count_matches(
            r"\bttg\.async_wait\b[^\n]*\bnum\s*=\s*1\s*:\s*i32", ttgir
        ),
        "async_wait_group_0_count": _count_matches(
            r"\bttg\.async_wait\b[^\n]*\bnum\s*=\s*0\s*:\s*i32", ttgir
        ),
        "explicit_async_wait_marker_count": _count_matches(
            r"\btle\.explicit_async_wait\b", ttgir
        ),
        "explicit_async_wait_group_1_count": _count_lines_matching_all(
            ttgir,
            (
                r"\bttg\.async_wait\b",
                r"\bnum\s*=\s*1\s*:\s*i32",
                r"\btle\.explicit_async_wait\b",
            ),
        ),
        "explicit_async_wait_group_0_count": _count_lines_matching_all(
            ttgir,
            (
                r"\bttg\.async_wait\b",
                r"\bnum\s*=\s*0\s*:\s*i32",
                r"\btle\.explicit_async_wait\b",
            ),
        ),
    }
    ptx_evidence = {
        "kernel_symbol_entry_count": _count_matches(
            rf"(?:\.visible\s+)?\.entry\s+{re.escape(expected_symbol)}\s*\(", ptx
        ),
        "cp_async_copy_count": _count_matches(
            r"^\s*(?:@!?%[A-Za-z0-9_$]+\s+)?"
            r"cp\.async\.(?:ca|cg)\.shared\.global(?:\.[^\s;]+)*\s+",
            ptx,
        ),
        "cp_async_commit_group_count": _count_matches(
            r"\bcp\.async\.commit_group\s*;", ptx
        ),
        "cp_async_wait_group_1_count": _count_matches(
            r"\bcp\.async\.wait_group\s+1\s*;", ptx
        ),
        "cp_async_wait_group_0_count": _count_matches(
            r"\bcp\.async\.wait_group\s+0\s*;", ptx
        ),
        "wait_group_1_then_barrier_count": (
            _count_ptx_wait_then_thread_rendezvous(ptx, 1)
        ),
        "wait_group_0_then_barrier_count": (
            _count_ptx_wait_then_thread_rendezvous(ptx, 0)
        ),
    }

    missing: list[str] = []
    missing.extend(
        f"TTGIR {name}"
        for name in (
            "kernel_symbol_definition_count",
            "async_copy_global_to_local_count",
            "async_commit_group_count",
            "async_wait_group_1_count",
            "async_wait_group_0_count",
            "explicit_async_wait_group_1_count",
            "explicit_async_wait_group_0_count",
        )
        if ttgir_evidence[name] < 1
    )
    if (
        ttgir_evidence["required_async_copy_marker_count"]
        < expected_async_execution_count
    ):
        missing.append(
            "TTGIR required_async_copy_marker_count>="
            f"{expected_async_execution_count}"
        )
    missing.extend(
        f"PTX {name}"
        for name in (
            "kernel_symbol_entry_count",
            "cp_async_copy_count",
            "cp_async_commit_group_count",
            "cp_async_wait_group_1_count",
            "cp_async_wait_group_0_count",
            "wait_group_1_then_barrier_count",
            "wait_group_0_then_barrier_count",
        )
        if ptx_evidence[name] < 1
    )
    if missing:
        raise ValueError(
            f"compiled cp.async gate failed for kernel {expected_symbol!r} in "
            f"{metadata_path.parent}: missing {', '.join(missing)}. Required async "
            "transport must fail lowering instead of being silently downgraded."
        )

    def file_evidence(path: Path) -> dict[str, Any]:
        return {
            "relative_path": path.relative_to(cache_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }

    return {
        "cache_key": metadata_path.parent.relative_to(cache_root).as_posix(),
        "actual_kernel_symbol": actual_symbol,
        "compiler_metadata": {
            "hash": metadata.get("hash"),
            "target": metadata.get("target"),
            "num_warps": metadata.get("num_warps"),
            "num_stages": metadata.get("num_stages"),
            "shared_bytes": metadata.get("shared"),
        },
        "files": {
            "metadata": file_evidence(metadata_path),
            "ttgir": file_evidence(ttgir_path),
            "ptx": file_evidence(ptx_path),
        },
        "ttgir_evidence": ttgir_evidence,
        "ptx_evidence": ptx_evidence,
    }


def inspect_compiled_pipeline_artifacts(
    cache_dir: Path, pipeline_gate: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate actual artifacts for async matrix symbols observed in this run.

    Ordinary matrix kernels are reported but can neither satisfy the gate nor lend
    their source or compiled instructions to an async execution. An async matrix
    symbol that was not launched is reported as unobserved.  At least one
    corresponding symbol must be observed and pass the complete TTGIR/PTX gate.
    """

    cache_root = cache_dir.resolve(strict=True)
    if not cache_root.is_dir():
        raise ValueError(f"Triton cache is not a directory: {cache_root}")
    raw_kernels = pipeline_gate.get("matrix_kernels")
    if not isinstance(raw_kernels, list) or not raw_kernels:
        raise ValueError(
            "compiled artifact gate requires non-empty matrix_kernels evidence "
            "from the generated manifest"
        )

    expectations: dict[str, Mapping[str, Any]] = {}
    async_expectations: dict[str, Mapping[str, Any]] = {}
    declared_execution_symbols: dict[str, str] = {}
    for index, kernel in enumerate(raw_kernels):
        path = f"pipeline_gate.matrix_kernels[{index}]"
        if not isinstance(kernel, Mapping):
            raise ValueError(f"{path} must be an object")
        symbol = _required_string(kernel, "kernel_symbol", path)
        if re.fullmatch(r"[A-Za-z_]\w*", symbol) is None:
            raise ValueError(
                f"{path}.kernel_symbol is not a Python identifier: {symbol!r}"
            )
        if symbol in expectations:
            raise ValueError(f"duplicate matrix kernel symbol {symbol!r}")
        execution_ids = kernel.get("async_execution_ids")
        if not isinstance(execution_ids, list) or any(
            not isinstance(execution_id, str) or not execution_id
            for execution_id in execution_ids
        ):
            raise ValueError(
                f"{path}.async_execution_ids must be an array of non-empty strings"
            )
        if len(execution_ids) != len(set(execution_ids)):
            raise ValueError(f"{path}.async_execution_ids contains duplicates")
        for execution_id in execution_ids:
            previous_symbol = declared_execution_symbols.get(execution_id)
            if previous_symbol is not None:
                raise ValueError(
                    f"async execution {execution_id!r} is ambiguously linked to matrix "
                    f"symbols {previous_symbol!r} and {symbol!r}"
                )
            declared_execution_symbols[execution_id] = symbol
        expectations[symbol] = kernel
        if execution_ids:
            async_expectations[symbol] = kernel

    status = pipeline_gate.get("status")
    if status == "passed" and not async_expectations:
        raise ValueError(
            "compiled artifact gate received passed status without an async matrix execution"
        )
    if status not in {"passed", "ordinary_baseline_allowed"}:
        raise ValueError(f"compiled artifact gate received invalid status {status!r}")
    declared_kernel_count = pipeline_gate.get("async_matrix_kernel_count")
    declared_execution_count = pipeline_gate.get("async_matrix_execution_count")
    if declared_kernel_count != len(async_expectations):
        raise ValueError(
            "pipeline gate async_matrix_kernel_count does not match matrix_kernels: "
            f"{declared_kernel_count!r} != {len(async_expectations)}"
        )
    if declared_execution_count != len(declared_execution_symbols):
        raise ValueError(
            "pipeline gate async_matrix_execution_count does not match "
            f"matrix_kernels: {declared_execution_count!r} != "
            f"{len(declared_execution_symbols)}"
        )

    metadata_by_symbol: dict[str, list[Path]] = {
        symbol: [] for symbol in async_expectations
    }
    expected_names = {f"{symbol}.json": symbol for symbol in async_expectations}
    for metadata_path in sorted(cache_root.rglob("*.json")):
        symbol = expected_names.get(metadata_path.name)
        if symbol is not None:
            metadata_by_symbol[symbol].append(metadata_path)

    kernels: list[dict[str, Any]] = []
    execution_artifact_bindings: list[dict[str, Any]] = []
    artifact_count = 0
    for symbol, expectation in expectations.items():
        execution_ids = list(expectation["async_execution_ids"])
        if not execution_ids:
            kernels.append(
                {
                    **dict(expectation),
                    "artifact_status": "not_applicable_ordinary",
                    "compiled_artifacts": [],
                }
            )
            continue
        metadata_paths = metadata_by_symbol[symbol]
        if not metadata_paths:
            kernels.append(
                {
                    **dict(expectation),
                    "artifact_status": "not_observed_in_isolated_run",
                    "compiled_artifacts": [],
                }
            )
            continue
        artifacts = [
            _compiled_artifact_evidence(
                cache_root,
                metadata_path,
                symbol,
                expected_async_execution_count=len(execution_ids),
            )
            for metadata_path in metadata_paths
        ]
        artifact_count += len(artifacts)
        cache_keys = [artifact["cache_key"] for artifact in artifacts]
        for execution_id in execution_ids:
            execution_artifact_bindings.append(
                {
                    "execution_id": execution_id,
                    "kernel_symbol": symbol,
                    "artifact_cache_keys": cache_keys,
                    "evidence_scope": "containing_kernel_artifact",
                }
            )
        kernels.append(
            {
                **dict(expectation),
                "artifact_status": "validated_cp_async",
                "compiled_artifacts": artifacts,
            }
        )

    validated_async_kernel_count = sum(
        kernel["artifact_status"] == "validated_cp_async" for kernel in kernels
    )
    if async_expectations and validated_async_kernel_count == 0:
        raise ValueError(
            f"isolated Triton cache {cache_root} has no compiled metadata/TTGIR/PTX "
            "for any declared async matrix kernel symbol; global historical caches "
            "are not searched"
        )

    return {
        "status": status,
        "cache_scope": "isolated_run",
        "cache_dir": str(cache_root),
        "global_cache_search": False,
        "matrix_kernel_count": len(kernels),
        "declared_async_matrix_kernel_count": len(async_expectations),
        "validated_async_matrix_kernel_count": validated_async_kernel_count,
        "unobserved_async_matrix_kernel_symbols": sorted(
            symbol for symbol in async_expectations if not metadata_by_symbol[symbol]
        ),
        "compiled_artifact_count": artifact_count,
        "declared_async_matrix_execution_count": len(declared_execution_symbols),
        "artifact_linked_async_execution_count": len(execution_artifact_bindings),
        "execution_evidence_scope": "containing_kernel_artifact",
        "kernels": kernels,
        "execution_artifact_bindings": execution_artifact_bindings,
    }


def inspect_generated_package(
    generated_dir: Path, *, allow_ordinary: bool = False
) -> PackageContract:
    """Validate the one-layer, single-token ABI and fingerprint the package."""

    generated_dir = generated_dir.resolve()
    if not (generated_dir / "__init__.py").is_file():
        raise FileNotFoundError(
            f"generated PyNTT package has no __init__.py: {generated_dir}"
        )

    metadata_path = generated_dir / "metadata.json"
    metadata = _read_json(metadata_path)
    if not isinstance(metadata, Mapping):
        raise ValueError(f"metadata.json must contain an object: {metadata_path}")
    target_machine = metadata.get("target_machine")
    if not isinstance(target_machine, str) or not target_machine:
        raise ValueError("metadata.json target_machine must be a non-empty string")
    functions = metadata.get("functions")
    if not isinstance(functions, list):
        raise ValueError("metadata.json has no functions array")
    entries = [item for item in functions if item.get("is_entry") is True]
    if len(entries) != 1:
        raise ValueError(
            f"expected exactly one generated entry function, found {len(entries)}"
        )
    entry = entries[0]
    inputs = entry.get("inputs")
    if not isinstance(inputs, list):
        raise ValueError("entry function has no inputs array")
    input_ids = [item for item in inputs if item.get("name") == "input_ids"]
    if len(input_ids) != 1:
        raise ValueError(
            f"expected exactly one input named 'input_ids', found {len(input_ids)}"
        )
    input_spec = input_ids[0]
    input_dtype = str(input_spec.get("dtype", ""))
    if input_dtype != "int64":
        raise ValueError(f"input_ids must have dtype int64, got {input_dtype!r}")

    raw_shape = input_spec.get("shape")
    if not isinstance(raw_shape, list) or not raw_shape:
        raise ValueError("input_ids must have a non-empty fixed shape")
    if any(
        isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
        for dim in raw_shape
    ):
        raise ValueError(
            f"input_ids must have a positive fixed shape, got {raw_shape!r}"
        )
    input_shape = tuple(int(dim) for dim in raw_shape)
    if math.prod(input_shape) != 1:
        raise ValueError(
            "this benchmark requires the static one-token Qwen3 package; "
            f"input_ids shape is {input_shape}"
        )

    layer_ids = tuple(
        sorted({str(value) for value in _collect_named_values(metadata, "layer_id")})
    )
    if layer_ids != ("0",):
        raise ValueError(
            "this benchmark requires a one-layer Qwen3 package; "
            f"detected layer IDs {list(layer_ids)}"
        )

    fingerprint_names = (
        "__init__.py",
        "generated_kernels.py",
        "kernel_params.json",
        "metadata.json",
        "model.py",
        "rdata.py",
        "runtime_config.py",
        "specs.py",
        "requirements.txt",
    )
    file_hashes = {
        name: _sha256_file(generated_dir / name)
        for name in fingerprint_names
        if (generated_dir / name).is_file()
    }
    code_manifest_digest = hashlib.sha256()
    for name, digest in sorted(file_hashes.items()):
        code_manifest_digest.update(name.encode("utf-8"))
        code_manifest_digest.update(b"\0")
        code_manifest_digest.update(digest.encode("ascii"))
        code_manifest_digest.update(b"\n")

    assets_dir = generated_dir / "assets"
    asset_file_bytes = {
        path.relative_to(generated_dir).as_posix(): path.stat().st_size
        for path in sorted(assets_dir.glob("**/*"))
        if path.is_file()
    }
    asset_file_sha256 = {
        path.relative_to(generated_dir).as_posix(): _sha256_file(path)
        for path in sorted(assets_dir.glob("**/*"))
        if path.is_file()
    }

    manifest_summary = {
        "pyntt_spec_version": metadata.get("pyntt_spec_version"),
        "target_kind": metadata.get("target_kind"),
        "target_machine": metadata.get("target_machine"),
        "backend": metadata.get("backend"),
        "strict": metadata.get("strict"),
        "function_count": len(functions),
    }
    kernel_params_path = generated_dir / "kernel_params.json"
    if not kernel_params_path.is_file():
        raise FileNotFoundError(
            f"required generated-package file not found: {kernel_params_path}"
        )
    kernel_params = _read_json(kernel_params_path)
    if not isinstance(kernel_params, Mapping):
        raise ValueError(
            f"kernel_params.json must contain an object: {kernel_params_path}"
        )
    manifest_summary["pyntt_codegen_manifest_version"] = kernel_params.get(
        "pyntt_codegen_manifest_version"
    )
    source_path = generated_dir / "generated_kernels.py"
    try:
        generated_source = source_path.read_text(encoding="utf-8-sig")
    except FileNotFoundError as ex:
        raise FileNotFoundError(
            f"required generated-package file not found: {source_path}"
        ) from ex
    pipeline_gate_summary = _validate_pipeline_gate(
        kernel_params, generated_source, source_path, allow_ordinary
    )

    return PackageContract(
        entry_function=str(entry.get("name", "")),
        input_name="input_ids",
        input_dtype=input_dtype,
        input_shape=input_shape,
        detected_layer_ids=layer_ids,
        manifest_summary=manifest_summary,
        pipeline_gate_summary=pipeline_gate_summary,
        source_file_sha256=dict(sorted(file_hashes.items())),
        asset_file_bytes=asset_file_bytes,
        asset_file_sha256=asset_file_sha256,
        code_manifest_sha256=code_manifest_digest.hexdigest(),
    )


def _enum_names(values: Iterable[Any]) -> list[str]:
    return [str(getattr(value, "name", value)) for value in values]


def _runner_config_metadata(runner: Any) -> dict[str, Any]:
    return {
        "model_layers": int(runner.num_layers),
        "num_kv_heads": int(runner.num_kv_heads),
        "head_dim": int(runner.head_dim),
        "kv_dtype": str(runner.kv_type),
        "block_size": int(runner.block_size),
        "num_blocks": int(runner.num_blocks),
        "max_sessions": int(runner.max_sessions),
        "max_model_len": int(runner.max_model_len),
        "hierarchy": [int(value) for value in runner.hierarchy],
        "key_cache_layout": _enum_names(runner.key_cache_layout),
        "value_cache_layout": _enum_names(runner.value_cache_layout),
        "key_vectorized_axes": _enum_names(runner.key_vectorized_axes),
        "value_vectorized_axes": _enum_names(runner.value_vectorized_axes),
        "key_lanes": [int(value) for value in runner.key_lanes],
        "value_lanes": [int(value) for value in runner.value_lanes],
        "sharding_axes": _enum_names(runner.sharding_axes),
        "axis_policies": [
            [int(axis) for axis in policy] for policy in runner.axis_policies
        ],
    }


def _git_metadata(repo: Path) -> dict[str, Any]:
    def run_git(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=repo,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    commit = run_git("rev-parse", "HEAD")
    status = run_git("status", "--short", "--untracked-files=no")
    return {
        "commit": commit,
        "tracked_worktree_dirty": bool(status) if status is not None else None,
    }


def _nvidia_driver_version(device_index: int) -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(device_index),
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as ex:
        raise RuntimeError("cannot determine NVIDIA driver version") from ex
    versions = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(versions) != 1:
        raise RuntimeError(
            f"expected one NVIDIA driver version for device {device_index}, got {versions}"
        )
    return versions[0]


def _runtime_metadata(torch: Any, device_index: int, repo: Path) -> dict[str, Any]:
    import _nncase
    import nncase
    import triton

    properties = torch.cuda.get_device_properties(device_index)
    capability = torch.cuda.get_device_capability(device_index)
    hardware_fields = (
        "total_memory",
        "multi_processor_count",
        "warp_size",
        "max_threads_per_multi_processor",
        "shared_memory_per_block",
        "shared_memory_per_multiprocessor",
        "regs_per_multiprocessor",
    )
    hardware = {
        "device_type": "cuda",
        "device_index": int(device_index),
        "name": str(properties.name),
        "compute_capability": [int(capability[0]), int(capability[1])],
        "driver_version": _nvidia_driver_version(device_index),
    }
    for field in hardware_fields:
        if hasattr(properties, field):
            hardware[field] = int(getattr(properties, field))

    capabilities = {
        "cuda_available": bool(torch.cuda.is_available()),
        "bfloat16_supported": bool(torch.cuda.is_bf16_supported()),
        "tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
        "triton_tle_available": importlib.util.find_spec("triton.experimental.tle")
        is not None,
        "static_single_token_input": True,
        "cuda_events": True,
    }
    software = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_cuda_runtime": str(torch.version.cuda),
        "cudnn": int(torch.backends.cudnn.version())
        if torch.backends.cudnn.version() is not None
        else None,
        "triton": str(getattr(triton, "__version__", "unknown")),
        "nncase": str(
            getattr(nncase, "__version__", getattr(_nncase, "__version__", "unknown"))
        ),
    }
    return {
        "hardware": hardware,
        "capabilities": capabilities,
        "software": software,
        "source": _git_metadata(repo),
    }


def _runner_override_config(scratch_root: Path) -> str:
    root = json.dumps(str(scratch_root))
    return f"""
root = {root}

[compile_opt]
dump_ir = false
shape_bucket_enable = false
shape_bucket_range_info = {{ }}
shape_bucket_segments_count = 0
shape_bucket_segments = {{ }}
shape_bucket_fix_var_map = {{ "sequence_length"=1 }}

[huggingface_options]
output_logits = true
output_hidden_states = false
num_layers = 1
tensor_type = "bfloat16"

[paged_attention_config]
kv_type = "bfloat16"
key_lanes = [8]
value_lanes = [8]

[generator]
[generator.inputs]
method = "text"
number = 1
batch = 1

[generator.inputs.text]
args = "tests/importer/huggingface_/prompt.txt"
sequence_length = 1

[generator.calibs]
method = "text"
number = 1
batch = 1

[generator.calibs.text]
args = "tests/importer/huggingface_/prompt.txt"
sequence_length = 1
"""


def _prepare_runner(repo: Path, model_dir: Path, scratch_root: Path, torch: Any) -> Any:
    tests_root = str(repo / "tests")
    pyntt_root = str(repo / "pyntt")
    for path in (tests_root, pyntt_root):
        if path not in sys.path:
            sys.path.insert(0, path)

    from huggingface_test_runner import HuggingfaceTestRunner

    previous_targets = os.environ.get("NNCASE_TEST_TARGETS")
    os.environ["NNCASE_TEST_TARGETS"] = "pyntt"
    try:
        runner = HuggingfaceTestRunner(
            "pyntt_qwen3_benchmark",
            overwrite_configs=_runner_override_config(scratch_root),
        )
        runner.parse_model(str(model_dir))
    finally:
        if previous_targets is None:
            os.environ.pop("NNCASE_TEST_TARGETS", None)
        else:
            os.environ["NNCASE_TEST_TARGETS"] = previous_targets

    # parse_model owns the canonical Qwen/tokenizer/paged-KV setup.  The HF
    # weights are not part of the measured path and must not compete for GPU
    # memory with the generated package.
    if hasattr(runner, "model"):
        del runner.model
    gc.collect()
    torch.cuda.empty_cache()
    return runner


def _read_prompt(prompt_file: Path, prompt_index: int) -> str:
    lines = [
        line.strip().strip('"')
        for line in prompt_file.read_text(encoding="utf-8").splitlines()
    ]
    prompts = [line for line in lines if line]
    if not prompts:
        raise ValueError(f"prompt file contains no non-empty prompts: {prompt_file}")
    if prompt_index < 0 or prompt_index >= len(prompts):
        raise ValueError(
            f"prompt index {prompt_index} is outside [0, {len(prompts) - 1}]"
        )
    return prompts[prompt_index]


def _tokenize_prompt(runner: Any, prompt: str) -> list[int]:
    messages = [
        {"role": "system", "content": "You are a assistant!"},
        {"role": "user", "content": prompt},
    ]
    text = runner.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    tokenized = runner.tokenizer([text], return_tensors="np")
    return [int(value) for value in tokenized.input_ids[0]]


def _first_output(output: Any) -> Any:
    if isinstance(output, (tuple, list)):
        if not output:
            raise RuntimeError("generated model returned an empty output sequence")
        return output[0]
    return output


def _make_invoker(
    model: Any,
    torch: Any,
    device_index: int,
    input_shape: tuple[int, ...],
) -> Callable[[int, Any, bool], StepMeasurement]:
    def invoke(
        input_token: int, kv_cache: Any, needs_prediction: bool
    ) -> StepMeasurement:
        input_tensor = torch.tensor(
            [int(input_token)],
            dtype=torch.int64,
            device=f"cuda:{device_index}",
        ).reshape(input_shape)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            start_event.record()
            wall_start_ns = time.perf_counter_ns()
            output = model(input_tensor, kv_cache)
            end_event.record()
            end_event.synchronize()
            wall_ms = (time.perf_counter_ns() - wall_start_ns) / 1_000_000.0
        cuda_ms = float(start_event.elapsed_time(end_event))

        predicted_token = None
        if needs_prediction:
            logits = _first_output(output)
            if logits.numel() == 0:
                raise RuntimeError("generated model returned empty logits")
            predicted_token = int(
                torch.argmax(logits.reshape(-1, logits.shape[-1])[-1]).item()
            )
        return StepMeasurement(cuda_ms, wall_ms, predicted_token)

    return invoke


def _run_request(
    model: Any,
    runner: Any,
    torch: Any,
    device_index: int,
    input_shape: tuple[int, ...],
    prompt_token_ids: Sequence[int],
    output_tokens: int,
) -> dict[str, Any]:
    import nncase

    scheduler = nncase.PagedAttentionScheduler(
        runner.kv_cache_config,
        runner.num_blocks,
        runner.max_model_len,
        runner.hierarchy,
    )
    torch.cuda.synchronize(device_index)
    invoke = _make_invoker(model, torch, device_index, input_shape)
    return drive_token_by_token(
        prompt_token_ids,
        output_tokens,
        lambda: scheduler.schedule([0], [1]),
        invoke,
    )


def _summarize_scenario(
    scenario: Scenario,
    requests: Sequence[Mapping[str, Any]],
    tokenizer: Any,
) -> dict[str, Any]:
    if not requests:
        raise ValueError("scenario requires at least one timed request")
    token_sequences = [list(request["generated_token_ids"]) for request in requests]
    if any(tokens != token_sequences[0] for tokens in token_sequences[1:]):
        raise RuntimeError(
            f"non-deterministic token output in scenario {scenario.name}: {token_sequences}"
        )

    prefill_cuda = [
        value for request in requests for value in request["prefill_cuda_ms"]
    ]
    prefill_wall = [
        value for request in requests for value in request["prefill_wall_ms"]
    ]
    decode_cuda = [value for request in requests for value in request["decode_cuda_ms"]]
    decode_wall = [value for request in requests for value in request["decode_wall_ms"]]
    total_cuda = [request["total_model_cuda_ms"] for request in requests]
    total_wall = [request["total_model_wall_ms"] for request in requests]
    ttft_cuda = [sum(request["prefill_cuda_ms"]) for request in requests]
    ttft_wall = [sum(request["prefill_wall_ms"]) for request in requests]

    total_cuda_summary = summarize_ms(total_cuda)
    result = {
        "name": scenario.name,
        "key": scenario.key,
        "prompt_tokens": scenario.prompt_tokens,
        "output_tokens": scenario.output_tokens,
        "model_calls_per_request": scenario.model_calls,
        "prefill_model_calls_per_request": scenario.prompt_tokens,
        "decode_model_calls_per_request": scenario.output_tokens - 1,
        "latency_ms": {
            "total_model_cuda": total_cuda_summary,
            "total_model_wall": summarize_ms(total_wall),
            "time_to_first_token_cuda": summarize_ms(ttft_cuda),
            "time_to_first_token_wall": summarize_ms(ttft_wall),
            "prefill_token_cuda": summarize_ms(prefill_cuda),
            "prefill_token_wall": summarize_ms(prefill_wall),
            "decode_token_cuda": summarize_ms(decode_cuda) if decode_cuda else None,
            "decode_token_wall": summarize_ms(decode_wall) if decode_wall else None,
        },
        "throughput_from_p50_cuda": {
            "model_calls_per_second": scenario.model_calls
            * 1000.0
            / total_cuda_summary["p50_ms"],
            "generated_tokens_per_second": scenario.output_tokens
            * 1000.0
            / total_cuda_summary["p50_ms"],
        },
        "generated_token_ids": token_sequences[0],
        "generated_tokens": [
            tokenizer.decode(token, skip_special_tokens=False)
            for token in token_sequences[0]
        ],
    }
    return result


def _package_contract_json(
    contract: PackageContract,
    generated_dir: Path,
    compiled_artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pipeline_gate = dict(contract.pipeline_gate_summary)
    if compiled_artifacts is not None:
        pipeline_gate["compiled_artifacts"] = dict(compiled_artifacts)
    return {
        "generated_dir": str(generated_dir.resolve()),
        "entry_function": contract.entry_function,
        "input_ids": {
            "name": contract.input_name,
            "dtype": contract.input_dtype,
            "shape": list(contract.input_shape),
            "tokens_per_call": 1,
        },
        "detected_layer_ids": list(contract.detected_layer_ids),
        "manifest": dict(contract.manifest_summary),
        "pipeline_gate": pipeline_gate,
        "source_file_sha256": dict(contract.source_file_sha256),
        "asset_file_bytes": dict(contract.asset_file_bytes),
        "asset_file_sha256": dict(contract.asset_file_sha256),
        "code_manifest_sha256": contract.code_manifest_sha256,
    }


def prepare_isolated_triton_cache(cache_dir: Path) -> Path:
    """Create an empty cache root; never admit stale or global artifacts."""

    cache_dir = Path(os.path.abspath(os.fspath(cache_dir.expanduser())))
    if cache_dir.is_symlink():
        raise ValueError(f"isolated Triton cache must not be a symlink: {cache_dir}")
    if cache_dir.exists():
        if not cache_dir.is_dir():
            raise ValueError(f"Triton cache path is not a directory: {cache_dir}")
        if any(cache_dir.iterdir()):
            raise ValueError(
                f"isolated Triton cache must be empty before the run: {cache_dir}"
            )
    else:
        cache_dir.mkdir(parents=True)
    return cache_dir.resolve(strict=True)


@contextmanager
def _isolated_triton_cache_environment(cache_dir: Path):
    updates = {
        "TRITON_CACHE_DIR": str(cache_dir),
        "TRITON_STORE_BINARY_ONLY": "0",
    }
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _benchmark_cache_dir(args: argparse.Namespace) -> Path:
    explicit = getattr(args, "triton_cache_dir", None)
    if explicit is not None:
        return Path(explicit)
    output_json = getattr(args, "output_json", None)
    if output_json is None:
        raise ValueError(
            "run_benchmark requires output_json or an explicit triton_cache_dir"
        )
    output_json = Path(output_json)
    return output_json.with_name(output_json.name + ".triton-cache")


def run_benchmark(args: argparse.Namespace, repo: Path) -> dict[str, Any]:
    """Run the CUDA benchmark in a fresh, inspectable Triton cache."""

    loaded_triton_modules = sorted(
        name for name in sys.modules if name == "triton" or name.startswith("triton.")
    )
    if loaded_triton_modules:
        raise RuntimeError(
            "the isolated Triton cache must be configured before importing Triton; "
            "run this benchmark from a fresh Python process (already loaded: "
            + ", ".join(loaded_triton_modules[:5])
            + ")"
        )
    cache_dir = prepare_isolated_triton_cache(_benchmark_cache_dir(args))
    with _isolated_triton_cache_environment(cache_dir):
        return _run_benchmark_in_cache(args, repo, cache_dir)


def _run_benchmark_in_cache(
    args: argparse.Namespace, repo: Path, cache_dir: Path
) -> dict[str, Any]:
    """Run after the cache environment is fixed, before importing GPU modules."""

    try:
        import torch
    except ImportError as ex:
        raise RuntimeError("PyNTT benchmark requires PyTorch") from ex
    if not torch.cuda.is_available():
        raise RuntimeError("PyNTT benchmark requires an available CUDA device")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")

    scenarios = parse_scenarios(args.scenario)
    annotations = parse_annotations(args.annotation)
    contract = inspect_generated_package(
        args.generated_dir, allow_ordinary=bool(args.allow_ordinary)
    )
    max_prompt_tokens = max(scenario.prompt_tokens for scenario in scenarios)
    max_sequence_tokens = max(
        scenario.prompt_tokens + scenario.output_tokens - 1 for scenario in scenarios
    )

    torch.cuda.set_device(args.device)
    with tempfile.TemporaryDirectory(prefix="nncase-pyntt-qwen3-benchmark-") as scratch:
        runner = _prepare_runner(repo, args.model_dir, Path(scratch), torch)
        if int(runner.num_layers) != 1:
            raise ValueError(
                f"runner configured {runner.num_layers} model layers, expected 1"
            )
        if max_sequence_tokens > int(runner.max_model_len):
            raise ValueError(
                f"scenario requires {max_sequence_tokens} KV tokens, but max_model_len is "
                f"{runner.max_model_len}"
            )

        prompt = _read_prompt(args.prompt_file, args.prompt_index)
        all_prompt_ids = _tokenize_prompt(runner, prompt)
        if len(all_prompt_ids) < max_prompt_tokens:
            raise ValueError(
                f"tokenized prompt has {len(all_prompt_ids)} tokens, but a scenario requires "
                f"{max_prompt_tokens}; choose a longer prompt"
            )

        generated_package = runner.load_pyntt_generated_package(
            str(args.generated_dir.resolve())
        )
        model = generated_package.load_model()

        scenario_results = []
        for scenario in scenarios:
            prompt_ids = all_prompt_ids[: scenario.prompt_tokens]
            for _ in range(args.warmup):
                _run_request(
                    model,
                    runner,
                    torch,
                    args.device,
                    contract.input_shape,
                    prompt_ids,
                    scenario.output_tokens,
                )
            requests = [
                _run_request(
                    model,
                    runner,
                    torch,
                    args.device,
                    contract.input_shape,
                    prompt_ids,
                    scenario.output_tokens,
                )
                for _ in range(args.iterations)
            ]
            scenario_results.append(
                _summarize_scenario(scenario, requests, runner.tokenizer)
            )

        runtime_metadata = _runtime_metadata(torch, args.device, repo)
        compiled_artifacts = inspect_compiled_pipeline_artifacts(
            cache_dir, contract.pipeline_gate_summary
        )
        report = {
            "schema_version": SCHEMA_VERSION,
            "benchmark": BENCHMARK_NAME,
            "label": args.label,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "measurement": {
                "instrumentation": "none",
                "timing": "CUDA events around generated model calls; synchronized wall time",
                "includes": [
                    "generated PyNTT model dispatch",
                    "generated CUDA kernels",
                ],
                "excludes": [
                    "model compilation and package import",
                    "paged-KV scheduler construction and schedule calls",
                    "host-to-device input token creation",
                    "logits argmax and token decode",
                ],
                "percentile_method": "linear interpolation over per-request samples",
                "fixed_output_tokens": True,
            },
            "run_config": {
                "warmup_iterations_per_scenario": args.warmup,
                "timed_iterations_per_scenario": args.iterations,
                "device_index": args.device,
                "model_dir": str(args.model_dir.resolve()),
                "model_config_sha256": _sha256_file(args.model_dir / "config.json"),
                "prompt_file": str(args.prompt_file.resolve()),
                "prompt_index": args.prompt_index,
                "prompt_text_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "tokenized_prompt_tokens_available": len(all_prompt_ids),
                "triton_cache_dir": str(cache_dir),
                "annotations": annotations,
            },
            "model_package": _package_contract_json(
                contract, args.generated_dir, compiled_artifacts
            ),
            "paged_attention_config": _runner_config_metadata(runner),
            **runtime_metadata,
            "scenarios": scenario_results,
        }
    return report


def _load_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as ex:
        raise FileNotFoundError(f"benchmark JSON not found: {path}") from ex
    except json.JSONDecodeError as ex:
        raise ValueError(f"invalid benchmark JSON {path}: {ex}") from ex
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported benchmark schema_version {report.get('schema_version')!r}"
        )
    if report.get("benchmark") != BENCHMARK_NAME:
        raise ValueError(f"unexpected benchmark {report.get('benchmark')!r}")
    scenarios = report.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("benchmark report must contain at least one scenario")
    for scenario in scenarios:
        if not isinstance(scenario, Mapping) or not scenario.get("key"):
            raise ValueError("each benchmark scenario must have a key")
        latency = scenario.get("latency_ms")
        if not isinstance(latency, Mapping):
            raise ValueError(f"scenario {scenario.get('key')} has no latency_ms object")


def _metric_summary(scenario: Mapping[str, Any], metric: str) -> Mapping[str, Any]:
    latency = scenario["latency_ms"]
    summary = latency.get(metric)
    if not isinstance(summary, Mapping):
        raise ValueError(f"scenario {scenario['key']} has no metric {metric!r}")
    for field in ("p50_ms", "p90_ms"):
        value = summary.get(field)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(
                f"scenario {scenario['key']} metric {metric!r} has invalid {field}"
            )
    if float(summary["p90_ms"]) < float(summary["p50_ms"]):
        raise ValueError(
            f"scenario {scenario['key']} metric {metric!r} has p90 below p50"
        )
    return summary


def _report_path(report: Mapping[str, Any], path: str) -> Any:
    value: Any = report
    for field in path.split("."):
        if not isinstance(value, Mapping) or field not in value:
            raise ValueError(f"benchmark report has no comparability field {path!r}")
        value = value[field]
    return value


def _validate_comparable_reports(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> None:
    fields = (
        "hardware.name",
        "hardware.compute_capability",
        "hardware.driver_version",
        "measurement",
        "run_config.warmup_iterations_per_scenario",
        "run_config.timed_iterations_per_scenario",
        "run_config.device_index",
        "run_config.model_config_sha256",
        "run_config.prompt_index",
        "run_config.prompt_text_sha256",
        "model_package.input_ids",
        "model_package.detected_layer_ids",
        "model_package.manifest.target_kind",
        "model_package.manifest.target_machine",
        "model_package.manifest.backend",
        "model_package.asset_file_sha256",
        "paged_attention_config",
    )
    differences = [
        path
        for path in fields
        if _report_path(candidate, path) != _report_path(baseline, path)
    ]
    candidate_scenarios = {
        scenario["key"]: (
            scenario.get("prompt_tokens"),
            scenario.get("output_tokens"),
            scenario.get("model_calls_per_request"),
        )
        for scenario in candidate["scenarios"]
    }
    baseline_scenarios = {
        scenario["key"]: (
            scenario.get("prompt_tokens"),
            scenario.get("output_tokens"),
            scenario.get("model_calls_per_request"),
        )
        for scenario in baseline["scenarios"]
    }
    if candidate_scenarios != baseline_scenarios:
        differences.append(
            "scenarios.prompt_tokens/output_tokens/model_calls_per_request"
        )
    candidate_pipeline_count = _report_path(
        candidate, "model_package.pipeline_gate.async_matrix_execution_count"
    )
    baseline_pipeline_count = _report_path(
        baseline, "model_package.pipeline_gate.async_matrix_execution_count"
    )
    if candidate_pipeline_count <= 0 or baseline_pipeline_count != 0:
        differences.append(
            "model_package.pipeline_gate.async_matrix_execution_count"
            "(candidate>0,baseline=0)"
        )
    if differences:
        raise ValueError(
            "baseline and candidate reports are not strictly comparable; differing "
            "fields: " + ", ".join(differences)
        )


def _nice_axis_max(value: float) -> float:
    if value <= 0.0:
        return 1.0
    exponent = math.floor(math.log10(value))
    magnitude = 10.0**exponent
    normalized = value / magnitude
    for candidate in (1.0, 2.0, 5.0, 10.0):
        if normalized <= candidate:
            return candidate * magnitude
    return 10.0 * magnitude


def render_comparison_svg(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any] | None = None,
    metric: str = "total_model_cuda",
    title: str = "Qwen3 0.6B · 1-layer PyNTT",
) -> str:
    """Render deterministic p50 bars and p90 markers for benchmark reports."""

    validate_report(candidate)
    if baseline is not None:
        validate_report(baseline)
    if metric not in SVG_METRICS:
        raise ValueError(f"unsupported SVG metric {metric!r}")

    candidate_scenarios = list(candidate["scenarios"])
    baseline_by_key = (
        {scenario["key"]: scenario for scenario in baseline["scenarios"]}
        if baseline is not None
        else {}
    )
    if baseline is not None:
        candidate_keys = {scenario["key"] for scenario in candidate_scenarios}
        baseline_keys = set(baseline_by_key)
        if candidate_keys != baseline_keys:
            raise ValueError(
                "baseline and candidate scenario keys differ: "
                f"candidate={sorted(candidate_keys)}, baseline={sorted(baseline_keys)}"
            )
        _validate_comparable_reports(candidate, baseline)

    series = []
    if baseline is not None:
        series.append(("baseline", baseline, "#F58518"))
    series.append(("candidate", candidate, "#4C78A8"))
    all_p90 = []
    for scenario in candidate_scenarios:
        all_p90.append(float(_metric_summary(scenario, metric)["p90_ms"]))
        if baseline is not None:
            all_p90.append(
                float(
                    _metric_summary(baseline_by_key[scenario["key"]], metric)["p90_ms"]
                )
            )

    axis_max = _nice_axis_max(max(all_p90) * 1.08)
    width = 1080
    left = 250
    right = 70
    top = 178
    row_height = 92 if baseline is not None else 66
    plot_width = width - left - right
    plot_height = len(candidate_scenarios) * row_height
    height = top + plot_height + 92
    baseline_label = str(baseline.get("label", "baseline")) if baseline else None
    candidate_label = str(candidate.get("label", "candidate"))
    metric_label = metric.replace("_", " ")

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "  <style>",
        "    text { font-family: ui-sans-serif, system-ui, sans-serif; fill: #1f2937; }",
        "    .title { font-size: 24px; font-weight: 700; }",
        "    .subtitle { font-size: 13px; fill: #4b5563; }",
        "    .axis { font-size: 12px; fill: #6b7280; }",
        "    .scenario { font-size: 14px; font-weight: 600; }",
        "    .value { font-size: 12px; }",
        "    .legend { font-size: 12px; }",
        "  </style>",
        f'  <rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'  <text class="title" x="32" y="42">{escape(title)}</text>',
        f'  <text class="subtitle" x="32" y="68">metric: {escape(metric_label)} · filled bar: p50 · line marker: p90 · lower is better</text>',
    ]

    legend_x = 32
    for role, report, color in series:
        label = baseline_label if role == "baseline" else candidate_label
        lines.extend(
            [
                f'  <rect x="{legend_x}" y="88" width="18" height="12" rx="2" fill="{color}"/>',
                f'  <text class="legend" x="{legend_x + 25}" y="99">{escape(str(label))}</text>',
            ]
        )
        legend_x += max(150, 45 + len(str(label)) * 8)

    for tick in range(6):
        value = axis_max * tick / 5.0
        x = left + plot_width * tick / 5.0
        lines.extend(
            [
                f'  <line x1="{x:.1f}" y1="{top - 22}" x2="{x:.1f}" y2="{top + plot_height}" stroke="#e5e7eb" stroke-width="1"/>',
                f'  <text class="axis" x="{x:.1f}" y="{top - 30}" text-anchor="middle">{value:.3g} ms</text>',
            ]
        )

    for row, candidate_scenario in enumerate(candidate_scenarios):
        y_base = top + row * row_height
        scenario_label = (
            f"{candidate_scenario.get('name', candidate_scenario['key'])}  "
            f"({candidate_scenario['prompt_tokens']}→{candidate_scenario['output_tokens']})"
        )
        lines.append(
            f'  <text class="scenario" x="{left - 16}" y="{y_base + 19}" text-anchor="end">{escape(scenario_label)}</text>'
        )

        row_series = []
        if baseline is not None:
            row_series.append(
                (
                    baseline_label,
                    baseline_by_key[candidate_scenario["key"]],
                    "#F58518",
                )
            )
        row_series.append((candidate_label, candidate_scenario, "#4C78A8"))
        for series_index, (label, scenario, color) in enumerate(row_series):
            summary = _metric_summary(scenario, metric)
            p50 = float(summary["p50_ms"])
            p90 = float(summary["p90_ms"])
            y = y_base + 4 + series_index * 27
            p50_width = plot_width * p50 / axis_max
            p90_x = left + plot_width * p90 / axis_max
            lines.extend(
                [
                    f'  <rect x="{left}" y="{y}" width="{p50_width:.2f}" height="16" rx="2" fill="{color}" fill-opacity="0.88"/>',
                    f'  <line x1="{p90_x:.2f}" y1="{y - 3}" x2="{p90_x:.2f}" y2="{y + 19}" stroke="{color}" stroke-width="3"/>',
                    f'  <text class="value" x="{left + p50_width + 7:.2f}" y="{y + 13}">{escape(str(label))}: {p50:.3f} / {p90:.3f} ms</text>',
                ]
            )

        if baseline is not None:
            baseline_p50 = float(
                _metric_summary(baseline_by_key[candidate_scenario["key"]], metric)[
                    "p50_ms"
                ]
            )
            candidate_p50 = float(_metric_summary(candidate_scenario, metric)["p50_ms"])
            delta = (
                "n/a"
                if baseline_p50 == 0.0
                else f"{(candidate_p50 / baseline_p50 - 1.0) * 100.0:+.1f}%"
            )
            lines.append(
                f'  <text class="axis" x="{width - right}" y="{y_base + 73}" text-anchor="end">candidate p50 Δ {delta}</text>'
            )

    hardware = candidate.get("hardware", {})
    device = str(hardware.get("name", "unknown device"))
    capability = hardware.get("compute_capability", ["?", "?"])
    package = candidate.get("model_package", {})
    package_hash = str(package.get("code_manifest_sha256", "unknown"))[:12]
    footer_y = top + plot_height + 38
    lines.extend(
        [
            f'  <text class="subtitle" x="32" y="{footer_y}">candidate device: {escape(device)} · compute capability: {escape(".".join(map(str, capability)))}</text>',
            f'  <text class="subtitle" x="32" y="{footer_y + 22}">candidate package: {escape(package_hash)} · benchmark schema: {SCHEMA_VERSION}</text>',
            "</svg>",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_svg(
    path: Path,
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    metric: str,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_comparison_svg(candidate, baseline, metric=metric, title=title),
        encoding="utf-8",
    )


def _add_svg_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--metric",
        choices=SVG_METRICS,
        default="total_model_cuda",
        help="Latency metric plotted as p50 bars and p90 markers.",
    )
    parser.add_argument(
        "--title",
        default="Qwen3 0.6B · 1-layer PyNTT",
        help="SVG chart title.",
    )


def build_argument_parser(repo: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run an uninstrumented CUDA benchmark.")
    run.add_argument("--generated-dir", type=Path, required=True)
    run.add_argument(
        "--model-dir",
        type=Path,
        default=repo / "tests/llm/Qwen/Qwen3-0.6B",
        help="Local Qwen3-0.6B model used for canonical tokenizer/KV configuration.",
    )
    run.add_argument(
        "--prompt-file",
        type=Path,
        default=repo / "tests/importer/huggingface_/prompt.txt",
    )
    run.add_argument("--prompt-index", type=int, default=0)
    run.add_argument(
        "--scenario",
        action="append",
        help="Repeatable NAME:PROMPT_TOKENS:OUTPUT_TOKENS; defaults to 1x3 and 20x3.",
    )
    run.add_argument("--warmup", type=int, default=5)
    run.add_argument("--iterations", type=int, default=20)
    run.add_argument("--device", type=int, default=0)
    run.add_argument("--label", default="candidate")
    run.add_argument(
        "--triton-cache-dir",
        type=Path,
        help=(
            "Fresh persistent Triton cache used for compiled TTGIR/PTX evidence. "
            "Defaults to <output-json>.triton-cache and must be empty."
        ),
    )
    run.add_argument(
        "--allow-ordinary",
        action="store_true",
        help=(
            "Allow an intentional ordinary non-pipelined baseline. By default at "
            "least one GEMM/GEMV matrix execution must pass the explicit two-slot "
            "cp.async artifact gate."
        ),
    )
    run.add_argument(
        "--annotation",
        action="append",
        default=[],
        help="Repeatable KEY=JSON configuration annotation, e.g. pipeline_depth=2.",
    )
    run.add_argument("--output-json", type=Path, required=True)
    run.add_argument("--baseline-json", type=Path)
    run.add_argument("--output-svg", type=Path)
    _add_svg_arguments(run)

    svg = subparsers.add_parser("svg", help="Render existing benchmark JSON.")
    svg.add_argument("--candidate-json", type=Path, required=True)
    svg.add_argument("--baseline-json", type=Path)
    svg.add_argument("--output-svg", type=Path, required=True)
    _add_svg_arguments(svg)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = build_argument_parser(repo)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            if args.baseline_json is not None and args.output_svg is None:
                raise ValueError("--baseline-json requires --output-svg")
            report = run_benchmark(args, repo)
            _write_json(args.output_json, report)
            if args.output_svg is not None:
                baseline = (
                    _load_report(args.baseline_json)
                    if args.baseline_json is not None
                    else None
                )
                _write_svg(args.output_svg, report, baseline, args.metric, args.title)
            print(f"json={args.output_json}")
            if args.output_svg is not None:
                print(f"svg={args.output_svg}")
        else:
            candidate = _load_report(args.candidate_json)
            baseline = (
                _load_report(args.baseline_json)
                if args.baseline_json is not None
                else None
            )
            _write_svg(args.output_svg, candidate, baseline, args.metric, args.title)
            print(f"svg={args.output_svg}")
    except (FileNotFoundError, RuntimeError, ValueError) as ex:
        parser.error(str(ex))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
