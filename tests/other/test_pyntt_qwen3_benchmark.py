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

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from tools.pyntt_qwen3_benchmark import BENCHMARK_NAME
from tools.pyntt_qwen3_benchmark import SCHEMA_VERSION
from tools.pyntt_qwen3_benchmark import StepMeasurement
from tools.pyntt_qwen3_benchmark import _package_contract_json
from tools.pyntt_qwen3_benchmark import build_argument_parser
from tools.pyntt_qwen3_benchmark import drive_token_by_token
from tools.pyntt_qwen3_benchmark import inspect_compiled_pipeline_artifacts
from tools.pyntt_qwen3_benchmark import inspect_generated_package
from tools.pyntt_qwen3_benchmark import main
from tools.pyntt_qwen3_benchmark import parse_annotations
from tools.pyntt_qwen3_benchmark import parse_scenario
from tools.pyntt_qwen3_benchmark import percentile
from tools.pyntt_qwen3_benchmark import prepare_isolated_triton_cache
from tools.pyntt_qwen3_benchmark import render_comparison_svg
from tools.pyntt_qwen3_benchmark import run_benchmark
from tools.pyntt_qwen3_benchmark import summarize_ms


_REGION_ID = "main_prim/pipeline_op0_packed_mat_mul__reduction0"


def test_latency_summary_uses_linear_percentiles():
    samples = [4.0, 1.0, 3.0, 2.0]

    assert percentile(samples, 50) == pytest.approx(2.5)
    assert percentile(samples, 90) == pytest.approx(3.7)
    summary = summarize_ms(samples)
    assert summary["count"] == 4
    assert summary["mean_ms"] == pytest.approx(2.5)
    assert summary["p50_ms"] == pytest.approx(2.5)
    assert summary["p90_ms"] == pytest.approx(3.7)


def test_scenario_and_annotation_parsing_are_strict():
    scenario = parse_scenario("prefill:20:3")
    assert scenario.prompt_tokens == 20
    assert scenario.output_tokens == 3
    assert scenario.model_calls == 22

    assert parse_annotations(
        ["pipeline_depth=2", 'policy="explicit"', "tag=nightly"]
    ) == {
        "pipeline_depth": 2,
        "policy": "explicit",
        "tag": "nightly",
    }
    with pytest.raises(ValueError, match="positive"):
        parse_scenario("bad:0:3")
    with pytest.raises(ValueError, match="duplicate"):
        parse_annotations(["n=1", "n=2"])


def test_token_driver_prefills_and_decodes_one_token_per_call():
    scheduled_contexts = []
    invocations = []
    predictions = iter([100, 101, 102])

    def schedule_one():
        context = len(scheduled_contexts)
        scheduled_contexts.append(context)
        return {"context": context, "query_len": 1}

    def invoke_one(token, kv_cache, needs_prediction):
        invocations.append((token, kv_cache, needs_prediction))
        predicted = next(predictions) if needs_prediction else None
        return StepMeasurement(1.0, 2.0, predicted)

    result = drive_token_by_token([10, 11, 12], 3, schedule_one, invoke_one)

    # Three prompt calls produce token 100; only tokens 100 and 101 are fed
    # back to obtain three output tokens in total.
    assert [item[0] for item in invocations] == [10, 11, 12, 100, 101]
    assert [item[2] for item in invocations] == [False, False, True, True, True]
    assert [item[1]["query_len"] for item in invocations] == [1, 1, 1, 1, 1]
    assert scheduled_contexts == [0, 1, 2, 3, 4]
    assert result["generated_token_ids"] == [100, 101, 102]
    assert result["prefill_cuda_ms"] == [1.0, 1.0, 1.0]
    assert result["decode_cuda_ms"] == [1.0, 1.0]
    assert result["total_model_cuda_ms"] == pytest.approx(5.0)


def _pipeline_execution(
    *,
    region_id=_REGION_ID,
    marker="__PYNTT_PIPELINE_EXECUTION_0__",
    descriptor_name="rhs_shared_buffer_0",
    arena_offset_bytes=0,
):
    allocation = {
        "buffer_name": "rhs_shared",
        "descriptor_name": descriptor_name,
        "stage_count": 2,
        "stage_physical_bytes": 64,
        "stage_stride_bytes": 64,
        "physical_bytes": 128,
        "arena_id": "pyntt_shared_arena",
        "arena_offset_bytes": arena_offset_bytes,
        "scalar_element_size_bytes": 2,
        "triton_dtype": "tl.float16",
        "logical_stage_shape": [32],
        "logical_stage_strides": [1],
        "vector_lane_shape": [1],
        "descriptor_shape": [2, 32],
        "storage_encoding": "linear",
        "nv_mma_shared_layout": False,
    }
    return {
        "marker": marker,
        "region_id": region_id,
        "schedule_id": "qwen3.packed-matmul.k.cp-async.n2",
        "template_id": "triton.loop.cp_async.n2.v1",
        "stage_count": 2,
        "prefetch_distance": 1,
        "partition": "full",
        "synchronization": {
            "asynchronous_produce": True,
            "requires_producer_commit": True,
            "requires_consumer_wait": True,
            "wait_provides_consumer_acquire": False,
            "requires_consumer_release": True,
        },
        "tail_policy": "serial",
        "loop_variable": "logical_sequence",
        "loop_start": "0",
        "loop_stop": "4",
        "loop_step": "1",
        "channels": [
            {
                "channel_id": "rhs",
                "source_memory_space": "gpu.block-global",
                "destination_memory_space": "gpu.shared",
                "allocation": allocation,
            }
        ],
        "produce_source": "\n".join(
            [
                "rhs_slot = "
                f"{descriptor_name}.slot(tl.cast(logical_sequence % 2, tl.int32))",
                "tle.gpu.copy(input0, rhs_slot, [32], is_async=True)",
            ]
        ),
        "consume_source": "\n".join(
            [
                "rhs_slot = "
                f"{descriptor_name}.slot(tl.cast(logical_sequence % 2, tl.int32))",
                "value = tl.load(rhs_slot)",
            ]
        ),
    }


def _strict_launch():
    return {
        "meta": {},
        "tuning": {"parameters": {}},
        "sharding": {
            "strategy": "replicated",
            "placement_axis": "b",
            "tensor_axis": 0,
            "extent": "1",
            "hierarchy": [1],
            "hierarchy_levels": "b",
            "global_shape": [],
        },
        "num_warps": 8,
        "num_stages": 1,
    }


def _matrix_render_kernel(symbol, *, async_pipeline):
    execution = _pipeline_execution() if async_pipeline else None
    return {
        "metadata": {
            "name": symbol,
            "op_kind": "packed_mat_mul",
            "inputs": ["input0", "input1"],
            "outputs": ["output0"],
            "attrs": {
                "ops": ["packed_mat_mul"],
                "block_microkernels": [
                    {
                        "helper": "packed_matmul_gemv_compute",
                        "family": "triton.gemv",
                    }
                ],
                "target_worker_width": 32,
                "target_threads_per_block": 256,
                "shared_memory_bytes": 128 if async_pipeline else 0,
                "shared_memory_allocation_size_policy": "granularity_aligned",
                "shared_memory_allocation_granularity_bytes": 1,
                "shared_memory_capacity_bytes": 48 * 1024,
            },
            "launch": _strict_launch(),
        },
        "helpers": [],
        "device_functions": [],
        "pipeline_executions": [execution] if execution is not None else [],
        "body_source": (
            f"# {execution['marker']}" if execution is not None else "pass"
        ),
    }


def _render_codegen_manifest(manifest):
    repo_root = Path(__file__).resolve().parents[2]
    package_root = str(repo_root / "pyntt")
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    from pyntt.codegen.render import render_manifest

    return render_manifest(manifest)


def _write_codegen_manifest(path, manifest):
    (path / "kernel_params.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (path / "generated_kernels.py").write_text(
        _render_codegen_manifest(manifest), encoding="utf-8"
    )


def _write_generated_package(
    path,
    input_shape=(1,),
    layer_ids=("0",),
    *,
    async_pipeline=True,
):
    path.mkdir()
    metadata = {
        "pyntt_spec_version": 4,
        "target_kind": "pyntt",
        "target_machine": "cuda_rtx5060",
        "backend": "triton",
        "strict": True,
        "functions": [
            {
                "name": "main_prim",
                "is_entry": True,
                "inputs": [
                    {
                        "name": "input_ids",
                        "dtype": "int64",
                        "shape": list(input_shape),
                    },
                    {"name": "kvCache", "dtype": "object", "shape": []},
                ],
                "generated_kernels": [
                    {"attrs": {"layer_id": layer_id}} for layer_id in layer_ids
                ],
            }
        ],
    }
    (path / "metadata.json").write_text(
        "\ufeff" + json.dumps(metadata), encoding="utf-8"
    )
    kernel_params = {
        "pyntt_codegen_manifest_version": 6,
        "target_kind": "pyntt",
        "backend": "triton",
        "functions": [
            {
                "id": 0,
                "name": "main_prim",
                "module_kind": "pyntt",
                "is_entry": True,
                "render_kernels": [
                    _matrix_render_kernel(
                        "main_prim_packed_mat_mul_0",
                        async_pipeline=async_pipeline,
                    )
                ],
            }
        ],
    }
    _write_codegen_manifest(path, kernel_params)
    (path / "__init__.py").write_text("# __init__.py\n", encoding="utf-8")
    (path / "model.py").write_text("# model.py\n", encoding="utf-8")
    (path / "assets").mkdir()
    (path / "assets" / "module_rdata.bin").write_bytes(b"weights")


def _append_matrix_render_kernel(path, symbol, *, async_pipeline):
    manifest_path = path / "kernel_params.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    kernel = _matrix_render_kernel(symbol, async_pipeline=async_pipeline)
    if async_pipeline:
        execution = kernel["pipeline_executions"][0]
        execution["region_id"] = f"main_prim/{symbol}__reduction0"
        execution["marker"] = "__PYNTT_PIPELINE_EXECUTION_1__"
        allocation = execution["channels"][0]["allocation"]
        old_descriptor = allocation["descriptor_name"]
        allocation["descriptor_name"] = f"{symbol}_rhs"
        execution["produce_source"] = execution["produce_source"].replace(
            old_descriptor, allocation["descriptor_name"]
        )
        execution["consume_source"] = execution["consume_source"].replace(
            old_descriptor, allocation["descriptor_name"]
        )
        kernel["body_source"] = f"# {execution['marker']}"
    manifest["functions"][0]["render_kernels"].append(kernel)
    _write_codegen_manifest(path, manifest)


def test_generated_package_contract_and_fingerprint(tmp_path):
    generated_dir = tmp_path / "generated"
    _write_generated_package(generated_dir)

    first = inspect_generated_package(generated_dir)
    second = inspect_generated_package(generated_dir)

    assert first.entry_function == "main_prim"
    assert first.input_dtype == "int64"
    assert first.input_shape == (1,)
    assert first.detected_layer_ids == ("0",)
    assert first.code_manifest_sha256 == second.code_manifest_sha256
    assert first.source_file_sha256 == second.source_file_sha256
    assert first.asset_file_bytes == {"assets/module_rdata.bin": 7}
    assert first.asset_file_bytes == second.asset_file_bytes
    assert len(first.asset_file_sha256["assets/module_rdata.bin"]) == 64
    assert first.asset_file_sha256 == second.asset_file_sha256
    assert first.manifest_summary["pyntt_codegen_manifest_version"] == 6

    gate = first.pipeline_gate_summary
    assert gate["status"] == "passed"
    assert gate["maximum_stage_count"] == 2
    assert gate["async_matrix_execution_count"] == 1
    assert gate["declared_async_execution_count"] == 1
    assert gate["physical_staged_capacity_bytes"] == 128
    assert gate["matrix_kernels"] == [
        {
            "manifest_path": "manifest.functions[0].render_kernels[0]",
            "kernel_symbol": "main_prim_packed_mat_mul_0",
            "pipeline_mode": "cp_async_n2",
            "async_execution_ids": [_REGION_ID],
            "schedule_ids": ["qwen3.packed-matmul.k.cp-async.n2"],
            "pipeline_execution_count": 1,
            "staged_buffer_count": 1,
            "unique_physical_range_count": 1,
            "physical_staged_capacity_bytes": 128,
            "pipeline_executions": [
                {
                    "region_id": _REGION_ID,
                    "schedule_id": "qwen3.packed-matmul.k.cp-async.n2",
                    "template_id": "triton.loop.cp_async.n2.v1",
                    "partition": "full",
                    "owner_kind": "render_kernel",
                    "owner_name": "main_prim_packed_mat_mul_0",
                    "staged_buffer_count": 1,
                    "stage_count": 2,
                }
            ],
            "source_evidence": {
                "async_copy": True,
                "async_commit_group": True,
                "async_wait_group_1": True,
                "async_wait_group_0": True,
                "staged_slot": True,
            },
        }
    ]


@pytest.mark.parametrize(
    "input_shape,layer_ids,message",
    [
        ((20,), ("0",), "static one-token"),
        ((1,), ("0", "1"), "one-layer"),
        ((1,), (), "one-layer"),
    ],
)
def test_generated_package_rejects_incompatible_contract(
    tmp_path, input_shape, layer_ids, message
):
    generated_dir = tmp_path / "generated"
    _write_generated_package(generated_dir, input_shape, layer_ids)

    with pytest.raises(ValueError, match=message):
        inspect_generated_package(generated_dir)


def test_generated_package_pipeline_gate_requires_v6(tmp_path):
    generated_dir = tmp_path / "generated"
    _write_generated_package(generated_dir)
    manifest_path = generated_dir / "kernel_params.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pyntt_codegen_manifest_version"] = 4
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="expected 6"):
        inspect_generated_package(generated_dir)


def test_generated_package_rejects_v4_pipeline_tables(tmp_path):
    generated_dir = tmp_path / "generated"
    _write_generated_package(generated_dir)
    manifest_path = generated_dir / "kernel_params.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    kernel = manifest["functions"][0]["render_kernels"][0]
    kernel["pipeline_regions"] = []
    kernel["staged_smem_allocations"] = []
    kernel["staged_smem_bindings"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected fields"):
        inspect_generated_package(generated_dir)


def test_generated_package_allows_only_explicit_ordinary_baseline(tmp_path):
    generated_dir = tmp_path / "generated"
    _write_generated_package(generated_dir, async_pipeline=False)

    with pytest.raises(ValueError, match="at least one GEMM/GEMV matrix execution"):
        inspect_generated_package(generated_dir)

    contract = inspect_generated_package(generated_dir, allow_ordinary=True)
    gate = contract.pipeline_gate_summary
    assert gate["status"] == "ordinary_baseline_allowed"
    assert gate["manifest_version"] == 6
    assert gate["matrix_kernel_count"] == 1
    assert gate["async_matrix_kernel_count"] == 0
    assert gate["ordinary_matrix_kernel_count"] == 1
    assert gate["async_matrix_execution_count"] == 0
    assert gate["declared_async_execution_count"] == 0
    assert gate["staged_buffer_count"] == 0
    assert gate["maximum_stage_count"] == 1
    assert gate["physical_staged_capacity_bytes"] == 0
    assert gate["matrix_kernels"][0]["pipeline_mode"] == "ordinary"
    assert gate["matrix_kernels"][0]["async_execution_ids"] == []
    assert not any(gate["matrix_kernels"][0]["source_evidence"].values())


def test_pipeline_gate_requires_one_async_matrix_execution_not_every_matrix_kernel(
    tmp_path,
):
    generated_dir = tmp_path / "generated"
    _write_generated_package(generated_dir)
    _append_matrix_render_kernel(
        generated_dir, "main_prim_packed_mat_mul_ordinary", async_pipeline=False
    )

    gate = inspect_generated_package(generated_dir).pipeline_gate_summary

    assert gate["matrix_kernel_count"] == 2
    assert gate["async_matrix_kernel_count"] == 1
    assert gate["ordinary_matrix_kernel_count"] == 1
    ordinary = next(
        kernel
        for kernel in gate["matrix_kernels"]
        if kernel["kernel_symbol"] == "main_prim_packed_mat_mul_ordinary"
    )
    assert ordinary["pipeline_mode"] == "ordinary"
    assert ordinary["async_execution_ids"] == []
    assert not any(ordinary["source_evidence"].values())


def test_pipeline_gate_does_not_borrow_async_source_from_another_function(tmp_path):
    generated_dir = tmp_path / "generated"
    _write_generated_package(generated_dir)
    source_path = generated_dir / "generated_kernels.py"
    source = source_path.read_text(encoding="utf-8")
    source_path.write_text(
        source.replace(
            "def main_prim_packed_mat_mul_0(",
            "def unrelated_async_kernel(",
            1,
        )
        + "\ndef main_prim_packed_mat_mul_0():\n    pass\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing evidence"):
        inspect_generated_package(generated_dir)


def test_pipeline_gate_follows_only_reachable_generated_helpers(tmp_path):
    generated_dir = tmp_path / "generated"
    _write_generated_package(generated_dir)
    source_path = generated_dir / "generated_kernels.py"
    source = source_path.read_text(encoding="utf-8")
    helper_source = source.replace(
        "def main_prim_packed_mat_mul_0(",
        "def reachable_async_helper(",
        1,
    )
    source_path.write_text(
        "def main_prim_packed_mat_mul_0():\n"
        "    reachable_async_helper()\n\n"
        + helper_source,
        encoding="utf-8",
    )

    gate = inspect_generated_package(generated_dir).pipeline_gate_summary

    assert all(gate["matrix_kernels"][0]["source_evidence"].values())


def test_pipeline_gate_does_not_count_non_matrix_async_execution(tmp_path):
    generated_dir = tmp_path / "generated"
    _write_generated_package(generated_dir, async_pipeline=False)
    _append_matrix_render_kernel(
        generated_dir, "main_prim_elementwise_async", async_pipeline=True
    )
    manifest_path = generated_dir / "kernel_params.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    non_matrix = manifest["functions"][0]["render_kernels"][1]
    non_matrix["metadata"]["op_kind"] = "elementwise"
    non_matrix["metadata"]["attrs"]["ops"] = ["relu"]
    non_matrix["metadata"]["attrs"]["block_microkernels"] = []
    _write_codegen_manifest(generated_dir, manifest)

    with pytest.raises(ValueError, match="at least one GEMM/GEMV matrix execution"):
        inspect_generated_package(generated_dir)


def test_pipeline_gate_requires_globally_unique_execution_ids(tmp_path):
    generated_dir = tmp_path / "generated"
    _write_generated_package(generated_dir)
    _append_matrix_render_kernel(
        generated_dir, "main_prim_packed_mat_mul_1", async_pipeline=True
    )
    manifest_path = generated_dir / "kernel_params.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    second = manifest["functions"][0]["render_kernels"][1]
    second["pipeline_executions"][0]["region_id"] = _REGION_ID
    _write_codegen_manifest(generated_dir, manifest)

    with pytest.raises(ValueError, match="globally unique"):
        inspect_generated_package(generated_dir)


@pytest.mark.parametrize(
    "source_edit,missing",
    [
        (lambda source: source.replace(", is_async=True", ""), "async_copy"),
        (
            lambda source: source.replace("tle.gpu.async_commit_group()", ""),
            "async_commit_group",
        ),
        (
            lambda source: source.replace("tle.gpu.async_wait_group(1)", "", 1),
            "async_wait_group_1",
        ),
        (
            lambda source: source.replace("tle.gpu.async_wait_group(0)", "", 1),
            "async_wait_group_0",
        ),
        (
            lambda source: source.replace(".slot(", ".not_slot("),
            "staged_slot",
        ),
    ],
)
def test_generated_package_pipeline_gate_requires_source_evidence(
    tmp_path, source_edit, missing
):
    generated_dir = tmp_path / "generated"
    _write_generated_package(generated_dir)
    source_path = generated_dir / "generated_kernels.py"
    source_path.write_text(
        source_edit(source_path.read_text(encoding="utf-8")), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=missing):
        inspect_generated_package(generated_dir)


@pytest.mark.parametrize(
    "forbidden_source,construct",
    [
        ("def bad():\n    tle.pipe()\n", r"tle\.pipe"),
        ("def bad():\n    tle.gpu.warp_specialize()\n", "warp_specialize"),
        (
            "def bad():\n    return tl.range(0, 4, warp_specialize=True)\n",
            "warp_specialize",
        ),
        (
            "def bad():\n    return tl.range(0, 4, num_stages=2)\n",
            r"tl\.range",
        ),
    ],
)
def test_generated_package_pipeline_gate_rejects_implicit_or_warp_pipeline(
    tmp_path, forbidden_source, construct
):
    generated_dir = tmp_path / "generated"
    _write_generated_package(generated_dir, async_pipeline=False)
    (generated_dir / "generated_kernels.py").write_text(
        forbidden_source, encoding="utf-8"
    )

    with pytest.raises(ValueError, match=construct):
        inspect_generated_package(generated_dir, allow_ordinary=True)


def test_run_parser_requires_opt_in_for_ordinary_baseline(tmp_path):
    parser = build_argument_parser(tmp_path)
    common = [
        "run",
        "--generated-dir",
        str(tmp_path / "generated"),
        "--output-json",
        str(tmp_path / "result.json"),
    ]
    assert parser.parse_args(common).allow_ordinary is False
    assert parser.parse_args([*common, "--allow-ordinary"]).allow_ordinary is True
    run_parser = next(
        action.choices["run"]
        for action in parser._actions
        if getattr(action, "choices", None) and "run" in action.choices
    )
    assert "ordinary non-pipelined baseline" in run_parser.format_help()


_SYNTHETIC_TTGIR = """
module {
  tt.func public @main_prim_packed_mat_mul_0() {
    %copy0 = ttg.async_copy_global_to_local %src0, %slot0 {tle.required_async_copy}
    %commit0 = ttg.async_commit_group tokens %copy0
    %wait1 = ttg.async_wait %commit0 {num = 1 : i32, tle.explicit_async_wait}
    %copy1 = ttg.async_copy_global_to_local %src1, %slot1 {tle.required_async_copy}
    %commit1 = ttg.async_commit_group tokens %copy1
    %wait0 = ttg.async_wait %commit1 {num = 0 : i32, tle.explicit_async_wait}
  }
}
"""

_SYNTHETIC_PTX = """
.visible .entry main_prim_packed_mat_mul_0() {
    cp.async.ca.shared.global [ %r1 ], [ %rd1 ], 0x10;
    cp.async.commit_group;
    cp.async.wait_group 1;
    bar.sync 0;
    ld.shared.b32 %r2, [ %r1 ];
    cp.async.ca.shared.global [ %r3 ], [ %rd2 ], 0x10;
    cp.async.commit_group;
    cp.async.wait_group 0;
    bar.sync 0;
    ld.shared.b32 %r4, [ %r3 ];
    ret;
}
"""


def _write_synthetic_compiled_artifact(
    cache_dir,
    *,
    symbol="main_prim_packed_mat_mul_0",
    cache_key="SYNTHETIC_CACHE_KEY",
    ttgir=_SYNTHETIC_TTGIR,
    ptx=_SYNTHETIC_PTX,
):
    ttgir = ttgir.replace("main_prim_packed_mat_mul_0", symbol)
    ptx = ptx.replace("main_prim_packed_mat_mul_0", symbol)
    artifact_dir = cache_dir / cache_key
    artifact_dir.mkdir()
    (artifact_dir / f"{symbol}.json").write_text(
        json.dumps({"name": symbol, "target": {"backend": "cuda", "arch": 90}}),
        encoding="utf-8",
    )
    (artifact_dir / f"{symbol}.ttgir").write_text(ttgir, encoding="utf-8")
    (artifact_dir / f"{symbol}.ptx").write_text(ptx, encoding="utf-8")


def test_compiled_pipeline_gate_uses_execution_linked_isolated_artifacts(tmp_path):
    generated_dir = tmp_path / "generated"
    _write_generated_package(generated_dir)
    contract = inspect_generated_package(generated_dir)
    cache_dir = prepare_isolated_triton_cache(tmp_path / "triton-cache")
    _write_synthetic_compiled_artifact(cache_dir)

    evidence = inspect_compiled_pipeline_artifacts(
        cache_dir, contract.pipeline_gate_summary
    )

    assert evidence["status"] == "passed"
    assert evidence["cache_scope"] == "isolated_run"
    assert evidence["global_cache_search"] is False
    assert evidence["compiled_artifact_count"] == 1
    assert evidence["validated_async_matrix_kernel_count"] == 1
    assert evidence["artifact_linked_async_execution_count"] == 1
    assert evidence["execution_artifact_bindings"] == [
        {
            "execution_id": _REGION_ID,
            "kernel_symbol": "main_prim_packed_mat_mul_0",
            "artifact_cache_keys": ["SYNTHETIC_CACHE_KEY"],
            "evidence_scope": "containing_kernel_artifact",
        }
    ]
    artifact = evidence["kernels"][0]["compiled_artifacts"][0]
    assert artifact["actual_kernel_symbol"] == "main_prim_packed_mat_mul_0"
    assert artifact["ttgir_evidence"]["kernel_symbol_definition_count"] == 1
    assert artifact["ttgir_evidence"]["async_copy_global_to_local_count"] == 2
    assert artifact["ttgir_evidence"]["required_async_copy_marker_count"] == 2
    assert artifact["ttgir_evidence"]["async_wait_group_1_count"] == 1
    assert artifact["ttgir_evidence"]["async_wait_group_0_count"] == 1
    assert artifact["ptx_evidence"]["cp_async_copy_count"] == 2
    assert artifact["ptx_evidence"]["kernel_symbol_entry_count"] == 1
    assert artifact["ptx_evidence"]["wait_group_1_then_barrier_count"] == 1
    assert artifact["ptx_evidence"]["wait_group_0_then_barrier_count"] == 1
    assert len(artifact["files"]["ttgir"]["sha256"]) == 64
    package_json = _package_contract_json(contract, generated_dir, evidence)
    assert package_json["pipeline_gate"]["compiled_artifacts"] == evidence


@pytest.mark.parametrize(
    "rendezvous",
    (
        "bar.sync 0;",
        "bar.cta.sync 0;",
        "barrier.sync.aligned 0;",
        "barrier.cta.sync.aligned 0;",
    ),
)
def test_compiled_pipeline_gate_accepts_ptx_cta_thread_rendezvous(tmp_path, rendezvous):
    generated_dir = tmp_path / "generated"
    _write_generated_package(generated_dir)
    contract = inspect_generated_package(generated_dir)
    cache_dir = prepare_isolated_triton_cache(tmp_path / "triton-cache")
    _write_synthetic_compiled_artifact(
        cache_dir, ptx=_SYNTHETIC_PTX.replace("bar.sync 0;", rendezvous)
    )

    evidence = inspect_compiled_pipeline_artifacts(
        cache_dir, contract.pipeline_gate_summary
    )

    ptx_evidence = evidence["kernels"][0]["compiled_artifacts"][0]["ptx_evidence"]
    assert ptx_evidence["wait_group_1_then_barrier_count"] == 1
    assert ptx_evidence["wait_group_0_then_barrier_count"] == 1


def test_compiled_gate_reports_ordinary_kernel_without_requiring_its_artifact(
    tmp_path,
):
    generated_dir = tmp_path / "generated"
    _write_generated_package(generated_dir)
    ordinary_symbol = "main_prim_packed_mat_mul_ordinary"
    _append_matrix_render_kernel(generated_dir, ordinary_symbol, async_pipeline=False)
    contract = inspect_generated_package(generated_dir)
    cache_dir = prepare_isolated_triton_cache(tmp_path / "triton-cache")
    _write_synthetic_compiled_artifact(cache_dir)

    evidence = inspect_compiled_pipeline_artifacts(
        cache_dir, contract.pipeline_gate_summary
    )

    assert evidence["validated_async_matrix_kernel_count"] == 1
    ordinary = next(
        kernel
        for kernel in evidence["kernels"]
        if kernel["kernel_symbol"] == ordinary_symbol
    )
    assert ordinary["artifact_status"] == "not_applicable_ordinary"
    assert ordinary["compiled_artifacts"] == []


def test_compiled_gate_allows_explicit_ordinary_baseline_without_async_artifact(
    tmp_path,
):
    generated_dir = tmp_path / "generated"
    _write_generated_package(generated_dir, async_pipeline=False)
    contract = inspect_generated_package(generated_dir, allow_ordinary=True)
    cache_dir = prepare_isolated_triton_cache(tmp_path / "triton-cache")

    evidence = inspect_compiled_pipeline_artifacts(
        cache_dir, contract.pipeline_gate_summary
    )

    assert evidence["status"] == "ordinary_baseline_allowed"
    assert evidence["validated_async_matrix_kernel_count"] == 0
    assert evidence["compiled_artifact_count"] == 0
    assert evidence["kernels"][0]["artifact_status"] == (
        "not_applicable_ordinary"
    )


def test_compiled_gate_does_not_borrow_ordinary_symbol_artifact(tmp_path):
    generated_dir = tmp_path / "generated"
    _write_generated_package(generated_dir)
    ordinary_symbol = "main_prim_packed_mat_mul_ordinary"
    _append_matrix_render_kernel(generated_dir, ordinary_symbol, async_pipeline=False)
    contract = inspect_generated_package(generated_dir)
    cache_dir = prepare_isolated_triton_cache(tmp_path / "triton-cache")
    _write_synthetic_compiled_artifact(cache_dir, symbol=ordinary_symbol)

    with pytest.raises(ValueError, match="no compiled metadata/TTGIR/PTX"):
        inspect_compiled_pipeline_artifacts(cache_dir, contract.pipeline_gate_summary)


def test_compiled_gate_requires_at_least_one_observed_async_matrix_symbol(tmp_path):
    generated_dir = tmp_path / "generated"
    _write_generated_package(generated_dir)
    unobserved_symbol = "main_prim_packed_mat_mul_unobserved"
    _append_matrix_render_kernel(generated_dir, unobserved_symbol, async_pipeline=True)
    contract = inspect_generated_package(generated_dir)
    cache_dir = prepare_isolated_triton_cache(tmp_path / "triton-cache")
    _write_synthetic_compiled_artifact(cache_dir)

    evidence = inspect_compiled_pipeline_artifacts(
        cache_dir, contract.pipeline_gate_summary
    )

    assert evidence["declared_async_matrix_kernel_count"] == 2
    assert evidence["validated_async_matrix_kernel_count"] == 1
    assert evidence["unobserved_async_matrix_kernel_symbols"] == [unobserved_symbol]


def test_compiled_pipeline_gate_does_not_search_another_cache(tmp_path):
    generated_dir = tmp_path / "generated"
    _write_generated_package(generated_dir)
    contract = inspect_generated_package(generated_dir)
    isolated_cache = prepare_isolated_triton_cache(tmp_path / "isolated")
    historical_cache = prepare_isolated_triton_cache(tmp_path / "historical")
    _write_synthetic_compiled_artifact(historical_cache)

    with pytest.raises(ValueError, match="global historical caches are not searched"):
        inspect_compiled_pipeline_artifacts(
            isolated_cache, contract.pipeline_gate_summary
        )


def test_compiled_pipeline_gate_rejects_missing_required_async_transport(tmp_path):
    generated_dir = tmp_path / "generated"
    _write_generated_package(generated_dir)
    contract = inspect_generated_package(generated_dir)
    cache_dir = prepare_isolated_triton_cache(tmp_path / "triton-cache")
    _write_synthetic_compiled_artifact(
        cache_dir,
        ttgir=_SYNTHETIC_TTGIR.replace(
            "ttg.async_copy_global_to_local", "ttg.local_store"
        ).replace("ttg.async_commit_group", "arith.constant"),
    )

    with pytest.raises(ValueError, match="Required async transport"):
        inspect_compiled_pipeline_artifacts(cache_dir, contract.pipeline_gate_summary)


def test_compiled_pipeline_gate_requires_marker_per_declared_async_execution(tmp_path):
    generated_dir = tmp_path / "generated"
    _write_generated_package(generated_dir)
    manifest_path = generated_dir / "kernel_params.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    kernel = manifest["functions"][0]["render_kernels"][0]
    second_execution = _pipeline_execution(
        region_id="main_prim/pipeline_op1_packed_mat_mul__reduction0",
        marker="__PYNTT_PIPELINE_EXECUTION_1__",
        descriptor_name="rhs_shared_buffer_1",
        arena_offset_bytes=128,
    )
    kernel["pipeline_executions"].append(second_execution)
    kernel["body_source"] += f"\n# {second_execution['marker']}"
    kernel["metadata"]["attrs"]["shared_memory_bytes"] = 256
    _write_codegen_manifest(generated_dir, manifest)
    contract = inspect_generated_package(generated_dir)
    cache_dir = prepare_isolated_triton_cache(tmp_path / "triton-cache")
    ttgir_with_one_required_marker = _SYNTHETIC_TTGIR.replace(
        " {tle.required_async_copy}", "", 1
    )
    _write_synthetic_compiled_artifact(cache_dir, ttgir=ttgir_with_one_required_marker)

    with pytest.raises(ValueError, match=r"required_async_copy_marker_count>=2"):
        inspect_compiled_pipeline_artifacts(cache_dir, contract.pipeline_gate_summary)


def test_compiled_pipeline_gate_requires_barrier_before_shared_consumer(tmp_path):
    generated_dir = tmp_path / "generated"
    _write_generated_package(generated_dir)
    contract = inspect_generated_package(generated_dir)
    cache_dir = prepare_isolated_triton_cache(tmp_path / "triton-cache")
    ptx_without_steady_barrier = _SYNTHETIC_PTX.replace(
        "    cp.async.wait_group 1;\n    bar.sync 0;\n    ld.shared.b32",
        "    cp.async.wait_group 1;\n    ld.shared.b32",
    )
    _write_synthetic_compiled_artifact(cache_dir, ptx=ptx_without_steady_barrier)

    with pytest.raises(ValueError, match="wait_group_1_then_barrier_count"):
        inspect_compiled_pipeline_artifacts(cache_dir, contract.pipeline_gate_summary)


def test_compiled_pipeline_gate_rejects_membar_cta_as_thread_rendezvous(tmp_path):
    generated_dir = tmp_path / "generated"
    _write_generated_package(generated_dir)
    contract = inspect_generated_package(generated_dir)
    cache_dir = prepare_isolated_triton_cache(tmp_path / "triton-cache")
    _write_synthetic_compiled_artifact(
        cache_dir, ptx=_SYNTHETIC_PTX.replace("bar.sync 0;", "membar.cta;")
    )

    with pytest.raises(ValueError, match="wait_group_1_then_barrier_count"):
        inspect_compiled_pipeline_artifacts(cache_dir, contract.pipeline_gate_summary)


def test_compiled_pipeline_gate_does_not_accept_cp_async_bulk_as_cp_async(tmp_path):
    generated_dir = tmp_path / "generated"
    _write_generated_package(generated_dir)
    contract = inspect_generated_package(generated_dir)
    cache_dir = prepare_isolated_triton_cache(tmp_path / "triton-cache")
    _write_synthetic_compiled_artifact(
        cache_dir,
        ptx=_SYNTHETIC_PTX.replace(
            "cp.async.ca.shared.global", "cp.async.bulk.shared.global"
        ),
    )

    with pytest.raises(ValueError, match="cp_async_copy_count"):
        inspect_compiled_pipeline_artifacts(cache_dir, contract.pipeline_gate_summary)


def test_isolated_triton_cache_rejects_preexisting_artifacts(tmp_path):
    cache_dir = tmp_path / "triton-cache"
    cache_dir.mkdir()
    (cache_dir / "stale.ptx").write_text("stale", encoding="utf-8")

    with pytest.raises(ValueError, match="must be empty"):
        prepare_isolated_triton_cache(cache_dir)


def test_benchmark_rejects_triton_imported_before_cache_isolation(
    tmp_path, monkeypatch
):
    parser = build_argument_parser(tmp_path)
    args = parser.parse_args(
        [
            "run",
            "--generated-dir",
            str(tmp_path / "generated"),
            "--output-json",
            str(tmp_path / "result.json"),
        ]
    )
    monkeypatch.setitem(sys.modules, "triton", object())

    with pytest.raises(RuntimeError, match="before importing Triton"):
        run_benchmark(args, tmp_path)


def _summary(p50, p90):
    return {
        "count": 10,
        "mean_ms": p50,
        "stdev_ms": 0.1,
        "min_ms": p50 - 0.1,
        "p50_ms": p50,
        "p90_ms": p90,
        "p99_ms": p90 + 0.1,
        "max_ms": p90 + 0.2,
    }


def _report(label, scale):
    is_baseline = "baseline" in label or label.startswith("n=1")
    scenarios = []
    for name, prompt_tokens, output_tokens, latency in (
        ("decode_1x3", 1, 3, 3.0),
        ("prefill_20x3", 20, 3, 12.0),
    ):
        scenarios.append(
            {
                "name": name,
                "key": f"prompt={prompt_tokens};output={output_tokens}",
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "model_calls_per_request": prompt_tokens + output_tokens - 1,
                "latency_ms": {
                    "total_model_cuda": _summary(
                        latency * scale, latency * scale * 1.1
                    ),
                    "time_to_first_token_cuda": _summary(
                        latency * scale * 0.8, latency * scale * 0.9
                    ),
                    "decode_token_cuda": _summary(
                        latency * scale * 0.1, latency * scale * 0.12
                    ),
                },
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "label": label,
        "hardware": {
            "name": "Test GPU <0>",
            "compute_capability": [9, 0],
            "driver_version": "999.1",
        },
        "measurement": {
            "timing": "cuda-events",
            "fixed_output_tokens": True,
        },
        "run_config": {
            "warmup_iterations_per_scenario": 5,
            "timed_iterations_per_scenario": 20,
            "device_index": 0,
            "model_config_sha256": "model-config",
            "prompt_index": 0,
            "prompt_text_sha256": "prompt-hash",
        },
        "model_package": {
            "code_manifest_sha256": "abcdef0123456789",
            "input_ids": {
                "name": "input_ids",
                "dtype": "int64",
                "shape": [1],
                "tokens_per_call": 1,
            },
            "detected_layer_ids": ["0"],
            "manifest": {
                "target_kind": "pyntt",
                "target_machine": "cuda_rtx5060",
                "backend": "triton",
            },
            "pipeline_gate": {
                "async_matrix_execution_count": 0 if is_baseline else 1,
            },
            "asset_file_sha256": {"assets/module_rdata.bin": "asset-hash"},
        },
        "paged_attention_config": {"block_size": 16, "model_layers": 1},
        "scenarios": scenarios,
    }


def test_svg_is_deterministic_and_compares_matching_scenarios():
    baseline = _report("n=1 & baseline", 1.0)
    candidate = _report("n=2 candidate", 0.8)

    first = render_comparison_svg(candidate, baseline)
    second = render_comparison_svg(candidate, baseline)

    assert first == second
    assert "candidate p50 Δ -20.0%" in first
    assert "n=1 &amp; baseline" in first
    assert "Test GPU &lt;0&gt;" in first
    ET.fromstring(first)


def test_svg_rejects_non_matching_baseline():
    baseline = _report("baseline", 1.0)
    candidate = _report("candidate", 0.8)
    candidate["scenarios"].pop()

    with pytest.raises(ValueError, match="scenario keys differ"):
        render_comparison_svg(candidate, baseline)


@pytest.mark.parametrize(
    "path,new_value,field",
    [
        (("hardware", "driver_version"), "different", "hardware.driver_version"),
        (
            ("run_config", "prompt_text_sha256"),
            "different",
            "run_config.prompt_text_sha256",
        ),
        (
            ("model_package", "asset_file_sha256"),
            {"assets/module_rdata.bin": "different"},
            "model_package.asset_file_sha256",
        ),
        (
            ("model_package", "manifest", "target_machine"),
            "different",
            "model_package.manifest.target_machine",
        ),
        (("paged_attention_config", "block_size"), 32, "paged_attention_config"),
    ],
)
def test_svg_rejects_non_comparable_ab_reports(path, new_value, field):
    baseline = _report("baseline", 1.0)
    candidate = _report("candidate", 0.8)
    target = candidate
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = new_value

    with pytest.raises(ValueError, match=field):
        render_comparison_svg(candidate, baseline)


def test_svg_subcommand_writes_deterministic_artifact(tmp_path):
    baseline_json = tmp_path / "baseline.json"
    candidate_json = tmp_path / "candidate.json"
    output_svg = tmp_path / "comparison.svg"
    baseline_json.write_text(json.dumps(_report("n=1", 1.0)), encoding="utf-8")
    candidate_json.write_text(json.dumps(_report("n=2", 0.8)), encoding="utf-8")

    arguments = [
        "svg",
        "--baseline-json",
        str(baseline_json),
        "--candidate-json",
        str(candidate_json),
        "--output-svg",
        str(output_svg),
    ]
    assert main(arguments) == 0
    first = output_svg.read_bytes()
    assert main(arguments) == 0
    assert output_svg.read_bytes() == first
